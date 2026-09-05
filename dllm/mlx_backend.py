"""MLX runtime for the Mac node's layer shard, on Apple's Metal via MLX instead of torch/mps.

Same Qwen2 decoder maths as dllm/model.py's Layer/Shard (RMSNorm, QKV with bias, rotate-half RoPE,
grouped-query attention with a causal mask, KV cache concat, o_proj, second RMSNorm, SwiGLU MLP) --
this module only changes which array library runs it. It is transliterated from dllm/np_node.py's
numpy implementation, which dllm/test_np.py checks against the torch path directly; the same
op-for-op structure here (repeat instead of repeat_interleave, full-tuple transpose instead of
torch's two-axis swap, explicit rms_norm/rope/softmax) is what makes that agreement carry over.

Public surface mirrors dllm.model.Shard exactly, so dllm/node.py can use either interchangeably:
.a, .b, .cache, __call__(x, pos, req=...), forward_batch(x, positions, reqs), reset(req),
reassign(shard_dir, a, b). x/pos/positions and the return value cross this boundary as torch
tensors -- dllm/node.py builds torch tensors off the wire and calls to_bytes on what it gets back
-- and are converted to and from mlx arrays via numpy internally.

fp32, int8 and int4 shards are all supported the same way the numpy phone node handles them:
dllm.quant.dequant() turns whatever is on disk into fp32 arrays at load time, because MLX has no
packed int8/int4 matmul kernel here worth using instead (same reasoning as quant.dequant's own
docstring). The torch node keeps its quantised weights packed and uses a fast int8 kernel instead;
this module trades that runtime speedup for simplicity, exactly as the numpy node already does.
"""
import math
import os
import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn

from dllm.quant import dequant, unpack4


def set_memory_cap(gb):
    """Hold MLX's Metal allocations under `gb` gigabytes. Returns the cap in bytes, or None.

    MLX allocates from unified memory, so an unbounded shard on a Mac that is also running the hub
    competes with it for the same RAM. The limit makes MLX reclaim its buffer cache and, past that,
    fail the allocation rather than push the machine into swap, which is the failure mode worth
    having: a node that says it cannot hold a range is recoverable, a machine that starts swapping
    takes the whole cluster's latency with it. The buffer cache gets a quarter of the ceiling so
    reuse still happens inside it.

    This is MLX's own accounting, not an OS limit: it bounds what MLX allocates, not the process."""
    if not gb:
        return None
    b = int(float(gb) * 2 ** 30)
    mx.set_memory_limit(b)
    mx.set_cache_limit(b // 4)
    return b


def memory_now():
    """(active, peak) MLX bytes, for reporting what the cap is actually holding back."""
    try:
        return int(mx.get_active_memory()), int(mx.get_peak_memory())
    except Exception:
        return None, None


class LazyLayer:
    """One layer's npz, dequantised a tensor at a time rather than all at once.

    A 14B layer is 275M parameters: 131 MB as the int4 shard on disk, 1050 MB expanded to fp32.
    Expanding a whole layer and keeping it is what makes 42 of them impossible on a 32 GB Mac, so
    the raw arrays are held (cheap, they are still int4) and each tensor is widened only for as long
    as it takes to hand it to MLX."""

    def __init__(self, path):
        with np.load(path) as z:
            self.raw = {k: z[k] for k in z.files}
        self.quantised = any(k.endswith(".scale") for k in self.raw)

    def __contains__(self, k):
        return k in self.raw

    def __getitem__(self, k):
        s = f"{k}.scale"
        if s not in self.raw:
            return np.asarray(self.raw[k], dtype=np.float32)
        w, sc = self.raw[k], self.raw[s]
        if w.dtype == np.uint8:                      # int4: two codes per byte, scale per group
            return unpack4(w, sc)
        return w.astype(np.float32) * sc[:, None]    # int8: one scale per output row


def _load_layer(shard_dir, i):
    return LazyLayer(f"{shard_dir}/layer_{i:02d}.npz")


class QW:
    """A projection weight in MLX's own grouped form, for shards that were quantised on disk.

    Re-quantising rather than keeping fp32 is what lets the Mac hold most of a 14B: ~150 MB a layer
    instead of 1050 MB. The fp32 expansion never outlives this constructor. Shards that are fp32 on
    disk stay dense, so the small models keep exactly the numerics they had."""

    __slots__ = ("wq", "scales", "biases", "group", "bits")

    # The shard on disk is already int4 grouped by 128, so re-quantising to 4 bits quantises twice
    # and measured ~3% relative error on a 14B layer, which compounds over the 42 layers a Mac
    # holds. 8 bits costs about 260 MB a layer instead of 150 and brings that back to noise.
    # DLLM_MLX_BITS=4 trades it back if memory is tighter than accuracy.
    BITS = int(os.environ.get("DLLM_MLX_BITS", "8"))

    def __init__(self, w_oi, group=64, bits=None):
        bits = QW.BITS if bits is None else bits
        a = mx.array(w_oi)                                   # (out, in), as nn.Linear stores it
        self.wq, self.scales, self.biases = mx.quantize(a, group_size=group, bits=bits)
        mx.eval(self.wq, self.scales, self.biases)
        del a
        self.group, self.bits = group, bits


def _to_mx(x, dtype=np.float32):
    """torch.Tensor | list | np.ndarray -> mx.array, via numpy. `x` may be a torch tensor (the
    wire boundary uses these), a plain python list (forward_batch's `positions` arrives as one),
    or already a numpy array -- one path handles all three."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return mx.array(np.ascontiguousarray(x, dtype=dtype))


def _to_torch(y):
    return torch.from_numpy(np.array(y, copy=True))


def _lin(x, w, b):
    """x @ w (+ b if this projection has one). A dense weight is stored pre-transposed to
    (in, out), the same shape convention dllm/np_node.py uses, so it is a plain matmul; a [QW] is
    kept in the (out, in) layout mx.quantized_matmul wants and transposes on the fly."""
    if isinstance(w, QW):
        y = mx.quantized_matmul(x, w.wq, w.scales, w.biases, transpose=True,
                                group_size=w.group, bits=w.bits)
    else:
        y = x @ w
    return y if b is None else y + b


def rms_norm(x, w, eps):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, pos, theta):
    # x: (b, h, n, d) mx.float32   pos: (n,) mx.float32 absolute positions
    d = x.shape[-1]
    inv = 1.0 / mx.power(theta, mx.arange(0, d, 2, dtype=mx.float32) / d)
    f = pos[:, None] * inv[None, :]                          # (n, d/2)
    emb = mx.concatenate([f, f], axis=-1)                    # (n, d)
    cos, sin = mx.cos(emb)[None, None], mx.sin(emb)[None, None]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin


class MlxLayer:
    """One decoder block, weights as mx arrays. Same maths as dllm.model.Layer."""

    def __init__(self, cfg, w):
        """w: a [LazyLayer], or any mapping of key -> fp32 numpy array. When the shard was
        quantised on disk the seven projections are held in MLX's grouped form instead of dense
        fp32; the norms and biases are tiny and stay dense either way."""
        self.cfg = cfg
        q = getattr(w, "quantised", False)
        g = lambda k: mx.array(w[k])
        gb = lambda k: mx.array(w[k]) if k in w else None
        T = lambda k: QW(w[k]) if q else mx.array(np.ascontiguousarray(w[k].T))
        self.ln1 = g("input_layernorm.weight")
        self.ln2 = g("post_attention_layernorm.weight")
        self.wq, self.bq = T("self_attn.q_proj.weight"), gb("self_attn.q_proj.bias")
        self.wk, self.bk = T("self_attn.k_proj.weight"), gb("self_attn.k_proj.bias")
        self.wv, self.bv = T("self_attn.v_proj.weight"), gb("self_attn.v_proj.bias")
        self.wo, self.bo = T("self_attn.o_proj.weight"), gb("self_attn.o_proj.bias")
        self.wg, self.bg = T("mlp.gate_proj.weight"), gb("mlp.gate_proj.bias")
        self.wu, self.bu = T("mlp.up_proj.weight"), gb("mlp.up_proj.bias")
        self.wd, self.bd = T("mlp.down_proj.weight"), gb("mlp.down_proj.bias")

    def __call__(self, x, pos, past_k=None, past_v=None):
        """x: (1, n, hidden) mx.float32. pos: (n,) mx.float32 absolute positions.
        past_k/past_v: (1, kv_heads, p, head_dim) or None. Returns (x, k, v) with cache appended."""
        c = self.cfg
        b, n, _ = x.shape
        h = rms_norm(x, self.ln1, c.eps)
        q = _lin(h, self.wq, self.bq).reshape(b, n, c.heads, c.head_dim).transpose(0, 2, 1, 3)
        k = _lin(h, self.wk, self.bk).reshape(b, n, c.kv_heads, c.head_dim).transpose(0, 2, 1, 3)
        v = _lin(h, self.wv, self.bv).reshape(b, n, c.kv_heads, c.head_dim).transpose(0, 2, 1, 3)
        q, k = rope(q, pos, c.theta), rope(k, pos, c.theta)
        if past_k is not None:
            k, v = mx.concatenate([past_k, k], axis=2), mx.concatenate([past_v, v], axis=2)
        rep = c.heads // c.kv_heads
        kk, vv = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
        p = k.shape[2] - n
        mask = mx.tril(mx.ones((n, p + n), dtype=mx.bool_), k=p)
        att = (q @ kk.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(c.head_dim))
        att = mx.softmax(mx.where(mask, att, float("-inf")), axis=-1)
        o = (att @ vv).transpose(0, 2, 1, 3).reshape(b, n, c.hidden)
        x = x + _lin(o, self.wo, self.bo)
        h2 = rms_norm(x, self.ln2, c.eps)
        gate, up = _lin(h2, self.wg, self.bg), _lin(h2, self.wu, self.bu)
        x = x + _lin(nn.silu(gate) * up, self.wd, self.bd)
        return x, k, v

    def forward_batch(self, x, positions, past):
        """One decode step for several independent requests at once, mirroring
        dllm.model.Layer.forward_batch: x (B, 1, hidden) one row per request, `positions` (B,)
        each row's own absolute position, `past` a list of B (k, v) pairs or None. Projections and
        the MLP run once over the whole batch; attention is looped because each row has its own
        history, exactly like the torch and numpy versions."""
        c = self.cfg
        B = x.shape[0]
        h = rms_norm(x, self.ln1, c.eps)
        q = _lin(h, self.wq, self.bq).reshape(B, 1, c.heads, c.head_dim).transpose(0, 2, 1, 3)
        k = _lin(h, self.wk, self.bk).reshape(B, 1, c.kv_heads, c.head_dim).transpose(0, 2, 1, 3)
        v = _lin(h, self.wv, self.bv).reshape(B, 1, c.kv_heads, c.head_dim).transpose(0, 2, 1, 3)
        rep = c.heads // c.kv_heads
        rows, new_cache = [], []
        for i in range(B):
            pos_i = positions[i : i + 1]
            qi = rope(q[i : i + 1], pos_i, c.theta)
            ki = rope(k[i : i + 1], pos_i, c.theta)
            vi = v[i : i + 1]
            if past is not None and past[i] is not None:
                pk, pv = past[i]
                ki, vi = mx.concatenate([pk, ki], axis=2), mx.concatenate([pv, vi], axis=2)
            new_cache.append((ki, vi))
            att = (qi @ mx.repeat(ki, rep, axis=1).transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(c.head_dim))
            att = mx.softmax(att, axis=-1)                    # one query, all keys: no mask needed
            rows.append((att @ mx.repeat(vi, rep, axis=1)).transpose(0, 2, 1, 3).reshape(1, 1, c.hidden))
        x = x + _lin(mx.concatenate(rows, axis=0), self.wo, self.bo)
        h2 = rms_norm(x, self.ln2, c.eps)
        gate, up = _lin(h2, self.wg, self.bg), _lin(h2, self.wu, self.bu)
        x = x + _lin(nn.silu(gate) * up, self.wd, self.bd)
        return x, new_cache


class MlxShard:
    """Contiguous layer range [a, b) on MLX/Metal. Same public surface as dllm.model.Shard, so
    dllm/node.py can use either interchangeably: .a, .b, .cache, __call__(x, pos, req=...),
    forward_batch(x, positions, reqs), reset(req), reassign(shard_dir, a, b). Holds its own KV
    cache per request id, keyed and shaped exactly like the torch Shard's."""

    def __init__(self, cfg, shard_dir, a, b, device=None):
        # MLX picks its own default device (Metal/GPU) on import; only an explicit "cpu" asks for
        # MLX's CPU backend instead. `device` exists for call-signature symmetry with dllm.model.Shard.
        mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
        self.cfg, self.a, self.b = cfg, a, b
        self.layers = [MlxLayer(cfg, _load_layer(shard_dir, i)) for i in range(a, b)]
        self.cache = {}  # req -> list[(k, v)] of mx arrays, one pair per layer

    def reassign(self, shard_dir, a, b):
        """Move to [a, b), keeping every layer already loaded -- same contract as
        dllm.model.Shard.reassign, including dropping the KV cache: its entries are indexed by
        position in the old range."""
        have = {self.a + i: layer for i, layer in enumerate(self.layers)}
        self.layers = [have[i] if i in have else MlxLayer(self.cfg, _load_layer(shard_dir, i))
                       for i in range(a, b)]
        self.a, self.b = a, b
        self.cache = {}

    def __call__(self, x, pos, req="default"):
        xm, posm = _to_mx(x), _to_mx(pos, dtype=np.float32)
        past = self.cache.get(req)
        new = []
        for i, layer in enumerate(self.layers):
            pk, pv = past[i] if past else (None, None)
            xm, k, v = layer(xm, posm, pk, pv)
            new.append((k, v))
        self.cache[req] = new
        return _to_torch(xm)

    def forward_batch(self, x, positions, reqs):
        """x: (B, 1, hidden) torch tensor, one row per request id in `reqs`, each at its own position."""
        xm, posm = _to_mx(x), _to_mx(positions, dtype=np.float32)
        caches = [self.cache.get(r) for r in reqs]
        for li, layer in enumerate(self.layers):
            past = [c[li] if c else None for c in caches]
            xm, new = layer.forward_batch(xm, posm, past)
            for bi, r in enumerate(reqs):
                if caches[bi] is None:
                    caches[bi] = [None] * len(self.layers)
                    self.cache[r] = caches[bi]
                caches[bi][li] = new[bi]
        return _to_torch(xm)

    def reset(self, req):
        self.cache.pop(req, None)
