"""Minimal Qwen2 decoder blocks with explicit KV cache. No transformers dependency at runtime.
Used by Mac nodes (torch) and as the ONNX export source for phone nodes."""
import json
import math
import torch
import torch.nn.functional as F
import numpy as np

from dllm.quant import unpack4


def load_npz(path):
    with np.load(path) as z:
        return {k: torch.from_numpy(z[k]) for k in z.files}


def _q(w, key):
    """(weight, scales) for `key`, whichever way the slicer wrote it.

    fp32 shards give (fp32, None). int8 shards (int8, one scale per row) are used as they are.
    int4 shards are unpacked and re-quantised to int8 here, at load: the fast kernel takes one
    scale per row, and measuring says a row scale reproduces int4's grouped codes exactly well
    enough that the layer error does not move. So int4 buys a smaller download and a smaller
    file for a phone to hold, and a torch node still runs the int8 path."""
    t = w[key]
    if t.dtype == torch.uint8:                                   # int4, two codes per byte
        fp = torch.from_numpy(unpack4(t.numpy(), w[f"{key}.scale"].numpy()))
        s = fp.abs().amax(1) / 127.0
        s[s == 0] = 1.0
        return (fp / s[:, None]).round().clamp(-127, 127).to(torch.int8), s
    if t.dtype == torch.int8:
        return t, w[f"{key}.scale"].float()
    return t.float(), None


class QLinear(torch.nn.Module):
    """F.linear over an int8 weight with per-output-row scales, or a plain fp32 one.

    Weight-only quantisation: activations stay fp32, so nothing upstream changes. int8 is not
    only a quarter of the bytes, it is also faster here, because a decode step is bound by how
    fast the weights can be read, not by the arithmetic over them."""

    def __init__(self, w, scale, bias):
        super().__init__()
        self.register_buffer("weight", w, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, x):
        if self.scale is None:
            return F.linear(x, self.weight, self.bias)
        y = torch.ops.aten._weight_int8pack_mm(x.reshape(-1, x.shape[-1]).contiguous(), self.weight, self.scale)
        y = y.view(*x.shape[:-1], -1)
        return y if self.bias is None else y + self.bias


class Cfg:
    def __init__(self, d):
        self.hidden = d["hidden_size"]
        self.heads = d["num_attention_heads"]
        self.kv_heads = d["num_key_value_heads"]
        self.head_dim = self.hidden // self.heads
        self.inter = d["intermediate_size"]
        self.eps = d["rms_norm_eps"]
        self.theta = d["rope_theta"]
        self.n_layers = d["num_hidden_layers"]
        self.vocab = d["vocab_size"]

    @staticmethod
    def load(path):
        return Cfg(json.load(open(path)))


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, theta):
    # x: (b, h, n, d)  pos: (n,)
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    f = pos.float()[:, None] * inv[None, :]            # (n, d/2)
    emb = torch.cat([f, f], -1)                          # (n, d)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


def sample(lg, temperature, top_p, top_k, seed=None):
    """temperature 0 means greedy, which is what the verification scripts rely on.
    top_k and top_p are applied in that order, matching what OpenAI clients expect. Ignoring them
    silently would make this endpoint quietly disagree with any caller that sets them."""
    if temperature <= 0:
        return int(lg.argmax())
    lg = lg / temperature
    if top_k:
        kth = lg.topk(min(top_k, lg.numel())).values[-1]
        lg = lg.masked_fill(lg < kth, float("-inf"))
    probs = torch.softmax(lg, -1)
    if top_p and top_p < 1.0:
        srt, idx = probs.sort(descending=True)
        keep = (srt.cumsum(-1) - srt) < top_p          # keeps the token that crosses the mass too
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[idx[keep]] = True
        probs = probs * mask
        probs = probs / probs.sum()
    g = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    return int(torch.multinomial(probs, 1, generator=g))


class Layer(torch.nn.Module):
    def __init__(self, cfg: Cfg, w: dict):
        super().__init__()
        self.cfg = cfg
        g = lambda k: torch.nn.Parameter(w[k].float(), requires_grad=False)
        lin = lambda k: QLinear(*_q(w, f"{k}.weight"), w[f"{k}.bias"].float() if f"{k}.bias" in w else None)
        self.ln1 = g("input_layernorm.weight")
        self.ln2 = g("post_attention_layernorm.weight")
        self.q, self.k, self.v = lin("self_attn.q_proj"), lin("self_attn.k_proj"), lin("self_attn.v_proj")
        self.o = lin("self_attn.o_proj")
        self.gate, self.up, self.down = lin("mlp.gate_proj"), lin("mlp.up_proj"), lin("mlp.down_proj")

    def forward(self, x, pos, past_k=None, past_v=None):
        """x: (1, n, hidden) fp32. pos: (n,) int64 absolute positions.
        past_k/past_v: (1, kv_heads, p, head_dim) or None. Returns (x, k, v) with cache appended."""
        c = self.cfg
        b, n, _ = x.shape
        h = rms_norm(x, self.ln1, c.eps)
        q = self.q(h).view(b, n, c.heads, c.head_dim).transpose(1, 2)
        k = self.k(h).view(b, n, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(h).view(b, n, c.kv_heads, c.head_dim).transpose(1, 2)
        q, k = rope(q, pos, c.theta), rope(k, pos, c.theta)
        if past_k is not None:
            k, v = torch.cat([past_k, k], 2), torch.cat([past_v, v], 2)
        rep = c.heads // c.kv_heads
        kk = k.repeat_interleave(rep, 1)
        vv = v.repeat_interleave(rep, 1)
        p = k.shape[2] - n
        mask = torch.ones(n, p + n, dtype=torch.bool, device=x.device).tril(diagonal=p)
        att = (q @ kk.transpose(-1, -2)) / math.sqrt(c.head_dim)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        o = (att @ vv).transpose(1, 2).reshape(b, n, c.hidden)
        x = x + self.o(o)
        h = rms_norm(x, self.ln2, c.eps)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x, k, v

    def forward_batch(self, x, positions, past):
        """One decode step for several independent requests at once.

        x: (B, 1, hidden), one row per request. positions: (B,) absolute position of each row.
        past: list of B (k, v) pairs or None, each that request's own cache.

        Why this is worth doing: every matmul below reads the same weights whichever B is used, and
        at B=1 a decode step reads 57 MB of weights to do 30 MFLOP of arithmetic. Batching pays for
        that traffic once and gets B times the work out of it. Attention is the exception, because
        each row attends over a different history, so it is looped. Attention is a small share of
        both the weights and the time, so the loop costs little.
        """
        c = self.cfg
        B = x.shape[0]
        h = rms_norm(x, self.ln1, c.eps)                                   # (B, 1, H)
        q = self.q(h).view(B, 1, c.heads, c.head_dim).transpose(1, 2)
        k = self.k(h).view(B, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(h).view(B, 1, c.kv_heads, c.head_dim).transpose(1, 2)

        rep_ = c.heads // c.kv_heads
        out = x.new_empty(B, 1, c.hidden)
        new_cache = []
        for i in range(B):
            pos_i = positions[i:i + 1]                                      # (1,)
            qi = rope(q[i:i + 1], pos_i, c.theta)
            ki = rope(k[i:i + 1], pos_i, c.theta)
            vi = v[i:i + 1]
            if past is not None and past[i] is not None:
                pk, pv = past[i]
                ki, vi = torch.cat([pk, ki], 2), torch.cat([pv, vi], 2)
            new_cache.append((ki, vi))
            att = (qi @ ki.repeat_interleave(rep_, 1).transpose(-1, -2)) / math.sqrt(c.head_dim)
            att = att.softmax(-1)                                           # one query, all keys: no mask needed
            out[i] = (att @ vi.repeat_interleave(rep_, 1)).transpose(1, 2).reshape(1, 1, c.hidden)

        x = x + self.o(out)
        h = rms_norm(x, self.ln2, c.eps)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x, new_cache


class Shard(torch.nn.Module):
    """Contiguous layer range [a, b). Holds its own KV cache per request id."""

    def __init__(self, cfg: Cfg, shard_dir, a, b, device="cpu"):
        super().__init__()
        self.cfg, self.a, self.b, self.device_ = cfg, a, b, device
        self.layers = torch.nn.ModuleList()
        for i in range(a, b):
            self.layers.append(Layer(cfg, load_npz(f"{shard_dir}/layer_{i:02d}.npz")))
        self.to(device)
        self.cache = {}  # req -> list[(k, v)]

    @torch.no_grad()
    def forward(self, x, pos, req="default"):
        x, pos = x.to(self.device_), pos.to(self.device_)
        past = self.cache.get(req)
        new = []
        for i, layer in enumerate(self.layers):
            pk, pv = past[i] if past else (None, None)
            x, k, v = layer(x, pos, pk, pv)
            new.append((k, v))
        self.cache[req] = new
        return x

    @torch.no_grad()
    def forward_batch(self, x, positions, reqs):
        """x: (B, 1, hidden), one row per request id in `reqs`, each at its own position."""
        x = x.to(self.device_)
        positions = torch.as_tensor(positions, device=self.device_)
        caches = [self.cache.get(r) for r in reqs]
        for li, layer in enumerate(self.layers):
            past = [c[li] if c else None for c in caches]
            x, new = layer.forward_batch(x, positions, past)
            for bi, r in enumerate(reqs):
                if caches[bi] is None:
                    caches[bi] = [None] * len(self.layers)
                    self.cache[r] = caches[bi]
                caches[bi][li] = new[bi]
        return x

    def reset(self, req):
        self.cache.pop(req, None)


class Head(torch.nn.Module):
    """Embedding + final norm + lm_head (tied). Lives on the coordinator."""

    def __init__(self, cfg: Cfg, shard_dir, device="cpu"):
        super().__init__()
        w = load_npz(f"{shard_dir}/head.npz")
        self.cfg, self.device_ = cfg, device
        e, es = _q(w, "model.embed_tokens.weight")
        self.embed = e.to(device)
        self.embed_scale = None if es is None else es.to(device)
        self.norm = w["model.norm.weight"].float().to(device)
        hk = "lm_head.weight" if "lm_head.weight" in w else "model.embed_tokens.weight"
        self.lm_head = QLinear(*_q(w, hk), None).to(device)

    @torch.no_grad()
    def embed_tokens(self, ids):
        ids = torch.as_tensor(ids, device=self.device_)
        rows = self.embed[ids]
        if self.embed_scale is None:
            return rows[None]
        return (rows.float() * self.embed_scale[ids][:, None])[None]

    @torch.no_grad()
    def logits(self, x):
        x = rms_norm(x.to(self.device_)[:, -1], self.norm, self.cfg.eps)
        return self.lm_head(x)[0]
