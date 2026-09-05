"""Minimal Qwen2 decoder blocks with explicit KV cache. No transformers dependency at runtime.
Used by Mac nodes (torch) and as the ONNX export source for phone nodes."""
import json
import math
import torch
import torch.nn.functional as F
import numpy as np


def load_npz(path):
    with np.load(path) as z:
        return {k: torch.from_numpy(z[k]) for k in z.files}


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
        self.ln1 = g("input_layernorm.weight")
        self.ln2 = g("post_attention_layernorm.weight")
        self.wq, self.bq = g("self_attn.q_proj.weight"), g("self_attn.q_proj.bias")
        self.wk, self.bk = g("self_attn.k_proj.weight"), g("self_attn.k_proj.bias")
        self.wv, self.bv = g("self_attn.v_proj.weight"), g("self_attn.v_proj.bias")
        self.wo = g("self_attn.o_proj.weight")
        self.wg, self.wu, self.wd = g("mlp.gate_proj.weight"), g("mlp.up_proj.weight"), g("mlp.down_proj.weight")

    def forward(self, x, pos, past_k=None, past_v=None):
        """x: (1, n, hidden) fp32. pos: (n,) int64 absolute positions.
        past_k/past_v: (1, kv_heads, p, head_dim) or None. Returns (x, k, v) with cache appended."""
        c = self.cfg
        b, n, _ = x.shape
        h = rms_norm(x, self.ln1, c.eps)
        q = F.linear(h, self.wq, self.bq).view(b, n, c.heads, c.head_dim).transpose(1, 2)
        k = F.linear(h, self.wk, self.bk).view(b, n, c.kv_heads, c.head_dim).transpose(1, 2)
        v = F.linear(h, self.wv, self.bv).view(b, n, c.kv_heads, c.head_dim).transpose(1, 2)
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
        x = x + F.linear(o, self.wo)
        h = rms_norm(x, self.ln2, c.eps)
        x = x + F.linear(F.silu(F.linear(h, self.wg)) * F.linear(h, self.wu), self.wd)
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
        q = F.linear(h, self.wq, self.bq).view(B, 1, c.heads, c.head_dim).transpose(1, 2)
        k = F.linear(h, self.wk, self.bk).view(B, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        v = F.linear(h, self.wv, self.bv).view(B, 1, c.kv_heads, c.head_dim).transpose(1, 2)

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

        x = x + F.linear(out, self.wo)
        h = rms_norm(x, self.ln2, c.eps)
        x = x + F.linear(F.silu(F.linear(h, self.wg)) * F.linear(h, self.wu), self.wd)
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
        self.embed = w["model.embed_tokens.weight"].float().to(device)
        self.norm = w["model.norm.weight"].float().to(device)
        self.lm_head = w.get("lm_head.weight", w["model.embed_tokens.weight"]).float().to(device)

    @torch.no_grad()
    def embed_tokens(self, ids):
        return self.embed[torch.as_tensor(ids, device=self.device_)][None]

    @torch.no_grad()
    def logits(self, x):
        x = rms_norm(x.to(self.device_)[:, -1], self.norm, self.cfg.eps)
        return F.linear(x, self.lm_head)[0]
