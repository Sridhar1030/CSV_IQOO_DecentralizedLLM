"""Export layers [a, b) of the sliced model as one .tflite the Hexagon HTP runs whole: hidden in, hidden out,
KV cache as explicit tensors. The same weights as dllm/np_node.py, straight from dist/layer_XX.npz.

Why the graph looks the way it does (every choice below was forced by a failure on the phone):
  * Static shapes only. The cache is a fixed buffer of S positions; a growing cache cannot be converted.
  * No integer arithmetic in the graph. The earlier shard wrote its cache with index_copy, which lowered to
    int64 Less/Select/StablehloScatter; the HTP rejected those ops and the runtime segfaulted while
    re-serialising the partially compiled model (how_torun_on_npu.md §7). Here the position enters as
    floats the host computes: RoPE cos/sin, the additive causal mask, and a one-hot write matrix. The cache
    update is then cache * keep + new @ write, two ops the HTP has always supported. --cache-write dus uses
    the TFLite DYNAMIC_UPDATE_SLICE builtin with an int32 position instead; both are exported for comparison.
  * Grouped-query attention without repeats. The 14 query heads are viewed as [2 KV groups, 7 heads * T tokens],
    so attention is two batch-matmuls against the [1, 2, ·, ·] cache with matching batch dims: no gather, no
    broadcast, no rank-5 tensor. The mask is tiled 7x on the host to match.
  * K is stored time-last, [1, KV, HD, S], V time-major, [1, KV, S, HD], so neither needs a transpose in the graph.
  * RMSNorm is the odml.rms_norm composite, which the Qualcomm plugin maps to the QNN RmsNorm op. The plain
    x*x -> mean form squares the residual stream, which reaches 1710 on this model; 1710^2 overflows fp16.
  * fp16 is what the HTP computes in. The census in npu/golden.py showed every intermediate under 3400, so no
    rescaling is needed, but the residual stream at 1700 has an fp16 ulp of 1.0: expect ~1e-3 relative error.

Signature I/O, the names an engine binds to:
    in   hidden [1,T,896]  cos,sin [1,T,1,32]  mask [1,1,7T,S]  write [1,2,T,S]  keep [1,1,1,S]
         kv_cache_k_i [1,2,64,S]  kv_cache_v_i [1,2,S,64]          (or input_pos [1] int32 with --cache-write dus)
    out  hidden_out  kv_cache_k_out_i  kv_cache_v_out_i

  .venv-litert/bin/python npu/export_shard.py --layers 8-16 --cache-len 512 --sigs decode,prefill32 --verify npu/golden
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

MASK_FILL = -1e4       # finite: -inf and -1e30 do not survive fp16, and exp(-1e4) is exactly 0 in every format


def read_cfg(path):
    d = json.load(open(path))
    return dict(hidden=d["hidden_size"], heads=d["num_attention_heads"], kv_heads=d["num_key_value_heads"],
                head_dim=d["hidden_size"] // d["num_attention_heads"], inter=d["intermediate_size"],
                eps=d["rms_norm_eps"], theta=d["rope_theta"], n_layers=d["num_hidden_layers"])


# ---------- host side: what the caller computes per step, in numpy, mirrored later in the engine ----------

def rope_tables(cfg, pos, T):
    """cos, sin of shape [1,T,1,HD/2] for absolute positions pos..pos+T-1; the same angles as dllm.model.rope."""
    d = cfg["head_dim"]
    inv = 1.0 / (np.float32(cfg["theta"]) ** (np.arange(0, d, 2, dtype=np.float32) / np.float32(d)))
    f = (np.arange(pos, pos + T, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)
    return np.cos(f)[None, :, None, :].astype(np.float32), np.sin(f)[None, :, None, :].astype(np.float32)


def causal_mask(cfg, pos, T, S):
    """[1,1,G*T,S] additive mask: row g*T+t may see cache slots 0..pos+t. Tiled over the G heads of a KV group."""
    G = cfg["heads"] // cfg["kv_heads"]
    slots = np.arange(S)[None, :]
    allowed = slots <= (pos + np.arange(T))[:, None]                      # [T, S]
    m = np.where(allowed, np.float32(0), np.float32(MASK_FILL)).astype(np.float32)
    return np.tile(m, (G, 1))[None, None]


def write_matrices(cfg, pos, T, S):
    """write [1,KV,T,S] one-hot (token t -> slot pos+t) and keep [1,1,1,S] (0 at the written slots, 1 elsewhere)."""
    w = np.zeros((T, S), np.float32)
    w[np.arange(T), pos + np.arange(T)] = 1.0
    keep = (1.0 - w.sum(0)).astype(np.float32)
    return np.broadcast_to(w, (1, cfg["kv_heads"], T, S)).copy(), keep[None, None, None, :].copy()


def host_inputs(cfg, pos, T, S, mode):
    cos, sin = rope_tables(cfg, pos, T)
    d = {"cos": cos, "sin": sin, "mask": causal_mask(cfg, pos, T, S)}
    if mode == "dus":
        d["input_pos"] = np.array([pos], np.int32)
    else:
        d["write"], d["keep"] = write_matrices(cfg, pos, T, S)
    return d


# ---------- the graph ----------

def rope_half(x, cos, sin):
    """x [1,T,heads,HD], cos/sin [1,T,1,HD/2]. Rotate-half, the form dllm.model.rope computes."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class NpuLayer(torch.nn.Module):
    def __init__(self, cfg, w, rms):
        super().__init__()
        self.c, self.rms = cfg, rms
        P = lambda k: torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(w[k], dtype=np.float32)), requires_grad=False)
        self.ln1, self.ln2 = P("input_layernorm.weight"), P("post_attention_layernorm.weight")
        self.wq, self.bq = P("self_attn.q_proj.weight"), P("self_attn.q_proj.bias")
        self.wk, self.bk = P("self_attn.k_proj.weight"), P("self_attn.k_proj.bias")
        self.wv, self.bv = P("self_attn.v_proj.weight"), P("self_attn.v_proj.bias")
        self.wo = P("self_attn.o_proj.weight")
        self.wg, self.wu, self.wd = P("mlp.gate_proj.weight"), P("mlp.up_proj.weight"), P("mlp.down_proj.weight")

    def norm(self, x, w):
        eps = self.c["eps"]
        if self.rms == "composite":
            from litert_torch.generative.layers.normalization import rms_norm_with_hlfb
            return rms_norm_with_hlfb(x, w, eps, 1.0)
        if self.rms == "scaled":       # keeps x*x under fp16's ceiling: (1710/16)^2 = 11424
            c = 1.0 / 16
            xs = x * c
            return x * (torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps * c * c) * c) * w
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def forward(self, x, cos, sin, mask, kc, vc, write, keep, pos0):
        c = self.c
        H, KV, HD = c["heads"], c["kv_heads"], c["head_dim"]
        G = H // KV
        T = x.shape[1]
        S = kc.shape[-1]
        h = self.norm(x, self.ln1)
        q = rope_half(F.linear(h, self.wq, self.bq).view(1, T, H, HD), cos, sin)
        k = rope_half(F.linear(h, self.wk, self.bk).view(1, T, KV, HD), cos, sin)
        v = F.linear(h, self.wv, self.bv).view(1, T, KV, HD)
        q = q.permute(0, 2, 1, 3).reshape(1, KV, G * T, HD)      # heads are group-major: head = g*G + j
        k = k.permute(0, 2, 3, 1)                                 # [1,KV,HD,T]  time last, like the cache
        v = v.permute(0, 2, 1, 3)                                 # [1,KV,T,HD]
        if pos0 is not None:
            from litert_torch.generative.custom_ops import dynamic_update_slice as tfl_dus
            z = torch.zeros((), dtype=torch.int32)
            kc = tfl_dus.dynamic_update_slice(kc, k, [z, z, z, pos0])
            vc = tfl_dus.dynamic_update_slice(vc, v, [z, z, pos0, z])
        else:
            kc = kc * keep + k @ write                            # [1,KV,HD,S]
            vc = vc * keep.reshape(1, 1, S, 1) + write.transpose(-1, -2) @ v
        att = torch.softmax((q @ kc) * (1.0 / math.sqrt(HD)) + mask, -1)   # [1,KV,G*T,S]
        o = (att @ vc).reshape(1, H, T, HD).permute(0, 2, 1, 3).reshape(1, T, c["hidden"])
        x = x + F.linear(o, self.wo)
        h = self.norm(x, self.ln2)
        x = x + F.linear(F.silu(F.linear(h, self.wg)) * F.linear(h, self.wu), self.wd)
        return x, kc, vc


class NpuShard(torch.nn.Module):
    def __init__(self, cfg, dist, a, b, rms="composite"):
        super().__init__()
        self.cfg, self.a, self.b = cfg, a, b
        layers = []
        for i in range(a, b):
            with np.load(f"{dist}/layer_{i:02d}.npz") as z:
                layers.append(NpuLayer(cfg, {k: z[k] for k in z.files}, rms))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden, cos, sin, mask, kv_cache_k, kv_cache_v, write=None, keep=None, input_pos=None):
        pos0 = input_pos.reshape(()) if input_pos is not None else None
        k_out, v_out = [], []
        for layer, kc, vc in zip(self.layers, kv_cache_k, kv_cache_v):
            hidden, kc, vc = layer(hidden, cos, sin, mask, kc, vc, write, keep, pos0)
            k_out.append(kc); v_out.append(vc)
        return {"hidden_out": hidden, "kv_cache_k_out": k_out, "kv_cache_v_out": v_out}


def sample_kwargs(cfg, n_layers, T, S, mode, pos=0, hidden=None):
    KV, HD = cfg["kv_heads"], cfg["head_dim"]
    kw = {"hidden": torch.zeros(1, T, cfg["hidden"]) if hidden is None else torch.as_tensor(hidden)}
    kw.update({k: torch.from_numpy(v) for k, v in host_inputs(cfg, pos, T, S, mode).items()})
    kw["kv_cache_k"] = [torch.zeros(1, KV, HD, S) for _ in range(n_layers)]
    kw["kv_cache_v"] = [torch.zeros(1, KV, S, HD) for _ in range(n_layers)]
    return kw


def flatten(kw):
    """Signature tensor names: lists become name_i, the convention litert_torch uses for kwargs."""
    out = {}
    for k, v in kw.items():
        if isinstance(v, list):
            for i, t in enumerate(v):
                out[f"{k}_{i}"] = t
        else:
            out[k] = v
    return {k: (v.detach().numpy() if torch.is_tensor(v) else v) for k, v in out.items()}


def parse_sigs(spec):
    sigs = {}
    for s in spec.split(","):
        s = s.strip()
        if s == "decode":
            sigs["decode"] = 1
        elif s.startswith("prefill"):
            sigs[s] = int(s[len("prefill"):])
        else:
            raise SystemExit(f"unknown signature {s!r}; use decode or prefillN")
    return sigs


def op_histogram(path):
    from ai_edge_litert import schema_py_generated as schema
    buf = bytearray(open(path, "rb").read())
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    hist = {}
    for sg in model.subgraphs:
        for op in sg.operators:
            code = model.operatorCodes[op.opcodeIndex]
            n = names.get(code.builtinCode, "?")
            if n == "CUSTOM":
                n = "CUSTOM:" + (code.customCode.decode() if isinstance(code.customCode, bytes) else str(code.customCode))
            hist[n] = hist.get(n, 0) + 1
    return len(model.subgraphs), dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def verify(path, cfg, a, b, S, sigs, mode, golden):
    """Run the golden prompt through the .tflite on the Mac's CPU (fp32) and diff against the torch reference.
    Decode-only: the 32 prefill rows go through one at a time, since causal rows are independent of what follows."""
    from ai_edge_litert import interpreter as I
    man = json.load(open(f"{golden}/manifest.json"))
    assert man["layers"] == [a, b], f"golden is for layers {man['layers']}, model is {a}-{b}"
    it = I.Interpreter(model_path=path); it.allocate_tensors()
    n = b - a
    KV, HD = cfg["kv_heads"], cfg["head_dim"]
    kc = [np.zeros((1, KV, HD, S), np.float32) for _ in range(n)]
    vc = [np.zeros((1, KV, S, HD), np.float32) for _ in range(n)]
    prefill_sig = next((s for s, T in sigs.items() if T > 1), None)
    P = man["prefill"]
    steps = []
    x_in, x_ref = np.load(f"{golden}/prefill_in.npy"), np.load(f"{golden}/prefill_out.npy")
    if prefill_sig and sigs[prefill_sig] == P:
        steps.append((prefill_sig, 0, x_in, x_ref))
    else:
        steps += [("decode", t, x_in[:, t:t + 1], x_ref[:, t:t + 1]) for t in range(P)]
    for st in man["steps"]:
        if st["tag"].startswith("dec"):
            steps.append(("decode", st["pos"], np.load(f"{golden}/{st['tag']}_in.npy"), np.load(f"{golden}/{st['tag']}_out.npy")))
    worst = 0.0
    for sig, pos, xin, xref in steps:
        T = xin.shape[1]
        feed = {"hidden": xin.astype(np.float32)}
        feed.update(host_inputs(cfg, pos, T, S, mode))
        for i in range(n):
            feed[f"kv_cache_k_{i}"], feed[f"kv_cache_v_{i}"] = kc[i], vc[i]
        out = it.get_signature_runner(sig)(**feed)
        for i in range(n):
            kc[i], vc[i] = out[f"kv_cache_k_out_{i}"], out[f"kv_cache_v_out_{i}"]
        y = out["hidden_out"]
        err = float(np.abs(y - xref).max()); rel = err / float(np.abs(xref).max())
        worst = max(worst, rel)
        print(f"    {sig:<10} pos={pos:<3} n={T:<3} max|err|={err:.3e} rel={rel:.2e} |ref|max={np.abs(xref).max():.1f}")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", default="dist")
    ap.add_argument("--layers", default="8-16", help="a-b half-open")
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--sigs", default="decode", help="comma list: decode, prefillN (N tokens)")
    ap.add_argument("--cache-write", choices=["matmul", "dus"], default="matmul")
    ap.add_argument("--rms", choices=["composite", "scaled", "plain"], default="composite")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", default=None, help="golden dir from npu/golden.py for the same layers")
    args = ap.parse_args()
    os.environ.setdefault("LITERT_TORCH_SHOW_PROGRESS", "n")
    import litert_torch

    a, b = (int(v) for v in args.layers.split("-"))
    cfg = read_cfg(f"{args.dist}/config.json")
    S = args.cache_len
    sigs = parse_sigs(args.sigs)
    out = args.out or f"npu/out/qwen05_L{a}-{b}_S{S}_{args.cache_write}_{args.rms}_{'-'.join(sigs)}.tflite"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    shard = NpuShard(cfg, args.dist, a, b, args.rms).eval()
    print(f"layers {a}-{b - 1}: {sum(p.numel() for p in shard.parameters()) / 1e6:.1f} M params, cache {S} slots, "
          f"write={args.cache_write} rms={args.rms} sigs={sigs}")
    conv = None
    for name, T in sigs.items():
        kw = sample_kwargs(cfg, b - a, T, S, args.cache_write)
        conv = litert_torch.signature(name, shard, sample_kwargs=kw) if conv is None else conv.signature(name, shard, sample_kwargs=kw)
    conv.convert().export(out)
    n_sg, hist = op_histogram(out)
    print(f"wrote {out}: {os.path.getsize(out) / 2**20:.1f} MB in {time.time() - t0:.0f}s, {n_sg} subgraph(s)")
    print(f"  ops: {hist}")
    if args.verify:
        print("  CPU interpreter vs torch reference on the golden prompt:")
        worst = verify(out, cfg, a, b, S, sigs, args.cache_write, args.verify)
        print(f"  worst relative error {worst:.2e} -> {'OK' if worst < 1e-4 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
