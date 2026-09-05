"""Export layers of a QUANTISED slice (dist14: Qwen2.5-14B int4) as NPU graphs the Hexagon runs.

Why this exists separately from npu/export_shard.py: that one reads fp32 .npz and bakes fp32
constants. A 14B layer is 275M parameters, so an fp32 graph is 1050 MB per layer and 49 GB for the
model, which fits neither the phone nor the disk. Two changes make 14B reachable:

  * Read the int4/int8 shards. dllm/quant.py packs int4 as uint8 nibbles with a scale per group of
    128 columns; dequant() turns a loaded npz back into fp32. The expansion is transient, one layer
    at a time, never the whole model.
  * Emit fp16 constants. The HTP computes in fp16 whatever the file says (the QNN validator lists
    FP16 among its supported datatype sets, and the measured error on the 0.5B matched fp16), so
    fp32 constants only cost file size and load time. fp16 halves the layer to 525 MB and changes
    no arithmetic the device actually performs.

Everything else — the graph shape, the host-computed RoPE/mask/one-hot-write tensors, the KV cache
as explicit I/O — is reused verbatim from npu/export_shard.py, so a layer exported here is driven by
exactly the same engine as a 0.5B layer.

  .venv-litert/bin/python npu/export14.py --dist dist14 --layers 0-1 --cache-len 512
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export_shard import NpuLayer, read_cfg, sample_kwargs, op_histogram


def load_fp32(path):
    """One layer's weights as fp32, whatever precision the shard was stored in."""
    from dllm.quant import dequant
    with np.load(path) as z:
        return dequant(z)


class Shard14(torch.nn.Module):
    """Same graph as export_shard.NpuShard, but built from already-dequantised weights."""

    def __init__(self, cfg, dist, a, b, rms="composite"):
        super().__init__()
        self.cfg, self.a, self.b = cfg, a, b
        layers = []
        for i in range(a, b):
            w = load_fp32(f"{dist}/layer_{i:02d}.npz")
            layers.append(NpuLayer(cfg, w, rms))
            del w
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden, cos, sin, mask, kv_cache_k, kv_cache_v, write=None, keep=None, input_pos=None):
        pos0 = input_pos.reshape(()) if input_pos is not None else None
        k_out, v_out = [], []
        for layer, kc, vc in zip(self.layers, kv_cache_k, kv_cache_v):
            hidden, kc, vc = layer(hidden, cos, sin, mask, kc, vc, write, keep, pos0)
            k_out.append(kc); v_out.append(vc)
        return {"hidden_out": hidden, "kv_cache_k_out": k_out, "kv_cache_v_out": v_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="dist14")
    ap.add_argument("--layers", default="0-1", help="a-b half-open; one file per layer")
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--precision", choices=["fp16", "fp32", "int8", "int4"], default="fp16")
    ap.add_argument("--out-dir", default=None, help="default: <dist>/")
    ap.add_argument("--single", action="store_true",
                    help="one graph for the whole range instead of one per layer; amortises the "
                         "per-context memory the HTP spends, at the cost of a fixed layer grouping")
    args = ap.parse_args()
    os.environ.setdefault("LITERT_TORCH_SHOW_PROGRESS", "n")
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes, quant_attrs

    a, b = (int(v) for v in args.layers.split("-"))
    cfg = read_cfg(f"{args.dist}/config.json")
    out_dir = args.out_dir or args.dist
    os.makedirs(out_dir, exist_ok=True)
    S = args.cache_len
    print(f"{args.dist}: hidden={cfg['hidden']} heads={cfg['heads']} kv={cfg['kv_heads']} "
          f"head_dim={cfg['head_dim']} inter={cfg['inter']} cache={S} precision={args.precision}")

    # One graph per layer keeps placement free: the planner can hand a node any range. --single
    # trades that for memory, packing the whole range into one QNN context.
    groups = [(a, b)] if args.single else [(i, i + 1) for i in range(a, b)]
    for lo, hi in groups:
        i = lo
        suffix = "" if args.precision == "fp16" else f".{args.precision}"
        out = (f"{out_dir}/layers_{lo:02d}-{hi:02d}{suffix}.tflite" if args.single
               else f"{out_dir}/layer_{i:02d}{suffix}.tflite")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"  layers {lo:02d}-{hi-1:02d}: exists ({os.path.getsize(out)/2**20:.0f} MB), skipping")
            continue
        t0 = time.time()
        shard = Shard14(cfg, args.dist, lo, hi).eval()
        params = sum(p.numel() for p in shard.parameters())
        kw = sample_kwargs(cfg, hi - lo, 1, S, "matmul")
        conv = litert_torch.signature("decode", shard, sample_kwargs=kw)
        # int4 is only valid dynamic-range and blockwise; BLOCKWISE_128 is the same group size
        # dllm/quant.py already uses, so the shard and the graph agree on how weights are grouped.
        qc = {"fp16": lambda: quant_recipes.full_fp16_recipe(),
              "int8": lambda: quant_recipes.full_weight_only_recipe(),
              "int4": lambda: quant_recipes.full_dynamic_recipe(
                  weight_dtype=quant_attrs.Dtype.INT4,
                  granularity=quant_attrs.Granularity.BLOCKWISE_128),
              "fp32": lambda: None}[args.precision]()
        conv.convert(quant_config=qc).export(out)
        del shard
        mb = os.path.getsize(out) / 2**20
        n_sg, hist = op_histogram(out)
        print(f"  layers {lo:02d}-{hi-1:02d}: {params/1e6:.0f}M params -> {mb:.0f} MB in {time.time()-t0:.0f}s, "
              f"{n_sg} subgraphs, {sum(hist.values())} ops", flush=True)


if __name__ == "__main__":
    main()
