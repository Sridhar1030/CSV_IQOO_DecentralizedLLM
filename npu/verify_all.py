"""Verify all 24 exported single-layer NPU graphs in dist/ against the torch reference, on the
host CPU interpreter only (no phone, no adb). Two checks:

  per-layer: each dist/layer_XX.tflite alone, fed the real hidden state that layer sees inside the
             full model (from npu/golden_all.py), through the decode signature exactly as the real
             deployment drives it -- one token at a time, including the 32-token prefill -- so the
             exported cache-update math is exercised across many sequential steps, not one vector.
  chain:     dist/layer_a.tflite .. dist/layer_{b-1}.tflite run back to back, each with its own KV
             cache, the output of one feeding the next -- the same wiring a real multi-layer node
             uses -- compared against the true torch output after those layers.

Reuses export_shard.read_cfg/host_inputs (the exact tensors the graph expects) and mirrors
export_shard.verify()'s step construction (decode-only signature, prefill looped one position at a
time), extended to report both max-abs and max-relative error and to chain several interpreters.

  .venv-litert/bin/python npu/verify_all.py --dist dist --golden npu/golden_all
  .venv-litert/bin/python npu/verify_all.py --cross-check   # sanity: matches export_shard.verify() on layer 8
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_shard import read_cfg, host_inputs, verify as reference_verify


def load_steps(golden, a, b):
    """(tag, pos, hidden-entering-layer-a, hidden-leaving-layer-{b-1}) for every step, decode order:
    the P prefill positions one at a time, then every recorded decode step. Mirrors
    export_shard.verify()'s own step construction for a decode-only signature."""
    man = json.load(open(f"{golden}/manifest.json"))
    P = man["prefill"]
    pre_in = np.load(f"{golden}/boundary_prefill_{a:02d}.npy")
    pre_out = np.load(f"{golden}/boundary_prefill_{b:02d}.npy")
    steps = [("prefill", t, pre_in[:, t:t + 1], pre_out[:, t:t + 1]) for t in range(P)]
    for st in man["steps"]:
        if st["tag"].startswith("dec"):
            tag = st["tag"]
            xin = np.load(f"{golden}/boundary_{tag}_{a:02d}.npy")
            xref = np.load(f"{golden}/boundary_{tag}_{b:02d}.npy")
            steps.append((tag, st["pos"], xin, xref))
    return man, steps


class LayerRunner:
    """One single-layer .tflite (decode signature) with its own KV cache state."""

    def __init__(self, path, cfg, S, mode):
        from ai_edge_litert import interpreter as I
        self.it = I.Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.cfg, self.S, self.mode = cfg, S, mode
        self.runner = self.it.get_signature_runner("decode")
        self._reset()

    def _reset(self):
        KV, HD = self.cfg["kv_heads"], self.cfg["head_dim"]
        self.kc = np.zeros((1, KV, HD, self.S), np.float32)
        self.vc = np.zeros((1, KV, self.S, HD), np.float32)

    def step(self, pos, xin):
        T = xin.shape[1]
        feed = {"hidden": xin.astype(np.float32), "kv_cache_k_0": self.kc, "kv_cache_v_0": self.vc}
        feed.update(host_inputs(self.cfg, pos, T, self.S, self.mode))
        out = self.runner(**feed)
        self.kc, self.vc = out["kv_cache_k_out_0"], out["kv_cache_v_out_0"]
        return out["hidden_out"]


def run_check(steps, run_fn):
    """run_fn(pos, xin) -> hidden_out. Returns (worst_abs, worst_rel, worst_step_desc, rows)."""
    worst_abs = worst_rel = 0.0
    worst_step = None
    rows = []
    for tag, pos, xin, xref in steps:
        y = run_fn(pos, xin)
        err = float(np.abs(y - xref).max())
        refmax = float(np.abs(xref).max())
        rel = err / refmax if refmax else 0.0
        rows.append((tag, pos, err, rel, refmax))
        if rel >= worst_rel:
            worst_rel = rel
            worst_step = f"{tag}@pos={pos}"
        worst_abs = max(worst_abs, err)
    return worst_abs, worst_rel, worst_step, rows


def verify_layer(i, dist, golden, cfg, S, mode):
    man, steps = load_steps(golden, i, i + 1)
    r = LayerRunner(f"{dist}/layer_{i:02d}.tflite", cfg, S, mode)
    return run_check(steps, r.step)


def verify_chain(a, b, dist, golden, cfg, S, mode):
    man, steps = load_steps(golden, a, b)
    runners = [LayerRunner(f"{dist}/layer_{i:02d}.tflite", cfg, S, mode) for i in range(a, b)]

    def run_fn(pos, xin):
        x = xin
        for r in runners:
            x = r.step(pos, x)
        return x

    return run_check(steps, run_fn)


def cross_check(dist, cfg, S, mode, golden_all="npu/golden_all"):
    """Sanity: my per-layer loop must agree with export_shard.py's own trusted verify() function
    on layer 8. export_shard.verify() reads the pre-existing npu/golden_8-9 (a single-layer golden
    set already in the repo, in golden.py's own file naming); verify_layer() reads golden_all's
    boundary files. The two golden sets were independently confirmed bit-identical (both are the
    deterministic torch reference), so this checks that the two *verification* code paths agree,
    not just that the data matches."""
    print("cross-check vs export_shard.verify() on layer 8 (npu/golden_8-9 vs npu/golden_all):")
    print("  export_shard.verify():")
    ref_worst = reference_verify(f"{dist}/layer_08.tflite", cfg, 8, 9, S, {"decode": 1}, mode, "npu/golden_8-9")
    wa, wr, ws, _ = verify_layer(8, dist, golden_all, cfg, S, mode)
    print(f"  verify_all.verify_layer(): worst_rel={wr:.3e} at {ws}  (max_abs={wa:.3e})")
    print(f"  export_shard.verify() worst_rel={ref_worst:.3e}")
    agree = abs(ref_worst - wr) < 1e-9 or (ref_worst > 0 and abs(ref_worst - wr) / ref_worst < 1e-3)
    print(f"  agreement: {'OK, numbers match' if agree else 'MISMATCH -- reimplementation disagrees with the trusted verify()!'}")
    print()
    return agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="dist")
    ap.add_argument("--golden", default="npu/golden_all")
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--cache-write", choices=["matmul", "dus"], default="matmul")
    ap.add_argument("--red-flag-rel", type=float, default=1e-5, help="CPU-interpreter relative error above this is called out")
    ap.add_argument("--cross-check", action="store_true", help="only run the sanity cross-check against export_shard.verify()")
    args = ap.parse_args()

    cfg = read_cfg(f"{args.dist}/config.json")
    S, mode = args.cache_len, args.cache_write
    n_layers = cfg["n_layers"]

    if args.cross_check:
        cross_check(args.dist, cfg, S, mode, args.golden)
        return

    ok = cross_check(args.dist, cfg, S, mode, args.golden)
    if not ok:
        print("!! aborting the sweep: the verification harness itself disagrees with the trusted export_shard.verify(). Fix the harness before trusting anything below.")
        sys.exit(2)

    man = json.load(open(f"{args.golden}/manifest.json"))
    print(f"golden: prompt {man['prompt']!r}, prefill {man['prefill']}, {len(man['steps'])} recorded steps, "
          f"generated {man.get('generated')} -> {man.get('text')!r}")
    print(f"cache_len={S} cache_write={mode} n_layers={n_layers}\n")

    print("=" * 88)
    print("PER-LAYER: each dist/layer_XX.tflite alone vs torch reference for that same layer")
    print("=" * 88)
    t0 = time.time()
    results = []
    for i in range(n_layers):
        wa, wr, ws, rows = verify_layer(i, args.dist, args.golden, cfg, S, mode)
        flag = "  <-- RED FLAG" if wr > args.red_flag_rel else ""
        print(f"layer {i:2d}  max|err|={wa:10.3e}  max_rel={wr:10.3e}  worst_step={ws:<14}{flag}")
        results.append((i, wa, wr, ws))
    print(f"({time.time() - t0:.1f}s for {n_layers} layers x {len(load_steps(args.golden, 0, 1)[1])} steps each)\n")

    worst_layer = max(results, key=lambda r: r[2])
    print(f"WORST LAYER: layer {worst_layer[0]:2d}  max|err|={worst_layer[1]:.3e}  max_rel={worst_layer[2]:.3e}  at {worst_layer[3]}\n")

    print("=" * 88)
    print("CHAIN: layer_XX.tflite files run back-to-back (own KV cache each) vs torch over the same range")
    print("=" * 88)
    chain_ranges = [(0, 8), (8, 16), (16, 24), (0, 24)]
    chain_results = []
    for a, b in chain_ranges:
        t0 = time.time()
        wa, wr, ws, rows = verify_chain(a, b, args.dist, args.golden, cfg, S, mode)
        flag = "  <-- RED FLAG" if wr > args.red_flag_rel else ""
        print(f"chain [{a:2d},{b:2d})  {b - a:2d} layers  max|err|={wa:10.3e}  max_rel={wr:10.3e}  worst_step={ws:<14}{flag}  ({time.time() - t0:.1f}s)")
        chain_results.append((a, b, wa, wr, ws))
    print()

    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    any_red = any(r[2] > args.red_flag_rel for r in results) or any(c[3] > args.red_flag_rel for c in chain_results)
    any_mismatch = any(r[2] > 1e-4 for r in results) or any(c[3] > 1e-4 for c in chain_results)
    if any_mismatch:
        print("MISMATCH: at least one layer or chain exceeds the 1e-4 relative-error threshold export_shard.py itself uses for OK/MISMATCH.")
    elif any_red:
        print(f"All layers/chains are under 1e-4 relative, but at least one exceeds the {args.red_flag_rel:.0e} red-flag line "
              f"the task called out as worth investigating on a CPU interpreter (reference: a single layer verified at ~2e-6 when this was built).")
    else:
        print("All 24 layers and all chains are under the red-flag line. Graphs look numerically faithful on the CPU interpreter.")


if __name__ == "__main__":
    main()
