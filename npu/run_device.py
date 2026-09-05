"""Run an exported shard on the attached phone's Hexagon NPU, step by step, against the golden prompt.

Each step pushes the real inputs (hidden state, RoPE tables, mask, write matrix, the KV cache carried from
the previous step) as .raw files, runs LiteRT's run_model with the Qualcomm dispatch + compiler plugin, pulls
the outputs back and diffs hidden_out against the torch reference in the golden dir. The cache never touches
the Mac's arithmetic: what the NPU wrote is what the next step reads, so 37 steps in a row is the real test.

  .venv/bin/python npu/run_device.py --model npu/out/qwen05_L8-16_S512_matmul_composite_decode.tflite --golden npu/golden
  .venv/bin/python npu/run_device.py --model ... --accelerator cpu       # same binary on the phone's CPU, for the comparison
"""
import argparse, json, os, re, shutil, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_shard import read_cfg, host_inputs

DEV = "/data/local/tmp/npu"
ENV = f"LD_LIBRARY_PATH={DEV} ADSP_LIBRARY_PATH={DEV}"


def sh(cmd, check=True, capture=True):
    r = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stdout}\n{r.stderr}")
    return r


def push_model(path):
    name = os.path.basename(path)
    size = os.path.getsize(path)
    have = sh(f"adb shell stat -c %s {DEV}/{name} 2>/dev/null", check=False).stdout.strip()
    if have != str(size):
        t = time.time(); sh(f"adb push {path} {DEV}/{name}"); print(f"pushed {name} ({size / 2**20:.0f} MB) in {time.time() - t:.0f}s")
    return f"{DEV}/{name}"


def parse_model(model_path, cfg):
    m = re.search(r"_L(\d+)-(\d+)_S(\d+)_(matmul|dus)_", os.path.basename(model_path))
    a, b, S, mode = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    sigs = os.path.basename(model_path).rsplit("_", 1)[1].replace(".tflite", "").split("-")
    return a, b, S, mode, sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", default="npu/golden")
    ap.add_argument("--dist", default="dist")
    ap.add_argument("--accelerator", default="npu", choices=["npu", "cpu", "gpu"])
    ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (0 = all)")
    ap.add_argument("--keep-cache", action="store_true", help="reuse the phone's compiler cache from an earlier run")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    cfg = read_cfg(f"{args.dist}/config.json")
    a, b, S, mode, sigs = parse_model(args.model, cfg)
    n, KV, HD = b - a, cfg["kv_heads"], cfg["head_dim"]
    man = json.load(open(f"{args.golden}/manifest.json"))
    assert man["layers"] == [a, b], f"golden {man['layers']} vs model {a}-{b}"
    dev_model = push_model(args.model)
    tag = os.path.basename(args.model).replace(".tflite", "") + f"_{args.accelerator}"
    log = open(args.log or f"npu/out/{tag}.device.log", "w")
    local_in, local_out = "npu/out/_in", "npu/out/_out"
    if not args.keep_cache:
        sh(f"adb shell rm -rf {DEV}/cache; adb shell mkdir -p {DEV}/cache")
    sh(f"adb shell 'rm -rf {DEV}/in {DEV}/out; mkdir -p {DEV}/in {DEV}/out'")

    P = man["prefill"]
    x_in, x_ref = np.load(f"{args.golden}/prefill_in.npy"), np.load(f"{args.golden}/prefill_out.npy")
    pre = next((s for s in sigs if s.startswith("prefill") and int(s[7:]) == P), None)
    steps = [(pre, 0, x_in, x_ref)] if pre else [("decode", t, x_in[:, t:t + 1], x_ref[:, t:t + 1]) for t in range(P)]
    for st in man["steps"]:
        if st["tag"].startswith("dec"):
            steps.append(("decode", st["pos"], np.load(f"{args.golden}/{st['tag']}_in.npy"), np.load(f"{args.golden}/{st['tag']}_out.npy")))
    if args.steps:
        steps = steps[:args.steps]

    kc = [np.zeros((1, KV, HD, S), np.float32) for _ in range(n)]
    vc = [np.zeros((1, KV, S, HD), np.float32) for _ in range(n)]
    worst, times = 0.0, []
    print(f"{tag}: layers {a}-{b - 1}, {len(steps)} steps on the phone's {args.accelerator}")
    for si, (sig, pos, xin, xref) in enumerate(steps):
        T = xin.shape[1]
        shutil.rmtree(local_in, ignore_errors=True); os.makedirs(local_in); shutil.rmtree(local_out, ignore_errors=True); os.makedirs(local_out)
        feed = {"hidden": xin.astype(np.float32)}
        feed.update(host_inputs(cfg, pos, T, S, mode))
        for i in range(n):
            feed[f"kv_cache_k_{i}"], feed[f"kv_cache_v_{i}"] = kc[i], vc[i]
        for k, v in feed.items():
            np.ascontiguousarray(v).tofile(f"{local_in}/{k}.raw")
        sh(f"adb push {local_in}/. {DEV}/in/ > /dev/null")
        acc = {"npu": "npu", "cpu": "cpu", "gpu": "gpu"}[args.accelerator]
        cmd = (f"adb shell 'cd {DEV} && {ENV} ./run_model --graph={dev_model} --accelerator={acc} "
               f"--dispatch_library_dir={DEV} --compiler_plugin_library_dir={DEV} --compiler_cache_dir={DEV}/cache "
               f"--signature_index={sigs.index(sig)} --input_dir={DEV}/in --output_dir={DEV}/out --iterations=1 "
               f"--qualcomm_log_level=warn 2>&1'")
        t0 = time.time()
        r = sh(cmd, check=False)
        dt = time.time() - t0
        log.write(f"\n===== step {si} {sig} pos={pos} =====\n{r.stdout}\n")
        log.flush()
        part = re.findall(r"Partitioned subgraph<\d+>, selected (\d+) ops, from a total of (\d+) ops", r.stdout)
        if si == 0 and part:
            print(f"  partition: {part[0][0]}/{part[0][1]} ops on the {args.accelerator}" + ("" if part[0][0] == part[0][1] else "  <-- NOT fully delegated"))
        if r.returncode != 0 or "Segmentation" in r.stdout or "Aborted" in r.stdout:
            print(f"  step {si} FAILED (exit {r.returncode}); tail of log:\n" + "\n".join(r.stdout.strip().splitlines()[-15:]))
            return 1
        sh(f"adb pull {DEV}/out/. {local_out}/ > /dev/null")
        y = np.fromfile(f"{local_out}/hidden_out.raw", np.float32).reshape(xref.shape)
        for i in range(n):
            kc[i] = np.fromfile(f"{local_out}/kv_cache_k_out_{i}.raw", np.float32).reshape(1, KV, HD, S)
            vc[i] = np.fromfile(f"{local_out}/kv_cache_v_out_{i}.raw", np.float32).reshape(1, KV, S, HD)
        err = float(np.abs(y - xref).max()); rel = err / float(np.abs(xref).max())
        cos = float((y * xref).sum() / (np.linalg.norm(y) * np.linalg.norm(xref) + 1e-12))
        worst = max(worst, rel)
        run_ms = re.findall(r"[Rr]un(?:ning)? (?:model )?(?:took|time)[^\d]*([\d.]+) ?ms", r.stdout)
        times.append(dt)
        print(f"  {sig:<10} pos={pos:<3} n={T:<3} max|err|={err:.3e} rel={rel:.2e} cos={cos:.6f} |ref|max={np.abs(xref).max():.1f}  wall {dt:.1f}s" + (f" run {run_ms[-1]} ms" if run_ms else ""))
    print(f"worst relative error over {len(steps)} steps: {worst:.2e}; first step wall {times[0]:.1f}s (includes JIT compile), later steps median {np.median(times[1:]) if len(times) > 1 else 0:.1f}s")
    print(f"device log: {log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
