"""Layer node. Holds layers [a, b) only. One persistent WebSocket to the hub. Never loads the full model.
python -m dllm.node --hub ws://192.168.1.10:8000/ws/node --name mac2 --code ABC123 [--layers 16-28] [--device mps|mlx]

--device mlx runs the shard on dllm.mlx_backend (Apple's Metal via MLX) instead of torch. If MLX
is not installed, or the shard fails to build or warm on it, the node prints why and falls back to
the torch Shard on CPU automatically -- see build_shard() below. Never crashes the node either way."""
import argparse, asyncio, hashlib, os, resource, subprocess, sys, time, traceback, urllib.request
import numpy as np
import torch, websockets
from dllm.model import Cfg, Shard
from dllm.wire import pack, unpack, to_bytes, from_bytes


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 1024, 1)  # macOS bytes, Linux KiB


def fingerprint(shard_dir, i):
    """Hash of the weights this node loaded, matching dllm.slicer.content_hash. Container-independent,
    so it can be checked against the manifest by a coordinator that holds no weights at all."""
    h = hashlib.sha256()
    with np.load(f"{shard_dir}/layer_{i:02d}.npz") as z:
        for k in sorted(z.files):
            a = np.ascontiguousarray(z[k])
            h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:16]


_cpu_last = None


def device_stats():
    """What this device can say about itself, for the hub to stamp on this node's hop spans.
    psutil if it is installed (the Mac venv), /proc otherwise (Termux on Android). Never raises."""
    global _cpu_last
    out = {}
    try:
        import psutil
        p, vm = psutil.Process(), psutil.virtual_memory()
        out.update(rss_bytes=p.memory_info().rss, sys_total_bytes=vm.total, sys_used_bytes=vm.used,
                   sys_available_bytes=vm.available, sys_percent=vm.percent, cpu_percent=p.cpu_percent(None))
        return out
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as f:
            out["rss_bytes"] = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["rss_bytes"] = int(r if sys.platform == "darwin" else r * 1024)
    try:
        m = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                m[k] = int(v.strip().split()[0]) * 1024
        total, avail = m["MemTotal"], m.get("MemAvailable", m["MemFree"])
        out.update(sys_total_bytes=total, sys_used_bytes=total - avail, sys_available_bytes=avail,
                   sys_percent=round(100 * (total - avail) / total, 1))
    except Exception:
        pass
    try:
        t = os.times(); now = time.time(); cpu = t.user + t.system
        if _cpu_last:
            dc, dw = cpu - _cpu_last[0], now - _cpu_last[1]
            if dw > 0:
                out["cpu_percent"] = round(100 * dc / dw, 1)
        _cpu_last = (cpu, now)
    except Exception:
        pass
    return out


def battery():
    try:  # macOS only; phones report from their own API
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=2).stdout
        return int(out.split("%")[0].split()[-1])
    except Exception:
        return None


def disk_layers(shard_dir):
    """Layer ids already on disk. The hub's planner counts these as free to assign here."""
    if not os.path.isdir(shard_dir):
        return []
    return sorted(int(f[6:8]) for f in os.listdir(shard_dir) if f.startswith("layer_") and f.endswith(".npz"))


def fetch_shards(http_base, shard_dir, a, b):
    """Download whatever [a, b) still lacks. Returns (bytes downloaded, seconds), which the hub
    turns into a bandwidth estimate; files already present cost nothing and count nothing.
    Downloads land in a .part file first so a socket drop mid-transfer never leaves a truncated
    shard that looks complete on the next pass. Downloads run in parallel (4 threads)."""
    os.makedirs(shard_dir, exist_ok=True)
    total, t0 = 0, time.time()
    needed = []
    for f in ["config.json"] + [f"layer_{i:02d}.npz" for i in range(a, b)]:
        dst = f"{shard_dir}/{f}"
        if not os.path.exists(dst):
            needed.append((f, dst))
    def _dl(item):
        f, dst = item
        print(f"downloading {f}", flush=True)
        urllib.request.urlretrieve(f"{http_base}/shards/{f}", dst + ".part")
        os.replace(dst + ".part", dst)
        return os.path.getsize(dst)
    if needed:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(needed))) as pool:
            total = sum(pool.map(_dl, needed))
    return total, time.time() - t0


def build_shard(cfg, shard_dir, a, b, device, shard, memory_cap=None):
    """Build or re-range the shard for [a, b) on `device`. Returns (shard, device_used).

    When `device` is "mlx", MLX is tried first; if it is not importable, or fails to construct or
    warm on the real weights in `shard_dir`, this prints one line saying why and falls back to the
    torch Shard on CPU instead of raising. A Mac node asked for a Metal backend it does not
    actually have must still come up and serve on CPU, not go missing. Any other `device` ("cpu",
    "mps") is unaffected: same construction as before, errors still propagate."""
    if device == "mlx":
        try:
            from dllm.mlx_backend import MlxShard, set_memory_cap, memory_now
            if memory_cap:
                # Before anything is allocated, so the first layer already loads under the ceiling.
                set_memory_cap(memory_cap)
            if isinstance(shard, MlxShard):
                shard.reassign(shard_dir, a, b)
            else:
                shard = MlxShard(cfg, shard_dir, a, b)
            # Prove it actually runs before committing to it: an import can succeed while the
            # Metal device init or a real forward still fails.
            shard(torch.zeros(1, 1, cfg.hidden), torch.tensor([0]), req="_mlx_probe")
            shard.reset("_mlx_probe")
            if memory_cap:
                act, peak = memory_now()
                if act is not None:
                    print(f"mlx holding {act/2**30:.2f} GB (peak {peak/2**30:.2f}) under a "
                          f"{float(memory_cap):.1f} GB cap", flush=True)
            return shard, "mlx"
        except Exception as e:
            print(f"--device mlx unavailable ({type(e).__name__}: {e}) -- falling back to torch on cpu", flush=True)
            device, shard = "cpu", None   # an MlxShard (or nothing) can't be torch's reassign target
    if isinstance(shard, Shard):
        shard.reassign(shard_dir, a, b)
    else:
        shard = Shard(cfg, shard_dir, a, b, device=device)
    return shard, device


def load_range(http_base, shard_dir, a, b, device="cpu", shard=None, memory_cap=None):
    """Fetch what is missing, build or re-range the shard, warm it, bench it. One function for the
    first assign and for a reassign, so both paths measure and report exactly the same things.
    Shards stay on disk whichever range this node holds: an unused file costs nothing at runtime,
    and the next plan that hands the layer back finds it here rather than downloading it again.
    Blocking by design: run it in a thread so heartbeats keep flowing during a long download."""
    dl_bytes, dl_s = fetch_shards(http_base, shard_dir, a, b)
    cfg = Cfg.load(f"{shard_dir}/config.json")
    t0 = time.time()
    shard, device = build_shard(cfg, shard_dir, a, b, device, shard, memory_cap)
    # warm up + bench: one 1-token forward through own layers
    shard(torch.zeros(1, 1, cfg.hidden), torch.tensor([0]), req="_bench"); shard.reset("_bench")
    load_s = time.time() - t0
    t1 = time.time(); shard(torch.zeros(1, 1, cfg.hidden), torch.tensor([0]), req="_bench"); shard.reset("_bench")
    ms_per_layer = (time.time() - t1) * 1000 / (b - a)
    print(f"layers {a}-{b-1} loaded in {load_s:.1f}s, {ms_per_layer:.2f} ms/layer/token", flush=True)
    fields = {"layers": [a, b], "ms_per_layer": ms_per_layer, "batch": True, "reassign": True,
              "rss_mb": rss_mb(), "shard_dir": os.path.abspath(shard_dir), "files": sorted(os.listdir(shard_dir)),
              "fingerprints": {i: fingerprint(shard_dir, i) for i in range(a, b)},
              "download_bytes": dl_bytes, "download_s": dl_s, "load_s": load_s, "device": device}
    return shard, cfg, fields


async def run(args):
    http_base = args.hub.replace("ws://", "http://").replace("wss://", "https://").split("/ws/")[0]
    shard = cfg = ready_fields = None
    had_ready = False
    tasks = set()

    def spawn(coro):
        # Loads and prefetches run beside the message loop, not in it: a forward that arrives while
        # nothing is loaded must get "not loaded" back, and a prefetch must not stall serving.
        t = asyncio.create_task(coro); tasks.add(t); t.add_done_callback(tasks.discard)

    async with websockets.connect(args.hub, max_size=None) as ws:
        # --memory-cap is what the planner is told this node has. Reporting the cap rather than the
        # machine's physical RAM is the whole mechanism: ram_layers() sizes a node from ram_gb, so
        # capping it here is what stops layers arriving in the first place. Enforcing the ceiling on
        # the MLX side as well only catches what slips past that.
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
        if args.memory_cap:
            ram_gb = round(min(ram_gb, float(args.memory_cap)), 1)
        hello = {"t": "hello", "name": args.name, "code": args.code, "device": args.device,
                 "ram_gb": ram_gb,
                 "reassign": True, "disk": disk_layers(args.shards)}
        if args.layers:
            hello["layers"] = [int(v) for v in args.layers.split("-")]
        await ws.send(pack(hello))

        async def heartbeat():
            while True:
                await ws.send(pack({"t": "hb", "battery": battery(), "cache_reqs": len(shard.cache) if shard else 0,
                                    "rss_mb": rss_mb(), "mem": device_stats()}))
                await asyncio.sleep(1)

        # One download or load at a time. Two loads on the same shard dir delete each other's files
        # and race on the same .part file, and whichever finished last would own `shard` even if
        # the hub had already committed the other range. An assign that is superseded while it
        # waits for the lock is skipped: only the newest one is worth loading.
        io_lock = asyncio.Lock()
        newest = [0]

        async def on_assign(hdr):
            nonlocal shard, cfg, ready_fields, had_ready
            a, b = hdr["layers"]
            gen = hdr.get("gen", 0)
            newest[0] += 1
            mine = newest[0]
            async with io_lock:
                if mine != newest[0]:
                    print(f"assign {a}-{b-1} gen {gen} superseded before it started", flush=True)
                    return
                if shard is not None and (shard.a, shard.b) == (a, b):
                    # Same range, already loaded: reclaim and no-op plans cost nothing. The file
                    # listing is re-read because a prefetch may have added files since the load.
                    ready_fields["files"] = sorted(os.listdir(args.shards))
                    await ws.send(pack({"t": "ready", **ready_fields, "gen": gen, "reassigned": True, "rss_mb": rss_mb()}))
                    return
                reassigned = had_ready
                prev, shard = shard, None          # forwards answer "not loaded" until the new range is up
                args.layers = f"{a}-{b}"           # reclaim this exact range if the connection drops and we retry
                try:
                    shard, cfg, ready_fields = await asyncio.to_thread(load_range, http_base, args.shards, a, b, args.device, prev, args.memory_cap)
                except Exception as e:
                    # Say so, or the hub sees a heartbeating node that never answers and waits out its deadline.
                    traceback.print_exc()
                    await ws.send(pack({"t": "ready", "gen": gen, "layers": [a, b], "error": f"{type(e).__name__}: {e}"}))
                    return
                await ws.send(pack({"t": "ready", **ready_fields, "gen": gen, "reassigned": reassigned}))
                had_ready = True

        async def on_prefetch(hdr):
            a, b = hdr["layers"]
            gen = hdr.get("gen", 0)
            try:
                async with io_lock:
                    n, secs = await asyncio.to_thread(fetch_shards, http_base, args.shards, a, b)
                await ws.send(pack({"t": "prefetched", "gen": gen, "layers": [a, b], "bytes": n, "s": secs}))
            except Exception as e:
                await ws.send(pack({"t": "prefetched", "gen": gen, "error": f"{type(e).__name__}: {e}"}))

        hb = asyncio.create_task(heartbeat())
        try:
            async for msg in ws:
                hdr, payload = unpack(msg)
                kind = hdr["t"]
                if kind == "assign":
                    spawn(on_assign(hdr))
                elif kind == "prefetch":
                    spawn(on_prefetch(hdr))
                elif kind == "standby":
                    print(f"standby: {hdr.get('reason', '')}", flush=True)
                elif kind == "error":
                    print(f"hub refused us: {hdr}", flush=True); return
                elif kind == "fwd":
                    if shard is None:
                        await ws.send(pack({"t": "fwd_out", "req": hdr["req"], "hop": hdr["hop"], "error": "not loaded"})); continue
                    n = hdr["n"]
                    dt = hdr.get("dtype", "bf16")
                    x = from_bytes(payload, (1, n, cfg.hidden), dt)
                    pos = torch.arange(hdr["pos"], hdr["pos"] + n)
                    t = time.time()
                    y = shard(x, pos, req=hdr["req"])
                    ms = (time.time() - t) * 1000
                    await ws.send(pack({"t": "fwd_out", "req": hdr["req"], "hop": hdr["hop"], "n": n, "ms": ms, "dtype": hdr.get("out_dtype", "bf16")},
                                       to_bytes(y, hdr.get("out_dtype", "bf16"))))
                elif kind == "fwd_batch":
                    if shard is None:
                        await ws.send(pack({"t": "fwd_batch_out", "key": hdr["key"], "error": "not loaded"})); continue
                    reqs, poss = hdr["reqs"], hdr["pos"]
                    x = from_bytes(payload, (len(reqs), 1, cfg.hidden), hdr.get("dtype", "bf16"))
                    t = time.time()
                    y = shard.forward_batch(x, poss, reqs)
                    ms = (time.time() - t) * 1000
                    out_dt = hdr.get("out_dtype", "bf16")
                    await ws.send(pack({"t": "fwd_batch_out", "key": hdr["key"], "batch": len(reqs),
                                        "ms": ms, "dtype": out_dt}, to_bytes(y, out_dt)))
                elif kind == "reset":
                    if shard is not None:
                        shard.reset(hdr["req"])
        finally:
            hb.cancel()
            for t in tasks:
                t.cancel()


async def forever(args):
    """Reconnect on any drop, so a node coming back rejoins without being restarted by hand."""
    while True:
        try:
            await run(args)
            print("hub closed the connection")
        except Exception as e:
            print(f"disconnected: {type(e).__name__}: {e}")
        print("retrying in 3s")
        await asyncio.sleep(3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hub", default="ws://127.0.0.1:8000/ws/node")
    p.add_argument("--name", default=os.uname().nodename)
    p.add_argument("--code", default="")
    p.add_argument("--layers", default=None, help="a-b, half-open. Omit to let the hub assign.")
    p.add_argument("--shards", default="shards")
    p.add_argument("--memory-cap", type=float, default=None, metavar="GB",
                   help="ceiling on the memory this node uses. Two effects: the planner is told "
                        "this instead of the machine's physical RAM, so it stops assigning layers "
                        "at the cap, and on --device mlx it is enforced with MLX's own memory "
                        "limit. Omit for no cap.")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                   help="cpu, mps, or mlx (Metal via MLX; falls back to cpu automatically if unavailable)")
    asyncio.run(forever(p.parse_args()))
