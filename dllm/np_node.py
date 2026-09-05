"""Phone node. Pure numpy + websockets, no torch, no safetensors. Runs in Termux.
Holds layers [a, b) only, downloads just those shards from the hub, keeps its own KV cache.

  pkg install python python-numpy && pip install websockets
  python np_node.py --hub ws://127.0.0.1:8000/ws/node --code ABC123 --name phoneA
"""
import argparse, asyncio, hashlib, json, os, resource, struct, sys, time, traceback, urllib.request
import numpy as np
import websockets

# ---------- wire (mirror of dllm/wire.py, bf16 as uint16) ----------

def pack(hdr, payload=b""):
    h = json.dumps(hdr, separators=(",", ":")).encode()
    return struct.pack(">I", len(h)) + h + payload

def unpack(buf):
    n = struct.unpack(">I", buf[:4])[0]
    return json.loads(buf[4:4 + n]), buf[4 + n:]

def to_wire(x, dtype="bf16"):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "fp32":
        return x.tobytes()
    u = x.view(np.uint32)
    r = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)   # round to nearest even
    return ((u + r) >> 16).astype(np.uint16).tobytes()

def from_wire(b, shape, dtype="bf16"):
    if dtype == "fp32":
        return np.frombuffer(b, dtype=np.float32).reshape(shape).copy()
    u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape)

# ---------- math ----------

def rms_norm(x, w, eps):
    return x * (np.float32(1.0) / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(eps))) * w

def rope(x, pos, theta):
    d = x.shape[-1]
    inv = (np.float32(1.0) / (np.float32(theta) ** (np.arange(0, d, 2, dtype=np.float32) / np.float32(d)))).astype(np.float32)
    f = pos.astype(np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([f, f], -1)
    cos, sin = np.cos(emb)[None, None], np.sin(emb)[None, None]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return (x * cos + np.concatenate([-x2, x1], -1) * sin).astype(np.float32, copy=False)

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class Layer:
    def __init__(self, cfg, w):
        self.c = cfg
        self.ln1, self.ln2 = w["input_layernorm.weight"], w["post_attention_layernorm.weight"]
        T = lambda k: np.ascontiguousarray(w[k].T)
        self.wq, self.bq = T("self_attn.q_proj.weight"), w["self_attn.q_proj.bias"]
        self.wk, self.bk = T("self_attn.k_proj.weight"), w["self_attn.k_proj.bias"]
        self.wv, self.bv = T("self_attn.v_proj.weight"), w["self_attn.v_proj.bias"]
        self.wo = T("self_attn.o_proj.weight")
        self.wg, self.wu, self.wd = T("mlp.gate_proj.weight"), T("mlp.up_proj.weight"), T("mlp.down_proj.weight")

    def __call__(self, x, pos, pk, pv):
        c = self.c
        b, n, _ = x.shape
        h = rms_norm(x, self.ln1, c["rms_norm_eps"])
        H, KV, D = c["heads"], c["kv_heads"], c["head_dim"]
        q = (h @ self.wq + self.bq).reshape(b, n, H, D).transpose(0, 2, 1, 3)
        k = (h @ self.wk + self.bk).reshape(b, n, KV, D).transpose(0, 2, 1, 3)
        v = (h @ self.wv + self.bv).reshape(b, n, KV, D).transpose(0, 2, 1, 3)
        q, k = rope(q, pos, c["rope_theta"]), rope(k, pos, c["rope_theta"])
        if pk is not None:
            k, v = np.concatenate([pk, k], 2), np.concatenate([pv, v], 2)
        rep = H // KV
        kk, vv = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
        p = k.shape[2] - n
        mask = np.tril(np.ones((n, p + n), dtype=bool), k=p)
        att = (q @ kk.transpose(0, 1, 3, 2)) * np.float32(1.0 / np.sqrt(D))
        att = softmax(np.where(mask, att, np.float32(-np.inf)))
        o = (att @ vv).transpose(0, 2, 1, 3).reshape(b, n, c["hidden"])
        x = x + o @ self.wo
        h = rms_norm(x, self.ln2, c["rms_norm_eps"])
        g = h @ self.wg
        x = x + ((g / (1.0 + np.exp(-g))) * (h @ self.wu)) @ self.wd
        return x, k, v


class Shard:
    def __init__(self, cfg, shard_dir, a, b):
        self.cfg, self.a, self.b = cfg, a, b
        self.layers = []
        for i in range(a, b):
            with np.load(f"{shard_dir}/layer_{i:02d}.npz") as z:
                self.layers.append(Layer(cfg, {k: z[k].astype(np.float32) for k in z.files}))
        self.cache = {}

    def __call__(self, x, pos, req="default"):
        past, new = self.cache.get(req), []
        for i, layer in enumerate(self.layers):
            pk, pv = past[i] if past else (None, None)
            x, k, v = layer(x, pos, pk, pv)
            assert x.dtype == np.float32, f"layer {i} returned {x.dtype}, expected float32"
            new.append((k, v))
        self.cache[req] = new
        return x

    def forward_batch(self, x, positions, reqs):
        """One decode step for several requests at once: x is (B, 1, hidden), one row each.

        The projections and the feedforward run over all rows in a single matmul, so the weights are
        read once for the whole batch. That is the entire point: a decode step is bound by moving
        weights, not by arithmetic. Attention is per row, because each request has its own history.
        """
        c = self.cfg
        B = x.shape[0]
        cached = [self.cache.get(r) for r in reqs]
        for li, layer in enumerate(self.layers):
            h = rms_norm(x, layer.ln1, c["rms_norm_eps"])
            H, KV, D = c["heads"], c["kv_heads"], c["head_dim"]
            q = (h @ layer.wq + layer.bq).reshape(B, 1, H, D).transpose(0, 2, 1, 3)
            k = (h @ layer.wk + layer.bk).reshape(B, 1, KV, D).transpose(0, 2, 1, 3)
            v = (h @ layer.wv + layer.bv).reshape(B, 1, KV, D).transpose(0, 2, 1, 3)
            rep_ = H // KV
            out = np.empty((B, 1, c["hidden"]), np.float32)
            for i in range(B):
                p = np.array([positions[i]])
                qi = rope(q[i:i + 1], p, c["rope_theta"])
                ki = rope(k[i:i + 1], p, c["rope_theta"])
                vi = v[i:i + 1]
                prev = cached[i][li] if cached[i] is not None and cached[i][li] is not None else None
                if prev is not None:
                    ki = np.concatenate([prev[0], ki], 2)
                    vi = np.concatenate([prev[1], vi], 2)
                if cached[i] is None:
                    cached[i] = [None] * len(self.layers)
                    self.cache[reqs[i]] = cached[i]
                cached[i][li] = (ki, vi)
                att = (qi @ np.repeat(ki, rep_, 1).transpose(0, 1, 3, 2)) * np.float32(1.0 / np.sqrt(D))
                att = softmax(att)                      # one query against all keys: nothing to mask
                out[i] = (att @ np.repeat(vi, rep_, 1)).transpose(0, 2, 1, 3).reshape(1, 1, c["hidden"])
            x = x + out @ layer.wo
            h = rms_norm(x, layer.ln2, c["rms_norm_eps"])
            g = h @ layer.wg
            x = x + ((g / (np.float32(1.0) + np.exp(-g))) * (h @ layer.wu)) @ layer.wd
        return x

    def reset(self, req):
        self.cache.pop(req, None)


def read_cfg(path):
    d = json.load(open(path))
    return {"hidden": d["hidden_size"], "heads": d["num_attention_heads"],
            "kv_heads": d["num_key_value_heads"], "head_dim": d["hidden_size"] // d["num_attention_heads"],
            "rms_norm_eps": d["rms_norm_eps"], "rope_theta": d["rope_theta"],
            "n_layers": d["num_hidden_layers"]}


def fetch(http_base, shard_dir, names):
    """Download whatever is missing. Returns (bytes downloaded, seconds), which the hub turns into a
    bandwidth estimate; files already present cost nothing and count nothing. Downloads land in a
    .part file first so a socket drop mid-transfer never leaves a truncated shard that looks complete."""
    os.makedirs(shard_dir, exist_ok=True)
    total, t0 = 0, time.time()
    for f in names:
        dst = f"{shard_dir}/{f}"
        if os.path.exists(dst):
            continue
        print("downloading", f, flush=True)
        urllib.request.urlretrieve(f"{http_base}/shards/{f}", dst + ".part")
        os.replace(dst + ".part", dst)
        total += os.path.getsize(dst)
    return total, time.time() - t0


def fetch_range(http_base, shard_dir, a, b):
    return fetch(http_base, shard_dir, ["config.json"] + [f"layer_{i:02d}.npz" for i in range(a, b)])


def disk_layers(shard_dir):
    """Layer ids already on disk. The hub's planner counts these as free to assign here."""
    if not os.path.isdir(shard_dir):
        return []
    return sorted(int(f[6:8]) for f in os.listdir(shard_dir) if f.startswith("layer_") and f.endswith(".npz"))


def load_range(http_base, shard_dir, a, b):
    """Fetch, drop foreign shards, build the shard, warm it, bench it. One function for the first
    assign and for a reassign, so both paths measure and report exactly the same things.
    Blocking by design: run it in a thread so heartbeats keep flowing during a long download."""
    dl_bytes, dl_s = fetch_range(http_base, shard_dir, a, b)
    drop_foreign_shards(shard_dir, a, b)
    cfg = read_cfg(f"{shard_dir}/config.json")
    t0 = time.time()
    shard = Shard(cfg, shard_dir, a, b)
    z = np.zeros((1, 1, cfg["hidden"]), np.float32)
    shard(z, np.array([0]), "_b"); shard.reset("_b")
    load_s = time.time() - t0
    t1 = time.time(); shard(z, np.array([0]), "_b"); shard.reset("_b")
    ms = (time.time() - t1) * 1000 / (b - a)
    print(f"layers {a}-{b-1} loaded in {load_s:.1f}s, {ms:.2f} ms/layer/token", flush=True)
    fields = {"layers": [a, b], "ms_per_layer": ms, "batch": True, "reassign": True,
              "rss_mb": rss_mb(), "shard_dir": os.path.abspath(shard_dir), "files": sorted(os.listdir(shard_dir)),
              "fingerprints": {i: fingerprint(shard_dir, i) for i in range(a, b)},
              "download_bytes": dl_bytes, "download_s": dl_s, "load_s": load_s}
    return shard, cfg, fields


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


def drop_foreign_shards(shard_dir, a, b):
    """A node keeps only the layers it owns. Anything left over from an earlier assignment goes,
    otherwise a node that has been reassigned quietly accumulates the whole model."""
    removed = []
    for f in sorted(os.listdir(shard_dir)):
        if f.startswith("layer_") and f.endswith(".npz") and not (a <= int(f[6:8]) < b):
            os.remove(f"{shard_dir}/{f}"); removed.append(f)
    if removed:
        print(f"removed {len(removed)} shard(s) outside layers {a}-{b-1}: {', '.join(removed)}", flush=True)
    return removed


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


_batt_ok = True

def battery():
    global _batt_ok
    if not _batt_ok:
        return None
    try:
        return int(json.loads(os.popen("termux-battery-status 2>/dev/null").read())["percentage"])
    except Exception:
        _batt_ok = False
        return None


async def run(args):
    http_base = args.hub.split("/ws/")[0].replace("ws://", "http://").replace("wss://", "https://")
    shard = cfg = ready_fields = None
    had_ready = False
    tasks = set()

    def spawn(coro):
        # Loads and prefetches run beside the message loop, not in it: a forward that arrives while
        # nothing is loaded must get "not loaded" back, and a prefetch must not stall serving.
        t = asyncio.ensure_future(coro); tasks.add(t); t.add_done_callback(tasks.discard)

    async with websockets.connect(args.hub, max_size=None, ping_interval=20) as ws:
        hello = {"t": "hello", "name": args.name, "code": args.code, "device": "phone-numpy",
                 "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
                 "reassign": True, "disk": disk_layers(args.shards)}
        if args.layers:
            hello["layers"] = [int(v) for v in args.layers.split("-")]
        await ws.send(pack(hello))

        async def hb():
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
                shard = None                       # forwards answer "not loaded" until the new range is up
                args.layers = f"{a}-{b}"           # reclaim this exact range if the connection drops and we retry
                try:
                    shard, cfg, ready_fields = await asyncio.to_thread(load_range, http_base, args.shards, a, b)
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
                    n, secs = await asyncio.to_thread(fetch_range, http_base, args.shards, a, b)
                await ws.send(pack({"t": "prefetched", "gen": gen, "layers": [a, b], "bytes": n, "s": secs}))
            except Exception as e:
                await ws.send(pack({"t": "prefetched", "gen": gen, "error": f"{type(e).__name__}: {e}"}))

        task = asyncio.ensure_future(hb())
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
                    x = from_wire(payload, (1, n, cfg["hidden"]), hdr.get("dtype", "bf16"))
                    pos = np.arange(hdr["pos"], hdr["pos"] + n)
                    t = time.time()
                    y = shard(x, pos, hdr["req"])
                    dt = (time.time() - t) * 1000
                    print(f"  fwd req={hdr['req']} n={n} pos={hdr['pos']} {dt:.0f} ms", flush=True)
                    await ws.send(pack({"t": "fwd_out", "req": hdr["req"], "hop": hdr["hop"], "n": n, "ms": dt, "dtype": hdr.get("out_dtype", "bf16")},
                                       to_wire(y, hdr.get("out_dtype", "bf16"))))
                elif kind == "fwd_batch":
                    if shard is None:
                        await ws.send(pack({"t": "fwd_batch_out", "key": hdr["key"], "error": "not loaded"})); continue
                    reqs, poss = hdr["reqs"], hdr["pos"]
                    x = from_wire(payload, (len(reqs), 1, cfg["hidden"]), hdr.get("dtype", "bf16"))
                    t = time.time()
                    y = shard.forward_batch(x, poss, reqs)
                    dt = (time.time() - t) * 1000
                    print(f"  batch of {len(reqs)} {dt:.0f} ms", flush=True)
                    out_dt = hdr.get("out_dtype", "bf16")
                    await ws.send(pack({"t": "fwd_batch_out", "key": hdr["key"], "batch": len(reqs),
                                        "ms": dt, "dtype": out_dt}, to_wire(y, out_dt)))
                elif kind == "reset":
                    if shard is not None:
                        shard.reset(hdr["req"])
        finally:
            task.cancel()
            for t in tasks:
                t.cancel()


async def forever(args):
    """Reconnect on any drop. Pulling the cable or the tunnel is a kill switch, restoring it rejoins."""
    while True:
        try:
            await run(args)
            print("hub closed the connection", flush=True)
        except Exception as e:
            print(f"disconnected: {type(e).__name__}: {e}", flush=True)
        print("retrying in 3s", flush=True)
        await asyncio.sleep(3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hub", default="ws://127.0.0.1:8000/ws/node")
    p.add_argument("--name", default="phone")
    p.add_argument("--code", default="")
    p.add_argument("--layers", default=None, help="a-b half-open; omit to let hub assign")
    p.add_argument("--shards", default="shards")
    asyncio.run(forever(p.parse_args()))
