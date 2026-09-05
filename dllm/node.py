"""Layer node. Holds layers [a, b) only. One persistent WebSocket to the hub. Never loads the full model.
python -m dllm.node --hub ws://192.168.1.10:8000/ws/node --name mac2 --code ABC123 [--layers 16-28] [--device mps]"""
import argparse, asyncio, os, subprocess, time, urllib.request
import torch, websockets
from dllm.model import Cfg, Shard
from dllm.wire import pack, unpack, to_bf16_bytes, from_bf16_bytes


def battery():
    try:  # macOS only; phones report from their own API
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=2).stdout
        return int(out.split("%")[0].split()[-1])
    except Exception:
        return None


def fetch_shards(http_base, shard_dir, a, b):
    os.makedirs(shard_dir, exist_ok=True)
    for f in ["config.json"] + [f"layer_{i:02d}.npz" for i in range(a, b)]:
        dst = f"{shard_dir}/{f}"
        if not os.path.exists(dst):
            print("downloading", f)
            urllib.request.urlretrieve(f"{http_base}/shards/{f}", dst)


async def run(args):
    http_base = args.hub.replace("ws://", "http://").replace("wss://", "https://").split("/ws/")[0]
    async with websockets.connect(args.hub, max_size=None) as ws:
        hello = {"t": "hello", "name": args.name, "code": args.code, "device": args.device,
                 "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)}
        if args.layers:
            hello["layers"] = [int(v) for v in args.layers.split("-")]
        await ws.send(pack(hello))
        hdr, _ = unpack(await ws.recv())
        assert hdr["t"] == "assign", hdr
        a, b = hdr["layers"]
        fetch_shards(http_base, args.shards, a, b)
        cfg = Cfg.load(f"{args.shards}/config.json")
        t0 = time.time()
        shard = Shard(cfg, args.shards, a, b, device=args.device)
        # warm up + bench: one 1-token forward through own layers
        shard(torch.zeros(1, 1, cfg.hidden), torch.tensor([0]), req="_bench"); shard.reset("_bench")
        t1 = time.time(); shard(torch.zeros(1, 1, cfg.hidden), torch.tensor([0]), req="_bench"); shard.reset("_bench")
        ms_per_layer = (time.time() - t1) * 1000 / (b - a)
        print(f"layers {a}-{b-1} loaded in {time.time()-t0:.1f}s, {ms_per_layer:.2f} ms/layer/token")
        await ws.send(pack({"t": "ready", "layers": [a, b], "ms_per_layer": ms_per_layer}))

        async def heartbeat():
            while True:
                await ws.send(pack({"t": "hb", "battery": battery(), "cache_reqs": len(shard.cache)}))
                await asyncio.sleep(1)
        hb = asyncio.create_task(heartbeat())
        try:
            async for msg in ws:
                hdr, payload = unpack(msg)
                if hdr["t"] == "fwd":
                    n = hdr["n"]
                    x = from_bf16_bytes(payload, (1, n, cfg.hidden))
                    pos = torch.arange(hdr["pos"], hdr["pos"] + n)
                    t = time.time()
                    y = shard(x, pos, req=hdr["req"])
                    ms = (time.time() - t) * 1000
                    await ws.send(pack({"t": "fwd_out", "req": hdr["req"], "hop": hdr["hop"], "n": n, "ms": ms}, to_bf16_bytes(y)))
                elif hdr["t"] == "reset":
                    shard.reset(hdr["req"])
        finally:
            hb.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hub", default="ws://127.0.0.1:8000/ws/node")
    p.add_argument("--name", default=os.uname().nodename)
    p.add_argument("--code", default="")
    p.add_argument("--layers", default=None, help="a-b, half-open. Omit to let the hub assign.")
    p.add_argument("--shards", default="shards")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    asyncio.run(run(p.parse_args()))
