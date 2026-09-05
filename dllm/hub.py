"""Coordinator: lobby + hub + embed/lm_head/sampling + OpenAI endpoint + shard server.
python -m dllm.hub --shards shards --expected 4 --port 8000"""
import argparse, asyncio, json, math, os, random, string, time, uuid
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse, PlainTextResponse
from transformers import AutoTokenizer
from dllm.model import Cfg, Head
from dllm.wire import pack, unpack, to_bf16_bytes, from_bf16_bytes

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="shards")
ap.add_argument("--expected", type=int, default=int(os.getenv("EXPECTED_NODES", 4)))
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--code", default="".join(random.choices(string.ascii_uppercase + string.digits, k=6)))
ap.add_argument("--device", default="cpu")
ARGS, _ = ap.parse_known_args()

app = FastAPI(title="dllm hub")
cfg = Cfg.load(f"{ARGS.shards}/config.json")
tok = AutoTokenizer.from_pretrained(ARGS.shards)
head = Head(cfg, ARGS.shards, ARGS.device)
gen_cfg = json.load(open(f"{ARGS.shards}/generation_config.json")) if os.path.exists(f"{ARGS.shards}/generation_config.json") else {}
EOS = set(gen_cfg.get("eos_token_id", [tok.eos_token_id]) if isinstance(gen_cfg.get("eos_token_id"), list) else [gen_cfg.get("eos_token_id", tok.eos_token_id)])

nodes = {}          # name -> dict(ws, layers, ready, ms_per_layer, battery, last_hb)
pending = {}        # (req, hop) -> Future
events = []         # list of asyncio.Queue for /events SSE listeners
gen_lock = asyncio.Lock()   # ponytail: one request at a time; batching is ADR-006, later


def emit(ev):
    ev["ts"] = time.time()
    for q in events:
        q.put_nowait(ev)


def pipeline():
    """Ordered ready nodes. Valid only if they tile [0, n_layers) exactly."""
    ready = sorted((n for n in nodes.values() if n["ready"]), key=lambda n: n["layers"][0])
    want = 0
    for n in ready:
        if n["layers"][0] != want:
            return None
        want = n["layers"][1]
    return ready if want == cfg.n_layers else None


def assign_layers(hello):
    if "layers" in hello:
        return hello["layers"]
    taken = sorted(n["layers"] for n in nodes.values())
    start = taken[-1][1] if taken else 0
    chunk = math.ceil(cfg.n_layers / ARGS.expected)
    return [start, min(start + chunk, cfg.n_layers)]


@app.websocket("/ws/node")
async def ws_node(ws: WebSocket):
    await ws.accept()
    hdr, _ = unpack(await ws.receive_bytes())
    if hdr.get("t") != "hello" or hdr.get("code") != ARGS.code:
        await ws.send_bytes(pack({"t": "error", "msg": "bad lobby code"})); await ws.close(); return
    name = hdr["name"]
    layers = assign_layers(hdr)
    nodes[name] = {"ws": ws, "layers": layers, "ready": False, "ms_per_layer": None, "battery": None,
                   "device": hdr.get("device"), "ram_gb": hdr.get("ram_gb"), "last_hb": time.time()}
    await ws.send_bytes(pack({"t": "assign", "layers": layers}))
    emit({"t": "join", "node": name, "layers": layers})
    try:
        while True:
            hdr, payload = unpack(await ws.receive_bytes())
            t = hdr["t"]
            if t == "ready":
                nodes[name].update(ready=True, layers=hdr["layers"], ms_per_layer=hdr["ms_per_layer"])
                emit({"t": "ready", "node": name, "layers": hdr["layers"], "ms_per_layer": hdr["ms_per_layer"]})
            elif t == "hb":
                nodes[name].update(battery=hdr.get("battery"), last_hb=time.time())
            elif t == "fwd_out":
                fut = pending.pop((hdr["req"], hdr["hop"]), None)
                if fut and not fut.done():
                    fut.set_result((hdr, payload))
    except WebSocketDisconnect:
        pass
    finally:
        nodes.pop(name, None)
        emit({"t": "leave", "node": name})
        for k, f in list(pending.items()):
            if not f.done():
                f.set_exception(RuntimeError(f"node {name} left"))


async def forward_all(req, x, pos):
    """Run hidden x (1, n, hidden) through every node in layer order. Returns final hidden."""
    pipe = pipeline()
    if not pipe:
        raise RuntimeError(f"pipeline incomplete: {[ (n['layers']) for n in nodes.values() ]}")
    n = x.shape[1]
    for hop, node in enumerate(pipe):
        fut = asyncio.get_event_loop().create_future()
        pending[(req, hop)] = fut
        t = time.time()
        await node["ws"].send_bytes(pack({"t": "fwd", "req": req, "hop": hop, "pos": pos, "n": n}, to_bf16_bytes(x)))
        hdr, payload = await asyncio.wait_for(fut, timeout=120)
        x = from_bf16_bytes(payload, (1, n, cfg.hidden))
        emit({"t": "hop", "req": req, "hop": hop, "node": [k for k, v in nodes.items() if v is node][0],
              "layers": node["layers"], "n": n, "compute_ms": hdr["ms"], "wire_ms": (time.time() - t) * 1000 - hdr["ms"]})
    return x


async def generate(ids, max_tokens=256, temperature=0.0):
    req = uuid.uuid4().hex[:8]
    pipe = pipeline()
    try:
        x = await forward_all(req, head.embed_tokens(ids), 0)
        pos = len(ids)
        for _ in range(max_tokens):
            lg = head.logits(x)
            if temperature > 0:
                nxt = int(torch.multinomial(torch.softmax(lg / temperature, -1), 1))
            else:
                nxt = int(lg.argmax())
            if nxt in EOS:
                break
            emit({"t": "token", "req": req, "id": nxt, "pos": pos})
            yield nxt
            x = await forward_all(req, head.embed_tokens([nxt]), pos)
            pos += 1
    finally:
        for node in pipe or []:
            try: await node["ws"].send_bytes(pack({"t": "reset", "req": req}))
            except Exception: pass


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    prompt = tok.apply_chat_template(body["messages"], add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, add_special_tokens=False).input_ids
    max_tokens, temp = body.get("max_tokens", 256), body.get("temperature", 0.0)
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
    if pipeline() is None:
        return JSONResponse({"error": "pipeline incomplete", "nodes": {k: v["layers"] for k, v in nodes.items()}}, 503)

    async def run_stream():
        async with gen_lock:
            async for t in generate(ids, max_tokens, temp):
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": "dllm",
                         "choices": [{"index": 0, "delta": {"content": tok.decode([t])}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'dllm', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

    if body.get("stream"):
        return StreamingResponse(run_stream(), media_type="text/event-stream")
    async with gen_lock:
        out = [t async for t in generate(ids, max_tokens, temp)]
    return {"id": cid, "object": "chat.completion", "created": created, "model": "dllm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(out)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "dllm", "object": "model"}]}


@app.get("/shards/{name}")
def shard_file(name: str):
    return FileResponse(f"{ARGS.shards}/{os.path.basename(name)}")


@app.get("/s", response_class=PlainTextResponse)
@app.get("/setup.sh", response_class=PlainTextResponse)
def setup_sh(layers: str = "", name: str = "phoneA"):
    """Phone bootstrap. In Termux:  pkg i -y curl && curl -s 127.0.0.1:8000/setup.sh | bash"""
    return f"""set -e
echo '== installing python + numpy (prebuilt, not pip) =='
pkg update -y >/dev/null 2>&1 || true
pkg install -y python python-numpy
python -c 'import numpy' || {{ echo 'numpy missing'; exit 1; }}
pip install --quiet websockets
echo '== fetching node source =='
curl -sO http://127.0.0.1:{ARGS.port}/node.py
echo '== joining cluster as {name} =='
exec python node.py --hub ws://127.0.0.1:{ARGS.port}/ws/node --code {ARGS.code} --name {name} {'--layers ' + layers if layers else ''}
"""


@app.get("/node.py")
def node_source():
    """Phone bootstrap: curl -O http://127.0.0.1:8000/node.py"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "np_node.py"), media_type="text/plain")


@app.get("/status")
def status():
    return {"code": ARGS.code, "n_layers": cfg.n_layers, "pipeline_ok": pipeline() is not None,
            "nodes": {k: {kk: vv for kk, vv in v.items() if kk != "ws"} for k, v in nodes.items()}}


@app.get("/events")
async def sse_events():
    q = asyncio.Queue(); events.append(q)
    async def gen():
        try:
            while True:
                yield f"data: {json.dumps(await q.get())}\n\n"
        finally:
            events.remove(q)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index():
    p = os.path.join(os.path.dirname(__file__), "status.html")
    return open(p).read() if os.path.exists(p) else f"<pre>dllm hub. lobby code {ARGS.code}. see /status</pre>"


if __name__ == "__main__":
    import uvicorn
    print(f"\n  LOBBY CODE: {ARGS.code}   expecting {ARGS.expected} nodes for {cfg.n_layers} layers\n")
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port, ws_max_size=None)
