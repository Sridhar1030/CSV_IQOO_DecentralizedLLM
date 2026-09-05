"""Coordinator: lobby + hub + embed/lm_head/sampling + OpenAI endpoint + shard server.
python -m dllm.hub --shards shards --expected 4 --port 8000"""
import contextlib
import argparse, asyncio, json, math, os, random, socket, string, time, uuid
import torch
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse, PlainTextResponse
from transformers import AutoTokenizer
from dllm.model import Cfg, Head, sample
from dllm.observe import Observability
from dllm.wire import pack, unpack, to_bytes, from_bytes

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="shards")
ap.add_argument("--dist", default=None,
                help="directory of layer shards to hand out to joining nodes. Kept separate from "
                     "--shards so the coordinator itself never has to hold the whole model. Delete "
                     "it once every node has loaded.")
ap.add_argument("--expected", type=int, default=int(os.getenv("EXPECTED_NODES", 4)))
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--code", default="".join(random.choices(string.ascii_uppercase + string.digits, k=6)))
ap.add_argument("--device", default="cpu")
ap.add_argument("--otlp", default=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                help="OTLP/HTTP endpoint for traces and metrics, normally the llmobs proxy "
                     "(http://localhost:8100). Unset disables telemetry entirely.")
ap.add_argument("--node-id", default=None, help="This coordinator's identity in telemetry. Default: hostname.")
ap.add_argument("--apk", default=os.path.join(os.path.dirname(__file__), "..", "android", "app", "build",
                                              "outputs", "apk", "debug", "app-debug.apk"),
                help="Android node app to offer on the join page, if the file exists.")
ARGS, _ = ap.parse_known_args()

app = FastAPI(title="dllm hub")
cfg = Cfg.load(f"{ARGS.shards}/config.json")
tok = AutoTokenizer.from_pretrained(ARGS.shards)
head = Head(cfg, ARGS.shards, ARGS.device)
gen_cfg = json.load(open(f"{ARGS.shards}/generation_config.json")) if os.path.exists(f"{ARGS.shards}/generation_config.json") else {}
EOS = set(gen_cfg.get("eos_token_id", [tok.eos_token_id]) if isinstance(gen_cfg.get("eos_token_id"), list) else [gen_cfg.get("eos_token_id", tok.eos_token_id)])

MODEL_NAME = (json.load(open(f"{ARGS.shards}/manifest.json")).get("repo", "dllm")
              if os.path.exists(f"{ARGS.shards}/manifest.json") else "dllm")

nodes = {}          # name -> dict(ws, layers, ready, ms_per_layer, battery, mem, last_hb)
pending = {}        # (req, hop) -> Future
events = []         # list of asyncio.Queue for /events SSE listeners
# Requests run concurrently. Every node already keys its KV cache by request id, and forward_all
# keys its pending futures by (request, hop), so two requests interleave on the wire without
# touching each other's state. The lock this replaces cost most of the cluster's throughput:
# with three devices, a serialised request leaves two of them idle at all times.
gen_lock = contextlib.nullcontext()


def emit(ev):
    ev["ts"] = time.time()
    for q in events:
        q.put_nowait(ev)


HB_TIMEOUT = 5.0  # a node that has not sent a heartbeat this recently is treated as gone
STALE_S = 30.0    # after this long silent, a node's layer range may be given to a newcomer


def live(n):
    return n["ready"] and (time.time() - n["last_hb"]) < HB_TIMEOUT


obs = Observability(endpoint=ARGS.otlp, model=MODEL_NAME, n_layers=cfg.n_layers, node_id=ARGS.node_id,
                    nodes=lambda: [(k, v, live(v)) for k, v in nodes.items()])


@app.on_event("shutdown")
def _flush_telemetry():
    obs.shutdown()


def missing_layers() -> list[int]:
    """Layers no live node holds. This is the whole reason a request gets refused, so say it."""
    have = set()
    for n in nodes.values():
        if live(n):
            have |= set(range(*n["layers"]))
    return sorted(set(range(cfg.n_layers)) - have)


def _ranges(xs: list[int]) -> str:
    """[8,9,10,20] -> '8-10,20'"""
    out, start = [], None
    for i, x in enumerate(xs):
        if start is None:
            start = x
        if i + 1 == len(xs) or xs[i + 1] != x + 1:
            out.append(str(start) if start == x else f"{start}-{x}")
            start = None
    return ",".join(out)


def pipeline():
    """Ordered live nodes. Valid only if they tile [0, n_layers) exactly.
    A node whose heartbeat has stopped is excluded even if its socket is still open, so a wedged
    or unreachable device fails the request instead of hanging it."""
    ready = sorted((n for n in nodes.values() if live(n)), key=lambda n: n["layers"][0])
    want = 0
    for n in ready:
        if n["layers"][0] != want:
            return None
        want = n["layers"][1]
    return ready if want == cfg.n_layers else None


assigned = {}   # node name -> layer range, kept across reconnects


def assign_layers(name, hello):
    """Sticky by name. A node that drops and comes back gets its own range, never a neighbour's.
    New ranges are packed after whatever is already handed out, including to absent nodes."""
    if "layers" in hello:
        return hello["layers"]
    if name in assigned:
        return assigned[name]
    held = set()
    for a, b in assigned.values():
        held |= set(range(a, b))
    free = [i for i in range(cfg.n_layers) if i not in held]
    if not free:
        # Every layer is spoken for, but some of it may belong to a node that has gone. A range
        # stays reserved through a brief reconnect; after this long without a heartbeat it is
        # released to whoever is joining now, which is how a replacement phone takes over.
        stale = [n for n in assigned if n not in nodes or time.time() - nodes[n]["last_hb"] > STALE_S]
        if not stale:
            return None
        gone = max(stale, key=lambda n: time.time() - nodes[n]["last_hb"] if n in nodes else 1e9)
        rng = assigned.pop(gone)
        nodes.pop(gone, None)
        print(f"  released layers {rng[0]}-{rng[1]-1} from {gone}, absent for over {STALE_S}s, to {name}")
        return rng
    start = free[0]                                        # the first gap, wherever it is
    end = start
    while end < cfg.n_layers and end in set(free):
        end += 1
    still_expected = max(1, ARGS.expected - len(assigned))
    chunk = math.ceil((end - start) / still_expected)
    return [start, min(start + chunk, end)]


@app.websocket("/ws/node")
async def ws_node(ws: WebSocket):
    await ws.accept()
    hdr, _ = unpack(await ws.receive_bytes())
    if hdr.get("t") != "hello" or hdr.get("code") != ARGS.code:
        await ws.send_bytes(pack({"t": "error", "msg": "bad lobby code"})); await ws.close(); return
    name = hdr["name"]
    layers = assign_layers(name, hdr)
    if layers is None:
        await ws.send_bytes(pack({"t": "error", "reason": "every layer is already assigned; nothing left for you"}))
        await ws.close(); return
    assigned[name] = layers
    nodes[name] = {"ws": ws, "layers": layers, "ready": False, "ms_per_layer": None, "battery": None,
                   "device": hdr.get("device"), "ram_gb": hdr.get("ram_gb"), "last_hb": time.time()}
    await ws.send_bytes(pack({"t": "assign", "layers": layers}))
    emit({"t": "join", "node": name, "layers": layers})
    try:
        while True:
            hdr, payload = unpack(await ws.receive_bytes())
            t = hdr["t"]
            if t == "ready":
                assigned[name] = hdr["layers"]
                nodes[name].update(batch=bool(hdr.get("batch")), ready=True, layers=hdr["layers"], ms_per_layer=hdr["ms_per_layer"],
                                   rss_mb=hdr.get("rss_mb"), shard_dir=hdr.get("shard_dir"),
                                   files=hdr.get("files", []), fingerprints=hdr.get("fingerprints", {}))
                emit({"t": "ready", "node": name, "layers": hdr["layers"], "ms_per_layer": hdr["ms_per_layer"]})
            elif t == "hb":
                nodes[name].update(battery=hdr.get("battery"), last_hb=time.time(),
                                   rss_mb=hdr.get("rss_mb", nodes[name].get("rss_mb")),
                                   mem=hdr.get("mem", nodes[name].get("mem")))
            elif t == "fwd_batch_out":
                fut = pending.pop(hdr["key"], None)
                if fut and not fut.done():
                    fut.set_result((hdr, payload))
            elif t == "fwd_out":
                fut = pending.pop((hdr["req"], hdr["hop"]), None)
                if fut and not fut.done():
                    fut.set_result((hdr, payload))
    except WebSocketDisconnect:
        pass
    finally:
        n = nodes.pop(name, None)
        if n is not None:
            print(f"  node {name} left (layers {n['layers'][0]}-{n['layers'][1]-1}, ready={n['ready']}) at {time.strftime('%H:%M:%S')}", flush=True)
        emit({"t": "leave", "node": name})
        for k, f in list(pending.items()):
            if not f.done():
                f.set_exception(RuntimeError(f"node {name} left"))


# --- batching -------------------------------------------------------------------------------------
# A decode step reads every weight in a node's shard in order to produce one token, which is half an
# operation of arithmetic per byte moved. Serving one request at a time throws almost all of that
# away. So decode steps waiting on the same node are merged into one frame and the node reads its
# weights once for the whole batch. Prefill is left alone: it already has many tokens to amortise.
#
# The wait below is the entire scheduler. Zero would batch nothing, because requests that could have
# merged have not arrived yet. Too long and every request pays the delay. A few milliseconds is well
# under the tens of milliseconds a hop costs, so it is invisible to one request but long enough for
# concurrent ones to find each other.
BATCH_WINDOW_S = 0.004
MAX_BATCH = 16

_batch_pending = {}   # node name -> list of (req, pos, x_row, future)
_batch_task = {}      # node name -> asyncio.Task


async def _batch_runner(name):
    """Collect decode rows for one node, send them as one frame, hand each row back to its request."""
    await asyncio.sleep(BATCH_WINDOW_S)
    queued = _batch_pending.get(name, [])
    items, _batch_pending[name] = queued[:MAX_BATCH], queued[MAX_BATCH:]
    _batch_task.pop(name, None)
    if _batch_pending.get(name):
        _batch_task[name] = asyncio.ensure_future(_batch_runner(name))
    if not items:
        return
    node = nodes.get(name)
    if node is None or not live(node):
        for *_, fut in items:
            if not fut.done():
                fut.set_exception(RuntimeError(f"node {name} left"))
        return
    reqs = [r for r, _, _, _ in items]
    poss = [p for _, p, _, _ in items]
    x = torch.cat([row for _, _, row, _ in items], 0)              # (B, 1, hidden)
    key = f"{name}:{_batch_seq()}"
    fut = asyncio.get_event_loop().create_future()
    pending[key] = fut
    hop, last = node["hop"], node["last_hop"]
    out_dt = "fp32" if hop == last else "bf16"
    t = time.time()
    try:
        await node["ws"].send_bytes(pack({"t": "fwd_batch", "key": key, "hop": hop,
                                          "reqs": reqs, "pos": poss,
                                          "dtype": "bf16", "out_dtype": out_dt}, to_bytes(x, "bf16")))
        hdr, payload = await asyncio.wait_for(fut, timeout=120)
    except BaseException as e:
        pending.pop(key, None)
        err = e if isinstance(e, Exception) else RuntimeError(str(e))
        for *_, f in items:
            if not f.done():
                f.set_exception(err)
        return
    y = from_bytes(payload, (len(items), 1, cfg.hidden), hdr.get("dtype", out_dt))
    ms, wire_ms = hdr["ms"], (time.time() - t) * 1000 - hdr["ms"]
    emit({"t": "hop", "req": reqs[0], "hop": hop, "node": name, "layers": node["layers"],
          "n": 1, "batch": len(items), "compute_ms": ms, "wire_ms": wire_ms})
    for i, (*_, f) in enumerate(items):
        if not f.done():
            f.set_result((y[i:i + 1], ms, wire_ms, len(items)))


_seq = 0


def _batch_seq():
    global _seq
    _seq += 1
    return _seq


async def _batched_hop(name, req, x, pos):
    """Queue one decode row for `name` and wait for that row's answer."""
    fut = asyncio.get_event_loop().create_future()
    _batch_pending.setdefault(name, []).append((req, pos, x, fut))
    if name not in _batch_task:
        _batch_task[name] = asyncio.ensure_future(_batch_runner(name))
    return await fut


async def forward_all(req, x, pos, trace=None):
    """Run hidden x (1, n, hidden) through every node in layer order. Returns final hidden."""
    pipe = pipeline()
    if not pipe:
        raise RuntimeError(f"pipeline incomplete: {[ (n['layers']) for n in nodes.values() ]}")
    n = x.shape[1]
    # Batch only if every node advertised it. A node from an older build ignores an unknown frame
    # and simply never answers, so one stale phone would hang the whole cluster. Falling back keeps
    # a mixed-version cluster correct, just without the throughput gain.
    if n == 1 and all(node.get("batch") for node in pipe):
        for hop, node in enumerate(pipe):
            name = next(k for k, v in nodes.items() if v is node)
            node["hop"], node["last_hop"] = hop, len(pipe) - 1
            t = time.time()
            x, ms, wire_ms, _ = await _batched_hop(name, req, x, pos)
            if trace:
                trace.hop(index=hop, name=name, node=node, n=1, pos=pos, started=t, ended=time.time(),
                          compute_ms=ms, wire_ms=wire_ms)
        return x

    dt = "bf16"
    for hop, node in enumerate(pipe):
        # the tail node's output feeds the lm_head, so it comes back in fp32
        out_dt = "fp32" if hop == len(pipe) - 1 else "bf16"
        fut = asyncio.get_event_loop().create_future()
        pending[(req, hop)] = fut
        t = time.time()
        await node["ws"].send_bytes(pack({"t": "fwd", "req": req, "hop": hop, "pos": pos, "n": n,
                                          "dtype": dt, "out_dtype": out_dt}, to_bytes(x, dt)))
        hdr, payload = await asyncio.wait_for(fut, timeout=120)
        t_end = time.time()
        x = from_bytes(payload, (1, n, cfg.hidden), hdr.get("dtype", out_dt))
        dt = out_dt
        name = next(k for k, v in nodes.items() if v is node)
        wire_ms = (t_end - t) * 1000 - hdr["ms"]
        emit({"t": "hop", "req": req, "hop": hop, "node": name,
              "layers": node["layers"], "n": n, "compute_ms": hdr["ms"], "wire_ms": wire_ms})
        if trace:
            trace.hop(index=hop, name=name, node=node, n=n, pos=pos, started=t, ended=t_end,
                      compute_ms=hdr["ms"], wire_ms=wire_ms)
    return x


# Prefill is sent in pieces. One message carrying hundreds of positions keeps a phone busy for tens
# of seconds, during which it cannot answer a ping and the hub drops it. Chunking changes no maths:
# each chunk extends the same KV cache at the next position.
PREFILL_CHUNK = 48


async def generate(ids, max_tokens=256, temperature=0.0, top_p=1.0, top_k=0, seed=None, trace=None):
    req = uuid.uuid4().hex[:8]
    pipe = pipeline()
    try:
        for i in range(0, len(ids), PREFILL_CHUNK):
            chunk = ids[i:i + PREFILL_CHUNK]
            x = await forward_all(req, head.embed_tokens(chunk), i, trace)
        pos = len(ids)
        for _ in range(max_tokens):
            lg = head.logits(x)
            nxt = sample(lg, temperature, top_p, top_k, None if seed is None else seed + pos)
            if nxt in EOS:
                break
            if trace:
                trace.first_token()
            emit({"t": "token", "req": req, "id": nxt, "pos": pos, "text": tok.decode([nxt])})
            yield nxt
            x = await forward_all(req, head.embed_tokens([nxt]), pos, trace)
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
    max_tokens = body.get("max_tokens", 256)
    temp = body.get("temperature") or 0.0
    top_p = body.get("top_p", 1.0)
    top_k = body.get("top_k", 0)
    seed = body.get("seed")
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
    if pipeline() is None:
        gaps = missing_layers()
        held = {k: f"{v['layers'][0]}-{v['layers'][1]-1}" for k, v in nodes.items() if live(v)}
        return JSONResponse({
            "error": "pipeline incomplete",
            "detail": (f"no live node holds layer(s) {_ranges(gaps)} of {cfg.n_layers}. "
                       f"Start a node for them, or scan {join_url()} on a phone."
                       if gaps else "nodes overlap or do not start at layer 0"),
            "missing_layers": _ranges(gaps),
            "live_nodes": held,
            "join_url": join_url(),
        }, 503)
    trace = obs.request(request_id=cid, input_tokens=len(ids), max_tokens=max_tokens, temperature=temp)
    finish = lambda n: "length" if n >= max_tokens else "stop"

    async def run_stream():
        n = 0
        try:
            with gen_lock:
                async for t in generate(ids, max_tokens, temp, top_p, top_k, seed, trace):
                    n += 1
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": "dllm",
                             "choices": [{"index": 0, "delta": {"content": tok.decode([t])}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'dllm', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish(n)}]})}\n\n"
                yield "data: [DONE]\n\n"
        except BaseException as e:
            trace.error(e, output_tokens=n)
            raise
        else:
            trace.finish(output_tokens=n, finish_reason=finish(n))

    if body.get("stream"):
        return StreamingResponse(run_stream(), media_type="text/event-stream")
    out = []
    try:
        with gen_lock:
            async for t in generate(ids, max_tokens, temp, top_p, top_k, seed, trace):
                out.append(t)
    except BaseException as e:
        trace.error(e, output_tokens=len(out))
        raise
    trace.finish(output_tokens=len(out), finish_reason=finish(len(out)))
    return {"id": cid, "object": "chat.completion", "created": created, "model": "dllm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(out)}, "finish_reason": finish(len(out))}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "dllm", "object": "model"}]}


@app.get("/shards/{name}")
def shard_file(name: str):
    name = os.path.basename(name)
    if ARGS.dist and os.path.exists(f"{ARGS.dist}/{name}"):
        return FileResponse(f"{ARGS.dist}/{name}")
    return FileResponse(f"{ARGS.shards}/{name}")


@app.get("/s/{name}/{layers}", response_class=PlainTextResponse)
def setup_sh_path(name: str, layers: str, request: Request):
    """Path form so it can be typed without ? or & :  curl 127.0.0.1:8000/s/phoneB/16-24 | sh"""
    return setup_sh(layers=layers, name=name, host=request.headers.get("host"))


# ---------------------------------------------------------------------------------------------
# Joining by camera. One URL does both jobs, chosen by what asked for it: a browser gets a page,
# curl gets the shell script. So the QR can carry a plain http:// link any camera app will open,
# and the same link piped through sh brings the node up.
# ---------------------------------------------------------------------------------------------
def lan_ip() -> str:
    """This machine's address on a network a phone can reach, rather than 127.0.0.1."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))      # sends nothing; this only picks the outbound route
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def join_url(host: str | None = None) -> str:
    return f"http://{host or f'{lan_ip()}:{ARGS.port}'}/j/{ARGS.code}"


def sh_error(message: str, status: int = 404) -> PlainTextResponse:
    """An error on this endpoint is usually about to be piped into a shell, so it has to BE a shell
    script. Returning prose gets you `sh: 1: wrong: not found`, which tells the user nothing."""
    safe = message.replace("'", "'\\''")
    return PlainTextResponse(f"echo '{safe}' >&2\nexit 1\n", status_code=status,
                             media_type="text/plain")


@app.get("/j/{code}")
def join(code: str, request: Request):
    """Scanned by a camera, or piped to sh. Same URL, different answer."""
    host = request.headers.get("host")
    wants_html = "text/html" in request.headers.get("accept", "")
    if code.strip().upper() != ARGS.code.upper():
        msg = f"dllm: wrong lobby code '{code}'. This cluster's code is {ARGS.code}. Try: curl -s {join_url(host)} | sh"
        return HTMLResponse(f"<pre>{msg}</pre>", status_code=404) if wants_html else sh_error(msg)
    name = next(f"node{i}" for i in range(1, 1000) if f"node{i}" not in assigned)   # never reuse a live name
    if not wants_html:
        return PlainTextResponse(setup_sh(name=name, host=host), media_type="text/plain")
    held = sum(b - a for a, b in assigned.values())
    apk_link = (f'<p><a href="/app.apk" style="display:block;text-align:center;border:1px solid #30363d;color:#e6edf3;'
                f'text-decoration:none;padding:12px;border-radius:8px">Get the app ({os.path.getsize(ARGS.apk) // 1048576} MB)</a>'
                f'<br><span style="color:#5b6773;font-size:13px">Android asks you to allow installs from this source. Say yes once.</span></p>'
                if os.path.exists(ARGS.apk) else '<p style="color:#5b6773">App not built yet on this coordinator.</p>')
    return HTMLResponse(f"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Join the cluster</title>
<style>body{{font:16px/1.55 system-ui;margin:0;padding:24px;background:#0b0d10;color:#e6edf3}}
h1{{font-size:20px;margin:0 0 4px}}p{{color:#9aa7b4;margin:6px 0 18px}}
code{{display:block;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;
font:14px/1.5 ui-monospace,monospace;color:#7ee787;word-break:break-all;user-select:all;-webkit-user-select:all}}
b{{color:#e6edf3}}ol{{padding-left:20px;color:#9aa7b4}}li{{margin:10px 0}}</style>
<h1>Join as <b>{name}</b></h1>
<p>Lobby <b>{ARGS.code}</b> &middot; {len(assigned)} node(s) already in &middot;
{cfg.n_layers - held} of {cfg.n_layers} layers still unassigned</p>
<p style="margin-top:22px"><a href="dllm://join?hub={host}&amp;code={ARGS.code}" style="display:block;text-align:center;
background:#1f6feb;color:#fff;text-decoration:none;padding:14px;border-radius:8px;font-weight:600">Open in the dllm node app</a></p>
{apk_link}
<p style="margin-top:22px;color:#5b6773">Developer route, Termux:</p>
<code>curl -s {join_url(host)} | sh</code>
""")


@app.get("/app.apk")
def app_apk():
    if not os.path.exists(ARGS.apk):
        return PlainTextResponse("no APK built on this coordinator\n", status_code=404)
    return FileResponse(ARGS.apk, media_type="application/vnd.android.package-archive", filename="dllm-node.apk")


@app.get("/qr.svg")
def qr_svg(request: Request):
    """The same join URL as a scannable image, for putting on a projector."""
    import segno
    return Response(segno.make(join_url(request.headers.get("host")), error="m").svg_inline(scale=8),
                    media_type="image/svg+xml")


@app.get("/s", response_class=PlainTextResponse)
@app.get("/setup.sh", response_class=PlainTextResponse)
def setup_sh(layers: str = "", name: str = "phoneA", host: str | None = None):
    """Phone bootstrap. In Termux:  curl -s 127.0.0.1:8000/s | sh"""
    host = host or f"127.0.0.1:{ARGS.port}"
    return f"""set -e
echo '== installing python + numpy (prebuilt, not pip) =='
pkg update -y >/dev/null 2>&1 || true
pkg install -y python python-numpy
python -c 'import numpy' || {{ echo 'numpy missing'; exit 1; }}
pip install --quiet websockets
echo '== fetching node source =='
curl -sO http://{host}/node.py
mkdir -p ~/bin
printf '#!%s/bin/sh\\ncurl -s "$1" | sh\\n' "$PREFIX" > ~/bin/termux-url-opener
chmod +x ~/bin/termux-url-opener
echo '== joining cluster as {name} =='
exec python node.py --hub ws://{host}/ws/node --code {ARGS.code} --name {name} {'--layers ' + layers if layers else ''}
"""


@app.exception_handler(404)
async def not_found(request: Request, exc):
    """Same reasoning as sh_error: /s/... and /j/... get piped into a shell."""
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(f"<pre>dllm: no such path {request.url.path}</pre>", status_code=404)
    return sh_error(f"dllm: no such path {request.url.path} on this hub")


@app.get("/node.py")
def node_source():
    """Phone bootstrap: curl -O http://127.0.0.1:8000/node.py"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "np_node.py"), media_type="text/plain")


@app.get("/status")
def status():
    now = time.time()
    return {"code": ARGS.code, "n_layers": cfg.n_layers, "pipeline_ok": pipeline() is not None,
            "missing_layers": _ranges(missing_layers()), "join_url": join_url(),
            "nodes": {k: {**{kk: vv for kk, vv in v.items() if kk not in ("ws", "fingerprints", "files")},
                          "live": live(v), "hb_age_s": round(now - v["last_hb"], 1)}
                      for k, v in nodes.items()}}


@app.get("/inventory")
def inventory():
    """Audit what each node holds against the manifest recorded at slice time. The coordinator holds
    no layer weights, so ground truth is the manifest's content hashes, not files on this disk."""
    man = json.load(open(f"{ARGS.shards}/manifest.json"))
    truth = {int(k): v for k, v in man["layer_hashes"].items()}
    out, claimed, problems = {}, [], []
    for name, n in nodes.items():
        if not live(n):
            continue
        got = {int(k): v for k, v in (n.get("fingerprints") or {}).items()}
        a, b = n["layers"]
        wrong = [i for i, h in got.items() if truth.get(i) != h]
        out[name] = {"layers": [a, b], "n_layers": b - a, "rss_mb": n.get("rss_mb"),
                     "shard_dir": n.get("shard_dir"),
                     "layer_files_present": sum(1 for f in n.get("files", []) if f.startswith("layer_")),
                     "files": n.get("files", []),
                     "fingerprints_verified": len(got) - len(wrong), "fingerprints_wrong": wrong,
                     "layer_hashes": {str(k): v for k, v in sorted(got.items())},
                     "holds_embedding_or_head": any(f.startswith("head") for f in n.get("files", []))}
        claimed += list(range(a, b))
        if set(got) != set(range(a, b)):
            problems.append(f"{name}: fingerprints {sorted(got)} != claimed range {a}-{b-1}")
        if wrong:
            problems.append(f"{name}: layers {wrong} do not match the real shard files")
        if out[name]["layer_files_present"] > b - a:
            problems.append(f"{name}: has {out[name]['layer_files_present']} layer files but only owns {b-a}")
    dupes = sorted({i for i in claimed if claimed.count(i) > 1})
    if dupes:
        problems.append(f"layers held by more than one node: {dupes}")
    missing = sorted(set(range(cfg.n_layers)) - set(claimed))
    coord_layers = [f for f in os.listdir(ARGS.shards) if f.startswith("layer_")]
    if coord_layers:
        problems.append(f"coordinator itself holds {len(coord_layers)} layer files")
    return {"n_layers": cfg.n_layers, "coordinator_holds_layers": len(coord_layers),
            "nodes": out, "duplicated_layers": dupes,
            "unclaimed_layers": missing, "problems": problems,
            "disjoint_and_complete": not problems and not dupes and not missing}


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
    try:
        import segno
        print()
        segno.make(join_url(), error="m").terminal(compact=True)   # writes to stdout, returns None
    except ImportError:
        print("\n  (pip install segno to print a scannable join code here)")
    print(f"  SCAN TO JOIN, or open:  {join_url()}")
    print(f"\n  LOBBY CODE: {ARGS.code}   expecting {ARGS.expected} nodes for {cfg.n_layers} layers")
    print(f"  telemetry: {'exporting to ' + ARGS.otlp if ARGS.otlp else 'off (set OTEL_EXPORTER_OTLP_ENDPOINT or --otlp)'}\n")
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port, ws_max_size=None)
