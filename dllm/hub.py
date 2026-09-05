"""Coordinator: lobby + hub + embed/lm_head/sampling + OpenAI endpoint + shard server + planner.
python -m dllm.hub --shards hub_shards --dist dist --utilization 0.8 --port 8000"""
import contextlib
import argparse, asyncio, collections, json, os, random, re, socket, string, time, traceback, uuid
import torch
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse, PlainTextResponse
from transformers import AutoTokenizer
from dllm import placement
from dllm.model import Cfg, Head, sample
from dllm.observe import Observability
from dllm.wire import pack, unpack, to_bytes, from_bytes

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="shards")
ap.add_argument("--dist", default=None,
                help="directory of layer shards to hand out to joining nodes. Kept separate from "
                     "--shards so the coordinator itself never has to hold the whole model. Delete "
                     "it once every node has loaded.")
ap.add_argument("--expected", type=int, default=int(os.getenv("EXPECTED_NODES", 0)),
                help="deprecated. Used as --min-nodes when that flag is not given.")
ap.add_argument("--utilization", type=float, default=float(os.getenv("UTILIZATION", placement.UTIL_DEFAULT)),
                help="0..1. Share of each device the planner may use: RAM when it allocates layers, "
                     "compute duty cycle when it dispatches. 0.6 keeps phones responsive.")
ap.add_argument("--min-layers", type=int, default=placement.MIN_LAYERS,
                help="a node holding fewer layers than this is not worth its wire hops")
ap.add_argument("--min-nodes", type=int, default=None, help="force at least this many nodes into the pipeline (demos)")
ap.add_argument("--concurrency", type=int, default=0,
                help="design point. 0 = follow the observed in-flight EMA, N = plan for N concurrent requests")
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
if not 0 < ARGS.utilization <= 1:
    ap.error(f"--utilization must be in (0, 1], got {ARGS.utilization}")
if ARGS.min_layers < 1:
    ap.error(f"--min-layers must be at least 1, got {ARGS.min_layers}")
if ARGS.min_nodes is None:
    ARGS.min_nodes = ARGS.expected if ARGS.expected > 0 else placement.MIN_NODES
    if ARGS.expected > 0:
        print(f"--expected is deprecated; using it as --min-nodes {ARGS.expected}")

app = FastAPI(title="dllm hub")
cfg = Cfg.load(f"{ARGS.shards}/config.json")
tok = AutoTokenizer.from_pretrained(ARGS.shards)
head = Head(cfg, ARGS.shards, ARGS.device)
gen_cfg = json.load(open(f"{ARGS.shards}/generation_config.json")) if os.path.exists(f"{ARGS.shards}/generation_config.json") else {}
EOS = set(gen_cfg.get("eos_token_id", [tok.eos_token_id]) if isinstance(gen_cfg.get("eos_token_id"), list) else [gen_cfg.get("eos_token_id", tok.eos_token_id)])

MODEL_NAME = (json.load(open(f"{ARGS.shards}/manifest.json")).get("repo", "dllm")
              if os.path.exists(f"{ARGS.shards}/manifest.json") else "dllm")


def _layer_bytes() -> int:
    """fp32 bytes of one layer shard, the unit the RAM planner works in. The manifest records file
    sizes at slice time, so the hub knows this without holding any layer itself."""
    man = f"{ARGS.shards}/manifest.json"
    if os.path.exists(man):
        files = json.load(open(man)).get("files", {})
        if "layer_00.npz" in files:
            return int(files["layer_00.npz"])
    for d in (ARGS.dist, ARGS.shards):
        if d and os.path.exists(f"{d}/layer_00.npz"):
            return os.path.getsize(f"{d}/layer_00.npz")
    print("  no manifest or layer_00.npz found; RAM planning assumes a 0.5B layer")
    return 59652874


LAYER_BYTES = _layer_bytes()
HB_TIMEOUT = 5.0  # a node that has not sent a heartbeat this recently is treated as gone
MAX_BATCH = placement.MAX_BATCH

nodes = {}          # name -> record (spec 2.2) plus ws, last_hb, hold, not_before, pending_gen, told
pending = {}        # (req, hop) or batch key -> Future for a forward reply
waiters = {}        # (kind, name, gen) -> Future; apply_plan waits here for "prefetched" and "ready" replies
events = []         # list of asyncio.Queue for /events SSE listeners
# Requests run concurrently. Every node keys its KV cache by request id and forward_all keys its
# pending futures by (request, hop), so two requests interleave on the wire without touching each
# other's state. The only thing that pauses them is a rebalance: gen_gate closes while the cluster
# drains, so a request never straddles two layer layouts.
gen_gate = asyncio.Event(); gen_gate.set()
inflight = 0
plan_lock = asyncio.Lock()
head_ms_ema = 5.0   # logits + sample + embed per decode token on the hub, a term of every prediction
c_ema = 1.0         # slow EMA of in-flight requests: the design point when --concurrency is 0
class_prior = {}    # device class -> EMA of ready.ms_per_layer, so the second phone is costed like the first
plan_state = {"gen": 0, "assignments": {}, "order": [], "standby": {}, "util": ARGS.utilization,
              "min_layers": ARGS.min_layers, "min_nodes": ARGS.min_nodes, "last_applied": 0.0,
              "last_plan": None, "last_reason": "", "rebalancing": False, "provisional": False}


def emit(ev):
    ev["ts"] = time.time()
    for q in events:
        q.put_nowait(ev)


def _ema(old, v, alpha=placement.EMA_ALPHA):
    return v if old is None else old + alpha * (v - old)


def _json_default(o):
    return sorted(o) if isinstance(o, (set, frozenset)) else str(o)


def live(n):
    return n["ws"] is not None and n["ready"] and (time.time() - n["last_hb"]) < HB_TIMEOUT


obs = Observability(endpoint=ARGS.otlp, model=MODEL_NAME, n_layers=cfg.n_layers, node_id=ARGS.node_id,
                    nodes=lambda: [(k, v, live(v)) for k, v in nodes.items() if v.get("layers")])


@app.on_event("shutdown")
def _flush_telemetry():
    obs.shutdown()


def _assigned_live():
    """Nodes of the committed plan that are live and hold exactly what the plan says, by start layer."""
    out = []
    for name, (a, b) in sorted(plan_state["assignments"].items(), key=lambda kv: kv[1][0]):
        n = nodes.get(name)
        if n is not None and live(n) and n["layers"] == [a, b]:
            out.append((a, b, n))
    return out


def missing_layers() -> list[int]:
    """Layers no live node of the committed plan holds. This is the whole reason a request gets refused, so say it."""
    have = set()
    for a, b, _ in _assigned_live():
        have |= set(range(a, b))
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
    """Ordered live nodes of the committed plan. Valid only if they tile [0, n_layers) exactly.
    Standby and joining nodes are never here, and a node whose heartbeat has stopped is excluded even
    if its socket is still open, so a wedged or unreachable device fails the request instead of hanging it."""
    want, out = 0, []
    for a, b, n in _assigned_live():
        if a != want:
            return None
        out.append(n); want = b
    return out if want == cfg.n_layers else None


def _fmt(assignments) -> str:
    return ", ".join(f"{m} {a}-{b - 1}" for m, (a, b) in sorted(assignments.items(), key=lambda kv: kv[1][0]))


# --- node records ---------------------------------------------------------------------------------
def _record(name, hdr):
    """A fresh node record. Everything the planner reads lives here, plus the socket and hub bookkeeping."""
    return {"name": name, "ws": None, "device": hdr.get("device"), "ram_gb": hdr.get("ram_gb"),
            "ms_per_layer": None, "battery": None, "thermal": None, "mem_pct": None, "mem_available_bytes": None,
            "mem": None, "rss_mb": None, "cache_reqs": None, "layers": None, "present": False, "ready": False,
            "ineligible": False, "ema_ms_per_layer": None, "ema_samples": 0, "ema_wire_ms": None, "ema_batch": {},
            "disk": set(), "reassign": False, "bw_bps": None, "load_s_per_layer": None, "ready_at": None,
            "role": "joining", "batch": False, "train": False, "files": [], "fingerprints": {},
            "last_hb": time.time(), "hold": collections.deque(), "not_before": 0.0, "pending_gen": 0,
            "told": None, "move_to": None, "absent_since": None, "mem_penalty_suppressed": False}


def _layer_ids(files) -> set[int]:
    return {int(m.group(1)) for f in files for m in [re.fullmatch(r"layer_(\d+)\.npz", f)] if m}


def _public(n):
    """A record as /status shows it: no socket, no hash tables, JSON-friendly containers."""
    out = {k: v for k, v in n.items() if k not in ("ws", "fingerprints", "files", "hold")}
    out["disk"] = sorted(n["disk"])
    return out


def recs():
    """Planner input: a copy of every record minus the socket, with live computed now."""
    out = {}
    for name, n in nodes.items():
        r = {k: v for k, v in n.items() if k not in ("ws", "hold")}
        r["live"] = live(n)
        out[name] = r
    return out


def _priors():
    """Class priors, with ms_per_layer replaced by what this cluster's nodes of that class have benched."""
    return {cls: {**p, "ms_per_layer": class_prior.get(cls, p["ms_per_layer"])} for cls, p in placement.PRIORS.items()}


def _observe_hop(name, compute_ms, wire_ms, n, batch):
    """Running cost per node from real hops. The join bench is one forward on an idle device; this is
    what the device does under the load it actually has, which is what the planner should believe."""
    rec = nodes.get(name)
    if rec is None or n != 1 or not rec["layers"]:
        return
    per = compute_ms / max(1, rec["layers"][1] - rec["layers"][0])
    rec["ema_batch"][batch] = _ema(rec["ema_batch"].get(batch), per)
    if batch == 1:
        rec["ema_ms_per_layer"] = _ema(rec["ema_ms_per_layer"], per)
        rec["ema_samples"] += 1
        rec["ema_wire_ms"] = _ema(rec["ema_wire_ms"], wire_ms)


def _on_ready(name, n, hdr):
    now = time.time()
    g = hdr.get("gen")
    if hdr.get("error"):
        # The node could not load what we asked for (404 shard, bad zip, OOM). It holds nothing now;
        # fail the switch waiter at once instead of letting it run out its deadline.
        print(f"  {name}: load of {hdr.get('layers')} failed: {hdr['error']}", flush=True)
        n.update(layers=None, ready=False, files=[], fingerprints={})
        fut = waiters.pop(("ready", name, n["pending_gen"] if g is None else g), None)
        if fut is not None and not fut.done():
            fut.set_exception(RuntimeError(f"{name}: {hdr['error']}"))
        return
    n["ms_per_layer"] = hdr.get("ms_per_layer")
    if n["ms_per_layer"] and n["ms_per_layer"] > 0:
        cls = placement.device_class(n["device"] or "")
        class_prior[cls] = _ema(class_prior.get(cls), n["ms_per_layer"], 0.5)
    if hdr.get("download_bytes", 0) > 0 and hdr.get("download_s"):
        n["bw_bps"] = _ema(n["bw_bps"], hdr["download_bytes"] / hdr["download_s"])
    layers = [int(v) for v in hdr["layers"]]
    if hdr.get("load_s") is not None and layers[1] > layers[0]:
        n["load_s_per_layer"] = _ema(n["load_s_per_layer"], hdr["load_s"] / (layers[1] - layers[0]))
    if g is not None and g not in (n["pending_gen"], plan_state["gen"]):
        # an answer to an assign we have since superseded; the bench above is still real
        print(f"  {name}: ready for gen {g} (want {n['pending_gen']}), bench noted, otherwise ignored", flush=True)
        return
    n.update(layers=layers, ready=True, ready_at=now, batch=bool(hdr.get("batch")), train=bool(hdr.get("train")),
             reassign=n["reassign"] or bool(hdr.get("reassign")), rss_mb=hdr.get("rss_mb"),
             shard_dir=hdr.get("shard_dir"), files=hdr.get("files", []), fingerprints=hdr.get("fingerprints", {}))
    # files is the true listing after drop_foreign_shards, so replace, never union: a stale superset
    # makes the planner cost a move back onto dropped layers at zero bytes and skip the prefetch.
    n["disk"] = _layer_ids(n["files"])
    if plan_state["assignments"].get(name) == layers:
        n["role"] = "active"
    elif not n["reassign"]:
        n["role"] = "standby (legacy)"   # loaded a range the plan has not given it; never routed to
    emit({"t": "ready", "node": name, "layers": layers, "ms_per_layer": n["ms_per_layer"], "gen": g})
    _resolve("ready", name, n["pending_gen"] if g is None else g, hdr)
    schedule_replan("ready", 0)


def _on_hb(n, hdr):
    now = time.time()
    n["last_hb"] = now
    mem = hdr.get("mem") or n["mem"] or {}
    hold = n["hold"]
    hold.append((now, hdr.get("battery"), hdr.get("thermal"), (hdr.get("mem") or {}).get("sys_percent")))
    while hold and now - hold[0][0] > placement.HOLD_S:
        hold.popleft()
    # Held values: the worst reading over the window, so one good heartbeat cannot un-evict a phone
    # that is about to throttle again.
    bats = [b for _, b, _, _ in hold if b is not None]
    ths = [t for _, _, t, _ in hold if t is not None]
    mps = [p for _, _, _, p in hold if p is not None]
    n.update(battery=min(bats) if bats else None, thermal=max(ths) if ths else None, mem_pct=max(mps) if mps else None,
             mem_available_bytes=mem.get("sys_available_bytes"), mem=hdr.get("mem", n["mem"]),
             rss_mb=hdr.get("rss_mb", n["rss_mb"]), cache_reqs=hdr.get("cache_reqs", n["cache_reqs"]))
    n["mem_penalty_suppressed"] = n["ready_at"] is not None and now - n["ready_at"] < placement.MEM_PENALTY_GRACE_S
    n["ineligible"] = placement.ineligible_next(n["ineligible"], n["battery"], n["thermal"])


def _waiter(kind, name, gen):
    fut = asyncio.get_event_loop().create_future()
    waiters[(kind, name, gen)] = fut
    return fut


def _resolve(kind, name, gen, hdr):
    fut = waiters.pop((kind, name, gen), None)
    if fut is not None and not fut.done():
        fut.set_result(hdr)


async def _send(n, hdr) -> bool:
    try:
        await n["ws"].send_bytes(pack(hdr))
        return True
    except Exception as e:
        print(f"  {n['name']}: send {hdr.get('t')} failed: {e!r}", flush=True)
        return False


async def _close(ws, code, reason):
    with contextlib.suppress(Exception):
        await ws.close(code=code, reason=reason)


def _left(name, ws, why):
    """One node's socket is gone. Idempotent by socket identity, because a reconnect can replace the
    socket before the old handler's finally block runs."""
    n = nodes.get(name)
    if n is None or n["ws"] is not ws:
        return
    n.update(ws=None, present=False, ready=False, role="absent", absent_since=time.time())
    rng = f"layers {n['layers'][0]}-{n['layers'][1] - 1}" if n["layers"] else "no layers"
    print(f"  node {name} left ({why}; {rng}) at {time.strftime('%H:%M:%S')}", flush=True)
    emit({"t": "leave", "node": name})
    if name in plan_state["assignments"]:
        # Only a pipeline member takes requests down with it. A standby, joining or parked legacy
        # node leaving changes nothing about the layout the in-flight requests are running on.
        for f in list(pending.values()):
            if not f.done():
                f.set_exception(RuntimeError(f"node {name} left"))
    for k in [k for k in waiters if k[1] == name]:
        f = waiters.pop(k)
        if not f.done():
            f.set_exception(RuntimeError(f"node {name} left"))
    schedule_replan("leave", placement.LEAVE_GRACE_S)


# --- join -----------------------------------------------------------------------------------------
def _evict_first_range():
    """The committed range of the member the planner would drop first. Only a legacy node with no
    other home ever asks for this, so a sort over the last plan's numbers is enough."""
    # ponytail: mirrors placement.evict_order's key; call it instead if this ever matters more.
    per = (plan_state["last_plan"] or {}).get("per_node", {})
    act = [m for m in plan_state["assignments"] if m in nodes]
    if not act:
        return None
    key = lambda m: (nodes[m]["ineligible"], per.get(m, {}).get("c_ms_per_layer", 0),
                     per.get(m, {}).get("busy_fraction", 0),
                     plan_state["assignments"][m][0] - plan_state["assignments"][m][1])
    return list(plan_state["assignments"][max(act, key=key)])


def _legacy_range(name):
    """A node from an older build must be told a range at hello and cannot be moved later, so pick
    the best guess now: what the planner would give it, else a gap, else a duplicate of the range the
    planner would evict first (it then sits as 'standby (legacy)', never routed to)."""
    try:
        p = compute_plan()
        if name in p["assignments"]:
            return list(p["assignments"][name])
    except Exception:
        traceback.print_exc()
    gaps = missing_layers()
    if gaps:
        end = gaps[0]
        while end + 1 in gaps:
            end += 1
        return [gaps[0], end + 1]
    return _evict_first_range() or [0, min(cfg.n_layers, max(plan_state["min_layers"], 1))]


async def _hello(name, hdr, ws):
    old = nodes.get(name)
    n = old if old is not None else _record(name, hdr)
    if old is not None and old["ws"] is not None and old["ws"] is not ws:
        asyncio.ensure_future(_close(old["ws"], 4002, "replaced by a new connection"))
    # A new socket means the node holds nothing we can vouch for until its ready says so. Keeping the
    # old range here made the planner credit RAM the node no longer uses and cost its reload at zero.
    n.update(ws=ws, present=True, ready=False, role="joining", last_hb=time.time(), absent_since=None,
             device=hdr.get("device", n["device"]), ram_gb=hdr.get("ram_gb", n["ram_gb"]),
             reassign=bool(hdr.get("reassign")), layers=None, files=[], fingerprints={})
    if "disk" in hdr:
        n["disk"] = {int(i) for i in hdr["disk"]}
    nodes[name] = n
    want = [int(v) for v in hdr["layers"]] if hdr.get("layers") else None
    mine = plan_state["assignments"].get(name)
    move, n["move_to"] = n["move_to"], None
    # 3.6: a committed member gets its range back at once (hello.layers is only a request, the plan
    # wins); a legacy node must be answered now because it cannot be parked or moved on a live socket;
    # everyone else heartbeats until the debounced replan answers with assign or standby.
    if move:
        rng = move
    elif mine is not None:
        rng = list(mine)
    elif not n["reassign"]:
        rng = want or _legacy_range(name)
    else:
        rng = None
    if rng is not None:
        n.update(told=rng, pending_gen=plan_state["gen"], role="active" if rng == mine else n["role"])
        await _send(n, {"t": "assign", "layers": rng, "gen": plan_state["gen"]})
    emit({"t": "join", "node": name, "layers": rng, "reassign": n["reassign"]})
    schedule_replan("join", placement.JOIN_DEBOUNCE_S)
    return n


@app.websocket("/ws/node")
async def ws_node(ws: WebSocket):
    await ws.accept()
    hdr, _ = unpack(await ws.receive_bytes())
    if hdr.get("t") != "hello" or hdr.get("code") != ARGS.code:
        await ws.send_bytes(pack({"t": "error", "msg": "bad lobby code"})); await ws.close(); return
    name = hdr["name"]
    n = await _hello(name, hdr, ws)
    try:
        while True:
            hdr, payload = unpack(await ws.receive_bytes())
            t = hdr["t"]
            if t == "ready":
                _on_ready(name, n, hdr)
            elif t == "hb":
                _on_hb(n, hdr)
            elif t in ("fwd_batch_out", "fwd_out"):
                fut = pending.pop(hdr["key"] if t == "fwd_batch_out" else (hdr["req"], hdr["hop"]), None)
                if fut and not fut.done():
                    if "error" in hdr:
                        fut.set_exception(RuntimeError(f"{name}: {hdr['error']}"))
                    else:
                        fut.set_result((hdr, payload))
            elif t == "prefetched":
                _resolve("prefetched", name, hdr.get("gen"), hdr)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"  {name}: socket error {e!r}", flush=True)
    finally:
        _left(name, ws, "socket closed")


# --- planning -------------------------------------------------------------------------------------
def design_point():
    """(B, C): what batch size and concurrency the planner should optimise for right now."""
    C = ARGS.concurrency if ARGS.concurrency > 0 else max(1, min(64, round(c_ema)))
    return min(C, MAX_BATCH), C


def compute_plan(util=None, min_nodes=None, concurrency=None, force=False, provisional=False, exclude=()):
    """Preview only: what the planner would do right now. Nothing here touches a node."""
    B, C = (min(concurrency, MAX_BATCH), concurrency) if concurrency else design_point()
    rs = recs()
    for name in exclude:
        if name in rs:
            rs[name]["ineligible"] = True
    return placement.plan(rs, {"assignments": dict(plan_state["assignments"]), "order": list(plan_state["order"])},
                          cfg.n_layers, LAYER_BYTES, util or plan_state["util"], plan_state["min_layers"],
                          min_nodes or plan_state["min_nodes"], B, C, head_ms_ema, _priors(),
                          time.time(), plan_state["last_applied"], provisional, force)


_replan = {"task": None, "due": 0.0}


def schedule_replan(trigger, delay):
    """One pending replan at a time. A sooner request replaces a later one, a later one is absorbed."""
    due = time.time() + delay
    t = _replan["task"]
    if t is not None and not t.done():
        if due >= _replan["due"]:
            return
        t.cancel()

    async def fire():
        await asyncio.sleep(delay)
        _replan["task"] = None
        if plan_lock.locked():
            schedule_replan(trigger, 2.0)   # a rebalance is in progress; retry soon rather than next periodic tick
            return
        # A node that just dropped gets its grace no matter what else asked for a plan: a WiFi blip
        # must not hand its layers to someone else one second before it reconnects.
        now = time.time()
        grace = max((n["absent_since"] + placement.LEAVE_GRACE_S - now
                     for n in nodes.values() if n["ws"] is None and n["absent_since"]), default=0.0)
        if grace > 0:
            schedule_replan(trigger, grace)
            return
        try:
            await replan(trigger)
        except Exception:
            traceback.print_exc()

    _replan["due"] = due
    _replan["task"] = asyncio.ensure_future(fire())


async def replan(trigger, util=None, min_nodes=None, concurrency=None, force=False, apply=True):
    # A plan built on class priors gets one free follow-up once the real bench arrives, so a phone
    # that benched slower than its class is corrected without waiting out margin and cooldown.
    waive = plan_state["provisional"] and trigger == "ready"
    if waive:
        plan_state["provisional"] = False
    plan = compute_plan(util, min_nodes, concurrency, force, provisional=waive)
    plan_state["last_plan"] = plan
    plan_state["last_reason"] = "; ".join(plan["reasons"])
    emit({"t": "plan", "trigger": trigger, "would_apply": plan["would_apply"], "reasons": plan["reasons"],
          "assignments": plan["assignments"], "standby": plan["standby"], "missing_layers": plan["missing_layers"]})
    if apply and plan["would_apply"]:
        await apply_plan(plan)
    else:
        await _park(plan)
    return plan


async def _park(plan):
    """Tell every eligible node the plan left out that it is on standby. Only nodes that are not in
    the committed pipeline: a preview that would evict someone must not park a serving member."""
    # excluded nodes (battery, thermal) are parked too: an evicted member must learn it is out
    for name in list(plan["standby"]) + [m for m in plan["excluded"] if m not in plan["standby"]]:
        n = nodes.get(name)
        if n is None or n["ws"] is None or name in plan_state["assignments"] or n["role"] == "reassigning":
            continue
        reason = plan["standby_reasons"].get(name) or plan["excluded"].get(name, "")
        if n["reassign"]:
            if n["role"] == "standby" and plan_state["standby"].get(name) == reason:
                continue
            n["role"] = "standby"
            await _send(n, {"t": "standby", "gen": plan_state["gen"], "reason": reason})
        elif n["layers"]:
            n["role"] = "standby (legacy)"
        else:
            continue
        plan_state["standby"][name] = reason
        emit({"t": "standby", "node": name, "reason": reason})


def _mig(plan, name, key):
    return float((plan.get("migration") or {}).get(name, {}).get(key, 0) or 0)


async def _switch(plan, G, changed, prefetched):
    """Phase 3: hand every changed member its new range and wait for the ready that answers it.
    Returns the names that did not make it."""
    waits, failed = {}, []
    for m in changed:
        n = nodes.get(m)
        a, b = plan["assignments"][m]
        if n is None or n["ws"] is None:
            failed.append(m); continue
        if n["reassign"]:
            n.update(ready=False, role="reassigning", pending_gen=G, told=[a, b])
            waits[m] = _waiter("ready", m, G)
            if not await _send(n, {"t": "assign", "layers": [a, b], "gen": G}):
                failed.append(m)
        elif n["told"] != [a, b] and n["layers"] != [a, b]:
            # a legacy node cannot be moved on a live socket; its reconnect hello picks up this range
            n.update(move_to=[a, b], ready=False)
            await _close(n["ws"], 4001, "reassign")
    deadline = {m: 20 + 2 * (_mig(plan, m, "reload_s") + (0 if m in prefetched else _mig(plan, m, "download_s")))
                for m in waits}
    results = await asyncio.gather(*(asyncio.wait_for(f, deadline[m]) for m, f in waits.items()), return_exceptions=True)
    for m, r in zip(waits, results):
        if isinstance(r, BaseException) and m not in failed:
            failed.append(m)
            print(f"  {m}: did not send ready within {deadline[m]:.0f} s ({r!r})", flush=True)
    return failed


async def apply_plan(plan) -> str:
    """Move the cluster to `plan` without corrupting a request in flight: prefetch on the side, drain,
    switch every changed member, commit. Returns "ok" or why not."""
    if plan["missing_layers"]:
        return f"failed: layers {_ranges(plan['missing_layers'])} have no home"
    async with plan_lock:
        G = plan_state["gen"] + 1
        A = plan["assignments"]
        changed = [m for m, rng in A.items()
                   if m in nodes and (plan_state["assignments"].get(m) != rng or nodes[m]["layers"] != rng)]
        # phase 1: prefetch while the old pipeline keeps serving
        pre = {}
        for m in changed:
            n = nodes[m]
            if n["reassign"] and n["ws"] is not None and not set(range(*A[m])) <= n["disk"]:
                pre[m] = _waiter("prefetched", m, G)
                await _send(n, {"t": "prefetch", "layers": list(A[m]), "gen": G})
        got = await asyncio.gather(*(asyncio.wait_for(f, 30 + 2 * _mig(plan, m, "download_s")) for m, f in pre.items()),
                                   return_exceptions=True)
        for m, hdr in zip(pre, got):
            if isinstance(hdr, BaseException) or "error" in hdr:
                for k in [k for k in waiters if k[2] == G]:
                    waiters.pop(k).cancel()
                why = f"prefetch failed: {m}" + ("" if isinstance(hdr, BaseException) else f" ({hdr['error']})")
                plan["reasons"].append(why); plan_state["last_reason"] = why
                print(f"  plan gen {G} abandoned: {why}", flush=True)
                return why
            nodes[m]["disk"] |= set(range(*A[m]))
        # phase 2: drain. New generations wait at gen_gate; in-flight ones finish on the old layout.
        plan_state["rebalancing"] = True
        gen_gate.clear()
        try:
            t0 = time.time()
            while inflight and time.time() - t0 < placement.DRAIN_TIMEOUT_S:
                await asyncio.sleep(0.05)
            plan_state["gen"] = G
            # phase 3: switch, with one retry that plans around whoever failed
            failed = await _switch(plan, G, changed, pre)
            outcome = "ok"
            if failed:
                for m in failed:
                    if m in nodes:
                        nodes[m]["role"] = "standby"
                    plan["standby_reasons"][m] = "reassign failed"
                outcome = f"failed: {', '.join(failed)} did not send ready in time"
                try:
                    p2 = compute_plan(exclude=failed, force=True)
                except Exception:
                    traceback.print_exc(); p2 = None
                if p2 is not None and p2["assignments"] and not p2["missing_layers"]:
                    p2["reasons"].append(f"replanned without {', '.join(failed)}: reassign failed")
                    for m in failed:
                        p2["standby_reasons"][m] = "reassign failed"
                        if m not in p2["standby"] and m not in p2["assignments"]:
                            p2["standby"].append(m)
                    changed2 = [m for m in p2["assignments"] if m in nodes and nodes[m]["layers"] != p2["assignments"][m]]
                    failed2 = await _switch(p2, G, changed2, set())
                    plan = p2
                    if not failed2:
                        outcome = "ok"
                    else:
                        for m in failed2:
                            if m in nodes:
                                nodes[m]["role"] = "standby"
                            plan["standby_reasons"][m] = "reassign failed"
            # phase 4: commit
            plan_state.update(assignments={m: list(r) for m, r in plan["assignments"].items()}, order=list(plan["order"]),
                              standby={m: plan["standby_reasons"].get(m, "") for m in plan["standby"]},
                              last_applied=time.time(), provisional=bool(plan["provisional"]), last_plan=plan,
                              last_reason="; ".join(plan["reasons"]))
            for m in plan["assignments"]:
                n = nodes.get(m)
                if n is not None and n["ws"] is not None and n["layers"] == plan_state["assignments"][m]:
                    n["role"] = "active"
        finally:
            plan_state["rebalancing"] = False
            gen_gate.set()
        await _park(plan)
        emit({"t": "plan", "applied": True, "gen": G, "assignments": plan_state["assignments"],
              "standby": plan_state["standby"], "reasons": plan["reasons"]})
        sb = "; standby " + ", ".join(f"{m} ({r})" for m, r in plan_state["standby"].items()) if plan_state["standby"] else ""
        print(f"  plan gen {G} applied: {_fmt(plan_state['assignments']) or 'nothing'}{sb}", flush=True)
        return outcome


async def stale_watch():
    """Every 2 s: close silent sockets, purge old absent records, feed the concurrency EMA, and replan
    periodically so battery, thermal and memory drift get a look even when nobody joins or leaves."""
    global c_ema
    last_periodic = time.time()
    while True:
        await asyncio.sleep(2)
        now = time.time()
        c_ema += (inflight - c_ema) * (2 / placement.C_EMA_TAU_S)
        for name, n in list(nodes.items()):
            if n["ws"] is not None:
                # ponytail: legacy builds only start heartbeating after their load, so give one that is
                # still loading a long leash. Drop this once every phone runs a build that heartbeats at hello.
                limit = placement.STALE_S if (n["reassign"] or n["ready"]) else 180.0
                if now - n["last_hb"] > limit:
                    ws = n["ws"]
                    _left(name, ws, f"no heartbeat for {now - n['last_hb']:.0f} s")
                    asyncio.ensure_future(_close(ws, 4000, "stale"))
            elif now - (n["absent_since"] or now) > placement.ABSENT_TTL_S:
                del nodes[name]
        if now - last_periodic >= placement.PERIODIC_S:
            last_periodic = now
            schedule_replan("periodic", 0)


@app.on_event("startup")
async def _start_watch():
    asyncio.ensure_future(stale_watch())


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

_batch_pending = {}   # node name -> list of (req, pos, x_row, future)
_batch_task = {}      # node name -> asyncio.Task


async def _pace(node):
    """The compute half of the utilization knob. A node is not sent the next decode frame until it
    has idled (1/u - 1) times as long as the last one took, so its duty cycle stays at u."""
    await asyncio.sleep(max(0.0, node["not_before"] - time.time()))


def _paced(node, ms):
    node["not_before"] = time.time() + (ms / 1000) * (1 / plan_state["util"] - 1)


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
    await _pace(node)
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
    _paced(node, ms)
    _observe_hop(name, ms, wire_ms, 1, len(items))
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


async def forward_all(req, x, pos, trace=None, gen=None):
    """Run hidden x (1, n, hidden) through every node in layer order. Returns final hidden."""
    if gen is not None and gen != plan_state["gen"]:
        # the layout changed under this request; failing fast beats feeding one node's KV cache
        # with activations that were meant for another
        raise RuntimeError("cluster rebalanced")
    pipe = pipeline()
    if not pipe:
        raise RuntimeError(f"pipeline incomplete: missing layers {_ranges(missing_layers())}")
    n = x.shape[1]
    # Batch only if every node advertised it. A node from an older build ignores an unknown frame
    # and simply never answers, so one stale phone would hang the whole cluster. Falling back keeps
    # a mixed-version cluster correct, just without the throughput gain.
    if n == 1 and all(node.get("batch") for node in pipe):
        for hop, node in enumerate(pipe):
            name = node["name"]
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
        if n == 1:
            await _pace(node)
        t = time.time()
        await node["ws"].send_bytes(pack({"t": "fwd", "req": req, "hop": hop, "pos": pos, "n": n,
                                          "dtype": dt, "out_dtype": out_dt}, to_bytes(x, dt)))
        hdr, payload = await asyncio.wait_for(fut, timeout=120)
        t_end = time.time()
        x = from_bytes(payload, (1, n, cfg.hidden), hdr.get("dtype", out_dt))
        dt = out_dt
        name = node["name"]
        wire_ms = (t_end - t) * 1000 - hdr["ms"]
        if n == 1:
            _paced(node, hdr["ms"])
        _observe_hop(name, hdr["ms"], wire_ms, n, 1)
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
    global inflight, head_ms_ema
    await gen_gate.wait()
    gen = plan_state["gen"]
    inflight += 1
    req = uuid.uuid4().hex[:8]
    pipe = pipeline()
    try:
        for i in range(0, len(ids), PREFILL_CHUNK):
            chunk = ids[i:i + PREFILL_CHUNK]
            x = await forward_all(req, head.embed_tokens(chunk), i, trace, gen)
        pos = len(ids)
        for _ in range(max_tokens):
            t0 = time.perf_counter()
            lg = head.logits(x)
            nxt = sample(lg, temperature, top_p, top_k, None if seed is None else seed + pos)
            if nxt in EOS:
                break
            if trace:
                trace.first_token()
            emit({"t": "token", "req": req, "id": nxt, "pos": pos, "text": tok.decode([nxt])})
            yield nxt
            emb = head.embed_tokens([nxt])
            head_ms_ema = _ema(head_ms_ema, (time.perf_counter() - t0) * 1000, 0.1)
            x = await forward_all(req, emb, pos, trace, gen)
            pos += 1
    finally:
        inflight -= 1
        for node in pipe or []:
            try: await node["ws"].send_bytes(pack({"t": "reset", "req": req}))
            except Exception: pass


def _incomplete_503():
    gaps = missing_layers()
    unhoused = list((plan_state["last_plan"] or {}).get("missing_layers") or [])
    held = {k: f"{v['layers'][0]}-{v['layers'][1] - 1}" for k, v in nodes.items() if live(v)}
    if unhoused:
        detail = (f"layers {_ranges(unhoused)} have no home: remaining RAM at utilization {plan_state['util']} holds "
                  f"{cfg.n_layers - len(unhoused)} of {cfg.n_layers} layers, {len(unhoused) * LAYER_BYTES / 1e6:.0f} MB "
                  f"more is needed; raise --utilization or join a device")
    elif gaps:
        detail = (f"no live node holds layer(s) {_ranges(gaps)} of {cfg.n_layers}. "
                  f"Start a node for them, or scan {join_url()} on a phone.")
    else:
        detail = "nodes overlap or do not start at layer 0"
    return JSONResponse({"error": "pipeline incomplete", "detail": detail, "missing_layers": _ranges(gaps),
                         "unhoused_layers": _ranges(unhoused), "live_nodes": held, "join_url": join_url()}, 503)


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
    try:
        # a rebalance in progress looks like a pause, not an outage
        await asyncio.wait_for(gen_gate.wait(), 120)
    except TimeoutError:
        return JSONResponse({"error": "rebalancing", "detail": "rebalancing", "retry_after_s": 5}, 503)
    if pipeline() is None:
        return _incomplete_503()
    trace = obs.request(request_id=cid, input_tokens=len(ids), max_tokens=max_tokens, temperature=temp)
    finish = lambda n: "length" if n >= max_tokens else "stop"

    async def run_stream():
        n = 0
        try:
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


# --- planner endpoints ----------------------------------------------------------------------------
def _changes(plan):
    out = []
    for name in sorted(set(plan_state["assignments"]) | set(plan["assignments"])):
        frm, to = plan_state["assignments"].get(name), plan["assignments"].get(name, "standby")
        if frm == to:
            continue
        m = (plan.get("migration") or {}).get(name, {})
        out.append({"node": name, "from": frm, "to": to, "download_mb": round(m.get("download_bytes", 0) / 1e6, 1),
                    "download_s": round(m.get("download_s", 0), 1), "reload_s": round(m.get("reload_s", 0), 1)})
    return out


def _plan_view(plan):
    return {**plan, "gen": plan_state["gen"], "rebalancing": plan_state["rebalancing"], "changes": _changes(plan)}


@app.get("/plan")
def plan_preview(utilization: float | None = None, concurrency: int | None = None, min_nodes: int | None = None):
    """What the planner would do right now, with reasons. Applies nothing."""
    if utilization is not None and not 0 < utilization <= 1:
        return JSONResponse({"error": "utilization must be in (0, 1]"}, 422)
    return _plan_view(compute_plan(utilization, min_nodes, concurrency))


@app.post("/rebalance")
async def rebalance(request: Request):
    raw = await request.body()
    body = json.loads(raw) if raw else {}
    util = body.get("utilization")
    if util is not None and not 0 < util <= 1:
        return JSONResponse({"error": "utilization must be in (0, 1]"}, 422)
    if plan_lock.locked():
        return JSONResponse({"error": "a rebalance is already in progress", "rebalancing": True}, 409)
    # any knob given becomes the running value, so the next automatic replan uses it too
    if util is not None:
        plan_state["util"] = float(util)
    if body.get("min_nodes") is not None:
        plan_state["min_nodes"] = int(body["min_nodes"])
    if body.get("concurrency") is not None:
        ARGS.concurrency = int(body["concurrency"])
    t0 = time.time()
    plan = await replan("manual", force=bool(body.get("force")), apply=False)
    view = _plan_view(plan)   # the diff this call is about to make, taken before commit moves the baseline
    applied, outcome = False, ""
    if plan["would_apply"]:
        outcome = await apply_plan(plan)
        applied = outcome == "ok"
        view.update(plan_state["last_plan"], gen=plan_state["gen"], rebalancing=plan_state["rebalancing"])
    elif plan["missing_layers"]:
        return JSONResponse({**view, "applied": False, "elapsed_s": round(time.time() - t0, 2),
                             "outcome": f"failed: layers {_ranges(plan['missing_layers'])} have no home"}, 503)
    elif "no change" in plan["reasons"]:
        outcome = "nothing to change"
    else:
        outcome = "deferred: " + "; ".join(plan["reasons"])
    return {**view, "applied": applied, "outcome": outcome, "elapsed_s": round(time.time() - t0, 2)}


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
    name = next(f"node{i}" for i in range(1, 1000) if f"node{i}" not in nodes)   # never reuse a known name
    if not wants_html:
        return PlainTextResponse(setup_sh(name=name, host=host), media_type="text/plain")
    held = sum(b - a for a, b in plan_state["assignments"].values())
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
<p>Lobby <b>{ARGS.code}</b> &middot; {len(plan_state["assignments"])} node(s) already in &middot;
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
    lp = plan_state["last_plan"] or {}
    per = lp.get("per_node") or {}
    # a proposed-but-deferred plan carries the committed pipeline's numbers under "current"
    src = lp["current"] if lp and not lp.get("would_apply") and lp.get("current") else lp
    rs, priors = recs(), _priors()

    def costs(k):
        pn = per.get(k, {})
        c, w = pn.get("c_ms_per_layer"), pn.get("wire_ms")
        if c is None or w is None:
            try:
                c = placement.effective_cost(rs[k], priors)[0]
                w = placement.wire_ms(rs[k], priors)
            except Exception:
                c = w = None
        return {"c_ms_per_layer": c, "wire_ms": w, "busy_fraction": pn.get("busy_fraction") if k in plan_state["assignments"] else None}

    B, C = design_point()
    return {"code": ARGS.code, "n_layers": cfg.n_layers, "pipeline_ok": pipeline() is not None,
            "missing_layers": _ranges(missing_layers()), "join_url": join_url(),
            "plan": {"gen": plan_state["gen"], "util": plan_state["util"], "min_layers": plan_state["min_layers"],
                     "min_nodes": plan_state["min_nodes"], "concurrency": C,
                     "design_point": lp.get("design_point") or {"batch": B, "concurrency": C, "util": plan_state["util"], "head_ms": head_ms_ema},
                     "predicted_ms_per_token": src.get("predicted_ms_per_token"), "predicted_tok_s": src.get("predicted_tok_s"),
                     "utilization_pct": None if src.get("utilization") is None else round(100 * src["utilization"], 1),
                     "best_achievable": lp.get("best_achievable"), "cap_cost": lp.get("cap_cost"),
                     "active": [f"{m} {a}-{b - 1}" for m, (a, b) in sorted(plan_state["assignments"].items(), key=lambda kv: kv[1][0])],
                     "standby": [f"{m} ({r})" for m, r in plan_state["standby"].items()],
                     "missing_layers": _ranges(list(lp.get("missing_layers") or [])),
                     "rebalancing": plan_state["rebalancing"], "provisional": plan_state["provisional"],
                     "last_applied": plan_state["last_applied"], "last_reason": plan_state["last_reason"]},
            "nodes": {k: {**_public(v), **costs(k), "live": live(v), "hb_age_s": round(now - v["last_hb"], 1)}
                      for k, v in nodes.items() if v["ws"] is not None and v["layers"]},
            "candidates": [{"name": k, "role": v["role"], "device": v["device"], "reassign": v["reassign"],
                            "ms_per_layer": v["ms_per_layer"], "hb_age_s": round(now - v["last_hb"], 1)}
                           for k, v in nodes.items() if v["ws"] is not None and not v["layers"]],
            "absent": {k: round(now - v["absent_since"], 1) for k, v in nodes.items() if v["ws"] is None and v["absent_since"]}}


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
                yield f"data: {json.dumps(await q.get(), default=_json_default)}\n\n"
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
    print(f"\n  LOBBY CODE: {ARGS.code}   {cfg.n_layers} layers of {LAYER_BYTES / 1e6:.0f} MB; utilization {ARGS.utilization}, "
          f"min {ARGS.min_layers} layers/node, min {ARGS.min_nodes} node(s), "
          f"concurrency {'auto' if ARGS.concurrency == 0 else ARGS.concurrency}")
    print(f"  telemetry: {'exporting to ' + ARGS.otlp if ARGS.otlp else 'off (set OTEL_EXPORTER_OTLP_ENDPOINT or --otlp)'}\n")
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port, ws_max_size=None)
