"""Layer placement: who holds which layers, and whether changing that is worth the churn.

Pure functions over plain dicts. No I/O, no asyncio, no torch, no hub imports, so the whole
planner runs under pytest in milliseconds and the hub only has to hand it records.

The cost model is small on purpose. One decode token is a chain of hops; a hop costs the node's
compute for its layers, one round trip of wire, and the batch window the hub sleeps before every
frame. Compute per layer starts from the node's own bench at join and is replaced by an EMA of
what the hub actually saw once there are enough samples. Batching helps a Mac a lot and a phone
hardly at all, so every node carries a slope for how compute grows with batch size. Wire is per
hop, not per layer, which is the whole reason ten phones with one layer each lose to one laptop.

Member selection, the join decision and eviction are one mechanism: score every subset of the
candidates and take the best, with a tie band that prefers fewer nodes and fewer bytes moved.
"""
import itertools
import math
import time

LAYER_OVERHEAD = 1.25          # fp32 shard bytes to resident bytes (KV, numpy transposes, load transient)
RESERVE_BYTES = 1.5 * 2 ** 30  # OS + runtime per device
HOP_OVERHEAD_MS = 4.0          # the BATCH_WINDOW_S sleep every decode hop pays; measured wire_ms never includes it
MAX_BATCH = 16                 # same as the hub
MIN_LAYERS = 2
MIN_NODES = 1
UTIL_DEFAULT = 0.8
APPLY_MARGIN = 0.15
PAYBACK_HORIZON_S = 600
COOLDOWN_S = 30
TIE_BAND = 0.05
MAX_CANDIDATES = 12
HOLD_S = 30
LEAVE_GRACE_S = 8
JOIN_DEBOUNCE_S = 3
STALE_S = 15
ABSENT_TTL_S = 600
DRAIN_TIMEOUT_S = 60
MEM_PENALTY_GRACE_S = 30
PERIODIC_S = 60
EMA_ALPHA = 0.2                # hop EMAs
C_EMA_TAU_S = 300              # design-point EMA

PRIORS = {  # per device class: ms_per_layer, batch slope, wire per hop, download bandwidth, shard load time
    "cpu":   dict(ms_per_layer=2.6, slope=0.30, wire_ms=2.0,  bw_bps=100e6, load_s_per_layer=0.1),
    "phone": dict(ms_per_layer=5.9, slope=0.87, wire_ms=25.0, bw_bps=12e6,  load_s_per_layer=0.6),
}
# Before a node has benched itself all we know is its device string. Runtimes we have measured
# get their number; anything else gets a pessimistic guess so a mystery device is not handed
# half the model on faith.
DEVICE_MS_PRIOR = {"mps": 2.0, "phone-numpy": 12.0}
UNKNOWN_MS_PRIOR = 8.0
NUMPY_SLOPE = 0.9


# --- per-node cost inputs -------------------------------------------------------------------------

def device_class(device):
    d = device or ""
    return "phone" if d.startswith(("android", "phone")) else "cpu"


def prior_ms_per_layer(device, priors=PRIORS):
    d = device or ""
    for prefix, ms in DEVICE_MS_PRIOR.items():
        if d.startswith(prefix):
            return ms
    if d.startswith(("android", "phone", "cpu")):
        return priors[device_class(d)]["ms_per_layer"]
    return UNKNOWN_MS_PRIOR


def effective_cost(rec, priors=PRIORS):
    """ms per layer for one decode token, with penalties that make a struggling node look slower
    than its bench so the search sheds it before it fails outright.
    Returns (c, from_prior, factors)."""
    from_prior = False
    # A zero or negative bench is a clock artefact, not a free node: treat it like no bench at all,
    # otherwise 1/c in the allocator and P/util in predict divide by zero and every replan dies.
    ema, bench = rec.get("ema_ms_per_layer"), rec.get("ms_per_layer")
    if rec.get("ema_samples", 0) >= 5 and ema is not None and ema > 0:
        base = ema                              # what the hub actually observed beats the join bench
    elif bench is not None and bench > 0:
        base = bench
    else:
        base, from_prior = prior_ms_per_layer(rec.get("device"), priors), True
    bat, th, mem = rec.get("battery"), rec.get("thermal"), rec.get("mem_pct")
    f = {"base": base,
         "battery": 0.5 if bat is not None and bat < 20 else 0.0,
         "thermal": (0.5 if th is not None and th >= 3 else 0.0) + (1.0 if th is not None and th >= 4 else 0.0),
         "mem": 0.3 if mem is not None and mem > 90 and not rec.get("mem_penalty_suppressed") else 0.0}
    f["pen"] = 1 + f["battery"] + f["thermal"] + f["mem"]
    return base * f["pen"], from_prior, f


def batch_slope(rec, priors=PRIORS):
    """How compute per layer grows with batch size: comp(B) = comp(1) * (1 + a*(B-1)).
    Fitted from the hop EMAs when the node has served both batch 1 and a bigger batch."""
    eb = {int(k): v for k, v in (rec.get("ema_batch") or {}).items()}
    big = [k for k in eb if k >= 2]
    if big and 1 in eb and eb[1]:
        bmax = max(big)
        return min(1.0, max(0.05, (eb[bmax] / eb[1] - 1) / (bmax - 1)))
    if "numpy" in (rec.get("device") or ""):
        return NUMPY_SLOPE
    return priors[device_class(rec.get("device"))]["slope"]


def wire_ms(rec, priors=PRIORS):
    w = rec.get("ema_wire_ms")
    if w is None:
        w = priors[device_class(rec.get("device"))]["wire_ms"]
    return max(0.5, w)


def comp_ms(L, c, a, B):
    return L * c * (1 + a * (B - 1))


def ineligible_next(prev, battery, thermal):
    """Hard exclusion with a re-entry band, so a phone hovering around 10% battery does not
    flap in and out of the pipeline every heartbeat."""
    if thermal is not None and thermal >= 5:
        return True
    if battery is not None and battery < 10:
        return True
    if prev:
        return not ((thermal is None or thermal <= 2) and (battery is None or battery >= 15))
    return False


def ineligible_why(rec):
    th, bat = rec.get("thermal"), rec.get("battery")
    if th is not None and th >= 5:
        return f"thermal {th}"
    if bat is not None and bat < 10:
        return f"battery {bat}%"
    return f"recovering (battery {bat}, thermal {th}): needs battery >= 15% and thermal <= 2"


def candidate(rec):
    return bool(rec.get("present")) and not rec.get("ineligible") and rec.get("role") != "absent"


def ram_layers(rec, layer_bytes, util):
    """How many layers this node may hold under the cap. The static bound comes from ram_gb; the
    live bound from what the OS says is free right now, crediting back what the node's own
    loaded layers occupy, so a phone with a game open is never planned as if it were empty."""
    per = layer_bytes * LAYER_OVERHEAD
    r = math.floor((util * (rec.get("ram_gb") or 0) * 2 ** 30 - RESERVE_BYTES) / per)
    avail = rec.get("mem_available_bytes")
    if avail is not None:
        held = (rec["layers"][1] - rec["layers"][0]) if rec.get("layers") else 0
        r = min(r, math.floor((avail + held * per - 0.5 * 2 ** 30) / per))
    return max(r, 0)


def node_costs(rec, priors, layer_bytes, util):
    c, from_prior, factors = effective_cost(rec, priors)
    return {"c": c, "a": batch_slope(rec, priors), "w": wire_ms(rec, priors),
            "ram": ram_layers(rec, layer_bytes, util), "from_prior": from_prior, "factors": factors,
            "ineligible": bool(rec.get("ineligible"))}


# --- allocation for a fixed member set ------------------------------------------------------------

def _largest_remainder(weights, total):
    """Integer shares proportional to weights that sum exactly to total."""
    s = sum(weights.values())
    ideal = {m: total * w / s for m, w in weights.items()}
    L = {m: math.floor(v) for m, v in ideal.items()}
    short = total - sum(L.values())
    for m in sorted(weights, key=lambda m: (-(ideal[m] - L[m]), m))[:short]:
        L[m] += 1
    return L


def allocate_balanced(S, n, B, min_layers, costs):
    """Layers in inverse proportion to batch-adjusted cost, so every node finishes its hop at
    about the same time. That minimises the cycle time under load. Nodes that would fall under
    min_layers or over their RAM get pinned there and the rest is re-solved."""
    if any(costs[m]["ram"] < min_layers for m in S) or min_layers * len(S) > n:
        return None
    k = {m: costs[m]["c"] * (1 + costs[m]["a"] * (B - 1)) for m in S}
    # Two kinds of pin. RAM is hard: a node cannot take more. min_layers is soft: a node pinned there
    # can still take more if layers are left over, which happens when the fast node hits its RAM
    # ceiling and everyone else was under the minimum in the same round.
    hard, soft, free = {}, {}, list(S)
    for _ in range(4 * len(S) + 4):
        rem = n - sum(hard.values()) - sum(soft.values())
        if not free:
            if rem == 0:
                return {**hard, **soft}
            if rem < 0 or not soft:
                return None
            free, soft = list(soft), {}     # hand the leftover back to the nodes held at the minimum
            continue
        L = _largest_remainder({m: 1 / k[m] for m in free}, rem)
        moved = False
        for m in free:
            if L[m] > costs[m]["ram"]:
                hard[m], moved = costs[m]["ram"], True
            elif L[m] < min_layers:
                soft[m], moved = min_layers, True
        if not moved:
            return {**hard, **soft, **L}
        free = [m for m in free if m not in hard and m not in soft]
    return None


def allocate_cheapest(S, n, min_layers, costs):
    """Everyone gets the minimum, then the cheapest node takes as much as it can hold, then the
    next. Minimises the serial sum, which is what one user waiting on one token feels."""
    if any(costs[m]["ram"] < min_layers for m in S):
        return None
    L = {m: min_layers for m in S}
    remaining = n - min_layers * len(S)
    if remaining < 0:
        return None
    for m in sorted(S, key=lambda m: (costs[m]["c"], m)):
        give = min(costs[m]["ram"] - L[m], remaining)
        if give > 0:
            L[m] += give
            remaining -= give
    return L if remaining == 0 else None


# --- layout, prediction, search -------------------------------------------------------------------

def _churn(order, counts, disk, layer_bytes):
    out, start = {}, 0
    for m in order:
        rng = set(range(start, start + counts[m]))
        out[m] = layer_bytes * len(rng - disk.get(m, set()))
        start += counts[m]
    return out


def layout(counts, current_order, disk, layer_bytes):
    """Contiguous ranges in an order that moves the fewest bytes: nodes already in the pipeline
    keep their relative order, newcomers slot in wherever their disk already has the files."""
    order = [m for m in current_order if m in counts]
    for m in sorted(m for m in counts if m not in order):
        best_slot, best_bytes = 0, None
        for slot in range(len(order) + 1):
            total = sum(_churn(order[:slot] + [m] + order[slot:], counts, disk, layer_bytes).values())
            if best_bytes is None or total <= best_bytes:      # <= so ties go to the end
                best_slot, best_bytes = slot, total
        order.insert(best_slot, m)
    ranges, start = {}, 0
    for m in order:
        ranges[m] = [start, start + counts[m]]
        start += counts[m]
    return ranges, _churn(order, counts, disk, layer_bytes)


def predict(ranges, costs, B, C, util, head_ms):
    """Per-token numbers for one layout at a design point of B rows per frame and C requests in
    flight. cycle is the time between successive frames at any node: the slowest node's compute
    (stretched by the cap, which the hub enforces by pacing) or, if there are not enough frames
    in flight to fill the pipeline, one lap of the whole chain."""
    if not ranges:
        return None
    L = {m: b - a for m, (a, b) in ranges.items()}
    comp = {m: comp_ms(L[m], costs[m]["c"], costs[m]["a"], B) for m in ranges}
    lat = head_ms + sum(comp[m] + costs[m]["w"] + HOP_OVERHEAD_MS for m in ranges)
    P = max(comp.values())
    cycle = max(P / util, lat / math.ceil(C / B))
    comp1 = {m: comp_ms(L[m], costs[m]["c"], costs[m]["a"], 1) for m in ranges}
    lat1 = head_ms + sum(comp1[m] + costs[m]["w"] + HOP_OVERHEAD_MS for m in ranges)
    p16 = max(comp_ms(L[m], costs[m]["c"], costs[m]["a"], MAX_BATCH) for m in ranges)
    per = {m: {"busy_ms": comp[m], "busy_fraction": comp[m] / cycle, "layers": list(ranges[m]), "n_layers": L[m]}
           for m in ranges}
    return {"ms_per_token": max(lat1, max(comp1.values()) / util),
            "ms_per_token_uncapped_pacing": lat1,
            "lat_b": lat, "cycle_ms": cycle, "tok_s": 1000 * B / cycle,
            "peak_tok_s": 1000 * MAX_BATCH / (p16 / util),
            "utilization": sum(p["busy_fraction"] for p in per.values()) / len(per),
            "per_node": per}


def prune_candidates(cands, costs, n):
    """The subset search is exponential, so past MAX_CANDIDATES keep the ones that would be
    cheapest at an even share. ponytail: a greedy add/drop search would lift the ceiling."""
    if len(cands) <= MAX_CANDIDATES:
        return sorted(cands)
    return sorted(sorted(cands, key=lambda m: (costs[m]["c"] * n / len(cands) + costs[m]["w"], m))[:MAX_CANDIDATES])


def pick_best(entries, cur_set):
    """Within TIE_BAND of the top score, prefer fewer nodes, then fewer bytes moved, then the set
    we already run. This is the anti-fragmentation rule and why a no-op beats a marginal shuffle."""
    top = max(e["score"] for e in entries)
    near = [e for e in entries if e["score"] >= (1 - TIE_BAND) * top]
    return min(near, key=lambda e: (len(e["members"]), e["churn"], 0 if set(e["members"]) == cur_set else 1, e["members"]))


def search(cands, current, n, min_layers, min_nodes, B, C, util, head_ms, costs, disk, layer_bytes):
    """Score every feasible member set. Returns (best entry or None, {name: best entry containing name})."""
    cands = prune_candidates(cands, costs, n)
    cur_assign = (current or {}).get("assignments") or {}
    cur_order = (current or {}).get("order") or sorted(cur_assign, key=lambda m: cur_assign[m][0])
    cur_set = set(cur_assign)
    cur_counts = {m: b - a for m, (a, b) in cur_assign.items()}
    entries, best_with = [], {}
    lo = max(1, min(min_nodes, len(cands)))
    hi = min(len(cands), n // min_layers)
    for k in range(lo, hi + 1):
        for S in itertools.combinations(cands, k):
            if sum(costs[m]["ram"] for m in S) < n:
                continue
            best_e = None
            for kind, counts in (("balanced", allocate_balanced(S, n, B, min_layers, costs)),
                                 ("cheapest", allocate_cheapest(S, n, min_layers, costs))):
                if counts is None:
                    continue
                # Dead band: EMA drift must never move a single layer. Within one layer of what
                # runs now, keep what runs now.
                if set(S) == cur_set and all(abs(counts[m] - cur_counts[m]) <= 1 for m in S):
                    counts = dict(cur_counts)
                ranges, churn = layout(counts, cur_order, disk, layer_bytes)
                pred = predict(ranges, costs, B, C, util, head_ms)
                e = {"members": S, "counts": counts, "ranges": ranges, "pred": pred, "score": pred["tok_s"],
                     "churn": sum(churn.values()), "churn_bytes": churn, "alloc": kind}
                if best_e is None or e["score"] > best_e["score"] or (
                        e["score"] == best_e["score"] and counts == cur_counts and best_e["counts"] != cur_counts):
                    best_e = e
            if best_e is None:
                continue
            entries.append(best_e)
            for m in S:
                if m not in best_with or best_e["score"] > best_with[m]["score"]:
                    best_with[m] = best_e
    if not entries:
        return None, best_with
    return pick_best(entries, cur_set), best_with


def evict_order(members, costs, pred):
    """Who goes first when there are more nodes than help: ineligible, then the slowest, then the
    busiest, then the one holding the least."""
    per = (pred or {}).get("per_node", {})
    return sorted(members, key=lambda m: (bool(costs[m].get("ineligible")), costs[m]["c"],
                                          per.get(m, {}).get("busy_fraction", 0.0),
                                          -per.get(m, {}).get("n_layers", 0)), reverse=True)


# --- when nothing fits, migration cost, hysteresis ------------------------------------------------

def partial_fill(current_ranges, live_members, costs, n):
    """What the cluster can still cover when RAM cannot house everything. Live members keep their
    ranges, neighbours stretch into the gaps up to their RAM, members without a range fill from
    the left. Informational only: the hub never applies a plan with missing layers."""
    ranges = {m: list(current_ranges[m]) for m in live_members if m in current_ranges}
    order = sorted(ranges, key=lambda m: ranges[m][0])
    room = lambda m: costs[m]["ram"] - (ranges[m][1] - ranges[m][0])
    for i, m in enumerate(order):
        left_end = ranges[order[i - 1]][1] if i else 0
        if ranges[m][0] > left_end:                                   # gap on my left: the left neighbour first
            if i:
                grow = min(room(order[i - 1]), ranges[m][0] - left_end)
                ranges[order[i - 1]][1] += grow
                left_end += grow
            ranges[m][0] -= min(room(m), ranges[m][0] - left_end)
    if order:
        last = order[-1]
        ranges[last][1] += min(room(last), n - ranges[last][1])
    covered = set()
    for a, b in ranges.values():
        covered |= set(range(a, b))
    for m in sorted(m for m in live_members if m not in ranges):
        free = [i for i in range(n) if i not in covered]
        if not free or costs[m]["ram"] <= 0:
            continue
        a = free[0]
        b = a
        while b < n and b in set(free) and b - a < costs[m]["ram"]:
            b += 1
        ranges[m] = [a, b]
        covered |= set(range(a, b))
    return ranges, [i for i in range(n) if i not in covered]


def migration(new_ranges, recs, priors, layer_bytes):
    """Seconds until the new layout serves, and the per-node work behind it. Nodes fetch in
    parallel so the max wins. ponytail: ignores the hub NIC; sum per link if the uplink saturates."""
    per = {}
    for m, (a, b) in new_ranges.items():
        r = recs[m]
        if list(r.get("layers") or []) == [a, b]:
            continue
        cls = priors[device_class(r.get("device"))]
        dl = layer_bytes * len(set(range(a, b)) - set(r.get("disk") or ()))
        per[m] = {"download_bytes": dl,
                  "download_s": dl / (r.get("bw_bps") or cls["bw_bps"]),
                  "reload_s": (b - a) * (r.get("load_s_per_layer") or cls["load_s_per_layer"])}
    return max((p["download_s"] + p["reload_s"] for p in per.values()), default=0.0), per


def should_apply(cur_score, new_score, migration_s, complete_now, changed, last_applied, now, provisional, force):
    """Hysteresis. A plan has to beat what runs by a margin, pay back its migration inside the
    horizon, and not follow the last change too closely; an incomplete pipeline skips all that."""
    if not changed:
        return False, ["no change"]
    if force:
        return True, ["forced"]
    if not complete_now:
        return True, ["pipeline incomplete: apply without margin"]
    reasons, ok = [], True
    gain = new_score / cur_score - 1
    if provisional:
        reasons.append("provisional: margin waived")
    elif gain >= APPLY_MARGIN:
        reasons.append(f"gain {gain * 100:.0f}% >= {APPLY_MARGIN * 100:.0f}%")
    else:
        reasons.append(f"gain {gain * 100:.0f}% < {APPLY_MARGIN * 100:.0f}%")
        ok = False
    limit = PAYBACK_HORIZON_S * max(gain, 0)
    if migration_s <= limit:
        reasons.append(f"payback: {migration_s:.0f} s <= {PAYBACK_HORIZON_S} * {max(gain, 0):.2f} = {limit:.0f} s")
    else:
        reasons.append(f"payback: {migration_s:.0f} s > {PAYBACK_HORIZON_S} * {max(gain, 0):.2f} = {limit:.0f} s")
        ok = False
    left = COOLDOWN_S - (now - last_applied)
    if not provisional and left > 0:
        reasons.append(f"cooldown {left:.0f} s left")
        ok = False
    return ok, reasons


# --- entry point ----------------------------------------------------------------------------------

def _tiles(assign, n):
    end = 0
    for m in sorted(assign, key=lambda m: assign[m][0]):
        a, b = assign[m]
        if a != end or b <= a:
            return False
        end = b
    return end == n


def _holds(r, rng):
    """Does this record hold `rng` for hysteresis purposes? Presence and a loaded range are what
    count, not the 5 s heartbeat liveness the router uses: one late heartbeat on a healthy member
    must not turn a deferred marginal plan into a margin-free reshuffle. The hub keeps a member's
    role at "active" until it moves it, parks it or loses its socket."""
    return bool(r.get("present")) and list(r.get("layers") or []) == list(rng) and (
        r.get("live") or r.get("role") == "active")


def _brief(pred):
    return f"{pred['ms_per_token']:.0f} ms/token, {pred['tok_s']:.1f} tok/s"


def _ranges_str(xs):
    """[8,9,10,20] -> '8-10,20'"""
    out, start = [], None
    for i, x in enumerate(xs):
        if start is None:
            start = x
        if i + 1 == len(xs) or xs[i + 1] != x + 1:
            out.append(str(start) if start == x else f"{start}-{x}")
            start = None
    return ",".join(out)


def plan(recs, current, n_layers, layer_bytes, util=UTIL_DEFAULT, min_layers=MIN_LAYERS, min_nodes=MIN_NODES,
         B=1, C=1, head_ms=5.0, priors=PRIORS, now=None, last_applied=0.0, provisional=False, force=False,
         _nested=False):
    """The planner. recs: node records (dict by name or a list), current: the hub's committed
    plan ({} on first boot). Returns the Plan dict; every key is always present."""
    recs = {r["name"]: r for r in (recs.values() if isinstance(recs, dict) else recs)}
    now = time.time() if now is None else now
    n = n_layers
    min_layers = max(1, int(min_layers))     # 0 would divide by zero below; the hub rejects it at startup too
    cur_assign = {m: list(r) for m, r in ((current or {}).get("assignments") or {}).items()}
    cur_order = list((current or {}).get("order") or sorted(cur_assign, key=lambda m: cur_assign[m][0]))
    cur = {"assignments": cur_assign, "order": cur_order}

    present = {m: r for m, r in recs.items() if r.get("present") and r.get("role") != "absent"}
    costs = {m: node_costs(r, priors, layer_bytes, util) for m, r in present.items()}
    disk = {m: set(r.get("disk") or ()) for m, r in present.items()}
    excluded = {}
    for m, r in recs.items():
        if m not in present:
            since = r.get("absent_since")
            excluded[m] = f"absent {now - since:.0f} s" if since is not None else "absent"
        elif r.get("ineligible"):
            excluded[m] = ineligible_why(r)
    cands = [m for m in recs if candidate(recs[m])]

    complete_now = bool(cur_assign) and _tiles(cur_assign, n) and all(
        m in costs and _holds(recs[m], cur_assign[m]) for m in cur_assign)
    cur_pred = predict(cur_assign, costs, B, C, util, head_ms) if complete_now else None

    reasons = []
    if len(cands) > n // min_layers:
        reasons.append(f"at most {n // min_layers} nodes: {len(cands)} would hold fewer than {min_layers} layers each")
    if len(cands) > MAX_CANDIDATES:
        reasons.append(f"searched the {MAX_CANDIDATES} cheapest of {len(cands)} candidates")
    best, best_with = search(cands, cur, n, min_layers, min_nodes, B, C, util, head_ms, costs, disk, layer_bytes)
    if best is None:
        # Last resort: a phone at thermal 5 still beats no pipeline at all.
        inel = [m for m in present if present[m].get("ineligible")]
        if inel:
            best, best_with = search(cands + inel, cur, n, min_layers, min_nodes, B, C, util, head_ms, costs, disk, layer_bytes)
            for m in (best["members"] if best else ()):
                if m in inel:
                    reasons.append(f"kept {m} despite {excluded.pop(m)}: nothing else can house the layers")

    missing, partial = [], {}
    if best is None:
        held = {m: cur_assign[m] for m in cands if m in cur_assign and _holds(recs[m], cur_assign[m])}
        partial, missing = partial_fill(held, cands, costs, n)
        if missing:
            reasons.append(f"layers {_ranges_str(missing)} have no home: RAM at utilization {util:g} holds "
                           f"{n - len(missing)} of {n} layers, {len(missing) * layer_bytes / 1e6:.0f} MB more is needed; "
                           f"raise --utilization or join a device")
        else:
            reasons.append(f"no split satisfies --min-layers {min_layers} and --min-nodes {min_nodes}")
        assignments = {}
    else:
        assignments = {m: list(r) for m, r in best["ranges"].items()}
    order = sorted(assignments, key=lambda m: assignments[m][0])
    pred = best["pred"] if best else None

    mig_s, mig = migration(assignments, recs, priors, layer_bytes)
    nodes_changed = [m for m in order if assignments[m] != cur_assign.get(m)
                     or list(recs[m].get("layers") or []) != assignments[m]]
    # A member that never loaded its range counts as a change even when the assignments match:
    # "no change" here would mean nobody ever re-sends the assign and the pipeline stays incomplete.
    changed = assignments != cur_assign or bool(nodes_changed)
    cur_score = cur_pred["tok_s"] if cur_pred else None
    # An ineligible member is a hard exclusion (thermal 5, battery under 10), so a plan that drops
    # it is not held to the margin: waiting 15% of gain to stop cooking a phone is the wrong trade.
    evicting = [m for m in cur_assign if complete_now and m in recs and recs[m].get("ineligible") and m not in assignments]
    for m in evicting:
        reasons.append(f"{m} ineligible ({excluded.get(m, '?')}): apply without margin")
    if best is None:
        would_apply = False
    elif evicting:
        would_apply = True
    elif len(cur_assign) < min_nodes and len(assignments) > len(cur_assign):
        would_apply = True
        reasons.append(f"active {len(cur_assign)} < min_nodes {min_nodes}: apply without margin")
    else:
        would_apply, rs = should_apply(cur_score, best["score"], mig_s, complete_now, changed,
                                       last_applied, now, provisional, force)
        reasons += rs

    standby = evict_order([m for m in cands if m not in assignments], costs, pred)
    standby_reasons = {}
    for m in standby:
        if costs[m]["ram"] < min_layers or m not in best_with:
            standby_reasons[m] = f"would hold fewer than {min_layers} layers"
        elif best is None:
            standby_reasons[m] = "no feasible plan"
        else:
            with_it = best_with[m]
            s = f"with it: {_brief(with_it['pred'])}; without: {_brief(pred)}"
            if with_it["score"] >= (1 - TIE_BAND) * best["score"]:
                s = "not needed: RAM fits without it; " + s
            standby_reasons[m] = s

    per_node = {}
    for m in present:
        rng = assignments.get(m)
        p = (pred or {}).get("per_node", {}).get(m, {}) if rng else {}
        per_node[m] = {"layers": rng, "n_layers": (rng[1] - rng[0]) if rng else 0,
                       "bytes": ((rng[1] - rng[0]) if rng else 0) * layer_bytes,
                       "busy_ms": p.get("busy_ms", 0.0), "busy_fraction": p.get("busy_fraction", 0.0),
                       "c_ms_per_layer": costs[m]["c"], "batch_slope": costs[m]["a"], "wire_ms": costs[m]["w"],
                       "ram_layers": costs[m]["ram"], "from_prior": costs[m]["from_prior"], "factors": costs[m]["factors"],
                       "role": "active" if rng else ("excluded" if m in excluded else "standby")}

    churn_bytes = best["churn_bytes"] if best else {}
    out = {
        "assignments": assignments, "order": order,
        "standby": standby, "standby_reasons": standby_reasons, "excluded": excluded,
        "predicted_ms_per_token": pred["ms_per_token"] if pred else None,
        "predicted_tok_s": pred["tok_s"] if pred else None,
        "peak_tok_s": pred["peak_tok_s"] if pred else None,
        "utilization": pred["utilization"] if pred else 0.0,
        "per_node": per_node,
        "design_point": {"batch": B, "concurrency": C, "util": util, "head_ms": head_ms},
        "reasons": reasons,
        "churn": {"bytes_to_move": sum(churn_bytes.get(m, 0) for m in nodes_changed), "nodes_changed": nodes_changed},
        "migration_s": mig_s, "migration": mig,
        "missing_layers": missing, "partial": partial,
        "improvement_vs_current": (best["score"] / cur_score - 1) if (best and cur_score) else None,
        "current": ({"predicted_ms_per_token": cur_pred["ms_per_token"], "predicted_tok_s": cur_pred["tok_s"],
                     "utilization": cur_pred["utilization"]} if cur_pred else None),
        "complete": bool(assignments) and _tiles(assignments, n) and not missing,
        "would_apply": would_apply,
        "provisional": any(costs[m]["from_prior"] for m in assignments),
    }
    # What the cap costs: the same question at util 1, prediction only. Computed once.
    if util < 1 and not _nested:
        ba = plan(recs, current, n, layer_bytes, 1.0, min_layers, min_nodes, B, C, head_ms, priors,
                  now, last_applied, provisional, force, _nested=True)
    else:
        ba = out
    out["best_achievable"] = {"util": 1.0, "predicted_ms_per_token": ba["predicted_ms_per_token"],
                              "predicted_tok_s": ba["predicted_tok_s"], "peak_tok_s": ba["peak_tok_s"],
                              "utilization": ba["utilization"]}
    have = pred and ba["predicted_tok_s"]
    out["cap_cost"] = {"latency_ms": (pred["ms_per_token"] - ba["predicted_ms_per_token"]) if have else None,
                       "throughput_pct": 100 * (1 - pred["tok_s"] / ba["predicted_tok_s"]) if have else None}
    return out


if __name__ == "__main__":
    # The measured cluster: Mac 8 layers + two phones 8 each, one user. Formula 184.2 vs measured 188.
    costs = {"mac": {"c": 2.6, "a": 0.3, "w": 2.0}, "phoneA": {"c": 5.9, "a": 0.87, "w": 25.0},
             "phoneB": {"c": 5.9, "a": 0.87, "w": 25.0}}
    p = predict({"mac": [0, 8], "phoneA": [8, 16], "phoneB": [16, 24]}, costs, 1, 1, 1.0, 5.0)
    assert abs(p["ms_per_token"] - 184.2) < 0.05 and abs(p["ms_per_token"] - 188) / 188 < 0.05, p
    solo = predict({"mac": [0, 24]}, costs, 1, 1, 1.0, 5.0)
    assert abs(solo["ms_per_token"] - 70) / 70 < 0.10, solo
    assert ram_layers({"ram_gb": 14.9}, 1860e6, 0.8) == 4          # a 32B layer on a phone
    assert ram_layers({"ram_gb": 16, "mem_available_bytes": 300e6}, 59652874, 0.8) == 0
    assert ineligible_next(True, 12, None) and not ineligible_next(True, 15, 2)
    print("placement ok")
