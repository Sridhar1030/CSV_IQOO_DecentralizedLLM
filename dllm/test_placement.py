"""Pins the planner to the measured cluster and to the decisions the spec asks for.

    .venv/bin/python -m pytest dllm/test_placement.py -q

"Mac" is the M1 Pro on torch cpu (2.6 ms/layer, 2 ms wire on the same machine); "phone" is an
iQOO on the Kotlin fp32 path (5.9 ms/layer, 25 ms wire over WiFi). Wire is set explicitly on
every record so the priors are never the thing under test.

Design points matter. With the spec's model, one batch of C rows (B = C) pays every hop's wire
serially, so phones never beat the laptop alone; with C requests as separate frames (B = 1) the
pipeline fills and phones share the bottleneck. Tests that need "a plan where the phone is worth
it" pin B and C explicitly rather than assume one.
"""
import math

from dllm import placement as P

LB, N, HEAD = 59652874, 24, 5.0


def rec(name, device, ms_per_layer, ram_gb=16, **kw):
    r = dict(name=name, device=device, ram_gb=ram_gb, ms_per_layer=ms_per_layer, battery=None, thermal=None,
             mem_pct=None, mem_available_bytes=None, layers=None, present=True, ineligible=False,
             ema_ms_per_layer=None, ema_samples=0, ema_wire_ms=None, ema_batch={}, disk=set(), reassign=True,
             bw_bps=None, load_s_per_layer=None, ready_at=None)
    r.update(kw)
    if r["layers"] and not r["disk"]:
        r["disk"] = set(range(*r["layers"]))
    r.setdefault("live", bool(r["layers"]) and r["present"])
    r.setdefault("role", "absent" if not r["present"] else "active" if r["layers"] else "joining")
    return r


def mac(**kw):
    return rec("mac", "cpu", 2.6, ema_wire_ms=2.0, **kw)


def phone(name, **kw):
    return rec(name, "android-cpu-fp32", 5.9, ema_wire_ms=25.0, **kw)


def current(**ranges):
    return {"assignments": {m: list(r) for m, r in ranges.items()},
            "order": sorted(ranges, key=lambda m: ranges[m][0])}


CUR_8_8_8 = current(mac=[0, 8], phoneA=[8, 16], phoneB=[16, 24])


def three(**kw):
    return [mac(layers=[0, 8], **kw), phone("phoneA", layers=[8, 16]), phone("phoneB", layers=[16, 24])]


def plan(recs, cur, **kw):
    kw.setdefault("now", 1000.0)
    kw.setdefault("head_ms", HEAD)
    return P.plan(recs, cur, kw.pop("n", N), kw.pop("layer_bytes", LB), kw.pop("util", 1.0), **kw)


def costs_of(recs, util=1.0, layer_bytes=LB):
    return {r["name"]: P.node_costs(r, P.PRIORS, layer_bytes, util) for r in recs}


def tiles(assign, n=N):
    return P._tiles(assign, n)


# --- the cost model against reality ---------------------------------------------------------------

def test_measured_cluster_prediction():
    p = plan(three(), CUR_8_8_8, B=1, C=1)
    assert abs(p["current"]["predicted_ms_per_token"] - 188) / 188 < 0.05
    pred = P.predict(CUR_8_8_8["assignments"], costs_of(three()), 1, 1, 1.0, HEAD)
    assert abs(pred["ms_per_token"] - 184.2) < 0.05


def test_cost_aware_beats_even():
    p = plan([mac(), phone("phoneA"), phone("phoneB")], {}, B=1, C=1)
    a = p["assignments"]
    assert a["mac"][1] - a["mac"][0] >= 20 or list(a) == ["mac"]
    assert p["predicted_ms_per_token"] < 130


def test_latency_first_mac_alone_at_util_1():
    p = plan([mac(), phone("phoneA"), phone("phoneB")], {}, B=1, C=1)
    assert p["assignments"] == {"mac": [0, 24]}
    assert sorted(p["standby"]) == ["phoneA", "phoneB"]
    assert all("ms/token" in p["standby_reasons"][m] for m in ("phoneA", "phoneB"))
    assert abs(p["predicted_ms_per_token"] - 70) / 70 < 0.10
    assert p["complete"] and p["would_apply"]


def test_throughput_recruits_phones():
    recs = [mac(), phone("phoneA"), phone("phoneB")]
    p = plan(recs, {}, B=1, C=8)
    assert any(m in p["assignments"] for m in ("phoneA", "phoneB"))
    alone = P.predict({"mac": [0, 24]}, costs_of(recs), 1, 8, 1.0, HEAD)
    assert p["predicted_tok_s"] > alone["tok_s"]
    assert tiles(p["assignments"])


def test_ten_devices_ten_layers_rejected():
    p = plan([phone(f"p{i}") for i in range(10)], {}, n=10, B=1, C=1, min_layers=2)
    assert 1 <= len(p["assignments"]) <= 5
    assert tiles(p["assignments"], 10)
    assert all(b - a >= 2 for a, b in p["assignments"].values())
    assert any("fewer than 2 layers" in s for s in p["reasons"])
    for m in p["standby"]:
        s = p["standby_reasons"][m]
        assert "fewer than 2 layers" in s or "ms/token" in s


def test_min_nodes_demo_knob():
    p = plan([mac(), phone("phoneA"), phone("phoneB")], {}, B=1, C=1, min_nodes=3)
    assert sorted(p["assignments"]) == ["mac", "phoneA", "phoneB"]
    assert all(b - a >= P.MIN_LAYERS for a, b in p["assignments"].values())
    assert tiles(p["assignments"])


# --- RAM -------------------------------------------------------------------------------------------

def test_ram_limit_forces_inclusion():
    big = 1860e6                                        # a 32B fp32 layer
    phones = [phone(f"p{i}", ram_gb=14.9) for i in range(3)]
    assert all(P.ram_layers(r, big, 0.8) == 4 for r in phones)
    p = plan(phones, {}, layer_bytes=big, util=0.8, B=1, C=1)
    assert p["assignments"] == {}
    assert p["missing_layers"] == list(range(12, 24))
    assert not p["complete"] and not p["would_apply"]
    assert any("no home" in s or "RAM" in s for s in p["reasons"])
    p = plan(phones, {}, n=12, layer_bytes=big, util=0.8, B=1, C=1)
    assert sorted(p["assignments"]) == ["p0", "p1", "p2"]
    assert all(b - a == 4 for a, b in p["assignments"].values())
    assert p["complete"] and p["missing_layers"] == []


def test_ram_clamp_by_available():
    assert P.ram_layers(phone("p", mem_available_bytes=300e6), LB, 0.8) == 0
    assert P.ram_layers(phone("p"), LB, 0.8) > 24


# --- leave, takeover, hysteresis ------------------------------------------------------------------

def test_leave_neighbour_extends():
    # The Mac reports 1.15 GB free with 8 layers loaded, so it can hold 16: it takes phoneA's
    # gap and phoneB stays exactly where it is.
    recs = [mac(layers=[0, 8], mem_available_bytes=1.15e9),
            phone("phoneA", layers=[8, 16], present=False, absent_since=959.0),
            phone("phoneB", layers=[16, 24])]
    p = plan(recs, CUR_8_8_8, util=0.8, B=1, C=1)
    assert p["current"] is None                        # complete_now is False
    assert p["would_apply"] and "pipeline incomplete: apply without margin" in p["reasons"]
    assert set(p["assignments"]) <= {"mac", "phoneB"} and tiles(p["assignments"])
    assert p["assignments"]["mac"][1] > 8
    assert p["assignments"]["phoneB"] == [16, 24]
    assert p["churn"]["nodes_changed"] == ["mac"]
    assert p["excluded"]["phoneA"].startswith("absent 41")
    assert p["migration"]["mac"]["download_bytes"] == 8 * LB


def test_standby_takeover_zero_bytes():
    recs = [mac(layers=[0, 8], mem_available_bytes=0.6e9),          # can hold 8, no more
            phone("phoneA", layers=[8, 16], present=False),
            phone("phoneB", layers=[16, 24]),
            phone("phoneC", disk=set(range(8, 16)), role="standby")]
    p = plan(recs, CUR_8_8_8, B=1, C=8)
    assert p["assignments"]["phoneC"] == [8, 16]
    assert p["assignments"]["mac"] == [0, 8] and p["assignments"]["phoneB"] == [16, 24]
    assert p["churn"]["bytes_to_move"] == 0
    assert p["migration_s"] == 8 * P.PRIORS["phone"]["load_s_per_layer"]   # reload only, nothing to fetch


def test_hysteresis_blocks_tiny_gain():
    # 20/4 runs; 22/2 would be 6% faster for one user but costs two layers of download.
    cur = current(mac=[0, 20], phoneA=[20, 24])
    p = plan([mac(layers=[0, 20]), phone("phoneA", layers=[20, 24])], cur, B=1, C=1, min_nodes=2)
    assert p["assignments"] == {"mac": [0, 22], "phoneA": [22, 24]}
    assert 0.05 < p["improvement_vs_current"] < 0.07
    assert not p["would_apply"]
    assert "gain 6% < 15%" in p["reasons"]
    assert p["churn"]["bytes_to_move"] == 2 * LB

    ok, why = P.should_apply(100, 106, 20, True, True, 0, 100, False, False)
    assert not ok and "gain 6% < 15%" in why
    ok, why = P.should_apply(100, 140, 20, True, True, 0, 100, False, False)
    assert ok and "gain 40% >= 15%" in why
    ok, why = P.should_apply(100, 140, 400, True, True, 0, 100, False, False)
    assert not ok and any(s.startswith("payback") and ">" in s for s in why)
    ok, why = P.should_apply(100, 140, 20, True, True, 90, 100, False, False)
    assert not ok and any(s.startswith("cooldown 20") for s in why)
    assert P.should_apply(100, 140, 20, True, False, 0, 100, False, False) == (False, ["no change"])
    assert P.should_apply(100, 101, 999, True, True, 99, 100, False, True) == (True, ["forced"])
    assert P.should_apply(None, 140, 20, False, True, 0, 100, False, False)[0]


def test_dead_band_no_change():
    cur = current(mac=[0, 13], phoneA=[13, 19], phoneB=[19, 24])
    recs = [mac(layers=[0, 13], ema_ms_per_layer=2.3, ema_samples=10),   # ideal drifts to 13.5/5.3/5.3
            phone("phoneA", layers=[13, 19]), phone("phoneB", layers=[19, 24])]
    p = plan(recs, cur, B=1, C=8)
    assert p["assignments"] == cur["assignments"]
    assert p["reasons"] == ["no change"]
    assert p["churn"] == {"bytes_to_move": 0, "nodes_changed": []}
    assert not p["would_apply"]


# --- the utilization knob -------------------------------------------------------------------------

def test_utilization_cap_reports_cost():
    p = plan([mac()], {}, util=0.6, B=1, C=1)
    lat1 = HEAD + 24 * 2.6 + 2.0 + P.HOP_OVERHEAD_MS
    assert math.isclose(p["predicted_ms_per_token"], max(lat1, 24 * 2.6 / 0.6))
    assert abs(p["predicted_ms_per_token"] - 104) < 1
    assert abs(p["best_achievable"]["predicted_ms_per_token"] - 70) / 70 < 0.10
    assert math.isclose(p["cap_cost"]["latency_ms"], 104 - lat1)
    assert abs(p["cap_cost"]["latency_ms"] - 34) < 5
    assert all(v["busy_fraction"] <= 0.6 + 1e-9 for v in p["per_node"].values())
    assert p["utilization"] <= 0.6 + 1e-9
    p = plan([mac(), phone("phoneA"), phone("phoneB")], {}, util=0.8, B=1, C=8)
    assert len(p["assignments"]) >= 2
    assert all(v["busy_fraction"] <= 0.8 + 1e-9 for v in p["per_node"].values())
    assert p["cap_cost"]["throughput_pct"] >= 0


# --- join decisions --------------------------------------------------------------------------------

def test_slow_joiner_rejected():
    # Two frames in flight (B=4, C=8): the phone earns its wire, a 12 ms/layer numpy phone does not.
    cur = current(mac=[0, 16], phoneA=[16, 24])
    recs = [mac(layers=[0, 16]), phone("phoneA", layers=[16, 24]),
            rec("phoneB", "phone-numpy", 12.0, ema_wire_ms=25.0)]
    p = plan(recs, cur, B=4, C=8)
    assert "phoneB" in p["standby"] and "phoneB" not in p["assignments"]
    why = p["standby_reasons"]["phoneB"]
    assert "with it" in why and "without" in why
    assert "phoneA" in p["assignments"]


def test_fast_joiner_takes_from_slow():
    cur = current(mac=[0, 13], phoneA=[13, 19], phoneB=[19, 24])
    recs = [mac(layers=[0, 13]), phone("phoneA", layers=[13, 19]), phone("phoneB", layers=[19, 24]),
            rec("mac2", "cpu", 2.6, ema_wire_ms=25.0)]
    p = plan(recs, cur, B=1, C=8)
    a = p["assignments"]
    assert "mac2" in a
    phones = sum(a[m][1] - a[m][0] for m in ("phoneA", "phoneB") if m in a)
    assert phones < 11
    assert p["improvement_vs_current"] > 0.15 and p["would_apply"]


def test_low_battery_evictable():
    recs = [mac(), phone("phoneA"), phone("phoneB", battery=15)]
    c, from_prior, f = P.effective_cost(recs[2])
    assert math.isclose(c, 1.5 * 5.9) and not from_prior and f["battery"] == 0.5
    costs = costs_of(recs)
    pred = P.predict({"mac": [0, 12], "phoneA": [12, 18], "phoneB": [18, 24]}, costs, 1, 8, 1.0, HEAD)
    order = P.evict_order(["mac", "phoneA", "phoneB"], costs, pred)
    assert order.index("phoneB") < order.index("phoneA") < order.index("mac")
    p = plan(recs, {}, B=1, C=1, min_nodes=2)          # one phone must stay: the healthy one
    assert "phoneA" in p["assignments"] and "phoneB" not in p["assignments"]
    p = plan(recs, {}, B=4, C=8)
    kept = [m for m in ("phoneA", "phoneB") if m in p["assignments"]]
    assert kept != ["phoneB"]


def test_ineligible_and_reentry():
    assert P.ineligible_next(False, 9, None)
    assert P.ineligible_next(True, 12, None)
    assert not P.ineligible_next(True, 15, 2)
    assert P.ineligible_next(False, 50, 5)
    assert P.ineligible_next(True, 50, 3)
    assert not P.ineligible_next(True, 50, 2)
    assert not P.ineligible_next(False, None, None)
    cur = current(mac=[0, 16], phoneA=[16, 24])
    p = plan([mac(layers=[0, 16]), phone("phoneA", layers=[16, 24], thermal=5, ineligible=True)], cur, B=1, C=1)
    assert "phoneA" not in p["assignments"] and p["excluded"]["phoneA"] == "thermal 5"
    assert p["per_node"]["phoneA"]["role"] == "excluded"
    assert p["would_apply"] and any("ineligible" in s for s in p["reasons"])


def test_stale_node_ignored():
    cur = current(mac=[0, 16], phoneA=[16, 24])
    p = plan([mac(layers=[0, 16]), phone("phoneA", layers=[16, 24], present=False, absent_since=970.0)], cur, B=1, C=1)
    assert p["excluded"]["phoneA"] == "absent 30 s"
    assert p["current"] is None and "pipeline incomplete: apply without margin" in p["reasons"]
    assert p["would_apply"] and p["assignments"] == {"mac": [0, 24]}
    # present but not live (mid-reassign): the pipeline is not complete either
    p = plan([mac(layers=[0, 16]), phone("phoneA", layers=[16, 24], live=False, role="reassigning")], cur, B=1, C=1)
    assert p["current"] is None and "pipeline incomplete: apply without margin" in p["reasons"]


def test_prior_marks_provisional():
    p = plan([rec("newphone", "android-cpu-fp32", None, ema_wire_ms=25.0)], {}, B=1, C=1)
    assert p["assignments"] == {"newphone": [0, 24]}
    assert p["per_node"]["newphone"]["from_prior"] and p["provisional"]
    assert p["per_node"]["newphone"]["c_ms_per_layer"] == P.PRIORS["phone"]["ms_per_layer"]
    p = plan([mac(), rec("newphone", "android-cpu-fp32", None, ema_wire_ms=25.0)], {}, B=1, C=1)
    assert not p["provisional"]                        # standby nodes do not make a plan provisional
    ok, why = P.should_apply(100, 106, 20, True, True, 90, 100, True, False)
    assert ok and "provisional: margin waived" in why and not any("cooldown" in s for s in why)
    ok, why = P.should_apply(100, 106, 400, True, True, 90, 100, True, False)
    assert not ok and any(s.startswith("payback") and ">" in s for s in why)


def test_tie_break_prefers_fewer_nodes_then_fewer_bytes():
    fewer = [{"members": ("mac",), "score": 96.0, "churn": 500},
             {"members": ("mac", "phoneA"), "score": 100.0, "churn": 0}]
    assert P.pick_best(fewer, set())["members"] == ("mac",)
    bytes_ = [{"members": ("mac", "phoneA"), "score": 100.0, "churn": 5 * LB},
              {"members": ("mac", "phoneB"), "score": 98.0, "churn": 0}]
    assert P.pick_best(bytes_, set())["members"] == ("mac", "phoneB")
    assert P.pick_best([fewer[1], {"members": ("mac",), "score": 90.0, "churn": 0}], set())["members"] == ("mac", "phoneA")

    # A phone on a bad link (60 ms wire) would score a few percent above the Mac alone at
    # B=4, C=8. Not worth a second device: the search picks the Mac.
    recs = [mac(), rec("phoneA", "android-cpu-fp32", 5.9, ema_wire_ms=60.0)]
    costs = costs_of(recs)
    best, best_with = P.search(["mac", "phoneA"], {}, N, 2, 1, 4, 8, 1.0, HEAD, costs, {}, LB)
    assert best["members"] == ("mac",)
    assert best["score"] < best_with["phoneA"]["score"] <= best["score"] / (1 - P.TIE_BAND)
    assert plan(recs, {}, B=4, C=8)["assignments"] == {"mac": [0, 24]}

    # Two identical phones; phoneB already has the files the plan would give it.
    recs = [mac(disk=set(range(24))), phone("phoneA"), phone("phoneB", disk=set(range(20, 24)))]
    p = plan(recs, {}, B=4, C=8)
    assert "phoneB" in p["assignments"] and "phoneA" not in p["assignments"]
    assert p["churn"]["bytes_to_move"] == 0


def test_migration_estimate():
    recs = {"mac": mac(layers=[0, 8], bw_bps=100e6, load_s_per_layer=0.1),
            "phoneB": phone("phoneB", layers=[16, 24])}
    s, per = P.migration({"mac": [0, 16], "phoneB": [16, 24]}, recs, P.PRIORS, LB)
    assert abs(per["mac"]["download_s"] - 4.8) < 0.05
    assert math.isclose(per["mac"]["reload_s"], 1.6)
    assert per["mac"]["download_bytes"] == 8 * LB
    assert "phoneB" not in per
    assert math.isclose(s, per["mac"]["download_s"] + per["mac"]["reload_s"])
    # priors fill in for a node that never reported bandwidth
    s, per = P.migration({"phoneB": [8, 24]}, recs, P.PRIORS, LB)
    assert math.isclose(per["phoneB"]["download_s"], 8 * LB / 12e6)


# --- the building blocks ---------------------------------------------------------------------------

def test_allocations_and_layout():
    costs = {"mac": {"c": 2.6, "a": 0.3, "w": 2, "ram": 100}, "p": {"c": 5.9, "a": 0.87, "w": 25, "ram": 100}}
    assert P.allocate_cheapest(("mac", "p"), 24, 2, costs) == {"mac": 22, "p": 2}
    assert P.allocate_balanced(("mac", "p"), 24, 1, 2, costs) == {"mac": 17, "p": 7}
    costs["p"]["ram"] = 3
    assert P.allocate_balanced(("mac", "p"), 24, 1, 2, costs) == {"mac": 21, "p": 3}     # pinned to RAM
    costs["p"]["ram"] = 1
    assert P.allocate_balanced(("mac", "p"), 24, 1, 2, costs) is None                      # under min_layers
    assert P.allocate_cheapest(("mac", "p"), 24, 2, costs) is None
    assert P.allocate_cheapest(("mac",), 24, 30, costs) is None
    ranges, churn = P.layout({"mac": 8, "p": 16}, ["p", "mac"], {"p": set(range(16)), "mac": set(range(16, 24))}, 1)
    assert ranges == {"p": [0, 16], "mac": [16, 24]} and churn == {"p": 0, "mac": 0}
    ranges, churn = P.layout({"mac": 8, "new": 8, "p": 8}, ["mac", "p"], {"new": set(range(8, 16))}, 1)
    assert ranges["new"] == [8, 16] and churn["new"] == 0


def test_partial_fill_and_batch_slope():
    costs = {"mac": {"ram": 12}, "phoneB": {"ram": 10}, "phoneC": {"ram": 2}}
    ranges, missing = P.partial_fill({"mac": [0, 8], "phoneB": [16, 24]}, ["mac", "phoneB", "phoneC"], costs, 24)
    assert ranges["mac"] == [0, 12] and ranges["phoneB"] == [14, 24] and ranges["phoneC"] == [12, 14]
    assert missing == []
    costs["phoneC"]["ram"] = 0
    ranges, missing = P.partial_fill({"mac": [0, 8], "phoneB": [16, 24]}, ["mac", "phoneB", "phoneC"], costs, 24)
    assert missing == [12, 13] and "phoneC" not in ranges
    assert P.batch_slope(mac()) == 0.30 and P.batch_slope(phone("p")) == 0.87
    assert P.batch_slope(rec("p", "phone-numpy", 12)) == 0.9
    assert math.isclose(P.batch_slope(mac(ema_batch={1: 2.6, 4: 2.6 * 1.9})), 0.3)
    assert P.batch_slope(mac(ema_batch={1: 2.6, 8: 2.6 * 20})) == 1.0
    assert P.device_class("android-cpu-fp32") == "phone" and P.device_class("mps") == "cpu"
    assert P.prior_ms_per_layer("mps") == 2.0 and P.prior_ms_per_layer("cuda") == 8.0
    assert P.wire_ms(rec("p", "cpu", 1, ema_wire_ms=0.1)) == 0.5


# --- from the adversarial review ------------------------------------------------------------------

def test_balanced_survives_everyone_pinned_in_one_round():
    # A fast Mac clamped to 10 layers by free memory, two slow numpy phones: the first round pins the
    # Mac at RAM and both phones under min_layers. The leftover must go back to the phones, not to None.
    costs = {"mac": {"c": 0.8, "a": 0.3, "w": 2, "ram": 10},
             "pA": {"c": 30.0, "a": 0.9, "w": 25, "ram": 100}, "pB": {"c": 30.0, "a": 0.9, "w": 25, "ram": 100}}
    assert P.allocate_balanced(("mac", "pA", "pB"), 24, 1, 2, costs) == {"mac": 10, "pA": 7, "pB": 7}
    recs = [rec("mac", "cpu", 0.8, ema_wire_ms=2.0, mem_available_bytes=10 * LB * P.LAYER_OVERHEAD + 0.5 * 2 ** 30),
            rec("pA", "phone-numpy", 30.0, ema_wire_ms=25.0), rec("pB", "phone-numpy", 30.0, ema_wire_ms=25.0)]
    assert P.ram_layers(recs[0], LB, 1.0) == 10
    p = plan(recs, {}, B=1, C=8)
    a = p["assignments"]
    assert "mac" in a and a["mac"][1] - a["mac"][0] == 10
    assert p["predicted_ms_per_token"] < 600      # the cheapest-only split was 783 ms/token


def test_member_that_never_loaded_is_a_change():
    # The plan equals the committed one but the member holds nothing: still a change, so the hub
    # re-sends the assign instead of reporting "no change" forever. force must not be swallowed either.
    cur = current(mac=[0, 24])
    recs = [mac(live=False, role="joining"), phone("phoneA", disk=set(range(24)), role="standby")]
    recs[0]["disk"] = set(range(24))
    p = plan(recs, cur, B=1, C=1)
    assert p["current"] is None and "pipeline incomplete: apply without margin" in p["reasons"]
    assert p["would_apply"] and "mac" in p["churn"]["nodes_changed"] and "no change" not in p["reasons"]
    p = plan(recs, cur, B=1, C=1, force=True)
    assert p["would_apply"] and "forced" in p["reasons"]
    # the 7B case from the review: identical assignments, one phone never loaded
    cur = current(pA=[0, 12], pB=[12, 24])
    recs = [phone("pA", layers=[0, 12]), phone("pB", live=False, role="reassigning", disk=set(range(12, 24)))]
    p = plan(recs, cur, B=1, C=1, min_nodes=2)
    assert p["assignments"] == cur["assignments"] and p["would_apply"] and p["churn"]["nodes_changed"] == ["pB"]


def test_zero_bench_does_not_crash():
    for recs in ([rec("z", "cpu", 0.0, ema_wire_ms=2.0)],
                 [mac(), rec("z", "cpu", 0.0, ema_wire_ms=2.0)],
                 [mac(ema_ms_per_layer=0.0, ema_samples=10)]):
        p = plan(recs, {}, B=1, C=8)
        assert p["assignments"] and p["predicted_ms_per_token"] > 0
    c, from_prior, _ = P.effective_cost(rec("z", "cpu", 0.0))
    assert from_prior and c == P.PRIORS["cpu"]["ms_per_layer"]
    c, from_prior, _ = P.effective_cost(mac(ema_ms_per_layer=-1.0, ema_samples=10))
    assert not from_prior and c == 2.6                  # falls back to the join bench


def test_late_heartbeat_keeps_hysteresis():
    # Three Macs, m3 a little faster than its share implies: deferred under the margin when all are
    # live. m2's heartbeat is 6 s late (present, still holding [9,17], role active): still deferred.
    cur = current(m1=[0, 9], m2=[9, 17], m3=[17, 24])
    def macs(**m2kw):
        return [rec("m1", "cpu", 2.6, ema_wire_ms=2.0, layers=[0, 9]),
                rec("m2", "cpu", 2.6, ema_wire_ms=2.0, layers=[9, 17], **m2kw),
                rec("m3", "cpu", 2.6, ema_wire_ms=2.0, layers=[17, 24], ema_ms_per_layer=2.2, ema_samples=10)]
    p = plan(macs(), cur, B=1, C=8, min_nodes=3)
    assert p["current"] is not None
    p2 = plan(macs(live=False, role="active"), cur, B=1, C=8, min_nodes=3)
    assert p2["current"] is not None and p2["would_apply"] == p["would_apply"] and p2["reasons"] == p["reasons"]
    assert "pipeline incomplete: apply without margin" not in p2["reasons"]
    # a member the hub is actually moving (role reassigning) still counts as incomplete
    p3 = plan(macs(live=False, role="reassigning"), cur, B=1, C=8, min_nodes=3)
    assert p3["current"] is None


def test_min_layers_zero_is_clamped():
    p = plan([mac()], {}, B=1, C=1, min_layers=0)
    assert p["assignments"] == {"mac": [0, 24]}
    p = plan([mac()], {}, B=1, C=1, min_layers=-1)
    assert p["assignments"] == {"mac": [0, 24]}
