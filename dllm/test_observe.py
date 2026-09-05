"""The hub's telemetry and the observability proxy agree on the contract.

Two halves. First, what the hub emits: one root span per request carrying gen_ai.* attributes and
one child span per hop, each stamped with the originating device's own readings. Second, what the
proxy in observability/ does with exactly those spans when they pass through: it must derive the
request metrics from the root span alone, stamp the root with the coordinator's RAM, and leave the
hop spans saying what the phone said.

    .venv/bin/python -m pytest dllm/test_observe.py -q
"""
import sys
import time
from pathlib import Path

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dllm import observe
from dllm.observe import Observability

# the proxy's own package, imported from source so the test needs no install step
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "observability" / "src"))

PHONE_MEM = {"rss_bytes": 557_000_000, "sys_total_bytes": 15_600_000_000, "sys_used_bytes": 9_000_000_000,
             "sys_available_bytes": 6_600_000_000, "sys_percent": 57.7, "cpu_percent": 91.0}
NODES = {
    "mac1":   {"layers": [0, 8],   "device": "mps",         "ms_per_layer": 5.3, "battery": 86,
               "mem": {"rss_bytes": 820_000_000, "sys_total_bytes": 34_000_000_000, "sys_percent": 60.0}},
    "phoneA": {"layers": [8, 16],  "device": "phone-numpy", "ms_per_layer": 1.1, "battery": None, "mem": PHONE_MEM},
    "phoneB": {"layers": [16, 24], "device": "phone-numpy", "ms_per_layer": 1.0, "battery": 40,   "mem": {}},
}


@pytest.fixture
def obs():
    spans, reader = InMemorySpanExporter(), InMemoryMetricReader()
    o = Observability(endpoint=None, model="Qwen/Qwen2.5-0.5B-Instruct", n_layers=24,
                      nodes=lambda: [(k, v, True) for k, v in NODES.items()], node_id="mac1-hub",
                      span_processors=[SimpleSpanProcessor(spans)], metric_readers=[reader])
    yield o, spans, reader
    o.shutdown()


def simulate_request(o, *, output_tokens=3, fail=False):
    """Prefill through three nodes, then decode steps, the way the hub drives it."""
    rt = o.request(request_id="chatcmpl-test", input_tokens=33, max_tokens=16, temperature=0.0)
    t = time.time()
    for step in range(1 + output_tokens):
        n, pos = (33, 0) if step == 0 else (1, 32 + step)
        for i, (name, node) in enumerate(NODES.items()):
            rt.hop(index=i, name=name, node=node, n=n, pos=pos, started=t, ended=t + 0.02,
                   compute_ms=15.0 if name == "mac1" else 48.0, wire_ms=4.6)
            t += 0.02
        if step >= 1:
            rt.first_token()
    if fail:
        rt.error(RuntimeError("node phoneB left"), output_tokens=output_tokens)
    else:
        rt.finish(output_tokens=output_tokens, finish_reason="stop")
    return rt


# -------------------------------------------------------------------------------------------------
# what the hub emits
# -------------------------------------------------------------------------------------------------
def test_disabled_is_a_true_no_op():
    o = Observability(endpoint=None, model="m", n_layers=24, nodes=lambda: [])
    assert not o.enabled
    simulate_request(o)          # every call must be safe with no SDK behind it
    o.shutdown()


def test_one_trace_per_request_with_a_child_span_per_hop(obs):
    o, spans, _ = obs
    simulate_request(o, output_tokens=3)
    finished = spans.get_finished_spans()
    root = [s for s in finished if s.parent is None]
    hops = [s for s in finished if s.parent is not None]
    assert len(root) == 1 and len(hops) == 4 * 3          # prefill + 3 decode steps, 3 nodes each
    assert {s.context.trace_id for s in finished} == {root[0].context.trace_id}
    assert all(s.parent.span_id == root[0].context.span_id for s in hops)


def test_root_span_is_a_model_call_in_the_proxys_terms(obs):
    o, spans, _ = obs
    simulate_request(o, output_tokens=3)
    a = next(s for s in spans.get_finished_spans() if s.parent is None).attributes
    assert a[observe.GEN_AI_SYSTEM] == "dllm"
    assert a[observe.GEN_AI_REQUEST_MODEL] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert a[observe.GEN_AI_INPUT_TOKENS] == 33 and a[observe.GEN_AI_OUTPUT_TOKENS] == 3
    assert a[observe.GEN_AI_FINISH_REASONS] == ("stop",)
    assert a[observe.TTFT_MS] > 0
    assert a[observe.PIPELINE_NODES] == 3
    assert a[observe.PIPELINE_LAYOUT] == "mac1:0-7,phoneA:8-15,phoneB:16-23"


def test_hop_spans_carry_the_originating_devices_own_readings(obs):
    o, spans, _ = obs
    simulate_request(o, output_tokens=0)
    hops = {s.attributes[observe.NODE_NAME]: s for s in spans.get_finished_spans() if s.parent is not None}
    a = hops["phoneA"].attributes
    assert a[observe.LAYERS_START] == 8 and a[observe.LAYERS_END] == 15
    assert a[observe.HOP_PHASE] == "prefill" and a[observe.HOP_TOKENS] == 33
    assert a[observe.MEM_PROCESS_RSS] == PHONE_MEM["rss_bytes"]
    assert a[observe.MEM_SYSTEM_PERCENT] == PHONE_MEM["sys_percent"]
    assert a[observe.CPU_PROCESS_PERCENT] == PHONE_MEM["cpu_percent"]
    assert observe.NODE_BATTERY not in a                     # phone reported none, so none is claimed
    assert hops["phoneB"].attributes[observe.NODE_BATTERY] == 40
    assert observe.MEM_PROCESS_RSS not in hops["phoneB"].attributes   # reported nothing, stamped nothing
    # a hop is infrastructure, not a model call: no gen_ai.* on it, so the proxy will leave it alone
    assert not any(k.startswith("gen_ai.") for k in a)


def test_per_node_metrics_have_bounded_labels(obs):
    o, _, reader = obs
    simulate_request(o, output_tokens=2)
    got = {}
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                got[m.name] = [(getattr(p, "sum", getattr(p, "value", None)), dict(p.attributes)) for p in m.data.data_points]
    compute = {(a[observe.NODE_NAME], a[observe.HOP_PHASE]) for _, a in got["dllm.hop.compute"]}
    assert compute == {(n, ph) for n in NODES for ph in ("prefill", "decode")}
    assert all(set(a) == {observe.NODE_NAME, observe.HOP_PHASE} for _, a in got["dllm.hop.compute"])
    up = {a[observe.NODE_NAME]: v for v, a in got["dllm.node.up"]}
    assert up == {"mac1": 1, "phoneA": 1, "phoneB": 1}
    assert got["dllm.pipeline.complete"][0][0] == 1
    layers = {a[observe.NODE_NAME]: v for v, a in got["dllm.node.layers"]}
    assert layers == {"mac1": 8, "phoneA": 8, "phoneB": 8}
    battery = {a[observe.NODE_NAME]: v for v, a in got["dllm.node.battery"]}
    assert battery == {"mac1": 0.86, "phoneB": 0.4}         # phoneA reported none, so no series


def test_a_failed_request_is_an_error_span(obs):
    o, spans, _ = obs
    simulate_request(o, output_tokens=2, fail=True)
    root = next(s for s in spans.get_finished_spans() if s.parent is None)
    assert root.status.status_code.name == "ERROR"
    assert root.attributes[observe.ERROR_TYPE] == "RuntimeError"
    assert root.attributes[observe.GEN_AI_OUTPUT_TOKENS] == 2


# -------------------------------------------------------------------------------------------------
# what the proxy does with them
# -------------------------------------------------------------------------------------------------
@pytest.fixture
def proxy_enricher(monkeypatch):
    from llmobs import telemetry as t
    from llmobs.enrich import Enricher, RollingStats
    monkeypatch.setenv("LLMOBS_OTLP_ENABLED", "false")
    t.reset_for_testing()
    reader = InMemoryMetricReader()
    tel = t.init(set_global=False, otlp_enabled=False, service_name="llmobs-proxy",
                 node_id="proxy-on-mac", node_role="proxy", relay_mode=True,
                 extra_metric_readers=[reader])
    yield Enricher(tel, RollingStats(window=100)), reader
    t.reset_for_testing()


def as_otlp(spans):
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    return encode_spans(spans)


def test_proxy_counts_the_request_once_and_respects_the_phones_readings(obs, proxy_enricher):
    from llmobs import otlp
    from llmobs import semconv as sc
    o, spans, _ = obs
    simulate_request(o, output_tokens=5)
    request = as_otlp(spans.get_finished_spans())
    enricher, reader = proxy_enricher

    result = enricher.process_traces(request)

    # one model call per request, however many hops it took
    assert result.model_calls == 1
    assert result.spans_seen == 1 + (1 + 5) * 3            # root + (prefill + 5 decode steps) x 3 nodes
    assert (result.input_tokens, result.output_tokens) == (33, 5)

    by_name = {s.name: s for rs in request.resource_spans for ss in rs.scope_spans for s in ss.spans}
    root = otlp.to_dict(by_name["chat Qwen/Qwen2.5-0.5B-Instruct"].attributes)
    phone = otlp.to_dict(by_name["layers 8-15 on phoneA"].attributes)
    other = otlp.to_dict(by_name["layers 16-23 on phoneB"].attributes)

    # the root ran on the coordinator, so the proxy's own RAM reading on it is the right one
    assert root[sc.MEM_PROCESS_RSS] > 0 and root[sc.MEM_PROCESS_RSS] != PHONE_MEM["rss_bytes"]
    # the phone's hop keeps the phone's numbers, exactly as reported
    assert phone[sc.MEM_PROCESS_RSS] == PHONE_MEM["rss_bytes"]
    assert phone[sc.MEM_SYSTEM_PERCENT] == PHONE_MEM["sys_percent"]
    # a hop with no readings gets nothing invented for it
    assert sc.MEM_PROCESS_RSS not in other

    # the app's identity on the resource wins over the proxy's
    res = otlp.to_dict(request.resource_spans[0].resource.attributes)
    assert res[sc.NODE_ID] == "mac1-hub" and res[sc.NODE_ROLE] == "gateway"

    # derived metrics: tokens attributed to our node id, model, system
    points = {}
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                points[m.name] = [(getattr(p, "value", getattr(p, "sum", None)), dict(p.attributes)) for p in m.data.data_points]
    tokens = {a[sc.GEN_AI_TOKEN_TYPE]: v for v, a in points[sc.M_TOKENS_TOTAL]}
    assert tokens == {"input": 33, "output": 5}
    labels = points[sc.M_TOKENS_TOTAL][0][1]
    assert labels[sc.NODE_ID] == "mac1-hub"
    assert labels[sc.GEN_AI_REQUEST_MODEL] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert labels[sc.GEN_AI_SYSTEM] == "dllm"
    assert points[sc.M_TTFT][0][0] > 0
