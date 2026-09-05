"""OpenTelemetry for the coordinator. The only module in dllm that knows OTel exists.

The observability layer in observability/ is an OTLP proxy: applications export plain OTel spans
carrying gen_ai.* attributes, the proxy stamps them with the RAM of the device it runs on, derives
token, latency and error metrics, and forwards to a Collector. Its integration contract is the wire
format, not a library, so nothing here imports llmobs. Attribute names are pinned as strings.

How this fits a cluster where the work happens on other devices
-----------------------------------------------------------------
One trace per chat request. The root span is the model call and carries the gen_ai.* attributes
the proxy keys off. Each hop through a node is a child span with that hop's compute and wire time.

The proxy's premise is "one proxy per device, RAM read where it runs". We cannot run a proxy on a
phone, but we do not need to: every node already reports its own memory over the heartbeat. Hop
spans are stamped here with the originating device's readings, under the exact attribute names the
proxy uses. The proxy never overwrites an attribute the application set, and it only stamps spans
that carry gen_ai.* markers, which hop spans deliberately do not. So a hop span says what the phone
said about itself, the root span gets the coordinator's own RAM from the proxy, and nothing is
attributed to the wrong machine.

Request metrics (tokens, duration, TTFT, errors) are the proxy's job and are derived from the root
span only. Per-node metrics are ours, namespaced dllm.*, emitted as ordinary OTel metrics that pass
through the proxy untouched.

With no endpoint configured everything here is a no-op and the hub runs exactly as before.
"""
from __future__ import annotations

import socket
import time
from typing import Callable, Iterable

from opentelemetry import metrics, trace
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.trace import SpanKind, Status, StatusCode

# --- GenAI semantic conventions, read by the proxy ---------------------------------------------
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_FINISH_REASONS = "gen_ai.response.finish_reasons"
ERROR_TYPE = "error.type"
TTFT_MS = "llmobs.response.time_to_first_token_ms"

# --- device readings, same keys the proxy stamps, so a pre-filled value is kept ---------------
MEM_PROCESS_RSS = "llmobs.process.memory.rss_bytes"
MEM_SYSTEM_TOTAL = "llmobs.system.memory.total_bytes"
MEM_SYSTEM_USED = "llmobs.system.memory.used_bytes"
MEM_SYSTEM_AVAILABLE = "llmobs.system.memory.available_bytes"
MEM_SYSTEM_PERCENT = "llmobs.system.memory.percent"
CPU_PROCESS_PERCENT = "llmobs.process.cpu.percent"
NODE_ID = "llmobs.node.id"
NODE_ROLE = "llmobs.node.role"

# --- ours ---------------------------------------------------------------------------------------
NODE_NAME = "dllm.node.name"
NODE_DEVICE = "dllm.node.device"
NODE_BATTERY = "dllm.node.battery_percent"
LAYERS_START = "dllm.layers.start"
LAYERS_END = "dllm.layers.end"
HOP_INDEX = "dllm.hop.index"
HOP_PHASE = "dllm.hop.phase"
HOP_TOKENS = "dllm.hop.tokens"
HOP_POSITION = "dllm.hop.position"
HOP_COMPUTE_MS = "dllm.hop.compute_ms"
HOP_WIRE_MS = "dllm.hop.wire_ms"
REQUEST_ID = "dllm.request.id"
PIPELINE_NODES = "dllm.pipeline.nodes"
PIPELINE_LAYOUT = "dllm.pipeline.layout"

# Heartbeat field -> span attribute. Only what the node actually reported gets stamped.
_MEM_FIELDS = {
    "rss_bytes": MEM_PROCESS_RSS,
    "sys_total_bytes": MEM_SYSTEM_TOTAL,
    "sys_used_bytes": MEM_SYSTEM_USED,
    "sys_available_bytes": MEM_SYSTEM_AVAILABLE,
    "sys_percent": MEM_SYSTEM_PERCENT,
    "cpu_percent": CPU_PROCESS_PERCENT,
}

# Hop timings sit between a millisecond and a couple of seconds. The SDK default buckets start at
# 5 ms and would put every phone decode step in the first two.
_HOP_BUCKETS_S = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

NodesView = Callable[[], Iterable[tuple[str, dict, bool]]]   # (name, node record, live)


def layout(nodes: Iterable[tuple[str, dict, bool]]) -> str:
    """'mac1:0-7,phoneA:8-15,phoneB:16-23' — bounded, so it is safe on a span."""
    parts = sorted(((n["layers"][0], f"{name}:{n['layers'][0]}-{n['layers'][1]-1}")
                    for name, n, ok in nodes if ok), key=lambda p: p[0])
    return ",".join(p[1] for p in parts)


class Observability:
    def __init__(self, *, endpoint: str | None, model: str, n_layers: int, nodes: NodesView,
                 service_name: str = "dllm-hub", node_id: str | None = None,
                 span_processors: Iterable = (), metric_readers: Iterable = ()):
        """endpoint: OTLP/HTTP base, normally the llmobs proxy (http://localhost:8100). None disables.
        span_processors / metric_readers: extra sinks, used by the tests to capture in memory."""
        self.model, self.n_layers, self.nodes = model, n_layers, nodes
        self.node_id = node_id or socket.gethostname()
        self.endpoint = endpoint
        self.enabled = bool(endpoint) or bool(span_processors) or bool(metric_readers)
        self._tp = self._mp = None

        if not self.enabled:
            self.tracer = trace.NoOpTracerProvider().get_tracer("dllm")
            self.meter = metrics.NoOpMeterProvider().get_meter("dllm")
            self._instruments()      # no-op instruments, so every call site stays unconditional
            return

        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name": service_name,
            NODE_ID: self.node_id,          # the app's own identity wins over the proxy's guess
            NODE_ROLE: "gateway",
        })
        self._tp = TracerProvider(resource=resource)
        readers = list(metric_readers)
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            base = endpoint.rstrip("/")
            self._tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces")))
            readers.append(PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=f"{base}/v1/metrics"), export_interval_millis=5000))
        for p in span_processors:
            self._tp.add_span_processor(p)
        self._mp = MeterProvider(resource=resource, metric_readers=readers, views=[
            View(instrument_name="dllm.hop.*", aggregation=ExplicitBucketHistogramAggregation(_HOP_BUCKETS_S)),
        ])
        self.tracer = self._tp.get_tracer("dllm")
        self.meter = self._mp.get_meter("dllm")
        self._instruments()

    # ------------------------------------------------------------------------------------------
    def _instruments(self):
        m = self.meter
        self.hop_compute = m.create_histogram("dllm.hop.compute", unit="s",
                                              description="Time a node spent running its layers for one hop.")
        self.hop_wire = m.create_histogram("dllm.hop.wire", unit="s",
                                           description="Round trip minus compute: serialisation plus the wire.")

        def per_node(getter, only_live=False):
            def cb(_: CallbackOptions):
                for name, n, ok in self.nodes():
                    if only_live and not ok:
                        continue
                    v = getter(n, ok)
                    if v is not None:
                        yield Observation(v, {NODE_NAME: name, NODE_DEVICE: str(n.get("device"))})
            return [cb]

        mem = lambda key: (lambda n, ok: (n.get("mem") or {}).get(key))
        m.create_observable_gauge("dllm.node.up", callbacks=per_node(lambda n, ok: 1 if ok else 0),
                                  description="1 while the node is ready and heartbeating.")
        m.create_observable_gauge("dllm.node.layers", callbacks=per_node(lambda n, ok: n["layers"][1] - n["layers"][0]),
                                  description="Number of transformer layers the node holds.")
        m.create_observable_gauge("dllm.node.layer_time", unit="s",
                                  callbacks=per_node(lambda n, ok: (n.get("ms_per_layer") or 0) / 1000 or None),
                                  description="Per-layer decode cost measured at join.")
        m.create_observable_gauge("dllm.node.memory.rss", unit="By", callbacks=per_node(mem("rss_bytes")),
                                  description="Resident memory of the node process, as reported by the node.")
        m.create_observable_gauge("dllm.node.memory.system.utilization", unit="1",
                                  callbacks=per_node(lambda n, ok: (lambda p: None if p is None else p / 100)(mem("sys_percent")(n, ok))),
                                  description="Fraction of the device's RAM in use, as reported by the node.")
        m.create_observable_gauge("dllm.node.battery", unit="1",
                                  callbacks=per_node(lambda n, ok: None if n.get("battery") is None else n["battery"] / 100),
                                  description="Battery level, 0 to 1. Absent on mains-powered nodes.")
        m.create_observable_gauge("dllm.pipeline.complete",
                                  callbacks=[lambda _: [Observation(1 if self._complete() else 0, {})]],
                                  description="1 when live nodes tile every layer exactly once.")

    def _complete(self) -> bool:
        want = 0
        for _, n, ok in sorted(self.nodes(), key=lambda t: t[1]["layers"][0]):
            if not ok or n["layers"][0] != want:
                return False
            want = n["layers"][1]
        return want == self.n_layers

    # ------------------------------------------------------------------------------------------
    def request(self, *, request_id: str, input_tokens: int, max_tokens: int, temperature: float) -> "RequestTrace":
        nodes = list(self.nodes())
        span = self.tracer.start_span(f"chat {self.model}", kind=SpanKind.SERVER, attributes={
            GEN_AI_OPERATION: "chat",
            GEN_AI_SYSTEM: "dllm",
            GEN_AI_REQUEST_MODEL: self.model,
            GEN_AI_REQUEST_MAX_TOKENS: max_tokens,
            GEN_AI_REQUEST_TEMPERATURE: temperature,
            GEN_AI_INPUT_TOKENS: input_tokens,
            REQUEST_ID: request_id,
            PIPELINE_NODES: sum(1 for _, _, ok in nodes if ok),
            PIPELINE_LAYOUT: layout(nodes),
        })
        return RequestTrace(self, span)

    def shutdown(self):
        for p in (self._tp, self._mp):
            if p is not None:
                p.force_flush()
                p.shutdown()


class RequestTrace:
    """One chat request: a root span plus one child span per hop through a node."""

    def __init__(self, obs: Observability, span):
        self.obs, self.span = obs, span
        self.t0 = time.time()
        self.ttft_ms = None
        self.hops = 0

    def hop(self, *, index: int, name: str, node: dict, n: int, pos: int,
            started: float, ended: float, compute_ms: float, wire_ms: float):
        a, b = node["layers"]
        phase = "prefill" if n > 1 else "decode"
        attrs = {
            NODE_NAME: name, NODE_DEVICE: str(node.get("device")),
            LAYERS_START: a, LAYERS_END: b - 1, HOP_INDEX: index, HOP_PHASE: phase,
            HOP_TOKENS: n, HOP_POSITION: pos,
            HOP_COMPUTE_MS: round(compute_ms, 3), HOP_WIRE_MS: round(wire_ms, 3),
        }
        # The device's own readings, from its last heartbeat. Stamped here so the proxy has nothing to add.
        for field, key in _MEM_FIELDS.items():
            v = (node.get("mem") or {}).get(field)
            if v is not None:
                attrs[key] = v
        if node.get("battery") is not None:
            attrs[NODE_BATTERY] = node["battery"]
        ctx = trace.set_span_in_context(self.span)
        s = self.obs.tracer.start_span(f"layers {a}-{b-1} on {name}", context=ctx, kind=SpanKind.CLIENT,
                                       start_time=int(started * 1e9), attributes=attrs)
        s.end(end_time=int(ended * 1e9))
        labels = {NODE_NAME: name, HOP_PHASE: phase}
        self.obs.hop_compute.record(compute_ms / 1000, labels)
        self.obs.hop_wire.record(wire_ms / 1000, labels)
        self.hops += 1

    def first_token(self):
        if self.ttft_ms is None:
            self.ttft_ms = (time.time() - self.t0) * 1000
            self.span.set_attribute(TTFT_MS, round(self.ttft_ms, 3))

    def finish(self, *, output_tokens: int, finish_reason: str):
        self.span.set_attribute(GEN_AI_OUTPUT_TOKENS, output_tokens)
        self.span.set_attribute(GEN_AI_RESPONSE_MODEL, self.obs.model)
        self.span.set_attribute(GEN_AI_FINISH_REASONS, [finish_reason])
        self.span.set_status(Status(StatusCode.OK))
        self.span.end()

    def error(self, exc: BaseException, *, output_tokens: int = 0):
        self.span.set_attribute(GEN_AI_OUTPUT_TOKENS, output_tokens)
        self.span.set_attribute(ERROR_TYPE, type(exc).__name__)
        self.span.record_exception(exc)
        self.span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
        self.span.end()
