"""The "extra things" the proxy adds in front of the Collector.

Your application's OTel SDK already produces spans. This walks the OTLP batch
on its way past and adds what the SDK cannot know or does not bother to emit:

  1. **Device RAM/CPU** stamped onto every LLM span, plus continuous gauges.
     The SDK has no idea how much memory the box had left; the proxy runs on
     that box and does.
  2. **Metrics derived from spans** - tokens in/out, request duration, TTFT,
     error rate - so you get aggregatable time series without the application
     having to emit a single metric.

Everything derived is namespaced `llmobs.*`. It deliberately does *not* reuse
the `gen_ai.client.*` instrument names: if your app also emits those, reusing
them would double-count silently.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from opentelemetry.proto.trace.v1 import trace_pb2

from . import otlp
from . import semconv as sc

# A span is LLM-related - worth stamping with device RAM - if it carries any
# of these.
_LLM_MARKERS = (
    sc.GEN_AI_OPERATION_NAME,
    sc.GEN_AI_REQUEST_MODEL,
    sc.GEN_AI_SYSTEM,
    sc.GEN_AI_USAGE_INPUT_TOKENS,
    sc.GEN_AI_USAGE_OUTPUT_TOKENS,
)

# ...but metrics are derived only from spans that identify a model or backend,
# i.e. an actual inference call. Gateways routinely copy token totals onto a
# parent routing span as well; counting both would double every token in the
# fleet. Per the GenAI semconv a real model call always carries
# gen_ai.request.model or gen_ai.system, so requiring one is a safe filter.
_MODEL_CALL_MARKERS = (sc.GEN_AI_REQUEST_MODEL, sc.GEN_AI_SYSTEM)

# Token counts, in the spellings different SDKs actually emit.
_INPUT_TOKEN_KEYS = (
    sc.GEN_AI_USAGE_INPUT_TOKENS,
    "gen_ai.usage.prompt_tokens",
    "llm.usage.prompt_tokens",
    "llm.token_count.prompt",
)
_OUTPUT_TOKEN_KEYS = (
    sc.GEN_AI_USAGE_OUTPUT_TOKENS,
    "gen_ai.usage.completion_tokens",
    "llm.usage.completion_tokens",
    "llm.token_count.completion",
)
# Time to first token, in milliseconds or seconds depending on the key.
_TTFT_MS_KEYS = (sc.TTFT_MS, "llm.time_to_first_token_ms")
_TTFT_S_KEYS = ("gen_ai.server.time_to_first_token", "llm.time_to_first_token")


@dataclass
class BatchResult:
    """What one OTLP batch contained, for logging and /v1/stats."""

    spans_seen: int = 0
    llm_spans: int = 0        # LLM-related: got a RAM stamp
    model_calls: int = 0      # actual inference calls: fed the metrics
    spans_enriched: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0


class Enricher:
    def __init__(self, telemetry, stats: "RollingStats | None" = None) -> None:
        self.telemetry = telemetry
        self.config = telemetry.config
        self.stats = stats

    # ------------------------------------------------------------------
    def process_traces(self, request) -> BatchResult:
        """Enrich a decoded ExportTraceServiceRequest in place."""
        result = BatchResult()
        # One RAM reading per batch: the whole batch describes roughly the same
        # moment, and psutil is not free.
        snapshot = self.telemetry.sampler.sample()

        for resource_spans in request.resource_spans:
            resource_attrs = self._enrich_resource(resource_spans.resource)
            node_id = resource_attrs.get(sc.NODE_ID) or self.config.node_id
            service_name = resource_attrs.get(sc.SERVICE_NAME) or self.config.service_name

            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    result.spans_seen += 1
                    is_llm = _is_llm_span(span)
                    if is_llm:
                        result.llm_spans += 1
                    if is_llm or self.config.enrich_all_spans:
                        self._stamp_resources(span, snapshot)
                        result.spans_enriched += 1
                    if _is_model_call(span):
                        result.model_calls += 1
                        self._derive_metrics(span, node_id, service_name, result)
        return result

    # ------------------------------------------------------------------
    def _enrich_resource(self, resource) -> dict[str, Any]:
        """Fill in what the app's SDK left blank about the device it runs on.

        Never overwrites: the application knows its own identity better than
        the proxy does.
        """
        attrs = resource.attributes
        otlp.set_attribute(attrs, sc.NODE_ID, self.config.node_id)
        otlp.set_attribute(attrs, sc.NODE_ROLE, self.config.node_role)
        for key, value in self._host_attrs.items():
            otlp.set_attribute(attrs, key, value)
        return otlp.to_dict(attrs)

    @property
    def _host_attrs(self) -> dict[str, Any]:
        if not hasattr(self, "_cached_host_attrs"):
            from .resources import host_attributes

            self._cached_host_attrs = host_attributes()
        return self._cached_host_attrs

    def _stamp_resources(self, span, snapshot) -> None:
        """Attach the device's RAM/CPU at the time this batch arrived."""
        if not self.config.memory_on_span:
            return
        for key, value in snapshot.as_span_attributes().items():
            otlp.set_attribute(span.attributes, key, value)

    # ------------------------------------------------------------------
    def _derive_metrics(self, span, node_id: str, service_name: str, result: BatchResult) -> None:
        """Turn one LLM span into the time series the span alone cannot answer."""
        attrs = span.attributes
        is_error = span.status.code == trace_pb2.Status.STATUS_CODE_ERROR

        # Low-cardinality label set only. Never trace/span/request ids here.
        labels: dict[str, Any] = {
            sc.NODE_ID: node_id,
            sc.SERVICE_NAME: service_name,
            sc.GEN_AI_OPERATION_NAME: otlp.get(attrs, sc.GEN_AI_OPERATION_NAME, "unknown"),
        }
        if model := otlp.get(attrs, sc.GEN_AI_REQUEST_MODEL):
            labels[sc.GEN_AI_REQUEST_MODEL] = model
        if system := otlp.get(attrs, sc.GEN_AI_SYSTEM):
            labels[sc.GEN_AI_SYSTEM] = system
        if is_error:
            labels[sc.ERROR_TYPE] = otlp.get(attrs, sc.ERROR_TYPE) or "unknown_error"
            result.errors += 1

        telemetry = self.telemetry

        # --- duration, straight from the span's own timestamps ----------
        duration_s = _duration_seconds(span)
        if duration_s is not None:
            telemetry.request_duration.record(duration_s, labels)

        telemetry.requests_total.add(1, {**labels, "outcome": "error" if is_error else "ok"})

        # --- tokens ------------------------------------------------------
        input_tokens = _first_int(attrs, _INPUT_TOKEN_KEYS)
        output_tokens = _first_int(attrs, _OUTPUT_TOKEN_KEYS)
        for count, token_type in (
            (input_tokens, sc.TOKEN_TYPE_INPUT),
            (output_tokens, sc.TOKEN_TYPE_OUTPUT),
        ):
            if count is None:
                continue
            token_labels = {**labels, sc.GEN_AI_TOKEN_TYPE: token_type}
            telemetry.tokens_total.add(count, token_labels)
            telemetry.tokens_per_request.record(count, token_labels)
        result.input_tokens += input_tokens or 0
        result.output_tokens += output_tokens or 0

        # --- time to first token ----------------------------------------
        ttft_s = _time_to_first_token_s(attrs)
        if ttft_s is not None:
            telemetry.time_to_first_token.record(ttft_s, labels)

        if self.stats is not None:
            self.stats.observe(
                span=span,
                node_id=node_id,
                model=labels.get(sc.GEN_AI_REQUEST_MODEL),
                duration_s=duration_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                is_error=is_error,
            )


# ----------------------------------------------------------------------
# span helpers
# ----------------------------------------------------------------------
def _is_llm_span(span) -> bool:
    keys = {kv.key for kv in span.attributes}
    return any(marker in keys for marker in _LLM_MARKERS)


def _is_model_call(span) -> bool:
    """Only real inference spans feed the metrics - see _MODEL_CALL_MARKERS."""
    keys = {kv.key for kv in span.attributes}
    return any(marker in keys for marker in _MODEL_CALL_MARKERS)


def _duration_seconds(span) -> float | None:
    if not span.end_time_unix_nano or not span.start_time_unix_nano:
        return None
    delta = span.end_time_unix_nano - span.start_time_unix_nano
    return delta / 1e9 if delta >= 0 else None


def _first_int(attributes, keys) -> int | None:
    for key in keys:
        value = otlp.as_int(otlp.get(attributes, key))
        if value is not None:
            return value
    return None


def _time_to_first_token_s(attributes) -> float | None:
    for key in _TTFT_MS_KEYS:
        value = otlp.get(attributes, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) / 1000.0
    for key in _TTFT_S_KEYS:
        value = otlp.get(attributes, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


# ----------------------------------------------------------------------
# rolling stats
# ----------------------------------------------------------------------
class RollingStats:
    """A bounded in-memory window so `/v1/stats` can answer "is anything
    flowing?" without standing up Prometheus. Lossy by design."""

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=window)
        self.total_spans = 0
        self.total_llm_spans = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_errors = 0
        self.batches_forwarded = 0
        self.batches_failed = 0
        self.started_at = time.time()

    def observe(self, *, span, node_id, model, duration_s, input_tokens, output_tokens, is_error):
        with self._lock:
            self.total_llm_spans += 1
            self.total_input_tokens += input_tokens or 0
            self.total_output_tokens += output_tokens or 0
            if is_error:
                self.total_errors += 1
            self._events.append(
                {
                    "ts": time.time(),
                    "trace_id": span.trace_id.hex(),
                    "span_name": span.name,
                    "node_id": node_id,
                    "model": model,
                    "duration_ms": round(duration_s * 1000, 3) if duration_s else None,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "status": "error" if is_error else "ok",
                }
            )

    def record_batch(self, result: BatchResult, forwarded: bool) -> None:
        with self._lock:
            self.total_spans += result.spans_seen
            if forwarded:
                self.batches_forwarded += 1
            else:
                self.batches_failed += 1

    def snapshot(self, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            recent = list(self._events)
            totals = {
                "spans_seen": self.total_spans,
                "llm_spans": self.total_llm_spans,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "errors": self.total_errors,
                "batches_forwarded": self.batches_forwarded,
                "batches_failed": self.batches_failed,
            }
        durations = sorted(e["duration_ms"] for e in recent if e["duration_ms"] is not None)
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "totals": totals,
            "window": {
                "size": len(recent),
                "capacity": self._window,
                "latency_ms": {
                    "p50": _percentile(durations, 0.50),
                    "p95": _percentile(durations, 0.95),
                    "p99": _percentile(durations, 0.99),
                    "max": durations[-1] if durations else None,
                },
                "nodes": sorted({e["node_id"] for e in recent if e["node_id"]}),
                "models": sorted({e["model"] for e in recent if e["model"]}),
            },
            "recent": recent[-limit:][::-1],
        }


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    index = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[index]
