"""Enrichment: RAM stamping and metric derivation from real OTLP spans."""

import pytest
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2 as ts
from opentelemetry.proto.trace.v1 import trace_pb2

from llmobs import otlp
from llmobs import semconv as sc
from llmobs.enrich import Enricher, RollingStats


def make_request(spans, resource_attrs=None):
    """Build an ExportTraceServiceRequest the way an OTel SDK would."""
    request = ts.ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    for key, value in (resource_attrs or {"service.name": "my-llm-app"}).items():
        resource_spans.resource.attributes.add(key=key, value=otlp.any_value(value))
    scope_spans = resource_spans.scope_spans.add()
    for spec in spans:
        span = scope_spans.spans.add()
        span.name = spec.get("name", "chat")
        span.trace_id = spec.get("trace_id", b"\x01" * 16)
        span.span_id = spec.get("span_id", b"\x02" * 8)
        span.start_time_unix_nano = spec.get("start", 1_000_000_000)
        span.end_time_unix_nano = spec.get("end", 1_500_000_000)  # 500ms
        if spec.get("error"):
            span.status.code = trace_pb2.Status.STATUS_CODE_ERROR
        for key, value in spec.get("attrs", {}).items():
            span.attributes.add(key=key, value=otlp.any_value(value))
    return request


LLM_SPAN = {
    "name": "chat llama3.1:8b",
    "attrs": {
        sc.GEN_AI_OPERATION_NAME: "chat",
        sc.GEN_AI_REQUEST_MODEL: "llama3.1:8b",
        sc.GEN_AI_SYSTEM: "ollama",
        sc.GEN_AI_USAGE_INPUT_TOKENS: 120,
        sc.GEN_AI_USAGE_OUTPUT_TOKENS: 340,
    },
}


@pytest.fixture
def enricher(harness):
    return Enricher(harness.telemetry, RollingStats(window=100))


def test_device_ram_is_stamped_onto_llm_spans(enricher, harness):
    request = make_request([LLM_SPAN])
    enricher.process_traces(request)

    span = request.resource_spans[0].scope_spans[0].spans[0]
    attrs = otlp.to_dict(span.attributes)
    assert attrs[sc.MEM_PROCESS_RSS] > 0
    assert attrs[sc.MEM_SYSTEM_TOTAL] > 0
    assert 0 <= attrs[sc.MEM_SYSTEM_PERCENT] <= 100
    # The app's own attributes are untouched.
    assert attrs[sc.GEN_AI_REQUEST_MODEL] == "llama3.1:8b"


def test_non_llm_spans_are_left_alone_by_default(enricher):
    request = make_request([{"name": "GET /health", "attrs": {"http.method": "GET"}}])
    result = enricher.process_traces(request)

    span = request.resource_spans[0].scope_spans[0].spans[0]
    assert sc.MEM_PROCESS_RSS not in otlp.to_dict(span.attributes)
    assert result.llm_spans == 0
    assert result.spans_seen == 1


def test_tokens_are_derived_into_metrics(enricher, harness):
    enricher.process_traces(make_request([LLM_SPAN]))

    points = harness.metric_points(sc.M_TOKENS_TOTAL)
    by_type = {a[sc.GEN_AI_TOKEN_TYPE]: v for v, a in points}
    assert by_type == {"input": 120, "output": 340}

    labels = points[0][1]
    assert labels[sc.GEN_AI_REQUEST_MODEL] == "llama3.1:8b"
    assert labels[sc.SERVICE_NAME] == "my-llm-app"


def test_duration_is_derived_from_span_timestamps(enricher, harness):
    enricher.process_traces(make_request([LLM_SPAN]))
    points = harness.metric_points(sc.M_REQUEST_DURATION)
    assert points and points[0][0] == pytest.approx(0.5)  # 500ms window


def test_derived_metrics_do_not_reuse_gen_ai_instrument_names(enricher, harness):
    """Reusing gen_ai.client.* would double-count for an app that emits them."""
    enricher.process_traces(make_request([LLM_SPAN]))
    for name in harness.metrics():
        assert not name.startswith("gen_ai."), f"{name} would collide with app metrics"


def test_metric_labels_stay_low_cardinality(enricher, harness):
    """A per-request id on a label creates one time series per request."""
    span = {**LLM_SPAN, "attrs": {**LLM_SPAN["attrs"], "llmobs.request.id": "req-abc"}}
    enricher.process_traces(make_request([span]))

    for name in (sc.M_TOKENS_TOTAL, sc.M_REQUEST_DURATION, sc.M_REQUESTS_TOTAL):
        for _value, labels in harness.metric_points(name):
            assert "llmobs.request.id" not in labels
            assert not any("trace" in k for k in labels)


def test_error_spans_produce_error_metrics(enricher, harness):
    span = {**LLM_SPAN, "error": True, "attrs": {**LLM_SPAN["attrs"], sc.ERROR_TYPE: "TimeoutError"}}
    enricher.process_traces(make_request([span]))

    outcomes = {
        a.get("outcome"): a.get(sc.ERROR_TYPE)
        for _v, a in harness.metric_points(sc.M_REQUESTS_TOTAL)
    }
    assert outcomes == {"error": "TimeoutError"}


def test_alternate_token_attribute_spellings_are_understood(enricher, harness):
    """Not every instrumentation library uses the current semconv names."""
    span = {
        "name": "chat",
        "attrs": {
            sc.GEN_AI_SYSTEM: "openai",
            "gen_ai.usage.prompt_tokens": 7,
            "gen_ai.usage.completion_tokens": 9,
        },
    }
    enricher.process_traces(make_request([span]))
    by_type = {a[sc.GEN_AI_TOKEN_TYPE]: v for v, a in harness.metric_points(sc.M_TOKENS_TOTAL)}
    assert by_type == {"input": 7, "output": 9}


def test_ttft_accepted_in_both_ms_and_seconds(enricher, harness):
    """TTFT is normalized to seconds whichever unit the span used.

    Histogram points report a cumulative sum, so the second assertion checks
    the delta the seconds-valued span contributed.
    """
    enricher.process_traces(
        make_request([{**LLM_SPAN, "attrs": {**LLM_SPAN["attrs"], sc.TTFT_MS: 250.0}}])
    )
    assert harness.metric_points(sc.M_TTFT)[0][0] == pytest.approx(0.25)

    enricher.process_traces(
        make_request(
            [{**LLM_SPAN, "attrs": {**LLM_SPAN["attrs"], "gen_ai.server.time_to_first_token": 0.4}}]
        )
    )
    harness.refresh()
    assert harness.metric_points(sc.M_TTFT)[0][0] == pytest.approx(0.65)  # 0.25 + 0.40


def test_node_identity_filled_in_without_overwriting_the_app(enricher):
    """The app knows its own identity better than the proxy does."""
    request = make_request(
        [LLM_SPAN], resource_attrs={"service.name": "app", sc.NODE_ID: "app-declared"}
    )
    enricher.process_traces(request)
    attrs = otlp.to_dict(request.resource_spans[0].resource.attributes)
    assert attrs[sc.NODE_ID] == "app-declared"
    assert attrs[sc.NODE_ROLE]  # filled in, since the app omitted it


def test_token_counts_as_strings_are_parsed(enricher, harness):
    """Some SDKs stringify numeric attributes."""
    span = {"name": "chat", "attrs": {sc.GEN_AI_SYSTEM: "vllm", sc.GEN_AI_USAGE_INPUT_TOKENS: "42"}}
    enricher.process_traces(make_request([span]))
    by_type = {a[sc.GEN_AI_TOKEN_TYPE]: v for v, a in harness.metric_points(sc.M_TOKENS_TOTAL)}
    assert by_type == {"input": 42}


def test_batch_result_summarizes_the_batch(enricher):
    result = enricher.process_traces(
        make_request([LLM_SPAN, LLM_SPAN, {"name": "GET /x", "attrs": {"http.method": "GET"}}])
    )
    assert result.spans_seen == 3
    assert result.llm_spans == 2
    assert result.input_tokens == 240
    assert result.output_tokens == 680


# ----------------------------------------------------------------------
# double-counting
# ----------------------------------------------------------------------
def test_parent_routing_span_does_not_double_count_tokens(enricher, harness):
    """A gateway that copies token totals onto its parent span must not make
    every token in the fleet count twice.

    The parent carries gen_ai.operation.name and the usage attributes but no
    model/system, so it is RAM-stamped yet excluded from the metrics.
    """
    parent = {
        "name": "route request",
        "attrs": {
            sc.GEN_AI_OPERATION_NAME: "route",
            sc.GEN_AI_USAGE_INPUT_TOKENS: 120,
            sc.GEN_AI_USAGE_OUTPUT_TOKENS: 340,
        },
    }
    result = enricher.process_traces(make_request([parent, LLM_SPAN]))

    assert result.llm_spans == 2      # both got a RAM stamp
    assert result.model_calls == 1    # only the real call fed the metrics

    by_type = {a[sc.GEN_AI_TOKEN_TYPE]: v for v, a in harness.metric_points(sc.M_TOKENS_TOTAL)}
    assert by_type == {"input": 120, "output": 340}, "tokens counted twice"

    # Every token series must be attributable to a model.
    for _v, labels in harness.metric_points(sc.M_TOKENS_TOTAL):
        assert labels.get(sc.GEN_AI_REQUEST_MODEL), "series with no model label"


def test_aggregate_span_still_gets_device_ram(enricher):
    parent = {"name": "route", "attrs": {sc.GEN_AI_OPERATION_NAME: "route"}}
    request = make_request([parent])
    enricher.process_traces(request)
    span = request.resource_spans[0].scope_spans[0].spans[0]
    assert otlp.to_dict(span.attributes)[sc.MEM_PROCESS_RSS] > 0
