"""Proxy behaviour: transparency to the SDK, and honest failure handling."""

import gzip

import httpx
import pytest
from fastapi.testclient import TestClient
from google.protobuf import json_format

from llmobs import otlp
from llmobs import semconv as sc
from llmobs.server import create_app
from tests.test_enrich import LLM_SPAN, make_request


class FakeUpstream:
    """Stands in for the real Collector. Records what it was handed."""

    def __init__(self, status_code=200, raises=None):
        self.status_code = status_code
        self.raises = raises
        self.requests: list[dict] = []

    async def post(self, url, content, headers):
        if self.raises:
            raise self.raises
        self.requests.append({"url": url, "content": content, "headers": headers})
        return httpx.Response(
            self.status_code,
            content=b"",
            headers={"content-type": headers.get("content-type", otlp.PROTOBUF)},
        )

    async def aclose(self):
        return None

    @property
    def last(self):
        return self.requests[-1]

    def decoded(self, message_type=None):
        req = self.last
        return otlp.decode(
            req["content"], req["headers"]["content-type"], message_type or otlp.TRACES_REQUEST
        )


@pytest.fixture
def proxy(harness):
    """A running proxy with its upstream replaced by a fake."""
    upstream = FakeUpstream()
    with TestClient(create_app(otlp_enabled=False, service_name="llmobs-proxy")) as client:
        from llmobs import server

        server._state["client"] = upstream
        yield client, upstream


def post_traces(client, request, content_type=otlp.PROTOBUF, headers=None, **kwargs):
    return client.post(
        "/v1/traces",
        content=otlp.encode(request, content_type),
        headers={"content-type": content_type, **(headers or {})},
        **kwargs,
    )


# ----------------------------------------------------------------------
# transparency
# ----------------------------------------------------------------------
def test_batch_is_forwarded_upstream(proxy):
    client, upstream = proxy
    response = post_traces(client, make_request([LLM_SPAN]))

    assert response.status_code == 200
    assert len(upstream.requests) == 1
    assert upstream.last["url"].endswith("/v1/traces")


def test_forwarded_spans_carry_the_enrichment(proxy):
    client, upstream = proxy
    post_traces(client, make_request([LLM_SPAN]))

    span = upstream.decoded().resource_spans[0].scope_spans[0].spans[0]
    attrs = otlp.to_dict(span.attributes)
    assert attrs[sc.MEM_PROCESS_RSS] > 0                    # added by the proxy
    assert attrs[sc.GEN_AI_REQUEST_MODEL] == "llama3.1:8b"  # app's own, intact
    assert span.name == "chat llama3.1:8b"


def test_json_encoding_survives_the_round_trip(proxy):
    """OTLP/HTTP allows JSON; the proxy must answer in the format it got."""
    client, upstream = proxy
    response = post_traces(client, make_request([LLM_SPAN]), content_type=otlp.JSON)

    assert response.status_code == 200
    assert upstream.last["headers"]["content-type"] == otlp.JSON
    parsed = json_format.Parse(
        upstream.last["content"].decode("utf-8"), otlp.TRACES_REQUEST()
    )
    span = parsed.resource_spans[0].scope_spans[0].spans[0]
    assert otlp.to_dict(span.attributes)[sc.MEM_PROCESS_RSS] > 0


def test_gzipped_bodies_are_accepted(proxy):
    client, upstream = proxy
    body = gzip.compress(otlp.encode(make_request([LLM_SPAN]), otlp.PROTOBUF))
    response = client.post(
        "/v1/traces",
        content=body,
        headers={"content-type": otlp.PROTOBUF, "content-encoding": "gzip"},
    )
    assert response.status_code == 200
    # Forwarded decompressed, so content-encoding must not claim otherwise.
    assert "content-encoding" not in {k.lower() for k in upstream.last["headers"]}
    span = upstream.decoded().resource_spans[0].scope_spans[0].spans[0]
    assert otlp.to_dict(span.attributes)[sc.MEM_PROCESS_RSS] > 0


def test_metrics_and_logs_pass_through_untouched(proxy):
    client, upstream = proxy
    payload = b"\x0a\x00"
    for path in ("/v1/metrics", "/v1/logs"):
        response = client.post(
            path, content=payload, headers={"content-type": otlp.PROTOBUF}
        )
        assert response.status_code == 200
        assert upstream.last["content"] == payload
        assert upstream.last["url"].endswith(path)


# ----------------------------------------------------------------------
# failure handling
# ----------------------------------------------------------------------
def test_upstream_outage_returns_503_so_the_sdk_retries(harness):
    """Swallowing a batch would silently lose telemetry the SDK would have
    happily re-sent."""
    with TestClient(create_app(otlp_enabled=False)) as client:
        from llmobs import server

        server._state["client"] = FakeUpstream(raises=httpx.ConnectError("refused"))
        response = post_traces(client, make_request([LLM_SPAN]))
    assert response.status_code == 503


def test_upstream_error_status_is_mirrored_not_masked(harness):
    with TestClient(create_app(otlp_enabled=False)) as client:
        from llmobs import server

        server._state["client"] = FakeUpstream(status_code=429)
        response = post_traces(client, make_request([LLM_SPAN]))
    assert response.status_code == 429


def test_undecodable_body_is_forwarded_rather_than_dropped(proxy):
    """We cannot enrich it, but the Collector may still understand it."""
    client, upstream = proxy
    garbage = b"\xff\xfe not otlp at all"
    response = client.post(
        "/v1/traces", content=garbage, headers={"content-type": otlp.PROTOBUF}
    )
    assert response.status_code == 200
    assert upstream.last["content"] == garbage


def test_enrichment_failure_still_forwards_the_batch(proxy, monkeypatch):
    client, upstream = proxy
    from llmobs import server

    def explode(_self, _request):
        raise RuntimeError("enrichment bug")

    monkeypatch.setattr(server._state["enricher"].__class__, "process_traces", explode)
    response = post_traces(client, make_request([LLM_SPAN]))

    assert response.status_code == 200
    assert len(upstream.requests) == 1  # forwarded unmodified


def test_empty_batch_is_harmless(proxy):
    client, upstream = proxy
    assert post_traces(client, otlp.TRACES_REQUEST()).status_code == 200
    assert len(upstream.requests) == 1


# ----------------------------------------------------------------------
# operational surface
# ----------------------------------------------------------------------
def test_stats_reflect_what_passed_through(proxy):
    client, _ = proxy
    for _ in range(5):
        post_traces(client, make_request([LLM_SPAN]))

    stats = client.get("/v1/stats").json()
    assert stats["totals"]["llm_spans"] == 5
    assert stats["totals"]["input_tokens"] == 600     # 5 x 120
    assert stats["totals"]["output_tokens"] == 1700   # 5 x 340
    assert stats["totals"]["batches_forwarded"] == 5
    assert stats["window"]["latency_ms"]["p50"] == 500.0
    assert stats["window"]["models"] == ["llama3.1:8b"]


def test_healthz_reports_live_device_memory(proxy):
    client, _ = proxy
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["memory"]["system_total_bytes"] > 0
    assert body["upstream"]


def test_api_key_gate(harness):
    with TestClient(create_app(otlp_enabled=False, ingest_api_key="s3cret")) as client:
        from llmobs import server

        server._state["client"] = FakeUpstream()
        request = make_request([LLM_SPAN])
        assert post_traces(client, request).status_code == 401
        ok = post_traces(client, request, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
