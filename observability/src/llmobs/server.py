"""The endpoint: an OTLP passthrough proxy that sits in front of the Collector.

Your application keeps using the OTel SDK exactly as it does today. The only
change is where it exports to:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://llmobs:8100

Every OTLP batch is decoded, enriched with the things the SDK cannot know
(device RAM) or does not emit (token/latency/error metrics derived from the
spans), then forwarded upstream in the same format it arrived in.

Design rules this follows:

  * **Transparent.** Same OTLP paths, same content types (protobuf or JSON,
    gzipped or not), same response messages. The SDK cannot tell it is there.
  * **Honest about failure.** If the upstream Collector rejects a batch, the
    proxy returns its status so the SDK retries. It never swallows a batch and
    reports success.
  * **Never the reason you lose data.** If enrichment itself throws, the batch
    is forwarded unmodified rather than dropped.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse

from . import otlp
from .config import Config, load_config
from .enrich import Enricher, RollingStats
from .telemetry import init

log = logging.getLogger("llmobs.server")

_state: dict[str, Any] = {}

# Headers that must not be copied to the upstream request: they describe the
# inbound connection or a body we may have re-encoded at a different length.
_HOP_BY_HOP = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "upgrade", "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "content-encoding",  # we forward decompressed
}


def _require_api_key(config: Config):
    """Shared-secret gate. Empty key = open, which is fine on a private mesh
    network and not fine on the internet."""

    async def dependency(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        if not config.ingest_api_key:
            return
        if x_api_key != config.ingest_api_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    return dependency


def create_app(**config_overrides) -> FastAPI:
    config = load_config(**config_overrides)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        telemetry = init(
            service_name=config.service_name,
            node_id=config.node_id,
            node_role=config.node_role,
            otlp_endpoint=config.otlp_endpoint,
            otlp_enabled=config.otlp_enabled,
            console_export=config.console_export,
            # The proxy emits metrics on behalf of the apps behind it, so it
            # must not claim llmobs.node.id on its own Resource.
            relay_mode=True,
        )
        stats = RollingStats(window=config.stats_window)
        _state.update(
            telemetry=telemetry,
            stats=stats,
            enricher=Enricher(telemetry, stats),
            client=httpx.AsyncClient(timeout=config.forward_timeout_s),
        )
        log.info(
            "llmobs proxy listening on %s:%s -> upstream %s",
            config.server_host, config.server_port, config.otlp_endpoint,
        )
        try:
            yield
        finally:
            await _state["client"].aclose()
            telemetry.flush()
            telemetry.shutdown()
            _state.clear()

    app = FastAPI(
        title="llmobs",
        version="0.1.0",
        description=(
            "OTLP passthrough proxy for LLM applications. Enriches spans with "
            "device RAM and derives token, latency and error metrics, then "
            "forwards to an OpenTelemetry Collector."
        ),
        lifespan=lifespan,
    )
    app.state.config = config
    api_key_dep = _require_api_key(config)

    # ------------------------------------------------------------------
    # OTLP signal endpoints
    # ------------------------------------------------------------------
    @app.post("/v1/traces", dependencies=[Depends(api_key_dep)])
    async def traces(request: Request) -> Response:
        """Enrich spans, derive metrics, forward upstream."""
        body = await request.body()
        content_type = request.headers.get("content-type", otlp.PROTOBUF)
        encoding = request.headers.get("content-encoding")

        enricher: Enricher = _state["enricher"]
        stats: RollingStats = _state["stats"]

        try:
            decoded = otlp.decode(body, content_type, otlp.TRACES_REQUEST, encoding)
        except otlp.DecodeError as exc:
            # Undecodable means we cannot enrich, but the Collector might still
            # want it. Pass it through untouched rather than dropping it.
            log.warning("llmobs: %s - forwarding unmodified", exc)
            return await _forward(request, body, "/v1/traces", otlp.TRACES_RESPONSE, content_type)

        try:
            result = enricher.process_traces(decoded)
            payload = otlp.encode(decoded, content_type)
        except Exception:
            # Enrichment is a bonus. Losing telemetry because of it is not.
            log.exception("llmobs: enrichment failed - forwarding unmodified")
            return await _forward(request, body, "/v1/traces", otlp.TRACES_RESPONSE, content_type)

        response = await _forward(
            request, payload, "/v1/traces", otlp.TRACES_RESPONSE, content_type
        )
        stats.record_batch(result, forwarded=response.status_code < 300)
        return response

    @app.post("/v1/metrics", dependencies=[Depends(api_key_dep)])
    async def metrics_signal(request: Request) -> Response:
        """Metrics the app emits itself pass straight through."""
        return await _forward(
            request, await request.body(), "/v1/metrics", otlp.METRICS_RESPONSE,
            request.headers.get("content-type", otlp.PROTOBUF),
        )

    @app.post("/v1/logs", dependencies=[Depends(api_key_dep)])
    async def logs_signal(request: Request) -> Response:
        """Logs pass straight through."""
        return await _forward(
            request, await request.body(), "/v1/logs", otlp.LOGS_RESPONSE,
            request.headers.get("content-type", otlp.PROTOBUF),
        )

    # ------------------------------------------------------------------
    async def _forward(
        request: Request, body: bytes, path: str, response_type, content_type: str
    ) -> Response:
        """Relay a body upstream and mirror the Collector's answer back.

        The app's SDK is the retry mechanism, so an upstream failure must
        surface as a non-2xx here, not as a fake success.
        """
        client: httpx.AsyncClient = _state["client"]
        url = f"{request.app.state.config.otlp_endpoint.rstrip('/')}{path}"
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _HOP_BY_HOP
        }
        headers["content-type"] = content_type

        try:
            upstream = await client.post(url, content=body, headers=headers)
        except httpx.HTTPError as exc:
            log.warning("llmobs: upstream %s unreachable: %s", url, exc)
            # 503 tells a well-behaved OTLP exporter to back off and retry.
            return Response(
                content=otlp.empty_response(response_type, content_type),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                media_type=content_type,
            )

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", content_type),
        )

    # ------------------------------------------------------------------
    # operational endpoints
    # ------------------------------------------------------------------
    @app.get("/v1/stats", summary="Rolling summary of what has passed through")
    async def get_stats(limit: int = 20) -> dict[str, Any]:
        stats: RollingStats | None = _state.get("stats")
        if stats is None:  # pragma: no cover
            raise HTTPException(status_code=503, detail="proxy not ready")
        return stats.snapshot(limit=max(0, min(limit, 200)))

    @app.get("/healthz", summary="Liveness probe with a live memory snapshot")
    async def healthz() -> JSONResponse:
        telemetry = _state.get("telemetry")
        snapshot = telemetry.sampler.sample() if telemetry else None
        return JSONResponse(
            {
                "status": "ok",
                "service": config.service_name,
                "node_id": config.node_id,
                "upstream": config.otlp_endpoint,
                "memory": snapshot.as_dict() if snapshot else None,
            }
        )

    @app.get("/metrics", summary="Prometheus exposition (optional extra)")
    async def prometheus_metrics() -> PlainTextResponse:
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail=(
                    "Requires the optional extra: pip install 'llmobs[prometheus]'. "
                    "The default path is OTLP -> Collector -> Prometheus scrape."
                ),
            )
        return PlainTextResponse(
            generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST
        )

    return app


app = create_app()
