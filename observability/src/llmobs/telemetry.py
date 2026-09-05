"""OpenTelemetry wiring: tracer provider, meter provider, and instruments.

One process calls `init()` once. Everything downstream (`record`, `ingest`,
`server`) pulls the already-built handles off the module-level singleton.
All exporters speak OTLP/HTTP so the only thing a node needs to reach is an
OpenTelemetry Collector (Apache-2.0).
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable, Sequence

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased, TraceIdRatioBased

from . import semconv as sc
from .config import Config, load_config
from .resources import ResourceSampler, host_attributes

log = logging.getLogger("llmobs")

_INSTRUMENTATION_SCOPE = "llmobs"
_SCOPE_VERSION = "0.1.0"


class Telemetry:
    """Holds the providers, instruments, and sampler for one process."""

    def __init__(
        self,
        config: Config,
        *,
        extra_span_processors: Sequence[SpanProcessor] = (),
        extra_metric_readers: Sequence[MetricReader] = (),
    ) -> None:
        self.config = config
        self.sampler = ResourceSampler(cache_ms=config.memory_cache_ms)
        self.resource = self._build_resource(config)
        self._tracer_provider: TracerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self._build_providers(extra_span_processors, extra_metric_readers)
        self.tracer = trace.get_tracer(
            _INSTRUMENTATION_SCOPE, _SCOPE_VERSION, tracer_provider=self._tracer_provider
        )
        self.meter = metrics.get_meter(
            _INSTRUMENTATION_SCOPE, _SCOPE_VERSION, meter_provider=self._meter_provider
        )
        self._build_instruments()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    @staticmethod
    def _build_resource(config: Config) -> Resource:
        attrs: dict[str, str | int] = {
            sc.SERVICE_NAME: config.service_name,
            sc.SERVICE_VERSION: config.service_version,
            sc.SERVICE_INSTANCE_ID: config.instance_id,
            sc.DEPLOYMENT_ENVIRONMENT: config.deployment_environment,
            sc.NODE_ROLE: config.node_role,
        }
        # A relay's identity must not shadow the origin node's on relayed data
        # points; see Config.relay_mode.
        attrs[sc.COLLECTOR_ID if config.relay_mode else sc.NODE_ID] = config.node_id
        attrs.update(host_attributes())
        return Resource.create(attrs)

    def _build_providers(
        self,
        extra_span_processors: Sequence[SpanProcessor] = (),
        extra_metric_readers: Sequence[MetricReader] = (),
    ) -> None:
        cfg = self.config

        sampler = (
            ALWAYS_ON
            if cfg.trace_sample_ratio >= 1.0
            else ParentBased(root=TraceIdRatioBased(cfg.trace_sample_ratio))
        )
        tp = TracerProvider(resource=self.resource, sampler=sampler)

        metric_readers = []
        if cfg.otlp_enabled:
            headers = cfg.otlp_header_dict()
            try:
                tp.add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(
                            endpoint=f"{cfg.otlp_endpoint.rstrip('/')}/v1/traces",
                            headers=headers or None,
                            timeout=cfg.otlp_timeout_s,
                        )
                    )
                )
                metric_readers.append(
                    PeriodicExportingMetricReader(
                        OTLPMetricExporter(
                            endpoint=f"{cfg.otlp_endpoint.rstrip('/')}/v1/metrics",
                            headers=headers or None,
                            timeout=cfg.otlp_timeout_s,
                        ),
                        export_interval_millis=cfg.export_interval_ms,
                    )
                )
            except Exception:  # pragma: no cover - bad endpoint should not kill the app
                log.exception("llmobs: OTLP exporter setup failed; continuing without it")

        if cfg.console_export:
            tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            metric_readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=cfg.export_interval_ms,
                )
            )

        for processor in extra_span_processors:
            tp.add_span_processor(processor)
        metric_readers.extend(extra_metric_readers)

        self._tracer_provider = tp
        self._meter_provider = MeterProvider(
            resource=self.resource,
            metric_readers=metric_readers,
            views=self._build_views(),
        )

    @staticmethod
    def _build_views() -> list[View]:
        """Override default histogram buckets - the SDK defaults stop at 10s,
        which lumps every slow generation into one overflow bucket."""
        return [
            View(
                instrument_name=sc.M_REQUEST_DURATION,
                aggregation=ExplicitBucketHistogramAggregation(
                    boundaries=sc.DURATION_BUCKETS_S
                ),
            ),
            View(
                instrument_name=sc.M_TTFT,
                aggregation=ExplicitBucketHistogramAggregation(
                    boundaries=sc.DURATION_BUCKETS_S
                ),
            ),
            View(
                instrument_name=sc.M_TOKENS_PER_REQUEST,
                aggregation=ExplicitBucketHistogramAggregation(
                    boundaries=sc.TOKEN_BUCKETS
                ),
            ),
        ]

    def _build_instruments(self) -> None:
        meter = self.meter

        # All derived from the spans passing through the proxy.
        self.tokens_total = meter.create_counter(
            name=sc.M_TOKENS_TOTAL,
            unit="{token}",
            description="Cumulative tokens in and out, split by input/output.",
        )
        self.tokens_per_request = meter.create_histogram(
            name=sc.M_TOKENS_PER_REQUEST,
            unit="{token}",
            description="Distribution of tokens per LLM request.",
        )
        self.request_duration = meter.create_histogram(
            name=sc.M_REQUEST_DURATION,
            unit="s",
            description="End-to-end LLM request duration, from the span timestamps.",
        )
        self.time_to_first_token = meter.create_histogram(
            name=sc.M_TTFT,
            unit="s",
            description="Time from request start to the first streamed token.",
        )
        self.requests_total = meter.create_counter(
            name=sc.M_REQUESTS_TOTAL,
            unit="{request}",
            description="Cumulative LLM requests, split by outcome.",
        )

        # --- resource gauges: pulled by the SDK on each export tick -----
        base = {sc.NODE_ID: self.config.node_id, sc.NODE_ROLE: self.config.node_role}

        def _gauge(getter):
            def callback(_options: CallbackOptions) -> Iterable[Observation]:
                snap = self.sampler.sample()
                yield Observation(getter(snap), base)

            return callback

        meter.create_observable_gauge(
            name=sc.M_PROC_MEM_RSS,
            callbacks=[_gauge(lambda s: s.process_rss_bytes)],
            unit="By",
            description="Resident set size of this LLM process.",
        )
        meter.create_observable_gauge(
            name=sc.M_PROC_MEM_PERCENT,
            callbacks=[_gauge(lambda s: s.process_memory_percent / 100.0)],
            unit="1",
            description="Fraction of host RAM held by this process.",
        )
        meter.create_observable_gauge(
            name=sc.M_SYS_MEM_USED,
            callbacks=[_gauge(lambda s: s.system_used_bytes)],
            unit="By",
            description="RAM in use on the host.",
        )
        meter.create_observable_gauge(
            name=sc.M_SYS_MEM_AVAILABLE,
            callbacks=[_gauge(lambda s: s.system_available_bytes)],
            unit="By",
            description="RAM available on the host.",
        )
        meter.create_observable_gauge(
            name=sc.M_SYS_MEM_TOTAL,
            callbacks=[_gauge(lambda s: s.system_total_bytes)],
            unit="By",
            description="Total RAM on the host.",
        )
        meter.create_observable_gauge(
            name=sc.M_SYS_MEM_PERCENT,
            callbacks=[_gauge(lambda s: s.system_percent / 100.0)],
            unit="1",
            description="Fraction of host RAM in use.",
        )
        meter.create_observable_gauge(
            name=sc.M_PROC_CPU_PERCENT,
            callbacks=[_gauge(lambda s: s.process_cpu_percent / 100.0)],
            unit="1",
            description="CPU utilization of this process (1.0 = one full core).",
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def flush(self, timeout_ms: int = 5000) -> None:
        """Force-export anything buffered. Call before a short-lived process exits."""
        if self._tracer_provider is not None:
            self._tracer_provider.force_flush(timeout_ms)
        if self._meter_provider is not None:
            self._meter_provider.force_flush(timeout_ms)

    def shutdown(self) -> None:
        if self._tracer_provider is not None:
            self._tracer_provider.shutdown()
        if self._meter_provider is not None:
            self._meter_provider.shutdown()


# ----------------------------------------------------------------------
# module-level singleton
# ----------------------------------------------------------------------
_lock = threading.Lock()
_telemetry: Telemetry | None = None


def init(
    *,
    set_global: bool = True,
    extra_span_processors: Sequence[SpanProcessor] = (),
    extra_metric_readers: Sequence[MetricReader] = (),
    **overrides,
) -> Telemetry:
    """Initialize telemetry for this process. Idempotent.

    Args:
        set_global: also install the providers as the OTel global ones, so
            third-party libraries already instrumented with OTel emit into
            the same pipeline.
        extra_span_processors: additional span processors (a second backend,
            an in-memory exporter for tests).
        extra_metric_readers: additional metric readers, e.g. a
            `PrometheusMetricReader` to expose /metrics directly.
        **overrides: any `Config` field.
    """
    global _telemetry
    with _lock:
        if _telemetry is not None:
            return _telemetry
        config = load_config(**overrides)
        telemetry = Telemetry(
            config,
            extra_span_processors=extra_span_processors,
            extra_metric_readers=extra_metric_readers,
        )
        if set_global:
            # set_*_provider is a no-op-with-warning if something already set one.
            trace.set_tracer_provider(telemetry._tracer_provider)
            metrics.set_meter_provider(telemetry._meter_provider)
        _telemetry = telemetry
        log.info(
            "llmobs initialized service=%s node=%s otlp=%s",
            config.service_name,
            config.node_id,
            config.otlp_endpoint if config.otlp_enabled else "disabled",
        )
        return telemetry


def get_telemetry() -> Telemetry:
    """Return the process telemetry, initializing with defaults if needed."""
    if _telemetry is None:
        return init()
    return _telemetry


def shutdown() -> None:
    global _telemetry
    with _lock:
        if _telemetry is not None:
            _telemetry.shutdown()
            _telemetry = None


def reset_for_testing() -> None:
    """Drop the singleton without touching OTel globals. Tests only."""
    global _telemetry
    with _lock:
        _telemetry = None
