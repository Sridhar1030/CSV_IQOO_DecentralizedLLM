"""Environment-driven configuration for the llmobs telemetry layer.

Every knob is settable via env var so a decentralized node can be configured
without code changes (containers, systemd units, k8s manifests).
"""

from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_node_id() -> str:
    """Stable-ish identity for this process on this host."""
    explicit = os.getenv("LLMOBS_NODE_ID")
    if explicit:
        return explicit
    host = socket.gethostname()
    # Include a short pid-derived suffix so two workers on one host stay distinct.
    return f"{host}-{os.getpid()}"


@dataclass
class Config:
    """Runtime configuration.

    Attributes that end up on *metrics* must stay low-cardinality; anything
    per-request (request ids, user ids, prompts) belongs on spans only.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("LLMOBS_ENABLED", True))

    # --- identity -------------------------------------------------------
    service_name: str = field(
        default_factory=lambda: os.getenv("LLMOBS_SERVICE_NAME", "llm-node")
    )
    service_version: str = field(
        default_factory=lambda: os.getenv("LLMOBS_SERVICE_VERSION", "0.1.0")
    )
    node_id: str = field(default_factory=_default_node_id)
    deployment_environment: str = field(
        default_factory=lambda: os.getenv("LLMOBS_ENV", "development")
    )
    # Logical role of this node in the decentralized mesh (gateway, worker, router...).
    node_role: str = field(default_factory=lambda: os.getenv("LLMOBS_NODE_ROLE", "worker"))

    # Relay mode: this process emits telemetry on behalf of OTHER nodes, so its
    # own identity goes on the Resource as `llmobs.collector.id` rather than
    # `llmobs.node.id`. Without this, a Collector configured with
    # `resource_to_telemetry_conversion` promotes the relay's resource
    # attributes to metric labels, silently overwriting the origin node id on
    # every data point - every edge node's tokens get attributed to the relay.
    relay_mode: bool = field(default_factory=lambda: _env_bool("LLMOBS_RELAY_MODE", False))

    # --- export ---------------------------------------------------------
    # The real OpenTelemetry Collector this proxy sits in front of. Used both
    # to forward the app's OTLP batches and to export the proxy's own derived
    # metrics. Signal paths (/v1/traces, /v1/metrics) are appended.
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "LLMOBS_UPSTREAM_ENDPOINT",
            os.getenv("LLMOBS_OTLP_ENDPOINT", "http://localhost:4318"),
        )
    )
    # How long to wait on the upstream Collector before failing a forward.
    # On failure the proxy returns 503 and the app's SDK retries - that is the
    # OTLP contract, and it is why the proxy must not silently swallow errors.
    forward_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("LLMOBS_FORWARD_TIMEOUT_S", "10"))
    )
    otlp_headers: str = field(
        default_factory=lambda: os.getenv("LLMOBS_OTLP_HEADERS", "")
    )
    otlp_timeout_s: int = field(
        default_factory=lambda: _env_int("LLMOBS_OTLP_TIMEOUT_S", 10)
    )
    export_interval_ms: int = field(
        default_factory=lambda: _env_int("LLMOBS_EXPORT_INTERVAL_MS", 15_000)
    )
    # Emit spans/metrics to stdout as well - handy when bringing a node up.
    console_export: bool = field(
        default_factory=lambda: _env_bool("LLMOBS_CONSOLE_EXPORT", False)
    )
    # Turn off the network exporters entirely (tests, air-gapped nodes).
    otlp_enabled: bool = field(
        default_factory=lambda: _env_bool("LLMOBS_OTLP_ENABLED", True)
    )

    # --- sampling -------------------------------------------------------
    # Fraction of traces to keep. 1.0 = record everything.
    trace_sample_ratio: float = field(
        default_factory=lambda: float(os.getenv("LLMOBS_TRACE_SAMPLE_RATIO", "1.0"))
    )
    # psutil readings are cached this long so a hot request path does not
    # re-walk /proc for every single call.
    memory_cache_ms: int = field(
        default_factory=lambda: _env_int("LLMOBS_MEMORY_CACHE_MS", 250)
    )
    # Attach a RAM snapshot to every LLM span (in addition to the gauges).
    memory_on_span: bool = field(
        default_factory=lambda: _env_bool("LLMOBS_MEMORY_ON_SPAN", True)
    )
    # By default only spans that look like LLM calls get a RAM stamp. Turn on
    # to stamp every span passing through - accurate, but it inflates every
    # HTTP/DB span in your app with eight extra attributes.
    enrich_all_spans: bool = field(
        default_factory=lambda: _env_bool("LLMOBS_ENRICH_ALL_SPANS", False)
    )

    # --- ingest server --------------------------------------------------
    server_host: str = field(default_factory=lambda: os.getenv("LLMOBS_HOST", "0.0.0.0"))
    server_port: int = field(default_factory=lambda: _env_int("LLMOBS_PORT", 8100))
    # Shared-secret gate for the ingest endpoint. Empty = open (dev only).
    ingest_api_key: str = field(default_factory=lambda: os.getenv("LLMOBS_API_KEY", ""))
    # How many recent events /v1/stats keeps in memory.
    stats_window: int = field(default_factory=lambda: _env_int("LLMOBS_STATS_WINDOW", 1000))

    # service.instance.id identifies this *process*. It defaults to a fresh id
    # per start, so every restart creates a new Prometheus series. Pin it via
    # LLMOBS_INSTANCE_ID when a node's identity should outlive its process.
    instance_id: str = field(
        default_factory=lambda: os.getenv("LLMOBS_INSTANCE_ID") or uuid.uuid4().hex[:12]
    )

    def otlp_header_dict(self) -> dict[str, str]:
        """Parse `k=v,k2=v2` into a header mapping."""
        headers: dict[str, str] = {}
        for pair in self.otlp_headers.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            key, _, value = pair.partition("=")
            headers[key.strip()] = value.strip()
        return headers


def load_config(**overrides) -> Config:
    """Build a Config from the environment, with explicit kwargs winning."""
    cfg = Config()
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise TypeError(f"unknown config option: {key!r}")
        setattr(cfg, key, value)
    return cfg
