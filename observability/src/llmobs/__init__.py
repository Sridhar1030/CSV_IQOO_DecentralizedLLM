"""llmobs - an OTLP proxy that sits in front of your OpenTelemetry Collector.

Your app keeps using the OTel SDK unchanged. Point its exporter here:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://llmobs:8100

and every batch on its way to the Collector gets:

  * device RAM/CPU stamped onto each LLM span,
  * token, latency, TTFT and error metrics derived from the spans themselves,
  * node identity filled in where the app left it blank.

Built on OpenTelemetry (Apache-2.0), psutil (BSD-3-Clause), FastAPI (MIT).
"""

from __future__ import annotations

from .config import Config, load_config
from .enrich import BatchResult, Enricher, RollingStats
from .resources import MemorySnapshot, ResourceSampler
from .server import create_app
from .telemetry import Telemetry, get_telemetry, init, shutdown

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "create_app",
    "Enricher",
    "BatchResult",
    "RollingStats",
    "init",
    "shutdown",
    "get_telemetry",
    "Telemetry",
    "Config",
    "load_config",
    "ResourceSampler",
    "MemorySnapshot",
]
