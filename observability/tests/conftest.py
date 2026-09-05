import sys
from pathlib import Path

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmobs import telemetry as telemetry_mod  # noqa: E402


class Harness:
    """A Telemetry wired to in-memory exporters so tests can assert on
    exactly what would have been shipped over OTLP."""

    def __init__(self, telemetry, span_exporter, metric_reader):
        self.telemetry = telemetry
        self.span_exporter = span_exporter
        self.metric_reader = metric_reader
        self._collected = None

    def spans(self):
        return self.span_exporter.get_finished_spans()

    def refresh(self):
        """Force a new collection cycle."""
        self._collected = None
        return self

    def metrics(self):
        """Flatten the collected metrics into {name: [(value, attrs), ...]}.

        Cached: every `get_metrics_data()` call runs a real collection cycle,
        and a gauge only reports on the cycle after its last `set()`, so
        re-collecting mid-assertion would make gauge points vanish.
        """
        if self._collected is not None:
            return self._collected
        data = self.metric_reader.get_metrics_data()
        out = {}
        if data is None:
            self._collected = out
            return out
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    points = out.setdefault(metric.name, [])
                    for point in metric.data.data_points:
                        value = getattr(point, "value", None)
                        if value is None:
                            value = getattr(point, "sum", None)
                        points.append((value, dict(point.attributes)))
        self._collected = out
        return out

    def metric_points(self, name):
        """(value, attributes) pairs. For histograms `value` is the cumulative
        SUM of observations, not an individual recording."""
        return self.metrics().get(name, [])


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setenv("LLMOBS_OTLP_ENABLED", "false")
    telemetry_mod.reset_for_testing()

    span_exporter = InMemorySpanExporter()
    metric_reader = InMemoryMetricReader()
    telemetry = telemetry_mod.init(
        set_global=False,
        otlp_enabled=False,
        service_name="test-node",
        node_id="node-test",
        node_role="worker",
        extra_span_processors=[SimpleSpanProcessor(span_exporter)],
        extra_metric_readers=[metric_reader],
    )
    yield Harness(telemetry, span_exporter, metric_reader)
    telemetry_mod.reset_for_testing()
