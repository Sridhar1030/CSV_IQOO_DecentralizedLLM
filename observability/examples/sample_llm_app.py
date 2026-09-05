"""A normal LLM app instrumented with the plain OpenTelemetry SDK.

Note what is NOT here: any import of llmobs. This is stock OTel. The only
thing that makes it observable through the proxy is where it exports to.

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:8100 python examples/sample_llm_app.py

Point it at the Collector directly (:4318) instead and you still get traces -
you just lose the device RAM on each span and all the derived metrics, which
is precisely what the proxy adds.
"""

from __future__ import annotations

import argparse
import os
import random
import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

MODELS = ["llama3.1:8b", "qwen2.5:7b", "phi3:mini"]

PROMPTS = [
    "Summarize the tradeoffs of running inference at the edge.",
    "Explain W3C trace context to a backend engineer.",
    "What causes a node to OOM during a long generation?",
]


def setup_tracing(endpoint: str, service_name: str, node_id: str) -> trace.Tracer:
    """Textbook OTel SDK setup - nothing llmobs-specific."""
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service_name,
                "service.version": "1.4.0",
                # Optional. If omitted, the proxy fills it in from the device
                # it runs on; if set, the app's value always wins.
                "llmobs.node.id": node_id,
            }
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("sample-llm-app")


def handle_request(tracer: trace.Tracer, request_id: str) -> None:
    """A gateway span with a nested model call - two spans, one trace."""
    model = random.choice(MODELS)

    with tracer.start_as_current_span("route request") as route_span:
        route_span.set_attribute("gen_ai.operation.name", "route")
        route_span.set_attribute("llmobs.request.id", request_id)

        # The LLM call. These gen_ai.* attributes are standard OTel semconv -
        # they are what the proxy keys off to derive token and latency metrics.
        with tracer.start_as_current_span(
            f"chat {model}", kind=trace.SpanKind.CLIENT
        ) as span:
            prompt = random.choice(PROMPTS)
            prompt_tokens = max(1, len(prompt) // 4)

            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", "ollama")
            span.set_attribute("gen_ai.request.model", model)
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)

            time.sleep(random.uniform(0.02, 0.09))               # prefill
            ttft_ms = random.uniform(30, 120)
            span.set_attribute("llmobs.response.time_to_first_token_ms", ttft_ms)

            completion_tokens = random.randint(60, 400)
            time.sleep(completion_tokens * 0.0006)                # decode

            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
            span.set_attribute("gen_ai.response.model", model)

            if random.random() < 0.06:
                span.set_attribute("error.type", "CudaOutOfMemoryError")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "cuda oom"))
                return

        route_span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
        route_span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:8100"),
        help="Where the OTel SDK exports. Point at the llmobs proxy.",
    )
    parser.add_argument("--service-name", default="sample-llm-app")
    parser.add_argument("--node-id", default=os.getenv("LLMOBS_NODE_ID", "app-node-1"))
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    tracer = setup_tracing(args.endpoint, args.service_name, args.node_id)
    print(f"  exporting to {args.endpoint} as {args.service_name}/{args.node_id}")

    for i in range(args.requests):
        handle_request(tracer, request_id=f"req-{i:04d}")
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{args.requests}] sent")
        time.sleep(args.delay)

    trace.get_tracer_provider().shutdown()   # flush the batch processor
    print("  done")


if __name__ == "__main__":
    main()
