# llmobs — Observability Layer

An OTLP proxy that sits between your LLM application and your OpenTelemetry
Collector, adding what an LLM workload needs but the OTel SDK does not provide:
**device RAM per request**, and **token / latency / TTFT / error metrics derived
from the spans you already emit**.

Your application changes one environment variable. Nothing else.

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://llmobs:8100
```

> Running and monitoring commands: **[COMMANDS.md](COMMANDS.md)**

---

## 1. System context

```
┌─────────────────────────────────────────────────────────────────────┐
│  NODE  (one physical/virtual machine running inference)             │
│                                                                     │
│   ┌──────────────────────┐          ┌───────────────────────────┐   │
│   │  Your LLM app        │  OTLP    │  llmobs proxy   :8100     │   │
│   │  ─────────────────   │ ───────▶ │  ──────────────────────   │   │
│   │  OTel SDK            │  HTTP    │  1. decode OTLP           │   │
│   │  gen_ai.* spans      │          │  2. stamp device RAM      │   │
│   │  (no llmobs import)  │          │  3. derive metrics        │   │
│   └──────────────────────┘          │  4. forward upstream      │   │
│                                     └────────────┬──────────────┘   │
│    psutil reads THIS box's RAM ─────────────────▶│                  │
└──────────────────────────────────────────────────┼──────────────────┘
                                                   │ OTLP/HTTP
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │  OpenTelemetry Collector    │
                                    │  batch · memory_limiter     │
                                    └──────┬───────────────┬──────┘
                                    traces │               │ metrics
                                           ▼               ▼
                                     ┌──────────┐   ┌──────────────┐
                                     │  Jaeger  │   │  Prometheus  │
                                     │  :16686  │   │    :9090     │
                                     └──────────┘   └──────────────┘
```

The application talks OTLP, not a custom SDK. That is the whole design premise:
**the integration contract is a wire protocol, not a library.** Any language's
OTel SDK works, and removing llmobs is a one-line rollback.

---

## 2. Why a proxy instead of a library

The obvious alternative is a Python package the app imports. It was built that
way first and then replaced, for three reasons:

| Concern | Library | Proxy |
|---|---|---|
| Language support | Python only | Any OTel SDK |
| App code changes | Import, wrap every call site | One env var |
| Rollback | Code change + redeploy | Re-point the env var |
| Version coupling | App pinned to our releases | Independent |
| Device RAM | Available | Available (runs on the device) |

The cost is one decode/enrich/re-encode per batch. Since the SDK's
`BatchSpanProcessor` already amortizes export off the request path, this does
not touch inference latency.

---

## 3. Request path

What happens to a single LLM call, end to end:

```
 1. App creates a span
      "chat llama3.1:8b"  { gen_ai.system, gen_ai.request.model,
                            gen_ai.usage.input_tokens, ... }

 2. BatchSpanProcessor buffers it, exports a batch to :8100/v1/traces
      Content-Type: application/x-protobuf   (or JSON, optionally gzipped)

 3. server.py     decodes the batch                        [otlp.py]
 4. enrich.py     samples device RAM once for the batch    [resources.py]
 5.               for each span:
                    • LLM-related?  → stamp RAM/CPU attributes
                    • model call?   → derive metrics       [telemetry.py]
 6. server.py     re-encodes in the SAME content type      [otlp.py]
 7.               POSTs to the Collector, mirrors its status back

 8. Collector → Jaeger (traces) and Prometheus (metrics)
```

Steps 3–7 are transparent: same OTLP paths, same content types, same response
messages, upstream status codes passed through unchanged. The SDK cannot tell
the proxy is there.

**Two different predicates in step 5**, and the distinction matters:

- *LLM-related* (has any `gen_ai.*` attribute) → gets the RAM stamp.
- *Model call* (has `gen_ai.request.model` **or** `gen_ai.system`) → feeds metrics.

A gateway that copies token totals onto its parent routing span is common. If
both spans fed the metrics, every token in the fleet would count twice. The
parent still gets RAM; it just does not get counted.

---

## 4. Module map

```
src/llmobs/
├── server.py      The endpoint. FastAPI app: receive OTLP → enrich → forward.
│                  Owns failure policy (what happens when upstream is down).
├── enrich.py      The "extra things". Walks the decoded batch, stamps device
│                  RAM, derives metrics. Also the rolling /v1/stats window.
├── otlp.py        OTLP codec. protobuf ⇄ JSON, gzip, AnyValue ⇄ Python,
│                  attribute get/set. Zero business logic.
├── telemetry.py   OTel providers + the instruments the proxy writes.
│                  Histogram buckets tuned for LLM latency (to 82s, not 10s).
├── resources.py   psutil sampling, TTL-cached so a hot path doesn't re-walk
│                  procfs for every span.
├── config.py      Env-driven configuration. No code change to reconfigure.
├── semconv.py     Attribute + instrument name constants.
└── __main__.py    `llmobs-server` CLI.
```

Dependency direction is one-way: `server → enrich → {otlp, telemetry, resources}`.
`otlp.py` knows nothing about LLMs; `enrich.py` knows nothing about HTTP.

---

## 5. Deployment topology

**Run one proxy per device, as a sidecar.** This is the single most important
operational fact.

```
   CORRECT                              WRONG
   ┌──────────────┐                     ┌──────────────┐
   │ node-a       │                     │ node-a       │──┐
   │  app         │                     │  app         │  │
   │  llmobs ─────┼──┐                  └──────────────┘  │
   └──────────────┘  │                  ┌──────────────┐  ├──▶ one central
   ┌──────────────┐  ├──▶ Collector     │ node-b       │  │    llmobs ──▶ Collector
   │ node-b       │  │                  │  app         │──┘
   │  app         │  │                  └──────────────┘
   │  llmobs ─────┼──┘                  every span gets the PROXY's RAM,
   └──────────────┘                     not the inference node's
   each span carries its
   own node's real RAM
```

The proxy reads memory with `psutil` on the machine it runs on. Centralized, it
would faithfully report its own idle container's memory for every node in the
fleet — technically working, completely useless.

Token, latency, TTFT and error enrichment **do** work centrally. Only RAM is
position-dependent. If you do not need RAM, a central deployment is fine.

---

## 6. Data model: spans vs metrics

The split is deliberate and is the difference between a metrics backend that
survives and one that falls over.

```
   SPAN  (Jaeger)                        METRIC  (Prometheus)
   ─────────────────                     ────────────────────
   unbounded cardinality OK              STRICTLY bounded labels
   one record per request                one time series per label combination

   request_id      ✓                     request_id      ✗  1 series/request
   trace_id        ✓                     trace_id        ✗
   prompt/response ✓                     model           ✓  bounded set
   RAM snapshot    ✓                     system          ✓
   model, system   ✓                     operation       ✓
   duration        ✓                     service_name    ✓
                                         node_id         ✓
                                         outcome         ✓
                                         error_type      ✓
```

Putting a per-request id on a metric label creates one time series per request.
`enrich.py` builds the metric label set explicitly rather than copying span
attributes, and a test asserts high-cardinality keys never leak into it.

**Instrument names are namespaced `llmobs.*`** rather than reusing
`gen_ai.client.token.usage`. If your app already emits the spec instruments,
reusing those names would double-count silently.

---

## 7. Failure model

The proxy sits in the telemetry path, so its failure behaviour is part of its
contract.

| Situation | Behaviour | Why |
|---|---|---|
| Collector unreachable | Return **503** | The SDK's exporter retries. Accepting and dropping would lose data the app would happily have re-sent. |
| Collector returns 4xx/5xx | Mirror the status | The SDK decides retry vs discard, as it would talking to the Collector directly. |
| Body is not valid OTLP | Forward **unmodified** | We cannot enrich it; the Collector may still understand it. |
| Enrichment throws | Forward **unmodified** | Enrichment is a bonus. Losing telemetry is not. |
| `psutil` read fails | Zeroed snapshot, span still flows | Never break the pipeline over a resource reading. |

The invariant: **llmobs is never the reason you lose telemetry.**

---

## 8. Metrics produced

| Name | Type | Notes |
|---|---|---|
| `llmobs_tokens_total` | counter | by `gen_ai_token_type` (input/output), model, system, service, node |
| `llmobs_tokens_per_request` | histogram | per-request distribution |
| `llmobs_request_duration_seconds` | histogram | from span timestamps; buckets to 82s |
| `llmobs_request_time_to_first_token_seconds` | histogram | ms or s on the span, normalized |
| `llmobs_requests_total` | counter | by `outcome`, `error_type` |
| `llmobs_process_memory_rss_bytes` | gauge | device process RSS |
| `llmobs_system_memory_utilization_ratio` | gauge | host RAM, 0–1 |
| `llmobs_system_memory_{used,available,total}_bytes` | gauge | host RAM |
| `llmobs_process_cpu_utilization_ratio` | gauge | 1.0 = one full core |

Recording rules and alerts (memory pressure, latency, error rate, silent node)
are in [`deploy/rules.yml`](deploy/rules.yml).

---

## 9. Instrumenting your app

Nothing llmobs-specific — standard OTel with the
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

```python
with tracer.start_as_current_span(f"chat {model}", kind=SpanKind.CLIENT) as span:
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.system", "ollama")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
    ...
    span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
```

Using an auto-instrumentation library (OpenLLMetry, OpenInference, the OTel
GenAI instrumentations)? You already emit these — there is nothing to do.

Alternate token spellings are also understood, since not every library tracks
the current semconv: `gen_ai.usage.prompt_tokens`, `llm.usage.completion_tokens`,
`llm.token_count.prompt`.

Optional extras the proxy picks up:

| Attribute | Effect |
|---|---|
| `llmobs.response.time_to_first_token_ms` | Feeds the TTFT histogram (`gen_ai.server.time_to_first_token` in seconds also works) |
| `error.type` | Becomes the `error_type` metric label |
| `llmobs.node.id` on the Resource | Node identity; the proxy fills it in if omitted, and never overwrites yours |

See [`examples/sample_llm_app.py`](examples/sample_llm_app.py) — plain OTel, no
llmobs import anywhere in it.

---

## 10. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/traces` | OTLP traces — enriched, then forwarded |
| `POST` | `/v1/metrics` | OTLP metrics — passthrough |
| `POST` | `/v1/logs` | OTLP logs — passthrough |
| `GET` | `/v1/stats` | Rolling summary of what passed through |
| `GET` | `/healthz` | Liveness + live device memory |
| `GET` | `/metrics` | Prometheus exposition (optional `[prometheus]` extra) |

---

## 11. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLMOBS_UPSTREAM_ENDPOINT` | `http://localhost:4318` | The real Collector to forward to |
| `LLMOBS_PORT` | `8100` | Listen port |
| `LLMOBS_NODE_ID` | `<hostname>-<pid>` | Node identity, used when the app omits it |
| `LLMOBS_NODE_ROLE` | `worker` | `gateway` / `router` / `worker` |
| `LLMOBS_MEMORY_ON_SPAN` | `true` | Stamp a RAM snapshot on LLM spans |
| `LLMOBS_ENRICH_ALL_SPANS` | `false` | Stamp every span, not just LLM ones |
| `LLMOBS_MEMORY_CACHE_MS` | `250` | TTL on psutil reads |
| `LLMOBS_FORWARD_TIMEOUT_S` | `10` | Upstream timeout before returning 503 |
| `LLMOBS_EXPORT_INTERVAL_MS` | `15000` | Derived-metric export cadence |
| `LLMOBS_API_KEY` | *(empty)* | Require `X-API-Key` on the OTLP endpoints |
| `LLMOBS_INSTANCE_ID` | random per start | Pin to keep series stable across restarts |
| `LLMOBS_CONSOLE_EXPORT` | `false` | Also print to stdout |

### Operational notes

- **Unauthenticated by default.** Fine on a private mesh network; set
  `LLMOBS_API_KEY` before exposing it further.
- **Dead nodes linger** for `metric_expiration` (`2m` in the Collector config)
  before their series disappear.
- **Restarts create new series** unless you pin `LLMOBS_INSTANCE_ID`, because
  `service.instance.id` identifies a *process*.
- **Jaeger stores traces in memory** in this compose file. Swap in Cassandra or
  Elasticsearch (both permissively licensed) before depending on it.
- **`rate()` needs sustained traffic.** A short burst that stops leaves every
  rate at 0 and `histogram_quantile` returning `NaN` — Prometheus only ever
  scraped the flat post-burst value.

---

## 12. Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest tests/ -q          # 27 tests
```

Tests build real OTLP protobuf payloads and assert on what the proxy actually
forwards, with in-memory OTel exporters capturing the derived metrics. No mocks
of our own code, no network.

Three of the tests exist because the bug happened, not because it was predicted:

- `test_parent_routing_span_does_not_double_count_tokens` — a gateway copying
  token totals onto its parent span doubled every token in the fleet.
- `test_metric_labels_stay_low_cardinality` — one time series per request.
- `test_upstream_outage_returns_503_so_the_sdk_retries` — silently swallowing a
  batch loses data the SDK would have re-sent.

---

## 13. Licenses

Every component is permissively licensed. No copyleft anywhere in the stack.

| Component | License |
|---|---|
| OpenTelemetry (API, SDK, proto, Collector) | Apache-2.0 |
| Prometheus | Apache-2.0 |
| Jaeger | Apache-2.0 |
| FastAPI, Pydantic | MIT |
| Uvicorn, httpx, psutil | BSD-3-Clause |
| **llmobs** (this project) | Apache-2.0 |

**Grafana is deliberately excluded.** Grafana, Loki and Tempo relicensed to
**AGPLv3** in 2021, which is not permissive. Jaeger's UI covers traces and
Prometheus' own UI covers metrics. If AGPL is acceptable, Grafana points at this
stack with no code changes — add a Prometheus source at `http://prometheus:9090`
and a Jaeger one at `http://jaeger:16686`. Fully permissive alternatives:
**Perses** (Apache-2.0) for dashboards, **VictoriaMetrics** (Apache-2.0) for
long-term metric storage.
