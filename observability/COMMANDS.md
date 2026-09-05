# Commands

Every command here was run against this stack. Run them from this
`observability/` directory.

---

## 1. First time setup

Create the virtualenv and install the package:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
```

No `uv`? Use plain pip:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e '.[dev]'
```

Run the tests to confirm everything works:

```bash
./.venv/bin/python -m pytest tests/ -q
```

---

## 2. Start everything

Build and start the proxy, Collector, Jaeger and Prometheus:

```bash
docker compose up -d --build
```

Check all four came up:

```bash
docker compose ps
```

| What | URL |
|---|---|
| **llmobs proxy** — point your app here | http://localhost:8100 |
| **Jaeger UI** — traces | http://localhost:16686 |
| **Prometheus** — metrics | http://localhost:9090 |
| Collector, direct OTLP | http://localhost:4318 |

Wait until the proxy answers before sending traffic:

```bash
curl -s localhost:8100/healthz | jq
```

---

## 3. Point your app at the proxy

This is the only change your application needs:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:8100
```

In Docker Compose, add to your app service:

```yaml
environment:
  OTEL_EXPORTER_OTLP_ENDPOINT: http://llmobs:8100
```

---

## 4. Generate some traffic

The sample app is plain OpenTelemetry — it does not import llmobs:

```bash
./.venv/bin/python examples/sample_llm_app.py --endpoint http://localhost:8100 --requests 120 --delay 0.4
```

Two apps at once, so you can see per-service breakdowns:

```bash
./.venv/bin/python examples/sample_llm_app.py --endpoint http://localhost:8100 --service-name app-a --node-id node-1 --requests 120 --delay 0.4 & ./.venv/bin/python examples/sample_llm_app.py --endpoint http://localhost:8100 --service-name app-b --node-id node-2 --requests 120 --delay 0.5 &
```

> **Send sustained traffic, not a short burst.** `rate()` needs the counter to
> increase *within* its window. A quick burst that stops leaves every rate at 0
> and `histogram_quantile` returning `NaN`. Use `--requests 120 --delay 0.4`.

---

## 5. Monitor — the quick check

Is anything flowing? This needs no Prometheus:

```bash
curl -s localhost:8100/v1/stats | jq '.totals'
```

```
{
  "spans_seen": 492,        # everything that passed through
  "llm_spans": 246,         # LLM-related, got a RAM stamp
  "input_tokens": 3100,
  "output_tokens": 58595,
  "errors": 11,
  "batches_forwarded": 33,
  "batches_failed": 0       # if this climbs, the Collector is unreachable
}
```

Latency percentiles over the recent window:

```bash
curl -s localhost:8100/v1/stats | jq '.window | {latency_ms, models, nodes}'
```

The last few requests, one line each:

```bash
curl -s 'localhost:8100/v1/stats?limit=5' | jq -r '.recent[] | "\(.duration_ms)ms  \(.model)  \(.input_tokens)->\(.output_tokens)  \(.status)"'
```

Live device RAM as the proxy sees it:

```bash
curl -s localhost:8100/healthz | jq '{node_id, upstream, rss_mb: (.memory.process_rss_bytes/1000000|floor), host_ram_pct: .memory.system_percent}'
```

Watch it update every 2 seconds:

```bash
watch -n2 "curl -s localhost:8100/v1/stats | jq -c '.totals'"
```

---

## 6. Monitor — traces (Jaeger)

Open the UI and pick your service from the dropdown:

```bash
open http://localhost:16686
```

Which services have reported:

```bash
curl -s localhost:16686/api/services | jq -r '.data[]'
```

The most recent trace for a service:

```bash
curl -s "localhost:16686/api/traces?service=sample-llm-app&limit=1&lookback=1h" | jq -r '.data[0] | "trace \(.traceID)  spans=\(.spans|length)"'
```

Its spans as a table — name, duration, output tokens, and the RAM the proxy added:

```bash
curl -s "localhost:16686/api/traces?service=sample-llm-app&limit=1&lookback=1h" | jq -r '.data[0].spans[] | [.operationName, (.duration/1000|floor|tostring)+"ms", ((.tags[]|select(.key=="gen_ai.usage.output_tokens").value|tostring)//"-"), ((.tags[]|select(.key=="llmobs.process.memory.rss_bytes").value/1000000|floor|tostring)+"MB")] | @tsv'
```

```
chat llama3.1:8b   133ms   143   66MB
route request      133ms   143   66MB
```

Confirm the proxy is actually enriching — these device attributes come only
from the proxy, never from your app:

```bash
curl -s "localhost:16686/api/traces?service=sample-llm-app&limit=1&lookback=1h" | jq -r '.data[0].spans[0].tags[] | select(.key|startswith("llmobs.process.") or startswith("llmobs.system.")) | "\(.key) = \(.value)"'
```

```
llmobs.process.memory.rss_bytes = 66338816
llmobs.system.memory.percent = 11
llmobs.system.memory.available_bytes = 7312158720
llmobs.process.cpu.percent = 0.5
```

If this returns nothing, your app is bypassing the proxy and exporting straight
to the Collector.

Find only the failed requests:

```bash
curl -s "localhost:16686/api/traces?service=sample-llm-app&lookback=1h&limit=20&tags=%7B%22error%22%3A%22true%22%7D" | jq -r '.data[] | .traceID'
```

---

## 7. Monitor — metrics (Prometheus)

Open the UI:

```bash
open http://localhost:9090
```

Both scrape targets should say `up`:

```bash
curl -s localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job)\t\(.health)\t\(.lastError // "")"'
```

Every metric the proxy produces:

```bash
curl -s localhost:9090/api/v1/label/__name__/values | jq -r '.data[] | select(startswith("llmobs"))'
```

### Query from the terminal

Define this helper once per shell — every query below uses it:

```bash
q() { curl -s --get localhost:9090/api/v1/query --data-urlencode "query=$1" | jq -r '.data.result[] | "\(.value[1])\t\(.metric | del(.__name__) | to_entries | map("\(.key)=\(.value)") | join(" "))"'; }
```

**Tokens in and out, per model:**

```bash
q 'sum by (gen_ai_token_type, gen_ai_request_model) (llmobs_tokens_total)'
```

**Token throughput (tokens/sec) per service:**

```bash
q 'sum by (service_name, gen_ai_token_type) (rate(llmobs_tokens_total[5m]))'
```

**p95 response time per model:**

```bash
q 'histogram_quantile(0.95, sum by (le, gen_ai_request_model) (rate(llmobs_request_duration_seconds_bucket[5m])))'
```

**p95 time to first token:**

```bash
q 'histogram_quantile(0.95, sum by (le, gen_ai_request_model) (rate(llmobs_request_time_to_first_token_seconds_bucket[5m])))'
```

**Requests per second, by outcome:**

```bash
q 'sum by (service_name, outcome) (rate(llmobs_requests_total[5m]))'
```

**Error ratio per node:**

```bash
q 'llmobs:error_ratio:5m'
```

**Host RAM utilization (0–1) per node — who is about to OOM:**

```bash
q 'topk(5, llmobs_system_memory_utilization_ratio)'
```

**Process memory in MB per node:**

```bash
q 'llmobs_process_memory_rss_bytes / 1000000'
```

**Average tokens per request:**

```bash
q 'sum by (gen_ai_request_model) (rate(llmobs_tokens_per_request_sum[5m])) / sum by (gen_ai_request_model) (rate(llmobs_tokens_per_request_count[5m]))'
```

---

## 8. Debugging

Follow the proxy's logs:

```bash
docker compose logs -f llmobs
```

Errors only, across every container:

```bash
docker compose logs --tail 200 | grep -iE "error|warn|refused|failed"
```

Collector health:

```bash
curl -s localhost:13133/ | jq
```

Raw Prometheus-format metrics the Collector is exposing:

```bash
curl -s localhost:8889/metrics | grep llmobs | head -20
```

Print every span to the proxy's stdout — useful when nothing is arriving:

```bash
docker compose run --rm -e LLMOBS_CONSOLE_EXPORT=true -p 8100:8100 llmobs
```

Run the proxy locally without Docker, against a Collector you already have:

```bash
./.venv/bin/llmobs-server --port 8100 --upstream http://localhost:4318 --log-level debug
```

### If nothing shows up

| Symptom | Check |
|---|---|
| `batches_failed` climbing in `/v1/stats` | The Collector is unreachable — `docker compose logs otel-collector` |
| Traces in Jaeger but no metrics | Your spans need `gen_ai.request.model` or `gen_ai.system` to be counted |
| Metrics exist but every `rate()` is 0 | Traffic stopped. Send sustained load, see §4 |
| Nothing at all | Your app must export to `:8100`, not `:4318` — `curl -s localhost:8100/v1/stats` |
| Spans present but no `llmobs.*` attributes | The app bypassed the proxy and went straight to the Collector |

---

## 9. Stop and clean up

Stop the containers, keep the metric history:

```bash
docker compose down
```

Stop and wipe Prometheus data too — a clean slate:

```bash
docker compose down -v
```

Restart one service after editing its config (`deploy/*.yaml` are bind mounts, so a restart reloads them):

```bash
docker compose restart otel-collector prometheus
```

Rebuild just the proxy after changing Python code:

```bash
docker compose up -d --build llmobs
```

---

## 10. Configuration

Set these on the `llmobs` service in `docker-compose.yml`, or export them when
running it locally.

```bash
LLMOBS_UPSTREAM_ENDPOINT=http://otel-collector:4318   # the real Collector
LLMOBS_PORT=8100                                      # listen port
LLMOBS_NODE_ID=worker-a                               # node identity
LLMOBS_NODE_ROLE=worker                               # gateway | router | worker
LLMOBS_MEMORY_ON_SPAN=true                            # stamp RAM on LLM spans
LLMOBS_ENRICH_ALL_SPANS=false                         # stamp every span, not just LLM
LLMOBS_API_KEY=""                                     # require X-API-Key if set
LLMOBS_EXPORT_INTERVAL_MS=15000                       # derived-metric export cadence
LLMOBS_FORWARD_TIMEOUT_S=10                           # upstream timeout before 503
LLMOBS_INSTANCE_ID=""                                 # pin to keep series stable
LLMOBS_CONSOLE_EXPORT=false                           # also print to stdout
```

With an API key set, callers must send it:

```bash
curl -s -H 'X-API-Key: your-secret' localhost:8100/v1/stats | jq '.totals'
```

> **Deploy one proxy per device.** It stamps spans with the RAM of the machine
> *it* runs on. Run it as a sidecar next to each inference process. A single
> central proxy reports its own memory for the whole fleet, which tells you
> nothing. Token, latency and trace enrichment work fine centrally — RAM does not.
