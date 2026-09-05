# DecentralizedLLM

One language model, cut by layer across a Mac and two phones. No device holds the whole model.
A prompt typed on the Mac runs through layers 0-7 there, layers 8-15 on one phone, layers 16-23 on
the other, and the answer comes back over the same USB cables.

Every claim here is checkable. Run `dllm/prove.py`, or read [PHONE-SETUP.md](PHONE-SETUP.md) and do
it by hand.

## Layout

| path | what |
|---|---|
| `dllm/model.py` | One transformer block, a shard of contiguous blocks with its own KV cache, the embedding + output head. Torch. |
| `dllm/np_node.py` | The phone node. Same math in numpy, one file, no torch. Runs in Termux. |
| `dllm/node.py` | The Mac node. Torch on Metal. Same wire protocol as the phone. |
| `dllm/hub.py` | Coordinator: lobby, routing, sampling, OpenAI-compatible endpoint, shard server, telemetry. |
| `dllm/slicer.py` | Build step: checkpoint → one `.npz` per layer + a manifest of content hashes. |
| `dllm/observe.py` | The hub's OpenTelemetry: one trace per request, one span per hop, per-node metrics. |
| `dllm/wire.py` | Framing: 4-byte header length, JSON header, raw bf16 or fp32 activations. |
| `dllm/prove.py` | Adversarial checks that the split is real. Severs a node and expects failure. |
| `dllm/test_split.py`, `decisive.py` | The laptop alone versus the laptop plus phones, same prompt. |
| `observability/` | OTLP proxy + Collector + Jaeger + Prometheus. See its README. |
| `BUILD-PLAN.md` | The four-device plan, cut line, and what was deliberately not built. |

## Run it

Build shards once on any machine that can hold the checkpoint, then get it off the cluster:

```bash
python -m dllm.slicer Qwen/Qwen2.5-0.5B-Instruct dist
```

Coordinator on the Mac. It holds the embedding table and output head, never a layer:

```bash
.venv/bin/python -m dllm.hub --shards hub_shards --dist dist --expected 3
```

Mac node, then each phone in Termux over `adb reverse tcp:8000 tcp:8000`:

```bash
.venv/bin/python -m dllm.node --code <LOBBY> --name mac1 --layers 0-8 --shards mac_shards --device mps
```

```bash
curl 127.0.0.1:8000/s/phoneA/8-16|sh
```

Delete `dist/` once every node reports ready. Then:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name one car."}],"max_tokens":16}'
```

`temperature`, `top_p`, `top_k`, `seed` and `stream` behave as an OpenAI client expects.

## Verify it

```bash
.venv/bin/python decisive.py "Name one planet."    # laptop alone = noise, laptop + phones = language
.venv/bin/python -m dllm.prove                     # 12 checks, including severing a node
.venv/bin/python -m dllm.test_split                # assertion form of the above
.venv/bin/python -m pytest dllm -q                 # unit + contract tests
```

## Watch it

The coordinator exports OpenTelemetry to the proxy in `observability/`, which stamps the request
with the Mac's memory, derives token, latency, time-to-first-token and error metrics, and forwards
to a Collector feeding Jaeger and Prometheus.

```bash
cd observability && docker compose up -d --build
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:8100 .venv/bin/python -m dllm.hub ...
```

Then open a request in Jaeger at http://localhost:16686. One trace per chat. The root span is the
model call with `gen_ai.*` attributes. Under it, one span per hop, named `layers 8-15 on phoneA`,
carrying that hop's compute and wire time and, crucially, **that phone's own RAM, CPU and battery**
as the phone reported them over its heartbeat.

That last part is why this integration is not the proxy's default topology. The proxy stamps spans
with the memory of the device it runs on, and it cannot run on a phone. So the hub pre-fills each
hop span with the originating device's readings under the proxy's own attribute names. The proxy
never overwrites an attribute the application set, and never touches spans without `gen_ai.*`
markers, which hop spans deliberately lack. Result: the root span carries the coordinator's RAM
from the proxy, every hop carries its own device's RAM from the device, and nothing is attributed to
the wrong machine. `dllm/test_observe.py` runs our spans through the real enricher to hold it there.

Per-node metrics are the hub's own, namespaced `dllm_*`, and pass through the proxy untouched:

| Prometheus name | meaning |
|---|---|
| `dllm_hop_compute_seconds` | histogram, by `dllm_node_name` and `dllm_hop_phase` (prefill or decode) |
| `dllm_hop_wire_seconds` | round trip minus compute |
| `dllm_node_up` | 1 while ready and heartbeating |
| `dllm_node_layers` | layers the node holds |
| `dllm_node_memory_rss_bytes`, `dllm_node_memory_system_utilization_ratio` | as reported by the device itself |
| `dllm_node_battery_ratio` | phones only |
| `dllm_pipeline_complete` | 1 when live nodes tile every layer exactly once |

Recording rules and alerts for the pipeline, including *which device is the bottleneck*, are the
`dllm-pipeline` group in `observability/deploy/rules.yml`. Prometheus is at http://localhost:9090.

Without `OTEL_EXPORTER_OTLP_ENDPOINT` set, telemetry is a no-op and the hub runs exactly as before.

## What this is not

It is slower than one device. Distributed decode always is; the argument is capacity and fit.
Activations cross the wire in bfloat16 except the final hop into the output head, which stays
fp32 because near-tied tokens flip under bfloat16 rounding. Weights are never quantized on the wire.
