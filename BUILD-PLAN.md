# Build plan: 30h hackathon, 2 Macs + 2 phones

Decisions locked from Q&A on 2026-09-05. Change here first, then in code.

## What we ship

One Qwen2.5-3B-Instruct split by layer across 2 Macs (M1 Pro, 32 GB) and 2 iQOO phones (12 GB+).
Coordinator on Mac 1 exposes an OpenAI-compatible endpoint. Phones join via a lobby code.
Extra device = hot backup. LoRA finetune runs pipeline-parallel across the 2 Macs, launched and
shown from a Kubeflow Pipelines UI.

## Model constants (Qwen2.5-3B-Instruct config.json, verified)

hidden 2048 | 36 layers | 16 heads | 2 KV heads | head_dim 128 | intermediate 11008 | vocab 151,936 |
tied embeddings | rope_theta 1e6 | rms_norm_eps 1e-6

Derived:
- per decoder layer ~77M params (attn 9.4M + mlp 67.6M)
- embed/lm_head 311M params = ~4 layers' worth of compute, 1.2 GB fp32, lives once on Mac 1
- activation on wire: 2048 x bf16 = 4 KB per token per hop
- KV cache: 2 KB per token per layer fp32 (1 KB bf16). 512 tokens x 36 layers = 36 MB. Cache everywhere.
- 9-layer phone shard: ~690M params = 2.8 GB fp32. Fits 12 GB phone. No quant needed.

## Topology

Hub. Every node (Mac or phone) keeps ONE persistent WebSocket to the coordinator. Coordinator forwards
4 KB frames hop to hop. Costs one extra LAN crossing per hop (~1 ms). Chosen because phones (React
Native) cannot cheaply accept inbound sockets and because lobby, heartbeat, failover and RTT
measurement all fall out of the hub for free. Slide must say this contradicts ADR decision 4 and why.

Initial static placement (re-balanced at hour ~20 from measured ms/layer per device):

| device | role | layers |
|---|---|---|
| Mac 1 | coordinator + embed + tail + lm_head + sampling | 28-35 |
| Mac 2 | middle | 16-27 |
| Phone A | head | 0-7 |
| Phone B | middle | 8-15 |

Sampling on Mac 1 where logits are produced. Token id (4 B) goes back into the embed on Mac 1. Zero
logit traffic on the wire.

## Node runtimes

- Mac node: Python, torch (MPS), loads pre-sliced safetensors shard. Also the LoRA trainer.
- Phone node: Expo dev build (not Expo Go) + `onnxruntime-react-native`, loads pre-sliced ONNX shard
  with KV cache as explicit inputs/outputs. float32.
- Same wire protocol for both. A node is "layers i..j, forward(hidden, pos) -> hidden, own KV".

## Wire protocol (JSON header + raw bytes over WS)

```
node -> hub:  {"t":"hello","name","device","ram_gb","bench_ms_per_layer"}
hub -> node:  {"t":"assign","layers":[a,b],"shard_url"}
hub -> node:  {"t":"fwd","req","pos","n","dtype":"bf16"} + 4*n KB payload
node -> hub:  {"t":"fwd_out","req","ms"} + payload
node -> hub:  {"t":"hb","battery","temp","queue"}
hub -> node:  {"t":"reset","req"}   (drop KV for this request)
```

## Endpoint

`POST /v1/chat/completions` on Mac 1, `stream: true` -> SSE, OpenAI shape. Works with curl and any
OpenAI client. `GET /` serves a status page: nodes, layers, ms per hop, token relay animation.

## Lobby

Coordinator prints 6-char code. Phone app: enter code + coordinator IP (or mDNS `_dllm._tcp`),
hub assigns layers, phone downloads its shard over HTTP from Mac 1, reports ready, joins pipeline.

## Backup + placement

5th device (or a Mac holding a replica) preloads a second copy of one shard. Heartbeat 1 s. Miss 3 ->
hub re-routes to replica and resets KV for in-flight requests. Placement score per device =
ms_per_layer measured at join + RTT + battery/thermal penalty. Battery/thermal aware placement is the
novel bit; keep it simple (one penalty term).

## Finetune (Macs only)

LoRA rank 8 on q,v projections. Mac 2 holds layers 0-27 in torch, Mac 1 holds 28-35 + head.
Forward activations and backward gradients cross the wire. ~50 steps on a tiny dataset, show loss
drop. Phones idle during training, say so.

Kubeflow Pipelines standalone on kind on Mac 1 (Docker Desktop present). One pipeline: prepare data
-> launch LoRA -> eval. Judges see the run in the KFP UI. Install starts at hour 0 in background.
Fallback if broken by hour 20: coordinator serves its own training dashboard and slide says
"KFP-ready pipeline spec".

## Timeline, 2 people (A = Python/back, B = React Native)

| hours | A | B |
|---|---|---|
| 0-1 | scaffold repo, download 3B, start KFP install | Expo dev build + onnxruntime-react-native hello |
| 1-6 | slicer (safetensors + ONNX per shard), torch node, hub + pipeline loop | load ONNX shard on phone, one forward, WS client |
| 6-12 | endpoint SSE, 2-Mac end to end | phone joins as node, status screen |
| 12-18 | lobby code, shard download, status page | lobby UI, per-token screen flash |
| 18-22 | replica failover, placement scoring | airplane-mode kill test, battery/thermal reporting |
| 22-27 | LoRA pipeline-parallel, KFP pipeline | polish, rehearse |
| 27-30 | rehearsal, slides, numbers tagged (measured) | same |

## Cut line

Must ship: sliced shards, torch node, phone node, hub, endpoint, lobby. Everything below the line
(failover, placement, LoRA, KFP) becomes slide if inference is not end to end by hour 14.

## Do not

- No int8 on activations. No wire compression. No speed claims. Argument is fit and capacity.
- No `from_pretrained` of the full model inside a node. Shards only.
- No number on a slide without (measured)/(derived)/(modelled) tag.
