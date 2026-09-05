"""Attribute and instrument names.

Attributes the proxy *reads* follow the OpenTelemetry GenAI semantic
conventions, so it understands spans from any OTel-instrumented LLM library
without configuration. The upstream Python constants live under
`opentelemetry.semconv._incubating`, a private module whose path moves between
releases, so the stable strings are pinned here.

Instruments the proxy *writes* are namespaced `llmobs.*` on purpose. Reusing
the `gen_ai.client.*` instrument names would silently double-count for any app
that already emits them itself.

Spec: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

# --- GenAI attributes the proxy reads off incoming spans ----------------
GEN_AI_SYSTEM = "gen_ai.system"                    # ollama, vllm, openai, llama.cpp...
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"    # chat, text_completion, embeddings
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"            # "input" | "output"

TOKEN_TYPE_INPUT = "input"
TOKEN_TYPE_OUTPUT = "output"

# --- resource / status --------------------------------------------------
SERVICE_NAME = "service.name"
SERVICE_VERSION = "service.version"
SERVICE_INSTANCE_ID = "service.instance.id"
DEPLOYMENT_ENVIRONMENT = "deployment.environment.name"
HOST_NAME = "host.name"
HOST_ARCH = "host.arch"
OS_TYPE = "os.type"
PROCESS_PID = "process.pid"
ERROR_TYPE = "error.type"

# --- node identity in the mesh -----------------------------------------
NODE_ID = "llmobs.node.id"
NODE_ROLE = "llmobs.node.role"
# A relay emits telemetry on behalf of *other* nodes, so it must not claim
# `llmobs.node.id` for itself - see Config.relay_mode.
COLLECTOR_ID = "llmobs.collector.id"

# --- device resources the proxy stamps onto spans -----------------------
MEM_PROCESS_RSS = "llmobs.process.memory.rss_bytes"
MEM_PROCESS_VMS = "llmobs.process.memory.vms_bytes"
MEM_PROCESS_PERCENT = "llmobs.process.memory.percent"
MEM_SYSTEM_TOTAL = "llmobs.system.memory.total_bytes"
MEM_SYSTEM_USED = "llmobs.system.memory.used_bytes"
MEM_SYSTEM_AVAILABLE = "llmobs.system.memory.available_bytes"
MEM_SYSTEM_PERCENT = "llmobs.system.memory.percent"
CPU_PROCESS_PERCENT = "llmobs.process.cpu.percent"

# Time-to-first-token, if the app put it on the span.
TTFT_MS = "llmobs.response.time_to_first_token_ms"

# --- instruments the proxy derives from spans ---------------------------
M_TOKENS_TOTAL = "llmobs.tokens.total"                    # counter   {token}
M_TOKENS_PER_REQUEST = "llmobs.tokens.per_request"        # histogram {token}
M_REQUEST_DURATION = "llmobs.request.duration"            # histogram s
M_TTFT = "llmobs.request.time_to_first_token"             # histogram s
M_REQUESTS_TOTAL = "llmobs.requests.total"                # counter   {request}

# --- device gauges, sampled on each export tick -------------------------
M_PROC_MEM_RSS = "llmobs.process.memory.rss"
M_PROC_MEM_PERCENT = "llmobs.process.memory.utilization"
M_SYS_MEM_USED = "llmobs.system.memory.used"
M_SYS_MEM_AVAILABLE = "llmobs.system.memory.available"
M_SYS_MEM_TOTAL = "llmobs.system.memory.total"
M_SYS_MEM_PERCENT = "llmobs.system.memory.utilization"
M_PROC_CPU_PERCENT = "llmobs.process.cpu.utilization"

# Duration buckets tuned for LLM inference (seconds). The OTel GenAI spec
# recommends this powers-of-two ladder; the SDK default tops out at 10s, which
# lumps every slow generation into one overflow bucket.
DURATION_BUCKETS_S = [
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28,
    2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
]

# Token-count buckets, per the GenAI spec.
TOKEN_BUCKETS = [
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
]
