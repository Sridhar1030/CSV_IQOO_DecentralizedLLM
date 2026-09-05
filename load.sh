#!/usr/bin/env bash
# Five long prompts against the cluster.
#
#   ./load.sh              all five at once, which is what makes the hub batch
#   ./load.sh serial       one after another, for comparison
#   ./load.sh both         serial then concurrent, and the speedup between them
#
# Env: HUB (default 127.0.0.1:8000), MAX_TOKENS (default 60)
set -u

HUB="${HUB:-127.0.0.1:8000}"
MAX_TOKENS="${MAX_TOKENS:-60}"
MODE="${1:-concurrent}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

P1="You are explaining a distributed system to a sceptical engineer. A single language model of 24 transformer layers has been split across three devices on a local network: layers 0 to 7 on a laptop, 8 to 15 on one phone, 16 to 23 on a second phone. The laptop also holds the embedding table and the output head. For every token, a vector of 896 numbers travels laptop to phone to phone and back. Explain in plain prose why no single one of those devices can produce text on its own, and what would happen if the second phone were switched off in the middle of a sentence."
P2="Explain, for someone who knows programming but not machine learning, why generating one word at a time from a language model is limited by memory bandwidth rather than by processing power. Use the fact that producing a single token requires reading every weight in the model but performing only about half an arithmetic operation for each byte read. Then explain why serving several people at once fixes this, and why it does not make any individual answer arrive sooner."
P3="Write a short technical brief comparing three ways to run a neural network on an Android phone: a plain interpreted loop in a managed language, a hand-written kernel using the processor's vector instructions, and offloading to the phone's dedicated neural accelerator. For each, describe the engineering effort required, what could go wrong, and the kind of speed difference to expect. Be concrete and avoid marketing language."
P4="A team is preparing a live demonstration in which a language model runs across a laptop and two phones over ordinary WiFi. List the things most likely to go wrong during that demonstration, in order of how likely they are, and give a specific mitigation for each. Consider the venue network, battery, thermal limits, devices sleeping, and what the audience will actually be able to see."
P5="Describe how you would prove to a doubtful observer that a language model really is split across several machines, rather than running entirely on one and only pretending. Propose several independent checks, order them from weakest to strongest, and explain precisely why the strongest one cannot be faked by a dishonest implementation."

ask() {   # ask <index> <prompt>
  local i="$1" prompt="$2" start end
  start=$(python3 -c 'import time;print(time.time())')
  python3 - "$HUB" "$MAX_TOKENS" "$prompt" > "$OUT/$i" 2>&1 <<'PY'
import json, sys, urllib.request, urllib.error
hub, maxt, prompt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": maxt}).encode()
req = urllib.request.Request(f"http://{hub}/v1/chat/completions", body, {"content-type": "application/json"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=900))
except urllib.error.HTTPError as e:
    print("ERROR", json.load(e).get("detail", e)); sys.exit(1)
u = d["usage"]
print(u["prompt_tokens"], u["completion_tokens"])
print(d["choices"][0]["message"]["content"])
PY
  end=$(python3 -c 'import time;print(time.time())')
  echo "$start $end" > "$OUT/$i.t"
}

run() {   # run <label> <concurrent|serial>
  local label="$1" how="$2" i t0 t1
  t0=$(python3 -c 'import time;print(time.time())')
  for i in 1 2 3 4 5; do
    eval "local p=\$P$i"
    if [ "$how" = concurrent ]; then ask "$i" "$p" & else ask "$i" "$p"; fi
  done
  [ "$how" = concurrent ] && wait
  t1=$(python3 -c 'import time;print(time.time())')
  python3 - "$OUT" "$label" "$t0" "$t1" <<'PY'
import os, sys
out, label, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
total_out = 0
print(f"\n=== {label} ===")
for i in "12345":
    body = open(f"{out}/{i}").read().split("\n", 1)
    if body[0].startswith("ERROR"):
        print(f"  {i}. FAILED: {body[0][:110]}"); continue
    pin, pout = map(int, body[0].split()); total_out += pout
    s, e = open(f"{out}/{i}.t").read().split()
    text = body[1].strip().replace("\n", " ")
    print(f"  {i}. prompt {pin:3} tokens, generated {pout:3} in {float(e)-float(s):5.1f}s")
    print(f"     {text[:150]}{'…' if len(text) > 150 else ''}")
wall = t1 - t0
print(f"  ---- {total_out} tokens in {wall:.1f}s  =  {total_out/wall:.2f} tokens/s across the cluster")
open(f"{out}/{label}.rate", "w").write(f"{total_out/wall} {wall}")
PY
}

# refuse to run against an incomplete cluster: the error would be confusing
status=$(curl -s -m 5 "http://$HUB/status")
if [ -z "$status" ]; then echo "cannot reach the hub at $HUB"; exit 1; fi
python3 - "$status" <<'PY' || exit 1
import json, sys
d = json.loads(sys.argv[1])
who = ", ".join(f"{k} {v['layers'][0]}-{v['layers'][1]-1}" for k, v in sorted(d["nodes"].items(), key=lambda x: x[1]["layers"][0]) if v["live"])
print(f"cluster: {who or 'nothing live'}")
if not d["pipeline_ok"]:
    print(f"pipeline incomplete, no node holds layers {d['missing_layers']}. Join a device first.")
    sys.exit(1)
PY

case "$MODE" in
  serial)     run "one at a time" serial ;;
  concurrent) run "all five at once" concurrent ;;
  both)
    run "one at a time" serial
    run "all five at once" concurrent
    python3 - "$OUT" <<'PY'
import sys
out = sys.argv[1]
a, awall = map(float, open(f"{out}/one at a time.rate").read().split())
b, bwall = map(float, open(f"{out}/all five at once.rate").read().split())
print(f"\n  concurrent is {b/a:.2f}x the throughput of serial ({awall:.1f}s -> {bwall:.1f}s)")
print("  the gain is requests overlapping across devices, plus the hub merging decode steps")
print("  waiting on the same node into one batched frame.")
PY
    ;;
  *) echo "usage: $0 [concurrent|serial|both]"; exit 2 ;;
esac
