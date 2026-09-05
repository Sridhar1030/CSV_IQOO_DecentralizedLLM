#!/bin/bash
# Export every layer of the sliced model as a single-layer NPU .tflite the phone app fetches and runs
# on the Hexagon HTP. One file per layer keeps the "download only your layers" contract: a node
# assigned layers 8-15 fetches layer_08.tflite .. layer_15.tflite and chains them.
# Decode signature only (n=1); the app drives prefill as a loop of decode steps.
set -e
V="$(cd "$(dirname "$0")/.." && pwd)/.venv-litert/bin/python"
CACHE=${1:-512}
N=$(python3 -c "import json;print(json.load(open('dist/config.json'))['num_hidden_layers'])")
for i in $(seq 0 $((N-1))); do
  a=$i; b=$((i+1))
  out="dist/layer_$(printf %02d $i).tflite"
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  $V npu/export_shard.py --layers $a-$b --cache-len $CACHE --sigs decode \
     --out "$out" 2>&1 | grep -E "wrote|MISMATCH|Error" || true
done
echo "done: $(ls dist/layer_*.tflite | wc -l) layer tflite files"
