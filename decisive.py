"""The decisive check: this laptop alone, versus this laptop plus the phone.

    .venv/bin/python decisive.py
    .venv/bin/python decisive.py "Name one car."
    .venv/bin/python decisive.py --local "Name one car."   # laptop half only, cluster not needed

This laptop holds only its own slice of the layers, so running just what it has gives noise.
The same prompt through the cluster, where the phones hold the rest, gives language.
"""
import json, os, sys, urllib.request
import torch
from transformers import AutoTokenizer
from dllm.model import Cfg, Shard, Head

args   = [a for a in sys.argv[1:] if a != "--local"]
LOCAL  = "--local" in sys.argv          # run only this laptop's layers, skip the cluster
PROMPT = args[0] if args else "Name one car."
MAX_TOKENS = 12

MAC_DIR = "mac_shards"
A, B = (lambda f: (int(f[0][6:8]), int(f[-1][6:8]) + 1))(
    sorted(x for x in os.listdir(MAC_DIR) if x.startswith("layer_")))   # whatever this laptop holds

cfg  = Cfg.load("hub_shards/config.json")
tok  = AutoTokenizer.from_pretrained("hub_shards")
head = Head(cfg, "hub_shards")              # embedding table + output head
mac  = Shard(cfg, MAC_DIR, A, B)            # ONLY the layers this laptop holds

text = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                               add_generation_prompt=True, tokenize=False)
ids = tok(text, add_special_tokens=False).input_ids

out, pos = [], 0
for step in range(MAX_TOKENS):
    t = ids if step == 0 else [out[-1]]
    h = mac(head.embed_tokens(t), torch.arange(pos, pos + len(t)), "solo")
    pos += len(t)
    out.append(int(head.logits(h).argmax()))

print(f"LAPTOP ALONE  (layers {A}-{B-1}, then straight to the output head)")
print("   ", repr(tok.decode(out)))

if LOCAL:
    sys.exit(0)

body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                   "max_tokens": MAX_TOKENS}).encode()
req  = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", body,
                              {"content-type": "application/json"})
try:
    answer = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
except Exception as e:
    print("LAPTOP + PHONE  unavailable:", e)
    print("    the phone is not connected, so the other 12 layers are missing entirely")
    sys.exit(1)
print(f"LAPTOP + PHONES  (layers {A}-{B-1} here, layers {B}-{cfg.n_layers-1} on the phones)")
print("   ", repr(answer))
