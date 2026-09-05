"""Golden reference activations at EVERY layer boundary, from the torch reference in dllm/model.py,
for a full verification sweep of all exported single-layer NPU graphs.

This is npu/golden.py's approach (same prompt, same prefill/decode split, same torch model),
extended to snapshot the hidden state entering every layer (boundary i = input to layer i =
output of layer i-1), not just the boundary of one shard. One full-model forward pass over the
prompt gives everything needed to verify all 24 single-layer .tflite graphs, and any contiguous
chain of them, without re-running the model 24 times.

Writes, per step (prefill and each decode token), boundary_{tag}_{i:02d}.npy for i in 0..n_layers
(hidden state right before layer i would run; i == n_layers is the final output after layer 23),
plus manifest.json describing the steps (mirrors npu/golden.py's manifest shape).

  .venv/bin/python npu/golden_all.py --dist dist --out npu/golden_all
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dllm.model import Cfg, Layer, Head, load_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="dist")
    ap.add_argument("--tokenizer", default="hub_shards")
    ap.add_argument("--prompt", default="Name one car.")
    ap.add_argument("--prefill", type=int, default=32, help="prompt tokens fed as one prefill; the rest decode one by one")
    ap.add_argument("--new", type=int, default=4, help="tokens to generate greedily after the prompt")
    ap.add_argument("--out", default="npu/golden_all")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    cfg = Cfg.load(f"{args.dist}/config.json")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True,
                                             tokenize=False), add_special_tokens=False)
    P = min(args.prefill, len(ids))
    print(f"prompt {args.prompt!r}: {len(ids)} tokens, prefill {P}, then {len(ids) - P} prompt tokens + {args.new} new, one at a time")
    head = Head(cfg, args.tokenizer)
    layers = [Layer(cfg, load_npz(f"{args.dist}/layer_{i:02d}.npz")) for i in range(cfg.n_layers)]
    caches = [(None, None)] * cfg.n_layers
    os.makedirs(args.out, exist_ok=True)
    manifest = {"n_layers": cfg.n_layers, "prompt": args.prompt, "ids": ids, "prefill": P, "steps": []}

    def save(tag, boundary, arr):
        arr = arr.detach().cpu().numpy().astype(np.float32)
        np.save(f"{args.out}/boundary_{tag}_{boundary:02d}.npy", arr)

    def run_all_boundaries(x, pos, tag):
        """Runs x through every layer, saving the hidden state at every boundary 0..n_layers."""
        save(tag, 0, x)
        for i in range(cfg.n_layers):
            x, k, v = layers[i](x, pos, *caches[i])
            caches[i] = (k, v)
            save(tag, i + 1, x)
        return x

    with torch.no_grad():
        # prefill P tokens
        x = head.embed_tokens(ids[:P]); pos = torch.arange(P)
        x = run_all_boundaries(x, pos, "prefill")
        manifest["steps"].append({"tag": "prefill", "pos": 0, "n": P})
        # the rest of the prompt, then greedy generation, one token per step
        nxt = None
        t = P
        gen = []
        while t < len(ids) + args.new:
            tid = ids[t] if t < len(ids) else nxt
            if t >= len(ids):
                gen.append(tid)
            x = head.embed_tokens([tid]); pos = torch.tensor([t])
            x = run_all_boundaries(x, pos, f"dec{t}")
            lg = head.logits(x)
            nxt = int(lg.argmax())
            manifest["steps"].append({"tag": f"dec{t}", "pos": t, "n": 1, "token": tid, "argmax": nxt})
            t += 1
    manifest["generated"] = gen
    manifest["text"] = tok.decode(gen)
    json.dump(manifest, open(f"{args.out}/manifest.json", "w"), indent=1)
    print(f"generated: {gen} -> {manifest['text']!r}")
    print(f"wrote {len(manifest['steps'])} steps x {cfg.n_layers + 1} boundaries to {args.out}/")


if __name__ == "__main__":
    main()
