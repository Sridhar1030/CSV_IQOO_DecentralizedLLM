"""Golden activations for one layer range, from the torch reference in dllm/model.py, on a real prompt.

Writes, for a shard holding layers [a, b):
  prefill:  the residual stream entering layer a for the first P prompt tokens, and leaving layer b-1
  decode t: the same for every later token, one at a time, through greedy generation of the full model

Every array is float32 .npy and also flat little-endian .raw (what LiteRT's run_model --input_dir reads),
so the same numbers verify the tflite on the Mac and on the phone.

  .venv/bin/python npu/golden.py --layers 8-16 --prefill 32 --new 4 --out npu/golden
"""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dllm.model import Cfg, Layer, Head, load_npz, rms_norm, rope
from transformers import AutoTokenizer


def step(layers, x, pos, caches):
    """One forward of x (1, n, hidden) at absolute positions pos through `layers`, appending to caches."""
    for i, L in enumerate(layers):
        pk, pv = caches[i]
        x, k, v = L(x, pos, pk, pv)
        caches[i] = (k, v)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="dist")
    ap.add_argument("--layers", default="8-16", help="a-b half-open")
    ap.add_argument("--prompt", default="Name one car.")
    ap.add_argument("--prefill", type=int, default=32, help="prompt tokens fed as one prefill; the rest decode one by one")
    ap.add_argument("--new", type=int, default=4, help="tokens to generate greedily after the prompt")
    ap.add_argument("--out", default="npu/golden")
    args = ap.parse_args()
    a, b = (int(v) for v in args.layers.split("-"))
    cfg = Cfg.load(f"{args.dist}/config.json")
    tok = AutoTokenizer.from_pretrained("hub_shards")
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True,
                                             tokenize=False), add_special_tokens=False)
    P = min(args.prefill, len(ids))
    print(f"prompt {args.prompt!r}: {len(ids)} tokens, prefill {P}, then {len(ids) - P} prompt tokens + {args.new} new, one at a time")
    head = Head(cfg, "hub_shards")
    layers = [Layer(cfg, load_npz(f"{args.dist}/layer_{i:02d}.npz")) for i in range(cfg.n_layers)]
    caches = [(None, None)] * cfg.n_layers
    os.makedirs(args.out, exist_ok=True)
    manifest = {"layers": [a, b], "prompt": args.prompt, "ids": ids, "prefill": P, "steps": []}

    def save(name, arr):
        arr = arr.detach().cpu().numpy().astype(np.float32)
        np.save(f"{args.out}/{name}.npy", arr)
        arr.tofile(f"{args.out}/{name}.raw")
        return list(arr.shape)

    def run(x, pos, tag):
        """Full model forward for x at pos; records the shard's input and output along the way."""
        with torch.no_grad():
            x = step(layers[:a], x, pos, caches[:a] and caches) if False else x  # placeholder, replaced below
        return x

    with torch.no_grad():
        # prefill P tokens
        x = head.embed_tokens(ids[:P]); pos = torch.arange(P)
        for i in range(a):
            x, k, v = layers[i](x, pos, *caches[i]); caches[i] = (k, v)
        s_in = save("prefill_in", x)
        for i in range(a, b):
            x, k, v = layers[i](x, pos, *caches[i]); caches[i] = (k, v)
        s_out = save("prefill_out", x)
        for i in range(b, cfg.n_layers):
            x, k, v = layers[i](x, pos, *caches[i]); caches[i] = (k, v)
        manifest["steps"].append({"tag": "prefill", "pos": 0, "n": P, "in": s_in, "out": s_out})
        # the rest of the prompt, then greedy generation, one token per step
        seq = list(ids)
        nxt = None
        t = P
        gen = []
        while t < len(ids) + args.new:
            tid = ids[t] if t < len(ids) else nxt
            if t >= len(ids):
                seq.append(tid); gen.append(tid)
            x = head.embed_tokens([tid]); pos = torch.tensor([t])
            for i in range(a):
                x, k, v = layers[i](x, pos, *caches[i]); caches[i] = (k, v)
            s_in = save(f"dec{t}_in", x)
            for i in range(a, b):
                x, k, v = layers[i](x, pos, *caches[i]); caches[i] = (k, v)
            s_out = save(f"dec{t}_out", x)
            for i in range(b, cfg.n_layers):
                x, k, v = layers[i](x, pos, *caches[i]); caches[i] = (k, v)
            lg = head.logits(x)
            nxt = int(lg.argmax())
            manifest["steps"].append({"tag": f"dec{t}", "pos": t, "n": 1, "in": s_in, "out": s_out, "token": tid, "argmax": nxt})
            t += 1
    manifest["generated"] = gen
    manifest["text"] = tok.decode(gen)
    json.dump(manifest, open(f"{args.out}/manifest.json", "w"), indent=1)
    print(f"generated: {gen} -> {manifest['text']!r}")
    print(f"wrote {len(manifest['steps'])} steps to {args.out}/")


if __name__ == "__main__":
    main()
