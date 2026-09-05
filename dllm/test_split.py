"""Prove the model is genuinely split across two devices, not quietly running on one.

    .venv/bin/python -m dllm.test_split
    .venv/bin/python -m dllm.test_split --prompt "Explain gravity." --max-tokens 24
    .venv/bin/python -m dllm.test_split --temperature 0.8 --top-p 0.9 --seed 7

Run it from the project root with the hub and both nodes up. Exits non-zero if anything fails.

The argument is simple. This laptop physically holds layers 0-11 and nothing else, so running only
what it has produces noise. The same prompt through the cluster, where a phone holds layers 12-23,
produces language. Neither device can do it alone.

Temperature defaults to 0, which is greedy. That is deliberate: a sampled run differs every time and
would prove nothing. Pass --temperature to sample, and --seed to make the cluster side repeatable.
"""
import argparse, json, os, sys, urllib.request
import torch
from transformers import AutoTokenizer
from dllm.model import Cfg, Shard, Head, sample

HUB_DIR, MAC_DIR = "hub_shards", "mac_shards"
MINE = (lambda f: (int(f[0][6:8]), int(f[-1][6:8]) + 1))(
    sorted(x for x in os.listdir(MAC_DIR) if x.startswith("layer_")))   # derived, not assumed
failures = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    if not ok:
        failures.append(name)
    return ok


def post(hub, body, timeout=300):
    req = urllib.request.Request(f"{hub}/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]["content"]


def local_only(a, prompt, n, temperature, top_p, top_k, seed):
    """Decode using only the layers on this machine, straight into the output head."""
    cfg = Cfg.load(f"{HUB_DIR}/config.json")
    tok = AutoTokenizer.from_pretrained(HUB_DIR)
    head, shard = Head(cfg, HUB_DIR), Shard(cfg, MAC_DIR, *a)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    out, pos = [], 0
    for step in range(n):
        t = ids if step == 0 else [out[-1]]
        h = shard(head.embed_tokens(t), torch.arange(pos, pos + len(t)), "solo")
        pos += len(t)
        out.append(sample(head.logits(h), temperature, top_p, top_k,
                          None if seed is None else seed + pos))
    return tok.decode(out)


# -------------------------------------------------------------------------------------------------
# concurrency: the cluster serves several requests at once without mixing them up
# -------------------------------------------------------------------------------------------------
def test_concurrent_requests_do_not_share_state(hub="http://127.0.0.1:8000"):
    """Each node keys its KV cache by request id, so two requests in flight must not touch each
    other. Greedy decoding makes any leak deterministic and visible: the text would drift."""
    from concurrent.futures import ThreadPoolExecutor
    prompts = ["Name one car.", "Name one fruit.", "What is the capital of Japan?"]
    ask = lambda p: post(hub, {"messages": [{"role": "user", "content": p}], "max_tokens": 12})
    alone = {p: ask(p) for p in prompts}
    with ThreadPoolExecutor(len(prompts)) as ex:
        together = dict(zip(prompts, ex.map(ask, prompts)))
    for p in prompts:
        assert together[p] == alone[p], f"{p!r}: {together[p]!r} concurrently vs {alone[p]!r} alone"
    return alone


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", default="Name one fruit.")
    p.add_argument("--max-tokens", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.0, help="0 is greedy. Above 0 samples.")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=None, help="makes sampled runs repeatable")
    p.add_argument("--hub", default="http://127.0.0.1:8000")
    a = p.parse_args()
    gen = dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=a.seed)

    print("\n1. What does this laptop actually hold?")
    mine = sorted(f for f in os.listdir(MAC_DIR) if f.startswith("layer_"))
    theirs = [f"layer_{i:02d}.npz" for i in range(*MINE)]
    check("its layer files form one contiguous run with no gaps",
          mine == theirs, f"{MAC_DIR}: {mine[0]} .. {mine[-1]}, {len(mine)} files")
    check("the coordinator holds no layers at all",
          not [f for f in os.listdir(HUB_DIR) if f.startswith("layer_")],
          f"{HUB_DIR}: {sorted(os.listdir(HUB_DIR))}")
    # Look at the disk directly. Calling from_pretrained here would download the very checkpoint
    # this check is asserting is absent, which is how an earlier version of this test fooled itself.
    repo = json.load(open(f"{HUB_DIR}/manifest.json"))["repo"]
    cache = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")) + "/hub/models--" + repo.replace("/", "--")
    found = []
    for root, _, files in os.walk(cache):
        found += [f"{root}/{f}" for f in files if f.endswith(".safetensors")]
    check("the undivided checkpoint is not on this machine", not found,
          f"no weights under {cache}" if not found else f"STILL CACHED: {found}")

    print("\n2. Is the cluster complete?")
    st = json.load(urllib.request.urlopen(f"{a.hub}/status", timeout=30))
    layout = {k: v["layers"] for k, v in st["nodes"].items()}
    if not check("every layer is claimed by exactly one live node", st["pipeline_ok"], f"{layout}"):
        print("\n  cluster is not up, nothing further to compare.\n")
        sys.exit(1)

    print(f"\n3. Decode {a.prompt!r} two ways")
    solo = local_only(MINE, a.prompt, a.max_tokens, a.temperature, a.top_p, a.top_k, a.seed)
    both = post(a.hub, {"messages": [{"role": "user", "content": a.prompt}],
                        "max_tokens": a.max_tokens, **{k: v for k, v in gen.items() if v is not None}})
    print(f"    laptop alone, layers {MINE[0]}-{MINE[1]-1}:\n       {solo!r}")
    print(f"    laptop + phone, all {st['n_layers']} layers:\n       {both!r}")
    check("the two disagree, so the missing layers are doing real work",
          solo.strip() != both.strip(), "identical output would mean the split changes nothing")
    printable = sum(c.isascii() and (c.isalnum() or c in " .,'!?-\n") for c in both) / max(len(both), 1)
    check("the full pipeline produces coherent text",
          printable > 0.9, f"{printable:.0%} of the answer is ordinary text")

    print("\n5. Several requests at once must not contaminate each other")
    try:
        answers = test_concurrent_requests_do_not_share_state(a.hub)
        check("concurrent requests give the same answers as one at a time", True,
              f"{len(answers)} prompts, identical served alone and together")
    except AssertionError as e:
        check("concurrent requests give the same answers as one at a time", False, str(e)[:120])

    if a.temperature == 0:
        again = post(a.hub, {"messages": [{"role": "user", "content": a.prompt}],
                             "max_tokens": a.max_tokens})
        check("greedy decoding is reproducible across runs", again == both, f"{again!r}")

    print(f"\n{'all checks passed' if not failures else 'FAILED: ' + ', '.join(failures)}\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
