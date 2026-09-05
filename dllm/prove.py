"""Adversarial check: is the model really split, or is one device quietly running all of it?

  .venv/bin/python -m dllm.prove --hub http://127.0.0.1:8000

Four tests. The last one is the only one that cannot be faked by a lying node.
"""
import argparse, json, subprocess, sys, time, urllib.request

OK, BAD = "PASS", "FAIL"
results = []


def get(hub, path):
    with urllib.request.urlopen(f"{hub}{path}", timeout=30) as r:
        return json.load(r)


def ask(hub, prompt, max_tokens=8, timeout=180):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{hub}/v1/chat/completions", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def check(name, passed, detail):
    results.append((name, passed, detail))
    print(f"  [{OK if passed else BAD}] {name}\n        {detail}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default="http://127.0.0.1:8000")
    # adb reverse --remove only blocks NEW connections, so it leaves an established socket alive.
    # Killing the adb server tears down the USB transport the existing socket rides on.
    ap.add_argument("--kill", default="adb kill-server",
                    help="command that severs a node's link to the hub")
    ap.add_argument("--restore", default="adb start-server && adb reverse tcp:8000 tcp:8000")
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()

    inv = get(a.hub, "/inventory")
    n_layers = inv["n_layers"]

    print("\n1. Does every layer live on exactly one node?")
    check("layers tile the model with no gaps and no duplicates",
          inv["disjoint_and_complete"],
          f"{n_layers} layers, duplicated={inv['duplicated_layers']}, unclaimed={inv['unclaimed_layers']}, "
          f"problems={inv['problems'] or 'none'}")

    print("\n2. Does each node's claim match the bytes it actually holds?")
    for name, n in inv["nodes"].items():
        check(f"{name} fingerprints match the real shard files",
              n["fingerprints_wrong"] == [] and n["fingerprints_verified"] == n["n_layers"],
              f"claims layers {n['layers'][0]}-{n['layers'][1]-1}, "
              f"{n['fingerprints_verified']}/{n['n_layers']} sha256 verified against the checkpoint")
        check(f"{name} has no shard files beyond the layers it owns",
              n["layer_files_present"] <= n["n_layers"],
              f"{n['layer_files_present']} layer files on disk in {n['shard_dir']}, owns {n['n_layers']}")
        check(f"{name} does not hold the embedding table or lm_head",
              not n["holds_embedding_or_head"],
              "no head file present" if not n["holds_embedding_or_head"] else "HAS a head file")

    print("\n3. Is each node's memory too small to hold the whole model?")
    full_mb = None
    try:
        man = get(a.hub, "/shards/manifest.json")
        full_mb = sum(man["files"].values()) / 2**20
    except Exception:
        pass
    for name, n in inv["nodes"].items():
        rss = n["rss_mb"]
        if rss is None:
            continue
        check(f"{name} resident memory is a fraction of the full model",
              full_mb is None or rss < full_mb * 0.9,
              f"{rss:.0f} MB resident" + (f", full checkpoint is {full_mb:.0f} MB on disk" if full_mb else ""))

    if a.skip_ablation:
        print("\n4. skipped")
    else:
        print("\n4. Remove one node. If the rest could answer alone, the split was theatre.")
        before = ask(a.hub, "Say hello", 8)
        check("baseline generation with every node present", bool(before.strip()), f"answered {before!r}")

        subprocess.run(a.kill, shell=True, check=False)
        for _ in range(20):
            time.sleep(1)
            if not get(a.hub, "/status")["pipeline_ok"]:
                break
        st = get(a.hub, "/status")["nodes"]
        gone = [k for k in inv["nodes"] if k not in st or not st[k]["live"]]
        try:
            out = ask(a.hub, "Say hello", 8, timeout=45)
            check("generation FAILS once a node is gone", False, f"it still answered {out!r} without {gone}")
        except Exception as e:
            check("generation fails once a node is gone", True,
                  f"node(s) {gone} left, request refused: {str(e)[:70]}")

        subprocess.run(a.restore, shell=True, check=False)
        rejoined = False
        for _ in range(60):
            time.sleep(2)
            if get(a.hub, "/status")["pipeline_ok"]:
                rejoined = True
                break
        if rejoined:
            after = ask(a.hub, "Say hello", 8)
            check("the node rejoins and the same answer comes back", after == before,
                  f"before {before!r}, after {after!r}")
        else:
            check("the node rejoins", False, "pipeline did not come back within 120s")

    n_pass = sum(1 for _, p, _ in results if p)
    print(f"\n{n_pass}/{len(results)} checks passed\n")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
