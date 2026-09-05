#!/usr/bin/env python3
"""Load benchmark for the cluster: checked arithmetic plus the five long prompts, at rising concurrency.

  ./bench.py                                   levels 1,2,4,8 with 16 requests each
  ./bench.py --concurrency 1,2 --requests 4    a quick smoke run
  ./bench.py --json                            machine output only

Talks to /status and /v1/chat/completions, nothing else. Refuses to run on an incomplete pipeline
because the 503s would only look like a slow cluster. stdlib only so it runs from any laptop.
"""
import argparse
import json
import queue
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

# The same five prompts as load.sh, verbatim, so numbers line up between the two scripts.
PROSE = [
    "You are explaining a distributed system to a sceptical engineer. A single language model of 24 transformer layers has been split across three devices on a local network: layers 0 to 7 on a laptop, 8 to 15 on one phone, 16 to 23 on a second phone. The laptop also holds the embedding table and the output head. For every token, a vector of 896 numbers travels laptop to phone to phone and back. Explain in plain prose why no single one of those devices can produce text on its own, and what would happen if the second phone were switched off in the middle of a sentence.",
    "Explain, for someone who knows programming but not machine learning, why generating one word at a time from a language model is limited by memory bandwidth rather than by processing power. Use the fact that producing a single token requires reading every weight in the model but performing only about half an arithmetic operation for each byte read. Then explain why serving several people at once fixes this, and why it does not make any individual answer arrive sooner.",
    "Write a short technical brief comparing three ways to run a neural network on an Android phone: a plain interpreted loop in a managed language, a hand-written kernel using the processor's vector instructions, and offloading to the phone's dedicated neural accelerator. For each, describe the engineering effort required, what could go wrong, and the kind of speed difference to expect. Be concrete and avoid marketing language.",
    "A team is preparing a live demonstration in which a language model runs across a laptop and two phones over ordinary WiFi. List the things most likely to go wrong during that demonstration, in order of how likely they are, and give a specific mitigation for each. Consider the venue network, battery, thermal limits, devices sleeping, and what the audience will actually be able to see.",
    "Describe how you would prove to a doubtful observer that a language model really is split across several machines, rather than running entirely on one and only pretending. Propose several independent checks, order them from weakest to strongest, and explain precisely why the strongest one cannot be faked by a dishonest implementation.",
]
NUM = " Answer with just the number."
LIST = " Answer with just the list."


def bank(rng, n):
    """n tasks in a fixed cycle: 16 requests is exactly 2 each of the five arithmetic kinds, one compare,
    one evens, one sort, three prose. Each task is {kind, prompt, expect}: expect is an int (first or last
    integer in the reply must match), a list (extracted integers must equal it exactly), or None (prose)."""
    def add():
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"What is {a} + {b}?" + NUM, a + b
    def sub():
        a = rng.randint(50, 99); b = rng.randint(10, a)
        return f"What is {a} - {b}?" + NUM, a - b
    def mul():
        a, b = rng.randint(2, 9), rng.randint(11, 99)
        return f"What is {a} times {b}?" + NUM, a * b
    def pct():
        p, m = rng.choice([10, 20, 25, 50]), rng.choice(range(20, 401, 20))
        return f"What is {p}% of {m}?" + NUM, p * m // 100
    def chain():
        a, b = rng.randint(20, 99), rng.randint(10, 50); c = rng.randint(1, b)
        return f"Start with {a}. Add {b}. Then subtract {c}. What is the result?" + NUM, a + b - c
    def compare():
        a, b = rng.sample(range(10, 1000), 2)
        return f"Which number is larger, {a} or {b}?" + NUM, max(a, b)
    def evens():
        a = rng.choice(range(2, 21, 2)); b = a + 8
        return f"List the even numbers from {a} to {b} inclusive, separated by commas." + LIST, list(range(a, b + 1, 2))
    def sort():
        xs = rng.sample(range(1, 100), 5)
        return f"Sort these numbers in ascending order: {', '.join(map(str, xs))}." + LIST, sorted(xs)
    prose_i = [0]
    def prose():
        p = PROSE[prose_i[0] % len(PROSE)]; prose_i[0] += 1
        return p, None
    cycle = [add, add, sub, sub, mul, mul, pct, pct, chain, chain, compare, evens, sort, prose, prose, prose]
    out = []
    for i in range(n):
        f = cycle[i % len(cycle)]
        prompt, expect = f()
        out.append({"kind": f.__name__, "prompt": prompt, "expect": expect})
    return out


def check(task, reply, ntok):
    """Prose only has to produce something; everything else is compared on the integers in the reply.
    Models write either '68' or '45 + 23 = 68', so first or last integer counts; garbage does not."""
    if task["expect"] is None:
        return ntok >= 5
    ints = [int(x) for x in re.findall(r"-?\d+", reply)]
    if isinstance(task["expect"], list):
        return ints == task["expect"]
    return bool(ints) and task["expect"] in (ints[0], ints[-1])


def ask(hub, prompt, max_tokens, timeout):
    """One streamed chat request. TTFT is the first chunk with text; tokens is the count of content chunks."""
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"http://{hub}/v1/chat/completions", body, {"content-type": "application/json"})
    r = {"start": time.perf_counter(), "ttft": None, "tokens": 0, "text": "", "error": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"]
                if "content" not in delta:
                    continue
                r["tokens"] += 1
                r["text"] += delta["content"]
                if r["ttft"] is None and delta["content"]:
                    r["ttft"] = time.perf_counter() - r["start"]
    except urllib.error.HTTPError as e:
        try: r["error"] = f"HTTP {e.code}: {json.load(e).get('detail', '')}"[:200]
        except Exception: r["error"] = f"HTTP {e.code}"
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"[:200]
    r["end"] = time.perf_counter()
    r["latency"] = r["end"] - r["start"]
    return r


def run_level(hub, tasks, n, max_tokens, timeout):
    """Exactly n workers pulling from one queue, so n requests stay in flight until the queue drains."""
    q = queue.Queue()
    for i in range(len(tasks)):
        q.put(i)
    res = [None] * len(tasks)

    def worker():
        while True:
            try: i = q.get_nowait()
            except queue.Empty: return
            res[i] = ask(hub, tasks[i]["prompt"], max_tokens, timeout)
    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads: t.start()
    for t in threads: t.join()
    return res


def pct(xs, p):
    """Nearest-rank percentile; fine for a handful of samples where interpolation would only invent precision."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))] if xs else None


def summarise(level, tasks, res, predicted):
    ok = [r for r in res if not r["error"]]
    wall = max(r["end"] for r in res) - min(r["start"] for r in res)
    tokens = sum(r["tokens"] for r in ok)
    ttft = [r["ttft"] * 1000 for r in ok if r["ttft"] is not None]
    lat = [r["latency"] * 1000 for r in ok]
    per_tok = [(r["latency"] - r["ttft"]) * 1000 / (r["tokens"] - 1) for r in ok if r["ttft"] is not None and r["tokens"] >= 2]
    checked = correct = 0
    failures = []
    for t, r in zip(tasks, res):
        if r["error"]:
            failures.append(f"{t['kind']}: {t['prompt'][:50]}... -> {r['error']}")
            continue
        good = check(t, r["text"], r["tokens"])
        if t["expect"] is None:
            if not good:
                failures.append(f"prose: {t['prompt'][:50]}... -> expected >= 5 tokens, got {r['tokens']}")
            continue
        checked += 1
        correct += good
        if not good:
            failures.append(f"{t['kind']}: {t['prompt'].rsplit(' Answer', 1)[0]} "
                            f"-> expected {t['expect']}, got '{r['text'].strip()[:60]}'")
    return {
        "concurrency": level, "requests": len(res), "errors": len(res) - len(ok), "wall_s": wall, "tokens": tokens,
        "tok_s": tokens / wall if wall else 0.0,
        "ttft_p50_ms": pct(ttft, 0.5), "ttft_p95_ms": pct(ttft, 0.95),
        "lat_p50_ms": pct(lat, 0.5), "lat_p95_ms": pct(lat, 0.95),
        "ms_per_token": statistics.median(per_tok) if per_tok else None,
        "predicted_ms_per_token": predicted if level == 1 else None,
        "checked": checked, "correct": correct,
        "correct_pct": 100.0 * correct / checked if checked else None,
    }, failures


def get_status(hub):
    try:
        return json.load(urllib.request.urlopen(f"http://{hub}/status", timeout=5))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hub", default="127.0.0.1:8000")
    p.add_argument("--concurrency", default="1,2,4,8", help="comma-separated levels")
    p.add_argument("--requests", "--tasks", type=int, default=16, help="requests per level")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300, help="per request, seconds")
    p.add_argument("--min-correct", type=float, default=0.3, help="fail the run if correctness at the lowest level is below this")
    p.add_argument("--json", action="store_true", help="machine output only")
    args = p.parse_args()
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    say = (lambda *a: print(*a, file=sys.stderr)) if args.json else print

    # Refuse early: a 503 per request would only read as a very slow cluster.
    status = get_status(args.hub)
    if status is None:
        say(f"cannot reach the hub at {args.hub}"); sys.exit(1)
    plan = status.get("plan") or {}
    deadline = time.time() + 120
    while plan.get("rebalancing") and time.time() < deadline:
        say("hub is rebalancing, waiting..."); time.sleep(2)
        status = get_status(args.hub) or status
        plan = status.get("plan") or {}
    if not status["pipeline_ok"]:
        why = f" {plan['last_reason']}." if plan.get("last_reason") else ""
        say(f"pipeline incomplete, no node holds layers {status['missing_layers']}.{why} Join a device first: {status['join_url']}")
        sys.exit(1)
    who = ", ".join(f"{k} {v['layers'][0]}-{v['layers'][1] - 1}"
                    for k, v in sorted(status["nodes"].items(), key=lambda x: x[1]["layers"][0]) if v["live"])
    say(f"cluster: {who or 'nothing live'}")
    predicted = plan.get("predicted_ms_per_token")
    if plan:
        say(f"plan gen {plan.get('gen')}: {', '.join(plan.get('active', []))}; util {plan.get('util')}; "
            f"predicted {predicted:.0f} ms/token, {plan.get('predicted_tok_s', 0):.1f} tok/s at concurrency {plan.get('concurrency')}")

    tasks = bank(random.Random(args.seed), args.requests)
    rows, failures = [], []
    for level in levels:
        say(f"concurrency {level}: {len(tasks)} requests...")
        res = run_level(args.hub, tasks, level, args.max_tokens, args.timeout)
        row, fails = summarise(level, tasks, res, predicted)
        rows.append(row)
        failures += [f"[c={level}] {f}" for f in fails]

    if args.json:
        print(json.dumps({"plan": status.get("plan"), "levels": rows, "failures": failures}, indent=1))
    else:
        f = lambda x: "-" if x is None else f"{x:.0f}"
        print(f"\n{'conc':>5} {'reqs':>5} {'err':>4} {'tok/s':>7}  {'ttft p50/p95 ms':>16}  {'lat p50/p95 ms':>16}  {'ms/tok':>7} {'pred':>5}  correct")
        for r in rows:
            corr = "-" if r["correct_pct"] is None else f"{r['correct_pct']:.0f}% ({r['correct']}/{r['checked']})"
            print(f"{r['concurrency']:>5} {r['requests']:>5} {r['errors']:>4} {r['tok_s']:>7.2f}  "
                  f"{f(r['ttft_p50_ms']):>7} /{f(r['ttft_p95_ms']):>7}  {f(r['lat_p50_ms']):>7} /{f(r['lat_p95_ms']):>7}  "
                  f"{f(r['ms_per_token']):>7} {f(r['predicted_ms_per_token']):>5}  {corr}")
        for line in failures:
            print("  " + line)

    first = rows[0]
    bad = first["correct_pct"] is not None and first["correct_pct"] < 100 * args.min_correct
    if bad:
        say(f"correctness {first['correct_pct']:.0f}% at concurrency {first['concurrency']} is below {100 * args.min_correct:.0f}%: check the weights")
    if any(r["errors"] > r["requests"] - r["errors"] for r in rows):
        say("a level had more errors than successes"); bad = True
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
