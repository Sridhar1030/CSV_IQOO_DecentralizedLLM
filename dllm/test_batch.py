"""A batched decode step must give exactly what the same rows give one at a time.

Batching is the largest throughput lever the cluster has, and it is also the easiest to get subtly
wrong: rows sit at different positions with different histories, so a shared cache or a shared
position would corrupt one request with another's context and still look plausible.
"""
import os
import numpy as np
import torch

from dllm.model import Cfg, Shard

OUT = os.getenv("DLLM_SHARDS", "mac_shards")


def _range():
    f = sorted(x for x in os.listdir(OUT) if x.startswith("layer_"))
    return int(f[0][6:8]), int(f[-1][6:8]) + 1


def test_batched_decode_matches_one_at_a_time():
    cfg = Cfg.load("hub_shards/config.json")
    a, b = _range()
    rng = np.random.default_rng(0)
    # three requests with different histories, so their caches and positions genuinely differ
    prompts = [rng.standard_normal((1, n, cfg.hidden), dtype=np.float32) for n in (5, 2, 9)]

    solo = Shard(cfg, OUT, a, b)
    want = []
    for i, p in enumerate(prompts):
        solo(torch.from_numpy(p), torch.arange(p.shape[1]), f"s{i}")          # prefill
        step = rng.standard_normal((1, 1, cfg.hidden), dtype=np.float32)
        prompts[i] = (p, step)
        want.append(solo(torch.from_numpy(step), torch.tensor([p.shape[1]]), f"s{i}"))

    batch = Shard(cfg, OUT, a, b)
    for i, (p, _) in enumerate(prompts):
        batch(torch.from_numpy(p), torch.arange(p.shape[1]), f"b{i}")          # same prefill
    x = torch.from_numpy(np.concatenate([s for _, s in prompts], 0))           # (3, 1, hidden)
    positions = [p.shape[1] for p, _ in prompts]
    got = batch.forward_batch(x, positions, [f"b{i}" for i in range(len(prompts))])

    for i in range(len(prompts)):
        w = want[i][0]
        err = (got[i] - w).abs().max().item() / w.abs().max().item()
        print(f"  request {i}: history {positions[i]:>2} tokens, peak |activation| {w.abs().max():6.1f}, "
              f"relative difference {err:.1e}")
        # Relative, not absolute: these hidden states reach 70, and a few channels are far larger
        # than the rest, so an absolute bound would flag ordinary float32 rounding.
        assert err < 1e-4, f"row {i} differs by {err}"

    # The sharper check. A batch of one must take the same path as decoding alone, so any difference
    # there is a real bug rather than accumulation order.
    single = Shard(cfg, OUT, a, b)
    single(torch.from_numpy(prompts[2][0]), torch.arange(prompts[2][0].shape[1]), "one")
    got_one = single.forward_batch(torch.from_numpy(prompts[2][1]), [prompts[2][0].shape[1]], ["one"])
    exact = (got_one[0] - want[2][0]).abs().max().item()
    print(f"  batch of one against decoding alone: {exact:.1e} (must be exactly zero)")
    assert exact == 0.0, f"batch of one is not the same computation: {exact}"
    print("  batched decode matches decoding each request alone")


if __name__ == "__main__":
    test_batched_decode_matches_one_at_a_time()
