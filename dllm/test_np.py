"""Check the numpy phone node's math against the torch node on identical weights, prefill + 2 decode with KV cache."""
import os
import numpy as np, torch
from dllm.model import Cfg, Shard as TShard
from dllm.np_node import Shard as NShard, read_cfg, to_wire, from_wire

OUT = os.getenv("DLLM_SHARDS", "mac_shards")   # any directory holding a contiguous run of layer_XX.npz


def main():
    tc, nc = Cfg.load(f"{OUT}/config.json"), read_cfg(f"{OUT}/config.json")
    files = sorted(f for f in os.listdir(OUT) if f.startswith("layer_"))
    a, b = int(files[0][6:8]), int(files[-1][6:8]) + 1        # whatever range this machine holds
    t, n = TShard(tc, OUT, a, b), NShard(nc, OUT, a, b)
    rng = np.random.default_rng(0)
    for step, (ntok, pos0) in enumerate([(5, 0), (1, 5), (1, 6)]):
        x = rng.standard_normal((1, ntok, tc.hidden), dtype=np.float32)
        yt = t(torch.from_numpy(x), torch.arange(pos0, pos0 + ntok), req="r").numpy()
        yn = n(x, np.arange(pos0, pos0 + ntok), req="r")
        err = np.abs(yt - yn).max()
        print(f"step {step}: n={ntok} pos={pos0} max|d|={err:.2e}")
        assert err < 2e-3, err
    # both wire encodings must survive a round trip, and fp32 must be exact
    x = rng.standard_normal((1, 3, 8), dtype=np.float32)
    assert np.abs(from_wire(to_wire(x), x.shape) - x).max() < 0.02
    assert np.abs(from_wire(to_wire(x, "fp32"), x.shape, "fp32") - x).max() == 0
    print("numpy node matches torch node")


def test_numpy_batched_decode_matches_torch():
    """The phone's numpy runtime must batch the same way the laptop's torch one does."""
    import os
    from dllm.model import Shard as TShard
    tc, nc = Cfg.load(f"{OUT}/config.json"), read_cfg(f"{OUT}/config.json")
    files = sorted(f for f in os.listdir(OUT) if f.startswith("layer_"))
    a, b = int(files[0][6:8]), int(files[-1][6:8]) + 1
    t, n = TShard(tc, OUT, a, b), NShard(nc, OUT, a, b)
    rng = np.random.default_rng(1)
    hist = [4, 1, 7]
    for i, hn in enumerate(hist):                       # give each row a different history
        p = rng.standard_normal((1, hn, tc.hidden), dtype=np.float32)
        t(torch.from_numpy(p), torch.arange(hn), f"r{i}")
        n(p, np.arange(hn), f"r{i}")
    step = rng.standard_normal((len(hist), 1, tc.hidden), dtype=np.float32)
    reqs = [f"r{i}" for i in range(len(hist))]
    yt = t.forward_batch(torch.from_numpy(step), hist, reqs).numpy()
    yn = n.forward_batch(step, hist, reqs)
    err = np.abs(yt - yn).max() / np.abs(yt).max()
    print(f"batched decode, numpy vs torch, {len(hist)} rows: relative difference {err:.1e}")
    assert err < 1e-4, err


if __name__ == "__main__":
    main()
    test_numpy_batched_decode_matches_torch()
