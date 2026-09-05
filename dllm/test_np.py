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


if __name__ == "__main__":
    main()
