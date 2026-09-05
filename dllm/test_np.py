"""Check the numpy phone node's math against the torch node on identical weights, prefill + 2 decode with KV cache."""
import numpy as np, torch
from dllm.model import Cfg, Shard as TShard
from dllm.np_node import Shard as NShard, read_cfg, to_bf16, from_bf16

OUT = "/tmp/dllm_test_shards"


def main():
    tc, nc = Cfg.load(f"{OUT}/config.json"), read_cfg(f"{OUT}/config.json")
    a, b = 12, 24
    t, n = TShard(tc, OUT, a, b), NShard(nc, OUT, a, b)
    rng = np.random.default_rng(0)
    for step, (ntok, pos0) in enumerate([(5, 0), (1, 5), (1, 6)]):
        x = rng.standard_normal((1, ntok, tc.hidden), dtype=np.float32)
        yt = t(torch.from_numpy(x), torch.arange(pos0, pos0 + ntok), req="r").numpy()
        yn = n(x, np.arange(pos0, pos0 + ntok), req="r")
        err = np.abs(yt - yn).max()
        print(f"step {step}: n={ntok} pos={pos0} max|d|={err:.2e}")
        assert err < 2e-3, err
    # bf16 codec round trip must survive the wire both directions
    x = rng.standard_normal((1, 3, 8), dtype=np.float32)
    assert np.abs(from_bf16(to_bf16(x), x.shape) - x).max() < 0.02
    print("numpy node matches torch node")


if __name__ == "__main__":
    main()
