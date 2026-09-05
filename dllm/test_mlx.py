"""Check: dllm.mlx_backend.MlxShard's output matches dllm.model.Shard's (torch, CPU) on the real
fp32 Qwen2.5-0.5B shards in dist/ -- prefill, single-token decode, and a multi-step decode that
exercises the KV cache across several calls, plus the batched-decode path and int8/int4 support.

Skips cleanly if MLX is not installed (Apple Silicon + Metal only), or if dist/ is not on disk.
"""
import os
import numpy as np
import pytest
import torch

from dllm.model import Cfg, Layer, Shard
from dllm.quant import dequant
from dllm.slicer import put

mlx_backend = pytest.importorskip("dllm.mlx_backend", reason="mlx is not installed")
MlxShard = mlx_backend.MlxShard
MlxLayer = mlx_backend.MlxLayer

DIST = "dist"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{DIST}/config.json"),
                                 reason=f"no shards on disk at {DIST}/")

# torch and MLX both run this in fp32; the gap between them is float rounding order across two
# different math libraries/hardware paths, not precision loss, so this stays tight.
REL_TOL = 1e-3


def _rel_err(a, b):
    return (a - b).norm().item() / a.norm().item()


@pytest.fixture(scope="module")
def cfg():
    return Cfg.load(f"{DIST}/config.json")


def test_prefill_then_decode_matches_torch(cfg):
    """One pair of shards, walked through prefill (n>1), a single-token decode continuing that
    same cache, and several more decode steps after it -- so the KV cache is exercised across
    calls, not just compared once right after prefill."""
    torch_shard, mlx_shard = Shard(cfg, DIST, 0, cfg.n_layers), MlxShard(cfg, DIST, 0, cfg.n_layers)
    g = torch.Generator().manual_seed(0)
    worst_rel = 0.0

    def step(n, pos0, label):
        nonlocal worst_rel
        x = torch.randn(1, n, cfg.hidden, generator=g)
        pos = torch.arange(pos0, pos0 + n)
        yt = torch_shard(x, pos, req="r")
        ym = mlx_shard(x, pos, req="r")
        err, rel = (yt - ym).abs().max().item(), _rel_err(yt, ym)
        worst_rel = max(worst_rel, rel)
        print(f"  {label}: n={n} pos={pos0} max|d|={err:.3e} rel={rel:.3e}")
        assert rel < REL_TOL, f"{label} drifted {rel:.2e} from torch"

    step(9, 0, "prefill        ")     # n > 1: the shape a request's prompt arrives as
    step(1, 9, "decode step 0  ")     # single new token against that prefill's cache
    for i in range(1, 5):             # several more steps: the cache must keep agreeing, not just once
        step(1, 9 + i, f"decode step {i}  ")

    torch_shard.reset("r"); mlx_shard.reset("r")
    print(f"prefill + 5-step decode: worst relative difference = {worst_rel:.3e}")


def test_forward_batch_matches_torch(cfg):
    """Batched decode across requests with different histories -- the path node.py's fwd_batch
    handler calls, and the largest throughput lever the cluster has."""
    torch_shard, mlx_shard = Shard(cfg, DIST, 0, cfg.n_layers), MlxShard(cfg, DIST, 0, cfg.n_layers)
    rng = np.random.default_rng(1)
    hist_lens = [4, 1, 7]
    prompts = [rng.standard_normal((1, n, cfg.hidden), dtype=np.float32) for n in hist_lens]
    for i, p in enumerate(prompts):
        torch_shard(torch.from_numpy(p), torch.arange(hist_lens[i]), req=f"b{i}")
        mlx_shard(torch.from_numpy(p), torch.arange(hist_lens[i]), req=f"b{i}")

    step = torch.from_numpy(rng.standard_normal((len(hist_lens), 1, cfg.hidden), dtype=np.float32))
    reqs = [f"b{i}" for i in range(len(hist_lens))]
    yt = torch_shard.forward_batch(step, hist_lens, reqs)
    ym = mlx_shard.forward_batch(step, hist_lens, reqs)
    err, rel = (yt - ym).abs().max().item(), _rel_err(yt, ym)
    print(f"forward_batch ({len(hist_lens)} rows, histories {hist_lens}): max|d|={err:.3e} rel={rel:.3e}")
    assert rel < REL_TOL, f"batched decode drifted {rel:.2e} from torch"


@pytest.mark.parametrize("scheme", ["int8", "int4"])
def test_quantised_layer_matches_torch_quantised_layer(cfg, scheme):
    """dllm.mlx_backend must support fp32/int8/int4 shards, dequantising at load the same way the
    numpy node does (dllm.quant.dequant). Checked against the torch Shard's own quantised path
    (dllm.model._q, its fast-int8-kernel route) on the same real layer_00 weights, the same way
    dllm/test_int8.py and dllm/test_int4.py check the torch path against fp32."""
    with np.load(f"{DIST}/layer_00.npz") as z:
        fp32 = {k: z[k].astype(np.float32) for k in z.files}
    q = {}
    for k, v in fp32.items():
        put(q, k, v, scheme)
    torch_layer = Layer(cfg, {k: torch.from_numpy(v) for k, v in q.items()})

    class _Npz:  # dequant() only needs .files and __getitem__ -- exactly what np.load(...) gives it
        files = list(q)
        def __getitem__(self, k):
            return q[k]

    mlx_layer = MlxLayer(cfg, dequant(_Npz()))
    x = torch.randn(1, 6, cfg.hidden, generator=torch.Generator().manual_seed(2))
    pos = torch.arange(6)
    with torch.no_grad():
        yt, _, _ = torch_layer(x, pos)
    xm, posm = mlx_backend._to_mx(x), mlx_backend._to_mx(pos, dtype=np.float32)
    ym, _, _ = mlx_layer(xm, posm)
    rel = _rel_err(yt, mlx_backend._to_torch(ym))
    print(f"{scheme} layer_00: mlx-dequantised vs torch quantised-kernel, rel={rel:.3e}")
    assert rel < 0.01, f"{scheme} layer drifted {rel:.4f} between backends"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
