"""Check: int8 shards decode to the same thing fp32 shards do.

The quantisation itself is one line, so what is worth testing is the wiring around it: that
`slicer.put` writes scales the loader recognises, that `QLinear` picks the int8 path, and that a
whole `Layer` built from a quantised shard still lands within noise of the fp32 one. Runs on the
real layer_00 weights when they are on disk, because a quantiser that looks fine on random normals
can still fall over on the outlier channels real weights have."""
import os
import numpy as np
import pytest
import torch

from dllm.model import Cfg, Layer, QLinear, _q
from dllm.slicer import put, quantize

SHARD = next((d for d in ("dist", "mac_shards", "shards") if os.path.exists(f"{d}/layer_00.npz")), None)


def test_quantize_round_trip_keeps_per_row_scale():
    a = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
    a[7] *= 1e-3                                            # a quiet row must keep its resolution
    q, s = quantize(a)
    assert q.dtype == np.int8 and s.shape == (64,)
    err = np.abs(q.astype(np.float32) * s[:, None] - a).max(1) / np.abs(a).max(1)
    assert err.max() < 0.01, err.max()


def test_all_zero_row_does_not_divide_by_zero():
    a = np.zeros((4, 8), np.float32)
    q, s = quantize(a)
    assert np.isfinite(s).all() and (q == 0).all()


def test_qlinear_int8_matches_dequantised_linear():
    w = torch.randn(256, 128)
    q, s = quantize(w.numpy())
    x = torch.randn(3, 5, 128)
    got = QLinear(torch.from_numpy(q), torch.from_numpy(s), None)(x)
    want = torch.nn.functional.linear(x, torch.from_numpy(q).float() * torch.from_numpy(s)[:, None])
    assert got.shape == (3, 5, 256)
    assert (got - want).abs().max() < 1e-3


def test_put_writes_scales_the_loader_finds():
    out = {}
    put(out, "self_attn.q_proj.weight", np.random.default_rng(1).standard_normal((16, 8)), int8=True)
    put(out, "self_attn.q_proj.bias", np.zeros(16), int8=True)          # 1D stays fp32
    assert out["self_attn.q_proj.weight"].dtype == np.int8
    assert out["self_attn.q_proj.bias"].dtype == np.float32
    assert "self_attn.q_proj.bias.scale" not in out
    w = {k: torch.from_numpy(v) for k, v in out.items()}
    tensor, scale = _q(w, "self_attn.q_proj.weight")
    assert tensor.dtype == torch.int8 and scale is not None


@pytest.mark.skipif(SHARD is None, reason="no sliced fp32 shard on disk")
def test_int8_layer_tracks_fp32_layer_on_real_weights():
    cfg = Cfg.load(f"{SHARD}/config.json")
    with np.load(f"{SHARD}/layer_00.npz") as z:
        fp32 = {k: z[k].astype(np.float32) for k in z.files}
    q8 = {}
    for k, v in fp32.items():
        put(q8, k, v, int8=True)

    t = lambda d: {k: torch.from_numpy(v) for k, v in d.items()}
    a, b = Layer(cfg, t(fp32)), Layer(cfg, t(q8))
    assert b.q.scale is not None and a.q.scale is None

    x = torch.randn(1, 12, cfg.hidden)
    pos = torch.arange(12)
    with torch.no_grad():
        ya, _, _ = a(x, pos)
        yb, _, _ = b(x, pos)
    rel = (ya - yb).norm() / ya.norm()
    assert rel < 0.02, f"int8 layer drifted {rel:.4f} from fp32"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
