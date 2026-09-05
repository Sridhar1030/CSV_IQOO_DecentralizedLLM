"""Check: int4 shards survive the round trip and still track fp32.

int4 has two things int8 does not: a packing (two codes per byte) and a group scale, and either
can be got subtly wrong in a way that still produces plausible-looking numbers. So the pack is
checked exactly, and the layer is checked against fp32 on real weights."""
import os
import numpy as np
import pytest
import torch

from dllm.model import Cfg, Layer, _q
from dllm.quant import GROUP, dequant, quantize4, unpack4
from dllm.slicer import put

SHARD = next((d for d in ("dist", "mac_shards", "shards") if os.path.exists(f"{d}/layer_00.npz")), None)


def test_pack_is_two_codes_per_byte():
    a = np.random.default_rng(0).standard_normal((8, 256)).astype(np.float32)
    packed, s = quantize4(a)
    assert packed.dtype == np.uint8 and packed.shape == (8, 128)
    assert s.shape == (8, 256 // GROUP)


def test_unpack_inverts_pack_exactly():
    """Values already on the grid must survive the nibbles untouched. The scale is derived from the
    data, so a group whose largest magnitude is 7 gets scale 1 and every code round-trips."""
    rng = np.random.default_rng(1)
    a = rng.integers(-7, 8, size=(6, 64)).astype(np.float32)
    a[:, 0] = 7.0                                      # pin the group max, so the scale comes out 1
    back = unpack4(*quantize4(a, g=64))
    assert np.abs(back - a).max() < 1e-4, np.abs(back - a).max()


def test_group_scale_confines_an_outlier_to_its_own_group():
    """This is the whole reason for grouping. One loud column still swamps the 127 columns sharing
    its scale -- 4 bits cannot span 1000:1 -- but with a row-wide scale it would have swamped the
    entire row, and here the next group is untouched."""
    a = np.ones((4, 2 * GROUP), np.float32)
    a[:, 0] = 1000.0
    back = unpack4(*quantize4(a))
    assert np.abs(back[:, GROUP:] - 1.0).max() < 0.1        # the group next door is fine
    assert np.abs(back[:, 1:GROUP] - 1.0).max() > 0.5       # its own group is not, and should not be


def test_loader_dispatches_on_dtype():
    out = {}
    put(out, "w4", np.random.default_rng(2).standard_normal((16, 256)), "int4")
    put(out, "w8", np.random.default_rng(3).standard_normal((16, 256)), "int8")
    assert out["w4"].dtype == np.uint8 and out["w4.scale"].ndim == 2
    assert out["w8"].dtype == np.int8 and out["w8.scale"].ndim == 1
    w = {k: torch.from_numpy(v) for k, v in out.items()}
    for key in ("w4", "w8"):
        t, s = _q(w, key)
        assert t.dtype == torch.int8 and s.ndim == 1, key    # both arrive as int8 for the fast kernel


def test_numpy_node_reads_int4_like_torch_does():
    a = np.random.default_rng(4).standard_normal((32, 256)).astype(np.float32)
    out = {}
    put(out, "w", a, "int4")
    np.savez("/tmp/dllm_int4_probe.npz", **out)
    with np.load("/tmp/dllm_int4_probe.npz") as z:
        viaNumpy = dequant(z)["w"]
    t, s = _q({k: torch.from_numpy(v) for k, v in out.items()}, "w")
    viaTorch = (t.float() * s[:, None]).numpy()
    assert np.abs(viaNumpy - viaTorch).max() / np.abs(viaNumpy).max() < 0.02


@pytest.mark.skipif(SHARD is None, reason="no sliced fp32 shard on disk")
def test_int4_layer_tracks_fp32_layer_on_real_weights():
    cfg = Cfg.load(f"{SHARD}/config.json")
    with np.load(f"{SHARD}/layer_00.npz") as z:
        fp32 = {k: z[k].astype(np.float32) for k in z.files}
    q4 = {}
    for k, v in fp32.items():
        put(q4, k, v, "int4")

    t = lambda d: {k: torch.from_numpy(v) for k, v in d.items()}
    a, b = Layer(cfg, t(fp32)), Layer(cfg, t(q4))
    x = torch.randn(1, 12, cfg.hidden)
    pos = torch.arange(12)
    with torch.no_grad():
        ya, _, _ = a(x, pos)
        yb, _, _ = b(x, pos)
    rel = ((ya - yb).norm() / ya.norm()).item()
    assert rel < 0.05, f"int4 layer drifted {rel:.4f} from fp32"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
