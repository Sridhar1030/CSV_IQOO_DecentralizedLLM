"""How a quantised shard is laid out, shared by the build step and both node runtimes.

numpy only, on purpose: `dllm.slicer` needs safetensors and `dllm.model` needs torch, but a phone
node running `dllm.np_node` has neither, and it still has to read the same files.

A quantised tensor keeps its own key and gains a sibling `<key>.scale`. The dtype says which
scheme was used, so no extra metadata rides along:

    fp32   float32 (O, I)     no scale
    int8   int8    (O, I)     scale float32 (O,)          one scale per output row
    int4   uint8   (O, I/2)   scale float32 (O, I/g)      two codes per byte, one scale per g columns
"""
import numpy as np

GROUP = 128     # int4 columns sharing one scale. 8 bits has the range to cover a whole row; 4 does not.


def quantize(a):
    """Symmetric int8 with one scale per output row. Rows are output channels, so a row with a
    small dynamic range keeps its resolution however large the rest of the matrix is."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    s = np.abs(a).max(axis=1) / 127.0
    s[s == 0] = 1.0                                        # an all-zero row would divide by zero
    q = np.rint(a / s[:, None]).clip(-127, 127).astype(np.int8)
    return q, s.astype(np.float32)


def quantize4(a, g=GROUP):
    """Symmetric int4, one scale per group of `g` columns, two codes packed per byte.

    A row-wide scale is enough for 8 bits but not for 4: one outlier column would flatten the
    whole row to a handful of levels. Grouping costs one fp32 per 128 weights, which is 3% on top
    of the 0.5 bytes, and buys back most of the accuracy."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    O, I = a.shape
    if I % g:
        g = I
    x = a.reshape(O, I // g, g)
    s = np.abs(x).max(-1, keepdims=True) / 7.0
    s[s == 0] = 1.0
    q = np.rint(x / s).clip(-8, 7).astype(np.int8).reshape(O, I)
    lo = (q[:, 0::2] + 8).astype(np.uint8)                 # +8 so a code lands in a nibble's 0..15
    hi = (q[:, 1::2] + 8).astype(np.uint8)
    return (hi << 4) | lo, s.squeeze(-1).astype(np.float32)


def unpack4(packed, scale):
    """Back to fp32. The group size is implied: one scale per (columns / scales-per-row) columns,
    so nothing extra has to be carried in the file."""
    O, half = packed.shape
    I = half * 2
    g = I // scale.shape[1]
    q = np.empty((O, I), np.int8)
    q[:, 0::2] = (packed & 0x0F).astype(np.int8) - 8
    q[:, 1::2] = (packed >> 4).astype(np.int8) - 8
    return (q.reshape(O, I // g, g).astype(np.float32) * scale[:, :, None]).reshape(O, I)


def dequant(z):
    """Every tensor in an npz as fp32, whichever scheme it used. What a numpy node wants: it has no
    int8 or int4 matmul worth using, so it spends the memory back at load time. The download is
    still a fraction of the size, which is the part a phone on wifi actually feels."""
    out = {}
    for k in z.files:
        if k.endswith(".scale"):
            continue
        s = f"{k}.scale"
        if s not in z.files:
            out[k] = z[k].astype(np.float32)
        elif z[k].dtype == np.uint8:
            out[k] = unpack4(z[k], z[s])
        else:
            out[k] = z[k].astype(np.float32) * z[s][:, None]
    return out
