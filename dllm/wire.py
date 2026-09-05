"""Frame = 4-byte big-endian header length + JSON header + raw payload. All WS messages are binary.
Activations travel as bf16 (2 B/elem); fp32 on both ends. Same format for Python and phone nodes."""
import json, struct
import numpy as np
import torch


def pack(hdr: dict, payload: bytes = b"") -> bytes:
    h = json.dumps(hdr, separators=(",", ":")).encode()
    return struct.pack(">I", len(h)) + h + payload


def unpack(buf: bytes):
    n = struct.unpack(">I", buf[:4])[0]
    return json.loads(buf[4 : 4 + n]), buf[4 + n :]


def to_bytes(x: torch.Tensor, dtype: str = "bf16") -> bytes:
    """bf16 halves the wire cost and is invisible between layers. The hop that feeds the lm_head
    must stay fp32: top-2 logit margins get down to 1e-4, well under bf16's ~4e-3 perturbation,
    so rounding there silently changes the sampled token."""
    if dtype == "fp32":
        return x.detach().to("cpu", torch.float32).numpy().tobytes()
    return x.detach().to("cpu", torch.bfloat16).view(torch.int16).numpy().tobytes()


def from_bytes(b: bytes, shape, dtype: str = "bf16") -> torch.Tensor:
    if dtype == "fp32":
        return torch.from_numpy(np.frombuffer(b, dtype=np.float32).copy()).reshape(shape)
    return torch.from_numpy(np.frombuffer(b, dtype=np.int16).copy()).view(torch.bfloat16).float().reshape(shape)


if __name__ == "__main__":
    x = torch.randn(1, 3, 8)
    h, p = unpack(pack({"t": "fwd", "n": 3}, to_bytes(x)))
    assert h["n"] == 3 and (from_bytes(p, x.shape) - x).abs().max() < 0.05
    assert (from_bytes(to_bytes(x, "fp32"), x.shape, "fp32") - x).abs().max() == 0
    print("wire ok")
