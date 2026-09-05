"""Build step on a Mac: HF checkpoint -> shards/layer_XX.npz + head.npz + config.json.
npz so a phone node needs only numpy, no safetensors/torch. Never runs on a node.

python -m dllm.slicer Qwen/Qwen2.5-0.5B-Instruct shards
python -m dllm.slicer Qwen/Qwen2.5-32B-Instruct dist --int8 --stream

--int8 stores every projection as int8 with one fp32 scale per output row, a quarter of the
bytes. --stream pulls one checkpoint file at a time and deletes it once no later layer needs it,
so slicing a 65 GB checkpoint never needs 65 GB of free disk.
"""
import argparse, collections, glob, hashlib, json, os, shutil
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

from dllm.quant import quantize, quantize4

INDEX = "model.safetensors.index.json"
SIDECAR = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
           "vocab.json", "merges.txt"]


def _np(t):
    return t.float().numpy() if t.dtype.is_floating_point else t.numpy()


def put(out, key, a, quant):
    """Store one tensor. 2D tensors are the projections, the embedding and lm_head, and are the
    only ones worth quantising; norms and biases are 1D and stay fp32."""
    if quant and a.ndim == 2:
        q, s = quantize4(a) if quant == "int4" else quantize(a)
        out[key], out[f"{key}.scale"] = q, s
    else:
        out[key] = np.ascontiguousarray(a, dtype=np.float32)


def content_hash(w):
    """Hash the weights themselves, not the .npz container. np.savez stamps zip timestamps, so two
    slicer runs produce different files for identical weights. This is stable across runs and
    machines, which lets a coordinator that holds no layers still verify what a node reports."""
    h = hashlib.sha256()
    for k in sorted(w):
        a = np.ascontiguousarray(w[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:16]


class Checkpoint:
    """Reads tensors by name out of an HF checkpoint, fetching each file the first time it is
    asked for. With `stream`, a file is deleted as soon as no later step needs it, which keeps
    peak disk at the output plus the couple of checkpoint files still in play."""

    def __init__(self, repo, stream):
        self.repo, self.stream, self.open_files = repo, stream, {}
        try:
            self.weight_map = json.load(open(hf_hub_download(repo, INDEX)))["weight_map"]
        except Exception:                                   # small models ship one unsharded file
            self.weight_map = None
        self.last_use = {}

    def file_of(self, key):
        return self.weight_map[key] if self.weight_map else "model.safetensors"

    def plan(self, steps):
        """steps: list of (name, layer, [tensor keys]). Records, per checkpoint file, the last step
        that reads it, so `release` knows when the file can go."""
        for i, (_, _, keys) in enumerate(steps):
            for k in keys:
                self.last_use[self.file_of(k)] = i

    def get(self, key):
        f = self.file_of(key)
        if f not in self.open_files:
            self.open_files[f] = safe_open(hf_hub_download(self.repo, f), "pt")
        return _np(self.open_files[f].get_tensor(key))

    def release(self, step):
        if not self.stream:
            return
        for f, last in list(self.last_use.items()):
            if last != step:
                continue
            self.open_files.pop(f, None)
            try:
                p = hf_hub_download(self.repo, f)
                os.unlink(os.path.realpath(p))
                if os.path.islink(p):
                    os.unlink(p)
            except OSError:
                pass
            self.last_use.pop(f, None)


def slice_model(repo, out, quant=None, stream=False):
    os.makedirs(out, exist_ok=True)
    if stream:
        for f in SIDECAR:
            try:
                shutil.copy(hf_hub_download(repo, f), out)
            except Exception:
                pass
    else:
        src = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
        for f in SIDECAR:
            if os.path.exists(f"{src}/{f}"):
                shutil.copy(f"{src}/{f}", out)
    n_layers = json.load(open(f"{out}/config.json"))["num_hidden_layers"]

    ck = Checkpoint(repo, stream)
    if ck.weight_map:
        keys = list(ck.weight_map)
    else:
        with safe_open(hf_hub_download(repo, "model.safetensors"), "pt") as f:
            keys = list(f.keys())
    per_layer = collections.defaultdict(list)
    head_keys = []
    for k in keys:
        if k.startswith("model.layers."):
            per_layer[int(k.split(".")[2])].append(k)
        else:
            head_keys.append(k)

    # The head first: it reads the checkpoint's first and last file, and getting it out of the way
    # lets those files be released as the layers march past them.
    steps = [("head", None, head_keys)] + [(f"layer_{i:02d}", i, per_layer[i]) for i in range(n_layers)]
    ck.plan(steps)

    hashes = {}
    for step, (name, layer, ks) in enumerate(steps):
        # The embedding and lm_head never go below int8. Measured on the 0.5B, taking them to int4
        # costs more than quantising all 24 layers does (perplexity 16.1 -> 18.4), while int8 costs
        # nothing against fp32. They are one shard the coordinator holds, so the bytes are cheap.
        how = "int8" if layer is None and quant else quant
        w = {}
        for k in ks:
            short = k if layer is None else k.split(f"model.layers.{layer}.", 1)[1]
            put(w, short, ck.get(k), how)
        np.savez(f"{out}/{name}.npz", **w)
        if layer is not None:
            hashes[str(layer)] = content_hash(w)
        del w
        ck.release(step)
        print(f"  {name}.npz  {os.path.getsize(f'{out}/{name}.npz')/2**20:.0f} MB", flush=True)

    files = {os.path.basename(p): os.path.getsize(p) for p in sorted(glob.glob(f"{out}/*.npz"))}
    json.dump({"repo": repo, "n_layers": n_layers, "quant": quant or "fp32",
               "files": files, "layer_hashes": hashes},
              open(f"{out}/manifest.json", "w"), indent=1)
    print(f"{n_layers} layer shards + head -> {out}  ({sum(files.values())/2**30:.2f} GB total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out", nargs="?", default="shards")
    ap.add_argument("--int8", action="store_true", help="projections as int8 with one scale per output row")
    ap.add_argument("--int4", action="store_true",
                    help="projections as int4, two per byte, one scale per 128 columns. Half the bytes of int8, so a phone can hold layers of a model that would not otherwise fit on one")
    ap.add_argument("--stream", action="store_true",
                    help="fetch one checkpoint file at a time and delete it once no later layer "
                         "needs it, so peak disk stays near the size of the output")
    a = ap.parse_args()
    slice_model(a.repo, a.out, quant="int4" if a.int4 else "int8" if a.int8 else None, stream=a.stream)
