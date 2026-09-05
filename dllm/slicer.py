"""Build step on a Mac: HF checkpoint -> shards/layer_XX.npz + head.npz + config.json.
npz so a phone node needs only numpy, no safetensors/torch. Never runs on a node.
python -m dllm.slicer Qwen/Qwen2.5-0.5B-Instruct shards"""
import glob, hashlib, json, os, shutil, sys
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open


def _np(t):
    return t.float().numpy() if t.dtype.is_floating_point else t.numpy()


def content_hash(w):
    """Hash the weights themselves, not the .npz container. np.savez stamps zip timestamps, so two
    slicer runs produce different files for identical weights. This is stable across runs and
    machines, which lets a coordinator that holds no layers still verify what a node reports."""
    h = hashlib.sha256()
    for k in sorted(w):
        a = np.ascontiguousarray(w[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:16]


def slice_model(repo, out):
    src = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
    os.makedirs(out, exist_ok=True)
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "vocab.json", "merges.txt"]:
        if os.path.exists(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", out)
    n_layers = json.load(open(f"{src}/config.json"))["num_hidden_layers"]
    buckets, head = {i: {} for i in range(n_layers)}, {}
    for path in sorted(glob.glob(f"{src}/*.safetensors")):
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if k.startswith("model.layers."):
                    i = int(k.split(".")[2])
                    buckets[i][k.split(f"model.layers.{i}.")[1]] = _np(f.get_tensor(k))
                else:
                    head[k] = _np(f.get_tensor(k))
    for i, w in buckets.items():
        np.savez(f"{out}/layer_{i:02d}.npz", **w)
    np.savez(f"{out}/head.npz", **head)
    files = {os.path.basename(p): os.path.getsize(p) for p in sorted(glob.glob(f"{out}/*.npz"))}
    hashes = {str(i): content_hash(w) for i, w in buckets.items()}
    json.dump({"repo": repo, "n_layers": n_layers, "files": files, "layer_hashes": hashes},
              open(f"{out}/manifest.json", "w"), indent=1)
    print(f"{n_layers} layer shards + head -> {out}  ({sum(files.values())/2**30:.2f} GB total)")


if __name__ == "__main__":
    slice_model(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "shards")
