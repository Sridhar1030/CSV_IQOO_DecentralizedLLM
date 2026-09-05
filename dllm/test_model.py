"""Check: our Shard+Head match transformers Qwen2ForCausalLM logits on a cached 0.5B model, prefill + 3 decode steps with KV cache."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dllm.model import Cfg, Shard, Head
from dllm.slicer import slice_model

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "/tmp/dllm_test_shards"


def main():
    slice_model(REPO, OUT)
    cfg = Cfg.load(f"{OUT}/config.json")
    tok = AutoTokenizer.from_pretrained(OUT)
    ref = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32).eval()
    head = Head(cfg, OUT)
    mid = cfg.n_layers // 2
    shards = [Shard(cfg, OUT, 0, mid), Shard(cfg, OUT, mid, cfg.n_layers)]  # two-node pipeline

    ids = tok("The capital of France is", return_tensors="pt").input_ids
    ours, theirs = [], []
    with torch.no_grad():
        past = None
        for step in range(4):
            if step == 0:
                x_ids, pos = ids[0], torch.arange(ids.shape[1])
                out = ref(ids, use_cache=True)
            else:
                x_ids, pos = torch.tensor([theirs[-1]]), torch.tensor([ids.shape[1] + step - 1])
                out = ref(x_ids[None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            theirs.append(int(out.logits[0, -1].argmax()))
            x = head.embed_tokens(x_ids)
            for s in shards:
                x = s(x, pos, req="t")
            lg = head.logits(x)
            ours.append(int(lg.argmax()))
            err = (lg - out.logits[0, -1]).abs().max().item()
            print(f"step {step}: ours={ours[-1]} ref={theirs[-1]} max|dlogit|={err:.4f}")
            assert err < 0.05, "logit mismatch"
    assert ours == theirs, (ours, theirs)
    print("OK:", tok.decode(ours))


if __name__ == "__main__":
    main()
