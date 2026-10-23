import json

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

SEQ_LEN = 512
CHUNK = SEQ_LEN + 1  # +1 token so inputs/labels can be shifted by one
HF_REPO = "seashello/babylm-strict-small-512"

tok = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-hf")
texts = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train")["text"]
n_words = sum(len(t.split()) for t in texts)

ids = []
for i in range(0, len(texts), 10_000):
    for seq in tok(texts[i : i + 10_000], add_special_tokens=False)["input_ids"]:
        ids.extend(seq + [tok.eos_token_id])
    print(i, flush=True)

n = len(ids) // CHUNK
full = Dataset.from_dict(
    {"input_ids": [ids[i * CHUNK : (i + 1) * CHUNK] for i in range(n)]}
)
split = full.train_test_split(test_size=0.05, seed=0)


def with_idx(ds):
    return Dataset.from_dict(
        {"input_ids": ds["input_ids"], "idx": list(range(len(ds)))}
    )


out = DatasetDict(
    {"train": with_idx(split["train"]), "validation": with_idx(split["test"])}
)
stats = {
    "seq_len": SEQ_LEN,
    "n_words_total": n_words,
    "n_tokens_total": len(ids),
    "n_train_chunks": len(out["train"]),
    "n_val_chunks": len(out["validation"]),
    "words_per_epoch": n_words * len(out["train"]) / n,
}
json.dump(stats, open("data_stats.json", "w"), indent=2)
print(stats)
out.save_to_disk("babylm_packed")
out.push_to_hub(HF_REPO)
