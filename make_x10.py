from datasets import DatasetDict, concatenate_datasets, load_from_disk

HF_REPO = "seashello/babylm-strict-small-512-x10"
d = load_from_disk("babylm_packed")
copies = [
    d["train"].shuffle(seed=e).flatten_indices() for e in range(10)
]  # 10 epochs, each shuffled
big = concatenate_datasets(copies)
big = big.remove_columns("idx").add_column("idx", list(range(len(big))))
print(big)  # expect 306,470 rows
DatasetDict({"train": big, "validation": d["validation"]}).push_to_hub(HF_REPO)
