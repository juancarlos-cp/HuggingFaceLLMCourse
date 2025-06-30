from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset(
    "json",
    data_files={
        "train": "./data/python/python/final/jsonl/train/*.jsonl.gz",
        "validation": "./data/python/python/final/jsonl/valid/*.jsonl.gz",
        "test": "./data/python/python/final/jsonl/test/*.jsonl.gz",
    }
)

# note hardcoded loaded dataset and "train"  I can do better
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["original_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

training_corpus = get_training_corpus()

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

new_tokenizer.save_pretrained("code-search-net-tokenizer")
