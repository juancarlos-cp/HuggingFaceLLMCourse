from pathlib import Path

from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

# the method below should write these
training_files = [
    "./data/wikitext_train.txt", 
    "./data/wikitext_test.txt", 
    "./data/wikitext_validation.txt"
] 

def write_dataset(ds, ds_name, splits):
    for split in splits:
        fp = f"./data/{ds_name}_{split}.txt"
        if not Path(fp).exists():
            print(f"writing {split} split")
            with open(fp, "w", encoding="utf-8") as f:
                for i in range(len(ds[split])):
                    f.write(ds[split][i]["text"] + "\n")


dataset = load_dataset("wikitext", name="wikitext-2-raw-v1")
splits = ["test", "train", "validation"]
ds_name = "wikitext"

# in retrospect, for now, this is a waste of time
# for a larger corpus, I would want to load preprocessed data locally
write_dataset(dataset, ds_name, splits)
training_dataset = dataset["train"]

def get_training_corpus():
    for i in range(0, len(training_dataset), 1000):
        yield training_dataset[i: i+1000]["text"]

# this is a WordPiece tokenizer model, not an LLM
# I wonder what pieces of the WordPiece model are retained
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
# the first step is normalization
# tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
# this is a little too convenient.  what if we want something custom?
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

# the next step is pre-tokenization
# tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# too convenient, you can do custom...
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# and more custom, but this implementation is very flawed
# pre_tokenizer = pre_tokenizers.Sequence(
#     [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
# )

# the next step is training
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
# we can do this
# tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
# or we can do this
tokenizer.train(training_files, trainer=trainer)

# the last step is post-processing
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
# magic chicken stratch
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

# oh, there is one more last step - the decoder
tokenizer.decoder = decoders.WordPiece(prefix="##")

tokenizer.save("tokenizer.json")
new_tokenizer = Tokenizer.from_file("tokenizer.json")

wrapped_tokenizer = PreTrainedTokenizerFast(
    # tokenizer_object=new_tokenizer,
    tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

