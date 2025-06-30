import collections
import math
import numpy as np
import torch

from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
)

from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model_checkpoint = "distilbert-base-uncased"

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def top5results(text):
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

imdb_dataset = load_dataset("imdb")

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)

# chunk_size = 128 # why not use the max?
chunk_size = 256
# chunk_size = 512 

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

# wait, are we going to use the unsupervised examples for text?  Which tokens do we mask?

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# oh, they already have something that does that.
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
# this is not 'smart' however, often adjacent tokens are masked and words that are
# assembled from multiple tokens have only one masked token - the rest are unmasked
# so we need to build one from scratch

wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

# train the native pytorch way

def evaluate_dataset(model, eval_dataloader):
    loss_metric = 0.0
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        loss_metric += loss.item()

    mean_loss = loss_metric/len(eval_dataloader)

    return mean_loss

# batch_size = 16
batch_size = 24

train_dataloader = DataLoader(
    lm_datasets["train"], 
    shuffle=True, 
    batch_size=batch_size, 
    collate_fn=whole_word_masking_data_collator,
)
test_dataloader = DataLoader(
    lm_datasets["test"], 
    shuffle=True, 
    batch_size=batch_size, 
    collate_fn=whole_word_masking_data_collator,
)
unsupervised_dataloader = DataLoader(
    lm_datasets["unsupervised"], 
    shuffle=True, 
    batch_size=batch_size, 
    collate_fn=whole_word_masking_data_collator,
)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(unsupervised_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    loss_metric = 0.0
    for batch in unsupervised_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        loss_metric += loss.item()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    test_mean_loss = evaluate_dataset(model, test_dataloader)
    mean_loss = loss_metric/len(unsupervised_dataloader)

    print(f'epoch {epoch+1}')
    print(f'train loss:{mean_loss:.4f} valid loss:{test_mean_loss:.4f}')
    print(f'train perplexity:{math.exp(mean_loss):.4f} valid perplexity:{math.exp(test_mean_loss):.2f}')

# I think I should save the model.  How do I do that?
save_path = "./models/fine_tune_masked"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

