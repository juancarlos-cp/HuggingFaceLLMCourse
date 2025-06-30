import math
import torch

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel, 
    AutoConfig, 
    get_scheduler,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)

raw_datasets = DatasetDict(
    {
        "train": ds_train.shuffle().select(range(2400)),
        "valid": ds_valid.shuffle().select(range(480))
    }
)

context_length = 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets.set_format("torch")

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    loss_type = "causal",
)

model = GPT2LMHeadModel(config)

train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=12, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=12)

model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 2

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        # Move tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"\nEpoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}  Perplexity: {math.exp(avg_train_loss)}")

    # Evaluation
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            eval_loss += outputs.loss.item()

    avg_eval_loss = eval_loss / len(eval_dataloader)
    print(f"Epoch {epoch+1} - Eval Loss: {avg_eval_loss:.4f}  Perplexity: {math.exp(avg_eval_loss)}\n")

save_path = "./models/causal_python"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# this model does a horrible job because it is not well-trained.
