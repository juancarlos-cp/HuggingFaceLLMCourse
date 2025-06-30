import evaluate
import torch

from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    DataCollatorWithPadding,
    get_scheduler,
)

from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    fn_kwargs={"tokenizer": bert_tokenizer},
    batched=True)

data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

def evaluate_dataset(model, eval_dataloader):
    loss_metric = 0.0
    glue_metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        loss_metric += loss.item()
        glue_metric.add_batch(predictions=predictions, references=batch["labels"])

    mean_loss = loss_metric/len(eval_dataloader)
    glue_compute = glue_metric.compute()

    return mean_loss, glue_compute


train_dataloader = DataLoader(
    tokenized_datasets["train"], 
    shuffle=True, 
    batch_size=8, 
    collate_fn=data_collator,
)

valid_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    shuffle=True, 
    batch_size=8, 
    collate_fn=data_collator,
)

test_dataloader = DataLoader(
    tokenized_datasets["test"], 
    shuffle=True, 
    batch_size=8, 
    collate_fn=data_collator,
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)

num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    loss_metric = 0.0
    glue_metric = evaluate.load("glue", "mrpc")
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loss_metric += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        glue_metric.add_batch(predictions=predictions, references=batch["labels"])
        # progress_bar.update(1)

    mean_loss = loss_metric/len(train_dataloader)
    glue_compute = glue_metric.compute()

    valid_mean_loss, valid_glue_compute = evaluate_dataset(model, valid_dataloader)

    glue_acc = glue_compute['accuracy']
    glue_f1 = glue_compute['f1']

    valid_glue_acc = valid_glue_compute['accuracy']
    valid_glue_f1 = valid_glue_compute['f1']

    print(f'epoch {epoch+1}')
    print(f'train loss:{mean_loss:.4f} valid loss:{valid_mean_loss:.4f}')
    print(f'train glue acc:{glue_acc:.4f} valid glue acc:{valid_glue_acc:.4f}')
    print(f'train glue f1:{glue_f1:.4f} valid glue f1:{valid_glue_f1:.4f}')


# I feel that this routine is incomplete,  I will repair this tomorrow.
# I want loss and validation every epoch

print()
mean_loss, test_glue_compute = evaluate_dataset(model, test_dataloader)

glue_acc = test_glue_compute['accuracy']
glue_f1 = test_glue_compute['f1']

print(f'test loss:{mean_loss:.4f}')
print(f'test glue acc:{glue_acc:.4f}')
print(f'test glue f1:{glue_f1:.4f}')

