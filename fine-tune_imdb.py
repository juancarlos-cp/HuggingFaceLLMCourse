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

save_path = "./models/fine_tune_masked"
model = AutoModelForSequenceClassification.from_pretrained(
    save_path,
    num_labels=2,
    ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(save_path)

imdb_dataset = load_dataset("imdb")

def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=True, padding=False)
    return result

tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

def evaluate_dataset(model, eval_dataloader):
    loss_metric = 0.0
    correct = 0.0
    total = 0.0
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        loss_metric += loss.item()
        labels = batch["labels"]
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    mean_loss = loss_metric/len(eval_dataloader)
    accuracy = correct/total

    return mean_loss, accuracy

batch_size = 24
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

train_dataloader = DataLoader(
    tokenized_datasets["train"], 
    shuffle=True, 
    batch_size=batch_size, 
    collate_fn=data_collator,
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], 
    shuffle=True, 
    batch_size=batch_size, 
    collate_fn=data_collator,
)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.to(device)

for epoch in range(num_epochs):
    # Training
    loss_metric = 0.0
    correct = 0.0
    total = 0.0
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        loss_metric += loss.item()
        labels = batch["labels"]
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.update(1)
    
    accuracy = correct / total
    mean_loss = loss_metric/len(train_dataloader)
    test_mean_loss, test_accuracy = evaluate_dataset(model, test_dataloader)


    print(f'epoch {epoch+1}')
    print(f'train loss:{mean_loss:.4f} test loss:{test_mean_loss:.4f}')
    print(f'train accuracy:{accuracy:.4f} test accuracy:{test_accuracy:.4f}')

save_path = "./models/fine_tune_imdb"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
