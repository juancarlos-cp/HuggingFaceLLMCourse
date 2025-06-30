import evaluate
import nltk
import numpy as np
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    get_scheduler,
)

from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def show_samples(dataset, num_samples=3, seed=1492):
    sample = dataset.shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Summary: {example['summary']}'")
        print(f"'>> Article: {example['document']}'")

def clean_text(example):
    cleaned_article = example["document"].replace('\n', ' ').strip()
    cleaned_summary = example["summary"].replace('\n', ' ').strip()
    return {
        "document": cleaned_article,
        "summary": cleaned_summary
    }


# dataset = load_dataset("EdinburghNLP/xsum")
# dataset.save_to_disk('./data/EdinburghNLP-xsum')
dataset = load_from_disk("./data/EdinburghNLP-xsum")

max_input_length = 512
max_target_length = 64

dataset = dataset.filter(lambda x: len(x['document']) < max_input_length)
dataset = dataset.map(clean_text)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
model.config.use_cache = False

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["document"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["summary"], 
        max_length=max_target_length, 
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

nltk.download("punkt")
nltk.download("punkt_tab")

def postprocess_text(preds, labels):

    predictions = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds = [pred.strip() for pred in decoded_preds]
    labels = [label.strip() for label in decoded_labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    dataset["train"].column_names
)

# how would I train a model to perform summarization?
# tokenized_datasets.set_format("torch")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

batch_size = 8
optimizer = AdamW(model.parameters(), lr=2e-5)

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    # tokenized_datasets["train"].shuffle().select(range(800)), 
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
valid_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    # tokenized_datasets["validation"].shuffle().select(range(200)), 
    collate_fn=data_collator, 
    batch_size=batch_size,
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], 
    collate_fn=data_collator, 
    batch_size=batch_size,
)

num_epochs = 12
# num_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.to(device)

for epoch in range(num_epochs):
    model.train()
    rouge_score = evaluate.load("rouge")
    loss_metric = 0.0
    valid_loss_metric = 0.0

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        loss_metric += loss.item()

        progress_bar.update(1)

    mean_loss = loss_metric/len(train_dataloader)

    model.eval()
    for batch in tqdm(valid_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            valid_loss_metric += loss.item()

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_target_length,
            )

            labels = batch["labels"]
            decoded_preds, decoded_labels = postprocess_text(generated, labels)
            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    mean_loss = loss_metric/len(train_dataloader)
    valid_loss = valid_loss_metric/len(valid_dataloader)
    rouge_result = rouge_score.compute()


    print(f'epoch {epoch+1} train loss:{mean_loss:.4f}')
    result = {key: value for key, value in rouge_result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(result)

save_path = './models/flan-t5-small-xsum'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


