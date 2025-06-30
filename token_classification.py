import evaluate
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer, 
    DataCollatorForTokenClassification,
    get_scheduler,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_LABEL = -100

raw_datasets = load_dataset("conll2003")
ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

# label_type "ner_tags", "pos_tags", "chunk_tags"
def display_labeled_tokens(dataset, index, label_type):
    example = dataset[index]
    words = example["tokens"]
    labels = example[label_type]
    line1 = ""
    line2 = ""
    label_names = dataset.features[label_type].feature.names
    for word, label in zip(words, labels):
        full_label = label_names[label]
        max_length = max(len(word), len(full_label))
        line1 += word + " " * (max_length - len(word) + 1)
        line2 += full_label + " " * (max_length - len(full_label) + 1)

    print(line1)
    print(line2)

display_labeled_tokens(raw_datasets["train"], 0, "ner_tags")

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = IGNORE_LABEL if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(IGNORE_LABEL)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            # if label % 2 == 1:
            #     label += 1
            # new_labels.append(label)
            # Or set to -100 so that the loss is not amplified
            new_labels.append(IGNORE_LABEL)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

metric = evaluate.load("seqeval")

# I really do not like metric being a global like this
# An object oriented approach would suit me so much better.
# ignore_label however would work as a global.  IGNORE_LABEL
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != IGNORE_LABEL] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != IGNORE_LABEL]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
model.to(device)

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    collate_fn=data_collator, 
    batch_size=8,
)

optimizer = AdamW(model.parameters(), lr=2e-5)

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != IGNORE_LABEL] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != IGNORE_LABEL]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        # Move everything in the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        true_predictions, true_labels = postprocess(predictions, labels)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

# I think I should save the model.  How do I do that?
save_path = "./models/bert_token_classification"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# It works, but does a really lousy job and not like in the example.  Maybe because 
# I am not using the hub?  Maybe because I am ignoring subsequent tokens? 
# All this accounting stuff is BORING!

# could do this
# model = AutoModelForTokenClassification.from_pretrained(save_path)
# tokenizer = AutoTokenizer.from_pretrained(save_path)

# but I did this

from transformers import pipeline

token_classifier = pipeline(
    "token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)

token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
token_classifier("My name is Inigo Montoya, you killed my father, prepare to die.")
