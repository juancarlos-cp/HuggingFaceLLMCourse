from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
) 

checkpoint = "bert-base-uncased"
bert_model = BertModel.from_pretrained(checkpoint)
bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "mrpc")

# I really do not like using a tokenize function and map()
# There is just something too glib about the whole thing
# I found what.  It uses a cache (somewhere?) so that all
# the data is not in memory.

# this is horrible because it assumes a tokenizer has already been declared
# I had to fix it
def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    fn_kwargs={"tokenizer": bert_tokenizer},
    batched=True)

data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

# simple example
# samples = tokenized_datasets["train"][:8]
# samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

# batch = data_collator(samples)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=bert_tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.create_model_card()
