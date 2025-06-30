import evaluate
import numpy as np
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import(
    Dataset,
    DatasetDict,
    load_dataset,
)
from transformers import(
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    get_scheduler,
)
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

en_fr_datasets = load_dataset(
    "text", 
    data_files={
        "lang1": "./data/kde4/en-fr/KDE4.en-fr.en",
        "lang2": "./data/kde4/en-fr/KDE4.en-fr.fr"
    }    
)

en_dataset = en_fr_datasets['lang1']
fr_dataset = en_fr_datasets['lang2']

translations = [
            {"en": en_example["text"], "fr": fr_example["text"]}
            for en_example, fr_example in zip(en_dataset, fr_dataset)
        ]
ids = list(range(len(translations)))

raw_datasets = DatasetDict({
    "train": Dataset.from_dict({
        "id": ids,
        "translation": translations 
    })
})

split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")

# useful code example
# index = next(i for i, example in enumerate(split_datasets["train"]) if example["id"] == 69796)

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, max_length=128)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

max_length = 128

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

metric = evaluate.load("sacrebleu")

gen_config = GenerationConfig(
    max_length=128,
    num_beams=4,
    bad_words_ids=None
)

model.generation_config = gen_config

# this is a very dubious function provided by Huggingface
def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels

# Again with the damn login to huggingface and the Trainer class.  I am not using those.

batch_size = 16

# reduce size of datasets to speed results
train_dataloader = DataLoader(
    tokenized_datasets["train"], 
    # tokenized_datasets["train"].shuffle().select(range(180000)),
    # tokenized_datasets["train"].shuffle().select(range(12000)),
    shuffle=True, 
    batch_size=batch_size, 
    collate_fn=data_collator,
)
valid_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    # tokenized_datasets["validation"].shuffle().select(range(16000)), 
    # tokenized_datasets["validation"].shuffle().select(range(800)), 
    batch_size=batch_size, 
    collate_fn=data_collator,
)

optimizer = AdamW(model.parameters(), lr=2e-5)
# num_epochs = 3
num_epochs = 12
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
    model.train()
    loss_metric = 0.0
    train_bleu = 0.0
    valid_loss_metric = 0.0
    valid_bleu = 0.0
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

    # insert validation set here for now
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
                max_length=128,
                num_beams=4,
                decoder_start_token_id=model.config.decoder_start_token_id,
                bad_words_ids = None,
            )

            labels = batch["labels"]
            decoded_preds, decoded_labels = postprocess(generated, labels)
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    mean_loss = loss_metric/len(train_dataloader)
    valid_loss = valid_loss_metric/len(valid_dataloader)
    valid_bleu = metric.compute()['score']

    print(f'epoch {epoch+1}')
    print(f'train loss:{mean_loss:.4f} train bleu:{train_bleu:.4f}')
    print(f'valid loss:{valid_loss:.4f} valid bleu:{valid_bleu:.4f}')

save_path = "./models/marian-finetuned-kde4-en-to-fr"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
