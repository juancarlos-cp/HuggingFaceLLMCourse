import pandas as pd
import torch

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel

def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }

issues_dataset = load_dataset("lewtun/github-issues", split="train")

issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)

columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)

issues_dataset = issues_dataset.remove_columns(columns_to_remove)

issues_dataset.set_format("pandas")
issues_df = issues_dataset[:]
comments_df = issues_df.explode("comments", ignore_index=True)

comments_dataset = Dataset.from_pandas(comments_df)

# filter out short comments and concatenate all content
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
comments_dataset = comments_dataset.map(concatenate_text)

# best tokenizer for semantic search
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

device = torch.device("cuda")
model.to(device)

# this code is horrible - assumes device, tokenizer and model are globals
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

embeddings_dataset.add_faiss_index(column="embeddings")

# this seems fine for a standalone example, but how does one make a system out of this?

def query_dataset(question, k=5):
    question_embedding = get_embeddings([question]).cpu().detach().numpy()
    question_embedding.shape
    scores, samples = embeddings_dataset.get_nearest_examples(
        "embeddings", question_embedding, k=5
    )

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    for _, row in samples_df.iterrows():
        print(f"COMMENT: {row.comments}")
        print(f"SCORE: {row.scores}")
        print(f"TITLE: {row.title}")
        print(f"URL: {row.html_url}")
        print("=" * 50)
        print()

question = "How can I load a dataset offline?"
query_dataset(question, 5)
