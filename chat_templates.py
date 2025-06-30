from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")

smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"},
# ]
# smol_chat = smol_tokenizer.apply_chat_template(messages, tokenize=False)

def convert_to_chatml(example):
    messages = example['messages']
    return {'messages': messages}

training_dataset = dataset['train'].map(convert_to_chatml, batched=True)

