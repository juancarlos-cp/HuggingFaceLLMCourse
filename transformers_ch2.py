import transformers

'''
# what am I getting out of this?
config = transformers.BertConfig()
model = transformers.BertModel(config)

# model is randomly initialized
# you must train it with gobs of data

print(config)

# this loads the weights for bert-base-cased
checkpoint = "bert-base-cased"
model = transformers.BertModel.from_pretrained(checkpoint)
tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

message = "Translating text to numbers is known as encoding. Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs."
tokens = tokenizer(message)
print(tokens)

decoded =tokenizer.decode(tokens['input_ids'])
print(decoded)

tokens = tokenizer.tokenize(message)
print(tokens)

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)
'''


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

model_inputs = tokenizer(sequence)
print(model_inputs['input_ids'])

input_ids = torch.tensor([ids])
# print(input_ids)

output = model(input_ids)
print(output.logits)

# output2 = model(torch.tensor([ids, ids]))
# print(output2.logits)

padding_id = tokenizer.pad_token_id
# print(padding_id)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
    ]

token_list = []
for r in raw_inputs:
    tokens = tokenizer.tokenize(r)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    token_list.append(torch.tensor(ids))

input_ids = torch.nn.utils.rnn.pad_sequence(
    token_list, 
    batch_first=True,
    padding_value=padding_id)

attention_mask = (input_ids != 0).long()

print(input_ids, attention_mask)

output = model(input_ids, attention_mask=attention_mask)
print(output.logits)
