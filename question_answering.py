import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

def answer_question(question, context):
    offsets = tokenizer(question, context, return_tensors="pt")
    outputs = model(**offsets)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    sequence_ids = offsets.sequence_ids()
    # Mask everything apart from the tokens of the context
    mask = [i != 1 for i in sequence_ids]
    # Unmask the [CLS] token
    mask[0] = False
    mask = torch.tensor(mask)[None]

    start_logits[mask] = -10000
    end_logits[mask] = -10000

    start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
    end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

    scores = start_probabilities[:, None] * end_probabilities[None, :]
    scores = torch.triu(scores)
    max_index = scores.argmax().item()
    start_index = max_index // scores.shape[1]
    end_index = max_index % scores.shape[1]

    inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
    offsets = inputs_with_offsets["offset_mapping"]
    start_char, _ = offsets[start_index]
    _, end_char = offsets[end_index]
    answer = context[start_char:end_char]

    result = {
        "answer": answer,
        "start": start_char,
        "end": end_char,
        "score": scores[start_index, end_index],
    }

    return result

context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"

result = answer_question(question, context)

print(result)
