from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load the NLI model
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Input sentence (the "premise")
premise = "This restaurant has amazing pasta."

# Candidate labels
candidate_labels = ["food", "service", "ambience"]

# Hypothesis template
template = "This text is about {}."

# Convert labels into hypotheses
hypotheses = [template.format(label) for label in candidate_labels]

# Tokenize each (premise, hypothesis) pair
inputs = tokenizer([premise] * len(hypotheses), hypotheses, return_tensors="pt", truncation=True, padding=True)

# Get model logits
with torch.no_grad():
    logits = model(**inputs).logits

# The logits are for [contradiction, neutral, entailment]
entailment_index = model.config.label2id["entailment"]

# Extract the entailment logits
entailment_logits = logits[:, entailment_index]

# Normalize across labels
label_probs = F.softmax(entailment_logits, dim=0)

# Show results
for label, score in zip(candidate_labels, label_probs):
    print(f"{label}: {score:.4f}")
