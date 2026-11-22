

!pip install -q datasets==3.2.0

from datasets import load_dataset

ds = load_dataset("thainq107/abte-restaurants")

ds

ds['train'][0]

"""## **Tokenizer**"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = []
    labels = []

    for tokens, tags in zip(examples['Tokens'], examples['Tags']):
        bert_tokens = []
        bert_tags = []

        for i in range(len(tokens)):
          t = tokenizer.tokenize(tokens[i])
          bert_tokens += t
          bert_tags += [int(tags[i])] * len(t)

        bert_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        tokenized_inputs.append(bert_ids)
        labels.append(bert_tags)
    return {
            'input_ids': tokenized_inputs,
            'labels': labels
        }

preprocessed_ds = ds.map(tokenize_and_align_labels, batched=True)

preprocessed_ds['train'][0]

len(tokenizer)

"""## **Data Collator**"""

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

"""## **Evaluate**"""

!pip install -q seqeval==1.2.2

id2label = {
    0: "O",
    1: "B-Term",
    2: "I-Term"
}
label2id = {
    "O": 0,
    "B-Term": 1,
    "I-Term": 2
}

import numpy as np
from seqeval.metrics import accuracy_score, f1_score

def compute_metrics(p):
    predictions, labels = p # tuple (predictions, labels)
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = f1_score(true_labels, true_predictions)


    return {"F1-score": results}

"""## **Model**"""

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=3, id2label=id2label, label2id=label2id
)

model

"""## **Training**"""

# Disable wandb
import os
os.environ['WANDB_DISABLED'] = 'true'

# # Use wandb
# import wandb
# wandb.init(
#     project="aspect-based-term-extraction",
#     name="distilbert-base-uncased"
# )

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="abte-restaurants-distilbert-base-uncased",
    learning_rate=2e-5,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=80,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="F1-score",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed_ds["train"],
    eval_dataset=preprocessed_ds["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

"""## **Inference**"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the tokenizer (same as used in training)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Load the trained ATE model from the best checkpoint
# Replace "abte-restaurants-distilbert-base-uncased/checkpoint-XXXX" with the actual checkpoint directory (e.g., the one with the best F1-score)
model = AutoModelForTokenClassification.from_pretrained("/content/abte-restaurants-distilbert-base-uncased/checkpoint-1200")

# Define label mapping (based on PDF: 0=O, 1=B-Term, 2=I-Term)
id2label = {0: "O", 1: "B-Term", 2: "I-Term"}

# Pipeline for token classification with aggregation to group aspects
ate_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Function to extract aspects from predictions
def extract_aspects(text):
    predictions = ate_pipe(text)
    aspects = []
    for entity in predictions:
        if entity['entity_group'] == 'Term':  # Aggregated label for B/I-Term
            aspects.append(entity['word'])
    return aspects

# Example usage for inference
text = "The price was too high, but the cab was amazing."
aspects = extract_aspects(text)
print(f"Extracted aspects: {aspects}")
# Expected output: ['price', 'cab'] or similar based on model training

