from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, pipeline
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].select(range(2000)),
    eval_dataset=tokenized["test"].select(range(500)),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate()
print("Evaluation results:", results)

model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

classifier = pipeline("sentiment-analysis", model="sentiment_model", tokenizer="sentiment_model")
print(classifier("The movie was absolutely amazing. I loved every part of it."))
