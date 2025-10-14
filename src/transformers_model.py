# src/transformers_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
import torch

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(pred.label_ids, preds, average='binary')
    acc = accuracy_score(pred.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train_transformer_model(X_train, y_train, X_val, y_val, model_name="distilbert-base-uncased"):
    print(f"Fine-tuning {model_name}...")

    # 1 Tokenizacija
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    # 2 Kreiraj HuggingFace Dataset
    train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
    val_dataset = Dataset.from_dict({"text": X_val, "label": y_val})
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    # 3 Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4 Trening argumenti
    training_args = TrainingArguments(
        output_dir="../models/transformers_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir="../logs",
        logging_steps=50,
    )

    # 5 Trener
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("../models/distilbert_finetuned")

    print("Model sačuvan u ../models/distilbert_finetuned")
    return model, tokenizer
