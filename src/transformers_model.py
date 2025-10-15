from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
from transformer_config import TRANSFORMER_MODELS
import torch
import os

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(pred.label_ids, preds, average='binary')
    acc = accuracy_score(pred.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train_transformer_model(X_train, y_train, X_val, y_val, model_key="distilbert"):
    cfg = TRANSFORMER_MODELS[model_key]
    model_name = cfg["name"]
    print(f"\n🔧 Fine-tuning modela: {model_name}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=cfg["max_length"]
        )

    train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
    val_dataset = Dataset.from_dict({"text": X_val, "label": y_val})
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    output_dir = f"../models/{model_key}_output"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        num_train_epochs=cfg["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir=f"../logs/{model_key}",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"../models/{model_key}_finetuned")
    print(f"✅ Model '{model_key}' sačuvan u ../models/{model_key}_finetuned\n")

    return model, tokenizer