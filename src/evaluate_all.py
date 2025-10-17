import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluate import evaluate_model_classical, evaluate_model_transformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================================================
#   KLASIČNI MODELI - koriste IMDB_reviews_cleaned.csv
# ============================================================

print("\n===== Evaluacija klasičnih modela =====")

df_classic = pd.read_csv("../data/IMDB_reviews_cleaned.csv")
X_classic = df_classic["clean_review"]
y_classic = df_classic["is_spoiler"]

X_train_val_c, X_test_c, y_train_val_c, y_test_c = train_test_split(
    X_classic, y_classic, test_size=0.15, stratify=y_classic, random_state=42
)
X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
    X_train_val_c, y_train_val_c, test_size=0.1765, stratify=y_train_val_c, random_state=42
)

nb_model = joblib.load("../models/naive_bayes_1.pkl")
lr_model = joblib.load("../models/log_reg_1.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer_1.pkl")

evaluate_model_classical(nb_model, X_test_c, y_test_c, vectorizer, "Naive Bayes")
evaluate_model_classical(lr_model, X_test_c, y_test_c, vectorizer, "Logistic Regression")


# ============================================================
#   TRANSFORMER MODELI - koriste IMDB_reviews_transformer_cleaned.csv

print("\n===== Evaluacija transformer modela =====")

df_trans = pd.read_csv("../data/IMDB_reviews_transformer_cleaned.csv")
df_trans = df_trans.sample(n=35000, random_state=42).reset_index(drop=True)
X_trans = df_trans["clean_review"]
y_trans = df_trans["is_spoiler"]

X_train_val_t, X_test_t, y_train_val_t, y_test_t = train_test_split(
    X_trans, y_trans, test_size=0.15, stratify=y_trans, random_state=42
)
X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(
    X_train_val_t, y_train_val_t, test_size=0.1765, stratify=y_train_val_t, random_state=42
)

MODEL_DIRS = {
    "distilbert": "../models/transformers/distilbert_finetuned",
}

results = []

for key, path in MODEL_DIRS.items():
    print(f"\n🔍 Evaluacija modela: {key}")
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    metrics = evaluate_model_transformer(model, tokenizer, X_test_t.tolist(), y_test_t.tolist())
    metrics["model"] = key
    results.append(metrics)


results_df = pd.DataFrame(results)
os.makedirs("../results", exist_ok=True)
results_df.to_csv("../results/evaluation_summary.csv", index=False)

print("\nRezime performansi:")
print(results_df)