import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers_model import train_transformer_model
from evaluate import evaluate_model_transformer
from preprocessing import clean_text_transformer
from transformer_config import TRANSFORMER_MODELS

# --- Dataset priprema ---
raw_path = "../data/IMDB_reviews.json"
cleaned_path = "../data/IMDB_reviews_transformer_cleaned.csv"

if os.path.exists(cleaned_path):
    df = pd.read_csv(cleaned_path)
    print("Učitano očišćeno za transformer!")
else:
    df = pd.read_json(raw_path, lines=True)
    print(f"Učitano {len(df)} recenzija iz originalnog JSON-a")
    df["clean_review"] = df["review_text"].apply(clean_text_transformer)
    df.to_csv(cleaned_path, index=False)
    print(f"Sačuvan očišćeni dataset u {cleaned_path}")

# uzorak radi brzine
df = df.sample(n=5000, random_state=42).reset_index(drop=True)
print(f"Koristim {len(df)} recenzija za trening i evaluaciju.")

# --- Split podataka ---
X = df["clean_review"]
y = df["is_spoiler"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# --- Treniraj više modela ---
results = []

for model_key in ["distilbert", "bert", "roberta"]:
    print("=" * 60)
    print(f"🚀 Treniranje modela: {model_key}")
    print("=" * 60)

    model, tokenizer = train_transformer_model(
        X_train.tolist(), y_train.tolist(),
        X_val.tolist(), y_val.tolist(),
        model_key=model_key
    )

    metrics = evaluate_model_transformer(model, tokenizer, X_test, y_test)
    results.append({"model": model_key, **metrics})

# --- Rezime rezultata ---
results_df = pd.DataFrame(results)
print("\nRezultati svih modela:")
print(results_df)

best = results_df.sort_values(by="f1", ascending=False).iloc[0]
print(f"\nNajbolji model: {best['model']} (F1={best['f1']:.4f})")

results_df.to_csv("../results/transformer_comparison.csv", index=False)
print("\nRezultati sačuvani u ../results/transformer_comparison.csv")
