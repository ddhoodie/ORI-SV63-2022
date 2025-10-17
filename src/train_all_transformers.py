import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers_model import train_transformer_model
from evaluate import evaluate_model_transformer
from preprocessing import clean_text_transformer

BASE_DIR = "."
RESULTS_DIR = f"{BASE_DIR}/results"
MODEL_KEYS = ["distilbert", "bert", "roberta"]
DATA_DIR = f"{BASE_DIR}./data"
RAW_DATA_PATH = f"{DATA_DIR}/IMDB_reviews.json"
CLEANED_DATA_PATH = f"{DATA_DIR}/IMDB_reviews_transformer_cleaned.csv"

# --- Dataset priprema ---
if os.path.exists(CLEANED_DATA_PATH):
    df = pd.read_csv(CLEANED_DATA_PATH)
    print("Učitano očišćeno za transformer!")
else:
    df = pd.read_json(RAW_DATA_PATH, lines=True)
    print(f"Učitano {len(df)} recenzija iz originalnog JSON-a")
    df["clean_review"] = df["review_text"].apply(clean_text_transformer)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Sačuvan očišćeni dataset u {CLEANED_DATA_PATH}")

# Uzorak radi brzine
df = df.sample(n=5000, random_state=42).reset_index(drop=True)
print(f"Koristim {len(df)} recenzija za trening i evaluaciju.")

# Uravnoteziti dataset
from sklearn.utils import resample

df_majority = df[df.is_spoiler == 0]
df_minority = df[df.is_spoiler == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,                     # uzorkovati sa ponavljanjem
    n_samples=len(df_majority),       # izjednačiti broj uzoraka
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Dataset uravnotežen!")
print(df["is_spoiler"].value_counts())

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

# --- Treniranje ---
results = []

for model_key in MODEL_KEYS:
    print("=" * 60)
    print(f"🚀 Treniranje modela: {model_key}")
    print("=" * 60)

    model, tokenizer = train_transformer_model(
        X_train.tolist(), y_train.tolist(),
        X_val.tolist(), y_val.tolist(),
        model_key=model_key
    )

    metrics = evaluate_model_transformer(model, tokenizer, X_test.tolist(), y_test.tolist())
    results.append({"model": model_key, **metrics})

# --- Rezime rezultata ---
os.makedirs(RESULTS_DIR, exist_ok=True)
results_df = pd.DataFrame(results)
print("\nRezultati svih modela:")
print(results_df)

best = results_df.sort_values(by="f1", ascending=False).iloc[0]
print(f"\nNajbolji model: {best['model']} (F1={best['f1']:.4f})")

results_path = f"{RESULTS_DIR}/transformer_comparison.csv"
results_df.to_csv(results_path, index=False)
print(f"\nRezultati sačuvani u {results_path}")
