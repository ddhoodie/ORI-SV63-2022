import os
import pandas as pd
from preprocessing import clean_text_transformer
from sklearn.model_selection import train_test_split
from transformers_model import train_transformer_model
from evaluate import evaluate_model_transformer

# putanja do fajlova
raw_path = "../data/IMDB_reviews.json"
cleaned_path = "../data/IMDB_reviews_transformer_cleaned.csv"

# Učitaj i očisti (ako već nije sačuvan)
if os.path.exists(cleaned_path):
    df = pd.read_csv(cleaned_path)
    print("Učitano očišćeno za transformer!")
else:

    # učitaj originalni JSON
    df = pd.read_json(raw_path, lines=True)
    print(f"Učitano {len(df)} recenzija iz originalnog JSON-a")

    # minimalno očisti tekst (samo HTML i višak whitespace-a)
    df["clean_review"] = df["review_text"].apply(clean_text_transformer)

    # sačuvaj očišćeni dataset
    df.to_csv(cleaned_path, index=False)
    print(f"Transformer-cleaned dataset sačuvan u {cleaned_path}")

df = df.sample(n=5000, random_state=42).reset_index(drop=True)
print(f"Koristim {len(df)} recenzija za trening i evaluaciju.")

# Priprema podataka
X = df["clean_review"]
y = df["is_spoiler"]

# Train / Val / Test split (70 / 15 / 15)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Trenira transformer model (npr. DistilBERT, BERT-base, itd.)
model, tokenizer = train_transformer_model(
    X_train.tolist(),
    y_train.tolist(),
    X_val.tolist(),
    y_val.tolist()
)

# (opciono)
# evaluate_model_transformer(model, tokenizer, X_test, y_test)
