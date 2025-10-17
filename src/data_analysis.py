import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

DATA_PATH = "../data/IMDB_reviews.json"
CLEANED_PATH = "../data/IMDB_reviews_cleaned.csv"

# --- Učitavanje podataka ---
if os.path.exists(CLEANED_PATH):
    df = pd.read_csv(CLEANED_PATH)
    print("Učitano očišćeno!")
else:
    df = pd.read_json(DATA_PATH, lines=True)
    print(f"Učitano {len(df)} recenzija iz originalnog JSON-a")
    # Čišćenje teksta
    df['clean_review'] = df['review_text'].str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
    df.to_csv(CLEANED_PATH, index=False)
    print("Očišćeni dataset sačuvan!")

# --- Dodavanje kolone sa brojem reči ---
df['word_count'] = df['clean_review'].apply(lambda x: len(str(x).split()))

# --- Osnovna statistika ---
print("\n--- Osnovna statistika ---")
print(f"Ukupno recenzija: {len(df)}")
print("Broj spoiler / non-spoiler recenzija:")
print(df['is_spoiler'].value_counts())
print("Procenat po klasama:")
print(df['is_spoiler'].value_counts(normalize=True) * 100)

# --- Vizualizacija distribucije dužine recenzija ---
plt.figure(figsize=(8,4))
sns.histplot(df['word_count'], bins=50, kde=False)
plt.title("Distribucija dužine recenzija")
plt.xlabel("Broj reči")
plt.ylabel("Broj recenzija")
plt.tight_layout()
plt.show()

# --- Boxplot dužine po klasi ---
plt.figure(figsize=(6,4))
sns.boxplot(x='is_spoiler', y='word_count', data=df)
plt.title("Dužina recenzija po klasi (0=non-spoiler, 1=spoiler)")
plt.xlabel("Is Spoiler")
plt.ylabel("Broj reči")
plt.tight_layout()
plt.show()