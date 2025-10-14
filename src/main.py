import joblib
import pandas as pd
from preprocessing import clean_text_classical
from sklearn.model_selection import train_test_split
from classical import train_classical_models
from evaluate import evaluate_model_classical
import os

# učitaj dataset
df = pd.read_json("../data/IMDB_reviews.json", lines=True)

# filtriraj recenzije koje imaju manje od 10 reči
# df['word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))
# df = df[df['word_count'] >= 10].reset_index(drop=True)
# print(f"Dataset nakon filtriranja: {len(df)} recenzija ostalo.")

# putanja gde ćemo sačuvati očišćeni dataset
cleaned_path = "../data/IMDB_reviews_cleaned.csv"

if os.path.exists(cleaned_path):
    # ako već postoji očišćeni dataset, samo učitaj
    df = pd.read_csv(cleaned_path)
    print("Učitano očišćeno!")
else:
    # kreiraj novu kolonu sa očišćenim tekstom
    df['clean_review'] = df['review_text'].apply(clean_text_classical)

    # sačuvaj očišćeni dataset za buduće korišćenje
    df.to_csv(cleaned_path, index=False)
    print("Očišćeni dataset sačuvan!")

# proveri rezultate
# print(df[['review_text', 'clean_review']].head())

X = df['clean_review']  # očišćeni tekst
y = df['is_spoiler']    # ciljni atribut (0/1)

# 70% train, 15% validation, 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val)
# 0.1765 * 0.85 ≈ 0.15 ukupno za validation


# print("\n--- Treniranje klasičnih modela ---")
# nb_model, lr_model, vectorizer = train_classical_models(X_train, X_val, y_train, y_val)

# print("\n--- Evaluacija sa već istreniranim modelima ---")
nb_model = joblib.load("../models/naive_bayes.pkl")
lr_model = joblib.load("../models/log_reg.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

nb_model1 = joblib.load("../models/naive_bayes_1.pkl")
lr_model1 = joblib.load("../models/log_reg_1.pkl")
vectorizer1 = joblib.load("../models/tfidf_vectorizer_1.pkl")

# evaluacija na test skupu
evaluate_model_classical(nb_model, X_test, y_test, vectorizer, "Naive Bayes")
evaluate_model_classical(lr_model, X_test, y_test, vectorizer, "Logistic Regression")

evaluate_model_classical(nb_model1, X_test, y_test, vectorizer1, "Naive Bayes")
evaluate_model_classical(lr_model1, X_test, y_test, vectorizer1, "Logistic Regression")