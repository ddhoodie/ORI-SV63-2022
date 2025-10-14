# src/classical.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib

def train_classical_models(X_train, X_val, y_train, y_val):
    print("🔧 Pokrećem TF-IDF i oversampling...")

    # 1 Oversampling da izjednačimo klase (ako je neuravnotežen dataset)
    ros = RandomOverSampler(random_state=42)
    X_train_df = X_train.to_frame(name="text")  # jer ROS očekuje DataFrame
    X_resampled, y_resampled = ros.fit_resample(X_train_df, y_train)
    X_train = X_resampled["text"]
    y_train = y_resampled

    # 2 TF-IDF vektorizacija (unapređena)
    vectorizer = TfidfVectorizer(
        max_features=50000,       # više reči = više konteksta
        ngram_range=(1, 3),       # unigrams, bigrams, trigrams
        min_df=3,                 # ignoriši reči koje se pojavljuju <3 puta
        sublinear_tf=True,        # smanjuje uticaj frekventnih reči
        lowercase=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # 3 Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_val_tfidf)

    print("\n=== Naive Bayes ===")
    print(classification_report(y_val, nb_pred, digits=3))

    # 4 Logistic Regression (optimizovana)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=2.0,                 # manji regularizacioni penal
        solver='liblinear',
        n_jobs=-1
    )
    lr.fit(X_train_tfidf, y_train)
    lr_pred = lr.predict(X_val_tfidf)

    print("\n=== Logistic Regression ===")
    print(classification_report(y_val, lr_pred, digits=3))

    # 5 Sačuvaj modele i vektorizator
    joblib.dump(vectorizer, "../models/tfidf_vectorizer_1.pkl")
    joblib.dump(nb, "../models/naive_bayes_1.pkl")
    joblib.dump(lr, "../models/log_reg_1.pkl")

    print("\nModeli i vektorizator sačuvani u /models/ direktorijum!")

    return nb, lr, vectorizer
