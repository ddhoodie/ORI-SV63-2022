import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, \
    precision_recall_curve
from evaluate import evaluate_model_classical, evaluate_model_transformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import shap

# # =========================
# #  Učitavanje klasičnih modela
# # =========================
# df_classic = pd.read_csv("../data/IMDB_reviews_cleaned.csv")
# X_classic = df_classic["clean_review"]
# y_classic = df_classic["is_spoiler"]
#
# X_train_val_c, X_test_c, y_train_val_c, y_test_c = train_test_split(
#     X_classic, y_classic, test_size=0.15, stratify=y_classic, random_state=42
# )
# X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
#     X_train_val_c, y_train_val_c, test_size=0.1765, stratify=y_train_val_c, random_state=42
# )
#
# nb_model = joblib.load("../models/naive_bayes_1.pkl")
# lr_model = joblib.load("../models/log_reg_1.pkl")
# vectorizer = joblib.load("../models/tfidf_vectorizer_1.pkl")
#
# # Evaluacija klasičnih modela i confusion matrica
# evaluate_model_classical(nb_model, X_test_c, y_test_c, vectorizer, "Naive Bayes")
# evaluate_model_classical(lr_model, X_test_c, y_test_c, vectorizer, "Logistic Regression")
#
#
# # =========================
# # Feature importance za NB i LR
# # =========================
# def show_top_features(model, vectorizer, n=20, model_name="Model"):
#     features = vectorizer.get_feature_names_out()
#     if hasattr(model, "coef_"):  # Logistic Regression
#         coefs = model.coef_[0]
#         top_positive_idx = np.argsort(coefs)[-n:]
#         top_negative_idx = np.argsort(coefs)[:n]
#         plt.figure(figsize=(12, 5))
#         plt.barh(features[top_positive_idx], coefs[top_positive_idx], color='green')
#         plt.barh(features[top_negative_idx], coefs[top_negative_idx], color='red')
#         plt.title(f"Top {n} pozitivnih i negativnih reči - {model_name}")
#         plt.show()
#     elif hasattr(model, "feature_log_prob_"):  # Naive Bayes
#         log_probs = model.feature_log_prob_[1] - model.feature_log_prob_[0]
#         top_idx = np.argsort(log_probs)
#         top_positive_idx = top_idx[-n:]
#         top_negative_idx = top_idx[:n]
#         plt.figure(figsize=(12, 5))
#         plt.barh(features[top_positive_idx], log_probs[top_positive_idx], color='green')
#         plt.barh(features[top_negative_idx], log_probs[top_negative_idx], color='red')
#         plt.title(f"Top {n} reči koje utiču na predikciju - {model_name}")
#         plt.show()
#
#
# show_top_features(nb_model, vectorizer, model_name="Naive Bayes")
# show_top_features(lr_model, vectorizer, model_name="Logistic Regression")
#
# # =========================
# # Analiza FP i FN za Naive Bayes
# # =========================
# X_test_tfidf = vectorizer.transform(X_test_c)
# y_pred_nb = nb_model.predict(X_test_tfidf)
# fp_idx = np.where((y_pred_nb == 1) & (y_test_c.values == 0))[0]
# fn_idx = np.where((y_pred_nb == 0) & (y_test_c.values == 1))[0]
#
# print("Primeri FP (lažno pozitivni):")
# for i in fp_idx[:5]:
#     print(f"- {X_test_c.iloc[i][:100]}...")
#
# print("\nPrimeri FN (lažno negativni):")
# for i in fn_idx[:5]:
#     print(f"- {X_test_c.iloc[i][:100]}...")
#
# # =========================
# # ROC i Precision-Recall kurve za klasične modele
# # =========================
# from sklearn.preprocessing import label_binarize
#
#
# def plot_roc_pr(model, X_test, y_test, vectorizer, model_name="Model"):
#     X_test_tfidf = vectorizer.transform(X_test)
#     if hasattr(model, "predict_proba"):
#         y_score = model.predict_proba(X_test_tfidf)[:, 1]
#     else:
#         y_score = model.decision_function(X_test_tfidf)
#
#     fpr, tpr, _ = roc_curve(y_test, y_score)
#     roc_auc = auc(fpr, tpr)
#
#     precision, recall, _ = precision_recall_curve(y_test, y_score)
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
#     plt.plot([0, 1], [0, 1], '--', color='gray')
#     plt.title(f"ROC curve - {model_name}")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(recall, precision)
#     plt.title(f"Precision-Recall curve - {model_name}")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.show()
#
#
# plot_roc_pr(nb_model, X_test_c, y_test_c, vectorizer, "Naive Bayes")
# plot_roc_pr(lr_model, X_test_c, y_test_c, vectorizer, "Logistic Regression")

# =========================
# =========================
# Evaluacija i interpretacija DistilBERT
# =========================
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

distilbert_path = "../models/transformers/distilbert_finetuned"
tokenizer = AutoTokenizer.from_pretrained(distilbert_path)
model = AutoModelForSequenceClassification.from_pretrained(distilbert_path)

# Evaluacija i confusion matrica
evaluate_model_transformer(model, tokenizer, X_test_t.tolist(), y_test_t.tolist(), "DistilBERT")

# =========================
# Interpretabilnost pomoću SHAP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Uzmemo manji sample
sample_texts = [str(t) for t in X_test_t.dropna().tolist()[:50]]


# Kreiramo wrapper koji pretvara tekst u model input
def predict_proba(texts):
    # Uvek konvertuj u listu stringova
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray) or isinstance(texts, (list, tuple)):
        texts = [str(t) for t in texts]
    else:
        texts = [str(texts)]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs[:, 1].detach().cpu().numpy()


# Kreiramo masker za tekst
masker = shap.maskers.Text(tokenizer)

# Kreiramo explainer koji zna da koristi našu predikciju
explainer = shap.Explainer(predict_proba, masker)

# Računamo SHAP vrednosti
shap_values = explainer(sample_texts)

# Prikazujemo prvih 5 primera
shap.plots.text(shap_values[:5])
with open("../results/shap_explanation.html", "w", encoding="utf-8") as f:
    for i in range(5):
        f.write(f"<h2>Primer {i+1}</h2>")
        f.write(shap.plots.text(shap_values[i], display=False))
        f.write("<hr>")
print("Sačuvano u shap_explanation.html")