import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib   # ✅ umesto pickle
import os

# === PODESI PUTANJE DO MODELA ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIC_MODELS_DIR = os.path.join(BASE_DIR, "../models")
TRANSFORMER_MODEL_DIR = os.path.join(BASE_DIR, "../models/transformers/distilbert_finetuned")

# === UČITAVANJE KLASIČNIH MODELA ===
def load_classic_model(model_name):
    model_path = os.path.join(CLASSIC_MODELS_DIR, f"{model_name}.pkl")
    vectorizer_path = os.path.join(CLASSIC_MODELS_DIR, "tfidf_vectorizer_1.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# === UČITAVANJE TRANSFORMER MODELA ===
def load_transformer_model():
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
    return model, tokenizer

# === PREDIKCIJE ===
def predict_classic(model, vectorizer, text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred

def predict_transformer(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

# === STREAMLIT UI ===
st.set_page_config(page_title="Spoiler Detector", page_icon="🎬")
st.title("🎬 Spoiler Detector App")

model_type = st.selectbox("Odaberi model:", ["Naive Bayes", "Logistic Regression", "DistilBERT"])

user_text = st.text_area("Unesi recenziju ovde:", height=200)

if st.button("Predvidi"):
    if not user_text.strip():
        st.warning("⚠️ Molim te unesi tekst recenzije.")
    else:
        with st.spinner("🔍 Analiziram..."):
            if model_type in ["Naive Bayes", "Logistic Regression"]:
                model, vectorizer  = load_classic_model(
                    "naive_bayes_1" if model_type == "Naive Bayes" else "log_reg_1"
                )
                pred = predict_classic(model, vectorizer, user_text)
            else:
                model, tokenizer = load_transformer_model()
                pred = predict_transformer(model, tokenizer, user_text)

        st.success("✅ Analiza završena!")
        if pred == 1:
            st.error("🚨 Ovo je SPOILER!")
        else:
            st.info("🟢 Ovo NIJE spoiler.")