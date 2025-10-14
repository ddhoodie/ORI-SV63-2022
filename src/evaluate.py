# src/evaluate.py
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def evaluate_model_classical(model, X_test, y_test, vectorizer, model_name="Model"):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    print(f"\n=== Evaluacija za {model_name} ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

def evaluate_model_transformer(model, tokenizer, X_test, y_test, model_name="Transformer Model", batch_size=16, device=None):
    """
    Evaluacija transformer modela na test skupu.
    """
    print(f"\n=== Evaluacija za {model_name} ===")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    preds = []
    true_labels = y_test.tolist()

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_texts = X_test[i:i+batch_size].tolist()
            encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = model(**encodings)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)

    print(classification_report(true_labels, preds))

    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

