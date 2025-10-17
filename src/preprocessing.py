# src/preprocessing.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# (odkomentariši prvi put)
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_classical(text):
    """
    Osnovno i poboljšano čišćenje teksta:
    - mala slova
    - uklanjanje HTML tagova i specijalnih karaktera
    - tokenizacija
    - uklanjanje stopwords i reči kraćih od 3 slova
    - lematizacija
    """
    if not isinstance(text, str):
        return ""

    # u mala slova
    text = text.lower()

    # ukloni HTML tagove
    text = re.sub(r'<.*?>', ' ', text)

    # ukloni specijalne karaktere i brojeve
    text = re.sub(r'[^a-z\s]', '', text)

    # tokenizacija
    tokens = nltk.word_tokenize(text)

    # ukloni stopwords, kratke reči i lematizuj
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return ' '.join(tokens)

def clean_text_transformer(text):
    """
    Minimalno čišćenje teksta za transformer modele:
    - uklanja samo HTML tagove i višak razmaka
    - NE dira interpunkciju, velika slova, ni stop words
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r'<.*?>', ' ', text)  # ukloni HTML tagove
    text = re.sub(r'\s+', ' ', text).strip()  # višak razmaka

    return text