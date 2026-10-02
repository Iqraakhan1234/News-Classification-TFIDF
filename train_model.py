import os
import re
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK data dependencies
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def main():
    # Load dataset (Make sure your train.csv has 'Class Index', 'Title', 'Description' or similar columns)
    print("Loading dataset...")
    if not os.path.exists("train.csv"):
        print("Error: train.csv not found in directory.")
        return

    df = pd.read_csv("train.csv", encoding="latin-1", header=None)
    # Assuming columns are: 0 -> Class Index, 1 -> Title, 2 -> Description
    # Clean up columns if headers exist
    if df.iloc[0, 0] == 'Class Index' or df.iloc[0, 0] == '1':
        df = df.iloc[1:].reset_index(drop=True)

    df['label'] = pd.to_numeric(df[0], errors='coerce')
    df['text'] = df[1].fillna('') + " " + df[2].fillna('')
    
    # Drop invalid rows
    df = df.dropna(subset=['label', 'text'])
    
    print("Preprocessing text data (this may take a minute)...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing text using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)

    print("Training classifier model...")
    classifier = LogisticRegression(max_iter=1000, C=1.0)
    classifier.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test_vec)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save artifacts
    print("Saving model and vectorizer...")
    joblib.dump(classifier, "news_classifier.pkl")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    print("Done! Files saved successfully.")

if __name__ == "__main__":
    main()