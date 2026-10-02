import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_data = pd.read_csv(
    "dataset/test.csv",
    encoding="latin-1"
)

test_data.columns = ["label", "title", "description"]

# Remove header row if it was accidentally loaded as data
test_data = test_data[
    test_data["label"].astype(str).str.strip().str.lower() != "class index"
].copy()

# Convert labels to integers
test_data["label"] = test_data["label"].astype(int)

# ============================================================
# COMBINE TITLE + DESCRIPTION
# ============================================================

test_data["text"] = (
    test_data["title"].fillna("") +
    " " +
    test_data["description"].fillna("")
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

classifier = joblib.load(
    "news_classifier.pkl"
)

tfidf_vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)

print("Model loaded successfully.")


# ============================================================
# IMPORTANT:
# USE THE SAME PREPROCESSING AS TRAINING
# ============================================================

import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = str(text)

    # Case folding
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove punctuation and special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stop-word removal
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ============================================================
# PREPROCESS TEST DATA
# ============================================================

print("\nPreprocessing test data...")

test_data["clean_text"] = test_data["text"].apply(
    preprocess_text
)

print("Test preprocessing complete.")


# ============================================================
# TF-IDF TRANSFORMATION
# ============================================================

print("\nConverting test data into TF-IDF features...")

X_test = tfidf_vectorizer.transform(
    test_data["clean_text"]
)

y_test = test_data["label"]


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = classifier.predict(
    X_test
)

print("Predictions complete.")


# ============================================================
# LABEL NAMES
# ============================================================

label_names = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
]


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL ACCURACY")
print("=" * 70)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# DISPLAY CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_names,
    yticklabels=label_names
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("News Category Classifier - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")