from flask import Flask, render_template, request
import joblib
import re
import nltk
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load Model & Vectorizer
classifier = joblib.load("news_classifier.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# NLTK resources setup
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

# Expanded category taxonomy mapping (Supports up to 8 classes)
label_names = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
    5: "Entertainment",
    6: "Health & Wellness",
    7: "Politics",
    8: "Environment & Climate"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    top_predictions = []
    word_count = 0
    title = ""
    news = ""

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        news = request.form.get("news", "").strip()
        combined_text = title + " " + news

        if combined_text.strip():
            word_count = len(combined_text.split())
            clean_text = preprocess_text(combined_text)
            text_vector = tfidf_vectorizer.transform([clean_text])

            predicted_label = classifier.predict(text_vector)[0]
            
            if predicted_label in label_names:
                prediction = label_names[predicted_label]
            else:
                prediction = f"Category {predicted_label}"

            # Calculate probabilities and top 3 matches
            if hasattr(classifier, "predict_proba"):
                probabilities = classifier.predict_proba(text_vector)[0]
                confidence = round(max(probabilities) * 100, 2)
                
                # Get class labels associated with the classifier
                classes = classifier.classes_
                top_indices = np.argsort(probabilities)[::-1][:3]
                
                top_predictions = []
                for idx in top_indices:
                    cls_label = classes[idx]
                    cat_name = label_names.get(cls_label, f"Class {cls_label}")
                    top_predictions.append({
                        "category": cat_name,
                        "probability": round(probabilities[idx] * 100, 2)
                    })

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        top_predictions=top_predictions,
        word_count=word_count,
        title=title,
        news=news
    )

if __name__ == "__main__":
    app.run(debug=True)