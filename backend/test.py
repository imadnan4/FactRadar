# First load required components
import joblib
import numpy as np
from scipy.sparse import hstack

# Load model and vectorizer
model = joblib.load("best_model_xgboost.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer_full.pkl")

# Text cleaning function
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Prediction function
def predict(text):
    cleaned = clean_text(text)
    text_tfidf = tfidf_vectorizer.transform([cleaned])
    dummy_features = np.zeros((1, len(feature_summary['feature_names'])))  # Need feature_summary
    features = hstack([text_tfidf, dummy_features])
    return model.predict(features)[0], model.predict_proba(features)[0]

# Example usage:
pred, proba = predict("Trump is the best president ever")
print(f"Prediction: {'FAKE' if pred == 1 else 'REAL'}")
print(f"Confidence: {max(proba):.3f}")
