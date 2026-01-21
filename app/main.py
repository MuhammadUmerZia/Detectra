from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import re
from collections import Counter

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    text: str

# Load models - CHOOSE ONE BASED ON YOUR ACTUAL FILES:
# Check what .pkl files you have in models/ folder
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/logreg_baseline.pkl")  # Default to Logistic Regression

# You can switch to LinearSVM if you have it:
# model = joblib.load("models/svm_model.pkl")  # Uncomment if you have this file

def preprocess_text(text):
    """Simple preprocessing to match training"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)      # Remove extra spaces
    return text.strip()

def get_text_features(text):
    """Extract simple features to help with edge cases"""
    words = text.split()
    sentences = text.split('.')
    
    features = {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if len(s.strip()) > 0]),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'avg_sentence_length': len(words)/max(len(sentences), 1),
        'unique_word_ratio': len(set(words))/max(len(words), 1)
    }
    return features

def adjust_probability_based_on_features(text, base_prob, features):
    """Adjust probability based on text features to reduce FPs/FNs"""
    
    adjusted_prob = base_prob
    
    # Medical/technical texts are often human but predicted as AI
    medical_terms = ['patient', 'diagnosis', 'prescribed', 'treatment', 'symptoms']
    legal_terms = ['contract', 'clause', 'legal', 'law', 'agreement']
    technical_terms = ['algorithm', 'system', 'model', 'data', 'analysis']
    
    text_lower = text.lower()
    
    # If text contains medical/legal terms but probability is high for AI
    # It might be a false positive (human text misclassified as AI)
    has_medical = any(term in text_lower for term in medical_terms)
    has_legal = any(term in text_lower for term in legal_terms)
    has_technical = any(term in text_lower for term in technical_terms)
    
    if (has_medical or has_legal or has_technical) and base_prob > 0.7:
        # Reduce AI probability for technical texts
        adjusted_prob = base_prob * 0.7
    
    # Very short or very repetitive text might be human but predicted as AI
    if features['word_count'] < 100 and features['unique_word_ratio'] < 0.3:
        if base_prob > 0.8:
            adjusted_prob = base_prob * 0.6
    
    # Very long, diverse text with high AI probability might be correct
    if features['word_count'] > 200 and features['unique_word_ratio'] > 0.7:
        if base_prob > 0.8:
            adjusted_prob = min(base_prob * 1.1, 0.99)  # Slight boost
    
    return max(0.01, min(0.99, adjusted_prob))  # Keep between 0.01 and 0.99

@app.post("/detect")
def detect(payload: Payload):
    # Preprocess text
    processed_text = preprocess_text(payload.text)
    
    # Get text features
    features = get_text_features(processed_text)
    
    # Transform using TF-IDF
    X = tfidf.transform([processed_text])
    
    # Get base prediction
    if hasattr(model, 'predict_proba'):
        base_proba = float(model.predict_proba(X)[0, 1])  # Probability of class 1 (AI)
    elif hasattr(model, 'decision_function'):
        # For LinearSVM
        decision_score = float(model.decision_function(X)[0])
        base_proba = 1 / (1 + np.exp(-decision_score))  # Convert to probability
    else:
        # For RidgeClassifier or others
        prediction = model.predict(X)[0]
        base_proba = 0.8 if prediction == 1 else 0.2
    
    # Adjust probability based on features
    final_proba = adjust_probability_based_on_features(payload.text, base_proba, features)
    
    # Final label
    label = int(final_proba >= 0.5)
    
    # Get top words for explainability
    feature_names = np.array(tfidf.get_feature_names_out())
    
    try:
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0]
            X_array = X.toarray()[0]
            present_indices = np.where(X_array > 0)[0]
            
            if len(present_indices) > 0:
                # Sort by absolute coefficient value
                top_indices = present_indices[np.argsort(np.abs(coefs[present_indices]))[-5:]][::-1]
                top_words = feature_names[top_indices].tolist()
            else:
                top_words = get_common_words(payload.text)
        else:
            top_words = get_common_words(payload.text)
    except:
        top_words = get_common_words(payload.text)
    
    # Add feature info for debugging
    debug_info = {
        "base_probability": round(base_proba, 4),
        "adjusted_probability": round(final_proba, 4),
        "word_count": features['word_count'],
        "sentence_count": features['sentence_count'],
        "unique_word_ratio": round(features['unique_word_ratio'], 3)
    }
    
    return {
        "label": label,
        "probability": final_proba,
        "top_words": top_words,
        "debug": debug_info,  # Optional: remove in production
        "model_used": model.__class__.__name__
    }

def get_common_words(text, n=5):
    """Fallback to get most common words"""
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = [word for word, count in Counter(words).most_common(n)]
    return common_words if common_words else ["analysis", "complete"]

@app.get("/health")
def health():
    return {"status": "ok", "model": model.__class__.__name__}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)