import os
from config.settings import VECTORIZER_PATH, MODEL_PATH
import joblib 


def load_model():
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
        return joblib.load(VECTORIZER_PATH), joblib.load(MODEL_PATH)
    return None