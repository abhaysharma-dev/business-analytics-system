import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "mysql"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "abhay"),
    "database": os.getenv("DB_NAME", "banalytics"),
}

# File names for model persistence

VECTORIZER_PATH = "models/vectorizer.pkl"
MODEL_PATH = "models/sentiment_model.pkl"
