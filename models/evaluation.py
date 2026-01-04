import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from nlp.preprocessing import spacy_lemmatizer_tokenizer

def get_cv_report(texts: pd.Series, labels: pd.Series, cv: int = 5) -> pd.DataFrame:
    y = labels.astype(str).str.lower().replace({
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
        "n": "negative",
        "p": "positive"
    })

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), tokenizer=spacy_lemmatizer_tokenizer),
        LogisticRegression(max_iter=200)
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
    }

    cv_results = cross_validate(
        model, texts, y, cv=cv, scoring=scoring, return_train_score=False
    )

    rows = []
    for metric in scoring.keys():
        values = cv_results[f"test_{metric}"]
        rows.append({
            "metric": metric,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
        })

    return pd.DataFrame(rows).set_index("metric")
