import numpy as np

def predict_sentiment(vectorizer, clf, raw_text: str):
    if not raw_text or not isinstance(raw_text, str):
        return "neutral", 0.5
    X = vectorizer.transform([raw_text])
    pred = clf.predict(X)[0]
    score = float(np.max(clf.predict_proba(X)))
    return pred, score
