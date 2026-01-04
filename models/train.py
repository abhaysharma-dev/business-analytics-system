from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nlp.preprocessing import spacy_lemmatizer_tokenizer

def train_final_model(texts, labels):
    y = labels.astype(str).str.lower().replace({
        "pos":"positive","neg":"negative","neu":"neutral","n":"negative","p":"positive"
    })
    vectorizer = TfidfVectorizer(ngram_range=(1,2), tokenizer=spacy_lemmatizer_tokenizer)
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    return vectorizer, clf
