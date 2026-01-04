import spacy
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
NEGATIONS = {"not", "no", "never", "none", "nor", "cannot", "can't", "won't", "don't", "dont"}
base_stopwords = nlp.Defaults.stop_words
custom_stop_words = set(base_stopwords) - NEGATIONS

LEMMA_CORRECTIONS = {
    "interested": "interest",
    "considering": "consider",
    "considered": "consider",
    "fees": "fee",
    "payments": "payment",
    "courses": "course",
    "students": "student",
    "options": "option",
}

def spacy_lemmatizer_tokenizer(text: str):
    if not isinstance(text, str) or text.strip() == "":
        return []
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower().strip()
        if token.is_punct or token.like_num or not token.is_alpha or lemma == "-pron-":
            continue
        if lemma in NEGATIONS:
            tokens.append(lemma)
            continue
        if lemma in custom_stop_words:
            continue
        if lemma in LEMMA_CORRECTIONS:
            lemma = LEMMA_CORRECTIONS[lemma]
        tokens.append(lemma)
    return tokens
