from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_hf_pipeline():
    # Fast, widely used binary sentiment model
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
