import whisper
import torch
import tempfile
import os
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    if whisper is None:
        raise RuntimeError("Whisper not installed. pip install openai-whisper and ensure ffmpeg is present.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    st.sidebar.success(f"Whisper will run on: {device.upper()}")
    return whisper.load_model(model_size, device=device)


def transcribe_with_whisper(audio_bytes: bytes, model, filename: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
    try:
        result = model.transcribe(path,language="en",
    task="transcribe")
        return result.get("text", "").strip()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
