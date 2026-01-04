
# Business Analytics System 🎧📊

A production-style Business Analytics System that performs:

- Audio call transcription using Whisper
- Sentiment analysis using TF-IDF + Logistic Regression
- HuggingFace fallback sentiment model
- Real-time (live) prediction on unseen calls
- Business analytics & insights
- MySQL database integration
- Streamlit-based interactive UI

---

## 🚀 Features

- Upload CSV call logs and audio recordings
- Automatic transcription (ASR)
- Sentiment prediction (batch + live)
- Cross-validation and accuracy evaluation
- Keyword-based negative sentiment analysis
- Business recommendations
- Export processed data

---

## 🏗️ Project Architecture

The project follows **separation of concerns**:

- `app.py` – Main Streamlit application
- `models/` – ML training, evaluation, prediction
- `nlp/` – Text preprocessing and NLP models
- `asr/` – Audio transcription (Whisper)
- `analytics/` – business insights
- `database/` – MySQL database operations
- `config/` – Configuration and constants

This modular design improves maintainability, scalability, and production readiness.

---

## Database credentials are loaded from environment variables using a `.env` file.

## ⚙️ How to Run

## 🔊 FFmpeg Requirement (Mandatory for Audio Transcription)

This project uses OpenAI Whisper for audio transcription, which requires **FFmpeg** to be installed on the system.

Install FFmpeg
for Windows:
Download FFmpeg from: https://ffmpeg.org/download.html
Extract the files
Add the bin folder to System PATH
Restart terminal

Notes:-

- Actual datasets and audio recordings are not included due to privacy.
- Models are generated at runtime.
- FFmpeg is required for audio transcription.

### Check if FFmpeg is installed
```bash
ffmpeg -version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run application
streamlit run app.py


