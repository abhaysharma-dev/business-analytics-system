# Business Analytics System

An end-to-end Business Analytics platform that performs **audio transcription, sentiment analysis, and analytics visualization**, fully containerized using Docker and deployed on AWS EC2.

This system converts audio into text using Whisper, analyzes sentiment using a trained Machine Learning model, stores results in MySQL, and displays analytics through an interactive Streamlit dashboard.

---

**Deployment Status**

- Deployed on AWS EC2 using Docker containers  
- Instance is stopped when not in use to avoid cloud charges  
- Live demo available on request  

---

## Tech Stack

- Python
- FastAPI (Backend API)
- Streamlit (Frontend Dashboard)
- Scikit-learn (TF-IDF + Logistic Regression)
- OpenAI Whisper (Speech-to-Text)
- MySQL (Database)
- Docker & Docker Compose (Containerization)
- AWS EC2 (Cloud Deployment)

---

## System Architecture

→ Audio Input  
→ Whisper Transcription  
→ Text Preprocessing  
→ TF-IDF Vectorization  
→ Logistic Regression Sentiment Prediction  
→ MySQL Database Storage  
→ Streamlit Analytics Dashboard  

---

## Features

- Audio transcription using Whisper
- Sentiment analysis using trained Machine Learning model
- REST API built with FastAPI
- Interactive analytics dashboard using Streamlit
- MySQL database integration
- Fully containerized using Docker
- Production deployment on AWS EC2
- Persistent database using Docker volumes

---

## Project Structure

```text

Business_Analytics_System/
│
├── api/        # FastAPI backend
├── models/     # ML model and vectorizer
├── database/   # Database related files
├── asr/        # Whisper transcription logic
├── nlp/        # Preprocessing logic
├── app.py      # Streamlit frontend
├── docker-compose.yml # Multi-container setup
├── Dockerfile  # Frontend container
├── requirements.txt
└── README.md

``` 

This modular design improves maintainability, scalability, and production readiness.

---

## Docker Services

The system runs using 3 Docker containers:

- frontend → Streamlit dashboard
- api → FastAPI ML inference service
- mysql → MySQL database

Managed using docker-compose.

---

## ⚙️ How to Run

### Build and start containers

```bash
docker compose up --build -d
```
### Stop Containers 

```bash
docker compose down
```

### Access the Application

- Frontend Dashboard:

http://localhost:8501


- API Endpoint:

http://localhost:8000

---

### AWS Deployment

This project is deployed on AWS EC2 using Docker.

### Deployment steps:

- Build Docker images
- Push images to Docker Hub
- Pull images on EC2
- Run using docker compose
- Access via EC2 public IP
Instance can be stopped when not in use to avoid charges.

---

## Machine Learning Model

Model: Logistic Regression  
Vectorizer: TF-IDF  
Library: Scikit-learn  

## Files:

- models/vectorizer.pkl  
- models/sentiment_model.pkl

---

### Key Highlights

- Production-ready Docker deployment
- Real-time ML inference using FastAPI
- End-to-end pipeline from audio to analytics
- Cloud deployment experience with AWS EC2
- Database integration with persistent storage

Notes:-
Actual datasets and audio recordings are not included due to privacy.

## Author

Abhay Sharma  
GitHub: https://github.com/abhaysharma-dev




