FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

WORKDIR /code 

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]