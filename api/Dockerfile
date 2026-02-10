FROM python:3.11-slim

WORKDIR /code

COPY api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "uvicorn","api.predict_endpoint:app","--host","0.0.0.0","--port","8000" ]