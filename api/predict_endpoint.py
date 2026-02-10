from fastapi import FastAPI
from fastapi.responses import JSONResponse
from api.schemas import PredictRequest, PredictResponse
from models.predict import predict_sentiment
from models.loader import load_model  
app = FastAPI(title="Sentiment Inference API")

vectorizer,clf = load_model()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_res(req: PredictRequest):
    sentiment, confidence = predict_sentiment(
        vectorizer=vectorizer,
        clf=clf,
        raw_text=req.transcript
    )

    return JSONResponse(status_code = 200,content = {"prediction":{"sentiment": sentiment,"confidence": round(confidence,2)}  })
