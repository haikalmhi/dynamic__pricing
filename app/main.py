from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version


app = FastAPI()


class FeaturesIn(BaseModel):
    features: List[float]

class PredictionOut(BaseModel):
    predicted_price: float

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: FeaturesIn):
    if len(payload.features) != 13:
        raise HTTPException(status_code=400, detail="The number of features must be 13")
    
    predicted_price = predict_pipeline(payload.features)
    return {"predicted_price": predicted_price}