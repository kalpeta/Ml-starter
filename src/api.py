from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import json

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ART = Path("artifacts")
MODEL_PATH = ART / "model.joblib"
META_PATH = ART / "meta.json"

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    probability: float
    prediction: int

model = None
expected_n_features: Optional[int] = None
expected_threshold: Optional[float] = None

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python -m src.train_real")
    return joblib.load(MODEL_PATH)

def load_meta():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {META_PATH}. Run: python -m src.train_real")
    meta = json.loads(META_PATH.read_text())
    n = int(meta["n_features"])
    t = float(meta.get("threshold", 0.5))
    return n, t

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, expected_n_features, expected_threshold
    model = load_model()
    expected_n_features, expected_threshold = load_meta()
    yield

app = FastAPI(title="ML Starter API", version="0.2.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global model, expected_n_features, expected_threshold

    if model is None or expected_n_features is None or expected_threshold is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(req.features) != expected_n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_n_features} features, got {len(req.features)}"
        )

    x = np.array(req.features, dtype=float).reshape(1, -1)
    prob = float(model.predict_proba(x)[0, 1])
    pred = int(prob >= float(expected_threshold))
    return PredictResponse(probability=prob, prediction=pred)
