"""
FastAPI backend for the CIFAKE real-vs-AI-generated image detector.

Serves:
  GET  /            -> the frontend (static/index.html)
  POST /predict      -> {label: "REAL"|"FAKE", confidence: float} for an uploaded image
  GET  /health       -> simple health check for the hosting platform
"""

import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "cifake_cnn.h5")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="CIFAKE Detector API")

# Allow the frontend to call this API even if it's ever hosted separately.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None  # loaded lazily so the app can still boot if the file is missing


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Model file not found at app/saved_model/cifake_cnn.h5. "
                       "Train it (see model/train_model.py) and add it to the repo.",
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    model = get_model()

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize((32, 32))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image.")

    arr = np.asarray(image, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)

    prob_real = float(model.predict(arr, verbose=0)[0][0])  # 0=FAKE, 1=REAL
    is_real = prob_real >= 0.5
    confidence = prob_real if is_real else (1.0 - prob_real)

    return {
        "label": "REAL" if is_real else "FAKE",
        "confidence": round(confidence * 100, 2),
    }


# Serve the frontend. Mounted last so it doesn't shadow the API routes above.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
