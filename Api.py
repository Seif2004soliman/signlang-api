import os
import cv2
import json
import base64
import logging
import tempfile
import argparse
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization, Multiply, Softmax
)
from tensorflow.keras.saving import register_keras_serializable

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 30
FEATURE_SIZE    = 258
SKIP_FACE       = True
TOP_K           = 5
MAX_FRAMES      = 300
MAX_UPLOAD_MB   = 50
NUM_POSE = 33 * 4
NUM_FACE = 468 * 3
NUM_HAND = 21 * 3
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

mp_holistic = mp.solutions.holistic
_model     = None
_idx2label = None

@register_keras_serializable()
class ReduceSumLayer(layers.Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

def build_model(num_classes: int) -> Model:
    """
    Architecture مطابقة للـ weights:
    - BN بعد كل BiLSTM (size=256)
    - dense attention: kernel فقط بدون bias (128,1)
    - dense_1: (128,256)+bias
    - dense_2: (256,128)+bias
    - dense_3: (128,num_classes)+bias
    """
    inp = Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE))

    # BiLSTM 1 → output: (batch, 30, 256)
    x = BatchNormalization()(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)

    # BiLSTM 2 → output: (batch, 30, 256)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)

    # BiLSTM 3 → output: (batch, 30, 128)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # Attention: Dense(1, use_bias=False) → Softmax → Multiply → ReduceSum
    score = Dense(1, use_bias=False, activation='tanh')(x)   # (batch,30,1)
    score = Softmax(axis=1)(score)
    x = Multiply()([x, score])
    x = ReduceSumLayer()(x)   # (batch, 128)

    # Classifier
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inp, out)

def extract_keypoints(results) -> np.ndarray:
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(NUM_POSE)
    )
    face = np.array([]) if SKIP_FACE else (
        np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(NUM_FACE)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(NUM_HAND)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(NUM_HAND)
    )
    return np.concatenate([pose, face, lh, rh])

def _normalize_sequence(frames_kp: list) -> np.ndarray:
    seq = np.array(frames_kp, dtype=np.float32)
    T = seq.shape[0]
    if T >= SEQUENCE_LENGTH:
        start = (T - SEQUENCE_LENGTH) // 2
        seq = seq[start: start + SEQUENCE_LENGTH]
    else:
        pad = np.zeros((SEQUENCE_LENGTH - T, FEATURE_SIZE), dtype=np.float32)
        seq = np.vstack([seq, pad])
    return seq[np.newaxis, ...]

def video_to_sequence(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames_kp = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as h:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            frames_kp.append(extract_keypoints(h.process(image)))
    cap.release()
    if not frames_kp:
        raise ValueError("No frames extracted from video")
    return _normalize_sequence(frames_kp)

def frames_b64_to_sequence(frames_b64: List[str]) -> np.ndarray:
    frames_kp = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as h:
        for idx, b64 in enumerate(frames_b64):
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(b64)
            except Exception:
                continue
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            frames_kp.append(extract_keypoints(h.process(image)))
    if not frames_kp:
        raise ValueError("No valid frames decoded")
    return _normalize_sequence(frames_kp)

def run_inference(seq: np.ndarray) -> list:
    probs = _model.predict(seq, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:TOP_K]
    return [
        {"rank": int(i+1), "label": _idx2label[str(idx)], "confidence": round(float(probs[idx])*100, 2)}
        for i, idx in enumerate(top_indices)
    ]

def _build_response(predictions: list) -> dict:
    return {"top_predictions": predictions, "best_label": predictions[0]["label"], "best_confidence": predictions[0]["confidence"]}

def load_model_and_labels(model_dir: str) -> None:
    global _model, _idx2label

    model_path = os.path.join(model_dir, "best_model.keras")
    label_path = os.path.join(model_dir, "label_map.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label map not found: {label_path}")

    with open(label_path, encoding="utf-8") as f:
        _idx2label = json.load(f)
    num_classes = len(_idx2label)
    logger.info(f"Labels loaded: {num_classes} classes")

    logger.info("Building model architecture ...")
    _model = build_model(num_classes)

    logger.info(f"Loading weights from {model_path} ...")
    _model.load_weights(model_path)
    logger.info("Model loaded OK")

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.environ.get("MODEL_DIR", r"D:\Project\model")
    load_model_and_labels(model_dir)
    yield
    logger.info("Server shutting down")

app = FastAPI(title="Sign Language Recognition API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"422 Validation Error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "hint": "Send JSON with key 'frames'"})

class FramesRequest(BaseModel):
    frames: List[str] = Field(..., min_length=5, max_length=MAX_FRAMES)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "num_classes": len(_idx2label) if _idx2label else 0}

@app.get("/labels")
def get_labels():
    if _idx2label is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"labels": list(_idx2label.values())}

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    suffix = os.path.splitext(file.filename or "")[-1].lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_VIDEO_EXTENSIONS}")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_UPLOAD_MB}MB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        predictions = run_inference(video_to_sequence(tmp_path))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Unexpected error processing video")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        os.unlink(tmp_path)
    return _build_response(predictions)

@app.post("/predict_frames")
async def predict_frames(body: FramesRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        predictions = run_inference(frames_b64_to_sequence(body.frames))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Unexpected error processing frames")
        raise HTTPException(status_code=500, detail="Internal server error")
    return _build_response(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=os.environ.get("MODEL_DIR", r"D:\Project\model"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    os.environ["MODEL_DIR"] = args.model_dir
    uvicorn.run(app, host=args.host, port=args.port)