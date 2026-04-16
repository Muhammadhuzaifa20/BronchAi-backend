"""
app.py — FastAPI backend for BronchAI lung cancer classification.

Endpoints:
    GET  /              — Root (Railway default health check)
    GET  /api/health    — Detailed health check
    POST /api/predict   — Run ensemble prediction + Grad-CAM on a CT scan image
"""

import os
import io
import logging
import base64
import numpy as np
import cv2
import httpx
from contextlib import asynccontextmanager
from PIL import Image

import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from model_loader import model_manager, PREPROCESS_FUNCS, GRADCAM_LAYERS, IMG_SIZE
from gradcam import gradcam_to_base64

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# LIFESPAN — Lazy loading: models loaded on first request, not startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server lifespan. Models are loaded lazily on first prediction request."""
    import tensorflow as tf
    import keras
    logger.info(f"🚀 Starting BronchAI ML Backend... (TF: {tf.__version__}, Keras: {keras.__version__})")
    logger.info("📦 Models will be loaded on first prediction request (lazy loading).")
    yield
    logger.info("👋 Shutting down BronchAI ML Backend.")


# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="BronchAI ML Backend",
    description="Lung Cancer Classification with Ensemble Voting & Grad-CAM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow your React frontend to call this API
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", 
    "https://bronchai.netlify.app,https://bronchai.app,https://www.bronchai.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SCHEMAS
# ============================================================
class PredictRequest(BaseModel):
    image_url: str
    scan_id: Optional[str] = None


class ModelResult(BaseModel):
    name: str
    prediction: str
    confidence: float


class PredictResponse(BaseModel):
    prediction: str
    confidence_score: float
    models: list[ModelResult]
    gradcam_base64: Optional[str] = None
    original_image_base64: Optional[str] = None
    scan_id: Optional[str] = None


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
async def download_image_from_url(url: str) -> np.ndarray:
    """Download an image from a URL and return as numpy array (RGB, 224x224)."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

        content = response.content
        try:
            img = Image.open(io.BytesIO(content))
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            return np.array(img, dtype=np.uint8)
        except Exception:
            # Fallback for DICOM
            import pydicom
            dicom_data = pydicom.dcmread(io.BytesIO(content), force=True)
            img = dicom_data.pixel_array
            
            # Normalize to 0-255
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min()) * 255.0
            img = img.astype(np.uint8)
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
                
            img = cv2.resize(img, IMG_SIZE)
            return img

    except Exception as e:
        logger.error(f"Failed to download or parse image from {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download or parse image: {str(e)}")


def process_uploaded_file(file_bytes: bytes) -> np.ndarray:
    """Process an uploaded file and return as numpy array (RGB, 224x224)."""
    try:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            return np.array(img, dtype=np.uint8)
        except Exception:
            import pydicom
            dicom_data = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            img = dicom_data.pixel_array
            
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min()) * 255.0
            img = img.astype(np.uint8)
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
                
            img = cv2.resize(img, IMG_SIZE)
            return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def array_to_base64_png(img_array: np.ndarray) -> str:
    """Convert numpy array (RGB) to base64 PNG string for frontend display."""
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


def ensure_models_loaded():
    """Lazy-load models on first prediction request."""
    if not model_manager.loaded:
        logger.info("📥 First prediction request — loading models now...")
        model_manager.load_all()


def generate_gradcam_for_best_model(img_rgb: np.ndarray) -> Optional[str]:
    """
    Generate Grad-CAM using the best available transfer learning model.
    Priority: VGG16 > ResNet50 > EfficientNetB0
    (Sequential CNN doesn't have a clean conv layer for Grad-CAM)
    """
    for model_name in ["VGG16", "ResNet50", "EfficientNetB0"]:
        if model_name in model_manager.models and model_name in GRADCAM_LAYERS:
            model = model_manager.models[model_name]
            layer_name = GRADCAM_LAYERS[model_name]
            preprocess_fn = PREPROCESS_FUNCS[model_name]

            # Preprocess for this specific model
            preprocessed = preprocess_fn(np.copy(img_rgb).astype("float32"))
            input_batch = np.expand_dims(preprocessed, axis=0)

            result = gradcam_to_base64(img_rgb, model, input_batch, layer_name)
            if result:
                return result

    return None


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint — Railway default health check."""
    return {"status": "ok", "service": "BronchAI ML Backend"}


@app.get("/api/health")
async def health_check():
    """Detailed health check endpoint."""
    import tensorflow as tf
    import keras
    loaded_models = list(model_manager.models.keys())
    return {
        "status": "ok",
        "models_loaded": len(loaded_models),
        "model_names": loaded_models,
        "models_ready": model_manager.loaded,
        "tf_version": tf.__version__,
        "keras_version": keras.__version__
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict_from_url(request: PredictRequest):
    """
    Run ensemble prediction on a CT scan image from URL.

    The frontend sends the Supabase Storage URL of the uploaded scan.
    """
    # Lazy-load models on first request
    ensure_models_loaded()

    if not model_manager.models:
        raise HTTPException(status_code=503, detail="Models failed to load. Check server logs.")

    # 1. Download the image
    img_rgb = await download_image_from_url(request.image_url)

    # 2. Run ensemble prediction
    result = model_manager.predict(img_rgb)

    # 3. Generate Grad-CAM and Base64 Original Image
    gradcam_base64 = generate_gradcam_for_best_model(img_rgb)
    original_image_base64 = array_to_base64_png(img_rgb)

    return PredictResponse(
        prediction=result["prediction"],
        confidence_score=result["confidence_score"],
        models=result["models"],
        gradcam_base64=gradcam_base64,
        original_image_base64=original_image_base64,
        scan_id=request.scan_id,
    )


@app.post("/api/predict_stream")
async def predict_stream_from_url(request: PredictRequest):
    """
    Run ensemble prediction on a CT scan image from URL using Server-Sent Events (NDJSON streaming).
    """
    # Lazy-load models on first request
    ensure_models_loaded()

    if not model_manager.models:
        raise HTTPException(status_code=503, detail="Models failed to load. Check server logs.")

    async def event_generator():
        try:
            # 1. Download the image
            yield json.dumps({"event": "progress", "step": "Downloading image..."}) + "\n"
            img_rgb = await download_image_from_url(request.image_url)

            # 2. Run ensemble prediction and stream progress
            final_result = None
            for model_event in model_manager.predict_stream(img_rgb):
                if model_event.get("event") == "complete":
                    final_result = model_event["result"]
                else:
                    yield json.dumps(model_event) + "\n"

            # 3. Generate Grad-CAM and Base64 Original Image
            if final_result:
                yield json.dumps({"event": "progress", "step": "Grad-CAM Generation"}) + "\n"
                gradcam_base64 = generate_gradcam_for_best_model(img_rgb)
                original_image_base64 = array_to_base64_png(img_rgb)
                
                # Yield final payload
                yield json.dumps({
                    "event": "done",
                    "result": {
                        "prediction": final_result["prediction"],
                        "confidence_score": final_result["confidence_score"],
                        "models": final_result["models"],
                        "gradcam_base64": gradcam_base64,
                        "original_image_base64": original_image_base64,
                        "scan_id": request.scan_id,
                    }
                }) + "\n"
        except Exception as e:
            logger.error(f"Error during streaming prediction: {e}")
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/api/predict/upload", response_model=PredictResponse)
async def predict_from_upload(
    file: UploadFile = File(...),
    scan_id: Optional[str] = Form(None),
):
    """
    Run ensemble prediction on a directly uploaded CT scan image.
    Alternative to URL-based prediction for testing.
    """
    # Lazy-load models on first request
    ensure_models_loaded()

    if not model_manager.models:
        raise HTTPException(status_code=503, detail="Models failed to load. Check server logs.")

    # 1. Read uploaded file
    file_bytes = await file.read()
    img_rgb = process_uploaded_file(file_bytes)

    # 2. Run ensemble prediction
    result = model_manager.predict(img_rgb)

    # 3. Generate Grad-CAM and Base64 Original Image
    gradcam_base64 = generate_gradcam_for_best_model(img_rgb)
    original_image_base64 = array_to_base64_png(img_rgb)

    return PredictResponse(
        prediction=result["prediction"],
        confidence_score=result["confidence_score"],
        models=result["models"],
        gradcam_base64=gradcam_base64,
        original_image_base64=original_image_base64,
        scan_id=scan_id,
    )


# ============================================================
# RUN (for local development)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
