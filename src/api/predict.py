"""
Prediction API for Gender Voice Detection
FastAPI backend for model inference
"""

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow import keras

# Handle imports
try:
    from ..preprocessing.audio_cleaner import AudioCleaner
    from ..preprocessing.feature_extractor import MFCCExtractor
    from ..utils.config import get_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.preprocessing.audio_cleaner import AudioCleaner
    from src.preprocessing.feature_extractor import MFCCExtractor
    from src.utils.config import get_config


# Initialize config
config = get_config()

# Create FastAPI app
app = FastAPI(
    title="Gender Voice Detection API",
    description="API untuk deteksi gender dari audio menggunakan Deep Learning",
    version="1.0.0",
)

# Add CORS middleware
cors_origins = os.getenv(
    "CORS_ORIGINS", '["http://localhost:7860", "http://localhost:3000"]'
)
if isinstance(cors_origins, str):
    import json

    cors_origins = json.loads(cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class PredictionResponse(BaseModel):
    """Response model for prediction"""

    prediction: str  # "Laki-laki" or "Perempuan"
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float]  # {"Laki-laki": 0.3, "Perempuan": 0.7}
    model_type: str  # "lstm", "rnn", or "gru"


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str
    message: str
    models_loaded: Dict[str, bool]


# Global model cache
_model_cache = {}


def load_model(model_type: str = "lstm") -> keras.Model:
    """
    Load trained model from disk with caching

    Args:
        model_type: Type of model to load ('lstm', 'rnn', 'gru')

    Returns:
        Loaded Keras model
    """
    model_type = model_type.lower()

    # Check cache first
    if model_type in _model_cache:
        return _model_cache[model_type]

    # Get model path
    model_path = os.getenv(
        f"MODEL_{model_type.upper()}_PATH", f"models/{model_type}_production.h5"
    )
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    print(f"📦 Loading {model_type.upper()} model from {model_path}...")
    model = keras.models.load_model(str(model_path))

    # Cache model
    _model_cache[model_type] = model

    print(f"✅ {model_type.upper()} model loaded successfully!")
    return model


def preload_models():
    """Preload all models at startup"""
    try:
        for model_type in ["lstm", "rnn", "gru"]:
            try:
                load_model(model_type)
            except FileNotFoundError:
                print(f"⚠️  {model_type.upper()} model not found, skipping...")
    except Exception as e:
        print(f"⚠️  Error preloading models: {str(e)}")


def process_audio_file(audio_data: bytes) -> np.ndarray:
    """
    Process uploaded audio file and extract MFCC features

    Args:
        audio_data: Raw audio file bytes

    Returns:
        MFCC features array (1, time_steps, n_mfcc)
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name

    try:
        # Initialize preprocessors
        cleaner = AudioCleaner()
        extractor = MFCCExtractor()

        # Extract MFCC features
        mfcc_features = extractor.extract_from_file(tmp_path, preprocess=True)

        # Add batch dimension
        mfcc_features = np.expand_dims(mfcc_features, axis=0)

        return mfcc_features

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("🚀 Starting Gender Voice Detection API...")
    preload_models()
    print("✅ API ready!")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    models_status = {}
    for model_type in ["lstm", "rnn", "gru"]:
        try:
            load_model(model_type)
            models_status[model_type] = True
        except:
            models_status[model_type] = False

    return {
        "status": "healthy",
        "message": "Gender Voice Detection API is running",
        "models_loaded": models_status,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await root()


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    model_type: str = Query("lstm", description="Model type: lstm, rnn, or gru"),
):
    """
    Predict gender from audio file

    Args:
        file: Uploaded audio file
        model_type: Type of model to use for prediction

    Returns:
        Prediction results with confidence scores
    """
    try:
        # Validate model type
        model_type = model_type.lower()
        if model_type not in ["lstm", "rnn", "gru"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type: {model_type}. Must be 'lstm', 'rnn', or 'gru'",
            )

        # Read audio file
        audio_data = await file.read()

        # Check file size
        max_size = int(os.getenv("MAX_UPLOAD_SIZE", 10485760))  # 10MB default
        if len(audio_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {max_size / 1024 / 1024:.1f}MB",
            )

        # Process audio and extract features
        print(f"🎵 Processing audio file: {file.filename}")
        mfcc_features = process_audio_file(audio_data)
        print(f"   MFCC shape: {mfcc_features.shape}")

        # Load model
        model = load_model(model_type)

        # Make prediction
        prediction_prob = model.predict(mfcc_features, verbose=0)[0][0]

        # Convert to binary prediction
        # 0 = Laki-laki, 1 = Perempuan
        is_perempuan = prediction_prob > 0.5
        prediction_label = "Perempuan" if is_perempuan else "Laki-laki"

        # Calculate confidence (distance from 0.5 threshold)
        confidence = abs(prediction_prob - 0.5) * 2  # Scale to 0-1

        # Prepare response
        response = {
            "prediction": prediction_label,
            "confidence": float(confidence),
            "probabilities": {
                "Laki-laki": float(1 - prediction_prob),
                "Perempuan": float(prediction_prob),
            },
            "model_type": model_type,
        }

        print(f"✅ Prediction: {prediction_label} (confidence: {confidence:.2%})")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models"""
    models = {}
    for model_type in ["lstm", "rnn", "gru"]:
        model_path = Path(f"models/{model_type}_production.h5")
        models[model_type] = {
            "available": model_path.exists(),
            "path": str(model_path),
            "loaded": model_type in _model_cache,
        }
    return {"models": models}


# ============================================================================
# TESTING CODE
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("Starting Gender Voice Detection API")
    print("=" * 80)

    # Get config
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    print(f"\n🚀 Server starting on http://{host}:{port}")
    print(f"📖 API docs: http://{host}:{port}/docs")
    print(f"📊 Health check: http://{host}:{port}/health")

    uvicorn.run("predict:app", host=host, port=port, reload=True, log_level="info")
