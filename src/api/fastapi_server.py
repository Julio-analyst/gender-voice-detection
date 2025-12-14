"""
FastAPI Service for Gender Voice Detection
Provides REST API endpoints for model inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
import io
from typing import List, Dict

# Initialize FastAPI app
app = FastAPI(
    title="Gender Voice Detection API",
    description="MLOps-based API for gender classification from voice",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODELS_DIR = Path("models")
models = {}

def load_models():
    """Load all production models"""
    global models
    try:
        models["lstm"] = tf.keras.models.load_model(MODELS_DIR / "lstm_production.h5")
        models["rnn"] = tf.keras.models.load_model(MODELS_DIR / "rnn_production.h5")
        models["gru"] = tf.keras.models.load_model(MODELS_DIR / "gru_production.h5")
        print("✅ All models loaded successfully")
    except Exception as e:
        print(f"⚠️  Error loading models: {e}")

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()

# Request/Response models
class PredictionRequest(BaseModel):
    model_type: str = "lstm"

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_used: str
    probabilities: Dict[str, float]

# Preprocessing function
def extract_features(audio_data, sr):
    """Extract MFCC features from audio"""
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    
    # Pad or truncate to fixed length (94 frames)
    target_length = 94
    if mfccs.shape[1] < target_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, target_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :target_length]
    
    # Transpose to (time_steps, features)
    mfccs = mfccs.T
    
    # Add batch dimension
    mfccs = np.expand_dims(mfccs, axis=0)
    
    return mfccs

# Endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Gender Voice Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "models": "/models"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys())
    }

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": list(models.keys()),
        "default_model": "lstm"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_type: str = "lstm"
):
    """
    Predict gender from audio file
    
    Args:
        file: Audio file (wav, mp3, etc.)
        model_type: Model to use (lstm, rnn, gru)
    
    Returns:
        Prediction result with confidence scores
    """
    # Validate model type
    if model_type.lower() not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Available: {list(models.keys())}"
        )
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Load audio with librosa
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # Extract features
        features = extract_features(audio_data, sr)
        
        # Make prediction
        model = models[model_type.lower()]
        prediction = model.predict(features, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0]) if predicted_class == 1 else float(1 - prediction[0][0])
        
        # Map to gender labels
        gender = "Female" if predicted_class == 1 else "Male"
        
        # Prepare probabilities
        probabilities = {
            "Male": float(1 - prediction[0][0]),
            "Female": float(prediction[0][0])
        }
        
        return PredictionResponse(
            prediction=gender,
            confidence=confidence,
            model_used=model_type.lower(),
            probabilities=probabilities
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/batch_predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    model_type: str = "lstm"
):
    """
    Batch prediction for multiple audio files
    
    Args:
        files: List of audio files
        model_type: Model to use
    
    Returns:
        List of predictions
    """
    results = []
    
    for file in files:
        try:
            result = await predict(file, model_type)
            results.append({
                "filename": file.filename,
                "prediction": result.prediction,
                "confidence": result.confidence
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total": len(files)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
