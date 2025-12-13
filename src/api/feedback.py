"""
Feedback Collection API
Collects user feedback for model improvement
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Handle imports
try:
    from ..utils.config import get_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config import get_config


# Initialize config
config = get_config()

# Create FastAPI app
app = FastAPI(
    title="Feedback Collection API",
    description="API untuk mengumpulkan feedback pengguna",
    version="1.0.0",
)


# Request/Response models
class FeedbackRequest(BaseModel):
    """Request model for feedback submission"""

    audio_filename: str = Field(..., description="Nama file audio yang diprediksi")
    predicted_label: str = Field(..., description="Hasil prediksi model")
    actual_label: str = Field(..., description="Label yang benar menurut user")
    model_type: str = Field(..., description="Model yang digunakan")
    confidence: float = Field(..., description="Confidence score prediksi")
    user_comment: Optional[str] = Field(None, description="Komentar tambahan user")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""

    status: str
    message: str
    feedback_id: str
    total_feedback: int
    threshold: int
    should_retrain: bool


class FeedbackStatsResponse(BaseModel):
    """Response model for feedback statistics"""

    total_feedback: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float
    by_model: dict
    threshold: int
    should_retrain: bool


def get_feedback_dir() -> Path:
    """Get feedback directory path"""
    feedback_dir = Path(os.getenv("FEEDBACK_DATA_DIR", "data/feedback"))
    feedback_dir.mkdir(parents=True, exist_ok=True)
    return feedback_dir


def get_feedback_file() -> Path:
    """Get feedback CSV file path"""
    return get_feedback_dir() / "feedback.csv"


def load_feedback() -> pd.DataFrame:
    """Load existing feedback data"""
    feedback_file = get_feedback_file()

    if feedback_file.exists():
        return pd.read_csv(feedback_file)
    else:
        # Create empty DataFrame with schema
        return pd.DataFrame(
            columns=[
                "feedback_id",
                "timestamp",
                "audio_filename",
                "predicted_label",
                "actual_label",
                "model_type",
                "confidence",
                "is_correct",
                "user_comment",
            ]
        )


def save_feedback(df: pd.DataFrame):
    """Save feedback data to CSV"""
    feedback_file = get_feedback_file()
    df.to_csv(feedback_file, index=False)


def get_feedback_threshold() -> int:
    """Get feedback threshold for auto-retrain"""
    return int(os.getenv("FEEDBACK_THRESHOLD", 20))


def check_should_retrain(total_feedback: int) -> bool:
    """Check if model should be retrained based on feedback count"""
    threshold = get_feedback_threshold()
    auto_retrain_enabled = os.getenv("AUTO_RETRAIN_ENABLED", "true").lower() == "true"

    return auto_retrain_enabled and total_feedback >= threshold


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback

    Args:
        feedback: Feedback data from user

    Returns:
        Feedback submission status and retrain trigger info
    """
    try:
        # Load existing feedback
        df = load_feedback()

        # Generate feedback ID
        feedback_id = f"FB_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Check if prediction was correct
        is_correct = feedback.predicted_label == feedback.actual_label

        # Create new feedback entry
        new_feedback = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "audio_filename": feedback.audio_filename,
            "predicted_label": feedback.predicted_label,
            "actual_label": feedback.actual_label,
            "model_type": feedback.model_type,
            "confidence": feedback.confidence,
            "is_correct": is_correct,
            "user_comment": feedback.user_comment or "",
        }

        # Append to DataFrame
        df = pd.concat([df, pd.DataFrame([new_feedback])], ignore_index=True)

        # Save to CSV
        save_feedback(df)

        # Get statistics
        total_feedback = len(df)
        should_retrain = check_should_retrain(total_feedback)
        threshold = get_feedback_threshold()

        print(f"✅ Feedback submitted: {feedback_id}")
        print(
            f"   Prediction: {feedback.predicted_label} → Actual: {feedback.actual_label}"
        )
        print(f"   Correct: {is_correct}")
        print(f"   Total feedback: {total_feedback}/{threshold}")

        if should_retrain:
            print(f"🔔 Feedback threshold reached! Auto-retrain should be triggered.")

        return {
            "status": "success",
            "message": "Feedback berhasil disimpan",
            "feedback_id": feedback_id,
            "total_feedback": total_feedback,
            "threshold": threshold,
            "should_retrain": should_retrain,
        }

    except Exception as e:
        print(f"❌ Error saving feedback: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats():
    """
    Get feedback statistics

    Returns:
        Statistics about collected feedback
    """
    try:
        df = load_feedback()

        if len(df) == 0:
            return {
                "total_feedback": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "accuracy": 0.0,
                "by_model": {},
                "threshold": get_feedback_threshold(),
                "should_retrain": False,
            }

        # Calculate statistics
        total_feedback = len(df)
        correct_predictions = int(df["is_correct"].sum())
        incorrect_predictions = total_feedback - correct_predictions
        accuracy = correct_predictions / total_feedback if total_feedback > 0 else 0.0

        # Statistics by model
        by_model = {}
        for model_type in df["model_type"].unique():
            model_df = df[df["model_type"] == model_type]
            by_model[model_type] = {
                "total": len(model_df),
                "correct": int(model_df["is_correct"].sum()),
                "accuracy": float(model_df["is_correct"].mean())
                if len(model_df) > 0
                else 0.0,
            }

        should_retrain = check_should_retrain(total_feedback)

        return {
            "total_feedback": total_feedback,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": incorrect_predictions,
            "accuracy": accuracy,
            "by_model": by_model,
            "threshold": get_feedback_threshold(),
            "should_retrain": should_retrain,
        }

    except Exception as e:
        print(f"❌ Error getting feedback stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.get("/feedback/list")
async def list_feedback(limit: int = 50):
    """
    List recent feedback entries

    Args:
        limit: Maximum number of entries to return

    Returns:
        List of recent feedback entries
    """
    try:
        df = load_feedback()

        # Sort by timestamp descending
        df = df.sort_values("timestamp", ascending=False)

        # Limit results
        df = df.head(limit)

        # Convert to dict
        feedback_list = df.to_dict("records")

        return {
            "total": len(load_feedback()),
            "returned": len(feedback_list),
            "feedback": feedback_list,
        }

    except Exception as e:
        print(f"❌ Error listing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing feedback: {str(e)}")


@app.delete("/feedback/clear")
async def clear_feedback(confirm: str = ""):
    """
    Clear all feedback data (admin only)

    Args:
        confirm: Must be "DELETE_ALL_FEEDBACK" to confirm

    Returns:
        Deletion status
    """
    if confirm != "DELETE_ALL_FEEDBACK":
        raise HTTPException(
            status_code=400,
            detail="Must provide confirmation: confirm=DELETE_ALL_FEEDBACK",
        )

    try:
        feedback_file = get_feedback_file()

        if feedback_file.exists():
            # Backup before deleting
            backup_file = (
                get_feedback_dir()
                / f"feedback_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            import shutil

            shutil.copy(feedback_file, backup_file)

            # Delete current file
            feedback_file.unlink()

            print(f"✅ Feedback cleared (backup saved to {backup_file})")

            return {
                "status": "success",
                "message": "Feedback data cleared",
                "backup_file": str(backup_file),
            }
        else:
            return {"status": "success", "message": "No feedback data to clear"}

    except Exception as e:
        print(f"❌ Error clearing feedback: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error clearing feedback: {str(e)}"
        )


# ============================================================================
# TESTING CODE
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("Starting Feedback Collection API")
    print("=" * 80)

    port = 8001  # Different port from main API

    print(f"\n🚀 Server starting on http://localhost:{port}")
    print(f"📖 API docs: http://localhost:{port}/docs")
    print(f"📊 Feedback stats: http://localhost:{port}/feedback/stats")

    uvicorn.run(
        "feedback:app", host="0.0.0.0", port=port, reload=True, log_level="info"
    )
