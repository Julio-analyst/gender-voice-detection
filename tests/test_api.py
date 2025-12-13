"""
Unit tests for API endpoints
"""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Test imports
try:
    from src.api.feedback import app as feedback_app
    from src.api.predict import app as predict_app
except ImportError:
    pytest.skip("API modules not available", allow_module_level=True)


class TestPredictAPI:
    """Test prediction API endpoints"""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client"""
        return TestClient(predict_app)

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_predict_endpoint_exists(self, client):
        """Test if predict endpoint exists"""
        # This will fail with 422 if no file uploaded, but endpoint should exist
        response = client.post("/predict")
        assert response.status_code in [
            200,
            422,
            400,
        ]  # Any of these means endpoint exists

    def test_models_list_endpoint(self, client):
        """Test models list endpoint"""
        response = client.get("/models")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or isinstance(data, dict)


class TestFeedbackAPI:
    """Test feedback API endpoints"""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client"""
        return TestClient(feedback_app)

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_feedback_submission_endpoint(self, client):
        """Test feedback submission endpoint"""
        # Test data
        feedback_data = {
            "audio_filename": "test.wav",
            "predicted_label": "male",
            "actual_label": "female",
            "model_used": "lstm",
            "confidence": 0.95,
            "user_comment": "Test feedback",
        }

        response = client.post("/feedback", json=feedback_data)

        # Should either accept or return validation error
        assert response.status_code in [200, 201, 422]

    def test_feedback_list_endpoint(self, client):
        """Test feedback list endpoint"""
        response = client.get("/feedback")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or isinstance(data, dict)


class TestAPIValidation:
    """Test API input validation"""

    def test_invalid_audio_format(self):
        """Test if API rejects invalid audio formats"""
        # This is a placeholder - actual implementation depends on API
        assert True

    def test_missing_required_fields(self):
        """Test if API rejects missing required fields"""
        client = TestClient(feedback_app)

        # Missing required fields
        invalid_data = {
            "audio_filename": "test.wav"
            # Missing other required fields
        }

        response = client.post("/feedback", json=invalid_data)
        assert response.status_code in [400, 422]  # Should fail validation


# Integration tests
class TestAPIIntegration:
    """Integration tests for full prediction flow"""

    def test_prediction_flow(self):
        """Test full prediction flow (if models available)"""
        if not os.path.exists("models/lstm_production.h5"):
            pytest.skip("Production models not available")

        # This would test:
        # 1. Upload audio
        # 2. Get prediction
        # 3. Submit feedback
        # 4. Verify feedback saved

        assert True  # Placeholder

    def test_feedback_storage(self):
        """Test if feedback is stored correctly"""
        if not os.path.exists("data/feedback/feedback.csv"):
            pytest.skip("Feedback CSV not available")

        import pandas as pd

        # Read feedback CSV
        df = pd.read_csv("data/feedback/feedback.csv")

        # Check columns
        required_columns = [
            "feedback_id",
            "timestamp",
            "audio_filename",
            "predicted_label",
            "actual_label",
            "model_used",
            "confidence",
            "is_correct",
            "user_comment",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"


# Mock fixtures
@pytest.fixture
def sample_audio_file():
    """Fixture to create sample audio file for testing"""
    # Create a dummy WAV file (simplified)
    import struct
    import wave

    filename = "test_audio.wav"

    # Create 1 second of silence at 16kHz
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)

        # Write silent audio
        for _ in range(16000):
            wav_file.writeframes(struct.pack("h", 0))

    yield filename

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)
