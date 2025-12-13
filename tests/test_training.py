"""
Unit tests for training modules
"""
import os
from pathlib import Path

import numpy as np
import pytest

# Test imports
try:
    from src.training.evaluate import ModelEvaluator
    from src.training.model import ModelBuilder
except ImportError:
    pytest.skip("Training modules not available", allow_module_level=True)


class TestModelBuilder:
    """Test model building functionality"""

    def test_lstm_model_creation(self):
        """Test if LSTM model can be created"""
        builder = ModelBuilder(input_shape=(469, 13))
        model = builder.build_lstm(units=64, dropout=0.3)

        assert model is not None
        assert len(model.layers) > 0
        assert model.input_shape == (None, 469, 13)
        assert model.output_shape == (None, 1)  # Binary classification

    def test_rnn_model_creation(self):
        """Test if RNN model can be created"""
        builder = ModelBuilder(input_shape=(469, 13))
        model = builder.build_rnn(units=64, dropout=0.3)

        assert model is not None
        assert len(model.layers) > 0
        assert model.input_shape == (None, 469, 13)
        assert model.output_shape == (None, 1)

    def test_gru_model_creation(self):
        """Test if GRU model can be created"""
        builder = ModelBuilder(input_shape=(469, 13))
        model = builder.build_gru(units=64, dropout=0.3)

        assert model is not None
        assert len(model.layers) > 0
        assert model.input_shape == (None, 469, 13)
        assert model.output_shape == (None, 1)

    def test_model_compilation(self):
        """Test if model compiles correctly"""
        builder = ModelBuilder(input_shape=(469, 13))
        model = builder.build_lstm(units=64, dropout=0.3)

        # Model should be compiled by default
        assert model.optimizer is not None
        assert model.loss is not None

    def test_invalid_input_shape(self):
        """Test if invalid input shape raises error"""
        with pytest.raises((ValueError, TypeError)):
            builder = ModelBuilder(input_shape=(13,))  # 1D shape should fail


class TestModelEvaluator:
    """Test model evaluation functionality"""

    def test_evaluator_initialization(self):
        """Test if ModelEvaluator initializes correctly"""
        evaluator = ModelEvaluator(model=None, model_type="lstm")
        assert evaluator.model_type == "lstm"

    def test_calculate_metrics(self):
        """Test metrics calculation"""
        # Create dummy predictions and labels
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])  # Perfect predictions

        evaluator = ModelEvaluator(model=None, model_type="test")
        metrics = evaluator.calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        # Perfect predictions should have accuracy = 1.0
        assert metrics["accuracy"] == 1.0

    def test_prediction_shape(self):
        """Test if predictions have correct shape"""
        # Create dummy model and data
        builder = ModelBuilder(input_shape=(469, 13))
        model = builder.build_lstm(units=32, dropout=0.2)

        # Dummy input
        X_test = np.random.rand(10, 469, 13)

        predictions = model.predict(X_test, verbose=0)

        assert predictions.shape == (10, 1)
        assert np.all((predictions >= 0) & (predictions <= 1))  # Sigmoid output


class TestModelSaving:
    """Test model saving and loading"""

    def test_save_and_load_model(self):
        """Test if model can be saved and loaded"""
        from tensorflow import keras

        # Create a simple model
        builder = ModelBuilder(input_shape=(469, 13))
        model = builder.build_lstm(units=32, dropout=0.2)

        # Save model
        test_path = "test_model.h5"
        model.save(test_path)

        assert os.path.exists(test_path)

        # Load model
        loaded_model = keras.models.load_model(test_path)

        assert loaded_model is not None
        assert loaded_model.input_shape == model.input_shape
        assert loaded_model.output_shape == model.output_shape

        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)


# Fixtures
@pytest.fixture
def sample_dataset():
    """Fixture to create sample dataset for testing"""
    X = np.random.rand(100, 469, 13)  # 100 samples
    y = np.random.randint(0, 2, size=100)  # Binary labels
    return X, y


@pytest.fixture
def trained_model():
    """Fixture to create a trained model for testing"""
    builder = ModelBuilder(input_shape=(469, 13))
    model = builder.build_lstm(units=32, dropout=0.2)

    # Create dummy data
    X_train = np.random.rand(50, 469, 13)
    y_train = np.random.randint(0, 2, size=50)

    # Quick training (1 epoch)
    model.fit(X_train, y_train, epochs=1, verbose=0, batch_size=16)

    return model
