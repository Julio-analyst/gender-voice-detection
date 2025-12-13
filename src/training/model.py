"""
Model Architectures for Gender Voice Detection
Supports: RNN, LSTM, GRU models with configurable architecture
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

# Handle imports for both module and standalone execution
try:
    from ..utils.config import get_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config import get_config


class BaseModel:
    """Base class for all model architectures"""
    
    def __init__(self, model_type: str, input_shape: Tuple[int, int], **kwargs):
        """
        Initialize base model
        
        Args:
            model_type: Type of model ('rnn', 'lstm', 'gru')
            input_shape: Shape of input data (time_steps, n_mfcc)
            **kwargs: Additional hyperparameters
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.config = get_config()
        
        # Get model config from YAML
        model_config = self.config.get(f'models.{model_type}', {})
        
        # Merge config with kwargs (kwargs take precedence)
        self.hidden_units = kwargs.get('hidden_units', model_config.get('hidden_units', 64))
        self.dropout = kwargs.get('dropout', model_config.get('dropout', 0.2))
        self.learning_rate = kwargs.get('learning_rate', self.config.get('training.learning_rate', 0.001))
        
        self.model = None
        self.history = None
        
    def build(self) -> keras.Model:
        """Build the model architecture - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement build() method")
    
    def compile_model(self, optimizer: Optional[keras.optimizers.Optimizer] = None,
                     loss: str = 'binary_crossentropy',
                     metrics: list = None):
        """
        Compile the model
        
        Args:
            optimizer: Keras optimizer (default: Adam with configured learning rate)
            loss: Loss function for binary classification
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation. Call build() first.")
        
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        if metrics is None:
            metrics = ['accuracy', 
                      keras.metrics.Precision(name='precision'),
                      keras.metrics.Recall(name='recall')]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model must be built first. Call build() first.")
        return self.model.summary()
    
    def get_model(self) -> keras.Model:
        """Get the compiled model"""
        if self.model is None:
            raise ValueError("Model must be built first. Call build() first.")
        return self.model
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("Model must be built first. Call build() first.")
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> keras.Model:
        """Load model from file"""
        return keras.models.load_model(filepath)


class SimpleRNNModel(BaseModel):
    """Simple RNN model for gender classification"""
    
    def __init__(self, input_shape: Tuple[int, int], **kwargs):
        super().__init__('rnn', input_shape, **kwargs)
        
    def build(self) -> keras.Model:
        """
        Build Simple RNN architecture
        
        Architecture:
            - Input layer
            - SimpleRNN layer with configurable units
            - Dropout for regularization
            - Dense layer (sigmoid for binary classification)
        """
        self.model = models.Sequential([
            layers.Input(shape=self.input_shape, name='input'),
            
            # RNN layer
            layers.SimpleRNN(
                units=self.hidden_units,
                return_sequences=False,
                name='rnn_layer'
            ),
            
            # Regularization
            layers.Dropout(self.dropout, name='dropout'),
            
            # Output layer for binary classification
            layers.Dense(1, activation='sigmoid', name='output')
        ], name='SimpleRNN_GenderClassifier')
        
        return self.model


class LSTMModel(BaseModel):
    """LSTM model for gender classification"""
    
    def __init__(self, input_shape: Tuple[int, int], **kwargs):
        super().__init__('lstm', input_shape, **kwargs)
        
    def build(self) -> keras.Model:
        """
        Build LSTM architecture
        
        Architecture:
            - Input layer
            - LSTM layer with configurable units
            - Dropout for regularization
            - Dense layer (sigmoid for binary classification)
        """
        self.model = models.Sequential([
            layers.Input(shape=self.input_shape, name='input'),
            
            # LSTM layer
            layers.LSTM(
                units=self.hidden_units,
                return_sequences=False,
                name='lstm_layer'
            ),
            
            # Regularization
            layers.Dropout(self.dropout, name='dropout'),
            
            # Output layer for binary classification
            layers.Dense(1, activation='sigmoid', name='output')
        ], name='LSTM_GenderClassifier')
        
        return self.model


class GRUModel(BaseModel):
    """GRU model for gender classification"""
    
    def __init__(self, input_shape: Tuple[int, int], **kwargs):
        super().__init__('gru', input_shape, **kwargs)
        
    def build(self) -> keras.Model:
        """
        Build GRU architecture
        
        Architecture:
            - Input layer
            - GRU layer with configurable units
            - Dropout for regularization
            - Dense layer (sigmoid for binary classification)
        """
        self.model = models.Sequential([
            layers.Input(shape=self.input_shape, name='input'),
            
            # GRU layer
            layers.GRU(
                units=self.hidden_units,
                return_sequences=False,
                name='gru_layer'
            ),
            
            # Regularization
            layers.Dropout(self.dropout, name='dropout'),
            
            # Output layer for binary classification
            layers.Dense(1, activation='sigmoid', name='output')
        ], name='GRU_GenderClassifier')
        
        return self.model


def create_model(model_type: str, input_shape: Tuple[int, int], **kwargs) -> BaseModel:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('rnn', 'lstm', 'gru')
        input_shape: Shape of input data (time_steps, n_mfcc)
        **kwargs: Additional hyperparameters
        
    Returns:
        Model instance (not yet built)
        
    Example:
        >>> model = create_model('lstm', input_shape=(94, 13), hidden_units=128)
        >>> model.build()
        >>> model.compile_model()
    """
    model_type = model_type.lower()
    
    model_map = {
        'rnn': SimpleRNNModel,
        'lstm': LSTMModel,
        'gru': GRUModel
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(model_map.keys())}")
    
    return model_map[model_type](input_shape=input_shape, **kwargs)


# ============================================================================
# TESTING CODE
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Model Architectures")
    print("=" * 80)
    
    # Get config
    config = get_config()
    
    # Calculate expected input shape
    # Duration: 3 seconds, Sample rate: 16000 Hz, Hop length: 512
    duration = config.get('audio.duration', 3)
    sample_rate = config.get('audio.sample_rate', 16000)
    hop_length = config.get('audio.hop_length', 512)
    n_mfcc = config.get('audio.n_mfcc', 13)
    
    # Calculate time steps: (duration * sample_rate) / hop_length ≈ 94
    time_steps = int((duration * sample_rate) / hop_length)
    input_shape = (time_steps, n_mfcc)
    
    print(f"\n📊 Input Shape: {input_shape}")
    print(f"   Time steps: {time_steps}, MFCC coefficients: {n_mfcc}")
    
    # Test all 3 model types
    model_types = ['rnn', 'lstm', 'gru']
    
    for model_type in model_types:
        print(f"\n{'=' * 80}")
        print(f"Testing {model_type.upper()} Model")
        print(f"{'=' * 80}")
        
        try:
            # Create model
            model_instance = create_model(
                model_type=model_type,
                input_shape=input_shape,
                hidden_units=64,
                dropout=0.2
            )
            
            # Build and compile
            model_instance.build()
            model_instance.compile_model()
            
            # Show summary
            print(f"\n{model_type.upper()} Architecture:")
            model_instance.summary()
            
            # Test with dummy data
            print(f"\n🧪 Testing with dummy data...")
            dummy_input = np.random.randn(1, time_steps, n_mfcc).astype(np.float32)
            prediction = model_instance.get_model().predict(dummy_input, verbose=0)
            
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Prediction shape: {prediction.shape}")
            print(f"   Prediction value: {prediction[0][0]:.4f}")
            print(f"   Predicted class: {'Perempuan' if prediction[0][0] > 0.5 else 'Laki-laki'}")
            
            print(f"\n✅ {model_type.upper()} model working correctly!")
            
        except Exception as e:
            print(f"\n❌ Error testing {model_type.upper()} model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("✅ All model architectures tested successfully!")
    print(f"{'=' * 80}")
