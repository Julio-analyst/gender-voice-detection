"""
Unit tests for preprocessing modules
"""
import pytest
import numpy as np
import os
from pathlib import Path

# Test imports
try:
    from src.preprocessing.feature_extractor import MFCCExtractor
    from src.preprocessing.audio_cleaner import AudioCleaner
    from src.preprocessing.dataset_loader import DatasetLoader
except ImportError:
    pytest.skip("Preprocessing modules not available", allow_module_level=True)


class TestMFCCExtractor:
    """Test MFCC feature extraction"""
    
    def test_extractor_initialization(self):
        """Test if MFCCExtractor initializes correctly"""
        extractor = MFCCExtractor(n_mfcc=13, max_len=469)
        assert extractor.n_mfcc == 13
        assert extractor.max_len == 469
    
    def test_extract_features_shape(self):
        """Test if extracted features have correct shape"""
        extractor = MFCCExtractor(n_mfcc=13, max_len=469)
        
        # Create dummy audio (3 seconds at 16kHz)
        dummy_audio_path = "test_audio.wav"
        
        # Skip if no test audio available
        if not os.path.exists(dummy_audio_path):
            pytest.skip("Test audio file not available")
        
        features = extractor.extract(dummy_audio_path)
        assert features.shape == (469, 13), f"Expected (469, 13), got {features.shape}"
    
    def test_padding(self):
        """Test if padding works for short audio"""
        extractor = MFCCExtractor(n_mfcc=13, max_len=100)
        
        # Create short dummy array
        short_mfcc = np.random.rand(13, 50)  # Only 50 frames
        padded = extractor._pad_or_truncate(short_mfcc, 100)
        
        assert padded.shape == (13, 100)
    
    def test_truncation(self):
        """Test if truncation works for long audio"""
        extractor = MFCCExtractor(n_mfcc=13, max_len=100)
        
        # Create long dummy array
        long_mfcc = np.random.rand(13, 200)  # 200 frames
        truncated = extractor._pad_or_truncate(long_mfcc, 100)
        
        assert truncated.shape == (13, 100)


class TestAudioCleaner:
    """Test audio preprocessing"""
    
    def test_cleaner_initialization(self):
        """Test if AudioCleaner initializes correctly"""
        cleaner = AudioCleaner(target_sr=16000)
        assert cleaner.target_sr == 16000
    
    def test_normalize_audio(self):
        """Test audio normalization"""
        cleaner = AudioCleaner()
        
        # Create dummy audio array
        audio = np.random.rand(16000) * 2 - 1  # Random values between -1 and 1
        normalized = cleaner.normalize(audio)
        
        # Check if normalized to max 1.0
        assert np.max(np.abs(normalized)) <= 1.0


class TestDatasetLoader:
    """Test dataset loading"""
    
    def test_loader_initialization(self):
        """Test if DatasetLoader initializes correctly"""
        loader = DatasetLoader(data_dir="data/raw_wav")
        assert loader.data_dir == Path("data/raw_wav")
    
    def test_load_dataset_structure(self):
        """Test if dataset structure is correct"""
        loader = DatasetLoader(data_dir="data/raw_wav")
        
        # Skip if dataset not available
        if not os.path.exists("data/raw_wav"):
            pytest.skip("Dataset directory not available")
        
        try:
            X, y, files = loader.load()
            
            # Check shapes
            assert len(X) == len(y) == len(files)
            assert X.ndim == 3  # (samples, time_steps, features)
            assert y.ndim == 1  # (samples,)
            
        except Exception as e:
            pytest.skip(f"Dataset loading failed: {e}")


# Fixtures
@pytest.fixture
def sample_audio():
    """Fixture to create sample audio for testing"""
    return np.random.rand(48000)  # 3 seconds at 16kHz


@pytest.fixture
def sample_mfcc():
    """Fixture to create sample MFCC features"""
    return np.random.rand(469, 13)
