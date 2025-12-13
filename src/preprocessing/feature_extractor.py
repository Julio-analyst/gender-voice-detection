"""
MFCC Feature Extraction Module
Extract Mel-Frequency Cepstral Coefficients from audio
Sesuai dengan parameter di notebook (13 MFCC, 16kHz, dll)
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List

# Handle both relative and absolute imports
try:
    from .audio_cleaner import AudioCleaner
    from ..utils.config import get_config
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.preprocessing.audio_cleaner import AudioCleaner
    from src.utils.config import get_config


class MFCCExtractor:
    """
    Extract MFCC features from audio
    Parameters sesuai config dan notebook
    """
    
    def __init__(self, use_cleaner: bool = True):
        """
        Initialize MFCC extractor
        
        Args:
            use_cleaner: Use AudioCleaner preprocessing before extraction
        """
        self.config = get_config()
        
        # Audio parameters
        self.sr = self.config.get('audio.sample_rate', 16000)
        self.n_mfcc = self.config.get('audio.n_mfcc', 13)
        self.n_fft = self.config.get('audio.n_fft', 2048)
        self.hop_length = self.config.get('audio.hop_length', 512)
        
        # Audio cleaner (optional)
        self.cleaner = AudioCleaner() if use_cleaner else None
    
    def extract_from_array(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract MFCC from audio array
        
        Args:
            audio: Audio signal array
            sr: Sample rate (uses config if None)
            
        Returns:
            MFCC features array (time, n_mfcc)
        """
        if sr is None:
            sr = self.sr
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Transpose to (time, mfcc) format sesuai notebook
        return mfcc.T
    
    def extract(self, audio: Union[np.ndarray, str, Path], sr: int = None) -> np.ndarray:
        """
        Extract MFCC from audio (array or file path)
        
        Args:
            audio: Audio array or file path
            sr: Sample rate (required if audio is array)
            
        Returns:
            MFCC features array (time, n_mfcc)
        """
        if isinstance(audio, (str, Path)):
            return self.extract_from_file(audio, preprocess=True)
        else:
            return self.extract_from_array(audio, sr=sr or self.sr)
    
    def extract_from_file(
        self,
        audio_path: Union[str, Path],
        preprocess: bool = True
    ) -> np.ndarray:
        """
        Extract MFCC from audio file
        
        Args:
            audio_path: Path to audio file
            preprocess: Apply audio cleaning preprocessing
            
        Returns:
            MFCC features array (time, n_mfcc)
        """
        # Load and preprocess if needed
        if preprocess and self.cleaner:
            audio, sr = self.cleaner.process_audio(str(audio_path))
        else:
            audio, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        
        # Extract MFCC
        return self.extract_from_array(audio, sr)
    
    def extract_batch(
        self,
        audio_files: List[Union[str, Path]],
        preprocess: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract MFCC from multiple files
        
        Args:
            audio_files: List of audio file paths
            preprocess: Apply preprocessing
            
        Returns:
            Tuple of (MFCC array, list of failed files)
        """
        mfcc_features = []
        failed_files = []
        
        for file_path in audio_files:
            try:
                mfcc = self.extract_from_file(file_path, preprocess)
                mfcc_features.append(mfcc)
            except Exception as e:
                print(f"Failed to extract MFCC from {file_path}: {e}")
                failed_files.append(str(file_path))
        
        # Return as numpy array (object type karena panjang bisa beda)
        return np.array(mfcc_features, dtype=object), failed_files
    
    def extract_from_directory(
        self,
        directory: Union[str, Path],
        file_pattern: str = "*.wav",
        preprocess: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract MFCC from all files in directory
        
        Args:
            directory: Directory path
            file_pattern: Glob pattern for files
            preprocess: Apply preprocessing
            
        Returns:
            Tuple of (MFCC array, list of file paths)
        """
        dir_path = Path(directory)
        audio_files = sorted(dir_path.glob(file_pattern))
        
        mfcc_features = []
        file_paths = []
        
        for file_path in audio_files:
            try:
                mfcc = self.extract_from_file(file_path, preprocess)
                mfcc_features.append(mfcc)
                file_paths.append(str(file_path))
                print(f"✓ Extracted MFCC from: {file_path.name} - Shape: {mfcc.shape}")
            except Exception as e:
                print(f"✗ Failed: {file_path.name} - {e}")
        
        return np.array(mfcc_features, dtype=object), file_paths


if __name__ == "__main__":
    # Test MFCC extractor
    print("=" * 60)
    print("MFCC Extractor Test")
    print("=" * 60)
    
    extractor = MFCCExtractor(use_cleaner=True)
    
    print(f"\nConfiguration:")
    print(f"  Sample Rate: {extractor.sr} Hz")
    print(f"  N_MFCC: {extractor.n_mfcc}")
    print(f"  N_FFT: {extractor.n_fft}")
    print(f"  Hop Length: {extractor.hop_length}")
    
    # Test with dummy audio (3 seconds)
    print(f"\nTesting with dummy audio (3 seconds)...")
    dummy_audio = np.random.randn(extractor.sr * 3)
    
    mfcc = extractor.extract_from_array(dummy_audio)
    
    print(f"  MFCC Shape: {mfcc.shape}")
    print(f"  Time steps: {mfcc.shape[0]}")
    print(f"  MFCC coefficients: {mfcc.shape[1]}")
    
    # Expected: ~94 time steps untuk 3 detik audio
    # Formula: (sample_rate * duration) / hop_length
    expected_steps = (extractor.sr * 3) // extractor.hop_length
    print(f"  Expected time steps: ~{expected_steps}")
    
    print(f"\n✅ MFCC Extractor ready!")
