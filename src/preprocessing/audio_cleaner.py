"""
Audio Preprocessing Module
Noise reduction, RMS normalization, audio cleaning
Refactored from Tubes_MLOPS.ipynb
"""

from pathlib import Path
from typing import Optional, Tuple

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

# Handle both relative and absolute imports
try:
    from ..utils.config import get_config
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config import get_config


class AudioCleaner:
    """
    Audio preprocessing: noise reduction & normalization
    Sesuai dengan preprocessing di notebook
    """

    def __init__(self):
        """Initialize dengan config"""
        self.config = get_config()
        self.sr = self.config.get("audio.sample_rate", 16000)
        self.target_db = self.config.get("audio.target_db", -20.0)

    def rms_db(self, y: np.ndarray) -> float:
        """
        Calculate RMS (Root Mean Square) in dB

        Args:
            y: Audio signal array

        Returns:
            RMS value in dB
        """
        rms = np.sqrt(np.mean(y**2))
        return 20 * np.log10(rms + 1e-9)

    def normalize_rms(
        self, y: np.ndarray, target_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize audio RMS to target dB level

        Args:
            y: Audio signal array
            target_db: Target RMS in dB (default from config)

        Returns:
            Normalized audio array
        """
        if target_db is None:
            target_db = self.target_db

        # Calculate current RMS
        cur = self.rms_db(y)

        # Calculate gain needed
        gain = target_db - cur
        y_norm = y * (10 ** (gain / 20))

        # Prevent clipping
        peak = np.max(np.abs(y_norm))
        if peak > 0.999:
            y_norm = y_norm / peak * 0.999

        return y_norm

    def segment_audio(self, audio: np.ndarray, sr: int, segment_duration: float = 3.0):
        """
        Split audio into fixed-length segments

        Args:
            audio: Audio signal array
            sr: Sample rate
            segment_duration: Duration of each segment in seconds

        Returns:
            List of audio segments
        """
        segment_samples = int(sr * segment_duration)
        total_samples = len(audio)

        segments = []
        start = 0

        while start + segment_samples <= total_samples:
            segment = audio[start : start + segment_samples]
            segments.append(segment)
            start += segment_samples

        return segments

    def process_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        reduce_noise: bool = True,
        apply_preemphasis: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Complete audio preprocessing pipeline

        Args:
            audio_path: Path to input audio file
            output_path: Path to save processed audio (optional)
            reduce_noise: Apply noise reduction
            apply_preemphasis: Apply high-pass filter (preemphasis)

        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        try:
            # Load audio to mono at target sample rate
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)

            # Step 1: Noise reduction
            if reduce_noise:
                audio = nr.reduce_noise(y=audio, sr=sr)

            # Step 2: High-pass filter (preemphasis)
            if apply_preemphasis:
                audio = librosa.effects.preemphasis(audio)

            # Step 3: RMS normalization
            audio = self.normalize_rms(audio)

            # Save if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output_path, audio, sr, subtype="PCM_16")

            return audio, sr

        except Exception as e:
            raise Exception(f"Error processing audio {audio_path}: {e}")

    def clean(self, audio_path: str, return_sr: bool = False):
        """
        Simple clean method for quick preprocessing

        Args:
            audio_path: Path to audio file
            return_sr: If True, return (audio, sr), else just audio

        Returns:
            audio or (audio, sr) depending on return_sr
        """
        audio, sr = self.process_audio(audio_path)
        return (audio, sr) if return_sr else audio

    def process_batch(
        self, input_dir: str, output_dir: str, file_pattern: str = "*.wav"
    ) -> int:
        """
        Process batch of audio files

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: Glob pattern for files

        Returns:
            Number of files processed
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob(file_pattern))
        processed = 0

        for file in files:
            try:
                # Determine output filename
                out_file = output_path / file.name

                # Process audio
                self.process_audio(str(file), str(out_file))

                processed += 1
                print(f"✓ Processed: {file.name}")

            except Exception as e:
                print(f"✗ Failed: {file.name} - {e}")

        return processed


if __name__ == "__main__":
    # Test audio cleaner
    print("=" * 60)
    print("Audio Cleaner Test")
    print("=" * 60)

    cleaner = AudioCleaner()

    print(f"\nConfiguration:")
    print(f"  Sample Rate: {cleaner.sr} Hz")
    print(f"  Target RMS: {cleaner.target_db} dB")

    # Test with dummy audio
    print(f"\nTesting with dummy audio...")
    dummy_audio = np.random.randn(cleaner.sr * 3)  # 3 seconds

    print(f"  Original RMS: {cleaner.rms_db(dummy_audio):.2f} dB")

    normalized = cleaner.normalize_rms(dummy_audio)
    print(f"  Normalized RMS: {cleaner.rms_db(normalized):.2f} dB")

    print(f"\n✅ Audio Cleaner ready!")
