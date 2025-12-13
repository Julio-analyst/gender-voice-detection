"""
Dataset Loader
Load and preprocess audio data from local folders
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Handle imports
try:
    from ..utils.config import get_config
    from .audio_cleaner import AudioCleaner
    from .feature_extractor import MFCCExtractor
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.preprocessing.audio_cleaner import AudioCleaner
    from src.preprocessing.feature_extractor import MFCCExtractor
    from src.utils.config import get_config


class DatasetLoader:
    """Load and preprocess gender voice dataset"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.cleaner = AudioCleaner()
        self.extractor = MFCCExtractor()

    def load_from_folders(
        self,
        data_dir,
        male_folder="cowo",
        female_folder="cewe",
        use_segmentation=True,
        segment_duration=3.0,
    ):
        """
        Load audio files from folder structure:
        data_dir/
            male_folder/  <- audio files for male
            female_folder/  <- audio files for female

        Args:
            data_dir: Path to directory containing male and female folders
            male_folder: Name of folder containing male audio files
            female_folder: Name of folder containing female audio files
            use_segmentation: Split long audio into 3-second segments
            segment_duration: Duration of each segment in seconds

        Returns:
            X: Features array (n_samples, time_steps, n_mfcc)
            y: Labels array (n_samples,) - 0 for male, 1 for female
            metadata: List of dicts with file info
        """
        data_dir = Path(data_dir)
        male_dir = data_dir / male_folder
        female_dir = data_dir / female_folder

        # Check directories exist
        if not male_dir.exists():
            raise FileNotFoundError(f"Male folder not found: {male_dir}")
        if not female_dir.exists():
            raise FileNotFoundError(f"Female folder not found: {female_dir}")

        print(f"\n{'='*60}")
        print(f"Loading Dataset from: {data_dir}")
        print(f"{'='*60}")

        # Collect all audio files
        audio_extensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

        male_files = []
        for ext in audio_extensions:
            male_files.extend(list(male_dir.glob(f"*{ext}")))
            male_files.extend(list(male_dir.glob(f"*{ext.upper()}")))

        female_files = []
        for ext in audio_extensions:
            female_files.extend(list(female_dir.glob(f"*{ext}")))
            female_files.extend(list(female_dir.glob(f"*{ext.upper()}")))

        print(f"\n📊 Found:")
        print(f"   - Male voices: {len(male_files)} files")
        print(f"   - Female voices: {len(female_files)} files")
        print(f"   - Total: {len(male_files) + len(female_files)} files")

        if len(male_files) == 0 or len(female_files) == 0:
            raise ValueError("No audio files found! Check folder structure.")

        # Process files
        X_list = []
        y_list = []
        metadata_list = []
        failed_files = []

        male_counter = 1
        female_counter = 1

        print(f"\n🔄 Processing male voices...")
        for audio_file in tqdm(male_files, desc="Male"):
            try:
                # Process audio with segmentation
                if use_segmentation:
                    segments_data = self._process_audio_with_segments(
                        audio_file,
                        gender="male",
                        counter=male_counter,
                        segment_duration=segment_duration,
                    )
                    for seg_features, seg_metadata in segments_data:
                        X_list.append(seg_features)
                        y_list.append(0)  # 0 = male
                        metadata_list.append(seg_metadata)
                else:
                    features = self._process_audio_file(audio_file)
                    if features is not None:
                        X_list.append(features)
                        y_list.append(0)
                        metadata_list.append(
                            {
                                "filename": audio_file.name,
                                "path": str(audio_file),
                                "label": "male",
                                "gender": "Laki-laki",
                                "renamed_as": f"L_{male_counter:03d}.wav",
                            }
                        )

                male_counter += 1

            except Exception as e:
                failed_files.append({"file": audio_file.name, "error": str(e)})

        print(f"\n🔄 Processing female voices...")
        for audio_file in tqdm(female_files, desc="Female"):
            try:
                # Process audio with segmentation
                if use_segmentation:
                    segments_data = self._process_audio_with_segments(
                        audio_file,
                        gender="female",
                        counter=female_counter,
                        segment_duration=segment_duration,
                    )
                    for seg_features, seg_metadata in segments_data:
                        X_list.append(seg_features)
                        y_list.append(1)  # 1 = female
                        metadata_list.append(seg_metadata)
                else:
                    features = self._process_audio_file(audio_file)
                    if features is not None:
                        X_list.append(features)
                        y_list.append(1)
                        metadata_list.append(
                            {
                                "filename": audio_file.name,
                                "path": str(audio_file),
                                "label": "female",
                                "gender": "Perempuan",
                                "renamed_as": f"P_{female_counter:03d}.wav",
                            }
                        )

                female_counter += 1

            except Exception as e:
                failed_files.append({"file": audio_file.name, "error": str(e)})

        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)

        # Print summary
        print(f"\n{'='*60}")
        print(f"✅ Dataset Loading Complete")
        print(f"{'='*60}")
        print(f"   - Successfully processed: {len(X)} files")
        print(f"   - Failed: {len(failed_files)} files")
        print(f"   - Male samples: {np.sum(y == 0)}")
        print(f"   - Female samples: {np.sum(y == 1)}")
        print(f"   - Features shape: {X.shape}")
        print(f"   - Labels shape: {y.shape}")

        if failed_files:
            print(f"\n⚠️  Failed files:")
            for item in failed_files[:5]:  # Show first 5
                print(f"   - {item['file']}: {item['error'][:50]}")
            if len(failed_files) > 5:
                print(f"   ... and {len(failed_files) - 5} more")

        return X, y, metadata_list

    def _process_audio_file(self, audio_path):
        """
        Process single audio file

        Args:
            audio_path: Path to audio file

        Returns:
            features: MFCC features (time_steps, n_mfcc)
        """
        # Clean audio
        cleaned_audio = self.cleaner.clean(str(audio_path))

        if cleaned_audio is None:
            return None

        # Extract MFCC features
        features = self.extractor.extract(cleaned_audio)

        return features

    def _process_audio_with_segments(
        self, audio_path, gender, counter, segment_duration=3.0
    ):
        """
        Process audio file and split into segments

        Args:
            audio_path: Path to audio file
            gender: 'male' or 'female'
            counter: File counter for naming (L_001, P_001, etc)
            segment_duration: Duration of each segment in seconds

        Returns:
            List of (features, metadata) tuples for each segment
        """
        segments_data = []

        try:
            # Load and clean audio
            audio, sr = self.cleaner.clean(str(audio_path), return_sr=True)

            if audio is None:
                return segments_data

            # Segment audio
            segments = self.cleaner.segment_audio(audio, sr, segment_duration)

            # Process each segment
            prefix = "L" if gender == "male" else "P"
            gender_id = "Laki-laki" if gender == "male" else "Perempuan"

            for seg_idx, segment in enumerate(segments, start=1):
                # Extract MFCC from segment
                features = self.extractor.extract(segment)

                # Create metadata
                metadata = {
                    "filename": audio_path.name,
                    "original_path": str(audio_path),
                    "label": gender,
                    "gender": gender_id,
                    "renamed_as": f"{prefix}_{counter:03d}_seg{seg_idx}.wav",
                    "segment_number": seg_idx,
                    "total_segments": len(segments),
                }

                segments_data.append((features, metadata))

        except Exception as e:
            print(f"   Error segmenting {audio_path.name}: {e}")

        return segments_data

    def save_processed_data(self, X, y, metadata, output_dir="data/processed"):
        """
        Save processed data to disk

        Args:
            X: Features array
            y: Labels array
            metadata: Metadata list
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save features and labels
        features_file = output_dir / f"features_{timestamp}.npy"
        labels_file = output_dir / f"labels_{timestamp}.npy"
        metadata_file = output_dir / f"metadata_{timestamp}.json"

        np.save(features_file, X)
        np.save(labels_file, y)

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Also save as latest (for easy loading)
        np.save(output_dir / "features_latest.npy", X)
        np.save(output_dir / "labels_latest.npy", y)
        with open(output_dir / "metadata_latest.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved processed data:")
        print(f"   - Features: {features_file}")
        print(f"   - Labels: {labels_file}")
        print(f"   - Metadata: {metadata_file}")
        print(f"   - Latest versions also saved")

        # Save info file
        info = {
            "timestamp": timestamp,
            "total_samples": len(X),
            "male_samples": int(np.sum(y == 0)),
            "female_samples": int(np.sum(y == 1)),
            "features_shape": list(X.shape),
            "labels_shape": list(y.shape),
            "n_mfcc": X.shape[2] if len(X.shape) > 2 else X.shape[1],
            "time_steps": X.shape[1] if len(X.shape) > 2 else None,
        }

        info_file = output_dir / f"dataset_info_{timestamp}.json"
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)

        return features_file, labels_file, metadata_file

    def load_processed_data(self, processed_dir="data/processed", use_latest=True):
        """
        Load previously processed data

        Args:
            processed_dir: Directory with processed data
            use_latest: If True, load *_latest.npy files

        Returns:
            X, y, metadata
        """
        processed_dir = Path(processed_dir)

        if use_latest:
            features_file = processed_dir / "features_latest.npy"
            labels_file = processed_dir / "labels_latest.npy"
            metadata_file = processed_dir / "metadata_latest.json"
        else:
            # Find most recent files
            features_files = list(processed_dir.glob("features_*.npy"))
            if not features_files:
                raise FileNotFoundError(f"No processed data found in {processed_dir}")

            features_file = max(features_files, key=lambda p: p.stat().st_mtime)
            timestamp = features_file.stem.replace("features_", "")
            labels_file = processed_dir / f"labels_{timestamp}.npy"
            metadata_file = processed_dir / f"metadata_{timestamp}.json"

        print(f"\n📂 Loading processed data:")
        print(f"   - Features: {features_file}")
        print(f"   - Labels: {labels_file}")

        X = np.load(features_file)
        y = np.load(labels_file)

        metadata = []
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        print(f"\n✅ Loaded:")
        print(f"   - Samples: {len(X)}")
        print(f"   - Features shape: {X.shape}")
        print(f"   - Male: {np.sum(y == 0)}, Female: {np.sum(y == 1)}")

        return X, y, metadata


def main():
    """CLI for dataset loading"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and preprocess gender voice dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing male and female folders",
    )
    parser.add_argument(
        "--male-folder",
        type=str,
        default="cowo",
        help="Name of folder with male voices (default: cowo)",
    )
    parser.add_argument(
        "--female-folder",
        type=str,
        default="cewe",
        help="Name of folder with female voices (default: cewe)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disable audio segmentation (use full audio)",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=3.0,
        help="Segment duration in seconds (default: 3.0)",
    )

    args = parser.parse_args()

    # Load and process
    loader = DatasetLoader()
    X, y, metadata = loader.load_from_folders(
        args.data_dir,
        male_folder=args.male_folder,
        female_folder=args.female_folder,
        use_segmentation=not args.no_segmentation,
        segment_duration=args.segment_duration,
    )

    # Save processed data
    loader.save_processed_data(X, y, metadata, output_dir=args.output_dir)

    print(f"\n✅ Done! Dataset ready for training.")
    print(f"   Use --use-processed flag when training to load this data")


if __name__ == "__main__":
    main()
