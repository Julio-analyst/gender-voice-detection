"""
End-to-End Pipeline Test
Tests complete workflow: preprocessing → training → evaluation
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.preprocessing.audio_cleaner import AudioCleaner
from src.preprocessing.feature_extractor import MFCCExtractor
from src.training.evaluate import ModelEvaluator
from src.training.model import create_model
from src.training.train import ModelTrainer
from src.utils.config import get_config


def generate_dummy_audio_data(n_samples: int = 200) -> tuple:
    """
    Generate dummy MFCC features and labels for testing

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (X, y) where X is MFCC features and y is labels
    """
    print(f"\n📊 Generating {n_samples} dummy audio samples...")

    config = get_config()

    # Get audio parameters
    duration = config.get("audio.duration", 3)
    sample_rate = config.get("audio.sample_rate", 16000)
    hop_length = config.get("audio.hop_length", 512)
    n_mfcc = config.get("audio.n_mfcc", 13)

    # Calculate time steps
    time_steps = int((duration * sample_rate) / hop_length)

    # Generate random MFCC features
    X = np.random.randn(n_samples, time_steps, n_mfcc).astype(np.float32)

    # Generate balanced binary labels
    y = np.array([i % 2 for i in range(n_samples)], dtype=np.float32)
    np.random.shuffle(y)

    print(f"   ✅ Generated features shape: {X.shape}")
    print(f"   ✅ Generated labels shape: {y.shape}")
    print(
        f"   ✅ Class distribution: Laki-laki={np.sum(y==0)}, Perempuan={np.sum(y==1)}"
    )

    return X, y


@pytest.fixture
def dummy_data():
    """Generate dummy data for testing"""
    return generate_dummy_audio_data(n_samples=200)


@pytest.mark.parametrize("model_type", ["lstm", "rnn", "gru"])
def test_single_model(model_type, dummy_data):
    """
    Test complete pipeline for a single model

    Args:
        model_type: Type of model ('rnn', 'lstm', 'gru')
        dummy_data: Fixture providing (X, y) test data
    """
    X, y = dummy_data
    print(f"\n{'='*80}")
    print(f"Testing {model_type.upper()} Model Pipeline")
    print(f"{'='*80}")

    try:
        # 1. Training
        print(f"\n🚀 Step 1: Training {model_type.upper()} model...")
        trainer = ModelTrainer(model_type=model_type)

        # Prepare data
        data_splits = trainer.prepare_data(X, y)
        X_train, y_train = data_splits["train"]
        X_val, y_val = data_splits["val"]
        X_test, y_test = data_splits["test"]

        # Train (just 5 epochs for testing)
        history = trainer.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=16)

        # Save model
        model_path = trainer.save_model()
        print(f"   ✅ Model saved: {model_path}")

        # 2. Evaluation
        print(f"\n📊 Step 2: Evaluating {model_type.upper()} model...")
        evaluator = ModelEvaluator(model_type=model_type)

        # Get predictions on test set
        y_pred = trainer.model.get_model().predict(X_test, verbose=0)

        # Generate full evaluation report
        report = evaluator.generate_full_report(y_test, y_pred)

        print(f"\n✅ {model_type.upper()} Pipeline Complete!")
        print(f"   📁 Report directory: {report['directory']}")
        print(f"   🎯 Test Accuracy: {report['metrics']['accuracy']:.4f}")
        print(f"   📊 AUC Score: {report['metrics']['auc']:.4f}")

        return True

    except Exception as e:
        print(f"\n❌ Error in {model_type.upper()} pipeline: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run complete end-to-end pipeline test"""
    print("=" * 80)
    print("END-TO-END PIPELINE TEST")
    print("Gender Voice Detection MLOps Project")
    print("=" * 80)

    # Generate dummy data
    X, y = generate_dummy_audio_data(n_samples=200)

    # Test all 3 model types
    model_types = ["lstm", "rnn", "gru"]
    results = {}

    for model_type in model_types:
        success = test_single_model(model_type, X, y)
        results[model_type] = success

    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE TEST SUMMARY")
    print(f"{'='*80}")

    for model_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {model_type.upper():8s}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\n{'='*80}")
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print(f"{'='*80}")

    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Pipeline is working correctly!")
        print("\n📋 Next Steps:")
        print("   1. Check DagsHub MLflow for experiment tracking:")
        print("      https://dagshub.com/Julio-analyst/gender-voice-detection.mlflow")
        print("   2. Review generated reports in reports/ directory")
        print("   3. Check saved models in models/ directory")
        print("   4. Ready to integrate with real audio data!")
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
