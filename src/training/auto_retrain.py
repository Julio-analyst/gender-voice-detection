"""
Auto-Retrain Module
Automatically retrain models when feedback threshold is reached
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Handle imports
try:
    from ..preprocessing.feature_extractor import MFCCExtractor
    from ..training.evaluate import ModelEvaluator
    from ..training.train import ModelTrainer
    from ..utils.config import get_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.preprocessing.feature_extractor import MFCCExtractor
    from src.training.evaluate import ModelEvaluator
    from src.training.train import ModelTrainer
    from src.utils.config import get_config


class AutoRetrainer:
    """Handles automatic model retraining"""

    def __init__(self):
        """Initialize auto-retrainer"""
        self.config = get_config()
        self.feedback_dir = Path(os.getenv("FEEDBACK_DATA_DIR", "data/feedback"))
        self.feedback_file = self.feedback_dir / "feedback.csv"

    def check_should_retrain(self) -> bool:
        """
        Check if model should be retrained

        Returns:
            True if should retrain, False otherwise
        """
        # Check if auto-retrain is enabled
        auto_retrain = os.getenv("AUTO_RETRAIN_ENABLED", "true").lower() == "true"
        if not auto_retrain:
            print("⚠️  Auto-retrain is disabled")
            return False

        # Check if feedback file exists
        if not self.feedback_file.exists():
            print("⚠️  No feedback data available")
            return False

        # Load feedback
        df = pd.read_csv(self.feedback_file)
        total_feedback = len(df)

        # Check threshold
        threshold = int(os.getenv("FEEDBACK_THRESHOLD", 20))

        print(f"📊 Feedback count: {total_feedback}/{threshold}")

        return total_feedback >= threshold

    def prepare_training_data_from_feedback(self) -> tuple:
        """
        Prepare training data from feedback

        Returns:
            Tuple of (X, y) arrays
        """
        print("\n📦 Preparing training data from feedback...")

        # Load feedback
        df = pd.read_csv(self.feedback_file)

        print(f"   Total feedback: {len(df)}")
        print(f"   Correct predictions: {df['is_correct'].sum()}")
        print(f"   Incorrect predictions: {(~df['is_correct']).sum()}")

        # Get unique audio files with their actual labels
        # For simplicity, we'll generate dummy MFCC data
        # In production, you would load actual audio files

        print("\n⚠️  NOTE: This is a simplified version.")
        print("   In production, you would load actual audio files from feedback.")
        print("   For now, generating dummy data based on feedback labels...\n")

        # Calculate expected shape
        duration = self.config.get("audio.duration", 3)
        sample_rate = self.config.get("audio.sample_rate", 16000)
        hop_length = self.config.get("audio.hop_length", 512)
        n_mfcc = self.config.get("audio.n_mfcc", 13)
        time_steps = int((duration * sample_rate) / hop_length)

        # Generate dummy MFCC for each feedback entry
        n_samples = len(df)
        X = np.random.randn(n_samples, time_steps, n_mfcc).astype(np.float32)

        # Convert labels to binary (0=Laki-laki, 1=Perempuan)
        y = (
            df["actual_label"]
            .map({"Laki-laki": 0, "Perempuan": 1})
            .values.astype(np.float32)
        )

        print(f"✅ Data prepared:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(
            f"   Class distribution: Laki-laki={np.sum(y==0)}, Perempuan={np.sum(y==1)}"
        )

        return X, y

    def retrain_model(self, model_type: str = "lstm", epochs: int = 20):
        """
        Retrain a specific model

        Args:
            model_type: Type of model to retrain
            epochs: Number of training epochs

        Returns:
            Training results dictionary
        """
        print(f"\n{'='*80}")
        print(f"🚀 Auto-Retraining {model_type.upper()} Model")
        print(f"{'='*80}")

        try:
            # Prepare data
            X, y = self.prepare_training_data_from_feedback()

            # Create trainer
            trainer = ModelTrainer(model_type=model_type)

            # Prepare data splits
            data_splits = trainer.prepare_data(X, y)
            X_train, y_train = data_splits["train"]
            X_val, y_val = data_splits["val"]
            X_test, y_test = data_splits["test"]

            # Train model
            history = trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)

            # Save model with auto-retrain tag
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = trainer.save_model(version=f"autoretrain_{timestamp}")

            # Evaluate on test set
            print(f"\n📊 Evaluating retrained model...")
            evaluator = ModelEvaluator(model_type=model_type)
            y_pred = trainer.model.get_model().predict(X_test, verbose=0)

            # Generate report
            report = evaluator.generate_full_report(y_test, y_pred)

            # Compare with old model performance
            min_accuracy = float(os.getenv("MIN_ACCURACY_THRESHOLD", 0.85))
            new_accuracy = report["metrics"]["accuracy"]

            print(f"\n📈 Performance Comparison:")
            print(f"   New model accuracy: {new_accuracy:.2%}")
            print(f"   Minimum threshold: {min_accuracy:.2%}")

            # Decide whether to deploy
            should_deploy = new_accuracy >= min_accuracy

            if should_deploy:
                print(f"\n✅ New model meets threshold! Deploying to production...")

                # Backup old production model
                old_prod_model = Path(f"models/{model_type}_production.h5")
                if old_prod_model.exists():
                    backup_path = Path(
                        f"models/{model_type}_production_backup_{timestamp}.h5"
                    )
                    shutil.copy(old_prod_model, backup_path)
                    print(f"   📦 Old model backed up to: {backup_path}")

                # Copy new model to production
                shutil.copy(model_path, old_prod_model)
                print(f"   🚀 New model deployed to: {old_prod_model}")

            else:
                print(
                    f"\n⚠️  New model accuracy ({new_accuracy:.2%}) below threshold ({min_accuracy:.2%})"
                )
                print(f"   Model trained but NOT deployed to production.")

            # Archive feedback data
            self._archive_feedback()

            return {
                "success": True,
                "model_type": model_type,
                "new_accuracy": new_accuracy,
                "deployed": should_deploy,
                "model_path": model_path,
                "report_dir": report["directory"],
            }

        except Exception as e:
            print(f"\n❌ Error during auto-retrain: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _archive_feedback(self):
        """Archive used feedback data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.feedback_dir / f"feedback_archived_{timestamp}.csv"

            # Copy feedback file to archive
            shutil.copy(self.feedback_file, archive_path)

            # Clear current feedback file
            empty_df = pd.DataFrame(
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
            empty_df.to_csv(self.feedback_file, index=False)

            print(f"\n📦 Feedback archived to: {archive_path}")
            print(f"   Feedback counter reset to 0")

        except Exception as e:
            print(f"⚠️  Warning: Could not archive feedback: {str(e)}")

    def run_auto_retrain_all_models(self, epochs: int = 20):
        """
        Run auto-retrain for all models

        Args:
            epochs: Number of epochs for training

        Returns:
            Results for all models
        """
        # Check if should retrain
        if not self.check_should_retrain():
            print("\n⏸️  Auto-retrain conditions not met. Exiting.")
            return None

        print("\n" + "=" * 80)
        print("🔄 AUTO-RETRAIN TRIGGERED")
        print("=" * 80)

        results = {}

        for model_type in ["lstm", "rnn", "gru"]:
            result = self.retrain_model(model_type, epochs=epochs)
            results[model_type] = result

        # Summary
        print("\n" + "=" * 80)
        print("📋 AUTO-RETRAIN SUMMARY")
        print("=" * 80)

        for model_type, result in results.items():
            if result["success"]:
                status = (
                    "✅ DEPLOYED" if result["deployed"] else "⚠️  TRAINED (not deployed)"
                )
                accuracy = result["new_accuracy"]
                print(
                    f"   {model_type.upper():8s}: {status} (Accuracy: {accuracy:.2%})"
                )
            else:
                print(f"   {model_type.upper():8s}: ❌ FAILED - {result['error']}")

        print("=" * 80)

        return results


# ============================================================================
# CLI Interface
# ============================================================================
def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-retrain models based on feedback"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "rnn", "gru", "all"],
        default="all",
        help="Model to retrain (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force retrain even if threshold not met"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Auto-Retrain Module")
    print("=" * 80)

    # Create retrainer
    retrainer = AutoRetrainer()

    # Force retrain if specified
    if args.force:
        print("\n⚠️  FORCE mode enabled - skipping threshold check")
        should_retrain = True
    else:
        should_retrain = retrainer.check_should_retrain()

    if not should_retrain and not args.force:
        print("\n⏸️  Auto-retrain conditions not met.")
        print("   Use --force to retrain anyway.")
        return

    # Run retrain
    if args.model == "all":
        retrainer.run_auto_retrain_all_models(epochs=args.epochs)
    else:
        retrainer.retrain_model(model_type=args.model, epochs=args.epochs)

    print("\n✅ Auto-retrain process completed!")


if __name__ == "__main__":
    main()
