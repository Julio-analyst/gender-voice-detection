"""
Training Script with MLflow Integration
Trains gender voice detection models with experiment tracking
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Handle imports for both module and standalone execution
try:
    from ..utils.config import get_config
    from .model import create_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.training.model import create_model
    from src.utils.config import get_config


class ModelTrainer:
    """Handles model training with MLflow tracking"""

    def __init__(self, model_type: str = "lstm", experiment_name: Optional[str] = None):
        """
        Initialize trainer

        Args:
            model_type: Type of model to train ('rnn', 'lstm', 'gru')
            experiment_name: MLflow experiment name (default from config)
        """
        self.config = get_config()
        self.model_type = model_type.lower()

        # Setup MLflow
        self._setup_mlflow(experiment_name)

        # Model will be created during training
        self.model = None
        self.history = None

    def _setup_mlflow(self, experiment_name: Optional[str] = None):
        """Setup MLflow tracking"""
        # Get MLflow URI from config/env
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            self.config.get("mlflow.tracking_uri", "http://localhost:5000"),
        )
        mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        exp_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            self.config.get("mlflow.experiment_name", "gender-voice-detection"),
        )
        mlflow.set_experiment(exp_name)

        print(f"🔗 MLflow Tracking URI: {tracking_uri}")
        print(f"🧪 MLflow Experiment: {exp_name}")

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: float = None,
        val_split: float = None,
        test_split: float = None,
        random_seed: int = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train/val/test sets

        Args:
            X: Feature array (n_samples, time_steps, n_features)
            y: Label array (n_samples,)
            train_split: Training set ratio (default from config)
            val_split: Validation set ratio (default from config)
            test_split: Test set ratio (default from config)
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'val', 'test' keys containing (X, y) tuples
        """
        # Get splits from config if not provided
        train_split = train_split or self.config.get("dataset.train_split", 0.8)
        val_split = val_split or self.config.get("dataset.val_split", 0.1)
        test_split = test_split or self.config.get("dataset.test_split", 0.1)
        random_seed = random_seed or self.config.get("training.random_seed", 42)

        # Validate splits
        assert (
            abs(train_split + val_split + test_split - 1.0) < 1e-6
        ), f"Splits must sum to 1.0, got {train_split + val_split + test_split}"

        # Set random seed
        np.random.seed(random_seed)

        # Shuffle data
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Calculate split points
        train_end = int(n_samples * train_split)
        val_end = train_end + int(n_samples * val_split)

        # Split data
        X_train, y_train = X_shuffled[:train_end], y_shuffled[:train_end]
        X_val, y_val = X_shuffled[train_end:val_end], y_shuffled[train_end:val_end]
        X_test, y_test = X_shuffled[val_end:], y_shuffled[val_end:]

        print(f"\n📊 Data Split:")
        print(f"   Train: {len(X_train)} samples ({train_split*100:.0f}%)")
        print(f"   Val:   {len(X_val)} samples ({val_split*100:.0f}%)")
        print(f"   Test:  {len(X_test)} samples ({test_split*100:.0f}%)")

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        hidden_units: int = None,
        dropout: float = None,
        callbacks: List[keras.callbacks.Callback] = None,
        **kwargs,
    ) -> keras.callbacks.History:
        """
        Train model with MLflow tracking

        Args:
            X_train: Training features (n_samples, time_steps, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs (default from config)
            batch_size: Batch size (default from config)
            learning_rate: Learning rate (default from config)
            hidden_units: Hidden units (default from config)
            dropout: Dropout rate (default from config)
            callbacks: Additional Keras callbacks
            **kwargs: Additional parameters

        Returns:
            Training history
        """
        # Get hyperparameters from config if not provided
        epochs = epochs or self.config.get("training.epochs", 50)
        batch_size = batch_size or self.config.get("training.batch_size", 16)
        learning_rate = learning_rate or self.config.get(
            "training.learning_rate", 0.001
        )
        hidden_units = hidden_units or self.config.get(
            f"models.{self.model_type}.hidden_units", 64
        )
        dropout = dropout or self.config.get(f"models.{self.model_type}.dropout", 0.2)

        # Get input shape
        input_shape = X_train.shape[1:]  # (time_steps, n_features)

        # Ensure any existing run is ended
        if mlflow.active_run():
            mlflow.end_run()

        # Start MLflow run
        with mlflow.start_run(
            run_name=f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("hidden_units", hidden_units)
            mlflow.log_param("dropout", dropout)
            mlflow.log_param("input_shape", str(input_shape))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))

            # Log additional parameters
            for key, value in kwargs.items():
                mlflow.log_param(key, value)

            print(f"\n{'='*80}")
            print(f"Training {self.model_type.upper()} Model")
            print(f"{'='*80}")
            print(f"📊 Hyperparameters:")
            print(f"   Epochs: {epochs}")
            print(f"   Batch size: {batch_size}")
            print(f"   Learning rate: {learning_rate}")
            print(f"   Hidden units: {hidden_units}")
            print(f"   Dropout: {dropout}")
            print(f"   Input shape: {input_shape}")

            # Create model
            self.model = create_model(
                model_type=self.model_type,
                input_shape=input_shape,
                hidden_units=hidden_units,
                dropout=dropout,
                learning_rate=learning_rate,
            )

            # Build and compile
            self.model.build()
            self.model.compile_model()

            # Setup callbacks
            if callbacks is None:
                callbacks = []

            # Add early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            )
            callbacks.append(early_stop)

            # Disable MLflow autologging (causing issues with TF version)
            # mlflow.tensorflow.autolog(
            #     log_models=True,
            #     log_datasets=False,
            #     disable=False
            # )

            # Train model
            print(f"\n🚀 Starting training...")
            self.history = self.model.get_model().fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )

            # Log final metrics
            final_metrics = {
                "final_train_loss": self.history.history["loss"][-1],
                "final_train_accuracy": self.history.history["accuracy"][-1],
                "final_val_loss": self.history.history["val_loss"][-1],
                "final_val_accuracy": self.history.history["val_accuracy"][-1],
            }

            for metric_name, metric_value in final_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            print(f"\n✅ Training completed!")
            print(
                f"   Final train accuracy: {final_metrics['final_train_accuracy']:.4f}"
            )
            print(f"   Final val accuracy: {final_metrics['final_val_accuracy']:.4f}")

        # Ensure run is ended
        mlflow.end_run()

        return self.history

    def save_model(self, save_dir: str = None, version: str = None):
        """
        Save trained model

        Args:
            save_dir: Directory to save model (default: models/)
            version: Model version string (default: timestamp)
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        # Get save directory
        if save_dir is None:
            # Try to get from config, fallback to 'models' directory
            save_dir = self.config.get("paths.models", None)
            if save_dir is None:
                # Fallback to models/ in project root
                project_root = Path(__file__).parent.parent.parent
                save_dir = project_root / "models"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_type}_model_{version}.h5"
        filepath = save_dir / filename

        # Save model
        self.model.save(str(filepath))

        # Also save as production model
        prod_filepath = save_dir / f"{self.model_type}_production.h5"
        self.model.save(str(prod_filepath))

        # Log to MLflow
        mlflow.log_artifact(str(filepath))

        print(f"\n💾 Model saved:")
        print(f"   {filepath}")
        print(f"   {prod_filepath}")

        return str(filepath)


# ============================================================================
# CLI INTERFACE
# ============================================================================
def main():
    """Main training function with CLI support"""
    import argparse

    parser = argparse.ArgumentParser(description="Train gender voice detection model")
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["lstm", "rnn", "gru"],
        help="Model type to train",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with raw audio (with cowo/cewe folders)",
    )
    parser.add_argument(
        "--use-processed",
        action="store_true",
        help="Use previously processed data from data/processed/",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default from config)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (default from config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default from config)",
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Run quick test with dummy data"
    )

    args = parser.parse_args()

    config = get_config()

    # ========================================
    # Load/Prepare Data
    # ========================================
    if args.test_mode:
        print("\n🧪 TEST MODE: Using dummy data")
        n_samples = 100
        time_steps = 93
        n_features = 13
        X = np.random.randn(n_samples, time_steps, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")

    elif args.use_processed:
        print("\n📂 Loading processed data...")
        from src.preprocessing.dataset_loader import DatasetLoader

        loader = DatasetLoader()
        X, y, metadata = loader.load_processed_data()

    elif args.data_dir:
        print(f"\n📁 Loading data from: {args.data_dir}")
        from src.preprocessing.dataset_loader import DatasetLoader

        loader = DatasetLoader()
        X, y, metadata = loader.load_from_folders(args.data_dir)

        # Save processed data
        print("\n💾 Saving processed data...")
        loader.save_processed_data(X, y, metadata)

    else:
        # Check if processed data exists
        processed_dir = Path("data/processed")
        if (processed_dir / "features_latest.npy").exists():
            print("\n📂 Found processed data, loading...")
            from src.preprocessing.dataset_loader import DatasetLoader

            loader = DatasetLoader()
            X, y, metadata = loader.load_processed_data()
        else:
            print("\n❌ No data found!")
            print("   Options:")
            print("   1. Use --data-dir to load from raw audio folder")
            print("   2. Use --test-mode for dummy data testing")
            print("   3. Run dataset loader first:")
            print("      python src/preprocessing/dataset_loader.py --data-dir <path>")
            return

    # ========================================
    # Train Model
    # ========================================
    print(f"\n{'='*80}")
    print(f"Training {args.model_type.upper()} Model")
    print(f"{'='*80}")

    try:
        # Create trainer
        trainer = ModelTrainer(model_type=args.model_type)

        # Prepare data splits
        data_splits = trainer.prepare_data(X, y)
        X_train, y_train = data_splits["train"]
        X_val, y_val = data_splits["val"]
        X_test, y_test = data_splits["test"]

        # Train
        train_kwargs = {}
        if args.epochs:
            train_kwargs["epochs"] = args.epochs
        if args.batch_size:
            train_kwargs["batch_size"] = args.batch_size
        if args.learning_rate:
            train_kwargs["learning_rate"] = args.learning_rate

        history = trainer.train(X_train, y_train, X_val, y_val, **train_kwargs)

        # Evaluate on test set
        print(f"\n📊 Evaluating on test set...")
        from src.training.evaluate import ModelEvaluator

        evaluator = ModelEvaluator(args.model_type)

        # Get predictions
        y_pred = trainer.model.get_model().predict(X_test)
        metrics = evaluator.evaluate(y_test, y_pred)

        # Save model
        model_path = trainer.save_model()

        print(f"\n✅ Training completed successfully!")
        print(f"   Model saved to: {model_path}")
        print(f"   Test accuracy: {metrics['accuracy']:.4f}")
        print(f"   Check MLflow for detailed metrics")

    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
