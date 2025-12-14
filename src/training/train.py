"""
Training Script with MLflow Integration
Trains gender voice detection models with experiment tracking
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
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
        """Setup MLflow tracking with DagsHub integration"""
        # Check if using local MLflow (for testing)
        use_local = os.getenv("USE_LOCAL_MLFLOW", "false").lower() == "true"
        
        if use_local:
            tracking_uri = "mlruns"
            print("⚠️  Using LOCAL MLflow (testing mode)")
        else:
            # Get DagsHub MLflow URI from config/env
            tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI",
                self.config.get("mlflow.tracking_uri", "https://dagshub.com/Julio-analyst/gender-voice-detection.mlflow"),
            )
            
            # Setup DagsHub credentials
            username = os.getenv("MLFLOW_TRACKING_USERNAME", "Julio-analyst")
            password = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN")
            
            if password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = password
                print(f"🔐 DagsHub credentials configured for user: {username}")
            else:
                print("⚠️  WARNING: No DagsHub token found! Set DAGSHUB_TOKEN or MLFLOW_TRACKING_PASSWORD")
                print("   Get token from: https://dagshub.com/user/settings/tokens")
        
        mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        exp_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            self.config.get("mlflow.experiment_name", "gender-voice-detection"),
        )
        mlflow.set_experiment(exp_name)

        print(f"🔗 MLflow Tracking URI: {tracking_uri}")
        print(f"🧪 MLflow Experiment: {exp_name}")
        
        # Enable autologging if configured
        if self.config.get("mlflow.autolog.enabled", True):
            mlflow.tensorflow.autolog(
                log_models=self.config.get("mlflow.autolog.log_models", True),
                log_datasets=self.config.get("mlflow.autolog.log_datasets", True),
            )
            print("✅ MLflow autologging enabled")

    def _balance_dataset(self, X: np.ndarray, y: np.ndarray, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset by undersampling majority class
        
        Args:
            X: Feature array
            y: Label array
            random_seed: Random seed for reproducibility
            
        Returns:
            Balanced X and y arrays
        """
        np.random.seed(random_seed)
        
        # Count samples per class
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        print(f"\n⚖️  Balancing Dataset:")
        print(f"   Before - Label 0 (Cewe): {class_counts.get(0, 0)}, Label 1 (Cowo): {class_counts.get(1, 0)}")
        
        # Find minimum class size
        min_samples = min(counts)
        
        # Get indices for each class
        balanced_indices = []
        for label in unique:
            label_indices = np.where(y == label)[0]
            # Random sample from this class
            sampled_indices = np.random.choice(label_indices, size=min_samples, replace=False)
            balanced_indices.extend(sampled_indices)
        
        # Shuffle balanced indices
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        # Verify balance
        unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
        print(f"   After  - Label 0 (Cewe): {counts_bal[0]}, Label 1 (Cowo): {counts_bal[1]}")
        print(f"   ✅ Balanced to {min_samples} samples per class")
        
        return X_balanced, y_balanced

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: float = None,
        val_split: float = None,
        test_split: float = None,
        random_seed: int = None,
        balance: bool = True,
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
            balance: Whether to balance classes (default True)

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

        # Balance dataset first if requested
        if balance:
            X, y = self._balance_dataset(X, y, random_seed)

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
        print(f"   Train: {len(X_train)} samples ({train_split*100:.0f}%) - Label 0: {np.sum(y_train==0)}, Label 1: {np.sum(y_train==1)}")
        print(f"   Val:   {len(X_val)} samples ({val_split*100:.0f}%) - Label 0: {np.sum(y_val==0)}, Label 1: {np.sum(y_val==1)}")
        print(f"   Test:  {len(X_test)} samples ({test_split*100:.0f}%) - Label 0: {np.sum(y_test==0)}, Label 1: {np.sum(y_test==1)}")

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

            # Log training plots
            self._log_training_plots()

            print(f"\n✅ Training completed!")
            print(
                f"   Final train accuracy: {final_metrics['final_train_accuracy']:.4f}"
            )
            print(f"   Final val accuracy: {final_metrics['final_val_accuracy']:.4f}")

        # NOTE: Do NOT end run here - let caller handle it
        # This allows adding evaluation metrics to the same run
        # mlflow.end_run()

        return self.history

    def _log_training_plots(self):
        """Create and log training visualization plots to MLflow"""
        if self.history is None:
            return

        try:
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot accuracy
            ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
            ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
            ax1.set_title(f'{self.model_type.upper()} - Model Accuracy', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            
            # Plot loss
            ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
            ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
            ax2.set_title(f'{self.model_type.upper()} - Model Loss', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot temporarily
            plot_path = f"training_curves_{self.model_type}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(plot_path, "plots")
            
            # Clean up temp file
            if os.path.exists(plot_path):
                os.remove(plot_path)
            
            print("   📊 Training plots logged to MLflow")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to create training plots: {e}")

    def log_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray):
        """Create and log confusion matrix to MLflow"""
        if self.model is None:
            print("   ⚠️  No model available for confusion matrix")
            return

        try:
            # Get predictions
            y_pred_proba = self.model.get_model().predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_true = y_test.astype(int)
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Labels
            classes = ['Female (0)', 'Male (1)']
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes,
                   title=f'{self.model_type.upper()} - Confusion Matrix',
                   ylabel='True label',
                   xlabel='Predicted label')
            
            # Rotate tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            cm_path = f"confusion_matrix_{self.model_type}.png"
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(cm_path, "plots")
            
            # Clean up
            if os.path.exists(cm_path):
                os.remove(cm_path)
            
            # Log classification report as text
            report = classification_report(y_true, y_pred, target_names=classes)
            report_path = f"classification_report_{self.model_type}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            mlflow.log_artifact(report_path, "reports")
            
            if os.path.exists(report_path):
                os.remove(report_path)
            
            print("   📊 Confusion matrix and classification report logged to MLflow")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to create confusion matrix: {e}")

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

        # Log model to MLflow with better tracking
        try:
            # Log the versioned model
            mlflow.log_artifact(str(filepath), "models")
            
            # Log model info
            mlflow.log_param("model_path", str(filepath))
            mlflow.log_param("model_size_mb", filepath.stat().st_size / (1024 * 1024))
            
            # Try to log as Keras model for better MLflow integration
            mlflow.keras.log_model(self.model.get_model(), "model")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to log model to MLflow: {e}")

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
            print("\n[*] Found processed data, loading...")
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

        # Train (this creates its own MLflow run)
        history = trainer.train(X_train, y_train, X_val, y_val, **train_kwargs)
        
        # Get the active run ID from training
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        
        # If training ended the run, restart it to add evaluation metrics
        if not mlflow.active_run() and run_id:
            mlflow.start_run(run_id=run_id)
        
        # Evaluate on test set and log to SAME run
        print(f"\n📊 Evaluating on test set...")
        from src.training.evaluate import ModelEvaluator

        evaluator = ModelEvaluator(args.model_type)

        # Get predictions
        y_pred = trainer.model.get_model().predict(X_test)
        metrics = evaluator.evaluate(y_test, y_pred)
        
        # Log test metrics to the SAME MLflow run
        mlflow.log_metric("test_accuracy", metrics['accuracy'])
        mlflow.log_metric("test_precision", metrics.get('precision', 0))
        mlflow.log_metric("test_recall", metrics.get('recall', 0))
        mlflow.log_metric("test_f1", metrics.get('f1_score', 0))
        
        # Log confusion matrix and classification report
        print(f"\n📊 Creating evaluation plots...")
        trainer.log_confusion_matrix(X_test, y_test)
        
        # End the run now
        mlflow.end_run()

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
