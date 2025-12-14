"""
Hyperparameter Tuning using Optuna
Automatically find optimal hyperparameters for LSTM/RNN/GRU models
Integrated with MLflow for experiment tracking
"""

import optuna
from optuna.integration import MLflowCallback
import numpy as np
import mlflow
import mlflow.keras
from pathlib import Path
import json
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Simple logger
def log(message):
    print(f"[TUNING] {message}")


class HyperparameterTuner:
    """Optuna-based hyperparameter tuning for voice gender detection"""
    
    def __init__(self, model_type='lstm', n_trials=20):
        """
        Initialize tuner
        
        Args:
            model_type: Type of model (lstm, rnn, gru)
            n_trials: Number of optimization trials
        """
        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.best_params = None
        self.best_value = None
        
        # Load preprocessed data
        log("Loading preprocessed data...")
        
        data_dir = Path("data/processed")
        features_file = data_dir / "features_latest.npy"
        labels_file = data_dir / "labels_latest.npy"
        
        if not features_file.exists() or not labels_file.exists():
            raise FileNotFoundError("Preprocessed data not found! Run preprocessing first.")
        
        X = np.load(features_file)
        y = np.load(labels_file)
        
        # Train/val split (80/20)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        log(f"Data loaded: {self.X_train.shape[0]} train, {self.X_val.shape[0]} val samples")
        log(f"Input shape: {self.X_train.shape[1:]}")
    
    def _build_model(self, params):
        """Build model with given hyperparameters"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=self.X_train.shape[1:]))
        
        # Recurrent layer
        layer_class = {
            'lstm': layers.LSTM,
            'rnn': layers.SimpleRNN,
            'gru': layers.GRU
        }[self.model_type]
        
        model.add(layer_class(
            units=params['units'],
            return_sequences=False
        ))
        
        # Dropout
        model.add(layers.Dropout(params['dropout_rate']))
        
        # Dense layer
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile
        optimizer_class = {
            'adam': keras.optimizers.Adam,
            'rmsprop': keras.optimizers.RMSprop
        }[params['optimizer']]
        
        model.compile(
            optimizer=optimizer_class(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def objective(self, trial):
        """
        Objective function for Optuna
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy
        """
        # Sample hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 20, 100, step=10),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'units': trial.suggest_categorical('units', [64, 128, 256]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop']),
        }
        
        log(f"Trial {trial.number}: Testing params {params}")
        
        # Build model with suggested params
        model = self._build_model(params)
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=0,
            callbacks=[
                optuna.integration.TFKerasPruningCallback(trial, 'val_accuracy')
            ]
        )
        
        # Get best validation accuracy
        val_accuracy = max(history.history['val_accuracy'])
        
        log(f"Trial {trial.number} completed: val_accuracy = {val_accuracy:.4f}")
        
        return val_accuracy
    
    def run_tuning(self, mlflow_tracking_uri=None):
        """
        Run hyperparameter tuning
        
        Args:
            mlflow_tracking_uri: MLflow tracking URI (optional, auto-loads from env if None)
            
        Returns:
            Best hyperparameters dict
        """
        log(f"Starting Optuna tuning for {self.model_type} model...")
        log(f"Number of trials: {self.n_trials}")
        
        # Auto-load DagsHub/MLflow config if not provided
        if mlflow_tracking_uri is None:
            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if mlflow_tracking_uri:
                log(f"✅ Using MLflow tracking: {mlflow_tracking_uri}")
                # Set credentials from env
                mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
                mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
                if mlflow_username and mlflow_password:
                    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
        
        # Setup MLflow callback
        mlflc = None
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(f"optuna_{self.model_type}_tuning")
            mlflc = MLflowCallback(
                tracking_uri=mlflow_tracking_uri,
                metric_name="val_accuracy"
            )
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=f"{self.model_type}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        # Run optimization
        callbacks = [mlflc] if mlflc else []
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        # Get best results
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        log(f"✅ Tuning completed!")
        log(f"Best validation accuracy: {self.best_value:.4f}")
        log(f"Best parameters: {self.best_params}")
        
        # Save best params
        self._save_best_params()
        
        # Save optimization history
        self._save_optimization_history(study)
        
        return self.best_params
    
    def _save_best_params(self):
        """Save best hyperparameters to JSON"""
        output_dir = Path("models/tuning")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{self.model_type}_best_params.json"
        
        result = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'best_val_accuracy': float(self.best_value),
            'n_trials': self.n_trials,
            'tuning_date': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        log(f"Best parameters saved to: {output_file}")
    
    def _save_optimization_history(self, study):
        """Save Optuna study optimization history"""
        output_dir = Path("reports/tuning")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trials dataframe
        df = study.trials_dataframe()
        df.to_csv(output_dir / f"{self.model_type}_trials_{timestamp}.csv", index=False)
        
        log(f"Optimization history saved to: {output_dir}")
    
    @staticmethod
    def load_best_params(model_type):
        """
        Load best hyperparameters from saved file
        
        Args:
            model_type: Type of model (lstm, rnn, gru)
            
        Returns:
            Best parameters dict or None
        """
        params_file = Path(f"models/tuning/{model_type.lower()}_best_params.json")
        
        if params_file.exists():
            with open(params_file, 'r') as f:
                data = json.load(f)
            return data['best_params']
        else:
            log(f"No tuned parameters found for {model_type}")
            return None


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning with Optuna')
    parser.add_argument('--model-type', type=str, default='lstm',
                        choices=['lstm', 'rnn', 'gru'],
                        help='Model type to tune')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of optimization trials')
    parser.add_argument('--mlflow-uri', type=str, default=None,
                        help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    # Run tuning
    tuner = HyperparameterTuner(
        model_type=args.model_type,
        n_trials=args.n_trials
    )
    
    best_params = tuner.run_tuning(mlflow_tracking_uri=args.mlflow_uri)
    
    print("\n" + "="*60)
    print(f"🎯 BEST HYPERPARAMETERS FOR {args.model_type.upper()}")
    print("="*60)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\n✅ Best Validation Accuracy: {tuner.best_value:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
