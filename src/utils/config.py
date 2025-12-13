"""
Configuration Loader
Load settings from config.yaml and .env files
Provides centralized access to all configuration
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class Config:
    """
    Centralized configuration management
    Loads from config.yaml and .env
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration"""
        # Load environment variables from .env
        load_dotenv()

        # Get project root (parent of src/)
        self.project_root = Path(__file__).parent.parent.parent

        # Load YAML config
        config_file = self.project_root / config_path
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Override with environment variables where applicable
        self._load_env_overrides()

    def _load_env_overrides(self):
        """Override config with environment variables"""
        # MLflow settings
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            if "mlflow" not in self.config:
                self.config["mlflow"] = {}
            self.config["mlflow"]["tracking_uri"] = mlflow_uri
            self.config["mlflow"]["username"] = os.getenv(
                "MLFLOW_TRACKING_USERNAME", ""
            )
            self.config["mlflow"]["password"] = os.getenv(
                "MLFLOW_TRACKING_PASSWORD", ""
            )

        # Training hyperparameters from env
        self.config["training"]["default"]["epochs"] = int(
            os.getenv("DEFAULT_EPOCHS", self.config["training"]["default"]["epochs"])
        )
        self.config["training"]["default"]["batch_size"] = int(
            os.getenv(
                "DEFAULT_BATCH_SIZE", self.config["training"]["default"]["batch_size"]
            )
        )
        self.config["training"]["default"]["learning_rate"] = float(
            os.getenv(
                "DEFAULT_LEARNING_RATE",
                self.config["training"]["default"]["learning_rate"],
            )
        )

        # Random seed
        self.config["dataset"]["random_seed"] = int(
            os.getenv("RANDOM_SEED", self.config["dataset"]["random_seed"])
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation
        Example: config.get('audio.sample_rate')
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_path(self, key: str) -> Path:
        """Get path from config and resolve relative to project root"""
        path_str = self.get(key)
        if path_str:
            path = Path(path_str)
            if not path.is_absolute():
                path = self.project_root / path
            return path
        return None

    @property
    def audio_config(self) -> Dict:
        """Get all audio processing configuration"""
        return self.config.get("audio", {})

    @property
    def training_config(self) -> Dict:
        """Get all training configuration"""
        return self.config.get("training", {})

    @property
    def model_config(self) -> Dict:
        """Get all model configuration"""
        return self.config.get("models", {})

    @property
    def dataset_config(self) -> Dict:
        """Get all dataset configuration"""
        return self.config.get("dataset", {})

    @property
    def mlflow_config(self) -> Dict:
        """Get MLflow configuration"""
        return self.config.get("mlflow", {})

    def __repr__(self):
        return f"Config(project={self.config['project']['name']}, version={self.config['project']['version']})"


# Global config instance
_config = None


def get_config() -> Config:
    """Get global config instance (singleton)"""
    global _config
    if _config is None:
        _config = Config()
    return _config


if __name__ == "__main__":
    # Test config loading
    config = get_config()

    print("=" * 60)
    print("Configuration Test")
    print("=" * 60)

    print(f"\nProject: {config.get('project.name')}")
    print(f"Version: {config.get('project.version')}")

    print(f"\nAudio Config:")
    print(f"  Sample Rate: {config.get('audio.sample_rate')} Hz")
    print(f"  Duration: {config.get('audio.duration')} seconds")
    print(f"  N_MFCC: {config.get('audio.n_mfcc')}")

    print(f"\nTraining Config:")
    print(f"  Epochs: {config.get('training.default.epochs')}")
    print(f"  Batch Size: {config.get('training.default.batch_size')}")
    print(f"  Learning Rate: {config.get('training.default.learning_rate')}")

    print(f"\nDataset Config:")
    print(f"  Train Split: {config.get('dataset.split.train')}")
    print(f"  Val Split: {config.get('dataset.split.val')}")
    print(f"  Test Split: {config.get('dataset.split.test')}")

    print(f"\nMLflow Config:")
    print(f"  Tracking URI: {config.mlflow_config.get('tracking_uri', 'Not set')}")
    print(f"  Experiment: {config.mlflow_config.get('experiment_name', 'Not set')}")

    print(f"\nPaths:")
    print(f"  Project Root: {config.project_root}")
    print(f"  Model Dir: {config.get_path('paths.model_dir')}")

    print("\n✅ Config loaded successfully!")
