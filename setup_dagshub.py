"""
DagsHub Setup and Configuration
Initialize DagsHub connection for MLflow tracking and DVC data versioning
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import get_config


def check_dagshub_credentials():
    """Check if DagsHub credentials are configured"""
    print("\n" + "=" * 70)
    print("🔍 Checking DagsHub Configuration")
    print("=" * 70)
    
    # Check for token
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    has_token = bool(dagshub_token or mlflow_password)
    
    print(f"\n✓ DAGSHUB_TOKEN: {'✅ Set' if dagshub_token else '❌ Not set'}")
    print(f"✓ MLFLOW_TRACKING_PASSWORD: {'✅ Set' if mlflow_password else '❌ Not set'}")
    print(f"✓ MLFLOW_TRACKING_USERNAME: {os.getenv('MLFLOW_TRACKING_USERNAME', '❌ Not set')}")
    
    return has_token


def setup_dagshub():
    """Setup DagsHub connection"""
    config = get_config()
    
    print("\n" + "=" * 70)
    print("🚀 Setting up DagsHub Integration")
    print("=" * 70)
    
    # Get repo info
    repo_owner = config.get("dagshub.repo_owner", "Julio-analyst")
    repo_name = config.get("dagshub.repo_name", "gender-voice-detection")
    
    print(f"\n📦 Repository: {repo_owner}/{repo_name}")
    print(f"🔗 DagsHub URL: https://dagshub.com/{repo_owner}/{repo_name}")
    print(f"📊 MLflow UI: https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    
    # Check credentials
    has_credentials = check_dagshub_credentials()
    
    if not has_credentials:
        print("\n" + "⚠️ " * 35)
        print("WARNING: No DagsHub credentials found!")
        print("⚠️ " * 35)
        print("\n📝 To setup DagsHub credentials:")
        print("\n1. Get your DagsHub token from:")
        print("   https://dagshub.com/user/settings/tokens")
        print("\n2. Create a .env file (copy from .env.example):")
        print("   cp .env.example .env")
        print("\n3. Add your token to .env:")
        print("   DAGSHUB_TOKEN=your_token_here")
        print("\n4. Or set environment variables:")
        print("   export DAGSHUB_TOKEN=your_token_here")
        print("   export MLFLOW_TRACKING_USERNAME=Julio-analyst")
        print("   export MLFLOW_TRACKING_PASSWORD=your_token_here")
        print("\n" + "=" * 70)
        return False
    
    # Test connection
    print("\n✅ Credentials configured!")
    print("\n🧪 Testing MLflow connection...")
    
    try:
        import mlflow
        
        mlflow_uri = config.get("mlflow.tracking_uri")
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Try to get experiments
        experiments = mlflow.search_experiments()
        print(f"✅ Connected to MLflow! Found {len(experiments)} experiment(s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPlease check:")
        print("1. Your token is correct")
        print("2. You have access to the repository")
        print("3. Your internet connection")
        return False


def init_dvc():
    """Initialize DVC for data versioning"""
    print("\n" + "=" * 70)
    print("📁 DVC Data Versioning Setup")
    print("=" * 70)
    
    # Check if DVC is installed
    try:
        import dvc  # noqa
        print("✅ DVC is installed")
    except ImportError:
        print("❌ DVC not installed")
        print("\n📝 To install DVC:")
        print("   pip install dvc dvc-s3")
        return False
    
    # Check if DVC is initialized
    if Path(".dvc").exists():
        print("✅ DVC already initialized")
    else:
        print("⚠️  DVC not initialized")
        print("\n📝 To initialize DVC:")
        print("   dvc init")
    
    # Check DVC remote
    config = get_config()
    repo_owner = config.get("dagshub.repo_owner", "Julio-analyst")
    repo_name = config.get("dagshub.repo_name", "gender-voice-detection")
    
    print(f"\n📝 To setup DagsHub DVC remote:")
    print(f"   dvc remote add dagshub https://dagshub.com/{repo_owner}/{repo_name}.dvc")
    print(f"   dvc remote modify dagshub --local auth basic")
    print(f"   dvc remote modify dagshub --local user {repo_owner}")
    print(f"   dvc remote modify dagshub --local password $DAGSHUB_TOKEN")
    
    return True


def main():
    """Main setup function"""
    print("\n" + "🎯 " * 35)
    print("DagsHub MLOps Platform Setup")
    print("🎯 " * 35)
    
    # Setup DagsHub
    dagshub_ok = setup_dagshub()
    
    # Setup DVC
    print()
    dvc_ok = init_dvc()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Setup Summary")
    print("=" * 70)
    print(f"DagsHub Connection: {'✅ Ready' if dagshub_ok else '❌ Not configured'}")
    print(f"DVC Setup: {'✅ Ready' if dvc_ok else '⚠️  Needs setup'}")
    
    if dagshub_ok:
        print("\n🚀 Ready to train models with DagsHub tracking!")
        print("\nNext steps:")
        print("1. Train a model: python src/training/train.py")
        print("2. View experiments: https://dagshub.com/Julio-analyst/gender-voice-detection.mlflow")
    else:
        print("\n⚠️  Please configure DagsHub credentials first")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
