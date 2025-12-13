"""
Launcher Script
Run different components of the MLOps system
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def launch_user_ui():
    """Launch user Gradio interface"""
    from src.ui.app import main
    main()


def launch_admin_panel():
    """Launch admin panel"""
    import subprocess
    subprocess.run([sys.executable, "admin_dashboard.py"])


def launch_api():
    """Launch FastAPI prediction API"""
    from src.api.predict import main
    # Note: This won't work directly, use: uvicorn src.api.predict:app --reload
    print("To launch API, run:")
    print("  uvicorn src.api.predict:app --host 0.0.0.0 --port 8000 --reload")


def run_auto_retrain():
    """Run auto-retrain process"""
    from src.training.auto_retrain import main
    main()


def show_menu():
    """Show interactive menu"""
    print("="*80)
    print("Gender Voice Detection MLOps - Launcher")
    print("="*80)
    print("\nPilih komponen yang ingin dijalankan:\n")
    print("1. 🎤 User Interface (Gradio) - Port 7860")
    print("2. 🔐 Admin Panel (Gradio) - Port 7861")
    print("3. 🚀 API Server (FastAPI) - Port 8000")
    print("4. 🔄 Auto-Retrain Module")
    print("5. ℹ️  Show System Info")
    print("0. ❌ Exit")
    print("\n" + "="*80)
    
    choice = input("\nMasukkan pilihan (0-5): ").strip()
    
    if choice == "1":
        print("\n🚀 Launching User Interface...")
        launch_user_ui()
    elif choice == "2":
        print("\n🚀 Launching Admin Panel...")
        launch_admin_panel()
    elif choice == "3":
        print("\n🚀 Launching API Server...")
        launch_api()
    elif choice == "4":
        print("\n🚀 Running Auto-Retrain...")
        run_auto_retrain()
    elif choice == "5":
        show_system_info()
        input("\nPress Enter to continue...")
        show_menu()
    elif choice == "0":
        print("\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\n❌ Pilihan tidak valid!")
        show_menu()


def show_system_info():
    """Display system information"""
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    
    # Check models
    print("\n📦 Available Models:")
    for model_type in ['lstm', 'rnn', 'gru']:
        model_path = Path(f'models/{model_type}_production.h5')
        status = "✅" if model_path.exists() else "❌"
        print(f"   {status} {model_type.upper()}: {model_path}")
    
    # Check feedback
    feedback_file = Path('data/feedback/feedback.csv')
    if feedback_file.exists():
        import pandas as pd
        df = pd.read_csv(feedback_file)
        print(f"\n💬 Feedback Data:")
        print(f"   Total: {len(df)}")
        print(f"   Threshold: {os.getenv('FEEDBACK_THRESHOLD', 20)}")
    else:
        print(f"\n💬 Feedback Data: No data yet")
    
    # Check reports
    reports_dir = Path('reports')
    if reports_dir.exists():
        report_folders = list(reports_dir.glob('*/'))
        print(f"\n📊 Reports: {len(report_folders)} folders")
    
    # Environment
    print(f"\n⚙️  Environment:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   MLflow URI: {os.getenv('MLFLOW_TRACKING_URI', 'Not set')}")
    
    print("\n" + "="*80)


def main():
    """Main launcher function"""
    if len(sys.argv) > 1:
        # CLI mode
        command = sys.argv[1]
        
        if command == "ui":
            launch_user_ui()
        elif command == "admin":
            launch_admin_panel()
        elif command == "api":
            launch_api()
        elif command == "retrain":
            run_auto_retrain()
        elif command == "info":
            show_system_info()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  ui       - Launch user interface")
            print("  admin    - Launch admin panel")
            print("  api      - Show API launch instructions")
            print("  retrain  - Run auto-retrain")
            print("  info     - Show system info")
    else:
        # Interactive mode
        show_menu()


if __name__ == "__main__":
    main()
