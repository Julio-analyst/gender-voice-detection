"""
Admin Dashboard - MLOps Gender Voice Detection
Features:
- Manual model retraining with custom parameters
- View feedback data
- Model performance comparison
- Training history
"""

import gradio as gr
import pandas as pd
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def load_feedback_data():
    """Load feedback CSV"""
    feedback_file = Path("data/feedback/feedback.csv")
    if feedback_file.exists():
        df = pd.read_csv(feedback_file)
        # Reorder columns for better display
        desired_columns = ['timestamp', 'audio_filename', 'predicted_label', 'actual_label', 'is_correct', 'confidence', 'model_used', 'user_comment']
        # Only keep columns that exist
        available_columns = [col for col in desired_columns if col in df.columns]
        if available_columns:
            df = df[available_columns]
        return df
    else:
        # Return empty dataframe with correct schema
        return pd.DataFrame(columns=[
            'feedback_id', 'timestamp', 'audio_filename', 'predicted_label', 
            'actual_label', 'is_correct', 'confidence', 'model_used', 'user_comment'
        ])

def get_feedback_stats():
    """Get feedback statistics"""
    df = load_feedback_data()
    if len(df) == 0:
        return "📊 **Feedback Statistics**\n\nNo feedback data yet."
    
    total = len(df)
    
    # Count by actual label
    if 'actual_label' in df.columns:
        male_count = len(df[df['actual_label'] == 'Laki-laki'])
        female_count = len(df[df['actual_label'] == 'Perempuan'])
    else:
        male_count = 0
        female_count = 0
    
    # Count correct predictions
    if 'is_correct' in df.columns:
        correct_count = len(df[df['is_correct'] == 'Yes'])
        accuracy = (correct_count / total * 100) if total > 0 else 0
    else:
        correct_count = 0
        accuracy = 0
    
    stats = f"""
    📊 **Feedback Statistics**
    
    - **Total Feedback:** {total}
    - **Male Corrections:** {male_count} ({male_count/total*100:.1f}% if total > 0 else 0)
    - **Female Corrections:** {female_count} ({female_count/total*100:.1f}% if total > 0 else 0)
    - **Correct Predictions:** {correct_count} ({accuracy:.1f}%)
    - **Incorrect Predictions:** {total - correct_count}
    - **Ready for Retrain:** {'✅ Yes' if total >= 20 else f'❌ No (need {20-total} more)'}
    
    **Model Performance from Feedback:**
    """
    
    # Add per-model stats if available
    if 'model_used' in df.columns:
        for model in df['model_used'].unique():
            if pd.notna(model):
                model_df = df[df['model_used'] == model]
                model_correct = len(model_df[model_df['is_correct'] == 'Yes']) if 'is_correct' in df.columns else 0
                model_total = len(model_df)
                model_acc = (model_correct / model_total * 100) if model_total > 0 else 0
                stats += f"\n    - **{model}:** {model_correct}/{model_total} correct ({model_acc:.1f}%)"
    
    return stats

def get_model_metrics():
    """Load model performance metrics"""
    metrics = []
    reports_dir = Path("reports")
    
    if reports_dir.exists():
        for model_dir in sorted(reports_dir.iterdir(), reverse=True):
            if model_dir.is_dir():
                metrics_file = model_dir / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        data['timestamp'] = model_dir.name
                        metrics.append(data)
    
    if metrics:
        df = pd.DataFrame(metrics)
        return df
    else:
        return pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1_score'])

def trigger_training(model_type, epochs, learning_rate, batch_size, use_feedback):
    """Trigger model training with custom parameters"""
    try:
        # Build command
        cmd = [
            sys.executable,
            "src/training/train.py",
            "--use-processed",
            "--model-type", model_type.lower(),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size)
        ]
        
        # Add learning rate if custom
        if learning_rate != 0.001:
            # Note: Need to add --learning-rate arg to train.py
            cmd.extend(["--learning-rate", str(learning_rate)])
        
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent)
        )
        
        if result.returncode == 0:
            # Load metrics from latest report
            latest_report = max(Path("reports").glob(f"{model_type.lower()}_*"), key=os.path.getmtime)
            metrics_file = latest_report / "metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                return f"""
                ✅ **Training Completed Successfully!**
                
                **Model:** {model_type}
                **Epochs:** {epochs}
                **Learning Rate:** {learning_rate}
                **Batch Size:** {batch_size}
                
                📊 **Results:**
                - Accuracy: {metrics.get('test_accuracy', 0)*100:.2f}%
                - Precision: {metrics.get('test_precision', 0)*100:.2f}%
                - Recall: {metrics.get('test_recall', 0)*100:.2f}%
                - F1-Score: {metrics.get('test_f1', 0)*100:.2f}%
                
                Model saved to: models/{model_type.lower()}_production.h5
                """
            else:
                return "✅ Training completed but metrics not found."
        else:
            return f"❌ Training failed:\n\n```\n{result.stderr}\n```"
            
    except Exception as e:
        return f"❌ Error triggering training: {str(e)}"

def clear_feedback():
    """Clear all feedback data"""
    feedback_file = Path("data/feedback/feedback.csv")
    if feedback_file.exists():
        # Backup first
        backup_file = feedback_file.parent / f"feedback_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        import shutil
        shutil.copy(feedback_file, backup_file)
        
        # Clear
        feedback_file.unlink()
        return f"✅ Feedback cleared! Backup saved to: {backup_file.name}"
    else:
        return "ℹ️ No feedback data to clear."

def export_feedback():
    """Export feedback data"""
    df = load_feedback_data()
    if len(df) > 0:
        export_file = Path("data/feedback") / f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(export_file, index=False)
        return f"✅ Feedback exported to: {export_file.name}"
    else:
        return "ℹ️ No feedback data to export."

# Create Gradio Interface
with gr.Blocks(title="Admin Dashboard - MLOps", theme=gr.themes.Soft()) as admin_ui:
    
    gr.Markdown(
        """
        # 🔧 Admin Dashboard - Gender Voice Detection MLOps
        
        **Control Panel untuk Manage Models, Training, dan Feedback**
        
        ⚠️ **Admin Only** - Pastikan Anda memiliki akses admin
        """
    )
    
    with gr.Tabs():
        
        # Tab 1: Model Training
        with gr.Tab("🚀 Model Training"):
            gr.Markdown("### Train Model dengan Custom Parameters")
            
            with gr.Row():
                with gr.Column():
                    model_type = gr.Radio(
                        choices=["LSTM", "RNN", "GRU"],
                        value="LSTM",
                        label="Model Type"
                    )
                    
                    epochs = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Epochs"
                    )
                    
                    learning_rate = gr.Slider(
                        minimum=0.0001,
                        maximum=0.01,
                        value=0.001,
                        step=0.0001,
                        label="Learning Rate"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=8,
                        maximum=128,
                        value=32,
                        step=8,
                        label="Batch Size"
                    )
                    
                    use_feedback = gr.Checkbox(
                        label="Include Feedback Data in Training",
                        value=False,
                        info="Retrain with user feedback corrections"
                    )
                    
                    train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                
                with gr.Column():
                    training_output = gr.Markdown(
                        value="*Training results will appear here...*"
                    )
            
            train_btn.click(
                fn=trigger_training,
                inputs=[model_type, epochs, learning_rate, batch_size, use_feedback],
                outputs=[training_output]
            )
        
        # Tab 2: Feedback Management
        with gr.Tab("💬 Feedback Management"):
            gr.Markdown("### User Feedback Data")
            
            with gr.Row():
                with gr.Column(scale=1):
                    feedback_stats = gr.Markdown(value=get_feedback_stats())
                    
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")
                    export_btn = gr.Button("📥 Export Feedback", size="sm")
                    clear_btn = gr.Button("🗑️ Clear Feedback (with backup)", size="sm", variant="stop")
                    
                    action_result = gr.Markdown()
                
                with gr.Column(scale=2):
                    feedback_table = gr.Dataframe(
                        value=load_feedback_data(),
                        label="Feedback History",
                        interactive=False
                    )
            
            refresh_stats_btn.click(
                fn=get_feedback_stats,
                outputs=[feedback_stats]
            )
            
            refresh_stats_btn.click(
                fn=load_feedback_data,
                outputs=[feedback_table]
            )
            
            export_btn.click(
                fn=export_feedback,
                outputs=[action_result]
            )
            
            clear_btn.click(
                fn=clear_feedback,
                outputs=[action_result]
            )
        
        # Tab 3: Model Performance
        with gr.Tab("📊 Model Performance"):
            gr.Markdown("### Model Metrics Comparison")
            
            metrics_table = gr.Dataframe(
                value=get_model_metrics(),
                label="Training History",
                interactive=False
            )
            
            refresh_metrics_btn = gr.Button("🔄 Refresh Metrics")
            
            refresh_metrics_btn.click(
                fn=get_model_metrics,
                outputs=[metrics_table]
            )
        
        # Tab 4: System Info
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown(
                """
                ### System Information
                
                **Project Structure:**
                ```
                C:/mlops/
                ├── data/
                │   ├── raw_wav/          # Audio files
                │   ├── processed/        # MFCC features
                │   └── feedback/         # User feedback
                ├── models/               # Trained models
                ├── reports/              # Training reports
                └── src/                  # Source code
                ```
                
                **Models:**
                - LSTM: Long Short-Term Memory
                - RNN: Recurrent Neural Network
                - GRU: Gated Recurrent Unit
                
                **Dataset:**
                - Total Samples: 1,052
                - Male: 478 (45.4%)
                - Female: 574 (54.6%)
                
                **Training:**
                - Framework: TensorFlow/Keras
                - Optimizer: Adam
                - Loss: Binary Crossentropy
                - Metrics: Accuracy, Precision, Recall, F1
                
                **Auto-Retrain Trigger:**
                - Minimum feedback: 20 samples
                - Current trigger: Manual from this dashboard
                """
            )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 Launching Admin Dashboard...")
    print("="*60)
    print("\n⚠️  Admin Access Only")
    print("📍 URL: http://127.0.0.1:7861")
    print("="*60 + "\n")
    
    admin_ui.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )
