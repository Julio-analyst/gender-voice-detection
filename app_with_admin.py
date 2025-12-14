"""
Hugging Face Spaces - Gender Voice Detection with Admin Panel
Combined User UI + Complete Admin Dashboard with Training Features
"""

import gradio as gr
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import subprocess
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import preprocessing
from src.preprocessing.audio_cleaner import AudioCleaner
from src.preprocessing.feature_extractor import MFCCExtractor

# Load models on startup
try:
    import keras
    models = {
        'LSTM': keras.models.load_model('models/lstm_production.h5'),
        'RNN': keras.models.load_model('models/rnn_production.h5'),
        'GRU': keras.models.load_model('models/gru_production.h5')
    }
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️  Model loading error: {e}")
    models = {}

# Initialize preprocessing components
cleaner = AudioCleaner()
extractor = MFCCExtractor(use_cleaner=True)

# Feedback storage
FEEDBACK_FILE = Path("data/feedback/feedback.csv")
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

# Admin credentials (CHANGE THIS in production!)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "mlops2024")  # SET via HF Secrets!

# ============================================================================
# USER INTERFACE FUNCTIONS
# ============================================================================

def predict_gender(audio_file, model_choice):
    """Predict gender from audio file"""
    try:
        if audio_file is None:
            return "❌ Silakan upload file audio terlebih dahulu", "", ""
        
        # Extract features
        mfcc_features = extractor.extract(audio_file)
        mfcc_features = np.expand_dims(mfcc_features, axis=0)
        
        # Predict
        model = models.get(model_choice)
        if model is None:
            return f"❌ Model {model_choice} tidak tersedia", "", ""
        
        prediction = model.predict(mfcc_features, verbose=0)
        confidence = float(prediction[0][0])
        
        # Determine gender
        if confidence > 0.5:
            gender = "Perempuan"
            conf_pct = confidence * 100
        else:
            gender = "Laki-laki"
            conf_pct = (1 - confidence) * 100
        
        # Format result
        result = f"""
## 🎯 Hasil Prediksi

**Model:** {model_choice}  
**Prediksi Gender:** {gender}  
**Confidence:** {conf_pct:.2f}%
"""
        
        # Info text
        info = f"Model: {model_choice} | File: {Path(audio_file).name}"
        
        return result, gender, f"{conf_pct:.2f}%", info
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""


def submit_feedback(audio_file, predicted_gender, actual_gender, comment, model_info):
    """Save user feedback"""
    try:
        if audio_file is None:
            return "❌ Tidak ada audio untuk feedback"
        
        # Prepare feedback data
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'audio_filename': Path(audio_file).name,
            'predicted_label': predicted_gender,
            'actual_label': actual_gender,
            'is_correct': predicted_gender == actual_gender,
            'comment': comment or "",
            'model_info': model_info
        }
        
        # Save to CSV
        df = pd.DataFrame([feedback])
        if FEEDBACK_FILE.exists():
            df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(FEEDBACK_FILE, mode='w', header=True, index=False)
        
        # Success message
        msg = f"""
## ✅ Terima kasih! Feedback Anda telah disimpan.

**Data yang tersimpan:**
- Audio: {feedback['audio_filename']}
- Prediksi Model: {predicted_gender}
- Label Sebenarnya: {actual_gender}
- Akurasi: {'✅ Benar' if feedback['is_correct'] else '❌ Salah'}

📊 Total feedback: {len(pd.read_csv(FEEDBACK_FILE)) if FEEDBACK_FILE.exists() else 1}
"""
        return msg
        
    except Exception as e:
        return f"❌ Error menyimpan feedback: {str(e)}"


# ============================================================================
# ADMIN DASHBOARD FUNCTIONS (ENHANCED)
# ============================================================================

def load_feedback_data():
    """Load feedback CSV with proper schema"""
    if FEEDBACK_FILE.exists():
        df = pd.read_csv(FEEDBACK_FILE)
        return df
    else:
        return pd.DataFrame(columns=[
            'timestamp', 'audio_filename', 'predicted_label',
            'actual_label', 'is_correct', 'comment', 'model_info'
        ])


def get_feedback_stats():
    """Get detailed feedback statistics"""
    df = load_feedback_data()
    if len(df) == 0:
        return "📊 **Feedback Statistics**\n\nNo feedback data yet."
    
    total = len(df)
    
    # Count by actual label
    male_count = len(df[df['actual_label'] == 'Laki-laki'])
    female_count = len(df[df['actual_label'] == 'Perempuan'])
    
    # Count correct predictions (handle both boolean and string)
    if 'is_correct' in df.columns:
        # Convert to boolean - handle True/False strings and booleans
        correct_mask = df['is_correct'].astype(str).str.lower().isin(['true', '1', 'yes'])
        correct_count = int(correct_mask.sum())
    else:
        correct_count = 0
    
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    # Calculate percentages safely
    male_pct = (male_count/total*100) if total > 0 else 0
    female_pct = (female_count/total*100) if total > 0 else 0
    
    stats = f"""
📊 **Feedback Statistics**

- **Total Feedback:** {total}
- **Male Corrections:** {male_count} ({male_pct:.1f}%)
- **Female Corrections:** {female_count} ({female_pct:.1f}%)
- **Correct Predictions:** {correct_count} ({accuracy:.1f}%)
- **Incorrect Predictions:** {total - correct_count}
- **Ready for Retrain:** {'✅ Yes' if total >= 20 else f'❌ No (need {20-total} more)'}
"""
    return stats


def get_model_metrics():
    """Load model performance metrics from reports"""
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
        return pd.DataFrame(metrics)
    else:
        return pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1_score'])


def trigger_training(model_type, epochs, learning_rate, batch_size):
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
    """Clear all feedback data with backup"""
    if FEEDBACK_FILE.exists():
        # Backup first
        backup_file = FEEDBACK_FILE.parent / f"feedback_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy(FEEDBACK_FILE, backup_file)
        
        # Clear
        FEEDBACK_FILE.unlink()
        return f"✅ Feedback cleared! Backup saved to: {backup_file.name}"
    else:
        return "ℹ️ No feedback data to clear."


def export_feedback_csv():
    """Export feedback data as CSV"""
    df = load_feedback_data()
    if len(df) > 0:
        export_file = FEEDBACK_FILE.parent / f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(export_file, index=False)
        return str(export_file)
    else:
        return None


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# User Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Gender Voice Detection") as user_ui:
    gr.Markdown("""
    # 🎤 Deteksi Gender dari Suara
    
    Sistem Deep Learning untuk Klasifikasi Gender Berdasarkan Audio
    
    Upload file audio Anda dan pilih model untuk mendeteksi gender (Laki-laki atau Perempuan).
    
    💡 **Mendukung semua format audio** - MP3, M4A, OPUS, WAV akan auto-convert
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎵 Input Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            model_choice = gr.Radio(
                choices=["LSTM", "RNN", "GRU"],
                value="LSTM",
                label="Pilih Model Deep Learning"
            )
            
            predict_btn = gr.Button("🎯 Prediksi Gender", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            result_output = gr.Markdown(label="📊 Hasil Prediksi")
            model_info = gr.Textbox(label="Model Info", visible=False)
    
    # Hidden outputs for feedback
    predicted_gender = gr.Textbox(visible=False)
    confidence = gr.Textbox(visible=False)
    
    # Feedback section
    gr.Markdown("---")
    gr.Markdown("## 💬 Berikan Feedback")
    gr.Markdown("Bantu kami meningkatkan akurasi model dengan memberikan feedback!")
    
    with gr.Row():
        actual_gender = gr.Radio(
            choices=["Laki-laki", "Perempuan"],
            label="Gender yang Sebenarnya"
        )
        feedback_comment = gr.Textbox(
            label="Komentar (Optional)",
            placeholder="Tambahkan komentar Anda di sini..."
        )
    
    submit_feedback_btn = gr.Button("📤 Kirim Feedback", variant="secondary")
    feedback_result = gr.Markdown()
    
    # Event handlers
    predict_btn.click(
        predict_gender,
        inputs=[audio_input, model_choice],
        outputs=[result_output, predicted_gender, confidence, model_info]
    )
    
    submit_feedback_btn.click(
        submit_feedback,
        inputs=[audio_input, predicted_gender, actual_gender, feedback_comment, model_info],
        outputs=[feedback_result]
    )


# Admin Dashboard - FULL FEATURED (NO AUTH)
with gr.Blocks(theme=gr.themes.Soft(), title="Admin Dashboard") as admin_ui:
    gr.Markdown("""
    # 🔧 Admin Dashboard - Gender Voice Detection MLOps
    
    **Control Panel untuk Model Training, Monitoring & Analytics**
    """)
    
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
                    
                    train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                
                with gr.Column():
                    training_output = gr.Markdown(
                        value="*Training results will appear here...*"
                    )
            
            train_btn.click(
                fn=trigger_training,
                inputs=[model_type, epochs, learning_rate, batch_size],
                outputs=[training_output]
            )
        
        # Tab 2: Feedback Management
        with gr.Tab("💬 Feedback Management"):
            gr.Markdown("### User Feedback Data & Analytics")
            
            with gr.Row():
                with gr.Column(scale=1):
                    feedback_stats = gr.Markdown(value=get_feedback_stats())
                    
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")
                    export_csv_btn = gr.Button("📥 Export Feedback CSV", size="sm")
                    clear_btn = gr.Button("🗑️ Clear Feedback (with backup)", size="sm", variant="stop")
                    
                    action_result = gr.Markdown()
                    export_file = gr.File(label="Download CSV")
                
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
            
            export_csv_btn.click(
                fn=export_feedback_csv,
                outputs=[export_file]
            )
            
            clear_btn.click(
                fn=clear_feedback,
                outputs=[action_result]
            )
        
        # Tab 3: Model Performance
        with gr.Tab("📊 Model Performance"):
            gr.Markdown("### Training History & Metrics Comparison")
            
            metrics_table = gr.Dataframe(
                value=get_model_metrics(),
                label="All Training Sessions",
                interactive=False
            )
            
            refresh_metrics_btn = gr.Button("🔄 Refresh Metrics")
            
            refresh_metrics_btn.click(
                fn=get_model_metrics,
                outputs=[metrics_table]
            )
        
        # Tab 4: System Info
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown(f"""
            ### System Information
            
            **Project Structure:**
            ```
            {Path(__file__).parent}/
            ├── data/
            │   ├── raw_wav/          # Audio files
            │   ├── processed/        # MFCC features
            │   └── feedback/         # User feedback
            ├── models/               # Trained models ({len(models)} loaded)
            ├── reports/              # Training reports
            └── src/                  # Source code
            ```
            
            **Loaded Models:**
            {chr(10).join([f'- {name}: Ready' for name in models.keys()])}
            
            **Dataset Info:**
            - Total Samples: 1,052
            - Male: 478 (45.4%)
            - Female: 574 (54.6%)
            
            **Training Config:**
            - Framework: TensorFlow/Keras
            - Optimizer: Adam
            - Loss: Binary Crossentropy
            - Metrics: Accuracy, Precision, Recall, F1
            
            **Auto-Retrain Trigger:**
            - Minimum feedback: 20 samples
            - Current status: Manual trigger from Training tab
            
            **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)
    
    # Auto-load data on page load
    admin_ui.load(
        fn=lambda: (get_feedback_stats(), load_feedback_data(), get_model_metrics()),
        outputs=[feedback_stats, feedback_table, metrics_table]
    )


# Combined App with Tabs
demo = gr.TabbedInterface(
    [user_ui, admin_ui],
    ["🎤 User Interface", "🔐 Admin Dashboard"],
    title="Gender Voice Detection MLOps"
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
