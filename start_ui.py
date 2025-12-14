"""
Simple UI Launcher - Real Model Predictions
Launches Gradio UI with trained models
"""

import gradio as gr
import numpy as np
from pathlib import Path
import sys
import os

# Add FFmpeg to PATH for Gradio audio processing
ffmpeg_paths = [
    r"C:\ProgramData\chocolatey\bin",
    r"C:\Program Files\FFmpeg\bin",
    r"C:\ffmpeg\bin",
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-full_build\bin")
]

for ffmpeg_path in ffmpeg_paths:
    if os.path.exists(ffmpeg_path):
        if ffmpeg_path not in os.environ['PATH']:
            os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']
            print(f"✅ Added FFmpeg to PATH: {ffmpeg_path}")
        break

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

# Global state untuk track last prediction
last_prediction = {
    'audio_file': None,
    'model': None,
    'prediction': None,
    'confidence': None,
    'timestamp': None
}

def predict_gender(audio_file, model_choice):
    """Real prediction using trained models - Supports all audio formats via librosa"""
    if audio_file is None:
        return "❌ Silakan upload file audio terlebih dahulu", gr.update(visible=False)
    
    try:
        # Preprocess audio directly - librosa supports WAV, MP3, M4A, OGG, FLAC, etc.
        cleaner = AudioCleaner()
        extractor = MFCCExtractor()
        
        # Clean and extract features (librosa handles format detection)
        audio, sr = cleaner.process_audio(audio_file)
        features = extractor.extract_from_array(audio, sr)
        
        # Reshape for model input (batch_size, time_steps, n_features)
        # MFCCExtractor returns shape (time_steps, n_mfcc) -> need (1, time_steps, n_mfcc)
        # Just add batch dimension at axis 0
        features = np.expand_dims(features, axis=0)
        
        print(f"✓ Feature shape: {features.shape}")
        
        # Get model
        model = models.get(model_choice)
        if model is None:
            return f"❌ Model {model_choice} not loaded", gr.update(visible=False)
        
        # Predict
        prediction_proba = model.predict(features, verbose=0)[0][0]
        prediction = "Perempuan" if prediction_proba >= 0.5 else "Laki-laki"
        confidence = prediction_proba if prediction_proba >= 0.5 else (1 - prediction_proba)
        
        result = f"""
        ### 🎯 Hasil Prediksi
        
        **Model:** {model_choice}  
        **Prediksi Gender:** {prediction}  
        **Confidence:** {confidence*100:.2f}%
        
        ---
        *Model: {model_choice} | File: {Path(audio_file).name}*
        """
        
        # Auto-save prediction to history
        from datetime import datetime
        import pandas as pd
        
        # Update global state
        global last_prediction
        last_prediction = {
            'audio_file': Path(audio_file).name,
            'model': model_choice,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        # Auto-save to prediction history
        history_file = Path("data/feedback/prediction_history.csv")
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        history_data = {
            'timestamp': last_prediction['timestamp'],
            'audio_filename': last_prediction['audio_file'],
            'model_used': model_choice,
            'predicted_label': prediction,
            'confidence': f"{confidence*100:.2f}%"
        }
        
        if history_file.exists():
            df = pd.read_csv(history_file)
            df = pd.concat([df, pd.DataFrame([history_data])], ignore_index=True)
        else:
            df = pd.DataFrame([history_data])
        
        df.to_csv(history_file, index=False)
        
        # Show feedback section
        return result, gr.update(visible=True)
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ Error: {str(e)}\n\n```\n{error_detail}\n```", gr.update(visible=False)


def submit_feedback(actual_gender, comment):
    """Save user feedback with complete prediction data"""
    import pandas as pd
    from datetime import datetime
    
    global last_prediction
    
    # Validation: Check if there's a recent prediction
    if not last_prediction or not last_prediction.get('audio_file'):
        return """❌ **Error: Tidak ada prediksi sebelumnya**
        
Silakan lakukan prediksi dulu sebelum memberikan feedback!
        
**Langkah:**
1. Upload audio file
2. Klik "Prediksi Gender"
3. Kemudian isi feedback
        """
    
    feedback_file = Path("data/feedback/feedback.csv")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create feedback with complete data
    feedback_data = {
        'feedback_id': f"FB_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'audio_filename': last_prediction['audio_file'],
        'predicted_label': last_prediction['prediction'],
        'actual_label': actual_gender,
        'model_used': last_prediction['model'],
        'confidence': f"{last_prediction['confidence']*100:.2f}%",
        'is_correct': 'Yes' if last_prediction['prediction'] == actual_gender else 'No',
        'user_comment': comment or ''
    }
    
    if feedback_file.exists():
        df = pd.read_csv(feedback_file)
        df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
    else:
        df = pd.DataFrame([feedback_data])
    
    df.to_csv(feedback_file, index=False)
    
    # Prepare summary
    summary = f"""✅ **Terima kasih! Feedback Anda telah disimpan.**
    
**Data yang tersimpan:**
- Audio: {last_prediction['audio_file']}
- Prediksi Model: {last_prediction['prediction']}
- Label Sebenarnya: {actual_gender}
- Akurasi: {'✅ Benar' if feedback_data['is_correct'] == 'Yes' else '❌ Salah'}

📊 Total feedback: {len(df)}
    """
    
    # Don't reset last_prediction immediately - keep it for potential re-submit
    
    return summary


# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gr-button-primary {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border: none;
}
.gr-button-secondary {
    background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    border: none;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Deteksi Gender dari Suara") as demo:
    
    gr.Markdown(
        """
        # 🎤 Deteksi Gender dari Suara
        ### Sistem Deep Learning untuk Klasifikasi Gender Berdasarkan Audio
        
        Upload file audio Anda dan pilih model untuk mendeteksi gender (Laki-laki atau Perempuan).
        
        💡 **Mendukung semua format audio** - MP3, M4A, OPUS, WAV akan auto-convert
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Input Audio")
            audio_input = gr.Audio(
                label="Upload Audio (Semua Format) atau Rekam Suara",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            model_choice = gr.Radio(
                choices=["LSTM", "RNN", "GRU"],
                value="LSTM",
                label="Pilih Model Deep Learning"
            )
            
            predict_btn = gr.Button("🔍 Prediksi Gender", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### 📊 Hasil Prediksi")
            prediction_output = gr.Markdown(value="*Hasil prediksi akan muncul di sini...*")
    
    # Feedback section (controlled by visibility state)
    with gr.Column(visible=False) as feedback_box:
        gr.Markdown("### 💬 Berikan Feedback")
        gr.Markdown("Bantu kami meningkatkan akurasi model dengan memberikan feedback!")
        
        actual_gender = gr.Radio(
            choices=["Laki-laki", "Perempuan"],
            label="Gender yang Sebenarnya"
        )
        comment_input = gr.Textbox(
            label="Komentar (Opsional)",
            placeholder="Tambahkan komentar Anda di sini...",
            lines=3
        )
        feedback_btn = gr.Button("📨 Kirim Feedback", variant="secondary")
        feedback_result = gr.Markdown()
    
    gr.Markdown(
        """
        ---
        ### 📋 Informasi
        - **Model:** LSTM, RNN, dan GRU untuk klasifikasi gender
        - **Input:** Semua format audio (MP3, M4A, OPUS, WAV, dll)
        - **Output:** Prediksi gender dengan confidence score
        - **Auto-convert:** File non-WAV akan otomatis dikonversi menggunakan FFmpeg
        
        **Catatan:** 
        - Feedback Anda akan digunakan untuk auto-retraining model ketika mencapai 20+ feedback
        """
    )
    
    # Event handlers
    predict_btn.click(
        fn=predict_gender,
        inputs=[audio_input, model_choice],
        outputs=[prediction_output, feedback_box]
    )
    
    feedback_btn.click(
        fn=submit_feedback,
        inputs=[actual_gender, comment_input],
        outputs=[feedback_result]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Gender Voice Detection UI...")
    print("="*60)
    
    if models:
        print(f"\n✅ Loaded {len(models)} models: {', '.join(models.keys())}")
    else:
        print("\n⚠️  Models not loaded - running in demo mode")
    
    # Disable sagemaker check to prevent AWS connection issues
    import os
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces for Docker compatibility
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
