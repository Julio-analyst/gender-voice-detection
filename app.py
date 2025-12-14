"""
Hugging Face Spaces - Gender Voice Detection
Gradio app for gender classification from voice
"""

import gradio as gr
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime

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


def predict_gender(audio_file, model_choice):
    """Predict gender from audio file"""
    if audio_file is None:
        return "❌ Please upload an audio file first", gr.update(visible=False)
    
    try:
        # Preprocess audio
        cleaner = AudioCleaner()
        extractor = MFCCExtractor()
        
        # Clean and extract features
        audio, sr = cleaner.process_audio(audio_file)
        features = extractor.extract_from_array(audio, sr)
        
        # Reshape for model input (add batch dimension)
        features = np.expand_dims(features, axis=0)
        
        # Get model
        model = models.get(model_choice)
        if model is None:
            return f"❌ Model {model_choice} not loaded", gr.update(visible=False)
        
        # Predict
        prediction_proba = model.predict(features, verbose=0)[0][0]
        prediction = "Female" if prediction_proba >= 0.5 else "Male"
        confidence = prediction_proba if prediction_proba >= 0.5 else (1 - prediction_proba)
        
        result = f"""
        ### 🎯 Prediction Result
        
        **Model:** {model_choice}  
        **Predicted Gender:** {prediction}  
        **Confidence:** {confidence*100:.2f}%
        
        ---
        *Model: {model_choice} | File: {Path(audio_file).name}*
        """
        
        # Show feedback section
        return result, gr.update(visible=True)
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}", gr.update(visible=False)


def submit_feedback(audio_file, model_choice, actual_gender, comment):
    """Submit user feedback"""
    if audio_file is None:
        return "⚠️ No prediction to give feedback on"
    
    try:
        import pandas as pd
        
        # Prepare feedback data
        feedback = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'audio_filename': Path(audio_file).name,
            'model': model_choice,
            'actual_label': actual_gender,
            'comment': comment or ''
        }
        
        # Save to CSV
        feedback_file = 'data/feedback/feedback.csv'
        os.makedirs('data/feedback', exist_ok=True)
        
        if os.path.exists(feedback_file):
            df = pd.read_csv(feedback_file)
            df = pd.concat([df, pd.DataFrame([feedback])], ignore_index=True)
        else:
            df = pd.DataFrame([feedback])
        
        df.to_csv(feedback_file, index=False)
        
        return "✅ Thank you for your feedback! Your input helps improve the model."
        
    except Exception as e:
        return f"❌ Error saving feedback: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Gender Voice Detection", theme=gr.themes.Soft()) as iface:
    gr.Markdown("""
    # 🎤 Deteksi Gender dari Suara
    
    **Sistem Deep Learning untuk Klasifikasi Gender Berdasarkan Audio**
    
    Upload file audio Anda dan pilih model untuk mendeteksi gender (Laki-laki atau Perempuan).
    
    💡 **Mendukung semua format audio** - MP3, M4A, OPUS, WAV akan auto-convert
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📥 Input Audio")
            audio_input = gr.Audio(
                label="Upload Audio (Semua Format) atau Rekam Suara",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            model_selector = gr.Radio(
                choices=["LSTM", "RNN", "GRU"],
                value="LSTM",
                label="Pilih Model Deep Learning"
            )
            
            predict_btn = gr.Button("🔮 Prediksi Gender", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Hasil Prediksi")
            result_output = gr.Markdown()
    
    # Feedback section (hidden by default)
    with gr.Group(visible=False) as feedback_section:
        gr.Markdown("## 💬 Berikan Feedback")
        gr.Markdown("Bantu kami meningkatkan akurasi model dengan memberikan feedback!")
        
        with gr.Row():
            actual_gender_input = gr.Radio(
                choices=["Laki-laki", "Perempuan"],
                label="Gender yang Sebenarnya",
                value="Perempuan"
            )
            comment_input = gr.Textbox(
                label="Komentar (Opsional)",
                placeholder="Tambahkan komentar Anda di sini..."
            )
        
        submit_feedback_btn = gr.Button("📤 Kirim Feedback", variant="secondary")
        feedback_output = gr.Markdown()
    
    # Examples
    gr.Markdown("## 📝 Informasi")
    gr.Markdown("""
    **Cara Penggunaan:**
    1. Upload file audio atau rekam suara Anda
    2. Pilih model (LSTM, RNN, atau GRU)
    3. Klik "Prediksi Gender" untuk mendapatkan hasil
    4. (Opsional) Berikan feedback untuk membantu improve model
    
    **Performa Model:**
    - LSTM: ~95% accuracy
    - RNN: ~86% accuracy  
    - GRU: ~100% accuracy (best)
    
    **Built with ❤️ by MLOps Team**
    """)
    
    # Event handlers
    predict_btn.click(
        fn=predict_gender,
        inputs=[audio_input, model_selector],
        outputs=[result_output, feedback_section]
    )
    
    submit_feedback_btn.click(
        fn=submit_feedback,
        inputs=[audio_input, model_selector, actual_gender_input, comment_input],
        outputs=feedback_output
    )

# Launch app
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
