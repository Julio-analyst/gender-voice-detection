"""
Gradio User Interface - Bahasa Indonesia
Interface untuk deteksi gender dari audio
"""

import gradio as gr
import numpy as np
import requests
from pathlib import Path
import sys
import os
from datetime import datetime

# Handle imports
try:
    from ..utils.config import get_config
    from ..preprocessing.audio_cleaner import AudioCleaner
    from ..preprocessing.feature_extractor import MFCCExtractor
    from ..training.model import create_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config import get_config
    from src.preprocessing.audio_cleaner import AudioCleaner
    from src.preprocessing.feature_extractor import MFCCExtractor
    from src.training.model import create_model

# Initialize config
config = get_config()

# Load models globally
_models = {}


def load_models():
    """Load all available models"""
    import tensorflow as tf
    from tensorflow import keras
    
    global _models
    
    print("📦 Loading models...")
    
    for model_type in ['lstm', 'rnn', 'gru']:
        model_path = Path(f'models/{model_type}_production.h5')
        
        if model_path.exists():
            try:
                _models[model_type] = keras.models.load_model(str(model_path))
                print(f"   ✅ {model_type.upper()} model loaded")
            except Exception as e:
                print(f"   ⚠️  Error loading {model_type.upper()}: {str(e)}")
        else:
            print(f"   ⚠️  {model_type.upper()} model not found")
    
    if len(_models) == 0:
        print("   ❌ No models loaded!")
    else:
        print(f"   ✅ {len(_models)} model(s) loaded successfully")


def predict_audio(audio_file, model_choice):
    """
    Predict gender from audio file
    
    Args:
        audio_file: Path to uploaded audio file
        model_choice: Selected model type
        
    Returns:
        Tuple of (result_text, confidence_text, probabilities_text)
    """
    try:
        if audio_file is None:
            return "❌ Silakan upload file audio terlebih dahulu", "", ""
        
        model_type = model_choice.lower()
        
        if model_type not in _models:
            return f"❌ Model {model_choice} tidak tersedia", "", ""
        
        # Get audio file path
        audio_path = audio_file if isinstance(audio_file, str) else audio_file.name
        
        print(f"🎵 Processing audio: {audio_path}")
        
        # Extract MFCC features
        extractor = MFCCExtractor()
        mfcc_features = extractor.extract_from_file(audio_path, preprocess=True)
        
        # Add batch dimension
        mfcc_features = np.expand_dims(mfcc_features, axis=0)
        
        # Get model
        model = _models[model_type]
        
        # Make prediction
        prediction_prob = model.predict(mfcc_features, verbose=0)[0][0]
        
        # Determine gender
        is_perempuan = prediction_prob > 0.5
        predicted_gender = "Perempuan 👩" if is_perempuan else "Laki-laki 👨"
        
        # Calculate confidence
        confidence = abs(prediction_prob - 0.5) * 2  # Scale to 0-1
        
        # Format results
        result_text = f"## Hasil Prediksi: **{predicted_gender}**"
        
        confidence_text = f"### Tingkat Keyakinan: **{confidence:.1%}**"
        
        probabilities_text = f"""
### Detail Probabilitas:
- **Laki-laki**: {(1 - prediction_prob):.2%}
- **Perempuan**: {prediction_prob:.2%}

*Model yang digunakan: {model_choice}*
"""
        
        print(f"✅ Prediction: {predicted_gender} (confidence: {confidence:.1%})")
        
        # Store for feedback
        global _last_prediction
        _last_prediction = {
            'audio_file': Path(audio_path).name,
            'prediction': predicted_gender.split()[0],  # Remove emoji
            'confidence': confidence,
            'model_type': model_type,
            'prob_perempuan': float(prediction_prob)
        }
        
        return result_text, confidence_text, probabilities_text
        
    except Exception as e:
        error_msg = f"❌ Terjadi kesalahan: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, "", ""


_last_prediction = None


def submit_feedback(actual_gender, feedback_comment):
    """
    Submit user feedback
    
    Args:
        actual_gender: Actual gender selected by user
        feedback_comment: Optional user comment
        
    Returns:
        Feedback submission status message
    """
    global _last_prediction
    
    try:
        if _last_prediction is None:
            return "❌ Tidak ada prediksi untuk diberikan feedback. Silakan lakukan prediksi terlebih dahulu."
        
        # Prepare feedback data
        feedback_data = {
            'audio_filename': _last_prediction['audio_file'],
            'predicted_label': _last_prediction['prediction'],
            'actual_label': actual_gender,
            'model_type': _last_prediction['model_type'],
            'confidence': _last_prediction['confidence'],
            'user_comment': feedback_comment or ""
        }
        
        # Save feedback locally
        feedback_dir = Path('data/feedback')
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing feedback
        import pandas as pd
        feedback_file = feedback_dir / 'feedback.csv'
        
        if feedback_file.exists():
            df = pd.read_csv(feedback_file)
        else:
            df = pd.DataFrame(columns=[
                'feedback_id', 'timestamp', 'audio_filename',
                'predicted_label', 'actual_label', 'model_type',
                'confidence', 'is_correct', 'user_comment'
            ])
        
        # Create new feedback entry
        feedback_id = f"FB_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        is_correct = feedback_data['predicted_label'] == feedback_data['actual_label']
        
        new_feedback = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            **feedback_data,
            'is_correct': is_correct
        }
        
        # Append and save
        df = pd.concat([df, pd.DataFrame([new_feedback])], ignore_index=True)
        df.to_csv(feedback_file, index=False)
        
        # Check if should retrain
        threshold = int(os.getenv('FEEDBACK_THRESHOLD', 20))
        total_feedback = len(df)
        
        print(f"✅ Feedback saved: {feedback_id}")
        print(f"   Total feedback: {total_feedback}/{threshold}")
        
        if is_correct:
            status_msg = "✅ Terima kasih! Prediksi model sudah benar."
        else:
            status_msg = "⚠️ Terima kasih atas koreksinya! Kami akan menggunakan feedback ini untuk meningkatkan akurasi model."
        
        status_msg += f"\n\n📊 Total feedback terkumpul: **{total_feedback}/{threshold}**"
        
        if total_feedback >= threshold:
            status_msg += "\n\n🔔 **Threshold tercapai! Model akan segera di-retrain secara otomatis.**"
        
        # Clear last prediction
        _last_prediction = None
        
        return status_msg
        
    except Exception as e:
        error_msg = f"❌ Gagal menyimpan feedback: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def create_demo():
    """Create Gradio demo interface"""
    
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
    
    with gr.Blocks(css=custom_css, title="Deteksi Gender dari Suara") as demo:
        
        gr.Markdown(
            """
            # 🎤 Deteksi Gender dari Suara
            ### Sistem Deep Learning untuk Klasifikasi Gender Berdasarkan Audio
            
            Upload file audio Anda dan pilih model untuk mendeteksi gender (Laki-laki atau Perempuan).
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Audio")
                
                audio_input = gr.Audio(
                    label="File Audio",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                model_choice = gr.Radio(
                    choices=["LSTM", "RNN", "GRU"],
                    value="LSTM",
                    label="Pilih Model"
                )
                
                predict_btn = gr.Button("🔮 Prediksi Gender", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    ---
                    **Format yang didukung:** WAV, MP3, FLAC, dll.
                    
                    **Tips:**
                    - Pastikan audio memiliki kualitas yang baik
                    - Durasi minimal 3 detik
                    - Hindari noise/background yang terlalu besar
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Hasil Prediksi")
                
                result_output = gr.Markdown(label="Hasil")
                confidence_output = gr.Markdown(label="Confidence")
                probabilities_output = gr.Markdown(label="Probabilities")
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    ## 💬 Berikan Feedback
                    
                    Bantu kami meningkatkan akurasi model dengan memberikan feedback!
                    """
                )
                
                actual_gender = gr.Radio(
                    choices=["Laki-laki", "Perempuan"],
                    label="Gender yang Benar",
                    info="Pilih gender yang sebenarnya dari audio"
                )
                
                feedback_comment = gr.Textbox(
                    label="Komentar (Opsional)",
                    placeholder="Tambahkan komentar jika ada...",
                    lines=3
                )
                
                feedback_btn = gr.Button("📮 Kirim Feedback", variant="secondary")
                
                feedback_status = gr.Markdown()
        
        # Event handlers
        predict_btn.click(
            fn=predict_audio,
            inputs=[audio_input, model_choice],
            outputs=[result_output, confidence_output, probabilities_output]
        )
        
        feedback_btn.click(
            fn=submit_feedback,
            inputs=[actual_gender, feedback_comment],
            outputs=[feedback_status]
        )
        
        gr.Markdown(
            """
            ---
            ### ℹ️ Informasi Model
            
            Model ini menggunakan arsitektur Recurrent Neural Network (RNN, LSTM, GRU) 
            yang dilatih dengan dataset audio untuk mengklasifikasikan gender berdasarkan 
            karakteristik suara.
            
            **Akurasi Model:** ~80-85% (tergantung model yang dipilih)
            
            **Dibuat oleh:** Tim MLOps Gender Voice Detection
            """
        )
    
    return demo


def main():
    """Main function to launch Gradio app"""
    print("="*80)
    print("Gender Voice Detection - Gradio UI")
    print("="*80)
    
    # Load models
    load_models()
    
    # Create demo
    demo = create_demo()
    
    # Launch
    host = os.getenv('GRADIO_SERVER_NAME', '0.0.0.0')
    port = int(os.getenv('GRADIO_SERVER_PORT', 7860))
    share = os.getenv('GRADIO_SHARE', 'false').lower() == 'true'
    
    print(f"\n🚀 Launching Gradio UI on http://{host}:{port}")
    
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True
    )


if __name__ == "__main__":
    main()
