---
title: Gender Voice Detection MLOps
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.7.1
app_file: start_ui.py
pinned: false
license: mit
---

# 🎤 Gender Voice Detection - MLOps Project

Deep Learning system untuk klasifikasi gender berdasarkan suara menggunakan LSTM, RNN, dan GRU models.

## 🎯 Features

- 🎙️ **Upload atau Record Audio** - Supports all formats (WAV, MP3, M4A, OPUS)
- 🤖 **3 Deep Learning Models** - LSTM, RNN, GRU (pilih yang terbaik!)
- 📊 **High Accuracy** - Up to 100% accuracy with GRU model
- 💬 **Feedback System** - Help improve the model dengan feedback
- 🚀 **Real-time Prediction** - Instant gender classification

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LSTM** | 95.16% | 95.83% | 94.79% | 95.31% |
| **RNN** | 85.85% | 86.27% | 85.42% | 85.84% |
| **GRU** | 100.00% | 100.00% | 100.00% | 100.00% |

## 🔧 Technology Stack

- **Deep Learning**: TensorFlow/Keras
- **Audio Processing**: Librosa, MFCC features
- **Web Interface**: Gradio
- **MLOps**: DagsHub (MLflow tracking), GitHub Actions (CI/CD)
- **Deployment**: Hugging Face Spaces

## 📊 Dataset

- Total Samples: 1,052 audio files
- Balanced Dataset: 478 female, 478 male
- Features: 13 MFCC coefficients
- Augmented with feedback data

## 🎓 How It Works

1. **Audio Preprocessing**: Clean noise, normalize volume
2. **Feature Extraction**: Extract 13 MFCC features
3. **Model Prediction**: Deep learning model classifies gender
4. **Confidence Score**: Returns probability (0-100%)

## 🚀 Usage

1. Upload audio file atau record your voice
2. Select model (LSTM/RNN/GRU)
3. Click "Prediksi Gender"
4. View results with confidence score
5. (Optional) Submit feedback

## 📝 Technical Details

- **Input**: Audio file (any format, 3-10 seconds recommended)
- **Processing**: MFCC extraction (13 coefficients, max 469 frames)
- **Output**: Gender (Male/Female) + Confidence (%)
- **Training**: Balanced dataset, 50 epochs, Adam optimizer

## 🔗 Links

- **GitHub**: [gender-voice-detection](https://github.com/Julio-analyst/gender-voice-detection)
- **DagsHub**: [Experiment Tracking](https://dagshub.com/Julio-analyst/gender-voice-detection.mlflow)
- **Documentation**: [Project Docs](https://github.com/Julio-analyst/gender-voice-detection#readme)

## 👥 Team

Built by MLOps Team - Sains Data ITERA

## 📄 License

MIT License - Free to use and modify
