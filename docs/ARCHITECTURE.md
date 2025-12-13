# 🏗️ Arsitektur Sistem MLOps - Gender Voice Detection

**Last Updated:** December 14, 2025  
**Status:** ✅ Production Ready

---

## 📊 Diagram Arsitektur Keseluruhan

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐        ┌──────────────────┐                   │
│  │  Gradio Web UI   │        │   FastAPI REST   │                   │
│  │  (Port 7860)     │        │   API (Port 800) │                   │
│  │                  │        │                   │                   │
│  │ • Audio Upload   │        │ • /predict       │                   │
│  │ • Model Select   │        │ • /feedback      │                   │
│  │ • Live Record    │        │ • /health        │                   │
│  │ • Feedback Form  │        │ • /models/list   │                   │
│  └────────┬─────────┘        └────────┬─────────┘                   │
│           │                           │                              │
└───────────┼───────────────────────────┼──────────────────────────────┘
            │                           │
            └───────────┬───────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                               │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                  PREDICTION PIPELINE                           │  │
│  │                                                                 │  │
│  │  Input Audio  →  Preprocessing  →  Inference  →  Post-process │  │
│  │     (Any)         (MFCC)           (Models)      (Confidence)  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐    │
│  │ Audio Cleaner   │  │ MFCC Extractor  │  │  Model Manager   │    │
│  │                 │  │                  │  │                  │    │
│  │ • Noise Reduce  │  │ • 13 MFCC Coef  │  │ • Load Models    │    │
│  │ • RMS Normalize │  │ • 16kHz SR      │  │ • Model Switch   │    │
│  │ • Preemphasis   │  │ • Hop: 512      │  │ • Version Ctrl   │    │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────────┐
│                       MODEL LAYER (Deep Learning)                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│   │  LSTM Model  │     │  RNN Model   │     │  GRU Model   │         │
│   │  (100% Acc)  │     │ (85.85% Acc) │     │  (100% Acc)  │         │
│   ├──────────────┤     ├──────────────┤     ├──────────────┤         │
│   │ Input: (T,13)│     │ Input: (T,13)│     │ Input: (T,13)│         │
│   │ LSTM: 64     │     │ RNN: 64      │     │ GRU: 64      │         │
│   │ Dense: 32    │     │ Dense: 32    │     │ Dense: 32    │         │
│   │ Output: 1    │     │ Output: 1    │     │ Output: 1    │         │
│   │ Sigmoid      │     │ Sigmoid      │     │ Sigmoid      │         │
│   └──────────────┘     └──────────────┘     └──────────────┘         │
│                                                                         │
│   Training Config:                                                     │
│   • Optimizer: Adam (lr=0.001)                                        │
│   • Loss: Binary Crossentropy                                         │
│   • Metrics: Accuracy, Precision, Recall                              │
│   • Epochs: 50 (Early Stopping patience=10)                           │
│   • Batch Size: 32                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────────┐
│                          DATA LAYER                                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────┐  ┌────────────────┐  ┌─────────────────────┐      │
│  │   Raw Audio   │  │   Processed    │  │   Feedback Data     │      │
│  │  data/raw_wav/│  │ data/processed/│  │ data/feedback.csv   │      │
│  ├───────────────┤  ├────────────────┤  ├─────────────────────┤      │
│  │ • 100 files   │  │ • 1,052 samples│  │ • User corrections  │      │
│  │ • WAV 16kHz   │  │ • MFCC features│  │ • Confidence scores │      │
│  │ • Mono        │  │ • Labels (0/1) │  │ • Timestamps        │      │
│  │ • Male: 50    │  │ • Metadata JSON│  │ • Auto-retrain @20+ │      │
│  │ • Female: 50  │  │                 │  │                     │      │
│  └───────────────┘  └────────────────┘  └─────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────────┐
│                    EXPERIMENT TRACKING LAYER                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                      MLflow + DagsHub                         │    │
│  │                                                                │    │
│  │  • Experiment Logging       • Hyperparameter Tuning          │    │
│  │  • Model Versioning         • Metrics Comparison             │    │
│  │  • Artifact Storage         • Dataset Versioning             │    │
│  │  • Model Registry           • Collaborative Tracking         │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────────┐
│                    CI/CD & AUTOMATION LAYER                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    GitHub Actions Workflows                     │  │
│  │                                                                 │  │
│  │  1. Auto-Training (train.yml)                                  │  │
│  │     • Trigger: Schedule / Manual / Data Push                   │  │
│  │     • Train LSTM, RNN, GRU                                     │  │
│  │     • Upload artifacts                                         │  │
│  │     • Post metrics to PR                                       │  │
│  │                                                                 │  │
│  │  2. Testing Pipeline (test.yml)                                │  │
│  │     • Unit tests (pytest)                                      │  │
│  │     • Integration tests                                        │  │
│  │     • Coverage reports                                         │  │
│  │                                                                 │  │
│  │  3. Data Validation (data-validation.yml)                      │  │
│  │     • Validate new audio files                                 │  │
│  │     • Format checks                                            │  │
│  │     • Quality metrics                                          │  │
│  │                                                                 │  │
│  │  4. Deployment (deploy.yml)                                    │  │
│  │     • Docker build                                             │  │
│  │     • Push to registry                                         │  │
│  │     • Deploy to cloud                                          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Hugging Face    │  │   Docker Hub    │  │  Cloud Platforms    │  │
│  │   Spaces        │  │                  │  │                     │  │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────────┤  │
│  │ • Gradio App    │  │ • Container     │  │ • AWS EC2/Lambda    │  │
│  │ • Public Access │  │ • Reproducible  │  │ • Google Cloud Run  │  │
│  │ • Free Tier     │  │ • Version Ctrl  │  │ • Azure Container   │  │
│  │ • Auto Deploy   │  │ • Easy Deploy   │  │ • Heroku/Render     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack Detail

### **1. Programming Languages**
- **Python 3.10+** - Main language
- **YAML** - Configuration & GitHub Actions
- **Markdown** - Documentation

### **2. Deep Learning Framework**
- **TensorFlow/Keras 2.x** - Model training & inference
  - LSTM (Long Short-Term Memory)
  - RNN (Recurrent Neural Network)
  - GRU (Gated Recurrent Unit)

### **3. Audio Processing**
- **Librosa 0.10.1** - Audio loading & feature extraction
- **SoundFile 0.12.1** - Audio file I/O
- **NoiseReduce 3.0.0** - Noise reduction
- **NumPy** - Array operations

### **4. Web Frameworks**
- **Gradio 4.7.1** - Interactive web UI
- **FastAPI 0.104.1** - REST API backend
- **Uvicorn 0.24.0** - ASGI server

### **5. MLOps Tools**
- **MLflow 2.8.1** - Experiment tracking
- **DagsHub 0.3.1** - Collaborative ML platform
- **GitHub Actions** - CI/CD automation

### **6. Data Science**
- **Pandas** - Data manipulation
- **Scikit-learn** - Metrics & evaluation
- **Matplotlib** - Visualization

### **7. DevOps & Deployment**
- **Docker** - Containerization (planned)
- **Git** - Version control
- **pytest** - Testing (planned)

---

## 🔄 Data Flow - Complete Pipeline

### **Phase 1: Data Preparation**
```
Raw Audio Files (M4A/OPUS)
    ↓
[FFmpeg Conversion]
    ↓
WAV Files (16kHz, Mono)
    ↓
[Dataset Loader]
    ↓
Segmentation (3-second chunks)
    ↓
1,052 Audio Segments
```

### **Phase 2: Preprocessing**
```
Audio Segment
    ↓
[AudioCleaner]
    ├─ Noise Reduction (noisereduce)
    ├─ RMS Normalization (-20dB target)
    └─ Preemphasis Filter (α=0.97)
    ↓
Clean Audio Array
    ↓
[MFCCExtractor]
    ├─ Sample Rate: 16kHz
    ├─ n_fft: 2048
    ├─ hop_length: 512
    ├─ n_mfcc: 13
    └─ Output: (time_steps, 13)
    ↓
MFCC Features
```

### **Phase 3: Training**
```
Processed Features (1,052 samples)
    ↓
[Train/Val/Test Split]
    ├─ Train: 80% (841)
    ├─ Validation: 10% (105)
    └─ Test: 10% (106)
    ↓
[Model Training]
    ├─ LSTM Model
    ├─ RNN Model
    └─ GRU Model
    ↓
Model Checkpoints (.h5)
    ↓
[Evaluation]
    ├─ Accuracy
    ├─ Precision/Recall
    ├─ Confusion Matrix
    └─ ROC Curve
    ↓
Production Models
```

### **Phase 4: Inference (Real-time)**
```
User Upload Audio
    ↓
[Gradio/FastAPI]
    ↓
[AudioCleaner] → Clean Audio
    ↓
[MFCCExtractor] → Features (T, 13)
    ↓
[Reshape] → (1, T, 13)
    ↓
[Model.predict()]
    ↓
Probability Score [0-1]
    ↓
[Threshold 0.5]
    ├─ ≥ 0.5 → Perempuan
    └─ < 0.5 → Laki-laki
    ↓
Result + Confidence
    ↓
Display to User
```

### **Phase 5: Feedback Loop**
```
User Provides Feedback
    ↓
[Save to feedback.csv]
    ↓
Check: Feedback Count ≥ 20?
    ├─ No → Wait for more
    └─ Yes → Trigger Auto-Retrain
        ↓
    [Retrain Models]
        ↓
    Update Production Models
        ↓
    Log to MLflow
```

---

## 📁 Project Structure - Explained

```
C:\mlops/
│
├── 📂 .github/workflows/          # GitHub Actions CI/CD
│   ├── train.yml                  # Auto-training pipeline
│   ├── test.yml                   # Testing automation
│   ├── data-validation.yml        # Data quality checks
│   └── deploy.yml                 # Deployment automation
│
├── 📂 data/
│   ├── raw/                       # Original M4A/OPUS files
│   ├── raw_wav/                   # Converted WAV files
│   │   ├── cewe/                  # Female samples (50)
│   │   └── cowo/                  # Male samples (50)
│   ├── processed/                 # Preprocessed features
│   │   ├── features_latest.npy    # MFCC features (1052, T, 13)
│   │   ├── labels_latest.npy      # Binary labels
│   │   └── metadata_latest.json   # Dataset info
│   └── feedback/
│       └── feedback.csv           # User corrections
│
├── 📂 models/                     # Trained models
│   ├── lstm_production.h5         # LSTM (100% acc)
│   ├── rnn_production.h5          # RNN (85.85% acc)
│   └── gru_production.h5          # GRU (100% acc)
│
├── 📂 src/
│   ├── preprocessing/             # Data preprocessing
│   │   ├── audio_cleaner.py       # Noise reduction, normalization
│   │   ├── feature_extractor.py   # MFCC extraction
│   │   └── dataset_loader.py      # Load & segment audio
│   │
│   ├── training/                  # Model training
│   │   ├── model.py               # Model architectures
│   │   ├── train.py               # Training script
│   │   ├── evaluate.py            # Metrics & evaluation
│   │   └── auto_retrain.py        # Auto-retraining logic
│   │
│   ├── api/                       # REST API
│   │   ├── predict.py             # Prediction endpoint
│   │   └── feedback.py            # Feedback endpoint
│   │
│   └── ui/                        # User interfaces
│       ├── app.py                 # Gradio web UI
│       └── admin.py               # Admin dashboard
│
├── 📂 reports/                    # Training reports
│   └── [model]_[timestamp]/
│       ├── metrics.json
│       ├── classification_report.txt
│       └── confusion_matrix.png
│
├── 📂 tests/                      # Unit & integration tests
│   ├── test_pipeline.py
│   └── test_integration.py
│
├── 📂 docs/                       # Documentation
│   ├── ARCHITECTURE.md            # This file
│   └── DEPLOYMENT.md              # Deployment guide
│
├── start_ui.py                    # Quick launch Gradio UI
├── launch.py                      # Multi-component launcher
├── config.yaml                    # Configuration
├── requirements.txt               # Dependencies
└── README.md                      # Project overview
```

---

## 🚀 Deployment Options

### **1. Hugging Face Spaces** ✅ RECOMMENDED
**Kenapa Hugging Face?**
- ✅ **Gratis** untuk public apps
- ✅ **Auto-deploy** dari GitHub
- ✅ **Gradio native support**
- ✅ **Public URL** instant
- ✅ **GPU support** (paid tier)

**Cara Deploy:**
```bash
# 1. Push to Hugging Face Space
git remote add hf https://huggingface.co/spaces/[username]/[space-name]
git push hf main

# 2. Space akan auto-detect Gradio app
# 3. URL: https://huggingface.co/spaces/[username]/[space-name]
```

**File yang dibutuhkan:**
- `app.py` (rename dari start_ui.py)
- `requirements.txt`
- `models/` folder

### **2. Docker Containerization**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860 8000
CMD ["python", "start_ui.py"]
```

**Build & Run:**
```bash
docker build -t gender-voice-detection .
docker run -p 7860:7860 -p 8000:8000 gender-voice-detection
```

### **3. Cloud Platforms**
| Platform | Pros | Cons |
|----------|------|------|
| **Google Cloud Run** | Auto-scale, Pay-as-you-go | Cold start latency |
| **AWS Lambda + API Gateway** | Serverless, cheap | 15min timeout limit |
| **Azure Container Instances** | Easy setup, GPU support | More expensive |
| **Render** | Free tier, auto-deploy | Limited resources |
| **Railway** | Simple, modern UI | Limited free tier |

---

## 🔐 DagsHub vs GitHub

### **DagsHub - Apa itu?**
**DagsHub** adalah platform kolaborasi untuk Data Science & MLOps, seperti "GitHub untuk ML"

**Key Features:**
```
┌─────────────────────────────────────────────────────┐
│              DagsHub Platform                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  📊 MLflow Integration                              │
│     • Experiment tracking                           │
│     • Model versioning                              │
│     • Metrics comparison                            │
│                                                      │
│  📦 Data Versioning (DVC)                           │
│     • Large file storage                            │
│     • Dataset versioning                            │
│     • S3-like storage                               │
│                                                      │
│  🤝 Collaboration                                   │
│     • Team workspace                                │
│     • Experiment sharing                            │
│     • Model registry                                │
│                                                      │
│  🔗 Git Integration                                 │
│     • Works with GitHub                             │
│     • Auto-sync repos                               │
│     • CI/CD friendly                                │
└─────────────────────────────────────────────────────┘
```

**DagsHub vs GitHub:**
| Feature | GitHub | DagsHub |
|---------|--------|---------|
| Code versioning | ✅ | ✅ |
| Large files (models) | ❌ (100MB limit) | ✅ (DVC) |
| ML experiment tracking | ❌ | ✅ (MLflow) |
| Dataset versioning | ❌ | ✅ |
| Metrics visualization | ❌ | ✅ |
| Model comparison | ❌ | ✅ |

**Setup DagsHub:**
```bash
# 1. Create DagsHub account
# 2. Connect GitHub repo
# 3. Set credentials
export MLFLOW_TRACKING_URI='https://dagshub.com/[username]/[repo].mlflow'
export MLFLOW_TRACKING_USERNAME='[username]'
export MLFLOW_TRACKING_PASSWORD='[token]'

# 4. Your train.py will auto-log to DagsHub
```

---

## 📈 Current Performance

### **Model Metrics (Test Set)**
| Model | Accuracy | Precision | Recall | F1-Score | Size |
|-------|----------|-----------|--------|----------|------|
| LSTM  | **100%** | 100% | 100% | 100% | 263 KB |
| RNN   | 85.85%   | 86%  | 85%  | 85%  | 241 KB |
| GRU   | **100%** | 100% | 100% | 100% | 268 KB |

### **Dataset Statistics**
- Total Samples: **1,052**
- Male: 478 (45.4%)
- Female: 574 (54.6%)
- Audio Duration: 3 seconds each
- Sample Rate: 16kHz
- MFCC Features: 13 coefficients

### **Infrastructure**
- Training Time: ~5 min/model
- Inference Time: ~200ms
- UI Response: <500ms
- Model Size: <300KB each

---

## 🎯 Next Steps (GitHub Actions)

Setelah dokumentasi ini, kita akan implement:

1. **Auto-Training Pipeline** - Train model otomatis saat ada data baru
2. **Testing Automation** - Pytest untuk semua components
3. **Data Validation** - Quality checks untuk audio files
4. **Deployment Automation** - Auto-deploy ke Hugging Face
5. **DagsHub Integration** - Tracking experiments

**Ready to proceed?** 🚀
