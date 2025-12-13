# 🔄 CI/CD Pipeline Documentation

**GitHub Actions Workflows - Complete Guide**

---

## 📋 Overview CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRIGGER EVENTS                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  • Push ke branch main                                               │
│  • Pull Request (PR)                                                 │
│  • Manual workflow dispatch                                          │
│  • Schedule (cron: daily/weekly)                                     │
│  • Data update (push ke data/)                                       │
│                                                                       │
└───────────────────────────┬───────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│  WORKFLOW 1:   │  │ WORKFLOW 2: │  │  WORKFLOW 3:    │
│  Code Quality  │  │  Testing    │  │  Training       │
└───────┬────────┘  └──────┬──────┘  └────────┬────────┘
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ • Lint (flake8)│  │ • Unit Tests│  │ • Load Dataset  │
│ • Format Check │  │ • Integration│  │ • Train Models  │
│ • Type Check   │  │ • Coverage  │  │ • Evaluate      │
└───────┬────────┘  └──────┬──────┘  └────────┬────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │  WORKFLOW 4:   │
                    │  Deployment    │
                    └───────┬────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │ Build Docker │ │ Push to HF │ │ Notify     │
    └──────────────┘ └────────────┘ └────────────┘
```

---

## 🔧 Workflow 1: Code Quality & Linting

**File:** `.github/workflows/lint.yml`

### Steps:
1. **Checkout Code**
   - Action: `actions/checkout@v4`
   - Purpose: Clone repository

2. **Setup Python**
   - Action: `actions/setup-python@v5`
   - Python Version: 3.10
   - Cache pip dependencies

3. **Install Linters**
   ```bash
   pip install flake8 black isort mypy
   ```

4. **Run Linting**
   - **flake8:** Check PEP 8 compliance
   - **black:** Check code formatting
   - **isort:** Check import sorting
   - **mypy:** Type checking (optional)

5. **Post Results**
   - Comment on PR if issues found
   - Block merge if critical errors

### Trigger:
- Every push
- Every pull request
- Manual dispatch

### Example Output:
```
✅ Linting passed - No issues found
❌ Found 3 issues:
   - src/training/train.py:45 - line too long
   - src/api/predict.py:12 - unused import
   - src/ui/app.py:89 - missing type hint
```

---

## 🧪 Workflow 2: Testing Pipeline

**File:** `.github/workflows/test.yml`

### Steps:
1. **Checkout Code**

2. **Setup Python 3.10**
   - Cache dependencies

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```

4. **Run Unit Tests**
   ```bash
   pytest tests/unit/ -v --cov=src
   ```
   - Test preprocessing
   - Test feature extraction
   - Test model loading

5. **Run Integration Tests**
   ```bash
   pytest tests/integration/ -v
   ```
   - Test full pipeline
   - Test API endpoints
   - Test UI components

6. **Generate Coverage Report**
   - Upload to Codecov
   - Comment coverage % on PR

7. **Upload Test Results**
   - Artifact: test-results.xml
   - Artifact: coverage.xml

### Trigger:
- Every push
- Every pull request
- Before deployment

### Example Output:
```
======= test session starts =======
collected 24 items

tests/test_preprocessing.py ........ [100%]
tests/test_training.py .......... [100%]
tests/test_api.py ...... [100%]

========== 24 passed in 12.5s ==========

Coverage: 87%
```

---

## 🚂 Workflow 3: Model Training

**File:** `.github/workflows/train.yml`

### Steps:
1. **Checkout Code**

2. **Setup Python 3.10**

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset** (if not in repo)
   ```bash
   # From DagsHub or cloud storage
   dvc pull data/processed/
   ```

5. **Train LSTM Model**
   ```bash
   python src/training/train.py \
     --model-type lstm \
     --epochs 50 \
     --use-processed
   ```

6. **Train RNN Model**
   ```bash
   python src/training/train.py \
     --model-type rnn \
     --epochs 50 \
     --use-processed
   ```

7. **Train GRU Model**
   ```bash
   python src/training/train.py \
     --model-type gru \
     --epochs 50 \
     --use-processed
   ```

8. **Evaluate Models**
   - Generate metrics
   - Create comparison report

9. **Upload Model Artifacts**
   - models/*.h5
   - reports/*/metrics.json
   - Retention: 30 days

10. **Send to DagsHub**
    ```bash
    # Log metrics to MLflow
    export MLFLOW_TRACKING_URI=${{ secrets.DAGSHUB_URI }}
    python src/training/train.py --log-to-dagshub
    ```

11. **Post Results to PR**
    ```markdown
    ## 📊 Training Results
    
    | Model | Accuracy | Precision | Recall |
    |-------|----------|-----------|--------|
    | LSTM  | 100%     | 100%      | 100%   |
    | RNN   | 85.85%   | 86%       | 85%    |
    | GRU   | 100%     | 100%      | 100%   |
    
    Training completed in 15 minutes.
    Models saved to artifacts.
    ```

### Trigger:
- Push to `data/` folder
- Manual workflow dispatch (with parameters)
- Schedule: Weekly (Sunday 00:00 UTC)

### Parameters (Manual Dispatch):
- **model_type:** LSTM/RNN/GRU/ALL
- **epochs:** 10-200
- **batch_size:** 8-128
- **learning_rate:** 0.0001-0.01
- **use_feedback:** true/false

### Example Dispatch:
```yaml
on:
  workflow_dispatch:
    inputs:
      model_type:
        description: 'Model to train'
        required: true
        default: 'LSTM'
        type: choice
        options:
          - LSTM
          - RNN
          - GRU
          - ALL
      epochs:
        description: 'Number of epochs'
        required: true
        default: '50'
      learning_rate:
        description: 'Learning rate'
        required: false
        default: '0.001'
```

---

## 📦 Workflow 4: Deployment

**File:** `.github/workflows/deploy.yml`

### Steps:
1. **Checkout Code**

2. **Build Docker Image**
   ```bash
   docker build -t gender-voice-detection:latest .
   docker tag gender-voice-detection:latest \
     ghcr.io/${{ github.repository }}:${{ github.sha }}
   ```

3. **Push to GitHub Container Registry**
   ```bash
   docker push ghcr.io/${{ github.repository }}:${{ github.sha }}
   docker push ghcr.io/${{ github.repository }}:latest
   ```

4. **Deploy to Hugging Face Spaces**
   ```bash
   git clone https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE
   cp -r models/ $HF_SPACE/
   cp app.py $HF_SPACE/
   cd $HF_SPACE
   git add .
   git commit -m "Auto-deploy from GitHub Actions"
   git push
   ```

5. **Update Model Registry**
   - Register new model version in MLflow
   - Tag as "Production" if metrics improved

6. **Send Notifications**
   - Slack/Discord webhook
   - Email notification
   - GitHub release notes

### Trigger:
- Push to `main` branch (after tests pass)
- Manual deployment
- Release tag (v1.0.0, v1.1.0, etc.)

### Example Output:
```
✅ Deployment Successful!

🐳 Docker Image: ghcr.io/user/repo:abc1234
🤗 Hugging Face: https://huggingface.co/spaces/user/gender-voice
📊 DagsHub: https://dagshub.com/user/repo
🔗 Production URL: https://gender-voice.app

Deployment took 5 minutes.
```

---

## 🔐 Secrets Required

Add these to **GitHub Repository Settings > Secrets**:

```yaml
DAGSHUB_TOKEN              # DagsHub API token
DAGSHUB_URI                # MLflow tracking URI
HF_TOKEN                   # Hugging Face API token
HF_USERNAME                # Hugging Face username
HF_SPACE                   # Hugging Face Space name
DOCKER_USERNAME            # Docker Hub username (optional)
DOCKER_TOKEN               # Docker Hub token (optional)
SLACK_WEBHOOK              # Slack notification (optional)
```

### How to Get Secrets:

**DagsHub:**
1. Go to https://dagshub.com/user/settings/tokens
2. Create new token
3. Copy token → Add to GitHub Secrets

**Hugging Face:**
1. Go to https://huggingface.co/settings/tokens
2. Create "Write" token
3. Copy token → Add to GitHub Secrets

---

## 📊 Workflow Status Badges

Add to README.md:

```markdown
![Training](https://github.com/USER/REPO/actions/workflows/train.yml/badge.svg)
![Tests](https://github.com/USER/REPO/actions/workflows/test.yml/badge.svg)
![Lint](https://github.com/USER/REPO/actions/workflows/lint.yml/badge.svg)
![Deploy](https://github.com/USER/REPO/actions/workflows/deploy.yml/badge.svg)
```

---

## 🎯 Complete CI/CD Flow Example

### Scenario: User pushes code to main

```
1. Developer pushes code
      ↓
2. Lint workflow starts (1 min)
   ✅ Pass → Continue
   ❌ Fail → Block merge
      ↓
3. Test workflow starts (3 min)
   ✅ Pass → Continue
   ❌ Fail → Block merge
      ↓
4. Training workflow starts (15 min)
   - Train models
   - Log to DagsHub
   - Upload artifacts
      ↓
5. Deploy workflow starts (5 min)
   - Build Docker
   - Push to Hugging Face
   - Update production
      ↓
6. Notifications sent
   - GitHub PR comment
   - Slack message
   - Email
      ↓
7. Deployment complete ✅
```

**Total Time:** ~25 minutes for full pipeline

---

## 🎛️ Manual Controls

### Admin Dashboard
**URL:** http://127.0.0.1:7861

**Features:**
- ✅ Custom training parameters (epochs, LR, batch size)
- ✅ Manual retrain trigger
- ✅ View feedback data
- ✅ Model performance comparison
- ✅ Training history

### GitHub Actions Manual Dispatch
1. Go to GitHub Actions tab
2. Select workflow (e.g., "Model Training")
3. Click "Run workflow"
4. Fill parameters
5. Click "Run"

---

## 🔄 Auto-Retrain Logic

```python
# When to trigger auto-retrain:

1. Feedback Count ≥ 20
   → Trigger training with feedback data

2. Scheduled Weekly Training
   → Every Sunday 00:00 UTC

3. Data Update Detected
   → Push to data/ folder

4. Manual Admin Trigger
   → From Admin Dashboard

5. Model Performance Drop
   → If accuracy < 95% on validation
```

---

## 📈 Monitoring & Alerts

### DagsHub Dashboard
- Track all experiments
- Compare model versions
- View training curves
- Dataset versioning

### GitHub Actions Logs
- Real-time training logs
- Error traces
- Performance metrics

### Notifications
- **Success:** Model trained successfully
- **Failure:** Training failed with error
- **Warning:** Low accuracy detected
- **Info:** Deployment completed

---

## 🚀 Quick Start Commands

```bash
# Launch Admin Dashboard
python admin_dashboard.py

# Manual training from terminal
python src/training/train.py --model-type lstm --epochs 50

# Run all tests locally
pytest tests/ -v --cov=src

# Lint code locally
flake8 src/
black --check src/

# Build Docker locally
docker build -t gender-voice .
docker run -p 7860:7860 gender-voice
```

---

## 📝 Best Practices

1. **Never commit secrets** to Git
2. **Always run tests** before pushing
3. **Use meaningful commit messages**
4. **Tag releases** with semantic versioning
5. **Document parameter changes**
6. **Monitor DagsHub** for experiment tracking
7. **Review PR checks** before merging
8. **Keep models < 100MB** for GitHub

---

**Next:** Implement actual workflow YAML files! 🎯
