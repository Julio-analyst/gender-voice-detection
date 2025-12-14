# 🚀 Deployment Guide - Gender Voice Detection MLOps

## 📋 Overview

Ada **2 versi deployment** yang tersedia:

### 1. **app.py** - User Interface Only
- ✅ Public access
- ✅ Prediksi gender dari audio
- ✅ Feedback form
- ❌ No admin panel

### 2. **app_with_admin.py** - User Interface + Admin Dashboard
- ✅ Public access untuk User UI
- ✅ Prediksi gender dari audio
- ✅ Feedback form
- ✅ **Secure Admin Dashboard** dengan login
- 🔐 Password protection

---

## 🔐 Admin Dashboard Security

**Default Credentials:**
- Username: `admin`
- Password: `mlops2024`

⚠️ **PENTING: Ganti password untuk production!**

### Cara Ganti Password di Hugging Face:

1. **Set via Environment Variables** (Recommended):
   - Go to: **Space Settings → Variables and secrets**
   - Add secrets:
     - `ADMIN_USERNAME` = your_username
     - `ADMIN_PASSWORD` = your_secure_password
   
2. **Edit code directly** (Not recommended):
   ```python
   # Di app_with_admin.py line 32-33
   ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "your_username")
   ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "your_strong_password")
   ```

---

## 📦 Deployment Options

### Option A: Hugging Face Spaces (Recommended)

#### **Versi 1: User Interface Only**

1. **Create new Space:**
   - Go to: https://huggingface.co/spaces
   - Click: **Create new Space**
   - Name: `gender-voice-detection`
   - SDK: **Gradio**
   - Visibility: **Public**

2. **Upload files:**
   ```
   app.py                    # User UI only
   requirements_hf.txt
   config.yaml
   models/
     ├── lstm_production.h5
     ├── rnn_production.h5
     └── gru_production.h5
   src/
     ├── preprocessing/
     ├── training/
     └── utils/
   data/
     └── feedback/
         └── feedback.csv
   ```

3. **Space will auto-deploy!** 🎉
   - URL: `https://huggingface.co/spaces/[username]/gender-voice-detection`

#### **Versi 2: User Interface + Admin Dashboard**

1. **Create new Space** (same as above)

2. **Upload files:**
   ```
   app_with_admin.py         # Rename to app.py
   requirements_hf.txt       # (already includes plotly)
   config.yaml
   models/ (same as above)
   src/ (same as above)
   data/ (same as above)
   ```

3. **Set Admin Credentials:**
   - Go to: **Settings → Variables and secrets**
   - Add:
     - `ADMIN_USERNAME` = `admin` (or your choice)
     - `ADMIN_PASSWORD` = `your_secure_password_here`

4. **Space will deploy with 2 tabs:**
   - **Tab 1:** 🎤 User Interface (public)
   - **Tab 2:** 🔐 Admin Dashboard (password protected)

---

### Option B: Docker Deployment

1. **Build image:**
   ```bash
   docker build -t gender-voice-detection .
   ```

2. **Run container:**
   ```bash
   docker run -p 7860:7860 gender-voice-detection
   ```

3. **Or use docker-compose:**
   ```bash
   docker-compose up -d
   ```

4. **Access:**
   - User UI: http://localhost:7860
   - Stop: `docker-compose down`

---

### Option C: GitHub Actions Auto-Deploy

1. **Add Hugging Face Token to GitHub Secrets:**
   - Go to: **GitHub Repo → Settings → Secrets**
   - Add: `HF_TOKEN` (get from https://huggingface.co/settings/tokens)
   - Add: `HF_USERNAME` (your HF username)

2. **Trigger workflow:**
   - Go to: **Actions → Deploy to Hugging Face**
   - Click: **Run workflow**
   - Select deployment version (user only / with admin)

3. **Workflow will:**
   - ✅ Build deployment package
   - ✅ Upload to Hugging Face
   - ✅ Auto-deploy Space

---

## 🎯 Testing Before Deployment

### Local Testing (User UI + Admin):

```bash
# Test app with admin
python app_with_admin.py

# Open browser:
# - User UI: http://localhost:7860 (tab 1)
# - Admin Dashboard: http://localhost:7860 (tab 2)
```

**Test Checklist:**

**User Interface:**
- [ ] Upload audio file
- [ ] Select model (LSTM/RNN/GRU)
- [ ] Click "Prediksi Gender"
- [ ] Verify prediction result
- [ ] Submit feedback

**Admin Dashboard:**
- [ ] Click "Admin Dashboard" tab
- [ ] Login with credentials (admin/mlops2024)
- [ ] View statistics
- [ ] Check accuracy charts
- [ ] Review feedback table
- [ ] Export CSV

### Docker Testing:

```bash
# Build
docker build -t gender-voice-test .

# Run
docker run -p 7860:7860 gender-voice-test

# Test at http://localhost:7860
```

---

## 📊 Admin Dashboard Features

### Statistics Displayed:
- ✅ Total feedback count
- ✅ Correct vs incorrect predictions
- ✅ Overall accuracy percentage
- ✅ Breakdown by gender (Male/Female)

### Visualizations:
- 📊 Pie chart: Prediction accuracy
- 📈 Line chart: Feedback timeline
- 📋 Table: Recent feedback (last 10)

### Actions:
- 🔄 Refresh data (real-time update)
- 📥 Export feedback CSV (download)

---

## 🔒 Security Best Practices

1. **Always change default password!**
   ```python
   ADMIN_PASSWORD = "use_strong_password_here"
   ```

2. **Use Environment Variables:**
   ```bash
   # Hugging Face Space Secrets
   ADMIN_USERNAME=your_admin_username
   ADMIN_PASSWORD=your_super_secure_password_123
   ```

3. **Regular password rotation:**
   - Change password monthly
   - Use password manager

4. **Monitor admin access:**
   - Check who accessed admin panel
   - Review feedback exports

---

## 🚨 Troubleshooting

### Issue: Models not loading
**Solution:** Verify `models/` folder contains:
- `lstm_production.h5`
- `rnn_production.h5`
- `gru_production.h5`

### Issue: Admin login fails
**Solution:**
1. Check credentials in Space Secrets
2. Verify environment variables loaded:
   ```python
   print(os.getenv("ADMIN_PASSWORD"))
   ```

### Issue: Feedback not saving
**Solution:**
1. Check `data/feedback/` folder exists
2. Verify write permissions
3. Check CSV file not corrupted

### Issue: Charts not displaying
**Solution:**
1. Verify `plotly` in requirements_hf.txt
2. Check feedback.csv has data
3. Refresh browser cache

---

## 📝 Deployment Checklist

Before pushing to production:

- [ ] Test locally (user UI works)
- [ ] Test admin login (credentials correct)
- [ ] Change default password
- [ ] Add HF secrets (ADMIN_USERNAME, ADMIN_PASSWORD)
- [ ] Verify all models present
- [ ] Test feedback submission
- [ ] Test admin export CSV
- [ ] Check charts render correctly
- [ ] Verify mobile responsive

---

## 🎉 Post-Deployment

After successful deployment:

1. **Share public URL:**
   - User: `https://huggingface.co/spaces/[username]/gender-voice-detection`
   - Admin: Same URL → "Admin Dashboard" tab

2. **Monitor usage:**
   - Check feedback submissions
   - Review prediction accuracy
   - Export feedback data regularly

3. **Update models:**
   - Retrain with feedback data
   - Replace `*_production.h5` files
   - Restart Space

4. **Security:**
   - Rotate admin password monthly
   - Monitor suspicious access
   - Backup feedback data

---

## 📚 Additional Resources

- **Hugging Face Docs:** https://huggingface.co/docs/hub/spaces
- **Gradio Docs:** https://gradio.app/docs
- **Repository:** https://github.com/Julio-analyst/gender-voice-detection
- **DagsHub MLflow:** https://dagshub.com/Julio-analyst/gender-voice-detection.mlflow

---

**Built with ❤️ by MLOps Team**
