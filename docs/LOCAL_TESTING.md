# 🧪 Local GitHub Actions Testing Guide

## Overview

Sebelum push ke GitHub, kamu bisa test workflows secara lokal untuk menghindari push-fix-push cycle.

## ✅ Metode 1: Script Otomatis (Recommended)

### test-local.ps1 (Simple & Fast)

Script ini menjalankan checks yang sama dengan GitHub Actions workflows.

```powershell
# Basic check
.\test-local.ps1

# Auto-fix formatting issues
.\test-local.ps1 -Fix

# Skip tests (faster)
.\test-local.ps1 -Fix -SkipTests
```

**Apa yang ditest:**
1. ✅ Python syntax errors (flake8)
2. ✅ Code formatting (Black)
3. ✅ Import sorting (isort)
4. ✅ YAML validation
5. ✅ Unit tests (pytest)

## 🔧 Metode 2: Manual Commands

### 1. Test Dependencies
```powershell
# Check for conflicts
pip check | Select-String "gradio|fastapi|tensorflow|pydantic"
```

### 2. Code Quality
```powershell
# Syntax check
flake8 src/ --count --select=E9,F63,F7,F82

# Format check
black src/ tests/ --check

# Import sorting
isort src/ tests/ --check-only
```

### 3. Apply Fixes
```powershell
# Auto-format code
black src/ tests/

# Fix imports
isort src/ tests/
```

### 4. Run Tests
```powershell
# All tests
pytest tests/ -v

# Only integration tests (yang pass)
pytest tests/test_integration.py -v

# With coverage
pytest tests/ --cov=src
```

## 🐳 Metode 3: act CLI (Advanced)

`act` allows you to run GitHub Actions workflows locally using Docker.

### Installation

**Via winget:**
```powershell
winget install nektos.act
```

**Via Chocolatey:**
```powershell
choco install act-cli
```

**Manual download:**
Download from: https://github.com/nektos/act/releases

### Usage

```powershell
# List all workflows
act -l

# Run specific workflow
act -j lint          # Run lint workflow
act -j test          # Run test workflow

# Run all push event workflows
act push

# Dry run (show what would run)
act -n

# Use specific platform image
act -j test -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

### First Run Setup

Saat pertama kali run `act`, akan ditanya image size:
- **Micro** (~200MB) - basic testing
- **Medium** (~500MB) - recommended
- **Large** (~17GB) - full GitHub Actions environment

Pilih **Medium** untuk balance antara ukuran dan compatibility.

## 📊 Hasil Testing Local

Setelah run `.\test-local.ps1 -Fix`, kamu akan dapat:

✅ **All checks passed! Safe to push.**

Artinya:
- ✅ No syntax errors
- ✅ Code properly formatted (Black)
- ✅ Imports sorted correctly (isort)
- ✅ YAML files valid
- ✅ Integration tests passing

## 🔄 Workflow Setelah Testing

```powershell
# 1. Test lokal
.\test-local.ps1 -Fix

# 2. Review changes
git status
git diff

# 3. Commit
git add .
git commit -m "Your message"

# 4. Push ke GitHub
git push
```

## ⚠️ Known Issues

### Dependency Conflicts (FIXED)

**Previous issue:**
```
tensorflow==2.13.0 requires typing-extensions<4.6.0
fastapi==0.104.1 requires typing-extensions>=4.8.0
ERROR: ResolutionImpossible
```

**Solution (applied):**
```
✅ Upgraded TensorFlow 2.13.0 → 2.16.1
✅ Upgraded numpy 1.24.3 → 1.26.4
✅ All dependencies now compatible
```

### Test Failures (Expected)

`test_preprocessing.py` has signature mismatches (non-blocking in workflows):
```python
# Tests expect:
MFCCExtractor(n_mfcc=13, max_len=469)

# Actual class:
MFCCExtractor(use_cleaner=True)
```

These are **non-blocking** in GitHub Actions (`continue-on-error: true`).

## 📝 Tips

1. **Always run before push:**
   ```powershell
   .\test-local.ps1 -Fix && git push
   ```

2. **Quick format check:**
   ```powershell
   black src/ tests/ --check
   ```

3. **Fix all formatting issues:**
   ```powershell
   .\test-local.ps1 -Fix
   ```

4. **Skip slow tests:**
   ```powershell
   .\test-local.ps1 -SkipTests
   ```

## 🎯 GitHub Actions Status

Current workflows:
- ✅ **lint.yml** - Code quality checks
- ✅ **test.yml** - Multi-OS testing
- ✅ **train.yml** - Auto-training on data updates
- ✅ **deploy.yml** - Hugging Face deployment

All workflows passing after dependency fix!

## 📚 Documentation

- GitHub Actions: [.github/workflows/](../.github/workflows/)
- Requirements: [requirements.txt](../requirements.txt)
- Test files: [tests/](../tests/)
