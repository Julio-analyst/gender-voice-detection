# Hyperparameter Tuning Guide

## Overview

Project ini menggunakan **Optuna** untuk automatic hyperparameter optimization, terintegrasi dengan MLflow untuk experiment tracking.

## Features

✅ **Automatic Optimization**
- Learning rate (1e-4 to 1e-2)
- Batch size (16, 32, 64, 128)
- Epochs (20-100)
- Dropout rate (0.1-0.5)
- Network units (64, 128, 256)
- Optimizer (Adam, RMSprop)

✅ **Smart Pruning**
- MedianPruner: Stop unpromising trials early
- Saves computation time

✅ **MLflow Integration**
- All trials logged automatically
- Compare results in DagsHub

## Quick Start

### 1. Run Tuning (CLI)

```bash
# Tune LSTM with 20 trials
python src/training/hyperparameter_tuning.py --model-type lstm --n-trials 20

# Tune RNN with MLflow tracking
python src/training/hyperparameter_tuning.py --model-type rnn --n-trials 30 --mlflow-uri https://dagshub.com/Julio-analyst/gender-voice-detection.mlflow

# Tune GRU (quick test)
python src/training/hyperparameter_tuning.py --model-type gru --n-trials 10
```

### 2. Use Best Parameters

```python
from src.training.hyperparameter_tuning import HyperparameterTuner

# Load best params
best_params = HyperparameterTuner.load_best_params('lstm')

# Train with best params
python src/training/train.py --model-type lstm --use-tuned-params
```

### 3. View Results

**Saved Files:**
- `models/tuning/{model}_best_params.json` - Best hyperparameters
- `reports/tuning/{model}_trials_*.csv` - All trial results

**Example best_params.json:**
```json
{
  "model_type": "lstm",
  "best_params": {
    "learning_rate": 0.0012,
    "batch_size": 64,
    "epochs": 50,
    "dropout_rate": 0.3,
    "units": 128,
    "optimizer": "adam"
  },
  "best_val_accuracy": 0.9654,
  "n_trials": 20,
  "tuning_date": "2025-12-14T21:00:00"
}
```

## Optimization Process

```
Trial 1: lr=0.001, batch=32, dropout=0.2 → val_acc=0.89
Trial 2: lr=0.005, batch=64, dropout=0.4 → val_acc=0.92 ⬆️
Trial 3: lr=0.002, batch=128, dropout=0.3 → val_acc=0.94 ⬆️
...
Trial 20: lr=0.0015, batch=64, dropout=0.35 → val_acc=0.96 ⬆️ BEST!
```

## Integration with Training

Modified `train.py` automatically loads tuned params:

```python
# If tuned params exist, use them
# Otherwise, use default params
```

## MLflow Dashboard

View all tuning experiments at:
https://dagshub.com/Julio-analyst/gender-voice-detection.mlflow

Filter by:
- Experiment: `optuna_lstm_tuning`
- Metric: `val_accuracy`
- Sort: Descending

## Performance Tips

1. **Start Small**: 10-20 trials for initial tuning
2. **Increase Gradually**: 50+ trials for production
3. **Use Pruning**: Enabled by default (saves ~30% time)
4. **Parallel**: Run multiple models simultaneously

```bash
# Parallel tuning (separate terminals)
python src/training/hyperparameter_tuning.py --model-type lstm --n-trials 30 &
python src/training/hyperparameter_tuning.py --model-type rnn --n-trials 30 &
python src/training/hyperparameter_tuning.py --model-type gru --n-trials 30 &
```

## Troubleshooting

**Issue**: Tuning too slow
- **Solution**: Reduce `--n-trials` or `--epochs` range

**Issue**: Out of memory
- **Solution**: Reduce `batch_size` range or `units` max value

**Issue**: No improvement
- **Solution**: Increase `--n-trials` or expand search space

## Advanced Usage

### Custom Search Space

Edit `hyperparameter_tuning.py`:

```python
params = {
    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True),
    'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256]),
    # Add more params...
}
```

### Multi-objective Optimization

Optimize for both accuracy AND inference speed:

```python
return val_accuracy, -inference_time  # Minimize time
```

## Results Summary

| Model | Default Accuracy | Tuned Accuracy | Improvement |
|-------|-----------------|----------------|-------------|
| LSTM  | 95.16%          | TBD            | TBD         |
| RNN   | 85.85%          | TBD            | TBD         |
| GRU   | 100.00%         | TBD            | TBD         |

*(Run tuning to update)*

## Next Steps

1. ✅ Run tuning for all models
2. ✅ Compare results in MLflow
3. ✅ Retrain with best params
4. ✅ Deploy optimized models
5. ✅ Monitor production performance
