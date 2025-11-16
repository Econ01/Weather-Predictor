# How to Store and Compare Model Versions

This guide explains how we've set up version control for model experiments.

---

## ✅ Current State is SAVED!

Your Phase 0 baseline is now safely stored for comparison.

### What's Been Saved:

```
baselines/phase0_initial/
├── best_model_temperature.pth         # Trained model weights
├── temperature_forecast_evaluation.png # Visualization
├── model_temperature.py                # Model code
└── RESULTS.md                          # Detailed metrics and configuration

model_results.json                      # Metrics database for comparisons
```

---

## 📊 Baseline Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Overall MAE | 23.29 (2.33°C) | ✅ Exceeds target (3.0°C) |
| 1-Day MAE | 14.91 (1.49°C) | ✅ Exceeds stretch goal (1.5°C) |
| 7-Day MAE | 26.33 (2.63°C) | ✅ Exceeds target (4.0°C) |
| R² | 0.7520 | ✅ Meets target (0.7) |

---

## 🔄 How to Compare After Changes

### Step 1: Make Your Improvements

Example: Add lagged features to `dataManager.py`

### Step 2: Train New Model

```bash
python model_temperature.py
```

### Step 3: Record New Results

```python
from compare_models import ModelComparison

tracker = ModelComparison()

# Add your new results
tracker.add_result('with_lagged_features', {
    'mae_overall': 21.5,  # Replace with your actual results
    'rmse_overall': 28.3,
    'r2_overall': 0.780,
    'mae_day1': 13.2,
    'mae_day2': 19.1,
    'mae_day3': 22.5,
    'mae_day4': 24.0,
    'mae_day5': 24.8,
    'mae_day6': 25.2,
    'mae_day7': 24.1,
    'rmse_day1': 18.5,
    'r2_day1': 0.910,
    'r2_day7': 0.720
}, config={
    'features': 30,  # 12 original + 18 new lagged/rolling features
    'hidden_dim': 256,
    'change': 'Added lagged features and rolling statistics'
})
```

### Step 4: Compare Models

```python
# View comparison table
tracker.compare(baseline='baseline')

# Generate comparison plot
tracker.plot_comparison(metric='mae_overall')
```

**Output Example:**
```
================================================================================
MODEL COMPARISON (Baseline: baseline)
================================================================================

[METRICS] Overall Metrics:
Model                          MAE          RMSE         R2
--------------------------------------------------------------------------------
baseline                       23.29        30.60        0.7520
with_lagged_features           21.50 (-1.79) 28.30 (-2.30) 0.7800 (+0.0280)

[PER-DAY] Per-Day MAE Comparison:
Model                          Day 1      Day 3      Day 7
--------------------------------------------------------------------------------
baseline                       14.91      23.79      26.33
with_lagged_features           13.20 (-1.71) 22.50 (-1.29) 24.10 (-2.23)

================================================================================
```

---

## 📁 File Organization Best Practices

### For Each Major Experiment:

1. **Save model weights**:
   ```bash
   mkdir baselines/experiment_name
   cp best_model_temperature.pth baselines/experiment_name/
   ```

2. **Save visualization**:
   ```bash
   cp temperature_forecast_evaluation.png baselines/experiment_name/
   ```

3. **Document changes**:
   Create `baselines/experiment_name/CHANGES.md`:
   ```markdown
   # Experiment: with_lagged_features

   ## Changes Made:
   - Added lagged features (LAG1, LAG7) for TG, TN, TX, RR, HU, FG
   - Added rolling statistics (7-day mean, std)
   - Total features increased from 12 to 30

   ## Results:
   - MAE improved from 23.29 to 21.50 (-7.7%)
   - R² improved from 0.752 to 0.780 (+3.7%)

   ## Conclusion:
   Lagged features significantly improved short-term forecasts
   ```

4. **Track in comparison system**:
   Use `tracker.add_result()` as shown above

---

## 🎯 Quick Reference Commands

### View All Saved Models:
```python
from compare_models import ModelComparison
tracker = ModelComparison()
print(tracker.results.keys())
```

### Compare Specific Models:
```python
tracker.compare(baseline='baseline', models=['with_lagged_features', 'with_lstm'])
```

### Plot Comparison:
```python
tracker.plot_comparison(metric='mae_overall')  # Overall MAE
tracker.plot_comparison(metric='r2_overall')    # R² score
tracker.plot_comparison(metric='mae_day1')      # Day 1 MAE
```

### View Baseline Anytime:
```bash
cat baselines/phase0_initial/RESULTS.md
```

---

## 📈 What to Track

**Essential Metrics:**
- `mae_overall` - Overall mean absolute error
- `rmse_overall` - Overall root mean squared error
- `r2_overall` - Overall R² score
- `mae_day1` through `mae_day7` - Per-day MAE
- `r2_day1`, `r2_day7` - First and last day R²

**Configuration Info:**
- Number of features
- Model architecture changes
- Hyperparameters
- Training time
- Dataset size

---

## ⚠️ Important Rules

1. **Never overwrite baseline** - Always create new entries
2. **Always compare** - Use `tracker.compare()` before claiming improvement
3. **Document everything** - Write clear CHANGES.md for each experiment
4. **Save artifacts** - Keep model weights + visualizations for important experiments
5. **Use descriptive names** - `with_lagged_features` not `experiment1`

---

## 🚀 Next Steps

Your Phase 0 baseline is saved. Now you can:

1. **Start Phase 1** - Pick an improvement from IMPROVEMENT_PLAN.md
2. **Implement changes** - Modify code
3. **Train** - Run model_temperature.py
4. **Compare** - Use tracker.compare()
5. **Iterate** - Keep best, try next improvement

**All improvements are tracked automatically in `model_results.json`**

Happy experimenting! 🔬
