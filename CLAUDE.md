# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based weather prediction system that uses a Seq2Seq (Encoder-Decoder) architecture with GRU layers to forecast weather conditions. The system processes historical weather data from the European Climate Assessment & Dataset (ECA&D) to predict 3-day forecasts for temperature (TG), wind speed (FG), and precipitation probability (IS_RAIN).

## Architecture

### Two-Module Design

1. **dataManager.py** - Data processing pipeline
   - Loads 12 different weather variables from ECA&D text files in `data/` directory
   - Merges data on DATE index, handles missing values with forward fill
   - Engineers features: cyclical date encoding (DAY_SIN, DAY_COS) and binary precipitation target (IS_RAIN)
   - Generates visualizations: correlation heatmaps (2D/3D), histograms, seasonal boxplots, trend plots
   - Returns: clean_df (DataFrame), FEATURE_COLS (list), TARGET_COLS (list)

2. **model.py** - Neural network training and evaluation
   - Creates sliding windows: 30-day input sequences → 3-day forecast sequences
   - Implements Seq2Seq architecture:
     - Encoder: 2-layer GRU (input_dim=14, hidden_dim=256, dropout=0.2)
     - Decoder: RepeatVector bridge + 2-layer GRU + Linear layer (output_dim=3)
   - Hybrid loss function: MSELoss for regression (TG, FG) + BCEWithLogitsLoss for classification (IS_RAIN)
   - Training: 80/10/10 split, early stopping (patience=5), device-agnostic (CUDA/MPS/CPU)
   - Outputs: TensorBoard logs to `logs/fit/`, saves `best_model.pth`, generates `Output.png` visualization

### Data Flow

```
Raw ECA&D files (data/*.txt)
  → dataManager.ReadAndMerge()
  → dataManager.CleanData()
  → dataManager.EngineerFeatures()
  → clean_df with 14 features + 3 targets
  → model.py sliding window creation (30-day input, 3-day output)
  → StandardScaler normalization (separate for features/targets/classification)
  → PyTorch DataLoaders (batch_size=32)
  → Seq2Seq training with hybrid loss
  → Best model checkpoint + evaluation plots
```

## Development Commands

### Running the Full Pipeline

```bash
# Process data with visualizations (run this first if data/ files are present)
python dataManager.py

# Train the model (requires processed data from dataManager)
python model.py
```

### Data Processing Only

```python
from dataManager import load_and_process_data

# Generate all plots and save CSV
clean_df, features, targets = load_and_process_data(
    run_plots=True,
    save_plots=True,
    show_plots=False,
    save_csv=True
)
```

### View Training Metrics

```bash
# TensorBoard is auto-launched by model.py, or manually:
tensorboard --logdir logs/fit --port 6006
```

## Key Design Decisions

### Sequence Windowing
The model uses a sliding window approach where each sample consists of:
- Input: 30 consecutive days of 14 weather features
- Output: 3 consecutive days of 3 target variables (TG, FG, IS_RAIN)

This creates overlapping sequences that preserve temporal dependencies while maximizing training data.

### Scaling Strategy
- Features (X): Single StandardScaler fitted on flattened training data
- Regression targets (TG, FG): Separate StandardScaler, excluding IS_RAIN
- Classification target (IS_RAIN): No scaling (binary 0/1 values preserved)

All scalers are fitted ONLY on training data to prevent data leakage.

### Device Handling
The code automatically detects and uses available compute:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

ROCm (AMD) is supported if PyTorch is built with ROCm (appears as CUDA).

## Important File Locations

- Data files: `data/*.txt` (12 weather variables + metadata)
- Generated visualizations: `figures/*.png`
- Model checkpoint: `best_model.pth`
- Training logs: `logs/fit/YYYYMMDD-HHMMSS/`
- Processed data: `clean_data.csv` (optional output)
- Evaluation output: `Output.png`

## Modifying the Architecture

### Changing Forecast Window
To predict more/fewer days, update in `model.py`:
```python
INPUT_DAYS = 30      # Historical window size
FORECAST_DAYS = 3    # Prediction window size
```

### Adjusting Model Capacity
Key hyperparameters in `model.py`:
```python
HIDDEN_DIM = 256     # GRU hidden state size
BATCH_SIZE = 32      # Training batch size
N_EPOCHS = 50        # Maximum epochs
PATIENCE = 5         # Early stopping patience
```

### Adding New Weather Variables
1. Add file mapping to `dataManager.py` FILE_MAPPINGS list
2. Place corresponding data file in `data/` directory
3. Update feature engineering logic if needed (features are auto-detected from merged DataFrame)

## Data Source

Weather data must be obtained from [ECA&D](https://www.ecad.eu/dailydata/index.php) and placed in `data/` with naming convention `{ELEMENT}_SOUID{ID}.txt` (e.g., `TG_SOUID121044.txt`).

Expected variables: TG (mean temp), TN (min temp), TX (max temp), RR (precipitation), PP (pressure), SS (sunshine), HU (humidity), FG (wind speed), FX (wind gust), CC (cloud cover), SD (snow depth), QQ (radiation).
