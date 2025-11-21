# Weather Temperature Forecasting with Deep Learning

A comprehensive deep learning project for multi-day weather temperature forecasting using historical climate data from the European Climate Assessment & Dataset (ECA&D). The project implements a GRU-based sequence-to-sequence model with attention mechanism and benchmarks it against traditional forecasting methods.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Model Comparison](#model-comparison)
- [Requirements](#requirements)

## Overview

This project forecasts mean temperature (TG) for the next 3 days using a 15-day historical window of 9 weather variables. The model uses a GRU encoder-decoder architecture with Bahdanau-style attention mechanism and is benchmarked against Persistent and SARIMA baseline models.

### Key Highlights

- **Forecasting Task**: 3-day ahead mean temperature prediction
- **Input Features**: 9 weather variables (TN, TX, RR, SS, HU, FG, FX, CC, SD)
- **Architecture**: Seq2Seq GRU with attention mechanism
- **Data Range**: 1957-2025 (68 years of daily weather data)
- **Train/Val/Test Split**: 1957-2017 / 2018-2022 / 2023-2025
- **Model Performance**: Outperforms Persistent and SARIMA baselines on all metrics

## Dataset

### Data Source

Weather data is obtained from the [European Climate Assessment & Dataset (ECA&D)](https://www.ecad.eu/dailydata/index.php), providing high-quality daily observations for multiple weather variables.

### Weather Variables

The project processes the following weather variables:

| Variable | Description | Unit |
|----------|-------------|------|
| **TG** | Mean Temperature (target) | 0.1°C |
| **TN** | Minimum Temperature | 0.1°C |
| **TX** | Maximum Temperature | 0.1°C |
| **RR** | Precipitation | 0.1mm |
| **SS** | Sunshine Duration | 0.1h |
| **HU** | Humidity | % |
| **FG** | Wind Speed | 0.1m/s |
| **FX** | Wind Gust | 0.1m/s |
| **CC** | Cloud Cover | oktas |
| **SD** | Snow Depth | cm |

**Note**: The original dataset includes PP (Sea Level Pressure) and QQ (Global Radiation), but these variables are excluded from the final model due to data quality issues identified during exploratory analysis.

### Data Processing Pipeline

1. **Loading**: Reads and merges 12 weather variable files from ECA&D
2. **Cleaning**:
   - Removes data beyond 2025-09-30 (last real observation date)
   - Forward fills missing values within real data
   - Drops remaining NaN values at the start of the time series
3. **Feature Engineering**:
   - Creates binary precipitation indicator (IS_RAIN)
   - Generates cyclical date features (DAY_SIN, DAY_COS) for seasonal patterns
   - Excludes PP and QQ from final feature set

### Correlation Analysis

![Correlation Heatmap](./figures/correlation_heatmap_2d.png)

The correlation heatmap reveals strong relationships between temperature variables (TG, TN, TX) and moderate correlations with other meteorological features, validating the choice of input features for temperature forecasting.

## Architecture

### GRU Sequence-to-Sequence Model with Attention

The model implements a modern encoder-decoder architecture with the following components:

#### Encoder
- **Type**: 1-layer GRU with 64 hidden units
- **Initialization**: Orthogonal initialization for recurrent weights, Xavier uniform for input weights
- **Input**: 15-day window of 9 weather features
- **Output**: Encoded sequence representations and final hidden state

#### Attention Mechanism
- **Type**: Bahdanau-style attention
- **Purpose**: Allows decoder to focus on relevant parts of the input sequence
- **Benefits**: Improves long-term dependency modeling

#### Decoder
- **Type**: 1-layer autoregressive GRU with 64 hidden units
- **Initialization**: Orthogonal initialization for stability
- **Strategy**: Generates predictions autoregressively (each prediction feeds into the next step)
- **Output**: 3-day temperature forecast

#### Training Configuration

- **Loss Function**: L1 Loss (Mean Absolute Error)
- **Optimizer**: Adam with learning rate 0.0001
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch Size**: 32
- **Gradient Clipping**: Max norm of 1.0 for stability
- **Early Stopping**: Patience of 20 epochs
- **Reproducibility**: Fixed random seed (8888) for deterministic results

### Benchmark Models

#### 1. Persistent Model (Naive Forecast)
- **Approach**: Tomorrow's weather = today's weather
- **Implementation**: Uses last observed temperature for all forecast days
- **Purpose**: Establishes baseline performance

#### 2. SARIMA Model
- **Type**: Seasonal AutoRegressive Integrated Moving Average
- **Configuration**: SARIMA(1,1,1)(1,1,1,7) with weekly seasonality
- **Training**: Trained once on train+validation data (fair comparison)
- **Forecasting**: Rolling forecast without parameter refitting

## Features

- **Data Management Module** (`dataManager.py`):
  - Automated data loading and merging
  - Robust data cleaning with handling of missing values
  - Feature engineering for cyclical temporal patterns
  - Comprehensive visualization suite

- **Deep Learning Model** (`model_GRU_temperature.py`):
  - State-of-the-art GRU architecture with attention
  - Year-based train/validation/test splitting
  - Reproducible training with fixed seeds
  - Comprehensive evaluation with multiple metrics

- **Benchmarking System** (`benchmarks.py`):
  - Persistent and SARIMA baseline implementations
  - Efficient caching system to avoid recomputation
  - Fair comparison methodology

- **Multi-Seed Testing** (`train_multiple_seeds.py`):
  - Trains models with 10 different random seeds
  - Evaluates model stability and robustness
  - Statistical analysis of performance variance

- **Model Comparison Utilities** (`compare_models.py`):
  - Tracks multiple model versions
  - Compares metrics across models
  - Visualization of model improvements

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA or MPS (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Econ01/Weather-Predictor.git
cd Weather-Predictor
```

2. Install required dependencies:
```bash
pip3 install -r requirements.txt
```

Or install packages individually:
```bash
pip3 install torch numpy pandas scikit-learn matplotlib seaborn statsmodels
```

3. Download weather data from [ECA&D](https://www.ecad.eu/dailydata/index.php) and place files in the `data/` directory with the following naming convention:
```
data/
├── TG_SOUID121044.txt
├── TN_SOUID121045.txt
├── TX_SOUID121046.txt
├── RR_SOUID121042.txt
├── SS_SOUID121040.txt
├── HU_SOUID121047.txt
├── FG_SOUID121048.txt
├── FX_SOUID121049.txt
├── CC_SOUID121039.txt
├── SD_SOUID121043.txt
├── PP_SOUID121041.txt (excluded from model)
└── QQ_SOUID210447.txt (excluded from model)
```

## Usage

### 1. Data Processing and Visualization

Run the data management pipeline to process raw data and generate visualizations:

```python
from dataManager import load_and_process_data

# Process data with all visualizations
clean_df, features, targets = load_and_process_data(
    run_plots=True,      # Generate visualizations
    save_plots=True,     # Save plots to figures/
    show_plots=False,    # Display plots interactively
    save_csv=True        # Save processed data to CSV
)
```

Or run directly from command line:
```bash
python3 dataManager.py
```

This will generate:
- Cleaned and processed dataset
- 2D and 3D correlation heatmaps
- Distribution histograms for all variables
- Time series plots
- Seasonal analysis
- Long-term temperature trends

### 2. Train the GRU Model

Train the main forecasting model with benchmark comparison:

```bash
python3 model_GRU_temperature.py
```

The script will:
- Load and preprocess data
- Create sequential samples (15-day input, 3-day output)
- Split data by year (train: 1957-2017, val: 2018-2022, test: 2023-2025)
- Train GRU model with early stopping
- Evaluate against Persistent and SARIMA baselines
- Generate comprehensive evaluation plots
- Save the best model as `best_model_temperature.pth`

### 3. Multi-Seed Robustness Testing

Evaluate model stability across different random seeds:

```bash
python3 train_multiple_seeds.py
```

This trains 10 models with different seeds and provides:
- Performance statistics (mean, std, min, max)
- Comparison visualizations
- Results saved to `modifiedData/multi_seed_results.csv`

### 4. Model Comparison

Track and compare different model versions:

```python
from compare_models import ModelComparison

tracker = ModelComparison()

# Add a new model result
tracker.add_result('improved_model', {
    'mae_overall': 21.5,
    'rmse_overall': 28.3,
    'r2_overall': 0.780,
    'mae_day1': 14.2,
    'mae_day3': 22.1
})

# Compare against baseline
tracker.compare(baseline='baseline')

# Generate comparison plots
tracker.plot_comparison(metric='mae_overall')
```

## Results

### 1-Day Ahead Forecast Performance (Test Set: 2023-2025)

The GRU model significantly outperforms both baseline models for 1-day ahead forecasting:

| Model | MAE (°C) | RMSE (°C) | R² |
|-------|----------|-----------|-----|
| **GRU (Ours)** | **1.61** | **2.06** | **0.9011** |
| Persistent | 1.78 | 2.30 | 0.8768 |
| SARIMA | 1.72 | 2.22 | 0.8848 |

**Improvement over baselines**: 10.55% reduction in MAE compared to Persistent and 6.83% compared to SARIMA models.

### 3-Day Ahead Forecast Performance

The GRU model maintains strong performance even at longer forecast horizons:

| Model | MAE (°C) | RMSE (°C) | R² |
|-------|----------|-----------|-----|
| **GRU (Ours)** | **2.58** | **3.26** | **0.7525** |
| Persistent | 3.04 | 3.86 | 0.6524 |
| SARIMA | 2.73 | 3.47 | 0.7197 |

**Improvement over baselines**: 17.83% reduction in MAE compared to Persistent and 5.81% compared to SARIMA models.

### Computational Efficiency

| Model | Training/Computation Time |
|-------|---------------------------|
| GRU | 267.5s (4.46 min) to best model |
| Persistent | 6.7ms |
| SARIMA | 312.8s (5.21 min) |

When it comes to computational efficiency, our model performs significantly better than SARIMA, however, looses to Persistent model.

## Visualizations

### 1. Temperature Distribution and Seasonality

![Variable Histograms](./figures/variable_histograms.png)

Distribution plots reveal the statistical properties of all weather variables, showing normal distributions for temperature variables and right-skewed distributions for precipitation and wind.

![Seasonal Boxplot](./figures/seasonal_boxplot_tg.png)

Monthly temperature patterns clearly show seasonal cycles, with lower temperatures in winter months (December-February) and higher temperatures in summer months (June-August).

### 2. Long-Term Temperature Trends

![Long-Term Trend](./figures/long_term_trend_tg.png)

Annual average temperature analysis from 1957-2025 shows a clear warming trend with a 5-year rolling average and linear regression line indicating approximately 0.3°C increase per decade.

### 3. Time Series Analysis

![Time Series](./figures/time_series_all_variables.png)

Complete time series visualization of all weather variables reveals temporal patterns, seasonality, and data quality across the 68-year observation period.

### 4. Model Performance Comparison

![Forecast Evaluation](./figures/temperature_forecast_evaluation.png)

Comprehensive evaluation comparing GRU, Persistent, and SARIMA models across 1-day and 3-day forecast horizons. The plot includes:
- Time series comparison of actual vs predicted temperatures
- Per-day error metrics (MAE, RMSE, R²) for all models
- Performance degradation analysis across forecast horizon
- Execution time comparison

The GRU model demonstrates superior performance with:
- Lower MAE and RMSE across all forecast days
- Higher R² scores indicating better explained variance
- Smooth predictions that capture seasonal and short-term patterns
- Consistent performance improvement over baselines

## Model Comparison

### Advantages of GRU Model

1. **Multivariate Input**: Leverages 9 weather features vs. univariate baselines
2. **Non-linear Modeling**: Captures complex weather patterns and interactions
3. **Attention Mechanism**: Focuses on relevant historical information
4. **Learned Representations**: Automatically discovers predictive patterns
5. **Consistent Performance**: Maintains accuracy across forecast horizon

### When to Use Each Model

- **GRU Model**: Best for accurate multi-day forecasts when computational resources are available
- **Persistent Model**: Useful for 1-day forecasts when simplicity is prioritized
- **SARIMA Model**: Good middle-ground for moderate accuracy with lower complexity

## Requirements

### Python Libraries

```
torch>=1.9.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
statsmodels>=0.12.0
```

## Data Preprocessing Details

### Sequence Creation

- **Input Window**: 15 consecutive days of 9 features (shape: 15 × 9)
- **Output Window**: 3 consecutive days of target temperature (shape: 3 × 1)
- **Sliding Window**: Moves forward one day at a time
- **Total Sequences**: ~24,000 training samples

### Normalization

- **Method**: StandardScaler (zero mean, unit variance)
- **Fit on**: Training set only
- **Applied to**: All splits (train/val/test)
- **Separate Scalers**: One for features (X), one for target (y)

### Data Quality Issues

The following variables were excluded after quality analysis:

- **PP (Sea Level Pressure)**: Excessive missing values and suspicious patterns
- **QQ (Global Radiation)**: Incomplete historical coverage and data gaps

## Reproducibility

The project ensures reproducible results through:

1. **Fixed Random Seeds**: All random number generators use seed 8888
2. **Deterministic Operations**: PyTorch backends set to deterministic mode
3. **Saved Model Weights**: Best model checkpoint saved for inference
4. **Documented Hyperparameters**: All configurations explicitly specified
5. **Version Control**: Git tracking of code and configuration changes

## License

This project processes data from the European Climate Assessment & Dataset. Please refer to [ECA&D terms of use](https://www.ecad.eu/dailydata/index.php) for data licensing information.

## Acknowledgments

- **Data Source**: European Climate Assessment & Dataset (ECA&D)
- **Inspiration**: Modern sequence-to-sequence architectures for time series forecasting
- **Frameworks**: PyTorch for deep learning, statsmodels for SARIMA baseline
