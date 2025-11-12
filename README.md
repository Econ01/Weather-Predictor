# Weather Predictor

A Python-based weather data analysis and prediction project that processes historical weather data from the European Climate Assessment & Dataset (ECA&D) and prepares it for machine learning applications.

## Overview

This project loads, cleans, and analyzes multiple weather variables including temperature, precipitation, humidity, wind speed, pressure, and more. It performs exploratory data analysis through various visualizations and engineers features suitable for weather prediction models.

## Features

- **Data Loading and Processing**: Automatically reads and merges 12 different weather variables from ECA&D dataset files
- **Data Cleaning**: Handles missing values using forward fill and drops incomplete records
- **Feature Engineering**: Creates cyclical date features and binary classification targets for precipitation prediction
- **Visualization Suite**: Generates comprehensive visualizations including:
  - 2D and 3D correlation heatmaps
  - Variable distribution histograms
  - Seasonal boxplots
  - Long-term trend analysis with rolling averages

## Weather Variables

The following weather variables are processed:

- **TG**: Mean Temperature
- **TN**: Minimum Temperature
- **TX**: Maximum Temperature
- **RR**: Precipitation
- **PP**: Sea Level Pressure
- **SS**: Sunshine Duration
- **HU**: Humidity
- **FG**: Wind Speed
- **FX**: Wind Gust
- **CC**: Cloud Cover
- **SD**: Snow Depth
- **QQ**: Global Radiation

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

## Data Source

Weather data should be obtained from the [European Climate Assessment & Dataset (ECA&D)](https://www.ecad.eu/dailydata/index.php).

Place the downloaded data files in the `data/` directory with the following naming convention:
- `TG_SOUID121044.txt`
- `TN_SOUID121045.txt`
- `TX_SOUID121046.txt`
- (and so on for other variables)

## Usage

Run the data processing pipeline:

```python
from dataManager import load_and_process_data

# Process data with all visualizations
clean_df, features, targets = load_and_process_data(
    run_plots=True,
    save_plots=True,
    show_plots=False,
    save_csv=True
)
```

Or run directly from command line:

```bash
python dataManager.py
```

### Parameters

- `run_plots`: Whether to generate visualizations (default: True)
- `save_plots`: Save visualization figures to `figures/` directory (default: True)
- `show_plots`: Display plots interactively (default: False)
- `save_csv`: Save processed data to `clean_data.csv` (default: True)

## Output

The processing pipeline generates:

1. **Visualizations** (saved in `figures/`):
   - `correlation_heatmap_2d.png`: 2D correlation matrix
   - `correlation_heatmap_3d.png`: 3D correlation visualization
   - `variable_histograms.png`: Distribution plots for key variables
   - `seasonal_boxplot_tg.png`: Temperature seasonality analysis
   - `long_term_trend_tg.png`: Annual temperature trends

2. **Processed Data**:
   - `clean_data.csv`: Final cleaned dataset with engineered features

## Project Structure

```
Weather-Predictor/
├── data/               # Raw weather data files
├── figures/            # Generated visualization outputs
├── dataManager.py      # Main data processing module
├── .gitignore
└── README.md
```

## Engineered Features

The pipeline creates additional features for machine learning:

- **IS_RAIN**: Binary target indicating precipitation > 0.1mm
- **DAY_SIN / DAY_COS**: Cyclical encoding of day of year for seasonal patterns

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn

## License

This project processes data from the European Climate Assessment & Dataset. Please refer to their [terms of use](https://www.ecad.eu/dailydata/index.php) for data licensing information.
