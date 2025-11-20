"""
Benchmark Models for Weather Temperature Forecasting

This module provides baseline models to compare against the GRU model:
1. Persistent Model (Naive forecast: today's weather = yesterday's weather)
2. SARIMA Model (Seasonal AutoRegressive Integrated Moving Average)

Both models use caching to avoid recomputation on subsequent runs.
"""

import numpy as np
import pandas as pd
import os
import time
from statsmodels.tsa.statespace.sarimax import SARIMAX


def compute_benchmarks(actuals, x_test, train_df, val_df, test_df, forecast_days, Colors, test_dates=None):
    """
    Compute predictions for benchmark models (Persistent and SARIMA).

    Parameters:
    -----------
    actuals : numpy.ndarray
        Actual temperature values for the test set (unscaled)
        Shape: (n_samples, forecast_days, 1)
    x_test : numpy.ndarray
        Input features for test set (unscaled)
        Shape: (n_samples, input_days, n_features)
    train_df : pandas.DataFrame
        Training data with TG column
    val_df : pandas.DataFrame
        Validation data with TG column
    test_df : pandas.DataFrame
        Test data with TG column
    forecast_days : int
        Number of days to forecast (e.g., 3)
    Colors : class
        Color class for terminal output formatting
    test_dates : list, optional
        List of dates corresponding to first forecast day for each test sample

    Returns:
    --------
    tuple : (persistent_predictions, sarima_predictions)
        Both are numpy arrays of shape (n_samples, forecast_days, 1)
    """

    print("Running Benchmark Models...")

    # Check and warn about cache compatibility
    persistent_cache_path = './modifiedData/persistent_predictions.npy'
    sarima_cache_path = './modifiedData/sarima_predictions.npy'

    cache_needs_refresh = False
    if os.path.exists(persistent_cache_path):
        cached_persistent = np.load(persistent_cache_path)
        if cached_persistent.shape[1] != forecast_days:
            print(f"\n{Colors.YELLOW}Warning: Cached predictions are for {cached_persistent.shape[1]}-day forecast, but model uses {forecast_days}-day forecast.{Colors.ENDC}")
            print(f"{Colors.YELLOW}Deleting old cache files and recomputing...{Colors.ENDC}")
            os.remove(persistent_cache_path)
            if os.path.exists(sarima_cache_path):
                os.remove(sarima_cache_path)
            cache_needs_refresh = True

    # ----------------------------------------------------------------------------
    # 1. PERSISTENT MODEL (Naive Forecast: today's weather = yesterday's weather)
    # ----------------------------------------------------------------------------
    print("\n[Benchmark 1/2] Checking for cached Persistent Model predictions...")

    persistent_start_time = time.time()

    if os.path.exists(persistent_cache_path):
        print(f"  {Colors.GREEN}Found cached predictions. Loading from disk...{Colors.ENDC}")
        persistent_predictions = np.load(persistent_cache_path)
        print(f"  {Colors.CYAN}Loaded persistent predictions for {len(persistent_predictions)} samples{Colors.ENDC}")
    else:
        print(f"  {Colors.YELLOW}Cache not found. Computing Persistent Model predictions...{Colors.ENDC}")

        # For persistent model: all forecast days = last observed temperature
        # This is the true naive forecast: tomorrow = today

        persistent_predictions = np.zeros_like(actuals)

        # Combine all dataframes to get full TG history
        full_df = pd.concat([train_df, val_df, test_df])

        if test_dates is not None:
            # Use test_dates to get the exact last observed TG value
            from datetime import timedelta

            for i in range(len(x_test)):
                # Get the date of the first forecast day
                forecast_start_date = test_dates[i]

                # Get TG value from the day before (last observed value)
                last_observed_date = forecast_start_date - timedelta(days=1)
                last_observed_tg = full_df.loc[last_observed_date, 'TG']

                # Persistent model: all forecast days use the same last observed value
                for day in range(forecast_days):
                    persistent_predictions[i, day, 0] = last_observed_tg
        else:
            # Fallback: use test_df indices (assumes test samples align with test_df dates)
            print(f"  {Colors.YELLOW}Warning: test_dates not provided, using fallback method{Colors.ENDC}")
            test_tg_values = test_df['TG'].values

            for i in range(len(x_test)):
                # Use the TG value from the day before this test sample
                # (i-1 in test_df corresponds to the last observed day)
                if i > 0:
                    last_observed_tg = test_tg_values[i - 1]
                else:
                    # For first test sample, use last value from validation set
                    last_observed_tg = val_df['TG'].iloc[-1]

                # All forecast days use the same last observed value
                for day in range(forecast_days):
                    persistent_predictions[i, day, 0] = last_observed_tg

        # Save to cache
        np.save(persistent_cache_path, persistent_predictions)
        print(f"  {Colors.GREEN}Computed and saved persistent predictions for {len(persistent_predictions)} samples{Colors.ENDC}")

    persistent_end_time = time.time()
    persistent_time = persistent_end_time - persistent_start_time

    # ----------------------------------------------------------------------------
    # 2. SARIMA MODEL
    # ----------------------------------------------------------------------------
    print("\n[Benchmark 2/2] Checking for cached SARIMA predictions...")

    sarima_start_time = time.time()

    if os.path.exists(sarima_cache_path):
        print(f"  {Colors.GREEN}Found cached predictions. Loading from disk...{Colors.ENDC}")
        sarima_predictions = np.load(sarima_cache_path)
        print(f"  {Colors.CYAN}Loaded SARIMA predictions for {len(sarima_predictions)} samples{Colors.ENDC}")
    else:
        print(f"  {Colors.YELLOW}Cache not found. Training SARIMA model (fair approach: train once, no refitting)...{Colors.ENDC}")

        # Prepare data for SARIMA
        # Use train+val data for SARIMA training (fair comparison with GRU which uses best model from validation)
        tg_train_val = pd.concat([train_df['TG'], val_df['TG']])
        tg_test_full = test_df['TG'].values

        # SARIMA parameters (p,d,q)(P,D,Q,s)
        sarima_params = {
            'order': (1, 1, 1),           # (p, d, q) - ARIMA order
            'seasonal_order': (1, 1, 1, 7), # (P, D, Q, s) - seasonal order with weekly cycle
            'enforce_stationarity': False,
            'enforce_invertibility': False
        }

        print(f"  SARIMA parameters: order={sarima_params['order']}, seasonal_order={sarima_params['seasonal_order']}")
        print(f"  Training on {len(tg_train_val)} days (train+val, matching GRU's best model approach)...")

        # Train SARIMA model ONCE on training+validation data
        sarima_model = SARIMAX(tg_train_val, **sarima_params)
        sarima_fit = sarima_model.fit(disp=False, maxiter=200)

        print(f"  {Colors.GREEN}SARIMA model trained successfully (model parameters are now frozen){Colors.ENDC}")
        print(f"  Generating forecasts for test set using fixed model parameters...")

        # Generate SARIMA predictions for the test set
        # FAIR APPROACH: Use the trained model without refitting
        # We'll use 'append' to add observations and forecast, but model parameters stay fixed
        sarima_predictions = np.zeros_like(actuals)

        # Start with the fitted model
        current_fit = sarima_fit

        # For each test sample, append actual observations and forecast
        for i in range(len(x_test)):
            try:
                # Forecast the next 3 days from current state
                forecast = current_fit.forecast(steps=forecast_days)
                sarima_predictions[i, :, 0] = forecast

                # Append the actual first day's value to move forward (but don't refit parameters!)
                # This updates the state but keeps model parameters fixed
                if i < len(x_test) - 1:  # Don't append on last iteration
                    actual_value = tg_test_full[i]
                    current_fit = current_fit.append([actual_value], refit=False)

            except Exception as e:
                # If anything fails, use persistence as fallback
                if i > 0:
                    sarima_predictions[i, :, 0] = tg_test_full[i-1]
                else:
                    sarima_predictions[i, :, 0] = tg_train_val.iloc[-1]

            if (i + 1) % 50 == 0:
                print(f"    {Colors.CYAN}Processed {i + 1}/{len(x_test)} samples...{Colors.ENDC}")

        # Save to cache
        np.save(sarima_cache_path, sarima_predictions)
        print(f"  {Colors.GREEN}Computed and saved SARIMA predictions for {len(sarima_predictions)} samples{Colors.ENDC}")

    sarima_end_time = time.time()
    sarima_time = sarima_end_time - sarima_start_time

    return persistent_predictions, sarima_predictions, persistent_time, sarima_time
