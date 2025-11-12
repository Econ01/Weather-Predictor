import numpy as np
import pandas as pd
import dataManager as dm
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# load the clean data from the dataManager
clean_df, FEATURE_COLS, TARGET_COLS = dm.load_and_process_data(run_plots=False, save_plots=False)

INPUT_DAYS = 30
FORECAST_DAYS = 3
N_FEATURES = len(FEATURE_COLS)
N_TARGETS = len(TARGET_COLS)

x_data = []
y_data = []

# Get the raw numpy array of our data
full_data_array = clean_df.values

for i in range(len(full_data_array) - INPUT_DAYS - FORECAST_DAYS + 1):
    # Input Window (x)
    # Get the 30-day input squence
    # This slices the feature columns (0 to N_FEATUREs)
    input_slice = full_data_array[i : i + INPUT_DAYS, 0 : N_FEATURES]
    x_data.append(input_slice)

    # Forecast Window (y)
    # Get the 3-day forecast sequence
    # This slices the target columns (N_FEATURES to end)
    target_slice = full_data_array[i + INPUT_DAYS : i + INPUT_DAYS + FORECAST_DAYS, N_FEATURES : ]
    y_data.append(target_slice)

# Convert lists of 2D arrays into two big 3D arrays
x_data = np.stack(x_data)
y_data = np.stack(y_data)

print("Sequence creation complete!")
print(f"X_data shape: {x_data.shape}")
print(f"y_data shape: {y_data.shape}")

# Data Splitting
# 80% for training, 10% for validation, 10% for testing
# Data MUST be chronological
n_samples = x_data.shape[0]
n_train = int(n_samples * 0.8)
n_val = int(n_samples * 0.9)

x_train, y_train = x_data[:n_train], y_data[:n_train]
x_val,   y_val   = x_data[n_train:n_val], y_data[n_train:n_val]
x_test,  y_test  = x_data[n_val:], y_data[n_val:]

# Scale X
# Reshape the array to a 2D array: (samples, 30, 14) -> (samples * 30, 14)
x_train_2d = x_train.reshape(-1, N_FEATURES)

# Apply standard fit (0 to 1)
x_scaler = StandardScaler().fit(x_train_2d)

# Transform and reshape all 3 sets
x_train_scaled = x_scaler.transform(x_train.reshape(-1, N_FEATURES)).reshape(x_train.shape)
x_val_scaled = x_scaler.transform(x_val.reshape(-1, N_FEATURES)).reshape(x_val.shape)
x_test_scaled = x_scaler.transform(x_test.reshape(-1, N_FEATURES)).reshape(x_test.shape)

# Scale Y
# Reshape the array to a 2D array: (samples, 30, 14) -> (samples * 30, 14)
y_train_2d = y_train.reshape(-1, N_TARGETS)
y_scaler = StandardScaler().fit(y_train_2d)

# Transform and reshape all 3 sets
y_train_scaled = y_scaler.transform(y_train.reshape(-1, N_TARGETS)).reshape(y_train.shape)
y_val_scaled = y_scaler.transform(y_val.reshape(-1, N_TARGETS)).reshape(y_val.shape)
y_test_scaled = y_scaler.transform(y_test.reshape(-1, N_TARGETS)).reshape(y_test.shape)

# Convert all numpy arrays to PyTorch tensors
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# Create DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Setting up compute device...")

# Check for CUDA (NVIDIA) or ROCm (AMD)
# A proper ROCm-enabled PyTorch build (for AMD GPUs on Linux)
# will make torch.cuda.is_available() return True.
if torch.cuda.is_available():
    device = torch.device("cuda")
# Check for Metal (Apple Silicon)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
# Default to CPU
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Define model constants
INPUT_DIM = N_FEATURES   # 14
OUTPUT_DIM = N_TARGETS   # 3
HIDDEN_DIM = 64          # Size of the "context vector"