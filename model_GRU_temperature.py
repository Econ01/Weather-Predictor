"""
Weather Temperature Prediction Model - Standard GRU
Single-task model for 3-day temperature forecasting

Features:
- 9 input variables (excludes PP and QQ due to data quality issues)
  - Base features: TN, TX, RR, SS, HU, FG, FX, CC, SD
- 3-day forecast horizon
- Year-based train/val/test split (1957-2017 train / 2018-2022 val / 2023-2025 test)
- Architecture: 1-layer GRU with 64 hidden units
- Autoregressive decoder with attention mechanism
- Gradient clipping for stability
- Reproducible results with seed=8888
- Comprehensive evaluation metrics and visualization
- Benchmarked against Persistent and SARIMA models
"""

import numpy as np
import pandas as pd
import dataManager as dm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
import warnings
import os
import time
import benchmarks
warnings.filterwarnings('ignore')

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Set random seeds for reproducibility
SEED = 8888
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed set to {SEED} for reproducibility")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1/9] Loading data...")
# Load the clean data from dataManager
clean_df, feature_cols, target_cols = dm.load_and_process_data(run_plots=False, save_plots=False)

# Use features from dataManager
FEATURE_COLS = feature_cols

# Target: TG (Mean Temperature)
TARGET_COL = 'TG'

# Configuration
INPUT_DAYS = 15
FORECAST_DAYS = 3
N_FEATURES = len(FEATURE_COLS)

print(f"Input features: {N_FEATURES}")
print(f"Features: {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")
print(f"Input window: {INPUT_DAYS} days")
print(f"Forecast horizon: {FORECAST_DAYS} days")

# Extract feature and target data
feature_data = clean_df[FEATURE_COLS].values
target_data = clean_df[[TARGET_COL]].values  # Keep as 2D array

# Get the date index for splitting
dates = clean_df.index

print(f"\nData shape:")
print(f"  Features: {feature_data.shape}")
print(f"  Target: {target_data.shape}")
print(f"  Date range: {dates[0]} to {dates[-1]}")

# ============================================================================
# SEQUENCE CREATION
# ============================================================================

print("\n[2/9] Creating sequences...")

x_data = []
y_data = []
date_indices = []  # Track which date each sequence corresponds to

for i in range(len(feature_data) - INPUT_DAYS - FORECAST_DAYS + 1):
    # Input: 30-day window of all features
    input_slice = feature_data[i : i + INPUT_DAYS, :]

    # Output: 3-day forecast of TG only
    target_slice = target_data[i + INPUT_DAYS : i + INPUT_DAYS + FORECAST_DAYS, :]

    x_data.append(input_slice)
    y_data.append(target_slice)
    date_indices.append(i + INPUT_DAYS)  # First forecast day

x_data = np.stack(x_data)
y_data = np.stack(y_data)

print(f"Sequence creation complete!")
print(f"  X_data shape: {x_data.shape}  # (samples, 30 days, {N_FEATURES} features)")
print(f"  y_data shape: {y_data.shape}  # (samples, {FORECAST_DAYS} days, 1 target)")

# ============================================================================
# YEAR-BASED DATA SPLITTING
# ============================================================================

print("\n[3/9] Splitting data by year...")

# Find split indices based on dates
train_end_date = pd.Timestamp('2017-12-31')  # Train: 1957-2017 (60 years)
val_end_date = pd.Timestamp('2022-12-31')    # Val: 2018-2022 (5 years)

# Find the indices
train_idx = None
val_idx = None

for i, date_i in enumerate(date_indices):
    date = dates[date_i]
    if train_idx is None and date > train_end_date:
        train_idx = i
    if val_idx is None and date > val_end_date:
        val_idx = i
        break

# If we didn't find the indices (shouldn't happen), fall back to end
if train_idx is None:
    train_idx = len(x_data)
if val_idx is None:
    val_idx = len(x_data)

# Split the data
x_train, y_train = x_data[:train_idx], y_data[:train_idx]
x_val, y_val = x_data[train_idx:val_idx], y_data[train_idx:val_idx]
x_test, y_test = x_data[val_idx:], y_data[val_idx:]

train_dates = [dates[date_indices[i]] for i in range(train_idx)]
val_dates = [dates[date_indices[i]] for i in range(train_idx, val_idx)]
test_dates = [dates[date_indices[i]] for i in range(val_idx, len(date_indices))]

print(f"Data split complete:")
print(f"  Train: {len(x_train)} samples ({train_dates[0].date()} to {train_dates[-1].date()})")
print(f"  Val:   {len(x_val)} samples ({val_dates[0].date()} to {val_dates[-1].date()})")
print(f"  Test:  {len(x_test)} samples ({test_dates[0].date()} to {test_dates[-1].date()})")

# ============================================================================
# SAVE DATASETS TO CSV
# ============================================================================

print("\n[4/9] Checking for existing datasets...")

# Check if CSV files already exist
train_csv_path = './modifiedData/train_data.csv'
val_csv_path = './modifiedData/val_data.csv'
test_csv_path = './modifiedData/test_data.csv'

if os.path.exists(train_csv_path) and os.path.exists(val_csv_path) and os.path.exists(test_csv_path):
    print(f"  {Colors.GREEN}Found existing CSV files. Loading from disk...{Colors.ENDC}")
    train_df = pd.read_csv(train_csv_path, index_col=0, parse_dates=True)
    val_df = pd.read_csv(val_csv_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_csv_path, index_col=0, parse_dates=True)

    print(f"  {Colors.CYAN}Loaded train_data.csv ({len(train_df)} rows, {len(train_df.columns)} columns){Colors.ENDC}")
    print(f"  {Colors.CYAN}Loaded val_data.csv ({len(val_df)} rows, {len(val_df.columns)} columns){Colors.ENDC}")
    print(f"  {Colors.CYAN}Loaded test_data.csv ({len(test_df)} rows, {len(test_df.columns)} columns){Colors.ENDC}")
    print(f"  Date range - Train: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"  Date range - Val:   {val_df.index[0].date()} to {val_df.index[-1].date()}")
    print(f"  Date range - Test:  {test_df.index[0].date()} to {test_df.index[-1].date()}")
else:
    print(f"  {Colors.YELLOW}CSV files not found. Creating new datasets...{Colors.ENDC}")

    # Create subset of clean_df excluding PP and QQ
    columns_to_keep = [col for col in clean_df.columns if col not in ['PP', 'QQ']]
    clean_df_filtered = clean_df[columns_to_keep]

    # Split by date
    train_df = clean_df_filtered[clean_df_filtered.index <= train_end_date]
    val_df = clean_df_filtered[(clean_df_filtered.index > train_end_date) & (clean_df_filtered.index <= val_end_date)]
    test_df = clean_df_filtered[clean_df_filtered.index > val_end_date]

    # Save to CSV
    os.makedirs('./modifiedData', exist_ok=True)
    train_df.to_csv(train_csv_path)
    val_df.to_csv(val_csv_path)
    test_df.to_csv(test_csv_path)

    print(f"  {Colors.GREEN}Saved train_data.csv ({len(train_df)} rows, {len(train_df.columns)} columns){Colors.ENDC}")
    print(f"  {Colors.GREEN}Saved val_data.csv ({len(val_df)} rows, {len(val_df.columns)} columns){Colors.ENDC}")
    print(f"  {Colors.GREEN}Saved test_data.csv ({len(test_df)} rows, {len(test_df.columns)} columns){Colors.ENDC}")
    print(f"  Date range - Train: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"  Date range - Val:   {val_df.index[0].date()} to {val_df.index[-1].date()}")
    print(f"  Date range - Test:  {test_df.index[0].date()} to {test_df.index[-1].date()}")

# ============================================================================
# DATA SCALING
# ============================================================================

print("\n[5/9] Scaling data...")

# Scale input features (X)
x_train_2d = x_train.reshape(-1, N_FEATURES)
x_scaler = StandardScaler().fit(x_train_2d)

x_train_scaled = x_scaler.transform(x_train.reshape(-1, N_FEATURES)).reshape(x_train.shape)
x_val_scaled = x_scaler.transform(x_val.reshape(-1, N_FEATURES)).reshape(x_val.shape)
x_test_scaled = x_scaler.transform(x_test.reshape(-1, N_FEATURES)).reshape(x_test.shape)

# Scale target (y) - only TG
y_train_2d = y_train.reshape(-1, 1)
y_scaler = StandardScaler().fit(y_train_2d)

y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

print(f"Scaling complete")
print(f"  X scaler - mean: {x_scaler.mean_[:3]}, std: {x_scaler.scale_[:3]}")
print(f"  y scaler - mean: {y_scaler.mean_[0]:.2f}, std: {y_scaler.scale_[0]:.2f}")

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create DataLoaders
BATCH_SIZE = 32
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================================
# DEVICE SETUP
# ============================================================================

print("\n[6/9] Setting up compute device...")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

print("\n[7/9] Building model architecture...")

HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.2

class Encoder(nn.Module):
    """Encoder with GRU that returns both outputs and hidden states"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Orthogonal initialization for better gradient flow
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:  # input-to-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # hidden-to-hidden (recurrent) weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # outputs: (batch, seq_len, hidden_dim) - all timesteps
        # hidden: (num_layers, batch, hidden_dim) - final hidden states
        outputs, hidden = self.gru(x)
        return outputs, hidden


class Attention(nn.Module):
    """Bahdanau-style attention mechanism"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        # Xavier initialization for attention layers
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim)

        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Repeat hidden state for each time step
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)

        # Apply attention weights
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class Decoder(nn.Module):
    """Autoregressive decoder with attention"""
    def __init__(self, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.attention = Attention(hidden_dim)

        # GRU input is output_dim + hidden_dim (prediction + context)
        self.gru = nn.GRU(
            output_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

        # Orthogonal initialization for decoder GRU
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:  # input-to-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # hidden-to-hidden (recurrent) weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Xavier initialization for output layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, encoder_outputs, encoder_hidden, forecast_days):
        batch_size = encoder_outputs.size(0)

        # Initialize decoder input (zeros for first time step)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim).to(encoder_outputs.device)

        # Use encoder's final hidden state as initial decoder hidden state
        decoder_hidden = encoder_hidden

        outputs = []

        for t in range(forecast_days):
            # Get context using attention (use last layer's hidden state)
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs)

            # Concatenate decoder input with context
            gru_input = torch.cat((decoder_input, context.unsqueeze(1)), dim=2)

            # GRU step
            gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)

            # Generate prediction
            prediction = self.fc(gru_output)  # (batch, 1, output_dim)

            outputs.append(prediction)

            # Use prediction as next input (autoregressive)
            decoder_input = prediction

        # Stack all predictions
        return torch.cat(outputs, dim=1)  # (batch, forecast_days, output_dim)


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model with encoder, decoder, and attention"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, forecast_days=FORECAST_DAYS):
        # Encode input sequence
        encoder_outputs, encoder_hidden = self.encoder(x)

        # Decode to generate forecast
        predictions = self.decoder(encoder_outputs, encoder_hidden, forecast_days)

        return predictions


# Instantiate model
encoder = Encoder(N_FEATURES, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
decoder = Decoder(HIDDEN_DIM, 1, NUM_LAYERS, DROPOUT).to(device)  # output_dim=1 for TG only
model = Seq2Seq(encoder, decoder).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model architecture:")
print(f"  Encoder: {NUM_LAYERS}-layer GRU with {HIDDEN_DIM} hidden units")
print(f"  Decoder: {NUM_LAYERS}-layer autoregressive with attention mechanism")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n[8/9] Setting up training...")

# Loss and optimizer
criterion = nn.L1Loss()  # MAE Loss (L1 norm)
LEARNING_RATE = 0.0001
PATIENCE = 20
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

print(f"Training configuration:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Loss function: L1Loss (MAE)")
print(f"  Optimizer: Adam")
print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n[9/9] Training model...")

# Start timing GRU (training + evaluation)
gru_start_time = time.time()

N_EPOCHS = 100
MAX_GRAD_NORM = 1.0

best_val_loss = float('inf')
epochs_no_improve = 0
prev_train_loss = float('inf')
prev_val_loss = float('inf')

for epoch in range(N_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Forward pass
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()

        train_loss += loss.item() * x_batch.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * x_batch.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)

    # Determine colors based on loss changes
    train_color = Colors.GREEN if avg_train_loss < prev_train_loss else Colors.RED if avg_train_loss > prev_train_loss else Colors.ENDC
    val_color = Colors.GREEN if avg_val_loss < prev_val_loss else Colors.RED if avg_val_loss > prev_val_loss else Colors.ENDC

    # Print progress with colors
    print(f"{Colors.BOLD}Epoch {epoch+1}/{N_EPOCHS}{Colors.ENDC} | "
          f"Train Loss: {train_color}{avg_train_loss:.6f}{Colors.ENDC} | "
          f"Val Loss: {val_color}{avg_val_loss:.6f}{Colors.ENDC} | "
          f"LR: {Colors.CYAN}{optimizer.param_groups[0]['lr']:.6f}{Colors.ENDC}")

    # Update previous losses
    prev_train_loss = avg_train_loss
    prev_val_loss = avg_val_loss

    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model_temperature.pth')
        best_model_time = time.time()  # Track time when best model was found
        print(f"  {Colors.GREEN}-> Best model saved (val_loss: {best_val_loss:.6f}){Colors.ENDC}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\n{Colors.YELLOW}Early stopping triggered after {epoch+1} epochs{Colors.ENDC}")
            break

print(f"\n{Colors.GREEN}{Colors.BOLD}Training complete!{Colors.ENDC}")

# ============================================================================
# EVALUATION
# ============================================================================

print("Evaluating...")

# Load best model
model.load_state_dict(torch.load('best_model_temperature.pth'))
model.eval()

print(f"\n{Colors.GREEN}{Colors.BOLD}Evaluation complete!{Colors.ENDC}")

# Get predictions on test set
all_predictions = []
all_actuals = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_pred = model(x_batch).cpu()
        all_predictions.append(y_pred)
        all_actuals.append(y_batch)

predictions_scaled = torch.cat(all_predictions, dim=0).numpy()
actuals_scaled = torch.cat(all_actuals, dim=0).numpy()

# Unscale predictions
predictions = y_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).reshape(predictions_scaled.shape)

actuals = y_scaler.inverse_transform(
    actuals_scaled.reshape(-1, 1)
).reshape(actuals_scaled.shape)

# End timing GRU
gru_time = best_model_time - gru_start_time

# ============================================================================
# BENCHMARK MODELS
# ============================================================================

# Compute benchmark predictions
persistent_predictions, sarima_predictions, persistent_time, sarima_time = benchmarks.compute_benchmarks(
    actuals=actuals,
    x_test=x_test,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    forecast_days=FORECAST_DAYS,
    Colors=Colors,
    test_dates=test_dates
)

# ============================================================================
# METRICS (FULL TEST SET 2023-2025)
# ============================================================================

print(f"\n{Colors.BOLD}{Colors.HEADER}Comprehensive Metrics (2023-2025 Full Test Set):{Colors.ENDC}")
print(Colors.HEADER + "-" * 80 + Colors.ENDC)

# Use all test samples (2023-2025)
n_samples_for_metrics = len(predictions)

# Calculate metrics for all three models on full test set
actual_flat = actuals[:n_samples_for_metrics].flatten()

# GRU Model
pred_flat_gru = predictions[:n_samples_for_metrics].flatten()
mae_gru = mean_absolute_error(actual_flat, pred_flat_gru)
rmse_gru = np.sqrt(mean_squared_error(actual_flat, pred_flat_gru))
r2_gru = r2_score(actual_flat, pred_flat_gru)

# Persistent Model
pred_flat_persistent = persistent_predictions[:n_samples_for_metrics].flatten()
mae_persistent = mean_absolute_error(actual_flat, pred_flat_persistent)
rmse_persistent = np.sqrt(mean_squared_error(actual_flat, pred_flat_persistent))
r2_persistent = r2_score(actual_flat, pred_flat_persistent)

# SARIMA Model
pred_flat_sarima = sarima_predictions[:n_samples_for_metrics].flatten()
mae_sarima = mean_absolute_error(actual_flat, pred_flat_sarima)
rmse_sarima = np.sqrt(mean_squared_error(actual_flat, pred_flat_sarima))
r2_sarima = r2_score(actual_flat, pred_flat_sarima)

print(f"\n{Colors.BOLD}{'Model':<20} {'MAE (0.1°C)':<15} {'MAE (°C)':<12} {'RMSE (0.1°C)':<16} {'RMSE (°C)':<12} {'R²':<10}{Colors.ENDC}")
print("-" * 100)
print(f"{Colors.BLUE}{Colors.BOLD}{'GRU (Ours)':<20}{Colors.ENDC} {Colors.GREEN}{mae_gru:<15.2f} {mae_gru/10:<12.2f} {rmse_gru:<16.2f} {rmse_gru/10:<12.2f}{Colors.ENDC} {Colors.CYAN}{r2_gru:<10.4f}{Colors.ENDC}")
print(f"{'Persistent':<20} {mae_persistent:<15.2f} {mae_persistent/10:<12.2f} {rmse_persistent:<16.2f} {rmse_persistent/10:<12.2f} {r2_persistent:<10.4f}")
print(f"{'SARIMA':<20} {mae_sarima:<15.2f} {mae_sarima/10:<12.2f} {rmse_sarima:<16.2f} {rmse_sarima/10:<12.2f} {r2_sarima:<10.4f}")

# Per-day metrics for all models
print(f"\n{Colors.BOLD}Per-Day Forecast Performance (2023-2025 Full Test Set):{Colors.ENDC}")
print(f"{Colors.BOLD}{'Day':<6} {'Model':<15} {'MAE (°C)':<12} {'RMSE (°C)':<12} {'R²':<10}{Colors.ENDC}")
print("-" * 60)
for day in range(FORECAST_DAYS):
    actual_day = actuals[:n_samples_for_metrics, day, 0]

    # GRU
    pred_day_gru = predictions[:n_samples_for_metrics, day, 0]
    mae_gru_day = mean_absolute_error(actual_day, pred_day_gru)
    rmse_gru_day = np.sqrt(mean_squared_error(actual_day, pred_day_gru))
    r2_gru_day = r2_score(actual_day, pred_day_gru)

    # Persistent
    pred_day_persistent = persistent_predictions[:n_samples_for_metrics, day, 0]
    mae_persistent_day = mean_absolute_error(actual_day, pred_day_persistent)
    rmse_persistent_day = np.sqrt(mean_squared_error(actual_day, pred_day_persistent))
    r2_persistent_day = r2_score(actual_day, pred_day_persistent)

    # SARIMA
    pred_day_sarima = sarima_predictions[:n_samples_for_metrics, day, 0]
    mae_sarima_day = mean_absolute_error(actual_day, pred_day_sarima)
    rmse_sarima_day = np.sqrt(mean_squared_error(actual_day, pred_day_sarima))
    r2_sarima_day = r2_score(actual_day, pred_day_sarima)

    print(f"{Colors.BOLD}{day+1:<6}{Colors.ENDC} {Colors.BLUE}{'GRU':<15}{Colors.ENDC} {Colors.GREEN}{mae_gru_day/10:<12.2f} {rmse_gru_day/10:<12.2f}{Colors.ENDC} {Colors.CYAN}{r2_gru_day:<10.4f}{Colors.ENDC}")
    print(f"{'':6} {'Persistent':<15} {mae_persistent_day/10:<12.2f} {rmse_persistent_day/10:<12.2f} {r2_persistent_day:<10.4f}")
    print(f"{'':6} {'SARIMA':<15} {mae_sarima_day/10:<12.2f} {rmse_sarima_day/10:<12.2f} {r2_sarima_day:<10.4f}")
    print()

# Execution Time Comparison
print(f"\n{Colors.BOLD}Execution Time (Training + Evaluation):{Colors.ENDC}")
print(f"{Colors.BOLD}{'Model':<20} {'Time':<25} {'Time (minutes)':<15}{Colors.ENDC}")
print("-" * 65)
print(f"{Colors.BLUE}{Colors.BOLD}{'GRU (Ours)':<20}{Colors.ENDC} {Colors.GREEN}{gru_time:<10.2f}s (to best model) {gru_time/60:<15.2f}{Colors.ENDC}")
print(f"{'Persistent':<20} {persistent_time*1000:<10.2f}ms               {persistent_time/60:<15.4f}")
print(f"{'SARIMA':<20} {sarima_time:<10.2f}s                {sarima_time/60:<15.2f}")
print()

# ============================================================================
# TEST SET VISUALIZATION WITH BENCHMARKS
# ============================================================================

print(f"\n{Colors.CYAN}Generating visualization with benchmarks...{Colors.ENDC}")

# Use all test samples (2023-2025)
n_samples_to_plot = len(predictions)

fig, axes = plt.subplots(3, 1, figsize=(18, 11))

# Plot 1: Day 1 forecast comparison
day1_actual = actuals[:n_samples_to_plot, 0, 0]
day1_pred_gru = predictions[:n_samples_to_plot, 0, 0]
day1_pred_persistent = persistent_predictions[:n_samples_to_plot, 0, 0]
day1_pred_sarima = sarima_predictions[:n_samples_to_plot, 0, 0]

day1_mae_gru = mean_absolute_error(day1_actual, day1_pred_gru)
day1_rmse_gru = np.sqrt(mean_squared_error(day1_actual, day1_pred_gru))
day1_r2_gru = r2_score(day1_actual, day1_pred_gru)
day1_mae_persistent = mean_absolute_error(day1_actual, day1_pred_persistent)
day1_rmse_persistent = np.sqrt(mean_squared_error(day1_actual, day1_pred_persistent))
day1_r2_persistent = r2_score(day1_actual, day1_pred_persistent)
day1_mae_sarima = mean_absolute_error(day1_actual, day1_pred_sarima)
day1_rmse_sarima = np.sqrt(mean_squared_error(day1_actual, day1_pred_sarima))
day1_r2_sarima = r2_score(day1_actual, day1_pred_sarima)

axes[0].plot(range(n_samples_to_plot), day1_actual,
             'k-', label='Actual', linewidth=1.5, alpha=0.7)
axes[0].plot(range(n_samples_to_plot), day1_pred_gru,
             'b-', label='GRU', linewidth=1, alpha=0.7)
axes[0].plot(range(n_samples_to_plot), day1_pred_persistent,
             'g--', label='Persistent', linewidth=1, alpha=0.6)
axes[0].plot(range(n_samples_to_plot), day1_pred_sarima,
             'r-.', label='SARIMA', linewidth=1, alpha=0.6)
axes[0].set_title('1-Day Ahead Temperature Forecast - Model Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Temperature (0.1°C)', fontsize=10)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Add error metrics text box
textstr = (f'GRU:        MAE={day1_mae_gru/10:.2f}°C, RMSE={day1_rmse_gru/10:.2f}°C, R²={day1_r2_gru:.4f}\n'
           f'Persistent: MAE={day1_mae_persistent/10:.2f}°C, RMSE={day1_rmse_persistent/10:.2f}°C, R²={day1_r2_persistent:.4f}\n'
           f'SARIMA:     MAE={day1_mae_sarima/10:.2f}°C, RMSE={day1_rmse_sarima/10:.2f}°C, R²={day1_r2_sarima:.4f}')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes, fontsize=8,
             verticalalignment='top', bbox=props, family='monospace')

# Plot 2: Day 3 forecast comparison
day3_actual = actuals[:n_samples_to_plot, 2, 0]
day3_pred_gru = predictions[:n_samples_to_plot, 2, 0]
day3_pred_persistent = persistent_predictions[:n_samples_to_plot, 2, 0]
day3_pred_sarima = sarima_predictions[:n_samples_to_plot, 2, 0]

day3_mae_gru = mean_absolute_error(day3_actual, day3_pred_gru)
day3_rmse_gru = np.sqrt(mean_squared_error(day3_actual, day3_pred_gru))
day3_r2_gru = r2_score(day3_actual, day3_pred_gru)
day3_mae_persistent = mean_absolute_error(day3_actual, day3_pred_persistent)
day3_rmse_persistent = np.sqrt(mean_squared_error(day3_actual, day3_pred_persistent))
day3_r2_persistent = r2_score(day3_actual, day3_pred_persistent)
day3_mae_sarima = mean_absolute_error(day3_actual, day3_pred_sarima)
day3_rmse_sarima = np.sqrt(mean_squared_error(day3_actual, day3_pred_sarima))
day3_r2_sarima = r2_score(day3_actual, day3_pred_sarima)

axes[1].plot(range(n_samples_to_plot), day3_actual,
             'k-', label='Actual', linewidth=1.5, alpha=0.7)
axes[1].plot(range(n_samples_to_plot), day3_pred_gru,
             'b-', label='GRU', linewidth=1, alpha=0.7)
axes[1].plot(range(n_samples_to_plot), day3_pred_persistent,
             'g--', label='Persistent', linewidth=1, alpha=0.6)
axes[1].plot(range(n_samples_to_plot), day3_pred_sarima,
             'r-.', label='SARIMA', linewidth=1, alpha=0.6)
axes[1].set_title('3-Day Ahead Temperature Forecast - Model Comparison', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Temperature (0.1°C)', fontsize=10)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Add error metrics text box
textstr = (f'GRU:        MAE={day3_mae_gru/10:.2f}°C, RMSE={day3_rmse_gru/10:.2f}°C, R²={day3_r2_gru:.4f}\n'
           f'Persistent: MAE={day3_mae_persistent/10:.2f}°C, RMSE={day3_rmse_persistent/10:.2f}°C, R²={day3_r2_persistent:.4f}\n'
           f'SARIMA:     MAE={day3_mae_sarima/10:.2f}°C, RMSE={day3_rmse_sarima/10:.2f}°C, R²={day3_r2_sarima:.4f}')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[1].text(0.02, 0.98, textstr, transform=axes[1].transAxes, fontsize=8,
             verticalalignment='top', bbox=props, family='monospace')

# Plot 3: MAE and RMSE comparison across forecast days
days = np.arange(1, FORECAST_DAYS + 1)
mae_gru_per_day = []
rmse_gru_per_day = []
mae_persistent_per_day = []
rmse_persistent_per_day = []
mae_sarima_per_day = []
rmse_sarima_per_day = []

for day in range(FORECAST_DAYS):
    actual_day = actuals[:n_samples_to_plot, day, 0]

    # GRU
    pred_gru_day = predictions[:n_samples_to_plot, day, 0]
    mae_gru_per_day.append(mean_absolute_error(actual_day, pred_gru_day) / 10)
    rmse_gru_per_day.append(np.sqrt(mean_squared_error(actual_day, pred_gru_day)) / 10)

    # Persistent
    pred_persistent_day = persistent_predictions[:n_samples_to_plot, day, 0]
    mae_persistent_per_day.append(mean_absolute_error(actual_day, pred_persistent_day) / 10)
    rmse_persistent_per_day.append(np.sqrt(mean_squared_error(actual_day, pred_persistent_day)) / 10)

    # SARIMA
    pred_sarima_day = sarima_predictions[:n_samples_to_plot, day, 0]
    mae_sarima_per_day.append(mean_absolute_error(actual_day, pred_sarima_day) / 10)
    rmse_sarima_per_day.append(np.sqrt(mean_squared_error(actual_day, pred_sarima_day)) / 10)

ax3_twin = axes[2].twinx()

# Plot MAE on left y-axis
axes[2].plot(days, mae_gru_per_day, 'b-o', label='GRU MAE', linewidth=2, markersize=6)
axes[2].plot(days, mae_persistent_per_day, 'g--s', label='Persistent MAE', linewidth=2, markersize=6)
axes[2].plot(days, mae_sarima_per_day, 'r-.^', label='SARIMA MAE', linewidth=2, markersize=6)

# Plot RMSE on right y-axis
ax3_twin.plot(days, rmse_gru_per_day, 'b:o', label='GRU RMSE', linewidth=2, markersize=6, alpha=0.6)
ax3_twin.plot(days, rmse_persistent_per_day, 'g:s', label='Persistent RMSE', linewidth=2, markersize=6, alpha=0.6)
ax3_twin.plot(days, rmse_sarima_per_day, 'r:^', label='SARIMA RMSE', linewidth=2, markersize=6, alpha=0.6)

axes[2].set_title('Model Performance Across Forecast Horizon', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Forecast Day', fontsize=10)
axes[2].set_ylabel('MAE (°C)', fontsize=10)
ax3_twin.set_ylabel('RMSE (°C)', fontsize=10)
axes[2].set_xticks(days)
axes[2].grid(True, alpha=0.3)

# Add execution time text box (bottom right)
time_textstr = (f'Execution Time:\n'
                f'GRU (to best):  {gru_time:.1f}s ({gru_time/60:.2f}min)\n'
                f'Persistent:     {persistent_time*1000:.1f}ms\n'
                f'SARIMA:         {sarima_time:.1f}s ({sarima_time/60:.2f}min)')
time_props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
axes[2].text(0.98, 0.02, time_textstr, transform=axes[2].transAxes, fontsize=8,
             verticalalignment='bottom', horizontalalignment='right', bbox=time_props, family='monospace')

# Combine legends
lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.suptitle(f'Temperature Forecast Evaluation with Benchmarks - Full Test Set ({test_dates[0].strftime("%Y-%m-%d")} to {test_dates[n_samples_to_plot-1].strftime("%Y-%m-%d")})',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('./figures/temperature_forecast_evaluation.png', dpi=800, bbox_inches='tight')
print(f"  {Colors.GREEN}Saved: ./figures/temperature_forecast_evaluation.png{Colors.ENDC}")

plt.show()
