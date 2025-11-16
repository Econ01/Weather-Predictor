"""
Weather Temperature Prediction Model - Phase 1
Single-task model for 7-day temperature forecasting

Features:
- 26 input variables (12 base + 12 lagged + 2 rolling statistics)
  - Base features: TG, TN, TX, RR, SS, HU, FG, FX, CC, SD, DAY_SIN, DAY_COS
  - Lagged features: LAG1 and LAG7 for TG, TN, TX, RR, HU, FG
  - Rolling statistics: 7-day mean and std for TG
  - Excludes PP and QQ due to data quality issues
- 7-day forecast horizon
- Year-based train/val/test split (1957-2022 / 2023 / 2024-2025)
- Autoregressive decoder with attention mechanism
- Gradient clipping for stability
- Comprehensive evaluation metrics and visualization
"""

import numpy as np
import pandas as pd
import dataManager as dm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import subprocess
import webbrowser
import time
import matplotlib.pyplot as plt

print("="*80)
print("PHASE 1: Temperature Prediction Model with Enhanced Features")
print("="*80)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1/9] Loading data...")
# Load the clean data from dataManager (now includes lagged and rolling features)
clean_df, feature_cols, target_cols = dm.load_and_process_data(run_plots=False, save_plots=False)

# Use features from dataManager (26 variables with lagged and rolling features)
FEATURE_COLS = feature_cols

# Target: Only TG (Mean Temperature)
TARGET_COL = 'TG'

# Configuration
INPUT_DAYS = 30
FORECAST_DAYS = 7  # Increased from 3
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

    # Output: 7-day forecast of TG only
    target_slice = target_data[i + INPUT_DAYS : i + INPUT_DAYS + FORECAST_DAYS, :]

    x_data.append(input_slice)
    y_data.append(target_slice)
    date_indices.append(i + INPUT_DAYS)  # First forecast day

x_data = np.stack(x_data)
y_data = np.stack(y_data)

print(f"Sequence creation complete!")
print(f"  X_data shape: {x_data.shape}  # (samples, 30 days, {N_FEATURES} features)")
print(f"  y_data shape: {y_data.shape}  # (samples, 7 days, 1 target)")

# ============================================================================
# YEAR-BASED DATA SPLITTING
# ============================================================================

print("\n[3/9] Splitting data by year...")

# Find split indices based on dates
train_end_date = pd.Timestamp('2022-12-31')
val_end_date = pd.Timestamp('2023-12-31')

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

print("\n[4/9] Saving datasets to CSV...")

# Create subset of clean_df excluding PP and QQ
columns_to_keep = [col for col in clean_df.columns if col not in ['PP', 'QQ']]
clean_df_filtered = clean_df[columns_to_keep]

# Split by date
train_df = clean_df_filtered[clean_df_filtered.index <= train_end_date]
val_df = clean_df_filtered[(clean_df_filtered.index > train_end_date) & (clean_df_filtered.index <= val_end_date)]
test_df = clean_df_filtered[clean_df_filtered.index > val_end_date]

# Save to CSV
train_df.to_csv('train_data.csv')
val_df.to_csv('val_data.csv')
test_df.to_csv('test_data.csv')

print(f"  Saved train_data.csv ({len(train_df)} rows, {len(train_df.columns)} columns)")
print(f"  Saved val_data.csv ({len(val_df)} rows, {len(val_df.columns)} columns)")
print(f"  Saved test_data.csv ({len(test_df)} rows, {len(test_df.columns)} columns)")
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

HIDDEN_DIM = 256
NUM_LAYERS = 2
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
print(f"  Decoder: Autoregressive with attention mechanism")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n[8/9] Setting up training...")

# Loss and optimizer
criterion = nn.MSELoss()
LEARNING_RATE = 0.001  # Increased from 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)
os.makedirs(os.path.dirname(log_dir), exist_ok=True)

print(f"Training configuration:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Loss function: MSELoss")
print(f"  Optimizer: Adam")
print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")

# Launch TensorBoard
print("\nLaunching TensorBoard...")
try:
    subprocess.Popen(['tensorboard', '--logdir', 'logs/fit', '--port', '6006'])
    time.sleep(3)
    webbrowser.open('http://localhost:6006/')
    print("  TensorBoard launched at http://localhost:6006/")
except Exception as e:
    print(f"  Could not auto-launch TensorBoard: {e}")
    print("  Run manually: tensorboard --logdir logs/fit")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n[9/9] Training model...")

N_EPOCHS = 100  # Increased from 50
PATIENCE = 15  # Increased from 5
MAX_GRAD_NORM = 1.0  # Gradient clipping

best_val_loss = float('inf')
epochs_no_improve = 0

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

    # Log to TensorBoard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # Print progress
    print(f"Epoch {epoch+1}/{N_EPOCHS} | "
          f"Train Loss: {avg_train_loss:.6f} | "
          f"Val Loss: {avg_val_loss:.6f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model_temperature.pth')
        print(f"  → Best model saved (val_loss: {best_val_loss:.6f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

writer.close()
print("\nTraining complete!")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

# Load best model
model.load_state_dict(torch.load('best_model_temperature.pth'))
model.eval()

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

# ============================================================================
# METRICS
# ============================================================================

print("\nComprehensive Metrics:")
print("-" * 80)

# Overall metrics
pred_flat = predictions.flatten()
actual_flat = actuals.flatten()

mae_overall = mean_absolute_error(actual_flat, pred_flat)
rmse_overall = np.sqrt(mean_squared_error(actual_flat, pred_flat))
r2_overall = r2_score(actual_flat, pred_flat)

print(f"Overall Performance:")
print(f"  MAE:  {mae_overall:.2f} (0.1°C units = {mae_overall/10:.2f}°C)")
print(f"  RMSE: {rmse_overall:.2f} (0.1°C units = {rmse_overall/10:.2f}°C)")
print(f"  R²:   {r2_overall:.4f}")

# Per-day metrics
print(f"\nPer-Day Forecast Performance:")
for day in range(FORECAST_DAYS):
    pred_day = predictions[:, day, 0]
    actual_day = actuals[:, day, 0]

    mae = mean_absolute_error(actual_day, pred_day)
    rmse = np.sqrt(mean_squared_error(actual_day, pred_day))
    r2 = r2_score(actual_day, pred_day)

    print(f"  Day {day+1}: MAE={mae:.2f} ({mae/10:.2f}°C), "
          f"RMSE={rmse:.2f} ({rmse/10:.2f}°C), R²={r2:.4f}")

# ============================================================================
# YEAR-LONG VISUALIZATION
# ============================================================================

print("\nGenerating year-long visualization...")

# Use first 365 samples from test set (or all if less)
n_samples_to_plot = min(365, len(predictions))

fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# Plot 1: Day 1 forecast (most accurate)
axes[0].plot(range(n_samples_to_plot), actuals[:n_samples_to_plot, 0, 0],
             'b-', label='Actual', linewidth=1.5, alpha=0.7)
axes[0].plot(range(n_samples_to_plot), predictions[:n_samples_to_plot, 0, 0],
             'r-', label='Predicted', linewidth=1, alpha=0.7)
axes[0].set_title('1-Day Ahead Temperature Forecast', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Temperature (0.1°C)', fontsize=10)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: Day 3 forecast
axes[1].plot(range(n_samples_to_plot), actuals[:n_samples_to_plot, 2, 0],
             'b-', label='Actual', linewidth=1.5, alpha=0.7)
axes[1].plot(range(n_samples_to_plot), predictions[:n_samples_to_plot, 2, 0],
             'r-', label='Predicted', linewidth=1, alpha=0.7)
axes[1].set_title('3-Day Ahead Temperature Forecast', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Temperature (0.1°C)', fontsize=10)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Plot 3: Day 7 forecast (most challenging)
axes[2].plot(range(n_samples_to_plot), actuals[:n_samples_to_plot, 6, 0],
             'b-', label='Actual', linewidth=1.5, alpha=0.7)
axes[2].plot(range(n_samples_to_plot), predictions[:n_samples_to_plot, 6, 0],
             'r-', label='Predicted', linewidth=1, alpha=0.7)
axes[2].set_title('7-Day Ahead Temperature Forecast', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Temperature (0.1°C)', fontsize=10)
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

# Plot 4: Residuals (errors) over time
residuals_day1 = actuals[:n_samples_to_plot, 0, 0] - predictions[:n_samples_to_plot, 0, 0]
residuals_day7 = actuals[:n_samples_to_plot, 6, 0] - predictions[:n_samples_to_plot, 6, 0]

axes[3].plot(range(n_samples_to_plot), residuals_day1,
             'g-', label='Day 1 Error', linewidth=0.8, alpha=0.6)
axes[3].plot(range(n_samples_to_plot), residuals_day7,
             'orange', label='Day 7 Error', linewidth=0.8, alpha=0.6)
axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[3].set_title('Prediction Errors Over Time', fontsize=12, fontweight='bold')
axes[3].set_xlabel('Sample Index (days)', fontsize=10)
axes[3].set_ylabel('Error (0.1°C)', fontsize=10)
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)

plt.suptitle(f'Temperature Forecast Evaluation ({test_dates[0].strftime("%Y-%m-%d")} to {test_dates[n_samples_to_plot-1].strftime("%Y-%m-%d")})',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('temperature_forecast_evaluation.png', dpi=300, bbox_inches='tight')
print(f"  Saved: temperature_forecast_evaluation.png")

plt.show()

print("\n" + "="*80)
print("PHASE 1 COMPLETE!")
print("="*80)
print(f"\nOutputs:")
print(f"  - Model: best_model_temperature.pth")
print(f"  - Training data: train_data.csv")
print(f"  - Validation data: val_data.csv")
print(f"  - Test data: test_data.csv")
print(f"  - Visualization: temperature_forecast_evaluation.png")
print(f"  - TensorBoard logs: {log_dir}")
print(f"\nNext steps:")
print(f"  1. Review metrics and visualization")
print(f"  2. Analyze errors to identify improvement areas")
print(f"  3. Consider hyperparameter tuning if needed")
print(f"  4. Apply learnings to other target variables")
