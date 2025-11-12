import numpy as np
import pandas as pd
import dataManager as dm
from sklearn.preprocessing import StandardScaler
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
y_scaler_reg = StandardScaler().fit(y_train[:, :, 0:2].reshape(-1, N_TARGETS - 1))

# We copy the original arrays to preserve the unscaled IS_RAIN column
y_train_scaled = y_train.copy()
y_val_scaled = y_val.copy()
y_test_scaled = y_test.copy()

# Transform and reshape ONLY the regression columns (0 and 1)
y_train_scaled[:, :, 0:2] = y_scaler_reg.transform(y_train[:, :, 0:2].reshape(-1, N_TARGETS - 1)).reshape(y_train.shape[0], FORECAST_DAYS, N_TARGETS - 1)
y_val_scaled[:, :, 0:2] = y_scaler_reg.transform(y_val[:, :, 0:2].reshape(-1, N_TARGETS - 1)).reshape(y_val.shape[0], FORECAST_DAYS, N_TARGETS - 1)
y_test_scaled[:, :, 0:2] = y_scaler_reg.transform(y_test[:, :, 0:2].reshape(-1, N_TARGETS - 1)).reshape(y_test.shape[0], FORECAST_DAYS, N_TARGETS - 1)

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
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
HIDDEN_DIM = 256          # Size of the "context vector"

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)

    def forward(self, x):
        # x shape: (batch_size, 30, 14)
        # hidden shape: (2, batch_size, 256) -> (num_layers, batch, hidden)
        _ , hidden = self.gru(x)
        # We pass the full hidden state (both layers) to the decoder
        return hidden

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        # The GRU's input will be the repeated context vector
        # The output will be the predicted sequence
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        # This linear layer acts like the "TimeDistributed(Dense)"
        # It maps the 64 hidden dims to our 3 target dims
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_context):
        # hidden_context shape: (2, batch_size, 256)
        
        # Bridge (Mimic Keras's RepeatVector)
        # We use the *last* layer of the hidden state
        # to create the input for the decoder.
        # hidden_context[-1] gives (batch_size, 256)
        # .unsqueeze(1) gives (batch_size, 1, 256)
        # .repeat(...) gives (batch_size, 3, 256)
        last_layer_hidden = hidden_context[-1]
        decoder_input = last_layer_hidden.unsqueeze(1).repeat(1, FORECAST_DAYS, 1)
        
        # Decoder
        # We feed in the "Bridge" input
        # and the *entire* (2-layer) hidden context from the encoder
        decoder_output, _ = self.gru(decoder_input, hidden_context)
        
        # Output
        # prediction shape: (batch_size, 3, 3)
        prediction = self.fc(decoder_output)
        return prediction

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # x shape: (batch_size, 30, 14)
        hidden_context = self.encoder(x)

        # hidden_context shape: (1, batch_size, 64)
        prediction = self.decoder(hidden_context)
        # prediction shape: (batch_size, 3, 3)
        return prediction
    
print("Model classes defined")

# ------------------------------------------------
# Instantiate the model
encoder = Encoder(INPUT_DIM, HIDDEN_DIM).to(device)
decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM).to(device)
model = Seq2Seq(encoder, decoder).to(device)

# Loss function and optimizer
regression_loss_fn = nn.MSELoss()  # For TG and FG (Regression)
classification_loss_fn = nn.BCEWithLogitsLoss() # For IS_RAIN (Classification)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# TensorBoard writer
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)
# Create the logs directory if it doesn't exist
os.makedirs(os.path.dirname(log_dir), exist_ok=True)

print("Launching TensorBoard in the background...")
# Start TensorBoard as a separate process
# This won't block our Python script
try:
    subprocess.Popen(['tensorboard', '--logdir', 'logs/fit', '--port', '6006'])
    
    # Give the server 5 seconds to start
    time.sleep(5)
    
    # Open the browser
    webbrowser.open('http://localhost:6006/')
    print("TensorBoard server started. Opening browser...")
except Exception as e:
    print(f"  Could not auto-launch TensorBoard: {e}")
    print("  Please start it manually by running: tensorboard --logdir logs/fit")

# Early stopping parameters
N_EPOCHS = 50
PATIENCE = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# Training Loop
for epoch in range(N_EPOCHS):
    # Set model to training mode
    model.train()
    train_loss = 0.0
    
    for x_batch, y_batch in train_loader:
        # Move data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Forward pass
        y_pred = model(x_batch)
        
        # Calculate loss
        # y_pred and y_batch have shape: (batch_size, 3, 3)

        # 1. Regression Loss (for TG and FG)
        # Select targets 0 and 1
        loss_reg = regression_loss_fn(y_pred[:, :, 0:2], y_batch[:, :, 0:2])

        # 2. Classification Loss (for IS_RAIN)
        # Select target 2
        # We must squeeze the last dim for BCEWithLogitsLoss
        loss_class = classification_loss_fn(y_pred[:, :, 2].squeeze(-1), y_batch[:, :, 2].squeeze(-1))

        # 3. Combine the losses (we can weight them, but 1:1 is a great start)
        loss = loss_reg + loss_class
        train_loss += loss.item() * x_batch.size(0)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # Validation Loop
    # Set model to evaluation mode
    model.eval()
    val_loss = 0.0

    # Disable gradient calculation
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)

            # 1. Regression Loss (for TG and FG)
            loss_reg = regression_loss_fn(y_pred[:, :, 0:2], y_batch[:, :, 0:2])

            # 2. Classification Loss (for IS_RAIN)
            loss_class = classification_loss_fn(y_pred[:, :, 2].squeeze(-1), y_batch[:, :, 2].squeeze(-1))

            # 3. Combine the losses
            loss = loss_reg + loss_class

            val_loss += loss.item() * x_batch.size(0)
            
    avg_val_loss = val_loss / len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{N_EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Log to TensorBoard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    
    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0

        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == PATIENCE:
            print("Early stopping!")
            break

writer.close()
print("Training complete.")

# Load the best model weights
model.load_state_dict(torch.load('best_model.pth'))

# ------------------------------------------------
# Evaluation & Visualization
model.eval()
all_y_pred_scaled = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        
        y_pred_batch = model(x_batch)
        
        # Move prediction back to CPU and append
        all_y_pred_scaled.append(y_pred_batch.cpu())

# Concatenate all batches into one big tensor
y_pred_scaled_tensor = torch.cat(all_y_pred_scaled, dim=0)

# Convert to numpy for scaling
y_pred_scaled = y_pred_scaled_tensor.numpy()

# Un-scale the Predictions
# We copy the arrays to preserve the IS_RAIN column
y_pred_unscaled = y_pred_scaled.copy()
y_test_unscaled = y_test_scaled.copy()

# Un-scale ONLY the regression columns (0 and 1)
y_pred_unscaled[:, :, 0:2] = y_scaler_reg.inverse_transform(y_pred_scaled[:, :, 0:2].reshape(-1, N_TARGETS - 1)).reshape(y_pred_scaled.shape[0], FORECAST_DAYS, N_TARGETS - 1)
y_test_unscaled[:, :, 0:2] = y_scaler_reg.inverse_transform(y_test_scaled[:, :, 0:2].reshape(-1, N_TARGETS - 1)).reshape(y_test_scaled.shape[0], FORECAST_DAYS, N_TARGETS - 1)

# Pick a random sample index from the test set
idx = 100 

# Get the 3-day prediction and the 3-day actual values
pred_3_day = y_pred_unscaled[idx]
actual_3_day = y_test_unscaled[idx]

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
days = ['Day 1', 'Day 2', 'Day 3']

# Plot for Mean Temperature (TG)
axes[0].plot(days, actual_3_day[:, 0], 'bo-', label='Actual (TG)')
axes[0].plot(days, pred_3_day[:, 0], 'ro--', label='Predicted (TG)')
axes[0].set_title('Mean Temperature Forecast')
axes[0].set_ylabel('Temp (0.1 C)')
axes[0].legend()

# Plot for Wind Speed (FG)
axes[1].plot(days, actual_3_day[:, 1], 'bo-', label='Actual (FG)')
axes[1].plot(days, pred_3_day[:, 1], 'ro--', label='Predicted (FG)')
axes[1].set_title('Mean Wind Speed Forecast')
axes[1].set_ylabel('Wind (0.1 m/s)')
axes[1].legend()

# Plot for Precipitation Probability (IS_RAIN)
axes[2].bar(days, actual_3_day[:, 2], label='Actual (IS_RAIN)', color='blue', alpha=0.6)
axes[2].plot(days, pred_3_day[:, 2], 'ro--', label='Predicted (IS_RAIN Prob.)')
axes[2].set_title('Precipitation Probability Forecast')
axes[2].set_ylabel('Probability')
axes[2].set_ylim(-0.1, 1.1)
axes[2].legend()

plt.tight_layout()
plt.savefig("Output.png", dpi=800)
plt.show()