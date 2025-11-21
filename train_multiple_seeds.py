"""
Multi-Seed Training and Comparison Script
Trains the GRU temperature forecasting model with 10 different random seeds
and compares their 1-day forecast performance.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import warnings
import os
import time
import dataManager as dm

warnings.filterwarnings('ignore')

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# ============================================================================
# CONFIGURATION
# ============================================================================

SEEDS = [42, 123, 456, 789, 1011, 2022, 3141, 5926, 8888, 9999]
INPUT_DAYS = 15
FORECAST_DAYS = 3
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
N_EPOCHS = 100
PATIENCE = 20

# ============================================================================
# MODEL ARCHITECTURE (copied from main script)
# ============================================================================

class Encoder(nn.Module):
    """Encoder with GRU that returns both outputs and hidden states"""
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
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
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
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
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class Decoder(nn.Module):
    """Autoregressive decoder with attention"""
    def __init__(self, hidden_dim, output_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(
            output_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Orthogonal initialization
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, encoder_outputs, encoder_hidden, forecast_days):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim).to(encoder_outputs.device)
        decoder_hidden = encoder_hidden
        outputs = []

        for t in range(forecast_days):
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs)
            gru_input = torch.cat([decoder_input, context.unsqueeze(1)], dim=2)
            gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
            prediction = self.fc(gru_output)
            outputs.append(prediction)
            decoder_input = prediction

        return torch.cat(outputs, dim=1)


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, forecast_days=FORECAST_DAYS):
        encoder_outputs, encoder_hidden = self.encoder(x)
        predictions = self.decoder(encoder_outputs, encoder_hidden, forecast_days)
        return predictions


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model_with_seed(seed, train_loader, val_loader, n_features, device):
    """Train a single model with a specific seed"""
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Build model
    encoder = Encoder(n_features, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, 1, NUM_LAYERS, DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # Setup training
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    train_start = time.time()

    for epoch in range(N_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            best_model_time = time.time()
            torch.save(model.state_dict(), f'temp_model_seed_{seed}.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    # Load best model
    model.load_state_dict(torch.load(f'temp_model_seed_{seed}.pth'))
    training_time = best_model_time - train_start

    return model, best_val_loss, best_epoch, training_time


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print(f"{Colors.BOLD}{Colors.HEADER}Multi-Seed Model Training and Comparison{Colors.ENDC}")
    print(f"Training {len(SEEDS)} models with different random seeds\n")

    # Load data
    print(f"{Colors.CYAN}[1/4] Loading and preparing data...{Colors.ENDC}")
    clean_df, feature_cols, _ = dm.EngineerFeatures(dm.CleanData(dm.ReadAndMerge()))

    # Extract feature and target data
    feature_data = clean_df[feature_cols].values
    target_data = clean_df[['TG']].values
    dates = clean_df.index

    # Create sequences
    print("Creating sequences...")
    x_data = []
    y_data = []
    date_indices = []

    for i in range(len(feature_data) - INPUT_DAYS - FORECAST_DAYS + 1):
        input_slice = feature_data[i : i + INPUT_DAYS, :]
        target_slice = target_data[i + INPUT_DAYS : i + INPUT_DAYS + FORECAST_DAYS, :]
        x_data.append(input_slice)
        y_data.append(target_slice)
        date_indices.append(i + INPUT_DAYS)

    X_data = np.stack(x_data)
    y_data_arr = np.stack(y_data)

    # Split data by year
    train_end_date = pd.Timestamp('2017-12-31')
    val_end_date = pd.Timestamp('2022-12-31')

    train_idx = None
    val_idx = None

    for i, date_i in enumerate(date_indices):
        date = dates[date_i]
        if train_idx is None and date > train_end_date:
            train_idx = i
        if val_idx is None and date > val_end_date:
            val_idx = i
            break

    # Split the data
    X_train = X_data[:train_idx]
    y_train = y_data_arr[:train_idx]
    X_val = X_data[train_idx:val_idx]
    y_val = y_data_arr[train_idx:val_idx]
    X_test = X_data[val_idx:]
    y_test = y_data_arr[val_idx:]
    test_dates = [dates[date_i] for date_i in date_indices[val_idx:]]

    # Scale data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = X_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}\n")

    # Train models with different seeds
    print(f"{Colors.CYAN}[2/4] Training {len(SEEDS)} models with different seeds...{Colors.ENDC}\n")

    results = []
    models = []

    for i, seed in enumerate(SEEDS):
        print(f"{Colors.BOLD}Training model {i+1}/{len(SEEDS)} (seed={seed})...{Colors.ENDC}")

        model, best_val_loss, best_epoch, training_time = train_model_with_seed(
            seed, train_loader, val_loader, X_train.shape[-1], device
        )

        # Evaluate on test set
        model.eval()
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

        # Unscale
        predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(predictions_scaled.shape)
        actuals = y_scaler.inverse_transform(actuals_scaled.reshape(-1, 1)).reshape(actuals_scaled.shape)

        # Calculate metrics (1-day forecast only)
        day1_actual = actuals[:, 0, 0]
        day1_pred = predictions[:, 0, 0]

        mae = mean_absolute_error(day1_actual, day1_pred)
        rmse = np.sqrt(mean_squared_error(day1_actual, day1_pred))
        r2 = r2_score(day1_actual, day1_pred)

        results.append({
            'seed': seed,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'training_time': training_time
        })

        models.append((model, predictions, actuals))

        print(f"  MAE: {mae/10:.3f}°C, RMSE: {rmse/10:.3f}°C, R²: {r2:.4f}")
        print(f"  Training time: {training_time:.1f}s, Best epoch: {best_epoch}\n")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Calculate statistics
    print(f"\n{Colors.CYAN}[3/4] Results Summary (1-Day Forecast):{Colors.ENDC}")
    print(f"{Colors.BOLD}{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}{Colors.ENDC}")
    print("-" * 65)
    print(f"{'MAE (°C)':<15} {results_df['mae'].mean()/10:<12.3f} {results_df['mae'].std()/10:<12.3f} {results_df['mae'].min()/10:<12.3f} {results_df['mae'].max()/10:<12.3f}")
    print(f"{'RMSE (°C)':<15} {results_df['rmse'].mean()/10:<12.3f} {results_df['rmse'].std()/10:<12.3f} {results_df['rmse'].min()/10:<12.3f} {results_df['rmse'].max()/10:<12.3f}")
    print(f"{'R²':<15} {results_df['r2'].mean():<12.4f} {results_df['r2'].std():<12.4f} {results_df['r2'].min():<12.4f} {results_df['r2'].max():<12.4f}")
    print(f"{'Time (s)':<15} {results_df['training_time'].mean():<12.1f} {results_df['training_time'].std():<12.1f} {results_df['training_time'].min():<12.1f} {results_df['training_time'].max():<12.1f}")

    # Save results
    results_df.to_csv('./modifiedData/multi_seed_results.csv', index=False)
    print(f"\n{Colors.GREEN}Results saved to: ./modifiedData/multi_seed_results.csv{Colors.ENDC}")

    # Visualization
    print(f"\n{Colors.CYAN}[4/4] Creating visualization...{Colors.ENDC}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: MAE comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(SEEDS)), results_df['mae']/10, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(y=results_df['mae'].mean()/10, color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["mae"].mean()/10:.3f}°C')
    ax1.set_xlabel('Model (by seed)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MAE (°C)', fontsize=11, fontweight='bold')
    ax1.set_title('1-Day Forecast MAE Across Different Seeds', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(SEEDS)))
    ax1.set_xticklabels([f'{s}' for s in SEEDS], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{results_df["mae"].iloc[i]/10:.3f}',
                ha='center', va='bottom', fontsize=8)

    # Plot 2: R² comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(SEEDS)), results_df['r2'], color='forestgreen', alpha=0.7, edgecolor='black')
    ax2.axhline(y=results_df['r2'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["r2"].mean():.4f}')
    ax2.set_xlabel('Model (by seed)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('R²', fontsize=11, fontweight='bold')
    ax2.set_title('1-Day Forecast R² Across Different Seeds', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(SEEDS)))
    ax2.set_xticklabels([f'{s}' for s in SEEDS], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([results_df['r2'].min() - 0.001, results_df['r2'].max() + 0.001])

    # Plot 3: Actual vs Predicted (first 100 days, best model)
    ax3 = axes[1, 0]
    best_idx = results_df['mae'].idxmin()
    best_model, best_predictions, best_actuals = models[best_idx]

    n_plot = min(100, len(best_actuals))
    x_axis = range(n_plot)
    ax3.plot(x_axis, best_actuals[:n_plot, 0, 0], 'k-', label='Actual', linewidth=1.5, alpha=0.7)
    ax3.plot(x_axis, best_predictions[:n_plot, 0, 0], 'b-', label=f'Best Model (seed={SEEDS[best_idx]})', linewidth=1, alpha=0.7)
    ax3.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Temperature (0.1°C)', fontsize=11, fontweight='bold')
    ax3.set_title(f'1-Day Forecast: Best Model (MAE={results_df["mae"].iloc[best_idx]/10:.3f}°C)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Training time comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(SEEDS)), results_df['training_time'], color='coral', alpha=0.7, edgecolor='black')
    ax4.axhline(y=results_df['training_time'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["training_time"].mean():.1f}s')
    ax4.set_xlabel('Model (by seed)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title('Training Time to Best Model', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(SEEDS)))
    ax4.set_xticklabels([f'{s}' for s in SEEDS], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Multi-Seed Comparison: {len(SEEDS)} Models with Different Random Seeds',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('./figures/multi_seed_comparison.png', dpi=300, bbox_inches='tight')
    print(f"{Colors.GREEN}Visualization saved to: ./figures/multi_seed_comparison.png{Colors.ENDC}")

    # Cleanup temp model files
    for seed in SEEDS:
        if os.path.exists(f'temp_model_seed_{seed}.pth'):
            os.remove(f'temp_model_seed_{seed}.pth')

    print(f"\n{Colors.BOLD}{Colors.GREEN}Done! Trained {len(SEEDS)} models.{Colors.ENDC}")
    print(f"{Colors.BOLD}Average 1-day MAE: {results_df['mae'].mean()/10:.3f}°C ± {results_df['mae'].std()/10:.3f}°C{Colors.ENDC}")
    print(f"{Colors.BOLD}Best model: seed={SEEDS[best_idx]}, MAE={results_df['mae'].min()/10:.3f}°C{Colors.ENDC}")


if __name__ == '__main__':
    main()
