# Weather Predictor Improvement Plan

**Created**: 2025-11-13
**Status**: Initial Analysis Complete

---

## Current Model Performance Analysis

### Observed Issues (from Output.png)
1. **Temperature Prediction**: Model outputs linear trend, ignoring actual fluctuations
2. **Wind Speed Prediction**: Nearly constant output (~30 m/s), close to dataset mean (32.2)
3. **Precipitation Probability**: Near-zero probabilities, failing to learn pattern
4. **Overall Pattern**: Model has collapsed to outputting mean values (classic training failure sign)

### Dataset Statistics
- Total samples: 24,929 days (1957-2025)
- Features: 14 (TG, TN, TX, RR, PP, SS, HU, FG, FX, CC, SD, QQ, DAY_SIN, DAY_COS)
- Mean Temperature: 103.6 (0.1°C units = 10.36°C)
- Mean Wind Speed: 32.2 (0.1 m/s units = 3.22 m/s)
- Precipitation days: 48.4% of dataset

---

## Critical Issues (Fix Immediately)

### ISSUE #1: Data Leakage ⚠️
**Location**: `dataManager.py:330-334`, `model.py:17-40`

**Problem**:
- Features include: TG, FG
- Targets include: TG, FG
- Same variables used as both input and output creates confusion

**Current Code**:
```python
FEATURE_COLS = ['TG', 'TN', 'TX', 'RR', 'PP', 'SS', 'HU', 'FG', 'FX', 'CC', 'SD', 'QQ', 'DAY_SIN', 'DAY_COS']
TARGET_COLS = ['TG', 'FG', 'IS_RAIN']
```

**Solution**: Remove overlap between features and targets
```python
FEATURE_COLS = ['TN', 'TX', 'RR', 'PP', 'SS', 'HU', 'FG', 'FX', 'CC', 'SD', 'QQ', 'DAY_SIN', 'DAY_COS']
TARGET_COLS = ['TEMP_RANGE', 'RR', 'HU']  # NEW targets
```

**Status**: ✅ User agreed to new targets

---

### ISSUE #2: Poor Decoder Design
**Location**: `model.py:142-172`

**Problem**:
```python
# Current: Repeats same hidden state 3 times
decoder_input = last_layer_hidden.unsqueeze(1).repeat(1, FORECAST_DAYS, 1)
```
- No temporal variation in decoder input
- No way for decoder to distinguish Day 1 vs Day 2 vs Day 3
- No autoregressive connection (predictions don't feed into next step)

**Solution Options**:

**A) Autoregressive Decoder (Recommended)**
```python
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(output_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_context, forecast_days):
        batch_size = hidden_context.size(1)
        outputs = []

        # Start with zeros
        decoder_input = torch.zeros(batch_size, 1, self.output_dim).to(hidden_context.device)
        decoder_hidden = hidden_context

        for t in range(forecast_days):
            output, decoder_hidden = self.gru(decoder_input, decoder_hidden)
            prediction = self.fc(output)
            outputs.append(prediction)
            decoder_input = prediction  # Feed prediction to next step

        return torch.cat(outputs, dim=1)
```

**B) Positional Encoding**
```python
# Add positional embeddings to tell decoder which day it's predicting
position_embedding = nn.Embedding(FORECAST_DAYS, HIDDEN_DIM)
positions = torch.arange(FORECAST_DAYS).to(device)
pos_emb = position_embedding(positions)  # (3, 256)
decoder_input = last_layer_hidden.unsqueeze(1) + pos_emb.unsqueeze(0)
```

**Status**: ⏳ Pending implementation

---

### ISSUE #3: No Attention Mechanism
**Location**: `model.py:130-187`

**Problem**:
- Encoder outputs 30 timesteps, but only final hidden state used
- Loses temporal granularity (e.g., "3 days ago was warmer than yesterday")
- Weather patterns benefit from attending to specific past days

**Solution**: Add Bahdanau Attention
```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention_weights = torch.softmax(attention, dim=1)

        # context: (batch, hidden_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights

# Modify Encoder to return all outputs
class Encoder(nn.Module):
    def forward(self, x):
        outputs, hidden = self.gru(x)
        return outputs, hidden  # Return BOTH outputs and hidden
```

**Status**: ⏳ Pending implementation

---

## High Priority Issues

### ISSUE #4: Unweighted Hybrid Loss
**Location**: `model.py:246-256`, `model.py:281-288`

**Problem**:
```python
loss = loss_reg + loss_class  # Equal weighting
```
- MSE and BCE have different scales
- MSE likely dominates, causing model to ignore classification task
- Harder tasks (precipitation) should be weighted more

**Solution**: Add loss weighting
```python
# Define loss weights
REGRESSION_WEIGHT = 1.0
CLASSIFICATION_WEIGHT = 3.0  # Weight harder task more

# Use pos_weight for imbalanced classification
classification_loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([2.0])  # Rain is important
)

# In training loop
loss_reg = regression_loss_fn(y_pred[:, :, 0:2], y_batch[:, :, 0:2])
loss_class = classification_loss_fn(y_pred[:, :, 2], y_batch[:, :, 2])
loss = REGRESSION_WEIGHT * loss_reg + CLASSIFICATION_WEIGHT * loss_class
```

**Status**: ⏳ Pending implementation

---

### ISSUE #5: Aggressive Early Stopping
**Location**: `model.py:224-311`

**Problem**:
```python
PATIENCE = 5  # Too aggressive
N_EPOCHS = 50
lr = 0.0001  # Too conservative
```
- Complex weather patterns need more time to learn
- Low learning rate + early stopping = insufficient training

**Solution**: Adjust hyperparameters
```python
N_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 0.001  # 10x higher

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# In validation loop
scheduler.step(avg_val_loss)
```

**Status**: ⏳ Pending implementation

---

### ISSUE #6: Insufficient Model Capacity
**Location**: `model.py:125-129`

**Problem**:
```python
HIDDEN_DIM = 256
# Encoder: 2-layer GRU, dropout=0.2
```
- 14 features with complex temporal patterns
- Strong seasonal, weekly, daily cycles
- 256 hidden units may be too small

**Solution**: Increase capacity
```python
HIDDEN_DIM = 512  # Double capacity
NUM_LAYERS = 3    # Add one more layer
DROPOUT = 0.3     # Increase dropout to prevent overfitting

# Consider LSTM instead of GRU for long-term dependencies
self.lstm = nn.LSTM(
    input_dim, hidden_dim,
    num_layers=NUM_LAYERS,
    batch_first=True,
    dropout=DROPOUT
)
```

**Status**: ⏳ Pending implementation

---

### ISSUE #7: Missing Gradient Clipping
**Location**: `model.py:259-266`

**Problem**:
- No gradient clipping
- RNNs prone to exploding gradients
- Can cause training instability

**Solution**: Add gradient clipping
```python
# After loss.backward(), before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Status**: ⏳ Pending implementation

---

## Medium Priority Issues

### ISSUE #8: Limited Temporal Features
**Location**: `dataManager.py:315-343`

**Problem**:
- Only cyclical encodings (DAY_SIN, DAY_COS)
- No lagged features (yesterday's values)
- No rolling statistics (trends)
- No derivative features (temperature change)

**Solution**: Engineer richer features
```python
def EngineerFeatures(master_df):
    # Existing: IS_RAIN
    master_df['IS_RAIN'] = (master_df['RR'] > 1).astype(int)

    # NEW: Lagged features (yesterday, last week)
    for col in ['TG', 'RR', 'HU', 'FG', 'PP']:
        master_df[f'{col}_LAG1'] = master_df[col].shift(1)
        master_df[f'{col}_LAG7'] = master_df[col].shift(7)

    # NEW: Rolling statistics (7-day windows)
    master_df['TG_ROLLING_7_MEAN'] = master_df['TG'].rolling(window=7).mean()
    master_df['TG_ROLLING_7_STD'] = master_df['TG'].rolling(window=7).std()
    master_df['RR_ROLLING_7_SUM'] = master_df['RR'].rolling(window=7).sum()
    master_df['HU_ROLLING_7_MEAN'] = master_df['HU'].rolling(window=7).mean()

    # NEW: Derivative features
    master_df['TEMP_CHANGE'] = master_df['TG'] - master_df['TG'].shift(1)
    master_df['TEMP_RANGE'] = master_df['TX'] - master_df['TN']
    master_df['TEMP_VOLATILITY'] = master_df['TG'].rolling(window=7).std()

    # NEW: Month encoding (seasonal patterns)
    month = master_df.index.month
    master_df['MONTH_SIN'] = np.sin(2 * np.pi * month / 12)
    master_df['MONTH_COS'] = np.cos(2 * np.pi * month / 12)

    # Existing: Day of year
    day_of_year = master_df.index.dayofyear
    master_df['DAY_SIN'] = np.sin(2 * np.pi * day_of_year / 365.25)
    master_df['DAY_COS'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Drop NaN rows created by shifts/rolling
    master_df.dropna(inplace=True)

    # Updated feature list
    FEATURE_COLS = [
        # Raw weather variables
        'TN', 'TX', 'RR', 'PP', 'SS', 'FG', 'FX', 'CC', 'SD', 'QQ',
        # Lagged features
        'TG_LAG1', 'TG_LAG7', 'RR_LAG1', 'RR_LAG7', 'HU_LAG1', 'HU_LAG7',
        'FG_LAG1', 'FG_LAG7', 'PP_LAG1', 'PP_LAG7',
        # Rolling statistics
        'TG_ROLLING_7_MEAN', 'TG_ROLLING_7_STD', 'RR_ROLLING_7_SUM', 'HU_ROLLING_7_MEAN',
        # Derivative features
        'TEMP_CHANGE', 'TEMP_RANGE', 'TEMP_VOLATILITY',
        # Cyclical encodings
        'MONTH_SIN', 'MONTH_COS', 'DAY_SIN', 'DAY_COS'
    ]

    TARGET_COLS = ['TEMP_RANGE', 'RR', 'HU']

    # ... rest of function
```

**Status**: ⏳ Pending implementation

---

### ISSUE #9: Suboptimal Train/Val/Test Split
**Location**: `model.py:51-60`

**Problem**:
```python
n_train = int(n_samples * 0.8)
n_val = int(n_samples * 0.9)
```
- Weather data has strong temporal autocorrelation
- Recent years may have climate shift patterns
- 10% validation might be too small for reliable early stopping

**Solution**: Consider alternative splits
```python
# Option A: Larger validation set
n_train = int(n_samples * 0.7)  # 70% train
n_val = int(n_samples * 0.85)   # 15% validation
# 15% test

# Option B: Time-based split (use recent years for validation)
# E.g., train on 1957-2010, validate on 2011-2018, test on 2019-2025
```

**Status**: ⏳ Pending implementation

---

### ISSUE #10: No Model Evaluation Metrics
**Location**: `model.py:315-382`

**Problem**:
- Only visualizes one random sample (idx=100)
- No quantitative metrics (MAE, RMSE, R², F1 for classification)
- Can't objectively compare model versions

**Solution**: Add comprehensive evaluation
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score

def evaluate_model(model, test_loader, y_scaler_reg, device):
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch).cpu()
            all_predictions.append(y_pred)
            all_actuals.append(y_batch)

    predictions = torch.cat(all_predictions, dim=0).numpy()
    actuals = torch.cat(all_actuals, dim=0).numpy()

    # Un-scale regression targets
    predictions[:, :, 0:2] = y_scaler_reg.inverse_transform(
        predictions[:, :, 0:2].reshape(-1, 2)
    ).reshape(predictions.shape[0], FORECAST_DAYS, 2)
    actuals[:, :, 0:2] = y_scaler_reg.inverse_transform(
        actuals[:, :, 0:2].reshape(-1, 2)
    ).reshape(actuals.shape[0], FORECAST_DAYS, 2)

    # Calculate metrics for each target
    print("\n=== Evaluation Metrics ===")

    # Temperature Range
    mae_temp = mean_absolute_error(actuals[:, :, 0].flatten(), predictions[:, :, 0].flatten())
    rmse_temp = np.sqrt(mean_squared_error(actuals[:, :, 0].flatten(), predictions[:, :, 0].flatten()))
    r2_temp = r2_score(actuals[:, :, 0].flatten(), predictions[:, :, 0].flatten())
    print(f"Temperature Range - MAE: {mae_temp:.2f}, RMSE: {rmse_temp:.2f}, R²: {r2_temp:.3f}")

    # Precipitation
    mae_precip = mean_absolute_error(actuals[:, :, 1].flatten(), predictions[:, :, 1].flatten())
    rmse_precip = np.sqrt(mean_squared_error(actuals[:, :, 1].flatten(), predictions[:, :, 1].flatten()))
    print(f"Precipitation - MAE: {mae_precip:.2f}, RMSE: {rmse_precip:.2f}")

    # Humidity (if classification)
    humidity_pred = (predictions[:, :, 2] > 0.5).astype(int)
    humidity_actual = actuals[:, :, 2].astype(int)
    acc_humidity = accuracy_score(humidity_actual.flatten(), humidity_pred.flatten())
    f1_humidity = f1_score(humidity_actual.flatten(), humidity_pred.flatten())
    print(f"Humidity - Accuracy: {acc_humidity:.3f}, F1: {f1_humidity:.3f}")

    return {
        'temp_mae': mae_temp, 'temp_rmse': rmse_temp, 'temp_r2': r2_temp,
        'precip_mae': mae_precip, 'precip_rmse': rmse_precip,
        'humidity_acc': acc_humidity, 'humidity_f1': f1_humidity
    }
```

**Status**: ⏳ Pending implementation

---

## Low Priority / Alternative Approaches

### ISSUE #11: GRU vs LSTM vs Transformer
**Location**: `model.py:130-187`

**Current**: GRU-based Seq2Seq

**Alternatives**:

**A) LSTM (Better for long-term dependencies)**
```python
self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0.3)
```

**B) Transformer (State-of-the-art for sequences)**
```python
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)
```

**C) CNN-LSTM Hybrid (Extract spatial then temporal patterns)**
```python
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True)
```

**Status**: 📋 Consider after baseline improvements

---

### ISSUE #12: Multi-Task Learning with Separate Heads
**Location**: `model.py:174-187`

**Idea**: Instead of one decoder for all targets, use specialized decoders

```python
class MultiTaskSeq2Seq(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        # Separate decoder heads for each target
        self.temp_range_decoder = Decoder(HIDDEN_DIM, 1)  # Temperature range
        self.precipitation_decoder = Decoder(HIDDEN_DIM, 1)  # Precipitation
        self.humidity_decoder = Decoder(HIDDEN_DIM, 1)  # Humidity

    def forward(self, x):
        hidden = self.encoder(x)

        temp_pred = self.temp_range_decoder(hidden)
        precip_pred = self.precipitation_decoder(hidden)
        humidity_pred = self.humidity_decoder(hidden)

        return torch.cat([temp_pred, precip_pred, humidity_pred], dim=2)
```

**Benefits**:
- Each task has specialized parameters
- Can use task-specific architectures (e.g., different activation functions)
- Easier to debug per-task performance

**Status**: 📋 Consider for advanced optimization

---

### ISSUE #13: Ensemble Methods
**Current**: Single model

**Alternative**: Train multiple models and ensemble predictions

```python
# Train 5 models with different random seeds
models = []
for seed in [42, 123, 456, 789, 999]:
    torch.manual_seed(seed)
    model = Seq2Seq(encoder, decoder).to(device)
    # ... train model ...
    models.append(model)

# Ensemble predictions
def ensemble_predict(models, x):
    predictions = [model(x) for model in models]
    return torch.stack(predictions).mean(dim=0)
```

**Status**: 📋 Consider for production deployment

---

## New Target Variables (Agreed)

### Updated Targets
```python
TARGET_COLS = ['TEMP_RANGE', 'RR', 'HU']
```

**1. TEMP_RANGE** (Temperature Range)
- Formula: `TX - TN` (max temp - min temp)
- Why: More informative than mean temperature
- Captures daily temperature variability
- Important for agriculture, health, energy planning

**2. RR** (Precipitation Amount)
- Direct measurement in 0.1mm units
- Why: More useful than binary classification
- Tells you HOW MUCH rain, not just IF rain
- Can still derive binary classification if needed

**3. HU** (Humidity)
- Direct humidity measurement
- Why: Influences human comfort, health, agriculture
- Correlates with precipitation and temperature
- Missing from original targets

### Feature Set (No Overlap)
```python
FEATURE_COLS = [
    'TN', 'TX',           # Min/max temp (can derive TEMP_RANGE from these)
    'PP', 'SS',           # Pressure, sunshine
    'FG', 'FX',           # Wind speed, wind gust
    'CC', 'SD', 'QQ',     # Cloud cover, snow depth, radiation
    # Note: RR and HU removed from features since they're targets
    # Note: TG removed (redundant with TN/TX)
    'DAY_SIN', 'DAY_COS'  # Temporal encoding
]
```

**Status**: ✅ Ready to implement

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Immediate)
- [ ] **Task 1.1**: Update targets to TEMP_RANGE, RR, HU
- [ ] **Task 1.2**: Remove target overlap from features
- [ ] **Task 1.3**: Fix decoder to use autoregressive approach
- [ ] **Task 1.4**: Add attention mechanism
- [ ] **Task 1.5**: Test baseline with new targets

### Phase 2: Training Improvements (Week 1)
- [ ] **Task 2.1**: Implement weighted loss function
- [ ] **Task 2.2**: Adjust hyperparameters (epochs, patience, LR)
- [ ] **Task 2.3**: Add learning rate scheduler
- [ ] **Task 2.4**: Add gradient clipping
- [ ] **Task 2.5**: Add comprehensive evaluation metrics

### Phase 3: Feature Engineering (Week 2)
- [ ] **Task 3.1**: Add lagged features (1-day, 7-day)
- [ ] **Task 3.2**: Add rolling statistics (mean, std, sum)
- [ ] **Task 3.3**: Add derivative features (change, volatility)
- [ ] **Task 3.4**: Add month encoding
- [ ] **Task 3.5**: Retrain and evaluate

### Phase 4: Architecture Optimization (Week 3)
- [ ] **Task 4.1**: Increase model capacity (512 hidden, 3 layers)
- [ ] **Task 4.2**: Try LSTM instead of GRU
- [ ] **Task 4.3**: Experiment with positional encoding
- [ ] **Task 4.4**: Compare with multi-task learning

### Phase 5: Advanced Techniques (Optional)
- [ ] **Task 5.1**: Implement Transformer architecture
- [ ] **Task 5.2**: Try CNN-LSTM hybrid
- [ ] **Task 5.3**: Ensemble multiple models
- [ ] **Task 5.4**: Hyperparameter tuning (grid search)

---

## Success Metrics

### Baseline (Current Model)
- Temperature: Predicts linear trend (FAILED)
- Wind: Constant ~30 m/s (FAILED)
- Precipitation: Near-zero probabilities (FAILED)

### Target Metrics (After Phase 1-2)
- **TEMP_RANGE**:
  - MAE < 30 (0.1°C units = 3°C)
  - R² > 0.5
- **Precipitation (RR)**:
  - MAE < 20 (0.1mm units = 2mm)
  - RMSE < 40 (4mm)
- **Humidity (HU)**:
  - MAE < 10 (humidity units)
  - R² > 0.6

### Stretch Goals (After Phase 3-4)
- **TEMP_RANGE**: MAE < 20, R² > 0.7
- **Precipitation**: MAE < 15, RMSE < 30
- **Humidity**: MAE < 8, R² > 0.7

---

## Notes and Observations

### Data Quality Checks Needed
- [ ] Verify units for all variables (especially HU)
- [ ] Check for outliers (e.g., TG min: -188, TN min: -234)
- [ ] Confirm -9999 values properly handled as NaN
- [ ] Validate date range (1957-2025 seems correct)

### Potential Issues to Monitor
- Class imbalance in precipitation (48% rain days - balanced)
- Weather regime changes over 68-year period (climate change)
- Missing data patterns (after ffill, some early years dropped)
- Seasonal bias in test set (check date distribution)

### Experiment Tracking
- Use TensorBoard for all runs
- Save model checkpoints with descriptive names
- Log hyperparameters and metrics to CSV
- Version control all code changes

---

## References and Resources

### Academic Papers
- Bahdanau et al. (2014) - "Neural Machine Translation by Jointly Learning to Align and Translate" (Attention)
- Sutskever et al. (2014) - "Sequence to Sequence Learning with Neural Networks" (Seq2Seq)
- Vaswani et al. (2017) - "Attention Is All You Need" (Transformer)

### Weather Forecasting with ML
- Consider looking at papers on weather prediction using deep learning
- Check if ECA&D has documentation on variable correlations
- Review meteorological domain knowledge for feature engineering

### PyTorch Resources
- [Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Attention Mechanisms](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Time Series Forecasting](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

---

**Last Updated**: 2025-11-13
**Next Review**: After Phase 1 completion
