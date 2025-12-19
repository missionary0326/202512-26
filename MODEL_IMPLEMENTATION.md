# Model Implementation Details

## Overview

This project implements two LSTM-based models for stock price prediction:

| Model | Input Features | Purpose |
|-------|----------------|---------|
| **Base Model** | OHLV (Price + Volume) | Univariate technical analysis |
| **Advanced Model** | OHLV + Sentiment | Multivariate analysis with news sentiment |

---

## 1. Base Model (Univariate)

### 1.1 Input Features
```
Features: [Open, High, Low, Volume]
Target: Close Price
```

### 1.2 Architecture
```
Input Shape: (20, 4) → 20 days of history, 4 features

┌─────────────────────────────────┐
│   LSTM Layer 1 (64 units)       │  ← return_sequences=True
│   Dropout (0.2)                 │
├─────────────────────────────────┤
│   LSTM Layer 2 (32 units)       │  ← return_sequences=False
│   Dropout (0.2)                 │
├─────────────────────────────────┤
│   Dense Layer (16 units, ReLU)  │
├─────────────────────────────────┤
│   Output Layer (1 unit)         │  → Predicted Close Price
└─────────────────────────────────┘
```

### 1.3 Data Pipeline

```
1. Load Data
   └── data/{TICKER}.csv → Date, Open, High, Low, Close, Volume

2. Temporal Split
   └── Training: 2023 data (~250 trading days)
   └── Testing:  2024 data (~250 trading days)

3. Preprocessing
   └── MinMaxScaler normalization (0-1 range)
   └── Fit on training data, transform both

4. Sequence Creation
   └── Sliding window of 20 days
   └── X: days [t-20, t-1], y: day [t]

5. Training
   └── Optimizer: Adam
   └── Loss: MSE
   └── Early Stopping: patience=10
```

### 1.4 Key Code Snippets

**Sequence Creation:**
```python
def create_sequences(features, target, seq_length=20):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])  # 20 days of history
        y.append(target[i + seq_length])       # Next day's close
    return np.array(X), np.array(y)
```

**Model Definition:**
```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(20, 4)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

---

## 2. Advanced Model (Multivariate with Sentiment)

### 2.1 Input Features
```
Features: [Open, High, Low, Volume, SentimentScore]
Target: Close Price
```

### 2.2 Architecture
```
Input Shape: (20, 5) → 20 days of history, 5 features (includes sentiment)

┌─────────────────────────────────┐
│   LSTM Layer 1 (64 units)       │  ← return_sequences=True
│   Dropout (0.2)                 │
├─────────────────────────────────┤
│   LSTM Layer 2 (32 units)       │  ← return_sequences=False
│   Dropout (0.2)                 │
├─────────────────────────────────┤
│   Dense Layer (16 units, ReLU)  │
├─────────────────────────────────┤
│   Output Layer (1 unit)         │  → Predicted Close Price
└─────────────────────────────────┘
```

### 2.3 Sentiment Data Processing

```
1. Load News Data
   └── data/{TICKER}_GNews_2023_2024_with_sentiment.csv
   └── Columns: Date, Text, sentiment_score

2. Daily Aggregation (Multiple articles per day)
   └── Method: Mean of all sentiment scores
   └── Example: Day has 5 articles → Average their scores

3. Merge with Stock Data
   └── Left join on Date
   └── Missing days filled with 0 (neutral sentiment)
```

**Sentiment Aggregation:**
```python
# Multiple news articles per day → Single daily score
daily_sentiment = df.groupby('Date').agg({
    'sentiment_score': 'mean'  # Average sentiment for the day
}).reset_index()
```

### 2.4 Data Pipeline

```
1. Load Stock Data
   └── data/{TICKER}.csv

2. Load Sentiment Data
   └── data/{TICKER}_GNews_2023_2024_with_sentiment.csv
   └── Aggregate to daily scores (mean)

3. Merge Data
   └── Join on Date
   └── Fill missing sentiment with 0

4. Temporal Split
   └── Training: 2023, Testing: 2024

5. Preprocessing
   └── MinMaxScaler on all 5 features
   └── Sentiment scaled from [-1, 1] to [0, 1]

6. Sequence Creation & Training
   └── Same as Base Model
```

---

## 3. Key Differences Summary

| Aspect | Base Model | Advanced Model |
|--------|------------|----------------|
| **Input Features** | 4 (OHLV) | 5 (OHLV + Sentiment) |
| **Input Shape** | (20, 4) | (20, 5) |
| **Data Sources** | Stock prices only | Stock prices + News sentiment |
| **Hypothesis** | Price patterns predict future | Price + Sentiment improve prediction |

---

## 4. Training Configuration

```python
# Shared Configuration
SEQUENCE_LENGTH = 20    # 20-day lookback window
EPOCHS = 100            # Maximum epochs
BATCH_SIZE = 32         # Mini-batch size
RANDOM_SEED = 42        # For reproducibility

# Early Stopping
EarlyStopping(
    monitor='loss',
    patience=10,          # Stop if no improvement for 10 epochs
    restore_best_weights=True
)
```

---

## 5. Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | Mean(\|Actual - Predicted\|) | Average dollar error |
| **RMSE** | √Mean((Actual - Predicted)²) | Penalizes large errors |
| **R²** | 1 - (SS_res / SS_tot) | Variance explained (0-1) |
| **MAPE** | Mean(\|Error\| / \|Actual\|) × 100 | **Best for comparison** (scale-independent) |

---

## 6. Output Files

### Base Model
```
output/
├── {TICKER}_base_predictions.csv   # Predictions for all dates
├── {TICKER}_base_metrics.csv       # MAE, RMSE, R², MAPE
└── base_model_summary.csv          # Summary across all tickers
```

### Advanced Model
```
output/
├── {TICKER}_advanced_predictions.csv  # Predictions + SentimentScore
├── {TICKER}_advanced_metrics.csv      # MAE, RMSE, R², MAPE
├── {TICKER}_finbert.csv               # Daily sentiment for UI
└── advanced_model_summary.csv         # Summary across all tickers
```

---

## 7. Prediction Flow Diagram

```
                    BASE MODEL
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Stock Data  │ →  │   LSTM (4)   │ →  │  Prediction  │
│    OHLV      │    │   Network    │    │   Close $    │
└──────────────┘    └──────────────┘    └──────────────┘

                   ADVANCED MODEL
┌──────────────┐
│  Stock Data  │ ┐
│    OHLV      │ │  ┌──────────────┐    ┌──────────────┐
└──────────────┘ ├→ │   LSTM (5)   │ →  │  Prediction  │
┌──────────────┐ │  │   Network    │    │   Close $    │
│  News Data   │ │  └──────────────┘    └──────────────┘
│  Sentiment   │ ┘
└──────────────┘
```

---

## 8. Usage

```bash
# Train Base Model (all tickers)
python scripts/base_model.py

# Train Advanced Model (all tickers)
python scripts/advanced_model.py
```

---

## 9. Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **LSTM** - Long Short-Term Memory networks for sequence modeling
- **scikit-learn** - MinMaxScaler, evaluation metrics
- **pandas** - Data manipulation
- **NumPy** - Numerical operations

