# E6893 Big Data Analytics - Final Project

## Market Correlation Analyzer

A comprehensive dashboard for comparing univariate Technical Analysis vs. multivariate Sentiment Analysis for stock price prediction. This project demonstrates the effectiveness of incorporating sentiment data alongside traditional price/volume features in LSTM-based stock prediction models.

## Features

- **Real-time Stock Data**: Fetches historical stock data using yfinance
- **Base Model (Univariate)**: LSTM model using only OHLCV (Open, High, Low, Close, Volume) features
- **Advanced Model (Multivariate)**: Incorporates sentiment analysis alongside price data
- **Interactive Dashboard**: Real-time visualization with date range filtering
- **Model Comparison**: Side-by-side comparison of prediction accuracy

## Prerequisites

- **Node.js** (for frontend)
- **Python 3.10+** (for data acquisition and model training)
- **uv** (Python package manager) - recommended, or use pip

## Project Structure

```
finsent_-market-correlation-analyzer/
├── data/                    # Stock data CSV files (generated)
├── output/                  # Model predictions and metrics (generated)
├── scripts/
│   ├── data_acquisition.py  # Downloads stock data using yfinance
│   └── base_model.py        # Trains LSTM base model
├── components/              # React components
├── services/                # Data services
└── App.tsx                  # Main application component
```

## Setup

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Install Python Dependencies

Using **uv** (recommended):
```bash
uv sync
```

Or using **pip**:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env.local` file and set:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Data Acquisition

### Download Stock Data

The first step is to download historical stock data for the tickers:

```bash
python scripts/data_acquisition.py
```

This script:
- Downloads data for: **META, AAPL, MSFT, AMZN, GOOGL**
- Date range: **2023-01-01 to 2024-12-31**
- Saves CSV files to the `data/` folder
- Format: `Date,Open,High,Low,Close,Volume`

**Output**: CSV files in `data/` folder (e.g., `META.csv`, `AAPL.csv`, etc.)

## Model Training

### Base Model Architecture

The base model uses a univariate LSTM architecture that processes only price/volume data:

```
Input Layer (t-60): OHLCV vectors (60 days of history)
    ↓
LSTM Layer 1: 50 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 2: 50 units
    ↓
Dropout: 0.2
    ↓
Dense Layer: 25 units (ReLU activation)
    ↓
Output: Predicted Close Price (t+1)
```

### Training the Base Model

Train the LSTM model for all tickers:

```bash
python scripts/base_model.py
```

Or using **uv**:
```bash
uv run python scripts/base_model.py
```

### Training Process

1. **Data Loading**: Loads stock data from `data/` folder
2. **Feature Preparation**: Extracts OHLCV features and normalizes using MinMaxScaler
3. **Sequence Creation**: Creates sequences of 60 days to predict the next day's close price
4. **Temporal Train/Test Split**: 
   - **Training**: Data from 2023-01-01 to 2023-12-31 (past data)
   - **Testing**: Data from 2024-01-01 to 2024-12-31 (future data)
   - **Why temporal split?** In time-series, we must train on past data and test on future data to avoid data leakage. Testing on the same period as training would give overly optimistic (and unrealistic) results.
5. **Model Training**: 
   - Batch size: 32
   - Max epochs: 100
   - Early stopping: Stops if validation loss doesn't improve for 15 epochs
   - Learning rate reduction: Reduces LR by 50% if validation plateaus
   - Validation split: 20% of training data used for validation during training
6. **Prediction Generation**: 
   - Generates predictions for test set (2024 data) - these are the **realistic** performance metrics
   - Also generates predictions for training set (2023 data) for visualization, but these are marked and should not be used for evaluation

### Training Output

For each ticker, the script generates:

1. **`{TICKER}_base_predictions.csv`**
   - Contains: Date, Open, High, Low, Close, Volume, BasePrediction, PredictionError, AbsoluteError
   - Includes predictions for all days (first 60 days have NaN as they need history)

2. **`{TICKER}_base_metrics.csv`**
   - Training metrics: Train_MAE, Train_RMSE, Train_R²
   - Test metrics: Test_MAE, Test_RMSE, Test_R²

3. **`base_model_summary.csv`**
   - Summary of test performance across all tickers

### Model Metrics

The model is evaluated using:
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices (in dollars)
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily (in dollars)
- **R² (Coefficient of Determination)**: Measures how well the model explains variance (0-1 scale)
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error between predicted and actual prices

#### Recommended Metric: MAPE

**MAPE is the best metric for evaluating stock prediction models** because:

1. **Scale-independent**: A $12 MAE means different things for a $50 stock vs a $500 stock. MAPE normalizes by price.
2. **Interpretable**: "7% average error" is intuitive and easy to communicate to stakeholders.
3. **Cross-stock comparison**: Allows fair comparison across stocks with different price levels (e.g., AAPL ~$190 vs META ~$500).
4. **Business relevance**: Investors think in percentages (returns, risk), making MAPE more actionable.

| Metric | Best For | Limitation |
|--------|----------|------------|
| MAE | Understanding average dollar error | Not comparable across different-priced stocks |
| RMSE | Penalizing large prediction errors | Same scale issue as MAE |
| R² | Measuring explained variance | Can be misleading for non-stationary data |
| **MAPE** | **Overall model comparison** | Undefined when actual price = 0 (rare for stocks) |

**Important**: The **Test Metrics** are the only reliable indicators of model performance, as they measure predictions on unseen future data. Training metrics are optimistic because the model has already seen that data.

Example output:
```
Model Performance for AAPL
============================================================
Training Metrics (2023 data - model has seen this):
  MAE:  $3.24
  RMSE: $4.00
  R²:   0.9256
  MAPE: 2.69%

Test Metrics (2024 data - UNSEEN FUTURE DATA):
  MAE:  $12.00
  RMSE: $13.99
  R²:   0.1649
  MAPE: 7.09%  ← Best metric for comparison (scale-independent)
```

**Why this matters**: 
- Training on 2023 and testing on 2024 simulates real-world usage (predicting future prices)
- If you trained and tested on the same period, metrics would be artificially high
- The test metrics (2024) show how well the model generalizes to new, unseen market conditions

## Prediction Workflow

### How Predictions Work

1. **Data Flow**:
   ```
   Stock Data (CSV) → Feature Extraction → Normalization → 
   Sequence Creation → LSTM Model → Prediction → Denormalization → Output
   ```

2. **Sequence Processing**:
   - Uses 60 days of historical data (t-60 to t-1)
   - Predicts the next day's close price (t+1)
   - First 60 days cannot have predictions (insufficient history)

3. **Temporal Validation**:
   - **Training Period (2023)**: Model is trained on this data
   - **Test Period (2024)**: Model predicts on unseen future data
   - Only test period predictions reflect real-world performance
   - Training period predictions are included for visualization but should not be used for evaluation

4. **Prediction Storage**:
   - Predictions are saved to `output/{TICKER}_base_predictions.csv`
   - Each row contains: Date, OHLCV, BasePrediction, PredictionError, AbsoluteError, IsTrainingData
   - `IsTrainingData` column marks which predictions are from training vs test period
   - UI automatically loads these predictions when available

### Using Predictions in UI

The dashboard automatically:
1. Loads stock data from `data/` folder
2. Loads model predictions from `output/` folder
3. Displays:
   - **Actual Price (GT)**: Real close prices (white line)
   - **Base Model Prediction**: LSTM predictions (red dashed line)
   - **Advanced Model Prediction**: Base prediction + sentiment adjustment (green line)

## Running the Application

### Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

## Usage

1. **Select a Ticker**: Click on META, AAPL, MSFT, AMZN, or GOOGL
2. **View Metrics**: 
   - Current Price with change percentage
   - Daily Sentiment score
   - Base Model MAE (Mean Absolute Error)
   - Advanced Model MAE
3. **Filter Date Range**: Use the date picker in "Model Comparison" section to filter data
4. **Hover for Details**: Hover over chart points to see:
   - Actual price for that day
   - Model predictions
   - Headlines and sentiment
   - Metrics update dynamically based on hovered day

## Model Comparison

### Base Model (Univariate)
- **Input**: OHLCV vectors only
- **Architecture**: 2-layer LSTM (50 units each) + Dense layer
- **Strengths**: Captures price trends and patterns
- **Limitations**: May lag behind sudden market changes

### Advanced Model (Multivariate)
- **Input**: OHLCV vectors + Sentiment scores
- **Architecture**: Enhanced LSTM with sentiment fusion
- **Strengths**: Incorporates market sentiment for better predictions
- **Improvement**: Typically shows lower MAE than base model

## Technical Details

### Technologies Used

**Frontend**:
- React 19
- TypeScript
- Vite
- Recharts (for charts)
- D3.js (for correlation visualization)
- Tailwind CSS

**Backend/ML**:
- Python 3.10+
- TensorFlow/Keras
- pandas, numpy
- scikit-learn
- yfinance

### Data Sources

- **Stock Data**: Yahoo Finance (via yfinance)
- **Date Range**: 2023-01-01 to 2024-12-31
- **Tickers**: META, AAPL, MSFT, AMZN, GOOGL
