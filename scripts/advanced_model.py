"""
Advanced Model Script for Stock Price Prediction using LSTM with Sentiment

Trains on 2023 data and predicts stock prices for 2024.
Incorporates real news sentiment scores as an additional feature.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Configuration
TICKERS = ['META', 'AAPL', 'MSFT', 'AMZN', 'GOOGL']
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
SEQUENCE_LENGTH = 20  # Lookback window for LSTM
EPOCHS = 200
BATCH_SIZE = 32


def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load stock data from CSV file."""
    filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def load_sentiment_data(ticker: str) -> pd.DataFrame:
    """
    Load real news sentiment data from CSV file.
    
    The sentiment data contains multiple news articles per day.
    This function aggregates them to get a single daily sentiment score.
    """
    filepath = os.path.join(DATA_DIR, f'{ticker}_GNews_2023_2024_with_sentiment.csv')
    
    if not os.path.exists(filepath):
        print(f"Warning: Sentiment file not found for {ticker}, using neutral sentiment.")
        return pd.DataFrame(columns=['Date', 'SentimentScore'])
    
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Aggregate multiple news articles per day
    # Using mean of sentiment scores for each day
    daily_sentiment = df.groupby('Date').agg({
        'sentiment_score': 'mean'  # Average sentiment for the day
    }).reset_index()
    
    # Rename column to match expected format
    daily_sentiment = daily_sentiment.rename(columns={'sentiment_score': 'SentimentScore'})
    
    print(f"  Loaded {len(df)} news articles, aggregated to {len(daily_sentiment)} daily scores")
    print(f"  Sentiment range: [{daily_sentiment['SentimentScore'].min():.3f}, {daily_sentiment['SentimentScore'].max():.3f}]")
    print(f"  Mean sentiment: {daily_sentiment['SentimentScore'].mean():.3f}")
    
    return daily_sentiment


def merge_data(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """
    Merge stock data with sentiment data on Date.
    
    Args:
        stock_df: Stock price data
        sentiment_df: Sentiment scores by date
        lag_days: Number of days to lag sentiment (use past sentiment to predict future price)
    """
    merged_df = pd.merge(stock_df, sentiment_df, on='Date', how='left')
    
    # Fill any missing sentiment scores with 0 (neutral)
    # This handles weekends/holidays where there might be news but no trading
    missing_count = merged_df['SentimentScore'].isna().sum()
    if missing_count > 0:
        print(f"  Filling {missing_count} missing sentiment values with 0 (neutral)")
    merged_df['SentimentScore'] = merged_df['SentimentScore'].fillna(0)
    
    # Lag the sentiment by specified days (use yesterday's sentiment to predict today)
    # This prevents look-ahead bias and reflects reality (news affects next day's price)
    if lag_days > 0:
        print(f"  Lagging sentiment by {lag_days} day(s)")
        merged_df['SentimentScore'] = merged_df['SentimentScore'].shift(lag_days)
        # Fill the first lag_days with 0 (neutral) since we don't have prior sentiment
        merged_df['SentimentScore'] = merged_df['SentimentScore'].fillna(0)
    
    return merged_df


def split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into 2023 (training) and 2024 (testing) sets."""
    train_df = df[df['Date'].dt.year == 2023].copy()
    test_df = df[df['Date'].dt.year == 2024].copy()
    return train_df, test_df


def create_sequences(features: np.ndarray, target: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM input."""
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: tuple) -> Sequential:
    """Build and compile LSTM model with sentiment-enhanced architecture."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    return {
        f'{prefix}_MAE': mae,
        f'{prefix}_RMSE': rmse,
        f'{prefix}_R2': r2,
        f'{prefix}_MAPE': mape
    }


def train_and_predict(ticker: str) -> tuple[pd.DataFrame, dict]:
    """Train LSTM model with sentiment and generate predictions for a single ticker."""
    # Load stock and sentiment data
    stock_df = load_stock_data(ticker)
    sentiment_df = load_sentiment_data(ticker)
    
    # Merge data
    df = merge_data(stock_df, sentiment_df)
    train_df, test_df = split_by_year(df)
    
    print(f"Training samples (2023): {len(train_df)}")
    print(f"Testing samples (2024): {len(test_df)}")
    
    # Prepare features and target (now including SentimentScore)
    price_feature_cols = ['Open', 'High', 'Low', 'Volume']
    sentiment_col = 'SentimentScore'
    target_col = 'Close'
    
    print(f"Price Features: {price_feature_cols}")
    print(f"Sentiment Feature: {sentiment_col} (lagged)")
    
    # Combine data for consistent scaling
    all_price_features = df[price_feature_cols].values
    all_sentiment = df[sentiment_col].values.reshape(-1, 1)
    all_target = df[target_col].values.reshape(-1, 1)
    
    # Use SEPARATE scalers for price features vs sentiment
    # This preserves the sentiment signal which is already in [-1, 1] range
    price_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()  # Scale sentiment separately
    target_scaler = MinMaxScaler()
    
    scaled_price_features = price_scaler.fit_transform(all_price_features)
    scaled_sentiment = sentiment_scaler.fit_transform(all_sentiment)
    scaled_target = target_scaler.fit_transform(all_target)
    
    # Combine scaled features
    scaled_features = np.hstack([scaled_price_features, scaled_sentiment])
    
    # Split scaled data back into train/test
    train_size = len(train_df)
    train_features = scaled_features[:train_size]
    train_target = scaled_target[:train_size]
    test_features = scaled_features[train_size:]
    test_target = scaled_target[train_size:]
    
    # Create sequences for training
    X_train, y_train = create_sequences(train_features, train_target.flatten(), SEQUENCE_LENGTH)
    
    print(f"Training sequences: {len(X_train)}")
    
    # Build and train model (5 features: 4 price + 1 sentiment)
    num_features = len(price_feature_cols) + 1  # +1 for sentiment
    model = build_lstm_model((SEQUENCE_LENGTH, num_features))
    
    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Generate predictions for all data
    # For training data: predict from sequence_length onwards
    # For test data: use last sequence_length from training + test data
    
    all_predictions = np.full(len(df), np.nan)
    
    # Predictions for training period (starting from SEQUENCE_LENGTH)
    train_pred_scaled = model.predict(X_train, verbose=0)
    train_pred = target_scaler.inverse_transform(train_pred_scaled).flatten()
    all_predictions[SEQUENCE_LENGTH:train_size] = train_pred
    
    # Predictions for test period
    # Create sequences using end of training data + test data
    combined_features = scaled_features[train_size - SEQUENCE_LENGTH:]
    combined_target = scaled_target[train_size - SEQUENCE_LENGTH:]
    
    X_test, y_test = create_sequences(combined_features, combined_target.flatten(), SEQUENCE_LENGTH)
    
    if len(X_test) > 0:
        test_pred_scaled = model.predict(X_test, verbose=0)
        test_pred = target_scaler.inverse_transform(test_pred_scaled).flatten()
        all_predictions[train_size:train_size + len(test_pred)] = test_pred
    
    # Calculate metrics
    # Training metrics (where we have predictions)
    train_actual = df[target_col].values[SEQUENCE_LENGTH:train_size]
    train_predicted = all_predictions[SEQUENCE_LENGTH:train_size]
    train_metrics = calculate_metrics(train_actual, train_predicted, 'Train')
    
    # Test metrics
    test_actual = df[target_col].values[train_size:train_size + len(test_pred)]
    test_predicted = all_predictions[train_size:train_size + len(test_pred)]
    test_metrics = calculate_metrics(test_actual, test_predicted, 'Test')
    
    all_metrics = {**train_metrics, **test_metrics}
    
    print(f"\nTraining Metrics:")
    print(f"  MAE: {train_metrics['Train_MAE']:.4f}")
    print(f"  RMSE: {train_metrics['Train_RMSE']:.4f}")
    print(f"  R2: {train_metrics['Train_R2']:.4f}")
    print(f"  MAPE: {train_metrics['Train_MAPE']:.4f}%")
    
    print(f"\nTest Metrics:")
    print(f"  MAE: {test_metrics['Test_MAE']:.4f}")
    print(f"  RMSE: {test_metrics['Test_RMSE']:.4f}")
    print(f"  R2: {test_metrics['Test_R2']:.4f}")
    print(f"  MAPE: {test_metrics['Test_MAPE']:.4f}%")
    
    # Create output DataFrame
    output_df = df.copy()
    output_df['AdvancedPrediction'] = all_predictions
    output_df['IsTrainingData'] = output_df['Date'].dt.year == 2023
    output_df['PredictionError'] = output_df['AdvancedPrediction'] - output_df['Close']
    output_df['AbsoluteError'] = output_df['PredictionError'].abs()
    
    # For rows without predictions (first SEQUENCE_LENGTH rows), set prediction columns to empty
    mask = output_df['AdvancedPrediction'].isna()
    output_df.loc[mask, 'PredictionError'] = np.nan
    output_df.loc[mask, 'AbsoluteError'] = np.nan
    
    return output_df, all_metrics


def save_predictions(df: pd.DataFrame, ticker: str) -> None:
    """Save predictions to CSV file."""
    filepath = os.path.join(OUTPUT_DIR, f'{ticker}_advanced_predictions.csv')
    
    # Format output to match expected format (including SentimentScore)
    output_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SentimentScore',
                    'AdvancedPrediction', 'IsTrainingData', 'PredictionError', 'AbsoluteError']].copy()
    
    output_df.to_csv(filepath, index=False)
    print(f"Saved predictions to {filepath}")


def save_metrics(metrics: dict, ticker: str) -> None:
    """Save metrics to CSV file."""
    filepath = os.path.join(OUTPUT_DIR, f'{ticker}_advanced_metrics.csv')
    
    metrics_df = pd.DataFrame([
        {'Metric': k, 'Value': v} for k, v in metrics.items()
    ])
    
    metrics_df.to_csv(filepath, index=False)
    print(f"Saved metrics to {filepath}")


def save_summary(all_metrics: dict) -> None:
    """Save summary of all tickers to CSV file."""
    filepath = os.path.join(OUTPUT_DIR, 'advanced_model_summary.csv')
    
    summary_data = []
    for ticker, metrics in all_metrics.items():
        row = {'Ticker': ticker}
        row['Train_MAE'] = metrics['Train_MAE']
        row['Test_MAE'] = metrics['Test_MAE']
        row['Test_RMSE'] = metrics['Test_RMSE']
        row['Test_R2'] = metrics['Test_R2']
        row['Test_MAPE'] = metrics['Test_MAPE']
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(filepath, index=False)
    print(f"\nSaved summary to {filepath}")


def save_sentiment_for_ui(ticker: str) -> None:
    """
    Save daily sentiment data to output folder for UI consumption.
    This creates the {TICKER}_finbert.csv file that the UI expects.
    """
    sentiment_df = load_sentiment_data(ticker)
    
    if len(sentiment_df) == 0:
        print(f"  No sentiment data to save for {ticker}")
        return
    
    filepath = os.path.join(OUTPUT_DIR, f'{ticker}_finbert.csv')
    sentiment_df.to_csv(filepath, index=False)
    print(f"  Saved sentiment data to {filepath}")


def main():
    """Main function to train models and generate predictions for all tickers."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("\n" + "="*60)
    print("ADVANCED MODEL: LSTM with Real News Sentiment")
    print("="*60)
    
    all_metrics = {}
    
    for ticker in TICKERS:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}")
            print('='*50)
            
            # Save sentiment data for UI first
            save_sentiment_for_ui(ticker)
            
            # Train and predict
            predictions_df, metrics = train_and_predict(ticker)
            
            # Save outputs
            save_predictions(predictions_df, ticker)
            save_metrics(metrics, ticker)
            
            all_metrics[ticker] = metrics
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            raise
    
    # Save summary
    save_summary(all_metrics)
    
    print("\n" + "="*50)
    print("All tickers processed successfully!")
    print("="*50)


if __name__ == '__main__':
    main()

