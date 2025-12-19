"""
Synthetic FinBERT Sentiment Generator

Generates realistic synthetic FinBERT sentiment scores for each trading day
across all stock tickers.
"""

import os
import numpy as np
import pandas as pd

# Configuration
TICKERS = ['META', 'AAPL', 'MSFT', 'AMZN', 'GOOGL']
DATA_DIR = 'data'
OUTPUT_DIR = 'output'

# Sentiment generation parameters
SENTIMENT_MEAN = 0.0  # Centered around neutral
SENTIMENT_STD = 0.3   # Standard deviation for realistic spread
AUTOCORRELATION = 0.7  # Day-to-day correlation (0-1)
RANDOM_SEED = 42


def load_dates(ticker: str) -> pd.Series:
    """Load dates from stock data CSV file."""
    filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df['Date']


def generate_sentiment_scores(n_days: int) -> np.ndarray:
    """
    Generate realistic synthetic FinBERT sentiment scores.
    
    Uses an AR(1) process to create autocorrelated sentiment scores
    that resemble real-world sentiment patterns.
    
    Args:
        n_days: Number of days to generate scores for
        
    Returns:
        Array of sentiment scores in range [-1, 1]
    """
    scores = np.zeros(n_days)
    
    # Generate first score
    scores[0] = np.random.normal(SENTIMENT_MEAN, SENTIMENT_STD)
    
    # Generate subsequent scores with autocorrelation (AR(1) process)
    noise_std = SENTIMENT_STD * np.sqrt(1 - AUTOCORRELATION**2)
    
    for i in range(1, n_days):
        # AR(1): x_t = phi * x_{t-1} + epsilon
        noise = np.random.normal(0, noise_std)
        scores[i] = AUTOCORRELATION * scores[i-1] + (1 - AUTOCORRELATION) * SENTIMENT_MEAN + noise
    
    # Clip to valid FinBERT range [-1, 1]
    scores = np.clip(scores, -1, 1)
    
    return scores


def generate_finbert_data(ticker: str) -> pd.DataFrame:
    """Generate FinBERT sentiment data for a single ticker."""
    print(f"Generating sentiment for {ticker}...")
    
    # Load dates from stock data
    dates = load_dates(ticker)
    n_days = len(dates)
    
    # Generate sentiment scores
    sentiment_scores = generate_sentiment_scores(n_days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'SentimentScore': sentiment_scores
    })
    
    # Print summary statistics
    print(f"  Days: {n_days}")
    print(f"  Mean sentiment: {sentiment_scores.mean():.4f}")
    print(f"  Std sentiment: {sentiment_scores.std():.4f}")
    print(f"  Min: {sentiment_scores.min():.4f}, Max: {sentiment_scores.max():.4f}")
    
    return df


def save_finbert_data(df: pd.DataFrame, ticker: str) -> None:
    """Save FinBERT sentiment data to CSV file."""
    filepath = os.path.join(OUTPUT_DIR, f'{ticker}_finbert.csv')
    df.to_csv(filepath, index=False)
    print(f"  Saved to {filepath}")


def main():
    """Main function to generate FinBERT sentiment data for all tickers."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    print("="*50)
    print("Generating Synthetic FinBERT Sentiment Data")
    print("="*50)
    
    for ticker in TICKERS:
        # Generate and save sentiment data
        df = generate_finbert_data(ticker)
        save_finbert_data(df, ticker)
        print()
    
    print("="*50)
    print("All tickers processed successfully!")
    print("="*50)


if __name__ == '__main__':
    main()

