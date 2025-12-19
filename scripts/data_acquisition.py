"""
Data Acquisition Script for Market Correlation Analyzer
Downloads stock data using yfinance and saves to CSV files in the data folder.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Configuration
TICKERS = ["META", "AAPL", "MSFT", "AMZN", "GOOGL"]
START = "2023-01-01"
END = "2024-12-31"

# Output directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def ensure_data_directory():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")

def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download stock data for a given ticker and date range.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with stock data
    """
    print(f"Downloading data for {ticker} from {start} to {end}...")
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        
        if df.empty:
            print(f"Warning: No data retrieved for {ticker}")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Rename columns to match expected format (Date, Open, High, Low, Close, Volume)
        df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }, inplace=True)
        
        # Format date column to YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Select only the columns we need
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"  ✓ Downloaded {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {str(e)}")
        return pd.DataFrame()

def save_to_csv(df: pd.DataFrame, ticker: str):
    """
    Save DataFrame to CSV file in the data directory.
    
    Args:
        df: DataFrame with stock data
        ticker: Stock ticker symbol
    """
    if df.empty:
        print(f"  ✗ Skipping CSV save for {ticker} (empty data)")
        return
    
    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(filepath, index=False)
    print(f"  ✓ Saved {ticker}.csv ({len(df)} rows)")

def main():
    """Main function to download and save stock data for all tickers."""
    print("=" * 60)
    print("Stock Data Acquisition")
    print("=" * 60)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Date Range: {START} to {END}")
    print()
    
    # Ensure data directory exists
    ensure_data_directory()
    print()
    
    # Download data for each ticker
    success_count = 0
    for ticker in TICKERS:
        df = download_stock_data(ticker, START, END)
        if not df.empty:
            save_to_csv(df, ticker)
            success_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print(f"Summary: {success_count}/{len(TICKERS)} tickers downloaded successfully")
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()

