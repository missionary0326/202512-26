"""
Sentiment Analysis Script

Analyzes the effectiveness of sentiment data for each stock ticker.
Calculates correlation between sentiment and stock returns to understand
why sentiment improves predictions for some stocks but not others.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

# Configuration
TICKERS = ['AAPL', 'META', 'MSFT', 'AMZN', 'GOOGL']
DATA_DIR = 'data'
OUTPUT_DIR = 'output'


def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load stock data from CSV file."""
    filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def load_sentiment_data(ticker: str) -> pd.DataFrame:
    """Load sentiment data from CSV file."""
    filepath = os.path.join(DATA_DIR, f'{ticker}_GNews_2023_2024_with_sentiment.csv')
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath, parse_dates=['Date'])
    return df


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from close prices."""
    df = df.copy()
    df['DailyReturn'] = df['Close'].pct_change() * 100  # Percentage return
    return df


def aggregate_daily_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple news articles per day into daily statistics."""
    if sentiment_df.empty:
        return pd.DataFrame()
    
    daily = sentiment_df.groupby('Date').agg({
        'sentiment_score': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    
    # Flatten column names
    daily.columns = ['Date', 'SentimentMean', 'SentimentStd', 'SentimentMin', 'SentimentMax', 'ArticleCount']
    
    # Fill NaN std with 0 (when only 1 article)
    daily['SentimentStd'] = daily['SentimentStd'].fillna(0)
    
    return daily


def analyze_ticker(ticker: str) -> dict:
    """Analyze sentiment effectiveness for a single ticker."""
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker}")
    print('='*60)
    
    # Load data
    stock_df = load_stock_data(ticker)
    sentiment_df = load_sentiment_data(ticker)
    
    if sentiment_df.empty:
        print(f"  No sentiment data found for {ticker}")
        return None
    
    # Calculate daily returns
    stock_df = calculate_daily_returns(stock_df)
    
    # Aggregate daily sentiment
    daily_sentiment = aggregate_daily_sentiment(sentiment_df)
    
    # Merge stock data with sentiment
    merged = pd.merge(stock_df, daily_sentiment, on='Date', how='inner')
    
    # Remove rows with NaN returns (first day)
    merged = merged.dropna(subset=['DailyReturn'])
    
    # Basic Statistics
    total_articles = len(sentiment_df)
    total_days_with_news = len(daily_sentiment)
    avg_articles_per_day = total_articles / total_days_with_news if total_days_with_news > 0 else 0
    
    print(f"\n  News Coverage:")
    print(f"    Total articles: {total_articles}")
    print(f"    Days with news: {total_days_with_news}")
    print(f"    Avg articles/day: {avg_articles_per_day:.2f}")
    
    # Sentiment Statistics
    print(f"\n  Sentiment Statistics:")
    print(f"    Mean: {daily_sentiment['SentimentMean'].mean():.4f}")
    print(f"    Std:  {daily_sentiment['SentimentMean'].std():.4f}")
    print(f"    Min:  {daily_sentiment['SentimentMean'].min():.4f}")
    print(f"    Max:  {daily_sentiment['SentimentMean'].max():.4f}")
    
    # Correlation Analysis
    # Same-day correlation
    same_day_corr, same_day_pval = stats.pearsonr(merged['SentimentMean'], merged['DailyReturn'])
    
    # Next-day correlation (does today's sentiment predict tomorrow's return?)
    merged['NextDayReturn'] = merged['DailyReturn'].shift(-1)
    next_day_df = merged.dropna(subset=['NextDayReturn'])
    if len(next_day_df) > 0:
        next_day_corr, next_day_pval = stats.pearsonr(next_day_df['SentimentMean'], next_day_df['NextDayReturn'])
    else:
        next_day_corr, next_day_pval = 0, 1
    
    print(f"\n  Sentiment-Return Correlation:")
    print(f"    Same-day:  r={same_day_corr:.4f} (p={same_day_pval:.4f})")
    print(f"    Next-day:  r={next_day_corr:.4f} (p={next_day_pval:.4f})")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if abs(same_day_corr) > 0.1 and same_day_pval < 0.05:
        print(f"    ✓ Significant same-day correlation - sentiment reflects market movement")
    else:
        print(f"    ✗ Weak same-day correlation")
    
    if abs(next_day_corr) > 0.05 and next_day_pval < 0.1:
        print(f"    ✓ Predictive power - sentiment helps predict next-day returns")
    else:
        print(f"    ✗ Limited predictive power for next-day returns")
    
    # Sentiment extreme analysis
    high_sentiment = merged[merged['SentimentMean'] > 0.3]
    low_sentiment = merged[merged['SentimentMean'] < -0.3]
    neutral_sentiment = merged[(merged['SentimentMean'] >= -0.3) & (merged['SentimentMean'] <= 0.3)]
    
    print(f"\n  Return by Sentiment Category:")
    if len(high_sentiment) > 0:
        print(f"    Positive sentiment (>0.3):  Avg return = {high_sentiment['DailyReturn'].mean():.3f}% ({len(high_sentiment)} days)")
    if len(neutral_sentiment) > 0:
        print(f"    Neutral sentiment:          Avg return = {neutral_sentiment['DailyReturn'].mean():.3f}% ({len(neutral_sentiment)} days)")
    if len(low_sentiment) > 0:
        print(f"    Negative sentiment (<-0.3): Avg return = {low_sentiment['DailyReturn'].mean():.3f}% ({len(low_sentiment)} days)")
    
    # Return results
    return {
        'Ticker': ticker,
        'TotalArticles': total_articles,
        'DaysWithNews': total_days_with_news,
        'AvgArticlesPerDay': avg_articles_per_day,
        'SentimentMean': daily_sentiment['SentimentMean'].mean(),
        'SentimentStd': daily_sentiment['SentimentMean'].std(),
        'SameDayCorrelation': same_day_corr,
        'SameDayPValue': same_day_pval,
        'NextDayCorrelation': next_day_corr,
        'NextDayPValue': next_day_pval,
        'PositiveSentimentReturn': high_sentiment['DailyReturn'].mean() if len(high_sentiment) > 0 else np.nan,
        'NeutralSentimentReturn': neutral_sentiment['DailyReturn'].mean() if len(neutral_sentiment) > 0 else np.nan,
        'NegativeSentimentReturn': low_sentiment['DailyReturn'].mean() if len(low_sentiment) > 0 else np.nan
    }


def generate_recommendations(results_df: pd.DataFrame) -> None:
    """Generate recommendations based on analysis."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Sort by next-day correlation (predictive power)
    sorted_df = results_df.sort_values('NextDayCorrelation', ascending=False)
    
    print("\n  Stocks Ranked by Sentiment Predictive Power:")
    print("  " + "-"*50)
    for _, row in sorted_df.iterrows():
        ticker = row['Ticker']
        corr = row['NextDayCorrelation']
        pval = row['NextDayPValue']
        
        if abs(corr) > 0.05 and pval < 0.1:
            status = "✓ Good candidate for sentiment"
        else:
            status = "✗ Sentiment may add noise"
        
        print(f"    {ticker}: r={corr:+.4f} (p={pval:.3f}) - {status}")
    
    print("\n  Why Sentiment Works Better for Some Stocks:")
    print("  " + "-"*50)
    
    # Find best and worst
    best = sorted_df.iloc[0]
    worst = sorted_df.iloc[-1]
    
    print(f"\n    Best: {best['Ticker']}")
    print(f"      - Next-day correlation: {best['NextDayCorrelation']:.4f}")
    print(f"      - News coverage: {best['AvgArticlesPerDay']:.1f} articles/day")
    
    print(f"\n    Worst: {worst['Ticker']}")
    print(f"      - Next-day correlation: {worst['NextDayCorrelation']:.4f}")
    print(f"      - News coverage: {worst['AvgArticlesPerDay']:.1f} articles/day")
    
    # Coverage analysis
    high_coverage = results_df[results_df['AvgArticlesPerDay'] > results_df['AvgArticlesPerDay'].median()]
    low_coverage = results_df[results_df['AvgArticlesPerDay'] <= results_df['AvgArticlesPerDay'].median()]
    
    print(f"\n  Coverage vs Correlation:")
    print(f"    High coverage stocks avg correlation: {high_coverage['NextDayCorrelation'].mean():.4f}")
    print(f"    Low coverage stocks avg correlation:  {low_coverage['NextDayCorrelation'].mean():.4f}")


def save_results(results: list) -> None:
    """Save analysis results to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    filepath = os.path.join(OUTPUT_DIR, 'sentiment_analysis.csv')
    results_df.to_csv(filepath, index=False)
    print(f"\nSaved detailed results to {filepath}")
    
    return results_df


def main():
    """Main function to analyze sentiment effectiveness for all tickers."""
    print("\n" + "="*60)
    print("SENTIMENT-RETURN CORRELATION ANALYSIS")
    print("="*60)
    print("\nAnalyzing how well sentiment predicts stock returns...")
    
    results = []
    
    for ticker in TICKERS:
        try:
            result = analyze_ticker(ticker)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
    
    if results:
        results_df = save_results(results)
        generate_recommendations(results_df)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

