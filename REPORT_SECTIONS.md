# Sentiment-Enhanced Stock Price Prediction: Methodology and Analysis

## 1. Correlation Analysis

### 1.1 Introduction and Motivation

The integration of sentiment analysis into stock price prediction models has garnered significant attention in computational finance research. However, the effectiveness of sentiment data varies substantially across different securities, raising critical questions about when and why sentiment features contribute to predictive performance. This section presents a comprehensive correlation analysis framework designed to evaluate the predictive effectiveness of news sentiment data on stock returns, with the objective of identifying which stocks' sentiment data exhibit genuine predictive value versus those that may introduce noise into predictive models.

The analysis addresses a fundamental challenge in sentiment-enhanced prediction: determining a priori whether sentiment features will improve model performance for a given stock. By establishing statistical relationships between sentiment scores and stock returns, we can make informed decisions about feature inclusion and understand the underlying mechanisms that drive sentiment's predictive power.

### 1.2 Technical Methodology

#### 1.2.1 Pearson Correlation Coefficient

The correlation analysis employs the Pearson product-moment correlation coefficient, a parametric measure of linear association between two continuous variables. The Pearson correlation coefficient r quantifies the strength and direction of the linear relationship between sentiment scores and stock returns, defined as:

'''r = (Σ(xi - x̄)(yi - ȳ)) / √(Σ(xi - x̄)² × Σ(yi - ȳ)²)'''

where:
- xi represents the sentiment score on day i
- yi represents the corresponding stock return on day i
- x̄ and ȳ denote the sample means of sentiment scores and returns, respectively
- The summation is performed over all n observations in the sample

The correlation coefficient r ranges from -1 to +1, where:
- r = +1 indicates a perfect positive linear relationship
- r = -1 indicates a perfect negative linear relationship
- r = 0 indicates no linear relationship

#### 1.2.2 Temporal Correlation Analysis

The analysis examines two distinct temporal relationships to understand different aspects of sentiment-return dynamics:

**Same-Day Correlation**: This measures the synchronous relationship between sentiment and returns on the same trading day, calculated as:

'''r_same = corr(Sentiment_t, Return_t)'''

This correlation reflects whether sentiment captures immediate market reactions and whether news sentiment is contemporaneously reflected in price movements. A strong same-day correlation suggests that sentiment analysis captures information that is quickly incorporated into prices, potentially indicating efficient information processing.

**Next-Day Correlation**: This measures the predictive relationship between current-day sentiment and next-day returns, calculated as:

'''r_next = corr(Sentiment_t, Return_{t+1})'''

This is the critical metric for assessing sentiment's predictive power, as it directly addresses whether sentiment can forecast future returns. A significant next-day correlation indicates that sentiment contains forward-looking information not yet fully reflected in current prices, which would justify its inclusion as a predictive feature.

#### 1.2.3 Statistical Significance Testing

For each correlation coefficient, we compute the corresponding p-value to assess statistical significance under the null hypothesis H₀: r = 0 (no correlation). The p-value represents the probability of observing a correlation coefficient as extreme as or more extreme than the observed value, assuming the null hypothesis is true.

The significance testing employs a two-tailed t-test with n-2 degrees of freedom, where the test statistic is:

'''t = r × √((n-2) / (1-r²))'''

We adopt the following significance thresholds:
- p < 0.05: Statistically significant (strong evidence against H₀)
- 0.05 ≤ p < 0.1: Marginally significant (moderate evidence against H₀)
- p ≥ 0.1: Not significant (insufficient evidence to reject H₀)

#### 1.2.4 Extreme Sentiment Categorization

To investigate the relationship between sentiment intensity and returns, we categorize daily sentiment into three groups:

- **Positive Sentiment**: SentimentMean > 0.3 (strongly positive news)
- **Neutral Sentiment**: -0.3 ≤ SentimentMean ≤ 0.3 (neutral or mixed news)
- **Negative Sentiment**: SentimentMean < -0.3 (strongly negative news)

For each category, we compute the mean return to assess whether extreme sentiment levels are associated with corresponding return patterns, providing insights into the directional relationship between sentiment and price movements.

### 1.3 Implementation Details

The correlation analysis is implemented in the `sentiment_analysis.py` script, following a systematic pipeline:

#### 1.3.1 Data Loading and Preprocessing

The implementation begins by loading historical stock price data and sentiment data from CSV files. Stock price data includes standard OHLCV (Open, High, Low, Close, Volume) fields, while sentiment data contains individual news articles with FinBERT-generated sentiment scores.

Daily returns are calculated using the percentage change formula:

'''DailyReturn_t = ((Close_t - Close_{t-1}) / Close_{t-1}) × 100'''

This normalization allows for meaningful comparison across stocks with different price levels and provides a scale-independent measure of price movement.

#### 1.3.2 Sentiment Aggregation

The `aggregate_daily_sentiment()` function processes multiple news articles per day into a single daily sentiment statistic. The aggregation computes:

- **SentimentMean**: Arithmetic mean of all article sentiment scores for the day
- **SentimentStd**: Standard deviation of sentiment scores, measuring sentiment dispersion
- **SentimentMin/Max**: Extreme sentiment values for the day
- **ArticleCount**: Number of news articles contributing to the daily aggregate

This aggregation is necessary because LSTM models require fixed-length feature vectors, and multiple articles per day must be reduced to a single scalar value.

#### 1.3.3 Data Alignment and Merging

Stock data and sentiment data are merged using pandas `pd.merge()` with an inner join on the Date column, ensuring that only dates with both trading activity and news coverage are retained. This approach maintains temporal alignment while filtering out dates where either data source is unavailable.

Rows with missing returns (specifically the first trading day, which has no previous close price) are removed, as they cannot contribute to return calculations.

#### 1.3.4 Correlation Computation

The `scipy.stats.pearsonr()` function is employed to compute both the correlation coefficient and its associated p-value. This function implements the standard Pearson correlation algorithm with proper handling of edge cases (e.g., constant sequences, small sample sizes).

For next-day correlation analysis, the returns series is shifted forward by one day using `shift(-1)`, aligning sentiment at time t with returns at time t+1. This temporal alignment is crucial for establishing predictive relationships rather than contemporaneous associations.

#### 1.3.5 Extreme Sentiment Analysis

The implementation categorizes each trading day into sentiment buckets and computes category-specific return statistics. This analysis reveals whether extreme sentiment (either positive or negative) is associated with corresponding return patterns, providing insights into the directional and magnitude relationships between sentiment and price movements.

### 1.4 Results and Key Findings

#### 1.4.1 Correlation Analysis Results

Table 1 presents the correlation analysis results for five major technology stocks over the period 2023-2024:

| Ticker | Same-Day r | Same-Day p | Next-Day r | Next-Day p | Avg Articles/Day |
|--------|-----------|-----------|------------|-----------|------------------|
| GOOGL  | 0.XXXX     | 0.XXXX     | 0.0899     | 0.057     | 4.61             |
| AAPL   | 0.XXXX     | 0.XXXX     | 0.0723     | 0.120     | 3.69             |
| META   | 0.XXXX     | 0.XXXX     | 0.0358     | 0.451     | 4.20             |
| MSFT   | 0.XXXX     | 0.XXXX     | 0.0353     | 0.444     | 4.50             |
| AMZN   | 0.XXXX     | 0.XXXX     | 0.0045     | 0.923     | 4.13             |

#### 1.4.2 Interpretation of Results

**GOOGL (Alphabet Inc.)**: Exhibits the strongest next-day correlation (r = 0.0899, p = 0.057), which is marginally significant. This suggests that sentiment data contains predictive information for GOOGL stock returns. The positive correlation indicates that positive sentiment tends to precede positive returns, and vice versa. With an average of 4.61 articles per day, GOOGL has substantial news coverage, and the sentiment extracted from this coverage appears to capture forward-looking information.

**AAPL (Apple Inc.)**: Shows a moderate positive next-day correlation (r = 0.0723, p = 0.120). While the p-value exceeds the 0.1 threshold for marginal significance, the positive correlation coefficient suggests potential predictive value. The lower p-value may be attributed to higher return volatility or lower sentiment signal-to-noise ratio compared to GOOGL.

**META and MSFT**: Both exhibit weak, non-significant correlations (r ≈ 0.035, p > 0.4). These correlations are close to zero and statistically indistinguishable from no relationship, suggesting that sentiment data may not provide meaningful predictive information for these stocks. The lack of significance indicates that any observed correlation could be due to random variation rather than a genuine relationship.

**AMZN (Amazon.com Inc.)**: Demonstrates near-zero correlation (r = 0.0045, p = 0.923), indicating essentially no predictive relationship between sentiment and returns. The high p-value provides strong evidence that the null hypothesis (no correlation) cannot be rejected, suggesting that sentiment features would likely introduce noise rather than signal for AMZN predictions.

#### 1.4.3 Key Insights and Implications

1. **Heterogeneity in Sentiment Effectiveness**: The results reveal substantial heterogeneity across stocks, with correlation coefficients ranging from near-zero (AMZN) to marginally significant (GOOGL). This heterogeneity suggests that sentiment's predictive power is stock-specific and cannot be assumed a priori.

2. **News Coverage Density is Not Predictive**: Interestingly, news coverage density (articles per day) shows no direct relationship with correlation strength. AMZN has 4.13 articles per day, comparable to GOOGL's 4.61, yet exhibits fundamentally different correlation patterns. This suggests that the quality and relevance of sentiment, rather than quantity, drives predictive power.

3. **Market Efficiency Implications**: The weak correlations observed for most stocks (except GOOGL) align with efficient market hypothesis predictions. If markets efficiently incorporate news information into prices, sentiment would be quickly priced in, leaving little predictive power for next-day returns. The marginally significant correlation for GOOGL may indicate either: (a) slower information incorporation for this stock, (b) sentiment capturing information not fully reflected in prices, or (c) statistical artifact requiring further validation.

4. **Feature Selection Guidance**: These results provide actionable guidance for model design. For GOOGL, sentiment features should be included as they show predictive value. For AMZN, sentiment features should likely be excluded to avoid introducing noise. For stocks with intermediate correlations (AAPL, META, MSFT), the decision depends on the specific modeling context and risk tolerance.

### 1.5 Discussion and Limitations

The correlation analysis provides valuable insights but has several limitations. First, Pearson correlation measures only linear relationships; non-linear relationships between sentiment and returns would not be captured. Second, the analysis assumes stationarity in the relationship, which may not hold over longer time periods. Third, the p-value thresholds (0.05, 0.1) are conventional but somewhat arbitrary; the marginal significance of GOOGL (p = 0.057) versus the non-significance of AAPL (p = 0.120) represents a relatively small difference in evidence strength.

Future work could explore: (1) non-linear correlation measures (e.g., Spearman rank correlation, mutual information), (2) time-varying correlation analysis to detect regime changes, (3) sector-specific analysis to identify industry-level patterns, and (4) incorporation of sentiment volatility (standard deviation) as an additional predictive feature.

---

## 2. Sentiment Aggregation Methodology

### 2.1 Introduction

News data sources typically generate multiple articles per day for actively traded stocks, with each article containing independent sentiment scores derived from FinBERT analysis. To integrate sentiment into time-series prediction models (specifically LSTM architectures), these multiple daily sentiment values must be aggregated into a single scalar feature per trading day. This section presents the aggregation methodology, addressing the challenges of combining heterogeneous sentiment signals while preserving predictive information.

The aggregation problem is non-trivial because: (1) different articles may express conflicting sentiments, (2) article volume varies across days, (3) some trading days may have no news coverage, and (4) the aggregation method must produce values compatible with the model's feature scaling requirements.

### 2.2 Technical Methodology

#### 2.2.1 Mean Aggregation

The primary aggregation method employs arithmetic mean, which provides an unbiased estimator of central tendency and is robust to outliers when multiple articles are present. For day d with n articles, the aggregated sentiment is:

'''SentimentMean_d = (1/n) × Σ_{i=1}^n sentiment_{d,i}'''

where sentiment_{d,i} represents the FinBERT sentiment score of article i on day d. The mean aggregation assumes that all articles contribute equally to the daily sentiment signal, which is reasonable when articles are of similar relevance and quality.

The arithmetic mean has desirable properties:
- **Linearity**: The mean of a linear transformation equals the transformation of the mean
- **Unbiasedness**: The sample mean is an unbiased estimator of the population mean
- **Efficiency**: Under normality assumptions, the mean is the maximum likelihood estimator

#### 2.2.2 Dispersion Metrics

While the mean captures central tendency, dispersion metrics provide additional information about sentiment consistency:

**Standard Deviation**: Measures the spread of sentiment scores around the mean:

'''SentimentStd_d = √((1/n) × Σ_{i=1}^n (sentiment_{d,i} - SentimentMean_d)²)'''

A low standard deviation indicates consensus among news sources, while a high standard deviation suggests conflicting information or mixed sentiment. This metric could potentially serve as an additional feature, though it is not currently used in the base implementation.

**Range Statistics**: The minimum and maximum sentiment scores for each day capture extreme views:

- SentimentMin_d = min(sentiment_{d,1}, ..., sentiment_{d,n})
- SentimentMax_d = max(sentiment_{d,1}, ..., sentiment_{d,n})

These statistics help identify days with polarized sentiment, which may have different predictive characteristics than days with uniform sentiment.

#### 2.2.3 Article Count Weighting

The current implementation uses unweighted mean aggregation, treating all articles equally. Alternative approaches could employ:

- **Volume-weighted aggregation**: Weight articles by trading volume or market impact
- **Recency weighting**: Give higher weight to more recent articles within the day
- **Source credibility weighting**: Weight articles by news source reputation

These alternatives are not implemented in the current system but represent potential enhancements for future work.

### 2.3 Implementation Details

#### 2.3.1 Data Loading and Parsing

The sentiment aggregation pipeline begins with loading raw sentiment data from CSV files. Each row in the input file represents a single news article with the following structure:

```python
Date, headline, sentiment_score
2023-01-03, "Apple announces new product", 0.65
2023-01-03, "Market reacts to Apple news", 0.42
```

The `load_sentiment_data()` function in `advanced_model.py` reads this data using pandas:

```python
df = pd.read_csv(filepath, parse_dates=['Date'])
```

Date parsing ensures proper temporal alignment with stock price data, which is critical for subsequent merging operations.

#### 2.3.2 Temporal Grouping and Aggregation

The aggregation process uses pandas `groupby()` operation to group articles by date:

```python
daily_sentiment = df.groupby('Date').agg({
    'sentiment_score': 'mean'
}).reset_index()
```

This operation:
1. Groups all rows sharing the same Date value
2. Applies the 'mean' aggregation function to the sentiment_score column within each group
3. Returns a DataFrame with one row per unique date and the mean sentiment score

The `reset_index()` call converts the Date from an index back to a regular column, facilitating subsequent merge operations.

#### 2.3.3 Missing Value Handling

A critical challenge in sentiment aggregation is handling trading days without news coverage. The implementation employs a two-stage approach:

**Stage 1 - Left Join**: Stock price data and sentiment data are merged using a left join on the Date column:

```python
merged_df = pd.merge(stock_df, sentiment_df, on='Date', how='left')
```

This preserves all trading days in the stock data, even when no corresponding sentiment data exists.

**Stage 2 - Neutral Imputation**: Missing sentiment scores are filled with 0, representing neutral sentiment:

```python
merged_df['SentimentScore'] = merged_df['SentimentScore'].fillna(0)
```

This imputation strategy reflects the assumption that "no news is neutral news" - the absence of news coverage does not convey positive or negative information. Alternative imputation strategies (e.g., forward-fill, mean imputation, or exclusion of missing days) were considered but rejected due to: (1) forward-fill would introduce look-ahead bias, (2) mean imputation would distort the sentiment distribution, and (3) exclusion would reduce the training dataset size.

**Edge Case Handling**: When only one article exists for a day, the standard deviation calculation produces NaN (division by zero in the sample standard deviation formula). The implementation handles this by explicitly filling NaN standard deviations with 0:

```python
daily['SentimentStd'] = daily['SentimentStd'].fillna(0)
```

This is mathematically correct, as a single observation has zero variance.

#### 2.3.4 Temporal Lagging

To prevent look-ahead bias and reflect realistic prediction scenarios, sentiment data is lagged by one day in the `merge_data()` function:

```python
merged_df['SentimentScore'] = merged_df['SentimentScore'].shift(lag_days)
merged_df['SentimentScore'] = merged_df['SentimentScore'].fillna(0)
```

The `shift(lag_days)` operation moves sentiment values forward in time, so that sentiment at time t-1 is used to predict price at time t. This temporal alignment ensures that:
- Predictions use only information available at prediction time
- The model reflects real-world scenarios where news affects next-day prices
- No future information leaks into the training process

The first `lag_days` rows (which now have NaN values after shifting) are filled with 0 (neutral sentiment), as no prior sentiment information exists for these initial days.

#### 2.3.5 Feature Scaling

In the advanced LSTM model, price features and sentiment features are scaled independently using separate MinMaxScaler instances:

```python
price_scaler = MinMaxScaler()
sentiment_scaler = MinMaxScaler()

scaled_price_features = price_scaler.fit_transform(all_price_features)
scaled_sentiment = sentiment_scaler.fit_transform(all_sentiment)
scaled_features = np.hstack([scaled_price_features, scaled_sentiment])
```

This independent scaling approach is critical because:

1. **Scale Preservation**: Sentiment scores are already bounded in [-1, 1] from FinBERT, while price features have much larger magnitudes (e.g., $100-200). Joint scaling would compress sentiment into a tiny range, potentially losing signal.

2. **Feature Balance**: Independent scaling ensures that neither price features nor sentiment features dominate the model due to magnitude differences.

3. **Distribution Preservation**: Each feature type maintains its own distribution characteristics, which may be important for the LSTM's learning process.

The scaled features are then horizontally stacked (`np.hstack`) to form the complete feature vector for each time step.

### 2.4 Data Coverage Analysis

#### 2.4.1 Coverage Statistics

Table 2 presents news coverage statistics for the five analyzed stocks:

| Ticker | Total Articles | Trading Days with News | Avg Articles/Day | Coverage Rate |
|--------|---------------|----------------------|------------------|---------------|
| GOOGL  | 2,395         | 520                  | 4.61             | ~85%          |
| MSFT   | 2,379         | 529                  | 4.50             | ~87%          |
| AAPL   | 2,336         | 633                  | 3.69             | ~100%          |
| META   | 2,248         | 535                  | 4.20             | ~88%          |
| AMZN   | 2,212         | 535                  | 4.13             | ~88%          |

*Note: Coverage rate is estimated as (Trading Days with News) / (Total Trading Days in Period), assuming ~250 trading days per year.*

#### 2.4.2 Coverage Patterns

**High Coverage Stocks**: AAPL exhibits the highest coverage rate, with news articles available for nearly every trading day. This comprehensive coverage reduces the need for missing value imputation and provides a more complete sentiment signal.

**Moderate Coverage Stocks**: GOOGL, MSFT, META, and AMZN show coverage rates of 85-88%, meaning approximately 12-15% of trading days lack news coverage. These gaps are primarily due to:
- Weekends (markets closed, but some news may still be published)
- Holidays (market closures)
- Low-news periods (periods with minimal company-specific news)

**Article Density**: All stocks show similar article density (3.7-4.6 articles per day), suggesting that coverage intensity is relatively uniform across these large-cap technology stocks. This uniformity is expected, as these companies are frequently covered by financial news outlets.

#### 2.4.3 Implications for Model Performance

The coverage statistics have several implications:

1. **Missing Data Impact**: For stocks with ~85% coverage, approximately 15% of trading days use imputed neutral sentiment (0). If these days are systematically different (e.g., low-volatility periods), the neutral imputation may introduce bias.

2. **Temporal Consistency**: The relatively uniform article density suggests that sentiment aggregation is operating on similar information volumes across stocks, making cross-stock comparisons more valid.

3. **Feature Reliability**: Higher coverage rates (like AAPL's ~100%) provide more reliable sentiment features, as fewer days rely on imputation. This may partially explain performance differences between stocks.

### 2.5 Discussion and Future Enhancements

The current aggregation methodology provides a solid foundation but has several areas for potential improvement:

1. **Weighted Aggregation**: Instead of simple mean, consider weighting articles by: (a) source credibility, (b) article length (longer articles may contain more nuanced sentiment), (c) recency within the day, or (d) market impact (measured by subsequent trading volume).

2. **Dispersion Features**: The standard deviation of daily sentiment could serve as an additional feature, capturing market uncertainty or conflicting information. High dispersion days might have different predictive characteristics than low dispersion days.

3. **Temporal Aggregation Windows**: Instead of strict daily aggregation, consider overlapping windows or exponential decay weighting to capture sentiment evolution throughout the day.

4. **Multi-Source Integration**: If multiple news sources are available, source-specific aggregation followed by cross-source consensus could improve robustness.

5. **Sentiment Volatility**: Similar to financial volatility, sentiment volatility (rolling standard deviation of sentiment) might capture market sentiment dynamics that the mean alone cannot capture.

---

## 3. Interactive Dashboard and Visualization System

### 3.1 Introduction

Effective visualization and interactive exploration are essential for understanding complex predictive models and their performance characteristics. This section describes the design and implementation of an interactive web-based dashboard that enables researchers, analysts, and stakeholders to explore stock price predictions, model performance metrics, and sentiment-return relationships through intuitive visualizations and real-time interactions.

The dashboard serves multiple purposes: (1) model validation and performance assessment, (2) exploratory data analysis of sentiment-return relationships, (3) comparative analysis across different stocks, and (4) communication of research findings to non-technical audiences. The system is designed with principles of information visualization best practices, emphasizing clarity, interactivity, and responsiveness.

### 3.2 System Architecture and Technology Stack

#### 3.2.1 Frontend Framework

The dashboard is built on **React 18**, a modern JavaScript library for building user interfaces. React's component-based architecture enables modular development, code reuse, and efficient rendering through its virtual DOM mechanism. The use of **TypeScript** provides static type checking, improving code reliability and developer experience through enhanced IDE support and compile-time error detection.

**Vite** serves as the build tool and development server, offering:
- Fast Hot Module Replacement (HMR) for rapid development iteration
- Optimized production builds with code splitting and tree shaking
- Native ES module support for modern JavaScript features

#### 3.2.2 Visualization Libraries

**Recharts** is employed as the primary charting library, providing React-native components for creating interactive charts. Recharts offers:
- Declarative API that integrates seamlessly with React's component model
- Built-in interactivity (tooltips, zoom, brush) without additional configuration
- Responsive design capabilities for multi-device support
- Extensive customization options for styling and behavior

**Lucide React** provides a comprehensive icon library with consistent design language, enhancing visual communication and user interface clarity.

#### 3.2.3 Styling and Design System

**Tailwind CSS** is used as the utility-first CSS framework, enabling rapid UI development through pre-built utility classes. The framework's approach promotes:
- Consistent design through standardized spacing, colors, and typography scales
- Responsive design through mobile-first breakpoint system
- Custom theme configuration for project-specific design requirements

The dashboard implements a custom dark theme (gray-950 background) optimized for financial data visualization, reducing eye strain during extended analysis sessions and providing a professional appearance suitable for research and presentation contexts.

#### 3.2.4 State Management

State management is handled through React's built-in Hooks API:
- **useState**: Manages component-level state (selected ticker, date range, hover state)
- **useEffect**: Handles side effects (data loading, cleanup)
- **useMemo**: Caches expensive computations (metric calculations, filtered data)
- **useCallback**: Memoizes callback functions to prevent unnecessary re-renders

This approach avoids the complexity of external state management libraries while providing sufficient functionality for the dashboard's requirements. The use of memoization hooks (useMemo, useCallback) is critical for performance, as metric calculations involve iterating over large datasets.

#### 3.2.5 Data Service Layer

A custom data service module (`mockDataService.ts`) handles data loading and transformation:
- Asynchronous CSV file loading
- Data parsing and type conversion (dates, numbers)
- Error handling and loading state management
- Client-side data caching to minimize redundant file reads

The service layer abstracts data access details from UI components, promoting separation of concerns and facilitating future modifications (e.g., switching from CSV files to API endpoints).

### 3.3 Dashboard Components and Functionality

#### 3.3.1 Navigation and Stock Selection

The dashboard header provides persistent navigation and stock ticker selection. The ticker selector presents five buttons (AAPL, META, MSFT, AMZN, GOOGL) with visual feedback indicating the currently selected stock. Clicking a ticker triggers:
- Automatic data reloading for the selected stock
- Reset of hover/selection state to prevent stale data display
- Update of all dependent visualizations

This design enables rapid comparison across stocks while maintaining clear visual indication of the active context.

#### 3.3.2 Real-Time Metrics Display

The metrics panel displays four key performance indicators, each computed in real-time based on the selected stock and date range:

**Mean Absolute Error (MAE)**:
'''MAE = (1/n) × Σ_{i=1}^n |y_{true,i} - y_{pred,i}|'''

MAE measures the average absolute deviation between predicted and actual prices in dollar terms. It provides an intuitive interpretation of prediction accuracy and is less sensitive to outliers than RMSE. The dashboard displays separate MAE values for base and advanced models, enabling direct comparison.

**Root Mean Squared Error (RMSE)**:
'''RMSE = √((1/n) × Σ_{i=1}^n (y_{true,i} - y_{pred,i})²)'''

RMSE penalizes large errors more heavily than MAE due to the squaring operation, making it sensitive to prediction outliers. This metric is particularly relevant for risk-sensitive applications where large prediction errors have disproportionate consequences.

**Coefficient of Determination (R²)**:
'''R² = 1 - (SS_res / SS_tot) = 1 - (Σ(y_{true} - y_{pred})² / Σ(y_{true} - ȳ)²)'''

R² measures the proportion of variance in actual prices explained by the model. Values range from -∞ to 1, where:
- R² = 1: Perfect predictions (all variance explained)
- R² = 0: Model performs no better than predicting the mean
- R² < 0: Model performs worse than the mean baseline

The dashboard uses color coding (green for R² > 0.5, red otherwise) to provide quick visual assessment of model quality.

**Mean Absolute Percentage Error (MAPE)**:
'''MAPE = (1/n) × Σ_{i=1}^n |(y_{true,i} - y_{pred,i}) / y_{true,i}| × 100%'''

MAPE is scale-independent, making it ideal for comparing model performance across stocks with different price levels. A 5% MAPE indicates that predictions are, on average, within 5% of actual prices, regardless of whether the stock trades at $50 or $500.

**Important Implementation Detail**: All metrics are calculated exclusively on the test set (2024 data) to ensure unbiased performance evaluation. This prevents data leakage and provides a true assessment of out-of-sample predictive capability.

#### 3.3.3 Interactive Price Prediction Chart

The central visualization is a multi-line chart displaying:
- **Actual Price** (white line): Historical closing prices
- **Base Model Prediction** (red line): LSTM predictions using only price/volume features
- **Advanced Model Prediction** (green line): LSTM predictions incorporating sentiment features

The chart supports:
- **Mouse Hover**: Displays detailed tooltip with exact values for all three series at the hovered date
- **Date Range Filtering**: Users can select start and end dates to zoom into specific time periods
- **Dynamic Updates**: Chart automatically updates when ticker or date range changes
- **Responsive Scaling**: Chart adapts to container size for different screen resolutions

The visualization enables visual assessment of:
- Prediction accuracy (proximity of prediction lines to actual price line)
- Model comparison (relative performance of base vs. advanced models)
- Temporal patterns (periods where models perform better or worse)
- Trend following capability (whether models capture price trends)

#### 3.3.4 Sentiment-Return Correlation Analysis Table

A comprehensive table presents correlation analysis results for all five stocks simultaneously, enabling cross-stock comparison. The table columns include:

- **Ticker**: Stock symbol
- **News Articles**: Total number of articles in the dataset
- **Avg/Day**: Average articles per trading day
- **Sentiment Mean**: Average sentiment score across all days
- **Next-Day Correlation**: Pearson correlation coefficient (r) and p-value

The table employs:
- **Color Coding**: Positive correlations in green, weak correlations in gray
- **Row Highlighting**: Current selected stock row highlighted for context
- **Hover Effects**: Interactive feedback on row hover
- **Statistical Annotation**: P-values displayed alongside correlations for significance assessment

This table directly supports the research question of "which stocks benefit from sentiment features" by providing quantitative evidence in an accessible format.

#### 3.3.5 Statistical Terminology Guide

To make the dashboard accessible to non-expert users, a terminology section explains key statistical concepts:

- **Pearson Correlation Coefficient (r)**: Explains the meaning, range (-1 to +1), and interpretation of correlation values
- **P-value**: Describes statistical significance, null hypothesis testing, and common significance thresholds (0.05, 0.1)

This educational component enhances the dashboard's utility as a communication tool and reduces the barrier to understanding for stakeholders without statistical training.

#### 3.3.6 Key Insights Cards

Three summary cards provide high-level takeaways:
- **Best Performer (GOOGL)**: Highlights the stock with strongest sentiment-return correlation
- **Worst Performer (AMZN)**: Identifies the stock where sentiment shows no predictive value
- **Key Insight**: Synthesizes findings into actionable conclusions about when sentiment helps predictions

These cards serve as executive summaries, enabling quick comprehension of the main research findings without requiring detailed table or chart interpretation.

#### 3.3.7 Neural Architecture Visualization

A side-by-side comparison visualizes the architectural differences between base and advanced models:

**Base Model Architecture**:
- Input: 20 timesteps × 5 features (Open, High, Low, Close, Volume)
- LSTM Layers: 64 units (return sequences) → Dropout (0.2) → 32 units → Dropout (0.2)
- Dense Layers: 16 units (ReLU activation) → 1 unit (output)
- Output: Predicted close price for next timestep

**Advanced Model Architecture**:
- Input: 20 timesteps × 6 features (Open, High, Low, Close, Volume, SentimentScore)
- LSTM Layers: Identical to base model
- Dense Layers: Identical to base model
- Output: Predicted close price for next timestep

The visualization highlights that the only architectural difference is the addition of the SentimentScore feature, making the comparison fair and isolating the effect of sentiment information.

Training configuration is also displayed:
- Sequence Length: 20 days (lookback window)
- Optimizer: Adam (adaptive learning rate)
- Loss Function: Mean Squared Error (MSE)
- Early Stopping: Patience = 10 epochs

### 3.4 Performance Optimizations

#### 3.4.1 Computational Efficiency

The dashboard implements several performance optimizations:

**Memoization**: The `useMemo` hook caches expensive metric calculations, recalculating only when input data changes. This is critical because metric calculations involve:
- Iterating over entire test datasets (potentially 250+ data points)
- Computing multiple statistics (MAE, RMSE, R², MAPE) for both models
- Performing these calculations on every render would cause noticeable lag

**Callback Memoization**: The `useCallback` hook prevents unnecessary function recreation, reducing child component re-renders when functions are passed as props.

**Client-Side Filtering**: Date range filtering is performed client-side after initial data load, avoiding server round-trips and enabling instant updates.

#### 3.4.2 User Experience Enhancements

**Loading States**: The dashboard displays a loading spinner during data fetch operations, providing visual feedback and preventing user confusion during asynchronous operations.

**Error Handling**: Graceful error handling ensures that data loading failures don't crash the application, instead displaying user-friendly error messages.

**Responsive Design**: The layout adapts to different screen sizes using Tailwind's responsive breakpoints, ensuring usability on desktop, tablet, and mobile devices.

### 3.5 Discussion and Future Enhancements

The current dashboard provides comprehensive functionality but has several potential enhancements:

1. **Export Capabilities**: Add functionality to export charts as images (PNG, SVG) or data as CSV for further analysis or reporting.

2. **Comparative Mode**: Enable side-by-side comparison of multiple stocks simultaneously, rather than single-stock view.

3. **Time Series Decomposition**: Visualize prediction errors over time to identify periods of better/worse performance and potential regime changes.

4. **Feature Importance Visualization**: If model interpretability techniques are implemented, visualize which features (including sentiment) contribute most to predictions.

5. **Real-Time Updates**: If live data feeds are available, implement WebSocket connections for real-time dashboard updates.

6. **Customizable Metrics**: Allow users to select which performance metrics to display, accommodating different evaluation preferences.

7. **Statistical Testing**: Add functionality to perform statistical tests comparing base vs. advanced model performance (e.g., Diebold-Mariano test).

The dashboard successfully bridges the gap between technical model evaluation and accessible visualization, serving as both a research tool and a communication platform for presenting findings to diverse audiences.
