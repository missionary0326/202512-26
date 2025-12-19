import { StockDataPoint, Ticker } from '../types';

const HEADLINES_POSITIVE = [
  "Analyst upgrades target price",
  "Quarterly earnings beat expectations",
  "New product launch receives praise",
  "Strategic partnership announced",
  "Market share expands in key regions",
  "CEO announces stock buyback program",
  "Tech breakthrough revealed at conference"
];

const HEADLINES_NEGATIVE = [
  "Regulatory scrutiny increases",
  "Supply chain disruptions reported",
  "Quarterly revenue misses estimates",
  "Analyst downgrades rating",
  "Executive leadership departs",
  "Product recall issued",
  "Growth concerns worry investors"
];

const HEADLINES_NEUTRAL = [
  "Company to hold annual meeting",
  "Industry report released",
  "Minor update to terms of service",
  "Board meeting scheduled",
  "Market awaits federal rate decision",
  "Trading volume remains steady",
  "Expansion plans discussed"
];

// --- Helpers ---

const getRandomHeadline = (sentiment: number) => {
  if (sentiment > 0.4) return HEADLINES_POSITIVE[Math.floor(Math.random() * HEADLINES_POSITIVE.length)];
  if (sentiment < -0.4) return HEADLINES_NEGATIVE[Math.floor(Math.random() * HEADLINES_NEGATIVE.length)];
  return HEADLINES_NEUTRAL[Math.floor(Math.random() * HEADLINES_NEUTRAL.length)];
};

const formatDate = (date: Date): string => {
  return date.toISOString().split('T')[0];
};

// --- Synthetic Logic Overlay ---
// Applies real or synthetic data on top of price data
// Now uses real base model predictions, advanced predictions, sentiment, and headlines if available
const augmentData = (
  baseData: Partial<StockDataPoint>[], 
  basePredictions?: Map<string, number>,
  advancedPredictions?: Map<string, number>,
  sentimentData?: Map<string, number>,
  newsData?: Map<string, { sentiment: number; headline: string }>
): StockDataPoint[] => {
  const result: StockDataPoint[] = [];
  
  // Seed initial sentiment (only used as fallback)
  let currentSentiment = (Math.random() - 0.5) * 0.5;

  baseData.forEach((day, index) => {
    const close = day.close || 100;
    const date = day.date || formatDate(new Date());

    // 1. Sentiment & Headline: Use real news data if available
    let sentiment: number;
    let headline: string;
    
    if (newsData && newsData.has(date)) {
      // Use real news data (sentiment and headline)
      const news = newsData.get(date)!;
      sentiment = news.sentiment;
      headline = news.headline;
    } else if (sentimentData && sentimentData.has(date)) {
      // Fallback: Use sentiment from finbert.csv but generate headline
      sentiment = sentimentData.get(date)!;
      headline = getRandomHeadline(sentiment);
    } else {
      // Fallback: Generate synthetic sentiment and headline
      const prevSentiment = index > 0 ? result[index-1].sentiment : currentSentiment;
      sentiment = prevSentiment * 0.85 + (Math.random() - 0.5) * 0.5;
      sentiment = Math.max(-1, Math.min(1, sentiment));
      headline = getRandomHeadline(sentiment);
    }

    // 2. Base Model Prediction: Use real prediction if available
    let basePred: number;
    if (basePredictions && basePredictions.has(date)) {
      basePred = basePredictions.get(date)!;
    } else {
      // Fallback: Simple Moving Average for days without predictions
      if (index > 3) {
        const past3 = [
          result[index-1]?.close || close,
          result[index-2]?.close || close,
          result[index-3]?.close || close
        ];
        const avg = past3.reduce((a, b) => a + b, 0) / 3;
        basePred = avg;
      } else {
        basePred = close;
      }
    }

    // 3. Advanced Model Prediction: Use real prediction if available
    let advPred: number;
    if (advancedPredictions && advancedPredictions.has(date)) {
      advPred = advancedPredictions.get(date)!;
    } else {
      // Fallback: Generate synthetic advanced prediction based on sentiment
      const sentimentBias = sentiment * (close * 0.02); 
      advPred = basePred + sentimentBias + (Math.random() - 0.5) * (close * 0.01);
    }

    result.push({
      date: date,
      open: day.open || 0,
      high: day.high || 0,
      low: day.low || 0,
      close: close,
      volume: day.volume || 0,
      headline,
      sentiment: parseFloat(sentiment.toFixed(2)),
      basePrediction: parseFloat(basePred.toFixed(2)),
      advancedPrediction: parseFloat(advPred.toFixed(2))
    });
  });

  return result;
};

// --- CSV Parser ---
const parseCsv = (csvText: string): Partial<StockDataPoint>[] => {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return [];

  // Robust header parsing
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  
  const dateIdx = headers.indexOf('date');
  const openIdx = headers.indexOf('open');
  const highIdx = headers.indexOf('high');
  const lowIdx = headers.indexOf('low');
  const closeIdx = headers.indexOf('close');
  const volIdx = headers.indexOf('volume');

  if (closeIdx === -1) return []; // Cannot parse without Close price

  const parsedData: Partial<StockDataPoint>[] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    // Handle CSV quoting slightly better, though standard numbers usually don't have commas
    const parts = line.split(',');

    if (parts.length < headers.length) continue;

    const closeVal = parseFloat(parts[closeIdx]);
    
    // Skip invalid rows
    if (isNaN(closeVal)) continue;

    parsedData.push({
      date: parts[dateIdx], 
      open: openIdx !== -1 ? parseFloat(parts[openIdx]) : closeVal,
      high: highIdx !== -1 ? parseFloat(parts[highIdx]) : closeVal,
      low: lowIdx !== -1 ? parseFloat(parts[lowIdx]) : closeVal,
      close: closeVal,
      volume: volIdx !== -1 ? parseFloat(parts[volIdx]) : 0
    });
  }
  return parsedData;
};

// --- Parse Predictions CSV ---
const parsePredictionsCsv = (csvText: string, predictionColumn: string = 'baseprediction'): Map<string, number> => {
  const predictions = new Map<string, number>();
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return predictions;

  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const dateIdx = headers.indexOf('date');
  const predIdx = headers.indexOf(predictionColumn.toLowerCase());

  if (dateIdx === -1 || predIdx === -1) return predictions;

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    const parts = line.split(',');
    if (parts.length < headers.length) continue;

    const date = parts[dateIdx];
    const predStr = parts[predIdx]?.trim();
    
    if (predStr && predStr !== '' && !isNaN(parseFloat(predStr))) {
      const prediction = parseFloat(predStr);
      if (!isNaN(prediction)) {
        predictions.set(date, prediction);
      }
    }
  }
  return predictions;
};

// --- Parse Sentiment CSV ---
const parseSentimentCsv = (csvText: string): Map<string, number> => {
  const sentiments = new Map<string, number>();
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return sentiments;

  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const dateIdx = headers.indexOf('date');
  const sentimentIdx = headers.indexOf('sentimentscore');

  if (dateIdx === -1 || sentimentIdx === -1) return sentiments;

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    const parts = line.split(',');
    if (parts.length < headers.length) continue;

    const date = parts[dateIdx];
    const sentimentStr = parts[sentimentIdx]?.trim();
    
    if (sentimentStr && sentimentStr !== '' && !isNaN(parseFloat(sentimentStr))) {
      const sentiment = parseFloat(sentimentStr);
      if (!isNaN(sentiment)) {
        sentiments.set(date, sentiment);
      }
    }
  }
  return sentiments;
};

// --- Parse News Headlines CSV (real news data) ---
const parseNewsCsv = (csvText: string): Map<string, { sentiment: number; headline: string }> => {
  const newsData = new Map<string, { sentiment: number; headline: string }>();
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return newsData;

  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const dateIdx = headers.indexOf('date');
  const textIdx = headers.indexOf('text');
  const sentimentIdx = headers.indexOf('sentiment_score');

  if (dateIdx === -1 || textIdx === -1 || sentimentIdx === -1) return newsData;

  // Group headlines by date, pick the one with strongest sentiment
  const dateHeadlines = new Map<string, { sentiment: number; headline: string; absScore: number }[]>();

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    // Handle CSV with commas in text field
    const match = line.match(/^(\d{4}-\d{2}-\d{2}),(.+),(-?[\d.]+)$/);
    if (!match) continue;

    const date = match[1];
    const headline = match[2].replace(/^"|"$/g, '').trim();
    const sentiment = parseFloat(match[3]);

    if (isNaN(sentiment)) continue;

    if (!dateHeadlines.has(date)) {
      dateHeadlines.set(date, []);
    }
    dateHeadlines.get(date)!.push({ sentiment, headline, absScore: Math.abs(sentiment) });
  }

  // For each date, pick the headline with the strongest sentiment (most impactful news)
  dateHeadlines.forEach((headlines, date) => {
    // Sort by absolute sentiment score (strongest first)
    headlines.sort((a, b) => b.absScore - a.absScore);
    const best = headlines[0];
    
    // Calculate average sentiment for the day
    const avgSentiment = headlines.reduce((sum, h) => sum + h.sentiment, 0) / headlines.length;
    
    newsData.set(date, { 
      sentiment: avgSentiment, 
      headline: best.headline 
    });
  });

  return newsData;
};

// --- Helper to get correct path for GitHub Pages ---
// Use Vite's BASE_URL or fallback to hardcoded value
const getBasePath = (): string => {
  // In browser, try to get base URL from current location or use default
  if (typeof window !== 'undefined') {
    // Extract base path from current URL
    const pathname = window.location.pathname;
    // If pathname starts with /202512-26/, use that
    if (pathname.startsWith('/202512-26/')) {
      return '/202512-26';
    }
  }
  // Fallback to hardcoded base path
  return '/202512-26';
};

const getResourcePath = (path: string): string => {
  const basePath = getBasePath();
  
  // If path already starts with base path, return as is
  if (path.startsWith(`${basePath}/`)) {
    return path;
  }
  // If path is absolute (starts with /), prepend base path
  if (path.startsWith('/')) {
    return `${basePath}${path}`;
  }
  // Relative paths work as is
  return path;
};

// --- Helper to fetch CSV safely ---
const fetchCsvSafely = async (url: string): Promise<string | null> => {
  // Try multiple paths: with base path first (for GitHub Pages), then original, then relative
  const basePath = getResourcePath(url);
  const relativePath = url.startsWith('/') ? `.${url}` : url;
  
  const pathsToTry = [
    basePath, // With base path (works in GitHub Pages) - try this first!
    url, // Original path (works in local dev)
    relativePath, // Relative path
  ];

  for (const finalUrl of pathsToTry) {
    try {
      const response = await fetch(finalUrl);
      
      if (!response.ok) {
        // Log for debugging (only in development)
        if (process.env.NODE_ENV === 'development') {
          console.log(`fetchCsvSafely: ${finalUrl} returned ${response.status}`);
        }
        continue; // Try next path
      }

      const contentType = response.headers.get("content-type");
      const isJsonOrHtml = contentType && 
        (contentType.includes("application/json") || contentType.includes("text/html"));

      if (!isJsonOrHtml) {
        const text = await response.text();
        // Check if it's HTML (404 page) or actual CSV
        const trimmedText = text.trim().toLowerCase();
        if (!trimmedText.startsWith("<!doctype") && 
            !trimmedText.startsWith("<html")) {
          console.log(`fetchCsvSafely: Successfully loaded ${finalUrl}`);
          return text;
        } else {
          // Got HTML instead of CSV, try next path
          if (process.env.NODE_ENV === 'development') {
            console.log(`fetchCsvSafely: ${finalUrl} returned HTML instead of CSV`);
          }
          continue;
        }
      } else {
        // Got JSON or HTML content type, try next path
        if (process.env.NODE_ENV === 'development') {
          console.log(`fetchCsvSafely: ${finalUrl} returned ${contentType}`);
        }
        continue;
      }
    } catch (error) {
      // Try next path
      if (process.env.NODE_ENV === 'development') {
        console.log(`fetchCsvSafely: Error fetching ${finalUrl}:`, error);
      }
      continue;
    }
  }
  
  // All paths failed
  console.warn(`fetchCsvSafely: Failed to load ${url} from all attempted paths:`, pathsToTry);
  return null;
};

// --- Parse Summary CSV (for metrics) ---
const parseSummaryCsv = (csvText: string): Map<string, { mae: number; rmse: number; r2: number; mape: number }> => {
  const metrics = new Map<string, { mae: number; rmse: number; r2: number; mape: number }>();
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) {
    console.warn('parseSummaryCsv: CSV has less than 2 lines');
    return metrics;
  }

  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  console.log('parseSummaryCsv headers:', headers);
  
  const tickerIdx = headers.indexOf('ticker');
  const maeIdx = headers.indexOf('test_mae');
  const rmseIdx = headers.indexOf('test_rmse');
  const r2Idx = headers.indexOf('test_r2');
  const mapeIdx = headers.indexOf('test_mape');

  if (tickerIdx === -1 || maeIdx === -1 || rmseIdx === -1 || r2Idx === -1 || mapeIdx === -1) {
    console.error('parseSummaryCsv: Missing required columns', {
      tickerIdx, maeIdx, rmseIdx, r2Idx, mapeIdx
    });
    return metrics;
  }

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    const parts = line.split(',');
    if (parts.length < headers.length) {
      console.warn(`parseSummaryCsv: Line ${i} has ${parts.length} parts, expected ${headers.length}`);
      continue;
    }

    const ticker = parts[tickerIdx]?.trim();
    const mae = parseFloat(parts[maeIdx]?.trim() || '0');
    const rmse = parseFloat(parts[rmseIdx]?.trim() || '0');
    const r2 = parseFloat(parts[r2Idx]?.trim() || '0');
    const mape = parseFloat(parts[mapeIdx]?.trim() || '0');

    if (ticker && !isNaN(mae) && !isNaN(rmse) && !isNaN(r2) && !isNaN(mape)) {
      metrics.set(ticker, { mae, rmse, r2, mape });
      console.log(`parseSummaryCsv: Added metrics for ${ticker}:`, { mae, rmse, r2, mape });
    } else {
      console.warn(`parseSummaryCsv: Skipping invalid row for ticker ${ticker}`, {
        mae, rmse, r2, mape,
        isValid: !isNaN(mae) && !isNaN(rmse) && !isNaN(r2) && !isNaN(mape)
      });
    }
  }
  return metrics;
};

// --- Load Model Metrics ---
export const getModelMetrics = async (): Promise<{
  base: Map<string, { mae: number; rmse: number; r2: number; mape: number }>;
  advanced: Map<string, { mae: number; rmse: number; r2: number; mape: number }>;
}> => {
  let baseMetrics = new Map<string, { mae: number; rmse: number; r2: number; mape: number }>();
  let advancedMetrics = new Map<string, { mae: number; rmse: number; r2: number; mape: number }>();

  // Load base model metrics
  const baseSummaryText = await fetchCsvSafely('/output/base_model_summary.csv');
  if (baseSummaryText) {
    baseMetrics = parseSummaryCsv(baseSummaryText);
    console.log(`Loaded base model metrics for ${baseMetrics.size} tickers:`, Array.from(baseMetrics.keys()));
    // Debug: log first metric
    if (baseMetrics.size > 0) {
      const firstTicker = Array.from(baseMetrics.keys())[0];
      console.log(`Sample base metric for ${firstTicker}:`, baseMetrics.get(firstTicker));
    }
  } else {
    console.error('Failed to load base_model_summary.csv');
  }

  // Load advanced model metrics
  const advSummaryText = await fetchCsvSafely('/output/advanced_model_summary.csv');
  if (advSummaryText) {
    advancedMetrics = parseSummaryCsv(advSummaryText);
    console.log(`Loaded advanced model metrics for ${advancedMetrics.size} tickers:`, Array.from(advancedMetrics.keys()));
    // Debug: log first metric
    if (advancedMetrics.size > 0) {
      const firstTicker = Array.from(advancedMetrics.keys())[0];
      console.log(`Sample advanced metric for ${firstTicker}:`, advancedMetrics.get(firstTicker));
    }
  } else {
    console.error('Failed to load advanced_model_summary.csv');
  }

  return { base: baseMetrics, advanced: advancedMetrics };
};

// --- Main Data Function ---

export const getStockData = async (ticker: Ticker): Promise<StockDataPoint[]> => {
  let basePredictions: Map<string, number> | undefined = undefined;
  let advancedPredictions: Map<string, number> | undefined = undefined;
  let sentimentData: Map<string, number> | undefined = undefined;
  let newsData: Map<string, { sentiment: number; headline: string }> | undefined = undefined;
  
  // 1. Try to load base model predictions from output folder
  const basePredText = await fetchCsvSafely(`/output/${ticker}_base_predictions.csv`);
  if (basePredText) {
    basePredictions = parsePredictionsCsv(basePredText, 'baseprediction');
    console.log(`Loaded ${basePredictions.size} base model predictions for ${ticker}.`);
  } else {
    console.log(`Base model predictions not found for ${ticker}, using fallback predictions.`);
  }

  // 2. Try to load advanced model predictions from output folder
  const advPredText = await fetchCsvSafely(`/output/${ticker}_advanced_predictions.csv`);
  if (advPredText) {
    advancedPredictions = parsePredictionsCsv(advPredText, 'advancedprediction');
    console.log(`Loaded ${advancedPredictions.size} advanced model predictions for ${ticker}.`);
  } else {
    console.log(`Advanced model predictions not found for ${ticker}, using fallback predictions.`);
  }

  // 3. Try to load sentiment data from output folder
  const sentimentText = await fetchCsvSafely(`/output/${ticker}_finbert.csv`);
  if (sentimentText) {
    sentimentData = parseSentimentCsv(sentimentText);
    console.log(`Loaded ${sentimentData.size} sentiment scores for ${ticker}.`);
  } else {
    console.log(`Sentiment data not found for ${ticker}, using synthetic sentiment.`);
  }

  // 4. Try to load real news data with headlines from data folder
  const newsText = await fetchCsvSafely(`/data/${ticker}_GNews_2023_2024_with_sentiment.csv`);
  if (newsText) {
    newsData = parseNewsCsv(newsText);
    console.log(`Loaded ${newsData.size} real news headlines for ${ticker}.`);
  } else {
    console.log(`News headlines not found for ${ticker}, using synthetic headlines.`);
  }

  // 5. Try to fetch real CSV data (created by python script)
  const priceText = await fetchCsvSafely(`/data/${ticker}.csv`);
  if (priceText) {
    const rawData = parseCsv(priceText);
    if (rawData.length > 0) {
      console.log(`Loaded ${rawData.length} rows for ${ticker} from CSV.`);
      return augmentData(rawData, basePredictions, advancedPredictions, sentimentData, newsData);
    }
  }

  // 6. Fallback: Generate Mock Data (Strict Date Range: 2023-01-01 to 2024-12-31)
  console.log("Using generated fallback data for", ticker);
  return generateFallbackData(ticker);
};

// Renamed from generateStockData to generateFallbackData
const generateFallbackData = (ticker: Ticker): StockDataPoint[] => {
  const data: Partial<StockDataPoint>[] = [];
  
  let currentPrice = 150;
  // Set initial price approx based on real Jan 2023 values
  switch(ticker) {
    case Ticker.AMZN: currentPrice = 85; break;
    case Ticker.GOOGL: currentPrice = 89; break;
    case Ticker.MSFT: currentPrice = 240; break;
    case Ticker.META: currentPrice = 125; break;
    case Ticker.AAPL: currentPrice = 125; break;
  }

  const startDate = new Date("2023-01-01");
  const endDate = new Date("2024-12-31");
  
  const currentDate = new Date(startDate);
  
  while (currentDate <= endDate) {
    // Skip weekends
    const day = currentDate.getDay();
    if (day !== 0 && day !== 6) {
      // Random Walk
      const volatility = 0.02;
      const changePercent = (Math.random() - 0.5) * volatility;
      const close = currentPrice * (1 + changePercent);
      const open = currentPrice;
      const high = Math.max(open, close) * 1.01;
      const low = Math.min(open, close) * 0.99;
      const volume = Math.floor(1000000 + Math.random() * 5000000);

      data.push({
        date: formatDate(currentDate),
        open, high, low, close, volume
      });
      currentPrice = close;
    }
    
    currentDate.setDate(currentDate.getDate() + 1);
  }

  return augmentData(data, undefined, undefined, undefined);
};

// Export alias for backward compatibility if needed, though we should prefer getStockData
export const generateStockData = generateFallbackData;