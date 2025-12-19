export interface StockDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  headline: string;
  sentiment: number; // -1 to 1
  basePrediction: number;
  advancedPrediction: number;
}

export const Ticker = {
  AAPL: 'AAPL',
  MSFT: 'MSFT',
  AMZN: 'AMZN',
  META: 'META',
  GOOGL: 'GOOGL',
} as const;

export type Ticker = typeof Ticker[keyof typeof Ticker];

export interface AnalysisResult {
  markdown: string;
  timestamp: string;
}
