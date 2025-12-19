import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { StockChart } from './components/StockChart';
import { CorrelationChart } from './components/CorrelationChart';
import { getStockData } from './services/mockDataService';
import { StockDataPoint, Ticker } from './types';
import { BarChart3, TrendingUp, TrendingDown, Activity, DollarSign, LayoutDashboard, ArrowDown, Network, RefreshCw } from 'lucide-react';

function App() {
  const [ticker, setTicker] = useState<Ticker>(Ticker.AAPL);
  const [data, setData] = useState<StockDataPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedDay, setSelectedDay] = useState<StockDataPoint | null>(null);
  const [dateRange, setDateRange] = useState<{ start: string; end: string }>({
    start: '2023-01-01',
    end: '2024-12-31'
  });

  useEffect(() => {
    let isMounted = true;
    
    const loadData = async () => {
      setLoading(true);
      try {
        const newData = await getStockData(ticker);
        if (isMounted) {
          setData(newData);
        }
      } catch (error) {
        console.error("Failed to load stock data", error);
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    loadData();

    return () => { isMounted = false; };
  }, [ticker]);

  // Reset selectedDay when ticker changes
  useEffect(() => {
    setSelectedDay(null);
  }, [ticker]);

  // Update date range when data loads
  // Default start date is 60 days after first date (when predictions begin)
  useEffect(() => {
    if (data.length > 0) {
      const SEQUENCE_LENGTH = 0; // Need 60 days of history for predictions
      const predictionStartIndex = Math.min(SEQUENCE_LENGTH, data.length - 1);
      const predictionStartDate = data[predictionStartIndex]?.date || data[0].date;
      
      setDateRange({
        start: predictionStartDate, // Start from when predictions are available
        end: data[data.length - 1].date
      });
    }
  }, [data]);

  // Filter data based on date range
  const filteredData = useMemo(() => {
    if (!data.length) return [];
    return data.filter(d => {
      const date = d.date;
      return date >= dateRange.start && date <= dateRange.end;
    });
  }, [data, dateRange]);

  // Memoize hover handlers
  const handleChartHover = useCallback((day: StockDataPoint) => {
    setSelectedDay(day);
  }, []);

  const handleChartLeave = useCallback(() => {
    setSelectedDay(null);
  }, []);

  // Calculate Metrics - use selectedDay if available, otherwise use latest
  const metrics = useMemo(() => {
    if (data.length === 0) return null;
    
    // Use selectedDay if hovering, otherwise use latest
    const currentDay = selectedDay || data[data.length - 1];
    const currentIndex = selectedDay 
      ? data.findIndex(d => d.date === selectedDay.date)
      : data.length - 1;
    const prevIndex = Math.max(0, currentIndex - 1);
    const prev = data[prevIndex];
    const change = currentDay.close - prev.close;
    const changePct = (change / prev.close) * 100;
    
    // Filter to TEST SET ONLY (2024 data) for model performance metrics
    const testData = data.filter(d => d.date >= '2024-01-01');
    const n = testData.length;
    
    if (n === 0) {
      // Fallback if no test data
      return { 
        currentDay, change, changePct, 
        baseError: 0, advError: 0, 
        baseMAE: 0, baseRMSE: 0, baseR2: 0, baseMAPE: 0,
        advMAE: 0, advRMSE: 0, advR2: 0, advMAPE: 0,
        isHovering: !!selectedDay 
      };
    }
    
    // Base Model Metrics (TEST SET ONLY)
    const baseErrors = testData.map(d => d.close - d.basePrediction);
    const baseAbsErrors = baseErrors.map(e => Math.abs(e));
    const baseSqErrors = baseErrors.map(e => e * e);
    const basePctErrors = testData.map(d => Math.abs((d.close - d.basePrediction) / d.close) * 100);
    
    const baseMAE = baseAbsErrors.reduce((a, b) => a + b, 0) / n;
    const baseRMSE = Math.sqrt(baseSqErrors.reduce((a, b) => a + b, 0) / n);
    const baseMAPE = basePctErrors.reduce((a, b) => a + b, 0) / n;
    
    // R² for Base Model (TEST SET ONLY)
    const meanClose = testData.reduce((a, d) => a + d.close, 0) / n;
    const ssTot = testData.reduce((a, d) => a + Math.pow(d.close - meanClose, 2), 0);
    const ssResBase = baseSqErrors.reduce((a, b) => a + b, 0);
    const baseR2 = 1 - (ssResBase / ssTot);
    
    // Advanced Model Metrics (TEST SET ONLY)
    const advErrors = testData.map(d => d.close - d.advancedPrediction);
    const advAbsErrors = advErrors.map(e => Math.abs(e));
    const advSqErrors = advErrors.map(e => e * e);
    const advPctErrors = testData.map(d => Math.abs((d.close - d.advancedPrediction) / d.close) * 100);
    
    const advMAE = advAbsErrors.reduce((a, b) => a + b, 0) / n;
    const advRMSE = Math.sqrt(advSqErrors.reduce((a, b) => a + b, 0) / n);
    const advMAPE = advPctErrors.reduce((a, b) => a + b, 0) / n;
    
    // R² for Advanced Model (TEST SET ONLY)
    const ssResAdv = advSqErrors.reduce((a, b) => a + b, 0);
    const advR2 = 1 - (ssResAdv / ssTot);
    
    // Daily Error for the current/hovered day
    const baseError = Math.abs(currentDay.close - currentDay.basePrediction);
    const advError = Math.abs(currentDay.close - currentDay.advancedPrediction);
    
    return { 
      currentDay, change, changePct, baseError, advError, 
      baseMAE, baseRMSE, baseR2, baseMAPE,
      advMAE, advRMSE, advR2, advMAPE,
      isHovering: !!selectedDay 
    };
  }, [data, selectedDay]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 pb-12">
      {/* Navbar */}
      <nav className="border-b border-gray-800 bg-gray-900/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <LayoutDashboard className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg tracking-tight">E6893 Big Data Analytics - Final Project</h1>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 pt-8">
        
        {/* Header & Controls */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Financial Sentiment and Market Correlation Analysis</h2>
            <p className="text-xs text-gray-500 mt-1 font-mono">Range: 2023-01-01 to 2024-12-31</p>
          </div>
          
          <div className="flex items-center gap-4 bg-gray-900 p-1.5 rounded-lg border border-gray-800">
            {Object.values(Ticker).map((t) => (
              <button
                key={t}
                onClick={() => setTicker(t)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  ticker === t 
                    ? 'bg-gray-800 text-white shadow-sm border border-gray-700' 
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="flex h-96 items-center justify-center">
            <div className="flex flex-col items-center gap-4 text-gray-500">
              <RefreshCw className="w-8 h-8 animate-spin text-indigo-500" />
              <p>Loading market data...</p>
            </div>
          </div>
        ) : metrics ? (
          <>
            {/* Top Row: Price & Sentiment */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              {/* Price Card */}
              <div className="bg-gray-900/50 border border-gray-800 p-5 rounded-xl">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm font-medium">Current Price</span>
                  <DollarSign className="w-4 h-4 text-gray-500" />
                </div>
                <div className="flex items-end gap-3">
                  <span className="text-3xl font-bold text-white">${metrics.currentDay.close.toFixed(2)}</span>
                  <div className={`flex items-center text-sm font-medium mb-1 ${metrics.change >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {metrics.change >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                    {Math.abs(metrics.changePct).toFixed(2)}%
                  </div>
                </div>
                {selectedDay && (
                  <div className="text-xs text-gray-500 mt-1">{selectedDay.date}</div>
                )}
              </div>

              {/* Sentiment Card */}
              <div className="bg-gray-900/50 border border-gray-800 p-5 rounded-xl">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm font-medium">Daily Sentiment</span>
                  <Activity className="w-4 h-4 text-gray-500" />
                </div>
                <div className="flex flex-col">
                  <span className={`text-3xl font-bold ${metrics.currentDay.sentiment > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {metrics.currentDay.sentiment > 0 ? '+' : ''}{metrics.currentDay.sentiment.toFixed(2)}
                  </span>
                  <span className="text-xs text-gray-500 mt-1 truncate">{metrics.currentDay.headline}</span>
                </div>
              </div>
            </div>

            {/* Model Performance Metrics */}
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5 mb-8">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" /> Model Performance Metrics
                </h3>
                <span className="text-[10px] bg-gray-800 text-gray-400 px-2 py-0.5 rounded border border-gray-700">Overall Test Set</span>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {/* MAE */}
                <div className="bg-gray-950/50 rounded-lg p-4 border border-gray-800">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-400 text-xs font-medium">MAE</span>
                    <span className="text-[9px] text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">Recommended</span>
                  </div>
                  <div className="flex justify-between items-end mb-2">
                    <div>
                      <span className="text-xs text-gray-500 block">Base</span>
                      <span className="text-lg font-bold text-rose-400">${metrics.baseMAE.toFixed(2)}</span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-gray-500 block">Advanced</span>
                      <span className="text-lg font-bold text-emerald-400">${metrics.advMAE.toFixed(2)}</span>
                    </div>
                  </div>
                  <p className="text-[10px] text-gray-500 leading-relaxed">Mean Absolute Error - Average dollar difference between predicted and actual prices</p>
                </div>

                {/* RMSE */}
                <div className="bg-gray-950/50 rounded-lg p-4 border border-gray-800">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-400 text-xs font-medium">RMSE</span>
                  </div>
                  <div className="flex justify-between items-end mb-2">
                    <div>
                      <span className="text-xs text-gray-500 block">Base</span>
                      <span className="text-lg font-bold text-rose-400">${metrics.baseRMSE.toFixed(2)}</span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-gray-500 block">Advanced</span>
                      <span className="text-lg font-bold text-emerald-400">${metrics.advRMSE.toFixed(2)}</span>
                    </div>
                  </div>
                  <p className="text-[10px] text-gray-500 leading-relaxed">Root Mean Squared Error - Penalizes large errors more heavily than MAE</p>
                </div>

                {/* R² */}
                <div className="bg-gray-950/50 rounded-lg p-4 border border-gray-800">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-400 text-xs font-medium">R²</span>
                  </div>
                  <div className="flex justify-between items-end mb-2">
                    <div>
                      <span className="text-xs text-gray-500 block">Base</span>
                      <span className={`text-lg font-bold ${metrics.baseR2 > 0.5 ? 'text-emerald-400' : 'text-rose-400'}`}>{metrics.baseR2.toFixed(3)}</span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-gray-500 block">Advanced</span>
                      <span className={`text-lg font-bold ${metrics.advR2 > 0.5 ? 'text-emerald-400' : 'text-rose-400'}`}>{metrics.advR2.toFixed(3)}</span>
                    </div>
                  </div>
                  <p className="text-[10px] text-gray-500 leading-relaxed">Coefficient of Determination - How much variance is explained (1.0 = perfect)</p>
                </div>

                {/* MAPE */}
                <div className="bg-gray-950/50 rounded-lg p-4 border border-gray-800 relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-transparent pointer-events-none"></div>
                  <div className="flex items-center justify-between mb-2 relative z-10">
                    <span className="text-gray-400 text-xs font-medium">MAPE</span>
                    <span className="text-[9px] text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">Best Metric</span>
                  </div>
                  <div className="flex justify-between items-end mb-2 relative z-10">
                    <div>
                      <span className="text-xs text-gray-500 block">Base</span>
                      <span className="text-lg font-bold text-rose-400">{metrics.baseMAPE.toFixed(2)}%</span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-gray-500 block">Advanced</span>
                      <span className="text-lg font-bold text-emerald-400">{metrics.advMAPE.toFixed(2)}%</span>
                    </div>
                  </div>
                  <p className="text-[10px] text-gray-500 leading-relaxed relative z-10">Mean Absolute % Error - Scale-independent, best for cross-stock comparison</p>
                </div>
              </div>

              {/* Daily Error Row */}
              <div className="mt-4 pt-4 border-t border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-gray-400 text-xs font-medium">Daily Error</span>
                  <span className="text-[10px] text-gray-500">
                    {metrics.isHovering ? `on ${metrics.currentDay.date}` : '(hover chart to see specific day)'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center justify-between bg-gray-950/30 rounded-lg px-4 py-2">
                    <span className="text-xs text-gray-400">Base Model</span>
                    <span className="text-lg font-bold text-rose-400">${metrics.baseError.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between bg-gray-950/30 rounded-lg px-4 py-2">
                    <span className="text-xs text-gray-400">Advanced Model</span>
                    <span className="text-lg font-bold text-emerald-400">${metrics.advError.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Charts Section */}
            <div className="mb-8">
              {/* Date Range Selector */}
              <div className="bg-gray-900/50 border border-gray-800 p-4 rounded-xl mb-4">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div>
                    <h3 className="text-gray-400 text-sm font-semibold mb-2 uppercase tracking-wider">
                      Model Comparison: Price vs Predictions
                    </h3>
                    <p className="text-xs text-gray-500">
                      Select a date range to filter the chart data
                      {filteredData.length > 0 && (
                        <span className="ml-2 text-gray-400">
                          ({filteredData.length} data points)
                        </span>
                      )}
                    </p>
                  </div>
                  <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
                    <div className="flex flex-col gap-1">
                      <label className="text-xs text-gray-400 font-medium">Start Date</label>
                      <input
                        type="date"
                        value={dateRange.start}
                        onChange={(e) => {
                          const newStart = e.target.value;
                          if (newStart <= dateRange.end) {
                            setDateRange(prev => ({ ...prev, start: newStart }));
                          }
                        }}
                        min={data.length > 0 ? data[0].date : '2023-01-01'}
                        max={dateRange.end}
                        className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      />
                    </div>
                    <div className="flex items-center text-gray-500 mt-6 sm:mt-0">
                      <span className="text-sm">to</span>
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs text-gray-400 font-medium">End Date</label>
                      <input
                        type="date"
                        value={dateRange.end}
                        onChange={(e) => {
                          const newEnd = e.target.value;
                          if (newEnd >= dateRange.start) {
                            setDateRange(prev => ({ ...prev, end: newEnd }));
                          }
                        }}
                        min={dateRange.start}
                        max={data.length > 0 ? data[data.length - 1].date : '2024-12-31'}
                        className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      />
                    </div>
                    <button
                      onClick={() => {
                        if (data.length > 0) {
                          setDateRange({
                            start: data[0].date,
                            end: data[data.length - 1].date
                          });
                        } else {
                          setDateRange({ start: '2023-01-01', end: '2024-12-31' });
                        }
                      }}
                      className="px-4 py-2 mt-6 sm:mt-0 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-md text-sm text-gray-300 hover:text-white transition-colors"
                    >
                      Reset
                    </button>
                  </div>
                </div>
              </div>
              
              <StockChart 
                data={filteredData} 
                onHover={handleChartHover}
                onLeave={handleChartLeave}
              />
            </div>

            {/* Sentiment Analysis Section */}
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 mb-8">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider flex items-center gap-2">
                  <Activity className="w-4 h-4" /> Sentiment-Return Correlation Analysis
                </h3>
                <span className="text-[10px] bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded border border-amber-500/20">Why Sentiment Works</span>
              </div>
              
              {/* Analysis Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-800">
                      <th className="text-left py-3 px-4 text-gray-400 font-medium">Ticker</th>
                      <th className="text-center py-3 px-4 text-gray-400 font-medium">News Articles</th>
                      <th className="text-center py-3 px-4 text-gray-400 font-medium">Avg/Day</th>
                      <th className="text-center py-3 px-4 text-gray-400 font-medium">Sentiment Mean</th>
                      <th className="text-center py-3 px-4 text-gray-400 font-medium">Next-Day Correlation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { ticker: 'GOOGL', articles: 2395, avgDay: 4.61, sentMean: 0.052, corr: 0.0899, pval: 0.057 },
                      { ticker: 'AAPL', articles: 2336, avgDay: 3.69, sentMean: 0.016, corr: 0.0723, pval: 0.120 },
                      { ticker: 'META', articles: 2248, avgDay: 4.20, sentMean: -0.084, corr: 0.0358, pval: 0.451 },
                      { ticker: 'MSFT', articles: 2379, avgDay: 4.50, sentMean: 0.164, corr: 0.0353, pval: 0.444 },
                      { ticker: 'AMZN', articles: 2212, avgDay: 4.13, sentMean: 0.009, corr: 0.0045, pval: 0.923 },
                    ].map((row, idx) => (
                      <tr key={row.ticker} className={`border-b border-gray-800/50 transition-colors cursor-pointer hover:bg-gray-800/50 ${ticker === row.ticker ? 'bg-indigo-500/10 hover:bg-indigo-500/20' : ''}`}>
                        <td className="py-3 px-4">
                          <span className={`font-medium ${ticker === row.ticker ? 'text-indigo-400' : 'text-gray-200'}`}>
                            {row.ticker}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center text-gray-300">{row.articles.toLocaleString()}</td>
                        <td className="py-3 px-4 text-center text-gray-300">{row.avgDay.toFixed(2)}</td>
                        <td className="py-3 px-4 text-center">
                          <span className={row.sentMean >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                            {row.sentMean >= 0 ? '+' : ''}{row.sentMean.toFixed(3)}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center">
                          <div className="flex items-center justify-center gap-2">
                            <span className={`font-mono ${row.corr > 0.05 ? 'text-emerald-400' : 'text-gray-400'}`}>
                              r={row.corr > 0 ? '+' : ''}{row.corr.toFixed(4)}
                            </span>
                            <span className="text-[10px] text-gray-500">(p={row.pval.toFixed(3)})</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              {/* Statistical Terms Note */}
              <div className="mt-4 flex flex-wrap gap-6 px-4 py-3 bg-gray-950/30 rounded-lg border border-gray-800/50">
                <div className="flex items-start gap-2">
                  <span className="text-xs font-bold text-indigo-400 bg-indigo-500/10 px-1.5 py-0.5 rounded">r</span>
                  <p className="text-[11px] text-gray-400 leading-relaxed max-w-xs">
                    <span className="font-medium text-gray-300">Pearson correlation coefficient</span> — measures the linear relationship between sentiment and next-day returns. Values range from -1 (perfect negative) to +1 (perfect positive), with 0 indicating no correlation.
                  </p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-xs font-bold text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">p</span>
                  <p className="text-[11px] text-gray-400 leading-relaxed max-w-xs">
                    <span className="font-medium text-gray-300">P-value (significance)</span> — probability that the correlation occurred by chance. Lower is better: p &lt; 0.05 is statistically significant, p &lt; 0.1 is marginally significant.
                  </p>
                </div>
              </div>
              
              {/* Insights */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-950/50 rounded-lg p-4 border border-gray-800">
                  <h4 className="text-emerald-400 text-xs font-semibold mb-2">Best: GOOGL</h4>
                  <p className="text-[11px] text-gray-400 leading-relaxed">
                    Highest next-day correlation (r=0.09, p=0.057). Sentiment has predictive power for next-day returns.
                  </p>
                </div>
                <div className="bg-gray-950/50 rounded-lg p-4 border border-gray-800">
                  <h4 className="text-rose-400 text-xs font-semibold mb-2">Worst: AMZN</h4>
                  <p className="text-[11px] text-gray-400 leading-relaxed">
                    Near-zero correlation (r=0.004, p=0.92). Sentiment does not predict returns - may add noise to model.
                  </p>
                </div>
                <div className="bg-gray-950/50 rounded-lg p-4 border border-gray-800">
                  <h4 className="text-amber-400 text-xs font-semibold mb-2">Key Insight</h4>
                  <p className="text-[11px] text-gray-400 leading-relaxed">
                    Sentiment improves predictions only when there's statistical correlation with returns. For efficient markets, news is priced in quickly.
                  </p>
                </div>
              </div>
            </div>

            {/* Secondary Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
               <CorrelationChart data={data} />
               
               {/* Enhanced System Architecture Comparison */}
               <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800 flex flex-col">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider flex items-center gap-2">
                      <Network className="w-4 h-4" /> Neural Architecture Comparison
                    </h3>
                    <span className="text-[10px] bg-gray-800 text-gray-400 px-2 py-0.5 rounded border border-gray-700">TensorFlow / Keras</span>
                  </div>

                  {/* Training Config Summary */}
                  <div className="flex flex-wrap gap-3 mb-4 text-[10px]">
                    <span className="bg-gray-800/50 text-gray-400 px-2 py-1 rounded border border-gray-700">Sequence: 20 days</span>
                    <span className="bg-gray-800/50 text-gray-400 px-2 py-1 rounded border border-gray-700">Optimizer: Adam</span>
                    <span className="bg-gray-800/50 text-gray-400 px-2 py-1 rounded border border-gray-700">Loss: MSE</span>
                    <span className="bg-gray-800/50 text-gray-400 px-2 py-1 rounded border border-gray-700">Early Stop: patience=10</span>
                  </div>

                  <div className="grid grid-cols-2 gap-4 flex-1">
                    {/* Base Model Column */}
                    <div className="flex flex-col gap-2 relative group">
                       {/* Header */}
                       <div className="text-center mb-1">
                         <span className="text-rose-400 text-xs font-bold bg-rose-500/10 px-2 py-1 rounded border border-rose-500/20">Base Model (5 features)</span>
                       </div>
                       
                       {/* Stack */}
                       <div className="flex-1 bg-gray-950/80 rounded-lg border border-gray-800 p-3 flex flex-col items-center gap-1.5 relative overflow-hidden">
                         {/* Input */}
                         <div className="w-full bg-gray-900 border border-gray-700 p-2 rounded text-center z-10">
                           <p className="text-[9px] text-gray-500 uppercase font-mono">Input Shape (20, 5)</p>
                           <p className="text-[11px] text-gray-300 font-medium">Open, High, Low, Close, Volume</p>
                         </div>
                         <ArrowDown className="w-3 h-3 text-gray-600" />
                         
                         {/* Hidden Layers */}
                         <div className="w-full bg-gray-800 border border-gray-700 p-2 rounded text-center z-10 shadow-lg shadow-rose-900/5">
                           <p className="text-[9px] text-gray-500 uppercase font-mono mb-1">LSTM Layers</p>
                           <p className="text-[11px] text-gray-200">LSTM (64) → Dropout (0.2)</p>
                           <div className="h-px w-full bg-gray-700 my-1"></div>
                           <p className="text-[11px] text-gray-200">LSTM (32) → Dropout (0.2)</p>
                           <div className="h-px w-full bg-gray-700 my-1"></div>
                           <p className="text-[11px] text-gray-200">Dense (16, ReLU)</p>
                         </div>
                         
                         <ArrowDown className="w-3 h-3 text-gray-600" />
                         
                         {/* Output */}
                         <div className="w-full bg-rose-500/10 border border-rose-500/30 p-2 rounded text-center z-10">
                           <p className="text-[9px] text-rose-400 uppercase font-mono">Output (1)</p>
                           <p className="text-xs font-bold text-rose-300">Close Price (t+1)</p>
                         </div>

                         {/* Background Effect */}
                         <div className="absolute inset-0 bg-gradient-to-b from-transparent via-rose-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
                       </div>
                    </div>

                    {/* Advanced Model Column */}
                    <div className="flex flex-col gap-2 relative group">
                       {/* Header */}
                       <div className="text-center mb-1">
                         <span className="text-emerald-400 text-xs font-bold bg-emerald-500/10 px-2 py-1 rounded border border-emerald-500/20">Advanced Model (6 features)</span>
                       </div>
                       
                       {/* Stack */}
                       <div className="flex-1 bg-gray-950/80 rounded-lg border border-gray-800 p-3 flex flex-col items-center gap-1.5 relative overflow-hidden">
                         {/* Input */}
                         <div className="w-full bg-gray-900 border border-gray-700 p-2 rounded text-center z-10 relative">
                           <p className="text-[9px] text-gray-500 uppercase font-mono">Input Shape (20, 6)</p>
                           <div className="flex items-center justify-center gap-1">
                             <span className="text-[11px] text-gray-300 font-medium">OHLCV</span>
                             <span className="text-[11px] text-emerald-400 font-bold">+</span>
                             <span className="text-[11px] text-emerald-300 font-medium">FinBERT</span>
                           </div>
                         </div>
                         <ArrowDown className="w-3 h-3 text-gray-600" />
                         
                         {/* Hidden Layers */}
                         <div className="w-full bg-gray-800 border border-gray-700 p-2 rounded text-center z-10 shadow-lg shadow-emerald-900/5">
                           <p className="text-[9px] text-gray-500 uppercase font-mono mb-1">LSTM Layers</p>
                           <p className="text-[11px] text-gray-200">LSTM (64) → Dropout (0.2)</p>
                           <div className="h-px w-full bg-gray-700 my-1"></div>
                           <p className="text-[11px] text-gray-200">LSTM (32) → Dropout (0.2)</p>
                           <div className="h-px w-full bg-gray-700 my-1"></div>
                           <p className="text-[11px] text-gray-200">Dense (16, ReLU)</p>
                         </div>
                         
                         <ArrowDown className="w-3 h-3 text-gray-600" />
                         
                         {/* Output */}
                         <div className="w-full bg-emerald-500/10 border border-emerald-500/30 p-2 rounded text-center z-10">
                           <p className="text-[9px] text-emerald-400 uppercase font-mono">Output (1)</p>
                           <p className="text-xs font-bold text-emerald-300">Close Price (t+1)</p>
                         </div>

                          {/* Background Effect */}
                         <div className="absolute inset-0 bg-gradient-to-b from-transparent via-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
                       </div>
                    </div>
                  </div>

                  {/* Key Difference Note */}
                  <div className="mt-4 pt-3 border-t border-gray-800">
                    <p className="text-[10px] text-gray-500 text-center">
                      <span className="text-gray-400 font-medium">Key Difference:</span> Advanced model adds daily FinBERT sentiment score (aggregated mean from news articles) as 5th input feature
                    </p>
                  </div>
               </div>
            </div>
          </>
        ) : (
          <div className="flex h-64 items-center justify-center text-gray-500">
             No Data Available
          </div>
        )}
      </main>
    </div>
  );
}

export default App;