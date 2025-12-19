import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { StockDataPoint } from '../types';

interface StockChartProps {
  data: StockDataPoint[];
  onHover?: (day: StockDataPoint) => void;
  onLeave?: () => void;
}

const CustomTooltip = ({ active, payload, label, onHover, onLeave }: any) => {
  React.useEffect(() => {
    if (active && payload && payload.length) {
      const dayData = payload[0].payload as StockDataPoint;
      onHover?.(dayData);
    } else {
      onLeave?.();
    }
  }, [active, payload, onHover, onLeave]);

  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 border border-gray-700 p-3 rounded shadow-xl text-xs">
        <p className="font-bold text-gray-200 mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }}></div>
            <span className="text-gray-400">{entry.name}:</span>
            <span className="font-mono font-medium text-gray-100">{entry.value.toFixed(2)}</span>
          </div>
        ))}
        <div className="mt-2 pt-2 border-t border-gray-700">
           <p className="text-gray-400 italic">Headline:</p>
           <p className="text-gray-300 max-w-xs">{payload[0].payload.headline}</p>
        </div>
      </div>
    );
  }
  return null;
};

export const StockChart: React.FC<StockChartProps> = ({ data, onHover, onLeave }) => {
  return (
    <div className="h-[400px] min-h-[400px] w-full min-w-0 bg-gray-900/50 p-4 rounded-xl border border-gray-800">
      <ResponsiveContainer width="100%" height="100%" minHeight={400}>
        <LineChart
          data={data}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
          onMouseLeave={onLeave}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
          <XAxis 
            dataKey="date" 
            stroke="#9ca3af" 
            fontSize={12} 
            tickFormatter={(val) => val.slice(5)} // Show MM-DD
          />
          <YAxis 
            domain={['auto', 'auto']} 
            stroke="#9ca3af" 
            fontSize={12}
          />
          <Tooltip content={<CustomTooltip onHover={onHover} onLeave={onLeave} />} />
          <Legend wrapperStyle={{ paddingTop: '10px' }} />
          
          <Line
            type="monotone"
            dataKey="close"
            name="Actual Close Price (GT)"
            stroke="#ffffff"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6 }}
          />
          <Line
            type="monotone"
            dataKey="basePrediction"
            name="Base Model (Univariate)"
            stroke="#ef4444" // Red
            strokeWidth={2}
            strokeDasharray="5 5" // Dashed
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="advancedPrediction"
            name="Advanced Model (Sentiment)"
            stroke="#10b981" // Green
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
