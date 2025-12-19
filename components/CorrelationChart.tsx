import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { StockDataPoint } from '../types';

interface CorrelationChartProps {
  data: StockDataPoint[];
}

export const CorrelationChart: React.FC<CorrelationChartProps> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!data || !svgRef.current) return;

    // Clear previous render
    d3.select(svgRef.current).selectAll("*").remove();

    // Dimensions
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const width = svgRef.current.clientWidth - margin.left - margin.right;
    const height = 350 - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Prepare Data: Calculate daily return % vs sentiment
    const plotData = data.slice(1).map((d, i) => {
      const prevClose = data[i].close; // i is shifted by 1 relative to slice, so this is previous day
      const dailyReturn = ((d.close - prevClose) / prevClose) * 100;
      return {
        sentiment: d.sentiment,
        return: dailyReturn,
        date: d.date
      };
    });

    // Scales
    const x = d3.scaleLinear()
      .domain([-1, 1])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([
        (d3.min(plotData, d => d.return) as number) - 0.5,
        (d3.max(plotData, d => d.return) as number) + 0.5
      ])
      .range([height, 0]);

    // Color scale based on sentiment
    const color = d3.scaleLinear<string>()
      .domain([-1, 0, 1])
      .range(["#ef4444", "#6b7280", "#10b981"]);

    // Axes
    svg.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(5))
      .attr("color", "#9ca3af")
      .append("text")
      .attr("x", width / 2)
      .attr("y", 35)
      .attr("fill", "#9ca3af")
      .text("Sentiment Score (-1 to +1)");

    svg.append("g")
      .call(d3.axisLeft(y).ticks(5))
      .attr("color", "#9ca3af")
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -40)
      .attr("x", -height / 2)
      .attr("dy", "1em")
      .attr("text-anchor", "middle")
      .attr("fill", "#9ca3af")
      .text("Daily Return (%)");

    // Grid lines
    svg.append("g")
      .attr("class", "grid")
      .attr("opacity", 0.1)
      .call(d3.axisLeft(y)
        .tickSize(-width)
        .tickFormat(() => "")
      );
    svg.append("g")
      .attr("class", "grid")
      .attr("opacity", 0.1)
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x)
        .tickSize(-height)
        .tickFormat(() => "")
      );

    // Center Lines (0,0)
    svg.append("line")
      .attr("x1", 0)
      .attr("y1", y(0))
      .attr("x2", width)
      .attr("y2", y(0))
      .attr("stroke", "#4b5563")
      .attr("stroke-dasharray", "4");

    svg.append("line")
      .attr("x1", x(0))
      .attr("y1", 0)
      .attr("x2", x(0))
      .attr("y2", height)
      .attr("stroke", "#4b5563")
      .attr("stroke-dasharray", "4");

    // Tooltip
    const tooltip = d3.select("body").append("div")
      .attr("class", "absolute hidden bg-gray-800 border border-gray-700 p-2 rounded text-xs text-gray-200 pointer-events-none shadow-lg")
      .style("opacity", 0);

    // Dots
    svg.selectAll("circle")
      .data(plotData)
      .enter()
      .append("circle")
      .attr("cx", d => x(d.sentiment))
      .attr("cy", d => y(d.return))
      .attr("r", 5)
      .attr("fill", d => color(d.sentiment))
      .attr("opacity", 0.8)
      .attr("stroke", "#1f2937")
      .attr("stroke-width", 1)
      .on("mouseover", (event, d) => {
        d3.select(event.currentTarget).attr("r", 8).attr("stroke", "#fff");
        tooltip.transition().duration(200).style("opacity", 1);
        tooltip.html(`
          Date: ${d.date}<br/>
          Sentiment: ${d.sentiment.toFixed(2)}<br/>
          Return: ${d.return.toFixed(2)}%
        `)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 28) + "px");
        tooltip.classed("hidden", false);
      })
      .on("mouseout", (event) => {
        d3.select(event.currentTarget).attr("r", 5).attr("stroke", "#1f2937");
        tooltip.transition().duration(500).style("opacity", 0);
        tooltip.classed("hidden", true);
      });

  }, [data]);

  return (
    <div className="h-[420px] w-full bg-gray-900/50 p-4 rounded-xl border border-gray-800">
      <h3 className="text-gray-400 text-sm font-semibold mb-2 uppercase tracking-wider">
        Sentiment vs. Price Correlation (D3.js)
      </h3>
      <svg ref={svgRef} className="w-full h-[350px] overflow-visible" />
    </div>
  );
};
