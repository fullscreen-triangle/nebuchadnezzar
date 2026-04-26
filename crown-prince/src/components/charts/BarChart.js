import { useState } from "react";
import { scaleBand, scaleLinear } from "d3-scale";
import { max } from "d3-array";

export default function BarChart({
  data, // [{ label, value, color, sub }]
  height = 280,
  yLabel = "",
  threshold,
  caption,
}) {
  const [hover, setHover] = useState(null);
  const margin = { top: 14, right: 14, bottom: 36, left: 44 };
  const width = 560;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const x = scaleBand()
    .domain(data.map((d) => d.label))
    .range([0, innerW])
    .padding(0.18);
  const y = scaleLinear()
    .domain([0, Math.max(max(data, (d) => d.value), threshold || 0) * 1.1])
    .range([innerH, 0]);

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full text-light/70">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {y.ticks(5).map((t) => (
            <line
              key={t}
              x1={0}
              x2={innerW}
              y1={y(t)}
              y2={y(t)}
              stroke="currentColor"
              opacity={0.07}
            />
          ))}
          {threshold !== undefined && (
            <line
              x1={0}
              x2={innerW}
              y1={y(threshold)}
              y2={y(threshold)}
              stroke="#EE6677"
              strokeDasharray="4 3"
              opacity={0.7}
            />
          )}
          {data.map((d, i) => (
            <g key={d.label}>
              <rect
                x={x(d.label)}
                y={y(d.value)}
                width={x.bandwidth()}
                height={innerH - y(d.value)}
                fill={d.color || "#58E6D9"}
                opacity={hover === null || hover === i ? 0.9 : 0.4}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
                style={{ cursor: "pointer" }}
              />
              <text
                x={x(d.label) + x.bandwidth() / 2}
                y={y(d.value) - 4}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={hover === i ? 0.9 : 0.5}
              >
                {d.value}
              </text>
            </g>
          ))}
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" opacity={0.3} />
          {data.map((d) => (
            <text
              key={`l-${d.label}`}
              x={x(d.label) + x.bandwidth() / 2}
              y={innerH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {d.label}
            </text>
          ))}
          {y.ticks(5).map((t) => (
            <text
              key={`yt-${t}`}
              x={-6}
              y={y(t) + 3}
              textAnchor="end"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {t}
            </text>
          ))}
          {yLabel && (
            <text
              x={-innerH / 2}
              y={-32}
              transform="rotate(-90)"
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.6}
            >
              {yLabel}
            </text>
          )}
        </g>
      </svg>
      <div className="mt-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover !== null ? `${data[hover].label} · ${data[hover].sub || data[hover].value}` : "hover a bar"}
        </span>
      </div>
    </div>
  );
}
