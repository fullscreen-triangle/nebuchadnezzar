import { useState } from "react";
import { scaleBand, scaleLinear } from "d3-scale";

// det(Hol_l) per cycle, with healthy band shaded.
export default function HolonomyBars({ data, height = 220, caption }) {
  const [hover, setHover] = useState(null);
  const margin = { top: 14, right: 14, bottom: 36, left: 50 };
  const width = 560;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const x = scaleBand()
    .domain(data.map((d) => d.name))
    .range([0, innerW])
    .padding(0.3);
  const yMin = Math.min(0.4, ...data.map((d) => d.det));
  const yMax = Math.max(1.6, ...data.map((d) => d.det));
  const y = scaleLinear().domain([yMin, yMax]).range([innerH, 0]).nice();

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full text-light/70">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* healthy band */}
          <rect
            x={0}
            y={y(1.1)}
            width={innerW}
            height={y(0.9) - y(1.1)}
            fill="#228833"
            opacity={0.08}
          />
          <line
            x1={0}
            x2={innerW}
            y1={y(1)}
            y2={y(1)}
            stroke="#228833"
            opacity={0.5}
            strokeDasharray="4 3"
          />
          {data.map((d, i) => {
            const c = d.healthy ? "#58E6D9" : "#EE6677";
            return (
              <g key={d.name}>
                <line
                  x1={x(d.name) + x.bandwidth() / 2}
                  x2={x(d.name) + x.bandwidth() / 2}
                  y1={y(1)}
                  y2={y(d.det)}
                  stroke={c}
                  strokeWidth={2}
                  opacity={hover === null || hover === i ? 1 : 0.4}
                />
                <circle
                  cx={x(d.name) + x.bandwidth() / 2}
                  cy={y(d.det)}
                  r={5}
                  fill={c}
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                  style={{ cursor: "pointer" }}
                />
              </g>
            );
          })}
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" opacity={0.3} />
          {data.map((d) => (
            <text
              key={`l-${d.name}`}
              x={x(d.name) + x.bandwidth() / 2}
              y={innerH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {d.name}
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
              {t.toFixed(1)}
            </text>
          ))}
          <text
            x={-innerH / 2}
            y={-38}
            transform="rotate(-90)"
            textAnchor="middle"
            fontSize={10}
            fill="currentColor"
            opacity={0.6}
          >
            det(Hol_ℓ)
          </text>
        </g>
      </svg>
      <div className="mt-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover !== null
            ? `${data[hover].name}  det=${data[hover].det.toFixed(3)}  ${data[hover].healthy ? "✓" : "✗"}`
            : "hover a cycle"}
        </span>
      </div>
    </div>
  );
}
