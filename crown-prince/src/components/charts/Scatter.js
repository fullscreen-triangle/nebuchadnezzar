import { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";
import { extent } from "d3-array";

export default function Scatter({
  data, // [{x, y, label, color}]
  xLabel = "x",
  yLabel = "y",
  height = 320,
  diagonal = false,
  caption,
}) {
  const [hover, setHover] = useState(null);
  const margin = { top: 14, right: 14, bottom: 36, left: 44 };
  const width = 560;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const xExt = useMemo(() => extent(data, (d) => d.x), [data]);
  const yExt = useMemo(() => extent(data, (d) => d.y), [data]);
  const lo = Math.min(xExt[0], yExt[0]);
  const hi = Math.max(xExt[1], yExt[1]);
  const xs = scaleLinear().domain(diagonal ? [lo, hi] : xExt).range([0, innerW]).nice();
  const ys = scaleLinear().domain(diagonal ? [lo, hi] : yExt).range([innerH, 0]).nice();

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full text-light/70">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* gridlines */}
          {ys.ticks(5).map((t) => (
            <line
              key={`y-${t}`}
              x1={0}
              x2={innerW}
              y1={ys(t)}
              y2={ys(t)}
              stroke="currentColor"
              opacity={0.08}
            />
          ))}
          {xs.ticks(5).map((t) => (
            <line
              key={`x-${t}`}
              x1={xs(t)}
              x2={xs(t)}
              y1={0}
              y2={innerH}
              stroke="currentColor"
              opacity={0.08}
            />
          ))}
          {/* y=x diagonal */}
          {diagonal && (
            <line
              x1={xs(lo)}
              y1={ys(lo)}
              x2={xs(hi)}
              y2={ys(hi)}
              stroke="#EE6677"
              strokeDasharray="4 3"
              strokeWidth={1.2}
            />
          )}
          {/* points */}
          {data.map((d, i) => {
            const isHover = hover === i;
            return (
              <circle
                key={i}
                cx={xs(d.x)}
                cy={ys(d.y)}
                r={isHover ? 6 : 4}
                fill={d.color || "#58E6D9"}
                opacity={hover === null || isHover ? 0.9 : 0.3}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
                style={{ cursor: "pointer" }}
              />
            );
          })}
          {/* axes */}
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" opacity={0.3} />
          <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" opacity={0.3} />
          {xs.ticks(5).map((t) => (
            <text
              key={`xt-${t}`}
              x={xs(t)}
              y={innerH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {t}
            </text>
          ))}
          {ys.ticks(5).map((t) => (
            <text
              key={`yt-${t}`}
              x={-6}
              y={ys(t) + 3}
              textAnchor="end"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {t}
            </text>
          ))}
          <text
            x={innerW / 2}
            y={innerH + 28}
            textAnchor="middle"
            fontSize={10}
            fill="currentColor"
            opacity={0.6}
          >
            {xLabel}
          </text>
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
        </g>
      </svg>
      <div className="mt-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover !== null
            ? `${data[hover].label} · ${data[hover].x.toFixed(2)}, ${data[hover].y.toFixed(2)}`
            : "hover a point"}
        </span>
      </div>
    </div>
  );
}
