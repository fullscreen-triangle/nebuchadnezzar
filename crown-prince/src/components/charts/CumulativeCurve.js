import { useState } from "react";
import { scaleLinear } from "d3-scale";
import { line as d3line, area as d3area, curveStepAfter } from "d3-shape";

export default function CumulativeCurve({ passed, threshold = 95, height = 280, caption }) {
  const [hover, setHover] = useState(null);
  const margin = { top: 14, right: 14, bottom: 36, left: 44 };
  const width = 560;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  // Build cumulative pass-rate sequence.
  const series = [];
  let p = 0;
  for (let i = 0; i < passed.length; i++) {
    if (passed[i]) p++;
    series.push({ i: i + 1, rate: (p / (i + 1)) * 100 });
  }

  const x = scaleLinear().domain([0, series.length]).range([0, innerW]);
  const y = scaleLinear().domain([60, 102]).range([innerH, 0]);

  const lineGen = d3line()
    .x((d) => x(d.i))
    .y((d) => y(d.rate))
    .curve(curveStepAfter);
  const areaGen = d3area()
    .x((d) => x(d.i))
    .y0(innerH)
    .y1((d) => y(d.rate))
    .curve(curveStepAfter);

  const final = series[series.length - 1];

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
          <line
            x1={0}
            x2={innerW}
            y1={y(threshold)}
            y2={y(threshold)}
            stroke="#EE6677"
            strokeDasharray="4 3"
            opacity={0.7}
          />
          <path d={areaGen(series)} fill="#58E6D9" opacity={0.12} />
          <path d={lineGen(series)} fill="none" stroke="#58E6D9" strokeWidth={1.6} />
          {hover !== null && (
            <g>
              <line x1={x(hover.i)} x2={x(hover.i)} y1={0} y2={innerH} stroke="#58E6D9" opacity={0.4} />
              <circle cx={x(hover.i)} cy={y(hover.rate)} r={4} fill="#58E6D9" />
            </g>
          )}
          <rect
            x={0}
            y={0}
            width={innerW}
            height={innerH}
            fill="transparent"
            onMouseLeave={() => setHover(null)}
            onMouseMove={(e) => {
              const pt = e.currentTarget.getBoundingClientRect();
              const px = e.clientX - pt.left;
              const idx = Math.max(
                0,
                Math.min(series.length - 1, Math.floor(x.invert(px)))
              );
              setHover(series[idx]);
            }}
          />
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" opacity={0.3} />
          <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" opacity={0.3} />
          {x.ticks(6).map((t) => (
            <text
              key={`xt-${t}`}
              x={x(t)}
              y={innerH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {t}
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
              {t}%
            </text>
          ))}
          <text x={innerW / 2} y={innerH + 28} textAnchor="middle" fontSize={10} fill="currentColor" opacity={0.6}>
            test index
          </text>
        </g>
      </svg>
      <div className="mt-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover ? `test ${hover.i} · ${hover.rate.toFixed(1)}%` : `final ${final.rate.toFixed(1)}%`}
        </span>
      </div>
    </div>
  );
}
