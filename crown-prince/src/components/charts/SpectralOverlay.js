import { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line as d3line } from "d3-shape";
import { max as d3max } from "d3-array";

// Three-spectrum overlay: ground IR, excited Raman, fluorescence emission.
export default function SpectralOverlay({ grid, series, height = 240, caption }) {
  const [hover, setHover] = useState(null);
  const margin = { top: 14, right: 14, bottom: 36, left: 44 };
  const width = 560;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const yMax = useMemo(
    () => Math.max(...series.map((s) => d3max(s.values) || 0)),
    [series]
  );
  const x = scaleLinear().domain([grid[0], grid[grid.length - 1]]).range([0, innerW]);
  const y = scaleLinear().domain([0, yMax * 1.05]).range([innerH, 0]);

  const lineGen = d3line()
    .x((_, i) => x(grid[i]))
    .y((v) => y(v));

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const omega = x.invert(px);
    const idx = Math.max(
      0,
      Math.min(grid.length - 1, Math.round((omega - grid[0]) / (grid[1] - grid[0])))
    );
    setHover({ omega: grid[idx], idx });
  };

  return (
    <div className="w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full text-light/70"
        onMouseLeave={() => setHover(null)}
      >
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* gridlines */}
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
          {/* curves */}
          {series.map((s) => (
            <path
              key={s.name}
              d={lineGen(s.values)}
              fill="none"
              stroke={s.color}
              strokeWidth={1.5}
              opacity={0.9}
            />
          ))}
          {/* hover line */}
          {hover && (
            <g>
              <line
                x1={x(hover.omega)}
                x2={x(hover.omega)}
                y1={0}
                y2={innerH}
                stroke="#58E6D9"
                opacity={0.4}
                strokeDasharray="3 3"
              />
              {series.map((s) => (
                <circle
                  key={s.name}
                  cx={x(hover.omega)}
                  cy={y(s.values[hover.idx])}
                  r={3}
                  fill={s.color}
                />
              ))}
            </g>
          )}
          {/* axes */}
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
          {y.ticks(4).map((t) => (
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
            x={innerW / 2}
            y={innerH + 28}
            textAnchor="middle"
            fontSize={10}
            fill="currentColor"
            opacity={0.6}
          >
            ω (cm⁻¹)
          </text>
          {/* legend */}
          <g transform={`translate(${innerW - 130}, 4)`}>
            <rect x={0} y={0} width={130} height={series.length * 14 + 6} fill="#1b1b1b" opacity={0.7} rx={3} />
            {series.map((s, i) => (
              <g key={s.name} transform={`translate(0, ${i * 14 + 4})`}>
                <line x1={6} y1={6} x2={20} y2={6} stroke={s.color} strokeWidth={2} />
                <text x={26} y={9} fontSize={9} fill="currentColor" opacity={0.75}>
                  {s.name}
                </text>
              </g>
            ))}
          </g>
          {/* invisible interaction layer */}
          <rect
            x={0}
            y={0}
            width={innerW}
            height={innerH}
            fill="transparent"
            onMouseMove={onMove}
          />
        </g>
      </svg>
      <div className="mt-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover
            ? `ω = ${hover.omega.toFixed(0)} cm⁻¹  ${series
                .map((s) => `${s.name[0]}=${s.values[hover.idx].toFixed(2)}`)
                .join("  ")}`
            : "hover to inspect"}
        </span>
      </div>
    </div>
  );
}
