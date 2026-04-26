import { useMemo, useState } from "react";
import { scaleBand, scaleSequential } from "d3-scale";
import { interpolateViridis } from "d3-scale-chromatic";

export default function Heatmap({
  data, // {rows: [labels], cols: [labels], values: number[rows][cols]}
  height = 360,
  caption,
}) {
  const [hover, setHover] = useState(null);
  const margin = { top: 10, right: 80, bottom: 40, left: 60 };

  const width = 560;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const x = useMemo(
    () => scaleBand().domain(data.cols).range([0, innerW]).padding(0.04),
    [data.cols, innerW]
  );
  const y = useMemo(
    () => scaleBand().domain(data.rows).range([0, innerH]).padding(0.04),
    [data.rows, innerH]
  );
  const color = useMemo(() => scaleSequential(interpolateViridis).domain([0, 1]), []);

  const cells = [];
  for (let i = 0; i < data.rows.length; i++) {
    for (let j = 0; j < data.cols.length; j++) {
      cells.push({ i, j, v: data.values[i][j] });
    }
  }

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full text-light/70">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {cells.map((c) => (
            <rect
              key={`${c.i}-${c.j}`}
              x={x(data.cols[c.j])}
              y={y(data.rows[c.i])}
              width={x.bandwidth()}
              height={y.bandwidth()}
              fill={color(c.v)}
              opacity={hover && (hover.i !== c.i && hover.j !== c.j) ? 0.25 : 1}
              onMouseEnter={() => setHover({ i: c.i, j: c.j, v: c.v })}
              onMouseLeave={() => setHover(null)}
            />
          ))}
          {/* row labels (every Nth) */}
          {data.rows.map((r, i) =>
            i % Math.ceil(data.rows.length / 14) === 0 ? (
              <text
                key={r}
                x={-6}
                y={y(r) + y.bandwidth() / 2}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.55}
              >
                {r}
              </text>
            ) : null
          )}
          {data.cols.map((c, j) =>
            j % Math.ceil(data.cols.length / 14) === 0 ? (
              <text
                key={c}
                x={x(c) + x.bandwidth() / 2}
                y={innerH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.55}
              >
                {c}
              </text>
            ) : null
          )}
          {/* legend */}
          <g transform={`translate(${innerW + 10}, 0)`}>
            <defs>
              <linearGradient id="hm-grad" x1="0" x2="0" y1="0" y2="1">
                {Array.from({ length: 11 }).map((_, k) => (
                  <stop
                    key={k}
                    offset={`${k * 10}%`}
                    stopColor={color(1 - k / 10)}
                  />
                ))}
              </linearGradient>
            </defs>
            <rect x={0} y={0} width={12} height={innerH} fill="url(#hm-grad)" />
            <text x={18} y={8} fontSize={9} fill="currentColor" opacity={0.6}>
              1.0
            </text>
            <text x={18} y={innerH - 2} fontSize={9} fill="currentColor" opacity={0.6}>
              0.0
            </text>
          </g>
        </g>
      </svg>
      <div className="mt-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover
            ? `${data.rows[hover.i]} × ${data.cols[hover.j]}  R = ${hover.v.toFixed(3)}`
            : "hover a cell"}
        </span>
      </div>
    </div>
  );
}
