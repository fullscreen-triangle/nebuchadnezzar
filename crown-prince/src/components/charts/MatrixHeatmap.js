import { useState } from "react";
import { scaleBand, scaleSequential } from "d3-scale";
import { interpolateRdBu } from "d3-scale-chromatic";

// Diverging-colour matrix heatmap (e.g. coupling matrix K_ij in [-1, 1]).
export default function MatrixHeatmap({
  matrix,
  labels,
  height = 360,
  caption,
  sym = "K",
  domain = [-1, 1],
}) {
  const [hover, setHover] = useState(null);
  const margin = { top: 10, right: 70, bottom: 40, left: 50 };
  const width = 480;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const N = matrix.length;
  const idx = labels.map((_, i) => i);
  const x = scaleBand().domain(idx).range([0, innerW]).padding(0.04);
  const y = scaleBand().domain(idx).range([0, innerH]).padding(0.04);
  const color = scaleSequential(interpolateRdBu).domain([domain[1], domain[0]]);

  const cells = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) cells.push({ i, j, v: matrix[i][j] });
  }

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full text-light/70">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {cells.map((c) => (
            <rect
              key={`${c.i}-${c.j}`}
              x={x(c.j)}
              y={y(c.i)}
              width={x.bandwidth()}
              height={y.bandwidth()}
              fill={color(c.v)}
              opacity={hover && hover.i !== c.i && hover.j !== c.j ? 0.3 : 1}
              onMouseEnter={() => setHover(c)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
          {labels.map((lab, i) => (
            <text
              key={`r-${i}`}
              x={-6}
              y={y(i) + y.bandwidth() / 2 + 3}
              textAnchor="end"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {lab}
            </text>
          ))}
          {labels.map((lab, j) => (
            <text
              key={`c-${j}`}
              x={x(j) + x.bandwidth() / 2}
              y={innerH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {lab}
            </text>
          ))}
          {/* Legend */}
          <g transform={`translate(${innerW + 10}, 0)`}>
            <defs>
              <linearGradient id="mh-grad" x1="0" x2="0" y1="0" y2="1">
                {Array.from({ length: 11 }).map((_, k) => (
                  <stop key={k} offset={`${k * 10}%`} stopColor={color(domain[1] - (k / 10) * (domain[1] - domain[0]))} />
                ))}
              </linearGradient>
            </defs>
            <rect x={0} y={0} width={12} height={innerH} fill="url(#mh-grad)" />
            <text x={18} y={8} fontSize={9} fill="currentColor" opacity={0.6}>{`+${domain[1]}`}</text>
            <text x={18} y={innerH / 2} fontSize={9} fill="currentColor" opacity={0.6}>0</text>
            <text x={18} y={innerH - 2} fontSize={9} fill="currentColor" opacity={0.6}>{domain[0]}</text>
          </g>
        </g>
      </svg>
      <div className="mt-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover
            ? `${sym}[${labels[hover.i]}, ${labels[hover.j]}] = ${hover.v.toFixed(2)}`
            : "hover a cell"}
        </span>
      </div>
    </div>
  );
}
