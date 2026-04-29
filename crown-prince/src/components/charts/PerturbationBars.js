import { useState } from "react";
import { scaleBand, scaleLinear } from "d3-scale";

// Sparse drug perturbation η: signed bars centred on zero.
export default function PerturbationBars({ data, l1, height = 240, caption }) {
  const [hover, setHover] = useState(null);
  const margin = { top: 14, right: 14, bottom: 50, left: 50 };
  const width = 560;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const x = scaleBand()
    .domain(data.map((d) => d.name))
    .range([0, innerW])
    .padding(0.25);
  const y = scaleLinear().domain([-1, 1]).range([innerH, 0]).nice();

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full text-light/70">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          <line x1={0} x2={innerW} y1={y(0)} y2={y(0)} stroke="currentColor" opacity={0.3} />
          {data.map((d, i) => {
            const c = d.value > 0 ? "#58E6D9" : d.value < 0 ? "#EE6677" : "#444";
            const yTop = y(Math.max(d.value, 0));
            const yBot = y(Math.min(d.value, 0));
            return (
              <rect
                key={d.name}
                x={x(d.name)}
                y={yTop}
                width={x.bandwidth()}
                height={yBot - yTop}
                fill={c}
                opacity={hover === null || hover === i ? 0.85 : 0.4}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
                style={{ cursor: "pointer" }}
              />
            );
          })}
          {data.map((d) => (
            <text
              key={`l-${d.name}`}
              x={x(d.name) + x.bandwidth() / 2}
              y={innerH + 14}
              textAnchor="middle"
              fontSize={8}
              fill="currentColor"
              opacity={0.55}
              transform={`rotate(-30 ${x(d.name) + x.bandwidth() / 2} ${innerH + 14})`}
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
            η_ij
          </text>
        </g>
      </svg>
      <div className="mt-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover !== null
            ? `${data[hover].name}  η = ${data[hover].value.toFixed(2)}`
            : `‖η‖₁ = ${l1.toFixed(2)}  ·  ${data.filter((d) => d.value !== 0).length} active edges`}
        </span>
      </div>
    </div>
  );
}
