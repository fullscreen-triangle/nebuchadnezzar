import { scaleLinear } from "d3-scale";

// Stacked horizontal bar visualising decomposition (e.g. Stokes shift = vibrational + solvent).
export default function StackedBars({ items, total, label, sub, color1 = "#58E6D9", color2 = "#AA3377" }) {
  const W = 560;
  const H = 56;
  const x = scaleLinear().domain([0, total]).range([0, W]);

  let acc = 0;
  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full text-light/70">
        {items.map((it, i) => {
          const w = x(it.value);
          const xPos = x(acc);
          acc += it.value;
          const c = i === 0 ? color1 : color2;
          return (
            <g key={it.label}>
              <rect x={xPos} y={20} width={w} height={20} fill={c} opacity={0.85} />
              <text
                x={xPos + w / 2}
                y={34}
                textAnchor="middle"
                fontSize={10}
                fontFamily="monospace"
                fill="#1b1b1b"
                fontWeight="bold"
              >
                {Math.round(it.value)}
              </text>
              <text
                x={xPos + w / 2}
                y={14}
                textAnchor="middle"
                fontSize={9}
                fill={c}
                opacity={0.85}
              >
                {it.label}
              </text>
            </g>
          );
        })}
        <line x1={0} y1={45} x2={W} y2={45} stroke="currentColor" opacity={0.3} />
        <text x={0} y={54} fontSize={9} fill="currentColor" opacity={0.55}>0</text>
        <text x={W} y={54} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.55}>
          {Math.round(total)} cm⁻¹
        </text>
      </svg>
      <div className="mt-1 flex items-baseline justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{label}</span>
        {sub && <span className="font-mono">{sub}</span>}
      </div>
    </div>
  );
}
