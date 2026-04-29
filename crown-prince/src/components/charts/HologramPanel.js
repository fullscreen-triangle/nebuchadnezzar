import { useState } from "react";
import { scaleLinear, scaleSequential } from "d3-scale";
import { interpolateInferno } from "d3-scale-chromatic";

// 2D hologram heatmap H(omega, t). Magnitude as intensity, phase optional overlay.
export default function HologramPanel({ hologram, height = 280, caption }) {
  const [hover, setHover] = useState(null);
  const margin = { top: 10, right: 60, bottom: 36, left: 50 };
  const width = 560;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const T = hologram.mag.length;
  const W = hologram.mag[0].length;
  // Find max mag for normalisation
  let maxMag = 0;
  for (let t = 0; t < T; t++)
    for (let w = 0; w < W; w++)
      if (hologram.mag[t][w] > maxMag) maxMag = hologram.mag[t][w];

  const x = scaleLinear()
    .domain([hologram.gridOmega[0], hologram.gridOmega[W - 1]])
    .range([0, innerW]);
  const y = scaleLinear().domain([0, 1]).range([0, innerH]); // t in [0,1]
  const color = scaleSequential(interpolateInferno).domain([0, 1]);

  const cellW = innerW / W;
  const cellH = innerH / T;

  return (
    <div className="w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full text-light/70"
        onMouseLeave={() => setHover(null)}
      >
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {hologram.mag.map((row, ti) =>
            row.map((m, wi) => (
              <rect
                key={`${ti}-${wi}`}
                x={wi * cellW}
                y={ti * cellH}
                width={cellW + 0.5}
                height={cellH + 0.5}
                fill={color(m / maxMag)}
              />
            ))
          )}
          <rect
            x={0}
            y={0}
            width={innerW}
            height={innerH}
            fill="transparent"
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const px = e.clientX - rect.left;
              const py = e.clientY - rect.top;
              const wi = Math.max(0, Math.min(W - 1, Math.floor((px / innerW) * W)));
              const ti = Math.max(0, Math.min(T - 1, Math.floor((py / innerH) * T)));
              setHover({
                omega: hologram.gridOmega[wi],
                t: hologram.gridT[ti],
                m: hologram.mag[ti][wi],
                phi: hologram.phase[ti][wi],
              });
            }}
          />
          {/* axes */}
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" opacity={0.3} />
          <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" opacity={0.3} />
          {x.ticks(6).map((t) => (
            <text
              key={t}
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
              {t.toFixed(2)}
            </text>
          ))}
          <text x={innerW / 2} y={innerH + 28} textAnchor="middle" fontSize={10} fill="currentColor" opacity={0.6}>
            ω (cm⁻¹)
          </text>
          <text
            x={-innerH / 2}
            y={-36}
            transform="rotate(-90)"
            textAnchor="middle"
            fontSize={10}
            fill="currentColor"
            opacity={0.6}
          >
            t (cycles)
          </text>
          {/* legend */}
          <g transform={`translate(${innerW + 10}, 0)`}>
            <defs>
              <linearGradient id="ho-grad" x1="0" x2="0" y1="0" y2="1">
                {Array.from({ length: 11 }).map((_, k) => (
                  <stop key={k} offset={`${k * 10}%`} stopColor={color(1 - k / 10)} />
                ))}
              </linearGradient>
            </defs>
            <rect x={0} y={0} width={10} height={innerH} fill="url(#ho-grad)" />
            <text x={14} y={8} fontSize={9} fill="currentColor" opacity={0.6}>|H|</text>
            <text x={14} y={innerH - 2} fontSize={9} fill="currentColor" opacity={0.6}>0</text>
          </g>
        </g>
      </svg>
      <div className="mt-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover
            ? `ω=${hover.omega.toFixed(0)}  t=${hover.t.toFixed(2)}  |H|=${hover.m.toFixed(2)}  φ=${hover.phi.toFixed(2)}`
            : "hover the hologram"}
        </span>
      </div>
    </div>
  );
}
