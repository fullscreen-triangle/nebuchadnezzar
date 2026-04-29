import { useMemo, useState } from "react";
import { scaleSequential } from "d3-scale";
import { interpolateInferno } from "d3-scale-chromatic";
import { fft2d, fftShift2d, magnitude2d } from "@/lib/fft";

// 2D FFT of the hologram magnitude — diffraction pattern revealing molecular symmetry.
export default function DiffractionPattern({ hologram, height = 320, caption }) {
  const [hover, setHover] = useState(null);

  const diffraction = useMemo(() => {
    // Build a power-of-2 grid from hologram magnitude
    const N = 64;
    const T = hologram.mag.length;
    const W = hologram.mag[0].length;
    const grid = Array.from({ length: N }, () => new Array(N));
    for (let i = 0; i < N; i++) {
      const ti = Math.floor((i / N) * T);
      for (let j = 0; j < N; j++) {
        const wi = Math.floor((j / N) * W);
        grid[i][j] = { re: hologram.mag[ti][wi], im: 0 };
      }
    }
    const F = fft2d(grid);
    const shifted = fftShift2d(F);
    const mag = magnitude2d(shifted);
    // log scale for visibility
    let maxLog = 0;
    const log = mag.map((row) =>
      row.map((m) => {
        const v = Math.log(1 + m);
        if (v > maxLog) maxLog = v;
        return v;
      })
    );
    return { log, max: maxLog };
  }, [hologram]);

  const margin = { top: 10, right: 50, bottom: 36, left: 50 };
  const width = 380;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const N = diffraction.log.length;
  const cellW = innerW / N;
  const cellH = innerH / N;
  const color = scaleSequential(interpolateInferno).domain([0, 1]);

  return (
    <div className="w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full text-light/70"
        onMouseLeave={() => setHover(null)}
      >
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {diffraction.log.map((row, i) =>
            row.map((v, j) => (
              <rect
                key={`${i}-${j}`}
                x={j * cellW}
                y={i * cellH}
                width={cellW + 0.5}
                height={cellH + 0.5}
                fill={color(v / diffraction.max)}
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
              const r = e.currentTarget.getBoundingClientRect();
              const px = e.clientX - r.left;
              const py = e.clientY - r.top;
              const j = Math.max(0, Math.min(N - 1, Math.floor((px / innerW) * N)));
              const i = Math.max(0, Math.min(N - 1, Math.floor((py / innerH) * N)));
              setHover({ i, j, v: diffraction.log[i][j] });
            }}
          />
          {/* axes labels */}
          <text
            x={innerW / 2}
            y={innerH + 24}
            textAnchor="middle"
            fontSize={9}
            fill="currentColor"
            opacity={0.55}
          >
            k_ω
          </text>
          <text
            x={-innerH / 2}
            y={-36}
            transform="rotate(-90)"
            textAnchor="middle"
            fontSize={9}
            fill="currentColor"
            opacity={0.55}
          >
            k_t
          </text>
          <text
            x={innerW / 2}
            y={-2}
            textAnchor="middle"
            fontSize={9}
            fill="currentColor"
            opacity={0.45}
          >
            FFT(|H(ω, t)|)
          </text>
        </g>
      </svg>
      <div className="mt-1 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-light/50">
        <span>{caption}</span>
        <span className="font-mono">
          {hover ? `k=(${hover.i}, ${hover.j})  log|F|=${hover.v.toFixed(2)}` : "hover the pattern"}
        </span>
      </div>
    </div>
  );
}
