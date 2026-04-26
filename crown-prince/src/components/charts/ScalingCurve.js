import { useMemo, useState } from "react";
import { scaleLog, scaleLinear } from "d3-scale";
import { line as d3line } from "d3-shape";

// Log-log query-cost vs N comparison.
// Slider controls the database size; both costs are reported in ops.
export default function ScalingCurve({ height = 320, k = 18, d = 1024, caption }) {
  const [N, setN] = useState(10000);
  const margin = { top: 14, right: 14, bottom: 36, left: 56 };
  const width = 560;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const ns = useMemo(() => {
    const arr = [];
    for (let e = 1; e <= 9; e += 0.1) arr.push(Math.pow(10, e));
    return arr;
  }, []);

  const x = useMemo(() => scaleLog().domain([10, 1e9]).range([0, innerW]), [innerW]);
  const y = useMemo(
    () => scaleLog().domain([1, 1e13]).range([innerH, 0]),
    [innerH]
  );

  const fpLine = d3line()
    .x((n) => x(n))
    .y((n) => y(Math.max(1, n * d)));
  const trieLine = d3line()
    .x((n) => x(n))
    .y((n) => y(k));

  const fpAtN = N * d;
  const trieAtN = k;
  const speedup = fpAtN / trieAtN;

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full text-light/70">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* gridlines (decade) */}
          {[1, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12].map((t) => (
            <line
              key={`y-${t}`}
              x1={0}
              x2={innerW}
              y1={y(t)}
              y2={y(t)}
              stroke="currentColor"
              opacity={0.07}
            />
          ))}
          {[10, 1e3, 1e5, 1e7, 1e9].map((t) => (
            <line
              key={`x-${t}`}
              x1={x(t)}
              x2={x(t)}
              y1={0}
              y2={innerH}
              stroke="currentColor"
              opacity={0.07}
            />
          ))}
          {/* fingerprint line */}
          <path d={fpLine(ns)} fill="none" stroke="#EE6677" strokeWidth={2} />
          <path d={trieLine(ns)} fill="none" stroke="#58E6D9" strokeWidth={2} />
          {/* selected N marker */}
          <line
            x1={x(N)}
            x2={x(N)}
            y1={0}
            y2={innerH}
            stroke="#58E6D9"
            strokeDasharray="3 3"
            opacity={0.6}
          />
          <circle cx={x(N)} cy={y(fpAtN)} r={5} fill="#EE6677" />
          <circle cx={x(N)} cy={y(trieAtN)} r={5} fill="#58E6D9" />
          {/* axes */}
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" opacity={0.3} />
          <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" opacity={0.3} />
          {[10, 1e3, 1e5, 1e7, 1e9].map((t) => (
            <text
              key={`xt-${t}`}
              x={x(t)}
              y={innerH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {fmt(t)}
            </text>
          ))}
          {[1, 1e3, 1e6, 1e9, 1e12].map((t) => (
            <text
              key={`yt-${t}`}
              x={-6}
              y={y(t) + 3}
              textAnchor="end"
              fontSize={9}
              fill="currentColor"
              opacity={0.55}
            >
              {fmt(t)}
            </text>
          ))}
          <text x={innerW / 2} y={innerH + 28} textAnchor="middle" fontSize={10} fill="currentColor" opacity={0.6}>
            database size N
          </text>
          <text x={-innerH / 2} y={-44} transform="rotate(-90)" textAnchor="middle" fontSize={10} fill="currentColor" opacity={0.6}>
            ops per query
          </text>
          {/* legend */}
          <g transform={`translate(${innerW - 160}, 8)`}>
            <rect x={0} y={0} width={160} height={36} fill="#1b1b1b" opacity={0.7} rx={3} />
            <line x1={6} y1={12} x2={20} y2={12} stroke="#EE6677" strokeWidth={2} />
            <text x={26} y={15} fontSize={9} fill="currentColor" opacity={0.7}>
              fingerprint O(Nd)
            </text>
            <line x1={6} y1={26} x2={20} y2={26} stroke="#58E6D9" strokeWidth={2} />
            <text x={26} y={29} fontSize={9} fill="currentColor" opacity={0.7}>
              empty dict O(k)
            </text>
          </g>
        </g>
      </svg>
      <div className="mt-3 flex items-center gap-4">
        <input
          type="range"
          min={1}
          max={9}
          step={0.05}
          value={Math.log10(N)}
          onChange={(e) => setN(Math.pow(10, parseFloat(e.target.value)))}
          className="flex-1 accent-primaryDark"
        />
        <span className="font-mono text-[10px] text-light/60 whitespace-nowrap">
          N = {fmt(N)} · speedup = {fmt(speedup)}×
        </span>
      </div>
      <p className="mt-2 text-[10px] uppercase tracking-[0.18em] text-light/50">{caption}</p>
    </div>
  );
}

function fmt(x) {
  if (x >= 1e9) return (x / 1e9).toFixed(1) + "G";
  if (x >= 1e6) return (x / 1e6).toFixed(1) + "M";
  if (x >= 1e3) return (x / 1e3).toFixed(0) + "k";
  return x.toFixed(0);
}
