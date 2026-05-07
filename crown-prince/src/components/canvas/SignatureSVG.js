/**
 * SignatureSVG — 2-D fallback for the (D, R) disease-signature space.
 * Used when WebGL is unavailable.
 */
import { scaleLinear } from "d3-scale";
import { DISEASE_SIGNATURES } from "@/lib/metabolism";

export default function SignatureSVG({ query }) {
  const margin = { top: 20, right: 20, bottom: 40, left: 44 };
  const W = 340, H = 300;
  const IW = W - margin.left - margin.right;
  const IH = H - margin.top - margin.bottom;

  const xs = scaleLinear().domain([0, 1]).range([0, IW]);
  const ys = scaleLinear().domain([0, 1]).range([IH, 0]);

  return (
    <div className="w-full rounded-md border border-light/10 bg-dark/40 p-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full text-light/60">
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* Grid */}
          {[0.2, 0.4, 0.6, 0.8].map((t) => (
            <g key={t}>
              <line x1={xs(t)} y1={0} x2={xs(t)} y2={IH} stroke="currentColor" opacity={0.07} />
              <line x1={0} y1={ys(t)} x2={IW} y2={ys(t)} stroke="currentColor" opacity={0.07} />
            </g>
          ))}

          {/* Disease state points */}
          {DISEASE_SIGNATURES.map((sig) => (
            <g key={sig.name}>
              <circle
                cx={xs(sig.D)} cy={ys(sig.R)} r={7}
                fill={sig.color} opacity={0.70}
              />
              <text
                x={xs(sig.D) + 9} y={ys(sig.R) + 4}
                fontSize={7.5} fill={sig.color} opacity={0.80}
              >
                {sig.name}
              </text>
            </g>
          ))}

          {/* User point */}
          {query && (
            <circle
              cx={xs(query.D)} cy={ys(query.R)} r={8}
              fill="#58E6D9" opacity={0.90}
            />
          )}

          {/* Axes */}
          <line x1={0} y1={IH} x2={IW} y2={IH} stroke="currentColor" opacity={0.25} />
          <line x1={0} y1={0}  x2={0}  y2={IH} stroke="currentColor" opacity={0.25} />
          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((t) => (
            <g key={t}>
              <text x={xs(t)} y={IH + 13} textAnchor="middle" fontSize={7} fill="currentColor" opacity={0.45}>{t}</text>
              <text x={-4}   y={ys(t) + 3} textAnchor="end"    fontSize={7} fill="currentColor" opacity={0.45}>{t}</text>
            </g>
          ))}
          <text x={IW / 2} y={IH + 28} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.45}>
            Hierarchical Depth D
          </text>
          <text
            x={-IH / 2} y={-30} transform="rotate(-90)"
            textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.45}
          >
            Coherence R
          </text>
        </g>
      </svg>
    </div>
  );
}
