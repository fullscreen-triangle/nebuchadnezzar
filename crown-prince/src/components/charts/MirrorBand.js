/**
 * MirrorBand — activity-sleep mirror coefficient stability diagram.
 * Shows the [0.8, 1.2] stable band and the error magnitude curve.
 */
import { scaleLinear } from "d3-scale";

const MU_MAX   = 2.6;
const ERR_MAX  = 1.5;
const N_CURVE  = 260;

export default function MirrorBand({ mu, stable, caption }) {
  const margin = { top: 24, right: 16, bottom: 36, left: 38 };
  const W = 500, H = 170;
  const IW = W - margin.left - margin.right;
  const IH = H - margin.top - margin.bottom;

  const xs = scaleLinear().domain([0, MU_MAX]).range([0, IW]);
  const ys = scaleLinear().domain([0, ERR_MAX]).range([IH, 0]);

  const errFn = (v) => (v >= 0.8 && v <= 1.2 ? 0 : Math.abs(v - 1));

  // Build SVG polyline for the error curve
  const pts = Array.from({ length: N_CURVE }, (_, i) => {
    const v = (i / (N_CURVE - 1)) * MU_MAX;
    return `${xs(v).toFixed(1)},${ys(errFn(v)).toFixed(1)}`;
  }).join(" ");

  const muClamp = Math.min(mu, MU_MAX);
  const muX     = xs(muClamp);
  const color   = stable ? "#2ECC71" : "#E74C3C";

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full text-light/70">
        <g transform={`translate(${margin.left},${margin.top})`}>

          {/* Stable band fill */}
          <rect
            x={xs(0.8)} y={0}
            width={xs(1.2) - xs(0.8)} height={IH}
            fill="#2ECC71" opacity={0.09}
          />

          {/* Band edges */}
          {[0.8, 1.2].map((v) => (
            <line
              key={v}
              x1={xs(v)} y1={0} x2={xs(v)} y2={IH}
              stroke="#2ECC71" strokeDasharray="3 3" opacity={0.35}
            />
          ))}

          {/* Identity (μ = 1) */}
          <line
            x1={xs(1)} y1={0} x2={xs(1)} y2={IH}
            stroke="#f5f5f5" opacity={0.10}
          />

          {/* Error curve */}
          <polyline
            points={pts}
            fill="none" stroke="#f5f5f5" strokeWidth={1.5} opacity={0.30}
          />

          {/* Stable segment of curve in green */}
          <polyline
            points={Array.from({ length: 40 }, (_, i) => {
              const v = 0.8 + (i / 39) * 0.4;
              return `${xs(v).toFixed(1)},${ys(0).toFixed(1)}`;
            }).join(" ")}
            fill="none" stroke="#2ECC71" strokeWidth={2} opacity={0.55}
          />

          {/* Current μ line */}
          <line
            x1={muX} y1={0} x2={muX} y2={IH}
            stroke={color} strokeWidth={2} opacity={0.9}
          />
          <circle cx={muX} cy={ys(errFn(muClamp))} r={4} fill={color} opacity={0.9} />

          {/* μ readout */}
          <text
            x={Math.min(muX + 5, IW - 90)} y={14}
            fontSize={10} fontFamily="monospace" fill={color} opacity={0.9}
          >
            μ = {mu.toFixed(3)}  {stable ? "stable" : "unstable"}
          </text>

          {/* Band label */}
          <text
            x={(xs(0.8) + xs(1.2)) / 2} y={-10}
            textAnchor="middle" fontSize={8}
            fill="#2ECC71" opacity={0.55}
            style={{ letterSpacing: "0.15em" }}
          >
            STABLE  [0.8, 1.2]
          </text>

          {/* Axes */}
          <line x1={0} y1={IH} x2={IW} y2={IH} stroke="currentColor" opacity={0.25} />
          <line x1={0} y1={0}  x2={0}  y2={IH} stroke="currentColor" opacity={0.25} />

          {[0, 0.5, 1.0, 1.5, 2.0, 2.5].map((t) => (
            <g key={t}>
              <text
                x={xs(t)} y={IH + 14}
                textAnchor="middle" fontSize={8}
                fill="currentColor" opacity={0.45}
              >
                {t}
              </text>
            </g>
          ))}
          {[0, 0.5, 1.0, 1.5].map((t) => (
            <text
              key={t} x={-4} y={ys(t) + 3}
              textAnchor="end" fontSize={8}
              fill="currentColor" opacity={0.45}
            >
              {t}
            </text>
          ))}

          <text
            x={IW / 2} y={IH + 28}
            textAnchor="middle" fontSize={9}
            fill="currentColor" opacity={0.45}
          >
            Mirror coefficient μ
          </text>
          <text
            x={-IH / 2} y={-26}
            transform="rotate(-90)"
            textAnchor="middle" fontSize={9}
            fill="currentColor" opacity={0.45}
          >
            Error magnitude
          </text>

        </g>
      </svg>
      <div className="mt-1 text-[10px] uppercase tracking-[0.15em] text-light/40 font-mono">
        {caption}
      </div>
    </div>
  );
}
