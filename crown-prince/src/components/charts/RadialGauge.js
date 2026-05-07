/**
 * RadialGauge — Kuramoto order-parameter arc gauge.
 * 270° sweep; zones: red [0,0.4], orange [0.4,0.6], yellow [0.6,0.7], green [0.7,1].
 */

// 0° = 12-o'clock; positive = clockwise
const toXY = (cx, cy, rr, deg) => {
  const rad = ((deg - 90) * Math.PI) / 180;
  return [cx + rr * Math.cos(rad), cy + rr * Math.sin(rad)];
};

function arcPath(cx, cy, rr, a1, a2) {
  const [x1, y1] = toXY(cx, cy, rr, a1);
  const [x2, y2] = toXY(cx, cy, rr, a2);
  const sweep = ((a2 - a1) + 360) % 360;
  const large  = sweep > 180 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${rr} ${rr} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

const START  = 135;   // degrees (bottom-left)
const SWEEP  = 270;   // total arc

const ZONES = [
  { from: 0,   to: 0.4, color: "#E74C3C" },
  { from: 0.4, to: 0.6, color: "#F39C12" },
  { from: 0.6, to: 0.7, color: "#CCBB44" },
  { from: 0.7, to: 1.0, color: "#2ECC71" },
];

const TICK_VALUES = [0, 0.25, 0.5, 0.75, 1.0];

function stateLabel(R) {
  if (R < 0.30) return "desynchronised";
  if (R < 0.50) return "fragmented";
  if (R < 0.65) return "partial sync";
  if (R < 0.75) return "awake focused";
  if (R < 0.80) return "deep focus";
  return "deep sleep";
}

export default function RadialGauge({ R, label = "Kuramoto  R" }) {
  const CX = 120, CY = 115, OR = 90, IR = 72;
  const needleDeg  = START + R * SWEEP;
  const [nx, ny]   = toXY(CX, CY, OR - 12, needleDeg);

  return (
    <div className="w-full max-w-[260px] mx-auto">
      <svg viewBox="0 0 240 168" className="w-full">
        {/* Background track */}
        <path
          d={arcPath(CX, CY, OR, START, START + SWEEP)}
          fill="none" stroke="#f5f5f5" strokeWidth={14} opacity={0.06}
          strokeLinecap="round"
        />

        {/* Coloured zones */}
        {ZONES.map((z, i) => (
          <path
            key={i}
            d={arcPath(CX, CY, OR, START + z.from * SWEEP, START + z.to * SWEEP)}
            fill="none" stroke={z.color} strokeWidth={14} opacity={0.55}
            strokeLinecap="round"
          />
        ))}

        {/* Active-value overlay */}
        <path
          d={arcPath(CX, CY, IR, START, needleDeg)}
          fill="none" stroke="#f5f5f5" strokeWidth={3} opacity={0.70}
          strokeLinecap="round"
        />

        {/* Tick labels */}
        {TICK_VALUES.map((v) => {
          const [tx, ty] = toXY(CX, CY, OR + 14, START + v * SWEEP);
          return (
            <text
              key={v}
              x={tx.toFixed(1)} y={(ty + 3).toFixed(1)}
              textAnchor="middle" fontSize={7}
              fill="#f5f5f5" opacity={0.30}
              fontFamily="monospace"
            >
              {v}
            </text>
          );
        })}

        {/* Needle */}
        <line
          x1={CX} y1={CY} x2={nx.toFixed(2)} y2={ny.toFixed(2)}
          stroke="#f5f5f5" strokeWidth={2} opacity={0.80}
          strokeLinecap="round"
        />
        <circle cx={CX} cy={CY} r={5} fill="#58E6D9" />

        {/* Centre readout */}
        <text
          x={CX} y={CY + 18}
          textAnchor="middle" fontSize={26}
          fontFamily="monospace" fill="#58E6D9" fontWeight="bold"
        >
          {R.toFixed(2)}
        </text>
        <text
          x={CX} y={CY + 34}
          textAnchor="middle" fontSize={8}
          fill="#f5f5f5" opacity={0.45}
        >
          {stateLabel(R)}
        </text>

        {/* Bottom label */}
        <text
          x={CX} y={158}
          textAnchor="middle" fontSize={8}
          fill="#f5f5f5" opacity={0.35}
          fontFamily="monospace"
          style={{ letterSpacing: "0.18em", textTransform: "uppercase" }}
        >
          {label}
        </text>
      </svg>
    </div>
  );
}
