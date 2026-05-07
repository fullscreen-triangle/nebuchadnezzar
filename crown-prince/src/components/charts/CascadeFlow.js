/**
 * CascadeFlow — 5-level metabolic cascade visualisation.
 * Shows horizontal flux bars for L1–L5, coloured active/inactive.
 */

const LEVELS = [
  { name: "L1", sub: "Glucose Transport" },
  { name: "L2", sub: "Glycolysis" },
  { name: "L3", sub: "TCA Cycle" },
  { name: "L4", sub: "OxPhos" },
  { name: "L5", sub: "Gene Expression" },
];

const THRESH = 0.10;
const ACTIVE_COLOR  = "#2ECC71";
const INACTIVE_COLOR = "#E74C3C";
const THRESH_COLOR  = "#CCBB44";

export default function CascadeFlow({ proxies, active, D, Itot }) {
  const W = 520, ROW = 46, H = ROW * 5 + 32;
  const LABEL_W = 62, BAR_X = LABEL_W + 4, BAR_MAX = 320;

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {LEVELS.map((lv, i) => {
          const v       = proxies[i];
          const isActive = active[i];
          const color   = isActive ? ACTIVE_COLOR : INACTIVE_COLOR;
          const barW    = Math.max(v * BAR_MAX, isActive ? 4 : 0);
          const y       = i * ROW + 6;

          return (
            <g key={lv.name}>
              {/* Level identifier */}
              <text
                x={2} y={y + 13}
                fontSize={11} fontFamily="monospace"
                fill="#f5f5f5" opacity={0.85}
              >
                {lv.name}
              </text>
              <text
                x={2} y={y + 26}
                fontSize={8}
                fill="#f5f5f5" opacity={0.38}
              >
                {lv.sub}
              </text>

              {/* Background track */}
              <rect
                x={BAR_X} y={y + 5}
                width={BAR_MAX} height={20} rx={3}
                fill="#f5f5f5" opacity={0.04}
              />

              {/* Flux bar */}
              <rect
                x={BAR_X} y={y + 5}
                width={barW} height={20} rx={3}
                fill={color} opacity={0.72}
              />

              {/* Threshold tick */}
              <line
                x1={BAR_X + THRESH * BAR_MAX} y1={y + 2}
                x2={BAR_X + THRESH * BAR_MAX} y2={y + 28}
                stroke={THRESH_COLOR} strokeDasharray="2 2" opacity={0.6}
              />

              {/* Value + status */}
              <text
                x={BAR_X + BAR_MAX + 10} y={y + 18}
                fontSize={9} fontFamily="monospace"
                fill={color} opacity={0.9}
              >
                {v.toFixed(2)} {isActive ? "✓" : "✗"}
              </text>
            </g>
          );
        })}

        {/* Summary row */}
        <text
          x={BAR_X} y={ROW * 5 + 22}
          fontSize={10} fontFamily="monospace"
          fill="#58E6D9" opacity={0.85}
        >
          D = {D.toFixed(2)}  ·  {active.filter(Boolean).length}/5 active  ·  I = {Itot.toFixed(3)} bits
        </text>

        {/* Threshold legend */}
        <line
          x1={BAR_X + THRESH * BAR_MAX} y1={ROW * 5 + 12}
          x2={BAR_X + THRESH * BAR_MAX} y2={ROW * 5 + 26}
          stroke={THRESH_COLOR} strokeDasharray="2 2" opacity={0.5}
        />
        <text
          x={BAR_X + THRESH * BAR_MAX + 3} y={ROW * 5 + 22}
          fontSize={7} fill={THRESH_COLOR} opacity={0.55}
        >
          10% threshold
        </text>
      </svg>
    </div>
  );
}
