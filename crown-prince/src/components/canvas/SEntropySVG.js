// SVG fallback for SEntropyCube. Static isometric projection of [0,1]^3
// with the 39 reference compounds and the query point.
import { COMPOUNDS, TYPE_COLOR } from "@/lib/compounds";
import { sEntropy } from "@/lib/sentropy";

const REFS = COMPOUNDS.map((c) => ({
  name: c.name,
  type: c.type,
  s: sEntropy(c.omega, c.bRot),
}));

// Isometric projection: (x, y, z) -> (x - z*cos(30), y + z*sin(30) - x*sin(30))
// scaled into a 320x320 viewport.
const SIZE = 320;
const PAD = 30;
function project(s) {
  const cos30 = Math.cos(Math.PI / 6);
  const sin30 = Math.sin(Math.PI / 6);
  const u = (s.Sk - s.Se * cos30) * 0.5 + 0.5;
  const v = 1 - (s.St + s.Se * sin30 - s.Sk * sin30) * 0.5 - 0.25;
  return { x: PAD + u * (SIZE - 2 * PAD), y: PAD + v * (SIZE - 2 * PAD) };
}

const CORNERS = [
  { Sk: 0, St: 0, Se: 0 },
  { Sk: 1, St: 0, Se: 0 },
  { Sk: 1, St: 1, Se: 0 },
  { Sk: 0, St: 1, Se: 0 },
  { Sk: 0, St: 0, Se: 1 },
  { Sk: 1, St: 0, Se: 1 },
  { Sk: 1, St: 1, Se: 1 },
  { Sk: 0, St: 1, Se: 1 },
];
const EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

export default function SEntropySVG({ query }) {
  const proj = CORNERS.map(project);
  const qp = query ? project(query) : null;
  return (
    <div className="aspect-square w-full overflow-hidden rounded-md border border-light/10 bg-dark/40">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full">
        {/* cube edges */}
        {EDGES.map(([a, b], i) => (
          <line
            key={i}
            x1={proj[a].x}
            y1={proj[a].y}
            x2={proj[b].x}
            y2={proj[b].y}
            stroke="#4a4a4a"
            strokeWidth={0.8}
          />
        ))}
        {/* reference cloud */}
        {REFS.map((r) => {
          const p = project(r.s);
          return (
            <circle
              key={r.name}
              cx={p.x}
              cy={p.y}
              r={2.5}
              fill={TYPE_COLOR[r.type]}
              opacity={0.7}
            />
          );
        })}
        {/* query */}
        {qp && (
          <g>
            <circle cx={qp.x} cy={qp.y} r={6} fill="#58E6D9" />
            <circle
              cx={qp.x}
              cy={qp.y}
              r={11}
              fill="none"
              stroke="#58E6D9"
              strokeWidth={1}
              opacity={0.5}
            />
          </g>
        )}
        {/* axis ticks at corners */}
        <text x={proj[1].x + 4} y={proj[1].y + 4} fontSize={8} fill="#EE6677">
          S_k
        </text>
        <text x={proj[3].x - 18} y={proj[3].y + 4} fontSize={8} fill="#228833">
          S_t
        </text>
        <text x={proj[4].x + 4} y={proj[4].y - 4} fontSize={8} fill="#4477AA">
          S_e
        </text>
      </svg>
    </div>
  );
}
