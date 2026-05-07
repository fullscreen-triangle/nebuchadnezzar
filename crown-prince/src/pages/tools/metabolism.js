import Head from "next/head";
import dynamic from "next/dynamic";
import { useState, useMemo } from "react";
import {
  computeBMR,
  computeCharges,
  computeDepth,
  computeMirror,
  computeEfficacy,
  prodromeMatch,
  DISEASE_SIGNATURES,
  INTERVENTIONS,
  I_TOT_MAX,
} from "@/lib/metabolism";
import BarChart from "@/components/charts/BarChart";
import Scatter from "@/components/charts/Scatter";
import CascadeFlow from "@/components/charts/CascadeFlow";
import RadialGauge from "@/components/charts/RadialGauge";
import MirrorBand from "@/components/charts/MirrorBand";

const SignatureCube = dynamic(
  () => import("@/components/canvas/SignatureCube"),
  {
    ssr: false,
    loading: () => (
      <div className="aspect-square w-full animate-pulse rounded-md border border-light/10 bg-dark/40" />
    ),
  }
);

// ─── Default profile (30 yr male, 185 cm, 83 kg, healthy params) ─────────────
const DEFAULTS = {
  weight: 83, height: 185, age: 30, sex: "M",
  rmssd: 59, heartRate: 86, remMin: 45, deepMin: 93, sleepEff: 0.78,
  cadence: 166, stepLength: 1.12, peakForce: 1920,
  l1: 0.90, l2: 0.85, l3: 0.80, l4: 0.75, l5: 0.70,
  R: 0.74, activityMetH: 1.5,
};

// ─── Tiny slider component ────────────────────────────────────────────────────
function Slider({ label, value, min, max, step, onChange, unit = "", fmt }) {
  const display = fmt ? fmt(value) : value;
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[11px]">
        <span className="text-light/55">{label}</span>
        <span className="font-mono text-primaryDark">
          {display}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-primaryDark"
      />
    </div>
  );
}

function SexToggle({ value, onChange }) {
  return (
    <div>
      <p className="mb-1 text-[11px] text-light/55">Sex</p>
      <div className="flex gap-2">
        {["M", "F"].map((s) => (
          <button
            key={s}
            onClick={() => onChange(s)}
            className={`rounded border px-3 py-1 text-xs uppercase tracking-[0.15em] transition ${
              value === s
                ? "border-primaryDark/60 bg-primaryDark/15 text-primaryDark"
                : "border-light/15 text-light/50 hover:text-light"
            }`}
          >
            {s === "M" ? "Male" : "Female"}
          </button>
        ))}
      </div>
    </div>
  );
}

function SectionTitle({ n, title }) {
  return (
    <div className="mb-4 mt-12 flex items-baseline gap-3">
      <span className="font-mono text-[11px] text-primaryDark">{n}</span>
      <h2 className="text-lg font-medium tracking-tight">{title}</h2>
      <div className="ml-3 h-px flex-1 bg-light/10" />
    </div>
  );
}

function Panel({ title, children }) {
  return (
    <div className="panel">
      <p className="panel-title">{title}</p>
      {children}
    </div>
  );
}

function ScalarRow({ label, value, unit = "", color = "#58E6D9", sub }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-light/55">{label}</span>
        <span className="font-mono" style={{ color }}>
          {value}
          {unit && <span className="ml-1 text-[10px] text-light/40">{unit}</span>}
        </span>
      </div>
      {sub && <p className="text-[10px] text-light/35">{sub}</p>}
    </div>
  );
}

function LevelProxy({ label, value, onChange }) {
  const active = value > 0.10;
  return (
    <div className="flex items-center gap-2">
      <span
        className="w-5 shrink-0 text-center font-mono text-[10px]"
        style={{ color: active ? "#2ECC71" : "#E74C3C" }}
      >
        {label}
      </span>
      <input
        type="range" min={0} max={1} step={0.01} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 accent-primaryDark"
      />
      <span className="w-10 text-right font-mono text-[10px] text-light/50">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function MetabolismEngine() {
  const [p, setP] = useState(DEFAULTS);
  const upd = (k) => (v) => setP((prev) => ({ ...prev, [k]: v }));

  // Computed values (all reactive)
  const bmr = useMemo(
    () => computeBMR(p.weight, p.height, p.age, p.sex),
    [p.weight, p.height, p.age, p.sex]
  );

  const charges = useMemo(
    () =>
      computeCharges({
        bmr,
        rmssd: p.rmssd,
        heartRate: p.heartRate,
        cadence: p.cadence,
        stepLength: p.stepLength,
        peakForce: p.peakForce,
      }),
    [bmr, p.rmssd, p.heartRate, p.cadence, p.stepLength, p.peakForce]
  );

  const depth = useMemo(
    () => computeDepth([p.l1, p.l2, p.l3, p.l4, p.l5]),
    [p.l1, p.l2, p.l3, p.l4, p.l5]
  );

  const mirror = useMemo(
    () =>
      computeMirror({
        activityMetH: p.activityMetH,
        deepMin: p.deepMin,
        remMin: p.remMin,
        sleepEff: p.sleepEff,
      }),
    [p.activityMetH, p.deepMin, p.remMin, p.sleepEff]
  );

  const ItotN = useMemo(
    () => Math.min(1, depth.Itot / I_TOT_MAX),
    [depth.Itot]
  );

  const matches = useMemo(
    () => prodromeMatch(depth.D, p.R, ItotN),
    [depth.D, p.R, ItotN]
  );

  const queryPoint = { D: depth.D, R: p.R, ItotN };

  // Chart data
  const chargeData = [
    { label: "Q_b", value: charges.Qb,  color: "#4477AA", sub: `baseline · ${charges.Pcog} W` },
    { label: "Q_m", value: charges.Qm,  color: "#2ECC71", sub: `motor · ${charges.Ploc} W` },
    { label: "Q_p", value: charges.Qp,  color: "#CCBB44", sub: `perception · f=${charges.fPerc}` },
    { label: "Q_t", value: charges.Qt,  color: "#58E6D9", sub: `thought · P_cog=${charges.Pcog} W` },
    { label: "Q_d", value: charges.Qd,  color: "#9B59B6", sub: `dream · 0.95×P_cog` },
  ];

  const etaData = INTERVENTIONS.map((d) => ({
    label: d.name,
    value: parseFloat(d.eta.toFixed(3)),
    color: d.color,
    sub: `D: ${d.Dpre}→${d.Dpost}`,
  }));

  const scatterData = INTERVENTIONS.map((d) => ({
    x: d.Dpre,
    y: d.Dpost,
    label: d.name,
    color: d.color,
  }));

  const signatureScatter = DISEASE_SIGNATURES.map((sig) => ({
    x: sig.D,
    y: sig.R,
    label: sig.name,
    color: sig.color,
  })).concat([{ x: depth.D, y: p.R, label: "You", color: "#58E6D9" }]);

  return (
    <>
      <Head>
        <title>Partitioned Metabolism Engine — Crown Prince</title>
      </Head>

      <section className="mx-auto max-w-7xl px-6 py-10 md:px-10">

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <header className="mb-8">
          <p className="text-[11px] uppercase tracking-[0.3em] text-primaryDark">
            Diagnostic-first · Charge · Depth · Coherence · six primitives
          </p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">
            Partitioned Metabolism Engine
          </h1>
          <p className="mt-3 max-w-3xl text-xs leading-relaxed text-light/60">
            Every physiological observable derived from one axiom — bounded phase
            space admitting hierarchical partitioning. Four orthogonal charge
            rates{" "}
            <span className="mono accent">Q_b Q_m Q_p Q_t</span> plus REM
            charge{" "}
            <span className="mono accent">Q_d</span>, five-level metabolic
            depth{" "}
            <span className="mono accent">D ∈ [0,1]</span>, and Kuramoto
            coherence{" "}
            <span className="mono accent">R</span> — twelve scalars, no fitted
            parameters. The framework measures. It does not prescribe.
          </p>
        </header>

        {/* ── Profile inputs ─────────────────────────────────────────────── */}
        <div className="panel mb-2">
          <p className="panel-title">Profile</p>
          <div className="grid gap-x-8 gap-y-4 sm:grid-cols-2 lg:grid-cols-4">
            <Slider label="Weight" value={p.weight} min={40} max={150} step={1}
              onChange={upd("weight")} unit=" kg" />
            <Slider label="Height" value={p.height} min={140} max={215} step={1}
              onChange={upd("height")} unit=" cm" />
            <Slider label="Age" value={p.age} min={18} max={80} step={1}
              onChange={upd("age")} unit=" yr" />
            <SexToggle value={p.sex} onChange={upd("sex")} />
          </div>
          <div className="mt-2 text-right font-mono text-[11px] text-primaryDark/70">
            BMR = {bmr} W · P_brain = {charges.Pbrain} W · P_cog = {charges.Pcog} W
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          {/* Sleep panel */}
          <div className="panel">
            <p className="panel-title">Sleep</p>
            <div className="space-y-3">
              <Slider label="RMSSD" value={p.rmssd} min={10} max={120} step={1}
                onChange={upd("rmssd")} unit=" ms" />
              <Slider label="Heart rate" value={p.heartRate} min={40} max={120} step={1}
                onChange={upd("heartRate")} unit=" bpm" />
              <Slider label="REM duration" value={p.remMin} min={10} max={130} step={5}
                onChange={upd("remMin")} unit=" min" />
              <Slider label="Deep duration" value={p.deepMin} min={20} max={200} step={5}
                onChange={upd("deepMin")} unit=" min" />
              <Slider label="Sleep efficiency" value={p.sleepEff} min={0.50} max={1.00} step={0.01}
                onChange={upd("sleepEff")} fmt={(v) => (v * 100).toFixed(0)} unit="%" />
            </div>
          </div>

          {/* Activity panel */}
          <div className="panel">
            <p className="panel-title">Activity (running bout)</p>
            <div className="space-y-3">
              <Slider label="Cadence" value={p.cadence} min={100} max={210} step={2}
                onChange={upd("cadence")} unit=" spm" />
              <Slider label="Step length" value={p.stepLength} min={0.5} max={1.6} step={0.01}
                onChange={upd("stepLength")} unit=" m" />
              <Slider label="Peak ground force" value={p.peakForce} min={600} max={3200} step={50}
                onChange={upd("peakForce")} unit=" N" />
              <Slider label="Activity MET-h" value={p.activityMetH} min={0.2} max={6} step={0.1}
                onChange={upd("activityMetH")} fmt={(v) => v.toFixed(1)} unit=" MET·h" />
            </div>
          </div>

          {/* Depth proxies + R */}
          <div className="panel">
            <p className="panel-title">Depth proxies  ·  Coherence</p>
            <div className="space-y-2">
              {[
                { key: "l1", label: "L1", hint: "CGM swing" },
                { key: "l2", label: "L2", hint: "post-prandial HR" },
                { key: "l3", label: "L3", hint: "RMSSD quality" },
                { key: "l4", label: "L4", hint: "VO₂ estimate" },
                { key: "l5", label: "L5", hint: "circadian coherence" },
              ].map(({ key, label, hint }) => (
                <div key={key}>
                  <LevelProxy
                    label={label}
                    value={p[key]}
                    onChange={upd(key)}
                  />
                  <p className="ml-7 text-[9px] text-light/30">{hint}</p>
                </div>
              ))}
            </div>
            <div className="mt-4 border-t border-light/8 pt-3">
              <Slider label="Kuramoto  R" value={p.R} min={0.10} max={0.95} step={0.01}
                onChange={upd("R")} fmt={(v) => v.toFixed(2)} />
            </div>
          </div>
        </div>

        {/* ══════════════════════════════════════════════════════════════════
            Section 01 — Charge Decomposition
        ═════════════════════════════════════════════════════════════════════ */}
        <SectionTitle n="01" title="Charge decomposition" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Five component charges  ·  Q = √(2 C P)  mC/s">
            <BarChart
              data={chargeData}
              yLabel="mC/s"
              caption="hover a bar · formula Q = sqrt(2 C P Delta_t=1s)"
            />
          </Panel>
          <Panel title={`Dream–thought ratio  ·  Q_d/Q_t = ${charges.ratio}  (target ${charges.ratioTarget})`}>
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <div className="mb-1 flex justify-between text-[11px] text-light/55">
                    <span>Q_d / Q_t</span>
                    <span className="font-mono text-primaryDark">{charges.ratio}</span>
                  </div>
                  <div className="relative h-3 w-full overflow-hidden rounded-full bg-light/8">
                    <div
                      className="absolute left-0 top-0 h-full rounded-full bg-primaryDark/70"
                      style={{ width: `${charges.ratio * 100}%` }}
                    />
                    {/* Target marker */}
                    <div
                      className="absolute top-0 h-full w-0.5 bg-[#CCBB44]/80"
                      style={{ left: `${charges.ratioTarget * 100}%` }}
                    />
                  </div>
                  <div className="mt-1 flex justify-between text-[9px] text-light/30">
                    <span>0</span>
                    <span style={{ marginLeft: `${charges.ratioTarget * 100 - 2}%` }} className="text-[#CCBB44]/60">
                      √0.95
                    </span>
                    <span>1</span>
                  </div>
                </div>
              </div>
              <div className="space-y-2 border-t border-light/8 pt-3">
                <ScalarRow label="Cardiac coupling  κ" value={charges.kappa} />
                <ScalarRow label="Perceptual fraction  f_perc" value={charges.fPerc} />
                <ScalarRow label="Locomotion power  P_loc" value={charges.Ploc} unit="W" />
                <ScalarRow label="Q_d / Q_t  (measured)" value={charges.ratio}
                  color={Math.abs(charges.ratio - charges.ratioTarget) < 0.05 ? "#2ECC71" : "#E74C3C"} />
                <ScalarRow label="Prediction  √0.95" value={charges.ratioTarget} color="#CCBB44" />
              </div>
            </div>
          </Panel>
        </div>

        {/* ══════════════════════════════════════════════════════════════════
            Section 02 — Hierarchical Depth
        ═════════════════════════════════════════════════════════════════════ */}
        <SectionTitle n="02" title="Hierarchical metabolic depth" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Five-level cascade  ·  threshold = 10% of L1">
            <CascadeFlow
              proxies={depth.proxies}
              active={depth.active}
              D={depth.D}
              Itot={depth.Itot}
            />
          </Panel>
          <Panel title="D across disease states  ·  prodrome threshold = 0.65">
            <BarChart
              data={[
                { label: "Healthy",   value: 1.0, color: "#2ECC71", sub: "all five levels" },
                { label: "T2D",       value: 0.6, color: "#F39C12", sub: "L1–L3 only" },
                { label: "Syndrome",  value: 0.4, color: "#E74C3C", sub: "L1–L2 only" },
                { label: "Post-Met.", value: parseFloat(computeEfficacy(0.4,0.8) > 0 ? (0.4+(1-0.4)*computeEfficacy(0.4,0.8)).toFixed(2) : "0.40"), color: "#3498DB", sub: "metformin L3–L4" },
                { label: "You",       value: depth.D, color: "#58E6D9", sub: `${depth.active.filter(Boolean).length}/5 active` },
              ]}
              threshold={0.65}
              yLabel="D"
              caption="D = fraction of active metabolic levels · red dashed = prodrome gate"
            />
          </Panel>
        </div>

        {/* ══════════════════════════════════════════════════════════════════
            Section 03 — Physiological Coherence
        ═════════════════════════════════════════════════════════════════════ */}
        <SectionTitle n="03" title="Physiological coherence" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Kuramoto order parameter  R">
            <div className="flex flex-col items-center gap-4">
              <RadialGauge R={p.R} />
              <div className="w-full space-y-2 border-t border-light/8 pt-3">
                {[
                  { label: "Awake focused",  R: 0.74, color: "#F39C12" },
                  { label: "Deep sleep",     R: 0.85, color: "#4477AA" },
                  { label: "Metab. syndrome",R: 0.52, color: "#E74C3C" },
                  { label: "Major depression",R: 0.25, color: "#9B59B6" },
                ].map((ref) => (
                  <div key={ref.label} className="flex items-center gap-2">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-light/8">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${ref.R * 100}%`, backgroundColor: ref.color, opacity: 0.65 }}
                      />
                    </div>
                    <span className="w-28 text-right text-[10px] text-light/45">{ref.label}</span>
                    <span className="w-8 font-mono text-[10px]" style={{ color: ref.color }}>{ref.R}</span>
                  </div>
                ))}
              </div>
            </div>
          </Panel>
          <Panel title="Disease signature space  (D, R, I_norm)  ·  3D orbit">
            <SignatureCube query={queryPoint} />
            <p className="mt-2 text-[10px] text-light/35">
              Red axis = D · green axis = R · blue axis = I_norm · teal = you · drag to orbit
            </p>
          </Panel>
        </div>

        {/* ══════════════════════════════════════════════════════════════════
            Section 04 — Mirror Law
        ═════════════════════════════════════════════════════════════════════ */}
        <SectionTitle n="04" title="Activity–sleep mirror law" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title={`Mirror coefficient  μ = ${mirror.mu}  ·  ${mirror.stable ? "stable" : "unstable"}`}>
            <MirrorBand
              mu={mirror.mu}
              stable={mirror.stable}
              caption="charge subtraction error ≤ 15% inside [0.8, 1.2]"
            />
          </Panel>
          <Panel title="Mirror readout">
            <div className="space-y-3">
              <ScalarRow
                label="μ"
                value={mirror.mu}
                color={mirror.stable ? "#2ECC71" : "#E74C3C"}
                sub="C_night / E_day  —  cleanup-to-error ratio"
              />
              <ScalarRow label="Stable band" value="[0.8, 1.2]" color="#2ECC71" />
              <ScalarRow
                label="Charge error"
                value={mirror.error}
                color={mirror.stable ? "#2ECC71" : "#E74C3C"}
                sub={mirror.stable ? "inside stable band — full confidence" : "outside band — Qt estimate degraded"}
              />
              <div className="border-t border-light/8 pt-3 text-[11px] text-light/50">
                Deep {p.deepMin} min · REM {p.remMin} min ·
                sleep eff {(p.sleepEff * 100).toFixed(0)}% ·
                activity {p.activityMetH} MET·h
              </div>
            </div>
          </Panel>
        </div>

        {/* ══════════════════════════════════════════════════════════════════
            Section 05 — Disease Signatures (Prodrome)
        ═════════════════════════════════════════════════════════════════════ */}
        <SectionTitle n="05" title="Disease signatures  ·  Prodrome" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="(D, R) signature plane  ·  your position in teal">
            <Scatter
              data={signatureScatter}
              xLabel="Hierarchical Depth  D"
              yLabel="Coherence  R"
              height={300}
              caption="Mahalanobis proximity to named disease prototypes"
            />
          </Panel>
          <Panel title="Nearest signature matches">
            <div className="space-y-3">
              {matches.slice(0, 4).map((m, i) => (
                <div key={m.name} className="flex items-start gap-3">
                  <span
                    className={`mt-1 h-2 w-2 shrink-0 rounded-full`}
                    style={{ backgroundColor: m.color }}
                  />
                  <div className="flex-1">
                    <div className="flex items-baseline justify-between text-xs">
                      <span style={{ color: m.color }}>{m.name}</span>
                      <span className="font-mono text-[10px] text-light/40">
                        dist = {m.dist}
                      </span>
                    </div>
                    <div className="mt-1 h-1 overflow-hidden rounded-full bg-light/8">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.max(0, (1 - m.dist / 1.2)) * 100}%`,
                          backgroundColor: m.color,
                          opacity: 0.65,
                        }}
                      />
                    </div>
                    <p className="mt-0.5 text-[9px] text-light/35">
                      D={m.D} · R={m.R} · I={m.ItotN}
                    </p>
                  </div>
                </div>
              ))}
              <div className="border-t border-light/8 pt-2 text-[10px] text-light/35">
                Your point: D={depth.D} · R={p.R.toFixed(2)} · I_norm={ItotN.toFixed(3)}
              </div>
            </div>
          </Panel>
        </div>

        {/* ══════════════════════════════════════════════════════════════════
            Section 06 — Intervention Efficacy
        ═════════════════════════════════════════════════════════════════════ */}
        <SectionTitle n="06" title="Intervention efficacy  η" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="η_drug = (D_post − D_pre) / (1 − D_pre)  ·  baseline D = 0.4">
            <BarChart
              data={etaData}
              threshold={0}
              yLabel="η"
              caption="GLP-1 η = 0 (input-throttling) · Metformin η = 0.67 (L3–L4 coupling)"
            />
          </Panel>
          <Panel title="D_pre vs D_post  ·  diagonal = no change  (η = 0)">
            <Scatter
              data={scatterData}
              xLabel="D_pre"
              yLabel="D_post"
              diagonal={true}
              height={300}
              caption="points above the diagonal restore hierarchical depth"
            />
          </Panel>
        </div>

      </section>
    </>
  );
}
