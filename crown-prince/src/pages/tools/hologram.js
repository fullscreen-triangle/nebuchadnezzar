import Head from "next/head";
import dynamic from "next/dynamic";
import { useState, useMemo, useEffect } from "react";
import {
  runKuramoto,
  classifyRegime,
  criticalCoupling,
  meanFieldRstar,
  trajectoryCount,
} from "@/lib/kuramoto";
import {
  buildPartitionField,
  fieldToSEntropy,
  organelleFrequencies,
  CELL_CONFIGS,
} from "@/lib/phasefield";
import SpectralOverlay from "@/components/charts/SpectralOverlay";
import BarChart from "@/components/charts/BarChart";

const VolumeRaymarch = dynamic(
  () => import("@/components/canvas/VolumeRaymarch"),
  {
    ssr: false,
    loading: () => (
      <div className="aspect-square w-full animate-pulse rounded-md border border-light/10 bg-dark/40" />
    ),
  }
);

// ── Cell type labels ──────────────────────────────────────────────────────────
const CELL_LABELS = {
  hepatocyte:    "Hepatocyte",
  neuron:        "Neuron",
  cardiomyocyte: "Cardiomyocyte",
  cancer:        "Tumour (HeLa)",
  macrophage:    "Macrophage",
  beta_cell:     "β-cell",
};

// Channel labels for the volume display
const CHANNELS = [
  { id: 0, label: "Composite",  sub: "n + ℓ + m + s" },
  { id: 1, label: "n(r)",       sub: "refractive index" },
  { id: 2, label: "ℓ(r)",       sub: "phase gradient" },
  { id: 3, label: "m(r)",       sub: "orientation" },
  { id: 4, label: "s(r)",       sub: "chirality" },
];

// ── Kuramoto coupling presets ─────────────────────────────────────────────────
const KAPPA_LABELS = ["0.5 Kc", "1.0 Kc", "1.5 Kc", "2.5 Kc", "5.0 Kc"];
const KAPPA_MULTS  = [0.5, 1.0, 1.5, 2.5, 5.0];

export default function HologramTool() {
  const [cellId,   setCellId]   = useState("hepatocyte");
  const [channel,  setChannel]  = useState(0);
  const [kappaMul, setKappaMul] = useState(3);   // index into KAPPA_MULTS

  // ── Partition field (expensive — only on cell type change) ────────────────
  const [volume, setVolume] = useState(null);
  useEffect(() => {
    setVolume(null);
    const timer = setTimeout(() => {
      setVolume(buildPartitionField(cellId, 64));
    }, 40);
    return () => clearTimeout(timer);
  }, [cellId]);

  // ── Field-derived S-entropy ───────────────────────────────────────────────
  const fieldCoords = useMemo(
    () => (volume ? fieldToSEntropy(volume.data, volume.size) : null),
    [volume]
  );

  // ── Kuramoto simulation ───────────────────────────────────────────────────
  const kuramoto = useMemo(() => {
    const placed = volume?.placed ?? [];
    const n = Math.max(6, placed.length);
    const freqs = placed.length > 0 ? organelleFrequencies(placed) : null;
    const seed = cellId.charCodeAt(0) * 137;

    // Compute Kc from actual organelle frequencies if available
    const { Kc: baseKc } = freqs
      ? criticalCoupling(freqs)
      : criticalCoupling(Array.from({ length: n }, (_, i) => 1 + (i - n / 2) * 0.1));

    const K = baseKc * KAPPA_MULTS[kappaMul];
    return { ...runKuramoto(n, K, 140, 0.05, seed), K, baseKc };
  }, [volume, cellId, kappaMul]);

  const regime = useMemo(() => classifyRegime(kuramoto.finalRens), [kuramoto]);

  // ── Trajectory count table ────────────────────────────────────────────────
  const D = volume?.organelleCount ?? 0;
  const trajCounts = useMemo(() => {
    if (!D) return [];
    return [3, 5, 8, 10].map((m) => ({
      label: `m=${m}`,
      value: parseFloat(Math.log10(trajectoryCount(m, D)).toFixed(1)),
      color: "#4477AA",
      sub: `log₁₀ T(${m},${D})`,
    }));
  }, [D]);

  // ── Regime row colors ─────────────────────────────────────────────────────
  const regimeRows = [
    { Rens: "< 0.30", label: "Turbulent",            cost: "O(R⁻²)",         color: "#EE6677" },
    { Rens: "0.30–0.50", label: "Aperture-Dominated",cost: "O(n²/R)",         color: "#CCBB44" },
    { Rens: "0.50–0.80", label: "Hierarchical Cascade",cost:"O(n log n / R²)","color": "#AA3377" },
    { Rens: "0.80–0.95", label: "Coherent",           cost: "O(log n)",       color: "#4477AA" },
    { Rens: "≥ 0.95",    label: "Phase-Locked",       cost: "0",              color: "#58E6D9" },
  ];

  return (
    <>
      <Head>
        <title>Phase-Holographic Cell — Crown Prince</title>
      </Head>
      <section className="mx-auto max-w-7xl px-6 py-10 md:px-10">

        {/* ── Header ── */}
        <header className="mb-8">
          <p className="text-[11px] uppercase tracking-[0.3em] text-primaryDark">
            Oscillator-referenced partition · Identify · Regulate · Protect
          </p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">
            Phase-Holographic Cell
          </h1>
          <p className="mt-3 max-w-3xl text-xs leading-relaxed text-light/60">
            Each organelle is a Kuramoto oscillator whose natural frequency is set
            by its refractive-index offset and structural gradient. The partition
            field{" "}
            <span className="mono accent">(n, ℓ, m, s)</span> is a normalised
            S-entropy coordinate in{" "}
            <span className="mono accent">[0, 100]⁴</span> (distributed-control-system.tex §3).
            Signals between organelles are bare timing pulses; the datum is the
            deviation{" "}
            <span className="mono accent">ΔP(k) = T_ref(k) − t_rec(k)</span>{" "}
            (temporal-programming.tex §2). At ensemble coherence{" "}
            <span className="mono accent">R ≥ 0.95</span> all inter-organelle
            coordination becomes structurally free (swarm-federations.tex Theorem 4.1).
          </p>
        </header>

        {/* ── Cell selector ── */}
        <div className="mb-8 flex flex-wrap items-center gap-3">
          <span className="text-[10px] uppercase tracking-[0.25em] text-light/50">Cell</span>
          {Object.entries(CELL_LABELS).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setCellId(id)}
              className={`rounded border px-3 py-1.5 text-xs uppercase tracking-[0.15em] transition ${
                cellId === id
                  ? "border-primaryDark/60 bg-primaryDark/15 text-primaryDark"
                  : "border-light/15 bg-transparent text-light/60 hover:text-light"
              }`}
            >
              {label}
            </button>
          ))}
          <span className="ml-auto font-mono text-[10px] text-light/40">
            {D > 0 ? `${D} organelles · D = ${D} channels` : "building field…"}
          </span>
        </div>

        {/* ── Section 01: Partition Field ── */}
        <SectionTitle n="01" title="(n, ℓ, m, s) partition field" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Volume ray march · composite (n, ℓ, m, s)">
            <VolumeRaymarch volume={volume} channel={channel} alpha={1.3} steps={96} />
            {/* Channel selector */}
            <div className="mt-3 flex flex-wrap gap-2">
              {CHANNELS.map((ch) => (
                <button
                  key={ch.id}
                  onClick={() => setChannel(ch.id)}
                  className={`rounded border px-2.5 py-1 text-[10px] uppercase tracking-[0.15em] transition ${
                    channel === ch.id
                      ? "border-primaryDark/50 bg-primaryDark/10 text-primaryDark"
                      : "border-light/15 text-light/50 hover:text-light"
                  }`}
                >
                  {ch.label}
                  <span className="ml-1.5 text-light/30">{ch.sub}</span>
                </button>
              ))}
            </div>
          </Panel>

          <Panel title="Field coordinates · S-entropy proxy">
            {fieldCoords ? (
              <div className="space-y-3">
                {[
                  { key: "n(r)", val: fieldCoords.Sk, color: "#58E6D9", sub: "mean refractive index offset" },
                  { key: "ℓ(r)", val: fieldCoords.St, color: "#CCBB44", sub: "mean phase gradient magnitude" },
                  { key: "m(r)", val: fieldCoords.Se, color: "#EE6677", sub: "mean phase orientation" },
                  { key: "s(r)", val: fieldCoords.Schir, color: "#AA3377", sub: "mean chirality" },
                ].map((r) => (
                  <div key={r.key}>
                    <div className="flex items-baseline justify-between font-mono text-xs">
                      <span style={{ color: r.color }}>{r.key}</span>
                      <span className="accent">{r.val.toFixed(4)}</span>
                    </div>
                    <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-light/5">
                      <div className="h-full" style={{ width: `${r.val * 100}%`, backgroundColor: r.color }} />
                    </div>
                    <p className="mt-0.5 text-[10px] text-light/40">{r.sub}</p>
                  </div>
                ))}
                <p className="mt-4 text-[10px] leading-relaxed text-light/40">
                  Coordinates normalised to [0, 1] per distributed-control-system.tex §3.
                  Full S-entropy address derivable from organelle frequency spectrum.
                </p>
              </div>
            ) : (
              <p className="text-xs text-light/40">Building field…</p>
            )}
          </Panel>
        </div>

        {/* ── Section 02: Kuramoto Phase Coupling ── */}
        <SectionTitle n="02" title="Kuramoto phase coupling · organelle ensemble" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title={`R_ens trajectory · K = ${kuramoto.K.toFixed(3)} · Kc = ${kuramoto.baseKc.toFixed(3)}`}>
            <SpectralOverlay
              grid={kuramoto.trajectory.map((_, i) => i)}
              series={[
                {
                  name: "R_ens(t)",
                  values: kuramoto.trajectory,
                  color: regime.color,
                },
                {
                  name: "R* (mean-field)",
                  values: kuramoto.trajectory.map(() =>
                    meanFieldRstar(kuramoto.K, kuramoto.baseKc)
                  ),
                  color: "#444",
                },
              ]}
              caption={`Final R_ens = ${kuramoto.finalRens.toFixed(3)}`}
            />
            {/* Kappa multiplier selector */}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="text-[10px] uppercase tracking-[0.2em] text-light/40">K/Kc</span>
              {KAPPA_LABELS.map((label, i) => (
                <button
                  key={i}
                  onClick={() => setKappaMul(i)}
                  className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-[0.15em] transition ${
                    kappaMul === i
                      ? "border-primaryDark/50 text-primaryDark"
                      : "border-light/15 text-light/50 hover:text-light"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </Panel>

          <Panel title="Coordination regime · five-regime landscape">
            {/* Current regime badge */}
            <div
              className="mb-4 rounded-md border p-4"
              style={{ borderColor: regime.color + "55", backgroundColor: regime.color + "0d" }}
            >
              <p className="text-[10px] uppercase tracking-[0.25em]" style={{ color: regime.color }}>
                Regime {regime.index} · {regime.label}
              </p>
              <p className="mt-1 font-mono text-2xl" style={{ color: regime.color }}>
                R = {kuramoto.finalRens.toFixed(3)}
              </p>
              <p className="mt-1 text-xs text-light/60">
                Coordination cost: <span className="font-mono accent">{regime.cost}</span>
              </p>
              {regime.index === 5 && (
                <p className="mt-2 text-[11px] text-light/60">
                  All inter-organelle coordination messages are structurally
                  redundant. Communication overhead = 0. (Theorem 4.1)
                </p>
              )}
            </div>
            {/* Regime table */}
            <table className="w-full text-[10px]">
              <tbody>
                {regimeRows.map((row) => (
                  <tr
                    key={row.label}
                    className="border-b border-light/5"
                    style={row.label === regime.label ? { backgroundColor: regime.color + "18" } : {}}
                  >
                    <td className="py-1 pr-3 font-mono" style={{ color: row.color }}>{row.Rens}</td>
                    <td className="py-1 pr-3 text-light/70">{row.label}</td>
                    <td className="py-1 font-mono text-light/50">{row.cost}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </div>

        {/* ── Section 03: ΔP Timing Space ── */}
        <SectionTitle n="03" title="ΔP timing space · trajectory distinguishability" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title={`T(m, D) = D · (D+1)^(m−1) · D = ${D} channels`}>
            {trajCounts.length > 0 ? (
              <BarChart
                data={trajCounts}
                yLabel="log₁₀ T(m, D)"
                height={200}
                caption={`${D}-channel organelle swarm: trajectory distinguishability grows as (D+1)^(m−1)`}
              />
            ) : (
              <p className="text-xs text-light/40">Building field…</p>
            )}
          </Panel>

          <Panel title="Temporal programming · organelle timing cells">
            <div className="space-y-2 text-xs leading-relaxed text-light/65">
              <p>
                Each organelle emits bare clock pulses. The datum is{" "}
                <span className="font-mono accent">ΔP(k) = T_ref(k) − t_rec(k)</span>,
                the signed deviation between the reference oscillator tick and the
                actual reception time. No semantic payload is ever transmitted.
              </p>
              <p>
                A timing cell <span className="font-mono accent">C ⊆ ΔP-space</span> with
                positive Lebesgue measure captures the operating condition. The
                cell-action map <span className="font-mono accent">𝒜: 𝒫 → ℱ</span> is
                compiled at deploy time and stored read-only. No runtime path leads
                from signal content to 𝒜 modification — structural incorruptibility
                follows architecturally (temporal-programming.tex Theorem 5.1).
              </p>
              <p>
                Replay attacks fail because the monotone oscillator counter
                ensures <span className="font-mono accent">ΔP₁ ≠ ΔP₀</span> for any
                replay at a later cycle (Theorem 5.3).
              </p>
              <div className="mt-3 rounded border border-light/10 bg-light/[0.02] p-3 font-mono text-[10px]">
                <p className="text-primaryDark">Structural incorruptibility</p>
                <p className="mt-1 text-light/55">No payload parser → no injection surface.</p>
                <p className="mt-0.5 text-light/55">Attack surface = |Cells(𝒜)| = {volume?.placed?.length ?? 0} pre-compiled cells.</p>
              </div>
            </div>
          </Panel>
        </div>

        {/* ── Section 04: Phase-Synchronous Control ── */}
        <SectionTitle n="04" title="Phase-synchronous distributed regulation" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Kuramoto coupling statistics">
            {kuramoto.omegas && (
              <BarChart
                data={(() => {
                  const bins = 12;
                  const ws = kuramoto.omegas;
                  const mn = Math.min(...ws), mx = Math.max(...ws);
                  const bw = (mx - mn) / bins || 0.1;
                  const counts = new Array(bins).fill(0);
                  for (const w of ws) {
                    const b = Math.min(bins - 1, Math.floor((w - mn) / bw));
                    counts[b]++;
                  }
                  const { Kc } = criticalCoupling(ws);
                  return counts.map((c, i) => ({
                    label: (mn + (i + 0.5) * bw).toFixed(2),
                    value: c,
                    color: "#4477AA",
                    sub: "",
                  }));
                })()}
                yLabel="count"
                height={180}
                caption={`σω = ${kuramoto.sigmaOmega?.toFixed(3) ?? "—"}  ·  Kc = ${kuramoto.baseKc.toFixed(3)}  ·  K = ${kuramoto.K.toFixed(3)}`}
              />
            )}
          </Panel>

          <Panel title="PSDR control properties">
            <div className="space-y-3 text-xs leading-relaxed text-light/65">
              <p>
                Physical channels — RI offset <span className="font-mono accent">n(r)</span>,
                gradient <span className="font-mono accent">ℓ(r)</span>, orientation{" "}
                <span className="font-mono accent">m(r)</span>, chirality{" "}
                <span className="font-mono accent">s(r)</span> — are each normalised to
                the unit interval via S-entropy affine maps. The resulting state
                space <span className="font-mono accent">ℋ = [0,1]⁴</span> admits a
                uniform piecewise Lyapunov stability certificate
                (distributed-control-system.tex Theorem 3.2).
              </p>
              <p>
                The phase-domain transfer matrix{" "}
                <span className="font-mono accent">G(ν)</span> is dimensionless;
                cross-channel composition requires no unit conversion. The timing
                deviation <span className="font-mono accent">ΔP(k)</span> serves as both
                measurement primitive and process output, collapsing the
                sensor–plant–observer abstraction into one scalar per channel.
              </p>
              <div className="mt-2 rounded border border-light/10 bg-light/[0.02] p-3 font-mono text-[10px] space-y-1">
                <div className="flex justify-between">
                  <span className="text-light/50">K / Kc</span>
                  <span className="accent">{(kuramoto.K / kuramoto.baseKc).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-light/50">R* (mean-field)</span>
                  <span className="accent">{meanFieldRstar(kuramoto.K, kuramoto.baseKc).toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-light/50">R_ens (simulated)</span>
                  <span className="accent">{kuramoto.finalRens.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-light/50">Regime</span>
                  <span style={{ color: regime.color }}>{regime.label}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-light/50">Coord. cost</span>
                  <span className="accent">{regime.cost}</span>
                </div>
              </div>
            </div>
          </Panel>
        </div>

      </section>
    </>
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
