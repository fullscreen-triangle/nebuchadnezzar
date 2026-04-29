import Head from "next/head";
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import {
  CELL_TYPES,
  buildSpectra,
  sampleSpectrum,
  buildHologram,
  couplingMatrix,
  franckCondon,
  stokesShift,
  holonomy,
  sparsePerturbation,
} from "@/lib/cellspectra";
import { sEntropy } from "@/lib/sentropy";
import { encode, sharedPrefix } from "@/lib/ternary";
import { COMPOUNDS } from "@/lib/compounds";

import SpectralOverlay from "@/components/charts/SpectralOverlay";
import HologramPanel from "@/components/charts/HologramPanel";
import MatrixHeatmap from "@/components/charts/MatrixHeatmap";
import BarChart from "@/components/charts/BarChart";
import StackedBars from "@/components/charts/StackedBars";
import DiffractionPattern from "@/components/charts/DiffractionPattern";
import HolonomyBars from "@/components/charts/HolonomyBars";
import PerturbationBars from "@/components/charts/PerturbationBars";

const SEntropyCube = dynamic(() => import("@/components/canvas/SEntropyCube"), {
  ssr: false,
  loading: () => <div className="aspect-square w-full animate-pulse rounded-md border border-light/10 bg-dark/40" />,
});
const PartitionShader = dynamic(() => import("@/components/canvas/PartitionShader"), {
  ssr: false,
  loading: () => <div className="aspect-[2/1] w-full animate-pulse rounded-md border border-light/10 bg-dark/40" />,
});

const FREQ_GRID = (() => {
  const arr = [];
  for (let w = 200; w <= 3800; w += 18) arr.push(w);
  return arr;
})();
const TIME_GRID = (() => {
  const arr = [];
  for (let t = 0; t <= 1; t += 1 / 31) arr.push(t);
  return arr;
})();

const DEPTH = 18;

export default function CellSpectralHologram() {
  const [cellId, setCellId] = useState("hepatocyte");
  const cell = { id: cellId, ...CELL_TYPES[cellId] };

  // --- Spectra: ground IR (red), excited Raman (teal), emission (yellow)
  const spectra = useMemo(() => buildSpectra(cell), [cellId]);
  const yIR = useMemo(() => sampleSpectrum(spectra.ground, FREQ_GRID), [spectra]);
  const yRaman = useMemo(() => sampleSpectrum(spectra.excited, FREQ_GRID), [spectra]);
  const yEm = useMemo(
    () =>
      FREQ_GRID.map((w) =>
        Math.exp(-Math.pow((w - (FREQ_GRID[0] + FREQ_GRID[FREQ_GRID.length - 1]) / 2) / 600, 2)) * 0.8
      ),
    []
  );

  // --- Hologram H(omega, t)
  const hologram = useMemo(
    () => buildHologram(cell, FREQ_GRID, TIME_GRID),
    [cellId]
  );

  // --- Coupling matrix
  const { K, labels } = useMemo(() => couplingMatrix(cell), [cellId]);

  // --- Franck-Condon
  const fc = useMemo(() => franckCondon(cell), [cellId]);

  // --- Stokes
  const stokes = useMemo(() => stokesShift(cell), [cellId]);

  // --- Holonomy + sparse drug design
  const hol = useMemo(() => holonomy(cell), [cellId]);
  const perturbation = useMemo(() => sparsePerturbation(cell, hol), [cellId, hol]);
  const diseasedCycles = hol.filter((c) => !c.healthy).length;

  // --- S-entropy address from the dominant ground-state spectrum
  const omegaForAddress = spectra.ground.map((m) => m.omega);
  const coords = useMemo(() => sEntropy(omegaForAddress), [omegaForAddress]);
  const trits = useMemo(() => encode(coords, DEPTH), [coords]);

  // Nearest neighbours among the 39 NIST compounds
  const neighbours = useMemo(() => {
    return COMPOUNDS.map((c) => {
      const cs = sEntropy(c.omega, c.bRot);
      const cTrits = encode(cs, DEPTH);
      return { name: c.formula, prefix: sharedPrefix(trits, cTrits) };
    })
      .sort((a, b) => b.prefix - a.prefix)
      .slice(0, 6);
  }, [trits]);

  return (
    <>
      <Head>
        <title>Cell Spectral Hologram — Crown Prince</title>
      </Head>
      <section className="mx-auto max-w-7xl px-6 py-10 md:px-10">
        <header className="mb-8">
          <p className="text-[11px] uppercase tracking-[0.3em] text-primaryDark">
            Cell-level partition state · Identify · Predict · Close
          </p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">
            Cell Spectral Hologram
          </h1>
          <p className="mt-3 max-w-3xl text-xs leading-relaxed text-light/60">
            A cell is a bounded oscillatory system whose vibrational structure is
            partitioned across three electronic states — ground (IR), excited
            (Raman), and emission (fluorescence). Superimposing the three
            spectra produces the spectral hologram{" "}
            <span className="mono accent">H(ω, t) = Σₙ cₙ(t) Sₙ(ω) e^(iφₙ(t))</span>,
            from which the vibrational coupling matrix, Franck–Condon factors,
            Stokes shift decomposition, and 2D diffraction symmetry are
            extractable in a single measurement. This tool implements the
            full pipeline on a synthetic cell, derives the cell's S-entropy
            address, computes its loop-holonomy diagnostic, and produces the
            sparse <span className="mono accent">η*</span> therapeutic
            perturbation.
          </p>
        </header>

        {/* Cell preset selector */}
        <div className="mb-8 flex flex-wrap items-center gap-3">
          <span className="text-[10px] uppercase tracking-[0.25em] text-light/50">
            Cell
          </span>
          {Object.entries(CELL_TYPES).map(([key, c]) => (
            <button
              key={key}
              onClick={() => setCellId(key)}
              className={`rounded border px-3 py-1.5 text-xs uppercase tracking-[0.15em] transition ${
                cellId === key
                  ? "border-primaryDark/60 bg-primaryDark/15 text-primaryDark"
                  : "border-light/15 bg-transparent text-light/60 hover:text-light"
              }`}
            >
              {c.label}
            </button>
          ))}
          <span className="ml-auto font-mono text-[10px] text-light/40">
            coherence = {cell.coherence.toFixed(2)} · M = {cell.partition_depth} ·{" "}
            {diseasedCycles ? (
              <span className="text-[#EE6677]">{diseasedCycles} diseased cycles</span>
            ) : (
              <span className="text-primaryDark">all cycles healthy</span>
            )}
          </span>
        </div>

        {/* ---- Section 1: three-state spectra and hologram ---- */}
        <SectionTitle n="01" title="Three-state spectra" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Ground IR · excited Raman · emission">
            <SpectralOverlay
              grid={FREQ_GRID}
              series={[
                { name: "IR (ground)", values: yIR, color: "#EE6677" },
                { name: "Raman (excited)", values: yRaman, color: "#58E6D9" },
                { name: "emission", values: yEm, color: "#CCBB44" },
              ]}
              caption="Three-state superposition basis"
            />
          </Panel>
          <Panel title="Spectral hologram H(ω, t)">
            <HologramPanel
              hologram={hologram}
              caption="|H| amplitude · t = oscillation cycle"
            />
          </Panel>
        </div>

        {/* ---- Section 2: vibrational structure ---- */}
        <SectionTitle n="02" title="Vibrational structure" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Coupling matrix K_ij">
            <MatrixHeatmap
              matrix={K}
              labels={labels}
              caption="Mode-pair coupling from ground/excited frequency shifts"
              sym="K"
            />
          </Panel>
          <Panel title={`Franck–Condon ladder · S = ${fc.S.toFixed(2)}`}>
            <BarChart
              data={fc.factors.map((v, n) => ({
                label: `0→${n}`,
                value: parseFloat(v.toFixed(3)),
                color: n === 0 ? "#58E6D9" : "#4477AA",
                sub: `|⟨${n}|0⟩|² = ${v.toFixed(3)}`,
              }))}
              yLabel="|⟨n|0⟩|²"
              height={240}
              caption={`Dominant mode ${fc.mode} cm⁻¹  · Huang–Rhys S = ${fc.S.toFixed(2)}`}
            />
          </Panel>
        </div>

        {/* ---- Section 3: thermodynamic decomposition + diffraction ---- */}
        <SectionTitle n="03" title="Stokes & molecular symmetry" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Stokes shift decomposition">
            <StackedBars
              total={stokes.total}
              items={[
                { label: "vibrational", value: stokes.vibrational },
                { label: "solvent", value: stokes.solvent },
              ]}
              label="Stokes shift = vibrational + solvent reorganisation"
              sub={`λ_reorg = ${stokes.lambdaReorg.toFixed(0)} cm⁻¹  ·  total ${stokes.total.toFixed(0)} cm⁻¹`}
              color1="#58E6D9"
              color2="#AA3377"
            />
            <p className="mt-3 text-[11px] leading-relaxed text-light/55">
              The vibrational component reflects intramolecular relaxation; the
              solvent component reflects environmental reorganisation. Diseased
              cells exhibit larger solvent reorganisation due to altered local
              dielectric and disrupted hydrogen-bond networks.
            </p>
          </Panel>
          <Panel title="2D diffraction · molecular symmetry">
            <DiffractionPattern
              hologram={hologram}
              caption="2D FFT of the hologram"
            />
          </Panel>
        </div>

        {/* ---- Section 4: cell partition state ---- */}
        <SectionTitle n="04" title="Cell partition state" />
        <div className="grid gap-6 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <Panel title="S-entropy cube">
              <SEntropyCube query={coords} />
              <p className="mt-2 text-[10px] uppercase tracking-[0.18em] text-light/50">
                Cell address · query in <span className="accent">teal</span>
              </p>
            </Panel>
          </div>
          <div className="lg:col-span-7 space-y-4">
            <Panel title="Coordinates and ternary address">
              <CoordReadout coords={coords} />
              <Address trits={trits} />
            </Panel>
            <Panel title="Nearest molecular neighbours by trit prefix">
              <ul className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs md:grid-cols-3">
                {neighbours.map((n, i) => (
                  <li key={n.name} className="flex items-baseline justify-between">
                    <span className={i === 0 ? "accent" : "text-light/80"}>{n.name}</span>
                    <span className="text-light/40">k = {n.prefix}</span>
                  </li>
                ))}
              </ul>
            </Panel>
          </div>
        </div>

        {/* ---- Section 5: partition observation shader ---- */}
        <SectionTitle n="05" title="Partition observation A(u; M)" />
        <Panel title="Fragment shader · 65 536 cell observations per draw call">
          <PartitionShader omega={omegaForAddress} coords={coords} />
        </Panel>

        {/* ---- Section 6: holonomy + therapy ---- */}
        <SectionTitle n="06" title="Loop holonomy and therapeutic perturbation" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="det(Hol_ℓ) per cellular cycle">
            <HolonomyBars
              data={hol}
              caption={
                diseasedCycles
                  ? "Cells out of band require closure"
                  : "All cycles within healthy band"
              }
            />
          </Panel>
          <Panel title="Sparse drug perturbation η*">
            <PerturbationBars
              data={perturbation.eta}
              l1={perturbation.l1}
              caption="Solution of arg min ‖η‖₁ s.t. Hol_ℓ(η) = Id"
            />
            <p className="mt-3 text-[11px] leading-relaxed text-light/55">
              Each non-zero edge corresponds to one drug-target intervention. By
              the Therapeutic Drug Signature theorem, drugs whose
              S-entropy-trajectory intersection pattern reproduces η* are
              candidate therapeutics; inverse synthetic retrieval enumerates
              them from the drug trie.
            </p>
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

function CoordReadout({ coords }) {
  const rows = [
    { key: "S_k", color: "#EE6677", val: coords.Sk, sub: "knowledge / spectral shape" },
    { key: "S_t", color: "#228833", val: coords.St, sub: "temporal / timescale span" },
    { key: "S_e", color: "#4477AA", val: coords.Se, sub: "evolution / harmonic density" },
  ];
  return (
    <div className="space-y-2">
      {rows.map((r) => (
        <div key={r.key}>
          <div className="flex items-baseline justify-between font-mono text-xs">
            <span style={{ color: r.color }}>{r.key}</span>
            <span className="accent">{r.val.toFixed(4)}</span>
          </div>
          <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-light/5">
            <div
              className="h-full"
              style={{ width: `${r.val * 100}%`, backgroundColor: r.color }}
            />
          </div>
          <p className="mt-0.5 text-[10px] text-light/40">{r.sub}</p>
        </div>
      ))}
    </div>
  );
}

function Address({ trits }) {
  const dimColor = ["#EE6677", "#228833", "#4477AA"];
  return (
    <div className="mt-3">
      <p className="mb-1 text-[10px] uppercase tracking-[0.18em] text-light/50">
        18-trit address
      </p>
      <div className="flex flex-wrap gap-1 font-mono text-[11px]">
        {trits.map((t, i) => (
          <span
            key={i}
            className="rounded border px-1.5 py-0.5"
            style={{
              borderColor: dimColor[i % 3] + "55",
              color: dimColor[i % 3],
            }}
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
