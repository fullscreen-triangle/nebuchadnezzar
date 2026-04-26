import Head from "next/head";
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { COMPOUNDS } from "@/lib/compounds";
import { sEntropy } from "@/lib/sentropy";
import { encode, formatTritLine, sharedPrefix } from "@/lib/ternary";

// R3F components are client-only.
const SEntropyCube = dynamic(() => import("@/components/canvas/SEntropyCube"), {
  ssr: false,
  loading: () => <div className="aspect-square w-full animate-pulse rounded-md border border-light/10 bg-dark/40" />,
});
const PartitionShader = dynamic(
  () => import("@/components/canvas/PartitionShader"),
  {
    ssr: false,
    loading: () => <div className="aspect-[2/1] w-full animate-pulse rounded-md border border-light/10 bg-dark/40" />,
  }
);

const DEPTH = 18;

export default function Spectrum() {
  const [presetName, setPresetName] = useState("H2O");
  const [text, setText] = useState(
    COMPOUNDS.find((c) => c.name === "H2O").omega.join("\n")
  );

  const omega = useMemo(() => parseSpectrum(text), [text]);
  const compound = useMemo(
    () => COMPOUNDS.find((c) => c.name === presetName) ?? null,
    [presetName]
  );
  const bRot = compound && compound.bRot ? compound.bRot : null;
  const coords = useMemo(() => sEntropy(omega, bRot), [omega, bRot]);
  const trits = useMemo(() => encode(coords, DEPTH), [coords]);

  const neighbours = useMemo(() => {
    const all = COMPOUNDS.map((c) => {
      const cs = sEntropy(c.omega, c.bRot);
      const cTrits = encode(cs, DEPTH);
      return {
        name: c.name,
        formula: c.formula,
        prefix: sharedPrefix(trits, cTrits),
      };
    });
    return all.sort((a, b) => b.prefix - a.prefix).slice(0, 6);
  }, [trits]);

  const onPreset = (name) => {
    setPresetName(name);
    const c = COMPOUNDS.find((x) => x.name === name);
    if (c) setText(c.omega.join("\n"));
  };

  return (
    <>
      <Head>
        <title>Spectrum Inspector — Crown Prince</title>
      </Head>
      <section className="mx-auto max-w-7xl px-6 py-10 md:px-10">
        <header className="mb-8 flex items-end justify-between gap-6">
          <div>
            <p className="text-[11px] uppercase tracking-[0.3em] text-primaryDark">
              Identify primitive
            </p>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">
              Spectrum Inspector
            </h1>
            <p className="mt-2 max-w-xl text-xs leading-relaxed text-light/60">
              Enter vibrational frequencies in cm⁻¹, one per line. The site
              computes <span className="mono accent">(S_k, S_t, S_e)</span>,
              encodes the {DEPTH}-trit ternary address, locates the molecule in
              the S-entropy cube, and renders the partition observation shader.
            </p>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-12">
          {/* Left column: input + readout */}
          <div className="space-y-4 lg:col-span-4">
            <PresetPicker value={presetName} onChange={onPreset} />
            <SpectrumInput value={text} onChange={setText} omegaCount={omega.length} />
            <Readout coords={coords} />
            <Address trits={trits} />
          </div>

          {/* Centre column: 3D cube */}
          <div className="lg:col-span-5">
            <p className="panel-title">S-entropy cube</p>
            <SEntropyCube query={coords} />
            <Legend />
          </div>

          {/* Right column: nearest neighbours */}
          <div className="lg:col-span-3">
            <Neighbours items={neighbours} />
          </div>
        </div>

        {/* Full-width: partition observation shader */}
        <div className="mt-10">
          <p className="panel-title">Partition observation A(u; M)</p>
          <PartitionShader omega={omega} coords={coords} />
          <p className="mt-2 text-[10px] uppercase tracking-[0.2em] text-light/40">
            Fragment shader · one draw call · {omega.length} mode
            {omega.length === 1 ? "" : "s"} · 65 536 cells
          </p>
        </div>
      </section>
    </>
  );
}

function parseSpectrum(text) {
  return text
    .split(/[\s,;]+/)
    .map((s) => parseFloat(s))
    .filter((x) => Number.isFinite(x) && x > 0);
}

function PresetPicker({ value, onChange }) {
  return (
    <div className="panel">
      <p className="panel-title">NIST preset</p>
      <select
        className="input-base appearance-none"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {COMPOUNDS.map((c) => (
          <option key={c.name} value={c.name}>
            {c.formula} · {c.type}
          </option>
        ))}
      </select>
    </div>
  );
}

function SpectrumInput({ value, onChange, omegaCount }) {
  return (
    <div className="panel">
      <div className="mb-2 flex items-center justify-between">
        <p className="panel-title !mb-0">Vibrational frequencies (cm⁻¹)</p>
        <span className="font-mono text-[10px] text-light/40">N = {omegaCount}</span>
      </div>
      <textarea
        rows={8}
        className="input-base"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
      />
    </div>
  );
}

function Readout({ coords }) {
  const rows = [
    { key: "S_k", color: "#EE6677", val: coords.Sk, sub: "knowledge / spectral shape" },
    { key: "S_t", color: "#228833", val: coords.St, sub: "temporal / timescale span" },
    { key: "S_e", color: "#4477AA", val: coords.Se, sub: "evolution / harmonic density" },
  ];
  return (
    <div className="panel">
      <p className="panel-title">S-entropy coordinates</p>
      <div className="space-y-3">
        {rows.map((r) => (
          <div key={r.key}>
            <div className="flex items-baseline justify-between font-mono text-xs">
              <span className="text-light/80" style={{ color: r.color }}>
                {r.key}
              </span>
              <span className="accent">{r.val.toFixed(4)}</span>
            </div>
            <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-light/5">
              <div
                className="h-full"
                style={{ width: `${r.val * 100}%`, backgroundColor: r.color }}
              />
            </div>
            <p className="mt-1 text-[10px] text-light/40">{r.sub}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function Address({ trits }) {
  const dimColor = ["#EE6677", "#228833", "#4477AA"];
  return (
    <div className="panel">
      <p className="panel-title">{trits.length}-trit address</p>
      <div className="flex flex-wrap gap-1 font-mono text-xs">
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
      <p className="mt-2 text-[10px] uppercase tracking-[0.2em] text-light/40">
        {formatTritLine(trits)}
      </p>
    </div>
  );
}

function Legend() {
  return (
    <div className="mt-2 flex items-center gap-4 text-[10px] uppercase tracking-[0.2em] text-light/50">
      <Dot c="#4477AA" t="diatomic" />
      <Dot c="#228833" t="triatomic" />
      <Dot c="#CCBB44" t="tetra" />
      <Dot c="#EE6677" t="poly" />
      <span className="ml-auto accent">▣ query</span>
    </div>
  );
}

function Dot({ c, t }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block h-2 w-2 rounded-full" style={{ background: c }} />
      {t}
    </span>
  );
}

function Neighbours({ items }) {
  return (
    <div className="panel">
      <p className="panel-title">Nearest by trit prefix</p>
      <ul className="space-y-1.5 font-mono text-xs">
        {items.map((n, i) => (
          <li key={n.name} className="flex items-baseline justify-between">
            <span className={i === 0 ? "accent" : "text-light/80"}>{n.formula}</span>
            <span className="text-light/40">k = {n.prefix}</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[10px] uppercase tracking-[0.2em] text-light/40">
        Shared prefix length = ternary similarity (Theorem 5.2)
      </p>
    </div>
  );
}
