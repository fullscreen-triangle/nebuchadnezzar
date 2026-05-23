import Head from "next/head";
import Link from "next/link";

const TOOLS = [
  {
    href: "/tools/spectrum",
    name: "Cell Spectral Hologram",
    sub: "three-state superposition · K_ij coupling · Franck–Condon · Stokes decomposition · 2D diffraction · cell partition state · loop holonomy · sparse therapeutic η*",
    op: "Identify · Predict · Close",
    status: "live",
  },
  {
    href: "/tools/metabolism",
    name: "Partitioned Metabolism Engine",
    sub: "Q_b Q_m Q_p Q_t Q_d · hierarchical depth D · Kuramoto R · six diagnostic primitives · GLP-1 critique · η_drug",
    op: "Charge · Depth · Diagnose",
    status: "live",
  },
  {
    href: "/tools/hologram",
    name: "Phase-Holographic Cell",
    sub: "(n, ℓ, m, s) partition field · volume ray march · Kuramoto organelle coupling · ΔP timing space · T(m,D) trajectory count · phase-locked regime classifier",
    op: "Identify · Regulate · Protect",
    status: "live",
  },
  {
    href: "/tools/observe",
    name: "Partition Observation",
    sub: "fragment shader as categorical instrument: 65 536 cell observations per draw call",
    op: "Predict",
    status: "soon",
  },
  {
    href: "/tools/interfere",
    name: "Interference Similarity",
    sub: "two textures multiplied with the visibility kernel — similarity as physical observable",
    op: "Similar",
    status: "soon",
  },
];

export default function Tools() {
  return (
    <>
      <Head>
        <title>Tools — Crown Prince</title>
      </Head>
      <section className="mx-auto max-w-5xl px-6 py-12 md:px-10">
        <p className="panel-title">Available primitives</p>
        <h1 className="mb-8 text-2xl font-semibold tracking-tight">Tools</h1>
        <div className="grid gap-4">
          {TOOLS.map((t) => (
            <ToolCard key={t.href} tool={t} />
          ))}
        </div>
      </section>
    </>
  );
}

function ToolCard({ tool }) {
  const live = tool.status === "live";
  const Wrapper = live ? Link : "div";
  const props = live ? { href: tool.href } : {};
  return (
    <Wrapper
      {...props}
      className={`panel flex items-start justify-between gap-6 ${
        live ? "transition hover:border-primaryDark/40" : "opacity-50"
      }`}
    >
      <div className="flex-1">
        <p className="text-[10px] uppercase tracking-[0.25em] text-primaryDark/80">
          Primitive · {tool.op}
        </p>
        <h3 className="mt-1.5 text-lg font-medium">{tool.name}</h3>
        <p className="mt-1.5 text-xs leading-relaxed text-light/60">{tool.sub}</p>
      </div>
      <span
        className={`mt-1 rounded px-2 py-0.5 text-[10px] uppercase tracking-[0.2em] ${
          live
            ? "border border-primaryDark/30 text-primaryDark"
            : "border border-light/20 text-light/40"
        }`}
      >
        {tool.status}
      </span>
    </Wrapper>
  );
}
