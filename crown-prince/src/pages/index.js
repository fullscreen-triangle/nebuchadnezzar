import Head from "next/head";
import Link from "next/link";

const TOOLS = [
  {
    href: "/tools/spectrum",
    name: "Spectrum Inspector",
    sub: "vibrational frequencies → S-entropy address",
    op: "Identify",
  },
  {
    href: "/tools/observe",
    name: "Partition Observation",
    sub: "fragment shader as categorical instrument",
    op: "Predict",
  },
  {
    href: "/tools/interfere",
    name: "Interference Similarity",
    sub: "two textures → visibility scalar",
    op: "Similar",
  },
];

export default function Home() {
  return (
    <>
      <Head>
        <title>Crown Prince — Geometric pharmacology in the browser</title>
      </Head>
      <section className="mx-auto max-w-5xl px-6 pt-16 pb-10 md:px-10">
        <p className="text-[11px] uppercase tracking-[0.3em] text-primaryDark">
          Bounded Phase Space Law
        </p>
        <h1 className="mt-4 text-4xl font-semibold leading-[1.1] tracking-tight md:text-5xl">
          Drugs and targets addressed in three coordinates.
          <br />
          <span className="text-light/60">No database required.</span>
        </h1>
        <p className="mt-6 max-w-2xl text-sm leading-relaxed text-light/70">
          Every stable molecule is a bounded oscillatory system whose identity is
          determined by three S-entropy coordinates{" "}
          <span className="mono accent">(S_k, S_t, S_e)</span>. This site implements the
          framework as a set of in-browser shader tools: the GPU is the
          categorical observation apparatus, not a graphics accelerator.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link href="/tools" className="btn">
            Open the tools
          </Link>
          <Link href="/framework" className="btn !border-light/20 !bg-transparent !text-light/70 hover:!text-light">
            Read the framework
          </Link>
        </div>
      </section>

      <section className="mx-auto max-w-5xl px-6 py-12 md:px-10">
        <p className="panel-title">Available tools</p>
        <div className="grid gap-4 md:grid-cols-3">
          {TOOLS.map((t) => (
            <Link
              key={t.href}
              href={t.href}
              className="panel group transition hover:border-primaryDark/40"
            >
              <p className="text-[10px] uppercase tracking-[0.25em] text-primaryDark/80">
                Primitive · {t.op}
              </p>
              <h3 className="mt-2 text-lg font-medium text-light group-hover:text-primaryDark">
                {t.name}
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-light/60">{t.sub}</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-5xl px-6 pb-20 md:px-10">
        <div className="grid gap-4 md:grid-cols-3">
          <Stat label="Storage" value="50 MB" sub="vs 12 GB DrugBank" />
          <Stat label="Query cost" value="O(k)" sub="independent of N" />
          <Stat label="Probe size" value="0.6 M params" sub="LoRA adapter" />
        </div>
      </section>
    </>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="panel">
      <p className="panel-title">{label}</p>
      <p className="font-mono text-2xl text-primaryDark">{value}</p>
      <p className="mt-1 text-xs text-light/50">{sub}</p>
    </div>
  );
}
