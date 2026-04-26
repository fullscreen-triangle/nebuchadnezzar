import Head from "next/head";
import Link from "next/link";
import { PAPERS } from "@/content/papers/_meta";

export default function Papers() {
  return (
    <>
      <Head>
        <title>Papers — Crown Prince</title>
      </Head>
      <section className="mx-auto max-w-5xl px-6 py-12 md:px-10">
        <p className="panel-title">Theoretical corpus</p>
        <h1 className="mb-2 text-2xl font-semibold tracking-tight">Papers</h1>
        <p className="mb-8 max-w-2xl text-sm text-light/60">
          Nine self-contained derivations from the Bounded Phase Space Law.
          Each paper is rendered with its full theorems, equations, validation
          tables, and interactive D3 charts; live papers can be navigated by
          section.
        </p>
        <ol className="space-y-3">
          {PAPERS.map((p) => (
            <li key={p.id}>
              <Link
                href={`/papers/${p.id}`}
                className="panel flex flex-col gap-3 transition hover:border-primaryDark/40 md:flex-row md:items-start md:justify-between"
              >
                <div className="flex-1">
                  <div className="flex items-baseline gap-3">
                    <span className="font-mono text-[10px] text-light/30">{p.n}</span>
                    <p className="text-[10px] uppercase tracking-[0.25em] text-primaryDark/80">
                      {p.kicker}
                    </p>
                  </div>
                  <h3 className="mt-1.5 text-base font-medium leading-snug text-light">
                    {p.title}
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-light/55">
                    {p.abstract}
                  </p>
                  <p className="mt-2 text-[10px] uppercase tracking-[0.18em] text-light/40">
                    Primitives · {p.primitives.join(" · ")}
                  </p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-1 text-right">
                  <span className="font-mono text-[11px] text-primaryDark">{p.tests}</span>
                  <span
                    className={`rounded px-2 py-0.5 text-[10px] uppercase tracking-[0.2em] ${
                      p.status === "live"
                        ? "border border-primaryDark/30 text-primaryDark"
                        : "border border-light/15 text-light/40"
                    }`}
                  >
                    {p.status}
                  </span>
                </div>
              </Link>
            </li>
          ))}
        </ol>
      </section>
    </>
  );
}
