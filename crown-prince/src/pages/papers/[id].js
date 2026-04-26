import Head from "next/head";
import Link from "next/link";
import { PAPERS, paperById } from "@/content/papers/_meta";
import PaperLayout from "@/components/paper/PaperLayout";

// Dynamically loadable paper modules.
const LIVE = {
  "synthentic-isomorphism-database": () =>
    import("@/content/papers/synthentic-isomorphism"),
};

export default function PaperPage({ id, isLive }) {
  const meta = paperById(id);
  if (!meta) {
    return (
      <section className="mx-auto max-w-3xl px-6 py-20 md:px-10">
        <h1 className="text-2xl">Unknown paper</h1>
      </section>
    );
  }

  if (!isLive) {
    return <Stub meta={meta} />;
  }

  return <LivePaper id={id} />;
}

function Stub({ meta }) {
  return (
    <>
      <Head>
        <title>{meta.title} — Crown Prince</title>
      </Head>
      <section className="mx-auto max-w-3xl px-6 py-12 md:px-10">
        <Link
          href="/papers"
          className="text-[10px] uppercase tracking-[0.25em] text-light/50 hover:text-light"
        >
          ← all papers
        </Link>
        <p className="mt-6 text-[11px] uppercase tracking-[0.3em] text-primaryDark">
          {meta.kicker} · {meta.tests}
        </p>
        <h1 className="mt-2 text-3xl font-semibold leading-tight tracking-tight">
          {meta.title}
        </h1>
        <div className="mt-6 rounded-md border border-light/10 bg-light/[0.03] p-5">
          <p className="mb-2 text-[10px] uppercase tracking-[0.25em] text-light/50">
            Abstract
          </p>
          <p className="text-sm leading-relaxed text-light/80">{meta.abstract}</p>
        </div>
        <p className="mt-3 text-[10px] uppercase tracking-[0.18em] text-light/40">
          Primitives · {meta.primitives.join(" · ")}
        </p>

        <div className="mt-12 rounded-md border border-dashed border-light/15 bg-light/[0.01] p-6">
          <p className="text-[10px] uppercase tracking-[0.25em] text-light/40">
            Status
          </p>
          <p className="mt-2 text-sm text-light/65">
            Full content with theorems, equations, and interactive D3 charts is
            forthcoming. The paper is fully written in the source corpus
            (TeX); rendering it for the web is queued in the same pattern as the{" "}
            <Link href="/papers/synthentic-isomorphism-database" className="accent underline">
              Synthentic Isomorphism Database
            </Link>{" "}
            paper.
          </p>
        </div>
      </section>
    </>
  );
}

function LivePaper({ id }) {
  // For the synthentic paper, we statically import to avoid dynamic-import
  // boundaries during SSR for KaTeX/d3.
  // (Could be replaced by next/dynamic if more papers go live.)
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const mod = require("@/content/papers/synthentic-isomorphism").default;
  const { meta, sections, Body } = mod;
  return (
    <PaperLayout meta={meta} sections={sections}>
      <Body />
    </PaperLayout>
  );
}

export async function getStaticPaths() {
  return {
    paths: PAPERS.map((p) => ({ params: { id: p.id } })),
    fallback: false,
  };
}

export async function getStaticProps({ params }) {
  const { id } = params;
  return {
    props: {
      id,
      isLive: !!LIVE[id],
    },
  };
}
