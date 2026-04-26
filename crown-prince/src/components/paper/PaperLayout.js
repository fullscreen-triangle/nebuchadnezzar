import Head from "next/head";
import Link from "next/link";

export default function PaperLayout({ meta, sections, children }) {
  return (
    <>
      <Head>
        <title>{meta.title} — Crown Prince</title>
        <meta name="description" content={meta.abstract?.slice(0, 160)} />
      </Head>
      <div className="mx-auto grid max-w-7xl gap-10 px-6 py-12 lg:grid-cols-12 md:px-10">
        {/* Sidebar TOC */}
        <aside className="lg:col-span-3">
          <div className="sticky top-6 space-y-6">
            <div>
              <Link
                href="/papers"
                className="text-[10px] uppercase tracking-[0.25em] text-light/50 hover:text-light"
              >
                ← all papers
              </Link>
              <p className="mt-3 text-[10px] uppercase tracking-[0.25em] text-primaryDark">
                Paper · {meta.id}
              </p>
              <p className="mt-1 font-mono text-[10px] text-light/50">
                {meta.tests || ""}
              </p>
            </div>
            <nav className="space-y-1.5 text-xs">
              {sections.map((s) => (
                <a
                  key={s.id}
                  href={`#${s.id}`}
                  className="block text-light/55 transition hover:text-primaryDark"
                >
                  <span className="mr-2 font-mono text-[10px] text-light/30">
                    {s.n}
                  </span>
                  {s.title}
                </a>
              ))}
            </nav>
          </div>
        </aside>

        {/* Main column */}
        <article className="lg:col-span-9">
          <header className="mb-8 border-b border-light/10 pb-8">
            <p className="text-[11px] uppercase tracking-[0.3em] text-primaryDark">
              {meta.kicker}
            </p>
            <h1 className="mt-3 text-3xl font-semibold leading-[1.15] tracking-tight md:text-4xl">
              {meta.title}
            </h1>
            <p className="mt-4 text-xs text-light/50">
              {meta.author} · {meta.year}
            </p>
            <div className="mt-6 rounded-md border border-light/10 bg-light/[0.03] p-5">
              <p className="mb-2 text-[10px] uppercase tracking-[0.25em] text-light/50">
                Abstract
              </p>
              <p className="text-sm leading-relaxed text-light/80">{meta.abstract}</p>
            </div>
            {meta.keywords && (
              <p className="mt-3 text-[10px] uppercase tracking-[0.18em] text-light/40">
                Keywords · {meta.keywords.join(" · ")}
              </p>
            )}
          </header>
          {children}
        </article>
      </div>
    </>
  );
}
