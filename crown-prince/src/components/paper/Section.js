function slug(s) {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

export function Section({ n, title, children }) {
  const id = slug(title);
  return (
    <section id={id} className="mt-12 scroll-mt-24">
      <h2 className="mb-4 flex items-baseline gap-3 text-xl font-semibold tracking-tight">
        <span className="font-mono text-[11px] text-primaryDark">{n}</span>
        <span>{title}</span>
      </h2>
      <div className="space-y-4 text-sm leading-relaxed text-light/80">
        {children}
      </div>
    </section>
  );
}

export function Sub({ title, children }) {
  const id = slug(title);
  return (
    <div id={id} className="mt-6 scroll-mt-24">
      <h3 className="mb-3 text-sm font-medium uppercase tracking-[0.15em] text-light/70">
        {title}
      </h3>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

export function P({ children }) {
  return <p className="text-sm leading-relaxed text-light/80">{children}</p>;
}

export function L({ children }) {
  return (
    <ul className="my-3 list-disc space-y-1.5 pl-5 text-sm text-light/80 marker:text-primaryDark/50">
      {children}
    </ul>
  );
}
