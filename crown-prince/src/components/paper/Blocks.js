// Theorem-style block primitives. Each renders a coloured rule + label.

function Block({ kind, label, children, accent }) {
  return (
    <div
      className="my-6 rounded-md border-l-2 bg-light/[0.02] p-4 pl-5"
      style={{ borderColor: accent }}
    >
      <p
        className="mb-2 text-[10px] uppercase tracking-[0.25em]"
        style={{ color: accent }}
      >
        {kind}
        {label && <span className="ml-2 text-light/60">· {label}</span>}
      </p>
      <div className="text-sm leading-relaxed text-light/80">{children}</div>
    </div>
  );
}

export const Axiom = ({ label, children }) => (
  <Block kind="Axiom" label={label} accent="#58E6D9">{children}</Block>
);
export const Theorem = ({ label, children }) => (
  <Block kind="Theorem" label={label} accent="#EE6677">{children}</Block>
);
export const Lemma = ({ label, children }) => (
  <Block kind="Lemma" label={label} accent="#CCBB44">{children}</Block>
);
export const Corollary = ({ label, children }) => (
  <Block kind="Corollary" label={label} accent="#AA3377">{children}</Block>
);
export const Proposition = ({ label, children }) => (
  <Block kind="Proposition" label={label} accent="#4477AA">{children}</Block>
);
export const Definition = ({ label, children }) => (
  <Block kind="Definition" label={label} accent="#228833">{children}</Block>
);
export const Remark = ({ label, children }) => (
  <Block kind="Remark" label={label} accent="#888">{children}</Block>
);

export function Proof({ children }) {
  return (
    <div className="my-4 border-l border-light/15 pl-4 text-xs leading-relaxed text-light/60">
      <p className="mb-1 italic text-light/50">Proof.</p>
      {children}
      <span className="ml-1 text-light/40">▪</span>
    </div>
  );
}
