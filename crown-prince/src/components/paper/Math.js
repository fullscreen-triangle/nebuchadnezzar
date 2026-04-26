import { useMemo } from "react";
import { renderMath } from "@/lib/math";

export function M({ children }) {
  const html = useMemo(() => renderMath(String(children), false), [children]);
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export function Eq({ children, label }) {
  const html = useMemo(() => renderMath(String(children), true), [children]);
  return (
    <div className="my-5 flex items-center justify-between gap-4">
      <div className="flex-1 overflow-x-auto" dangerouslySetInnerHTML={{ __html: html }} />
      {label && (
        <span className="font-mono text-[10px] text-light/40 whitespace-nowrap">
          ({label})
        </span>
      )}
    </div>
  );
}
