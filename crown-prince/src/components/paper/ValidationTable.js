export function ValidationTable({ rows, summary }) {
  return (
    <div className="my-6 overflow-hidden rounded-md border border-light/10">
      <table className="w-full text-xs">
        <thead className="bg-light/[0.04] text-[10px] uppercase tracking-[0.18em] text-light/50">
          <tr>
            <th className="px-3 py-2 text-left">Test group</th>
            <th className="px-3 py-2 text-right">Passed</th>
            <th className="px-3 py-2 text-right">Total</th>
            <th className="px-3 py-2 text-right">Rate</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {rows.map((r) => {
            const rate = r.total ? r.passed / r.total : 0;
            const ok = rate >= 0.9;
            return (
              <tr key={r.name} className="border-t border-light/10">
                <td className="px-3 py-1.5 text-light/80">{r.name}</td>
                <td className="px-3 py-1.5 text-right text-light/80">{r.passed}</td>
                <td className="px-3 py-1.5 text-right text-light/50">{r.total}</td>
                <td
                  className="px-3 py-1.5 text-right"
                  style={{ color: ok ? "#58E6D9" : "#EE6677" }}
                >
                  {(rate * 100).toFixed(0)}%
                </td>
              </tr>
            );
          })}
          {summary && (
            <tr className="border-t border-light/20 bg-light/[0.03]">
              <td className="px-3 py-2 text-[10px] uppercase tracking-[0.18em] text-primaryDark">
                Aggregate
              </td>
              <td className="px-3 py-2 text-right text-primaryDark">{summary.passed}</td>
              <td className="px-3 py-2 text-right text-light/60">{summary.total}</td>
              <td className="px-3 py-2 text-right text-primaryDark">
                {((summary.passed / summary.total) * 100).toFixed(1)}%
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
