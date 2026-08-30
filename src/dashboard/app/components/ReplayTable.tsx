"use client";

import type { ReplayRow } from "@/app/lib/types";

export function ReplayTable({ rows }: { rows: ReplayRow[] }) {
  const recent = rows.slice(-25).reverse();

  return (
    // Fixed-height internal scroll instead of letting the table grow the
    // whole page as rows accumulate -- a log/terminal feed scrolls in its
    // own pane, it doesn't push the rest of the layout down. Header stays
    // pinned (sticky) while the body scrolls underneath it.
    <div className="max-h-96 overflow-y-auto overflow-x-auto">
      <table className="min-w-full border-collapse font-mono text-xs">
        <thead className="sticky top-0 z-10 bg-panel">
          <tr className="text-text-secondary">
            <th className="border-b border-border px-3 py-2 text-left font-medium tracking-widest uppercase">
              Flow
            </th>
            <th className="border-b border-border px-3 py-2 text-left font-medium tracking-widest uppercase">
              True
            </th>
            <th className="border-b border-border px-3 py-2 text-left font-medium tracking-widest uppercase">
              Pred
            </th>
            <th className="border-b border-border px-3 py-2 text-right font-medium tracking-widest uppercase">
              Conf
            </th>
            <th className="border-b border-border px-3 py-2 text-right font-medium tracking-widest uppercase">
              Risk
            </th>
            <th className="border-b border-border px-3 py-2 text-left font-medium tracking-widest uppercase">
              State
            </th>
            <th className="border-b border-border px-3 py-2 text-center font-medium tracking-widest uppercase">
              OK
            </th>
          </tr>
        </thead>
        <tbody>
          {recent.map((row) => (
            <tr
              key={row.id}
              className={`motion-safe:animate-row-in border-b border-border/60 border-l-2 ${
                row.is_malicious ? "border-l-amber" : "border-l-transparent"
              }`}
            >
              <td className="whitespace-nowrap px-3 py-1.5 text-text-secondary">{row.id}</td>
              <td className="whitespace-nowrap px-3 py-1.5 text-text-primary">{row.true_label}</td>
              <td
                className={`whitespace-nowrap px-3 py-1.5 ${
                  row.is_malicious ? "text-amber" : "text-text-primary"
                }`}
              >
                {row.prediction}
              </td>
              <td className="whitespace-nowrap px-3 py-1.5 text-right tabular-nums text-text-primary">
                {row.confidence.toFixed(3)}
              </td>
              <td
                className={`whitespace-nowrap px-3 py-1.5 text-right tabular-nums ${
                  row.is_malicious ? "text-amber" : "text-text-secondary"
                }`}
              >
                {row.risk_score.toFixed(4)}
              </td>
              <td className="whitespace-nowrap px-3 py-1.5">
                {row.is_malicious ? (
                  <span className="border border-amber/40 bg-amber/10 px-1.5 py-0.5 text-amber">
                    MALICIOUS
                  </span>
                ) : (
                  <span className="text-text-secondary">benign</span>
                )}
              </td>
              <td className="px-3 py-1.5 text-center text-text-secondary">
                {row.correct ? "✓" : "✗"}
              </td>
            </tr>
          ))}
          {recent.length === 0 && (
            <tr>
              <td className="px-3 py-6 text-center text-text-secondary" colSpan={7}>
                — awaiting replay —
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
