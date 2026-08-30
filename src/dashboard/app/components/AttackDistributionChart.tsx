"use client";

import { Bar, BarChart, Cell, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ReplayRow } from "@/app/lib/types";

const GRID = "#22304a";
const TEXT_SECONDARY = "#8a93a6";
const TEXT_PRIMARY = "#e4e8f1";
const AMBER = "#f5a623";
const PANEL = "#141b2c";
const BORDER = "#22304a";

export function AttackDistributionChart({ rows }: { rows: ReplayRow[] }) {
  const counts = new Map<string, number>();
  for (const row of rows) {
    counts.set(row.prediction, (counts.get(row.prediction) ?? 0) + 1);
  }
  const data = Array.from(counts.entries())
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);

  return (
    <div className="h-64 w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          {/* interval={0}: Recharts defaults category axes to
              interval="preserveEnd", which silently thins ticks to avoid
              overlap -- with 9 classes that dropped several labels
              entirely. Forcing every tick to render, then truncating to 3
              chars (same convention as the confusion matrix's row/column
              labels) keeps all 9 visible without rotation; the full name
              is still available via the bar's own tooltip on hover. */}
          <XAxis
            dataKey="label"
            interval={0}
            tickFormatter={(label: string) => label.slice(0, 3)}
            fontSize={11}
            fontFamily="var(--font-mono)"
            stroke={TEXT_SECONDARY}
            tick={{ fill: TEXT_SECONDARY }}
            height={30}
          />
          <YAxis
            allowDecimals={false}
            fontSize={11}
            fontFamily="var(--font-mono)"
            stroke={TEXT_SECONDARY}
            tick={{ fill: TEXT_SECONDARY }}
          />
          {/* itemStyle: recharts' DefaultTooltipContent falls back to
              `entry.color || '#000'` for the value-row text color, and
              since <Bar> itself has no fill prop (only its per-class
              <Cell> children do), that fallback -- black -- was winning,
              making the count unreadable against the dark panel. */}
          <Tooltip
            contentStyle={{
              background: PANEL,
              border: `1px solid ${BORDER}`,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              color: TEXT_PRIMARY,
            }}
            labelStyle={{ color: TEXT_PRIMARY }}
            itemStyle={{ color: TEXT_PRIMARY }}
            cursor={{ fill: BORDER, opacity: 0.3 }}
          />
          <Bar dataKey="count">
            {data.map((entry) => (
              <Cell key={entry.label} fill={entry.label === "BENIGN" ? TEXT_SECONDARY : AMBER} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
