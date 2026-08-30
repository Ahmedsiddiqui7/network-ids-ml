"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/app/components/Card";
import type { ModelInfo } from "@/app/lib/types";

const ZOOM_FPR_MAX = 0.05;
const PR_TAIL_STEPS = 15;

const GRID = "#22304a";
const TEXT_SECONDARY = "#8a93a6";
const TEXT_PRIMARY = "#e4e8f1";
const AMBER = "#f5a623";
const PANEL = "#141b2c";
const BORDER = "#22304a";
const MONO = "var(--font-mono)";

// Recharts' XAxis `domain` only controls the visible scale -- it doesn't
// reliably clip a Line whose underlying `data` array spans far outside
// that domain (points outside the domain can end up excluded from the
// rendered path entirely rather than clipped at the boundary), which is
// exactly why the zoomed ROC/PR panels rendered with no visible line.
// Fix: give each zoomed chart its OWN pre-filtered `data` array -- every
// point already within [min, max], plus the single nearest point just
// outside each edge (if one exists) so the line still reaches the domain
// boundary instead of stopping short of it. `data` must already be sorted
// ascending by `key` (true for both rocData and the sorted prData).
function sliceToDomain<T extends Record<string, number>>(
  data: T[],
  key: keyof T,
  min: number,
  max: number,
): T[] {
  const inRange = data.filter((d) => d[key] >= min && d[key] <= max);
  const before = [...data].reverse().find((d) => d[key] < min);
  const after = data.find((d) => d[key] > max);
  return [...(before ? [before] : []), ...inRange, ...(after ? [after] : [])];
}

// Neutral signal-magnitude heatmap -- this is a density reading, not a
// good/bad judgment, so it stays in the slate-blue family. Amber is
// reserved for the malicious/attack signal elsewhere on the dashboard.
function cellShade(value: number, rowMax: number): string {
  if (rowMax === 0 || value === 0) return "";
  const intensity = value / rowMax;
  if (intensity < 0.01) return "bg-border/15";
  if (intensity < 0.1) return "bg-border/35";
  if (intensity < 0.5) return "bg-border/70";
  return "bg-text-secondary text-bg";
}

function ConfusionMatrix({ info }: { info: ModelInfo }) {
  const { confusion_matrix: cm, confusion_matrix_labels: labels } = info;
  return (
    <div className="overflow-x-auto">
      <table className="font-mono text-xs">
        <thead>
          <tr>
            <th className="px-1 py-1" />
            {labels.map((l) => (
              <th
                key={l}
                className="whitespace-nowrap px-1 py-1 font-medium text-text-secondary"
                title={l}
              >
                {l.slice(0, 3)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => {
            const rowMax = Math.max(...row);
            return (
              <tr key={labels[i]}>
                <td
                  className="whitespace-nowrap px-1 py-1 text-right font-medium text-text-secondary"
                  title={labels[i]}
                >
                  {labels[i].slice(0, 3)}
                </td>
                {row.map((value, j) => (
                  <td
                    key={j}
                    className={`px-1 py-1 text-center tabular-nums text-text-primary ${cellShade(value, rowMax)}`}
                  >
                    {value}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Recharts only computes "nice" rounded tick values when an explicit
// tickCount is set; without one, a type="number" axis falls back to
// labeling ticks with the raw underlying data values -- e.g. the zoomed
// PR chart's boundary point from sliceToDomain() (0.9482021821141775),
// not a clean 0.95. Display-only formatting, doesn't touch which points
// are plotted or how the domain is sliced.
const formatTick = (v: number) => v.toFixed(3);

const axisProps = {
  fontSize: 10,
  fontFamily: MONO,
  stroke: TEXT_SECONDARY,
  tick: { fill: TEXT_SECONDARY },
  tickFormatter: formatTick,
};

const tooltipContentStyle = {
  background: PANEL,
  border: `1px solid ${BORDER}`,
  fontFamily: MONO,
  fontSize: 12,
  color: TEXT_PRIMARY,
};

function RocChart({
  data,
  domain,
  operatingPoint,
}: {
  data: { fpr: number; tpr: number }[];
  domain: [number, number];
  operatingPoint: { x: number; y: number } | null;
}) {
  return (
    <div className="h-40 w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="fpr" type="number" domain={domain} {...axisProps} />
          <YAxis dataKey="tpr" type="number" domain={[0, 1]} {...axisProps} />
          <Tooltip
            formatter={(v) => Number(v).toFixed(4)}
            contentStyle={tooltipContentStyle}
            labelStyle={{ color: TEXT_PRIMARY }}
          />
          <Line
            type="monotone"
            dataKey="tpr"
            stroke={AMBER}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            className="trace-glow"
          />
          {operatingPoint && (
            <ReferenceDot
              x={operatingPoint.x}
              y={operatingPoint.y}
              r={4}
              fill={TEXT_PRIMARY}
              stroke={AMBER}
              strokeWidth={2}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function PrChart({
  data,
  domain,
}: {
  data: { recall: number; precision: number }[];
  domain: [number, number];
}) {
  return (
    <div className="h-40 w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="recall" type="number" domain={domain} {...axisProps} />
          <YAxis dataKey="precision" type="number" domain={[0, 1]} {...axisProps} />
          <Tooltip
            formatter={(v) => Number(v).toFixed(4)}
            contentStyle={tooltipContentStyle}
            labelStyle={{ color: TEXT_PRIMARY }}
          />
          <Line
            type="monotone"
            dataKey="precision"
            stroke={AMBER}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            className="trace-glow"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// The recall-zoomed PR inset (domain [0.95, 1]) rendered as a flat line:
// of the 101 points with recall >= 0.95, 61 share the exact same recall
// value (0.9999843464) and another 12 share another -- 72% of the window
// collapse onto ~2 x-positions. The real precision cliff (0.999 -> 0.169)
// happens between that cluster and the single point at recall = 1.0,
// which sits 0.03% of the window's width away -- sub-pixel. No y-axis
// range fixes this; recall genuinely can't separate two points that share
// the same recall value. Plotting the last N threshold steps by rank
// instead of by recall value spaces them evenly regardless of how
// clustered their recall is, which is what actually makes the slope
// visible.
function prTail(prData: { recall: number; precision: number }[], n: number) {
  const tail = prData.slice(-n);
  return tail.map((d, i) => ({
    index: i, // 0 = start of window (left) ... n-1 = final threshold step (right)
    stepsFromEnd: tail.length - 1 - i,
    recall: d.recall,
    precision: d.precision,
  }));
}

function PrTailChart({ data, steps }: { data: ReturnType<typeof prTail>; steps: number }) {
  return (
    <div className="h-40 w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ bottom: 14 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis
            dataKey="index"
            type="number"
            domain={[0, steps - 1]}
            allowDecimals={false}
            {...axisProps}
            tickFormatter={(idx: number) => String(steps - 1 - idx)}
            label={{
              value: "threshold steps from end",
              position: "insideBottom",
              offset: -8,
              fill: TEXT_SECONDARY,
              fontFamily: MONO,
              fontSize: 10,
            }}
          />
          <YAxis dataKey="precision" type="number" domain={[0, 1]} {...axisProps} />
          <Tooltip
            formatter={(value, _name, item) => [
              Number(value).toFixed(4),
              `precision (recall ${item.payload.recall.toFixed(6)})`,
            ]}
            labelFormatter={(idx) => `${steps - 1 - Number(idx)} steps from end`}
            contentStyle={tooltipContentStyle}
            labelStyle={{ color: TEXT_PRIMARY }}
          />
          <Line
            type="monotone"
            dataKey="precision"
            stroke={AMBER}
            strokeWidth={1.5}
            dot={{ r: 2, fill: AMBER, strokeWidth: 0 }}
            isAnimationActive={false}
            className="trace-glow"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PerformancePanel({ info }: { info: ModelInfo }) {
  const rocData = info.roc_curve.fpr.map((fpr, i) => ({ fpr, tpr: info.roc_curve.tpr[i] }));

  // recharts' numeric-axis LineChart expects ascending X order to position
  // its hover tooltip correctly. sklearn's precision_recall_curve returns
  // `recall` in descending order (1.0 -> 0.0) -- unlike roc_curve's `fpr`,
  // which is already ascending -- so this needs an explicit sort, or the
  // PR chart's tooltip doesn't track the cursor correctly.
  const prData = info.pr_curve.recall
    .map((recall, i) => ({ recall, precision: info.pr_curve.precision[i] }))
    .sort((a, b) => a.recall - b.recall);

  const operatingPoint = {
    x: info.operating_threshold.val_fpr_at_threshold,
    y: info.operating_threshold.val_recall_at_threshold,
  };

  const rocZoomData = sliceToDomain(rocData, "fpr", 0, ZOOM_FPR_MAX);
  const prTailData = prTail(prData, PR_TAIL_STEPS);

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
      <Card title="Confusion matrix // test split" className="min-w-0">
        <ConfusionMatrix info={info} />
      </Card>

      <Card title="ROC // attack vs. BENIGN" className="min-w-0">
        <p className="mb-1 font-mono text-xs text-text-secondary">full range</p>
        <RocChart data={rocData} domain={[0, 1]} operatingPoint={operatingPoint} />
        <p className="mt-3 mb-1 font-mono text-xs text-text-secondary">
          zoomed &middot; fpr 0&ndash;{ZOOM_FPR_MAX}
        </p>
        <RocChart data={rocZoomData} domain={[0, ZOOM_FPR_MAX]} operatingPoint={operatingPoint} />
        <p className="mt-2 font-mono text-xs text-text-secondary">
          marker: operating threshold ({info.operating_threshold.threshold.toFixed(5)}, fpr budget{" "}
          {info.operating_threshold.fpr_budget})
        </p>
      </Card>

      <Card title="Precision / recall // attack vs. BENIGN" className="min-w-0">
        <p className="mb-1 font-mono text-xs text-text-secondary">full range</p>
        <PrChart data={prData} domain={[0, 1]} />
        <p className="mt-3 mb-1 font-mono text-xs text-text-secondary">
          zoomed &middot; last {PR_TAIL_STEPS} threshold steps, indexed by rank &mdash; not recall
        </p>
        <PrTailChart data={prTailData} steps={PR_TAIL_STEPS} />
        <p className="mt-2 font-mono text-xs text-text-secondary">
          Precision drops sharply from 0.999 to 0.169 in the final few threshold steps &mdash; chasing
          the last bit of recall costs significant accuracy.
        </p>
      </Card>
    </div>
  );
}
