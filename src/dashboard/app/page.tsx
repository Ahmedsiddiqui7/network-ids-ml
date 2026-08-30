"use client";

import { useEffect, useRef, useState } from "react";
import { AttackDistributionChart } from "@/app/components/AttackDistributionChart";
import { Card } from "@/app/components/Card";
import { PerformancePanel } from "@/app/components/PerformancePanel";
import { ReplayControls } from "@/app/components/ReplayControls";
import { ReplayTable } from "@/app/components/ReplayTable";
import { TopBar } from "@/app/components/TopBar";
import type { ModelInfo, ReplayRow } from "@/app/lib/types";

export default function Home() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [allRows, setAllRows] = useState<ReplayRow[]>([]);
  const [revealedCount, setRevealedCount] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rowsPerSecond, setRowsPerSecond] = useState(5);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Kept in sync via its own effect (ref writes aren't allowed during
  // render) so the interval callback below always reads the latest count
  // instead of a stale value from whenever that effect last ran.
  const revealedCountRef = useRef(revealedCount);
  useEffect(() => {
    revealedCountRef.current = revealedCount;
  }, [revealedCount]);

  useEffect(() => {
    fetch("/api/model-info")
      .then((res) => res.json())
      .then(setInfo)
      .catch((err) => setError(`Failed to load model info: ${err}`));

    fetch("/api/replay")
      .then((res) => res.json())
      .then((body) => {
        if (body.error) {
          setError(`Failed to load replay data: ${body.error}`);
          return;
        }
        setAllRows(body.rows);
      })
      .catch((err) => setError(`Failed to load replay data: ${err}`));
  }, []);

  // setRevealedCount and setPlaying are called as independent, sibling
  // statements inside the setInterval callback -- an external-system
  // callback, which is exactly where React's own guidance says setState
  // belongs (https://react.dev/learn/you-might-not-need-an-effect).
  // Two things this deliberately avoids, both of which produced
  // "Maximum update depth exceeded": calling setPlaying from inside
  // setRevealedCount's functional updater (updater functions can be
  // invoked more than once -- Strict Mode double-invokes them
  // specifically to catch impurities like this), and calling setPlaying
  // synchronously in a useEffect body watching revealedCount (a separate
  // anti-pattern the react-hooks lint rule flags directly). Reading
  // revealedCountRef instead of the closed-over revealedCount avoids a
  // stale value, since this effect only re-runs when playing/speed/total
  // change, not on every tick.
  useEffect(() => {
    // Belt-and-suspenders: clear any interval this ref might still be
    // holding *before* deciding whether to create a new one, not just in
    // the cleanup function. A stray leftover interval here (from dev-mode
    // Fast Refresh not tearing down cleanly across repeated edits, or any
    // other path that skips the effect's own cleanup) would run alongside
    // a freshly-created one, and two overlapping intervals independently
    // calling setRevealedCount is exactly the kind of thing that can
    // eventually cascade into "Maximum update depth exceeded" -- not on
    // every tick, only when their firings happen to land close together,
    // which matches it showing up late and inconsistently rather than at
    // a fixed point.
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (!playing) {
      return;
    }
    intervalRef.current = setInterval(() => {
      const next = Math.min(revealedCountRef.current + 1, allRows.length);
      setRevealedCount(next);
      if (next >= allRows.length) {
        setPlaying(false);
      }
    }, 1000 / rowsPerSecond);
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [playing, rowsPerSecond, allRows.length]);

  const revealedRows = allRows.slice(0, revealedCount);

  // Recharts' CartesianAxis has a known issue with data changing faster
  // than React can settle (recharts issue #7563, confirmed in its own
  // source comment) -- at 50 rows/s the raw tick rate (20ms) reproduces
  // "Maximum update depth exceeded" inside AttackDistributionChart's
  // XAxis. useDeferredValue was tried first and wasn't sufficient: it
  // only defers when React perceives actual scheduling contention, which
  // isn't guaranteed -- on a machine fast enough to keep up with 20ms
  // ticks, there's no backlog for it to defer, so the chart still
  // re-renders on nearly every tick and still hits the recharts bug.
  // A fixed-rate poll is deterministic instead of heuristic: the chart's
  // data updates on its own timer, decoupled from rowsPerSecond entirely,
  // so it never sees updates faster than this regardless of replay speed.
  // Scoped to the chart only -- ReplayTable isn't a recharts consumer and
  // should keep showing every row immediately, since the log's
  // completeness matters there.
  const CHART_THROTTLE_MS = 200;
  const revealedRowsRef = useRef(revealedRows);
  useEffect(() => {
    revealedRowsRef.current = revealedRows;
  }, [revealedRows]);
  const [throttledRevealedRows, setThrottledRevealedRows] = useState(revealedRows);
  useEffect(() => {
    const id = setInterval(() => {
      setThrottledRevealedRows((prev) =>
        prev.length === revealedRowsRef.current.length ? prev : revealedRowsRef.current,
      );
    }, CHART_THROTTLE_MS);
    return () => clearInterval(id);
  }, []);

  const handleRestart = () => {
    setRevealedCount(0);
    setPlaying(true);
  };

  const status =
    allRows.length === 0
      ? "connecting"
      : playing
        ? "replaying"
        : revealedCount === 0
          ? "idle"
          : revealedCount === allRows.length
            ? "complete"
            : "paused";

  return (
    <>
      <TopBar modelVersion={info?.model_version ?? null} commitHash={info?.commit_hash ?? null} status={status} />

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-3 p-4">
        {error && (
          <div className="border border-amber/40 bg-amber/10 px-3 py-2 font-mono text-xs text-amber">
            {error}
          </div>
        )}

        <Card>
          <ReplayControls
            playing={playing}
            onTogglePlay={() => setPlaying((p) => !p)}
            onRestart={handleRestart}
            rowsPerSecond={rowsPerSecond}
            onSpeedChange={setRowsPerSecond}
            revealed={revealedCount}
            total={allRows.length}
            disabled={allRows.length === 0}
          />
        </Card>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          {/* min-w-0 is required here: CSS grid items default to min-width:
              auto, so without it the grid track won't shrink below the
              table's intrinsic content width, and the table's own
              overflow-x-auto never gets a chance to engage. */}
          <Card title="Replay feed" className="min-w-0 lg:col-span-2">
            <ReplayTable rows={revealedRows} />
          </Card>
          <Card title="Predicted distribution" className="min-w-0">
            <AttackDistributionChart rows={throttledRevealedRows} />
          </Card>
        </div>

        {info && <PerformancePanel info={info} />}
      </main>
    </>
  );
}
