"use client";

const SPEED_OPTIONS = [1, 5, 20, 50] as const;

export function ReplayControls({
  playing,
  onTogglePlay,
  onRestart,
  rowsPerSecond,
  onSpeedChange,
  revealed,
  total,
  disabled = false,
}: {
  playing: boolean;
  onTogglePlay: () => void;
  onRestart: () => void;
  rowsPerSecond: number;
  onSpeedChange: (n: number) => void;
  revealed: number;
  total: number;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-4 font-mono text-xs">
      <button
        onClick={onTogglePlay}
        disabled={disabled}
        className="border border-border bg-bg px-4 py-1.5 font-medium tracking-widest text-text-primary uppercase transition-colors hover:border-amber hover:text-amber focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-border disabled:hover:text-text-primary"
      >
        {playing ? "Pause" : disabled ? "Loading" : "Play"}
      </button>
      <button
        onClick={onRestart}
        disabled={disabled || revealed === 0}
        className="border border-border bg-bg px-4 py-1.5 font-medium tracking-widest text-text-secondary uppercase transition-colors hover:border-amber hover:text-amber focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-border disabled:hover:text-text-secondary"
      >
        Restart
      </button>
      <label className="flex items-center gap-2 text-text-secondary">
        <span className="tracking-widest uppercase">Speed</span>
        <select
          value={rowsPerSecond}
          onChange={(e) => onSpeedChange(Number(e.target.value))}
          className="border border-border bg-bg px-2 py-1 text-text-primary focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none"
        >
          {SPEED_OPTIONS.map((n) => (
            <option key={n} value={n}>
              {n} rows/s
            </option>
          ))}
        </select>
      </label>
      <span className="text-text-secondary tabular-nums">
        {revealed} / {total} flows replayed
      </span>
    </div>
  );
}
