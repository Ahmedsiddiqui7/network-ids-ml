"use client";

type Status = "connecting" | "idle" | "replaying" | "paused" | "complete";

function statusLabel(status: Status): string {
  switch (status) {
    case "connecting":
      return "connecting";
    case "idle":
      return "idle";
    case "replaying":
      return "replaying";
    case "paused":
      return "paused";
    case "complete":
      return "complete";
  }
}

export function TopBar({
  modelVersion,
  commitHash,
  status,
}: {
  modelVersion: string | null;
  commitHash: string | null;
  status: Status;
}) {
  return (
    <div className="sticky top-0 z-10 border-b border-border bg-panel">
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 font-mono text-xs">
        <div className="flex items-center gap-3">
          <span className="font-medium tracking-widest text-text-primary uppercase">
            NIDS &middot; Replay Console
          </span>
          {modelVersion && (
            <span className="text-text-secondary">
              {modelVersion}
              {commitHash && <span className="text-text-secondary/70"> @{commitHash.slice(0, 8)}</span>}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-text-secondary">
          <span
            className={`h-1.5 w-1.5 rounded-full bg-text-secondary ${
              status === "replaying" ? "motion-safe:animate-pulse" : ""
            }`}
            aria-hidden="true"
          />
          <span className="tracking-widest uppercase">{statusLabel(status)}</span>
        </div>
      </div>
    </div>
  );
}
