import { NextResponse } from "next/server";
import replayFlows from "@/data/replay_flows.json";
import { apiBaseUrl } from "@/app/lib/apiBase";
import type { PredictionResponse, ReplayRow } from "@/app/lib/types";

// Cycle 5 (MAD 3.4): calls the Cycle 4 API's existing /predict/batch ONCE
// with the whole replay sample -- no API changes needed. The client
// reveals these rows on a timer (the "replay speed" control); this route
// just supplies the real, already-computed predictions to reveal.
export async function GET() {
  const flows = replayFlows.map((row) => row.features);

  const res = await fetch(`${apiBaseUrl()}/predict/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ flows }),
    cache: "no-store",
  });

  if (!res.ok) {
    const detail = await res.text();
    return NextResponse.json(
      { error: "upstream /predict/batch failed", status: res.status, detail },
      { status: 502 },
    );
  }

  const body: { predictions: PredictionResponse[] } = await res.json();

  const rows: ReplayRow[] = body.predictions.map((prediction, i) => ({
    ...prediction,
    id: replayFlows[i].id,
    true_label: replayFlows[i].true_label,
    correct: prediction.prediction === replayFlows[i].true_label,
  }));

  return NextResponse.json({ rows });
}
