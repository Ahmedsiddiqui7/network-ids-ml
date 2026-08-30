"""Cycle 5 -- builds the small, checked-in replay fixture the dashboard
replays (MAD 3.4: "a table of flows replayed from the held-out test set").

data/processed/test.parquet is gitignored (the real dataset) and won't
exist inside the dashboard's Docker image, so this draws a small,
fixed-seed, stratified-by-class sample from it and writes a self-contained
JSON fixture that IS checked in -- small and derived, same category as
tests/fixtures/sample_flows.csv already being an exception to the
gitignore's blanket data/ rule.

Deliberately reads data/processed/test.parquet, not train or val: this is
the held-out split Cycle 5's own roadmap entry names, and per-split
Heartbleed counts differ (train=8, val=2, test=1) -- sampling from the
wrong split would silently pull from data the model was fit on or
selected against, and change which rows "Heartbleed: 1 (all)" refers to.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, REPO_ROOT

SAMPLE_COUNTS = {
    "BENIGN": 40,
    "DoS": 25,
    "DDoS": 25,
    "PortScan": 25,
    "Brute Force": 20,
    "Web Attack": 15,
    "Botnet": 15,
    "Infiltration": 6,
    "Heartbleed": 1,
}

OUTPUT_PATH = REPO_ROOT / "src" / "dashboard" / "data" / "replay_flows.json"


def build_sample(
    test_df: pd.DataFrame, feature_list: list[str], seed: int = RANDOM_SEED
) -> list[dict]:
    rows = []
    for label, count in SAMPLE_COUNTS.items():
        class_df = test_df[test_df["label_family"] == label]
        n = min(count, len(class_df))
        sampled = class_df.sample(n=n, random_state=seed)
        for i, (_, row) in enumerate(sampled.iterrows()):
            rows.append(
                {
                    "id": f"{label.replace(' ', '_').lower()}-{i}",
                    "true_label": label,
                    "features": {f: float(row[f]) for f in feature_list},
                }
            )
    # Shuffle across classes -- without this, rows sit in solid per-class
    # blocks (SAMPLE_COUNTS' iteration order), so the dashboard's replay
    # (which reveals a growing prefix of this list) shows only the first
    # class or two until playback is nearly done. Each row moves as a
    # complete {id, true_label, features} unit, so this only reorders --
    # it can't cross-contaminate a row's own fields.
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return rows


def run(
    processed_dir=PROCESSED_DIR,
    models_dir=MODELS_DIR,
    output_path=OUTPUT_PATH,
) -> list[dict]:
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    feature_list = json.loads((models_dir / "xgboost_v1" / "feature_list.json").read_text())

    rows = build_sample(test_df, feature_list)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {output_path} ({len(rows)} flows)")
    return rows


if __name__ == "__main__":
    run()
