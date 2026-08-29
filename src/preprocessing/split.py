"""Cycle 1 — leak-free train/val/test split.

Per the Master Architecture Document 2.5, a naive random row shuffle can
inflate measured performance. This dataset export has no Flow ID / IP /
Timestamp columns to build a true session-level group split from, and
several attack classes exist in only one source day (e.g. Infiltration is
36 rows, all from Thursday-Afternoon) so a whole-day holdout would erase
classes from train or test entirely. Instead we stratify by
(source_day, label_family): every (day, class) combination is split
proportionally across train/val/test, and leakage is guarded by the
full-dataset deduplication clean.py already performed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

DEFAULT_RATIOS = (0.7, 0.15, 0.15)


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str = "source_day",
    label_col: str = "label_family",
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df within each (group_col, label_col) stratum.

    Strata too small to place at least one row in every split fall back to
    a best-effort proportional assignment rather than raising, since
    genuinely rare classes (e.g. Heartbleed=11 rows) can't be split three
    ways cleanly — this is documented in the class-balance report, not
    silently hidden.
    """
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    rng = np.random.default_rng(seed)

    train_parts, val_parts, test_parts = [], [], []
    for _, stratum in df.groupby([group_col, label_col], sort=False):
        n = len(stratum)
        idx = rng.permutation(n)
        shuffled = stratum.iloc[idx]

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # remainder goes to test, guarantees all rows are assigned
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train : n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val :])

    def _finalize(parts: list[pd.DataFrame]) -> pd.DataFrame:
        out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return _finalize(train_parts), _finalize(val_parts), _finalize(test_parts)


def run(
    interim_dir: Path = INTERIM_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(interim_dir / "cleaned.parquet")
    train_df, val_df, test_df = stratified_group_split(df)

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    split_counts = {
        "train": train_df["label_family"].value_counts().to_dict(),
        "val": val_df["label_family"].value_counts().to_dict(),
        "test": test_df["label_family"].value_counts().to_dict(),
    }

    report_path = interim_dir / "class_balance_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["splits"] = split_counts
    report_path.write_text(json.dumps(report, indent=2))

    print(f"train: {len(train_df)} rows, val: {len(val_df)} rows, test: {len(test_df)} rows")
    for split_name, counts in split_counts.items():
        print(f"  {split_name}: {counts}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    run()
