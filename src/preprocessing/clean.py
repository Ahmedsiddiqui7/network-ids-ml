"""Cycle 1 — load, clean, and deduplicate raw CICIDS2017 CSVs.

No imputation and no scaling happens here: those steps must be fit on the
training fold only (Cycle 2), so doing them before the split would leak
val/test statistics into training.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"

RATE_COLUMNS = ["Flow Bytes/s", "Flow Packets/s"]

# Raw label -> attack family, per Master Architecture Document 2.4.
# Raw labels are matched case-sensitively after whitespace/mojibake cleanup.
LABEL_FAMILY_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack Brute Force": "Web Attack",
    "Web Attack XSS": "Web Attack",
    "Web Attack Sql Injection": "Web Attack",
    "Infiltration": "Infiltration",
    "Bot": "Botnet",
    "Heartbleed": "Heartbleed",
}


def _source_day(csv_path: Path) -> str:
    return csv_path.stem.replace(".pcap_ISCX", "")


def load_raw_csvs(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every CSV in raw_dir, strip header whitespace, tag with source_day."""
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["source_day"] = _source_day(path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _normalize_label_text(raw_label: str) -> str:
    label = raw_label.strip()
    # Collapse any run of non-ASCII / mojibake filler (e.g. a lost en-dash
    # decoded as "\x96" or "�") between words into a single space.
    label = re.sub(r"[^\x00-\x7F]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Fix mojibake/whitespace in Label and add a label_family column (2.4)."""
    df = df.copy()
    df["Label"] = df["Label"].astype(str).map(_normalize_label_text)

    unmapped = set(df["Label"].unique()) - set(LABEL_FAMILY_MAP.keys())
    if unmapped:
        raise ValueError(f"Unmapped labels found, update LABEL_FAMILY_MAP: {unmapped}")

    df = df.rename(columns={"Label": "label_detailed"})
    df["label_family"] = df["label_detailed"].map(LABEL_FAMILY_MAP)
    return df


def _class_counts(df: pd.DataFrame, label_col: str = "label_family") -> dict:
    return df[label_col].value_counts().to_dict()


def clean_flows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replace Inf with NaN in rate columns and drop exact duplicate rows.

    Returns the cleaned dataframe plus a report dict with per-class counts
    before and after, so a silent wipeout of a rare class is visible.
    """
    report = {"before": _class_counts(df)}

    df = df.copy()
    present_rate_cols = [c for c in RATE_COLUMNS if c in df.columns]
    df[present_rate_cols] = df[present_rate_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    dedup_subset = [c for c in df.columns if c != "source_day"]
    df = df.drop_duplicates(subset=dedup_subset).reset_index(drop=True)

    report["after"] = _class_counts(df)

    zeroed = [k for k, v in report["before"].items() if report["after"].get(k, 0) == 0]
    if zeroed:
        raise RuntimeError(f"Cleaning zeroed out class(es): {zeroed}")

    return df, report


def run(raw_dir: Path = RAW_DIR, interim_dir: Path = INTERIM_DIR) -> pd.DataFrame:
    interim_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_csvs(raw_dir)
    df = normalize_labels(df)
    df, report = clean_flows(df)

    df.to_parquet(interim_dir / "cleaned.parquet", index=False)
    with open(interim_dir / "class_balance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Class balance before -> after cleaning:")
    for label, before_count in report["before"].items():
        after_count = report["after"].get(label, 0)
        print(f"  {label:15s} {before_count:>8d} -> {after_count:>8d}")

    return df


if __name__ == "__main__":
    run()
