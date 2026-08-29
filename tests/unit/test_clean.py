import shutil
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.clean import clean_flows, load_raw_csvs, normalize_labels

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_flows.csv"


@pytest.fixture
def raw_dir(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "sample_flows.csv")
    return tmp_path


@pytest.fixture
def loaded_df(raw_dir):
    return load_raw_csvs(raw_dir)


def test_headers_are_stripped_of_whitespace(loaded_df):
    assert "Destination Port" in loaded_df.columns
    assert " Destination Port" not in loaded_df.columns
    assert "Label" in loaded_df.columns


def test_source_day_tag_added(loaded_df):
    assert "source_day" in loaded_df.columns
    assert (loaded_df["source_day"] == "sample_flows").all()


def test_mojibake_label_normalizes_to_web_attack_family(loaded_df):
    df = normalize_labels(loaded_df)
    web_attack_rows = df[df["label_detailed"] == "Web Attack XSS"]
    assert len(web_attack_rows) > 0
    assert (web_attack_rows["label_family"] == "Web Attack").all()


def test_inf_in_rate_columns_becomes_nan_not_dropped(loaded_df):
    df = normalize_labels(loaded_df)
    n_before = len(df)
    assert np.isinf(df["Flow Bytes/s"]).any()

    cleaned, _ = clean_flows(df)

    assert not np.isinf(cleaned["Flow Bytes/s"]).any()
    assert cleaned["Flow Bytes/s"].isna().any()
    # the Inf row itself must survive cleaning, just with NaN in place
    assert len(cleaned) >= n_before - 3  # minus the intentional exact duplicates


def test_exact_duplicate_rows_are_removed(loaded_df):
    df = normalize_labels(loaded_df)
    n_before = len(df)

    cleaned, report = clean_flows(df)

    assert len(cleaned) < n_before
    assert not cleaned.duplicated(subset=[c for c in cleaned.columns if c != "source_day"]).any()


def test_no_class_row_count_drops_to_zero_after_cleaning(loaded_df):
    df = normalize_labels(loaded_df)
    _, report = clean_flows(df)

    for label, before_count in report["before"].items():
        assert before_count > 0
        assert report["after"].get(label, 0) > 0, f"class {label} was wiped out by cleaning"


def test_rare_class_survives_cleaning(loaded_df):
    df = normalize_labels(loaded_df)
    _, report = clean_flows(df)

    assert report["after"].get("Heartbleed", 0) > 0
