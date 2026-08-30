import inspect
import shutil
from pathlib import Path

import pandas as pd
import pytest

import src.preprocessing.split as split_module
from src.preprocessing.clean import clean_flows, load_raw_csvs, normalize_labels
from src.preprocessing.split import stratified_group_split

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_flows.csv"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cleaned_df(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "sample_flows.csv")
    df = load_raw_csvs(tmp_path)
    df = normalize_labels(df)
    df, _ = clean_flows(df)
    return df


def _row_identity(df: pd.DataFrame) -> pd.Series:
    """Full-row content hash used as a leakage-detection identity, since
    this CICIDS2017 export has no Flow ID / IP / Timestamp columns."""
    feature_cols = [c for c in df.columns if c != "source_day"]
    return pd.util.hash_pandas_object(df[feature_cols], index=False)


def test_zero_overlap_between_splits(cleaned_df):
    train_df, val_df, test_df = stratified_group_split(cleaned_df, seed=42)

    train_ids = set(_row_identity(train_df))
    val_ids = set(_row_identity(val_df))
    test_ids = set(_row_identity(test_df))

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_every_input_class_present_in_train(cleaned_df):
    train_df, _, _ = stratified_group_split(cleaned_df, seed=42)

    input_classes = set(cleaned_df["label_family"].unique())
    train_classes = set(train_df["label_family"].unique())

    assert input_classes == train_classes


def test_split_row_counts_sum_to_input(cleaned_df):
    train_df, val_df, test_df = stratified_group_split(cleaned_df, seed=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(cleaned_df)


def test_split_proportions_approximate_ratios(cleaned_df):
    train_df, val_df, test_df = stratified_group_split(
        cleaned_df, ratios=(0.7, 0.15, 0.15), seed=42
    )
    total = len(cleaned_df)

    assert len(train_df) / total == pytest.approx(0.7, abs=0.1)
    assert len(val_df) / total == pytest.approx(0.15, abs=0.1)
    assert len(test_df) / total == pytest.approx(0.15, abs=0.1)


def test_split_module_has_no_preprocessor_fitting_dependency():
    """Cycle 1's split step must not depend on or trigger any fitted
    scaler/preprocessor -- that logic belongs to Cycle 2's pipeline.py,
    fit strictly after the split. Asserted structurally (no import/name
    coupling to fitting code) rather than by checking whether
    artifacts/preprocessor/preprocessor_v1.pkl exists on disk, since once
    Cycle 2 has legitimately run that file exists from then on."""
    source = inspect.getsource(split_module)
    for forbidden in ("Preprocessor", "RobustScaler", "SimpleImputer", "sklearn"):
        assert forbidden not in source, f"split.py must not reference {forbidden}"
