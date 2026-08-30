"""Cycle 3 verification: regression floor on the primary model's
macro-recall, and the day-based leakage smell test threshold.

Both depend on a real training run's artifacts (a synthetic fixture can't
stand in for "does the actual trained model still clear the bar") -- skip
with a clear message if they haven't been produced yet.
"""
import json

import numpy as np
import pytest
import xgboost as xgb

from src.config import ARTIFACTS_DIR, MODELS_DIR
from src.models.evaluate import compute_metrics

XGBOOST_DIR = MODELS_DIR / "xgboost_v1"

# Set from this implementation's own first successful training run
# (2026-08-30): observed macro_recall_stable_classes on the held-out test
# split was 0.99793. Floor set with a safety margin below that observed
# value, per MAD Cycle 3's "a floor you set from your own first successful
# run, not an externally assumed number."
MACRO_RECALL_STABLE_FLOOR = 0.97


@pytest.fixture
def label_map():
    path = ARTIFACTS_DIR / "label_map.json"
    if not path.exists():
        pytest.skip(f"{path} not found -- run `python -m src.preprocessing.pipeline` first")
    return json.loads(path.read_text())


def _require(path):
    if not path.exists():
        pytest.skip(f"{path} not found -- run `python -m src.models.train` first")


def test_macro_recall_above_floor(label_map):
    model_path = XGBOOST_DIR / "model.json"
    features_path = MODELS_DIR.parents[1] / "data" / "processed" / "test_features.npz"
    _require(model_path)
    _require(features_path)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    data = np.load(features_path)
    X_test, y_test = data["X"], data["y"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba, label_map)

    assert metrics["macro_recall_stable_classes"] >= MACRO_RECALL_STABLE_FLOOR


def test_leakage_smell_test_gap_within_threshold():
    smell_test_path = XGBOOST_DIR / "leakage_smell_test.json"
    _require(smell_test_path)

    result = json.loads(smell_test_path.read_text())

    assert result["absolute_gap"] <= result["threshold"], (
        f"day-based leakage smell test flagged: stratified-test FPR="
        f"{result['stratified_test_fpr']:.4f}, Monday-LODO FPR="
        f"{result['lodo_monday_fpr']:.4f}, gap={result['absolute_gap']:.4f} "
        f"exceeds threshold={result['threshold']}"
    )
    assert result["flagged"] is False
