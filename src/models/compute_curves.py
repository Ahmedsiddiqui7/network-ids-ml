"""Cycle 5 -- supplies ROC/PR curve coordinates for the dashboard's
model-performance panel (MAD 3.4). metrics.json (Cycle 3) only has scalar
per-class PR-AUC/ROC-AUC, no curve points to plot. Loads the already-trained
xgboost_v1 model and test_features.npz; does not retrain.

Binary framing (attack vs. BENIGN), not per-class: this matches the
framing already established by aggregate_benign_fpr (src/models/evaluate.py)
and the Cycle 4 operating threshold (risk_score = 1 - P(BENIGN), see
src/models/tune_threshold.py) -- so the dashboard can mark the actual
serving-time operating point on this same ROC curve.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR

CURVE_POINTS = 150


def _thin(*arrays: np.ndarray, n_points: int = CURVE_POINTS) -> list[np.ndarray]:
    length = len(arrays[0])
    if length <= n_points:
        idx = np.arange(length)
    else:
        idx = np.linspace(0, length - 1, n_points).round().astype(int)
    return [arr[idx] for arr in arrays]


def run(
    artifacts_dir: Path = ARTIFACTS_DIR,
    models_dir: Path = MODELS_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> dict:
    xgboost_dir = models_dir / "xgboost_v1"
    label_map = json.loads((artifacts_dir / "label_map.json").read_text())
    benign_id = label_map["BENIGN"]

    model = xgb.XGBClassifier()
    model.load_model(xgboost_dir / "model.json")

    test = np.load(processed_dir / "test_features.npz")
    X_test, y_test = test["X"], test["y"]

    proba = model.predict_proba(X_test)
    p_attack = 1.0 - proba[:, benign_id]
    y_binary = (y_test != benign_id).astype(int)

    fpr, tpr, _ = roc_curve(y_binary, p_attack)
    precision, recall, _ = precision_recall_curve(y_binary, p_attack)

    fpr_t, tpr_t = _thin(fpr, tpr)
    precision_t, recall_t = _thin(precision, recall)

    result = {
        "roc": {"fpr": fpr_t.tolist(), "tpr": tpr_t.tolist()},
        "pr": {"precision": precision_t.tolist(), "recall": recall_t.tolist()},
    }

    (xgboost_dir / "curves.json").write_text(json.dumps(result))
    print(f"wrote {xgboost_dir / 'curves.json'} ({len(fpr_t)} ROC pts, {len(precision_t)} PR pts)")
    return result


if __name__ == "__main__":
    run()
