"""Cycle 4 — implements Pillar 4.3's deferred threshold-tuning decision:
pick a serving-time operating threshold on the validation set that
maximizes recall subject to an explicit FPR budget, rather than the
default 0.5 per-class cutoff. Loads the already-trained xgboost_v1 model;
does not retrain.

Deliberate deviation from 4.3's literal "use the validation-set PR curve":
`sklearn.metrics.precision_recall_curve` returns (precision, recall,
thresholds) -- it has no FPR axis at all. The stated goal is "maximize
recall subject to an explicit FPR budget", which is a constraint on FPR
directly. Only `roc_curve`'s (fpr, tpr, thresholds) output lets you filter
candidate thresholds by FPR and then pick the best recall (tpr) among
survivors -- picking a threshold off the PR curve would mean choosing by
precision and hoping it happens to also satisfy the FPR budget, not
actually enforcing it. PR-AUC still leads in *evaluation reporting* (4.2,
see src/models/evaluate.py); this is specifically about *threshold
selection*, where FPR-budget-satisfaction requires the ROC curve's axes.
Same treatment as the aucpr -> mlogloss deviation documented in
src/config.py for Cycle 3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve

from src.config import ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR, THRESHOLD_FPR_BUDGET


def choose_threshold(y_binary: np.ndarray, p_attack: np.ndarray, fpr_budget: float) -> dict:
    fpr, tpr, thresholds = roc_curve(y_binary, p_attack)
    within_budget = fpr <= fpr_budget
    if not within_budget.any():
        raise ValueError(f"no threshold achieves FPR <= {fpr_budget}")

    best = np.argmax(tpr[within_budget])
    return {
        "fpr_budget": fpr_budget,
        "threshold": float(thresholds[within_budget][best]),
        "val_recall_at_threshold": float(tpr[within_budget][best]),
        "val_fpr_at_threshold": float(fpr[within_budget][best]),
    }


def run(
    artifacts_dir: Path = ARTIFACTS_DIR,
    models_dir: Path = MODELS_DIR,
    processed_dir: Path = PROCESSED_DIR,
    fpr_budget: float = THRESHOLD_FPR_BUDGET,
) -> dict:
    xgboost_dir = models_dir / "xgboost_v1"
    label_map = json.loads((artifacts_dir / "label_map.json").read_text())
    benign_id = label_map["BENIGN"]

    model = xgb.XGBClassifier()
    model.load_model(xgboost_dir / "model.json")

    val = np.load(processed_dir / "val_features.npz")
    X_val, y_val = val["X"], val["y"]

    proba = model.predict_proba(X_val)
    p_attack = 1.0 - proba[:, benign_id]
    y_binary = (y_val != benign_id).astype(int)

    result = choose_threshold(y_binary, p_attack, fpr_budget)

    (xgboost_dir / "threshold.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
