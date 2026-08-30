"""Cycle 3 — training orchestration: RF baseline + XGBoost primary model,
class-imbalance handling, evaluation, and MODEL_CARD.md generation.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SMOTE_MIN_SAMPLES,
    SMOTE_TARGET_COUNTS,
)
from src.models.baseline_rf import train_rf
from src.models.evaluate import compute_metrics
from src.models.xgboost_model import train_xgboost

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_split(name: str, processed_dir: Path = PROCESSED_DIR):
    data = np.load(processed_dir / f"{name}_features.npz", allow_pickle=True)
    return data["X"], data["y"], list(data["feature_names"])


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, label_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE for genuinely rare classes only (MAD 3.2), run strictly after
    the split, on the training fold alone. Classes below SMOTE_MIN_SAMPLES
    (e.g. Heartbleed, 8 train rows) are left alone -- oversampling can't
    manufacture signal that isn't there, and MAD 3.2 says not to pretend
    otherwise. Returns the resampled (X, y); callers must derive any
    balanced sample_weight from the *returned* y, not the original."""
    id_to_label = {v: k for k, v in label_map.items()}
    counts = {i: int((y_train == i).sum()) for i in id_to_label}

    sampling_strategy = {}
    for class_name, target in SMOTE_TARGET_COUNTS.items():
        class_id = label_map[class_name]
        current = counts[class_id]
        if current < SMOTE_MIN_SAMPLES:
            continue
        if current >= target:
            continue
        sampling_strategy[class_id] = target

    if not sampling_strategy:
        return X_train, y_train

    min_current = min(counts[cid] for cid in sampling_strategy)
    k_neighbors = min(5, min_current - 1)
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=RANDOM_SEED,
    )
    return smote.fit_resample(X_train, y_train)


def run(
    processed_dir: Path = PROCESSED_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict:
    X_train, y_train, feature_names = _load_split("train", processed_dir)
    X_val, y_val, _ = _load_split("val", processed_dir)
    X_test, y_test, _ = _load_split("test", processed_dir)

    label_map = json.loads((artifacts_dir / "label_map.json").read_text())
    num_class = len(label_map)

    # SMOTE first, THEN derive sample weights from the resampled labels --
    # weighting on the pre-SMOTE y_train would double-correct Infiltration
    # (once via oversampling, again via a stale scarcity-based weight) and
    # would also mismatch X_train_res's row count. See src/config.py.
    X_train_res, y_train_res = apply_smote(X_train, y_train, label_map)
    sample_weight = compute_sample_weight("balanced", y_train_res)

    print(f"train: {X_train.shape} -> after SMOTE: {X_train_res.shape}")

    rf_model = train_rf(X_train_res, y_train_res)
    xgb_model = train_xgboost(X_train_res, y_train_res, X_val, y_val, sample_weight, num_class)

    results = {}
    for model_name, model in [("rf_baseline", rf_model), ("xgboost", xgb_model)]:
        split_metrics = {}
        for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            split_metrics[split_name] = compute_metrics(y, y_pred, y_proba, label_map)
        results[model_name] = split_metrics
        print(
            f"{model_name}: test macro_recall_stable_classes="
            f"{split_metrics['test']['macro_recall_stable_classes']:.4f}, "
            f"macro_pr_auc={split_metrics['test']['macro_pr_auc']:.4f}"
        )

    commit_hash = _git_commit_hash()

    # every class in train, whether or not it was SMOTE-eligible, so the
    # <10-sample exclusion (e.g. Heartbleed) is recorded even though it was
    # never a SMOTE_TARGET_COUNTS entry to begin with
    excluded = sorted(
        name
        for name, class_id in label_map.items()
        if int((y_train == class_id).sum()) < SMOTE_MIN_SAMPLES
    )
    imbalance_config = {
        "class_weight": "balanced",
        "smote_target_counts": SMOTE_TARGET_COUNTS,
        "smote_min_samples": SMOTE_MIN_SAMPLES,
        "smote_excluded_classes": excluded,
    }

    xgboost_dir = models_dir / "xgboost_v1"
    xgboost_dir.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(xgboost_dir / "model.json")
    (xgboost_dir / "feature_list.json").write_text(json.dumps(feature_names, indent=2))
    xgboost_metrics = {
        "model": "xgboost_v1",
        "commit_hash": commit_hash,
        "feature_count": len(feature_names),
        "imbalance_config": imbalance_config,
        "val": results["xgboost"]["val"],
        "test": results["xgboost"]["test"],
    }
    (xgboost_dir / "metrics.json").write_text(json.dumps(xgboost_metrics, indent=2))

    rf_dir = models_dir / "rf_baseline_v1"
    rf_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, rf_dir / "model.pkl")
    rf_metrics = {
        "model": "rf_baseline_v1",
        "commit_hash": commit_hash,
        "feature_count": len(feature_names),
        "imbalance_config": {"class_weight": "balanced"},
        "val": results["rf_baseline"]["val"],
        "test": results["rf_baseline"]["test"],
    }
    (rf_dir / "metrics.json").write_text(json.dumps(rf_metrics, indent=2))

    return {"xgboost": xgboost_metrics, "rf_baseline": rf_metrics, "label_map": label_map}


if __name__ == "__main__":
    run()
