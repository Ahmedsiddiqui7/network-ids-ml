"""Cycle 3 — Random Forest baseline: fast, interpretable, a sanity floor
for the primary XGBoost model (Master Architecture Document 3.2)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_SEED, RF_BASELINE_MAX_DEPTH, RF_BASELINE_N_ESTIMATORS


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=RF_BASELINE_N_ESTIMATORS,
        max_depth=RF_BASELINE_MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf
