"""Cycle 3 — primary XGBoost model (Master Architecture Document 3.2).

eval_metric deviates from the MAD's literal 'aucpr' (PR-AUC) instruction:
that's a binary-only XGBoost metric and this is a 9-class problem, so
'mlogloss' is used for early stopping instead -- the standard multi-class
substitute. The metrics actually reported (Pillar 4) are computed
independently via sklearn after training and still lead with PR-AUC.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from src.config import (
    RANDOM_SEED,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_EVAL_METRIC,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)


def build_xgboost(num_class: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        eval_metric=XGB_EVAL_METRIC,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray,
    num_class: int,
) -> xgb.XGBClassifier:
    model = build_xgboost(num_class)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model
