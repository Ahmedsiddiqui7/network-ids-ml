"""Cycle 3 verification — day-based leakage smell test (Master Architecture
Document 2.5's original day-based-split intent).

Every day in this raw CICIDS2017 export is BENIGN plus exactly one attack
family, except Monday, which is 100% BENIGN. That makes a true
leave-one-day-out split well-posed only for Monday: holding out any other
day erases that day's entire attack family from training, leaving its
recall undefined rather than a meaningful generalization signal. So this
smell test holds out Monday and compares false-positive rate (the one
metric well-defined on an all-BENIGN holdout) between a model trained
without ever seeing Monday and the primary model's stratified-test FPR.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.config import (
    ARTIFACTS_DIR,
    INTERIM_DIR,
    LEAKAGE_SMELL_FPR_GAP_THRESHOLD,
    LODO_HELD_OUT_DAY,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.models.evaluate import aggregate_benign_fpr
from src.models.model_card import write_model_card
from src.models.xgboost_model import build_xgboost
from src.preprocessing.feature_engineering import Winsorizer


def build_lodo_split(cleaned_df: pd.DataFrame, held_out_day: str = LODO_HELD_OUT_DAY):
    lodo_train_df = cleaned_df[cleaned_df["source_day"] != held_out_day].reset_index(drop=True)
    lodo_holdout_df = cleaned_df[cleaned_df["source_day"] == held_out_day].reset_index(drop=True)
    return lodo_train_df, lodo_holdout_df


def run(
    interim_dir: Path = INTERIM_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict:
    xgboost_dir = models_dir / "xgboost_v1"
    feature_list = json.loads((xgboost_dir / "feature_list.json").read_text())
    primary_metrics = json.loads((xgboost_dir / "metrics.json").read_text())
    label_map = json.loads((artifacts_dir / "label_map.json").read_text())

    cleaned_df = pd.read_parquet(interim_dir / "cleaned.parquet")
    lodo_train_df, lodo_holdout_df = build_lodo_split(cleaned_df)

    # Reuse the already-chosen feature_list from preprocessor_v1.pkl (no
    # re-running correlation pruning / RF importance here -- that list is
    # already empirically justified and this is a diagnostic check, not a
    # shipped artifact). Fit a fresh imputer/winsorizer/scaler on LODO-train
    # only -- never touching the held-out day.
    X_train_raw = lodo_train_df[feature_list].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    winsorizer = Winsorizer()
    X_train = winsorizer.fit_transform(X_train)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)

    y_train = lodo_train_df["label_family"].map(label_map).to_numpy()

    X_holdout_raw = lodo_holdout_df[feature_list].to_numpy(dtype=float)
    X_holdout = scaler.transform(winsorizer.transform(imputer.transform(X_holdout_raw)))
    y_holdout = lodo_holdout_df["label_family"].map(label_map).to_numpy()

    model = build_xgboost(num_class=len(label_map))
    # no validation fold carved out of LODO-train for early stopping here --
    # this is a diagnostic comparison model, not a shipped artifact, so a
    # fixed round count keeps it simple; reuse the primary model's chosen
    # n_estimators as a fixed budget instead of early stopping
    model.set_params(early_stopping_rounds=None)
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_holdout)
    benign_id = label_map["BENIGN"]
    lodo_fpr = aggregate_benign_fpr(y_holdout, y_pred, benign_id)
    stratified_test_fpr = primary_metrics["test"]["aggregate_benign_fpr"]

    absolute_gap = abs(lodo_fpr - stratified_test_fpr)
    flagged = absolute_gap > LEAKAGE_SMELL_FPR_GAP_THRESHOLD

    result = {
        "held_out_day": LODO_HELD_OUT_DAY,
        "held_out_day_rows": int(len(lodo_holdout_df)),
        "stratified_test_fpr": stratified_test_fpr,
        "lodo_monday_fpr": lodo_fpr,
        "absolute_gap": absolute_gap,
        "threshold": LEAKAGE_SMELL_FPR_GAP_THRESHOLD,
        "flagged": flagged,
        "note": (
            "Every non-Monday day in this dataset is BENIGN plus exactly one "
            "attack family, so a class-complete leave-one-day-out comparison "
            "is only well-posed for Monday (100% BENIGN). FPR on Monday's "
            "unseen benign traffic is compared against the primary model's "
            "stratified-test FPR; macro-recall is not used here since it is "
            "undefined for any other day's held-out attack class."
        ),
    }

    (xgboost_dir / "leakage_smell_test.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    card_path = write_model_card(artifacts_dir=artifacts_dir, models_dir=models_dir)
    print(f"wrote {card_path}")

    return result


if __name__ == "__main__":
    run()
