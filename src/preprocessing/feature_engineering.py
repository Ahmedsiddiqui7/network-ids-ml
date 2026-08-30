"""Cycle 2 — feature selection: exact-duplicate removal, correlation
pruning, and RF-importance ranking, all computed on the training fold only.

This module empirically justifies the top-N feature set referenced in
Master Architecture Document 2.3, rather than asserting it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.config import (
    CORRELATION_THRESHOLD,
    RANDOM_SEED,
    RF_IMPORTANCE_MAX_DEPTH,
    RF_IMPORTANCE_N_ESTIMATORS,
    RF_IMPORTANCE_SAMPLE_SIZE,
    TOP_N_FEATURES,
    WINSORIZE_LOWER_Q,
    WINSORIZE_UPPER_Q,
)

# Master Architecture Document 2.3 calls these out by name as one of the few
# features that can catch "low and slow" attacks (Infiltration) that
# deliberately mimic idle legitimate traffic. The empirical RF importance
# ranking demotes them below the top-N cutoff -- but that's an artifact of
# Infiltration having only 25 rows in train, not evidence the features are
# actually uninformative: a single rare class's signal is easily swamped by
# high-volume classes like DoS/DDoS in an aggregate importance score. Force
# them into the final feature list regardless of rank, so the model still
# has access to the burstiness signal 2.3 identifies as necessary.
FORCE_INCLUDE_FEATURES = [
    "Active Mean",
    "Idle Mean",
]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips each column to bounds learned at fit time (train fold only)."""

    def __init__(self, lower_q: float = WINSORIZE_LOWER_Q, upper_q: float = WINSORIZE_UPPER_Q):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X, self.lower_q, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def drop_duplicate_columns(
    df: pd.DataFrame, candidate_cols: list[str], protect: tuple[str, ...] = ()
) -> list[str]:
    """Return candidate_cols with exact-duplicate columns removed.

    Detects duplication by content (byte-equal values), not by name, since
    CICIDS2017's raw export has at least one column duplicated under a
    pandas-mangled name (`Fwd Header Length.1`).

    Columns in `protect` are always kept, regardless of duplication — see
    FORCE_INCLUDE_FEATURES.
    """
    kept: list[str] = []
    for col in candidate_cols:
        if col in protect:
            kept.append(col)
            continue
        is_duplicate = any(df[col].equals(df[kept_col]) for kept_col in kept)
        if not is_duplicate:
            kept.append(col)
    return kept


def prune_correlated_features(
    df: pd.DataFrame,
    candidate_cols: list[str],
    threshold: float = CORRELATION_THRESHOLD,
    protect: tuple[str, ...] = (),
) -> list[str]:
    """Greedy correlation pruning: drop the later column of any pair whose
    absolute Pearson correlation exceeds threshold, computed on df (train).

    Columns in `protect` are always kept, regardless of correlation with an
    already-kept column — see FORCE_INCLUDE_FEATURES. They still count
    normally when deciding whether a *later* column is redundant.
    """
    corr = df[candidate_cols].corr().abs()

    kept: list[str] = []
    for col in candidate_cols:
        if col in protect:
            kept.append(col)
            continue
        if any(corr.loc[col, kept_col] > threshold for kept_col in kept):
            continue
        kept.append(col)
    return kept


def rank_by_importance(
    df: pd.DataFrame,
    candidate_cols: list[str],
    label_col: str,
    sample_size: int = RF_IMPORTANCE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.Series:
    """Fit a quick baseline RF and return feature importances, descending."""
    if len(df) > sample_size:
        sample_df = df.groupby(label_col, group_keys=False)[df.columns].apply(
            lambda g: g.sample(
                n=max(1, int(round(len(g) * sample_size / len(df)))),
                random_state=seed,
            )
        )
    else:
        sample_df = df

    X = sample_df[candidate_cols].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    y = sample_df[label_col].to_numpy()

    rf = RandomForestClassifier(
        n_estimators=RF_IMPORTANCE_N_ESTIMATORS,
        max_depth=RF_IMPORTANCE_MAX_DEPTH,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)

    return pd.Series(rf.feature_importances_, index=candidate_cols).sort_values(ascending=False)


def choose_feature_list(
    train_df: pd.DataFrame,
    candidate_cols: list[str],
    label_col: str = "label_family",
    top_n: int = TOP_N_FEATURES,
) -> tuple[list[str], pd.Series]:
    """Orchestrate dedup -> correlation pruning -> importance ranking -> top_n.

    Impute + winsorize candidate columns temporarily (train-fit) purely to
    get sane values for the correlation/importance computations; this fit
    is discarded — the Preprocessor re-fits its own imputer/winsorizer on
    the final, much smaller feature_list.
    """
    imputer = SimpleImputer(strategy="median")
    winsorizer = Winsorizer()

    temp = train_df[candidate_cols].astype(float)
    temp[:] = imputer.fit_transform(temp)
    temp[:] = winsorizer.fit_transform(temp)
    temp[label_col] = train_df[label_col].to_numpy()

    protect = tuple(f for f in FORCE_INCLUDE_FEATURES if f in candidate_cols)
    deduped_cols = drop_duplicate_columns(temp, candidate_cols, protect=protect)
    pruned_cols = prune_correlated_features(temp, deduped_cols, protect=protect)
    importances = rank_by_importance(temp, pruned_cols, label_col)

    final_cols = importances.head(top_n).index.tolist()

    missing = [f for f in FORCE_INCLUDE_FEATURES if f not in importances.index]
    if missing:
        raise ValueError(
            f"force-include features not found among candidate columns after "
            f"dedup/correlation pruning: {missing}"
        )
    for forced in FORCE_INCLUDE_FEATURES:
        if forced not in final_cols:
            final_cols.append(forced)

    return final_cols, importances
