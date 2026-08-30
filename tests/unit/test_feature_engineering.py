import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_engineering import (
    FORCE_INCLUDE_FEATURES,
    Winsorizer,
    choose_feature_list,
    drop_duplicate_columns,
    prune_correlated_features,
)
from src.config import REPO_ROOT
from src.preprocessing.pipeline import Preprocessor, build_label_map, decode_labels, encode_labels


def _synthetic_flows(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    feat_a = rng.normal(100, 20, n)
    feat_b = feat_a * 2 + rng.normal(0, 0.01, n)  # near-perfectly correlated with feat_a
    feat_c = rng.normal(0, 1, n)  # independent
    feat_d = rng.integers(1, 1000, n)  # independent, integer dtype like most real CICFlowMeter columns
    # low-importance-by-construction (pure noise), like the real Active/Idle
    # Mean columns force-included despite ranking below the top-N cutoff
    active_mean = rng.normal(0, 1, n)
    idle_mean = rng.normal(0, 1, n)

    labels = rng.choice(
        ["BENIGN", "DoS", "PortScan", "Botnet"], size=n, p=[0.6, 0.2, 0.15, 0.05]
    )

    df = pd.DataFrame(
        {
            "Feat A": feat_a,
            "Feat B": feat_b,
            "Feat B Duplicate": feat_b,  # exact duplicate column
            "Feat C": feat_c,
            "Feat D": feat_d,
            "Active Mean": active_mean,
            "Idle Mean": idle_mean,
            "source_day": "Monday",
            "label_detailed": labels,
            "label_family": labels,
        }
    )
    return df


@pytest.fixture
def train_df():
    return _synthetic_flows(n=300, seed=0)


@pytest.fixture
def val_df():
    return _synthetic_flows(n=100, seed=1)


def test_winsorizer_clips_using_train_bounds_not_val_bounds():
    train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    winsorizer = Winsorizer(lower_q=0.1, upper_q=0.9)
    winsorizer.fit(train)

    val_with_extreme_outlier = np.array([[-1000.0], [5.0], [1000.0]])
    transformed = winsorizer.transform(val_with_extreme_outlier)

    assert transformed[0, 0] == winsorizer.lower_bounds_[0]
    assert transformed[2, 0] == winsorizer.upper_bounds_[0]
    assert transformed[0, 0] > -1000.0
    assert transformed[2, 0] < 1000.0


def test_drop_duplicate_columns_removes_exact_duplicate(train_df):
    candidates = ["Feat A", "Feat B", "Feat B Duplicate", "Feat C", "Feat D"]
    kept = drop_duplicate_columns(train_df, candidates)

    assert "Feat B" in kept
    assert "Feat B Duplicate" not in kept
    assert "Feat A" in kept and "Feat C" in kept and "Feat D" in kept


def test_prune_correlated_features_drops_correlated_pair(train_df):
    candidates = ["Feat A", "Feat B", "Feat C", "Feat D"]
    kept = prune_correlated_features(train_df, candidates, threshold=0.95)

    # Feat A and Feat B are near-perfectly correlated by construction; only one survives
    assert not ({"Feat A", "Feat B"} <= set(kept))
    assert "Feat C" in kept
    assert "Feat D" in kept


def test_prune_correlated_features_protect_keeps_correlated_forced_column(train_df):
    """Reproduces the real-data case: Idle Mean is ~0.99 correlated with
    Idle Max/Idle Min and would normally be pruned as redundant -- but a
    force-included feature must survive anyway."""
    candidates = ["Feat A", "Feat B", "Feat C", "Feat D"]
    kept = prune_correlated_features(train_df, candidates, threshold=0.95, protect=("Feat B",))

    assert "Feat A" in kept  # processed first, keeps its slot
    assert "Feat B" in kept  # protected, survives despite correlating with Feat A


def test_preprocessor_imputes_using_train_median_not_val_median(train_df, val_df):
    train_df = train_df.copy()
    val_df = val_df.copy()

    rng = np.random.default_rng(2)
    train_df.loc[:, "Feat C"] = rng.normal(1.0, 0.05, len(train_df))
    train_df.loc[0, "Feat C"] = np.nan
    train_median = train_df["Feat C"].median()

    val_df.loc[:, "Feat C"] = rng.normal(999.0, 0.05, len(val_df))  # very different distribution
    val_df.loc[0, "Feat C"] = np.nan  # val's own median would be ~999.0

    pre = Preprocessor()
    pre.fit(train_df)

    if "Feat C" not in pre.feature_list:
        pytest.skip("Feat C was pruned by feature selection on this synthetic sample")

    col_idx = pre.feature_list.index("Feat C")

    # the fitted imputer's stored statistic must be train's median, regardless
    # of what val/test data later gets passed through transform()
    assert pre.imputer.statistics_[col_idx] == pytest.approx(train_median)

    val_transformed = pre.transform(val_df)
    # val's non-missing rows (~999) sit far outside train's winsorize bounds
    # (~1.0), so they get clipped to the same upper bound as the imputed row
    # (train median, also outside train bounds on the low side is not the
    # case here -- instead assert the imputed row's *raw* value equals the
    # train median before scaling, via a fresh imputer-only transform)
    raw_val = val_df[pre.feature_list].to_numpy(dtype=float)
    raw_imputed = pre.imputer.transform(raw_val)
    assert raw_imputed[0, col_idx] == pytest.approx(train_median)
    assert raw_imputed[0, col_idx] != pytest.approx(999.0, abs=1.0)


def test_transformed_output_has_no_nan_or_inf(train_df, val_df):
    pre = Preprocessor()
    pre.fit(train_df)

    X_train = pre.transform(train_df)
    X_val = pre.transform(val_df)

    assert not np.isnan(X_train).any()
    assert not np.isinf(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isinf(X_val).any()
    assert X_train.shape == (len(train_df), len(pre.feature_list))


def test_label_map_round_trip(train_df):
    label_map = build_label_map(train_df)

    assert set(label_map.keys()) == set(train_df["label_family"].unique())

    original_labels = train_df["label_family"].tolist()
    ids = encode_labels(original_labels, label_map)
    decoded = decode_labels(ids, label_map)

    assert decoded == original_labels


def test_preprocessor_round_trip_persistence(train_df, tmp_path):
    """The literal Cycle 2 verification requirement: a reloaded artifact
    must transform a known sample identically to the in-memory version
    used during fitting."""
    pre = Preprocessor()
    pre.fit(train_df)

    known_sample = train_df.iloc[:5]
    in_memory_output = pre.transform(known_sample)

    artifact_path = tmp_path / "preprocessor_v1.pkl"
    pre.save(artifact_path)

    reloaded = Preprocessor.load(artifact_path)
    reloaded_output = reloaded.transform(known_sample)

    np.testing.assert_allclose(in_memory_output, reloaded_output)
    assert reloaded.feature_list == pre.feature_list


def test_save_raises_if_class_pickled_under_main(train_df, tmp_path, monkeypatch):
    """Regression test for the real bug hit in Cycle 4: pipeline.py run via
    `python -m src.preprocessing.pipeline` binds Preprocessor's __module__
    to "__main__", producing an artifact that silently fails to load from
    any other entrypoint (e.g. the inference API). save() must refuse
    loudly instead of writing a broken artifact."""
    pre = Preprocessor()
    pre.fit(train_df)

    monkeypatch.setattr(Preprocessor, "__module__", "__main__")
    with pytest.raises(RuntimeError, match="__main__"):
        pre.save(tmp_path / "preprocessor_v1.pkl")


def test_saved_artifact_loads_from_independent_subprocess(train_df, tmp_path):
    """The portability regression this bug class needs: unpickle the saved
    artifact from a genuinely separate interpreter (not the process that
    wrote it), which is exactly the scenario that silently broke before
    save()'s __main__ guard existed -- a same-process round trip (as in
    test_preprocessor_round_trip_persistence above) can't catch it, since
    the writer's own module identity is still in scope there."""
    pre = Preprocessor()
    pre.fit(train_df)

    artifact_path = tmp_path / "preprocessor_v1.pkl"
    pre.save(artifact_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.preprocessing.pipeline import Preprocessor; "
            f"p = Preprocessor.load({str(artifact_path)!r}); "
            "print('loaded OK', len(p.feature_list))",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "loaded OK" in result.stdout


def test_force_include_features_survive_top_n_cutoff(train_df):
    """Active Mean / Idle Mean are pure noise by construction in this
    fixture, so a low top_n should normally exclude them -- but they must
    appear anyway, since MAD 2.3 flags them as necessary for detecting
    Infiltration's low-and-slow behavior regardless of aggregate RF rank."""
    candidate_cols = [c for c in train_df.columns if c not in ("source_day", "label_detailed", "label_family")]

    final_cols, importances = choose_feature_list(train_df, candidate_cols, top_n=1)

    for forced in FORCE_INCLUDE_FEATURES:
        assert forced in final_cols

    # top_n=1 alone could only ever keep 1 column by rank; the extra columns
    # present prove the force-include override actually added features
    # beyond the importance cutoff, not just that they happened to rank high
    assert len(final_cols) > 1
