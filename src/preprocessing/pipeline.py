"""Cycle 2 — reusable preprocessing pipeline: a single versioned artifact
(imputer + winsorizer + scaler + feature_list), fit on train only, applied
identically to train/val/test and, later, to API inference requests.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.config import PREPROCESSOR_DIR, PROCESSED_DIR, TOP_N_FEATURES
from src.preprocessing.feature_engineering import Winsorizer, choose_feature_list

METADATA_COLUMNS = ["source_day", "label_detailed", "label_family"]


class Preprocessor:
    """Bundles the fitted imputer, winsorizer, scaler, and final feature
    list into one object so training and serving apply the identical
    transform, loaded from the same artifact."""

    def __init__(self):
        self.feature_list: list[str] | None = None
        self.imputer: SimpleImputer | None = None
        self.winsorizer: Winsorizer | None = None
        self.scaler: RobustScaler | None = None

    def fit(self, train_df: pd.DataFrame, label_col: str = "label_family") -> "Preprocessor":
        candidate_cols = [c for c in train_df.columns if c not in METADATA_COLUMNS]
        self.feature_list, self.importances_ = choose_feature_list(
            train_df, candidate_cols, label_col=label_col, top_n=TOP_N_FEATURES
        )

        X = train_df[self.feature_list].to_numpy(dtype=float)

        self.imputer = SimpleImputer(strategy="median")
        X = self.imputer.fit_transform(X)

        self.winsorizer = Winsorizer()
        X = self.winsorizer.fit_transform(X)

        self.scaler = RobustScaler()
        self.scaler.fit(X)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_list is None:
            raise RuntimeError("Preprocessor must be fit (or loaded) before transform")

        X = df[self.feature_list].to_numpy(dtype=float)
        X = self.imputer.transform(X)
        X = self.winsorizer.transform(X)
        X = self.scaler.transform(X)
        return X

    def fit_transform(self, train_df: pd.DataFrame, label_col: str = "label_family") -> np.ndarray:
        self.fit(train_df, label_col=label_col)
        return self.transform(train_df)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "Preprocessor":
        return joblib.load(path)


def build_label_map(train_df: pd.DataFrame, label_col: str = "label_family") -> dict:
    """One JSON label-map artifact, loaded by both training and the API
    (Master Architecture Document 5.3) — kept separate from the pickled
    Preprocessor so the API doesn't need to unpickle sklearn objects."""
    classes = sorted(train_df[label_col].unique())
    return {label: idx for idx, label in enumerate(classes)}


def encode_labels(labels, label_map: dict) -> np.ndarray:
    return np.array([label_map[label] for label in labels])


def decode_labels(ids, label_map: dict) -> list[str]:
    id_to_label = {v: k for k, v in label_map.items()}
    return [id_to_label[i] for i in ids]


def run(
    processed_dir=PROCESSED_DIR,
    preprocessor_dir=PREPROCESSOR_DIR,
    artifacts_dir=None,
) -> Preprocessor:
    from src.config import ARTIFACTS_DIR

    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    preprocessor_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    preprocessor = Preprocessor()
    preprocessor.fit(train_df)

    label_map = build_label_map(train_df)
    with open(artifacts_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    preprocessor.save(preprocessor_dir / "preprocessor_v1.pkl")

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X = preprocessor.transform(split_df)
        y = encode_labels(split_df["label_family"], label_map)
        np.savez_compressed(
            processed_dir / f"{split_name}_features.npz",
            X=X,
            y=y,
            feature_names=np.array(preprocessor.feature_list),
        )
        print(f"{split_name}: X={X.shape}, y={y.shape}")

    print("\nTop features by importance:")
    print(preprocessor.importances_.head(TOP_N_FEATURES))
    print(f"\nlabel_map: {label_map}")

    return preprocessor


if __name__ == "__main__":
    run()
