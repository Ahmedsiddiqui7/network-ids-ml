"""Cycle 4 -- loads the trained preprocessor + model + supporting artifacts
once at construction (MAD 3.3: "loading the model and preprocessor once at
startup, not per-request") and serves predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.api.schemas import FEATURE_LIST, FlowFeatures, utc_timestamp
from src.config import API_MODEL_VERSION, ARTIFACTS_DIR, MODELS_DIR, PREPROCESSOR_DIR
from src.preprocessing.pipeline import Preprocessor


class ModelService:
    def __init__(
        self,
        preprocessor_dir: Path = PREPROCESSOR_DIR,
        artifacts_dir: Path = ARTIFACTS_DIR,
        models_dir: Path = MODELS_DIR,
        model_version: str = API_MODEL_VERSION,
    ):
        self.model_version = model_version
        model_dir = models_dir / model_version

        self.preprocessor = Preprocessor.load(preprocessor_dir / "preprocessor_v1.pkl")

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_dir / "model.json")

        self.label_map: dict = json.loads((artifacts_dir / "label_map.json").read_text())
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        self.benign_id = self.label_map["BENIGN"]

        self.feature_list: list[str] = json.loads((model_dir / "feature_list.json").read_text())
        self.metrics: dict = json.loads((model_dir / "metrics.json").read_text())
        self.threshold_info: dict = json.loads((model_dir / "threshold.json").read_text())
        self.threshold: float = self.threshold_info["threshold"]

    def _to_dataframe(self, flows: list[FlowFeatures]) -> pd.DataFrame:
        rows = [flow.model_dump(by_alias=True) for flow in flows]
        return pd.DataFrame(rows, columns=FEATURE_LIST)

    def _score(self, flows: list[FlowFeatures]) -> list[dict]:
        df = self._to_dataframe(flows)
        X = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X)

        results = []
        timestamp = utc_timestamp()
        for row_proba in proba:
            pred_id = int(row_proba.argmax())
            p_attack = 1.0 - float(row_proba[self.benign_id])
            results.append(
                {
                    "prediction": self.id_to_label[pred_id],
                    "is_malicious": p_attack >= self.threshold,
                    "confidence": float(row_proba.max()),
                    "risk_score": p_attack,
                    "model_version": self.model_version,
                    "timestamp": timestamp,
                }
            )
        return results

    def predict_one(self, flow: FlowFeatures) -> dict:
        return self._score([flow])[0]

    def predict_batch(self, flows: list[FlowFeatures]) -> list[dict]:
        return self._score(flows)

    def info(self) -> dict:
        test_metrics = self.metrics["test"]
        return {
            "model_version": self.model_version,
            "feature_count": self.metrics["feature_count"],
            "commit_hash": self.metrics["commit_hash"],
            "test_metrics": {
                "macro_recall_stable_classes": test_metrics["macro_recall_stable_classes"],
                "macro_f1": test_metrics["macro_f1"],
                "macro_pr_auc": test_metrics["macro_pr_auc"],
                "aggregate_benign_fpr": test_metrics["aggregate_benign_fpr"],
            },
            "operating_threshold": self.threshold_info,
        }
