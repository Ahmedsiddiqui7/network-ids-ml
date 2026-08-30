"""Cycle 4 -- request/response schemas for the inference API (MAD 3.3).

The request model (`FlowFeatures`) is built dynamically from
`artifacts/models/xgboost_v1/feature_list.json` -- the exact feature list
the trained preprocessor expects -- rather than hand-typed, so the schema
cannot silently drift from what the model was trained on (MAD 5.3's
feature-name-drift trap). Field names are the raw CICFlowMeter column
names (e.g. "Flow Duration", "Init_Win_bytes_forward"); Pydantic aliases
let the JSON body use those exact keys.

Negative-value validation is not a blanket rule -- checked directly
against data/interim/cleaned.parquet before choosing bounds:
- `Init_Win_bytes_forward`/`Init_Win_bytes_backward` legitimately take
  `-1` in ~40-48% of real rows: CICFlowMeter's documented sentinel for
  "no TCP window info available", not a physically impossible value.
  Allowed down to -1.
- Every other feature (including `Flow Duration`, `Flow Bytes/s`,
  `Flow Packets/s`, `Fwd IAT Min`, all of which have a handful of
  negative rows in the raw dataset from a known CICFlowMeter clock-skew
  bug) is constrained to >= 0, per MAD 3.3's explicit example of
  "negative duration" as a physically-impossible, reject-with-422 value.
  This means a tiny fraction of CICIDS2017's raw rows (well under 0.01%)
  would fail this API's own validation if replayed -- intentional: the
  API is a stricter boundary than the historical dataset, not a mirror
  of its flaws.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.config import MODELS_DIR

FEATURE_LIST_PATH = MODELS_DIR / "xgboost_v1" / "feature_list.json"
FEATURE_LIST: list[str] = json.loads(FEATURE_LIST_PATH.read_text())

# CICFlowMeter's documented "no TCP window info" sentinel -- legitimately
# -1, not a physically-impossible negative value.
SENTINEL_NEGATIVE_ONE_FEATURES = {"Init_Win_bytes_forward", "Init_Win_bytes_backward"}


def _python_safe_name(raw_name: str) -> str:
    return (
        raw_name.replace("/", "_per_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _build_flow_features_model():
    fields = {}
    for raw_name in FEATURE_LIST:
        min_value = -1.0 if raw_name in SENTINEL_NEGATIVE_ONE_FEATURES else 0.0
        fields[_python_safe_name(raw_name)] = (
            float,
            Field(..., alias=raw_name, ge=min_value),
        )
    return create_model(
        "FlowFeatures",
        __config__=ConfigDict(populate_by_name=True, extra="forbid"),
        **fields,
    )


FlowFeatures = _build_flow_features_model()


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flows: list[FlowFeatures] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    prediction: str = Field(
        ...,
        description=(
            "Most likely class, by argmax of the model's per-class probabilities. "
            "Independent of `is_malicious` -- the two fields can disagree; see "
            "`is_malicious`'s description."
        ),
    )
    is_malicious: bool = Field(
        ...,
        description=(
            "Whether P(attack) = 1 - P(BENIGN) crosses the tuned operating threshold "
            "(Pillar 4.3: recall-maximizing point at an explicit FPR budget). This can "
            "be True even when `prediction` is 'BENIGN': no single attack class needs "
            "to individually outscore BENIGN for the combined attack probability mass "
            "to still cross the alerting threshold. Treat `is_malicious` as the "
            "alerting signal and `prediction`/`confidence` as attribution, not gating."
        ),
    )
    confidence: float = Field(
        ..., description="Max per-class probability -- how sure the model is of `prediction`."
    )
    risk_score: float = Field(
        ..., description="1 - P(BENIGN) -- the value `is_malicious` is thresholded against."
    )
    model_version: str
    timestamp: str = Field(..., description="UTC ISO-8601, generated at request time.")


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
