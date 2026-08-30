"""Cycle 4 verification (MAD 3.3 / 4.3): FastAPI TestClient integration
tests. Fixtures are real rows pulled from data/processed/{test,val}.parquet
(via src/models/tune_threshold.py's own borderline-search logic), not
hand-invented values, so validation and threshold behavior are exercised
against genuine flow statistics.

Depends on a completed Cycle 3 training run (artifacts/models/xgboost_v1/)
and a completed `python -m src.models.tune_threshold` (threshold.json) --
skips with a clear message if those artifacts aren't present yet.
"""
import json

import pytest

from src.config import ARTIFACTS_DIR, MODELS_DIR

XGBOOST_DIR = MODELS_DIR / "xgboost_v1"

if not (XGBOOST_DIR / "threshold.json").exists():
    pytest.skip(
        f"{XGBOOST_DIR / 'threshold.json'} not found -- run "
        "`python -m src.models.train` then `python -m src.models.tune_threshold` first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

client = TestClient(app)

# A genuine BENIGN row from data/processed/test.parquet -- also a natural
# example of Init_Win_bytes_backward's legitimate -1 sentinel.
BENIGN_FLOW = {
    "Bwd Packet Length Max": 0.0,
    "Packet Length Variance": 0.0,
    "Max Packet Length": 37.0,
    "Total Length of Fwd Packets": 74.0,
    "Total Length of Bwd Packets": 0.0,
    "Packet Length Mean": 37.0,
    "Fwd Packet Length Max": 37.0,
    "Fwd Packet Length Mean": 37.0,
    "Init_Win_bytes_backward": -1.0,
    "Total Fwd Packets": 2.0,
    "Fwd IAT Std": 0.0,
    "Idle Mean": 0.0,
    "Destination Port": 443.0,
    "Flow IAT Mean": 64040141.0,
    "Flow IAT Std": 0.0,
    "Flow Packets/s": 0.0312304122,
    "act_data_pkt_fwd": 1.0,
    "PSH Flag Count": 0.0,
    "Bwd Packets/s": 0.0,
    "Total Backward Packets": 0.0,
    "min_seg_size_forward": 20.0,
    "Init_Win_bytes_forward": 65535.0,
    "Flow Duration": 64040141.0,
    "Flow Bytes/s": 1.155525251,
    "Bwd Packet Length Min": 0.0,
    "ACK Flag Count": 1.0,
    "Active Max": 0.0,
    "Fwd Packet Length Min": 37.0,
    "Fwd IAT Min": 64040141.0,
    "Bwd IAT Total": 0.0,
    "Active Mean": 0.0,
}

# A genuine PortScan row from data/processed/test.parquet.
ATTACK_FLOW = {
    "Bwd Packet Length Max": 6.0,
    "Packet Length Variance": 12.0,
    "Max Packet Length": 6.0,
    "Total Length of Fwd Packets": 0.0,
    "Total Length of Bwd Packets": 6.0,
    "Packet Length Mean": 2.0,
    "Fwd Packet Length Max": 0.0,
    "Fwd Packet Length Mean": 0.0,
    "Init_Win_bytes_backward": 0.0,
    "Total Fwd Packets": 1.0,
    "Fwd IAT Std": 0.0,
    "Idle Mean": 0.0,
    "Destination Port": 5999.0,
    "Flow IAT Mean": 51.0,
    "Flow IAT Std": 0.0,
    "Flow Packets/s": 39215.68627,
    "act_data_pkt_fwd": 0.0,
    "PSH Flag Count": 1.0,
    "Bwd Packets/s": 19607.84314,
    "Total Backward Packets": 1.0,
    "min_seg_size_forward": 40.0,
    "Init_Win_bytes_forward": 29200.0,
    "Flow Duration": 51.0,
    "Flow Bytes/s": 117647.0588,
    "Bwd Packet Length Min": 6.0,
    "ACK Flag Count": 0.0,
    "Active Max": 0.0,
    "Fwd Packet Length Min": 0.0,
    "Fwd IAT Min": 0.0,
    "Bwd IAT Total": 0.0,
    "Active Mean": 0.0,
}

# A genuine BENIGN row (val split, row index 138, Thursday-Afternoon-
# Infiltration day) where the model's top class is still BENIGN
# (P(BENIGN)=0.9928) but the combined attack-class probability mass
# (risk_score=0.0072) crosses the tuned 0.01-FPR-budget threshold
# (~0.00403) -- found by re-running tune_threshold.py's scoring logic
# against the val set and searching for prediction=="BENIGN" and
# is_malicious==True. Confirms the "flagged despite benign top label"
# behavior documented in src/api/schemas.py is real and reachable.
BORDERLINE_FLOW = {
    "Bwd Packet Length Max": 6.0,
    "Packet Length Variance": 0.0,
    "Max Packet Length": 6.0,
    "Total Length of Fwd Packets": 6.0,
    "Total Length of Bwd Packets": 6.0,
    "Packet Length Mean": 6.0,
    "Fwd Packet Length Max": 6.0,
    "Fwd Packet Length Mean": 6.0,
    "Init_Win_bytes_backward": 256.0,
    "Total Fwd Packets": 1.0,
    "Fwd IAT Std": 0.0,
    "Idle Mean": 0.0,
    "Destination Port": 52399.0,
    "Flow IAT Mean": 188.0,
    "Flow IAT Std": 0.0,
    "Flow Packets/s": 10638.29787,
    "act_data_pkt_fwd": 0.0,
    "PSH Flag Count": 0.0,
    "Bwd Packets/s": 5319.148936,
    "Total Backward Packets": 1.0,
    "min_seg_size_forward": 20.0,
    "Init_Win_bytes_forward": 351.0,
    "Flow Duration": 188.0,
    "Flow Bytes/s": 63829.78723,
    "Bwd Packet Length Min": 6.0,
    "ACK Flag Count": 1.0,
    "Active Max": 0.0,
    "Fwd Packet Length Min": 6.0,
    "Fwd IAT Min": 0.0,
    "Bwd IAT Total": 0.0,
    "Active Mean": 0.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info_matches_loaded_artifact():
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()

    on_disk = json.loads((XGBOOST_DIR / "metrics.json").read_text())
    assert body["model_version"] == "xgboost_v1"
    assert body["commit_hash"] == on_disk["commit_hash"]
    assert body["feature_count"] == on_disk["feature_count"]


def test_predict_happy_path_benign():
    r = client.post("/predict", json=BENIGN_FLOW)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "BENIGN"
    assert body["is_malicious"] is False
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["model_version"] == "xgboost_v1"
    assert "timestamp" in body


def test_predict_happy_path_attack():
    r = client.post("/predict", json=ATTACK_FLOW)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "PortScan"
    assert body["is_malicious"] is True


def test_predict_borderline_flagged_despite_benign_top_label():
    r = client.post("/predict", json=BORDERLINE_FLOW)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "BENIGN"
    assert body["is_malicious"] is True


def test_predict_malformed_payload_type_returns_422():
    bad_flow = dict(BENIGN_FLOW)
    bad_flow["Flow Duration"] = "not-a-number"
    r = client.post("/predict", json=bad_flow)
    assert r.status_code == 422


def test_predict_missing_required_field_returns_422():
    bad_flow = dict(BENIGN_FLOW)
    del bad_flow["Flow Duration"]
    r = client.post("/predict", json=bad_flow)
    assert r.status_code == 422


def test_predict_unexpected_extra_field_returns_422():
    bad_flow = dict(BENIGN_FLOW)
    bad_flow["Some Unexpected Field"] = 1.0
    r = client.post("/predict", json=bad_flow)
    assert r.status_code == 422


def test_predict_physically_impossible_negative_duration_returns_422():
    bad_flow = dict(BENIGN_FLOW)
    bad_flow["Flow Duration"] = -5.0
    r = client.post("/predict", json=bad_flow)
    assert r.status_code == 422


def test_predict_init_win_bytes_sentinel_negative_one_accepted():
    flow = dict(BENIGN_FLOW)
    flow["Init_Win_bytes_forward"] = -1.0
    r = client.post("/predict", json=flow)
    assert r.status_code == 200


def test_predict_init_win_bytes_below_sentinel_returns_422():
    bad_flow = dict(BENIGN_FLOW)
    bad_flow["Init_Win_bytes_forward"] = -2.0
    r = client.post("/predict", json=bad_flow)
    assert r.status_code == 422


def test_predict_batch_returns_matching_length():
    r = client.post("/predict/batch", json={"flows": [BENIGN_FLOW, ATTACK_FLOW]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["prediction"] == "BENIGN"
    assert body["predictions"][1]["prediction"] == "PortScan"


def test_predict_batch_empty_list_returns_422():
    r = client.post("/predict/batch", json={"flows": []})
    assert r.status_code == 422
