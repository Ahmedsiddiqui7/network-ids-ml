"""Shared path constants and pipeline hyperparameters."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = REPO_ROOT / "data" / "raw"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PREPROCESSOR_DIR = ARTIFACTS_DIR / "preprocessor"
MODELS_DIR = ARTIFACTS_DIR / "models"

RANDOM_SEED = 42

# Feature engineering (Pillar 7 Cycle 2 / Pillar 3.1)
CORRELATION_THRESHOLD = 0.95
TOP_N_FEATURES = 30
WINSORIZE_LOWER_Q = 0.01
WINSORIZE_UPPER_Q = 0.99
RF_IMPORTANCE_N_ESTIMATORS = 100
RF_IMPORTANCE_MAX_DEPTH = 10
RF_IMPORTANCE_SAMPLE_SIZE = 200_000

# Model training (Pillar 7 Cycle 3 / Pillar 3.2)
RF_BASELINE_N_ESTIMATORS = 300
RF_BASELINE_MAX_DEPTH = 20

XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 7
XGB_LEARNING_RATE = 0.08
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_EVAL_METRIC = "mlogloss"  # 'aucpr' (per MAD 3.2) is binary-only in XGBoost

# Imbalance handling
SMOTE_MIN_SAMPLES = 10  # below this, MAD 3.2 says don't pretend SMOTE fixes it
SMOTE_TARGET_COUNTS = {"Infiltration": 300}

# Day-based leakage smell test (Cycle 3 verification, MAD 2.5)
LODO_HELD_OUT_DAY = "Monday-WorkingHours"
LEAKAGE_SMELL_FPR_GAP_THRESHOLD = 0.02

# Inference API (Pillar 7 Cycle 4 / Pillar 3.3 / Pillar 4.3)
API_MODEL_VERSION = "xgboost_v1"
THRESHOLD_FPR_BUDGET = 0.01
