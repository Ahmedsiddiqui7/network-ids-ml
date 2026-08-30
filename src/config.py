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
