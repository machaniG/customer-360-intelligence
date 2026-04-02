"""Project-wide configuration constants.

Keep canonical data and model paths and the canonical feature column list here so
other modules import a single source of truth.
"""
from pathlib import Path
from config import PROCESSED_PATH, FEATURE_COLS


# Paths

# Data path (produced by ETL)
DATA_PATH = PROCESSED_PATH

# Model artifact directory
MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "churn_model.pkl"

# Output path
from config import OUTPUTS_PATH
OUTPUT_PATH = OUTPUTS_PATH / "churn_risk_scores.csv"


# Scoring configuration

# Default scoring date (can be overridden at runtime)
DEFAULT_SCORE_DATE = "2010-10-01"
