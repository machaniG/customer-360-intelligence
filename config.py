import os
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "artifacts/models"))
OUTPUTS_PATH = Path(os.getenv("OUTPUTS_PATH", BASE_DIR / "outputs"))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

RAW_PATH = BASE_DIR / "data/raw/cleaned_transactions.csv"      # Input dataset
PROCESSED_PATH = BASE_DIR / "data/processed/transactions.csv" # Output processed dataset
FEATURE_PATH = BASE_DIR / "data/features/customer_features.csv" # Output features dataset
MODELS_PATH = BASE_DIR / "models" # Directory to save trained models


#legacy scores folder left for compatibility, but outputs is authoritative
SCORES_PATH = OUTPUTS_PATH
CONFIG_PATH = BASE_DIR / "config.yaml"

# Feature configuration

FEATURE_COLS = [
    "recency_days",
    "frequency",
    "tenure_days",
    "return_rate",
    "avg_basket_quantity",
    "avg_unique_items_per_basket",
    "avg_basket_monetary_value",
    "avg_days_between",
    "std_days_between",
    "purchase_rate_monthly",
]