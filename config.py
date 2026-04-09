import os
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data/raw/cleaned_transactions.csv"      # Input dataset
PROCESSED_PATH = BASE_DIR / "data/processed/transactions.csv" # Output processed dataset
FEATURE_PATH = BASE_DIR / "data/features/customer_features.csv" # Output features dataset
MODELS_PATH = BASE_DIR / "models" # Directory to save trained models
OUTPUTS_PATH = BASE_DIR / "outputs"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

# NOTE: legacy scores folder left for compatibility, but outputs is authoritative
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

import os
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data/raw/cleaned_transactions.csv"      # Input dataset
PROCESSED_PATH = BASE_DIR / "data/processed/transactions.csv" # Output processed dataset
FEATURE_PATH = BASE_DIR / "data/features/customer_features.csv" # Output features dataset
MODELS_PATH = BASE_DIR / "models" # Directory to save trained models
SCORES_PATH = BASE_DIR / "scores" # Directory to save customer scores
CONFIG_PATH = BASE_DIR / "config.yaml"