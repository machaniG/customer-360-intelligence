from pathlib import Path

# Directories

MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "segmentation_model.pkl"

from config import OUTPUTS_PATH

# Output path
OUTPUT_PATH = OUTPUTS_PATH / "customer_value_segments.csv"

# Feature Configuration

CLUSTERING_FEATURES = [
    "gross_spend",
    "avg_basket_quantity",
    "avg_unique_items_per_basket",
    "avg_basket_monetary_value",
]

# Model Hyperparameters

N_CLUSTERS = 4
RANDOM_STATE = 42
SCALE_FEATURES = True

