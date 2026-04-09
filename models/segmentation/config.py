from pathlib import Path

# Directories

MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "segmentation_model.pkl"

from config import OUTPUTS_PATH

# Output path
OUTPUT_PATH = OUTPUTS_PATH / "customer_value_segments.csv"

# Segment labels ordered from BEST to WORST.
# After clustering, centroids are ranked by a composite RFM score
# and labels are assigned in this order (index 0 = best cluster)
SEGMENT_LABELS = [
    "Champions",    # Best: High recency, frequency, monetary
    "Loyal",        # Good: Moderate recency, high frequency/monetary
    "At Risk",      # Concerning: Low recency, moderate frequency/monetary
    "Lost"          # Worst: Low recency, frequency, monetary
]

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

