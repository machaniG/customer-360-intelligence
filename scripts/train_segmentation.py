import sys
from pathlib import Path

# ensure project root is on path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
from etl import load_processed_data
from features import build_customer_features
from models.segmentation import train_segmentation_model
from models.segmentation.config import CLUSTERING_FEATURES, MODEL_PATH, OUTPUT_PATH
from config import PROCESSED_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


AS_OF_DATE = "2010-09-30"  # feature cutoff date for segmentation model

def main():

    logger.info("Loading processed transaction data...")
    transactions_df = load_processed_data()

    logger.info("Building customer-level features...")
    features_df = build_customer_features(transactions_df, as_of_date=AS_OF_DATE)

    logger.info("Training segmentation model...")
    model, segment_summary = train_segmentation_model(features_df)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    profile_path = OUTPUT_PATH / "segment_profiles.csv"
    segment_summary.to_csv(profile_path)

    logger.info(f"Segment profiles saved to {profile_path}")
    logger.info("Segmentation training completed successfully.")


if __name__ == "__main__":
    main()

# The script performs the following steps:
# 1. Configures logging to output detailed information about the training process.
# 2. Loads the processed transaction data from a specified path.
# 3. Builds customer-level features from the transaction data.  
# 4. Trains a segmentation model using the built features.
# 5. Saves the segment profiles to a specified output directory for later analysis or use in the application.
# run it with `python scripts/train_segmentation.py` from the command line. or `python -m scripts.train_segmentation` if you are in the root directory of the project.