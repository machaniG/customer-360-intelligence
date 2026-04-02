import logging
import pandas as pd

from etl import load_processed_data
from features import build_customer_features
from models.segmentation import assign_segments
from config import SCORES_PATH, PROCESSED_PATH
from models.segmentation.config import CLUSTERING_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AS_OF_DATE = "2010-10-01"

def main():

    logger.info("Loading processed data...")
    transactions_df = load_processed_data()

    logger.info("Building customer features...")
    customer_df = build_customer_features(transactions_df, as_of_date=AS_OF_DATE)

    logger.info("Assigning segments...")
    segmented_df = assign_segments(customer_df)

    output_path = SCORES_PATH / "customer_segments.csv"
    segmented_df.to_csv(output_path, index=False)

    logger.info(f"Segmented customers saved to {output_path}")


if __name__ == "__main__":
    main()

# The script performs the following steps:
# 1. Loads the processed transaction data.  
# 2. Builds customer-level features from the transaction data.
# 3. Assigns customers to segments using a clustering model.
# 4. Saves the resulting customer segments to a CSV file in the scores directory.
# To run the script, execute `python scoring/score_segments.py` from the command line, or `
# python -m scoring.score_segments if you are in the root directory of the project.