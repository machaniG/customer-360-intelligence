# scripts/train_model.py

import sys
from pathlib import Path
import os
import joblib
import pandas as pd

# ensure project root is on path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etl import load_processed_data
from features import build_customer_features, validate_features
from labeling import build_churn_labels
from models.churn import train_churn_model 
from models.churn.config import FEATURE_COLS, MODEL_PATH
from config import PROCESSED_PATH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    


# Configuration

AS_OF_DATE = "2010-09-30"        # feature cutoff (9 months)
LABEL_END_DATE = "2010-10-31"    # churn observation window (next 30 days)

# Ensure model directory exists (MODEL_PATH comes from churn.config)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)



# Training pipeline

def main():

    logger.info("Loading data...")
    # read csv and parse InvoiceDate to datetime; assume upstream data uses
    # the canonical column names (user will clean data to match these).
    df = load_processed_data()

    logger.info("Building customer features...")
    # build_customer_features expects the transactions dataframe as the
    # first positional argument (named `transactions` in the pipeline). Use
    # a positional call here to avoid mismatched keyword names.
    features_df = build_customer_features(df, as_of_date=AS_OF_DATE)

    logger.info("Building churn labels...")
    labels_df = build_churn_labels(
        df=df,
        as_of_date=AS_OF_DATE,
        horizon_months=1
    )

    logger.info("Merging features & labels...")
    train_df = features_df.merge(
        labels_df,
        on="Customer ID",
        how="inner",
    )

    # Validate that the merged training frame contains the canonical feature
    # set AND the churn label. Training should fail loudly if required
    # features/labels are missing so errors are caught during development/CI.
    validate_features(train_df, FEATURE_COLS, context="training:train_df", raise_on_missing=True, logger=logger)

    if 'churn_label' not in train_df.columns:
        raise ValueError("Label column 'churn_label' not found in training DataFrame after merge")

    logger.info(f"Training samples: {len(train_df)}")

    logger.info("Training churn model...")
    model, feature_cols = train_churn_model(train_df)

    logger.info("Saving model...")
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "train_as_of_date": AS_OF_DATE, # training reference date
            "label_end_date": LABEL_END_DATE, # training label end date
        },
        MODEL_PATH,
    )

    logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

# The script performs the following steps:
# 1. Configures logging to output to the console.
# 2. Loads the processed dataset from a specified path.
# 3. Builds customer features using a specified cutoff date.
# 4. Builds churn labels using the same cutoff date and a specified horizon.
# 5. Merges the features and labels into a single training DataFrame.
# 6. Validates that the required feature columns and label are present in the training DataFrame.
# 7. Trains a churn prediction model using the training DataFrame.
# 8. Saves the trained model and metadata to a specified path.
# Run the script with `python scripts/train_churn.py` from the command line, or `python -m scripts.train_churn` if you are in the root directory of the project.
