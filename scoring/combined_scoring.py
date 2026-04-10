import logging
from pathlib import Path

import pandas as pd

from features import build_customer_features
from models.churn.score_customers import score_customers
from scoring.score_clv import score_clv
from models.segmentation.predict_segments import assign_segments
from models.churn.config import MODEL_PATH as CHURN_MODEL_PATH, DEFAULT_SCORE_DATE as CHURN_DEFAULT_DATE
from models.clv.config import MODEL_PATH as CLV_MODEL_PATH
from models.segmentation.config import MODEL_PATH as SEGMENT_MODEL_PATH
from config import PROCESSED_PATH, OUTPUTS_PATH

COMBINED_OUTPUT_PATH = OUTPUTS_PATH / "combined_customer_scores.csv"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_combined_scoring(
    score_date: str = CHURN_DEFAULT_DATE,
    transaction_csv_path: str = None,
    churn_model_path: str = None,
    clv_bgf_model_path: str = None,
    clv_ggf_model_path: str = None,
    segmentation_model_path: str = None,
    output_path: str = None,
):
    if transaction_csv_path is None:
        transaction_csv_path = str(PROCESSED_PATH)

    if churn_model_path is None:
        churn_model_path = str(CHURN_MODEL_PATH)

    if clv_bgf_model_path is None:
        clv_bgf_model_path = str(CLV_MODEL_PATH.parent / "bgf_model.pickle")
    if clv_ggf_model_path is None:
        clv_ggf_model_path = str(CLV_MODEL_PATH.parent / "ggf_model.pickle")

    if segmentation_model_path is None:
        segmentation_model_path = str(SEGMENT_MODEL_PATH)

    if output_path is None:
        output_path = str(COMBINED_OUTPUT_PATH)

    # 1) Load transaction data once and build features
    df = pd.read_csv(transaction_csv_path, parse_dates=["InvoiceDate"])
    features_df = build_customer_features(df, as_of_date=score_date)

    # Save built features immediately for downstream DB loading
    features_output_path = Path(output_path).parent / "customer_features.csv"
    features_output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_output_path, index=False)
    logger.info(f"Customer features saved to {features_output_path}")

    # 2) churn scoring using shared features
    churn_df = score_customers(
        transactions_path=transaction_csv_path,
        model_path=churn_model_path,
        score_date=score_date,
        features_df=features_df,
    )

    # 3) clv scoring using shared features
    clv_df = score_clv(
        score_date=score_date,
        feature_df=features_df,
    )

    # 4) segmentation
    segment_df = assign_segments(features_df)

    # merge outputs
    merged = churn_df.merge(
        clv_df[['Customer ID', 'CLV_6_month', 'CLV_12_month', 'clv_tier']],
        on='Customer ID',
        how='outer'
    )

    merged = merged.merge(
        segment_df[['Customer ID', 'segment', 'segment_label']],
        on='Customer ID',
        how='outer'
    )

    # Calculate revenue at risk: Expected Value Loss = predicted CLV x churn risk
    merged['revenue_at_risk'] = merged['CLV_12_month'].fillna(0) * merged['churn_risk'].fillna(0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info(f"Combined scoring output file saved to {output_path}")

    return merged


if __name__ == "__main__":
    run_combined_scoring()

# The script performs the following steps:
# 1. It defines a function `run_combined_scoring` that takes optional parameters for paths to transaction data, models, and output. If not provided, it uses default paths defined in the configuration.
# 2. It calls the `score_customers` function to get churn predictions, the `score_clv` function to get CLV predictions, and the `assign_segments` function to get customer segments.
# 3. It merges these three datasets on the "Customer ID" column to create a comprehensive dataset that includes churn risk, CLV predictions, and segment assignments for each customer.
# 4. Finally, it saves the merged dataset to a CSV file at the specified output path. To run the script, execute `python scoring/combined_scoring.py` from the command line, or `
# python -m scoring.combined_scoring ` if you are in the root directory of the project.
# run script with date argument to score for a specific date (defaults to churn model default date if not provided):
# python -m scoring.combined_scoring --score_date 2010-10-31    