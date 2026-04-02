# churn/scoring/score_customers.py

import joblib
import pandas as pd
import logging
import argparse

from features.feature_pipeline import build_customer_features
from models.churn.config import (
    DATA_PATH,
    MODEL_PATH,
    FEATURE_COLS,
    OUTPUT_PATH,
    DEFAULT_SCORE_DATE
)
from features.feature_validation import validate_features


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Scoring function

def score_customers(transactions_path, model_path, score_date):
    logger.info(f"Starting scoring for score_date = {score_date}")

    # Load transactions
    df = pd.read_csv(transactions_path, parse_dates=["InvoiceDate"])

    # Load model artifact
    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_cols = artifact.get("feature_cols", FEATURE_COLS)
        train_as_of_date = artifact.get("as_of_date")

        logger.info(f"Model trained with as_of_date = {train_as_of_date}")

    else:
        model = artifact
        feature_cols = FEATURE_COLS
        logger.warning("Legacy model detected (no metadata stored)")

    # uses score_date to build features for scoring
    X = build_customer_features(df, as_of_date=score_date)

    # Validate features
    validate_features(
        X,
        feature_cols,
        context="scoring",
        raise_on_missing=False,
        logger=logger,
    )

    # Predict churn probability
    X["churn_risk"] = model.predict_proba(X[feature_cols])[:, 1]

    logger.info(f"Scoring completed for {len(X)} customers")

    return X[["Customer ID", "churn_risk"]].sort_values("churn_risk",
        ascending=False
    )


# CLI entry point

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Score customers for churn risk")

    parser.add_argument(
        "--score-date",
        type=str,
        default=DEFAULT_SCORE_DATE,
        help="Scoring date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_PATH,
        help="Path to transactions CSV",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to trained model",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=OUTPUT_PATH,
        help="Output file path",
    )

    args = parser.parse_args()

    scores = score_customers(
        transactions_path=args.data_path,
        model_path=args.model_path,
        score_date=args.score_date,
    )

    scores.to_csv(args.output_path, index=False)

    logger.info(f"Churn scores saved to {args.output_path}")


# how to run this script:
"""
# airflow 
BashOperator(
    task_id="score_customers",
    bash_command="python -m churn.scoring.score_customers --score-date {{ ds }}"
)

# automation like : python score_customers.py --score-date $(date +%Y-%m-01)

# manual run: python score_customers.py --score-date 2010-10-01

# override data/model paths: python score_customers.py --score-date 2010-10-01 --data-path path/to/data.csv --model-path path/to/model.pkl --output-path path/to/scores.csv

# simple run with defaults: 
python -m churn.scoring.score_customers

# Override only the score date: 
python -m churn.scoring.score_customers --score-date 2010-10-01

# Override all parameters:
python -m churn.scoring.score_customers \
    --score-date 2010-10-01 \
    --data-path path/to/data.csv \
    --model-path path/to/model.pkl \
    --output-path path/to/scores.csv

    python -m churn.scoring.score_customers \
    --score-date 2010-10-01 \
    --data-path data/processed/transactions.csv \
    --model-path artifacts/models/churn_model.pkl \
    --output-path outputs/churn_scores_2010_10.csv

"""