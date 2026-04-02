import pandas as pd
import joblib
import logging

from etl import load_processed_data
from features import build_customer_features, validate_features
from models.churn.config import (
    MODEL_PATH,
    FEATURE_COLS,
    OUTPUT_PATH,
    DEFAULT_SCORE_DATE
)
from config import SCORES_PATH, PROCESSED_PATH


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default monthly score date

score_date = DEFAULT_SCORE_DATE


def run_monthly_scoring():

    logger.info(f"Starting monthly scoring for {score_date}")

    # Load transactions
    df = load_processed_data()

    # Build features for THIS scoring date
    X_today = build_customer_features(df, as_of_date=score_date)

    # Load model
    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_cols = artifact.get("feature_cols", FEATURE_COLS)
        train_date = artifact.get("as_of_date")
        logger.info(f"Model trained with as_of_date = {train_date}")
    else:
        model = artifact
        feature_cols = FEATURE_COLS
        logger.warning("Legacy model detected (no metadata stored)")

    # Validate features
    validate_features(
        X_today,
        feature_cols,
        context="monthly_scoring",
        raise_on_missing=False,
        logger=logger,
    )

    # Score customers
    X_today["churn_risk"] = model.predict_proba(
        X_today[feature_cols]
    )[:, 1]

    # Prepare output
    output_cols = ["Customer ID", "churn_risk"] + feature_cols
    available_cols = [c for c in output_cols if c in X_today.columns]

    results = X_today[available_cols].sort_values(
        "churn_risk",
        ascending=False
    )

    # Save
    results.to_csv(OUTPUT_PATH, index=False)

    logger.info(f"Monthly churn scores saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_monthly_scoring()


# train once: python scripts/train_model.py --as-of-date 2010-09-01
# score monthly: python -m churn.scoring.monthly_scoring
# python -m scripts.monthly_scoring --score-date 2010-10-01
# later, evaluate with: python scripts/evaluate_scores.py
# score later: run_monthly_scoring(score_date="2010-11-01")
# in real production, scoring is done as: python -m churn.scoring.score_customers --score-date 2024-02-01
# so monthly_scoring.py is unnecessary in production but serves as a convenient script for generating monthly scores for evaluation and development.