

import logging
import dill
import joblib
import pandas as pd
from config import PROCESSED_PATH, SCORES_PATH
from etl import load_processed_data
from features import build_customer_features
from models.clv.config import BETA_FEATURES, CLV_TIER_QUANTILES, GAMMA_FEATURES, MODEL_PATH, OUTPUT_PATH
from models.clv import predict_clv
from lifetimes import BetaGeoFitter, GammaGammaFitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assign_clv_tier(clv_values: pd.Series, tier_quantiles: dict = CLV_TIER_QUANTILES) -> pd.Series:
    """Assign Bronze/Silver/Gold/Platinum tiers based on 12-month CLV and configurable quantiles."""
    if clv_values.empty:
        return pd.Series(dtype="object")

    thresholds = {
        tier: clv_values.quantile(q)
        for tier, q in tier_quantiles.items()
    }

    def tier(value):
        if pd.isna(value):
            return pd.NA
        if value <= thresholds["Bronze"]:
            return "Bronze"
        if value <= thresholds["Silver"]:
            return "Silver"
        if value <= thresholds["Gold"]:
            return "Gold"
        return "Platinum"

    return clv_values.apply(tier)


def score_clv(
    transactions_path: str = PROCESSED_PATH,
    score_date: str = "2010-10-01",
    feature_df: pd.DataFrame = None
) -> pd.DataFrame:
    logger.info("Starting CLV prediction...")

    if feature_df is None:
        df = load_processed_data() if transactions_path is None else pd.read_csv(transactions_path, parse_dates=["InvoiceDate"])
        feature_df = build_customer_features(df, as_of_date=score_date)
        logger.info("Built customer features for CLV scoring")
    else:
        feature_df = feature_df.copy()
        logger.info("Using provided feature DataFrame for CLV scoring")

    clv_cols = ["Customer ID"] + []
    for col in GAMMA_FEATURES + BETA_FEATURES:
        if col not in clv_cols:
            clv_cols.append(col)
    clv_feature_df = feature_df[clv_cols].copy()
    clv_feature_df = clv_feature_df.set_index("Customer ID")

    # Load models
    # Load BG/NBD model
    with open(MODEL_PATH.parent / "bgf_model.pickle", "rb") as f:
        bgf_model = dill.load(f)

    # Load Gamma-Gamma model
    try:
        with open(MODEL_PATH.parent / "ggf_model.pickle", "rb") as f:
            ggf_model = dill.load(f)
        if ggf_model is None:
            logger.warning("Gamma-Gamma model is None - CLV predictions will only use purchase predictions")
    except FileNotFoundError:
        logger.warning("Gamma-Gamma model file not found - CLV predictions will only use purchase predictions")
        ggf_model = None  

    # make predictions
    logger.info("Predicting CLV...")
    clv_predictions_df = predict_clv(
        customer_features=clv_feature_df,
        bgf_model=bgf_model,
        ggf_model=ggf_model
    ).reset_index()

    clv_predictions_df["clv_tier"] = assign_clv_tier(clv_predictions_df["CLV_12_month"])

    # Use only CLV predictions in output to avoid redundant duplicate columns
    output_df = clv_predictions_df.copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"CLV predictions saved to {OUTPUT_PATH}")

    return output_df


def main():
    score_clv()


if __name__ == "__main__":
    main()

# The script performs the following steps:
# 1. Configures logging to output to the console.
# 2. Defines a main function that loads processed transaction data, loads pre-trained BG/NBD and Gamma-Gamma models, builds customer features, predicts CLV, and saves the predictions to a CSV file.
# 3. The script includes error handling for loading models and saving predictions, and ensures that the features are in the correct format for prediction.
# To run the script, execute `python scoring/score_clv.py` from the command line, 
# or python -m scoring.score_clv  if you are in the root directory of the project.