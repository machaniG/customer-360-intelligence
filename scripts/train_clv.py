import sys
from pathlib import Path

# ensure project root is on path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dill
import pandas as pd
from etl import load_processed_data
from features import build_customer_features
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from models.clv.config import GAMMA_FEATURES, BETA_FEATURES
from models.clv.config import MODEL_PATH
from config import PROCESSED_PATH
import logging
import joblib

from models.clv import train_clv_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AS_OF_DATE = "2010-09-30"        # feature cutoff


def main():
    logger.info("Loading data...")
    df = load_processed_data()

    logger.info("Building customer features...")
    customer_features = build_customer_features(df, as_of_date=AS_OF_DATE)

    # Train CLV models
    bgf, ggf, _ = train_clv_model(customer_features)

    # Save models to model path
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    bgf_path = MODEL_PATH.parent / "bgf_model.pickle"
    ggf_path = MODEL_PATH.parent / "ggf_model.pickle"

    with open(bgf_path, "wb") as f:
        dill.dump(bgf, f)
        logger.info(f"Saved BG/NBD model to {bgf_path}")
    
    if ggf is not None:
        with open(ggf_path, "wb") as f:
            dill.dump(ggf, f)
        logger.info(f"Saved Gamma-Gamma model to {ggf_path}")
    else:
        logger.warning("Gamma-Gamma model was not fitted due to lack of returning customers.")

    logger.info("CLV models saved successfully.")


if __name__ == "__main__":
    main()

# The script performs the following steps:
# 1. Loads pre-calculated customer features from the processed dataset.
# 2. Trains the CLV models (Beta-Geometric and Gamma-Gamma) using the loaded customer features.
# 3. Saves the trained models to the specified model path for later use in prediction or evaluation.
# run it with `python scripts/train_clv.py` from the command line. or `
# python -m scripts.train_clv ` if you are in the root directory of the project.
