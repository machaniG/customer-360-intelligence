import logging
import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from models.clv.config import GAMMA_FEATURES, BETA_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_clv_model(customer_features: pd.DataFrame):
    """
    Train BG/NBD and Gamma-Gamma models safely.
    Args:
        customer_features (pd.DataFrame): DataFrame containing customer-level features, including
                                         'lifetimes_frequency', 'lifetimes_recency', 'lifetimes_T', 'lifetimes_monetary_value'
    Returns:
        Tuple[BetaGeoFitter, GammaGammaFitter, pd.DataFrame]: Fitted BG/NBD model, fitted Gamma-Gamma model (or None if not fitted), and the cleaned customer_features DataFrame with predictions.
    """

    required_cols = GAMMA_FEATURES + BETA_FEATURES
    missing_cols = [col for col in required_cols if col not in customer_features.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ✅ Convert types WITHOUT forcing invalid values to 0
    for col in required_cols:
        customer_features[col] = pd.to_numeric(customer_features[col], errors='coerce')

    # ─────────────────────────────────────────────────────────────
    # 1. Fit BG/NBD (more tolerant, but still needs valid values)
    # ─────────────────────────────────────────────────────────────
    bgf = BetaGeoFitter(penalizer_coef=0.1)

    bgf_data = customer_features[
        (customer_features['lifetimes_frequency'] >= 0) &
        (customer_features['lifetimes_recency'] >= 0) &
        (customer_features['lifetimes_T'] > 0)
    ].dropna(subset=BETA_FEATURES)

    bgf.fit(
        bgf_data[BETA_FEATURES[0]],
        bgf_data[BETA_FEATURES[1]],
        bgf_data[BETA_FEATURES[2]]
    )

    logger.info(f"BG/NBD fitted on {len(bgf_data)} customers")

    # ─────────────────────────────────────────────────────────────
    # 2. Fit Gamma-Gamma (STRICT filtering required)
    # ─────────────────────────────────────────────────────────────
    ggf = None

    gg_data = customer_features[
        (customer_features['lifetimes_frequency'] > 0) &
        (customer_features['lifetimes_monetary_value'] > 0)
    ].dropna(subset=GAMMA_FEATURES)

    logger.info(f"Gamma-Gamma training customers: {len(gg_data)} / {len(customer_features)}")

    if not gg_data.empty:
        ggf = GammaGammaFitter(penalizer_coef=0.1)
        ggf.fit(
            gg_data[GAMMA_FEATURES[0]],
            gg_data[GAMMA_FEATURES[1]]
        )
        logger.info("Gamma-Gamma model fitted successfully")
    else:
        logger.warning("No valid data for Gamma-Gamma. Model not fitted.")

    return bgf, ggf, customer_features