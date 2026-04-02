import logging
import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from models.clv.config import GAMMA_FEATURES, BETA_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_clv_model(customer_features: pd.DataFrame):
    """
    Train CLV models (BG/NBD and Gamma-Gamma) using pre-calculated lifetimes features.

    Args:
        customer_features (pd.DataFrame): DataFrame containing customer-level features,
                                         including 'lifetimes_frequency', 'lifetimes_recency',
                                         'lifetimes_T', and 'lifetimes_monetary_value'.

    Returns:
        tuple: A tuple containing the fitted BetaGeoFitter model, GammaGammaFitter model,
               and the customer_features DataFrame with CLV predictions (or just models).
    """
    # Validate input data
    required_cols = GAMMA_FEATURES + BETA_FEATURES
    missing_cols = [col for col in required_cols if col not in customer_features.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure data types are correct
    for col in required_cols:
        customer_features[col] = pd.to_numeric(customer_features[col], errors='coerce').fillna(0)

    # Initialize models
    bgf = BetaGeoFitter(penalizer_coef=0.1) # Added a small penalizer_coef as good practice
    ggf = GammaGammaFitter(penalizer_coef=0.1) # Added a small penalizer_coef as good practice

    # 1. Fit the BetaGeoFitter model
    try:
        bgf.fit(
            customer_features[BETA_FEATURES[0]], # lifetimes_frequency
            customer_features[BETA_FEATURES[1]], # lifetimes_recency
            customer_features[BETA_FEATURES[2]]   # lifetimes_T
        )
        logger.info("BG/NBD model fitted successfully")
    except Exception as e:
        logger.error(f"Failed to fit BG/NBD model: {e}")
        raise

    # 2. Filter for customers with repeat purchases for GammaGammaFitter
    # (lifetimes_frequency > 0 customers are required for Gamma-Gamma)
    returning_customers_features = customer_features[customer_features['lifetimes_frequency'] > 0]

    # Also filter for positive monetary values (required for Gamma-Gamma)
    returning_customers_features = returning_customers_features[returning_customers_features['lifetimes_monetary_value'] > 0]

    # If there are no returning customers with positive monetary values, handle this case
    if returning_customers_features.empty:
        logger.warning("No returning customers with positive monetary values found. Gamma-Gamma model cannot be fitted.")
        ggf = None # Or handle as appropriate
    else:
        # 3. Fit the GammaGammaFitter model
        try:
            ggf.fit(
                returning_customers_features[GAMMA_FEATURES[0]], # lifetimes_frequency
                returning_customers_features[GAMMA_FEATURES[1]]  # lifetimes_monetary_value
            )
            logger.info("Gamma-Gamma model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit Gamma-Gamma model: {e}")
            ggf = None

    return bgf, ggf, customer_features # Returning customer_features for potential future use or just the models