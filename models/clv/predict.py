import pandas as pd
import numpy as np
import warnings
from lifetimes import BetaGeoFitter, GammaGammaFitter
from models.clv.config import GAMMA_FEATURES, BETA_FEATURES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_clv(
    customer_features: pd.DataFrame,
    bgf_model: BetaGeoFitter,
    ggf_model: GammaGammaFitter,
    prediction_period_6_month: int = 180,
    prediction_period_12_month: int = 365,
    monetary_margin: float = 1.0
) -> pd.DataFrame:
    """
    Predict CLV using fitted BG/NBD and Gamma-Gamma models.      
    Args:
        customer_features (pd.DataFrame): DataFrame containing customer-level features, including
                                         'lifetimes_frequency', 'lifetimes_recency', 'lifetimes_T', 'lifetimes_monetary_value'
        bgf_model (BetaGeoFitter): Fitted BG/NBD model.
        ggf_model (GammaGammaFitter): Fitted Gamma-Gamma model.
        prediction_period_6_month (int): Days to predict for 6-month CLV.
        prediction_period_12_month (int): Days to predict for 12-month CLV.
        monetary_margin (float): Margin to apply to one-time buyers' monetary value.
        Returns:
            pd.DataFrame: DataFrame with predicted CLV values.
    """

    df = customer_features.copy()

    # Ensure numeric types (no fillna!)
    for col in GAMMA_FEATURES + BETA_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ─────────────────────────────────────────────────────────────
    # 1. Predict purchase frequency (BG/NBD)
    # ─────────────────────────────────────────────────────────────
    df['predicted_purchases_6_month'] = bgf_model.conditional_expected_number_of_purchases_up_to_time(
        prediction_period_6_month,
        df[BETA_FEATURES[0]],
        df[BETA_FEATURES[1]],
        df[BETA_FEATURES[2]]
    )

    df['predicted_purchases_12_month'] = bgf_model.conditional_expected_number_of_purchases_up_to_time(
        prediction_period_12_month,
        df[BETA_FEATURES[0]],
        df[BETA_FEATURES[1]],
        df[BETA_FEATURES[2]]
    )

    # ─────────────────────────────────────────────────────────────
    # 2. Predict monetary value
    # ─────────────────────────────────────────────────────────────
    df['predicted_monetary_value'] = np.nan

    # ✅ Returning customers (valid for Gamma-Gamma)
    mask_gg = (
        (df['lifetimes_frequency'] > 0) &
        (df['lifetimes_monetary_value'] > 0)
    )

    if ggf_model is not None and mask_gg.any():
        df.loc[mask_gg, 'predicted_monetary_value'] = ggf_model.conditional_expected_average_profit(
            df.loc[mask_gg, GAMMA_FEATURES[0]],
            df.loc[mask_gg, GAMMA_FEATURES[1]]
        )

    # ✅ One-time buyers
    mask_one_time = (
        (df['lifetimes_frequency'] == 0) &
        (df['lifetimes_monetary_value'] > 0)
    )

    df.loc[mask_one_time, 'predicted_monetary_value'] = (
        df.loc[mask_one_time, 'lifetimes_monetary_value'] * monetary_margin
    )

    # ✅ Fallback for invalid customers (VERY IMPORTANT)
    mask_invalid = df['predicted_monetary_value'].isna()

    df.loc[mask_invalid, 'predicted_monetary_value'] = (
        df.loc[mask_invalid, 'lifetimes_monetary_value']
    )

    # ─────────────────────────────────────────────────────────────
    # 3. Final CLV calculation
    # ─────────────────────────────────────────────────────────────
    df['CLV_6_month'] = df['predicted_purchases_6_month'] * df['predicted_monetary_value']
    df['CLV_12_month'] = df['predicted_purchases_12_month'] * df['predicted_monetary_value']

    # ─────────────────────────────────────────────────────────────
    # 4. Diagnostics (keep this for now)
    # ─────────────────────────────────────────────────────────────
    logger.info(f"CLV negative count: {(df['CLV_12_month'] < 0).sum()}")
    logger.info(f"CLV zero count: {(df['CLV_12_month'] == 0).sum()}")

    return df