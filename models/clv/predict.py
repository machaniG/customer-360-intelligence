import pandas as pd
import numpy as np
import warnings
from lifetimes import BetaGeoFitter, GammaGammaFitter
from models.clv.config import GAMMA_FEATURES, BETA_FEATURES

def predict_clv(
    customer_features: pd.DataFrame,
    bgf_model: BetaGeoFitter,
    ggf_model: GammaGammaFitter,
    prediction_period_6_month: int = 180, # days
    prediction_period_12_month: int = 365, # days
    monetary_margin: float = 1.0 # margin for one-time buyer baseline
) -> pd.DataFrame:
    """
    Predicts Customer Lifetime Value (CLV) for given customer features using fitted BG/NBD and Gamma-Gamma models.

    Args:
        customer_features (pd.DataFrame): DataFrame containing customer-level features, including
                                         'lifetimes_frequency', 'lifetimes_recency', 'lifetimes_T',
                                         and 'lifetimes_monetary_value'.
        bgf_model (BetaGeoFitter): Fitted BetaGeoFitter model.
        ggf_model (GammaGammaFitter): Fitted GammaGammaFitter model.
        prediction_period_6_month (int): Number of days for 6-month CLV prediction.
        prediction_period_12_month (int): Number of days for 12-month CLV prediction.
        monetary_margin (float): Multiplier applied to one-time buyer monetary value.

    Returns:
        pd.DataFrame: customer_features DataFrame with added 'CLV_6_month' and 'CLV_12_month' columns.
    """
    # Suppress the non-critical log(0) warnings from lifetimes library
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered in log')

        # 1. Pre-prediction cleaning
        # ensure frequency, recency, T, and monetary_value are all non-negative and numeric
        for col in GAMMA_FEATURES + BETA_FEATURES:
            customer_features[col] = pd.to_numeric(customer_features[col], errors='coerce').fillna(0)
            customer_features[col] = customer_features[col].clip(lower=0)

        # 2. Predict Expected Number of Purchases (BG/NBD) first
        customer_features['predicted_purchases_6_month'] = bgf_model.conditional_expected_number_of_purchases_up_to_time(
            prediction_period_6_month,
            customer_features[BETA_FEATURES[0]],
            customer_features[BETA_FEATURES[1]],
            customer_features[BETA_FEATURES[2]]
        )

        customer_features['predicted_purchases_12_month'] = bgf_model.conditional_expected_number_of_purchases_up_to_time(
            prediction_period_12_month,
            customer_features[BETA_FEATURES[0]],
            customer_features[BETA_FEATURES[1]],
            customer_features[BETA_FEATURES[2]]
        )

        # 3. Predict Expected Monetary Value (Gamma-Gamma) and handle one-time buyers
        customer_features['predicted_monetary_value'] = 0.0

        # returning customers (have repeat purchases) can use Gamma-Gamma prediction if model exists
        mask_returning = customer_features['lifetimes_frequency'] > 0
        mask_returning_ok = mask_returning & (customer_features['lifetimes_monetary_value'] > 0)

        if ggf_model is not None and not customer_features[mask_returning_ok].empty:
            customer_features.loc[mask_returning_ok, 'predicted_monetary_value'] = ggf_model.conditional_expected_average_profit(
                customer_features.loc[mask_returning_ok, GAMMA_FEATURES[0]],
                customer_features.loc[mask_returning_ok, GAMMA_FEATURES[1]]
            )

        # one-time buyers: use current observed monetary_value as baseline and optional margin
        mask_one_time = (customer_features['lifetimes_frequency'] == 0) & (customer_features['lifetimes_monetary_value'] > 0)
        customer_features.loc[mask_one_time, 'predicted_monetary_value'] = (
            customer_features.loc[mask_one_time, 'lifetimes_monetary_value'] * monetary_margin
        )

        # 4. Calculate Final CLVs
        customer_features['CLV_6_month'] = customer_features['predicted_purchases_6_month'] * customer_features['predicted_monetary_value']
        customer_features['CLV_12_month'] = customer_features['predicted_purchases_12_month'] * customer_features['predicted_monetary_value']

    return customer_features