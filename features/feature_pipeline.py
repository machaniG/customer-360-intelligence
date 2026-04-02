# features/feature_pipeline.py

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys

from features import (LifetimesRFMTransformer, RFTFeatureBuilder, 
                      BasketFeatureBuilder, ReturnFeatureBuilder, 
                      InterpurchaseFeatureBuilder)
from config import FEATURE_COLS

# default name for the unit/price column in upstream data. Users can override
# by passing price_col to build_customer_features, e.g. price_col='Price'

UNIT_PRICE_COL = "UnitPrice"

# module logger
logger = logging.getLogger(__name__)



def build_customer_features(
    transactions: pd.DataFrame,
    as_of_date: str,
    price_col: str = UNIT_PRICE_COL,
) -> pd.DataFrame:

    # instantiate builders with a consistent as_of_date parameter
    builders = [
        RFTFeatureBuilder(as_of_date),
        BasketFeatureBuilder(as_of_date),
        ReturnFeatureBuilder(as_of_date),
        InterpurchaseFeatureBuilder(as_of_date),
        LifetimesRFMTransformer(
            customer_id_col='Customer ID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalPrice',
            observation_period_end=pd.to_datetime(as_of_date)
        )
    ]

    feature_dfs = [b.transform(transactions) for b in builders]
    
    from functools import reduce
    from pandas import merge

    customer_features = reduce(
        lambda left, right: pd.merge(left, right, on='Customer ID', how='outer'),
        feature_dfs
    )

    # ensure the canonical feature columns are present. If any feature the
    # model expects isn't produced by the builders, add it as a zero-filled
    # column so downstream training/scoring remains deterministic.
    for col in FEATURE_COLS:
        if col not in customer_features.columns:
            customer_features[col] = 0

    return customer_features.fillna(0)
