# basic features: recency, frequency, tenure
import grp
from pyexpat import features

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RFTFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, as_of_date):
        self.as_of_date = pd.to_datetime(as_of_date)

    def fit(self, X, y=None):
        return self

    def transform(self, X): 
        df = X.copy()
    
        # 1. Ensure InvoiceDate is datetime objects for subtraction
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        as_of_date = pd.to_datetime(self.as_of_date)

        # Filter by date and type
        df = df[(df['InvoiceDate'] <= as_of_date) & 
             (df['transaction_type'] == 'purchase')]

        grp = df.groupby('Customer ID')

        # 2. Calculate base metrics
        # Using .max() and .min() on the group
        max_date = grp['InvoiceDate'].max()
        min_date = grp['InvoiceDate'].min()

        features = pd.DataFrame({
            'recency_days': (as_of_date - max_date).dt.days,
            'frequency': grp['InvoiceNo'].nunique(),
            'tenure_days': (as_of_date - min_date).dt.days,
            'gross_spend': grp['TotalPrice'].sum()
            })

        # 3. Fix the 'tenure_months' calculation
        # Instead of .dt.month (which fails), we use days divided by average month length
        # We use max(1, ...) or .clip(lower=1) to prevent DivisionByZero for same-day purchases
        features['tenure_months'] = (features['tenure_days'] / 30.44).clip(lower=1)

        # 4. Calculate purchase rate
        features['purchase_rate_monthly'] = (features['frequency'] / features['tenure_months']).round(2)
    
        return features.reset_index()