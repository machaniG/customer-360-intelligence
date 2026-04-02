# basic features: recency, frequency, tenure
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RFTFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, as_of_date):
        self.as_of_date = pd.to_datetime(as_of_date)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # use the consistent attribute name as_of_date
        df = df[(df['InvoiceDate'] <= self.as_of_date) & 
                 (df['transaction_type'] == 'purchase')]

        grp = df.groupby('Customer ID')

        features = pd.DataFrame({
            'recency_days': (self.as_of_date - grp['InvoiceDate'].max()).dt.days,
            'frequency': grp['InvoiceNo'].nunique(),
            'tenure_days': (self.as_of_date - grp['InvoiceDate'].min()).dt.days,
            'gross_spend': grp['TotalPrice'].sum()
        })

        return features.reset_index()