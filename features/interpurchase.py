# interpurchase timing features
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InterpurchaseFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, as_of_date):
        self.as_of_date = pd.to_datetime(as_of_date)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X[(X['InvoiceDate'] <= self.as_of_date) & (X['transaction_type'] == 'purchase')]

        df = df.sort_values(['Customer ID', 'InvoiceDate'])

        df['days_between'] = (
            df.groupby('Customer ID')['InvoiceDate']
            .diff()
            .dt.days
        )

        features = df.groupby('Customer ID')['days_between'].agg(
            avg_days_between='mean',
            std_days_between='std'
        ).fillna(0)

        return features.reset_index()