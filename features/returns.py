# return behavioral features
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ReturnFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, as_of_date, cap_rate=2.0):
        self.as_of_date = pd.to_datetime(as_of_date)
        self.cap_rate = cap_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        past = X[X['InvoiceDate'] <= self.as_of_date]

        purchases = past[past['transaction_type'] == 'purchase']
        returns = past[past['transaction_type'] == 'return']

        purchased_qty = purchases.groupby('Customer ID')['Quantity'].sum()
        returned_qty = returns.groupby('Customer ID')['Quantity'].sum().abs()
        return_events = returns.groupby('Customer ID')['InvoiceNo'].nunique()
        return_value = returns.groupby('Customer ID')['TotalPrice'].sum().abs()

        df = pd.DataFrame({
            'purchased_quantity': purchased_qty,
            'returned_quantity': returned_qty,
            'return_events': return_events,
            'return_value': return_value
        }).fillna(0)

        df['return_rate'] = (df['returned_quantity'] / df['purchased_quantity']).clip(0, self.cap_rate)

        return df.reset_index()