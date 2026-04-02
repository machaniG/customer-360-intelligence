# purchase behavioral features
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BasketFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, as_of_date):
        self.as_of_date = pd.to_datetime(as_of_date)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X[(X['InvoiceDate'] <= self.as_of_date) & 
         (X['transaction_type'] == 'purchase')]

        basket = (
            df.groupby(['Customer ID', 'InvoiceNo'])
            .agg(
                basket_qty=('Quantity', 'sum'),
                unique_items=('StockCode', 'nunique'),
                basket_value=('TotalPrice', 'sum')
            )
            .reset_index()
        )

        features = basket.groupby('Customer ID').agg(
            avg_basket_quantity=('basket_qty', 'mean'),
            avg_unique_items_per_basket=('unique_items', 'mean'),
            avg_basket_monetary_value=('basket_value', 'mean')
        )

        return features.reset_index()