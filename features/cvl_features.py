import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.base import BaseEstimator, TransformerMixin

class LifetimesRFMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='Customer ID', datetime_col='InvoiceDate',
                 monetary_value_col='TotalPrice', datetime_format='%Y-%m-%d %H:%M:%S',
                 observation_period_end=None):
        self.customer_id_col = customer_id_col
        self.datetime_col = datetime_col
        self.monetary_value_col = monetary_value_col
        self.datetime_format = datetime_format
        self.observation_period_end = observation_period_end # Can be explicitly set or derived in fit

    def fit(self, X, y=None):
        # If observation_period_end is not explicitly provided, determine it from the max date in the training data
        if self.observation_period_end is None:
            self.as_of_date_ = X[self.datetime_col].max()
        else:
            self.as_of_date_ = self.observation_period_end
        return self

    def transform(self, X):
        # Ensure as_of_date_ is set
        if not hasattr(self, 'as_of_date_'):
            if self.observation_period_end is not None:
                self.as_of_date_ = self.observation_period_end
            else:
                self.as_of_date_ = X[self.datetime_col].max()
        
        # The utility function handles filtering based on observation_period_end and calculates RFM
        lifetimes_features = summary_data_from_transaction_data(
            X,
            customer_id_col=self.customer_id_col,
            datetime_col=self.datetime_col,
            monetary_value_col=self.monetary_value_col,
            datetime_format=self.datetime_format,
            observation_period_end=self.as_of_date_
        )
        # Rename columns to avoid conflicts and clearly identify them as lifetimes features
        lifetimes_features = lifetimes_features.rename(columns={
            'frequency': 'lifetimes_frequency',
            'recency': 'lifetimes_recency',
            'T': 'lifetimes_T',
            'monetary_value': 'lifetimes_monetary_value'
        })
        
        # Add epsilon to prevent log(0) in lifetimes library
        epsilon = 1e-10
        lifetimes_features['lifetimes_monetary_value'] = lifetimes_features['lifetimes_monetary_value'].clip(lower=epsilon)
        
        return lifetimes_features.reset_index()