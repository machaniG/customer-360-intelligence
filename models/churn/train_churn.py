# churn/models/train_model.py

from pyexpat import model
from typing import List, Tuple
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from models.churn.config import FEATURE_COLS


TARGET_COL = "churn_label"
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    random_state=42,
)


# Training function

def train_churn_model(
    df: pd.DataFrame,
) -> Tuple[Pipeline, List[str]]:
    """
    Train churn model using predefined feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe including target column

    Returns
    -------
    model : sklearn Pipeline
        Trained churn model
    feature_cols : list
        Feature columns used by the model
    """

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    pipeline.fit(X, y)

    return pipeline, FEATURE_COLS
