import logging
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .config import (
    MODEL_DIR,
    MODEL_PATH,
    CLUSTERING_FEATURES,
    N_CLUSTERS,
    RANDOM_STATE,
    SCALE_FEATURES
)

logger = logging.getLogger(__name__)


def train_segmentation_model(features_df):
    """
    Train KMeans segmentation model on customer-level features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Customer-level aggregated feature dataframe.

    Returns
    -------
    pipeline : sklearn Pipeline
        Trained segmentation pipeline.
    segment_summary : pd.DataFrame
        Cluster-level feature averages.
    """

    logger.info("Selecting clustering features...")
    X = features_df[CLUSTERING_FEATURES]

    steps = []

    if SCALE_FEATURES:
        steps.append(("scaler", StandardScaler()))

    steps.append((
        "kmeans",
        KMeans(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_STATE,
            n_init="auto"
        )
    ))

    pipeline = Pipeline(steps)

    logger.info("Training segmentation model...")
    pipeline.fit(X)

    logger.info("Assigning segment labels...")
    segments = pipeline.predict(X)

    segment_summary = (
        features_df
        .assign(segment=segments)
        .groupby("segment")[CLUSTERING_FEATURES]
        .mean()
        .round(2)
    )

    # Save artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    logger.info(f"Segmentation model saved to {MODEL_PATH}")

    return pipeline, segment_summary
