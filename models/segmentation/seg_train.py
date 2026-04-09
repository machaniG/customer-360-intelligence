import logging
import joblib
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .config import (
    MODEL_DIR,
    MODEL_PATH,
    CLUSTERING_FEATURES,
    N_CLUSTERS,
    RANDOM_STATE,
    SCALE_FEATURES,
    SEGMENT_LABELS
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

    # Add segments to features_df for RFM calculation
    features_with_segments = features_df.assign(segment=segments)

    segment_summary = (
        features_with_segments
        .groupby("segment")[CLUSTERING_FEATURES]
        .mean()
        .round(2)
    )

    # Compute composite RFM score for each centroid to rank segments
    # RFM score: higher frequency and monetary better, lower recency better
    centroids = pipeline.named_steps['kmeans'].cluster_centers_

    # Map centroids back to original feature space if scaled
    if SCALE_FEATURES:
        centroids = pipeline.named_steps['scaler'].inverse_transform(centroids)

    # Create DataFrame for centroids
    centroid_df = pd.DataFrame(centroids, columns=CLUSTERING_FEATURES)

    # Add RFM features: assume recency_days, frequency, gross_spend are in features_df
    rfm_features = ['recency_days', 'frequency', 'gross_spend']
    centroid_rfm = features_with_segments.groupby("segment")[rfm_features].mean()

    # Composite RFM score: (frequency * gross_spend) / (recency_days + 1)
    centroid_rfm['rfm_score'] = (centroid_rfm['frequency'] * centroid_rfm['gross_spend']) / (centroid_rfm['recency_days'] + 1)

    # Rank segments by RFM score descending (higher score = better segment)
    ranked_segments = centroid_rfm.sort_values('rfm_score', ascending=False).index.tolist()

    # Create label mapping: best segment (highest score) gets first label
    segment_label_map = {seg: SEGMENT_LABELS[i] for i, seg in enumerate(ranked_segments)}

    logger.info(f"Segment label mapping: {segment_label_map}")

    # Save artifact with model and label mapping
    artifact = {
        "model": pipeline,
        "segment_label_map": segment_label_map,
        "features": CLUSTERING_FEATURES
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    logger.info(f"Segmentation model saved to {MODEL_PATH}")

    return pipeline, segment_summary
