import logging
import joblib
import pandas as pd

from models.segmentation.config import MODEL_PATH

logger = logging.getLogger(__name__)


def assign_segments(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign customer segments using trained segmentation model.

    Parameters
    ----------
    features_df : pd.DataFrame
        Customer-level feature dataframe used for clustering.

    Returns
    -------
    pd.DataFrame
        DataFrame with assigned segment column.
    """

    logger.info("Loading segmentation model...")
    artifact = joblib.load(MODEL_PATH)

    # If you saved full pipeline only:
    # pipeline = artifact

    # If you saved dict with metadata (better future-proofing):
    if isinstance(artifact, dict):
        pipeline = artifact["model"]
        segment_label_map = artifact.get("segment_label_map", {})
        trained_features = artifact.get("features", None)
    else:
        pipeline = artifact
        segment_label_map = {}
        trained_features = None

    # Align feature columns if metadata exists
    from .config import CLUSTERING_FEATURES

    if trained_features is not None:
        X = features_df[trained_features]
    else:
        # If we have full feature set, select clustering feature subset
        X = features_df[CLUSTERING_FEATURES]

    logger.info("Predicting segments...")
    segments = pipeline.predict(X)

    result = features_df.copy()
    result["segment"] = segments
    result["segment_label"] = result["segment"].map(segment_label_map).fillna("Unknown")

    return result
