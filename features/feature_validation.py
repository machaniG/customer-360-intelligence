"""Feature validation utilities.

Functions here are small, dependency-free helpers that compare a DataFrame's
columns to the canonical `FEATURE_COLS`. They log a concise summary and can
optionally raise on missing required features (used by training to fail fast).
"""
from typing import Dict, List, Any, Iterable
import logging


def validate_features(
    df,
    expected_cols: Iterable[str],
    context: str | None = None,
    raise_on_missing: bool = False,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """Validate that `df` contains the expected feature columns.

    Parameters
    ----------
    df : pandas.DataFrame-like
        DataFrame whose columns will be validated.
    expected_cols : iterable[str]
        Iterable of expected feature column names (order is preserved in the
        returned dict).
    context : str, optional
        Short context string used in log messages (e.g. 'training' or
        'scoring').
    raise_on_missing : bool
        If True, raise ValueError when any expected columns are missing.
    logger : logging.Logger, optional
        Logger to use; by default the module logger will be used.

    Returns
    -------
    dict
        Dictionary with keys: 'expected', 'present', 'missing', 'extra'.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    cols = set(df.columns)
    expected = list(expected_cols)
    present = [c for c in expected if c in cols]
    missing = [c for c in expected if c not in cols]
    # extras are any columns that are not in expected; we keep Customer ID and
    # label/risk columns out of the interesting "extra" list by filtering
    ignore = {"Customer ID", "churn_label", "churn_risk"}
    extra = [c for c in df.columns if c not in expected and c not in ignore]

    ctx = f"[{context}] " if context else ""
    logger.info("%sfeature validation: expected=%d present=%d missing=%d extra=%d", ctx, len(expected), len(present), len(missing), len(extra))

    if missing:
        logger.warning("%smissing feature columns: %s", ctx, missing)
    if extra:
        logger.debug("%sextra columns: %s", ctx, extra)

    if raise_on_missing and missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return {"expected": expected, "present": present, "missing": missing, "extra": extra}
