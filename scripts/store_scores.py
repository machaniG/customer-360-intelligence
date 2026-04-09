# store_scores.py
#
# Writes customer scores and features to PostgreSQL after a scoring run.
# Replaces rows for the same as_of_date on re-run (upsert behaviour) so
# the table stays clean across multiple scoring runs.
#
# Usage — run after score_customers.py:
#
#   python store_scores.py \
#       --scores_path output/customer_scores.csv \
#       --features_path output/customer_features.csv \
#       --as_of_date 2011-10-31
#
# Or call store_scores() directly from score_customers.py.

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from db.config import FEATURES_TABLE, SCORES_TABLE
from db.schema import create_all_tables, get_engine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/store_scores.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def store_scores(
    scores_df: pd.DataFrame,
    features_df: pd.DataFrame,
    as_of_date: str,
) -> None:
    """Write scores and features DataFrames to PostgreSQL.

    Deletes existing rows for the same as_of_date before inserting so
    re-runs are idempotent.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of CombinedScorer.score() — one row per customer.
    features_df : pd.DataFrame
        Output of build_customer_features() — one row per customer.
    as_of_date : str
        The feature observation date used for this scoring run (YYYY-MM-DD).
        Used as a partition key so multiple scoring snapshots can coexist.
    """
    logger.info("Connecting to PostgreSQL and ensuring schema…")
    create_all_tables()
    engine = get_engine()

    scored_at = datetime.now(timezone.utc).replace(tzinfo=None)

    try:
        with engine.begin() as conn:
            # ── Delete existing rows for this as_of_date ───────────────────────
            # This makes re-runs safe without accumulating duplicates.
            for table in (SCORES_TABLE, FEATURES_TABLE):
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"DELETE FROM {table} WHERE as_of_date = :aod"
                    ),
                    {"aod": as_of_date},
                )
                logger.info("Cleared existing rows in '%s' for as_of_date=%s", table, as_of_date)

            # ── Write scores ───────────────────────────────────────────────────
            scores_to_write = _prepare_scores(scores_df, as_of_date, scored_at)
            scores_to_write.to_sql(
                SCORES_TABLE,
                conn,
                if_exists="append",
                index=False,
                chunksize=100,
            )
            logger.info(
                "Wrote %d rows to '%s'", len(scores_to_write), SCORES_TABLE
            )

            # ── Write features ─────────────────────────────────────────────────
            features_to_write = _prepare_features(features_df, as_of_date)
            features_to_write.to_sql(
                FEATURES_TABLE,
                conn,
                if_exists="append",
                index=False,
                chunksize=100,
            )
            logger.info(
                "Wrote %d rows to '%s'", len(features_to_write), FEATURES_TABLE
            )

    except SQLAlchemyError as e:
        logger.error("Database write failed: %s", e)
        raise
    finally:
        engine.dispose()

    logger.info("store_scores() complete for as_of_date=%s", as_of_date)


# ── DataFrame preparation helpers ──────────────────────────────────────────────

def _prepare_scores(
    df: pd.DataFrame,
    as_of_date: str,
    scored_at: datetime,
) -> pd.DataFrame:
    """Rename columns and add metadata columns before DB write."""
    out = df.rename(columns={"Customer ID": "customer_id"}).copy()
    out.columns = [c.lower().replace(" ", "_") for c in out.columns]
    out["as_of_date"] = as_of_date
    out["scored_at"] = scored_at

    # Keep only the columns that exist in the schema
    keep = [
        "customer_id", "churn_risk",
        "clv_6_month", "clv_12_month", "clv_tier", "segment", "segment_label",
        "revenue_at_risk", "scored_at", "as_of_date",
    ]
    return out[[c for c in keep if c in out.columns]]


def _prepare_features(df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """Rename columns and add as_of_date before DB write."""
    out = df.rename(columns={"Customer ID": "customer_id"}).copy()
    out.columns = [c.lower().replace(" ", "_") for c in out.columns]
    out["as_of_date"] = as_of_date

    # Drop any columns not in the features schema to avoid write errors
    schema_cols = [
        "customer_id", "as_of_date",
        "recency_days", "frequency", "tenure_days", "gross_spend",
        "avg_basket_quantity", "avg_unique_items_per_basket", "avg_basket_monetary_value",
        "purchased_quantity", "returned_quantity", "return_events", "return_rate",
        "avg_days_between", "std_days_between", "purchase_rate_monthly",
        "lifetimes_frequency", "lifetimes_recency", "lifetimes_T", "lifetimes_monetary_value",
    ]
    return out[[c for c in schema_cols if c in out.columns]]


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write customer scores and features to PostgreSQL."
    )
    parser.add_argument("--scores_path", required=True, help="Path to scores CSV.")
    parser.add_argument("--features_path", required=True, help="Path to features CSV.")
    parser.add_argument("--as_of_date", required=True, help="Scoring snapshot date (YYYY-MM-DD).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    scores_df = pd.read_csv(args.scores_path, dtype={"Customer ID": str})
    features_df = pd.read_csv(args.features_path, dtype={"Customer ID": str})

    store_scores(scores_df, features_df, as_of_date=args.as_of_date)

# run it
    # Example usage:
"""" 
python store_scores.py\ 
    --scores_path output/customer_scores.csv\
        --features_path output/customer_features.csv\ 
            --as_of_date 2011-10-31

python -m scripts.store_scores \
    --scores_path outputs/combined_customer_scores.csv \
    --features_path outputs/customer_features.csv \
    --as_of_date 2011-10-31
 """