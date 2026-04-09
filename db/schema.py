# db/schema.py
#
# SQLAlchemy table definitions.
# Call create_all_tables() once to initialise the schema in PostgreSQL.
# Subsequent calls are safe — tables are created only if they don't exist.

import logging
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    DateTime,
    create_engine,
    text,
)
from sqlalchemy.exc import SQLAlchemyError

from db.config import DATABASE_URL, FEATURES_TABLE, SCORES_TABLE

logger = logging.getLogger(__name__)

metadata = MetaData()

# ── customer_scores ────────────────────────────────────────────────────────────
# One row per customer per scoring run.
# scored_at allows tracking score history across multiple runs.
customer_scores_table = Table(
    SCORES_TABLE,
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", String, nullable=False, index=True),
    Column("churn_risk", Float),
    Column("clv_6_month", Float),       
    Column("clv_12_month", Float),
    Column("clv_tier", String),              # Bronze / Silver / Gold / Platinum
    Column("segment", Integer),
    Column("segment_label", String),         # Champions / Loyal / At-Risk / Inactive
    Column("revenue_at_risk", Float),
    Column("scored_at", DateTime, nullable=False),
    Column("as_of_date", String, nullable=False),  # the feature observation date
)

# ── customer_features ──────────────────────────────────────────────────────────
# Stores the feature store snapshot alongside scores so the agent can
# explain WHY a customer was flagged without re-running the feature pipeline.
customer_features_table = Table(
    FEATURES_TABLE,
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", String, nullable=False, index=True),
    Column("as_of_date", String, nullable=False),
    Column("recency_days", Float),
    Column("frequency", Float),
    Column("tenure_days", Float),
    Column("gross_spend", Float),
    Column("avg_basket_quantity", Float),
    Column("avg_unique_items_per_basket", Float),
    Column("avg_basket_monetary_value", Float),
    Column("purchased_quantity", Float),
    Column("returned_quantity", Float),
    Column("return_events", Float),
    Column("return_rate", Float),
    Column("avg_days_between", Float),
    Column("std_days_between", Float),
    Column("purchase_rate_monthly", Float),
    Column("lifetimes_frequency", Float),
    Column("lifetimes_recency", Float),
    Column("lifetimes_T", Float),
    Column("lifetimes_monetary_value", Float),
)


def get_engine():
    """Create and return a SQLAlchemy engine."""
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def create_all_tables() -> None:
    """Create all tables in PostgreSQL if they don't already exist."""
    engine = get_engine()
    try:
        metadata.create_all(engine)
        logger.info(
            "Tables ensured: '%s', '%s'", SCORES_TABLE, FEATURES_TABLE
        )
    except SQLAlchemyError as e:
        logger.error("Failed to create tables: %s", e)
        raise
    finally:
        engine.dispose()
