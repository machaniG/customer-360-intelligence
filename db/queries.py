import logging
from typing import Optional
import pandas as pd
from sqlalchemy import text
from db.config import FEATURES_TABLE, SCORES_TABLE
from db.schema import get_engine

logger = logging.getLogger(__name__)

def _latest_as_of_date(conn) -> str:
    """Return the most recent as_of_date present in the scores table."""
    result = conn.execute(
        text(f"SELECT MAX(as_of_date) AS latest FROM {SCORES_TABLE}")
    ).fetchone()
    if result is None or result.latest is None:
        raise ValueError(f"No data found in '{SCORES_TABLE}'.")
    return str(result.latest)

def fetch_priority_customers(
    top_n: int = 10,
    risk_band: Optional[str] = None,
    min_clv: Optional[float] = None,
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        aod = as_of_date or _latest_as_of_date(conn)
        
        # Base Query - Updated column names to match schema.py
        query = f"""
            SELECT 
                s.customer_id, s.churn_risk, s.clv_12_month, s.segment_label, s.revenue_at_risk,
                f.recency_days, f.frequency, f.tenure_days, f.purchase_rate_monthly
            FROM {SCORES_TABLE} s
            JOIN {FEATURES_TABLE} f ON s.customer_id = f.customer_id AND s.as_of_date = f.as_of_date
            WHERE s.as_of_date = :aod
        """
        
        params = {"aod": aod, "limit": top_n}
        
        # risk_band logic mapped to 'churn_risk'
        if risk_band == 'high':
            query += " AND s.churn_risk > 0.4"
        elif risk_band == 'medium':
            query += " AND s.churn_risk BETWEEN 0.3 AND 0.4"
        elif risk_band == 'low':
            query += " AND s.churn_risk < 0.3"
            
        if min_clv:
            query += " AND s.clv_12_month >= :min_clv"
            params["min_clv"] = min_clv
            
        query += " ORDER BY s.revenue_at_risk DESC LIMIT :limit"
        
        return pd.read_sql(text(query), conn, params=params)

def fetch_revenue_at_risk(top_n: int = 10) -> pd.DataFrame:
    """Fetch high-value customers with churn_risk > 0.4."""
    engine = get_engine()
    with engine.connect() as conn:
        aod = _latest_as_of_date(conn)
        query = f"""
            SELECT customer_id, churn_risk, clv_12_month, revenue_at_risk, segment_label
            FROM {SCORES_TABLE}
            WHERE as_of_date = :aod AND churn_risk > 0.4
            ORDER BY revenue_at_risk DESC LIMIT :limit
        """
        return pd.read_sql(text(query), conn, params={"aod": aod, "limit": top_n})

def fetch_customer_profile(customer_id: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        query = f"""
            SELECT s.*, f.*
            FROM {SCORES_TABLE} s
            JOIN {FEATURES_TABLE} f ON s.customer_id = f.customer_id AND s.as_of_date = f.as_of_date
            WHERE s.customer_id = :cid
            ORDER BY s.as_of_date DESC LIMIT 1
        """
        return pd.read_sql(text(query), conn, params={"cid": customer_id})

def fetch_segment_summary() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        aod = _latest_as_of_date(conn)
        # Updated to use churn_risk, CLV_12_month and purchase_rate_monthly
        query = f"""
            SELECT 
                s.segment_label,
                COUNT(*) AS customer_count,
                ROUND(AVG(s.churn_risk)::numeric, 3) AS avg_churn_risk,
                ROUND(AVG(s.clv_12_month)::numeric, 2) AS avg_clv_12m,
                ROUND(SUM(s.revenue_at_risk)::numeric, 2) AS total_revenue_at_risk,
                ROUND(AVG(f.purchase_rate_monthly)::numeric, 2) AS avg_purchase_rate
            FROM {SCORES_TABLE} s
            LEFT JOIN {FEATURES_TABLE} f ON s.customer_id = f.customer_id AND s.as_of_date = f.as_of_date
            WHERE s.as_of_date = :aod
            GROUP BY s.segment_label
            ORDER BY avg_clv_12m DESC
        """
        return pd.read_sql(text(query), conn, params={"aod": aod})
    
# fetch total revenue at risk for KPI card
def fetch_revenue_summary() -> dict:
    """Aggregate stats for all customers with churn_risk > 0.4."""
    engine = get_engine()
    with engine.connect() as conn:
        aod = _latest_as_of_date(conn)
        query = f"""
            SELECT 
                COUNT(*) FILTER (WHERE revenue_at_risk > 0 AND churn_risk > 0.4) as total_customers,
                SUM(revenue_at_risk) FILTER (WHERE revenue_at_risk > 0) as total_revenue_at_risk,
                AVG(clv_12_month) FILTER (WHERE clv_12_month > 0) as avg_clv
            FROM {SCORES_TABLE}
            WHERE as_of_date = :aod AND churn_risk > 0.4
        """
        row = conn.execute(text(query), {"aod": aod}).fetchone()
        return {
            "total_customers": int(row.total_customers),
            "total_revenue_at_risk": float(row.total_revenue_at_risk or 0),
            "avg_clv": float(row.avg_clv or 0)
        }