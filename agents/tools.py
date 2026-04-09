# agent/tools.py

import json
import logging
from typing import Optional
from langchain.tools import tool
from db.queries import (
    fetch_customer_profile,
    fetch_priority_customers,
    fetch_revenue_at_risk,
    fetch_segment_summary,
)

logger = logging.getLogger(__name__)

def _df_to_json(df) -> str:
    if df.empty:
        return json.dumps([])
    return df.to_json(orient="records", indent=2, default_handler=str)

@tool
def get_priority_customers(top_n: int = 10, risk_band: Optional[str] = None, min_clv: Optional[float] = None) -> str:
    """
    Fetch top customers ranked by revenue at risk.
    risk_band options: 
      - 'high': churn_risk > 0.5
      - 'medium': churn_risk between 0.3 and 0.5
      - 'low': churn_risk < 0.3
    """
    logger.info("Tool: get_priority_customers(top_n=%d, risk_band=%s, min_clv=%s)", top_n, risk_band, min_clv)
    df = fetch_priority_customers(top_n=top_n, risk_band=risk_band, min_clv=min_clv)
    return _df_to_json(df)

@tool
def get_revenue_at_risk(top_n: int = 10) -> str:
    """Fetch high-value customers with churn_risk > 0.5, ordered by CLV_12_month."""
    logger.info("Tool: get_revenue_at_risk(top_n=%d)", top_n)
    df = fetch_revenue_at_risk(top_n=top_n)
    return _df_to_json(df)

@tool
def get_customer_profile(customer_id: str) -> str:
    """Fetch profile including purchase_rate_monthly, return_rate, and tenure_days."""
    logger.info("Tool: get_customer_profile(customer_id=%s)", customer_id)
    df = fetch_customer_profile(customer_id=str(customer_id))
    return _df_to_json(df)

@tool
def get_segment_summary() -> str:
    """Get aggregated metrics (avg_purchase_rate_monthly, avg_clv_12m) by segment."""
    logger.info("Tool: get_segment_summary()")
    df = fetch_segment_summary()
    return _df_to_json(df)

AGENT_TOOLS = [
    get_priority_customers,
    get_revenue_at_risk,
    get_customer_profile,
    get_segment_summary,
]