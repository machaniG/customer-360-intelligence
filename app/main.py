# app/main.py (FastAPI)

from fastapi import FastAPI
from app.database import db_connection
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()



# HEALTH CHECK (Product-style)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Customer 360 API is running"}



# TOP CUSTOMERS

@app.get("/top-customers")
def top_customers(risk_band: str = None, min_clv: float = None):
    logger.info(f"Fetching top customers | risk_band={risk_band}, min_clv={min_clv}")

    from db.queries import fetch_priority_customers
    df = fetch_priority_customers(risk_band=risk_band, min_clv=min_clv)

    logger.info(f"Returned {len(df)} customers")

    return {
        "count": len(df),
        "data": df.to_dict(orient="records")
    }



# REVENUE AT RISK

@app.get("/revenue-at-risk")
def revenue_at_risk(top_n: int = 10):
    logger.info(f"Fetching top {top_n} customers by revenue at risk")

    from db.queries import fetch_revenue_at_risk
    df = fetch_revenue_at_risk(top_n=top_n)

    total_risk = df["revenue_at_risk"].sum()

    logger.info(f"Total revenue at risk (top {top_n}): £{total_risk:,.2f}")

    return {
        "top_n": top_n,
        "total_revenue_at_risk": float(total_risk),
        "data": df.to_dict(orient="records")
    }



# CUSTOMER PROFILE

@app.get("/customer-profile/{customer_id}")
def customer_profile(customer_id: str):
    logger.info(f"Fetching profile for customer_id: {customer_id}")

    from db.queries import fetch_customer_profile
    df = fetch_customer_profile(customer_id)

    if df.empty:
        return {"status": "not_found", "customer_id": customer_id}

    return {
        "customer_id": customer_id,
        "data": df.to_dict(orient="records")
    }



# AI AGENT

@app.post("/ask-agent")
def ask_ai_agent(question: str):
    logger.info(f"Agent question: {question}")

    from app.agent import ask_agent
    response = ask_agent(question)

    return {
        "question": question,
        "response": response
    }



# SCORING PIPELINE (CORE ENGINE)

@app.post("/score-customers")
def score_customers_endpoint(score_date: str = None):
    logger.info(f"Starting scoring job | score_date={score_date}")

    # 1. Export transactions
    from app.database import export_transactions_to_csv
    engine = db_connection()

    logger.info("Exporting transactions from database...")
    transactions_path = export_transactions_to_csv(
        engine,
        "transactions.csv"  
    )
    logger.info(f"Transactions exported to {transactions_path}")

    # 2. Run scoring
    from scoring.combined_scoring import run_combined_scoring

    try:
        scores_df, features_df = run_combined_scoring(
            transaction_csv_path=transactions_path,
            score_date=score_date
        )
        logger.info("Scoring pipeline completed")

    except Exception as e:
        logger.error(f"Scoring failed: {str(e)}")
        return {"status": "error", "message": str(e)}

    
    # VALIDATION GUARDS

    logger.info("Running validation checks...")

    if features_df.empty:
        raise ValueError("Features DataFrame is empty")

    if scores_df.empty:
        raise ValueError("Scores DataFrame is empty")

    required_cols = ["Customer ID", "churn_risk", "CLV_12_month", "revenue_at_risk"]

    missing_cols = [col for col in required_cols if col not in scores_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in scores_df: {missing_cols}")


    # BUSINESS LOGGING AND METRICS

    total_customers = len(scores_df)
    total_revenue_at_risk = scores_df["revenue_at_risk"].sum()

    logger.info(f"Customers scored: {total_customers}")
    logger.info(f"Total revenue at risk: £{total_revenue_at_risk:,.2f}")


    # STORE RESULTS IN DB

    from scripts.store_scores import store_scores

    logger.info("Writing results to database...")

    store_scores(
        features_df=features_df,
        scores_df=scores_df,
        as_of_date=score_date
    )

    logger.info("Database write complete")

    return {
        "status": "success",
        "customers_scored": total_customers,
        "total_revenue_at_risk": float(total_revenue_at_risk),
        "score_date": score_date
    }