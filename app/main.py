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

# Endpoint to get top customers
@app.get("/top-customers")
def top_customers(risk_band: str = None, min_clv: float = None):
    logger.info(f"Fetching top customers to prioritize with risk_band={risk_band} and min_clv={min_clv}")
    from db.queries import fetch_priority_customers
    df = fetch_priority_customers(risk_band=risk_band, min_clv=min_clv)
    return df.to_dict(orient="records") 

# Endpoint to ask AI agent
@app.post("/ask-agent")
def ask_ai_agent(question: str):
    logger.info(f"Received question for AI agent: {question}")
    from app.agent import ask_agent
    response = ask_agent(question)
    return {"response": response} 

# Endpoint to run agent in interactive mode
@app.get("/run-agent")
def run_agent():
    logger.info("Starting AI agent in interactive mode...")
    from app.agent import run_agent
    run_agent()
    return {"status": "Agent interactive mode started. Type 'exit' to quit."}

# Endpoint to fetch high-risk customers with revenue at risk
@app.get("/revenue-at-risk")
def revenue_at_risk(top_n: int = 10):
    logger.info(f"Fetching top {top_n} customers by revenue at risk")
    from db.queries import fetch_revenue_at_risk
    df = fetch_revenue_at_risk(top_n=top_n)
    return df.to_dict(orient="records") 

# Endpoint to fetch customer profile
@app.get("/customer-profile/{customer_id}")
def customer_profile(customer_id: str):
    logger.info(f"Fetching profile for customer_id: {customer_id}")
    from db.queries import fetch_customer_profile
    return fetch_customer_profile(customer_id).to_dict(orient="records")  

# Endpoint to run combined scoring (for testing purposes, not typically exposed in production)
@app.post("/score-customers")
def score_customers_endpoint(score_date: str = None):
    logger.info("Starting scoring job")

    # 1. Export transactions from DB
    from app.database import db_connection, export_transactions_to_csv
    engine = db_connection()
    logger.info("Exporting transactions from database...")

    transactions_path = export_transactions_to_csv(
        engine,
        "/tmp/transactions.csv"
    )
    logger.info(f"Transactions exported to {transactions_path}")

    # 2. Run scoring pipeline (models loaded internally)
    logger.info("Running combined scoring...")
    from scoring.combined_scoring import run_combined_scoring

    try: 
        scores_df, features_df = run_combined_scoring(
            transaction_csv_path=transactions_path,
            score_date=score_date
        )
        logger.info("Combined scoring completed successfully.")

    except Exception as e:
        logger.error(f"Scoring failed: {str(e)}")
        return {"status": "error", "message": str(e)}
    
    # 3. Debug 
    print("===== DEBUG: SCORES DF =====")
    print(scores_df.head())
    print(scores_df.columns)
    print(scores_df.isna().sum())
    
    # 4. Save outputs to DB
    from scripts.store_scores import store_scores
    logger.info("Storing scores and features in the database...")
    store_scores(
        features_df=features_df,
        scores_df=scores_df,
        as_of_date=score_date
    )
    logger.info("Scores and features stored in the database successfully.")

    return {"status": "Scoring complete"}
 
# To run the app, execute `uvicorn main:app --host 0.0.0.0 --port 8000` 
# This will start the FastAPI server, and you can access the endpoints at `http://localhost:8000`. 
# For example, to get top customers, navigate to `http://localhost:8000/top-customers` or 
# use query parameters like `http://localhost:8000/top-customers?risk_band=high&min_clv=100`. 
# To ask the AI agent a question, send a POST request to `http://localhost:8000/ask-agent` 
# with a JSON body like `{"question": "What is the churn risk for customer 123?"}`.
# To fetch a customer profile, navigate to `http://localhost:8000/customer-profile/{customer_id}`, replacing `{customer_id}` with the actual ID. 
# The `/run-scoring` endpoint can be triggered to execute the combined scoring process and store results in the database, which is useful for testing and demonstration purposes.   
# In a production environment, you would typically schedule the combined scoring to run as a batch job rather than exposing it as an API endpoint.  
# Make sure to have the database set up and the necessary tables created before running the API, as it relies on fetching data from the database.   
# You can also add error handling and validation as needed to make the API more robust.
#        