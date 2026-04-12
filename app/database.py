# app/database.py
# DB connection and utility functions for writing scores and features to PostgreSQL.

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError  
from db.config import (
    DATABASE_URL, FEATURES_TABLE, SCORES_TABLE, 
    DB_HOST,DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL)

def db_connection():
    """connect to PostgreSQL database."""
    # connect to the database
    try:
        #engine = engine.connect()
        logger.info("Connection to PostgreSQL database successful!")
        return engine
    except SQLAlchemyError as e:
        logger.error(f"Error connecting to PostgreSQL database: {e}")  
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return None

def export_transactions_to_csv(engine, output_path):
    query = "SELECT * FROM transactions"
    df = pd.read_sql(query, engine)
    df.to_csv(output_path, index=False)
    return output_path

