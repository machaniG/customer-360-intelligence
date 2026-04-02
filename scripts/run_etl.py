import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

from etl.etl import add_transaction_type, add_total_price
from config import RAW_PATH, PROCESSED_PATH

# Ensure the logs directory exists before configuring the file handler
Path("logs").mkdir(parents=True, exist_ok=True)

# Ensure the raw and processed folders exist
RAW_PATH = RAW_PATH      # Input dataset
PROCESSED_PATH = PROCESSED_PATH  # Output dataset
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
RAW_PATH.parent.mkdir(parents=True, exist_ok=True)


# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/etl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


UNIT_PRICE_COL = "UnitPrice"

# Define your global cutoff here or pass it as an argument
CUTOFF_DATE = "2010-11-01"


# run etl
def run_etl():
    logger.info("Starting ETL process...")
    # Load raw data
    df = pd.read_csv(RAW_PATH, parse_dates=["InvoiceDate"])
    
    # 1. Ensure date format and drop missing IDs
    df = df[df['Customer ID'].notna()]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # 2. THE TRUNCATION LAYER (Crucial for CLV Math)
    # This prevents 'Future Leakage' where Recency > T
    original_count = len(df)
    df = df[df['InvoiceDate'] <= pd.to_datetime(CUTOFF_DATE)]
    logger.info(f"Truncated data to {CUTOFF_DATE}. Rows reduced from {original_count} to {len(df)}.")

    # 3. Apply transformations
    df = add_transaction_type(df)
    df = add_total_price(df)

    # 4. Identify one-time-off customers BASED ONLY on truncated data
    # Now a 'one-time' customer is someone who only bought once BEFORE the cutoff
    invoice_counts = df.groupby('Customer ID')["InvoiceNo"].nunique()
    one_time_off_customers = invoice_counts[invoice_counts == 1].index
    df = df[~df['Customer ID'].isin(one_time_off_customers)]
    
    logger.info(f"Removed {len(one_time_off_customers)} one-time customers active before {CUTOFF_DATE}.")

    # Save processed data
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"Processed data saved to {PROCESSED_PATH}.")

if __name__ == "__main__":
    run_etl()

# The script performs the following steps:
# 1. Configures logging to output to both a file and the console.
# 2. Loads the raw dataset from a specified path.
# 3. Drops rows with missing Customer ID and ensures the InvoiceDate is in datetime format.
# 4. Applies transformations to add transaction type and total price.
# 5. Identifies and removes one-time-off customers who only have one purchase.
# 6. Saves the processed dataset to a specified path.
# run it with `python scripts/run_etl.py` from the command line. or `python -m scripts.run_etl` if you are in the root directory of the project.
# or provide output file name as an argument `python scripts/run_etl.py --output processed_data.csv` to specify a custom output file name.