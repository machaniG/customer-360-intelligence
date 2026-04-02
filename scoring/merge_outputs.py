import pandas as pd
import logging
import os
from config import SCORES_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists("SCORES_PATH"):
    os.makedirs("SCORES_PATH")

churn_df = pd.read_csv("SCORES_PATH/churn_predictions.csv")
segment_df = pd.read_csv("SCORES_PATH/customer_value_segments.csv")
clv_df = pd.read_csv("SCORES_PATH/clv_predictions.csv")

final_df = churn_df.merge(segment_df, on="customer_id", how="left")
final_df = final_df.merge(clv_df, on="customer_id", how="left")

final_df.to_csv("SCORES_PATH/customer_360_view.csv", index=False)

# The script performs the following steps:
# 1. Configures logging to output to the console.
# 2. Loads the churn predictions, customer segments, and CLV predictions from their respective CSV files in the scores directory.
# 3. Merges these datasets on the "customer_id" column to create a comprehensive customer 360 view.
# 4. Saves the final merged dataset to a new CSV file called "customer_360_view.csv" in the scores directory. 
# To run the script, execute `python scoring/merge_outputs.py` from the command line, or `
# python -m scoring.merge_outputs ` if you are in the root directory of the project.