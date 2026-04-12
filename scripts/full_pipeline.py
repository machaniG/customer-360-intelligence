import logging
import subprocess
import sys
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/full_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status."""
    logger.info(f"Starting: {description}")
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent.parent)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=env
        )
        logger.info(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error output: {e.stderr}")
        return False


def main():
    """Run the complete Mini CRM pipeline: ETL → Train Models → Score."""
    logger.info("Starting Mini CRM Full Pipeline")

    # Step 1: Run ETL
    if not run_command("python scripts/run_etl.py", "ETL Process"):
        sys.exit(1)

    # Step 2: Train Churn Model
    if not run_command("python scripts/train_churn.py", "Train Churn Model"):
        sys.exit(1)

    # Step 3: Train CLV Model
    if not run_command("python scripts/train_clv.py", "Train CLV Model"):
        sys.exit(1)

    # Step 4: Train Segmentation Model
    if not run_command("python scripts/train_segmentation.py", "Train Segmentation Model"):
        sys.exit(1)

    # Step 5: Run Monthly Scoring
    if not run_command("python scoring/monthly_scoring.py", "Monthly Scoring"):
        sys.exit(1)

    logger.info("Mini CRM Full Pipeline completed successfully!")


if __name__ == "__main__":
    main()

""" This script serves as an orchestrator for the entire Mini CRM pipeline. It sequentially executes the ETL process, trains the churn, CLV, and segmentation models, and finally runs the monthly scoring script. Each step is logged for monitoring and debugging purposes. If any step fails, the pipeline will stop and log the error details. To run the full pipeline, 
# execute `python scripts/full_pipeline.py` from the command line. Make sure to have all the necessary dependencies installed and configured before running the pipeline.
or from the root directory of the project, you can also run it as a module:
`python -m scripts.full_pipeline`  
for retraining models with new date, run the full pipeline by updating as_of_date in the config or by passing it as an argument to the individual training and scoring scripts.
 python -m scripts.full_pipeline --as_of_date 2010-10-31
"""