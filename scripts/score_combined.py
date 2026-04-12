import logging
import sys
from pathlib import Path

# ensure project root is on path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scoring.combined_scoring import run_combined_scoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting combined scoring")
    run_combined_scoring()
    logger.info("Combined scoring finished")


if __name__ == "__main__":
    main()

# how to run this script:
# python -m scripts.score_combined  or python -m scripts.score_combined --score_date 2010-10-31
# from another script or notebook 
# with new dates, run the individual scoring scripts in scoring/ first, then this one to combine the results.          
