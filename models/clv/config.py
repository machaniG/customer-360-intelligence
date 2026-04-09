from pathlib import Path
import yaml

# Directories

BASE_DIR = Path("artifacts")

MODEL_DIR = BASE_DIR / "models" / "clv"
MODEL_PATH = MODEL_DIR / "clv_model.pkl"

# Import OUTPUTS_PATH from global config to avoid circular import
import sys
from pathlib import Path as GlobalPath
_global_config_path = GlobalPath(__file__).resolve().parent.parent / "config.py"
if str(_global_config_path.parent) not in sys.path:
    sys.path.insert(0, str(_global_config_path.parent))

from config import OUTPUTS_PATH, CONFIG_PATH
OUTPUT_PATH = OUTPUTS_PATH / "customer_life_value.csv"

# Load CLV config from YAML
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    CLV_TIER_QUANTILES = config.get('clv', {}).get('tier_quantiles', {
        "Bronze": 0.50,
        "Silver": 0.75,
        "Gold": 0.90,
    })
except (FileNotFoundError, yaml.YAMLError):
    # Fallback to defaults if YAML not found or invalid
    CLV_TIER_QUANTILES = {
        "Bronze": 0.50,
        "Silver": 0.75,
        "Gold": 0.90,
    }

# Feature Configuration

GAMMA_FEATURES = [
    "lifetimes_frequency",
    "lifetimes_monetary_value"
]

BETA_FEATURES = [
    "lifetimes_frequency",
    "lifetimes_recency",
    "lifetimes_T"
]

