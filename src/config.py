"""All project settings live here, so you only change them in one place."""
from pathlib import Path

# Folder paths (built from this file's location, so they work on any computer)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset settings
TARGET_COLUMN = "Class"      # 1 = fraud, 0 = genuine
TEST_SIZE = 0.2              # 20% of rows are kept aside for testing
RANDOM_STATE = 42            # fixed seed so results are the same every run

# Rule-based detector: a transaction above this amount is suspicious
HIGH_AMOUNT_THRESHOLD = 2000.0

# Alert manager: what to do based on the fraud probability
BLOCK_THRESHOLD = 0.90       # probability >= 0.90 -> block the transaction
REVIEW_THRESHOLD = 0.50      # probability >= 0.50 -> send for manual review
