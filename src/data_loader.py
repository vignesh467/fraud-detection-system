"""The DataLoader class: reads the CSV file and describes it."""
from pathlib import Path

import pandas as pd

from src import config


class DataLoader:
    """Loads the dataset and gives a quick summary of it."""

    def __init__(self, file_path=config.DATA_PATH):
        self._file_path = Path(file_path)
        self._data = None   # filled in by load()

    def load(self):
        """Read the CSV file into a pandas DataFrame."""
        if not self._file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._file_path}. "
                "Download creditcard.csv from Kaggle and put it in data/raw/"
            )
        self._data = pd.read_csv(self._file_path)
        return self._data

    def _require_data(self):
        if self._data is None:
            raise RuntimeError("Call load() before using this method")

    def summary(self):
        """Basic facts: size, missing values, duplicates, fraud count."""
        self._require_data()
        counts = self._data[config.TARGET_COLUMN].value_counts()
        fraud = int(counts.get(1, 0))
        return {
            "rows": len(self._data),
            "columns": self._data.shape[1],
            "missing_values": int(self._data.isnull().sum().sum()),
            "duplicate_rows": int(self._data.duplicated().sum()),
            "genuine": int(counts.get(0, 0)),
            "fraud": fraud,
            "fraud_percent": round(100 * fraud / len(self._data), 3),
        }

    def compare_classes(self, top_n=6):
        """Show which columns differ most between fraud and genuine rows.

        This helps you choose rules for the RuleBasedDetector.
        """
        self._require_data()
        means = self._data.groupby(config.TARGET_COLUMN).mean().T
        means.columns = ["genuine_mean", "fraud_mean"]
        # Divide by the column's spread (std) so columns on different scales compare fairly
        spread = self._data.drop(columns=[config.TARGET_COLUMN]).std()
        means["difference_in_stds"] = (means["fraud_mean"] - means["genuine_mean"]).abs() / spread
        means = means.drop(index="Time")
        return means.sort_values("difference_in_stds", ascending=False).head(top_n).round(3)
