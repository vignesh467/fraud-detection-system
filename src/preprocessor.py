"""The Preprocessor class: cleans the data and splits it for training and testing."""
from sklearn.model_selection import train_test_split

from src import config


class Preprocessor:
    """Prepares raw data so the detectors can learn from it.

    Note: feature scaling is NOT done here. Each ML detector scales the data
    inside its own pipeline, so the scaler only ever learns from training data.
    """

    def __init__(self, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
        self._test_size = test_size
        self._random_state = random_state


    def clean(self, data):
        """Remove duplicate rows and rows with missing values."""
        before = len(data)
        data = data.drop_duplicates().dropna()
        print(f"Cleaning: removed {before - len(data)} rows, {len(data)} remain")
        return data

    def split(self, data):
        """Split into features (X) and labels (y), then into train and test sets.

        stratify=y keeps the same fraud percentage in both sets. Without it,
        the test set could end up with very few frauds by bad luck.
        """
        X = data.drop(columns=[config.TARGET_COLUMN])
        y = data[config.TARGET_COLUMN]
        return train_test_split(
            X, y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y,
        )

    def prepare(self, data):
        """Run the full preparation: clean, then split."""
        data = self.clean(data)
        X_train, X_test, y_train, y_test = self.split(data)
        print(f"Train rows: {len(X_train)} (frauds: {int(y_train.sum())})")
        print(f"Test rows:  {len(X_test)} (frauds: {int(y_test.sum())})")
        return X_train, X_test, y_train, y_test
