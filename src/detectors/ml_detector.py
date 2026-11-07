"""Machine learning detectors. Shows INHERITANCE and POLYMORPHISM."""
from abc import abstractmethod

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config
from src.detectors.base_detector import FraudDetector


class MLDetector(FraudDetector):
    """Parent class for every detector that learns from data.

    It builds a pipeline: scale the data -> (optionally) balance with SMOTE -> model.
    Child classes only decide WHICH model to use, by writing _create_model().
    """

    def __init__(self, name, use_smote=False):
        super().__init__(name)
        self._use_smote = use_smote
        self._pipeline = None

    @abstractmethod
    def _create_model(self):
        """Return an untrained scikit-learn model."""

    def _class_weight(self):
        # If SMOTE already balances the data, we do not also need class weights
        return None if self._use_smote else "balanced"

    def _build_pipeline(self):
        steps = [("scaler", StandardScaler())]
        if self._use_smote:
            # SMOTE runs only during fit(), never on test data
            steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
        steps.append(("model", self._create_model()))
        return Pipeline(steps)

    def fit(self, X, y):
        print(f"Training {self._name} ...")
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        self._is_trained = True
        return self

    def predict_proba(self, X):
        self._check_trained()
        return self._pipeline.predict_proba(X)[:, 1]   # column 1 = fraud

    def save(self, path):
        """Save the whole trained detector (scaler + model) to a file."""
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class LogisticRegressionDetector(MLDetector):
    """A simple, fast model that draws a boundary between fraud and genuine."""

    def __init__(self, use_smote=False):
        name = "Logistic Regression" + (" + SMOTE" if use_smote else "")
        super().__init__(name, use_smote)

    def _create_model(self):
        return LogisticRegression(max_iter=1000, class_weight=self._class_weight())


class RandomForestDetector(MLDetector):
    """Many decision trees that vote together. Usually stronger but slower."""

    def __init__(self, n_trees=100, use_smote=False):
        name = "Random Forest" + (" + SMOTE" if use_smote else "")
        super().__init__(name, use_smote)
        self._n_trees = n_trees

    def _create_model(self):
        return RandomForestClassifier(
            n_estimators=self._n_trees,
            class_weight=self._class_weight(),
            random_state=config.RANDOM_STATE,
            n_jobs=-1,   # use all CPU cores
        )
