"""HybridDetector: combines an ML detector with rules. Shows COMPOSITION."""
import numpy as np

from src.detectors.base_detector import FraudDetector


class HybridDetector(FraudDetector):
    """HAS-A machine learning detector and HAS-A rule-based detector.

    Final score = the higher of:
      - the ML model's probability, and
      - rule_weight x the rule score (0 or 1).
    With rule_weight = 0.5, a rule alone can send a transaction for review,
    but only the ML model can push it high enough to be blocked.
    """

    def __init__(self, ml_detector, rule_detector, rule_weight=0.5):
        super().__init__(f"Hybrid ({ml_detector.name} + Rules)")
        self._ml_detector = ml_detector
        self._rule_detector = rule_detector
        self._rule_weight = rule_weight

    def fit(self, X, y):
        self._ml_detector.fit(X, y)
        self._rule_detector.fit(X, y)
        self._is_trained = True
        return self

    def predict_proba(self, X):
        self._check_trained()
        ml_score = self._ml_detector.predict_proba(X)
        rule_score = self._rule_detector.predict_proba(X) * self._rule_weight
        return np.maximum(ml_score, rule_score)
