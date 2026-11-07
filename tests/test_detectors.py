import numpy as np
import pandas as pd
import pytest

from src.alert_manager import AlertManager, Decision
from src.detectors import (
    DetectorFactory,
    FraudDetector,
    HybridDetector,
    LogisticRegressionDetector,
    Rule,
    RuleBasedDetector,
)
from src.transaction import Transaction


@pytest.fixture
def small_data():
    """A tiny fake dataset with the same columns as the real one."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    X.insert(0, "Time", np.arange(n, dtype=float))
    X["Amount"] = rng.uniform(1, 500, size=n)
    y = pd.Series(np.zeros(n, dtype=int))
    y[:40] = 1
    X.loc[:39, "V14"] -= 4   # make the frauds easy to spot
    return X, y


def test_abstract_class_cannot_be_created():
    with pytest.raises(TypeError):
        FraudDetector("test")


def test_rule_based_flags_high_amounts(small_data):
    X, y = small_data
    X = X.copy()
    X.loc[0, "Amount"] = 5000
    detector = RuleBasedDetector().fit(X, y)
    predictions = detector.predict(X)
    assert predictions[0] == 1
    assert predictions[1:].sum() == 0


def test_custom_rule_can_be_added(small_data):
    X, y = small_data
    detector = RuleBasedDetector(rules=[])
    detector.add_rule(Rule("Low V14", "V14", "<", -3))
    detector.fit(X, y)
    assert detector.predict(X)[:40].mean() > 0.5


def test_untrained_detector_raises(small_data):
    X, _ = small_data
    with pytest.raises(RuntimeError):
        LogisticRegressionDetector().predict_proba(X)


def test_ml_detector_outputs_probabilities(small_data):
    X, y = small_data
    detector = LogisticRegressionDetector().fit(X, y)
    probabilities = detector.predict_proba(X)
    assert len(probabilities) == len(X)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_hybrid_score_is_at_least_ml_score(small_data):
    X, y = small_data
    ml = LogisticRegressionDetector()
    hybrid = HybridDetector(ml, RuleBasedDetector()).fit(X, y)
    assert (hybrid.predict_proba(X) >= ml.predict_proba(X)).all()


def test_factory_creates_every_detector():
    for name in DetectorFactory.AVAILABLE:
        assert isinstance(DetectorFactory.create(name), FraudDetector)


def test_factory_rejects_unknown_name():
    with pytest.raises(ValueError):
        DetectorFactory.create("magic")


class FixedScoreDetector(FraudDetector):
    """A fake detector that always returns the same score (for testing)."""

    def __init__(self, score):
        super().__init__("Fixed")
        self._score = score
        self._is_trained = True

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.full(len(X), self._score)


@pytest.mark.parametrize("score, expected", [
    (0.95, Decision.BLOCK),
    (0.60, Decision.REVIEW),
    (0.10, Decision.ALLOW),
])
def test_alert_manager_decisions(score, expected):
    transaction = Transaction("t1", 50, 0, {f"V{i}": 0.0 for i in range(1, 29)})
    decision, _ = AlertManager(FixedScoreDetector(score)).decide(transaction)
    assert decision == expected
