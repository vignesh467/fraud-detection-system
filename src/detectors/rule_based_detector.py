"""Rule and RuleBasedDetector: fraud detection with human-written rules."""
import numpy as np

from src import config
from src.detectors.base_detector import FraudDetector


class Rule:
    """One simple rule, for example: Amount > 2000."""

    _OPERATORS = {
        ">": lambda column, value: column > value,
        "<": lambda column, value: column < value,
    }

    def __init__(self, name, column, operator, value):
        if operator not in self._OPERATORS:
            raise ValueError("operator must be '>' or '<'")
        self.name = name
        self.column = column
        self.operator = operator
        self.value = value

    def check(self, X):
        """Return True for every row where the rule fires."""
        return self._OPERATORS[self.operator](X[self.column], self.value).to_numpy()

    def __repr__(self):
        return f"Rule({self.column} {self.operator} {self.value})"


class RuleBasedDetector(FraudDetector):
    """INHERITANCE: a FraudDetector that uses fixed rules instead of learning.

    COMPOSITION: it HAS a list of Rule objects.
    A transaction is flagged if ANY rule fires.
    """

    def __init__(self, rules=None):
        super().__init__("Rule-Based")
        if rules is None:
            rules = [Rule("High amount", "Amount", ">", config.HIGH_AMOUNT_THRESHOLD)]
        self._rules = list(rules)

    def add_rule(self, rule):
        self._rules.append(rule)

    @property
    def rules(self):
        return list(self._rules)

    def fit(self, X, y):
        # Rules do not learn anything, but we keep the same interface
        # as every other detector so they can all be used the same way.
        self._is_trained = True
        return self

    def predict_proba(self, X):
        self._check_trained()
        fired = np.zeros(len(X), dtype=bool)
        for rule in self._rules:
            fired = fired | rule.check(X)
        return fired.astype(float)   # 1.0 if any rule fired, else 0.0
