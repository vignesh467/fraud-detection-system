"""FraudDetector: the abstract parent class of every detector. Shows ABSTRACTION."""
from abc import ABC, abstractmethod


class FraudDetector(ABC):
    """A blueprint that says what every detector MUST be able to do.

    You cannot create a FraudDetector object directly. You can only create
    child classes that fill in the abstract methods fit() and predict_proba().
    """

    def __init__(self, name):
        self._name = name
        self._is_trained = False

    @property
    def name(self):
        return self._name

    @property
    def is_trained(self):
        return self._is_trained

    @abstractmethod
    def fit(self, X, y):
        """Learn from training data. Every child class must write this."""

    @abstractmethod
    def predict_proba(self, X):
        """Return the probability of fraud (0 to 1) for each row."""

    def predict(self, X, threshold=0.5):
        """Shared by all children: turn probabilities into 0 (genuine) or 1 (fraud)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError(f"{self._name} must be trained with fit() first")

    def __str__(self):
        return f"{self.__class__.__name__}(name='{self._name}')"
