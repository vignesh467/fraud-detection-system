"""AlertManager: decides what to do with each transaction."""
from enum import Enum

import pandas as pd

from src import config


class Decision(Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"   # a human from the bank checks it
    BLOCK = "BLOCK"


class AlertManager:
    """HAS-A detector (composition) and turns its score into an action."""

    def __init__(self, detector,
                 block_threshold=config.BLOCK_THRESHOLD,
                 review_threshold=config.REVIEW_THRESHOLD):
        if review_threshold > block_threshold:
            raise ValueError("review_threshold must be lower than block_threshold")
        self._detector = detector
        self._block_threshold = block_threshold
        self._review_threshold = review_threshold
        self._log = []   # history of every decision made

    def decide(self, transaction):
        """Score one Transaction object and return (decision, probability)."""
        row = pd.DataFrame([transaction.to_dict()])
        probability = float(self._detector.predict_proba(row)[0])

        if probability >= self._block_threshold:
            decision = Decision.BLOCK
        elif probability >= self._review_threshold:
            decision = Decision.REVIEW
        else:
            decision = Decision.ALLOW

        self._log.append((transaction.transaction_id, decision, probability))
        return decision, probability

    def decision_counts(self):
        counts = {d.value: 0 for d in Decision}
        for _, decision, _ in self._log:
            counts[decision.value] += 1
        return counts
