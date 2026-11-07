"""Makes the detector classes easy to import, e.g.
from src.detectors import RandomForestDetector
"""
from src.detectors.base_detector import FraudDetector
from src.detectors.factory import DetectorFactory
from src.detectors.hybrid_detector import HybridDetector
from src.detectors.ml_detector import (
    LogisticRegressionDetector,
    MLDetector,
    RandomForestDetector,
)
from src.detectors.rule_based_detector import Rule, RuleBasedDetector
