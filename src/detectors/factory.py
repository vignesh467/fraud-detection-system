"""DetectorFactory: creates detectors by name. This is the FACTORY design pattern."""
from src.detectors.hybrid_detector import HybridDetector
from src.detectors.ml_detector import LogisticRegressionDetector, RandomForestDetector
from src.detectors.rule_based_detector import RuleBasedDetector


class DetectorFactory:
    """The rest of the code asks for a detector by name and does not need
    to know how each one is built."""

    AVAILABLE = ["rule", "logistic", "logistic_smote", "random_forest", "hybrid"]

    @staticmethod
    def create(name):
        if name == "rule":
            return RuleBasedDetector()
        if name == "logistic":
            return LogisticRegressionDetector()
        if name == "logistic_smote":
            return LogisticRegressionDetector(use_smote=True)
        if name == "random_forest":
            return RandomForestDetector()
        if name == "hybrid":
            return HybridDetector(RandomForestDetector(), RuleBasedDetector())
        raise ValueError(f"Unknown detector '{name}'. Choose from {DetectorFactory.AVAILABLE}")
