"""Runs the whole project: load -> prepare -> train -> evaluate -> save -> demo."""
from src import config
from src.alert_manager import AlertManager
from src.data_loader import DataLoader
from src.detectors import DetectorFactory, MLDetector
from src.evaluator import Evaluator
from src.preprocessor import Preprocessor
from src.transaction import Transaction


def main():
    # 1. Load the data
    loader = DataLoader()
    data = loader.load()
    print("Dataset summary:", loader.summary())

    # 2. Clean and split
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare(data)

    # 3. Create every detector with the factory
    detectors = [DetectorFactory.create(name) for name in DetectorFactory.AVAILABLE]

    # 4. Train and evaluate. POLYMORPHISM: the same two calls work for every type.
    evaluator = Evaluator()
    for detector in detectors:
        detector.fit(X_train, y_train)
        evaluator.evaluate(detector, X_test, y_test)

    print("\nResults on the test set:")
    print(evaluator.results_table().to_string(index=False))
    evaluator.save(config.RESULTS_DIR)

    # 5. Save the best machine learning detector to a file
    best_name = evaluator.best_detector_name(metric="F1")
    best = next(d for d in detectors if d.name == best_name)
    print(f"\nBest detector by F1: {best.name}")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(best, MLDetector):
        best.save(config.MODELS_DIR / "best_detector.joblib")
        print("Saved to models/best_detector.joblib")

    # 6. Demo: send a few test transactions through the AlertManager
    alert_manager = AlertManager(best)
    fraud_rows = X_test[y_test == 1].head(3)
    genuine_rows = X_test[y_test == 0].head(3)
    print("\nDemo decisions:")
    for label, rows in [("fraud", fraud_rows), ("genuine", genuine_rows)]:
        for index, row in rows.iterrows():
            transaction = Transaction.from_row(index, row)
            decision, probability = alert_manager.decide(transaction)
            print(f"  {transaction} actually {label:7s} -> {decision.value:6s} (p={probability:.3f})")
    print("Decision counts:", alert_manager.decision_counts())


if __name__ == "__main__":
    main()
