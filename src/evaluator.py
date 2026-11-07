"""The Evaluator class: measures how good each detector is."""
import matplotlib
matplotlib.use("Agg")   # save plots to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluator:
    """Tests detectors on unseen data and keeps all their scores."""

    def __init__(self):
        self._results = []
        self._matrices = {}

    def evaluate(self, detector, X_test, y_test, threshold=0.5):
        """POLYMORPHISM: works with ANY FraudDetector, whatever its type."""
        probabilities = detector.predict_proba(X_test)
        predictions = (probabilities >= threshold).astype(int)

        result = {
            "Detector": detector.name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "PR-AUC": average_precision_score(y_test, probabilities),
        }
        self._results.append(result)
        self._matrices[detector.name] = confusion_matrix(y_test, predictions)
        return result

    def results_table(self):
        return pd.DataFrame(self._results).round(4)

    def best_detector_name(self, metric="F1"):
        table = self.results_table()
        return table.loc[table[metric].idxmax(), "Detector"]

    def save(self, folder):
        """Save the results table as CSV and one confusion-matrix picture per detector."""
        folder.mkdir(parents=True, exist_ok=True)
        self.results_table().to_csv(folder / "results.csv", index=False)
        for name, matrix in self._matrices.items():
            display = ConfusionMatrixDisplay(matrix, display_labels=["Genuine", "Fraud"])
            display.plot(cmap="Blues", values_format="d")
            plt.title(name)
            # Keep only letters and digits so the name is safe to use as a file name
            safe_name = "".join(c if c.isalnum() else "_" for c in name.lower())
            plt.savefig(folder / f"confusion_{safe_name}.png", dpi=120, bbox_inches="tight")
            plt.close()
