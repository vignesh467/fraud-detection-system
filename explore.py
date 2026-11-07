"""Step 1 of the project: look at the data before building any model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data_loader import DataLoader


def main():
    loader = DataLoader()
    data = loader.load()

    print("First 5 rows:")
    print(data.head())
    print("\nSummary:", loader.summary())
    print("\nColumns that differ most between fraud and genuine (use these for rules):")
    print(loader.compare_classes())

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: how many fraud vs genuine rows
    counts = data[config.TARGET_COLUMN].value_counts().sort_index()
    plt.bar(["Genuine (0)", "Fraud (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    plt.yscale("log")   # log scale, otherwise the fraud bar is invisible
    plt.title("Class balance (log scale)")
    plt.ylabel("Number of transactions")
    plt.savefig(config.RESULTS_DIR / "class_balance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Plot 2: transaction amounts for each class
    for label, colour in [(0, "#4C72B0"), (1, "#C44E52")]:
        amounts = data.loc[data[config.TARGET_COLUMN] == label, "Amount"]
        plt.hist(amounts, bins=50, range=(0, 1000), alpha=0.6, density=True,
                 color=colour, label="Fraud" if label else "Genuine")
    plt.legend()
    plt.title("Transaction amounts (up to 1000)")
    plt.xlabel("Amount")
    plt.savefig(config.RESULTS_DIR / "amount_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\nPlots saved in the results folder.")


if __name__ == "__main__":
    main()
