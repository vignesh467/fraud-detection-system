# Credit Card Fraud Detection System (OOP + Machine Learning)

A modular, object-oriented system that detects fraudulent credit card transactions.
It compares a rule-based detector, Logistic Regression, Random Forest and a hybrid
detector on a highly imbalanced real-world dataset, and turns each prediction into
an action: ALLOW, REVIEW or BLOCK.

## Problem
Card fraud costs banks and customers money. Fraud is very rare (about 0.17% of
transactions), so a model that predicts "genuine" every time would be 99.8% accurate
and completely useless. This project focuses on precision and recall instead.

## Dataset
Credit Card Fraud Detection (ULB Machine Learning Group), Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

284,807 transactions, 492 frauds. Columns: Time, V1-V28 (anonymised with PCA), Amount, Class.
The dataset is not included in this repository. Download `creditcard.csv` and place it in `data/raw/`.

## OOP design
| Concept | Where |
|---|---|
| Encapsulation | `Transaction` (private attributes, validated setter) |
| Abstraction | `FraudDetector` abstract base class |
| Inheritance | `RuleBasedDetector`, `MLDetector` -> `LogisticRegressionDetector`, `RandomForestDetector` |
| Polymorphism | `main.py` and `Evaluator` call `fit()` / `predict_proba()` on any detector |
| Composition | `HybridDetector` has an ML detector and a rule detector; `AlertManager` has a detector |
| Factory pattern | `DetectorFactory.create(name)` |

## Project structure
```
fraud-detection-system/
├── data/raw/            # creditcard.csv goes here (not committed)
├── models/              # saved trained model (not committed)
├── results/             # results.csv and plots
├── src/
│   ├── config.py
│   ├── transaction.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── evaluator.py
│   ├── alert_manager.py
│   └── detectors/
│       ├── base_detector.py
│       ├── rule_based_detector.py
│       ├── ml_detector.py
│       ├── hybrid_detector.py
│       └── factory.py
├── tests/
├── explore.py
├── main.py
└── requirements.txt
```

## How to run
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python explore.py            # look at the data, saves plots
python main.py               # train, evaluate, save results and the best model
pytest                       # run the unit tests
```

## Results
Test set: 56,746 transactions (95 frauds), decision threshold 0.5, `RANDOM_STATE = 42`.
Full numbers are in `results/results.csv`, with one confusion matrix per detector in `results/`.

| Detector | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Rule-Based | 0.000 | 0.000 | 0.000 | 0.002 |
| Logistic Regression | 0.056 | 0.874 | 0.106 | 0.672 |
| Logistic Regression + SMOTE | 0.053 | 0.874 | 0.100 | 0.675 |
| **Random Forest** | **0.945** | **0.726** | **0.821** | **0.801** |
| Hybrid (Random Forest + Rules) | 0.321 | 0.726 | 0.445 | 0.746 |

Random Forest is the best detector by F1 and is saved to `models/best_detector.joblib`.
With only 95 frauds in the test set, each fraud is worth about 1% of recall, so small
differences between detectors are within noise.

## Future work
MLOps: experiment tracking (MLflow), REST API (FastAPI), Docker, CI with GitHub Actions.

## Team
- Name (roll number)
