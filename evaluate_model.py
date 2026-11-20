"""
Reproducible evaluation for the Uganda Credit Risk Advisory prototype.

Run this in the same Colab environment used to build the model package:
    python evaluate_model.py

It recreates the stratified 80/20 German Credit split (random_state=42),
uses the fitted preprocessing/model objects in credit_risk_model.joblib,
and writes model_evaluation.json.

This evaluates a public benchmark dataset, not Ugandan bank lending data.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

PACKAGE_PATH = "credit_risk_model.joblib"
OUTPUT_PATH = "model_evaluation.json"
RANDOM_STATE = 42
TEST_SIZE = 0.20

package = joblib.load(PACKAGE_PATH)
data = fetch_openml(name="credit-g", version=1, as_frame=True)
X = data.data[package["original_features"]].copy()
y = (data.target.astype(str).str.lower() == "bad").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

preprocessor = package["preprocessor"]
X_test_transformed = preprocessor.transform(X_test)
feature_names = list(package["feature_names"])
X_test_transformed = pd.DataFrame(
    X_test_transformed, columns=feature_names
)

models = {
    "Gradient Boosting": package["gradient_model"],
    "Logistic Regression": package["logistic_model"]
}

results = {
    "dataset": "German Credit (OpenML credit-g)",
    "dataset_rows": int(len(X)),
    "test_rows": int(len(X_test)),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "target_definition": "1 = bad credit; 0 = good credit",
    "positive_class": 1,
    "models": {},
    "fairness": {},
    "limitations": [
        "Evaluation uses a public German Credit benchmark, not Ugandan bank lending data.",
        "A random hold-out split is internal validation, not external validation.",
        "Group metrics are descriptive and do not establish legal or ethical fairness.",
        "Small subgroup sizes can make group metrics unstable."
    ]
}

y_true = np.asarray(y_test, dtype=int)

for name, model in models.items():
    predictions = np.asarray(model.predict(X_test_transformed), dtype=int)
    probabilities = np.asarray(model.predict_proba(X_test_transformed))[:, 1]
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    results["models"][name] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": cm.tolist()
    }

# Descriptive group analysis for selected categorical fields.
# The metrics concern model predictions, not causal or legal fairness.
primary_model = models["Gradient Boosting"]
primary_predictions = np.asarray(
    primary_model.predict(X_test_transformed), dtype=int
)
group_fields = ["personal_status", "foreign_worker"]

for field in group_fields:
    if field not in X_test.columns:
        continue
    groups = {}
    values = X_test[field].fillna("Missing").astype(str)
    for group in sorted(values.unique()):
        mask = (values == group).to_numpy()
        n = int(mask.sum())
        if n < 10:
            groups[group] = {
                "n": n,
                "suppressed": True,
                "note": "Not reported because fewer than 10 test cases are in this group."
            }
            continue
        actual = y_true[mask]
        predicted = primary_predictions[mask]
        positives = int((predicted == 1).sum())
        true_positives = int(((predicted == 1) & (actual == 1)).sum())
        false_positives = int(((predicted == 1) & (actual == 0)).sum())
        actual_bad = int((actual == 1).sum())
        actual_good = int((actual == 0).sum())
        groups[group] = {
            "n": n,
            "bad_class_prediction_rate": float(positives / n),
            "bad_class_recall": (
                float(true_positives / actual_bad) if actual_bad else None
            ),
            "false_positive_rate": (
                float(false_positives / actual_good) if actual_good else None
            )
        }
    results["fairness"][field] = groups

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, ensure_ascii=False)

print(f"Evaluation complete. Saved {OUTPUT_PATH}")
for model_name, metrics in results["models"].items():
    print(
        f"{model_name}: accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}"
    )
