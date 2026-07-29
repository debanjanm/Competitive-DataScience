"""v1: naive rule-based baselines + Logistic Regression for PaySim fraud detection.

Excludes oldbalanceOrg/newbalanceOrig/oldbalanceDest/newbalanceDest per the
dataset author's explicit leakage warning (fraud transactions are cancelled,
so these balances reflect post-cancellation state, not what actually
happened at transaction time). See PROBLEM.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent / "data" / "PS_20174392719_1491204439457_log.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["isMerchantDest"] = df["nameDest"].str.startswith("M").astype(int)
    return df


def report(name: str, y_true, y_pred, y_score=None) -> None:
    print(f"\n--- {name} ---")
    print(
        classification_report(
            y_true, y_pred, target_names=["legit", "fraud"], digits=4
        )
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred))
    if y_score is not None:
        print(f"PR-AUC:  {average_precision_score(y_true, y_score):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")


def main():
    df = load_data()

    print(f"Rows: {len(df):,}  |  Fraud rate: {df['isFraud'].mean():.4%}")
    print("\nFraud by transaction type:")
    print(df.groupby("type")["isFraud"].agg(["sum", "count"]))

    features = ["step", "type", "amount", "isMerchantDest"]
    X = df[features]
    y = df["isFraud"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Stage 1: naive baselines ---
    majority_pred = np.zeros(len(y_val), dtype=int)
    report("Baseline: majority class (always legit)", y_val, majority_pred)

    flagged_pred = df.loc[X_val.index, "isFlaggedFraud"].to_numpy()
    report("Baseline: isFlaggedFraud rule as-is", y_val, flagged_pred)

    # --- Stage 2: Logistic Regression ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("type_ohe", OneHotEncoder(handle_unknown="ignore"), ["type"])
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_score = model.predict_proba(X_val)[:, 1]
    report(
        "Logistic Regression (class_weight=balanced)", y_val, val_pred, val_score
    )

    print(f"\nValidation F1 (fraud class): {f1_score(y_val, val_pred):.4f}")


if __name__ == "__main__":
    main()
