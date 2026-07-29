"""v3: feature engineering (PLAN.md Stage 4).

Same model class as v1/v2 (Logistic Regression, class_weight="balanced")
so the effect of features is isolated from model choice. Two parts:

  Part A — safe features: temporal, amount transforms, destination
  transaction-frequency (computed from train split only, to avoid leaking
  val-set info into the aggregate).

  Part B — deliberate leakage demonstration: adds the balance-consistency
  features the dataset author explicitly warned against (see PROBLEM.md).
  Shown side by side with Part A specifically to teach what a leakage red
  flag looks like (an implausibly large score jump) — NOT adopted for the
  final model.
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
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent / "data" / "PS_20174392719_1491204439457_log.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["isMerchantDest"] = df["nameDest"].str.startswith("M").astype(int)
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24
    df["amount_log"] = np.log1p(df["amount"])
    df["highAmountFlag"] = (df["amount"] > 200_000).astype(int)
    # Leakage-demo features only — NOT used in the final safe model.
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
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


def add_dest_txn_count(train_idx_df, X_train, X_val):
    """Destination transaction frequency, fit on train only (no val leakage)."""
    counts = train_idx_df["nameDest"].value_counts()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train["destTxnCount"] = train_idx_df["nameDest"].map(counts).to_numpy()
    X_val["destTxnCount"] = (
        X_val["_nameDest"].map(counts).fillna(0).to_numpy()
    )
    return X_train, X_val


def fit_eval(X_train, y_train, X_val, y_val, feature_cols, label):
    preprocessor = ColumnTransformer(
        transformers=[
            ("type_ohe", OneHotEncoder(handle_unknown="ignore"), ["type"])
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train[feature_cols], y_train)
    pred = model.predict(X_val[feature_cols])
    score = model.predict_proba(X_val[feature_cols])[:, 1]
    report(label, y_val, pred, score)
    return average_precision_score(y_val, score), roc_auc_score(y_val, score)


def main():
    df = load_data()

    y = df["isFraud"]
    idx_train, idx_val = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )
    train_df = df.loc[idx_train]
    val_df = df.loc[idx_val]

    X_train = train_df.copy()
    X_val = val_df.copy()
    X_val["_nameDest"] = val_df["nameDest"]
    X_train, X_val = add_dest_txn_count(train_df, X_train, X_val)

    y_train, y_val = y.loc[idx_train], y.loc[idx_val]

    safe_features = [
        "step", "type", "amount", "isMerchantDest",
        "hour", "day", "amount_log", "highAmountFlag", "destTxnCount",
    ]
    v1_features = ["step", "type", "amount", "isMerchantDest"]
    leaky_features = safe_features + ["errorBalanceOrig", "errorBalanceDest"]

    pr_auc_v1, roc_auc_v1 = fit_eval(
        X_train, y_train, X_val, y_val, v1_features,
        "Reference: v1 feature set (for comparison)",
    )
    pr_auc_safe, roc_auc_safe = fit_eval(
        X_train, y_train, X_val, y_val, safe_features,
        "Part A: v3 safe engineered features",
    )
    pr_auc_leaky, roc_auc_leaky = fit_eval(
        X_train, y_train, X_val, y_val, leaky_features,
        "Part B: safe features + balance-consistency (LEAKAGE DEMO — do not use)",
    )

    print("\n=== Summary (PR-AUC / ROC-AUC) ===")
    print(f"{'v1 features (reference)':45s} PR-AUC={pr_auc_v1:.4f}  ROC-AUC={roc_auc_v1:.4f}")
    print(f"{'v3 safe features':45s} PR-AUC={pr_auc_safe:.4f}  ROC-AUC={roc_auc_safe:.4f}")
    print(f"{'v3 + leakage demo features':45s} PR-AUC={pr_auc_leaky:.4f}  ROC-AUC={roc_auc_leaky:.4f}")


if __name__ == "__main__":
    main()
