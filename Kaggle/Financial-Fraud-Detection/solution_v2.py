"""v2: imbalance-handling techniques on top of the v1 feature set/model class.

Holds features and model class (Logistic Regression) constant vs v1 so the
effect of each imbalance technique is isolated:
  1. Threshold tuning on v1's class_weight="balanced" model.
  2. Random undersampling of the majority class + plain LR.
  3. SMOTE oversampling of the minority class + plain LR.

Same leakage exclusions as v1 (see PROBLEM.md / SOLUTION_v1.md): balance
columns dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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


def best_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_idx = np.argmax(f1[:-1])  # last point has no corresponding threshold
    return thresholds[best_idx], f1[best_idx]


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("type_ohe", OneHotEncoder(handle_unknown="ignore"), ["type"])
        ],
        remainder="passthrough",
    )


def main():
    df = load_data()

    features = ["step", "type", "amount", "isMerchantDest"]
    X = df[features]
    y = df["isFraud"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = make_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)

    # --- Technique 1: threshold tuning on v1's balanced-weight model ---
    balanced_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    balanced_model.fit(X_train_enc, y_train)
    val_score = balanced_model.predict_proba(X_val_enc)[:, 1]

    default_pred = (val_score >= 0.5).astype(int)
    report(
        "Reference: class_weight=balanced, default threshold 0.5 (= v1)",
        y_val,
        default_pred,
        val_score,
    )

    best_thresh, best_f1 = best_f1_threshold(y_val, val_score)
    tuned_pred = (val_score >= best_thresh).astype(int)
    report(
        f"Technique 1: threshold tuned to {best_thresh:.4f} (max F1 on val)",
        y_val,
        tuned_pred,
        val_score,
    )

    # --- Technique 2: random undersampling + plain LR ---
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train_enc, y_train)
    print(f"\nAfter undersampling: {len(y_rus):,} rows, "
          f"fraud rate {y_rus.mean():.2%}")

    rus_model = LogisticRegression(max_iter=1000)
    rus_model.fit(X_rus, y_rus)
    rus_pred = rus_model.predict(X_val_enc)
    rus_score = rus_model.predict_proba(X_val_enc)[:, 1]
    report("Technique 2: random undersampling + plain LR", y_val, rus_pred, rus_score)

    # --- Technique 3: SMOTE oversampling + plain LR ---
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train_enc, y_train)
    print(f"\nAfter SMOTE: {len(y_sm):,} rows, fraud rate {y_sm.mean():.2%}")

    sm_model = LogisticRegression(max_iter=1000)
    sm_model.fit(X_sm, y_sm)
    sm_pred = sm_model.predict(X_val_enc)
    sm_score = sm_model.predict_proba(X_val_enc)[:, 1]
    report("Technique 3: SMOTE oversampling + plain LR", y_val, sm_pred, sm_score)

    # --- Summary ---
    print("\n=== Summary (fraud class) ===")
    for name, pred in [
        ("v1 reference (balanced, thresh=0.5)", default_pred),
        (f"threshold tuned ({best_thresh:.3f})", tuned_pred),
        ("undersampling", rus_pred),
        ("SMOTE", sm_pred),
    ]:
        print(f"{name:38s} F1={f1_score(y_val, pred):.4f}")


if __name__ == "__main__":
    main()
