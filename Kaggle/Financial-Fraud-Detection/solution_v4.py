"""v4: tree ensembles (PLAN.md Stage 5) on the v3 safe feature set.

Random Forest, XGBoost, LightGBM compared against the v3 Logistic
Regression reference — same features, same split, only model class
changes. Each ensemble gets both default-threshold and F1-tuned-threshold
(Stage 3 technique) reporting for a fair comparison.

Note: this version reliably segfaulted (SIGSEGV, no Python traceback)
while in development. Root cause, in case it recurs: on macOS,
multiprocessing's default `spawn` start method re-imports the `__main__`
module in every worker process it creates — but only when the script runs
as a real file (`python3 solution_v4.py`), not under `python3 -c "..."`.
`RandomForestClassifier(n_jobs>1)` spawns worker processes via joblib's
loky backend, so with `xgboost`/`lightgbm` imported at module level, every
RF worker was *also* re-importing them and initializing their bundled
OpenMP runtimes — pure waste, since those workers only ever fit sklearn
trees. Leftover state from that duplicate init corrupted the process
enough to crash it the moment the real XGBoost/LightGBM fits ran later.
Fix: import xgboost/lightgbm lazily (inside main(), only right before
they're needed), and explicitly shut down the loky worker pool after the
Random Forest step — see shutdown_loky_workers() below.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24
    df["amount_log"] = np.log1p(df["amount"])
    df["highAmountFlag"] = (df["amount"] > 200_000).astype(int)
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
    best_idx = np.argmax(f1[:-1])
    return thresholds[best_idx], f1[best_idx]


def shutdown_loky_workers() -> None:
    gc.collect()
    get_reusable_executor().shutdown(wait=True)


def add_dest_txn_count(train_dest, X_train, X_val, val_dest):
    counts = train_dest.value_counts()
    X_train["destTxnCount"] = train_dest.map(counts).to_numpy()
    X_val["destTxnCount"] = val_dest.map(counts).fillna(0).to_numpy()
    return X_train, X_val


def eval_model(model, X_train, y_train, X_val, y_val, feature_cols, label):
    preprocessor = ColumnTransformer(
        transformers=[
            ("type_ohe", OneHotEncoder(handle_unknown="ignore"), ["type"])
        ],
        remainder="passthrough",
    )
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", model)])
    pipe.fit(X_train[feature_cols], y_train)

    score = pipe.predict_proba(X_val[feature_cols])[:, 1]
    default_pred = (score >= 0.5).astype(int)
    report(f"{label} (default threshold 0.5)", y_val, default_pred, score)

    thresh, _ = best_f1_threshold(y_val, score)
    tuned_pred = (score >= thresh).astype(int)
    report(f"{label} (F1-tuned threshold {thresh:.4f})", y_val, tuned_pred)

    return {
        "label": label,
        "pr_auc": average_precision_score(y_val, score),
        "roc_auc": roc_auc_score(y_val, score),
        "f1_default": f1_score(y_val, default_pred),
        "f1_tuned": f1_score(y_val, tuned_pred),
    }


def main():
    base_features = [
        "step", "type", "amount", "isMerchantDest",
        "hour", "day", "amount_log", "highAmountFlag",
    ]

    # Trim to only the columns this version needs and drop the raw,
    # untrimmed frame (~1.5GB, mostly nameOrig/nameDest strings) — see the
    # module docstring for why this matters.
    raw = load_data()
    df = raw[base_features + ["isFraud", "nameDest"]].copy()
    del raw
    gc.collect()

    y = df["isFraud"]
    idx_train, idx_val = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )
    X_train = df.loc[idx_train, base_features].copy()
    X_val = df.loc[idx_val, base_features].copy()
    train_dest, val_dest = df.loc[idx_train, "nameDest"], df.loc[idx_val, "nameDest"]
    y_train, y_val = y.loc[idx_train], y.loc[idx_val]
    del df
    gc.collect()

    X_train, X_val = add_dest_txn_count(train_dest, X_train, X_val, val_dest)
    del train_dest, val_dest
    gc.collect()

    safe_features = base_features + ["destTxnCount"]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    results = []

    # --- v3 reference: Logistic Regression on the same feature set ---
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    results.append(
        eval_model(lr, X_train, y_train, X_val, y_val, safe_features,
                   "Reference: Logistic Regression (= v3 safe features)")
    )

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        n_jobs=4, random_state=42,
    )
    results.append(
        eval_model(rf, X_train, y_train, X_val, y_val, safe_features, "Random Forest")
    )
    del rf
    shutdown_loky_workers()

    # --- XGBoost --- (imported lazily, see module docstring)
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, tree_method="hist",
        eval_metric="aucpr", random_state=42, n_jobs=4,
    )
    results.append(
        eval_model(xgb, X_train, y_train, X_val, y_val, safe_features, "XGBoost")
    )

    # --- LightGBM --- (imported lazily, see module docstring)
    from lightgbm import LGBMClassifier

    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        class_weight="balanced", random_state=42, n_jobs=4, verbose=-1,
    )
    results.append(
        eval_model(lgbm, X_train, y_train, X_val, y_val, safe_features, "LightGBM")
    )

    print("\n=== Summary ===")
    print(f"{'Model':45s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'F1@0.5':>8s} {'F1@tuned':>9s}")
    for r in results:
        print(f"{r['label']:45s} {r['pr_auc']:8.4f} {r['roc_auc']:8.4f} "
              f"{r['f1_default']:8.4f} {r['f1_tuned']:9.4f}")


if __name__ == "__main__":
    main()
