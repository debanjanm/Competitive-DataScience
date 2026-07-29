"""v5: unsupervised anomaly detection (PLAN.md Stage 7).

Three approaches, all trained on LEGIT-ONLY transactions (labels never
seen during training, only for evaluation) — a genuinely different framing
from v1-v4's supervised classifiers:

  1. Isolation Forest — tree-based, isolates anomalies via short random
     partition paths.
  2. One-Class SVM — boundary around "normal" in feature space. O(n^2-n^3)
     training cost makes full-scale (5M+ rows) infeasible, so this one
     runs on a subsample; noted explicitly, not an apples-to-apples
     comparison with the others.
  3. Autoencoder (PyTorch) — MLP trained to reconstruct legit
     transactions; high reconstruction error on val = anomalous.

Compared against v4's best supervised model (XGBoost, PR-AUC 0.6091) to
see how much signal unsupervised methods recover without ever seeing a
fraud label during training.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib.externals.loky import get_reusable_executor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM

DATA_PATH = Path(__file__).parent / "data" / "PS_20174392719_1491204439457_log.csv"
XGB_REFERENCE_PR_AUC = 0.6091  # v4 supervised best, for comparison


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


def score_and_report(label, y_val, anomaly_score):
    default_pred = (anomaly_score >= np.percentile(anomaly_score, 99.87)).astype(int)
    report(f"{label} (top 0.13% most-anomalous flagged)", y_val, default_pred, anomaly_score)

    thresh, _ = best_f1_threshold(y_val, anomaly_score)
    tuned_pred = (anomaly_score >= thresh).astype(int)
    report(f"{label} (F1-tuned threshold)", y_val, tuned_pred)

    return {
        "label": label,
        "pr_auc": average_precision_score(y_val, anomaly_score),
        "roc_auc": roc_auc_score(y_val, anomaly_score),
        "f1_tuned": f1_score(y_val, tuned_pred),
    }


def train_autoencoder(X_legit: np.ndarray, input_dim: int, epochs: int = 8):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(input_dim, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 16), nn.ReLU(),
        nn.Linear(16, input_dim),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_legit, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f"  autoencoder epoch {epoch + 1}/{epochs}  "
              f"MSE={total_loss / len(X_legit):.5f}", flush=True)

    return model


def main():
    base_features = [
        "step", "type", "amount", "isMerchantDest",
        "hour", "day", "amount_log", "highAmountFlag",
    ]

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

    counts = train_dest.value_counts()
    X_train["destTxnCount"] = train_dest.map(counts).to_numpy()
    X_val["destTxnCount"] = val_dest.map(counts).fillna(0).to_numpy()
    del train_dest, val_dest
    gc.collect()

    safe_features = base_features + ["destTxnCount"]
    numeric_cols = [c for c in safe_features if c != "type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("type_ohe", OneHotEncoder(handle_unknown="ignore"), ["type"]),
            ("scale", StandardScaler(), numeric_cols),
        ]
    )
    # Fit preprocessing on LEGIT-ONLY training rows — consistent with the
    # "model normal behavior, flag deviations" framing of this version.
    legit_mask = (y_train == 0).to_numpy()
    preprocessor.fit(X_train.loc[legit_mask, safe_features])

    X_train_legit = preprocessor.transform(
        X_train.loc[legit_mask, safe_features]
    ).astype(np.float32)
    X_val_all = preprocessor.transform(X_val[safe_features]).astype(np.float32)
    print(f"Legit training rows: {len(X_train_legit):,}  |  "
          f"Feature dim: {X_train_legit.shape[1]}", flush=True)

    results = []

    # --- Isolation Forest ---
    iso = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42, n_jobs=4
    )
    iso.fit(X_train_legit)
    iso_score = -iso.score_samples(X_val_all)  # higher = more anomalous
    results.append(score_and_report("Isolation Forest", y_val, iso_score))
    del iso
    shutdown_loky_workers()

    # --- One-Class SVM (subsample — O(n^2-n^3) training cost) ---
    rng = np.random.RandomState(42)
    train_sub_idx = rng.choice(len(X_train_legit), size=5000, replace=False)
    X_train_svm = X_train_legit[train_sub_idx]

    val_fraud_mask = (y_val == 1).to_numpy()
    val_legit_idx = np.where(~val_fraud_mask)[0]
    val_legit_sub = rng.choice(val_legit_idx, size=5000, replace=False)
    val_sub_idx = np.concatenate([val_legit_sub, np.where(val_fraud_mask)[0]])
    X_val_svm = X_val_all[val_sub_idx]
    y_val_svm = y_val.to_numpy()[val_sub_idx]

    ocsvm = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
    ocsvm.fit(X_train_svm)
    ocsvm_score = -ocsvm.decision_function(X_val_svm)
    print(f"\n[One-Class SVM evaluated on a {len(X_val_svm):,}-row subsample "
          f"of val, NOT the full val set — see SOLUTION_v5.md]")
    results.append(score_and_report("One-Class SVM (subsample eval)", y_val_svm, ocsvm_score))

    # --- Autoencoder (PyTorch) ---
    print("\nTraining autoencoder...", flush=True)
    model = train_autoencoder(X_train_legit, input_dim=X_train_legit.shape[1])
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X_val_all, dtype=torch.float32)).numpy()
    recon_error = np.mean((X_val_all - recon) ** 2, axis=1)
    results.append(score_and_report("Autoencoder (reconstruction error)", y_val, recon_error))

    print("\n=== Summary ===")
    print(f"{'Model':40s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'F1@tuned':>9s}")
    for r in results:
        print(f"{r['label']:40s} {r['pr_auc']:8.4f} {r['roc_auc']:8.4f} {r['f1_tuned']:9.4f}")
    print(f"\n(Reference — v4 XGBoost, supervised, full val set: "
          f"PR-AUC={XGB_REFERENCE_PR_AUC:.4f})")


if __name__ == "__main__":
    main()
