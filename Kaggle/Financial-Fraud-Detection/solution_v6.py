"""v6: deep learning + graph-inspired features (PLAN.md Stage 8).

Two parts:

  Part A — MLP (PyTorch) supervised classifier on v3's safe feature set,
  compared against v4's XGBoost. Tests the plan's stated hypothesis that
  tabular deep learning rarely beats gradient boosting.

  Part B — graph-inspired account features fed into XGBoost (the Stage 5
  winner), per PLAN.md's stated lighter-weight alternative to a full GNN
  (PyTorch Geometric wasn't installed and pulls in fragile compiled
  extensions — not worth the setup risk for this scope). Two features
  tested:
    - destPlaysOrigRole: has this nameDest ever acted as a nameOrig
      elsewhere in training (money passing back out — mule pattern)?
    - origPlaysDestRole: has this nameOrig ever acted as a nameDest
      elsewhere in training (this account previously received money)?
  A quick EDA pass (see SOLUTION_v6.md) found both barely differ between
  fraud and legit in this dataset — included anyway to quantify precisely
  rather than assume, and because a negative result here is itself a
  useful finding.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
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

DATA_PATH = Path(__file__).parent / "data" / "PS_20174392719_1491204439457_log.csv"
XGB_V4_REFERENCE_PR_AUC = 0.6091  # v4 XGBoost, safe features only


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


def score_and_report(label, y_val, score):
    default_pred = (score >= 0.5).astype(int)
    report(f"{label} (default threshold 0.5)", y_val, default_pred, score)

    thresh, _ = best_f1_threshold(y_val, score)
    tuned_pred = (score >= thresh).astype(int)
    report(f"{label} (F1-tuned threshold {thresh:.4f})", y_val, tuned_pred)

    return {
        "label": label,
        "pr_auc": average_precision_score(y_val, score),
        "roc_auc": roc_auc_score(y_val, score),
        "f1_tuned": f1_score(y_val, tuned_pred),
    }


def train_mlp(X_train, y_train, input_dim, epochs=8):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(input_dim, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        print(f"  MLP epoch {epoch + 1}/{epochs}  loss={total_loss / len(X_train):.5f}",
              flush=True)
    return model


def main():
    base_features = [
        "step", "type", "amount", "isMerchantDest",
        "hour", "day", "amount_log", "highAmountFlag",
    ]

    raw = load_data()
    df = raw[base_features + ["isFraud", "nameOrig", "nameDest"]].copy()
    del raw
    gc.collect()

    y = df["isFraud"]
    idx_train, idx_val = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )
    X_train = df.loc[idx_train, base_features].copy()
    X_val = df.loc[idx_val, base_features].copy()
    train_orig = df.loc[idx_train, "nameOrig"]
    train_dest, val_dest = df.loc[idx_train, "nameDest"], df.loc[idx_val, "nameDest"]
    val_orig = df.loc[idx_val, "nameOrig"]
    y_train, y_val = y.loc[idx_train], y.loc[idx_val]
    del df
    gc.collect()

    # --- destTxnCount (carried over from v3/v4) ---
    dest_counts = train_dest.value_counts()
    X_train["destTxnCount"] = train_dest.map(dest_counts).to_numpy()
    X_val["destTxnCount"] = val_dest.map(dest_counts).fillna(0).to_numpy()

    # --- graph-inspired role-overlap features (train-only sets, see docstring) ---
    train_orig_set = set(train_orig)
    train_dest_set = set(train_dest)
    X_train["destPlaysOrigRole"] = train_dest.isin(train_orig_set).astype(int).to_numpy()
    X_train["origPlaysDestRole"] = train_orig.isin(train_dest_set).astype(int).to_numpy()
    X_val["destPlaysOrigRole"] = val_dest.isin(train_orig_set).astype(int).to_numpy()
    X_val["origPlaysDestRole"] = val_orig.isin(train_dest_set).astype(int).to_numpy()

    print("Role-overlap feature rates by class (train):")
    print(pd.DataFrame({
        "destPlaysOrigRole": X_train["destPlaysOrigRole"],
        "origPlaysDestRole": X_train["origPlaysDestRole"],
        "isFraud": y_train.to_numpy(),
    }).groupby("isFraud").mean())

    del train_orig, train_dest, val_dest, val_orig, train_orig_set, train_dest_set
    gc.collect()

    safe_features = base_features + ["destTxnCount"]
    graph_features = safe_features + ["destPlaysOrigRole", "origPlaysDestRole"]
    numeric_cols = [c for c in graph_features if c != "type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("type_ohe", OneHotEncoder(handle_unknown="ignore"), ["type"]),
            ("scale", StandardScaler(), numeric_cols),
        ]
    )
    X_train_enc = preprocessor.fit_transform(X_train[graph_features]).astype(np.float32)
    X_val_enc = preprocessor.transform(X_val[graph_features]).astype(np.float32)

    results = []

    # --- Part A: MLP on v3 safe features (scaled subset of the encoding above) ---
    print("\nTraining MLP...", flush=True)
    mlp = train_mlp(X_train_enc, y_train, input_dim=X_train_enc.shape[1])
    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.tensor(X_val_enc, dtype=torch.float32))
        mlp_score = torch.sigmoid(logits).numpy().ravel()
    results.append(score_and_report("MLP (safe + graph features)", y_val, mlp_score))

    # --- Part B: XGBoost with graph features added (imported lazily) ---
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline

    def make_pre(cols):
        return ColumnTransformer(
            transformers=[("type_ohe", OneHotEncoder(handle_unknown="ignore"), ["type"])],
            remainder="passthrough",
        )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg / pos, tree_method="hist",
        eval_metric="aucpr", random_state=42, n_jobs=4,
    )

    xgb_graph = Pipeline([
        ("pre", make_pre(graph_features)),
        ("clf", XGBClassifier(**xgb_params)),
    ])
    xgb_graph.fit(X_train[graph_features], y_train)
    xgb_graph_score = xgb_graph.predict_proba(X_val[graph_features])[:, 1]
    results.append(score_and_report("XGBoost + graph features", y_val, xgb_graph_score))

    print("\n=== Summary ===")
    print(f"{'Model':38s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'F1@tuned':>9s}")
    for r in results:
        print(f"{r['label']:38s} {r['pr_auc']:8.4f} {r['roc_auc']:8.4f} {r['f1_tuned']:9.4f}")
    print(f"\n(Reference — v4 XGBoost, safe features only: "
          f"PR-AUC={XGB_V4_REFERENCE_PR_AUC:.4f})")


if __name__ == "__main__":
    main()
