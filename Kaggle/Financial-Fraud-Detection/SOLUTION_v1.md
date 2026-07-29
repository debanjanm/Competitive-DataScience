# Solution v1 — Naive Baselines + Logistic Regression

Covers [PLAN.md](PLAN.md) Stage 0 (EDA), Stage 1 (naive baselines), Stage 2
(first classic ML baseline).

## Summary

First working pipeline. Establishes the floor with two naive baselines,
then a Logistic Regression as the first real model. No resampling, no
threshold tuning, no engineered features yet — those come in later
versions per the plan. Point of this version is to see *why* accuracy is
useless here and get PR-AUC/ROC-AUC on the board as the real scoreboard.

## Data / EDA

- 6,362,620 transactions, fraud rate **0.1291%** — extreme imbalance.
- Confirmed: fraud occurs **only** in `TRANSFER` (4,097 of 532,909) and
  `CASH_OUT` (4,116 of 2,237,500). Zero fraud in `CASH_IN`, `DEBIT`,
  `PAYMENT`. Matches PaySim's documented fraud scenario.
- `isFlaggedFraud` almost never fires (amount > 200,000 rule) — 0
  predictions in the validation split. As a detector it's dead weight.

## Features (v1)

`step`, `type` (one-hot), `amount`, `isMerchantDest` (derived from
`nameDest` prefix `M`).

**Deliberately excluded**: `oldbalanceOrg`, `newbalanceOrig`,
`oldbalanceDest`, `newbalanceDest`. The dataset author states these "must
not be used" for fraud detection — fraud transactions are cancelled, so
these columns reflect post-cancellation state, not what happened at
transaction time. See [PROBLEM.md](PROBLEM.md). `nameOrig`/`nameDest`
themselves dropped too (high-cardinality IDs, not usable as raw
categoricals).

## Pipeline

1. Load full CSV, derive `isMerchantDest`.
2. Stratified 80/20 train/val split on `isFraud`, `random_state=42`.
3. **Baseline 1**: majority class (always predict legit).
4. **Baseline 2**: `isFlaggedFraud` used directly as the prediction.
5. **Model**: `OneHotEncoder(type)` → `LogisticRegression(class_weight="balanced")`.
6. Report precision/recall/F1 (fraud class), confusion matrix, PR-AUC,
   ROC-AUC for each.

## Results

| Approach | Precision (fraud) | Recall (fraud) | F1 (fraud) | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Majority class | 0.0000 | 0.0000 | 0.0000 | — | — |
| `isFlaggedFraud` as-is | 0.0000 | 0.0000 | 0.0000 | — | — |
| Logistic Regression (balanced) | 0.0052 | **0.8819** | 0.0104 | **0.0207** | 0.9194 |

Confusion matrix, Logistic Regression:

```
[[TN 996026   FP 274855]
 [FN    194   TP   1449]]
```

## Design Notes / Why These Choices

- **Both naive baselines score 0 on every metric except accuracy**
  (99.87%) — textbook demonstration of why accuracy is meaningless under
  0.13% base rate. Anchors the whole plan's emphasis on PR-AUC/F1 over
  accuracy.
- **`class_weight="balanced"` was necessary**: without it, Logistic
  Regression converges to essentially the majority-class baseline (near-
  zero recall). Balancing trades precision for recall — visible in the
  274,855 false positives.
- **ROC-AUC (0.92) vs PR-AUC (0.02) tell very different stories**. ROC-AUC
  looks strong because true negatives dominate and inflate it under
  extreme imbalance; PR-AUC (which ignores TN) shows the model is still
  far from usable at the default 0.5 threshold. This gap is *the* lesson
  of v1 — confirms Stage 6 of the plan before we even get there.
- **High recall (0.88), terrible precision (0.005)**: model finds most
  fraud but drowns it in false alarms — 275k flagged transactions to catch
  1,449 fraud cases isn't operationally usable. Motivates Stage 3
  (threshold tuning, cost-sensitive learning) directly.

## Known Limitations / Next Steps

- No resampling (SMOTE/undersampling) tried yet — Stage 3.
- No decision-threshold tuning — currently using sklearn's default 0.5,
  which is arbitrary here; a precision/recall-cost-aware threshold should
  cut false positives substantially without losing much recall.
- No engineered features (balance-consistency, behavioral aggregates,
  temporal) — Stage 4.
- Only linear model tried — tree ensembles (Stage 5) expected to handle
  the `amount`/`type` interaction structure much better than a linear
  decision boundary.
- Candidate v2 direction: keep Logistic Regression as reference, add
  Stage 3 imbalance techniques (resampling + threshold tuning) on the same
  feature set to isolate their effect before touching features or model
  class.
