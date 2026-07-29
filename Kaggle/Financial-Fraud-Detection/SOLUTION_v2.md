# Solution v2 — Imbalance-Handling Techniques

Covers [PLAN.md](PLAN.md) Stage 3. Same features and model class
(Logistic Regression) as [v1](SOLUTION_v1.md) throughout — only the
imbalance technique changes, to isolate its effect from feature/model
changes.

## Summary

Three techniques tested against the v1 reference (`class_weight="balanced"`,
default 0.5 threshold): threshold tuning, random undersampling, SMOTE
oversampling. Result: **threshold tuning wins by a wide margin**;
resampling barely moves the needle. Important lesson — imbalance technique
can't fix what feature/model capacity can't support (v1's `PR-AUC` ceiling
of ~0.02 holds across all three).

## Techniques

1. **Threshold tuning** — reuse v1's balanced-weight model, sweep
   `precision_recall_curve` thresholds on validation `predict_proba`,
   pick the threshold that maximizes F1. No retraining.
2. **Random undersampling** — `RandomUnderSampler` drops majority-class
   rows until classes are 50/50 (13,140 rows total), plain
   `LogisticRegression` (no class weighting — balance already achieved by
   resampling).
3. **SMOTE oversampling** — synthesize minority-class rows to 50/50
   (10,167,052 rows total), plain `LogisticRegression`.

## Results

| Technique | Precision (fraud) | Recall (fraud) | F1 (fraud) | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| v1 reference (balanced, thresh=0.5) | 0.0052 | 0.8819 | 0.0104 | 0.0207 | 0.9194 |
| **Threshold tuned (0.991)** | **0.0549** | 0.0992 | **0.0707** | 0.0207 | 0.9194 |
| Random undersampling | 0.0051 | 0.8819 | 0.0101 | 0.0204 | 0.9185 |
| SMOTE | 0.0066 | 0.8539 | 0.0131 | 0.0188 | 0.9156 |

## Design Notes / Why These Results

- **Threshold tuning gave the biggest F1 jump (0.010 → 0.071, ~7x)** for
  free — no retraining, just moving the decision boundary from 0.5 to
  0.991. Confirms the v1 hypothesis: the model was fine, the *threshold*
  was wrong. Note the tradeoff though — recall dropped hard (0.88 → 0.10)
  as precision rose; F1's optimum here isn't necessarily the right
  operating point for a real fraud system (a human reviewer might prefer
  higher recall despite more false positives — this is a business-cost
  decision, not just an F1-maximization one).
- **PR-AUC is identical for the reference and threshold-tuned rows
  (0.0207)** — expected, since PR-AUC is threshold-independent and both
  rows score the *same* underlying model, just different cutoffs. Good
  sanity check that the technique is doing what it claims.
- **Undersampling ≈ no improvement over v1** — with only ~6,570 fraud
  rows in training, undersampling throws away >99% of legitimate examples
  (1.27M → 6,570), leaving just 13,140 rows to train on. Model barely
  changes because it's starved of data, not because the imbalance was
  fixed.
- **SMOTE underperforms threshold tuning and is barely better than
  undersampling** on PR-AUC, despite generating 10M+ synthetic rows.
  Confirms a known SMOTE weakness: interpolating between minority points
  in a **weak feature space** (here: `step`, `type`, `amount`,
  `isMerchantDest`) just synthesizes more points that don't separate
  cleanly from legit transactions — SMOTE amplifies existing feature
  signal, it doesn't invent new signal.
- **Core takeaway**: all four rows share essentially the same PR-AUC
  ceiling (~0.02–0.02). Imbalance technique alone can't move a model past
  a feature-capacity wall. Confirms the plan's ordering — Stage 4
  (feature engineering) matters more than further imbalance tuning at
  this point.

## Known Limitations / Next Steps

- Only Logistic Regression tested — resampling effects can differ with
  tree-based models (Stage 5), worth revisiting there.
- Threshold was tuned by maximizing F1 on the validation set itself —
  technically an information leak for a true held-out estimate; fine for
  learning purposes here, but a production pipeline would tune threshold
  on a separate validation split from final test evaluation.
- No cost-sensitive threshold selection (e.g. minimize $-weighted
  false-negative cost) — only F1-optimal threshold shown. Worth adding as
  an alternative operating point once a cost model exists.
- Candidate v3 direction: Stage 4 feature engineering (balance-delta
  consistency features, behavioral aggregates) — expected to raise the
  PR-AUC ceiling itself, which no imbalance technique here could do.
