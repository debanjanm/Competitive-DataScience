# Solution v3 — Feature Engineering (+ Leakage Demonstration)

Covers [PLAN.md](PLAN.md) Stage 4. Same model class as v1/v2 (Logistic
Regression, `class_weight="balanced"`, default 0.5 threshold) throughout —
only the feature set changes, to isolate feature effect.

## Summary

Two experiments against the v1 feature-set reference:

- **Part A (safe features)**: temporal (`hour`, `day`), `amount_log`,
  `highAmountFlag`, `destTxnCount` (destination transaction frequency,
  computed from the training split only). Real, usable improvement.
- **Part B (leakage demo)**: adds `errorBalanceOrig` /
  `errorBalanceDest` — the balance-consistency features PLAN.md flagged as
  leakage-risk and the dataset author explicitly said "must not be used."
  Included deliberately, clearly separated, to show what a leakage red
  flag looks like in practice — **not adopted for the model going
  forward.**

## Features

| Feature | Part | Notes |
|---|---|---|
| `step`, `type`, `amount`, `isMerchantDest` | A (carried from v1) | baseline set |
| `hour` = `step % 24` | A | time-of-day |
| `day` = `step // 24` | A | day of 30-day sim |
| `amount_log` = `log1p(amount)` | A | compress heavy right skew for linear model |
| `highAmountFlag` = `amount > 200000` | A | mirrors `isFlaggedFraud` rule as a feature, not a prediction |
| `destTxnCount` | A | # transactions to this `nameDest` — fit on **train split only**, unseen dest in val → 0 (no val-info leak) |
| `errorBalanceOrig` = `oldbalanceOrg - amount - newbalanceOrig` | B — leakage demo | uses banned balance columns |
| `errorBalanceDest` = `oldbalanceDest + amount - newbalanceDest` | B — leakage demo | uses banned balance columns |

`nameOrig` transaction count was considered and dropped: 6,353,307 unique
values across 6,362,620 rows — almost every origin account transacts
exactly once, so a per-origin count feature carries near-zero signal.
`nameDest` is different — up to 113 repeats per account — so
`destTxnCount` was kept.

## Results

| Feature set | Precision (fraud) | Recall (fraud) | F1 (fraud) | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| v1 reference | 0.0052 | 0.8819 | 0.0104 | 0.0207 | 0.9194 |
| **v3 safe features** | 0.0057 | 0.8740 | 0.0114 | **0.0359** | 0.9345 |
| v3 + leakage demo | 0.0288 | 0.9464 | 0.0559 | **0.5695** | 0.9920 |

## Design Notes / Why These Results

- **Safe features nearly double PR-AUC (0.0207 → 0.0359, +74% relative)**
  with the same model and same imbalance handling as v1/v2 — this is the
  gain Stage 3 (v2) couldn't deliver. Confirms the plan's ordering:
  features move the ceiling, imbalance technique just moves you along it.
- **`destTxnCount` and `amount_log` are doing the real work here** — a
  linear model benefits from the log transform smoothing `amount`'s skew,
  and destination frequency gives the model a proxy for "is this a
  first-time/one-off recipient," which correlates with the fraud
  pattern (drain-and-cash-out to a fresh account).
- **The leakage demo is the real lesson of this version.** PR-AUC jumps
  from 0.036 to **0.57** — a ~16x jump — from adding two features. That
  kind of implausible leap on a genuinely hard problem (fraud detection
  rarely gets easy) is *the* signature of leakage, not of good feature
  engineering. Mechanism here: PaySim appears to reset/adjust balances on
  cancelled (fraudulent) transactions in a way that makes
  `errorBalanceOrig`/`errorBalanceDest` deviate sharply from the ~0
  they'd show for a normal, uncancelled transaction — the feature is
  almost directly encoding the label. This is exactly why the dataset
  author's warning exists, and why Stage 4 of the plan said to
  investigate leakage before trusting balance-derived features.
- **Rule of thumb learned here**: a feature that produces a jump this
  large, this cheaply, on a problem this hard, should be treated as
  suspicious by default and traced back to its construction before use —
  not celebrated as a win.
- Minor: one of the three fits threw an `lbfgs` `ConvergenceWarning`
  (likely Part B, due to `errorBalance*` having a much larger/differently
  scaled range than the other features). Didn't invalidate the result for
  teaching purposes here, but flags that feature scaling
  (`StandardScaler`) is overdue — deferred to the point where a model
  actually needs it (Stage 5 tree ensembles are scale-invariant and don't).

## Known Limitations / Next Steps

- No feature scaling yet — fine for tree ensembles (Stage 5, scale-
  invariant), but would matter if Logistic Regression continues past v3.
- `destTxnCount` computed as a flat historical count, not a rolling/
  time-aware window — a production system would need point-in-time
  correctness (don't count future transactions relative to the row being
  scored). Acceptable simplification for this offline learning exercise.
- Threshold not re-tuned for v3's new PR curve (still default 0.5) —
  combining v2's threshold-tuning technique with v3's features is a
  natural quick win, left for whichever version does the full Stage 5/6
  pass.
- Candidate v4 direction: Stage 5 — tree ensembles (Random Forest /
  XGBoost / LightGBM) on the **safe** v3 feature set. Expect a bigger
  jump than v3's linear-model gain, since trees capture the
  `amount`/`type`/`destTxnCount` interactions a linear boundary can't.
