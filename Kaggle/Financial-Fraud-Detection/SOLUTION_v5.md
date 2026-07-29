# Solution v5 — Unsupervised Anomaly Detection

Covers [PLAN.md](PLAN.md) Stage 7. Three methods, all trained on
**legit-only** transactions — fraud labels never seen during training,
only used for evaluation. A genuinely different framing from v1-v4's
supervised classifiers: "model what normal looks like, flag deviations"
instead of "learn to separate the two classes directly."

## Summary

**All three unsupervised methods underperform v4's supervised XGBoost
(PR-AUC 0.609) by a wide margin on a fair comparison.** Isolation Forest
gets some signal (PR-AUC 0.049). The autoencoder essentially fails
(PR-AUC 0.0015, ROC-AUC 0.42 — worse than random). One-Class SVM's raw
number looks strong (PR-AUC 0.73) but that's an artifact of its
necessarily-small, fraud-heavy evaluation subsample — **not comparable**
to the other rows, explained below.

## Method / Features

Same [v3](SOLUTION_v3.md) safe feature set (`step`, `type`, `amount`,
`isMerchantDest`, `hour`, `day`, `amount_log`, `highAmountFlag`,
`destTxnCount`). One-hot encode `type`, `StandardScaler` the rest — fit
on **legit-only training rows** (5,083,526 of them), consistent with the
"learn normal" framing. 13-dim feature space after encoding.

1. **Isolation Forest** — `n_estimators=200`, scores full val set
   (1,272,524 rows).
2. **One-Class SVM** — `kernel="rbf"`, `nu=0.01`. Training cost is
   O(n²)-O(n³), infeasible at 5M+ rows, so trained on a 5,000-row legit
   subsample and evaluated on a 6,643-row subsample (5,000 legit + all
   1,643 val fraud) — **not the full val set**.
3. **Autoencoder** (PyTorch) — `13 → 16 → 8 → 16 → 13`, ReLU, MSE loss,
   Adam, 8 epochs, batch size 4096. Reconstruction error = anomaly score
   on the full val set.

## Results

| Model | Eval set | PR-AUC | ROC-AUC | F1@tuned |
|---|---|---|---|---|
| Isolation Forest | full val (1,272,524 rows) | 0.0488 | 0.8992 | 0.1166 |
| One-Class SVM | **6,643-row subsample only** | 0.7316* | 0.8464 | 0.6500 |
| Autoencoder | full val (1,272,524 rows) | 0.0015 | 0.4220 | 0.0139 |
| *(reference)* v4 XGBoost, supervised | full val | **0.6091** | 0.9815 | 0.6189 |

*\*One-Class SVM's PR-AUC is not comparable to the other rows — see below.*

## Design Notes / Why These Results

- **One-Class SVM's 0.73 PR-AUC is inflated by its own evaluation set,
  not a real win.** PR-AUC's baseline (a random classifier's score)
  equals the positive-class prevalence. The full val set is 0.13% fraud;
  the OCSVM subsample is 24.7% fraud (1,643 of 6,643) by construction —
  nearly 200x denser. A PR-AUC of 0.73 at 24.7% prevalence reflects a much
  easier discrimination task than the same score would at 0.13%. This is
  exactly the kind of comparison trap PLAN.md's evaluation stage warns
  about — same lesson as v3's leakage demo, different mechanism: an
  unfair evaluation set instead of an unfair feature.
- **Isolation Forest gets real but weak signal** (PR-AUC 0.049 — better
  than v1's naive baselines, far below any supervised model since v2). It
  isolates *some* fraud via short partition paths, but fraud in this
  dataset isn't a strong outlier along these particular feature axes.
- **The autoencoder essentially failed** — ROC-AUC 0.42 is *worse than
  random guessing* (0.5). Reconstruction error converged to a very low,
  flat MSE (0.10 → 0.0001 within 2 epochs) and doesn't discriminate fraud
  from legit at all. Plausible explanation: fraud transactions here
  (`TRANSFER`/`CASH_OUT`, specific amount ranges) aren't structurally
  unusual along `step`/`type`/`amount`/etc. — legit transactions span
  *more* diversity across all 5 types, so the autoencoder may reconstruct
  the comparatively narrow, common fraud pattern *better* than the full
  diversity of legit traffic, inverting the expected error direction.
- **The core lesson of v5**: anomaly detection assumes anomalies are
  statistically distinct points in feature space. This fraud pattern is
  a specific *behavioral sequence* (account takeover → drain → cash out)
  that isn't necessarily an outlier point-in-time — it's outlier only in
  context (e.g., a large transfer from an account with no prior large
  transfers). None of v5's features capture that sequential/contextual
  structure; v3's `destTxnCount` was the only step in that direction.
  Supervised learning (v2-v4) exploits the label directly to find that
  needle; unsupervised methods without behavioral/sequential features
  are flying blind on exactly the signal that matters most.

## Known Limitations / Next Steps

- Isolation Forest and Autoencoder trained on the full 13-dim static
  feature set — no sequential/behavioral features (e.g., time since
  account's last transaction, deviation from that account's own typical
  amount) that might make fraud a genuine outlier for these methods.
  Worth revisiting if pursuing unsupervised methods further.
- One-Class SVM's subsample size (5,000 train / 6,643 eval) was chosen
  purely for tractability, not tuned — a proper comparison would need
  either a much larger legit sample (with a faster SVM approximation,
  e.g. `SGDOneClassSVM`) or evaluation against the full val set's true
  prevalence to be meaningful alongside the other rows.
- Autoencoder architecture/training was not tuned (fixed hidden sizes,
  8 epochs) — given how badly it failed, more capacity or a different
  objective (e.g. weighting reconstruction toward `amount`/`destTxnCount`)
  might help, but the more fundamental issue (missing behavioral
  features) would likely dominate any architecture tweak.
- Candidate v6 direction: PLAN.md Stage 8 — deep learning / graph-based
  features. `nameOrig`/`nameDest` form a transaction graph; graph
  features (degree, connected-component structure) are a more direct way
  to capture the "unusual for this account" signal that v5's static
  features missed.
