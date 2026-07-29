# Learning Plan: Fraud Detection on PaySim

Ordered ladder from trivial baseline to advanced techniques. Goal is
breadth of technique exposure, not just leaderboard score — each stage
should teach something the previous one didn't. Work top to bottom; don't
skip a stage just because a later one scores higher, the point is to know
*why* it scores higher.

## Stage 0 — Setup & EDA

- Load data, check dtypes, memory footprint (6.3M rows — consider `dtype`
  downcasting or chunked/`polars` loading).
- Class balance: count/percentage of `isFraud=1` vs `0`.
- Confirm fraud only occurs in `TRANSFER`/`CASH_OUT` (per PaySim docs) —
  filter and verify.
- Distribution of `amount`, `step` (time-of-day/day patterns), `type`.
- Check `isFlaggedFraud` overlap with `isFraud` — is the rule-based flag
  even a decent baseline on its own?
- Train/test split strategy: **stratified** on `isFraud` (imbalance), and
  consider a **time-based split** on `step` (more realistic — don't let
  future transactions leak into training for past ones).

## Stage 1 — Naive / Rule-Based Baselines

Establish the floor. Any real model must beat these.

| Baseline | What it is |
|---|---|
| Majority-class predictor | always predict `isFraud=0` |
| Random predictor (stratified) | predict fraud at the base rate |
| `isFlaggedFraud` as-is | use the existing rule (amount > 200,000) directly as the prediction |
| Simple threshold rule | e.g. flag all `TRANSFER`/`CASH_OUT` above some amount percentile |

Metric on all of these: precision/recall/F1 on the fraud class, PR-AUC.
This is where you learn *why* accuracy is a useless metric here (99%+
accuracy from just predicting "not fraud").

## Stage 2 — Classic ML Baselines

Minimal feature set (raw numeric + one-hot `type`), no leakage-prone
engineering yet.

- **Logistic Regression** (with `class_weight='balanced'`) — interpretable
  coefficients, good first real model.
- **Decision Tree** (shallow, e.g. `max_depth=5`) — visualize it, see what
  splits it picks. Good intuition builder before ensembles.
- **k-NN** — optional, mostly to learn why it struggles at this scale/
  dimensionality (curse of dimensionality, slow inference).

Learn: precision-recall tradeoff, ROC vs PR curves, confusion matrix
reading, choosing a decision threshold instead of default 0.5.

## Stage 3 — Handling Class Imbalance

Fraud is a tiny minority — this stage is about techniques for that, not
new models.

- **Class weighting** (`class_weight='balanced'`, `scale_pos_weight` in
  boosted trees).
- **Resampling**: random undersampling of majority class, `SMOTE` /
  `ADASYN` oversampling of minority class (via `imbalanced-learn`).
- **Threshold tuning**: move decision threshold based on precision/recall
  tradeoff or business cost, instead of retraining.
- **Cost-sensitive learning**: assign asymmetric misclassification cost
  (false negative fraud >> false positive review).

Compare each technique against the Stage 2 baseline on the *same* model to
isolate the effect of the imbalance technique from the model choice.

## Stage 4 — Feature Engineering

⚠ Recall the leakage caveat in [PROBLEM.md](PROBLEM.md): balance columns
reflect post-cancellation state for fraud rows. Engineer carefully.

- **Balance-consistency features**: `oldbalanceOrg - amount - newbalanceOrig`
  (should be ~0 for legit; discrepancies are informative but investigate
  whether they leak the fraud outcome itself before trusting them).
- **Destination type**: merchant (`nameDest` starts with `M`) vs customer.
- **Behavioral/aggregate features**: transaction count per `nameOrig` /
  `nameDest`, rolling amount stats per account, time-since-last-transaction.
- **Temporal features**: hour-of-day / day derived from `step`.
- **Zero-balance flags**: origin or destination balance is exactly 0
  before/after — common fraud fingerprint in this dataset.

This is the stage where feature quality usually matters more than model
choice — worth spending real time here.

## Stage 5 — Tree Ensembles

- **Random Forest** — bagging, feature importance, less prone to overfit
  than a single tree.
- **Gradient boosting**: `XGBoost`, `LightGBM`, `CatBoost` — the practical
  workhorses for tabular fraud data. Learn `scale_pos_weight`, early
  stopping, learning-rate/depth tuning.
- Compare training speed and handling of categoricals across the three
  (CatBoost handles `type`/categoricals natively; XGBoost/LightGBM need
  encoding or native categorical support).

## Stage 6 — Model Evaluation, Properly

- PR-AUC as primary metric (imbalance-appropriate); ROC-AUC as secondary.
- Confusion matrix at chosen threshold, cost-weighted.
- **Calibration**: are predicted probabilities meaningful, or just
  ranking scores? (`CalibratedClassifierCV`, reliability diagrams.)
- Cross-validation strategy for imbalanced + potentially time-dependent
  data: `StratifiedKFold`, or time-series-aware CV using `step`.

## Stage 7 — Unsupervised / Anomaly Detection

Fraud is rare — treat it as an anomaly-detection problem, not just
classification. Useful when labels are scarce/unreliable in practice
(this dataset has labels, but the technique matters for real-world fraud
work where labels lag or don't exist).

- **Isolation Forest**
- **One-Class SVM** (on legit-only training data)
- **Autoencoder** — train on legit transactions, flag high reconstruction
  error as anomalous.
- Compare anomaly-detection recall/precision against the supervised models
  from Stage 5 — where does unsupervised catch fraud supervised misses (or
  vice versa)?

## Stage 8 — Advanced / Deep Learning

- **MLP / simple feed-forward net** on engineered features — mostly to
  confirm tabular deep learning rarely beats gradient boosting here, and
  understand why (inductive bias of trees fits tabular data better).
- **Graph-based approach**: `nameOrig`/`nameDest` form a transaction graph.
  Build a graph (nodes = accounts, edges = transactions), try:
  - Graph features (degree, PageRank-style centrality) fed into Stage 5
    models.
  - A proper **Graph Neural Network** (e.g. `PyTorch Geometric` GCN/
    GraphSAGE) — fraud rings often show up as connected-component/degree
    anomalies that flat tabular features miss.
- **Ensembling/stacking**: combine supervised (Stage 5), unsupervised
  (Stage 7), and graph-based (Stage 8) model outputs as meta-features into
  a final blender.

## Stage 9 — Explainability

- **SHAP values** on the best tree ensemble — global feature importance +
  per-transaction explanation ("why was this flagged?").
- **Partial dependence plots** for key features (`amount`, balance-delta).
- Important for fraud specifically: analysts need to justify a flag, not
  just get a score.

## Stage 10 — Production-Mindset Extras (optional, conceptual)

Not required for a Kaggle-style notebook, but worth knowing about since
this is a "learn everything" pass:

- **Real-time scoring constraints**: batch model vs. low-latency
  inference, feature-store lookups for aggregate features.
- **Concept drift**: fraud patterns evolve; monitor for model decay,
  retraining cadence.
- **Active learning / feedback loop**: analyst-reviewed false
  positives/negatives feeding back into retraining.

## Suggested Order of Execution

1. Stage 0 (EDA) → Stage 1 (naive baselines) → Stage 2 (classic ML) —
   get a working pipeline end to end fast.
2. Stage 3 (imbalance handling) + Stage 4 (features) — biggest score
   jumps live here, do them before reaching for fancier models.
3. Stage 5 (tree ensembles) + Stage 6 (proper evaluation) — this is
   likely your best practical model.
4. Stage 7–9 as exploration/learning once the practical model is solid —
   compare, don't just chase score.
5. Stage 10 read-only, for conceptual completeness.
