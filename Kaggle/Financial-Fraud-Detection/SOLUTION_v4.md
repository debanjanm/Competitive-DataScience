# Solution v4 — Tree Ensembles

Covers [PLAN.md](PLAN.md) Stage 5. Random Forest, XGBoost, LightGBM vs. the
v3 Logistic Regression reference — same feature set ([v3](SOLUTION_v3.md)'s
safe features), same split, only model class changes. Each model reported
at both default (0.5) and F1-tuned threshold (Stage 3 technique).

## Summary

Tree ensembles blow past the linear baseline: PR-AUC goes from **0.036**
(Logistic Regression) to **0.60–0.61** (XGBoost/LightGBM) on the exact
same features. This is the jump the plan predicted — linear models can't
capture the `amount`/`type`/`destTxnCount` interactions that trees split
on natively. **XGBoost is the best model so far** on every metric.

## Results

| Model | PR-AUC | ROC-AUC | F1@0.5 | F1@tuned |
|---|---|---|---|---|
| Logistic Regression (v3 reference) | 0.0359 | 0.9345 | 0.0114 | 0.1049 |
| Random Forest | 0.4041 | 0.9793 | 0.0584 | 0.4240 |
| **XGBoost** | **0.6091** | **0.9815** | 0.0584 | **0.6189** |
| LightGBM | 0.6022 | 0.9816 | 0.0534 | 0.6048 |

XGBoost at its tuned threshold (0.9958): 824/1643 fraud caught, only 196
false positives — a precision/recall balance no linear model in this
project came close to.

## Design Notes / Why These Results

- **PR-AUC ~17x over v1's original baseline (0.021 → 0.61 combining v3
  features + v4 model)** — this confirms the plan's ordering claim from
  v2/v3: features moved the ceiling once (linear model), model class
  moved it again (trees), and neither alone got here.
- **RF vs. XGBoost/LightGBM gap (0.40 vs. 0.61 PR-AUC)** is expected —
  gradient boosting corrects previous trees' errors sequentially, bagging
  (RF) just averages independent trees. Boosting typically wins on
  structured/tabular fraud data for exactly this reason.
- **XGBoost and LightGBM are close (0.609 vs 0.602)** — both are
  gradient-boosted trees with similar capacity on this feature set;
  neither pulled meaningfully ahead. Differences here are more about
  training speed and categorical handling than raw score.
- **F1@0.5 barely moves for the ensembles (0.058, 0.058, 0.053)** —
  default threshold is still badly miscalibrated for 0.13% base rate
  regardless of model. Threshold tuning remains essential; it isn't a
  v2-only technique, it should be paired with every model going forward.

## Engineering Note: a segfault trap worth knowing about

This version reliably crashed (SIGSEGV, no Python traceback) while in
development, and it's a genuinely useful thing to have hit once. Root
cause: on macOS, Python's multiprocessing `spawn` start method
**re-imports the whole `__main__` module in every worker process it
creates — but only when the script runs as a real file**
(`python3 solution_v4.py`), not under `python3 -c "..."`.
`RandomForestClassifier(n_jobs>1)` spawns workers via joblib's loky
backend. With `xgboost`/`lightgbm` imported at module level, every one of
those RF worker processes was *also* re-importing them and initializing
their bundled OpenMP runtimes — despite those workers only ever needing
sklearn's tree-fitting code. Leftover state from the duplicate
initialization corrupted the process enough to crash it the moment the
real XGBoost/LightGBM fits ran later in the main process.

Two-part fix, both applied in `solution_v4.py`:

1. Import `xgboost`/`lightgbm` **lazily** — inside `main()`, right before
   they're used — instead of at module level, so RF's spawned workers
   never touch them.
2. Explicitly shut down joblib's loky worker pool
   (`get_reusable_executor().shutdown(wait=True)`) right after the Random
   Forest step, rather than relying on garbage collection to do it.

General takeaway: **heavy native-library imports at module level get
pulled into every multiprocessing worker a script spawns**, even ones
that never call into that library. Worth checking for on any script that
mixes `n_jobs>1` scikit-learn estimators with other native-code ML
libraries in the same process.

## Known Limitations / Next Steps

- No hyperparameter tuning — `n_estimators`/`max_depth`/`learning_rate`
  were reasonable defaults, not searched. Worth a pass (grid/random
  search, or `Optuna`) once a model is chosen as the production candidate.
- Random Forest and both boosters used `n_jobs=4` (not `-1`) to sidestep
  the segfault above — leaves some parallelism on the table; revisit if
  runtime becomes a bottleneck once the crash-avoidance fix is confirmed
  stable across environments.
- No feature importance / SHAP inspection yet — natural fit for Stage 9,
  and would help sanity-check that `destTxnCount`/`amount_log` are really
  doing the work v3 attributed to them.
- Candidate v5 direction: PLAN.md Stage 7 — unsupervised anomaly
  detection (Isolation Forest, autoencoder) as a different angle on the
  same problem, compared against XGBoost's supervised result rather than
  chasing further supervised score gains immediately.
