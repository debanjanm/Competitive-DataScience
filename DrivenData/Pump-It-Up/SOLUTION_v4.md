# Solution v4 — Pump it Up: Data Mining the Water Table

Improves on [SOLUTION_v3.md](SOLUTION_v3.md) / `nbs_v3.py`. Same problem, see [PROBLEM.md](PROBLEM.md).

## What changed vs v3, and why

| v3 | v4 | Why it should help |
|---|---|---|
| Frequency-encoded categoricals | Native `category` dtype, XGBoost `enable_categorical=True` | Frequency encoding collapses a category to a single number (its rate) — two different categories with the same rarity become indistinguishable. Native categorical splits keep the actual identity. |
| Global median imputation for lat/long/gps_height/population/construction_year | Per-region median (`geo_impute_numeric`), global median only as fallback | Tanzania's regions differ hugely in altitude, population density, and well-drilling era — a single national median flattens real signal. |
| Missing values imputed and forgotten | `*_missing` boolean flags added before imputing (`add_missingness_flags`) | Whether `scheme_name`/`permit`/`public_meeting` was recorded at all is itself correlated with how well-managed (and likely functional) a pump is. |
| Top-15 features by mutual information, rest discarded | All features kept; MI still logged for insight only | Tree ensembles handle irrelevant/weak features fine — the top-15 cutoff was throwing away signal for no accuracy benefit. |
| No class-imbalance handling | `compute_sample_weight('balanced', ...)` passed to `fit` | `functional needs repair` is ~7% of the data — unweighted training under-learns it. |
| Single 70/30 holdout split, one model | 5-fold `StratifiedKFold` CV, mean ± std accuracy reported, submission = average of all 5 fold-models' predicted probabilities | One split gives a noisy accuracy estimate and throws away 30% of training data from the final model. CV gives a trustworthy estimate; averaging fold predictions (bagging) is more robust than any single fold's model. |
| `StandardScaler` on continuous features | Removed | Tree models split on raw thresholds — scaling changes nothing for XGBoost, it was dead weight. |

## Pipeline steps

1. **Load data** — same guard checks as v3 (mislabeled/swapped raw file detection).
2. **Drop unneeded columns** — same list as v3 (`recorded_by`, redundant coarse/fine duplicates, etc).
3. **Fix placeholder zeros** — same as v3: `0` → `NaN` for lat/long/construction_year/population/gps_height.
4. **Missingness flags** — new. Any column with nulls in train gets a `<col>_missing` boolean column, added to both train and test before imputation.
5. **Geo-aware imputation** — new. Numeric columns filled by their train-set median *within the same region*, falling back to the train-set global median when a region has no data (or is unseen in test).
6. **Feature engineering** — `pump_age`, `record_year`, `record_month`, same as v3, now computed from geo-imputed (cleaner) `construction_year`.
7. **Categorical imputation** — most-frequent, same as v3.
8. **Numeric safety net** — plain median imputer catches any leftover NaN (should be near-zero after step 5); asserts zero nulls remain.
9. **Cast to `category` dtype** — categorical columns become pandas categoricals with train-derived category sets; unseen test categories become `NaN`, which XGBoost's native categorical handling treats as missing (same as any other missing value).
10. **Mutual information logging** — informational ranking only, doesn't filter features.
11. **5-fold CV + ensemble**:
    - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
    - Each fold: `XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', enable_categorical=True)`, trained with balanced sample weights.
    - Log per-fold accuracy, final fold's confusion matrix + classification report, and CV mean ± std.
    - Predicted probabilities on the real test set averaged across all 5 fold-models.
12. **Submission** — `argmax` of averaged probabilities, inverse-transformed to label strings, written to `submission_v4.csv` (`id, status_group` only).

## Bug fixed after first run

- **`public_meeting`/`permit` crashed XGBoost's native categorical encoder.** Both columns hold Python `True`/`False`, not strings. Casting them to `category` dtype and handing them to `enable_categorical=True` made XGBoost try to arrow-encode boolean category values as strings, raising `TypeError: object of type 'numpy.bool' has no len()`. Fixed by `cast_boolean_columns`: map `{True: 1.0, False: 0.0}` (NaN stays NaN) right after the zero-placeholder fix, so these two columns flow through as plain numeric features instead of categoricals.

## Caveats / things not done here

- **No hyperparameter search.** `n_estimators=400, max_depth=8, learning_rate=0.05` are reasonable defaults, not tuned. A `RandomizedSearchCV` or Optuna pass over `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight` would likely help further but costs real runtime — worth doing once this version's CV number is confirmed better than v3's.
- **High-cardinality categoricals** (`scheme_name`, `installer`, `funder`, `ward`, `subvillage` — thousands of unique values) use XGBoost's default categorical split threshold (`max_cat_threshold`). Worth revisiting if training is slow or these columns dominate feature importance oddly.
- **No geo-clustering feature** (e.g. KMeans on lat/long to capture spatial pump density) — a plausible next feature, not added yet.
- **No stacking/blending** with a second model family (e.g. LightGBM/CatBoost) — the 5-fold ensemble here is same-model bagging, not model diversity.

## How to run

```bash
python nbs_v4.py
```

Outputs `submission_v4.csv`. Logs (INFO level) print shape/column counts, missingness-flag columns, geo-imputation coverage, MI ranking, and per-fold + overall CV metrics.

## How to tell if it actually helped

Compare CV mean accuracy (v4, printed as `CV accuracy: X +/- Y`) against v3's single holdout accuracy — CV mean is the fairer number since it's averaged over 5 different splits instead of one lucky/unlucky 30%. Submit both `submission_v3.csv` and `submission_v4.csv` to DrivenData's leaderboard to confirm real generalization, not just local CV.
