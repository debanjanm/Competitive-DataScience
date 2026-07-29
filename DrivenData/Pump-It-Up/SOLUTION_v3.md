# Solution v3 — Pump it Up: Data Mining the Water Table

Plain Python pipeline (`nbs_v3.py`), no notebook/marimo — see [PROBLEM.md](PROBLEM.md) for problem definition.

## Pipeline steps

1. **Load data** (`load_data`)
   - Read `Training set values.csv`, `Training set labels.csv`, `Test set values.csv` from `data/raw/`.
   - Merge train values + labels on `id`.
   - Sanity checks: labels file must be exactly `[id, status_group]`; train/test feature files must have matching column counts — catches mislabeled/swapped raw files early with a clear error instead of a downstream `KeyError`.

2. **Drop unneeded columns** (`drop_unneeded_columns`)
   - `recorded_by` — constant across all rows, zero signal.
   - `wpt_name`, `num_private` — high-cardinality / near-empty.
   - Coarse-to-fine duplicate groups, keep one granularity each:
     - `extraction_type`, `extraction_type_group` → keep `extraction_type_class`
     - `quality_group` → keep `water_quality`
     - `quantity_group` → keep `quantity`
     - `source_type`, `source_class` → keep `source`
     - `waterpoint_type_group` → keep `waterpoint_type`
     - `payment_type` → keep `payment`
   - `id` dropped from features (kept separately for submission).

3. **Fix placeholder zeros** (`fix_placeholder_zeros`)
   - `longitude`, `latitude`, `construction_year`, `population`, `gps_height` use `0` as a missing-value placeholder, not a real value.
   - Replace `0` → `NaN` so imputation treats them as actually missing.

4. **Feature engineering** (`engineer_features` / `add_pump_age`)
   - `pump_age` = `date_recorded.year - construction_year`
   - `record_year`, `record_month` from `date_recorded`
   - Drop raw `date_recorded` after deriving.

5. **Missing value imputation** (`impute_missing_values`)
   - Continuous columns → median (`SimpleImputer`).
   - Categorical columns → most frequent.
   - Fit on train, transform train + test with same imputer.
   - Asserts zero remaining nulls before continuing.

6. **Encoding + scaling** (`encode_and_scale`)
   - Continuous columns → `StandardScaler`, fit on train only.
   - Categorical columns → `FrequencyEncoder` (custom): maps each category to its train-set frequency; unseen test categories fall back to `0`.

7. **Feature selection** (`select_features`)
   - Target encoded via `LabelEncoder` (`status_group` → int).
   - Rank all features by `mutual_info_classif` against target.
   - Keep top 15 (`TOP_N_FEATURES`).

8. **Modeling** (`train_model`)
   - Stratified 70/30 train/holdout split, `random_state=42`.
   - `XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1)`.

9. **Evaluation** (`evaluate`)
   - Predict on holdout split.
   - Log accuracy, confusion matrix, classification report (per-class precision/recall/F1) — checked *before* touching the real test set.

10. **Submission** (`make_submission`)
    - Refit `clf` on full training data (train + holdout combined) before predicting real test set.
    - Predict, inverse-transform labels back to string (`functional` / `functional needs repair` / `non functional`).
    - Write `id, status_group` only, to `submission_v3.csv`.

## Bugs fixed vs earlier notebook versions

- **Imputation never ran** (v1 notebook: delete-columns called twice instead of impute).
- **Label mapping swapped** (`LabelEncoder` sorts alphabetically — manual `{0:..,1:..,2:..}` dict had 1/2 backwards). Fixed by using `le.inverse_transform`.
- **Submission had extra `target` column** — now only `id, status_group`.
- **Redundant grouped columns fed into model unfiltered** — deduped, one granularity per group.
- **`0` silently mean-imputed as a real value** for lat/long/construction_year/population/gps_height — now converted to `NaN` first.
- **`date_recorded` dropped/ignored** — now engineered into `pump_age`/`record_year`/`record_month`.
- **Model was KNN despite XGBoost/LightGBM scoring higher** in `reports/models.csv` — switched to `XGBClassifier`.
- **No holdout evaluation printed** before submitting — added accuracy/confusion matrix/classification report.
- **Model was fit only on 70% split before predicting real test set** — now refits on full data first.
- **Raw data files were mislabeled/swapped on disk** (`Test set values.csv` actually contained training labels, and vice versa) — files renamed to correct pairing; load-time guard now catches this class of error immediately with a clear message.

## How to run

```bash
python nbs_v3.py
```

Outputs `submission_v3.csv` in the working directory. Logs (INFO level) print shape/column counts, missing-value rates, MI feature ranking, and eval metrics at each stage.
