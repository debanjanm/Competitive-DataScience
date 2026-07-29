# Solution v2 — Feature Engineering + Tuned TF-IDF + Model Selection

## Summary

Builds on [v1](SOLUTION_v1.md)'s TF-IDF + Logistic Regression baseline by
adding `keyword`/`location` signal, text-stat features, a small hyperparameter
search over the TF-IDF/LR config, and a comparison against LinearSVC,
ComplementNB, and a hard-voting ensemble of all three — picking whichever
scores best on the held-out validation split.

## Pipeline

1. **Load data** — same `train.csv`/`test.csv` as v1.
2. **Feature engineering** (`build_features`)
   - `text_clean` — same cleaning as v1 (URL/mention/hashtag-symbol strip,
     contraction/acronym expansion, non-alpha strip).
   - `keyword_clean` — `keyword` column lowercased, `%20` → space, NaN → `""`.
   - `text_combined` — `keyword_clean` prepended to `text_clean`, so TF-IDF
     vocabulary picks up the keyword directly instead of discarding it.
   - `has_location` — binary indicator, `location` present or not. Raw
     location text was **not** vectorized: ~33% missing, and the non-missing
     values are largely free-text noise (`"Est. September 2012 - Bristol"`,
     `"milky way"`, `"World Wide!!"`) with too little repetition per value to
     generalize — a presence flag captures the "is this a located account"
     signal without the overfit risk of thousands of one-off categories.
   - `word_count`, `char_count` — length of cleaned text, cheap numeric
     signal (real disaster reports tend to be denser/more detail-packed).
3. **Split** — 80/20 stratified on `target`, `random_state=42` (same seed as
   v1, for comparability).
4. **Hyperparameter search** — grid over:
   - `TfidfVectorizer(max_features ∈ {10000, 20000, 30000}, ngram_range ∈ {(1,1), (1,2)}, min_df=2)`
   - `LogisticRegression(C ∈ {0.5, 1.0, 3.0})`
   
   18 combos total, each scored by validation F1. Numeric features
   (`has_location`, `word_count`, `char_count`) horizontally stacked onto the
   TF-IDF matrix via `scipy.sparse.hstack` before fitting.
5. **Model comparison** — using the winning TF-IDF config, train and score:
   - `LogisticRegression` (best `C` from search)
   - `LinearSVC` (`C=1.0`)
   - `ComplementNB` — suited to TF-IDF's non-negative sparse counts
   - `VotingClassifier` (hard vote across all three)
   
   Best of the four on validation F1 is selected as the final model.
6. **Refit + predict** — refit chosen vectorizer config + model on full
   training data, predict on test set, write `submission_v2.csv`.

## Result

| Model | Validation F1 |
|---|---|
| Logistic Regression (tuned) | **0.7793** |
| Linear SVC | 0.7595 |
| Complement NB | 0.7673 |
| Hard-vote ensemble | 0.7752 |
| **v1 baseline (untuned LR)** | 0.7686 |

Best config: `max_features=10000, ngram_range=(1,1), C=3.0`. Logistic
Regression alone won — selected as final model.

## Design Notes / Why These Choices

- **Keyword folded into text, not one-hot encoded**: keyword vocabulary
  overlaps heavily with tweet text vocabulary already; folding it in lets
  TF-IDF weight it naturally alongside other tokens rather than adding a
  separate ~200-dim sparse block for marginal gain.
- **Location reduced to a binary flag**: tested the cost/benefit of raw
  location text — too sparse and unstructured (freeform, multilingual,
  emoji-laden) to vectorize without overfitting on a ~7600-row train set.
  A presence flag is the safe subset of that signal.
- **Grid search kept small and CPU-cheap**: 18 combos × dataset size (~8.5k
  rows) runs in seconds; no need for `GridSearchCV`/cross-validation
  machinery at this scale — a plain loop is simpler and just as correct
  here.
- **Unigrams beat unigrams+bigrams this time**: contrary to v1's use of
  bigrams, the search picked `ngram_range=(1,1)` — the added feature
  engineering (keyword folding, numeric stats) apparently supplies enough
  of what bigrams were compensating for in v1, and bigrams add noise/sparsity
  at this dataset size.
- **Ensemble underperformed the single tuned model**: hard voting across LR/
  SVM/NB (0.7752) scored below tuned LR alone (0.7793) — NB and SVM likely
  pull the vote toward inputs where they're systematically weaker, and hard
  voting has no way to weight by per-model confidence. Kept in the script
  for comparison, but not selected.

## Known Limitations / Next Steps

- Grid search is still coarse (3×2×3) — not exhaustive over C or
  regularization type (L1/L2), and doesn't tune SVM/NB hyperparameters at
  all.
- `location` binary flag is a blunt instrument — geocoding or
  country/region extraction could recover more signal.
- Still no semantic understanding — TF-IDF treats "blaze" and "fire" as
  unrelated tokens. This remains the strongest lever for a v3.
- Candidate v3 direction unchanged from v1's note: fine-tune
  DistilBERT/BERT — deferred here in favor of exhausting classic-ML
  feature engineering first, per current project sequencing.
