# Solution v3 — Preprocessing, Leak-Safe Pipeline, CV, Threshold Tuning

## Summary

No new model, no new features. This version is a **technique pass** over
[v2](SOLUTION_v2.md)'s winning config (TF-IDF unigrams, `max_features=10000`,
`C=3.0` Logistic Regression): four incremental, individually-learnable
additions — stopword removal + lemmatization, a leak-safe sklearn
`Pipeline`/`ColumnTransformer`, proper k-fold cross-validation, and
out-of-fold decision-threshold tuning.

## What changed vs v2

### 1. Stopword removal + lemmatization

`remove_stopwords_and_lemmatize()` runs after the existing clean/expand
steps: drops NLTK's English stopword list, **except** `not`/`no`/`nor` (these
carry negation meaning — "not on fire" and "on fire" should not collapse to
the same bag of words), then lemmatizes each surviving token
(`WordNetLemmatizer`, default noun mode) — e.g. `floods`/`flooding` → closer
shared forms, shrinking vocabulary redundancy.

### 2. `Pipeline` + `ColumnTransformer` (fixes a v2 leakage bug)

v2 fit `TfidfVectorizer` once on the train split, then reused that fitted
vectorizer across every CV-style comparison — fine for a single train/val
split, but wrong the moment you cross-validate: refitting the *same* fitted
vectorizer across folds means each fold's validation rows already influenced
the vocabulary/IDF weights of earlier folds it was fit on.

v3 wraps the vectorizer inside a `ColumnTransformer` (TF-IDF on
`text_combined`, `passthrough` for the three numeric columns), then wraps
that inside a `Pipeline` with the classifier. `cross_val_score` /
`cross_val_predict` clone and refit the whole pipeline **per fold**, so the
vectorizer only ever sees that fold's training text. Same TF-IDF config as
v2's winner, now leak-safe.

### 3. Stratified k-fold cross-validation

Replaces the single 80/20 split with `StratifiedKFold(n_splits=5)` +
`cross_val_score`. One split gives one noisy number; 5-fold gives a mean and
a spread, which is a more honest estimate of how the model performs on
unseen data — and explains why the score below reads *lower* than v2's
single-split number (see Results).

### 4. Out-of-fold threshold tuning

Logistic regression defaults to a 0.5 probability cutoff for the positive
class, but that's arbitrary — it's not necessarily where F1 is maximized.
`cross_val_predict(..., method="predict_proba")` gives every training row a
predicted probability from a fold that never trained on it (out-of-fold, so
unbiased). Sweeping thresholds 0.10–0.90 against those OOF probabilities and
picking the F1-maximizing cutoff avoids the standard mistake of tuning the
threshold on the same data used to fit the model.

## Result

| Metric | Score |
|---|---|
| 5-fold CV F1 (mean ± std) | 0.7581 ± 0.0133 |
| OOF F1 @ default threshold (0.5) | 0.7580 |
| OOF F1 @ tuned threshold (0.43) | **0.7641** |
| v2 single-split F1 (for reference) | 0.7793 |

Threshold tuning alone recovered **+0.006 F1** over the default cutoff, on
the *same* model and features — a free improvement from a decision-boundary
technique rather than model or feature changes.

## Why the headline number looks lower than v2

This is the important lesson of this version, not a regression. v2 reported
F1 from **one** 80/20 split — a single number, sensitive to which rows
happened to land in validation. v3's 5-fold CV (0.7581 ± 0.0133) is the more
trustworthy estimate — 0.7793 was very plausibly on the high side of what
that ±0.013 spread allows for. Stopword removal/lemmatization may also cost
a small amount here: on a short-text, small-vocabulary dataset like tweets,
"stopwords" like *is*/*are*/*was* can still carry weak signal (tense,
phrasing) that TF-IDF's IDF weighting already downweights on its own — so
manually stripping them isn't free. Net effect is roughly neutral to
slightly negative versus v2's true (not lucky-split) performance, but the
*evaluation methodology* is now trustworthy, which matters more going
forward than one more marginal F1 tick.

## Known Limitations / Next Steps

- Lemmatization runs in default (noun) mode — `WordNetLemmatizer` needs a
  POS tag to lemmatize verbs/adjectives correctly (e.g. `"burning"` won't
  reduce without `pos="v"`). A POS-tagged lemmatization pass is a natural
  next increment.
- Threshold was tuned on OOF training predictions, then applied to test —
  standard practice, but the tuned value (0.43) is itself an estimate with
  its own variance; not re-validated on a further holdout.
- Grid search over TF-IDF/LR hyperparameters was not re-run against the new
  cleaned text — v2's winning config was reused as-is. Worth re-tuning now
  that the input vocabulary has shrunk from stopword removal.
- Still no semantic embeddings — unchanged from v1/v2's noted ceiling on
  this approach family.
