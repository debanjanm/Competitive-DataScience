# Solution v5 — Char N-Gram TF-IDF + Feature Scaling

## Summary

Two more items pulled straight from [v4](SOLUTION_v4.md)'s Next Steps: a
character n-gram TF-IDF block run in parallel with the existing word-level
TF-IDF, and `StandardScaler` on the numeric feature block. Word-level TF-IDF
config is frozen at v4's winner (`max_features=10000, ngram_range=(1,2)`);
the grid search this round only covers the new char-ngram hyperparameters
and `C`, keeping search scope small.

## What changed vs v4

### 1. Character n-gram TF-IDF block

Added `TfidfVectorizer(analyzer="char_wb")` as a second, independent
transformer inside the same `ColumnTransformer`, both reading
`text_combined` and their outputs concatenated (`ColumnTransformer` does
this automatically — one row's feature vector is word-TF-IDF columns +
char-TF-IDF columns + numeric columns, side by side).

Word-level TF-IDF treats `"fire"` and a typo like `"firee"` as two
completely unrelated tokens — one hit, one miss, zero shared signal.
Character n-grams (`char_wb` = char n-grams within word boundaries) instead
break `"fire"` into overlapping chunks like `_fi`, `fir`, `ire`, `re_`, most
of which a misspelling like `"firee"` still shares. This gives the model
partial credit on typos, hashtag mashups, and informal slang the word-level
vectorizer drops entirely.

Grid searched `ngram_range ∈ {(3,5), (2,4)}` and `max_features ∈ {10000,
20000}` for this block; word block and lemmatization/stopword pipeline
unchanged from v4.

### 2. `StandardScaler` on numeric features

`has_location` (0/1), `word_count` (~0–30), `char_count` (~0–150) were
riding unscaled next to TF-IDF weights that are L2-normalized into roughly
`[0, 1]`. For an L2-regularized `LogisticRegression`, unscaled features with
larger raw magnitude get penalized more/less arbitrarily relative to their
actual predictive value — regularization strength isn't comparable across
features on different scales. `StandardScaler` (zero mean, unit variance)
puts all three numeric columns on the same footing as everything else
before the classifier sees them.

## Result

| Metric | Score |
|---|---|
| Best params | `char ngram_range=(3,5), char max_features=20000, C=3.0` |
| 5-fold CV F1 (GridSearchCV best) | **0.7643** |
| OOF F1 @ tuned threshold (0.46) | **0.7653** |
| v4 5-fold CV F1 (for reference) | 0.7623 |

+0.002 F1 from the two changes combined. Char n-grams picked the widest
allowed range (3–5), suggesting there's real signal in that block beyond
what word-level TF-IDF already had, even if the gain is modest at this
dataset size (~7600 rows) — char n-gram vocabularies get very large,
very fast, and 7600 rows isn't much to estimate 20000 extra weights from.

## Known Limitations / Next Steps

- Char n-gram vocabulary (20000 features) roughly doubles total feature
  count versus v4 — training/search time went from ~20s to ~25s. Fine here,
  but a sign this combination doesn't scale cheaply; `TruncatedSVD` (LSA)
  on the combined sparse matrix would be a natural next technique to learn
  (dimensionality reduction on sparse TF-IDF).
- Word-level TF-IDF hyperparameters weren't re-searched jointly with the
  char-ngram addition — only re-searched the new block in isolation. A
  joint search might find a different optimum (e.g. fewer word features
  once char n-grams cover some of the same ground).
- `class_weight='balanced'` still untried — worth checking whether the
  target's class balance is skewed enough to matter (not yet measured in
  this series).
- Semantic ceiling unchanged — still no embeddings; remains the eventual
  bigger step once incremental technique practice on this TF-IDF family is
  done.
