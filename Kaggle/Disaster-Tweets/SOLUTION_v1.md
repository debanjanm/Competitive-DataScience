# Solution v1 — TF-IDF + Logistic Regression Baseline

## Summary

Baseline model for the disaster-tweet binary classification task. Classic
sparse-vector pipeline: clean text → TF-IDF vectorize → logistic regression.
No deep learning, no pretrained embeddings — establishes a floor score to
beat with later versions.

## Pipeline

1. **Load data** — `train.csv` (8561 rows), `test.csv` (3699 rows) from `data/raw/`.
2. **Text cleaning** (`clean_text`)
   - Lowercase
   - Strip URLs (`http(s)://...`, `www...`)
   - Strip `@mentions`
   - Strip `#` symbol (keep hashtag word itself — often carries signal, e.g. `#earthquake`)
   - Expand contractions via `english_contractions_lowercase.json` (e.g. `don't` → `do not`)
   - Expand acronyms via `english_acronyms_lowercase.json` (e.g. slang → full form)
   - Strip non-alphabetic characters
   - Collapse whitespace
3. **Split** — 80/20 train/validation, stratified on `target`, `random_state=42`.
4. **Vectorize** — `TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2)`.
   Unigrams + bigrams capture some local phrase context (e.g. "car crash").
5. **Model** — `LogisticRegression(max_iter=1000, C=1.0)`. Fast, interpretable,
   strong baseline for sparse high-dim text features.
6. **Validate** — score on held-out 20% via F1 (competition metric).
7. **Refit + predict** — refit vectorizer + model on full training set, predict
   on test set, write `submission_v1.csv` (`id,target`).

## Result

| Metric | Score |
|---|---|
| Validation F1 | **0.7686** |

## Design Notes / Why These Choices

- **TF-IDF over raw counts**: downweights common tokens, boosts distinctive
  disaster-vocabulary terms.
- **Bigrams included**: single words like "fire" are ambiguous ("fire" a
  photo vs. actual fire); short phrase context helps a linear model without
  needing a neural encoder.
- **Contraction/acronym expansion before stripping punctuation**: prevents
  losing negation signal (e.g. "don't" → "do not" keeps "not" as a token
  instead of collapsing to "dont" and losing it to `min_df` pruning).
- **Hashtag symbol stripped, not the word**: hashtags in this dataset (e.g.
  `#earthquake`) are often the disaster keyword itself — dropping the whole
  token would throw away signal.
- **Refit on full data before test prediction**: standard practice — use all
  labeled data for the final model once validation score is captured,
  since held-out split was only needed to estimate generalization.

## Known Limitations / Next Steps

- No `keyword` / `location` metadata used — both columns are sparse/noisy but
  may carry auxiliary signal (e.g. `location` correlates with real news
  agencies).
- No spelling correction — tweets contain typos/informal contractions not in
  the lookup tables.
- No pretrained embeddings — TF-IDF misses semantic similarity (e.g. "blaze"
  vs. "fire" treated as unrelated tokens).
- Candidate v2 direction: fine-tune DistilBERT/BERT on raw or lightly cleaned
  text, likely large F1 gain given the small unigram vocabulary limits above.
