# Solution v4 — POS-Tagged Lemmatization + GridSearchCV Re-Tune

## Summary

Closes two limitations [v3](SOLUTION_v3.md) named explicitly: lemmatization
was noun-only (verbs/adjectives didn't reduce properly), and the TF-IDF/LR
hyperparameters were still v2's — never re-tuned after stopword removal and
lemmatization changed the vocabulary. Same model family, same features,
same leak-safe `Pipeline` structure as v3.

## What changed vs v3

### 1. POS-tagged lemmatization

v3's `WordNetLemmatizer.lemmatize(word)` defaults to noun mode, so verb
forms like `"burning"` or `"crashed"` weren't reduced to their lemma —
lemmatization only actually fired on words that happened to already look
like plural nouns.

v4 runs `nltk.pos_tag()` on the full token sequence first (POS tagging
needs sentence context, so this happens *before* stopword removal), maps
each Treebank tag to a WordNet POS category (`to_wordnet_pos`: adjective /
verb / adverb / default-noun), then lemmatizes each surviving token with its
actual POS. `"burning"` (VBG) → `"burn"`; a plain noun still lemmatizes as
before.

### 2. `GridSearchCV` replaces the manual tuning loop

v2 tuned TF-IDF/LR with a hand-written triple `for` loop over a single
train/val split. v4 uses `GridSearchCV` over the same parameter space —
`max_features ∈ {10000, 20000, 30000}`, `ngram_range ∈ {(1,1), (1,2)}`,
`C ∈ {0.5, 1.0, 3.0}` — but scored via the 5-fold `StratifiedKFold` from v3,
against text that's now been stopword-stripped and POS-lemmatized. Because
the vectorizer lives inside the `Pipeline`, `GridSearchCV` refits it
per-fold per-combo automatically — no separate leak-safety code needed,
`GridSearchCV` + `Pipeline` gives it for free.

Threshold tuning is unchanged from v3 (`cross_val_predict` on a fresh clone
at the winning params, sweep 0.10–0.90, pick the F1-max cutoff) — reused as-is
since it's orthogonal to what changed here.

## Result

| Metric | Score |
|---|---|
| Best params | `max_features=10000, ngram_range=(1,2), C=3.0` |
| 5-fold CV F1 (GridSearchCV best) | **0.7623** |
| OOF F1 @ tuned threshold (0.43) | **0.7635** |
| v3 5-fold CV F1 (for reference) | 0.7581 |

+0.0042 F1 from properly-tagged lemmatization and re-tuning together. Note
the grid picked `ngram_range=(1,2)` this time — bigrams, which v2's search
had picked but v3 (unretuned) was still running on the old unigram-only
winner; re-tuning let bigrams back in now that the shrunk, lemmatized
vocabulary changes what a "useful bigram" looks like.

## Why the gain is modest

This is a **finer-improvement** version by design — no new features, no new
model family. The gain is the ceiling of what "tune the same linear model
more carefully" can buy on top of already-decent preprocessing. Consistent
with the pattern across v1→v4: most of the score lives in the TF-IDF +
linear-model + reasonable-cleaning combination itself; incremental technique
passes move the needle by low single-digit tenths of an F1 point, not
whole points.

## Known Limitations / Next Steps

- `GridSearchCV` here is exhaustive (18 combos × 5 folds = 90 fits) — fine
  at this dataset size (~20s total) but wouldn't scale; `RandomizedSearchCV`
  or `Optuna` would be the next tool to learn once search spaces grow.
- POS tagging via `nltk.pos_tag` is an off-the-shelf, not domain-tuned,
  tagger — tweet grammar (fragments, no punctuation-based sentence
  boundaries) is exactly what generic POS taggers handle worst. Some
  mistagged tokens likely lemmatize incorrectly.
- Numeric features (`has_location`, `word_count`, `char_count`) are still
  unscaled while sitting next to L2-regularized TF-IDF weights — feature
  scaling (`StandardScaler` on the numeric block via `ColumnTransformer`) is
  an unexplored, cheap next technique.
- Semantic ceiling from v1–v3 still applies — TF-IDF-family approaches are
  close to exhausted for this dataset; a transformer fine-tune remains the
  next *big* step whenever that's the goal instead of incremental technique
  practice.
