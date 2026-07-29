# Solution v6 — TruncatedSVD (LSA) + class_weight (Negative Result)

## Summary

Last planned classical-ML round before moving to embeddings/deep learning.
Two techniques tested: `TruncatedSVD` (LSA) dimensionality reduction on both
TF-IDF blocks, and `class_weight='balanced'` to address the train set's mild
57/43 class split. **Result: this version is worse than v5, and the
regression traces entirely to SVD.** Recommend continuing forward from
[v5](SOLUTION_v5.md)'s sparse-feature approach, not this one — kept and
documented anyway, because knowing what *doesn't* help is as much a
technique-learning outcome as what does.

## What changed vs v5

### 1. TruncatedSVD (LSA) on each TF-IDF block

Each TF-IDF block (`word_tfidf`, `char_tfidf`) got a `TruncatedSVD` step
appended inside its own `Pipeline`, collapsing v5's sparse ~30000-dim
combined TF-IDF space down to a dense 100–200 component "latent semantic"
space per block. Motivation: sparse high-dimensional TF-IDF next to 3 dense
scaled numeric columns is a lopsided feature space; LSA is the standard way
to bring bag-of-words text down to a compact dense representation, and dense
low-rank features are also what most non-linear-model families (kNN,
non-linear SVM kernels, gradient boosting) actually need to work well with
text.

### 2. `class_weight` tuning

Train target is ~57% class 0 / 43% class 1 — mild but real imbalance, never
previously checked. Added `class_weight ∈ {None, "balanced"}` to the grid.

## Result

| Run | 5-fold CV F1 |
|---|---|
| v5 (sparse TF-IDF, no SVD, `class_weight=None`) | 0.7643 |
| v6 grid best (SVD n_components=200, `class_weight="balanced"`, C=3.0) | 0.7467 |
| **Isolation check**: v5 features + `class_weight="balanced"`, no SVD | 0.7643 |

The isolation check (run separately, not part of the original grid) removes
the ambiguity: `class_weight="balanced"` alone reproduces v5's score
exactly — it's a no-op here, because 57/43 isn't imbalanced enough for
`LogisticRegression`'s decision boundary to need reweighting. The full
**-0.018 F1 drop is attributable to `TruncatedSVD` alone.**

## Why SVD hurt here

- **Information loss vs. feature count**: TF-IDF's sparse ~30000 dims are
  individually meaningful (each is a literal word or char n-gram);
  compressing to 100–200 dense components necessarily discards variance,
  and short, noisy tweet text doesn't have enough shared latent structure
  at this dataset size (~7600 rows) to make up for it.
- **Linear model, not a distance/kernel method**: LSA compression pays off
  most for methods that are expensive or ill-behaved in high sparse
  dimensions (kNN, SVM with RBF kernel, clustering). `LogisticRegression`
  with L2 regularization already handles high-dimensional sparse input
  natively and efficiently — SVD doesn't remove a bottleneck this model
  actually had.
- **Two independent SVDs, not one joint one**: word and char blocks were
  reduced separately, so nothing enforces their latent components to align
  with each other's semantic space — this fragmentation likely costs
  additional signal beyond what a single combined-then-reduced approach
  would.

## Known Limitations / Next Steps

- Didn't test a joint SVD over the concatenated word+char sparse matrix
  (single `TruncatedSVD` after `hstack`, not one per block) — might recover
  some of the loss, but not worth chasing further given classical TF-IDF
  approaches are being wound down here regardless.
- Didn't test `class_weight="balanced"` combined with SVD in isolation from
  the SVD grid noise — but moot, since SVD is the confirmed culprit.
- **This is the planned end of the classical-ML technique series** for this
  problem (v1–v6). v5's config (`submission_v5.csv`) is the best classical
  result — 0.7643 CV F1, 0.7653 with threshold tuning. Next step: move to
  embeddings/deep learning (word embeddings + small neural net, or a
  transformer fine-tune) as a new technique family rather than further
  classical tuning.
