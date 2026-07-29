# Solution v8 — 5-Fold CV for the GloVe + Mean-Pool NN

## Summary

Deliberately the *simplest* possible next step after [v7](SOLUTION_v7.md),
picked over fine-tuning embeddings or swapping in an LSTM: wrap v7's exact
model/features/training loop in `StratifiedKFold(5)`, matching the rigor
v3–v6 already established for the classical series. No model change, no
feature change — only the evaluation protocol changes. This also makes v7's
result directly comparable to the classical series for the first time.

## What changed vs v7

- **`StratifiedKFold(5)` instead of one `train_test_split`.** Each of the 5
  folds trains its own `MeanPoolClassifier` from scratch, with the same
  early-stopping-on-val-F1 logic v7 used, just applied per fold
  (`train_one_fold()`).
- **Out-of-fold (OOF) probabilities** — each training row gets a prediction
  from the one fold where it was held out, so every row is scored by a
  model that never trained on it. Same technique as v3–v6's
  `cross_val_predict`, just driven by a manual loop since there's no
  sklearn estimator to hand to `cross_val_predict` here.
- **Test predictions = average of all 5 fold models**, not one final refit.
  This is standard practice for k-fold neural net training — it uses every
  model that got trained anyway (a free bagging-style ensemble) rather than
  paying for a 6th full-data training run.

## Result

| Fold | Val F1 @ 0.5 |
|---|---|
| 1 | 0.7863 |
| 2 | 0.7566 |
| 3 | 0.7538 |
| 4 | 0.7645 |
| 5 | 0.7831 |

| Metric | Score |
|---|---|
| **5-fold OOF F1 (mean ± std)** | **0.7690 ± 0.0134** |
| Best threshold | 0.50 (no change from default) |
| v7 single-split F1 (for reference) | 0.7825 |
| v5 5-fold CV F1 (best classical, for reference) | 0.7643 |

## The eval-rigor payoff

This is the same lesson v3 taught about the classical series, now confirmed
for the NN family too: v7's 0.7825 was a **single lucky split** — the honest
5-fold estimate is 0.7690 ± 0.0134, meaningfully lower and with real spread
(fold 3's 0.7538 vs. fold 1's 0.7863 is a 3+ point swing on the *same*
model/data, different row partition). v7's headline number was never wrong,
exactly — it's just one draw from a distribution that v8 now actually shows.

With this fix, **v8's NN (0.7690) and v5's classical result (0.7643) are
now genuinely comparable** — both 5-fold CV means. The gap is real but
modest (+0.0047), not the ~0.02 gap v7's single-split number implied.

## Known Limitations / Next Steps

- Training cost is now 5x — ~66s total here vs. v7's ~53s (which included a
  one-time embedding load v8 also pays once, shared across folds). Fine at
  this dataset size; would need `RandomizedSearchCV`-style budget thinking
  at larger scale.
- Still frozen embeddings, still mean pooling — v7's other two deferred
  items (fine-tuning, sequence-aware pooling) are unaddressed by this
  version on purpose, since this round was scoped to eval rigor only.
- The 5 fold models differ only by data split (same architecture, same
  hyperparameters) — a true ensemble diversity benefit would come from
  varying architecture or hyperparameters per fold too, not attempted here.
- Next natural steps, roughly increasing in complexity: fine-tune embeddings
  (unfreeze, small separate LR), then sequence-aware pooling (LSTM/GRU or
  attention) to recover the word-order signal both TF-IDF and mean pooling
  discard, then eventually a full transformer fine-tune.
