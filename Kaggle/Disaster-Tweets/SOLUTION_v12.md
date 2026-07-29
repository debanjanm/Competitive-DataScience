# Solution v12 — 5-Fold CV for Fine-Tuned DistilBERT

## Summary

Closes [v11](SOLUTION_v11.md)'s scope caveat: wraps the exact same
DistilBERT fine-tune (same tokenization, same hyperparameters, same
early-stopping-on-val-F1 logic) in `StratifiedKFold(5)`, mirroring the
v7→v8 pattern for the GloVe+NN series. No model change, no feature change
— only the evaluation protocol changes, and for the first time this series
has an honest CV estimate for the transformer approach.

## What changed vs v11

- **`StratifiedKFold(5)` instead of one `train_test_split`.**
- **A fresh DistilBERT per fold** — `AutoModelForSequenceClassification.from_pretrained()`
  is called inside `train_one_fold()`, once per fold, re-loading the
  original pretrained checkpoint each time. Reusing one fine-tuned model
  across folds would leak fold 1's learned weights into fold 2's
  supposedly-held-out evaluation — same leakage concern v3's `Pipeline`
  fix addressed for TF-IDF, applied here to model weights instead of a
  fitted vectorizer.
- **Out-of-fold probabilities** collected across all 5 folds for threshold
  tuning — same technique as v3–v6, v8, v9, v10, just via a manual loop
  since there's no sklearn estimator to hand to `cross_val_predict`.
- **Test predictions = average of all 5 fold models**, same bagging-style
  approach v8/v9/v10 used.
- **`torch.mps.empty_cache()` after each fold** — new, and necessary here
  specifically: 5 fresh ~268MB DistilBERT models get created and discarded
  in sequence, and without explicitly releasing MPS memory between folds,
  GPU memory pressure builds up across a run this long.

## Result

| Fold | Val F1 @ 0.5 |
|---|---|
| 1 | 0.8155 |
| 2 | 0.7971 |
| 3 | 0.8009 |
| 4 | 0.8049 |
| 5 | 0.7963 |

| Metric | Score |
|---|---|
| **5-fold OOF F1 (mean ± std)** | **0.8029 ± 0.0070** |
| OOF F1 @ tuned threshold (0.54) | **0.8040** |
| v11 single-split F1 (for reference) | 0.8187 |
| v9 best GloVe+NN result (for reference) | 0.7796 |
| v5 best classical result (for reference) | 0.7643 |
| Wall-clock time | **5h 22m 43s** (5x v11's ~37min, as predicted) |

## The eval-rigor payoff, one more time

Same lesson as v2→v3 and v7→v8, now confirmed for the transformer family
too: v11's 0.8187 was a lucky single split. The honest 5-fold estimate is
0.8029 ± 0.0070 — lower, and with real (if modest) spread across folds.
This is now the **third time** this exact pattern has shown up in this
series (classical TF-IDF, GloVe+NN, and now DistilBERT) — a single
train/val split consistently overstates performance by roughly 1-2 F1
points on this dataset size, regardless of model family. That consistency
is itself a useful thing to have learned: it's not a quirk of any one
technique, it's a property of evaluating on ~1,500-3,000 held-out rows.

Even accounting for that correction, **v12 is still the best result in the
entire series** — 0.8029 OOF vs. v9's 0.7796 (GloVe+NN) and v5's 0.7643
(classical TF-IDF), all now on the same 5-fold CV footing and genuinely
comparable for the first time.

## Cost note

5h22m is a large jump from every prior version — even v10's BiLSTM (the
previous most expensive, ~9min) is dwarfed by this. The `2% cpu` figure in
the `time` output (410s user + 131s system against 5h23m wall) says most of
that wall-clock time was GPU compute/overhead on MPS, not CPU-bound work —
consistent with v11's same observation, just amplified 5x. This is the
practical tradeoff of CV rigor on a technique this expensive: worth it once,
to establish a trustworthy final number, but not something to re-run casually
while iterating on hyperparameters.

## Known Limitations / Next Steps

- This is a reasonable stopping point for the "learn all the techniques"
  arc that started at v1 — classical ML (v1–v6), embeddings + neural nets
  (v7–v10), and transformer fine-tuning (v11–v12) have each been tried,
  evaluated rigorously, and compared on equal footing.
- If continuing further: hyperparameter search for DistilBERT (learning
  rate, batch size, epoch count) was never attempted — v11/v12 used
  standard fine-tuning conventions, not a tuned config, and a search here
  costs proportionally more than any GridSearchCV run in this series given
  the ~5h price of just one CV pass.
- A domain-pretrained variant (BERTweet, pretrained on tweets specifically
  rather than general web text) is the most likely remaining lever for a
  further F1 gain without changing technique family.
- `submission_v12.csv` is the recommended final submission for this
  competition given the full series' results.
