# Solution v10 — Bidirectional LSTM Pooling (Flat Result)

## Summary

Replaces v9's mean pooling with a bidirectional LSTM, to let word order and
short-range dependencies (e.g. "not on fire" vs "on fire") influence the
sentence representation instead of collapsing every tweet into an unordered
bag of embeddings. Everything else — fine-tuned GloVe embeddings, 5-fold
CV, numeric features, OOF threshold tuning, fold-averaged test predictions
— unchanged from [v9](SOLUTION_v9.md).

**Result: essentially flat versus v9**, at roughly 7x the training cost.
Documented in full because — same spirit as [v6](SOLUTION_v6.md)'s
TruncatedSVD experiment — knowing a more sophisticated technique *didn't*
help, and having a concrete reason why, is real signal for what to try next.

## What changed vs v9

- **`LSTMClassifier`** replaces `MeanPoolClassifier`: token embeddings feed
  a single-layer `nn.LSTM(bidirectional=True)` instead of being averaged
  directly.
- **`pack_padded_sequence`** — real per-row sequence lengths are computed
  (count of non-pad tokens) and used to pack the batch before the LSTM, so
  the recurrence skips padding positions entirely rather than running
  extra (meaningless) steps over zero-vector pad embeddings.
- **Pooling**: the final forward-direction hidden state (`h_n[0]`, the
  state after reading left-to-right) and the final backward-direction
  hidden state (`h_n[1]`, after reading right-to-left) are concatenated
  into one `128`-dim vector — this replaces the `100`-dim mean-pooled
  vector as input to the same dense head (now `130`-dim input: 128 LSTM +
  3 numeric, vs. v9's `103`).

## Result

| Fold | Val F1 @ 0.5 |
|---|---|
| 1 | 0.7947 |
| 2 | 0.7739 |
| 3 | 0.7670 |
| 4 | 0.7711 |
| 5 | 0.7876 |

| Metric | v9 (mean pool) | v10 (BiLSTM) | Δ |
|---|---|---|---|
| 5-fold OOF F1 (mean ± std) | 0.7792 ± 0.0086 | 0.7789 ± 0.0105 | **-0.0003** |
| OOF F1 @ tuned threshold | 0.7796 | 0.7797 | +0.0001 |
| Wall-clock training time | ~80s | **~547s (9m7s)** | ~7x slower |

Within noise of each other — no meaningful gain, and fold-to-fold std
actually got slightly *worse* (0.0086 → 0.0105), not better.

## Why the LSTM didn't help here

- **Tweets are short.** Median cleaned length is ~15 tokens (see
  [PROBLEM.md](PROBLEM.md) data notes) — most of the "does word order
  matter" cases a sequence model is built to capture (long-range
  dependencies, clause structure) barely exist at this length. Mean
  pooling already captures "which words are present," and for short,
  fragment-like text that may just be most of the signal available.
- **Small data, bigger model.** The LSTM adds a real parameter block
  (input-to-hidden and hidden-to-hidden weight matrices, ×4 gates,
  ×2 directions) on top of the same ~6,000-row-per-fold training set that
  was already enough to fine-tune the embedding table in v9. More capacity
  without more data tends to buy overfitting risk, not accuracy — consistent
  with the flat-to-slightly-worse std here.
- **Disaster-tweet classification may just not be order-sensitive.**
  Whether a tweet is about a real disaster often hinges on *which* words
  appear (`"earthquake"`, `"evacuated"`, `"casualties"`) more than on their
  arrangement — different from tasks like sentiment/sarcasm detection where
  negation placement and clause order carry the label.

## Known Limitations / Next Steps

- Single-layer, single-direction-pair LSTM — didn't try stacking layers or
  a GRU (cheaper per step, sometimes comparable accuracy) as alternatives.
- Didn't try attention pooling (a weighted combination over all LSTM
  outputs, not just the final hidden state) — attention can sometimes
  recover value a final-hidden-state summary loses, though given the mean
  pooling comparison here, expected upside looks limited for this dataset.
- Given the cost/benefit here, **the roadmap should treat "sequence
  modeling doesn't move this dataset" as a working conclusion** rather than
  iterating further on RNN variants — the next meaningfully different step
  is a transformer fine-tune (contextual embeddings, not just sequence
  order, are the actual thing TF-IDF/GloVe/LSTM all still lack).
- v9 remains the better cost-adjusted choice of the two — recommend it over
  v10 as the standing "best NN" result unless a future test shows the LSTM
  helps a metric this OOF-F1 comparison doesn't capture (e.g. precision/
  recall balance at a fixed threshold).
