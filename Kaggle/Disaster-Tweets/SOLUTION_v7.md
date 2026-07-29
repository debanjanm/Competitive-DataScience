# Solution v7 — GloVe Embeddings + Mean-Pooled Feedforward NN

## Summary

First deep-learning-family version, moving off the TF-IDF approach that
plateaued and then regressed in [v5](SOLUTION_v5.md)/[v6](SOLUTION_v6.md).
Replaces bag-of-words TF-IDF with pretrained GloVe word embeddings, and
replaces `LogisticRegression` with a small PyTorch feedforward network.
New tooling family entirely: `torch`, `gensim`, manual training loop instead
of `sklearn` `Pipeline`/`GridSearchCV`.

## Pipeline

1. **Text cleaning** — lighter than v3–v6: lowercase, strip URLs/mentions/
   hashtag-symbol, expand contractions/acronyms, strip non-alpha, collapse
   whitespace. **No stopword removal or lemmatization** — GloVe's vocabulary
   was trained on raw Twitter text, so tokens need to stay in the form that
   vocabulary actually indexes (lemmatizing `"floods"` to `"flood"` risks
   mapping to a *different* embedding than the one the model actually
   needs, or off-vocabulary entirely).
2. **Pretrained embeddings** — `glove-twitter-100` (gensim-hosted,
   Stanford's GloVe vectors trained on 2B tweets, 1.19M-word vocab, 100
   dimensions). Domain-matched to this dataset on purpose — Wikipedia/
   Gigaword GloVe would carry vectors for cleaner, more formal English,
   which under-serves tweet slang and abbreviations.
3. **Vocabulary + embedding matrix** — build a vocab from train+test tokens
   (`<pad>`=0, `<unk>`=1, rest indexed by first appearance), then build a
   `(vocab_size, 100)` matrix: GloVe's vector where the token is in GloVe's
   vocabulary, small random init otherwise. **87.1% coverage** (14,941/
   17,145 tokens) — the other ~13% (misspellings, rare hashtags, usernames
   that survived cleaning) fall back to random vectors.
4. **Model** (`MeanPoolClassifier`)
   - `nn.Embedding.from_pretrained(..., freeze=True)` — embeddings fixed,
     not fine-tuned, for this first pass.
   - Mask-aware mean pooling over the token sequence (ignores `<pad>`
     positions) — collapses variable-length tweets into one 100-dim vector.
   - Concatenate with the same 3 scaled numeric features used since v2
     (`has_location`, `word_count`, `char_count`).
   - `Linear(103→64) → ReLU → Dropout(0.3) → Linear(64→1)` → sigmoid.
5. **Training** — manual PyTorch loop, `BCEWithLogitsLoss`, Adam
   (`lr=1e-3`), batch size 64, up to 30 epochs with early stopping
   (patience=4 on validation F1). Runs on MPS (Apple GPU) if available.
6. **Threshold tuning** — same technique as v3–v6: sweep 0.10–0.90 against
   validation predictions, pick the F1-maximizing cutoff.

## Evaluation protocol note (read before comparing to v1–v6)

This version uses a **single 80/20 train/val split**, not the 5-fold
`StratifiedKFold` CV used since v3. Running full k-fold CV around a
from-scratch training loop (5x the training time, needs its own early-
stopping-per-fold logic) is a bigger lift than the sklearn-`Pipeline` case,
where `cross_val_score` handles it for free — deferred here to keep this
first NN pass tractable. **The F1 numbers below are therefore not directly
apples-to-apples with v3–v6's CV means** — a single split carries more
variance (recall v3's finding that single-split numbers can run ~0.02
optimistic/pessimistic vs. the 5-fold mean).

## Result

| Metric | Score |
|---|---|
| Embedding coverage | 87.1% (14,941 / 17,145 tokens) |
| Best validation F1 (early-stopped, epoch 8) | **0.7825** |
| Threshold tuning | no change — 0.50 was already optimal |
| v5 5-fold CV F1 (best classical, for reference) | 0.7643 |

Directionally ahead of the best classical result, though the single-split
caveat above means this isn't a fully rigorous comparison yet.

## Design Notes / Why These Choices

- **Frozen embeddings, not fine-tuned**: fine-tuning a 1.7M-parameter
  embedding table (17,145 vocab × 100 dims) on ~6,000 training rows risks
  overfitting badly and losing the pretrained semantic structure that's the
  whole point of using GloVe. Freezing first establishes what the pretrained
  vectors alone are worth; fine-tuning is a clean next increment to measure
  in isolation.
- **Mean pooling, not RNN/LSTM**: simplest fixed-size representation of a
  variable-length token sequence. It discards word order entirely (same
  bag-of-embeddings problem TF-IDF had, just with dense semantic vectors
  instead of sparse counts) — a sequence model is the natural next step once
  this baseline is understood.
- **Early stopping on val F1, not val loss**: F1 is the competition metric;
  BCE loss and F1 don't always move together near the decision boundary, so
  optimizing checkpoint selection directly on F1 avoids picking a "lower
  loss but worse F1" epoch.
- **Numeric features still included**: kept the same 3 engineered features
  from the classical series for continuity/comparability, not because the
  NN specifically needs them — worth an ablation later to see how much they
  actually contribute here versus in the linear-model context.

## Known Limitations / Next Steps

- **No k-fold CV** — single split, as noted above. Natural v8 candidate:
  wrap this training loop in `StratifiedKFold` for a trustworthy mean ± std,
  matching v3's rigor.
- **Frozen embeddings** — try `freeze=False` (fine-tuning) as an isolated
  ablation, likely with a lower learning rate on the embedding layer than
  the dense head to avoid catastrophic forgetting of pretrained structure.
- **Mean pooling discards word order** — an LSTM/GRU or attention-based
  pooling would let the model use sequence information TF-IDF and this
  version both throw away.
- **13% OOV tokens fall back to random vectors** — a subword-aware
  embedding (fastText, which builds vectors for unseen words from character
  n-grams) would close this gap without needing more preprocessing.
- **Still not a full transformer fine-tune** — DistilBERT/BERT remains the
  eventual larger step; this version is the deliberate middle ground between
  classical ML and that.
