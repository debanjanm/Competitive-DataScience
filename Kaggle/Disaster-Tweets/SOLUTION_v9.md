# Solution v9 — Fine-Tuned (Unfrozen) GloVe Embeddings

## Summary

Next item off [v8](SOLUTION_v8.md)'s list, isolated to one change: unfreeze
the GloVe embeddings so they update during training instead of staying
fixed. Everything else — 5-fold CV, mean pooling, dense head, numeric
features, OOF threshold tuning, test-time fold averaging — identical to v8.

## What changed vs v8

- **`nn.Embedding.from_pretrained(..., freeze=False)`** — v7/v8 froze the
  embedding table; v9 lets it train.
- **Two-speed optimizer** via Adam param groups: embedding layer at
  `lr=1e-4`, dense head at `lr=1e-3` (10x faster). Rationale: the head
  starts from random init and needs to move fast to learn anything at all;
  the embeddings start from *already-good* pretrained GloVe vectors and
  should only drift a little to specialize for this task, not get
  overwritten by noisy early gradients at the head's learning rate.

## Result

| Fold | Val F1 @ 0.5 |
|---|---|
| 1 | 0.7880 |
| 2 | 0.7679 |
| 3 | 0.7715 |
| 4 | 0.7784 |
| 5 | 0.7896 |

| Metric | v8 (frozen) | v9 (fine-tuned) | Δ |
|---|---|---|---|
| 5-fold OOF F1 (mean ± std) | 0.7690 ± 0.0134 | **0.7792 ± 0.0086** | **+0.0102** |
| OOF F1 @ tuned threshold | 0.7690 | **0.7796** (threshold 0.49) | +0.0106 |

Biggest single-change gain since the original v1→v2 jump. Fold-to-fold
spread also tightened (std 0.0134 → 0.0086) — fine-tuning didn't just raise
the mean, it made the model's performance more consistent across different
data splits.

## Why this helped

GloVe-Twitter was trained on general Twitter chatter, not specifically on
disaster-vs-not framing. A word like `"crash"` or `"wreckage"` sits in
GloVe's embedding space based on broad co-occurrence patterns across all of
Twitter, not this task's specific decision boundary. Letting the embeddings
move (a little, at low LR) lets the model nudge those vectors toward
directions that separate this dataset's two classes better — while the low
relative LR keeps it from drifting so far that the pretrained semantic
structure (the actual reason GloVe was worth using) gets destroyed. The
tightened std is consistent with this: better-adapted embeddings likely
also reduce sensitivity to which specific rows end up in a given fold.

## Known Limitations / Next Steps

- The 10x LR ratio (1e-4 vs 1e-3) was picked by convention/intuition, not
  swept — an ablation over the ratio (or a learning-rate schedule/warmup)
  is a further, smaller refinement.
- No weight decay or gradient clipping on the embedding layer specifically
  — with only ~6,000 training rows per fold, some overfitting risk on the
  now-trainable 17,145 × 100 embedding table is plausible, just not
  measured here (train/val loss curves weren't inspected for this version).
- Still mean pooling — word order is still discarded going into the dense
  head, unchanged from v7/v8.
- Next natural step, per the v7/v8 roadmap: sequence-aware pooling
  (LSTM/GRU or attention) to recover ordering signal, before the eventual
  full transformer fine-tune.
