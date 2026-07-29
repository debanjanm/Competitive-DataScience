# Solution v11 — Fine-Tuned DistilBERT

## Summary

The "big jump" flagged since [v1](SOLUTION_v1.md)'s original Approach
Notes, taken now that TF-IDF (v1–v6, ceiling ~0.764) and GloVe+NN (v7–v10,
ceiling ~0.780) both plateaued. Fine-tunes `distilbert-base-uncased` end to
end on this task — genuinely different technique family, not another
increment on the previous ones.

**Result: val F1 0.8187 — the largest single jump in this entire series**,
clearing the GloVe+NN ceiling by ~0.04 F1.

## What's actually different here

Every prior version (v1–v10) shares one limitation regardless of model
family: the representation of a word is **fixed**, independent of context.
TF-IDF gives `"fire"` one column; GloVe gives `"fire"` one 100-dim vector,
same value whether the tweet is `"the building is on fire"` or `"you're
fired"`. DistilBERT's self-attention layers instead compute a
**contextual** representation — the vector for `"fire"` is a function of
every other token in the sentence, so the model can in principle actually
resolve the "ABLAZE" metaphor-vs-literal ambiguity the competition
description opens with.

Concretely, new techniques this version introduces:

1. **Subword (WordPiece) tokenization** — no OOV problem the way v7–v10
   had 13% of tokens fall back to random-initialized vectors. Unknown
   words get broken into known sub-pieces instead of dropped.
2. **Minimal preprocessing, deliberately** — only URLs/mentions stripped,
   `#` symbol removed. No stopword removal, no lemmatization, no
   lowercasing (the tokenizer's "uncased" variant does that internally).
   BERT-family models are pretrained on naturally-written text; the heavy
   cleaning used for TF-IDF/GloVe actively fights what this tokenizer and
   pretrained weights expect to see.
3. **Fine-tuning a pretrained transformer** — `AutoModelForSequenceClassification`
   loads DistilBERT's pretrained encoder plus a freshly-initialized
   classification head (`pre_classifier`/`classifier` — the warning about
   these being "newly initialized" at startup is expected, not an error).
   The whole model, encoder included, updates during training.
4. **Linear warmup + decay LR schedule**
   (`get_linear_schedule_with_warmup`) — standard for transformer
   fine-tuning: ramps the learning rate up over the first 10% of steps,
   then decays linearly, rather than using a constant rate throughout.
5. **Gradient clipping** (`clip_grad_norm_`, max norm 1.0) — also standard
   practice for transformer fine-tuning, caps gradient magnitude to avoid
   destabilizing the pretrained weights early in training.

## Result

| Epoch | Train loss | Val F1 @ 0.5 |
|---|---|---|
| 1 | 0.4839 | 0.8095 |
| 2 | 0.3496 | 0.8070 |
| 3 | 0.2665 | **0.8146** |
| 4 | 0.2090 | 0.8065 |

| Metric | Score |
|---|---|
| Best val F1 (epoch 3, threshold 0.5) | 0.8146 |
| **Best val F1 (tuned threshold 0.41)** | **0.8187** |
| v9 best NN result (for reference) | 0.7796 |
| v5 best classical result (for reference) | 0.7643 |

Train loss keeps falling through epoch 4 while val F1 peaks at epoch 3 and
dips after — classic fine-tuning overfitting signature on a ~6,000-row
training split; early stopping (patience=2) is doing real work here.

## Scope caveat (same pattern as v7 before v8)

Single 80/20 split, not 5-fold CV — fine-tuning a transformer 5x costs
5x the ~37-minute run below, which wasn't judged worth it for this exploratory
first pass. Following the same sequencing v7→v8 established: get one credible
result first, add CV rigor as a deliberate next step if this technique looks
worth investing further in. **Treat 0.8187 as directionally strong but not
yet the trustworthy mean/std estimate v3–v6, v8, v9, v10 all have.**

## Practical cost note

Wall-clock time was **~37 minutes** for 4 epochs on this machine (MPS/Apple
GPU) — vastly more than any prior version (v10's BiLSTM, the previous
slowest, took ~9 minutes). Reported CPU time (`87s user + 30s system`) is
far below the wall-clock total, consistent with most of the time being GPU
compute/data-transfer overhead rather than CPU-bound work. This cost is the
real tradeoff for the F1 jump: worth it for a competition submission,
expensive for iterating on hyperparameters quickly.

## Known Limitations / Next Steps

- No k-fold CV yet, per the scope caveat above — natural next step if this
  result should be trusted more rigorously, mirroring v7→v8.
- No hyperparameter search — learning rate (2e-5), batch size (16), epoch
  count, and max sequence length (64) were all set from common BERT
  fine-tuning conventions, not tuned for this dataset specifically.
- `keyword` is folded into the input text (same pattern as v2–v10);
  `location`, `word_count`, `char_count` engineered features from the
  classical/GloVe series are **not** used here — DistilBERT's own
  representation may already subsume what those approximated, but this
  wasn't tested directly (an ablation could confirm).
- Only 4 epochs attempted, with early stopping already triggering — a
  smaller learning rate or per-layer LR decay (lower rate for earlier
  transformer layers) might get more useful epochs before overfitting sets
  in, worth trying before concluding this is DistilBERT's ceiling on this
  dataset.
- `distilbert-base-uncased` is a general-purpose 6-layer distillation of
  BERT-base — a Twitter-pretrained variant (e.g. BERTweet) or the full
  12-layer BERT-base could plausibly do better, at proportionally more
  training cost.
