# Natural Language Processing with Disaster Tweets

## Overview

Twitter is a key real-time channel during emergencies — smartphones let people
broadcast disasters as they happen, so relief agencies and news orgs want to
monitor it programmatically. Problem: language is ambiguous. A tweet with
"ABLAZE" might report a real fire, or just be a metaphor. Obvious to a human,
not to a machine.

**Task**: binary classification — predict whether a tweet describes a real
disaster (`target=1`) or not (`target=0`).

**Disclaimer**: dataset contains profane/vulgar/offensive text.

## Data

- `data/raw/train.csv` — 8561 labeled tweets
- `data/raw/test.csv` — 3699 unlabeled tweets (predictions submitted for these)
- `data/raw/english_contractions_lowercase.json` — contraction expansion map (preprocessing aid)
- `data/raw/english_acronyms_lowercase.json` — acronym expansion map (preprocessing aid)

### Columns

| Column | Description |
|---|---|
| `id` | unique tweet identifier |
| `keyword` | keyword from tweet (may be blank) |
| `location` | location tweet sent from (may be blank, often noisy/free-text) |
| `text` | tweet text |
| `target` | (train only) 1 = real disaster, 0 = not |

## Evaluation

Metric: **F1 score** between predicted and ground-truth labels.

## Submission Format

CSV with columns:

```
id,target
```

## Approach Notes

- Text-heavy, tabular metadata (`keyword`, `location`) sparse/noisy — likely
  useful as auxiliary features, not primary signal.
- Contraction/acronym JSON maps suggest planned preprocessing: normalize
  slang before tokenization.
- Candidate approaches: classic TF-IDF + linear model baseline, then
  transformer fine-tune (e.g. BERT/DistilBERT) for final score.
