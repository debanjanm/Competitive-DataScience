# Synthetic Financial Datasets For Fraud Detection

## Overview

Public financial-transaction datasets are scarce — real transaction logs are
private by nature, which hamstrings fraud-detection research. **PaySim**
fills that gap: a mobile-money simulator built from one month of real
transaction logs from a mobile financial service operating in 14+ countries
(sample from an African country deployment). PaySim reproduces normal
transaction behavior and injects malicious agents to generate a dataset
usable for fraud-detection benchmarking.

This dataset is a synthetic run scaled to 1/4 of the original PaySim output,
released for Kaggle.

**Task**: binary classification — predict whether a transaction is
fraudulent (`isFraud=1`) or not (`isFraud=0`).

**Important**: fraudulent transactions are cancelled by the system, so
`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` are
already end-state values, not what the actor saw in real time. Using them
naively as features leaks information about the outcome — leakage risk, not
free signal. Treat with care in feature engineering.

## Data

- `data/PS_20174392719_1491204439457_log.csv` — 6,362,620 transactions (~471MB)

### Columns

| Column | Description |
|---|---|
| `step` | time unit = 1 hour; 744 steps total (30-day simulation) |
| `type` | `CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | transaction amount, local currency |
| `nameOrig` | customer initiating the transaction |
| `oldbalanceOrg` | origin balance before transaction (⚠ leakage risk — see above) |
| `newbalanceOrig` | origin balance after transaction (⚠ leakage risk) |
| `nameDest` | recipient customer; `M` prefix = merchant |
| `oldbalanceDest` | recipient balance before transaction (⚠ leakage risk; not populated for merchants) |
| `newbalanceDest` | recipient balance after transaction (⚠ leakage risk; not populated for merchants) |
| `isFraud` | target — 1 = fraudulent agent transaction (account takeover → drain via transfer → cash-out), 0 = legit |
| `isFlaggedFraud` | system flag for illegal attempts — single transfer > 200,000 |

## Evaluation

No official Kaggle leaderboard/metric for this dataset. Given severe class
imbalance (fraud is a small minority of transactions), prefer **PR-AUC**
(average precision) or **F1 on the fraud class** over raw accuracy or ROC-AUC
alone.

## Approach Notes

- Fraud only occurs in `TRANSFER` and `CASH_OUT` types (per PaySim's
  documented fraud scenario: takeover → transfer → cash-out) — worth
  confirming empirically and possibly modeling those types separately.
- Heavy class imbalance expected — consider stratified sampling,
  class-weighting, or resampling (SMOTE etc.) rather than naive fit.
- `nameOrig`/`nameDest` are high-cardinality IDs — not directly usable as
  categorical features; could derive aggregate/behavioral features (e.g.
  transaction frequency, merchant vs. customer destination) instead.
- Balance-delta features (`oldbalance - newbalance` vs. `amount`) may reveal
  inconsistencies characteristic of fraud, but must be engineered carefully
  given the leakage caveat above.
- `isFlaggedFraud` is a rule-based flag, not a label to predict — useful at
  most as a feature or as a sanity baseline, not the target.

## Citation

E. A. Lopez-Rojas, A. Elmir, and S. Axelsson. "PaySim: A financial mobile
money simulator for fraud detection." *The 28th European Modeling and
Simulation Symposium-EMSS*, Larnaca, Cyprus. 2016.
