---
pretty_name: "PaMIR: Credit-Default Benchmark with Scarce and Delayed Labels"
language:
  - en
license: apache-2.0
task_categories:
  - tabular-classification
tags:
  - credit-risk
  - credit-scoring
  - benchmark
  - streaming
  - tabular
  - finance
  - default-prediction
  - label-delay
  - risk-management
  - pamir
size_categories:
  - 1M<n<10M
---

> **This page hosts no data.** PaMIR does not redistribute the datasets. It is a
> Python package (`pip install "pamir[data]"`) that fetches each dataset
> from its original source and harmonizes it locally. The `apache-2.0` tag
> covers the **package code only**; each dataset stays under its own license.

# PaMIR

**P**ublic **A**rrival-ordered **M**easurement for **I**nference in **R**isk

An open benchmark for **credit-default prediction when labels are scarce and
arrive late**: **19 public credit-default datasets** (1.24M loans, firms and
card accounts from nine countries, default rates 3%–41%), rebuilt from pinned
source snapshots by one leakage-audited recipe and never redistributed; to our
knowledge it is the one of its kind as of today.  Every model is
scored under **two evaluation protocols**: a label-delayed stream, in which each
application is scored on arrival by a model trained only on outcomes that have
matured, with AUC by label budget, and a repeated i.i.d. split for comparison
with other tabular benchmarks.  Fleet means are withheld unless every dataset
is scored.  A leakage-controlled synthetic-data harness tests generated
training rows.  This card hosts no data.

## Quick use

```python
# pip install "pamir[data]"
# Fetches gmsc from its original source and harmonizes it locally on first use.
from pamir import load_dataset, evaluate, evaluate_iid, gbdt_fit, gbdt_baseline
X, y, meta = load_dataset("gmsc")
results = evaluate(gbdt_fit)                          # streaming, reference setting

results_iid = evaluate_iid(gbdt_baseline, n_seeds=5)  # conventional i.i.d.
```

## What makes this benchmark different

General tabular benchmarks contain credit tasks but evaluate them on random,
temporal or grouped splits.  A credit model is trained on outcomes that mature
months or years after origination and starts with no labels; a temporal split
orders the data by time but can still train on outcomes that matured after its
cut-off.  PaMIR's streaming protocol scores each application on arrival, with a
model trained only on outcomes that had matured by then, and reports how the
AUC grows with the number of labels.  Rows are replayed in a fixed random order
(most sources carry no usable dates), so it measures learning under scarce,
delayed labels, not robustness to temporal shift.

| Benchmark | Datasets | Domain | Evaluation |
|---|---|---|---|
| OpenML-CC18 | 72 | General | i.i.d. cross-validation |
| TabArena | 51 | General | i.i.d. cross-validation |
| MultiTab | 196 | General | i.i.d. splits |
| TabReD | 8 | Industrial | time-based splits |
| BeyondArena | 142 | General | i.i.d., temporal and grouped splits |
| Lessmann et al. (2015) | 8 (4 public) | Credit | i.i.d. splits |
| **PaMIR** | **19** | **Credit default** | **i.i.d. and label-delayed streaming** |

## Datasets

| id | name | rows | DR | geography | product | license |
|---|---|---|---|---|---|---|
| bankruptcy | Taiwanese Bankruptcy Prediction | 6,819 | 3.2% | Taiwan | Corporate | Data files © Original Authors (Kaggle) |
| bondora | Bondora P2P Lending | 266,482 | 40.9% | Estonia, Finland, Spain | P2P consumer | CC0 1.0 (Kaggle reuploader) |
| conorsully | Conor Sully Credit Score | 1,000 | 28.4% | Synthetic / educational | Consumer | CC0 1.0 |
| dish | Automobile Loan Default | 121,856 | 8.1% | Unspecified | Vehicle finance | CC0 1.0 (Kaggle reuploader) |
| gastonstat | Gaston Sanchez Credit Scoring | 4,454 | 28.1% | Unknown | Consumer | No license (GitHub, no LICENSE file) |
| gmsc | Give Me Some Credit | 150,000 | 6.7% | USA | Consumer revolving + installment | Unknown (Kaggle reuploader) |
| laotse | Laotse Credit Risk | 32,581 | 21.8% | Unspecified | Consumer | CC0 1.0 (Kaggle reuploader) |
| lc_clean | Lending Club (cleaned, 2007–2014) | 150,000 | 20.2% | USA | P2P consumer | No license (HuggingFace repo, none stated) |
| lc_my | Lending Club (Malaysian variant) | 100,000 | 22.6% | Unspecified | Consumer | Unknown (Kaggle reuploader) |
| lc_small | Lending Club (small, 9,578 loans) | 9,578 | 16.0% | USA | Consumer | ODbL (Open Database License) |
| lt_vehicle | L&T Vehicle Loan Default | 233,154 | 21.7% | India | Vehicle finance | Other (specified in description) |
| pakdd | PAKDD 2010 Credit Data | 50,000 | 26.1% | Brazil | Consumer | No license (competition archive) |
| poland_1yr | Polish Companies Bankruptcy (1-year horizon) | 7,027 | 3.9% | Poland | Corporate | CC-BY-4.0 (UCI) |
| poland_3yr | Polish Companies Bankruptcy (3-year horizon) | 10,503 | 4.7% | Poland | Corporate | CC-BY-4.0 (UCI) |
| poland_5yr | Polish Companies Bankruptcy (5-year horizon) | 5,910 | 6.9% | Poland | Corporate | CC-BY-4.0 (UCI) |
| prosper | Prosper Marketplace Loans | 55,084 | 30.9% | USA | P2P consumer | CC0 1.0 (Kaggle reuploader) |
| sba | U.S. SBA Loan Defaults | 2,102 | 32.6% | USA | Small business | CC0 1.0 (Kaggle reuploader) |
| south_german | South German Credit (corrected) | 1,000 | 30.0% | Germany | Consumer | CC-BY-4.0 (UCI) |
| taiwan | Taiwan Credit Card Default | 30,000 | 22.1% | Taiwan | Credit card | CC0 1.0 (Kaggle “UCI ML” reuploader) |

**Total:** 1,237,550 rows, 281,766 defaults across 19 datasets.

## Streaming protocol (summary)

1. Applications arrive in stored order (a fixed random permutation).
2. The outcome of the application at position *t* is revealed at *t + lag*
   (default 1000, capped at 20% of the stream).
3. The model is refitted after every *k* newly resolved defaults (default 10).
4. Each application is scored once, on arrival, by the latest model; scores are final.
5. Metric: ROC AUC over all scored applications, and AUC by label budget;
   the fleet mean is withheld unless every dataset is scored.

Full protocol specification and evaluation code: `pip install pamir`
([paper](https://arxiv.org/abs/2610.03259), [documentation](https://pamir-docs.pages.dev), [code](https://github.com/zypl-ai/pamir)).

A repeated i.i.d. split protocol (`evaluate_iid`) is also
provided for comparison with other tabular benchmarks.

## Data provenance

PaMIR ships no data. `pamir.download(id)` fetches each dataset from its
original source and harmonizes it with a reproducible recipe: derive the binary
target, drop id/date/constant/surrogate columns, drop post-outcome columns by
the elicitation's semantic cut (`day_zero_available = false`) plus per-dataset
extra drops, and shuffle rows with a fixed seed. The source snapshot is pinned,
and the rebuilt table must match its data contract in `expected.json` exactly
(columns, counts, SHA-256 of the raw file and of the target vector) or it is
refused.
Per-dataset recipe: `pamir.dataset_info(id)` in the Python package.

## License

The PaMIR **package code** is Apache 2.0. PaMIR does **not** redistribute the
datasets — each is fetched from its original source at the user's request and
remains under its own license and terms (listed in the table above and in
`pamir.dataset_info(id)`). Users are responsible for complying with each
source's terms and citing its original authors.

## Citation

```bibtex
@misc{liashkov2026pamir,
  title         = {PaMIR: Open Benchmark of Public Credit-Default Datasets},
  author        = {Liashkov, Mikhail and Varshavskiy, Ilyas and Khalilbekov, Shuhratjon and
                   Azimi, Azizjon and Boboeva, Bonu},
  year          = {2026},
  eprint        = {2610.03259},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.03259},
}

@software{pamir2026,
  title   = {PaMIR: Public Arrival-ordered Measurement for Inference in Risk},
  author  = {Liashkov, Mikhail and Varshavskiy, Ilyas and Boboeva, Bonu and
             Khalilbekov, Shuhrat and Azimi, Azizjon},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/zypl-ai/pamir},
}
```
