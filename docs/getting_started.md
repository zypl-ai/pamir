# Getting started

## Installation

PaMIR is not on PyPI yet, so install from a clone:

```bash
git clone https://github.com/zypl-ai/pamir-credit
cd pamir-credit
pip install -e ".[data]"    # add [dev] for pytest, [synthetic] for pamir.synthetic
```

The `[data]` extra installs the fetch dependencies (`kagglehub`,
`huggingface-hub`, `platformdirs`). PaMIR **does not distribute the datasets
itself** — it ships the catalog, the harmonization recipe, and the code that
fetches each dataset from its original source and harmonizes it locally.

On macOS/arm64, import `xgboost` before anything that pulls in `torch` (`sdv`
does) — the reverse order segfaults.  `import pamir.synthetic` handles this
for you.

## Load a dataset

```python
from pamir import load_catalog, load_dataset

# Browse all 19 datasets (metadata only — no download)
catalog = load_catalog()
print(catalog[["name", "rows", "DR", "geography", "license"]])

# Load one — fetched from its original source and harmonized on first use,
# then cached locally (default: platformdirs user cache; override with $PAMIR_CACHE)
X, y, meta = load_dataset("taiwan")
print(f"{meta['name']}: {meta['n_rows']:,} rows, DR={meta['DR']:.1%}")
```

Most datasets are hosted on Kaggle, so fetching them needs Kaggle credentials
(`KAGGLE_USERNAME` + `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`). UCI, GitHub and
HuggingFace sources need no credentials. You can pre-fetch explicitly:

```python
import pamir
pamir.download("taiwan")          # fetch + harmonize + cache one dataset
```

### No Kaggle account? Start with the credential-free set

7 datasets come from UCI / GitHub / HuggingFace and need **no credentials** —
they work on a fresh install with zero setup:

```python
pamir.open_datasets()   # ['gastonstat', 'lc_clean', 'pakdd', 'poland_1yr', ...]
pamir.download_open()   # fetch all of them
```

### Command line

```bash
pamir list --open           # datasets that need no credentials
pamir info taiwan           # metadata + recipe for one dataset
pamir download poland_1yr   # fetch + harmonize into the cache
pamir cache                 # where the cache lives
```

## Define your model

A model comes in one of two forms, and the reference baselines ship in both:

- **`fit_fn(X_train, y_train) → scorer`**, with **`scorer(X_rows) → scores`** —
  the form the streaming protocol calls: a refit trains on the matured labels,
  and the scorer scores rows as they arrive;
- **`predict_fn(X_train, y_train, X_test) → scores`** — the form the i.i.d.
  protocol calls.

Scores have one entry per row and higher means more likely to default.  Most of
the datasets carry `object` or `bool` columns, so encode them rather than
dropping them, using the training rows' levels only:

```python
from pamir.baselines import apply_encoder, fit_encoder

def my_fit(X_train, y_train):
    from sklearn.ensemble import HistGradientBoostingClassifier
    levels = fit_encoder(X_train)                    # levels from the training rows
    clf = HistGradientBoostingClassifier(max_iter=150)
    clf.fit(apply_encoder(levels, X_train), y_train)
    return lambda X: clf.predict_proba(apply_encoder(levels, X))[:, 1]

def my_model(X_train, y_train, X_test):              # the same model as a predict_fn
    return my_fit(X_train, y_train)(X_test)
```

A scorer must score each row on its own: the streaming protocol checks this on
every batch it scores (see [Evaluation protocols](protocol.md)).

Two baselines ship ready-made if you only need a reference point:

```python
from pamir import logistic_fit, gbdt_fit                 # fit_fn form
from pamir import logistic_baseline, gbdt_baseline       # predict_fn form
```

## i.i.d. evaluation (conventional)

Standard random train/test split, repeated across multiple seeds:

```python
from pamir import evaluate_iid, evaluate_iid_one

# One dataset
result = evaluate_iid_one(my_model, "taiwan", n_seeds=5)
print(f"AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")

# Full fleet
results = evaluate_iid(my_model, n_seeds=5)
```

## Streaming evaluation (with label delay)

Applications arrive one at a time, each is scored once on arrival, outcomes
are revealed only after a maturation delay, and the model is refitted as
labels arrive:

```python
from pamir import evaluate, evaluate_one

# One dataset
result = evaluate_one(my_fit, "taiwan")
print(f"AUC: {result['auc_final']:.4f} ({result['n_refits']} refits, "
      f"lag {result['lag_effective']})")
print({k: v for k, v in result.items() if k.startswith("auc_budget_")})

# Full fleet
results = evaluate(my_fit)
```

## Compare both protocols

```python
from pamir import evaluate, evaluate_iid

results_iid = evaluate_iid(my_model, n_seeds=5, verbose=False)
results_stream = evaluate(my_fit, verbose=False)

import pandas as pd
comparison = results_iid[["dataset", "auc_mean"]].rename(columns={"auc_mean": "iid"})
comparison = comparison.merge(
    results_stream[["dataset", "auc_final"]].rename(columns={"auc_final": "streaming"}),
    on="dataset")
print(comparison.to_string(index=False))
```

By default the i.i.d. protocol uses the full table and the streaming protocol
the first 20,000 rows; pass the same `max_n` to both for a like-for-like
comparison.

## Check the summary before quoting a number

If the model raises, the affected rows go unscored.  Both protocols record
every failure, and `fleet_summary` withholds the headline mean unless the run
is valid:

```python
from pamir import evaluate, fleet_summary, gbdt_fit

results = evaluate(gbdt_fit)
summary = fleet_summary(results)

summary["complete"]         # False if anything failed
summary["contract_ok"]      # False if a table is not the benchmark table
summary["row_independent"]  # False if a scorer depended on its batch
summary["auc_mean"]         # None unless all three hold
summary["failed_datasets"]  # which ones; results["errors"] says why
```

An average over the datasets a model survived is selected by the model's own
failures, and can read *higher* than a working model's average over all of
them — which is why it is never reported as `auc_mean`.

While developing, pass `on_error="raise"` to get the traceback instead of a
warning.

## Report your protocol parameters

`mode`, `lag`, `max_lag_frac`, `k_refit` and `max_n` change the score, not
just the runtime.  The package defaults — `mode="arrival"`, `lag=1000`,
`max_lag_frac=0.2`, `k_refit=10`, `max_n=20000` — are the reference setting.
Report them with any number, and state any deviation.

Cost scales with the refit count: every refit is a fit plus the scoring of the
rows that arrived since the last one.  Raise `k_refit` (and say so) to trade
resolution for time.
