# Evaluation protocols

PaMIR evaluates every model on the same datasets under two protocols: a
conventional i.i.d. split, and a label-delayed streaming protocol.  A model is
written once, in one of two forms:

- **`predict_fn(X_train, y_train, X_test) → scores`** — fit and score in one
  call.  Used by the i.i.d. protocol and by the streaming protocol's refresh
  mode.
- **`fit_fn(X_train, y_train) → scorer`**, with **`scorer(X_rows) → scores`** —
  fit once, then score rows as they arrive.  Used by the streaming protocol's
  arrival mode, the reference protocol since 0.4.0.  A `predict_fn` is accepted
  there too and wrapped (`pamir.from_predict_fn`), at the cost of one fit per
  scoring batch.

Higher scores must mean a higher probability of default.  The reference
baselines exist in both forms: `logistic_fit` / `gbdt_fit` and
`logistic_baseline` / `gbdt_baseline` (the same models).

## i.i.d. protocol

The conventional train/test split used by OpenML-CC18, TabArena and most
tabular benchmarks, included for comparison with their published results.

```python
from pamir import evaluate_iid, gbdt_baseline
results = evaluate_iid(gbdt_baseline, n_seeds=5)
```

1. Permute the dataset with a seed (42, 43, …).
2. The first `train_frac` (default 70%) of rows train the model; the rest are
   scored in one `predict_fn` call.
3. Compute ROC AUC on the scored rows.
4. Repeat for `n_seeds` (default 5) permutations; report mean ± std.

| Parameter | Default | Description |
|---|---|---|
| `train_frac` | 0.7 | Fraction of rows used for training |
| `n_seeds` | 5 | Number of random permutations |
| `max_n` | None | Truncate the dataset before splitting |

## Streaming protocol

Applications arrive one at a time, outcomes are revealed only after a
maturation delay, and the model is refitted as resolved defaults accumulate.

```python
from pamir import evaluate, fleet_summary, gbdt_fit
results = evaluate(gbdt_fit)        # arrival mode, reference setting
fleet_summary(results)
```

### What it models, and what it does not

Credit models face three constraints that an i.i.d. split does not capture:

1. **No labels at the start.**  A new portfolio has no resolved outcomes on day
   one; the first scores come from a model trained on a handful of labels.
2. **Label-maturation delay.**  An outcome is known only `lag` positions after
   the application; at any moment the most recent `lag` applications are
   unlabelled.
3. **A score given at application is final.**

The rows are replayed in their stored order, which is a fixed random
permutation of the source: most sources carry no usable origination dates, and
file order is an export artefact.  The stream is therefore exchangeable — there
is no calendar drift — and the delay is measured in stream positions, not
months.  The protocol measures how well a model learns from a growing,
delayed, refit-limited label set; it does not measure robustness to temporal
shift.

### Arrival mode (reference)

```
scorer = None; pending = None
for t in 0, 1, ..., N-1:
    # row t arrives; it will be scored by the current scorer
    resolved_end = max(0, t + 1 - L)                 # rows [0, resolved_end) are labelled
    if t >= L and y[t - L] == 1: count a newly resolved default

    first   = (resolved defaults == min_defaults) and no refit attempted yet
    regular = (resolved defaults >= min_defaults) and (new defaults since last refit >= k_refit)
    if (first or regular) and resolved_end > 0 and t + 1 < N \
            and resolved non-defaults >= min_defaults:
        score rows [pending, t] with scorer           # the rows that arrived under it
        scorer  = fit_fn(X[0:resolved_end], y[0:resolved_end])   # kept if the fit raises
        pending = t + 1
score rows [pending, N) with scorer
AUC over every row that received a score
```

Every row is scored exactly once, by the scorer that was current when it
arrived, so a score is final; and that scorer was trained only on labels that
had matured by then.  Rows that arrive before the first refit are not scored
(the cold start) and are not in the AUC.

For speed, the rows that arrived under one scorer are passed to it as one
batch, after the fact.  That is equivalent to scoring each row on arrival only
if the scorer scores rows independently, so the harness checks: in every batch
it re-scores a random subset of up to 64 rows on its own and compares.  A
scorer whose output depends on the other rows of its batch — for example one
that encodes categories against the batch's own levels — is flagged
`row_independent = False`, and the run gets no fleet mean.
`batch_scoring=False` calls the scorer one row at a time instead.

### Refresh mode (0.3.0)

`mode="refresh"` is the 0.3.0 protocol, kept so that 0.3.0 numbers can be
reproduced.  At each refit the `predict_fn` is called with the resolved prefix
and the **whole remaining stream** as `X_test`, and those scores overwrite any
earlier ones; a row is evaluated on the last score committed before its label
resolves.  The model therefore sees, at every refit, the features of rows that
have not yet arrived, and the final score of most rows comes from a model
trained on almost every earlier row, so the delay binds mainly for the last
`L` rows.  0.3.0 numbers are reproduced by
`evaluate(..., mode="refresh", max_lag_frac=None)` with
`logistic_baseline_v03` / `gbdt_baseline_v03`.

### Delay cap

A delay as long as the stream leaves nothing to learn from: at `lag=1000`, a
1,000-row dataset never resolves a label.  The delay actually applied is
`min(lag, floor(max_lag_frac · N))`, with `max_lag_frac = 0.2` by default, and
it is reported per dataset as `lag_effective`.  At the reference setting it
binds for the four datasets with fewer than 5,000 rows.

### Streaming parameters

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"arrival"` | `"arrival"` (reference) or `"refresh"` (0.3.0) |
| `lag` | 1000 | Label-maturation delay in stream positions, before the cap |
| `max_lag_frac` | 0.2 | Delay cap as a fraction of the stream; `None` disables it |
| `k_refit` | 10 | Refit after every *k* newly resolved defaults |
| `min_defaults` | 3 | Resolved defaults (and non-defaults) before the first refit |
| `max_n` | 20000 | Truncate each stream to its first *N* rows |
| `batch_scoring` | True | Arrival mode: score arrived rows in batches (probed) |
| `probe` | True | Arrival mode: check row independence on every batch |
| `on_error` | `"warn"` | What to do when the model raises |
| `strict` | True | Refuse tables that do not match their data contract |

### These defaults are the reference setting

`mode`, `lag`, `max_lag_frac`, `k_refit` and `max_n` change the **score**, not
only the runtime: a smaller `k_refit` means more refits and fresher scorers.
Two runs are comparable only when all of them match.  Report them alongside any
number, and state any deviation.

Cost: in arrival mode a refit is one fit plus the scoring of the rows that
arrived since the previous refit; in refresh mode it is one fit plus a scoring
pass over the whole remaining stream.

### What the model must not do

- **Use labels that have not matured.**  Enforced structurally: the training
  rows are the resolved prefix.
- **Use rows that have not arrived.**  Enforced structurally in arrival mode;
  refresh mode hands them over, which is why it is not the reference.
- **Let a row's score depend on other rows being scored.**  Checked by the
  row-independence probe in arrival mode.
- **Compute target statistics on scoring rows.**  The scoring rows come without
  labels; nothing else may be inferred from them.

## Metric

Both protocols report **ROC AUC**; the Gini coefficient (2·AUC − 1) is
reported alongside it.

- **i.i.d.**: AUC on the held-out rows, averaged over seeds.
- **Streaming**: `auc_final`, the AUC over every row that received a score.
  Arrival mode adds the **label-budget curve**: rows are grouped by the number
  of resolved labels their scorer was trained on — [0, 100), [100, 300),
  [300, 1,000), [1,000, 3,000), [3,000, 10,000), ≥ 10,000 — and each group gets
  its own AUC (`auc_budget_0`, `auc_budget_100`, …).  `refit_points` records
  the cumulative AUC over resolved rows at every refit.

The fleet metric is the **unweighted mean across all datasets**, and it is
defined only for a valid run (below).  `fleet_summary(...)["auc_by_budget"]`
gives, per budget group, the mean over the datasets that reach it and their
number; groups contain different datasets, so the curve is descriptive.

### Coverage, contract and row independence are part of the result

If the model raises, the affected rows go unscored and the dataset may yield no
AUC.  Averaging over what is left would report a number the model selected by
failing: a model that crashes on the hard datasets can read *higher* than one
that scores them all.  `pamir.fleet_summary` therefore reports `auc_mean` only
when

- every dataset was scored (`complete`),
- every table matched its data contract (`contract_ok`; see
  [Datasets](datasets.md)), and
- in arrival mode, every scorer scored rows independently (`row_independent`).

The partial average is exposed as `auc_mean_scored_only`, a diagnostic that is
not comparable across models.  Every result frame carries `n_calls`,
`n_failures`, `n_refits` and `n_refits_ok`; `evaluate_one` also returns the
distinct error messages under `errors`.

A submission must report full coverage, or it is not a fleet result.
