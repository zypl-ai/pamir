# Reference results for PaMIR 0.4.0 — analysis plan

This plan is fixed before any reference number is computed.  Every run listed
here is reported, whatever it shows; a run that fails is reported as failed.
Deviations from the plan are listed in the results file with their reason.

## Data

All datasets of the 0.4.0 catalogue, rebuilt from the pinned source snapshots
and accepted by the data contract (`strict=True`: exact columns, row and default
counts, raw-file and target digests).  A run on a table that fails its contract
is not a reference result.

## Models

| id | streaming (`fit_fn`) | i.i.d. (`predict_fn`) |
|---|---|---|
| `logistic` | `pamir.logistic_fit` | `pamir.logistic_baseline` |
| `gbdt` | `pamir.gbdt_fit` | `pamir.gbdt_baseline` |

Both are the package baselines with their default hyper-parameters; nothing is
tuned.  One tabular foundation model is added after its weights' licence has
been checked; its configuration (checkpoint, context limit, subsampling) is
fixed in an addendum to this plan **before** it is run.

## Protocols

| id | evaluator | parameters | role |
|---|---|---|---|
| `arrival` | `evaluate_one` | reference setting: `mode="arrival"`, `lag=1000`, `max_lag_frac=0.2`, `k_refit=10`, `min_defaults=3`, `max_n=20000`, batch scoring with probe | primary |
| `iid_20k` | `evaluate_iid_one` | `train_frac=0.7`, seeds 42–46, `max_n=20000` | primary comparison (same rows as `arrival`) |
| `iid_full` | `evaluate_iid_one` | `train_frac=0.7`, seeds 42–46, full table | comparison with general benchmarks |
| `refresh` | `evaluate_one` | `mode="refresh"`, otherwise as `arrival` | the 0.3.0 protocol semantics, for the protocol comparison |
| `arrival_lag250` | `evaluate_one` | as `arrival`, `lag=250` | delay sensitivity |

Each combination of model and protocol is one invocation of
`run_reference.py`; the command, the git commit and the library versions are
recorded in its `summary.json`.

## Quantities reported

Per dataset: AUC (and Gini), its standard deviation over seeds (i.i.d.), the
label-budget AUCs (`arrival`), `n_refits`, `n_refits_ok`, `n_calls`,
`n_failures`, `n_scored`, `lag_effective`, `contract_ok`, `row_independent`,
wall time.  Per run: `fleet_summary` (fleet mean only for a valid run).

## Comparisons, fixed in advance

1. **Headroom**: GBDT − logistic, per dataset and in fleet mean, under
   `arrival` and under `iid_20k`.
2. **Protocol gap**: `iid_20k` − `arrival`, per model, per dataset and in fleet
   mean.
3. **Ranking change**: the number of datasets on which the better of the two
   models differs between `iid_20k` and `arrival`.
4. **Label-budget curve**: fleet mean AUC per budget group under `arrival`, per
   model, with the number of datasets in each group.
5. **Delay**: `arrival` − `arrival_lag250`, per model, in fleet mean.
6. **Refresh versus arrival**: `refresh` − `arrival`, per model, in fleet mean.

For comparisons 1, 2, 5 and 6 a paired Wilcoxon signed-rank test over the
datasets is reported alongside the per-dataset differences, as a description of
consistency, without a correction for multiple comparisons and without a claim
of significance beyond it.  No dataset is excluded from any comparison after
the results are seen.

## Not in this plan

Hyper-parameter tuning, model selection on the reference results, synthetic
augmentation runs, and any evaluation of a generator or model developed at
zypl.ai.
