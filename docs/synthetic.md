# Synthetic augmentation

`pamir.synthetic` answers one question under the benchmark's own protocol: **does
training on synthetic data help, and how good is that synthetic data?**

It mixes real and synthetic rows in stated proportions, across several generators
at once, fits every generator inside the cross-validation fold, and reports both
the AUC delta and a full fidelity battery.

```python
from sdv.single_table import GaussianCopulaSynthesizer

from pamir import load_dataset
from pamir.synthetic import CrossValidatedAugmentation, SDVAdapter, SyntheticMixer

mixer = SyntheticMixer(
    generators={"copula": SDVAdapter(GaussianCopulaSynthesizer, name="copula")},
    synthetic_share=0.5,                   # 50% synthetic / 50% real
)

X, y, _ = load_dataset("south_german")
result = CrossValidatedAugmentation(mixer, n_splits=5, seed=42).run(X, y)

result.summary()             # baseline vs augmented OOF AUC, and the delta
result.leakage_frame()       # the probes that say whether to believe that delta
result.fidelity_headline()   # the few fidelity numbers worth reading first
```

`SDVAdapter` takes any SDV synthesizer and needs only the `synthetic` extra, so
it is the path to start on.  Several generators mix the same way — the synthetic
part is split between them by `weights`:

```python
from sdv.single_table import CTGANSynthesizer

mixer = SyntheticMixer(
    generators={"copula": SDVAdapter(GaussianCopulaSynthesizer, name="copula"),
                "ctgan": SDVAdapter(CTGANSynthesizer, name="ctgan")},
    weights={"copula": 0.5, "ctgan": 0.5},   # the synthetic part splits evenly
    synthetic_share=0.7,                     # 70% synthetic / 30% real
)
```

When neither `synthetic_share` nor `n_synthetic` is given, the share is 0.5.

The four result views:

| view | what it answers |
|---|---|
| `summary()` | baseline vs augmented OOF AUC, and the delta |
| `fold_frame()` | per-fold composition and AUC |
| `leakage_frame()` | did a synthetic row reproduce a held-out row? |
| `fidelity_headline()` | the shortlist of fidelity numbers |
| `fidelity_frame()` | every fidelity metric, per fold and generator (66 columns) |

Read `leakage_frame()` before anything else.  `synth_holdout_only_overlap`
above zero means a synthetic row exactly reproduced a held-out row that is not
in training, and the delta is void however good the rest of the battery looks.

## Proportions

Two independent knobs:

* `synthetic_share` — the share of the *mixed frame* that is synthetic.
* `weights` — how that synthetic share splits across generators.

`synthetic_share=0.7` with `weights={"zgan": 0.5, "zedge": 0.5}` gives a frame
that is 30% real, 35% zGAN and 35% zEDGE.  Weights need not sum to 1; they are
normalized.  Row counts are resolved by largest remainder, so the parts always
sum to the total exactly and a tie breaks deterministically by name.

`sizing` decides what the share is a share *of*:

| `sizing` | behaviour | use for |
|---|---|---|
| `keep_real` (default) | every real row is kept, synthetic rows are added on top until the share is met | "does adding synthetic data help?" |
| `fixed_total` | the frame is pinned to `n_total` rows and the real part is subsampled | "at equal training size, is synthetic data as good as real?" |

A generator whose plan comes to zero rows — weighted to `0`, or with
`n_synthetic=0` — is not fitted at all, since for a GAN or a diffusion model
that would be a full training run per fold thrown away.

Note that `allow_replacement` exists on two levels and they govern different
steps: `PoolAdapter(allow_replacement=...)` decides whether the pool may hand
back the same row twice, while `SyntheticMixer(allow_replacement=...)` decides
the same for the class-balance selection.  Setting the mixer's to `False` does
not stop a pool from repeating rows.

`synthetic_target_rate` forces the synthetic part to a chosen positive rate —
the usual reason to reach for a generator on a 3%-default book.  It needs
`oversample > 1` so there is slack to select from, except for a `PoolAdapter`,
where the whole pool is the candidate set and oversampling is skipped.

### A count instead of a share

`n_synthetic=` replaces `synthetic_share=` when an exact number of rows is
wanted.  The two are mutually exclusive — naming both raises rather than letting
one quietly lose.  The difference matters under cross-validation: a share
resolves against each fold's train split and therefore varies between folds,
while a count is the same in every fold.

```python
SyntheticMixer(generators=..., synthetic_share=0.7)   # 70% of each fold's mix
SyntheticMixer(generators=..., n_synthetic=5000)      # exactly 5000 rows per fold
```

With `sizing="fixed_total"` a count is checked against `n_total`, so
`n_total=10000, n_synthetic=6000` leaves 4000 real rows and raises if the fold
cannot supply them.  `plan.requested_share` and `plan.requested_n_synthetic`
record which of the two was asked for, and `composition` carries both.

## Fidelity

`fidelity_report` runs three layers and never lets one failing metric take the
report down — anything that raises lands in `report.errors`.

**SDV / SDMetrics.** `QualityReport` (Column Shapes, Column Pair Trends) and
`DiagnosticReport` (Data Validity, Data Structure); per column `KSComplement`,
`TVComplement`, `CSTest`, `BoundaryAdherence`, `CategoryAdherence`,
`CategoryCoverage`, `RangeCoverage`, `MissingValueSimilarity` and
`StatisticSimilarity` for mean, median and std; per pair
`CorrelationSimilarity` (Pearson and Spearman), `ContingencySimilarity`,
`ContinuousKLDivergence`, `DiscreteKLDivergence`; per table `NewRowSynthesis`,
`TableStructure`, `LogisticDetection`, and with `heavy=True` also `SVCDetection`
and `GMLogLikelihood`.  With a target column, the ML-efficacy arms
`BinaryLogisticRegression`, `BinaryAdaBoostClassifier`,
`BinaryDecisionTreeClassifier` (and `BinaryMLPClassifier` when heavy).

**Engine-native.** Whatever the generator's own library ships, through
`adapter.native_fidelity`.  `ZganAdapter` reports zGAN's own
`evaluate_ks_complement`, `evaluate_tv_complement`, `evaluate_tv_complement_pairs`
and `correlation_similarity` from `app/utils/zgan_utils.py`.  zEDGE ships no
fidelity metrics of its own — only Prometheus operational counters — so there is
nothing to collect from it.

**Dependency-free.** Normalized Wasserstein, total-variation distance and
Jensen–Shannon per column; correlation-matrix delta; **C2ST AUC** (how well a
gradient-boosted classifier separates synthetic from real — 0.5 is
indistinguishable, and the scale is two-sided: **below** 0.5 means synthetic
rows duplicate real ones, not that they are better than indistinguishable, so
rank on the reported `C2ST_deviation` = `|AUC - 0.5|` and read a sub-0.5 AUC
next to `dcr_zero_share`) and **distance to closest record** (mean, median, 5th
percentile and the share of exact matches).  These run with nothing installed
beyond numpy, pandas, scipy and scikit-learn, so a report is never empty.

DCR caps both sides at 2 000 rows for cost.  Thinning the real side can only
remove candidate neighbours, so on a larger table the distances are an upper
bound — memorisation looks milder than it is, never worse.  Raise `sample` when
DCR is being read as a privacy statement rather than a utility one.

## The leakage contract

A generator that has seen the test split inflates every number downstream.  Five
mechanisms hold the line:

1. **Split first, fit second.** `CrossValidatedAugmentation` slices the fold and
   hands `SyntheticMixer` the training rows only.  Held-out rows reach the
   fidelity report solely as a leakage probe, never as reference data.
2. **A fresh generator per fold.** The mixer is cloned and each adapter reset, so
   nothing learned on fold *k* survives into fold *k+1* where those rows are test.
3. **The test split is never augmented.** Synthetic rows enter training only.
4. **Pools and artifacts are fold-keyed.** A pre-generated pool or a trained
   zEDGE artifact is refused unless registered per fold (`fold_pools=`,
   `fold_artifacts=`) or the caller explicitly vouches for it with
   `allow_shared_pool` / `allow_shared_artifact`.  One pool generated from the
   whole table has seen every fold's test rows.
5. **Probes, not promises.** Every fold reports
   `leak.n_synth_matching_holdout_only` — synthetic rows that exactly match a
   held-out row which is *not* also in train.  Anything above zero means the
   contract broke upstream.  Read `result.leakage_frame()` before believing a delta.

   Rows are hashed after normalisation, because `hash_pandas_object` keys on
   dtype as well as value: `int64(1)`, `float64(1.0)`, `True` and `"1"` all hash
   differently.  The schema is taken from the **real** frame — a column is
   whatever the real table says it is — and all three frames are normalised
   under it, so a copy that comes back as floats, or as text from a CSV pool,
   is still recognised as a copy.  Without that step a generator reproducing
   held-out rows *verbatim* reads as perfectly clean.  Shares are counted over
   synthetic rows, not distinct hashes, so a generator that emits one memorised
   row a hundred times is charged for a hundred.

   The probes are **not** switched off with `fidelity=False`: they cost
   milliseconds against the battery's seconds, and a delta reported without
   them cannot be checked.  `leakage_probes=False` is the only thing that stops
   them, in either mode, and then `leakage_frame()` comes back **empty** —
   empty always means the probes did not run, never that they ran and found
   nothing.

Column typing (`prepare`) is stateless — numerics coerced, categoricals cast to
string — so applying it before the split leaks nothing.  Note that the cast runs
before any sentinel could be applied, so a missing categorical becomes the
literal `"nan"` or `"None"` rather than a single `__NA__` level; this is kept
deliberately, because relabelling categories would change the numbers between
releases.  In the model arm, categorical *levels* are unioned across train and
test purely so both frames share one encoding; no value, statistic or label from
the test rows enters the fit.

## Protocol

The default protocol: `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)`, XGBoost (`n_estimators=400, max_depth=6, learning_rate=0.08`),
out-of-fold predictions pooled into a single AUC.  Without `xgboost` installed
the arm falls back to scikit-learn's `HistGradientBoostingClassifier`
(300 iterations), which gives different numbers — state which arm and which seed
a run used.  `predict_fn` takes the same
`(X_train, y_train, X_test) -> scores` signature as `pamir.evaluate`, so any
model can be dropped in.

`y` must be coded 0/1 and is checked before the first fold, so a `{1, 2}` or
`{-1, 1}` encoding is reported as what it is rather than surfacing later as a
complaint about the mixed frame.  The runner owns the target column name, so a
mixer naming a different one is refused instead of being silently overridden.

## Generators

| Adapter | Wraps | Notes |
|---|---|---|
| `ZganAdapter` | `zyplGANSynthesizer` from the zgan repo | It is an SDV `BaseSingleTableSynthesizer`, so metadata is detected per fold |
| `ZedgeAdapter` | `zgn_latdiff` latent diffusion | `mode="pipeline"` trains per fold (needs torch, realistically a GPU); `mode="artifact"` samples from an artifact already trained on that fold |
| `SDVAdapter` | any SDV synthesizer | `GaussianCopulaSynthesizer`, `CTGANSynthesizer`, … |
| `PoolAdapter` | a CSV/parquet/DataFrame of rows generated elsewhere | The offline path; use `fold_pools=` |
| `CallableAdapter` | anything with `fit`/`sample` | The escape hatch |

`as_adapter` picks the right wrapper, so `generators={"zgan": SomeSynthesizer}`
works without naming an adapter at all.

### Passing the engine's own parameters

Extra keyword arguments reach the engine's constructor:
`ZganAdapter(repo_path=..., epochs=300)`, `SDVAdapter(CTGANSynthesizer,
batch_size=500)`.  `ZedgeAdapter` forwards its extras to `run_full_pipeline`
instead, and `CallableAdapter` takes three dictionaries, each with exactly one
destination: `init_kwargs` to the model's constructor, `fit_kwargs` to
`fit(frame, ...)`, `sample_kwargs` to `sample(n, ...)`.  The destination does
not depend on whether a class or an instance was passed, nor on how many times
the adapter has been fitted.

Two rules keep a configuration from evaporating unnoticed:

* **An engine that rejects a parameter raises.**  The adapter does not retry
  with an empty parameter set, because a whole benchmark run on defaults nobody
  chose is worse than a failure at construction.  The only key it may withdraw
  on its own is the `target` extra it injected itself, which credit-specific
  engines such as zGAN accept and generic SDV synthesizers do not.
* **Constructor parameters given alongside an already-built instance raise.**
  They could not be applied — the engine exists already — so pass the class and
  let the adapter build it, or configure the instance before handing it over.
  The same holds for `CallableAdapter`'s `init_kwargs`.

Refitting an adapter, or calling `mixer.build()` twice, rebuilds the engine from
the prototype, so a second run never inherits the first one's state.

## Installing the optional parts

```bash
pip install "pamir-credit[synthetic]"   # sdv, sdmetrics, xgboost
```

Without them the mixer still runs and the fidelity report falls back to its
dependency-free layer; the SDV metrics are then listed in `report.errors` as
not installed.  The leakage probes need nothing optional at all, so the check
that validates a delta is always available.

Paths accept a tilde: `repo_path="~/path/to/repo"`, `fold_artifacts`,
`artifact_dir`, `work_dir` and pool paths are all expanded before use.
