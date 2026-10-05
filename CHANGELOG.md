# Changelog

Versions were not published to a package index before 0.4.0; the dates below
are those of the commits on `dev` that completed each version.

## 0.4.0 (unreleased)

Breaking changes to the streaming protocol, the data contract and the
baselines; 0.3.0 numbers are reproducible with the options noted below.

- **Streaming protocol: arrival mode (new reference).** `evaluate` /
  `evaluate_one` take `mode="arrival"` by default.  The model is a
  `fit_fn(X_train, y_train) -> scorer`; every row is scored once, on arrival, by
  the scorer that was current then, which was trained only on labels that had
  matured.  In 0.3.0 every refit re-scored the whole remaining stream, so the
  model saw the features of rows that had not yet arrived and the delay bound
  mainly for the last `lag` rows; that behaviour is kept as `mode="refresh"`.
  Scorers are called on batches of arrived rows, and a probe re-scores a random
  subset of every batch on its own: a scorer whose output depends on the rest
  of its batch is flagged `row_independent=False` and the run gets no fleet
  mean (`batch_scoring=False` scores one row at a time).  A three-argument
  `predict_fn` is accepted and wrapped (`pamir.from_predict_fn`).
- **Delay cap.** The delay applied is `min(lag, floor(max_lag_frac * n))`,
  `max_lag_frac=0.2` by default, reported as `lag_effective`.  At `lag=1000`
  the two 1,000-row datasets previously resolved no label, so the reference
  setting could never produce a fleet mean.
- **Label-budget curve.** Arrival mode reports the AUC of the rows scored by
  models trained on [0, 100), [100, 300), …, ≥ 10,000 labels (`auc_budget_*`);
  `fleet_summary(...)["auc_by_budget"]` averages each group over the datasets
  that reach it.
- **Data contract (`pamir.contract`).** A rebuilt table must match
  `expected.json` exactly (feature columns, row and default counts and, once
  recorded, SHA-256 of the raw file and of the target vector in stored order).
  `download(..., strict=True)` raises `pamir.ContractError` and caches nothing
  on a mismatch; in 0.3.0 the mismatch was only printed (and not at all with
  `quiet=True`) and the table was cached.  `load_dataset` re-checks cached
  tables on every load, so caches built by an older release or from a drifted
  snapshot are caught.  `strict=False` keeps such a table flagged
  `contract_ok=False`; results carry `contract_ok`, and `fleet_summary`
  withholds the fleet mean unless it holds everywhere.  Cached parquet files
  carry a provenance record.  CLI: `pamir download ... --no-strict`.
- **Pinned sources.** Download specs take a Kaggle `version` and a Hugging Face
  `revision`; the catalog's values are recorded with this release.
- **Baselines.** `logistic_fit` / `gbdt_fit` (new, `fit_fn` form) and
  `logistic_baseline` / `gbdt_baseline` (same models, `predict_fn` form) encode
  non-numeric columns against the training rows' levels only
  (`pamir.baselines.fit_encoder` / `apply_encoder`; unseen level `-2`, missing
  `-1`).  The 0.3.0 baselines, which shared levels between train and test, are
  `logistic_baseline_v03` / `gbdt_baseline_v03`.  To reproduce a 0.3.0 number:
  `evaluate(logistic_baseline_v03, mode="refresh", max_lag_frac=None, ...)`.
- **Result fields.** Streaming results add `mode`, `lag_effective`, `k_refit`,
  `n_refits_ok` (refits whose fit succeeded; `n_refits` counts attempts),
  `n_scored`, `contract_ok`, `row_independent`; refit points add `ok`.
- **Catalog.** `fetched_from` is derived from the download spec; `source` names
  the origin.  Corrected: `source` of `conorsully` (the GitHub repository it
  named does not exist), `sba` and `lc_clean`; `source_url` of `conorsully` and
  `lc_small` (the latter pointed to a different dataset); the name of
  `lc_small`; `notes` of `bondora` (31 columns removed by the day-zero cut, not
  23; `LossGivenDefault` is kept), `prosper` (11, not 10) and `sba`.
- **Tests.** Positive controls for the three guards that had none (univariate
  separability, row order, day-zero); the dataset count is read from the
  catalog; streaming invariants on synthetic streams, including that the
  refresh mode reproduces 0.3.0; contract tests; a release-consistency test
  (`pyproject.toml`, `CITATION.cff`, `docs/conf.py`, `CHANGELOG.md`).
- **Docs.** `protocol.md` rewritten for both modes (it described a
  "pre-commitment" that 0.3.0 did not implement, and omitted the first-refit
  rule); `contributing.md` lists every contract field, the pinning, the
  changelog step and how the guards run for Kaggle-hosted additions; README and
  dataset card no longer claim to reflect production conditions, compare
  against BeyondArena and TabReD, drop unsourced credit-dataset counts, add a
  conflict-of-interest note and the author list; private paths and internal
  references removed from the synthetic-module docs.

## 0.3.0 (2026-09-24)

- **Failure accounting in the evaluation protocols.** `predict_fn` failures
  were swallowed by a bare `except` and the fleet mean used `.dropna()`, so a
  model that crashed on the hard datasets could report a *higher* fleet AUC
  than one that worked everywhere. `pamir.failures` (`FailureLog`; `on_error`
  = `warn`/`raise`/`ignore`) counts every call, and `pamir.fleet_summary`
  withholds `auc_mean` unless coverage is complete (the partial average is
  `auc_mean_scored_only`). Result frames gain `n_calls`/`n_failures` and drop
  the nested `refit_points` column that blocked `to_csv`. The i.i.d. protocol
  moves to `pamir.evaluate_iid`.
- **Reference baselines (`pamir.baselines`).** `encode_features`,
  `logistic_baseline` and `gbdt_baseline` (scikit-learn, no optional
  dependency) run on all 19 datasets unmodified.
- **`pamir.synthetic` fixes.** Import `xgboost` before `sdv`/`torch` to avoid a
  macOS/arm64 OpenMP segfault; `HEADLINE_FIDELITY_COLUMNS` +
  `CVResult.fidelity_headline()` for the 66-column fidelity frame; examples now
  lead with `SDVAdapter` (needs only the `synthetic` extra) rather than the
  private zGAN/zEDGE repos.
- Docs: reference-protocol sensitivity (`lag`/`k_refit`/`max_n`),
  install-from-clone, a macOS import-order note, the Apache 2.0 `LICENSE`
  file, dtype/`notes` corrections, and a runnable
  `examples/pamir_walkthrough.ipynb`.

## 0.2.0 (2026-09-18)

- **Synthetic augmentation (`pamir.synthetic`).** A leakage-controlled layer
  that mixes real and synthetic training rows in stated proportions across
  several generators at once, fits every generator **inside** the
  cross-validation fold (the test split is never augmented), and reports both
  the AUC delta and a full fidelity battery. Exact-overlap leakage probes run
  even with `fidelity=False` and hash rows under the real table's schema, so a
  reproduced hold-out row cannot slip through. Generator adapters cover the SDV
  engines plus the zGAN / zEDGE repositories; the mixer still runs on its
  dependency-free metrics when the optional `synthetic` extra is absent.
  Install with `pip install "pamir[synthetic]"`. See `docs/synthetic.md`
  (guide) and `docs/synthetic_design.md` (design record).
- Docs render the design-record diagrams via `sphinxcontrib.mermaid`.

## 0.1.0 (2026-09-07)

First public-ready cut of PaMIR — a benchmark of 19 public credit-default
datasets with a streaming (label-delay) protocol and a conventional i.i.d.
protocol.

- **No data shipped.** The package contains only the catalog, the harmonization
  recipe, and the code; `pamir.download(id)` fetches each dataset from its
  original source and harmonizes it locally, validated against `expected.json`.
- **19 datasets** across 9 named countries (Poland, Taiwan, Brazil, Estonia,
  Finland, Spain, Germany, India, USA) — a mix of clean-licensed (CC-BY / CC0)
  and source-fetched competition datasets. See `docs/licenses.md`. Seven
  candidates were rejected during the public audit rather than patched:
  `uz_fintech` and `sba_foia` (label recoverable from a feature), `hmeq`
  (rule-generated label), `heloc`, `thomas` and `home_credit` (redistribution
  or competition-rules restrictions), and `mortgage` (columns independently
  shuffled at source — no signal survives).
- **Credential-free starter set**: `pamir.download_open()` / `pamir list --open`
  fetch the UCI / GitHub / HuggingFace datasets with no Kaggle account.
- **CLI**: `pamir list|info|download|cache`.
- Leakage-free harmonization: semantic day-zero cut, surrogate-key/id drops
  (by content, not name), fixed-seed shuffle, plus CI guards from the public
  audit — 1-D AUC, row-order, divisibility, day-zero, missingness-indicator,
  and a raw-file scan for rule-defined labels that runs before the drops.
  Each guard has a positive control asserting it fires on a reconstruction of
  the leak it was written for, so a broken check cannot pass as clean data.
- Docs (Sphinx + `llms.txt`), per-dataset citations, and packaging metadata.
