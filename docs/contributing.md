# Contributing

## Adding a dataset

PaMIR ships **no data** — a dataset is added as a **recipe** that fetches it
from its original source and harmonizes it locally. You never commit a data
file.

1. Check it meets the [inclusion criteria](datasets.md): binary default target,
   publicly downloadable, ≥1,000 rows, ≥3% default rate, and either a flat table
   or one the recipe can reduce to a flat table (a documented join/aggregation
   is fine — see `bondora`).
2. Add an entry to `pamir/data/catalog.json` with the metadata (`name`, `rows`,
   `defaults`, `DR`, `features`, `geography`, `product`, `target_definition`,
   `license`, `source`, `source_url`, and — where the license or academic
   origin requires it — `attribution`), plus:
   - **`download`**: `{kind, locator, file, sep, encoding, …}` — where the raw
     file comes from. `kind` is one of:
     `kaggle` (slug), `uci_zip` / `github_zip` (a zip URL; a member is
     extracted, one level of nesting supported), `github_raw` (a direct
     file URL), or `hf` (a HuggingFace dataset repo). For a headerless raw file
     add `"header": "none"` + `"columns": [...]`; for ARFF add `"format":
     "arff"`. **Pin the snapshot**: a `"version"` for Kaggle, a `"revision"`
     (commit hash) for Hugging Face, a commit hash instead of a branch name in a
     GitHub URL.  `source` names where the data originate; what the
     downloader fetches is derived from `download` (`fetched_from`).
   - **`harmonize`**: `{target, target_rule, day_zero_drop, extra_drop,
     drop_prefix, shuffle_seed, …}`. `day_zero_drop` is the list of post-outcome
     columns (the leakage-free semantic cut); `extra_drop` / `drop_prefix`
     remove further leaks or identifiers; `shuffle_seed` fixes the row order.
3. If it needs a bespoke read, target rule, or rescale, add the branch in
   `pamir/harmonize.py` and reference it via `target_rule` (see `bondora`,
   `prosper`, `south_german`).
4. Run `pamir.download("<id>", strict=False)`, confirm it harmonizes cleanly,
   then freeze the result into `pamir/data/expected.json`: `columns` (name →
   type), `n_cat`, `n_features`, `n_rows`, `n_defaults`, `DR`, `raw_sha256`
   (digest of the raw file) and `target_sha256` (digest of the target vector
   in stored order; `pamir.contract.target_sha256`).  This is the data
   contract: every later download and every load is checked against it
   exactly, and a mismatch is an error.
5. Cite the source: put the paper / repository that introduced or used the
   dataset in the catalog `attribution` field and in
   [Licenses & attribution](licenses.md).
6. `pamir/data/*.parquet` is git-ignored — never commit the dataset itself.
7. Run the full test suite **with the new dataset cached** and paste the output
   into the merge request.  Continuous integration downloads only the
   credential-free datasets, so for a Kaggle-hosted dataset the leakage guards
   run on your machine and nowhere else.
8. Add an entry to `CHANGELOG.md` under the next version (dataset added,
   removed, or recipe changed); maintainers bump the version and
   `CITATION.cff` at release.
9. Open a merge request.

## Running tests

```bash
pip install -e ".[dev,data]"
pytest tests/ -v
```

Tests that need data skip cleanly when a dataset is not cached (e.g. CI without
Kaggle credentials); catalog, recipe and protocol tests always run. The suite
checks:

- **Catalog & recipe integrity** — the dataset list matches `catalog.json`,
  all metadata fields, every dataset has a valid `download`/`harmonize` spec,
  `expected.json` agrees with the catalog, counts in `notes` match the drop
  lists, and the version agrees across `pyproject.toml`, `CITATION.cff`,
  `docs/conf.py` and `CHANGELOG.md`.
- **Data contract** — exact validation, strict refusal on download and on load,
  provenance record in the cached parquet.
- **Harmonizer/downloader internals** — synthetic end-to-end harmonize, cache
  location, pinned Kaggle handles.
- **Leakage guards** (from the audit) — no single-column AUC outside
  (0.05, 0.95), rows shuffled (|ρ(position, target)| < 0.1), no `day_zero_drop`
  column survives, no surrogate-key column, no divisibility leak, no
  missingness leak, no pure level, raw-file scan; each guard has a positive
  control that reconstructs its leak and asserts that the guard fires.
- **Streaming protocol** (synthetic streams, no data needed) — in arrival mode
  a scorer sees only rows that arrived after its refit and labels that had
  matured, every score is final, the delay cap, the row-independence probe,
  failure accounting; the refresh mode reproduces 0.3.0.
- **Data quality** — no duplicate rows, no constant features, licenses and
  attribution present where required.

All tests must pass before a PR is merged.
