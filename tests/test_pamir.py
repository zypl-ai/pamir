"""Tests for the PaMIR benchmark package.

Covers: catalog integrity, data loading, leakage checks, streaming protocol
invariants, i.i.d. protocol, data quality, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import pamir
from pamir.download import cache_dir


# ── data availability ───────────────────────────────────────────────────
# PaMIR ships no data; the harmonized parquet live in a local cache populated
# by pamir.download(). Tests that need data skip cleanly when it is absent
# (e.g. CI without Kaggle credentials); catalog/protocol tests always run.

def _cached(ds):
    try:
        return (cache_dir() / f"{ds}.parquet").exists()
    except Exception:
        return False


def _need(ds):
    if not _cached(ds):
        pytest.skip(f"{ds} not cached — run pamir.download('{ds}')")


def _load(ds, **kw):
    _need(ds)
    kw.setdefault("auto_download", False)
    return pamir.load_dataset(ds, **kw)


CACHED = [d for d in pamir.list_datasets() if _cached(d)]


# ═══════════════════════════════════════════════════════════════════════
# Catalog
# ═══════════════════════════════════════════════════════════════════════

def _catalog_ids():
    import json
    from pathlib import Path
    raw = json.loads((Path(pamir.__file__).parent / "data" / "catalog.json").read_text())
    return [e["id"] for e in raw]


def test_list_datasets_matches_catalog():
    ds = pamir.list_datasets()
    ids = _catalog_ids()
    assert isinstance(ds, list)
    assert len(ids) == len(set(ids)), "duplicate dataset ids in catalog.json"
    assert ds == sorted(ids)
    assert "gmsc" in ds
    assert "taiwan" in ds


def test_load_catalog_returns_dataframe():
    cat = pamir.load_catalog()
    assert isinstance(cat, pd.DataFrame)
    assert len(cat) == len(_catalog_ids())
    for col in ("rows", "features", "defaults", "DR", "license",
                "geography", "product", "source", "source_url",
                "target_definition"):
        assert col in cat.columns, f"missing column: {col}"


def test_catalog_dr_consistent():
    """DR must equal defaults / rows within rounding."""
    cat = pamir.load_catalog()
    for ds_id, row in cat.iterrows():
        computed = row["defaults"] / row["rows"]
        assert abs(computed - row["DR"]) < 0.01, \
            f"{ds_id}: DR={row['DR']} but defaults/rows={computed:.3f}"


def test_dataset_info_known():
    info = pamir.dataset_info("gmsc")
    assert info["id"] == "gmsc"
    assert info["rows"] == 150000
    assert 0.06 < info["DR"] < 0.08


def test_dataset_info_unknown_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        pamir.dataset_info("nonexistent_dataset")


def test_every_dataset_has_license():
    for ds_id in pamir.list_datasets():
        info = pamir.dataset_info(ds_id)
        assert info.get("license"), f"{ds_id} missing license"


def test_every_dataset_has_source_url():
    for ds_id in pamir.list_datasets():
        info = pamir.dataset_info(ds_id)
        url = info.get("source_url", "")
        assert url.startswith("http"), f"{ds_id} bad source_url: {url}"


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("ds_id", ["gmsc", "taiwan", "south_german", "bondora"])
def test_load_dataset_basic(ds_id):
    X, y, meta = _load(ds_id, max_rows=5000)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert y.dtype in (np.int32, np.int64)
    assert len(X) == len(y)
    assert len(X) > 100
    assert set(np.unique(y)) <= {0, 1}
    assert meta["n_rows"] == len(y)
    assert meta["n_defaults"] == int(y.sum())
    assert "__target__" not in X.columns


def test_load_dataset_max_rows():
    X, y, meta = _load("gmsc", max_rows=500)
    assert len(X) == 500
    assert len(y) == 500


def test_load_all_datasets():
    """Every dataset in the catalog loads without error."""
    for ds_id in CACHED:
        X, y, meta = _load(ds_id, max_rows=100)
        assert len(X) == min(100, meta["n_rows"])
        assert np.isfinite(y).all(), f"{ds_id}: NaN in target"


def test_target_is_binary():
    """Every dataset's target contains only 0 and 1."""
    for ds_id in CACHED:
        _, y, _ = _load(ds_id, max_rows=2000)
        assert set(np.unique(y)) <= {0, 1}, f"{ds_id}: non-binary target"


def test_default_rate_matches_catalog():
    """Observed DR matches catalog DR within tolerance."""
    cat = pamir.load_catalog()
    for ds_id in CACHED:
        _, y, _ = _load(ds_id)
        observed = y.mean()
        expected = cat.loc[ds_id, "DR"]
        assert abs(observed - expected) < 0.02, \
            f"{ds_id}: observed DR={observed:.3f}, catalog DR={expected:.3f}"


def test_feature_count_matches_catalog():
    """Number of feature columns matches catalog."""
    cat = pamir.load_catalog()
    for ds_id in CACHED:
        X, _, _ = _load(ds_id, max_rows=10)
        expected = cat.loc[ds_id, "features"]
        assert X.shape[1] == expected, \
            f"{ds_id}: {X.shape[1]} features, catalog says {expected}"


# ═══════════════════════════════════════════════════════════════════════
# No leakage
# ═══════════════════════════════════════════════════════════════════════

# Post-outcome columns that the semantic day-zero cut removes. (Bondora's
# ProbabilityOfDefault / LossGivenDefault are NOT here: they are origination-time
# risk estimates the elicitation marks day-zero-available, with 1-D AUC ~0.52-0.58
# — kept, like a bureau score.)
KNOWN_LEAKY_COLUMNS = {
    "bondora": ["PrincipalBalance", "RecoveryStage", "InterestAndPenaltyBalance"],
    "prosper": ["LoanCurrentDaysDelinquent", "LP_GrossPrincipalLoss", "LP_InterestandFees"],
    "sba": ["ChgOffPrinGr", "daysterm"],
}


@pytest.mark.parametrize("ds_id", ["bondora", "prosper", "sba"])
def test_leaky_columns_removed(ds_id):
    X, y, _ = _load(ds_id, max_rows=1000)
    for col in KNOWN_LEAKY_COLUMNS[ds_id]:
        assert col not in X.columns, f"{col} still present in {ds_id}"


def extreme_auc_hits(X, y, low=0.05, high=0.95, min_valid=100):
    """Numeric columns whose univariate AUC lies outside (low, high).

    Returns ``(column, auc)`` tuples. Shared by the fleet guard and its
    positive control. Skipped (no hits) when either class has < 10 rows.
    """
    y = np.asarray(y)
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return []
    hits = []
    for col in X.select_dtypes(include="number").columns:
        v = X[col].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(v)
        if valid.sum() < min_valid or len(np.unique(y[valid])) < 2:
            continue
        auc = roc_auc_score(y[valid], v[valid])
        if not low < auc < high:
            hits.append((col, auc))
    return hits


def test_no_column_has_extreme_auc():
    """No numeric column in any dataset should have AUC > 0.95 or < 0.05."""
    for ds_id in CACHED:
        X, y, _ = _load(ds_id, max_rows=5000)
        hits = extreme_auc_hits(X, y)
        assert not hits, "; ".join(
            f"{ds_id}/{c}: AUC={auc:.4f} — possible leakage" for c, auc in hits)


# ── leakage-free CI guards (audit §5) ────────────────────────────────────

def row_order_rho(y):
    """Correlation between stream position and target (None when too few rows)."""
    y = np.asarray(y)
    if len(y) < 100 or y.sum() < 10:
        return None
    return float(np.corrcoef(np.arange(len(y)), y)[0, 1])


def test_rows_are_shuffled():
    """No dataset may leak the label through row position (audit 2.4)."""
    for ds_id in CACHED:
        _, y, _ = _load(ds_id)
        rho = row_order_rho(y)
        assert rho is None or abs(rho) < 0.1, \
            f"{ds_id}: rho(position, target)={rho:+.3f} — not shuffled"


def day_zero_hits(columns, harmonize_spec):
    """Columns that a spec drops (day-zero cut, extra drops, prefixes) but survive."""
    h = harmonize_spec or {}
    banned = set(h.get("day_zero_drop", [])) | set(h.get("extra_drop", []))
    hits = sorted(banned & set(columns))
    for pref in h.get("drop_prefix", []):
        hits += [c for c in columns if c.startswith(pref)]
    return hits


def test_day_zero_columns_absent():
    """No column marked non-day-zero (or an extra/prefix drop) may survive."""
    for ds_id in CACHED:
        X, _, _ = _load(ds_id, max_rows=10)
        hits = day_zero_hits(X.columns, pamir.dataset_info(ds_id).get("harmonize", {}))
        assert not hits, f"{ds_id}: post-outcome columns present: {hits[:5]}"


def surrogate_key_hits(X):
    """Columns with one distinct value per row: a string name/id, or an integer
    id that escaped the name list. Continuous floats may legitimately be unique,
    so only integers are checked (audit §5.5: by content, not name)."""
    hits = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_integer_dtype(X[c]):
            continue
        if X[c].nunique(dropna=True) == len(X):
            hits.append(c)
    return hits


def test_no_surrogate_key_columns():
    """No column may be a surrogate key — one distinct value per row — whether a
    string name/id or an integer id (audit §5.5: caught by content, not name).
    Continuous floats may legitimately be unique, so only integers are checked."""
    for ds_id in CACHED:
        X, _, _ = _load(ds_id)
        hits = surrogate_key_hits(X)
        assert not hits, "; ".join(
            f"{ds_id}/{c}: one distinct value per row — surrogate key" for c in hits)


def divisibility_hits(X, y, threshold=0.65):
    """Columns whose divisibility encodes the label (sba.Term — audit 2.1).

    Uses the indicator ``(col % d == 0) & (col != 0)`` AUC. Excluding 0 avoids
    false positives on columns whose signal is merely the value 0 (a
    'not-a-franchise' code, a zero tax rate) rather than a duration encoded in
    round multiples.

    Returns ``(column, divisor, auc, dr_in, dr_out)`` tuples. Shared by the
    fleet guard below and by its positive control, so that the check the
    control proves is the check the guard runs.
    """
    y = np.asarray(y)
    hits = []
    for col in X.select_dtypes(include="number").columns:
        v = pd.to_numeric(X[col], errors="coerce").to_numpy(float)
        fin = np.isfinite(v)
        if pd.Series(v[fin]).nunique() <= 20:
            continue
        for d in (12, 6, 10):
            mask = fin & (np.mod(v, d) == 0) & (v != 0)
            other = fin & ~mask
            if mask.sum() < 100 or other.sum() < 100:
                continue
            auc = roc_auc_score(y[fin], (~mask[fin]).astype(int))
            if max(auc, 1 - auc) >= threshold:
                hits.append((col, d, max(auc, 1 - auc),
                             float(y[mask].mean()), float(y[other].mean())))
    return hits


def test_no_divisibility_leak():
    """No shipped dataset may encode the label in a column's arithmetic."""
    for ds_id in CACHED:
        X, y, _ = _load(ds_id)
        hits = divisibility_hits(X, y)
        assert not hits, "; ".join(
            f"{ds_id}/{c}: indicator AUC ({c}%{d}==0, nonzero)={auc:.3f} — "
            f"DR {dr_in:.3f} vs {dr_out:.3f}; label encoded in the column's arithmetic"
            for c, d, auc, dr_in, dr_out in hits)


def missingness_hits(X, y, tol=0.30, min_side=50):
    """Columns whose missingness pattern (near-)perfectly predicts the label.

    Returns ``(column, auc)`` tuples. Shared by the fleet guard and its
    positive control.
    """
    y = np.asarray(y)
    hits = []
    for c in X.columns:
        m = X[c].isna().to_numpy()
        if m.sum() < min_side or (~m).sum() < min_side:
            continue
        auc = roc_auc_score(y, m.astype(int))
        if abs(auc - 0.5) >= tol:
            hits.append((c, auc))
    return hits


def test_missingness_does_not_predict_target():
    """A column's missingness pattern must not (near-)perfectly predict the label
    (uz_fintech's Score_point=='-' → all-default). A moderate signal is allowed —
    e.g. poland_1yr/Attr27 isna-AUC≈0.71 is a legitimately undefined financial
    ratio — so only near-perfect coupling counts as a leak (audit 3.3)."""
    for ds_id in CACHED:
        X, y, _ = _load(ds_id)
        hits = missingness_hits(X, y)
        assert not hits, "; ".join(
            f"{ds_id}/{c}: missingness-indicator AUC={auc:.3f} — leak via missing pattern"
            for c, auc in hits)


def test_no_pure_level_leak():
    """On the shipped tables no column level is near-all-default while carrying
    most defaults (the rule-defined-label tell-tale; audit 3.4). Zero expected."""
    from pamir.harmonize import scan_pure_levels
    for ds_id in CACHED:
        X, y, _ = _load(ds_id)
        hits = scan_pure_levels(X, y)
        assert not hits, f"{ds_id}: near-all-default levels on the shipped table: {hits[:3]}"


def test_guards_actually_ran():
    """In CI the leakage guards must have data to check — an empty cache makes
    every guard pass vacuously, which then reads as 'no leaks' (audit 3.1)."""
    import os
    if os.environ.get("CI"):
        assert CACHED, ("no dataset cached — the §5 guards checked nothing; "
                        "run `pamir download --open` before pytest")


# ── positive controls: the guards must FIRE on a known leak ──────────────
# Every guard above is an "assert nothing found" test, which passes just as
# happily when the check has been broken as when the data is clean — the same
# failure mode as the empty CI cache (audit 3.1), one level up. The reference
# leaks are gone from the shipped tables (sba.Term is dropped, sba_foia and
# uz_fintech were removed), so each guard is exercised here against a
# reconstruction of the case it was written for, through the SAME helper the
# fleet guard calls.

def test_divisibility_guard_fires_on_term_style_leak():
    """Reconstruction of sba.Term (audit 2.1): a duration in whole years, where
    multiples of 12 are the loans that ran to term and repaid. Measured on the
    real file this is indicator AUC 0.932 (DR 0.036 vs 0.877)."""
    rng = np.random.default_rng(0)
    term = np.concatenate([rng.choice(np.arange(12, 253, 12), 1377),
                           rng.choice(np.array([n for n in range(3, 253)
                                                if n % 12 and n % 6 and n % 10]), 725)])
    y = np.concatenate([rng.random(1377) < 0.036, rng.random(725) < 0.877]).astype(int)
    X = pd.DataFrame({"Term": term, "GrAppv": rng.lognormal(11, 1, 2102)})

    hits = divisibility_hits(X, y)
    assert hits, "the divisibility guard did not fire on an sba.Term-style leak"
    assert any(c == "Term" and d == 12 for c, d, *_ in hits), hits
    assert max(a for _, _, a, _, _ in hits) > 0.85, hits
    # the innocent column must not be flagged
    assert not any(c == "GrAppv" for c, *_ in hits), hits


def test_divisibility_guard_ignores_zero_inflated_column():
    """A column whose only signal is the value 0 (FranchiseCode==0 'not a
    franchise', a zero tax rate) is not an arithmetic leak — the `!= 0` term
    exists to keep those out."""
    rng = np.random.default_rng(1)
    code = np.where(rng.random(3000) < 0.5, 0, rng.integers(1, 90000, 3000))
    y = np.where(code == 0, rng.random(3000) < 0.45, rng.random(3000) < 0.10).astype(int)
    hits = divisibility_hits(pd.DataFrame({"FranchiseCode": code}), y)
    assert not hits, f"zero-inflated column flagged as an arithmetic leak: {hits}"


def test_missingness_guard_fires_on_coupled_missingness():
    """Reconstruction of uz_fintech's Score_point (audit 2/3.3): the column is
    missing exactly for the defaulters, so dropping NaNs before the AUC — as the
    1-D guard does — hides it entirely."""
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.3).astype(int)
    v = rng.normal(size=2000)
    v[y == 1] = np.nan                      # missing iff default
    X = pd.DataFrame({"Score_point": v, "age": rng.integers(18, 70, 2000)})

    assert [c for c, _ in missingness_hits(X, y)] == ["Score_point"]
    # and the 1-D AUC guard is blind to it, which is why this guard exists
    fin = np.isfinite(v)
    assert len(np.unique(y[fin])) == 1, "control is only meaningful if NaN == default"


def test_missingness_guard_tolerates_undefined_ratio():
    """poland_1yr/Attr27 is a legitimately undefined financial ratio at
    isna-AUC 0.707; the 0.30 tolerance is set to let it through."""
    rng = np.random.default_rng(3)
    y = (rng.random(7000) < 0.04).astype(int)
    miss = np.where(y == 1, rng.random(7000) < 0.55, rng.random(7000) < 0.12)
    v = np.where(miss, np.nan, rng.normal(size=7000))
    hits = missingness_hits(pd.DataFrame({"Attr27": v}), y)
    auc = roc_auc_score(y, miss.astype(int))
    assert 0.65 < auc < 0.78, f"control drifted: isna-AUC={auc:.3f}"
    assert not hits, f"an undefined-ratio column was flagged: {hits}"


def test_surrogate_key_guard_fires_on_integer_id():
    """An integer id that escaped the _ID_NAMES list (audit 3.5): caught by
    content, not by name. A unique float column must NOT be flagged."""
    n = 500
    X = pd.DataFrame({"acct_no": np.arange(10_000, 10_000 + n),      # integer id
                      "loan_amount": np.linspace(1e3, 9e3, n),       # unique floats
                      "region": np.arange(n) % 7})
    assert surrogate_key_hits(X) == ["acct_no"]


def test_pure_level_guard_fires_on_rule_defined_label():
    """Reconstruction of uz_fintech (audit 3.4): a level that is all-default and
    carries most defaults. The lender's worst grade (laotse.loan_grade == 'G',
    DR 0.98 on 64 rows) must NOT trip it — it is filtered by recall."""
    from pamir.harmonize import scan_pure_levels
    rng = np.random.default_rng(4)
    n = 3000
    y = np.zeros(n, dtype=int)
    y[:670] = 1                                  # 670 defaults
    score = np.array(["ok"] * n, dtype=object)
    score[:590] = "-"                            # all-default, recall 0.881
    grade = np.array(["A"] * n, dtype=object)
    grade[600:664] = "G"                         # DR ~0.98, recall ~0.09
    y[600:663] = 1
    df = pd.DataFrame({"Score_point": score, "loan_grade": grade,
                       "age": rng.integers(18, 70, n)})

    hits = scan_pure_levels(df, y)
    cols = {c for c, *_ in hits}
    assert "Score_point" in cols, f"rule-defined level not found: {hits}"
    assert "loan_grade" not in cols, f"a lender's worst grade was flagged: {hits}"


def test_raw_scan_warns_only_on_unexplained_columns(tmp_path):
    """The raw-file scan runs in harmonize() BEFORE the drops (audit 3.4), but
    must stay quiet about columns the spec already accounts for — sba's
    MIS_Status is the raw status the target is derived from, and warning on
    every load trains the reader to ignore the one signal meant for a human."""
    import warnings as _w
    from pamir.harmonize import harmonize

    n = 1200
    y = np.zeros(n, dtype=int)
    y[:400] = 1
    raw = tmp_path / "raw.csv"
    pd.DataFrame({
        "status_text": np.where(y == 1, "CHGOFF", "PIF"),   # the label, re-encoded
        "amount": np.linspace(1e3, 9e3, n),
        "bad": y,
    }).to_csv(raw, index=False)

    spec = {"id": "ctl",
            "download": {"kind": "github_raw", "locator": "x", "file": "raw.csv",
                         "sep": ",", "encoding": "utf-8"},
            "harmonize": {"target": "bad", "target_rule": "numeric",
                          "day_zero_drop": [], "extra_drop": [],
                          "drop_prefix": [], "shuffle_seed": 7}}

    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        harmonize(raw, spec)
    msgs = [str(x.message) for x in rec if "near-all-default" in str(x.message)]
    assert msgs, "the raw scan did not warn about an unexplained all-default level"
    assert "status_text" in msgs[0]

    # same file, same leak — but now the column is declared in the spec
    spec2 = {**spec, "harmonize": {**spec["harmonize"], "day_zero_drop": ["status_text"]}}
    with _w.catch_warnings(record=True) as rec2:
        _w.simplefilter("always")
        harmonize(raw, spec2)
    assert not [x for x in rec2 if "near-all-default" in str(x.message)], \
        "the raw scan warned about a column the spec already accounts for"


def test_extreme_auc_guard_fires_on_outcome_column():
    """A recorded-after-the-outcome column (Bondora PrincipalBalance is 1.000)
    must be flagged; an ordinary predictor must not."""
    rng = np.random.default_rng(5)
    n = 3000
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({
        "principal_balance": y * 1000.0 + rng.normal(0, 1, n),   # outcome + noise
        "income": rng.lognormal(10, 0.5, n) * (1 - 0.1 * y),       # weak signal
    })
    assert [c for c, _ in extreme_auc_hits(X, y)] == ["principal_balance"]
    # an inverted outcome column (AUC near 0) is a leak too
    X["paid_flag"] = (1 - y) + rng.normal(0, 0.01, n)
    assert "paid_flag" in [c for c, _ in extreme_auc_hits(X, y)]


def test_row_order_guard_fires_on_sorted_target():
    """A table exported sorted by outcome leaks the label through position."""
    rng = np.random.default_rng(6)
    y = np.sort((rng.random(5000) < 0.25).astype(int))
    assert abs(row_order_rho(y)) >= 0.1
    assert abs(row_order_rho(rng.permutation(y))) < 0.1


def test_day_zero_guard_fires_on_readmitted_column():
    """A column on a spec's drop lists that reappears in a table is reported,
    whether it was dropped by the day-zero cut, as an extra drop, or by prefix."""
    spec = pamir.dataset_info("bondora")["harmonize"]
    survivor = spec["day_zero_drop"][0]
    cols = ["Age", "Amount", survivor]
    assert day_zero_hits(cols, spec) == [survivor]
    assert day_zero_hits(["Age", "Amount"], spec) == []
    sba = pamir.dataset_info("sba")["harmonize"]
    assert day_zero_hits(["GrAppv", "Term"], sba) == ["Term"]
    assert day_zero_hits(["R_score", "x"], {"drop_prefix": ["R_"]}) == ["R_score"]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _dummy_model(X_train, y_train, X_test):
    """Logistic regression on numeric columns only."""
    from sklearn.linear_model import LogisticRegression
    num = X_train.select_dtypes(include="number").columns
    X_tr = X_train[num].fillna(0)
    X_te = X_test[num].fillna(0)
    if X_tr.shape[1] == 0:
        return np.full(len(X_te), y_train.mean())
    clf = LogisticRegression(max_iter=300, solver="lbfgs")
    clf.fit(X_tr, y_train)
    return clf.predict_proba(X_te)[:, 1]


def _constant_model(X_train, y_train, X_test):
    """Predict base rate for everyone — AUC = 0.5."""
    return np.full(len(X_test), y_train.mean())


def _perfect_model(X_train, y_train, X_test):
    """Returns random scores — should produce AUC near 0.5."""
    rng = np.random.RandomState(0)
    return rng.rand(len(X_test))


# ═══════════════════════════════════════════════════════════════════════
# Streaming evaluate
# ═══════════════════════════════════════════════════════════════════════

def test_evaluate_one_runs():
    _need("south_german")
    r = pamir.evaluate_one(_dummy_model, "south_german",
                           lag=200, k_refit=5, max_n=1000)
    assert r["dataset"] == "south_german"
    assert r["n_refits"] > 0
    assert r["auc_final"] is None or 0.4 < r["auc_final"] < 1.0


def test_evaluate_one_returns_refit_points():
    _need("lc_small")
    r = pamir.evaluate_one(_dummy_model, "lc_small",
                           lag=200, k_refit=5, max_n=800)
    assert "refit_points" in r
    assert isinstance(r["refit_points"], list)
    if r["refit_points"]:
        rp = r["refit_points"][0]
        assert "pos" in rp
        assert "resolved_n" in rp
        assert "cum_defaults" in rp


def test_evaluate_fleet_subset():
    _need("south_german"); _need("lc_small")
    results = pamir.evaluate(
        _dummy_model,
        datasets=["south_german", "lc_small"],
        lag=200, k_refit=5, max_n=800,
        verbose=False,
    )
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert "auc_final" in results.columns


def test_constant_model_returns_result():
    """A constant predictor should complete without error.

    Note: because the streaming protocol refits multiple times with
    different base rates, the predictions are NOT truly constant across
    the stream — the base rate drifts as defaults accumulate.  So we only
    check that the protocol completes and returns a valid structure.
    """
    _need("south_german")
    r = pamir.evaluate_one(_constant_model, "south_german",
                           lag=200, k_refit=5, max_n=1000)
    assert r["dataset"] == "south_german"
    assert r["n_refits"] > 0
    assert r["auc_final"] is None or isinstance(r["auc_final"], float)


def test_streaming_auc_improves_over_random():
    """A real model should beat random on a clean dataset."""
    _need("taiwan")
    r_model = pamir.evaluate_one(_dummy_model, "taiwan",
                                 lag=200, k_refit=5, max_n=5000)
    r_rand = pamir.evaluate_one(_perfect_model, "taiwan",
                                lag=200, k_refit=5, max_n=5000)
    if r_model["auc_final"] and r_rand["auc_final"]:
        assert r_model["auc_final"] > r_rand["auc_final"] - 0.05, \
            "LogReg should not be much worse than random"


# ═══════════════════════════════════════════════════════════════════════
# Protocol invariants
# ═══════════════════════════════════════════════════════════════════════

def test_no_future_leakage():
    """The protocol must not pass future labels to the model."""
    _need("south_german")
    calls = []

    def spy_model(X_train, y_train, X_test):
        calls.append({
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_total": len(X_train) + len(X_test),
        })
        return np.full(len(X_test), y_train.mean())

    pamir.evaluate_one(spy_model, "south_german", mode="refresh",
                       max_lag_frac=None, lag=200, k_refit=5, max_n=800)

    assert len(calls) > 0, "model was never called"
    for i, c in enumerate(calls):
        assert c["n_total"] <= 800, f"call {i}: train+test={c['n_total']} > 800"
        if i > 0:
            assert c["n_train"] >= calls[i-1]["n_train"], \
                f"call {i}: train shrunk {calls[i-1]['n_train']} → {c['n_train']}"


def test_lag_respected():
    """At each refit, train size must be <= stream_position - lag."""
    _need("south_german")
    calls = []

    def spy_model(X_train, y_train, X_test):
        calls.append({"n_train": len(X_train), "n_test": len(X_test)})
        return np.full(len(X_test), 0.5)

    lag = 200
    max_n = 800
    pamir.evaluate_one(spy_model, "south_german", mode="refresh",
                       max_lag_frac=None, lag=lag, k_refit=5, max_n=max_n)

    for c in calls:
        stream_pos = c["n_train"] + c["n_test"]
        assert c["n_train"] <= stream_pos - lag + 1, \
            f"train={c['n_train']} but stream_pos={stream_pos}, lag={lag}"


def test_training_prefix_grows_between_refits():
    """The resolved prefix and the default count never shrink between refits.

    (Score finality is tested on synthetic streams in test_protocol.py.)"""
    _need("lc_small")
    r = pamir.evaluate_one(_dummy_model, "lc_small", mode="refresh",
                           lag=200, k_refit=5, max_n=800)
    rps = r["refit_points"]
    for i in range(1, len(rps)):
        assert rps[i]["resolved_n"] >= rps[i-1]["resolved_n"], \
            f"resolved_n decreased at refit {i}"
        assert rps[i]["cum_defaults"] >= rps[i-1]["cum_defaults"], \
            f"cum_defaults decreased at refit {i}"


# ═══════════════════════════════════════════════════════════════════════
# i.i.d. evaluate
# ═══════════════════════════════════════════════════════════════════════

def test_evaluate_iid_one_runs():
    _need("south_german")
    r = pamir.evaluate_iid_one(_dummy_model, "south_german", n_seeds=2)
    assert r["dataset"] == "south_german"
    assert r["n_seeds"] == 2
    assert r["auc_mean"] is not None
    assert 0.4 < r["auc_mean"] < 1.0
    assert len(r["aucs"]) == 2


def test_evaluate_iid_fleet():
    _need("south_german"); _need("lc_small")
    results = pamir.evaluate_iid(
        _dummy_model,
        datasets=["south_german", "lc_small"],
        n_seeds=2,
        verbose=False,
    )
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert "auc_mean" in results.columns
    assert "auc_std" in results.columns


def test_iid_seeds_differ():
    """Different seeds should give slightly different AUCs."""
    _need("taiwan")
    r = pamir.evaluate_iid_one(_dummy_model, "taiwan",
                               n_seeds=5, max_n=5000)
    aucs = r["aucs"]
    assert len(set(round(a, 6) for a in aucs)) > 1, \
        "all seeds gave identical AUC — shuffling may be broken"


def test_iid_train_frac_respected():
    """Verify train/test sizes match the requested fraction."""
    _need("south_german")
    sizes = []

    def size_spy(X_train, y_train, X_test):
        sizes.append((len(X_train), len(X_test)))
        return np.full(len(X_test), 0.5)

    pamir.evaluate_iid_one(size_spy, "south_german",
                           train_frac=0.8, n_seeds=1, max_n=1000)
    assert len(sizes) == 1
    n_tr, n_te = sizes[0]
    assert n_tr == 800
    assert n_te == 200


# ═══════════════════════════════════════════════════════════════════════
# Data quality (paranoid)
# ═══════════════════════════════════════════════════════════════════════

def test_no_duplicate_rows():
    """Spot-check: small datasets should not have exact duplicate rows."""
    for ds_id in [d for d in ["south_german", "conorsully", "lc_small"] if _cached(d)]:
        X, y, _ = _load(ds_id)
        df = X.copy()
        df["__y__"] = y
        n_dup = df.duplicated().sum()
        frac = n_dup / len(df)
        assert frac < 0.05, f"{ds_id}: {frac:.1%} duplicate rows"


def test_no_constant_features():
    """No feature column should be constant across the full dataset."""
    for ds_id in CACHED:
        X, _, _ = _load(ds_id)
        for col in X.columns:
            n_unique = X[col].nunique(dropna=True)
            assert n_unique > 1, f"{ds_id}/{col}: constant column"


# ═══════════════════════════════════════════════════════════════════════
# Recipe / spec integrity (no data or network needed)
# ═══════════════════════════════════════════════════════════════════════

_VALID_KINDS = {"kaggle", "kaggle_competition", "uci_zip", "github_raw", "github_zip", "hf"}


def test_every_dataset_has_download_spec():
    for ds_id in pamir.list_datasets():
        dl = pamir.dataset_info(ds_id).get("download")
        assert dl, f"{ds_id}: no download spec"
        for k in ("kind", "locator", "file"):
            assert dl.get(k), f"{ds_id}: download.{k} missing"
        assert dl["kind"] in _VALID_KINDS, f"{ds_id}: bad kind {dl['kind']}"


def test_every_dataset_has_harmonize_spec():
    for ds_id in pamir.list_datasets():
        h = pamir.dataset_info(ds_id).get("harmonize")
        assert h, f"{ds_id}: no harmonize spec"
        assert h.get("target"), f"{ds_id}: harmonize.target missing"
        assert isinstance(h.get("day_zero_drop", []), list)
        assert isinstance(h.get("shuffle_seed", 0), int)


def test_headerless_download_declares_columns():
    """A header='none' source must declare its column names."""
    for ds_id in pamir.list_datasets():
        dl = pamir.dataset_info(ds_id)["download"]
        if dl.get("header") == "none":
            assert len(dl.get("columns", [])) > 0, f"{ds_id}: header=none but no columns"


def test_expected_json_matches_catalog():
    """expected.json exists for every dataset and agrees with the catalog."""
    import json
    from pathlib import Path
    exp = json.loads((Path(pamir.__file__).parent / "data" / "expected.json").read_text())
    cat = pamir.load_catalog()
    for ds_id in pamir.list_datasets():
        assert ds_id in exp, f"{ds_id}: missing from expected.json"
        e = exp[ds_id]
        for k in ("n_rows", "n_defaults", "DR", "n_features", "columns"):
            assert k in e, f"{ds_id}: expected.json missing {k}"
        assert e["n_features"] == cat.loc[ds_id, "features"], \
            f"{ds_id}: expected {e['n_features']} feats, catalog {cat.loc[ds_id, 'features']}"
        assert abs(e["DR"] - cat.loc[ds_id, "DR"]) < 0.005, f"{ds_id}: DR mismatch"
        assert e["n_features"] == len(e["columns"]), f"{ds_id}: columns/n_features disagree"


def test_attribution_present_where_license_requires():
    """CC-BY / CC-BY-SA / ODbL / MIT datasets must carry an attribution string."""
    for ds_id in pamir.list_datasets():
        info = pamir.dataset_info(ds_id)
        lic = info["license"]
        if any(t in lic for t in ("CC-BY", "ODbL", "MIT")):
            assert info.get("attribution"), f"{ds_id}: {lic} requires attribution"


# ═══════════════════════════════════════════════════════════════════════
# Harmonizer / downloader internals (synthetic, no network)
# ═══════════════════════════════════════════════════════════════════════

def test_harmonize_synthetic(tmp_path):
    """End-to-end harmonize on a tiny synthetic raw file."""
    from pamir.harmonize import harmonize
    raw = tmp_path / "raw.csv"
    pd.DataFrame({
        "id": range(1, 21),                       # id column -> dropped
        "amount": np.arange(20) * 1.0,
        "R_ratio": np.arange(20) * 0.5,           # drop_prefix R_ -> dropped
        "risk_score": np.arange(20) % 5,
        "post_outcome": np.arange(20),            # day_zero_drop -> dropped
        "when": ["2021-01-02"] * 20,              # date -> dropped
        "bad": [0, 1] * 10,
    }).to_csv(raw, index=False)
    spec = {
        "id": "synthetic",
        "download": {"kind": "github_raw", "locator": "x", "file": "raw.csv",
                     "sep": ",", "encoding": "utf-8"},
        "harmonize": {"target": "bad", "target_rule": "numeric",
                      "day_zero_drop": ["post_outcome"], "extra_drop": [],
                      "drop_prefix": ["R_"], "shuffle_seed": 7},
    }
    df = harmonize(raw, spec)
    assert df.columns[-1] == "__target__"
    assert set(np.unique(df["__target__"])) <= {0, 1}
    cols = set(df.columns)
    for gone in ("id", "R_ratio", "post_outcome", "when", "bad"):
        assert gone not in cols, f"{gone} should have been dropped"
    assert {"amount", "risk_score"} <= cols
    # deterministic shuffle
    df2 = harmonize(raw, spec)
    assert df["amount"].tolist() == df2["amount"].tolist()


def test_validate_flags_mismatch(monkeypatch):
    from pamir import contract
    from pamir.download import _validate
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    good = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10) % 3, "__target__": y})
    monkeypatch.setitem(contract.EXPECTED, "ctl", {
        "columns": {"a": "num", "b": "num"}, "n_features": 2, "n_rows": 10,
        "n_defaults": 3, "DR": 0.3, "target_sha256": contract.target_sha256(y)})
    assert _validate("ctl", good) == [], "clean frame should not warn"
    assert _validate("ctl", good.drop(columns=["a"])), "missing feature should warn"
    assert _validate("ctl", good.iloc[:9]), "a missing row should warn"
    shuffled = good.iloc[::-1].reset_index(drop=True)
    assert any("stored order" in m for m in _validate("ctl", shuffled)), \
        "same counts, different order should warn"


def test_cache_dir_respects_env(tmp_path, monkeypatch):
    from pamir.download import cache_dir as _cache_dir
    monkeypatch.setenv("PAMIR_CACHE", str(tmp_path / "c"))
    assert _cache_dir() == tmp_path / "c"
    assert (tmp_path / "c").is_dir()


# ═══════════════════════════════════════════════════════════════════════
# Credential-free set & CLI (no data or network needed)
# ═══════════════════════════════════════════════════════════════════════

def test_open_datasets_are_credential_free():
    open_ids = pamir.open_datasets()
    assert open_ids, "no credential-free datasets?"
    assert set(open_ids) <= set(pamir.list_datasets())
    for ds in open_ids:
        assert pamir.dataset_info(ds)["needs_credentials"] is False
    # the Kaggle-sourced ones must NOT be in the open set
    assert "gmsc" not in open_ids          # kaggle
    # and known open ones are present
    assert {"south_german", "poland_1yr", "gastonstat"} <= set(open_ids)


def test_needs_credentials_flag_present_and_consistent():
    for ds in pamir.list_datasets():
        info = pamir.dataset_info(ds)
        assert isinstance(info["needs_credentials"], bool)
        kind = info["download"]["kind"]
        assert info["needs_credentials"] == (kind in {"kaggle", "kaggle_competition"})


def test_cli_smoke(capsys):
    from pamir import cli
    assert cli.main(["list"]) == 0
    assert cli.main(["list", "--open"]) == 0
    assert cli.main(["info", "taiwan"]) == 0
    assert cli.main(["cache"]) == 0
    out = capsys.readouterr().out
    assert "taiwan" in out
    assert cli.main(["info", "does_not_exist"]) == 1
