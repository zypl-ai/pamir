"""Invariants of the streaming protocol, on synthetic streams (no data needed).

The arrival mode must never hand a scorer a label that had not matured when a
row arrived, nor a row that had not arrived; every score must be final; the
delay cap must keep short streams scoreable; a scorer whose output depends on
the rest of its batch must be caught.  The refresh mode must reproduce 0.3.0.
"""

import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import pamir

ev = importlib.import_module("pamir.evaluate")


def _synthetic(n=3000, seed=0, n_feat=4):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_feat)), columns=[f"x{i}" for i in range(n_feat)])
    p = 1 / (1 + np.exp(-(1.5 * X["x0"].to_numpy() - 1.2)))
    y = (rng.random(n) < p).astype(int)
    return X, y


@pytest.fixture
def stream(monkeypatch):
    """Register synthetic datasets that evaluate_one loads instead of the cache."""
    data = {}

    def fake_load(dataset_id, max_rows=None, strict=True, **kw):
        X, y = data[dataset_id]
        if max_rows is not None:
            X, y = X.iloc[:max_rows], y[:max_rows]
        return X, y, {"contract_ok": True}

    monkeypatch.setattr(ev, "load_dataset", fake_load)

    def add(name, **kw):
        data[name] = _synthetic(**kw)
        return data[name]

    return add


def _x0_predict(X_train, y_train, X_test):
    return X_test["x0"].to_numpy(dtype=float)


# ── arrival mode ─────────────────────────────────────────────────────────

def test_arrival_scores_only_arrived_rows_with_matured_labels(stream):
    X, y = stream("s", n=3000)
    lag = 300
    fits = []   # (n_train, indices scored by that scorer)

    def fit_fn(X_train, y_train):
        assert np.array_equal(np.asarray(y_train), y[:len(X_train)])
        record = (len(X_train), [])
        fits.append(record)

        def scorer(X_rows):
            record[1].extend(X_rows.index.tolist())
            return X_rows["x0"].to_numpy(dtype=float)
        return scorer

    r = pamir.evaluate_one(fit_fn, "s", lag=lag, k_refit=10, max_n=None,
                           max_lag_frac=None, probe=False)
    assert r["n_refits_ok"] == len(fits) > 3
    for j, (n_train, rows) in enumerate(fits):
        if not rows:
            continue
        # the refit that produced this scorer ran at step t = n_train + lag - 1;
        # it may score only rows that arrive after t ...
        assert min(rows) >= n_train + lag, (j, n_train, min(rows))
        # ... and before the next refit
        if j + 1 < len(fits):
            assert max(rows) <= fits[j + 1][0] + lag - 1, (j, max(rows))
    scored = [i for _, rows in fits for i in rows]
    assert len(scored) == len(set(scored)), "a row was scored twice"
    assert r["n_scored"] == len(scored)


def test_arrival_scores_are_final_and_budget_is_recorded(stream):
    stream("s", n=2000)
    r = pamir.evaluate_one(pamir.logistic_fit, "s", lag=200, k_refit=10,
                           max_n=None, max_lag_frac=None)
    train_n = np.asarray(r["train_n"])
    scored = train_n >= 0
    assert scored.sum() == r["n_scored"]
    # a row's budget is a resolved prefix that ended at least `lag` rows before it
    idx = np.flatnonzero(scored)
    assert np.all(train_n[idx] <= idx + 1 - 200)
    assert r["row_independent"] is True
    assert any(r[c] is not None for c in ev.BUDGET_COLUMNS)


def test_delay_cap_keeps_short_streams_scoreable(stream):
    stream("short", n=1000)
    capped = pamir.evaluate_one(pamir.logistic_fit, "short", lag=1000, max_n=None)
    assert capped["lag_effective"] == 200
    assert capped["auc_final"] is not None
    uncapped = pamir.evaluate_one(pamir.logistic_fit, "short", lag=1000, max_n=None,
                                  max_lag_frac=None)
    assert uncapped["lag_effective"] == 1000
    assert uncapped["n_refits"] == 0 and uncapped["auc_final"] is None


def test_probe_flags_a_batch_dependent_scorer(stream):
    stream("s", n=2000)

    def rank_in_batch_fit(X_train, y_train):
        def scorer(X_rows):            # depends on the other rows of the batch
            return X_rows["x0"].rank(pct=True).to_numpy()
        return scorer

    r = pamir.evaluate_one(rank_in_batch_fit, "s", lag=200, max_n=None)
    assert r["row_independent"] is False
    df = pd.DataFrame([{k: r[k] for k in ev.RESULT_COLUMNS}])
    s = pamir.fleet_summary(df)
    assert s["auc_mean"] is None and s["row_dependent_datasets"] == ["s"]

    # scored one row at a time, the same scorer cannot see other rows
    r1 = pamir.evaluate_one(rank_in_batch_fit, "s", lag=200, max_n=None,
                            batch_scoring=False)
    assert r1["row_independent"] is True


def test_wrapped_predict_fn_counts_scoring_calls(stream):
    stream("s", n=1500)

    def always_raises(X_train, y_train, X_test):
        raise RuntimeError("broken")

    r = pamir.evaluate_one(always_raises, "s", lag=150, max_n=None, on_error="ignore")
    assert r["n_calls"] > 0 and r["n_failures"] == r["n_calls"]
    assert r["auc_final"] is None and r["n_scored"] == 0


def test_failed_refit_keeps_the_previous_scorer(stream):
    stream("s", n=3000)
    calls = {"n": 0}

    def flaky_fit(X_train, y_train):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second fit fails")
        return pamir.logistic_fit(X_train, y_train)

    r = pamir.evaluate_one(flaky_fit, "s", lag=300, max_n=None, on_error="ignore",
                           max_lag_frac=None)
    assert r["n_refits_ok"] == r["n_refits"] - 1
    assert [p["ok"] for p in r["refit_points"]][1] is False
    assert r["n_scored"] > 0 and r["auc_final"] is not None


# ── refresh mode: 0.3.0 semantics ────────────────────────────────────────

def _evaluate_v030(predict_fn, X, y, lag, k_refit, min_defaults=3):
    """The 0.3.0 loop of pamir.evaluate.evaluate_one, verbatim in substance."""
    n = len(y)
    preds = np.full(n, np.nan)
    cum = since = 0
    refits = []
    for t in range(n):
        r = max(0, t + 1 - lag)
        if t >= lag and y[t - lag] == 1:
            cum += 1
            since += 1
        need = cum >= min_defaults and since >= k_refit
        if cum == min_defaults and not refits:
            need = True
        if not (need and r > 0 and t + 1 < n):
            continue
        if r - cum < min_defaults:
            continue
        preds[r:] = predict_fn(X.iloc[:r], y[:r], X.iloc[r:])
        since = 0
        refits.append(t)
    valid = ~np.isnan(preds)
    return float(roc_auc_score(y[valid], preds[valid])), len(refits)


@pytest.mark.parametrize("lag,k_refit", [(200, 5), (500, 40), (1000, 10)])
def test_refresh_mode_reproduces_v030(stream, lag, k_refit):
    X, y = stream("s", n=4000, seed=3)
    auc_old, refits_old = _evaluate_v030(_x0_predict, X, y, lag, k_refit)
    r = pamir.evaluate_one(_x0_predict, "s", mode="refresh", max_lag_frac=None,
                           lag=lag, k_refit=k_refit, max_n=None)
    assert r["n_refits"] == refits_old
    assert r["auc_final"] == pytest.approx(auc_old, abs=1e-12)
    assert r["row_independent"] is None


def test_refresh_mode_passes_unarrived_rows(stream):
    """Documented 0.3.0 behaviour: X_test is the whole remaining stream."""
    stream("s", n=1200)
    seen = []

    def spy(X_train, y_train, X_test):
        seen.append((len(X_train), len(X_test)))
        return np.zeros(len(X_test))

    pamir.evaluate_one(spy, "s", mode="refresh", lag=200, max_n=None, max_lag_frac=None)
    assert seen and all(a + b == 1200 for a, b in seen)


def test_unknown_mode_is_rejected(stream):
    stream("s", n=500)
    with pytest.raises(ValueError, match="mode"):
        pamir.evaluate_one(_x0_predict, "s", mode="online")


# ── summaries ────────────────────────────────────────────────────────────

def test_fleet_summary_withholds_mean_on_contract_or_row_dependence():
    base = pd.DataFrame({"dataset": ["a", "b"], "auc_final": [0.7, 0.8],
                         "n_failures": [0, 0]})
    assert pamir.fleet_summary(base)["auc_mean"] == pytest.approx(0.75)
    bad_contract = base.assign(contract_ok=[True, False])
    s = pamir.fleet_summary(bad_contract)
    assert s["complete"] and s["auc_mean"] is None and s["contract_failed_datasets"] == ["b"]
    refresh_rows = base.assign(contract_ok=[True, True], row_independent=[None, None])
    assert pamir.fleet_summary(refresh_rows)["auc_mean"] == pytest.approx(0.75)
    dependent = base.assign(contract_ok=[True, True], row_independent=[True, False])
    assert pamir.fleet_summary(dependent)["auc_mean"] is None


def test_auc_by_budget_bins_rows_by_training_size():
    rng = np.random.default_rng(7)
    n = 600
    y = (rng.random(n) < 0.3).astype(int)
    preds = y + rng.normal(0, 1.0, n)
    train_n = np.repeat([-1, 50, 150, 500, 2000, 5000], 100)
    out = ev.auc_by_budget(y, preds, train_n)
    assert set(out) == set(ev.BUDGET_COLUMNS)
    assert out["auc_budget_10000"] is None
    for lo, block in zip((0, 100, 300, 1000, 3000), range(1, 6)):
        sl = slice(block * 100, (block + 1) * 100)
        assert out[f"auc_budget_{lo}"] == pytest.approx(roc_auc_score(y[sl], preds[sl]))


def test_effective_lag():
    assert ev.effective_lag(1000, 20000, 0.2) == 1000
    assert ev.effective_lag(1000, 1000, 0.2) == 200
    assert ev.effective_lag(1000, 1000, None) == 1000
