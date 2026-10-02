"""Tests for failure accounting in both evaluation protocols.

A benchmark that silently drops the datasets a model failed on reports an
average over a subset the model itself selected, which flatters a broken
model over a working one.  These tests pin the accounting that prevents it.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import pamir


SMALL = dict(lag=200, k_refit=5, max_n=800)
PAIR = ["south_german", "gastonstat"]


def _working_model(X_train, y_train, X_test):
    """Score on the first numeric column — enough to produce a real AUC."""
    num = X_train.select_dtypes(include="number")
    if num.shape[1] == 0:
        return np.full(len(X_test), 0.5)
    col = num.columns[0]
    return X_test[col].fillna(0).to_numpy(dtype=float)


def _always_raises(X_train, y_train, X_test):
    raise RuntimeError("model is broken")


def _raises_on_wide(X_train, y_train, X_test):
    if X_train.shape[1] > 15:
        raise ValueError("too many features")
    return _working_model(X_train, y_train, X_test)


# ═══════════════════════════════════════════════════════════════════════
# evaluate_one — failure accounting
# ═══════════════════════════════════════════════════════════════════════

def test_evaluate_one_reports_call_and_failure_counts():
    r = pamir.evaluate_one(_working_model, "south_german",
                           on_error="ignore", **SMALL)
    assert r["n_calls"] > 0
    assert r["n_failures"] == 0
    assert r["errors"] == []


def test_evaluate_one_counts_every_failure():
    r = pamir.evaluate_one(_always_raises, "south_german",
                           on_error="ignore", **SMALL)
    assert r["n_calls"] > 0
    assert r["n_failures"] == r["n_calls"]
    assert r["auc_final"] is None


def test_evaluate_one_records_the_error_message():
    r = pamir.evaluate_one(_always_raises, "south_german",
                           on_error="ignore", **SMALL)
    assert r["errors"], "the failure message must be retained"
    assert "RuntimeError" in r["errors"][0]
    assert "model is broken" in r["errors"][0]


def test_evaluate_one_warns_by_default():
    with pytest.warns(UserWarning, match="raised"):
        pamir.evaluate_one(_always_raises, "south_german", **SMALL)


def test_evaluate_one_on_error_raise_propagates():
    with pytest.raises(RuntimeError, match="model is broken"):
        pamir.evaluate_one(_always_raises, "south_german",
                           on_error="raise", **SMALL)


def test_evaluate_one_on_error_ignore_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = pamir.evaluate_one(_always_raises, "south_german",
                               on_error="ignore", **SMALL)
    assert r["n_failures"] > 0


def test_evaluate_one_rejects_unknown_on_error():
    with pytest.raises(ValueError, match="on_error"):
        pamir.evaluate_one(_working_model, "south_german",
                           on_error="explode", **SMALL)


def test_working_model_is_never_flagged():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = pamir.evaluate_one(_working_model, "south_german", **SMALL)
    assert r["n_failures"] == 0


# ═══════════════════════════════════════════════════════════════════════
# evaluate — fleet-level coverage
# ═══════════════════════════════════════════════════════════════════════

def test_evaluate_exposes_failure_columns():
    df = pamir.evaluate(_working_model, datasets=PAIR, verbose=False,
                        on_error="ignore", **SMALL)
    for col in ("n_calls", "n_failures", "auc_final"):
        assert col in df.columns


def test_evaluate_omits_nested_refit_points_column():
    """Nested lists block to_csv and were never documented."""
    df = pamir.evaluate(_working_model, datasets=PAIR, verbose=False,
                        on_error="ignore", **SMALL)
    assert "refit_points" not in df.columns
    df.to_csv(index=False)  # must not raise


def test_fleet_summary_reports_full_coverage():
    summary = pamir.fleet_summary(
        pamir.evaluate(_working_model, datasets=PAIR, verbose=False,
                       on_error="ignore", **SMALL))
    assert summary["n_datasets"] == 2
    assert summary["n_scored"] == 2
    assert summary["complete"] is True
    assert summary["auc_mean"] is not None


def test_fleet_summary_withholds_mean_when_incomplete():
    """The headline number is undefined if the model did not score everything."""
    df = pamir.evaluate(_raises_on_wide, datasets=PAIR, verbose=False,
                        on_error="ignore", **SMALL)
    summary = pamir.fleet_summary(df)
    assert summary["complete"] is False
    assert summary["n_scored"] < summary["n_datasets"]
    assert summary["auc_mean"] is None, (
        "a mean over the datasets a model survived is not a fleet mean")
    assert summary["auc_mean_scored_only"] is not None


def test_incomplete_fleet_run_warns():
    with pytest.warns(UserWarning, match="did not score"):
        pamir.evaluate(_raises_on_wide, datasets=PAIR, verbose=False, **SMALL)


def test_broken_model_cannot_outscore_a_working_one():
    """The regression this whole module exists to prevent."""
    working = pamir.fleet_summary(
        pamir.evaluate(_working_model, datasets=PAIR, verbose=False,
                       on_error="ignore", **SMALL))
    broken = pamir.fleet_summary(
        pamir.evaluate(_raises_on_wide, datasets=PAIR, verbose=False,
                       on_error="ignore", **SMALL))

    assert working["complete"] is True
    assert broken["complete"] is False
    assert working["auc_mean"] is not None
    assert broken["auc_mean"] is None


# ═══════════════════════════════════════════════════════════════════════
# i.i.d. protocol — same accounting
# ═══════════════════════════════════════════════════════════════════════

def test_evaluate_iid_one_reports_failures():
    r = pamir.evaluate_iid_one(_always_raises, "south_german",
                               n_seeds=3, max_n=800, on_error="ignore")
    assert r["n_calls"] == 3
    assert r["n_failures"] == 3
    assert r["auc_mean"] is None
    assert r["errors"]


def test_evaluate_iid_one_on_error_raise_propagates():
    with pytest.raises(RuntimeError):
        pamir.evaluate_iid_one(_always_raises, "south_german",
                               n_seeds=2, max_n=800, on_error="raise")


def test_evaluate_iid_exposes_failure_columns():
    df = pamir.evaluate_iid(_working_model, datasets=PAIR, n_seeds=2,
                            max_n=800, verbose=False, on_error="ignore")
    for col in ("n_calls", "n_failures", "auc_mean"):
        assert col in df.columns


def test_fleet_summary_handles_iid_frame():
    df = pamir.evaluate_iid(_working_model, datasets=PAIR, n_seeds=2,
                            max_n=800, verbose=False, on_error="ignore")
    summary = pamir.fleet_summary(df)
    assert summary["n_scored"] == 2
    assert summary["complete"] is True


def test_fleet_summary_rejects_a_frame_with_no_auc_column():
    with pytest.raises(ValueError, match="auc"):
        pamir.fleet_summary(pd.DataFrame({"dataset": ["a"]}))
