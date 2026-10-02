"""Tests for the reference baselines and the feature encoder they share.

Most of the datasets ship `object` or `bool` columns.  A baseline that a new
user is told to copy must run on all of them without edits.
"""

import numpy as np
import pandas as pd
import pytest

import pamir
from pamir.baselines import encode_features, gbdt_baseline, logistic_baseline


CATEGORICAL_DATASETS = ["bondora", "prosper", "south_german", "pakdd"]
BASELINES = [logistic_baseline, gbdt_baseline]


def _split(dataset_id, n_train=600, n_total=900):
    X, y, _ = pamir.load_dataset(dataset_id, max_rows=n_total)
    return X.iloc[:n_train], y[:n_train], X.iloc[n_train:]


# ═══════════════════════════════════════════════════════════════════════
# encode_features
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("ds_id", CATEGORICAL_DATASETS)
def test_encode_features_returns_all_numeric(ds_id):
    X_train, _, X_test = _split(ds_id)
    train, test = encode_features(X_train, X_test)
    assert all(np.issubdtype(t, np.number) for t in train.dtypes)
    assert all(np.issubdtype(t, np.number) for t in test.dtypes)


def test_encode_features_preserves_shape_and_columns():
    X_train, _, X_test = _split("bondora")
    train, test = encode_features(X_train, X_test)
    assert train.shape == X_train.shape
    assert test.shape == X_test.shape
    assert list(train.columns) == list(X_train.columns)


def test_encode_features_uses_one_shared_level_map():
    """The same category must encode to the same code in train and test."""
    train_raw = pd.DataFrame({"c": ["a", "b", "a", "c"]})
    test_raw = pd.DataFrame({"c": ["c", "a", "b", "b"]})
    train, test = encode_features(train_raw, test_raw)

    mapping = dict(zip(train_raw["c"], train["c"]))
    for raw, code in zip(test_raw["c"], test["c"]):
        assert mapping[raw] == code


def test_encode_features_handles_unseen_test_categories():
    train_raw = pd.DataFrame({"c": ["a", "b"]})
    test_raw = pd.DataFrame({"c": ["z"]})
    train, test = encode_features(train_raw, test_raw)
    assert test["c"].notna().all()
    assert (test["c"] >= 0).all()


def test_encode_features_does_not_mutate_its_inputs():
    X_train, _, X_test = _split("bondora")
    before_train = X_train.copy()
    before_test = X_test.copy()
    encode_features(X_train, X_test)
    pd.testing.assert_frame_equal(X_train, before_train)
    pd.testing.assert_frame_equal(X_test, before_test)


def test_encode_features_leaves_numeric_missing_values_alone():
    """NaN is information; the encoder must not silently fill it."""
    train_raw = pd.DataFrame({"n": [1.0, np.nan, 3.0]})
    test_raw = pd.DataFrame({"n": [np.nan, 2.0, 4.0]})
    train, test = encode_features(train_raw, test_raw)
    assert train["n"].isna().sum() == 1
    assert test["n"].isna().sum() == 1


def test_encode_features_rejects_mismatched_columns():
    with pytest.raises(ValueError, match="column"):
        encode_features(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))


# ═══════════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model", BASELINES)
@pytest.mark.parametrize("ds_id", CATEGORICAL_DATASETS)
def test_baseline_runs_on_datasets_with_string_columns(model, ds_id):
    X_train, y_train, X_test = _split(ds_id)
    scores = model(X_train, y_train, X_test)
    assert len(scores) == len(X_test)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("model", BASELINES)
def test_baseline_returns_scores_in_unit_interval(model):
    X_train, y_train, X_test = _split("taiwan")
    scores = model(X_train, y_train, X_test)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


@pytest.mark.parametrize("model", BASELINES)
def test_baseline_beats_random_on_a_clean_dataset(model):
    from sklearn.metrics import roc_auc_score

    X, y, _ = pamir.load_dataset("taiwan", max_rows=6000)
    scores = model(X.iloc[:4000], y[:4000], X.iloc[4000:])
    assert roc_auc_score(y[4000:], scores) > 0.6


@pytest.mark.parametrize("model", BASELINES)
def test_baseline_satisfies_the_protocol_contract(model):
    """The real check: it must survive a fleet run without a single failure."""
    df = pamir.evaluate(model, datasets=["south_german", "gastonstat"],
                        lag=200, k_refit=20, max_n=800, verbose=False,
                        on_error="raise")
    assert (df["n_failures"] == 0).all()
    assert df["auc_final"].notna().all()


def test_baselines_run_on_every_dataset_in_the_fleet():
    """One fit per dataset, every one, no edits — the claim the README makes."""
    for ds_id in pamir.list_datasets():
        X_train, y_train, X_test = _split(ds_id, n_train=400, n_total=600)
        scores = logistic_baseline(X_train, y_train, X_test)
        assert len(scores) == len(X_test), ds_id
        assert np.isfinite(scores).all(), ds_id
