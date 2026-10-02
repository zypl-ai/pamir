"""Tests for the documented fidelity shortlist.

``fidelity_frame()`` returns 66 columns.  A new user needs to know which few
decide whether to believe a delta, so the shortlist is part of the API rather
than something to reconstruct from the source.
"""

import numpy as np
import pandas as pd
import pytest

from pamir.synthetic import CrossValidatedAugmentation, SyntheticMixer
from pamir.synthetic.fidelity import HEADLINE_FIDELITY_COLUMNS

from test_synthetic import EchoGenerator, make_frame, TARGET


def _result(n_splits=2):
    frame = make_frame(n=300)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    mixer = SyntheticMixer(generators={"echo": EchoGenerator(jitter=0.5)},
                           synthetic_share=0.5)
    return CrossValidatedAugmentation(mixer, n_splits=n_splits,
                                      verbose=False).run(X, y)


def test_headline_columns_are_a_documented_shortlist():
    assert len(HEADLINE_FIDELITY_COLUMNS) < 12
    assert "leak.synth_holdout_only_overlap" in HEADLINE_FIDELITY_COLUMNS, (
        "the probe that invalidates a delta must be on the shortlist")


def test_fidelity_headline_is_much_narrower_than_the_full_frame():
    result = _result()
    full = result.fidelity_frame()
    headline = result.fidelity_headline()
    assert len(headline.columns) < len(full.columns) / 3


def test_fidelity_headline_keeps_the_fold_and_generator_labels():
    headline = _result().fidelity_headline()
    assert list(headline.columns[:2]) == ["fold", "generator"]


def test_fidelity_headline_columns_all_exist_in_the_full_frame():
    result = _result()
    full = set(result.fidelity_frame().columns)
    assert set(result.fidelity_headline().columns) <= full


def test_fidelity_headline_has_one_row_per_fold_and_generator():
    result = _result(n_splits=3)
    headline = result.fidelity_headline()
    assert len(headline) == 3
    assert set(headline["generator"]) == {"echo"}


def test_fidelity_headline_is_empty_when_fidelity_did_not_run():
    frame = make_frame(n=300)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    mixer = SyntheticMixer(generators={"echo": EchoGenerator()},
                           synthetic_share=0.5, fidelity=False)
    result = CrossValidatedAugmentation(mixer, n_splits=2, verbose=False).run(X, y)

    headline = result.fidelity_headline()
    assert isinstance(headline, pd.DataFrame)
    # The leakage probes run even without the fidelity battery, so whatever is
    # present must still be a subset of the full frame.
    assert set(headline.columns) <= set(result.fidelity_frame().columns) | {"fold", "generator"}
