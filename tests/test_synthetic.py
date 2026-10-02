"""Tests for pamir.synthetic: proportions, leakage contract, fidelity, CV."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

# The SDV engines and the sdmetrics fidelity metrics live behind the optional
# `synthetic` extra; the tests that exercise them skip when it is not installed.
requires_sdv = pytest.mark.skipif(
    importlib.util.find_spec("sdv") is None,
    reason="requires the optional 'synthetic' extra (sdv)",
)
requires_sdmetrics = pytest.mark.skipif(
    importlib.util.find_spec("sdmetrics") is None,
    reason="requires the optional 'synthetic' extra (sdmetrics)",
)

from pamir.synthetic import (
    CallableAdapter,
    CrossValidatedAugmentation,
    PoolAdapter,
    SDVAdapter,
    SyntheticMixer,
    as_adapter,
    fidelity_report,
    leakage_report,
    plan_mixture,
)
from pamir.synthetic.adapters import GeneratorAdapter, _expand

TARGET = "__target__"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 1, n)
    return pd.DataFrame({
        "income": rng.gamma(4, 1200, n),
        "age": rng.integers(20, 70, n),
        "score": score,
        "region": rng.choice(["north", "south", "east"], n),
        TARGET: (score + rng.normal(0, 1, n) > 0.7).astype(int),
    })


class EchoGenerator(GeneratorAdapter):
    """A generator that resamples its training rows with noise.

    Deliberately trivial: it can only ever reproduce what it was fitted on, so
    if a test row shows up in its output, the test row reached fit().
    """

    def __init__(self, name="echo", jitter=0.0):
        super().__init__(name)
        self.jitter = jitter
        self._train = None

    def _reset(self):
        self._train = None

    def _fit(self, data, target, discrete_columns):
        self._train = data.copy()

    def _sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self._train), size=n, replace=True)
        out = self._train.iloc[idx].reset_index(drop=True)
        if self.jitter:
            for c in out.select_dtypes("number").columns:
                if c != TARGET:
                    out[c] = out[c] + rng.normal(0, self.jitter * out[c].std(), n)
        return out


# --------------------------------------------------------------------------- #
# proportions
# --------------------------------------------------------------------------- #
def test_plan_keep_real_hits_the_requested_share():
    plan = plan_mixture(1000, 0.7, {"zgan": 0.5, "zedge": 0.5})
    assert plan.n_real == 1000
    assert plan.n_synthetic == 2333
    # an exact tie on the remainder breaks by name, deterministically
    assert plan.per_generator == {"zgan": 1166, "zedge": 1167}
    assert plan.realized_share == pytest.approx(0.7, abs=1e-4)


def test_plan_splits_synthetic_by_weight():
    plan = plan_mixture(1000, 0.5, {"a": 0.75, "b": 0.25})
    assert plan.per_generator["a"] == 750 and plan.per_generator["b"] == 250
    assert sum(plan.per_generator.values()) == plan.n_synthetic


def test_plan_normalizes_weights_that_do_not_sum_to_one():
    plan = plan_mixture(100, 0.5, {"a": 3, "b": 1})
    assert plan.weights == {"a": 0.75, "b": 0.25}


def test_plan_fixed_total_pins_the_frame_size():
    plan = plan_mixture(1000, 0.7, {"a": 1.0}, sizing="fixed_total", n_total=500)
    assert plan.n_total == 500
    assert plan.n_synthetic == 350 and plan.n_real == 150


def test_plan_rejects_impossible_requests():
    with pytest.raises(ValueError, match="synthetic_share"):
        plan_mixture(100, 1.0, {"a": 1.0})
    with pytest.raises(ValueError, match="only 100 are available"):
        plan_mixture(100, 0.1, {"a": 1.0}, sizing="fixed_total", n_total=1000)


# --------------------------------------------------------------------------- #
# mixing
# --------------------------------------------------------------------------- #
def test_mixer_builds_the_requested_composition():
    train = make_frame()
    mixer = SyntheticMixer(
        generators={"a": EchoGenerator("a"), "b": EchoGenerator("b")},
        weights={"a": 0.5, "b": 0.5},
        synthetic_share=0.7,
        target=TARGET,
        fidelity=False,
    )
    mix = mixer.build(train)
    comp = mix.composition
    assert comp["n_real"] == len(train)
    assert comp["realized_synthetic_share"] == pytest.approx(0.7, abs=0.001)
    assert comp["per_generator"]["a"] == comp["per_generator"]["b"] + 1 or \
           comp["per_generator"]["a"] == comp["per_generator"]["b"]
    assert len(mix.data) == comp["n_total"]
    assert set(mix.sources.unique()) == {"real", "a", "b"}
    assert list(mix.data.columns) == list(train.columns)


def test_mixer_is_deterministic_under_a_fixed_seed():
    train = make_frame()
    kw = dict(generators={"a": EchoGenerator("a")}, synthetic_share=0.5,
              target=TARGET, fidelity=False, random_state=7)
    first = SyntheticMixer(**kw).build(train).data
    second = SyntheticMixer(**kw).build(train).data
    pd.testing.assert_frame_equal(first, second)


def test_mixer_can_force_a_synthetic_class_balance():
    train = make_frame()
    mixer = SyntheticMixer(
        generators={"a": EchoGenerator("a")},
        synthetic_share=0.5, target=TARGET, synthetic_target_rate=0.5,
        oversample=3.0, fidelity=False,
    )
    mix = mixer.build(train)
    assert mix.composition["target_rate"]["a"] == pytest.approx(0.5, abs=0.01)


def test_mixer_rejects_a_missing_target_column():
    mixer = SyntheticMixer(generators={"a": EchoGenerator("a")}, target="nope", fidelity=False)
    with pytest.raises(ValueError, match="not a column"):
        mixer.build(make_frame())


def test_weights_must_name_the_generators():
    with pytest.raises(ValueError, match="No weight given"):
        SyntheticMixer(generators={"a": EchoGenerator("a"), "b": EchoGenerator("b")},
                       weights={"a": 1.0})


# --------------------------------------------------------------------------- #
# the leakage contract
# --------------------------------------------------------------------------- #
def test_generators_never_see_held_out_rows():
    """The echo generator can only emit rows it was fitted on."""
    frame = make_frame(300)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    mixer = SyntheticMixer(generators={"echo": EchoGenerator("echo")},
                           synthetic_share=0.5, fidelity=False)
    runner = CrossValidatedAugmentation(mixer, n_splits=3, seed=42, verbose=False)
    result = runner.run(X, y)

    seen = []

    class Recorder(EchoGenerator):
        def _fit(self, data, target, discrete_columns):
            seen.append(data.copy())
            super()._fit(data, target, discrete_columns)

    mixer2 = SyntheticMixer(generators={"echo": Recorder("echo")},
                            synthetic_share=0.5, fidelity=False)
    CrossValidatedAugmentation(mixer2, n_splits=3, seed=42, verbose=False).run(X, y)

    assert len(seen) == 3
    for fitted in seen:
        assert len(fitted) < len(frame)          # a fold's train split, not the table
    assert result.fold_frame().shape[0] == 3


def test_each_fold_gets_a_fresh_generator():
    frame = make_frame(300)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    fits = []

    class Counter(EchoGenerator):
        def _fit(self, data, target, discrete_columns):
            fits.append(id(self))
            super()._fit(data, target, discrete_columns)

    mixer = SyntheticMixer(generators={"echo": Counter("echo")},
                           synthetic_share=0.5, fidelity=False)
    CrossValidatedAugmentation(mixer, n_splits=3, verbose=False).run(X, y)
    assert len(set(fits)) == 3, "a generator instance was reused across folds"


def test_shared_pool_is_rejected_across_folds():
    """A pool generated once from the whole table has seen every test fold."""
    frame = make_frame(300)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    mixer = SyntheticMixer(generators={"pool": PoolAdapter(pool=frame, name="pool")},
                           synthetic_share=0.5, fidelity=False)
    with pytest.raises(ValueError, match="reused across CV folds"):
        CrossValidatedAugmentation(mixer, n_splits=3, verbose=False).run(X, y)


def test_fold_keyed_pools_are_accepted():
    frame = make_frame(300)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    pools = {k: frame.sample(100, random_state=k) for k in range(3)}
    mixer = SyntheticMixer(
        generators={"pool": PoolAdapter(fold_pools=pools, name="pool")},
        synthetic_share=0.3, fidelity=False,
    )
    result = CrossValidatedAugmentation(mixer, n_splits=3, verbose=False).run(X, y)
    assert len(result.folds) == 3


def test_leakage_probe_flags_a_generator_that_saw_the_holdout():
    train = make_frame(200, seed=1)
    holdout = make_frame(200, seed=2)
    leaked = pd.concat([holdout.iloc[:50], train.iloc[:50]], ignore_index=True)
    rep = fidelity_report(train, leaked, target=TARGET, holdout=holdout)
    assert rep.leakage["n_synth_matching_holdout_only"] == 50
    assert rep.leakage["synth_holdout_only_overlap"] == pytest.approx(0.5, abs=0.01)


def test_leakage_probe_is_clean_for_an_honest_generator():
    train = make_frame(200, seed=1)
    holdout = make_frame(200, seed=2)
    honest = train.sample(80, random_state=0)
    rep = fidelity_report(train, honest, target=TARGET, holdout=holdout)
    assert rep.leakage["n_synth_matching_holdout_only"] == 0


# --------------------------------------------------------------------------- #
# fidelity
# --------------------------------------------------------------------------- #
def test_fidelity_report_covers_columns_pairs_and_table():
    train = make_frame(300, seed=1)
    synth = make_frame(300, seed=2)
    rep = fidelity_report(train, synth, target=TARGET)

    metrics = set(rep.columns["metric"])
    assert {"JensenShannon", "WassersteinNormalized", "TVDistance"} <= metrics
    assert "C2ST_AUC" in rep.table
    assert 0.0 <= rep.table["C2ST_AUC"] <= 1.0
    assert "dcr_mean" in rep.table
    assert "target_rate_delta" in rep.overall
    assert not rep.column_matrix().empty


def _noise_frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "income": rng.normal(0, 1, n),
        "age": rng.normal(0, 1, n),
        "score": rng.normal(50, 10, n),
        "region": rng.choice(["x", "y"], n),
        TARGET: rng.integers(0, 2, n),
    })


@requires_sdmetrics
def test_c2st_is_two_sided_and_the_deviation_ranks_honestly():
    """Raw C2ST puts a memoriser ahead of an ideal generator; the deviation does not."""
    train = make_frame(400, seed=1)
    memorised = train.sample(300, random_state=0, replace=True)
    independent = make_frame(300, seed=9)      # same distribution, fresh draw
    noise = _noise_frame()

    mem = fidelity_report(train, memorised, target=TARGET).table
    ind = fidelity_report(train, independent, target=TARGET).table
    noi = fidelity_report(train, noise, target=TARGET).table

    # an honest sample sits at chance; both failure modes sit away from it,
    # duplicates *below* and a bad fit above
    assert abs(ind["C2ST_AUC"] - 0.5) < 0.10
    assert mem["C2ST_AUC"] < 0.45
    assert noi["C2ST_AUC"] > 0.9

    # reading the raw AUC as "lower is better" would crown the memoriser
    assert mem["C2ST_AUC"] < ind["C2ST_AUC"]

    # the deviation is the number that ranks them the way a reader means
    assert ind["C2ST_deviation"] < mem["C2ST_deviation"] < noi["C2ST_deviation"]

    # and the duplication is visible where it belongs.  NewRowSynthesis is 0.0
    # (no genuinely new rows) on the sdmetrics the battery targets; newer
    # sdmetrics (>=0.31) report nan for an all-duplicate sample — either way it
    # is not a positive "new rows" signal.
    assert mem["dcr_zero_share"] > 0.9 and (
        mem["NewRowSynthesis"] == 0.0 or pd.isna(mem["NewRowSynthesis"]))
    assert ind["dcr_zero_share"] == 0.0


def test_continuous_values_in_a_categorical_column_are_reported():
    """A generator that jitters an integer-coded category violates the schema.

    The column is categorical in the metadata and continuous in fact.  Left
    undetected this silently invalidates every category-based metric on it, and
    inflates the contingency tables behind the pair arms by orders of magnitude.
    """
    train = make_frame(300)
    train["band"] = np.repeat([1, 2, 3, 4], 75)
    synth = train.sample(300, replace=True, random_state=0).reset_index(drop=True)
    synth["band"] = synth["band"] + np.random.default_rng(0).normal(0, 0.1, 300)

    rep = fidelity_report(train, synth, target=TARGET)
    assert "band" in rep.type_violations
    assert rep.type_violations["band"]["real_cardinality"] == 4
    assert rep.type_violations["band"]["synthetic_cardinality"] > 200
    assert rep.summary()["n_type_violations"] == 1


def test_a_faithful_generator_raises_no_type_violation():
    train = make_frame(300)
    train["band"] = np.repeat([1, 2, 3, 4], 75)
    synth = train.sample(300, replace=True, random_state=0)
    assert fidelity_report(train, synth, target=TARGET).type_violations == {}


def test_a_failing_metric_does_not_kill_the_report():
    train = make_frame(100)
    broken = train.copy()
    broken["income"] = np.nan
    rep = fidelity_report(train, broken, target=TARGET)
    assert isinstance(rep.summary(), pd.Series)


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
def test_as_adapter_wraps_a_pool_and_a_duck_typed_model():
    frame = make_frame(50)
    assert isinstance(as_adapter(frame, name="p"), PoolAdapter)

    class Duck:
        def fit(self, data):
            self.data = data

        def sample(self, n):
            return self.data.iloc[:n]

    adapter = as_adapter(Duck(), name="duck")
    adapter.fit(frame)
    assert len(adapter.sample(10)) == 10


def test_as_adapter_accepts_an_uninstantiated_class():
    """The caller may hand in the model class itself, not an instance."""
    class Duck:
        def __init__(self, keep=1.0):
            self.keep = keep

        def fit(self, data):
            self.data = data

        def sample(self, n):
            return self.data.iloc[:n]

    adapter = as_adapter(Duck, name="duck", init_kwargs={"keep": 0.5})
    adapter.fit(make_frame(50))
    assert adapter.model.keep == 0.5
    assert len(adapter.sample(10)) == 10


def test_adapter_rejects_sampling_before_fitting():
    gen = EchoGenerator()
    with pytest.raises(RuntimeError, match="before fit"):
        gen.sample(5)


def test_adapter_conforms_the_generated_schema():
    class Wrong(EchoGenerator):
        def _sample(self, n, seed=None):
            return super()._sample(n, seed).drop(columns=["region"])

    gen = Wrong()
    gen.fit(make_frame(50))
    with pytest.raises(ValueError, match="missing columns"):
        gen.sample(5)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_cross_validated_run_reports_baseline_and_delta():
    frame = make_frame(400)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    mixer = SyntheticMixer(
        generators={"a": EchoGenerator("a", jitter=0.05), "b": EchoGenerator("b", jitter=0.2)},
        weights={"a": 0.5, "b": 0.5}, synthetic_share=0.7, fidelity=True,
    )
    result = CrossValidatedAugmentation(mixer, n_splits=3, verbose=False).run(X, y, dataset="toy")

    folds = result.fold_frame()
    assert len(folds) == 3
    assert folds["realized_share"].between(0.69, 0.71).all()
    assert np.isfinite(result.auc_baseline) and np.isfinite(result.auc_augmented)
    assert result.summary()["dataset"] == "toy"

    fid = result.fidelity_frame()
    assert set(fid["generator"]) >= {"a", "b", "__pooled__"}
    leak = result.leakage_frame()
    assert (leak["leak.n_synth_matching_holdout_only"] == 0).all()


# --------------------------------------------------------------------------- #
# the probe must not be fooled by representation
# --------------------------------------------------------------------------- #
class Memoriser(GeneratorAdapter):
    """Emits verbatim training rows through a float representation.

    This is what a latent or copula engine does to an integer-coded column, and
    it used to make the exact-overlap probe read clean on a generator that had
    copied every row it was given.
    """

    def __init__(self, name="mem", source=None, as_float=True):
        super().__init__(name)
        self.source = source
        self.as_float = as_float
        self._t = None

    def _reset(self):
        self._t = None

    def _fit(self, data, target, discrete_columns):
        self._t = (self.source if self.source is not None else data).copy()

    def _sample(self, n, seed=None):
        out = self._t.sample(n, replace=True, random_state=seed).reset_index(drop=True)
        if self.as_float:
            for c in out.select_dtypes("integer").columns:
                out[c] = out[c].astype(float)
        return out


@pytest.mark.parametrize("as_float", [False, True])
def test_probe_sees_memorisation_through_a_dtype_change(as_float):
    train = make_frame(200, seed=1)
    holdout = make_frame(200, seed=2)
    synth = train.sample(60, random_state=0).reset_index(drop=True)
    if as_float:
        for c in synth.select_dtypes("integer").columns:
            synth[c] = synth[c].astype(float)

    rep = fidelity_report(train, synth, target=TARGET, holdout=holdout)
    assert rep.leakage["synth_train_exact_overlap"] == 1.0, (
        "a verbatim copy of the training rows must read as full memorisation "
        "whether the engine emits ints or floats"
    )


@pytest.mark.parametrize("as_float", [False, True])
def test_probe_flags_held_out_rows_through_a_dtype_change(as_float):
    train = make_frame(200, seed=1)
    holdout = make_frame(200, seed=2)
    leaked = holdout.iloc[:40].copy()
    if as_float:
        for c in leaked.select_dtypes("integer").columns:
            leaked[c] = leaked[c].astype(float)

    rep = fidelity_report(train, leaked, target=TARGET, holdout=holdout)
    assert rep.leakage["n_synth_matching_holdout_only"] == 40


def test_overlap_shares_count_rows_not_distinct_hashes():
    """A generator emitting one memorised row a hundred times is charged for a hundred."""
    train = make_frame(200, seed=1)
    holdout = make_frame(200, seed=2)
    repeated = pd.concat([train.iloc[[0]]] * 100, ignore_index=True)
    rep = leakage_report(train, repeated, holdout)
    assert rep.leakage["synth_train_exact_overlap"] == 1.0
    assert rep.leakage["n_synth_matching_train_only"] == 100


def test_leakage_probes_survive_fidelity_being_switched_off():
    """Turning the battery off for a fleet sweep must not turn the check off."""
    frame = make_frame(300)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    mixer = SyntheticMixer(
        generators={"leaky": Memoriser("leaky")},
        synthetic_share=0.5, fidelity=False,
    )
    result = CrossValidatedAugmentation(mixer, n_splits=3, verbose=False).run(X, y)
    leak = result.leakage_frame()
    assert not leak.empty, "a delta was reported with no leakage column to check it against"
    assert set(leak["fold"]) == {0, 1, 2}
    assert (leak["leak.n_synth_matching_holdout_only"] == 0).all()


@pytest.mark.parametrize("fidelity", [False, True])
def test_leakage_probes_can_be_switched_off_explicitly(fidelity):
    """The flag is a master switch: it must bite in both modes, not just one."""
    frame = make_frame(200)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    mixer = SyntheticMixer(generators={"echo": EchoGenerator("echo")},
                           synthetic_share=0.5, fidelity=fidelity,
                           leakage_probes=False, fidelity_kwargs={"max_pairs": 2})
    result = CrossValidatedAugmentation(mixer, n_splits=2, verbose=False).run(X, y)
    assert result.leakage_frame().empty


def test_probe_sees_a_row_reproduced_in_a_different_representation():
    """Real rows define the schema, so a stringified copy is still a copy."""
    train = make_frame(200, seed=1)
    holdout = make_frame(200, seed=2)
    leaked = holdout.iloc[:30].astype(str)     # every column arrives as text
    rep = leakage_report(train, leaked, holdout)
    assert rep.leakage["n_synth_matching_holdout_only"] == 30


def test_a_holdout_missing_a_column_does_not_kill_the_probe():
    train = make_frame(120, seed=1)
    holdout = make_frame(120, seed=2).drop(columns=["region"])
    rep = leakage_report(train, train.iloc[:20], holdout)
    assert rep.leakage["n_synth_matching_holdout_only"] == 0
    assert rep.leakage["synth_train_exact_overlap"] == 1.0


# --------------------------------------------------------------------------- #
# a corrupt synthetic target is not silently relabelled
# --------------------------------------------------------------------------- #
def test_a_target_not_coded_zero_one_is_refused_up_front():
    """The caller's label coding is diagnosed as the caller's, not blamed on a generator."""
    frame = make_frame(200)
    X = frame.drop(columns=[TARGET])
    y = np.where(frame[TARGET].to_numpy() == 1, 2, 1)   # a {1,2} binary coding
    mixer = SyntheticMixer(generators={"echo": EchoGenerator("echo")},
                           synthetic_share=0.5, fidelity=False)
    with pytest.raises(ValueError, match="must be coded 0/1"):
        CrossValidatedAugmentation(mixer, n_splits=2, verbose=False).run(X, y)


def test_a_non_binary_synthetic_target_is_refused():
    frame = make_frame(200)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()

    class BadTarget(EchoGenerator):
        def _sample(self, n, seed=None):
            out = super()._sample(n, seed)
            out.loc[out.index[:3], TARGET] = np.nan
            return out

    mixer = SyntheticMixer(generators={"bad": BadTarget("bad")},
                           synthetic_share=0.5, fidelity=False)
    with pytest.raises(ValueError, match="neither 0 nor 1"):
        CrossValidatedAugmentation(mixer, n_splits=2, verbose=False).run(X, y)


# --------------------------------------------------------------------------- #
# adapter details
# --------------------------------------------------------------------------- #
def test_oversample_is_skipped_for_a_fixed_pool():
    """A pool cannot generate slack, so asking for it must not fail a valid request.

    150 rows are needed at a 30% positive rate and the 200-row pool can supply
    them, but oversample=1.5 used to ask the pool for 225 rows and fail.
    """
    frame = make_frame(350)
    pool = frame.sample(200, random_state=1)
    mixer = SyntheticMixer(
        generators={"p": PoolAdapter(fold_pools={0: pool}, name="p")},
        synthetic_share=0.3, target=TARGET, synthetic_target_rate=0.3,
        oversample=1.5, fidelity=False,
    )
    mix = mixer.build(frame, fold=0)
    assert mix.composition["per_generator"]["p"] == 150
    assert mix.composition["target_rate"]["p"] == pytest.approx(0.3, abs=0.01)


def test_a_pool_too_thin_for_the_requested_balance_says_so():
    frame = make_frame(350)
    pool = frame.sample(200, random_state=1)
    mixer = SyntheticMixer(
        generators={"p": PoolAdapter(fold_pools={0: pool}, name="p")},
        synthetic_share=0.3, target=TARGET, synthetic_target_rate=0.9,
        fidelity=False,
    )
    with pytest.raises(ValueError, match="Enlarge the pool"):
        mixer.build(frame, fold=0)


def test_repo_and_pool_paths_expand_a_tilde():
    assert _expand("~/x").is_absolute()
    assert "~" not in str(_expand("~/x"))


class _DuckModel:
    """Minimal fit/sample object, for the CallableAdapter path."""

    def fit(self, data):
        self._d = data

    def sample(self, n):
        return self._d.sample(n, replace=True).reset_index(drop=True)


def test_sampling_does_not_reseed_the_callers_global_rng():
    frame = make_frame(100)
    gen = as_adapter(_DuckModel(), name="duck")
    gen.fit(frame)
    np.random.seed(1234)
    before = np.random.rand()
    np.random.seed(1234)
    gen.sample(10, seed=7)
    assert np.random.rand() == before


# --------------------------------------------------------------------------- #
# paths the documentation advertises but nothing exercised
# --------------------------------------------------------------------------- #
def test_fixed_total_sizing_through_the_full_cv_path():
    """The plan is unit-tested; this pins the sizing end to end."""
    frame = make_frame(400)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    mixer = SyntheticMixer(generators={"echo": EchoGenerator("echo")},
                           synthetic_share=0.5, sizing="fixed_total", n_total=200,
                           fidelity=False)
    folds = CrossValidatedAugmentation(mixer, n_splits=2, verbose=False).run(X, y).fold_frame()
    assert (folds["n_real"] == 100).all() and (folds["n_synthetic"] == 100).all()
    assert (folds["realized_share"] == 0.5).all()


def test_a_pool_smaller_than_the_request_needs_allow_replacement():
    frame = make_frame(400)
    small = frame.sample(40, random_state=0)
    def mixer_with(**kw):
        return SyntheticMixer(
            generators={"p": PoolAdapter(fold_pools={0: small}, name="p", **kw)},
            synthetic_share=0.5, fidelity=False)
    with pytest.raises(ValueError, match="allow_replacement"):
        mixer_with().build(frame, fold=0)
    mix = mixer_with(allow_replacement=True).build(frame, fold=0)
    assert mix.composition["per_generator"]["p"] == 400


def test_a_mixer_naming_a_different_target_is_refused():
    mixer = SyntheticMixer(generators={"echo": EchoGenerator("echo")}, target="my_label")
    with pytest.raises(ValueError, match="the mixer names its target"):
        CrossValidatedAugmentation(mixer, verbose=False)


def test_run_fleet_isolates_a_failing_dataset():
    from pamir.synthetic import run_fleet

    mixer = SyntheticMixer(generators={"echo": EchoGenerator("echo")},
                           synthetic_share=0.3, fidelity=False)
    out = run_fleet(mixer, datasets=["south_german", "no_such_table"],
                    max_rows=300, n_splits=2, verbose=False)
    assert len(out) == 2
    ok = out[out["dataset"] == "south_german"].iloc[0]
    assert np.isfinite(ok["auc_baseline"]) and np.isfinite(ok["auc_augmented"])
    bad = out[out["dataset"] == "no_such_table"].iloc[0]
    assert isinstance(bad["error"], str) and bad["error"]


# --------------------------------------------------------------------------- #
# engine construction: settings are honoured or reported, never dropped
# --------------------------------------------------------------------------- #
class FussyEngine:
    """Takes metadata and epochs and nothing else, like a real engine."""

    def __init__(self, metadata=None, epochs=10):
        self.metadata, self.epochs = metadata, epochs

    def fit(self, data):
        self._d = data

    def sample(self, num_rows=1):
        return self._d.sample(num_rows, replace=True).reset_index(drop=True)


class TargetAwareEngine(FussyEngine):
    """Takes the credit-specific ``target`` extra, as zGAN does."""

    def __init__(self, metadata=None, epochs=10, target=None):
        super().__init__(metadata, epochs)
        self.target = target


@requires_sdv
def test_engine_parameters_reach_the_engine():
    adapter = SDVAdapter(FussyEngine, name="f", epochs=300)
    adapter.fit(make_frame(50))
    assert adapter.synthesizer.epochs == 300


@requires_sdv
def test_one_unsupported_parameter_does_not_silently_discard_the_rest():
    """The old fallback built a default engine and said nothing."""
    adapter = SDVAdapter(FussyEngine, name="f", epochs=300, batch_size=512)
    with pytest.raises(TypeError, match="rejected the parameters"):
        adapter.fit(make_frame(50))


@requires_sdv
def test_the_rejection_names_what_the_caller_asked_for():
    adapter = SDVAdapter(FussyEngine, name="f", epochs=300, batch_size=512)
    with pytest.raises(TypeError, match=r"\['batch_size', 'epochs'\]"):
        adapter.fit(make_frame(50))


@requires_sdv
def test_an_injected_target_is_withdrawn_quietly_but_a_requested_one_is_not():
    """``target`` is the adapter's own extra, so only it may be given up."""
    plain = SDVAdapter(FussyEngine, name="f", epochs=7)
    plain.fit(make_frame(50), target=TARGET)          # engine takes no target
    assert plain.synthesizer.epochs == 7

    aware = SDVAdapter(TargetAwareEngine, name="f", epochs=7)
    aware.fit(make_frame(50), target=TARGET)
    assert aware.synthesizer.target == TARGET

    asked = SDVAdapter(FussyEngine, name="f", target="mine")
    with pytest.raises(TypeError, match="rejected the parameters"):
        asked.fit(make_frame(50), target=TARGET)


@requires_sdv
def test_parameters_alongside_a_built_instance_are_refused():
    adapter = SDVAdapter(FussyEngine(epochs=10), name="f", epochs=300)
    with pytest.raises(ValueError, match="already-built"):
        adapter.fit(make_frame(50))


@requires_sdv
def test_an_adapter_can_be_refitted():
    """The guard must not mistake the engine the adapter built for the caller's."""
    adapter = SDVAdapter(FussyEngine, name="f", epochs=300)
    adapter.fit(make_frame(50))
    first = adapter.synthesizer
    adapter.fit(make_frame(50, seed=3))
    assert adapter.synthesizer.epochs == 300
    assert adapter.synthesizer is not first, "a refit must start from the prototype"


@requires_sdv
def test_a_mixer_can_be_rebuilt():
    frame = make_frame(80)
    mixer = SyntheticMixer(generators={"f": SDVAdapter(FussyEngine, name="f", epochs=300)},
                           synthetic_share=0.5, target=TARGET, fidelity=False)
    assert len(mixer.build(frame).data) == len(mixer.build(frame).data)


def test_an_all_synthetic_frame_reports_no_real_target_rate():
    frame = make_frame(200)
    mixer = SyntheticMixer(generators={"echo": EchoGenerator("echo")},
                           sizing="fixed_total", n_total=100, n_synthetic=100,
                           target=TARGET, fidelity=False)
    comp = mixer.build(frame).composition
    assert comp["n_real"] == 0 and comp["n_synthetic"] == 100
    assert "real" not in comp["target_rate"], "a rate over zero rows is not a rate"
    assert "echo" in comp["target_rate"]


def test_zero_synthetic_rows_is_a_valid_request():
    frame = make_frame(200)
    mixer = SyntheticMixer(generators={"echo": EchoGenerator("echo")},
                           n_synthetic=0, target=TARGET, fidelity=False)
    mix = mixer.build(frame)
    assert mix.composition["n_synthetic"] == 0
    assert mix.composition["realized_synthetic_share"] == 0.0
    assert len(mix.data) == len(frame)


@requires_sdv
def test_a_built_instance_without_parameters_is_fine():
    adapter = SDVAdapter(FussyEngine(epochs=42), name="f")
    adapter.fit(make_frame(50))
    assert adapter.synthesizer.epochs == 42
    assert len(adapter.sample(10)) == 10


# --------------------------------------------------------------------------- #
# an exact synthetic row count, as an alternative to a share
# --------------------------------------------------------------------------- #
def test_plan_takes_an_exact_count_instead_of_a_share():
    plan = plan_mixture(1000, weights={"a": 0.5, "b": 0.5}, n_synthetic=300)
    assert plan.n_real == 1000
    assert plan.n_synthetic == 300
    assert plan.per_generator == {"a": 150, "b": 150}
    assert plan.requested_share is None
    assert plan.requested_n_synthetic == 300


def test_plan_takes_an_exact_count_under_fixed_total():
    plan = plan_mixture(1000, weights={"a": 1.0}, sizing="fixed_total",
                        n_total=500, n_synthetic=200)
    assert (plan.n_real, plan.n_synthetic) == (300, 200)


def test_plan_rejects_a_count_larger_than_the_frame():
    with pytest.raises(ValueError, match="cannot be larger than the frame"):
        plan_mixture(1000, weights={"a": 1.0}, sizing="fixed_total",
                     n_total=500, n_synthetic=600)


def test_a_share_and_a_count_cannot_both_be_given():
    with pytest.raises(ValueError, match="cannot both hold"):
        plan_mixture(1000, 0.5, {"a": 1.0}, n_synthetic=300)
    with pytest.raises(ValueError, match="cannot both hold"):
        SyntheticMixer(generators={"e": EchoGenerator("e")},
                       synthetic_share=0.5, n_synthetic=300)


def test_the_default_is_still_a_half_share_when_neither_is_given():
    plan = plan_mixture(1000, weights={"a": 1.0})
    assert plan.n_synthetic == 1000 and plan.realized_share == 0.5


def test_an_exact_count_is_identical_in_every_fold():
    """A share resolves against each fold's train split; a count does not."""
    frame = make_frame(403)                     # deliberately not divisible by 3
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].to_numpy()
    mixer = SyntheticMixer(generators={"a": EchoGenerator("a"), "b": EchoGenerator("b")},
                           weights={"a": 0.25, "b": 0.75}, n_synthetic=200,
                           fidelity=False)
    folds = CrossValidatedAugmentation(mixer, n_splits=3, verbose=False).run(X, y).fold_frame()
    assert (folds["n_synthetic"] == 200).all()
    comp = CrossValidatedAugmentation(mixer, n_splits=3, verbose=False).run(X, y)
    assert comp.folds[0].composition["per_generator"] == {"a": 50, "b": 150}
    assert comp.folds[0].composition["requested_n_synthetic"] == 200
    assert comp.folds[0].composition["requested_synthetic_share"] is None


# --------------------------------------------------------------------------- #
# CallableAdapter: one destination per dictionary
# --------------------------------------------------------------------------- #
class Recorded:
    """Records what reached the constructor, fit() and sample()."""

    def __init__(self, epochs=1):
        self.epochs, self.fit_seen, self.sample_seen = epochs, None, None

    def fit(self, data, **kw):
        self._d, self.fit_seen = data, kw

    def sample(self, n, **kw):
        self.sample_seen = kw
        return self._d.sample(n, replace=True).reset_index(drop=True)


def test_init_kwargs_go_to_the_constructor_and_fit_kwargs_to_fit():
    adapter = CallableAdapter(Recorded, name="m", init_kwargs={"epochs": 50},
                              fit_kwargs={"warm": True}, sample_kwargs={"t": 2})
    adapter.fit(make_frame(40))
    assert adapter.model.epochs == 50
    assert adapter.model.fit_seen == {"warm": True}
    adapter.sample(5)
    assert adapter.model.sample_seen == {"t": 2}


def test_fit_kwargs_mean_the_same_thing_for_a_class_and_an_instance():
    from_class = CallableAdapter(Recorded, name="m", fit_kwargs={"warm": True})
    from_class.fit(make_frame(40))
    from_instance = CallableAdapter(Recorded(epochs=1), name="m", fit_kwargs={"warm": True})
    from_instance.fit(make_frame(40))
    assert from_class.model.fit_seen == from_instance.model.fit_seen == {"warm": True}


def test_callable_kwargs_do_not_change_destination_on_a_refit():
    """The second fit used to send fit_kwargs somewhere else than the first."""
    adapter = CallableAdapter(Recorded, name="m", init_kwargs={"epochs": 50},
                              fit_kwargs={"warm": True})
    adapter.fit(make_frame(40))
    first = (adapter.model.epochs, adapter.model.fit_seen)
    adapter.fit(make_frame(40, seed=3))
    assert (adapter.model.epochs, adapter.model.fit_seen) == first == (50, {"warm": True})


def test_init_kwargs_alongside_a_built_instance_are_refused():
    adapter = CallableAdapter(Recorded(epochs=1), name="m", init_kwargs={"epochs": 50})
    with pytest.raises(ValueError, match="already-built"):
        adapter.fit(make_frame(40))


def test_run_fleet_refuses_a_misconfigured_mixer_before_the_sweep():
    from pamir.synthetic import run_fleet

    mixer = SyntheticMixer(generators={"echo": EchoGenerator("echo")}, target="mine")
    with pytest.raises(ValueError, match="the mixer names its target"):
        run_fleet(mixer, datasets=["south_german"], max_rows=200, n_splits=2, verbose=False)


# --------------------------------------------------------------------------- #
# no training a generator whose planned contribution is zero
# --------------------------------------------------------------------------- #
class CountingGenerator(EchoGenerator):
    """Records every fit, standing in for an engine where fitting is the cost."""

    log: list = []

    def _fit(self, data, target, discrete_columns):
        CountingGenerator.log.append(self.name)
        super()._fit(data, target, discrete_columns)


def test_a_generator_weighted_to_zero_rows_is_not_trained():
    CountingGenerator.log = []
    frame = make_frame(200)
    mixer = SyntheticMixer(
        generators={"used": CountingGenerator("used"), "unused": CountingGenerator("unused")},
        weights={"used": 1.0, "unused": 0.0},
        synthetic_share=0.5, target=TARGET, fidelity=False)
    mix = mixer.build(frame)
    assert mix.composition["per_generator"] == {"used": 200, "unused": 0}
    assert CountingGenerator.log == ["used"], "a generator drawing no rows was trained"


def test_no_generator_is_trained_when_no_synthetic_rows_are_wanted():
    CountingGenerator.log = []
    mixer = SyntheticMixer(generators={"a": CountingGenerator("a"), "b": CountingGenerator("b")},
                           n_synthetic=0, target=TARGET, fidelity=False)
    mixer.build(make_frame(200))
    assert CountingGenerator.log == []


def test_fit_on_its_own_still_trains_everything():
    """The public fit() keeps its contract; only build() narrows the set."""
    CountingGenerator.log = []
    mixer = SyntheticMixer(generators={"a": CountingGenerator("a"), "b": CountingGenerator("b")},
                           weights={"a": 1.0, "b": 0.0}, target=TARGET, fidelity=False)
    mixer.fit(make_frame(200))
    assert sorted(CountingGenerator.log) == ["a", "b"]


# --------------------------------------------------------------------------- #
# failures that name what is actually wrong
# --------------------------------------------------------------------------- #
def test_a_pool_adapter_without_a_pool_says_so():
    mixer = SyntheticMixer(generators={"p": PoolAdapter(name="p")},
                           synthetic_share=0.5, target=TARGET, fidelity=False)
    with pytest.raises(ValueError, match="no rows to draw from"):
        mixer.build(make_frame(100))


def test_a_missing_repo_path_is_reported_and_not_put_on_sys_path():
    import sys
    from pamir.synthetic import ZedgeAdapter

    missing = "/nonexistent/pamir/audit/repo"
    adapter = ZedgeAdapter(repo_path=missing, mode="artifact", artifact_dir="/nowhere")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        adapter.fit(make_frame(50))
    assert missing not in sys.path, "a path that does not exist was added to sys.path"
