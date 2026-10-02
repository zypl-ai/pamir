"""Mix real and synthetic training rows in stated proportions, and score them.

    mixer = SyntheticMixer(
        generators={"zgan": zgan_adapter, "zedge": zedge_adapter},
        weights={"zgan": 0.5, "zedge": 0.5},   # split of the synthetic part
        synthetic_share=0.7,                   # 70% synthetic / 30% real
        target="__target__",
    )
    mix = mixer.build(train_frame)             # train rows of ONE fold
    mix.data                                   # the mixed training frame
    mix.composition                            # what was actually combined
    mix.fidelity["zgan"].summary()             # fidelity of that engine's rows

The mixer is deliberately fold-local: it fits generators on the frame it is
handed and nothing else.  Cross-validation lives in :mod:`pamir.synthetic.cv`,
which clones the mixer per fold so no generator state survives a fold boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from pamir.synthetic.adapters import GeneratorAdapter, as_adapter
from pamir.synthetic.fidelity import (
    FidelityReport,
    build_metadata,
    fidelity_report,
    leakage_report,
)
from pamir.synthetic.mixing import (
    KEEP_REAL,
    MixturePlan,
    plan_mixture,
    stratified_subsample,
    take_rows,
)

REAL = "real"


@dataclass
class MixResult:
    """A built training frame plus the record of how it was built."""

    data: pd.DataFrame
    sources: pd.Series
    plan: MixturePlan
    composition: Dict[str, Any]
    fidelity: Dict[str, FidelityReport] = field(default_factory=dict)
    synthetic: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:
        """Features of the mixed frame (target dropped)."""
        tgt = self.composition.get("target")
        return self.data.drop(columns=[tgt]) if tgt and tgt in self.data.columns else self.data

    @property
    def y(self) -> Optional[np.ndarray]:
        """Target of the mixed frame."""
        tgt = self.composition.get("target")
        if not tgt or tgt not in self.data.columns:
            return None
        return self.data[tgt].to_numpy()

    def fidelity_frame(self) -> pd.DataFrame:
        """One row per generator with every headline fidelity number."""
        if not self.fidelity:
            return pd.DataFrame()
        rows = {name: rep.summary() for name, rep in self.fidelity.items()}
        return pd.DataFrame(rows).T.rename_axis("generator").reset_index()

    def to_dict(self) -> Dict:
        return {
            "composition": self.composition,
            "plan": self.plan.to_dict(),
            "fidelity": {k: v.to_dict() for k, v in self.fidelity.items()},
        }


class SyntheticMixer:
    """Build a training frame from real rows plus several synthetic engines.

    Parameters
    ----------
    generators : mapping
        ``{name: generator}``.  Each value may be a
        :class:`~pamir.synthetic.adapters.GeneratorAdapter`, an SDV synthesizer
        class or instance (zGAN is one), a pre-generated pool, or any object
        with ``fit``/``sample`` — :func:`as_adapter` wraps it.
    weights : mapping, optional
        How the synthetic part splits across generators, e.g.
        ``{"zgan": 0.5, "zedge": 0.5}``.  Defaults to equal weights.
    synthetic_share : float, optional
        Share of synthetic rows in the mixed frame, in [0, 1).  Defaults to
        ``0.5`` when neither this nor ``n_synthetic`` is given.
    n_synthetic : int, optional
        Exact number of synthetic rows, in place of a share.  Under
        cross-validation this is per fold, so the count is identical in every
        fold, whereas a share resolves against each fold's train split and
        therefore varies with it.  Mutually exclusive with ``synthetic_share``.
    target : str, optional
        Target column.  Enables stratification, class-balance control and the
        ML-efficacy fidelity arm.
    sizing : {"keep_real", "fixed_total"}
        ``keep_real`` adds synthetic rows on top of every real row.
        ``fixed_total`` pins the frame to ``n_total`` rows.
    stratify : bool
        Preserve the target distribution when subsampling real or synthetic rows.
    synthetic_target_rate : float, optional
        Force the synthetic part to this positive rate (e.g. ``0.5`` to rebalance
        a 3%-default book).  Requires oversampling; raises if the pool cannot
        supply enough rows of a class.
    oversample : float
        Multiplier on how many rows each generator is asked for, so that
        class-balance selection has slack.  Skipped for a fixed pool, which
        cannot generate more rows than it holds.
    fidelity : bool
        Compute fidelity metrics for each generator's rows against the real
        training rows.  A full report costs roughly 11 s per generator per fold
        on a 1k x 20 table, so switch it off for a large fleet sweep and
        measure fidelity separately.  The leakage probes are *not* switched off
        with it — see ``leakage_probes``.
    leakage_probes : bool
        Run the exact-overlap probes whenever a holdout frame is given, whatever
        ``fidelity`` is set to.  They cost milliseconds, and a delta reported
        without them cannot be checked, so switching the battery off for a fleet
        sweep leaves them running.  Setting this to False is the only way to
        stop them, in either mode.
    pooled_fidelity : bool
        Also score the concatenation of every generator's rows as ``__pooled__``.
        Switch off to save one report per fold.
    random_state : int
        Seed for every selection step; generators receive derived seeds.
    """

    def __init__(
        self,
        generators: Mapping[str, Any],
        weights: Optional[Mapping[str, float]] = None,
        synthetic_share: Optional[float] = None,
        n_synthetic: Optional[int] = None,
        target: Optional[str] = None,
        sizing: str = KEEP_REAL,
        n_total: Optional[int] = None,
        stratify: bool = True,
        synthetic_target_rate: Optional[float] = None,
        oversample: float = 1.5,
        allow_replacement: bool = False,
        fidelity: bool = True,
        fidelity_kwargs: Optional[Dict] = None,
        pooled_fidelity: bool = True,
        leakage_probes: bool = True,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        if not generators:
            raise ValueError("At least one generator is required.")
        self.adapters: Dict[str, GeneratorAdapter] = {
            name: as_adapter(gen, name=name) for name, gen in generators.items()
        }
        self.weights = dict(weights) if weights else {name: 1.0 for name in self.adapters}
        unknown = set(self.weights) - set(self.adapters)
        if unknown:
            raise ValueError(f"Weights name generators that were not passed: {sorted(unknown)}")
        missing = set(self.adapters) - set(self.weights)
        if missing:
            raise ValueError(f"No weight given for generators: {sorted(missing)}")

        if synthetic_share is not None and n_synthetic is not None:
            raise ValueError(
                f"synthetic_share={synthetic_share} and n_synthetic={n_synthetic} were "
                "both given, and they cannot both hold. Pass a share or a count."
            )
        self.synthetic_share = synthetic_share
        self.n_synthetic = n_synthetic
        self.target = target
        self.sizing = sizing
        self.n_total = n_total
        self.stratify = stratify
        self.synthetic_target_rate = synthetic_target_rate
        self.oversample = max(1.0, float(oversample))
        self.allow_replacement = allow_replacement
        self.fidelity = fidelity
        self.fidelity_kwargs = dict(fidelity_kwargs or {})
        self.pooled_fidelity = pooled_fidelity
        self.leakage_probes = leakage_probes
        self.shuffle = shuffle
        self.random_state = random_state

    # -- lifecycle --------------------------------------------------------- #
    def clone(self) -> "SyntheticMixer":
        """A copy with unfitted generators, for the next CV fold."""
        fresh = SyntheticMixer(
            generators={name: ad.clone() for name, ad in self.adapters.items()},
            weights=self.weights,
            synthetic_share=self.synthetic_share,
            n_synthetic=self.n_synthetic,
            target=self.target,
            sizing=self.sizing,
            n_total=self.n_total,
            stratify=self.stratify,
            synthetic_target_rate=self.synthetic_target_rate,
            oversample=self.oversample,
            allow_replacement=self.allow_replacement,
            fidelity=self.fidelity,
            fidelity_kwargs=self.fidelity_kwargs,
            pooled_fidelity=self.pooled_fidelity,
            leakage_probes=self.leakage_probes,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        return fresh

    def fit(self, train: pd.DataFrame, fold: Optional[int] = None,
            only: Optional[Sequence[str]] = None) -> "SyntheticMixer":
        """Fit the generators on ``train`` — and only on ``train``.

        ``only`` restricts the work to the named generators.  :meth:`build`
        passes the ones its plan actually draws from, so a generator weighted to
        zero rows is not trained for nothing: for a GAN or a diffusion model
        that is a full training run per fold, thrown away.
        """
        names = list(self.adapters) if only is None else list(only)
        for name in names:
            adapter = self.adapters[name]
            if hasattr(adapter, "fold"):
                adapter.fold = fold
            adapter.fit(train, target=self.target)
        return self

    # -- building ---------------------------------------------------------- #
    def build(
        self,
        train: pd.DataFrame,
        holdout: Optional[pd.DataFrame] = None,
        fold: Optional[int] = None,
        fit: bool = True,
    ) -> MixResult:
        """Fit (optionally), generate, mix, and score.

        Parameters
        ----------
        train : DataFrame
            Real training rows including the target column.  This is the only
            data any generator ever sees.
        holdout : DataFrame, optional
            Held-out rows, passed to the fidelity report for leakage probes
            only — never used to fit, generate or score fidelity itself.
        fold : int, optional
            Fold index, forwarded to pool-based adapters so they pick the pool
            that belongs to this fold.
        fit : bool
            Set False to reuse generators already fitted on this same frame.
        """
        if self.target and self.target not in train.columns:
            raise ValueError(
                f"target '{self.target}' is not a column of the training frame; "
                f"got {list(train.columns)[:8]}..."
            )
        rng = np.random.default_rng(self.random_state)
        train = train.reset_index(drop=True)

        plan = plan_mixture(
            n_real=len(train),
            synthetic_share=self.synthetic_share,
            weights=self.weights,
            sizing=self.sizing,
            n_total=self.n_total,
            n_synthetic=self.n_synthetic,
        )

        if fit:
            drawn = [name for name, n_rows in plan.per_generator.items() if n_rows > 0]
            self.fit(train, fold=fold, only=drawn)

        strata = train[self.target] if (self.stratify and self.target) else None
        real_part = stratified_subsample(train, plan.n_real, rng, strata)

        synthetic: Dict[str, pd.DataFrame] = {}
        for i, (name, n_rows) in enumerate(plan.per_generator.items()):
            if n_rows <= 0:
                synthetic[name] = train.iloc[:0].copy()
                continue
            synthetic[name] = self._draw(self.adapters[name], n_rows, seed=self.random_state + i, rng=rng)

        frames = [real_part] + [f for f in synthetic.values() if len(f)]
        labels = [pd.Series([REAL] * len(real_part))] + [
            pd.Series([name] * len(f)) for name, f in synthetic.items() if len(f)
        ]
        mixed = pd.concat(frames, ignore_index=True)
        sources = pd.concat(labels, ignore_index=True)

        if self.shuffle:
            order = rng.permutation(len(mixed))
            mixed = mixed.iloc[order].reset_index(drop=True)
            sources = sources.iloc[order].reset_index(drop=True)

        composition = self._composition(plan, real_part, synthetic, fold)
        probe_holdout = holdout if self.leakage_probes else None
        if self.fidelity:
            reports = self._fidelity(train, synthetic, probe_holdout)
        elif probe_holdout is not None:
            reports = self._probes(train, synthetic, probe_holdout)
        else:
            reports = {}

        return MixResult(
            data=mixed,
            sources=sources,
            plan=plan,
            composition=composition,
            fidelity=reports,
            synthetic=synthetic,
        )

    # -- internals --------------------------------------------------------- #
    def _draw(self, adapter: GeneratorAdapter, n_rows: int, seed: int, rng) -> pd.DataFrame:
        """Ask one generator for ``n_rows``, honouring the class-balance request."""
        if self.synthetic_target_rate is None:
            return adapter.sample(n_rows, seed=seed)

        if not self.target:
            raise ValueError("synthetic_target_rate requires target to be set.")
        if adapter.fixed_pool:
            # A pool cannot generate slack, so oversampling it only turns a
            # satisfiable request into a failure.  The pool itself is the
            # candidate set: draw all of it and select the class quotas there.
            draw = max(adapter.n_available or n_rows, n_rows)
        else:
            draw = int(np.ceil(n_rows * self.oversample))
        pool = adapter.sample(draw, seed=seed)
        y = pd.to_numeric(pool[self.target], errors="coerce")
        n_pos = int(round(n_rows * self.synthetic_target_rate))
        n_neg = n_rows - n_pos
        pos, neg = pool[y == 1], pool[y == 0]
        if len(pos) < n_pos or len(neg) < n_neg:
            remedy = ("Enlarge the pool or ask for a rate it can supply."
                      if adapter.fixed_pool else "Raise oversample.")
            raise ValueError(
                f"[{adapter.name}] drew {len(pos)} positive / {len(neg)} negative rows "
                f"from {len(pool)} but {n_pos}/{n_neg} are needed for "
                f"synthetic_target_rate={self.synthetic_target_rate}. {remedy}"
            )
        picked = pd.concat([
            take_rows(pos, n_pos, rng, allow_replacement=self.allow_replacement),
            take_rows(neg, n_neg, rng, allow_replacement=self.allow_replacement),
        ], ignore_index=True)
        return picked

    def _composition(self, plan, real_part, synthetic, fold) -> Dict[str, Any]:
        comp: Dict[str, Any] = {
            "fold": fold,
            "target": self.target,
            "sizing": self.sizing,
            "requested_synthetic_share": plan.requested_share,
            "requested_n_synthetic": plan.requested_n_synthetic,
            "weights": plan.weights,
            "n_real": len(real_part),
            "n_synthetic": int(sum(len(f) for f in synthetic.values())),
            "per_generator": {k: len(v) for k, v in synthetic.items()},
        }
        comp["n_total"] = comp["n_real"] + comp["n_synthetic"]
        comp["realized_synthetic_share"] = (
            comp["n_synthetic"] / comp["n_total"] if comp["n_total"] else 0.0
        )
        if self.target:
            rates = {}
            if len(real_part):
                rates[REAL] = float(pd.to_numeric(real_part[self.target], errors="coerce").mean())
            for name, frame in synthetic.items():
                if len(frame):
                    rates[name] = float(pd.to_numeric(frame[self.target], errors="coerce").mean())
            comp["target_rate"] = rates
        return comp

    def _probes(self, train, synthetic, holdout) -> Dict[str, FidelityReport]:
        """Leakage probes only — the cheap half of the report, always affordable."""
        return {
            name: leakage_report(train, frame, holdout)
            for name, frame in synthetic.items() if len(frame)
        }

    def _fidelity(self, train, synthetic, holdout) -> Dict[str, FidelityReport]:
        """Score each engine's rows, then the pooled synthetic part, against train."""
        metadata = build_metadata(train, target=self.target)
        reports: Dict[str, FidelityReport] = {}
        for name, frame in synthetic.items():
            if not len(frame):
                continue
            reports[name] = fidelity_report(
                train, frame, metadata=metadata, target=self.target,
                holdout=holdout, **self.fidelity_kwargs
            )
            native = self.adapters[name].native_fidelity(train, frame)
            if native:
                reports[name].native[name] = native

        nonempty = [f for f in synthetic.values() if len(f)]
        if self.pooled_fidelity and len(nonempty) > 1:
            pooled = pd.concat(nonempty, ignore_index=True)
            reports["__pooled__"] = fidelity_report(
                train, pooled, metadata=metadata, target=self.target,
                holdout=holdout, **self.fidelity_kwargs
            )
        return reports
