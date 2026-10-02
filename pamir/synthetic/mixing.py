"""Proportion arithmetic and row assembly for a real/synthetic training mix.

The caller states two things: how much of the training frame should be
synthetic, and how that synthetic part splits across generators.  Everything
else — exact row counts, remainder handling, class balance — is resolved here
so that the mixer itself never does arithmetic inline.

    plan = plan_mixture(n_real=1000, synthetic_share=0.7,
                        weights={"zgan": 0.5, "zedge": 0.5})
    plan.n_real            # 1000   (every real row kept)
    plan.per_generator     # {"zgan": 1166, "zedge": 1167}  (2333 rows in total)
    plan.realized_share    # 0.69997 — 0.7 to the nearest whole row
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

KEEP_REAL = "keep_real"
FIXED_TOTAL = "fixed_total"
SIZINGS = (KEEP_REAL, FIXED_TOTAL)


@dataclass(frozen=True)
class MixturePlan:
    """How many rows come from where, resolved to exact integers."""

    n_real: int
    per_generator: Dict[str, int]
    sizing: str
    requested_share: Optional[float]
    weights: Dict[str, float]
    n_real_available: int
    requested_n_synthetic: Optional[int] = None

    @property
    def n_synthetic(self) -> int:
        return int(sum(self.per_generator.values()))

    @property
    def n_total(self) -> int:
        return self.n_real + self.n_synthetic

    @property
    def realized_share(self) -> float:
        """Synthetic share actually achieved after integer rounding."""
        return self.n_synthetic / self.n_total if self.n_total else 0.0

    def to_dict(self) -> Dict:
        return {
            "sizing": self.sizing,
            "n_real_available": self.n_real_available,
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "n_total": self.n_total,
            "per_generator": dict(self.per_generator),
            "requested_share": self.requested_share,
            "requested_n_synthetic": self.requested_n_synthetic,
            "realized_share": round(self.realized_share, 6),
            "weights": dict(self.weights),
        }


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Scale generator weights to sum to 1, preserving key order."""
    if not weights:
        raise ValueError("At least one generator weight is required.")
    items = list(weights.items())
    for name, w in items:
        if w < 0:
            raise ValueError(f"Weight for '{name}' is negative ({w}).")
    total = float(sum(w for _, w in items))
    if total <= 0:
        raise ValueError("Generator weights sum to zero.")
    return {name: float(w) / total for name, w in items}


def _largest_remainder(total: int, weights: Mapping[str, float]) -> Dict[str, int]:
    """Split ``total`` across weights so the parts sum to exactly ``total``."""
    exact = {name: total * w for name, w in weights.items()}
    floors = {name: int(np.floor(v)) for name, v in exact.items()}
    short = total - sum(floors.values())
    if short:
        order = sorted(exact, key=lambda k: (-(exact[k] - floors[k]), k))
        for name in order[:short]:
            floors[name] += 1
    return floors


def plan_mixture(
    n_real: int,
    synthetic_share: Optional[float] = None,
    weights: Optional[Mapping[str, float]] = None,
    sizing: str = KEEP_REAL,
    n_total: Optional[int] = None,
    n_synthetic: Optional[int] = None,
) -> MixturePlan:
    """Resolve a proportion — or a row count — into exact row counts.

    Parameters
    ----------
    n_real : int
        Real training rows available (one fold's train split, not the table).
    synthetic_share : float, optional
        Target share of synthetic rows in the mixed frame, in [0, 1).
        ``0.7`` means 70% synthetic / 30% real.  Defaults to ``0.5`` when
        neither this nor ``n_synthetic`` is given.
    weights : mapping
        Relative weights per generator, e.g. ``{"zgan": 0.5, "zedge": 0.5}``.
        Need not sum to 1; they are normalized.
    sizing : {"keep_real", "fixed_total"}
        ``keep_real`` (default) keeps every real row and adds synthetic rows on
        top until the share is met — the mix never throws real data away.
        ``fixed_total`` pins the mixed frame to ``n_total`` rows and subsamples
        the real part, which is what an ablation at constant training size needs.
    n_total : int, optional
        Required for ``fixed_total``.
    n_synthetic : int, optional
        Exact number of synthetic rows, in place of a share.  Under CV this is
        per fold, so the count is identical in every fold whereas a share is
        not.  Mutually exclusive with ``synthetic_share``: naming both would
        mean one of them silently losing.

    Returns
    -------
    MixturePlan
    """
    if sizing not in SIZINGS:
        raise ValueError(f"sizing must be one of {SIZINGS}, got {sizing!r}.")
    if synthetic_share is not None and n_synthetic is not None:
        raise ValueError(
            f"synthetic_share={synthetic_share} and n_synthetic={n_synthetic} were "
            "both given, and they cannot both hold. Pass a share or a count, not both."
        )
    if n_synthetic is not None and n_synthetic < 0:
        raise ValueError(f"n_synthetic must not be negative, got {n_synthetic}.")
    if synthetic_share is None and n_synthetic is None:
        synthetic_share = 0.5
    if synthetic_share is not None and not 0.0 <= synthetic_share < 1.0:
        raise ValueError(
            f"synthetic_share must be in [0, 1), got {synthetic_share}. "
            "A share of 1.0 means no real rows at all; use fixed_total with "
            "n_real=0 if that is really what you want."
        )
    if n_real <= 0:
        raise ValueError(f"n_real must be positive, got {n_real}.")

    w = normalize_weights(weights)

    if sizing == KEEP_REAL:
        keep_real_rows = n_real
        n_synth = (int(n_synthetic) if n_synthetic is not None
                   else int(round(n_real * synthetic_share / (1.0 - synthetic_share))))
    else:
        if n_total is None:
            raise ValueError("sizing='fixed_total' requires n_total.")
        if n_total <= 0:
            raise ValueError(f"n_total must be positive, got {n_total}.")
        n_synth = (int(n_synthetic) if n_synthetic is not None
                   else int(round(n_total * synthetic_share)))
        if n_synth > n_total:
            raise ValueError(
                f"n_synthetic={n_synth} exceeds n_total={n_total}; the synthetic part "
                "cannot be larger than the frame it belongs to."
            )
        keep_real_rows = n_total - n_synth
        if keep_real_rows > n_real:
            asked = (f"n_synthetic={n_synth}" if n_synthetic is not None
                     else f"synthetic_share={synthetic_share}")
            raise ValueError(
                f"n_total={n_total} at {asked} needs "
                f"{keep_real_rows} real rows but only {n_real} are available."
            )

    per_gen = _largest_remainder(n_synth, w) if n_synth else {k: 0 for k in w}
    return MixturePlan(
        n_real=keep_real_rows,
        per_generator=per_gen,
        sizing=sizing,
        requested_share=None if synthetic_share is None else float(synthetic_share),
        weights=w,
        n_real_available=int(n_real),
        requested_n_synthetic=None if n_synthetic is None else int(n_synthetic),
    )


def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    strata: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Take ``n`` rows, preserving the distribution of ``strata`` when given."""
    if n >= len(df):
        return df
    if strata is None:
        idx = rng.choice(len(df), size=n, replace=False)
        return df.iloc[np.sort(idx)]

    strata = pd.Series(np.asarray(strata), index=df.index)
    counts = strata.value_counts()
    quota = _largest_remainder(n, {k: v / len(strata) for k, v in counts.items()})
    picks = []
    for level, take in quota.items():
        pool = np.flatnonzero((strata == level).to_numpy())
        take = min(take, len(pool))
        if take:
            picks.append(rng.choice(pool, size=take, replace=False))
    if not picks:
        return df.iloc[:0]
    pos = np.sort(np.concatenate(picks))
    return df.iloc[pos]


def take_rows(
    pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    strata: Optional[pd.Series] = None,
    allow_replacement: bool = False,
) -> pd.DataFrame:
    """Draw ``n`` rows from ``pool``, with replacement only if explicitly allowed."""
    if n <= 0:
        return pool.iloc[:0]
    if n <= len(pool):
        return stratified_subsample(pool, n, rng, strata)
    if not allow_replacement:
        raise ValueError(
            f"Pool holds {len(pool)} rows but {n} were requested. "
            "Generate a larger pool or pass allow_replacement=True."
        )
    idx = rng.choice(len(pool), size=n, replace=True)
    return pool.iloc[idx]
