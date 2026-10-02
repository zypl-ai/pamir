"""Leak-safe cross-validated augmentation under a fixed protocol.

The protocol: stratified 5-fold, ``shuffle=True``, seed 42, out-of-fold
predictions pooled into a single AUC.
This module wraps a fold loop around :class:`SyntheticMixer` so the question
"does synthetic data help?" is answered under exactly that protocol.

What makes it leak-safe
-----------------------
1. The split happens **first**.  A generator is fitted inside the fold, on the
   training rows of that fold only; the held-out rows are never passed to
   ``fit``, to ``sample``, or to the fidelity report as reference data.
2. A **fresh generator per fold**.  The mixer is cloned and every adapter reset,
   so nothing learned on fold *k*'s training rows survives into fold *k+1*,
   where those rows are test data.
3. **The test split is never augmented.**  Synthetic rows enter training only;
   scoring always happens on real held-out rows.
4. **Pools and artifacts are fold-keyed.**  An offline pool or a pre-trained
   zEDGE artifact is rejected unless it is registered per fold or the caller
   explicitly vouches that it came from this fold's training split.
5. **Probes, not promises.**  Each fold reports the share of synthetic rows that
   exactly match a held-out row that is not also in train.  Anything above zero
   means the contract was broken somewhere upstream.

Column typing (``prepare``): numerics coerced with ``errors="coerce"`` and
infinities nulled, categoricals cast to ``str`` — which turns a missing value into
the literal ``"nan"`` or ``"None"``.  That typing is stateless — it reads no statistic from
the data — so applying it before the split leaks nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pamir.synthetic.fidelity import (
    HEADLINE_FIDELITY_COLUMNS,
    FidelityReport,
)
from pamir.synthetic.mixer import SyntheticMixer

TARGET = "__target__"


# --------------------------------------------------------------------------- #
# typing and the default model arm
# --------------------------------------------------------------------------- #
def split_columns(df: pd.DataFrame, target: str = TARGET) -> tuple:
    """Numeric and categorical feature names, by upstream's rule."""
    feats = [c for c in df.columns if c != target]
    cat = [c for c in feats
           if df[c].dtype == "object" or str(df[c].dtype) in ("category", "string", "bool")]
    num = [c for c in feats if c not in cat]
    return num, cat


def prepare(df: pd.DataFrame, num: Sequence[str], cat: Sequence[str]) -> pd.DataFrame:
    """Stateless column typing, so split-order safe.

    Note that ``astype(str)`` runs *before* any missing-value sentinel could be
    applied, so a missing categorical becomes the literal string ``"nan"`` or
    ``"None"`` rather than a single ``__NA__`` level.  This is kept deliberately:
    changing it would relabel categories and change the numbers between
    releases.  It costs one extra level when a column carries both ``None`` and
    ``NaN``.
    """
    X = df[list(num) + list(cat)].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in cat:
        X[c] = X[c].astype(str)
    return X


def xgb_predict(X_train, y_train, X_test, cat: Sequence[str], seed: int = 42) -> np.ndarray:
    """The default XGBoost arm, falling back to sklearn when xgboost is absent.

    Category *levels* are unioned across train and test purely so both frames
    share one encoding; no value, statistic or label from the test rows enters
    the fit.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return _hist_gb_predict(X_train, y_train, X_test, cat, seed)

    Xtr, Xte = X_train.copy(), X_test.copy()
    for c in cat:
        levels = pd.Index(sorted(set(Xtr[c].astype(str)) | set(Xte[c].astype(str))))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=levels)
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=levels)
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
        tree_method="hist", enable_categorical=True,
        eval_metric="auc", verbosity=0, random_state=seed,
    )
    model.fit(Xtr, y_train)
    return model.predict_proba(Xte)[:, 1]


def _hist_gb_predict(X_train, y_train, X_test, cat, seed=42) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xtr, Xte = X_train.copy(), X_test.copy()
    for c in cat:
        levels = pd.Index(sorted(set(Xtr[c].astype(str)) | set(Xte[c].astype(str))))
        mapping = {v: i for i, v in enumerate(levels)}
        Xtr[c] = Xtr[c].astype(str).map(mapping)
        Xte[c] = Xte[c].astype(str).map(mapping)
    Xtr_np = Xtr.to_numpy(dtype=float)
    Xte_np = Xte.to_numpy(dtype=float)
    # scikit-learn's histogram binning raises on a column with a single distinct
    # value; drop constant columns (no signal) before fitting.
    keep = [i for i in range(Xtr_np.shape[1])
            if np.unique(Xtr_np[~np.isnan(Xtr_np[:, i]), i]).size >= 2]
    if not keep:
        return np.full(len(X_test), float(np.mean(y_train)) if len(y_train) else 0.5)
    model = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    model.fit(Xtr_np[:, keep], y_train)
    return model.predict_proba(Xte_np[:, keep])[:, 1]


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class FoldOutcome:
    """Everything one fold produced."""

    fold: int
    n_train: int
    n_test: int
    composition: Dict[str, Any]
    auc_augmented: float
    auc_baseline: float
    fidelity: Dict[str, FidelityReport] = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return self.auc_augmented - self.auc_baseline


@dataclass
class CVResult:
    """Pooled out-of-fold outcome of an augmentation run."""

    folds: List[FoldOutcome]
    oof_augmented: np.ndarray
    oof_baseline: np.ndarray
    y: np.ndarray
    dataset: Optional[str] = None

    @property
    def auc_augmented(self) -> float:
        return float(roc_auc_score(self.y, self.oof_augmented))

    @property
    def auc_baseline(self) -> float:
        """NaN when the run was made without a baseline arm."""
        if np.all(np.isnan(self.oof_baseline)):
            return float("nan")
        return float(roc_auc_score(self.y, self.oof_baseline))

    @property
    def delta(self) -> float:
        return self.auc_augmented - self.auc_baseline

    def fold_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "fold": f.fold,
            "n_train": f.n_train,
            "n_test": f.n_test,
            "n_real": f.composition.get("n_real"),
            "n_synthetic": f.composition.get("n_synthetic"),
            "realized_share": f.composition.get("realized_synthetic_share"),
            "auc_baseline": f.auc_baseline,
            "auc_augmented": f.auc_augmented,
            "delta": f.delta,
        } for f in self.folds])

    def fidelity_frame(self) -> pd.DataFrame:
        """Per-fold, per-generator fidelity, one row each."""
        rows = []
        for f in self.folds:
            for name, rep in f.fidelity.items():
                row = {"fold": f.fold, "generator": name}
                row.update(rep.summary().to_dict())
                rows.append(row)
        return pd.DataFrame(rows)

    def fidelity_headline(self) -> pd.DataFrame:
        """The few fidelity columns worth reading first, per fold and generator.

        ``fidelity_frame`` returns everything the battery computed — 66 columns
        on a full SDV run.  This is the documented shortlist from
        :data:`~pamir.synthetic.fidelity.HEADLINE_FIDELITY_COLUMNS`, narrowed to
        the columns actually present, so it is also safe on a dependency-free
        run where most of the battery did not execute.
        """
        fid = self.fidelity_frame()
        if fid.empty:
            return fid
        present = [c for c in HEADLINE_FIDELITY_COLUMNS if c in fid.columns]
        return fid[["fold", "generator"] + present]

    def leakage_frame(self) -> pd.DataFrame:
        """The leakage probes alone — the frame to read before believing a delta.

        Empty means the probes did not run, never that they ran and found
        nothing: a frame of fold/generator labels with no ``leak.`` column would
        read as a clean bill of health that nobody issued.
        """
        fid = self.fidelity_frame()
        if fid.empty:
            return fid
        probes = [c for c in fid.columns if c.startswith("leak.")]
        if not probes:
            return pd.DataFrame()
        return fid[["fold", "generator"] + probes]

    def summary(self) -> pd.Series:
        fold_deltas = [f.delta for f in self.folds]
        return pd.Series({
            "dataset": self.dataset,
            "n_folds": len(self.folds),
            "auc_baseline": round(self.auc_baseline, 4),
            "auc_augmented": round(self.auc_augmented, 4),
            "delta_oof": round(self.delta, 4),
            "delta_fold_mean": round(float(np.mean(fold_deltas)), 4),
            "delta_fold_sd": round(float(np.std(fold_deltas)), 4),
            "folds_positive": int(sum(d > 0 for d in fold_deltas)),
        })


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
class CrossValidatedAugmentation:
    """Run a :class:`SyntheticMixer` inside stratified k-fold, without leakage.

    Parameters
    ----------
    mixer : SyntheticMixer
        Template.  It is cloned per fold; the original is never fitted.
    n_splits : int
        Folds (default 5).
    seed : int
        Seed for the fold split **and** for the default model arm, so changing it
        moves the delta on unchanged data — state which seed a run used before
        comparing two runs.  A ``predict_fn`` of
        your own receives no seed; seed it yourself if it needs one.
    predict_fn : callable, optional
        ``predict_fn(X_train, y_train, X_test) -> scores``, the same convention
        as :func:`pamir.evaluate_iid`.  Defaults to the XGBoost arm.
    baseline : bool
        Also fit on the real training rows alone, to report a delta rather than
        a bare number.
    """

    def __init__(
        self,
        mixer: SyntheticMixer,
        n_splits: int = 5,
        seed: int = 42,
        predict_fn: Optional[Callable] = None,
        baseline: bool = True,
        target: str = TARGET,
        verbose: bool = True,
    ):
        if mixer.target is not None and mixer.target != target:
            raise ValueError(
                f"the mixer names its target {mixer.target!r} but the runner builds "
                f"its frame with {target!r}. The runner owns the column name, so the "
                "mixer's would be silently overridden; pass the same name to both, or "
                "leave the mixer's target unset."
            )
        self.mixer = mixer
        self.n_splits = n_splits
        self.seed = seed
        self.predict_fn = predict_fn
        self.baseline = baseline
        self.target = target
        self.verbose = verbose

    def run(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        dataset: Optional[str] = None,
    ) -> CVResult:
        """Fit, mix and score fold by fold.  ``X`` holds features only."""
        y = np.asarray(y).astype(int)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}.")
        labels = set(np.unique(y).tolist())
        if len(labels) < 2:
            raise ValueError("Target has a single class; AUC is undefined.")
        if not labels <= {0, 1}:
            raise ValueError(
                f"y must be coded 0/1, got labels {sorted(labels)}. The protocol's "
                "model arm and every downstream check assume that coding; recode "
                "before calling rather than letting the mixed frame be blamed for it."
            )

        num, cat = split_columns(X, target=self.target)
        Xp = prepare(X, num, cat).reset_index(drop=True)
        frame = Xp.copy()
        frame[self.target] = y

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        oof_aug = np.full(len(y), np.nan)
        oof_base = np.full(len(y), np.nan)
        outcomes: List[FoldOutcome] = []

        for k, (tr, te) in enumerate(skf.split(Xp, y)):
            if set(tr) & set(te):
                raise AssertionError(f"fold {k}: train and test indices overlap.")

            train_frame = frame.iloc[tr].reset_index(drop=True)
            test_frame = frame.iloc[te].reset_index(drop=True)
            X_test, y_test = Xp.iloc[te], y[te]

            fold_mixer = self.mixer.clone()
            fold_mixer.target = self.target   # the runner owns the column name
            mix = fold_mixer.build(train_frame, holdout=test_frame, fold=k)

            X_aug = mix.data.drop(columns=[self.target])
            y_raw = pd.to_numeric(mix.data[self.target], errors="coerce")
            bad = ~y_raw.isin([0, 1])
            if bad.any():
                # y was checked above, so only a generator can be responsible.
                offenders = mix.sources[bad.to_numpy()].value_counts().to_dict()
                raise ValueError(
                    f"fold {k}: {int(bad.sum())} synthetic rows carry a target that "
                    f"is neither 0 nor 1, by generator {offenders}. Coercing them to "
                    "the negative class would move the delta without saying so; fix "
                    "the generator or filter its output before mixing."
                )
            y_aug = y_raw.astype(int).to_numpy()
            X_aug = prepare(X_aug, num, cat)

            predict = self.predict_fn or (lambda a, b, c: xgb_predict(a, b, c, cat, self.seed))
            oof_aug[te] = predict(X_aug, y_aug, X_test)

            if self.baseline:
                oof_base[te] = predict(Xp.iloc[tr], y[tr], X_test)

            auc_aug = float(roc_auc_score(y_test, oof_aug[te]))
            auc_base = float(roc_auc_score(y_test, oof_base[te])) if self.baseline else float("nan")
            outcomes.append(FoldOutcome(
                fold=k, n_train=len(tr), n_test=len(te),
                composition=mix.composition, auc_augmented=auc_aug,
                auc_baseline=auc_base, fidelity=mix.fidelity,
            ))
            if self.verbose:
                print(f"  fold {k}: real={mix.composition['n_real']} "
                      f"synth={mix.composition['n_synthetic']} "
                      f"share={mix.composition['realized_synthetic_share']:.2f}  "
                      f"base={auc_base:.4f} aug={auc_aug:.4f} "
                      f"delta={auc_aug - auc_base:+.4f}", flush=True)

        return CVResult(folds=outcomes, oof_augmented=oof_aug, oof_baseline=oof_base,
                        y=y, dataset=dataset)

    def run_dataset(self, dataset_id: str, max_rows: Optional[int] = None) -> CVResult:
        """Same, on a PaMIR dataset loaded by id."""
        from pamir.loader import load_dataset

        X, y, _ = load_dataset(dataset_id, max_rows=max_rows)
        if self.verbose:
            print(f"{dataset_id}: {len(y)} rows, DR={y.mean():.3f}", flush=True)
        return self.run(X, y, dataset=dataset_id)


def run_fleet(
    mixer: SyntheticMixer,
    datasets: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = 20000,
    **kwargs,
) -> pd.DataFrame:
    """Run the augmentation across PaMIR datasets and return one summary row each."""
    from pamir.catalog import list_datasets

    datasets = list(datasets) if datasets else list_datasets()
    # Built once: it carries no per-dataset state, and a misconfiguration should
    # surface before the sweep starts rather than on every table in turn.
    runner = CrossValidatedAugmentation(mixer, **kwargs)
    rows = []
    for ds in datasets:
        try:
            result = runner.run_dataset(ds, max_rows=max_rows)
            rows.append(result.summary())
        except Exception as exc:  # noqa: BLE001 - one bad table must not stop the fleet
            rows.append(pd.Series({"dataset": ds, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}))
    return pd.DataFrame(rows)
