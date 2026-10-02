"""The streaming evaluation protocol.

Simulates a lender scoring an arriving stream of applications whose outcomes
are revealed only after a maturation delay of ``lag`` stream positions.  Rows
are replayed in their stored order, a fixed random permutation of the source
(the sources carry no usable origination dates), so the stream has no calendar
drift.  See :mod:`pamir.evaluate_iid` for the conventional train/test split.

Two modes
---------
``mode="arrival"`` (default, the reference protocol since 0.4.0)
    The model is a ``fit_fn(X_train, y_train) -> scorer``.  A refit trains on
    the labels resolved at that step; the resulting ``scorer(X_rows)`` scores
    every row that arrives after the refit and before the next one, and a
    score, once given, is final.  The model therefore never sees a label that
    had not matured when a row arrived, nor the features of a row that had not
    arrived, and the delay ``lag`` binds for every row.  Scorers are called on
    batches of already-arrived rows for speed; a probe re-scores a random subset
    of each batch on its own, and a scorer whose scores depend on the other rows
    of the batch is flagged ``row_independent = False`` (the run then has no
    fleet mean).  ``batch_scoring=False`` calls the scorer one row at a time.
    A plain ``predict_fn(X_train, y_train, X_test)`` is accepted and wrapped
    (one ``predict_fn`` call per scoring batch).

``mode="refresh"`` (the 0.3.0 protocol, kept for reproducibility)
    The model is a ``predict_fn``.  Every refit re-scores all rows whose label
    is unresolved, *including rows that have not yet arrived*, whose features
    the model therefore sees in advance; a row is evaluated on the last score
    committed before its label resolves.  0.3.0 numbers are reproduced with
    ``mode="refresh", max_lag_frac=None`` and the ``*_v03`` baselines.

Delay cap
---------
A delay as long as the stream leaves no label to learn from: at ``lag=1000``
the 1,000-row datasets would never be scored.  The effective delay is
``min(lag, floor(max_lag_frac * n))`` with ``max_lag_frac=0.2`` by default, and
is reported per dataset as ``lag_effective``.

Every call into user code is counted and failures are kept — see
:mod:`pamir.failures`.
"""

import inspect
import warnings
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pamir.catalog import list_datasets
from pamir.failures import FailureLog
from pamir.loader import load_dataset
from pamir.summary import fleet_summary, format_fleet_summary

MODES = ("arrival", "refresh")

# Columns of the per-dataset frame returned by `evaluate`.  `refit_points` and
# the per-row arrays are deliberately absent: they are lists per row, which
# block `to_csv` and every other tabular export.  Read them from `evaluate_one`.
RESULT_COLUMNS = ["dataset", "mode", "n_rows", "n_defaults", "DR", "lag_effective",
                  "k_refit", "n_refits", "n_refits_ok", "n_calls", "n_failures",
                  "n_scored", "contract_ok", "row_independent", "auc_final"]

# Label budgets: rows are grouped by the number of resolved labels their scorer
# was trained on, [0, 100), [100, 300), ..., [10000, inf).
BUDGET_EDGES = (0, 100, 300, 1000, 3000, 10000)
BUDGET_COLUMNS = [f"auc_budget_{lo}" for lo in BUDGET_EDGES]

# Minimum scored rows and defaults before an AUC is meaningful.
_MIN_SCORED_ROWS = 10
_MIN_SCORED_DEFAULTS = 3

# Row-independence probe: rows re-scored per batch, and the tolerance.
_PROBE_ROWS = 64
_PROBE_RTOL, _PROBE_ATOL = 1e-6, 1e-9


def effective_lag(lag: int, n: int, max_lag_frac: Optional[float]) -> int:
    """The delay actually applied to a stream of ``n`` rows."""
    if max_lag_frac is None:
        return int(lag)
    return int(max(1, min(int(lag), int(np.floor(max_lag_frac * n)))))


def from_predict_fn(predict_fn: Callable) -> Callable:
    """Turn ``predict_fn(X_train, y_train, X_test)`` into a ``fit_fn``.

    The returned ``fit_fn`` stores the training data; its scorer calls
    ``predict_fn`` once per scoring batch, so a model is refitted for every
    batch it scores.  Correct, but slower than a native ``fit_fn``.
    """
    def fit_fn(X_train, y_train):
        def scorer(X_rows):
            return predict_fn(X_train, y_train, X_rows)
        return scorer

    fit_fn._pamir_wrapped = True
    fit_fn.__name__ = f"from_predict_fn({getattr(predict_fn, '__name__', 'model')})"
    return fit_fn


def _n_required_positional(fn: Callable) -> Optional[int]:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(1 for p in params if p.kind in kinds and p.default is inspect.Parameter.empty)


def _as_fit_fn(model: Callable) -> Callable:
    """Accept a fit_fn as is; wrap a three-argument predict_fn."""
    if getattr(model, "_pamir_wrapped", False):
        return model
    if _n_required_positional(model) == 3:
        return from_predict_fn(model)
    return model


def _refit_due(cum_defaults: int, since_refit: int, n_attempts: int,
               min_defaults: int, k_refit: int) -> bool:
    """First refit at ``min_defaults`` resolved defaults, then every ``k_refit``."""
    if cum_defaults == min_defaults and n_attempts == 0:
        return True
    return cum_defaults >= min_defaults and since_refit >= k_refit


def _safe_auc(y: np.ndarray, preds: np.ndarray) -> Optional[float]:
    """ROC AUC over scored rows, or None when too few to be meaningful."""
    valid = ~np.isnan(preds)
    n_pos = y[valid].sum()
    if (valid.sum() <= _MIN_SCORED_ROWS or n_pos < _MIN_SCORED_DEFAULTS
            or n_pos == valid.sum()):
        return None
    return float(roc_auc_score(y[valid], preds[valid]))


def _refit_point(t: int, resolved_end: int, cum_defaults: int, y: np.ndarray,
                 preds: np.ndarray, ok: bool) -> Dict:
    """Record cumulative AUC over everything resolved (and scored) so far."""
    point = {"pos": t + 1, "resolved_n": resolved_end,
             "cum_defaults": cum_defaults, "ok": bool(ok)}
    auc = _safe_auc(y[:resolved_end], preds[:resolved_end])
    if auc is not None:
        point["cum_auc"] = auc
    return point


def auc_by_budget(y: np.ndarray, preds: np.ndarray, train_n: np.ndarray,
                  edges: Sequence[int] = BUDGET_EDGES) -> Dict[str, Optional[float]]:
    """AUC of the rows scored by models trained on a given number of labels.

    ``train_n[i]`` is the number of resolved labels behind row ``i``'s score
    (``-1`` for unscored rows).  Bins with too few rows or defaults are None.
    """
    out: Dict[str, Optional[float]] = {}
    uppers = list(edges[1:]) + [np.inf]
    for lo, hi in zip(edges, uppers):
        mask = (train_n >= lo) & (train_n < hi)
        out[f"auc_budget_{lo}"] = _safe_auc(y[mask], preds[mask]) if mask.any() else None
    return out


# ---------------------------------------------------------------------------
# refresh mode (0.3.0)
# ---------------------------------------------------------------------------

def _run_refresh(predict_fn, X, y, lag, k_refit, min_defaults, log):
    n = len(y)
    preds = np.full(n, np.nan, dtype=np.float64)
    cum_defaults = 0
    since_refit = 0
    refit_points: List[Dict] = []

    def score_and_commit(X_ctx, y_ctx, X_rem, start):
        """Call the model and commit its scores, validating the contract."""
        scores = np.asarray(predict_fn(X_ctx, y_ctx, X_rem), dtype=np.float64)
        if scores.shape != (len(X_rem),):
            raise ValueError(
                f"predict_fn returned scores of shape {scores.shape} for "
                f"{len(X_rem)} rows; expected ({len(X_rem)},).")
        preds[start:] = scores
        return True

    for t in range(n):
        resolved_end = max(0, t + 1 - lag)
        if t >= lag and y[t - lag] == 1:
            cum_defaults += 1
            since_refit += 1
        if not (_refit_due(cum_defaults, since_refit, len(refit_points),
                           min_defaults, k_refit)
                and resolved_end > 0 and t + 1 < n):
            continue
        if resolved_end - cum_defaults < min_defaults:
            continue
        ok = log.call(score_and_commit, X.iloc[:resolved_end], y[:resolved_end],
                      X.iloc[resolved_end:], resolved_end) is not None
        since_refit = 0
        refit_points.append(_refit_point(t, resolved_end, cum_defaults, y, preds, ok))

    return preds, refit_points, None, None


# ---------------------------------------------------------------------------
# arrival mode
# ---------------------------------------------------------------------------

def _run_arrival(fit_fn, X, y, lag, k_refit, min_defaults, log,
                 batch_scoring, probe, seed=0):
    n = len(y)
    wrapped = getattr(fit_fn, "_pamir_wrapped", False)
    preds = np.full(n, np.nan, dtype=np.float64)
    train_n = np.full(n, -1, dtype=np.int64)
    rng = np.random.RandomState(seed)
    state = {"scorer": None, "scorer_n": 0, "pending": None, "row_independent": True}
    cum_defaults = 0
    since_refit = 0
    refit_points: List[Dict] = []

    def fit(X_ctx, y_ctx):
        scorer = fit_fn(X_ctx, y_ctx)
        if not callable(scorer):
            raise TypeError(
                "in arrival mode the model must be fit_fn(X_train, y_train) returning "
                f"a callable scorer(X_rows); got {type(scorer).__name__}. A "
                "three-argument predict_fn is accepted too (see from_predict_fn).")
        return scorer

    def score(start, stop):
        X_rows = X.iloc[start:stop]
        scores = np.asarray(state["scorer"](X_rows), dtype=np.float64)
        if scores.shape != (len(X_rows),):
            raise ValueError(
                f"scorer returned scores of shape {scores.shape} for "
                f"{len(X_rows)} rows; expected ({len(X_rows)},).")
        if probe and len(X_rows) >= 2:
            k = min(_PROBE_ROWS, max(1, len(X_rows) // 2))
            sub = np.sort(rng.choice(len(X_rows), size=k, replace=False))
            again = np.asarray(state["scorer"](X_rows.iloc[sub]), dtype=np.float64)
            if again.shape != (k,) or not np.allclose(
                    again, scores[sub], rtol=_PROBE_RTOL, atol=_PROBE_ATOL,
                    equal_nan=True):
                state["row_independent"] = False
        preds[start:stop] = scores
        train_n[start:stop] = state["scorer_n"]
        return True

    def flush(stop):
        """Score rows [pending, stop) with the scorer they arrived under."""
        start = state["pending"]
        if state["scorer"] is None or start is None or stop <= start:
            return
        if batch_scoring:
            log.call(score, start, stop)
        else:
            for i in range(start, stop):
                log.call(score, i, i + 1)

    n_attempts = 0
    for t in range(n):
        # Within step t: row t arrives (scored by the current scorer); then the
        # label of row t - lag resolves; then a refit may fire, and the new
        # scorer applies from row t + 1 on.
        resolved_end = max(0, t + 1 - lag)
        if t >= lag and y[t - lag] == 1:
            cum_defaults += 1
            since_refit += 1
        if not (_refit_due(cum_defaults, since_refit, n_attempts, min_defaults, k_refit)
                and resolved_end > 0 and t + 1 < n):
            continue
        if resolved_end - cum_defaults < min_defaults:
            continue
        flush(t + 1)
        n_attempts += 1
        if wrapped:   # nothing of the user's runs at fit time; scoring calls are counted
            new = fit(X.iloc[:resolved_end], y[:resolved_end])
        else:
            new = log.call(fit, X.iloc[:resolved_end], y[:resolved_end])
        since_refit = 0
        if new is not None:
            state["scorer"], state["scorer_n"] = new, resolved_end
        state["pending"] = t + 1
        refit_points.append(_refit_point(t, resolved_end, cum_defaults, y, preds,
                                         new is not None))
    flush(n)
    return preds, refit_points, train_n, state["row_independent"]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def evaluate_one(
    model: Callable,
    dataset_id: str,
    lag: int = 1000,
    k_refit: int = 10,
    max_n: Optional[int] = 20000,
    min_defaults: int = 3,
    on_error: str = "warn",
    mode: str = "arrival",
    max_lag_frac: Optional[float] = 0.2,
    batch_scoring: bool = True,
    probe: bool = True,
    strict: bool = True,
) -> Dict:
    """Run the streaming protocol on one dataset.

    Parameters
    ----------
    model : callable
        ``mode="arrival"``: ``fit_fn(X_train, y_train) -> scorer`` with
        ``scorer(X_rows) -> np.ndarray`` (higher = more likely to default); a
        three-argument ``predict_fn`` is wrapped with :func:`from_predict_fn`.
        ``mode="refresh"``: ``predict_fn(X_train, y_train, X_test)``, where
        ``X_test`` is the remaining stream.
    dataset_id : str
        PaMIR dataset id.
    lag : int
        Label-maturation delay: the outcome of the row at position *t* is
        revealed at position *t + lag*.  Capped by ``max_lag_frac``.
    k_refit : int
        Refit after every *k_refit* newly resolved defaults.  This changes the
        score as well as the cost, so report it.
    max_n : int or None
        Truncate the stream to at most this many rows.
    min_defaults : int
        Resolved defaults (and non-defaults) required before the first refit.
    on_error : {"warn", "raise", "ignore"}
        What to do when the model raises.  See :mod:`pamir.failures`.
    mode : {"arrival", "refresh"}
        See the module docstring.
    max_lag_frac : float or None
        Cap on the delay as a fraction of the stream length; None disables it.
    batch_scoring, probe : bool
        Arrival mode only: score arrived rows in batches, and probe each batch
        for row independence (see the module docstring).
    strict : bool
        Refuse a table that does not match its data contract (default True).

    Returns
    -------
    dict with the keys of ``RESULT_COLUMNS``, plus ``lag``, ``errors``,
    ``refit_points`` (list of dicts with pos, resolved_n, cum_defaults, ok,
    cum_auc), and in arrival mode the ``auc_budget_*`` keys and ``train_n``
    (per-row label budget, -1 where unscored).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    X, y, meta = load_dataset(dataset_id, max_rows=max_n, strict=strict)
    n = len(y)
    lag_eff = effective_lag(lag, n, max_lag_frac)
    log = FailureLog(on_error, context=dataset_id)

    if mode == "refresh":
        preds, refit_points, train_n, row_independent = _run_refresh(
            model, X, y, lag_eff, k_refit, min_defaults, log)
    else:
        preds, refit_points, train_n, row_independent = _run_arrival(
            _as_fit_fn(model), X, y, lag_eff, k_refit, min_defaults, log,
            batch_scoring=batch_scoring, probe=probe)
    log.warn_if_failed()

    result = {
        "dataset": dataset_id,
        "mode": mode,
        "n_rows": n,
        "n_defaults": int(y.sum()),
        "DR": float(y.mean()),
        "lag": int(lag),
        "lag_effective": lag_eff,
        "k_refit": int(k_refit),
        "n_refits": len(refit_points),
        "n_refits_ok": sum(1 for p in refit_points if p["ok"]),
        **log.as_dict(),
        "n_scored": int(np.isfinite(preds).sum()),
        "contract_ok": meta["contract_ok"],
        "row_independent": row_independent,
        "auc_final": _safe_auc(y, preds),
        "refit_points": refit_points,
    }
    if train_n is not None:
        result.update(auc_by_budget(y, preds, train_n))
        result["train_n"] = train_n.tolist()
    return result


def evaluate(
    model: Callable,
    datasets: Optional[Sequence[str]] = None,
    lag: int = 1000,
    k_refit: int = 10,
    max_n: Optional[int] = 20000,
    min_defaults: int = 3,
    verbose: bool = True,
    on_error: str = "warn",
    mode: str = "arrival",
    max_lag_frac: Optional[float] = 0.2,
    batch_scoring: bool = True,
    probe: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Run the streaming protocol across the PaMIR fleet.

    Parameters are those of :func:`evaluate_one`; ``datasets`` is a subset of
    dataset ids (default: every dataset in the catalogue).

    Returns
    -------
    DataFrame with one row per dataset: the columns in ``RESULT_COLUMNS`` and,
    in arrival mode, ``BUDGET_COLUMNS``.  Pass it to :func:`pamir.fleet_summary`
    for the headline numbers — a mean taken by hand over this frame silently
    excludes the datasets the model failed on.
    """
    ds_ids = list(datasets) if datasets else list_datasets()
    rows = []
    for ds in ds_ids:
        if verbose:
            print(f"  {ds}...", end=" ", flush=True)
        result = evaluate_one(model, ds, lag=lag, k_refit=k_refit, max_n=max_n,
                              min_defaults=min_defaults, on_error=on_error,
                              mode=mode, max_lag_frac=max_lag_frac,
                              batch_scoring=batch_scoring, probe=probe,
                              strict=strict)
        if verbose:
            print(_format_dataset_line(result))
        rows.append(result)

    columns = RESULT_COLUMNS + (BUDGET_COLUMNS if mode == "arrival" else [])
    df = pd.DataFrame(rows)[columns]
    summary = fleet_summary(df)

    if verbose and len(rows) > 1:
        print(format_fleet_summary(summary))
    if on_error == "warn":
        _warn_no_headline(summary)
    return df


def _warn_no_headline(summary: Dict) -> None:
    if not summary["complete"]:
        warnings.warn(
            f"the model did not score {summary['n_datasets'] - summary['n_scored']} "
            f"of {summary['n_datasets']} datasets "
            f"({', '.join(summary['failed_datasets'])}). There is no fleet mean "
            "for a partial run — see pamir.fleet_summary.",
            UserWarning, stacklevel=3)
    elif not summary["contract_ok"]:
        warnings.warn(
            "every dataset was scored, but these tables do not match their data "
            f"contract: {', '.join(summary['contract_failed_datasets'])}. There is "
            "no fleet mean for a run on non-benchmark tables — see pamir.contract.",
            UserWarning, stacklevel=3)
    elif not summary.get("row_independent", True):
        warnings.warn(
            "the scorer's output depended on the other rows of a scoring batch on "
            f"{', '.join(summary['row_dependent_datasets'])}; in arrival mode that "
            "lets a row's score use rows that arrived after it. There is no fleet "
            "mean — fix the scorer, or pass batch_scoring=False.",
            UserWarning, stacklevel=3)


def _format_dataset_line(result: Dict) -> str:
    auc = result["auc_final"]
    head = f"AUC={auc:.4f}" if auc is not None else "NO AUC"
    tail = f"({result['n_refits']} refits, lag {result['lag_effective']}"
    if result["n_failures"]:
        tail += f", {result['n_failures']}/{result['n_calls']} calls FAILED"
    if result.get("row_independent") is False:
        tail += ", ROW-DEPENDENT SCORER"
    if result.get("contract_ok") is False:
        tail += ", CONTRACT MISMATCH"
    return f"{head} {tail})"
