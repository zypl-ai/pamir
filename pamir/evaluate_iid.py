"""The conventional i.i.d. train/test protocol.

Included so PaMIR numbers can be compared with other tabular benchmarks.  It
is **not** the PaMIR-native protocol: it hands the model a shuffled split with
every label resolved, a condition that does not exist in production.  Use
:mod:`pamir.evaluate` for the streaming protocol.
"""

import warnings
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pamir.catalog import list_datasets
from pamir.failures import FailureLog
from pamir.loader import load_dataset
from pamir.summary import fleet_summary, format_fleet_summary

RESULT_COLUMNS = ["dataset", "n_rows", "n_defaults", "DR", "train_frac",
                  "n_seeds", "n_calls", "n_failures", "contract_ok",
                  "auc_mean", "auc_std"]

_BASE_SEED = 42
_MIN_TEST_DEFAULTS = 3


def evaluate_iid_one(
    predict_fn: Callable,
    dataset_id: str,
    train_frac: float = 0.7,
    max_n: Optional[int] = None,
    n_seeds: int = 5,
    on_error: str = "warn",
    strict: bool = True,
) -> Dict:
    """Evaluate one dataset with a standard i.i.d. train/test split.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(X_train, y_train, X_test) -> np.ndarray`` of scores
        (higher = more likely to default).
    dataset_id : str
        PaMIR dataset id.
    train_frac : float
        Fraction of data used for training (default 0.7).
    max_n : int or None
        Truncate dataset before splitting.
    n_seeds : int
        Number of random shuffles (default 5).  The reported AUC is the mean.
    on_error : {"warn", "raise", "ignore"}
        What to do when ``predict_fn`` raises.  See :mod:`pamir.failures`.
    strict : bool
        Refuse a table that does not match its data contract (default True);
        see :func:`pamir.load_dataset`.

    Returns
    -------
    dict with keys: dataset, n_rows, n_defaults, DR, train_frac, n_seeds,
    n_calls, n_failures, errors, contract_ok, auc_mean, auc_std, aucs.
    """
    X, y, meta = load_dataset(dataset_id, max_rows=max_n, strict=strict)
    n = len(y)
    split = int(n * train_frac)

    log = FailureLog(on_error, context=dataset_id)
    aucs = []

    def score_split(X_train, y_train, X_test, y_test):
        scores = np.asarray(predict_fn(X_train, y_train, X_test), dtype=np.float64)
        if scores.shape != (len(X_test),):
            raise ValueError(
                f"predict_fn returned scores of shape {scores.shape} for "
                f"{len(X_test)} rows; expected ({len(X_test)},)."
            )
        if y_test.sum() >= _MIN_TEST_DEFAULTS:
            aucs.append(float(roc_auc_score(y_test, scores)))

    for seed in range(_BASE_SEED, _BASE_SEED + n_seeds):
        idx = np.random.RandomState(seed).permutation(n)
        X_shuffled = X.iloc[idx].reset_index(drop=True)
        y_shuffled = y[idx]
        log.call(score_split,
                 X_shuffled.iloc[:split], y_shuffled[:split],
                 X_shuffled.iloc[split:], y_shuffled[split:])

    log.warn_if_failed()

    return {
        "dataset": dataset_id,
        "n_rows": n,
        "n_defaults": int(y.sum()),
        "DR": float(y.mean()),
        "train_frac": train_frac,
        "n_seeds": n_seeds,
        **log.as_dict(),
        "contract_ok": meta["contract_ok"],
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if len(aucs) > 1 else None,
        "aucs": aucs,
    }


def evaluate_iid(
    predict_fn: Callable,
    datasets: Optional[Sequence[str]] = None,
    train_frac: float = 0.7,
    max_n: Optional[int] = None,
    n_seeds: int = 5,
    verbose: bool = True,
    on_error: str = "warn",
    strict: bool = True,
) -> pd.DataFrame:
    """Run the i.i.d. protocol across the PaMIR fleet.

    Returns
    -------
    DataFrame with one row per dataset and the columns in ``RESULT_COLUMNS``.
    Pass it to :func:`pamir.fleet_summary` for the headline numbers.
    """
    ds_ids = list(datasets) if datasets else list_datasets()
    rows = []
    for ds in ds_ids:
        if verbose:
            print(f"  {ds}...", end=" ", flush=True)
        result = evaluate_iid_one(predict_fn, ds, train_frac=train_frac,
                                  max_n=max_n, n_seeds=n_seeds,
                                  on_error=on_error, strict=strict)
        if verbose:
            print(_format_dataset_line(result))
        rows.append(result)

    df = pd.DataFrame(rows)[RESULT_COLUMNS]
    summary = fleet_summary(df)

    if verbose and len(rows) > 1:
        print(format_fleet_summary(summary))
    if not summary["complete"] and on_error == "warn":
        warnings.warn(
            f"the model did not score {summary['n_datasets'] - summary['n_scored']} "
            f"of {summary['n_datasets']} datasets "
            f"({', '.join(summary['failed_datasets'])}). There is no fleet mean "
            "for a partial run — see pamir.fleet_summary.",
            UserWarning,
            stacklevel=2,
        )
    if summary["complete"] and not summary["contract_ok"] and on_error == "warn":
        warnings.warn(
            "every dataset was scored, but these tables do not match their data "
            f"contract: {', '.join(summary['contract_failed_datasets'])}. There is "
            "no fleet mean for a run on non-benchmark tables — see pamir.contract.",
            UserWarning,
            stacklevel=2,
        )
    return df


def _format_dataset_line(result: Dict) -> str:
    auc, std = result["auc_mean"], result["auc_std"]
    if auc is None:
        head = "NO AUC"
    elif std is None:
        head = f"AUC={auc:.4f}"
    else:
        head = f"AUC={auc:.4f}+/-{std:.4f}"
    if result["n_failures"]:
        head += f"  ({result['n_failures']}/{result['n_calls']} calls FAILED)"
    return head
