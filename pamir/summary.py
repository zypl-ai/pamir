"""Fleet-level summary of a results frame, with coverage as a first-class field.

The headline number of a benchmark run is only defined when the model scored
every dataset it was given.  ``fleet_summary`` therefore reports ``auc_mean``
only for a complete run, and puts the partial average in a separately named
field so it cannot be mistaken for one.
"""

from typing import Dict, List, Optional

import pandas as pd

# The AUC column is named by the protocol that produced the frame.
AUC_COLUMNS = ("auc_final", "auc_mean")


def _auc_column(results: pd.DataFrame) -> str:
    for col in AUC_COLUMNS:
        if col in results.columns:
            return col
    raise ValueError(
        f"no auc column found: expected one of {AUC_COLUMNS}, "
        f"got {list(results.columns)}. Pass a frame returned by "
        "pamir.evaluate or pamir.evaluate_iid."
    )


def fleet_summary(results: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Summarize a fleet run, withholding the mean when coverage is incomplete.

    Parameters
    ----------
    results : DataFrame
        As returned by :func:`pamir.evaluate` or :func:`pamir.evaluate_iid`.

    Returns
    -------
    dict with keys:

    ``n_datasets``, ``n_scored``, ``coverage``, ``complete``
        How much of the fleet the model actually scored.
    ``auc_mean``, ``gini_mean``
        The headline numbers — ``None`` unless every dataset was scored and,
        when the frame carries ``contract_ok``, every table matched its data
        contract.
    ``auc_mean_scored_only``
        The average over the datasets that did score.  Diagnostic only: it is
        an average over a subset the model chose by failing, so it is not
        comparable across models.
    ``n_failures``, ``failed_datasets``
        What went wrong and where.
    ``contract_ok``, ``contract_failed_datasets``
        Whether every table was the benchmark table (see :mod:`pamir.contract`).
    ``row_independent``, ``row_dependent_datasets``
        Arrival mode: whether every scorer scored rows independently of the
        other rows in its batch.
    ``auc_by_budget``
        Arrival mode: per label-budget bin, the mean AUC over the datasets that
        reach it and their number (``{}`` for other protocols).
    """
    auc_col = _auc_column(results)
    scored = results[auc_col].notna()

    n_datasets = len(results)
    n_scored = int(scored.sum())
    complete = n_datasets > 0 and n_scored == n_datasets
    partial_mean = float(results.loc[scored, auc_col].mean()) if n_scored else None

    has_names = "dataset" in results.columns
    failed: List[str] = results.loc[~scored, "dataset"].tolist() if has_names else []

    contract_ok, contract_failed = True, []
    if "contract_ok" in results.columns:
        ok = results["contract_ok"].fillna(False).astype(bool)
        contract_ok = bool(ok.all())
        contract_failed = results.loc[~ok, "dataset"].tolist() if has_names else []

    # Arrival mode records whether the scorer scored rows independently; refresh
    # and i.i.d. rows carry None (not applicable), which does not block.
    row_independent, row_dependent = True, []
    if "row_independent" in results.columns:
        dependent = results["row_independent"].map(
            lambda v: v is not None and not pd.isna(v) and not bool(v)).astype(bool)
        row_independent = not bool(dependent.any())
        row_dependent = results.loc[dependent, "dataset"].tolist() if has_names else []
    headline = complete and contract_ok and row_independent

    # Label-budget curve (arrival mode): per budget bin, the mean AUC over the
    # datasets that reach that bin, with their count.  Bins differ in which
    # datasets they contain, so the curve is descriptive, not a headline.
    budget = {}
    for col in [c for c in results.columns if c.startswith("auc_budget_")]:
        vals = pd.to_numeric(results[col], errors="coerce").dropna()
        budget[col] = {"mean": float(vals.mean()) if len(vals) else None,
                       "n_datasets": int(len(vals))}

    return {
        "n_datasets": n_datasets,
        "n_scored": n_scored,
        "coverage": n_scored / n_datasets if n_datasets else 0.0,
        "complete": complete,
        "auc_mean": partial_mean if headline else None,
        "gini_mean": 2 * partial_mean - 1 if headline and partial_mean else None,
        "auc_mean_scored_only": partial_mean,
        "n_failures": (int(results["n_failures"].sum())
                       if "n_failures" in results.columns else None),
        "failed_datasets": failed,
        "contract_ok": contract_ok,
        "contract_failed_datasets": contract_failed,
        "row_independent": row_independent,
        "row_dependent_datasets": row_dependent,
        "auc_by_budget": budget,
    }


def format_fleet_summary(summary: Dict) -> str:
    """One or two lines of human-readable summary, for ``verbose=True``."""
    if summary["complete"] and summary["auc_mean"] is not None:
        return (f"\n  Fleet mean AUC: {summary['auc_mean']:.4f}  "
                f"Gini: {summary['gini_mean']:.3f}  "
                f"({summary['n_datasets']} datasets, all scored)")
    if summary["complete"] and not summary.get("contract_ok", True):
        names = ", ".join(summary.get("contract_failed_datasets") or []) or "—"
        return (f"\n  NO FLEET MEAN: every dataset was scored, but these tables do "
                f"not match their data contract: {names}.")
    if summary["complete"]:
        names = ", ".join(summary.get("row_dependent_datasets") or []) or "—"
        return (f"\n  NO FLEET MEAN: every dataset was scored, but the scorer was "
                f"not row-independent on: {names}.")

    failed = ", ".join(summary["failed_datasets"]) or "—"
    partial = summary["auc_mean_scored_only"]
    partial_text = f"{partial:.4f}" if partial is not None else "n/a"
    return (
        f"\n  NO FLEET MEAN: {summary['n_scored']} of {summary['n_datasets']} "
        f"datasets scored (failed: {failed})."
        f"\n  Mean over the scored subset only: {partial_text} — not comparable "
        "with a model that scored the whole fleet."
    )
