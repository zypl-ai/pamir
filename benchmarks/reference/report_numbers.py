"""Illustrative numbers quoted in the report (Sections 4.4, 4.5 and 6).

These are the package-walkthrough settings, not reference results: a short
stream (``lag=500``, ``max_n=5000``) that runs in minutes.  Each number is
computed twice, under the 0.4.0 arrival mode with the 0.4.0 baselines and under
the 0.3.0 refresh rule with the ``*_v03`` baselines; the second set must
reproduce the numbers printed in release 0.3.0 of the report.

    python benchmarks/reference/report_numbers.py --out results/report_numbers.json
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import pamir
from pamir import (evaluate, evaluate_one, fleet_summary, gbdt_baseline_v03,
                   gbdt_fit, logistic_baseline_v03, logistic_fit)

FLEET = ["gmsc", "taiwan", "gastonstat", "south_german", "bondora", "prosper"]
ARRIVAL = dict(mode="arrival")                        # max_lag_frac=0.2 (default)
REFRESH = dict(mode="refresh", max_lag_frac=None)     # the 0.3.0 rule


def crashes_on_wide_fit(X_train, y_train):
    """Works on narrow tables, raises on wide ones (as in the walkthrough)."""
    if X_train.shape[1] > 15:
        raise RuntimeError("some genuine bug in my model")
    return logistic_fit(X_train, y_train)


def crashes_on_wide_v03(X_train, y_train, X_test):
    if X_train.shape[1] > 15:
        raise RuntimeError("some genuine bug in my model")
    return logistic_baseline_v03(X_train, y_train, X_test)


def hgb60_fit(X_train, y_train):
    return gbdt_fit(X_train, y_train, max_iter=60)


def hgb60_v03(X_train, y_train, X_test):
    return gbdt_baseline_v03(X_train, y_train, X_test, max_iter=60)


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def cadence():
    """Section 4.4: logistic regression on taiwan at three refit cadences."""
    rows = []
    for label, model, kw in (("arrival", logistic_fit, ARRIVAL),
                             ("refresh_v03", logistic_baseline_v03, REFRESH)):
        for k in (10, 40, 100):
            t0 = time.time()
            r = evaluate_one(model, "taiwan", lag=500, k_refit=k, max_n=5000, **kw)
            rows.append({"mode": label, "k_refit": k, "auc_final": _r(r["auc_final"]),
                         "n_refits": r["n_refits"], "n_refits_ok": r["n_refits_ok"],
                         "lag_effective": r["lag_effective"],
                         "wall_s": round(time.time() - t0, 1)})
            print("cadence", rows[-1], flush=True)
    return rows


def coverage():
    """Section 4.5: a complete run against one that fails on wide tables."""
    out = {}
    for label, good, bad, kw in (("arrival", logistic_fit, crashes_on_wide_fit, ARRIVAL),
                                 ("refresh_v03", logistic_baseline_v03,
                                  crashes_on_wide_v03, REFRESH)):
        res = {}
        for name, model in (("complete", good), ("crashes_on_wide", bad)):
            frame = evaluate(model, datasets=FLEET, lag=500, k_refit=40, max_n=5000,
                             verbose=False, **kw)
            s = fleet_summary(frame)
            res[name] = {"auc_mean": _r(s["auc_mean"]),
                         "auc_mean_scored_only": _r(s["auc_mean_scored_only"]),
                         "n_scored": s["n_scored"], "n_datasets": s["n_datasets"],
                         "failed_datasets": s["failed_datasets"],
                         "per_dataset": {d: _r(a) for d, a in
                                         zip(frame["dataset"], frame["auc_final"])}}
        out[label] = res
        print("coverage", label, {k: (v["auc_mean"], v["auc_mean_scored_only"])
                                  for k, v in res.items()}, flush=True)
    return out


def cost():
    """Section 6: bondora, 5,000 rows, k_refit=10, a 60-tree HGB model."""
    rows = []
    for label, model, kw in (("arrival", hgb60_fit, ARRIVAL),
                             ("refresh_v03", hgb60_v03, REFRESH)):
        t0 = time.time()
        r = evaluate_one(model, "bondora", k_refit=10, max_n=5000, **kw)
        rows.append({"mode": label, "n_refits": r["n_refits"], "n_calls": r["n_calls"],
                     "auc_final": _r(r["auc_final"]), "lag_effective": r["lag_effective"],
                     "wall_s": round(time.time() - t0, 1)})
        print("cost", rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/report_numbers.json")
    args = ap.parse_args()
    out = {
        "pamir_version": pamir.__version__,
        "git_commit": os.environ.get("PAMIR_COMMIT"),
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "cadence": cadence(),
        "coverage": coverage(),
        "cost": cost(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
