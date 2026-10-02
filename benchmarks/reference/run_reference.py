"""Reference results for PaMIR 0.4.0 (see PLAN.md in this folder).

One invocation runs one model under one protocol on every dataset (or a
subset), one dataset per worker process, and writes

    <out>/<model>__<protocol>/<dataset>.json   full evaluate_one / evaluate_iid_one result
    <out>/<model>__<protocol>/table.csv        one row per dataset
    <out>/<model>__<protocol>/summary.json     fleet_summary + run manifest

A dataset whose JSON already exists is skipped, so an interrupted run resumes.

    python benchmarks/reference/run_reference.py --model gbdt --protocol arrival --out results
    python benchmarks/reference/run_reference.py --model logistic --protocol iid_full --out results
    python benchmarks/reference/run_reference.py --smoke --out smoke    # 7 open datasets, short streams
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS = ("logistic", "gbdt")
PROTOCOLS = {
    # name: (evaluator, keyword arguments) — the reference setting is "arrival".
    "arrival": ("stream", dict(mode="arrival")),
    "arrival_lag250": ("stream", dict(mode="arrival", lag=250)),
    "refresh": ("stream", dict(mode="refresh")),
    "iid_full": ("iid", dict(max_n=None)),
    "iid_20k": ("iid", dict(max_n=20000)),
}
# Larger, higher-default-rate streams refit most; start them first.
_COST_ORDER = ["bondora", "prosper", "pakdd", "lc_my", "taiwan", "laotse", "lt_vehicle",
               "lc_clean", "dish", "gmsc", "lc_small", "gastonstat", "poland_3yr", "sba",
               "poland_5yr", "poland_1yr", "bankruptcy", "conorsully", "south_german"]


def _model(name, kind):
    import pamir
    if name == "logistic":
        return pamir.logistic_fit if kind == "fit" else pamir.logistic_baseline
    if name == "gbdt":
        return pamir.gbdt_fit if kind == "fit" else pamir.gbdt_baseline
    raise ValueError(name)


def _run_one(model_name, protocol, dataset, smoke):
    import pamir
    evaluator, kw = PROTOCOLS[protocol]
    kw = dict(kw)
    t0 = time.time()
    if evaluator == "stream":
        if smoke:
            kw.update(max_n=2000, lag=min(kw.get("lag", 1000), 300))
        kind = "predict" if kw.get("mode") == "refresh" else "fit"
        res = pamir.evaluate_one(_model(model_name, kind), dataset, on_error="warn", **kw)
    else:
        if smoke:
            kw.update(max_n=2000, n_seeds=2)
        res = pamir.evaluate_iid_one(_model(model_name, "predict"), dataset,
                                     on_error="warn", **kw)
    res["wall_seconds"] = round(time.time() - t0, 2)
    res["protocol_kwargs"] = kw
    return res


def _git_commit():
    env = os.environ.get("PAMIR_COMMIT")
    if env:
        return env
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _manifest(args, started):
    import numpy, pandas, sklearn, pamir
    return {
        "pamir_version": pamir.__version__,
        "git_commit": _git_commit(),
        "model": args.model, "protocol": args.protocol,
        "protocol_kwargs": PROTOCOLS[args.protocol][1], "smoke": args.smoke,
        "python": platform.python_version(), "numpy": numpy.__version__,
        "pandas": pandas.__version__, "scikit_learn": sklearn.__version__,
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "jobs": args.jobs, "threads_per_job": args.threads,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="PaMIR 0.4.0 reference results")
    ap.add_argument("--model", choices=MODELS, default="logistic")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="arrival")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=8, help="datasets run in parallel")
    ap.add_argument("--threads", type=int, default=4, help="BLAS/OpenMP threads per job")
    ap.add_argument("--smoke", action="store_true",
                    help="7 credential-free datasets, 2,000-row streams (pipeline check only)")
    args = ap.parse_args(argv)

    # Set before any worker imports numpy / scikit-learn.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    sys.path.insert(0, str(ROOT))
    import pamir

    datasets = args.datasets or (pamir.open_datasets() if args.smoke else pamir.list_datasets())
    order = {d: i for i, d in enumerate(_COST_ORDER)}
    datasets = sorted(datasets, key=lambda d: order.get(d, len(order)))
    out = Path(args.out) / f"{args.model}__{args.protocol}"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    todo = [d for d in datasets if not (out / f"{d}.json").exists()]
    print(f"{args.model} / {args.protocol}: {len(datasets)} datasets, {len(todo)} to run",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_one, args.model, args.protocol, d, args.smoke): d
                   for d in todo}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # an evaluator error, not a model failure
                res = {"dataset": d, "harness_error": f"{type(exc).__name__}: {exc}"}
            (out / f"{d}.json").write_text(json.dumps(res, default=str) + "\n")
            auc = res.get("auc_final", res.get("auc_mean"))
            print(f"  {d}: auc={auc} wall={res.get('wall_seconds')}s "
                  f"{res.get('harness_error', '')}", flush=True)

    import pandas as pd
    rows = [json.loads((out / f"{d}.json").read_text()) for d in datasets]
    if PROTOCOLS[args.protocol][0] == "stream":
        from pamir.evaluate import BUDGET_COLUMNS, RESULT_COLUMNS
        cols = RESULT_COLUMNS + (BUDGET_COLUMNS if rows and "auc_budget_0" in rows[0] else [])
    else:
        from pamir.evaluate_iid import RESULT_COLUMNS
        cols = list(RESULT_COLUMNS)
    cols = cols + ["wall_seconds"]
    table = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    table.to_csv(out / "table.csv", index=False)
    summary = pamir.fleet_summary(table)
    (out / "summary.json").write_text(json.dumps(
        {"summary": summary, "manifest": _manifest(args, started)}, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_scored", "n_datasets", "complete",
                                               "contract_ok", "row_independent", "auc_mean")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
