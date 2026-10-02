"""The comparisons fixed in PLAN.md, computed from the reference runs.

Reads ``<results>/<model>__<protocol>/{table.csv,summary.json}`` as written by
``run_reference.py`` and writes ``comparisons.json`` plus two LaTeX tables
(per-dataset AUC, label-budget curve) next to them.  Nothing here selects or
excludes a dataset: every comparison uses every dataset the two runs share.

    python benchmarks/reference/compare.py --results results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MODELS = ("logistic", "gbdt")
PROTOCOLS = ("arrival", "iid_20k", "iid_full", "refresh", "arrival_lag250")
BUDGETS = (0, 100, 300, 1000, 3000, 10000)


def load(root: Path):
    runs = {}
    for m in MODELS:
        for p in PROTOCOLS:
            d = root / f"{m}__{p}"
            if not (d / "table.csv").exists():
                continue
            t = pd.read_csv(d / "table.csv")
            t["auc"] = t["auc_final"] if "auc_final" in t else t["auc_mean"]
            runs[(m, p)] = {"table": t.set_index("dataset"),
                            "summary": json.loads((d / "summary.json").read_text())}
    return runs


def paired(a: pd.Series, b: pd.Series):
    """a - b over the datasets both runs scored."""
    d = (a - b).dropna()
    out = {"n_datasets": int(len(d)), "mean_diff": float(d.mean()) if len(d) else None,
           "median_diff": float(d.median()) if len(d) else None,
           "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
           "per_dataset": {k: round(float(v), 4) for k, v in d.items()}}
    if len(d) >= 5 and (d != 0).any():
        res = wilcoxon(d)
        out["wilcoxon_statistic"] = float(res.statistic)
        out["wilcoxon_p"] = float(res.pvalue)
    return out


def fleet(runs, m, p):
    s = runs[(m, p)]["summary"]["summary"]
    return {"auc_mean": s["auc_mean"], "auc_mean_scored_only": s["auc_mean_scored_only"],
            "complete": s["complete"], "contract_ok": s["contract_ok"],
            "row_independent": s["row_independent"], "n_scored": s["n_scored"]}


def compare(runs):
    have = lambda m, p: (m, p) in runs
    auc = lambda m, p: runs[(m, p)]["table"]["auc"]
    out = {"fleet": {f"{m}__{p}": fleet(runs, m, p) for (m, p) in runs}}
    out["headroom_gbdt_minus_logistic"] = {
        p: paired(auc("gbdt", p), auc("logistic", p))
        for p in ("arrival", "iid_20k") if have("gbdt", p) and have("logistic", p)}
    out["protocol_gap_iid20k_minus_arrival"] = {
        m: paired(auc(m, "iid_20k"), auc(m, "arrival"))
        for m in MODELS if have(m, "iid_20k") and have(m, "arrival")}
    if all(have(m, p) for m in MODELS for p in ("arrival", "iid_20k")):
        better = {p: (auc("gbdt", p) > auc("logistic", p)) for p in ("arrival", "iid_20k")}
        both = pd.concat(better, axis=1).dropna()
        changed = both.index[both["arrival"] != both["iid_20k"]].tolist()
        out["ranking_change"] = {"n_datasets": int(len(both)), "n_changed": len(changed),
                                 "datasets": changed,
                                 "gbdt_better_arrival": int(both["arrival"].sum()),
                                 "gbdt_better_iid_20k": int(both["iid_20k"].sum())}
    out["label_budget"] = {
        m: runs[(m, "arrival")]["summary"]["summary"].get("auc_by_budget")
        for m in MODELS if have(m, "arrival")}
    out["delay_arrival_minus_lag250"] = {
        m: paired(auc(m, "arrival"), auc(m, "arrival_lag250"))
        for m in MODELS if have(m, "arrival") and have(m, "arrival_lag250")}
    out["refresh_minus_arrival"] = {
        m: paired(auc(m, "refresh"), auc(m, "arrival"))
        for m in MODELS if have(m, "refresh") and have(m, "arrival")}
    out["manifests"] = {f"{m}__{p}": runs[(m, p)]["summary"]["manifest"] for (m, p) in runs}
    return out


def _fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def latex_tables(runs, order):
    cols = [(m, p) for m in MODELS for p in ("arrival", "refresh", "iid_20k")]
    lines = []
    for d in order:
        cells = [_fmt(runs[c]["table"]["auc"].get(d)) if c in runs else "--" for c in cols]
        lag = runs[("logistic", "arrival")]["table"]["lag_effective"].get(d) \
            if ("logistic", "arrival") in runs else None
        lines.append(f"\\texttt{{{d.replace('_', chr(92) + '_')}}} & "
                     f"{'--' if lag is None else int(lag)} & " + " & ".join(cells) + r" \\")
    means = [_fmt(fleet(runs, *c)["auc_mean"]) if c in runs else "--" for c in cols]
    lines.append(r"\midrule")
    lines.append(r"\textbf{Fleet mean} & & " + " & ".join(means) + r" \\")
    per_dataset = "\n".join(lines)

    blines = []
    for lo, hi in zip(BUDGETS, list(BUDGETS[1:]) + [None]):
        label = f"{lo:,}--{hi - 1:,}" if hi else f"$\\geq$ {lo:,}"
        cells = []
        for m in MODELS:
            b = (runs.get((m, "arrival"), {}).get("summary", {}).get("summary", {})
                 .get("auc_by_budget", {}).get(f"auc_budget_{lo}") or {})
            cells += [_fmt(b.get("mean")), str(b.get("n_datasets", "--"))]
        blines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return per_dataset, "\n".join(blines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--order", default=None,
                    help="comma-separated dataset order for the LaTeX rows (default: catalogue)")
    args = ap.parse_args()
    root = Path(args.results)
    runs = load(root)
    out = compare(runs)
    (root / "comparisons.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    import pamir
    cat = pamir.load_catalog()
    wanted = args.order.split(",") if args.order else list(cat.index)
    order = [d for d in wanted if d in runs[("logistic", "arrival")]["table"].index] \
        if ("logistic", "arrival") in runs else []
    per_dataset, budget = latex_tables(runs, order)
    (root / "tab_reference_rows.tex").write_text(per_dataset + "\n")
    (root / "tab_budget_rows.tex").write_text(budget + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("manifests",)},
                     indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
