"""Fidelity metrics for synthetic training data.

Three layers, all optional-import safe:

1. **SDV / SDMetrics** — the full battery: ``QualityReport`` and
   ``DiagnosticReport``, every applicable ``single_column``, ``column_pairs``
   and ``single_table`` metric, plus ML-efficacy and detection arms.
2. **Engine-native** — whatever the generator library ships itself, collected
   through :meth:`GeneratorAdapter.native_fidelity` (zGAN's own KS/TV/
   correlation helpers, for instance).
3. **Fallbacks** — scipy/scikit-learn implementations (KS, Wasserstein, TVD,
   Jensen–Shannon, correlation delta, C2ST, distance-to-closest-record) that
   run with no optional dependency at all, so a report is never empty.

Every metric is computed defensively: a metric that raises is recorded in
``errors`` and the rest of the report still lands.

All comparisons are *train-side only*.  Pass ``holdout`` to additionally probe
for leakage — the held-out fold is used to count overlaps, never to score
fidelity.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

NUMERICAL = "numerical"
CATEGORICAL = "categorical"
DATETIME = "datetime"

# The few columns of a 66-column fidelity frame that decide whether to believe
# an AUC delta.  Read the leakage probe first: anything above zero means a
# synthetic row reproduced a held-out row, and the delta is void regardless of
# how good the rest of the battery looks.
HEADLINE_FIDELITY_COLUMNS = (
    "leak.synth_holdout_only_overlap",  # >0 invalidates the delta outright
    "quality_score",                    # SDV overall, 1.0 = perfect
    "table.C2ST_AUC",                   # 0.5 = indistinguishable from real
    "table.NewRowSynthesis",            # 1.0 = no synthetic row copies a real one
    "table.dcr_p05",                    # 5th pct distance to nearest real row
    "target_rate_delta",                # drift in the default rate
    "n_errors",                         # metrics that failed to compute
)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def infer_sdtype(series: pd.Series, max_categories: int = 20) -> str:
    """Column type in SDV's vocabulary, using PaMIR's 20-distinct-value rule."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME
    if pd.api.types.is_bool_dtype(series):
        return CATEGORICAL
    if pd.api.types.is_numeric_dtype(series):
        return NUMERICAL if series.nunique(dropna=True) > max_categories else CATEGORICAL
    return CATEGORICAL


def build_metadata(
    data: pd.DataFrame,
    target: Optional[str] = None,
    max_categories: int = 20,
) -> Dict:
    """SDV single-table metadata as a plain dict (the form every metric takes)."""
    columns = {}
    for col in data.columns:
        sdtype = CATEGORICAL if col == target else infer_sdtype(data[col], max_categories)
        columns[col] = {"sdtype": sdtype}
    return {"columns": columns}


def _sdv_metadata_object(metadata: Dict):
    """SingleTableMetadata instance, for engine helpers that expect the object."""
    from sdv.metadata import SingleTableMetadata

    return SingleTableMetadata.load_from_dict(metadata)


def _columns_by_type(metadata: Dict, sdtype: str) -> List[str]:
    return [c for c, spec in metadata["columns"].items() if spec.get("sdtype") == sdtype]


# --------------------------------------------------------------------------- #
# categorical type violations
# --------------------------------------------------------------------------- #
def cardinality_violations(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: Dict,
    max_cardinality: int = 200,
    ratio: float = 10.0,
) -> Dict[str, Dict]:
    """Categorical columns the generator emitted as continuous values.

    A credit table codes most of its categoricals as small integers.  A
    generator that adds noise, or decodes a latent space without rounding,
    turns such a column into hundreds of distinct floats.  The column is then
    categorical in the schema and continuous in fact.

    Two consequences follow, and both matter.  Statistically, every
    category-based metric on that column is measuring the wrong thing.
    Computationally, the contingency tables behind the pair metrics and SDV's
    aggregate reports grow with the product of the cardinalities: on a 1k x 20
    credit table this is the difference between a 2.4 s report and a 177 s one.

    A column is flagged when its synthetic cardinality exceeds both
    ``max_cardinality`` and ``ratio`` times the real cardinality.
    """
    flagged: Dict[str, Dict] = {}
    for col in _columns_by_type(metadata, CATEGORICAL):
        if col not in real.columns or col not in synthetic.columns:
            continue
        n_real = int(real[col].nunique(dropna=True))
        n_syn = int(synthetic[col].nunique(dropna=True))
        if n_syn > max_cardinality and n_syn > ratio * max(n_real, 1):
            flagged[col] = {"real_cardinality": n_real, "synthetic_cardinality": n_syn}
    return flagged


def _retype(metadata: Dict, columns) -> Dict:
    """Metadata copy with the named columns treated as numerical."""
    out = {"columns": {c: dict(spec) for c, spec in metadata["columns"].items()}}
    for col in columns:
        if col in out["columns"]:
            out["columns"][col]["sdtype"] = NUMERICAL
    return out


# --------------------------------------------------------------------------- #
# report container
# --------------------------------------------------------------------------- #
@dataclass
class FidelityReport:
    """Everything measured about one (real, synthetic) pair."""

    overall: Dict[str, float] = field(default_factory=dict)
    columns: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["column", "sdtype", "metric", "score"]))
    pairs: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["column_1", "column_2", "metric", "score"]))
    table: Dict[str, float] = field(default_factory=dict)
    leakage: Dict[str, float] = field(default_factory=dict)
    native: Dict[str, Dict] = field(default_factory=dict)
    type_violations: Dict[str, Dict] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    n_real: int = 0
    n_synthetic: int = 0

    def summary(self) -> pd.Series:
        """Headline numbers, flattened for a results table."""
        out = dict(self.overall)
        out.update({f"table.{k}": v for k, v in self.table.items()})
        out.update({f"leak.{k}": v for k, v in self.leakage.items()})
        for engine, metrics in self.native.items():
            out.update({f"native.{engine}.{k}": v for k, v in metrics.items()})
        out["n_real"] = self.n_real
        out["n_synthetic"] = self.n_synthetic
        out["n_type_violations"] = len(self.type_violations)
        out["n_errors"] = len(self.errors)
        return pd.Series(out)

    def column_matrix(self) -> pd.DataFrame:
        """Per-column scores pivoted to column x metric."""
        if self.columns.empty:
            return pd.DataFrame()
        return self.columns.pivot_table(index="column", columns="metric", values="score")

    def to_dict(self) -> Dict:
        return {
            "overall": self.overall,
            "table": self.table,
            "leakage": self.leakage,
            "native": self.native,
            "type_violations": self.type_violations,
            "columns": self.columns.to_dict("records"),
            "pairs": self.pairs.to_dict("records"),
            "errors": self.errors,
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
        }


def _safe(report: FidelityReport, label: str, fn, default=np.nan):
    """Run one metric; record the failure instead of losing the whole report."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = fn()
        return float(value) if value is not None and np.isscalar(value) else value
    except Exception as exc:  # noqa: BLE001 - a metric must never kill the run
        report.errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:160]}")
        return default


# --------------------------------------------------------------------------- #
# fallback metrics (no optional dependency)
# --------------------------------------------------------------------------- #
def _js_distance(real: pd.Series, synth: pd.Series, bins: int = 30) -> float:
    """Jensen–Shannon distance between two columns (0 = identical)."""
    from scipy.spatial.distance import jensenshannon

    if pd.api.types.is_numeric_dtype(real):
        lo = float(np.nanmin([real.min(), synth.min()]))
        hi = float(np.nanmax([real.max(), synth.max()]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(real.dropna(), bins=edges)
        q, _ = np.histogram(synth.dropna(), bins=edges)
    else:
        levels = pd.Index(sorted(set(real.dropna().astype(str)) | set(synth.dropna().astype(str))))
        p = real.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy()
        q = synth.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy()
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def _normalized_wasserstein(real: pd.Series, synth: pd.Series) -> float:
    """Wasserstein distance scaled by the real column's IQR (scale-free)."""
    from scipy.stats import wasserstein_distance

    r = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    s = pd.to_numeric(synth, errors="coerce").dropna().to_numpy()
    if len(r) == 0 or len(s) == 0:
        return np.nan
    scale = np.subtract(*np.percentile(r, [75, 25])) or np.std(r) or 1.0
    return float(wasserstein_distance(r, s) / abs(scale))


def _tv_distance(real: pd.Series, synth: pd.Series) -> float:
    """Total-variation distance between two categorical columns."""
    levels = pd.Index(sorted(set(real.dropna().astype(str)) | set(synth.dropna().astype(str))))
    p = real.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0)
    q = synth.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0)
    return float(0.5 * np.abs(p - q).sum())


def _encode_for_model(real: pd.DataFrame, synth: pd.DataFrame, metadata: Dict) -> tuple:
    """Numeric design matrices for real and synthetic, encoded on the union."""
    cats = [c for c in _columns_by_type(metadata, CATEGORICAL) if c in real.columns]
    nums = [c for c in _columns_by_type(metadata, NUMERICAL) if c in real.columns]
    both = pd.concat([real[nums + cats], synth[nums + cats]], ignore_index=True)
    enc = pd.DataFrame(index=both.index)
    for c in nums:
        col = pd.to_numeric(both[c], errors="coerce")
        enc[c] = col.fillna(col.median())
    for c in cats:
        codes = pd.Categorical(both[c].astype(str)).codes
        enc[c] = codes
    return enc.iloc[: len(real)].to_numpy(), enc.iloc[len(real):].to_numpy()


def c2st_auc(real: pd.DataFrame, synth: pd.DataFrame, metadata: Dict, seed: int = 0) -> float:
    """Classifier two-sample test: AUC of telling synthetic from real.

    The scale is **two-sided** and runs over [0, 1], not [0.5, 1]:

    * ``≈ 0.5`` — indistinguishable to a strong learner, which is the goal.
    * ``> 0.5`` — the samples differ; above roughly 0.8 the synthetic rows are
      trivially identifiable and their value as training data is doubtful.
    * ``< 0.5`` — **duplicates**, not quality.  When synthetic rows repeat real
      ones, cross-validation puts a row's twin in the training fold carrying the
      opposite label, the classifier is systematically wrong on the held-out
      twin, and the AUC falls below chance.  A pure memoriser scores near 0.1.
      Read it with ``dcr_zero_share`` and ``NewRowSynthesis``, and never as
      "better than indistinguishable".

    :func:`fidelity_report` therefore also records ``C2ST_deviation``, the
    distance ``|AUC - 0.5|``, which is the number to rank generators by: it is
    0 for an ideal generator and rises for both failure modes.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    Xr, Xs = _encode_for_model(real, synth, metadata)
    X = np.vstack([Xr, Xs])
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xs))]
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=120, random_state=seed)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def distance_to_closest_record(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    metadata: Dict,
    sample: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """DCR: how close synthetic rows sit to real ones (privacy / memorisation).

    Both sides are capped at ``sample`` rows for cost.  Thinning the *real* side
    can only remove candidate neighbours, so on a table larger than ``sample``
    the reported distances are an upper bound: memorisation looks milder than it
    is, never worse.  Raise ``sample`` when DCR is being read as a privacy
    statement rather than a utility one.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    Xr, Xs = _encode_for_model(real, synth, metadata)
    if len(Xr) > sample:
        Xr = Xr[rng.choice(len(Xr), sample, replace=False)]
    if len(Xs) > sample:
        Xs = Xs[rng.choice(len(Xs), sample, replace=False)]
    scaler = StandardScaler().fit(Xr)
    nn = NearestNeighbors(n_neighbors=1).fit(scaler.transform(Xr))
    dist, _ = nn.kneighbors(scaler.transform(Xs))
    d = dist.ravel()
    return {
        "dcr_mean": float(np.mean(d)),
        "dcr_median": float(np.median(d)),
        "dcr_p05": float(np.percentile(d, 5)),
        "dcr_zero_share": float(np.mean(d < 1e-9)),
    }


def correlation_delta(real: pd.DataFrame, synth: pd.DataFrame, metadata: Dict) -> Dict[str, float]:
    """Mean and max absolute difference of the numeric correlation matrices."""
    nums = [c for c in _columns_by_type(metadata, NUMERICAL) if c in real.columns]
    if len(nums) < 2:
        return {}
    cr = real[nums].apply(pd.to_numeric, errors="coerce").corr()
    cs = synth[nums].apply(pd.to_numeric, errors="coerce").corr()
    diff = (cr - cs).abs().to_numpy()
    iu = np.triu_indices_from(diff, k=1)
    vals = diff[iu]
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {}
    return {"corr_delta_mean": float(vals.mean()), "corr_delta_max": float(vals.max())}


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def fidelity_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: Optional[Dict] = None,
    target: Optional[str] = None,
    holdout: Optional[pd.DataFrame] = None,
    heavy: bool = False,
    max_pairs: int = 60,
    max_category_cardinality: int = 200,
    max_contingency_cells: int = 20_000,
    seed: int = 0,
) -> FidelityReport:
    """Score synthetic rows against the real rows they were trained on.

    Parameters
    ----------
    real : DataFrame
        The real training rows the generator saw — never a test split.
    synthetic : DataFrame
        Generated rows with the same schema.
    metadata : dict, optional
        SDV-style ``{"columns": {name: {"sdtype": ...}}}``; inferred when absent.
    target : str, optional
        Target column name.  Enables ML-efficacy metrics and the target-rate check.
    holdout : DataFrame, optional
        Held-out real rows, used *only* for leakage probes (exact-row overlap).
    heavy : bool
        Also run the slow arms (SVC detection, MLP efficacy, GMLogLikelihood).
    max_pairs : int
        Cap on the number of column pairs scored, to bound O(p^2) work.
    max_category_cardinality : int
        A categorical column whose synthetic cardinality exceeds this (and ten
        times the real cardinality) is reported as a type violation and scored
        as numerical by the pair, table and aggregate arms.  See
        :func:`cardinality_violations`.
    max_contingency_cells : int
        Skip a categorical pair whose contingency table would exceed this many
        cells.  A bound on cost, not a statistical choice.

    Returns
    -------
    FidelityReport
    """
    shared = [c for c in real.columns if c in synthetic.columns]
    real = real[shared].reset_index(drop=True)
    synthetic = synthetic[shared].reset_index(drop=True)
    metadata = metadata or build_metadata(real, target=target)
    metadata = {"columns": {c: metadata["columns"][c] for c in shared if c in metadata["columns"]}}

    rep = FidelityReport(n_real=len(real), n_synthetic=len(synthetic))

    # Columns the generator emitted as continuous but the schema calls
    # categorical are measured on the declared typing (so the violation shows
    # up in TVComplement and CategoryAdherence) and scored as numerical
    # everywhere else, where the declared typing would be both wrong and slow.
    rep.type_violations = cardinality_violations(
        real, synthetic, metadata, max_category_cardinality)
    rep.overall["n_type_violations"] = float(len(rep.type_violations))
    effective = _retype(metadata, rep.type_violations) if rep.type_violations else metadata

    declared_cat = [c for c in _columns_by_type(metadata, CATEGORICAL) if c in shared]
    declared_num = [c for c in _columns_by_type(metadata, NUMERICAL) if c in shared]
    num_cols = [c for c in _columns_by_type(effective, NUMERICAL) if c in shared]
    cat_cols = [c for c in _columns_by_type(effective, CATEGORICAL) if c in shared]

    _sdv_reports(rep, real, synthetic, effective)
    _column_metrics(rep, real, synthetic, metadata, declared_num, declared_cat)
    _pair_metrics(rep, real, synthetic, num_cols, cat_cols, max_pairs, max_contingency_cells)
    _table_metrics(rep, real, synthetic, effective, target, heavy)
    _fallback_metrics(rep, real, synthetic, effective, target, seed)
    if holdout is not None:
        probe_cols = [c for c in shared if c in holdout.columns]
        _leakage_probes(rep, real[probe_cols], synthetic[probe_cols], holdout[probe_cols])

    return rep


def leakage_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame,
) -> FidelityReport:
    """The leakage probes alone, with no distributional metric computed.

    The probes cost milliseconds while the full battery costs seconds, so they
    are deliberately available on their own: switching fidelity off for a fleet
    sweep must never switch off the check that says whether to believe the
    resulting delta.
    """
    shared = [c for c in real.columns
              if c in synthetic.columns and c in holdout.columns]
    rep = FidelityReport(n_real=len(real), n_synthetic=len(synthetic))
    _leakage_probes(rep, real[shared], synthetic[shared], holdout[shared])
    return rep


def _sdv_reports(rep, real, synthetic, metadata) -> None:
    """SDV's own aggregate Quality and Diagnostic reports."""
    try:
        from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
    except ImportError:
        rep.errors.append("sdmetrics reports: not installed")
        return

    def _run(cls, prefix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = cls()
            report.generate(real, synthetic, metadata, verbose=False)
            rep.overall[f"{prefix}_score"] = float(report.get_score())
            for row in report.get_properties().to_dict("records"):
                key = str(row["Property"]).lower().replace(" ", "_")
                rep.overall[f"{prefix}.{key}"] = float(row["Score"])

    _safe(rep, "QualityReport", lambda: _run(QualityReport, "quality"))
    _safe(rep, "DiagnosticReport", lambda: _run(DiagnosticReport, "diagnostic"))


def _column_metrics(rep, real, synthetic, metadata, num_cols, cat_cols) -> None:
    """Every applicable sdmetrics single-column metric, per column."""
    rows: List[Dict] = []
    try:
        from sdmetrics import single_column as sc
    except ImportError:
        rep.errors.append("sdmetrics.single_column: not installed")
        sc = None

    plan = []
    if sc is not None:
        for col in num_cols:
            plan += [
                (col, NUMERICAL, "KSComplement", lambda c=col: sc.KSComplement.compute(real[c], synthetic[c])),
                (col, NUMERICAL, "BoundaryAdherence", lambda c=col: sc.BoundaryAdherence.compute(real[c], synthetic[c])),
                (col, NUMERICAL, "RangeCoverage", lambda c=col: sc.RangeCoverage.compute(real[c], synthetic[c])),
                (col, NUMERICAL, "StatisticSimilarity.mean", lambda c=col: sc.StatisticSimilarity.compute(real[c], synthetic[c], statistic="mean")),
                (col, NUMERICAL, "StatisticSimilarity.median", lambda c=col: sc.StatisticSimilarity.compute(real[c], synthetic[c], statistic="median")),
                (col, NUMERICAL, "StatisticSimilarity.std", lambda c=col: sc.StatisticSimilarity.compute(real[c], synthetic[c], statistic="std")),
            ]
        for col in cat_cols:
            plan += [
                (col, CATEGORICAL, "TVComplement", lambda c=col: sc.TVComplement.compute(real[c], synthetic[c])),
                (col, CATEGORICAL, "CSTest", lambda c=col: sc.CSTest.compute(real[c], synthetic[c])),
                (col, CATEGORICAL, "CategoryCoverage", lambda c=col: sc.CategoryCoverage.compute(real[c], synthetic[c])),
                (col, CATEGORICAL, "CategoryAdherence", lambda c=col: sc.CategoryAdherence.compute(real[c], synthetic[c])),
            ]
        for col in num_cols + cat_cols:
            plan.append((col, None, "MissingValueSimilarity",
                         lambda c=col: sc.MissingValueSimilarity.compute(real[c], synthetic[c])))

    # dependency-free additions that SDV does not provide per column
    for col in num_cols:
        plan.append((col, NUMERICAL, "WassersteinNormalized",
                     lambda c=col: _normalized_wasserstein(real[c], synthetic[c])))
    for col in cat_cols:
        plan.append((col, CATEGORICAL, "TVDistance", lambda c=col: _tv_distance(real[c], synthetic[c])))
    for col in num_cols + cat_cols:
        plan.append((col, None, "JensenShannon", lambda c=col: _js_distance(real[c], synthetic[c])))

    for col, sdtype, name, fn in plan:
        score = _safe(rep, f"{name}[{col}]", fn)
        rows.append({"column": col, "sdtype": sdtype, "metric": name, "score": score})

    if rows:
        rep.columns = pd.DataFrame(rows)
        for metric, grp in rep.columns.groupby("metric"):
            vals = pd.to_numeric(grp["score"], errors="coerce").dropna()
            if len(vals):
                rep.overall[f"col.{metric}.mean"] = float(vals.mean())
                rep.overall[f"col.{metric}.min"] = float(vals.min())


def _pair_metrics(rep, real, synthetic, num_cols, cat_cols, max_pairs,
                  max_contingency_cells: int = 20_000) -> None:
    """Column-pair trends: correlation, contingency, KL divergence."""
    try:
        from sdmetrics import column_pairs as cp
    except ImportError:
        rep.errors.append("sdmetrics.column_pairs: not installed")
        return

    from itertools import combinations

    rows: List[Dict] = []
    num_pairs = list(combinations(num_cols, 2))[:max_pairs]
    cat_pairs = list(combinations(cat_cols, 2))[:max_pairs]

    for a, b in num_pairs:
        for name, fn in (
            ("CorrelationSimilarity.Pearson",
             lambda: cp.CorrelationSimilarity.compute(real[[a, b]], synthetic[[a, b]], coefficient="Pearson")),
            ("CorrelationSimilarity.Spearman",
             lambda: cp.CorrelationSimilarity.compute(real[[a, b]], synthetic[[a, b]], coefficient="Spearman")),
            ("ContinuousKLDivergence",
             lambda: cp.ContinuousKLDivergence.compute(real[[a, b]], synthetic[[a, b]])),
        ):
            rows.append({"column_1": a, "column_2": b, "metric": name,
                         "score": _safe(rep, f"{name}[{a},{b}]", fn)})

    def _cells(a, b) -> int:
        levels = lambda c: max(real[c].nunique(dropna=True), synthetic[c].nunique(dropna=True))
        return levels(a) * levels(b)

    for a, b in cat_pairs:
        if _cells(a, b) > max_contingency_cells:
            rep.errors.append(
                f"contingency[{a},{b}]: skipped, {_cells(a, b)} cells exceed "
                f"max_contingency_cells={max_contingency_cells}")
            continue
        for name, fn in (
            ("ContingencySimilarity",
             lambda: cp.ContingencySimilarity.compute(real[[a, b]], synthetic[[a, b]])),
            ("DiscreteKLDivergence",
             lambda: cp.DiscreteKLDivergence.compute(real[[a, b]], synthetic[[a, b]])),
        ):
            rows.append({"column_1": a, "column_2": b, "metric": name,
                         "score": _safe(rep, f"{name}[{a},{b}]", fn)})

    if rows:
        rep.pairs = pd.DataFrame(rows)
        for metric, grp in rep.pairs.groupby("metric"):
            vals = pd.to_numeric(grp["score"], errors="coerce").dropna()
            if len(vals):
                rep.overall[f"pair.{metric}.mean"] = float(vals.mean())


def _table_metrics(rep, real, synthetic, metadata, target, heavy) -> None:
    """Whole-table metrics: novelty, structure, detection, ML efficacy."""
    try:
        from sdmetrics import single_table as st
    except ImportError:
        rep.errors.append("sdmetrics.single_table: not installed")
        return

    rep.table["NewRowSynthesis"] = _safe(
        rep, "NewRowSynthesis",
        lambda: st.NewRowSynthesis.compute(
            real, synthetic, metadata,
            numerical_match_tolerance=0.01,
            synthetic_sample_size=min(len(synthetic), 2000)),
    )
    rep.table["TableStructure"] = _safe(
        rep, "TableStructure", lambda: st.TableStructure.compute(real, synthetic))
    rep.table["LogisticDetection"] = _safe(
        rep, "LogisticDetection", lambda: st.LogisticDetection.compute(real, synthetic, metadata))

    if heavy:
        rep.table["SVCDetection"] = _safe(
            rep, "SVCDetection", lambda: st.SVCDetection.compute(real, synthetic, metadata))
        num_only = [c for c in _columns_by_type(metadata, NUMERICAL) if c in real.columns]
        if len(num_only) >= 2:
            rep.table["GMLogLikelihood"] = _safe(
                rep, "GMLogLikelihood",
                lambda: st.GMLogLikelihood.compute(real[num_only], synthetic[num_only]))

    if target and target in real.columns and real[target].nunique() == 2:
        arms = [("BinaryLogisticRegression", st.BinaryLogisticRegression),
                ("BinaryAdaBoostClassifier", st.BinaryAdaBoostClassifier),
                ("BinaryDecisionTreeClassifier", st.BinaryDecisionTreeClassifier)]
        if heavy:
            arms.append(("BinaryMLPClassifier", st.BinaryMLPClassifier))
        for name, metric in arms:
            rep.table[f"efficacy.{name}"] = _safe(
                rep, name,
                lambda m=metric: m.compute(test_data=real, train_data=synthetic,
                                           target=target, metadata=metadata),
            )


def _fallback_metrics(rep, real, synthetic, metadata, target, seed) -> None:
    """Dependency-free metrics that run whatever else is installed."""
    rep.table["C2ST_AUC"] = _safe(rep, "C2ST", lambda: c2st_auc(real, synthetic, metadata, seed))
    auc = rep.table["C2ST_AUC"]
    # Distance from indistinguishable. Ranking on the raw AUC would put a
    # memoriser (≈0.1) ahead of an ideal generator (≈0.5); this does not.
    rep.table["C2ST_deviation"] = abs(auc - 0.5) if auc == auc else np.nan
    rep.table.update(_safe(rep, "DCR",
                           lambda: distance_to_closest_record(real, synthetic, metadata, seed=seed),
                           default={}) or {})
    rep.overall.update(_safe(rep, "correlation_delta",
                             lambda: correlation_delta(real, synthetic, metadata),
                             default={}) or {})
    if target and target in real.columns:
        r = pd.to_numeric(real[target], errors="coerce")
        s = pd.to_numeric(synthetic[target], errors="coerce")
        if r.notna().any() and s.notna().any():
            rep.overall["target_rate_real"] = float(r.mean())
            rep.overall["target_rate_synthetic"] = float(s.mean())
            rep.overall["target_rate_delta"] = float(s.mean() - r.mean())


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]


def _row_hashes(df: pd.DataFrame, numeric_cols: Optional[Sequence[str]] = None) -> pd.Series:
    """Stable per-row hash, used for exact-overlap probes.

    Values are normalised before hashing because ``hash_pandas_object`` keys on
    dtype as well as on value: ``int64(1)``, ``float64(1.0)``, ``True`` and the
    string ``"1"`` all hash to different numbers.  Engines routinely change a
    column's representation — a float for an integer-coded column, a string for
    everything when the pool arrived as CSV — so hashing the frames as they
    come would report a generator that reproduces held-out rows *verbatim* as
    perfectly clean.

    ``numeric_cols`` names the columns to read as numbers; the caller passes the
    *real* frame's numeric columns so that every frame in a comparison is
    normalised under one schema rather than under its own dtypes.  Left unset,
    each frame is judged on its own, which is only safe for a single frame.
    """
    numeric = set(_numeric_columns(df) if numeric_cols is None else numeric_cols)
    norm = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in numeric:
            norm[col] = pd.to_numeric(df[col], errors="coerce").astype("float64").round(10)
        else:
            norm[col] = df[col].astype(str)
    return pd.util.hash_pandas_object(norm, index=False)


def _leakage_probes(rep, real, synthetic, holdout) -> None:
    """Confirm the generator never reproduced held-out rows.

    A synthetic row matching a *training* row is memorisation — expected to be
    small but not alarming.  A synthetic row matching a *held-out* row that is
    not also in train can only come from the generator having seen test data.

    Shares are counted over synthetic *rows*, not over distinct row hashes, so a
    generator that emits the same memorised row a hundred times is charged for a
    hundred rows.
    """
    try:
        # One schema for all three frames, taken from the real rows: a column is
        # whatever the real table says it is, whatever the generator returned.
        numeric = _numeric_columns(real)
        syn = _row_hashes(synthetic, numeric)
        h_real = set(_row_hashes(real, numeric))
        h_out = set(_row_hashes(holdout, numeric))
        train_only = h_real - h_out
        holdout_only = h_out - h_real
        n_syn = max(len(syn), 1)
        n_train_match = int(syn.isin(train_only).sum())
        n_holdout_match = int(syn.isin(holdout_only).sum())
        rep.leakage = {
            "synth_train_exact_overlap": n_train_match / n_syn,
            "synth_holdout_only_overlap": n_holdout_match / n_syn,
            "n_synth_matching_train_only": float(n_train_match),
            "n_synth_matching_holdout_only": float(n_holdout_match),
        }
    except Exception as exc:  # noqa: BLE001
        rep.errors.append(f"leakage_probe: {type(exc).__name__}: {str(exc)[:160]}")
