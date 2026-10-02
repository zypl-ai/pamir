"""Harmonize a raw source file into the PaMIR (features, target) table.

The logic here is a port of the reference loader used to build the benchmark:
a single generic pipeline driven by each dataset's ``harmonize`` spec in
``catalog.json``, plus a small number of per-dataset special cases (bespoke
target rules, encodings, one CR/LF fix, one rescale).

PaMIR does not ship the datasets themselves.  ``pamir.download`` fetches the
raw file from its original source; this module turns that raw file into the
exact table the benchmark evaluates, leakage-free:

1. read the raw file (dataset-specific separator / encoding),
2. derive the binary ``__target__`` (numeric parse, or a dataset target rule),
3. drop target, id, date/timestamp and unnamed-index columns,
4. coerce numeric-looking string columns; drop constant / all-missing columns,
5. drop post-outcome columns the elicitation marked ``day_zero_available=False``
   (the leakage-free semantic cut), plus any dataset-specific extra drops,
6. shuffle rows with a fixed seed (file order carries no origination time).
"""

import re
import warnings
from io import StringIO
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def scan_pure_levels(df: pd.DataFrame, y, min_n: int = 30, min_dr: float = 0.9,
                     min_recall: float = 0.5, max_levels: int = 500) -> List:
    """Column levels that are almost entirely defaults AND carry most of them.

    A tell-tale of a label defined by a RULE rather than observed (as in the
    rejected uz_fintech, where ``Field == 0`` and ``Score_point == '-'`` each
    covered ~90% of defaults). Meant to be run on the RAW file, before any
    columns are dropped — the harmonized table no longer contains the defining
    columns, so the leakage-free guards cannot see it there.

    Returns ``(column, level, dr, n, recall)`` tuples. A hit is not
    automatically a defect (a lender's worst grade can be near-all-default);
    it means *explain this level* before shipping the dataset.
    """
    y = np.asarray(y)
    n_def = int(y.sum())
    hits = []
    for c in df.columns:
        s = df[c].astype(str).fillna("__NA__")
        if s.nunique() > max_levels:
            continue
        for lvl in s.unique():
            m = s.eq(lvl).to_numpy()
            if m.sum() < min_n:
                continue
            dr = y[m].mean()
            recall = y[m].sum() / max(n_def, 1)
            if dr >= min_dr and recall >= min_recall:
                hits.append((c, lvl, float(dr), int(m.sum()), float(recall)))
    return hits

_DATE_RE = re.compile(
    r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}[-/ ][A-Za-z]{3,9}[-/ ]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
    r"([ T]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?)?\s*Z?\s*$")

_ID_NAMES = {"id", "customer_id", "cust_id", "loanid", "uniqueid",
             "loan id", "customer id", "listingkey", "listingnumber",
             "loannumber", "selected", "loannr_chkdgt", "user_id",
             "loankey", "memberkey", "groupkey", "id_client", "sk_id_curr"}


def _detect_date_cols(df: pd.DataFrame):
    """Date/timestamp columns, recognised by the shape of their values."""
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        sv = s.dropna()
        if len(sv) == 0:
            continue
        samp = sv.astype(str).sample(min(300, len(sv)), random_state=0)
        if samp.map(lambda v: bool(_DATE_RE.match(v))).mean() >= 0.8:
            out.append(c)
    return out


def _read_raw(path: Path, dl: Dict) -> pd.DataFrame:
    sep = dl.get("sep", ",")
    enc = dl.get("encoding", "utf-8")
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    if str(path).endswith(".arff") or dl.get("format") == "arff":
        from scipy.io import arff as _arff
        data, _meta = _arff.loadarff(
            StringIO(Path(path).read_text(encoding=enc, errors="replace")))
        df = pd.DataFrame(data)
        for c in df.columns:  # ARFF nominal values come back as bytes
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
        return df
    if dl.get("fix_crlf"):
        text = Path(path).read_text(encoding=enc).replace("\r\n", "\n").replace("\r", "\n")
        return pd.read_csv(StringIO(text), sep=sep)
    if dl.get("header") == "none":
        # headerless raw file; column names come from the spec (e.g. pakdd,
        # whose raw modeling data is rebuilt against its variable list)
        return pd.read_csv(path, sep=sep, encoding=enc, header=None,
                           names=dl["columns"], low_memory=False)
    return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)


def _apply_target(df: pd.DataFrame, h: Dict) -> pd.DataFrame:
    """Attach a binary __target__ (1 = default) and drop rows without one."""
    rule = h.get("target_rule", "numeric")
    tcol = h["target"]
    if rule == "bondora":
        df = df[df["Status"].isin(["Repaid", "Late"])].copy()
        df["__target__"] = (df["Status"] == "Late").astype(int)
    elif rule == "prosper":
        df = df[df["LoanStatus"].isin(["Completed", "Chargedoff", "Defaulted"])].copy()
        df["__target__"] = df["LoanStatus"].isin(["Chargedoff", "Defaulted"]).astype(int)
    elif rule == "lc_my":
        df = df[df["Loan Status"].isin(["Fully Paid", "Charged Off"])].copy()
        df["__target__"] = (df["Loan Status"] == "Charged Off").astype(int)
    elif rule == "gastonstat":
        df = df[df["Status"] != 0].copy()
        df["__target__"] = (df["Status"] == 2).astype(int)
    elif rule == "south_german":
        df = df.copy()
        df["__target__"] = (df["kredit"] == 0).astype(int)
    elif rule == "label_bad0":
        # label: 1 = good credit, 0 = bad -> default is the 0 class
        df = df.copy()
        lab = pd.to_numeric(df[tcol], errors="coerce")
        df["__target__"] = np.where(lab == 0, 1.0, np.where(lab == 1, 0.0, np.nan))
    else:  # numeric
        df = df.copy()
        df["__target__"] = pd.to_numeric(df[tcol], errors="coerce")
    return df


def harmonize(raw_path, spec: Dict) -> pd.DataFrame:
    """Turn a raw source file into the harmonized ``(features + __target__)`` table.

    Parameters
    ----------
    raw_path : path
        Local path to the raw file fetched from the dataset's original source.
    spec : dict
        The dataset's catalog entry (must contain ``download`` and ``harmonize``).

    Returns
    -------
    DataFrame with feature columns followed by ``__target__`` (0/1) as the last
    column; rows shuffled with the spec's fixed seed.
    """
    dl, h = spec["download"], spec["harmonize"]
    df = _read_raw(Path(raw_path), {**dl, **{k: h[k] for k in ("fix_crlf",) if k in h}})

    df.columns = [str(c).strip() for c in df.columns]

    if h.get("rescale_credit_score_gt850") and "Credit Score" in df.columns:
        cs = pd.to_numeric(df["Credit Score"], errors="coerce")
        df["Credit Score"] = np.where(cs > 850, cs / 10.0, cs)

    df = _apply_target(df, h)

    # raw-file check (before any columns are dropped): a column level that is
    # nearly all-default and carries most defaults suggests a rule-defined label.
    #
    # Scan every column, but warn only about columns the spec does not already
    # account for. A column listed in day_zero_drop / extra_drop has been looked
    # at and explained -- sba's MIS_Status, for instance, is the raw status the
    # target is derived from, so its CHGOFF level reproduces the label exactly
    # (DR 1.00, recall 1.00) and warning about it every load would train the
    # reader to ignore the one signal meant to be read by a human. Columns are
    # still scanned rather than skipped, so silencing a genuine hit means adding
    # it to the spec -- a deliberate, reviewable change to catalog.json, not the
    # default outcome.
    _raw_feat = df.drop(columns=[c for c in ("__target__", h["target"]) if c in df.columns],
                        errors="ignore")
    _explained = set(h.get("day_zero_drop", [])) | set(h.get("extra_drop", []))
    for _pref in h.get("drop_prefix", []):
        _explained |= {c for c in _raw_feat.columns if c.startswith(_pref)}
    pure = [hit for hit in scan_pure_levels(_raw_feat, df["__target__"].to_numpy())
            if hit[0] not in _explained]
    if pure:
        warnings.warn(
            f"[{spec.get('id', '?')}] near-all-default levels in the raw file "
            f"(possible rule-defined label): "
            + "; ".join(f"{c}=={lvl!r} DR={dr:.2f} n={n} recall={r:.2f}"
                        for c, lvl, dr, n, r in pure[:5]),
            stacklevel=2)

    # columns to drop: target, ids, dates, unnamed index
    drop_cols = {"__target__", h["target"]}
    for c in df.columns:
        if c.lower().strip() in _ID_NAMES:
            drop_cols.add(c)
    for c in _detect_date_cols(df):
        if c != h["target"]:
            drop_cols.add(c)
    if len(df.columns) and (df.columns[0] == "" or str(df.columns[0]).startswith("Unnamed")):
        drop_cols.add(df.columns[0])

    y = df["__target__"].to_numpy(dtype=float)
    keep = np.isfinite(y)
    feats = df.loc[keep].drop(columns=[c for c in drop_cols if c in df.columns],
                              errors="ignore").reset_index(drop=True)
    y = y[keep].astype(int)

    # coerce numeric-looking string columns (>20 distinct, >=99% parseable)
    for c in list(feats.columns):
        if pd.api.types.is_numeric_dtype(feats[c]):
            continue
        raw = feats[c].dropna().astype(str).str.strip()
        if raw.empty or raw.nunique() <= 20:
            continue

        def _strip_seps(s):
            for ch in (" ", "\t", " ", ","):
                s = s.str.replace(ch, "", regex=False)
            return s

        if float(pd.to_numeric(_strip_seps(raw), errors="coerce").notna().mean()) >= 0.99:
            feats[c] = pd.to_numeric(_strip_seps(feats[c].astype(str).str.strip()),
                                     errors="coerce")

    # drop constant / all-missing columns
    const = [c for c in feats.columns if feats[c].nunique(dropna=True) <= 1]
    if const:
        feats = feats.drop(columns=const)

    # drop surrogate-key columns (one distinct value per row: a name or id that
    # carries no generalizable signal, a row-order leakage channel). Caught by
    # CONTENT, not name: string columns, or integer columns (a numeric id that
    # escaped the _ID_NAMES list). Continuous floats may legitimately be unique.
    surrogate = [c for c in feats.columns
                 if feats[c].nunique(dropna=True) == len(feats)
                 and (not pd.api.types.is_numeric_dtype(feats[c])
                      or pd.api.types.is_integer_dtype(feats[c]))]
    if surrogate:
        feats = feats.drop(columns=surrogate)

    # leakage-free semantic cut + dataset-specific drops
    extra = set(h.get("day_zero_drop", [])) | set(h.get("extra_drop", []))
    for pref in h.get("drop_prefix", []):
        extra |= {c for c in feats.columns if c.startswith(pref)}
    feats = feats.drop(columns=[c for c in extra if c in feats.columns], errors="ignore")

    feats["__target__"] = y
    # shuffle rows deterministically (file order is not an origination sequence)
    feats = feats.sample(frac=1.0, random_state=int(h.get("shuffle_seed", 1234))
                         ).reset_index(drop=True)
    cols = [c for c in feats.columns if c != "__target__"] + ["__target__"]
    return feats[cols]
