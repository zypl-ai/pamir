"""Load a PaMIR dataset as (X, y, meta).

PaMIR does not ship the datasets; the harmonized parquet lives in a local
cache and is fetched from the dataset's original source on first use (see
:mod:`pamir.download`).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pamir.catalog import dataset_info


def load_dataset(
    dataset_id: str,
    max_rows: Optional[int] = None,
    auto_download: bool = True,
    strict: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Load a single dataset from the local cache.

    Parameters
    ----------
    dataset_id : str
        One of the PaMIR dataset ids (see ``pamir.list_datasets()``).
    max_rows : int, optional
        Truncate to at most this many rows (in stream order).
    auto_download : bool
        If the dataset is not cached, fetch and harmonize it from source
        (default True).  Set False to require an explicit ``pamir.download()``.
    strict : bool
        The cached table is checked against its contract in ``expected.json``
        on every load (a cache built by an older release, or from a drifted
        source snapshot, is caught here).  If True (default), a mismatch raises
        :class:`pamir.contract.ContractError`; if False, the table is returned
        with ``meta["contract_ok"] = False``.

    Returns
    -------
    X : DataFrame
        Feature columns exactly as stored: ``int64``, ``float64``, ``bool`` and
        ``object`` all occur, and most of the datasets carry at least one
        non-numeric column.  Nothing is encoded or imputed for you — see
        :func:`pamir.encode_features` for the glue most models need.
    y : ndarray of int
        Binary target (1 = default, 0 = non-default).
    meta : dict
        Dataset metadata from the catalog, plus ``n_rows``, ``n_defaults``,
        ``n_features`` (after truncation), ``contract_ok``,
        ``contract_issues`` and ``snapshot_verified`` (True when the cache
        records the raw-file digest and it equals the pinned one).  The
        catalog's ``notes`` key is present only on the datasets that required
        harmonization, so read it with ``.get()``.
    """
    from pamir.contract import EXPECTED, ContractError, check_table, read_record
    from pamir.download import cache_dir, download

    info = dataset_info(dataset_id)
    pq = cache_dir() / f"{dataset_id}.parquet"
    if not pq.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"'{dataset_id}' is not cached at {pq}. "
                f"Run `pamir.download('{dataset_id}')` (needs pamir[data]; "
                f"Kaggle sources also need Kaggle credentials)."
            )
        pq = download(dataset_id, strict=strict)
    df = pd.read_parquet(pq)
    if "__target__" not in df.columns:
        raise ValueError(f"No __target__ column in {pq}")

    record = read_record(pq)
    issues = check_table(dataset_id, df, raw_sha256=record.get("raw_sha256"))
    if issues and strict:
        raise ContractError(
            dataset_id, issues,
            hint=(f"The cached table at {pq} is not the benchmark table. Rebuild it "
                  f"with pamir.download('{dataset_id}', force=True), or pass "
                  "strict=False to load it anyway (flagged contract_ok=False)."))
    pinned_raw = EXPECTED.get(dataset_id, {}).get("raw_sha256")

    y = df["__target__"].to_numpy().astype(int)
    X = df.drop(columns=["__target__"])

    if max_rows is not None:
        X = X.iloc[:max_rows]
        y = y[:max_rows]

    meta = dict(info)
    meta["n_rows"] = len(y)
    meta["n_defaults"] = int(y.sum())
    meta["n_features"] = X.shape[1]
    meta["contract_ok"] = not issues
    meta["contract_issues"] = issues
    meta["snapshot_verified"] = bool(pinned_raw) and record.get("raw_sha256") == pinned_raw
    return X, y, meta
