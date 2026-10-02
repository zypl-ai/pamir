"""The data contract: what a rebuilt PaMIR table must be.

PaMIR ships recipes, not data, so every user rebuilds every table from its
source.  ``data/expected.json`` freezes the result of that rebuild per dataset:
the feature columns, the row and default counts and, once recorded, two
SHA-256 digests — of the raw source file (``raw_sha256``) and of the target
vector in stored order (``target_sha256``).  A table that does not match is a
different benchmark, so by default it is refused rather than evaluated:

* :func:`pamir.download` raises :class:`ContractError` before caching a
  mismatching table (``strict=False`` caches it anyway, flagged);
* :func:`pamir.load_dataset` re-checks a cached table on every load, which
  catches caches built by an older release or from a drifted snapshot;
* the evaluators carry ``contract_ok`` into every result row, and
  :func:`pamir.fleet_summary` withholds the fleet mean unless it is true for
  every dataset.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_EXPECTED_PATH = Path(__file__).parent / "data" / "expected.json"
EXPECTED: Dict[str, Dict] = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))

# Key under which PaMIR stores its provenance record in the parquet schema.
PARQUET_META_KEY = b"pamir"


class ContractError(ValueError):
    """A rebuilt or cached table does not match ``expected.json``."""

    def __init__(self, dataset_id: str, issues: List[str], hint: str = ""):
        self.dataset_id = dataset_id
        self.issues = list(issues)
        message = (f"'{dataset_id}' does not match its data contract "
                   f"(expected.json): " + "; ".join(self.issues))
        if hint:
            message += f". {hint}"
        super().__init__(message)


def file_sha256(path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def target_sha256(y) -> str:
    """SHA-256 of a 0/1 target vector in its stored order.

    The harmonizer shuffles rows with a fixed seed, so equal digests mean the
    same rows, with the same labels, in the same stream order.
    """
    return hashlib.sha256(np.asarray(y, dtype=np.int8).tobytes()).hexdigest()


def check_table(dataset_id: str, df: pd.DataFrame,
                raw_sha256: Optional[str] = None) -> List[str]:
    """Compare a harmonized table with its frozen contract.

    Parameters
    ----------
    dataset_id : str
        PaMIR dataset id.
    df : DataFrame
        The full harmonized table (features plus ``__target__``), untruncated.
    raw_sha256 : str, optional
        Digest of the raw file the table was built from, when known.

    Returns
    -------
    list of str
        One message per mismatch; empty when the table matches.
    """
    exp = EXPECTED.get(dataset_id)
    if exp is None:
        return ["no expected.json entry, so the table cannot be validated"]
    if "__target__" not in df.columns:
        return ["no __target__ column"]

    issues: List[str] = []
    got = [c for c in df.columns if c != "__target__"]
    want = list(exp["columns"])
    if len(got) != len(want) or set(got) != set(want):
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        issues.append(f"features {len(got)} != expected {len(want)} "
                      f"(missing {missing[:6]}, extra {extra[:6]})")

    y = pd.to_numeric(df["__target__"], errors="coerce").to_numpy()
    if len(df) != exp["n_rows"]:
        issues.append(f"rows {len(df):,} != expected {exp['n_rows']:,}")
    n_defaults = int(np.nansum(y))
    if n_defaults == 0:
        issues.append("target parsed as all-zero (harmonization or source problem)")
    elif n_defaults != exp["n_defaults"]:
        issues.append(f"defaults {n_defaults:,} != expected {exp['n_defaults']:,}")

    frozen_target = exp.get("target_sha256")
    if frozen_target and not issues and target_sha256(y) != frozen_target:
        issues.append("target vector in stored order differs from the frozen table "
                      "(same counts, different rows or order)")

    frozen_raw = exp.get("raw_sha256")
    if frozen_raw and raw_sha256 is not None and raw_sha256 != frozen_raw:
        issues.append(f"raw file sha256 {raw_sha256[:12]} != pinned {frozen_raw[:12]} "
                      "(the source snapshot changed)")
    return issues


def write_table(df: pd.DataFrame, out: Path, record: Dict) -> None:
    """Write a harmonized table as parquet with PaMIR's provenance record.

    The file is written next to ``out`` and renamed into place, so an
    interrupted write never leaves a truncated cache entry behind.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[PARQUET_META_KEY] = json.dumps(record, sort_keys=True).encode("utf-8")
    tmp = Path(str(out) + ".tmp")
    pq.write_table(table.replace_schema_metadata(metadata), tmp)
    os.replace(tmp, out)


def read_record(path: Path) -> Dict:
    """PaMIR's provenance record from a cached parquet ({} for older caches)."""
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(PARQUET_META_KEY)
    return json.loads(raw.decode("utf-8")) if raw else {}
