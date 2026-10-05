"""Fetch a dataset from its original source and harmonize it locally.

PaMIR ships code and a recipe, not data.  ``download(id)`` fetches the raw
file from the snapshot of the dataset's source that the catalog pins, into a
local cache; harmonizes it (see :mod:`pamir.harmonize`); checks the result
against the data contract in ``expected.json`` (see :mod:`pamir.contract`);
and only then writes the parquet.  A table that does not match the contract
raises :class:`pamir.contract.ContractError` and is not cached, unless the
caller passes ``strict=False``.  ``load_dataset`` calls this automatically on
first use.

Sources and their requirements:

- ``kaggle``     — needs ``kagglehub``; kagglehub 1.0 fetches public datasets
  anonymously, older versions need Kaggle credentials (``KAGGLE_USERNAME`` +
  ``KAGGLE_KEY``, or ``~/.kaggle/kaggle.json``).  The catalog pins a dataset
  version.
- ``uci_zip``    — public download, no credentials.
- ``github_raw`` / ``github_zip`` — public download, no credentials.
- ``hf``         — needs ``huggingface_hub``; the catalog pins a revision where
  one is recorded.

Install the optional fetch dependencies with ``pip install pamir[data]``.
"""

import io
import os
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen

import pandas as pd

from pamir.catalog import dataset_info, fetched_from, open_datasets
from pamir.contract import (EXPECTED as _EXPECTED, ContractError, check_table,
                            file_sha256, write_table)
from pamir.harmonize import harmonize


def cache_dir() -> Path:
    """Directory where fetched raw files and harmonized parquet are cached."""
    env = os.environ.get("PAMIR_CACHE")
    if env:
        base = Path(env)
    else:
        try:
            from platformdirs import user_cache_dir
            base = Path(user_cache_dir("pamir"))
        except Exception:
            base = Path.home() / ".cache" / "pamir"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _http_get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; pamir)"})
    with urlopen(req, timeout=120) as r:  # noqa: S310 (trusted, https)
        return r.read()


def _extract_from_zip(blob: bytes, fname: str, _depth: int = 0):
    """Return the bytes of ``fname`` from a zip, recursing one level into
    nested zips (the PAKDD archive nests the modeling data inside another zip)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        member = next((m for m in z.namelist() if m.endswith(fname)), None)
        if member is not None:
            return z.read(member)
        if _depth == 0:
            for m in z.namelist():
                if m.lower().endswith(".zip"):
                    got = _extract_from_zip(z.read(m), fname, _depth + 1)
                    if got is not None:
                        return got
    return None


def _fetch_raw(spec: Dict, raw_dir: Path) -> Path:
    """Fetch the raw source file into ``raw_dir`` and return its path."""
    dl = spec["download"]
    kind, loc, fname = dl["kind"], dl["locator"], dl["file"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / fname

    if kind == "github_raw":
        target.write_bytes(_http_get(loc))
        return target

    if kind in ("uci_zip", "github_zip"):
        data = _extract_from_zip(_http_get(loc), fname)
        if data is None:
            raise FileNotFoundError(f"{fname} not found in zip for '{spec['id']}'")
        target.write_bytes(data)
        return target

    if kind == "kaggle_competition":
        try:
            import kagglehub
        except ImportError as e:
            raise ImportError(
                "kaggle competition source needs `kagglehub` — pip install pamir[data]"
            ) from e
        # Single-file download (path=) — avoids pulling the whole ~700MB
        # competition, and kagglehub gives a clear message if the rules were not
        # accepted (a bare 401 otherwise). Rules must be accepted once at
        # https://www.kaggle.com/competitions/<comp>/rules
        try:
            got = Path(kagglehub.competition_download(loc, path=fname))
        except TypeError:  # older kagglehub without path=
            got = Path(kagglehub.competition_download(loc))
        if got.is_file():
            return got
        hit = next((p for p in got.rglob("*") if p.name == fname), None)
        if hit is None:
            raise FileNotFoundError(
                f"{fname} not found in competition '{loc}' (under {got})")
        return hit

    if kind == "kaggle":
        try:
            import kagglehub
        except ImportError as e:
            raise ImportError(
                "kaggle source needs `kagglehub` — pip install pamir[data]"
            ) from e
        handle = kaggle_handle(dl)
        path = Path(kagglehub.dataset_download(handle))
        hit = next((p for p in path.rglob("*") if p.name == fname), None)
        if hit is None:
            raise FileNotFoundError(
                f"{fname} not found in Kaggle dataset '{handle}' for '{spec['id']}'")
        return hit

    if kind == "hf":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "hf source needs `huggingface_hub` — pip install pamir[data]"
            ) from e
        return Path(hf_hub_download(repo_id=loc, filename=fname, repo_type="dataset",
                                    revision=dl.get("revision")))

    raise ValueError(f"unknown download.kind {kind!r} for '{spec['id']}'")


def kaggle_handle(dl: Dict) -> str:
    """The kagglehub handle for a download spec, pinned when a version is set."""
    version = dl.get("version")
    return f"{dl['locator']}/versions/{int(version)}" if version else dl["locator"]


def _validate(dataset_id: str, df: pd.DataFrame,
              raw_sha256: Optional[str] = None) -> List[str]:
    """Compare a harmonized table with expected.json; return the mismatches.

    Kept for backward compatibility; see :func:`pamir.contract.check_table`.
    """
    return check_table(dataset_id, df, raw_sha256=raw_sha256)


def download(dataset_id: str, force: bool = False, quiet: bool = False,
             strict: bool = True) -> Path:
    """Fetch and harmonize one dataset; return the path to the cached parquet.

    Parameters
    ----------
    dataset_id : str
        A PaMIR dataset id (see :func:`pamir.list_datasets`).
    force : bool
        Re-fetch and re-harmonize even if a cached parquet exists.
    quiet : bool
        Suppress progress messages.  Contract mismatches are never silent: with
        ``strict=False`` they are emitted as warnings regardless of ``quiet``.
    strict : bool
        If True (default), a rebuilt table that does not match its contract in
        ``expected.json`` raises :class:`pamir.contract.ContractError` and
        nothing is cached.  If False, the table is cached with its provenance
        record marked ``contract_ok: false``; evaluations on it carry
        ``contract_ok = False`` and get no fleet mean.

    Returns
    -------
    Path to ``<cache>/<dataset_id>.parquet``.
    """
    from pamir._version import __version__

    spec = dataset_info(dataset_id)
    out = cache_dir() / f"{dataset_id}.parquet"
    if out.exists() and not force:
        return out

    def say(msg):
        if not quiet:
            print(f"[pamir] {dataset_id}: {msg}", flush=True)

    say(f"fetching from {fetched_from(spec)} …")
    raw = _fetch_raw(spec, cache_dir() / "raw" / dataset_id)
    raw_sha = file_sha256(raw)
    say("harmonizing …")
    df = harmonize(raw, spec)
    issues = check_table(dataset_id, df, raw_sha256=raw_sha)
    if issues and strict:
        raise ContractError(
            dataset_id, issues,
            hint=("Nothing was cached. The pinned source snapshot may be "
                  "unavailable or the harmonizer may behave differently in this "
                  "environment; pass strict=False to cache the table anyway "
                  "(its results are then flagged contract_ok=False)."))
    for issue in issues:
        warnings.warn(f"[pamir] {dataset_id}: contract mismatch: {issue}",
                      UserWarning, stacklevel=2)
    write_table(df, out, {
        "dataset": dataset_id,
        "pamir_version": __version__,
        "contract_ok": not issues,
        "issues": issues,
        "raw_sha256": raw_sha,
        "fetched_from": fetched_from(spec),
    })
    say(f"cached {len(df):,} rows, {df.shape[1] - 1} features -> {out}")
    return out


def download_open(force: bool = False, quiet: bool = False,
                  strict: bool = True) -> list:
    """Download every dataset whose source needs no credentials.

    A zero-setup starter set (UCI / GitHub / HuggingFace) — no
    Kaggle account required. Returns the list of cached parquet paths.
    """
    paths = []
    for ds in open_datasets():
        paths.append(download(ds, force=force, quiet=quiet, strict=strict))
    return paths
