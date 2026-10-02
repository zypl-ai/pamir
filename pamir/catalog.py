"""Dataset catalog: metadata for every PaMIR dataset.

Two provenance fields answer different questions and are kept apart:
``source`` says where the data originate (an institution, a paper, a
competition), and ``fetched_from`` says what :func:`pamir.download` actually
retrieves.  ``fetched_from`` is computed from the ``download`` spec rather than
stored, so it cannot drift from what the downloader does.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"
_catalog = None


def _load_raw() -> List[Dict]:
    global _catalog
    if _catalog is None:
        _catalog = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    return _catalog


def load_catalog() -> pd.DataFrame:
    """Return a DataFrame with one row per dataset.

    Columns: id, name, rows, features, defaults, DR, source, fetched_from,
    license, geography, product, target_definition, and the recipe fields.
    """
    rows = [dict(e, fetched_from=fetched_from(e)) for e in _load_raw()]
    return pd.DataFrame(rows).set_index("id")


def fetched_from(entry: Dict) -> str:
    """Human-readable description of what the downloader fetches for an entry."""
    dl = entry.get("download", {})
    kind, loc = dl.get("kind"), dl.get("locator", "")
    if kind == "kaggle":
        version = dl.get("version")
        return f"Kaggle dataset {loc}" + (f" (version {version})" if version else "")
    if kind == "kaggle_competition":
        return f"Kaggle competition {loc}"
    if kind == "hf":
        revision = dl.get("revision")
        return f"Hugging Face dataset {loc}" + (f" (revision {revision[:12]})" if revision else "")
    if kind == "uci_zip":
        return f"UCI archive {loc}"
    if kind in ("github_raw", "github_zip"):
        return f"GitHub file {loc}"
    return f"{kind} {loc}"


def list_datasets() -> List[str]:
    """Return sorted list of dataset ids."""
    return sorted(e["id"] for e in _load_raw())


# Download sources that require credentials to fetch.
_CREDENTIAL_KINDS = {"kaggle", "kaggle_competition"}


def _needs_credentials(entry: Dict) -> bool:
    return entry.get("download", {}).get("kind") in _CREDENTIAL_KINDS


def open_datasets() -> List[str]:
    """Ids whose source needs **no credentials** to download.

    These come from UCI / GitHub / HuggingFace and work on a
    fresh install with no setup; the rest are on Kaggle, which kagglehub 1.0
    fetches anonymously and older kagglehub versions only with Kaggle
    credentials (see ``pamir.download``).
    """
    return sorted(e["id"] for e in _load_raw() if not _needs_credentials(e))


def dataset_info(dataset_id: str) -> Dict:
    """Return full metadata dict for one dataset.

    Raises KeyError if the id is not in the catalog.
    """
    for e in _load_raw():
        if e["id"] == dataset_id:
            d = dict(e)
            d["needs_credentials"] = _needs_credentials(e)
            d["fetched_from"] = fetched_from(e)
            return d
    raise KeyError(
        f"Unknown dataset '{dataset_id}'. "
        f"Available: {', '.join(list_datasets())}"
    )
