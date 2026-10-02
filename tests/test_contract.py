"""The data contract (expected.json) and release consistency — no data needed."""

import importlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pamir
from pamir import contract

dl_mod = importlib.import_module("pamir.download")
ROOT = Path(pamir.__file__).resolve().parent.parent


def _frame(n=10, n_def=3):
    y = np.zeros(n, dtype=int)
    y[:n_def] = 1
    return pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n) % 3,
                         "__target__": y})


@pytest.fixture
def ctl_contract(monkeypatch):
    good = _frame()
    monkeypatch.setitem(contract.EXPECTED, "south_german", {
        "columns": {"a": "num", "b": "num"}, "n_features": 2, "n_rows": 10,
        "n_defaults": 3, "DR": 0.3,
        "target_sha256": contract.target_sha256(good["__target__"]),
        "raw_sha256": "f" * 64})
    return good


def test_check_table_reports_each_kind_of_mismatch(ctl_contract):
    good = ctl_contract
    assert contract.check_table("south_german", good) == []
    msgs = contract.check_table("south_german", good.iloc[1:])
    assert any("rows" in m for m in msgs) and any("defaults" in m for m in msgs)
    assert any("features" in m for m in contract.check_table(
        "south_german", good.rename(columns={"a": "a2"})))
    assert any("raw file" in m for m in contract.check_table(
        "south_german", good, raw_sha256="0" * 64))
    assert contract.check_table("no_such_dataset", good)


def test_strict_download_refuses_and_caches_nothing(ctl_contract, tmp_path, monkeypatch):
    monkeypatch.setenv("PAMIR_CACHE", str(tmp_path))
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"not the pinned snapshot")
    monkeypatch.setattr(dl_mod, "_fetch_raw", lambda spec, d: raw)
    monkeypatch.setattr(dl_mod, "harmonize", lambda path, spec: _frame(n=9))
    with pytest.raises(pamir.ContractError) as exc:
        pamir.download("south_german", force=True, quiet=True)
    assert "rows 9" in str(exc.value)
    assert not (tmp_path / "south_german.parquet").exists()

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = pamir.download("south_german", force=True, quiet=True, strict=False)
    assert out.exists() and any("contract mismatch" in str(w.message) for w in rec)
    record = contract.read_record(out)
    assert record["contract_ok"] is False and record["raw_sha256"] == contract.file_sha256(raw)


def test_load_rechecks_a_cached_table(ctl_contract, tmp_path, monkeypatch):
    monkeypatch.setenv("PAMIR_CACHE", str(tmp_path))
    out = tmp_path / "south_german.parquet"
    contract.write_table(ctl_contract.iloc[::-1].reset_index(drop=True), out, {})
    with pytest.raises(pamir.ContractError, match="stored order"):
        pamir.load_dataset("south_german", auto_download=False)
    X, y, meta = pamir.load_dataset("south_german", auto_download=False, strict=False)
    assert meta["contract_ok"] is False and meta["snapshot_verified"] is False

    contract.write_table(ctl_contract, out, {"raw_sha256": "f" * 64})
    X, y, meta = pamir.load_dataset("south_german", auto_download=False)
    assert meta["contract_ok"] is True and meta["snapshot_verified"] is True


def test_record_roundtrip(tmp_path):
    out = tmp_path / "t.parquet"
    contract.write_table(_frame(), out, {"dataset": "x", "contract_ok": True})
    assert contract.read_record(out) == {"dataset": "x", "contract_ok": True}
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "old.parquet")
    assert contract.read_record(tmp_path / "old.parquet") == {}


def test_kaggle_handle_is_pinned_when_a_version_is_recorded():
    assert dl_mod.kaggle_handle({"locator": "o/d"}) == "o/d"
    assert dl_mod.kaggle_handle({"locator": "o/d", "version": 387}) == "o/d/versions/387"


def test_fetched_from_describes_the_download_kind():
    prefix = {"kaggle": "Kaggle dataset", "kaggle_competition": "Kaggle competition",
              "hf": "Hugging Face", "uci_zip": "UCI", "github_raw": "GitHub",
              "github_zip": "GitHub"}
    for ds in pamir.list_datasets():
        info = pamir.dataset_info(ds)
        assert info["fetched_from"].startswith(prefix[info["download"]["kind"]]), ds


def test_notes_counts_match_the_drop_lists():
    """A count in `notes` must be the length of the list it describes (D6)."""
    raw = json.loads((ROOT / "pamir" / "data" / "catalog.json").read_text())
    for e in raw:
        m = re.match(r"(\d+) post-outcome columns removed", e.get("notes") or "")
        if m:
            assert int(m.group(1)) == len(e["harmonize"]["day_zero_drop"]), e["id"]


def test_version_is_consistent_across_release_files():
    v = pamir.__version__
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert re.search(rf'^version\s*=\s*"{re.escape(v)}"', pyproject, re.M), "pyproject.toml"
    cff = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf'^version:\s*"?{re.escape(v)}"?\s*$', cff, re.M), "CITATION.cff"
    conf = (ROOT / "docs" / "conf.py").read_text()
    assert re.search(rf'^release\s*=\s*"{re.escape(v)}"', conf, re.M), "docs/conf.py"
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"## {v}" in changelog, "CHANGELOG.md has no entry for this version"
