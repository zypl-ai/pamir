"""Record the source snapshot behind every cached PaMIR table (maintainer tool).

Run after ``pamir download`` has rebuilt every table under the current contract
(row, default and column counts match ``expected.json``).  For each dataset it
records

* the snapshot actually fetched: the Kaggle dataset version, the Hugging Face
  revision, or the GitHub commit of a raw/zip URL (UCI archives are static and
  are pinned by digest only);
* ``raw_sha256``: the digest of the raw file the harmonizer read;
* ``target_sha256``: the digest of the cached table's target vector in stored
  order;

and writes them to a JSON file.  ``--apply`` writes the pins into
``pamir/data/catalog.json`` (download spec) and ``pamir/data/expected.json``
(contract digests).  Nothing is applied for a dataset whose cached table fails
its contract, or whose pinned GitHub URL serves a different file.

    python tools/pin_sources.py --out pins.json
    python tools/pin_sources.py --from pins.json --apply
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from pamir.catalog import dataset_info, list_datasets  # noqa: E402
from pamir.contract import check_table, file_sha256, target_sha256  # noqa: E402
from pamir.download import _fetch_raw, cache_dir  # noqa: E402

_GITHUB_RAW = re.compile(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)")


def _github_json(url: str):
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req, timeout=60) as r:  # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def _github_pin(locator: str):
    """(commit sha, pinned URL) for a raw.githubusercontent.com URL on a branch."""
    m = _GITHUB_RAW.match(locator)
    if not m:
        return None, None
    owner, repo, ref, path = m.groups()
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref, locator
    commits = _github_json(
        f"https://api.github.com/repos/{owner}/{repo}/commits"
        f"?sha={ref}&path={quote(unquote(path))}&per_page=1")
    sha = commits[0]["sha"]
    return sha, f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}"


def discover(ids):
    pins = {}
    for ds in ids:
        spec = dataset_info(ds)
        dl = spec["download"]
        pq = cache_dir() / f"{ds}.parquet"
        entry = {"kind": dl["kind"]}
        if not pq.exists():
            entry["error"] = "not cached"
            pins[ds] = entry
            continue
        df = pd.read_parquet(pq)
        entry["contract_issues"] = check_table(ds, df)
        entry["target_sha256"] = target_sha256(df["__target__"].to_numpy())
        with tempfile.TemporaryDirectory() as tmp:
            raw = _fetch_raw(spec, Path(tmp))
            entry["raw_sha256"] = file_sha256(raw)
            raw_str = str(raw)
        if dl["kind"] == "kaggle":
            m = re.search(r"/versions/(\d+)(/|$)", raw_str)
            entry["version"] = int(m.group(1)) if m else None
        elif dl["kind"] == "hf":
            m = re.search(r"/snapshots/([0-9a-f]{40})/", raw_str)
            entry["revision"] = m.group(1) if m else None
        elif dl["kind"] in ("github_raw", "github_zip"):
            sha, pinned = _github_pin(dl["locator"])
            entry["commit"], entry["pinned_locator"] = sha, pinned
            if pinned and pinned != dl["locator"]:
                with tempfile.TemporaryDirectory() as tmp:
                    raw2 = _fetch_raw({**spec, "download": {**dl, "locator": pinned}},
                                      Path(tmp))
                    entry["pinned_matches"] = file_sha256(raw2) == entry["raw_sha256"]
        pins[ds] = entry
        print(ds, {k: v for k, v in entry.items() if k not in ("target_sha256", "raw_sha256")},
              flush=True)
    return pins


def apply(pins):
    cat_path = ROOT / "pamir" / "data" / "catalog.json"
    exp_path = ROOT / "pamir" / "data" / "expected.json"
    catalog = json.loads(cat_path.read_text(encoding="utf-8"))
    expected = json.loads(exp_path.read_text(encoding="utf-8"))
    applied, skipped = [], []
    for e in catalog:
        p = pins.get(e["id"])
        if not p or p.get("error") or p.get("contract_issues"):
            skipped.append(e["id"])
            continue
        dl = e["download"]
        if p["kind"] == "kaggle":
            if not p.get("version"):
                skipped.append(e["id"])
                continue
            dl["version"] = p["version"]
        elif p["kind"] == "hf":
            if not p.get("revision"):
                skipped.append(e["id"])
                continue
            dl["revision"] = p["revision"]
        elif p["kind"] in ("github_raw", "github_zip"):
            if not p.get("pinned_locator") or p.get("pinned_matches") is False:
                skipped.append(e["id"])
                continue
            dl["locator"] = p["pinned_locator"]
        expected[e["id"]]["raw_sha256"] = p["raw_sha256"]
        expected[e["id"]]["target_sha256"] = p["target_sha256"]
        applied.append(e["id"])
    # Keep each file's existing formatting, so the diff shows only the pins.
    cat_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    exp_path.write_text(json.dumps(expected, indent=2, ensure_ascii=True, sort_keys=True),
                        encoding="utf-8")
    return applied, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="pins.json")
    ap.add_argument("--from", dest="src", help="apply pins from this file instead of discovering")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("datasets", nargs="*")
    args = ap.parse_args(argv)
    if args.src:
        pins = json.loads(Path(args.src).read_text())
    else:
        pins = discover(args.datasets or list_datasets())
        Path(args.out).write_text(json.dumps(pins, indent=2) + "\n")
    if args.apply:
        applied, skipped = apply(pins)
        print("applied:", applied)
        print("skipped:", skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
