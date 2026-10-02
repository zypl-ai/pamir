"""Command-line interface for PaMIR.

    pamir list [--open]          list datasets (optionally only credential-free ones)
    pamir info <id>              show a dataset's metadata / recipe
    pamir download <id> ...      fetch + harmonize datasets into the cache
    pamir download --open        fetch every credential-free dataset
    pamir download ... --no-strict  cache tables that fail their data contract (flagged)
    pamir cache                  print the cache directory
"""

import argparse
import sys


def _cmd_list(args):
    from pamir.catalog import list_datasets, open_datasets, dataset_info
    ids = open_datasets() if args.open else list_datasets()
    for ds in ids:
        info = dataset_info(ds)
        creds = "" if not info.get("needs_credentials") else "  [needs credentials]"
        print(f"{ds:14s} {info['rows']:>8,} rows  DR {info['DR']:.1%}  "
              f"{info['geography']}{creds}")
    print(f"\n{len(ids)} datasets" + (" (credential-free)" if args.open else ""))


def _cmd_info(args):
    from pamir.catalog import dataset_info
    try:
        info = dataset_info(args.dataset)
    except KeyError as e:
        print(e, file=sys.stderr)
        return 1
    for k in ("name", "geography", "product", "rows", "features", "defaults",
              "DR", "target_definition", "license", "attribution", "citation",
              "source", "source_url", "fetched_from", "needs_credentials"):
        if k in info:
            print(f"{k:20s} {info[k]}")
    return 0


def _cmd_download(args):
    from pamir.download import download, download_open
    if args.open:
        paths = download_open(force=args.force, strict=not args.no_strict)
        print(f"\ncached {len(paths)} credential-free datasets")
        return 0
    if not args.datasets:
        print("give dataset ids, or --open", file=sys.stderr)
        return 1
    for ds in args.datasets:
        download(ds, force=args.force, strict=not args.no_strict)
    return 0


def _cmd_cache(args):
    from pamir.download import cache_dir
    print(cache_dir())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pamir", description="PaMIR credit-risk benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list datasets")
    pl.add_argument("--open", action="store_true", help="only credential-free datasets")
    pl.set_defaults(func=_cmd_list)

    pi = sub.add_parser("info", help="show a dataset's metadata")
    pi.add_argument("dataset")
    pi.set_defaults(func=_cmd_info)

    pd_ = sub.add_parser("download", help="fetch + harmonize into the cache")
    pd_.add_argument("datasets", nargs="*", help="dataset ids")
    pd_.add_argument("--open", action="store_true", help="all credential-free datasets")
    pd_.add_argument("--force", action="store_true", help="re-fetch even if cached")
    pd_.add_argument("--no-strict", action="store_true",
                     help="cache a table even if it fails its data contract (results are flagged)")
    pd_.set_defaults(func=_cmd_download)

    pc = sub.add_parser("cache", help="print the cache directory")
    pc.set_defaults(func=_cmd_cache)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
