"""PaMIR — Public Arrival-ordered Measurement for Inference in Risk.

An open benchmark of public credit-default datasets, rebuilt from their
original sources under a leakage audit, with a label-delayed streaming protocol
(each row scored on arrival, by a model trained only on outcomes that had
matured by then) and a conventional i.i.d. protocol.

Quick start::

    from pamir import load_catalog, load_dataset, evaluate, fleet_summary, gbdt_fit

    catalog = load_catalog()                  # one row per dataset
    X, y, meta = load_dataset("gmsc")         # features, binary target, metadata
    results = evaluate(gbdt_fit)              # streaming AUC across the fleet
    fleet_summary(results)                    # headline numbers, with coverage

Report ``mode``, ``lag``, ``k_refit``, ``max_n`` and ``max_lag_frac`` alongside
any number produced here: they change the score, so runs are only comparable
when they match.
"""

from pamir._version import __version__

from pamir.catalog import load_catalog, list_datasets, dataset_info, open_datasets
from pamir.contract import ContractError
from pamir.loader import load_dataset
from pamir.download import download, download_open, cache_dir
from pamir.evaluate import evaluate, evaluate_one, from_predict_fn
from pamir.evaluate_iid import evaluate_iid, evaluate_iid_one
from pamir.baselines import (encode_features, gbdt_baseline, gbdt_baseline_v03,
                             gbdt_fit, logistic_baseline, logistic_baseline_v03,
                             logistic_fit)
from pamir.summary import fleet_summary

__all__ = [
    "ContractError",
    "load_catalog",
    "list_datasets",
    "dataset_info",
    "open_datasets",
    "load_dataset",
    "download",
    "download_open",
    "cache_dir",
    "evaluate",
    "evaluate_one",
    "from_predict_fn",
    "evaluate_iid",
    "evaluate_iid_one",
    "fleet_summary",
    "encode_features",
    "logistic_baseline",
    "gbdt_baseline",
    "logistic_fit",
    "gbdt_fit",
    "logistic_baseline_v03",
    "gbdt_baseline_v03",
]
