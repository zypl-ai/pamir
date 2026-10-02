API Reference
=============

Catalog
-------

.. autofunction:: pamir.load_catalog

.. autofunction:: pamir.list_datasets

.. autofunction:: pamir.dataset_info

Data loading
------------

.. autofunction:: pamir.load_dataset

Downloading data
----------------

PaMIR ships no data. These fetch each dataset from its original source and
harmonize it locally (see :doc:`datasets`).

.. autofunction:: pamir.download

.. autofunction:: pamir.cache_dir

Data contract
-------------

.. autoclass:: pamir.ContractError

.. autofunction:: pamir.contract.check_table

Baselines
---------

.. autofunction:: pamir.logistic_fit

.. autofunction:: pamir.gbdt_fit

.. autofunction:: pamir.logistic_baseline

.. autofunction:: pamir.gbdt_baseline

.. autofunction:: pamir.baselines.fit_encoder

.. autofunction:: pamir.baselines.apply_encoder

.. autofunction:: pamir.encode_features

.. autofunction:: pamir.logistic_baseline_v03

.. autofunction:: pamir.gbdt_baseline_v03

i.i.d. evaluation
-----------------

.. autofunction:: pamir.evaluate_iid

.. autofunction:: pamir.evaluate_iid_one

Streaming evaluation
--------------------

.. autofunction:: pamir.evaluate

.. autofunction:: pamir.evaluate_one

.. autofunction:: pamir.from_predict_fn

Results
-------

.. autofunction:: pamir.fleet_summary

Synthetic augmentation
----------------------

.. autoclass:: pamir.synthetic.SyntheticMixer
   :members: fit, build, clone

.. autoclass:: pamir.synthetic.CrossValidatedAugmentation
   :members: run, run_dataset

.. autoclass:: pamir.synthetic.CVResult
   :members: summary, fold_frame, fidelity_frame, fidelity_headline, leakage_frame

.. autofunction:: pamir.synthetic.run_fleet

.. autofunction:: pamir.synthetic.plan_mixture

Generator adapters
------------------

.. autoclass:: pamir.synthetic.GeneratorAdapter
   :members: fit, sample, clone, native_fidelity, fixed_pool, n_available

.. autoclass:: pamir.synthetic.ZganAdapter

.. autoclass:: pamir.synthetic.ZedgeAdapter

.. autoclass:: pamir.synthetic.SDVAdapter

.. autoclass:: pamir.synthetic.PoolAdapter

.. autoclass:: pamir.synthetic.CallableAdapter

.. autofunction:: pamir.synthetic.as_adapter

Fidelity
--------

.. autofunction:: pamir.synthetic.fidelity_report

.. autofunction:: pamir.synthetic.leakage_report

.. autoclass:: pamir.synthetic.FidelityReport
   :members: summary, column_matrix, to_dict

.. autofunction:: pamir.synthetic.c2st_auc

.. autofunction:: pamir.synthetic.build_metadata
