"""Synthetic-data augmentation for PaMIR.

Mix real and synthetic training rows in stated proportions, across several
generators at once, and measure how good the synthetic rows are — under the
a fixed stratified 5-fold protocol, with the
generators fitted inside each fold so no held-out row ever reaches them.

    from sdv.single_table import GaussianCopulaSynthesizer

    from pamir import load_dataset
    from pamir.synthetic import CrossValidatedAugmentation, SDVAdapter, SyntheticMixer

    mixer = SyntheticMixer(
        generators={"copula": SDVAdapter(GaussianCopulaSynthesizer, name="copula")},
        synthetic_share=0.5,                   # 50% synthetic / 50% real
    )

    X, y, _ = load_dataset("south_german")
    result = CrossValidatedAugmentation(mixer).run(X, y)
    result.summary()            # baseline vs augmented OOF AUC, and the delta
    result.leakage_frame()      # the probes that say whether to believe the delta
    result.fidelity_headline()  # the fidelity numbers worth reading first
    result.fidelity_frame()     # all 66 fidelity columns, per fold and generator

``SDVAdapter`` takes any SDV synthesizer, so it needs nothing but the
``synthetic`` extra.  ``ZganAdapter`` and ``ZedgeAdapter`` mix in generators
that live in their own repos; ``weights`` then splits the synthetic part
between them.
"""

# Load xgboost before anything that pulls in torch.  `sdv` imports torch, and
# on macOS/arm64 a torch-first import order makes the next xgboost.train() die
# with a bare `Segmentation fault: 11` and no traceback — both libraries ship
# their own OpenMP runtime and only the first one loaded survives.  Importing
# xgboost here fixes the order for anyone who imports pamir.synthetic before
# sdv.  If you import sdv first in your own script, do the same there.
try:  # pragma: no cover - import-order guard, not logic
    import xgboost as _xgboost  # noqa: F401
except ImportError:  # xgboost is optional; nothing to order
    pass

from pamir.synthetic.adapters import (
    CallableAdapter,
    GeneratorAdapter,
    PoolAdapter,
    SDVAdapter,
    ZedgeAdapter,
    ZganAdapter,
    as_adapter,
)
from pamir.synthetic.cv import (
    CrossValidatedAugmentation,
    CVResult,
    FoldOutcome,
    run_fleet,
)
from pamir.synthetic.fidelity import (
    HEADLINE_FIDELITY_COLUMNS,
    FidelityReport,
    build_metadata,
    c2st_auc,
    fidelity_report,
    leakage_report,
)
from pamir.synthetic.mixer import MixResult, SyntheticMixer
from pamir.synthetic.mixing import MixturePlan, plan_mixture

__all__ = [
    "SyntheticMixer",
    "MixResult",
    "CrossValidatedAugmentation",
    "CVResult",
    "FoldOutcome",
    "run_fleet",
    "GeneratorAdapter",
    "ZganAdapter",
    "ZedgeAdapter",
    "SDVAdapter",
    "PoolAdapter",
    "CallableAdapter",
    "as_adapter",
    "fidelity_report",
    "leakage_report",
    "FidelityReport",
    "HEADLINE_FIDELITY_COLUMNS",
    "build_metadata",
    "c2st_auc",
    "plan_mixture",
    "MixturePlan",
]
