"""One fit/sample contract over different synthetic-data engines.

The mixer never learns which engine produced a row.  All it needs is:

    adapter.fit(train_frame)      # rows of ONE fold's train split, with target
    adapter.sample(n, seed)       # n rows carrying the training schema
    adapter.native_fidelity(...)  # metrics the engine ships itself, if any

``as_adapter`` accepts what a caller naturally has to hand — an adapter, an SDV
synthesizer class or instance (zGAN's ``zyplGANSynthesizer`` is one), a zEDGE
artifact directory, an already-generated pool, or any object with
``fit``/``sample`` — and wraps it.

Leakage contract
----------------
``fit`` is the only entry point that receives real rows, and the cross-validated
runner calls it with the training split alone.  No adapter reads a file the
caller did not name, and no adapter holds state between folds: the runner clones
a fresh adapter per fold via :meth:`GeneratorAdapter.clone`.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from pamir.synthetic.fidelity import build_metadata


def _expand(path: Any) -> Path:
    """Resolve ``~`` in a user-supplied path.

    Repository and artifact paths are habitually written as ``~/path/to/repo``,
    and neither ``sys.path`` nor ``Path.exists`` expands a tilde, so without
    this the documented invocation fails with an import error.
    """
    return Path(path).expanduser()


@contextmanager
def _seeded(seed: Optional[int]):
    """Seed the global RNGs for one call, then put them back as they were.

    Several engines draw from ``numpy.random``'s legacy global state, so
    reproducible sampling means seeding it globally.  Leaving it seeded would
    silently reseed the caller's whole session, so the previous state is
    restored on the way out.
    """
    if seed is None:
        yield
        return
    np_state = np.random.get_state()
    try:
        import torch

        torch_state = torch.random.get_rng_state()
    except Exception:  # noqa: BLE001 - torch is optional
        torch, torch_state = None, None
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)


class GeneratorAdapter(ABC):
    """Uniform interface every generator is wrapped in."""

    def __init__(self, name: str):
        self.name = name
        self._fitted = False
        self._schema: Optional[pd.Index] = None
        self._target: Optional[str] = None

    # -- contract ---------------------------------------------------------- #
    @abstractmethod
    def _fit(self, data: pd.DataFrame, target: Optional[str], discrete_columns: Sequence[str]) -> None:
        ...

    @abstractmethod
    def _sample(self, n: int, seed: Optional[int]) -> pd.DataFrame:
        ...

    def fit(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        discrete_columns: Optional[Sequence[str]] = None,
    ) -> "GeneratorAdapter":
        """Train on real rows — the fold's training split, target column included."""
        if data.empty:
            raise ValueError(f"[{self.name}] cannot fit on an empty frame.")
        self._schema = data.columns
        self._target = target
        if discrete_columns is None:
            md = build_metadata(data, target=target)
            discrete_columns = [c for c, s in md["columns"].items() if s["sdtype"] == "categorical"]
        self._fit(data, target, list(discrete_columns))
        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Draw ``n`` synthetic rows with the training schema."""
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] sample() called before fit().")
        if n <= 0:
            return pd.DataFrame(columns=self._schema)
        out = self._sample(n, seed)
        return self._conform(out)

    def native_fidelity(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        """Metrics shipped by the engine's own library.  Empty when it ships none."""
        return {}

    def clone(self) -> "GeneratorAdapter":
        """An unfitted copy, so each CV fold trains from scratch."""
        fresh = copy.deepcopy(self)
        fresh._fitted = False
        fresh._schema = None
        fresh._reset()
        return fresh

    def _reset(self) -> None:
        """Drop fitted state; overridden by adapters holding a model object."""

    # -- helpers ----------------------------------------------------------- #
    def _conform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Force the generated frame onto the training schema."""
        if self._schema is None:
            return frame
        missing = [c for c in self._schema if c not in frame.columns]
        if missing:
            raise ValueError(
                f"[{self.name}] generated frame is missing columns: {missing[:8]}"
            )
        return frame.loc[:, list(self._schema)].reset_index(drop=True)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def fixed_pool(self) -> bool:
        """True when the engine can only hand back rows it already holds.

        The mixer treats such an engine differently when a class balance is
        requested: there is no point asking a finite pool for oversampled slack
        it cannot make, so the whole pool becomes the candidate set instead.
        """
        return False

    @property
    def n_available(self) -> Optional[int]:
        """Rows the engine can supply, when that number is bounded.  None if not."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, fitted={self._fitted})"


class CallableAdapter(GeneratorAdapter):
    """Wrap any object exposing ``fit(frame)`` and ``sample(n)``.

    This is the escape hatch for a model class that already follows the usual
    synthesizer convention but is neither zGAN nor zEDGE.

    Each dictionary has exactly one destination, whatever is passed and however
    many times the adapter is fitted:

    ``init_kwargs``
        The model's constructor.  Only meaningful when ``model`` is a class;
        given alongside an already-built instance they cannot be applied, so
        they are refused rather than ignored.
    ``fit_kwargs``
        ``model.fit(frame, **fit_kwargs)``.
    ``sample_kwargs``
        ``model.sample(n, **sample_kwargs)``.
    """

    def __init__(self, model: Any, name: str = "generator",
                 init_kwargs: Optional[Dict] = None, fit_kwargs: Optional[Dict] = None,
                 sample_kwargs: Optional[Dict] = None, pass_discrete: bool = False):
        super().__init__(name)
        self.model = model
        self.init_kwargs = dict(init_kwargs or {})
        self.fit_kwargs = dict(fit_kwargs or {})
        self.sample_kwargs = dict(sample_kwargs or {})
        self.pass_discrete = pass_discrete
        self._prototype = model
        self._own_instance = False   # True once _fit built the model itself

    def _reset(self) -> None:
        self.model = copy.deepcopy(self._prototype)
        self._own_instance = False

    def _fit(self, data, target, discrete_columns) -> None:
        if self._own_instance:
            # Built by an earlier fit; go back to the prototype so a refit starts
            # clean and the guard below only sees instances the caller supplied.
            self.model = copy.deepcopy(self._prototype)
            self._own_instance = False
        if isinstance(self.model, type):
            self.model = self.model(**self.init_kwargs)
            self._own_instance = True
        elif self.init_kwargs:
            raise ValueError(
                f"[{self.name}] was given an already-built "
                f"{type(self.model).__name__} together with init_kwargs "
                f"{sorted(self.init_kwargs)}, which can no longer be applied — the "
                "model is constructed already. Pass the class instead of an "
                "instance, or configure the instance before handing it over."
            )
        kwargs = dict(self.fit_kwargs)
        if self.pass_discrete:
            kwargs.setdefault("discrete_columns", discrete_columns)
        self.model.fit(data, **kwargs)

    def _sample(self, n, seed) -> pd.DataFrame:
        with _seeded(seed):
            return pd.DataFrame(self.model.sample(n, **self.sample_kwargs))


class SDVAdapter(GeneratorAdapter):
    """Any SDV ``BaseSingleTableSynthesizer`` — which is what zGAN subclasses.

    Metadata is detected from the fold's training frame, so nothing about the
    held-out rows reaches the generator, not even a column's value range.
    """

    def __init__(self, synthesizer: Any, name: str = "sdv", max_categories: int = 20, **params):
        super().__init__(name)
        self.synthesizer = synthesizer
        self.params = params
        self.max_categories = max_categories
        self._prototype = synthesizer
        self._metadata_obj = None
        self._own_instance = False   # True once _fit built the engine itself

    def _reset(self) -> None:
        # deepcopy, not assignment: fitting an SDV instance mutates it in place,
        # so a bare prototype reference would carry a fitted model into the
        # next fold.
        self.synthesizer = copy.deepcopy(self._prototype)
        self._own_instance = False

    def _build_metadata(self, data: pd.DataFrame, target: Optional[str]):
        from sdv.metadata import SingleTableMetadata

        md = SingleTableMetadata()
        md.detect_from_dataframe(data)
        inferred = build_metadata(data, target=target, max_categories=self.max_categories)
        for col, spec in inferred["columns"].items():
            try:
                md.update_column(column_name=col, sdtype=spec["sdtype"])
            except Exception:  # noqa: BLE001 - keep SDV's own detection on conflict
                pass
        return md

    def _fit(self, data, target, discrete_columns) -> None:
        self._metadata_obj = self._build_metadata(data, target)
        if self._own_instance:
            # A previous fit built this engine from the prototype. Go back to the
            # prototype so a refit starts clean, and so the guard below only ever
            # sees an instance the caller actually supplied.
            self.synthesizer = copy.deepcopy(self._prototype)
            self._own_instance = False
        if isinstance(self.synthesizer, type):
            params = dict(self.params)
            injected = ()
            if target is not None and "target" not in params:
                # Ours to withdraw if the engine will not take it, unlike
                # anything the caller asked for.
                params["target"] = target
                injected = ("target",)
            self.synthesizer = self._instantiate(
                self.synthesizer, self._metadata_obj, params, injected)
            self._own_instance = True
        elif self.params:
            raise ValueError(
                f"[{self.name}] was given an already-built {type(self.synthesizer).__name__} "
                f"together with {sorted(self.params)}, which can no longer be applied — "
                "the engine is constructed already. Pass the class instead of an "
                "instance, or configure the instance before handing it over."
            )
        self.synthesizer.fit(data)

    @staticmethod
    def _instantiate(cls, metadata, params, injected: Sequence[str] = ()):
        """Build the engine, refusing to quietly drop the caller's settings.

        Only keys in ``injected`` — extras this adapter added itself, such as
        the ``target`` that credit-specific engines like zGAN take — may be
        withdrawn when the engine does not accept them.  Everything the caller
        asked for is either honoured or reported: silently falling back to a
        default-configured engine because of one unsupported keyword would run
        a whole benchmark on settings nobody chose, and every number downstream
        would describe a generator the caller never configured.
        """
        attempts = [dict(params)]
        trimmed = {k: v for k, v in params.items() if k not in injected}
        if trimmed != params:
            attempts.append(trimmed)
        error = None
        for kwargs in attempts:
            try:
                return cls(metadata=metadata, **kwargs)
            except TypeError as exc:
                error = exc
        asked = sorted(k for k in params if k not in injected)
        raise TypeError(
            f"{cls.__name__} rejected the parameters it was given "
            f"({asked or 'none'}): {error}. They are not dropped silently, because "
            "a run on defaults nobody chose is worse than a failure here. Check the "
            "engine's signature, or hand over an already-configured instance."
        ) from error

    def _sample(self, n, seed) -> pd.DataFrame:
        with _seeded(seed):
            return self.synthesizer.sample(num_rows=n)


class ZganAdapter(SDVAdapter):
    """zGAN (``zyplGANSynthesizer``) from the zypl zgan repository.

    ``repo_path`` is the checkout root; the class is imported from
    ``app.zgan.zgan_lib`` unless ``module`` says otherwise.  zGAN's own fidelity
    helpers in ``app.utils.zgan_utils`` are reported through
    :meth:`native_fidelity`.
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        name: str = "zgan",
        module: str = "app.zgan.zgan_lib",
        utils_module: str = "app.utils.zgan_utils",
        synthesizer: Any = None,
        **params,
    ):
        self.repo_path = repo_path
        self.module = module
        self.utils_module = utils_module
        super().__init__(synthesizer or self._import_class(), name=name, **params)

    def _ensure_path(self) -> None:
        if self.repo_path:
            base = _expand(self.repo_path)
            if not base.exists():
                raise FileNotFoundError(
                    f"[{self.name}] repo_path {base} does not exist. It is not added "
                    "to sys.path, so the import would fail with a less obvious error."
                )
            root = str(base)
            if root not in sys.path:
                sys.path.insert(0, root)

    def _import_class(self):
        self._ensure_path()
        import importlib

        mod = importlib.import_module(self.module)
        return getattr(mod, "zyplGANSynthesizer")

    def _import_utils(self):
        self._ensure_path()
        import importlib

        return importlib.import_module(self.utils_module)

    def native_fidelity(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        """zGAN's shipped KS / TV / correlation helpers, on its own metadata object."""
        out: Dict[str, float] = {}
        try:
            utils = self._import_utils()
            md = self._metadata_obj or self._build_metadata(real, self._target)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}

        helpers = {
            "ks_complement": "evaluate_ks_complement",
            "tv_complement": "evaluate_tv_complement",
            "tv_complement_pairs": "evaluate_tv_complement_pairs",
            "correlation_similarity": "correlation_similarity",
        }
        for label, fname in helpers.items():
            fn = getattr(utils, fname, None)
            if fn is None:
                continue
            try:
                out[label] = float(fn(real, synthetic, md))
            except Exception as exc:  # noqa: BLE001
                out[f"{label}_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
        return out


class PoolAdapter(GeneratorAdapter):
    """A pool of rows generated elsewhere — the offline path.

    Use it when the engine runs outside this process (a GPU box, the zEDGE
    service, a nightly job) and all that comes back is a CSV or a DataFrame.

    The pool must have been generated from the same fold's training split.
    ``fold_pools`` keyed by fold index enforces that per fold; a single pool
    reused across folds is a leak and is rejected unless ``allow_shared_pool``.
    """

    def __init__(
        self,
        pool: Any = None,
        name: str = "pool",
        fold_pools: Optional[Mapping[int, Any]] = None,
        allow_shared_pool: bool = False,
        allow_replacement: bool = False,
    ):
        super().__init__(name)
        self.pool = pool
        self.fold_pools = dict(fold_pools) if fold_pools else None
        self.allow_shared_pool = allow_shared_pool
        self.allow_replacement = allow_replacement
        self.fold: Optional[int] = None
        self._frame: Optional[pd.DataFrame] = None

    @staticmethod
    def _read(pool: Any) -> pd.DataFrame:
        if isinstance(pool, pd.DataFrame):
            return pool.copy()
        path = _expand(pool)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    @property
    def fixed_pool(self) -> bool:
        return True

    @property
    def n_available(self) -> Optional[int]:
        return None if self._frame is None else len(self._frame)

    def _reset(self) -> None:
        self._frame = None

    def _fit(self, data, target, discrete_columns) -> None:
        if self.fold_pools is None and self.pool is None:
            raise ValueError(
                f"[{self.name}] has no rows to draw from: pass pool= for a single "
                "frame or file, or fold_pools={fold: pool} for one per CV fold."
            )
        if self.fold_pools is not None:
            if self.fold is None or self.fold not in self.fold_pools:
                raise ValueError(
                    f"[{self.name}] no pool registered for fold {self.fold}. "
                    "Provide one pool per fold, each generated from that fold's "
                    "training split only."
                )
            self._frame = self._read(self.fold_pools[self.fold])
        else:
            if self.fold is not None and not self.allow_shared_pool:
                raise ValueError(
                    f"[{self.name}] one pool is being reused across CV folds. A pool "
                    "generated once from the whole table has seen every fold's test "
                    "rows. Pass fold_pools={fold: pool}, or allow_shared_pool=True if "
                    "the pool provably came from this fold's training split."
                )
            self._frame = self._read(self.pool)

    def _sample(self, n, seed) -> pd.DataFrame:
        from pamir.synthetic.mixing import take_rows

        rng = np.random.default_rng(seed)
        return take_rows(self._frame, n, rng, allow_replacement=self.allow_replacement)


class ZedgeAdapter(GeneratorAdapter):
    """zEDGE / zGN-LatentDiff (latent diffusion) from the zgn_latdiff repository.

    Two ways to run it:

    ``mode="pipeline"``
        Train on the fold's split in-process: the training frame is written to a
        temporary CSV, ``run_full_pipeline`` builds the artifact, and
        ``generate_samples`` draws the pool.  Needs torch and, realistically, a GPU.

    ``mode="artifact"``
        Reuse an artifact directory already trained on this fold's split and only
        sample from it.  ``fit`` then verifies nothing but the artifact's presence,
        so the caller carries the burden of the leakage contract — which is why
        ``fold_artifacts`` is the keyed form to prefer.
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        fold_artifacts: Optional[Mapping[int, str]] = None,
        name: str = "zedge",
        mode: str = "pipeline",
        device: str = "cpu",
        temperature: float = 2.0,
        num_steps: int = 50,
        work_dir: Optional[str] = None,
        oversample: float = 1.2,
        allow_shared_artifact: bool = False,
        **pipeline_kwargs,
    ):
        super().__init__(name)
        if mode not in ("pipeline", "artifact"):
            raise ValueError("mode must be 'pipeline' or 'artifact'.")
        self.repo_path = repo_path
        self.artifact_dir = artifact_dir
        self.fold_artifacts = dict(fold_artifacts) if fold_artifacts else None
        self.mode = mode
        self.device = device
        self.temperature = temperature
        self.num_steps = num_steps
        self.work_dir = work_dir
        self.oversample = oversample
        self.allow_shared_artifact = allow_shared_artifact
        self.pipeline_kwargs = pipeline_kwargs
        self.fold: Optional[int] = None
        self._artifact: Optional[str] = None
        self._tmp: Optional[tempfile.TemporaryDirectory] = None

    def _ensure_path(self) -> None:
        if self.repo_path:
            base = _expand(self.repo_path)
            if not base.exists():
                raise FileNotFoundError(
                    f"[{self.name}] repo_path {base} does not exist. It is not added "
                    "to sys.path, so the import would fail with a less obvious error."
                )
            src = base / "src"
            root = str(src if src.exists() else base)
            if root not in sys.path:
                sys.path.insert(0, root)

    def _reset(self) -> None:
        self._artifact = None
        self._tmp = None

    def _resolve_artifact(self) -> str:
        if self.fold_artifacts is not None:
            if self.fold is None or self.fold not in self.fold_artifacts:
                raise ValueError(
                    f"[{self.name}] no artifact registered for fold {self.fold}."
                )
            return str(_expand(self.fold_artifacts[self.fold]))
        if self.artifact_dir is None:
            raise ValueError(f"[{self.name}] mode='artifact' needs artifact_dir.")
        if self.fold is not None and not self.allow_shared_artifact:
            raise ValueError(
                f"[{self.name}] one artifact is being reused across CV folds. An "
                "artifact trained on the whole table has seen every fold's test rows. "
                "Pass fold_artifacts={fold: dir}, or allow_shared_artifact=True if the "
                "artifact provably came from this fold's training split."
            )
        return str(_expand(self.artifact_dir))

    def _fit(self, data, target, discrete_columns) -> None:
        self._ensure_path()
        if self.mode == "artifact":
            self._artifact = self._resolve_artifact()
            if not (Path(self._artifact) / "data" / "info.json").exists():
                raise FileNotFoundError(
                    f"[{self.name}] {self._artifact} does not look like a zEDGE artifact "
                    "(no data/info.json)."
                )
            return

        from zgn_latdiff.pipeline.orchestrator import run_full_pipeline

        self._tmp = tempfile.TemporaryDirectory(
            dir=str(_expand(self.work_dir)) if self.work_dir else None,
            prefix="pamir_zedge_")
        root = Path(self._tmp.name)
        train_csv = root / "train.csv"
        data.to_csv(train_csv, index=False)

        num_cols = [c for c in data.columns if c not in discrete_columns and c != target]
        result = run_full_pipeline(
            train_csv=str(train_csv),
            target_col=target,
            num_cols=num_cols,
            dataset_id=f"pamir_{self.name}_fold{self.fold}",
            artifact_root=str(root / "artifacts"),
            device=self.device,
            **self.pipeline_kwargs,
        )
        self._artifact = str(result["artifact_dir"] if isinstance(result, dict) else result)

    def _sample(self, n, seed) -> pd.DataFrame:
        self._ensure_path()
        from zgn_latdiff.pipeline.sample import generate_samples

        art = Path(self._artifact)
        out_dir = art / "synthetic"
        draw = int(np.ceil(n * self.oversample))
        result = generate_samples(
            data_dir=str(art / "data"),
            ckpt_dir=str(art / "ckpt"),
            output_dir=str(out_dir),
            temperature=self.temperature,
            num_samples=draw,
            num_steps=self.num_steps,
            device=self.device,
        )
        path = result.get("output_path") if isinstance(result, dict) else None
        if path is None:
            candidates = sorted(out_dir.glob("synthetic_temp*.csv"),
                                key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(f"[{self.name}] sampling wrote no pool in {out_dir}.")
            path = candidates[-1]
        frame = pd.read_csv(path)
        if len(frame) < n:
            raise ValueError(
                f"[{self.name}] pool holds {len(frame)} rows but {n} were requested; "
                "raise oversample."
            )
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(frame), size=n, replace=False)
        return frame.iloc[np.sort(idx)]


def as_adapter(obj: Any, name: str = "generator", **kwargs) -> GeneratorAdapter:
    """Wrap whatever the caller has into a :class:`GeneratorAdapter`.

    Accepts an adapter (returned unchanged), a DataFrame or CSV path (pool), an
    SDV synthesizer class or instance, or any object with ``fit`` and ``sample``.
    """
    if isinstance(obj, GeneratorAdapter):
        return obj
    if isinstance(obj, (pd.DataFrame, Path)) or (isinstance(obj, str) and _expand(obj).exists()):
        return PoolAdapter(pool=obj, name=name, **kwargs)

    target_cls = obj if isinstance(obj, type) else type(obj)
    if any(base.__name__ == "BaseSingleTableSynthesizer" for base in target_cls.__mro__):
        return SDVAdapter(obj, name=name, **kwargs)
    if hasattr(obj, "fit") and hasattr(obj, "sample"):
        return CallableAdapter(obj, name=name, **kwargs)
    raise TypeError(
        f"Cannot adapt {obj!r}: expected a GeneratorAdapter, an SDV synthesizer, "
        "a pool (DataFrame or path), or an object with fit()/sample()."
    )
