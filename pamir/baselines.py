"""Reference baselines that run on every PaMIR dataset without edits.

Most of the datasets ship ``object`` or ``bool`` feature columns.  A model that
only selects numeric columns silently discards them; one that hands them
straight to an estimator raises.  The baselines ordinal-encode non-numeric
columns against the levels seen in the **training** rows (an unseen level gets
its own code, ``-2``; a missing value ``-1``), so a row is scored without
reference to any other row being scored.

Each baseline comes in two forms with identical models:

* ``logistic_fit`` / ``gbdt_fit`` — ``fit_fn(X_train, y_train) -> scorer``, the
  form the arrival-mode streaming protocol calls;
* ``logistic_baseline`` / ``gbdt_baseline`` — ``predict_fn(X_train, y_train,
  X_test)``, for the i.i.d. protocol and the refresh mode.

    from pamir import evaluate, fleet_summary, gbdt_fit

    results = evaluate(gbdt_fit)          # arrival mode, reference setting
    fleet_summary(results)

The 0.3.0 baselines, which encoded against the levels of ``X_train`` and
``X_test`` together, are kept as ``logistic_baseline_v03`` / ``gbdt_baseline_v03``
to reproduce 0.3.0 numbers.  :func:`encode_features` (the shared-level encoder)
is unchanged; it is legal in the i.i.d. protocol, where the model is handed the
rows it must score, but not in arrival mode, where it would let a row's code
depend on rows that arrived after it.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

def _non_numeric_columns(frame: pd.DataFrame) -> List[str]:
    # Anything an estimator cannot take as a float needs an ordinal code first.
    # Test against numeric-ness rather than a fixed dtype list so that
    # pyarrow-backed strings (pandas >= 3 loads parquet text as ``str``, not
    # ``object``), categoricals and the like are all caught.  ``bool`` is
    # numeric and casts to float directly, so it is intentionally left alone.
    return [c for c in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[c])]


def encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ordinal-encode non-numeric columns against one shared level set.

    Both frames are returned as float, so a level present in only one of them
    still maps to the same code in both.  Numeric missing values are left as
    ``NaN`` — that is information, and filling it is the caller's decision.

    Parameters
    ----------
    X_train, X_test : DataFrame
        Feature frames with identical columns, in identical order.

    Returns
    -------
    (train, test) : tuple of DataFrame
        New frames; the inputs are not modified.

    Raises
    ------
    ValueError
        If the two frames do not carry the same columns in the same order.
    """
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "X_train and X_test must carry the same column names in the same "
            f"order; got {list(X_train.columns)[:5]}... and "
            f"{list(X_test.columns)[:5]}..."
        )

    train, test = X_train.copy(), X_test.copy()
    for col in _non_numeric_columns(train):
        # Use the nullable string dtype so the level set is pure ``str`` (no
        # mixed float/NA that would make ``sorted`` raise on pandas >= 3), and
        # take levels from the non-missing values only.  Missing entries then
        # map to code -1, distinct from every real level.
        tr = train[col].astype("string")
        te = test[col].astype("string")
        levels = pd.Index(sorted(set(tr.dropna()) | set(te.dropna())))
        train[col] = pd.Categorical(tr, categories=levels).codes
        test[col] = pd.Categorical(te, categories=levels).codes

    return train.astype("float64"), test.astype("float64")


def _constant_scores(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Base-rate scores, for a training split that holds a single class.

    Not an error: with one class there is nothing to discriminate on, so the
    honest answer is a constant.  The protocol scores it as AUC 0.5.
    """
    return np.full(n_test, float(np.mean(y_train)) if len(y_train) else 0.5)


def logistic_baseline_v03(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """0.3.0 logistic baseline (levels shared between X_train and X_test)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = np.asarray(y_train)
    if len(np.unique(y_train)) < 2:
        return _constant_scores(y_train, len(X_test))

    train, test = encode_features(X_train, X_test)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(train, y_train)
    return model.predict_proba(test)[:, 1]


def gbdt_baseline_v03(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    max_iter: int = 150,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """0.3.0 gradient-boosting baseline (levels shared between X_train and X_test)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    y_train = np.asarray(y_train)
    if len(np.unique(y_train)) < 2:
        return _constant_scores(y_train, len(X_test))

    train, test = encode_features(X_train, X_test)
    # scikit-learn's histogram binning raises on a column with a single distinct
    # value; drop constant columns (they carry no signal) before fitting.
    keep = train.columns[train.nunique(dropna=True) >= 2]
    if len(keep) == 0:
        return _constant_scores(y_train, len(X_test))
    train, test = train[keep], test[keep]
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, random_state=seed)
    model.fit(train, y_train)
    return model.predict_proba(test)[:, 1]


# ---------------------------------------------------------------------------
# 0.4.0 baselines: encoder fitted on the training rows only
# ---------------------------------------------------------------------------

def fit_encoder(X_train: pd.DataFrame) -> Dict[str, pd.Index]:
    """Level sets of the non-numeric columns, taken from the training rows."""
    return {col: pd.Index(sorted(set(X_train[col].astype("string").dropna())))
            for col in _non_numeric_columns(X_train)}


def apply_encoder(levels: Dict[str, pd.Index], X: pd.DataFrame) -> pd.DataFrame:
    """Ordinal codes for the encoded columns: -1 missing, -2 unseen level."""
    out = X.copy()
    for col, cats in levels.items():
        s = X[col].astype("string")
        codes = pd.Categorical(s, categories=cats).codes.astype("float64")
        codes[(codes == -1) & s.notna().to_numpy()] = -2.0
        out[col] = codes
    return out.astype("float64")


def _constant_scorer(y_train: np.ndarray):
    rate = float(np.mean(y_train)) if len(y_train) else 0.5
    return lambda X: np.full(len(X), rate)


def logistic_fit(X_train: pd.DataFrame, y_train: np.ndarray):
    """Median-imputed, standardized logistic regression; returns a scorer.

    The cheap reference point: fast enough to run the whole fleet under the
    default protocol parameters.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = np.asarray(y_train)
    if len(np.unique(y_train)) < 2:
        return _constant_scorer(y_train)
    columns = list(X_train.columns)
    levels = fit_encoder(X_train)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(apply_encoder(levels, X_train), y_train)

    def scorer(X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(apply_encoder(levels, X[columns]))[:, 1]

    return scorer


def gbdt_fit(X_train: pd.DataFrame, y_train: np.ndarray, max_iter: int = 150,
             learning_rate: float = 0.1, seed: int = 42):
    """Histogram gradient boosting with native missing-value handling; returns a scorer.

    Uses scikit-learn rather than XGBoost so it needs no optional dependency,
    and so it cannot hit the OpenMP clash documented in the installation notes.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    y_train = np.asarray(y_train)
    if len(np.unique(y_train)) < 2:
        return _constant_scorer(y_train)
    levels = fit_encoder(X_train)
    train = apply_encoder(levels, X_train)
    # scikit-learn's histogram binning raises on a column with a single distinct
    # value; drop constant columns (they carry no signal) before fitting.
    keep = list(train.columns[train.nunique(dropna=True) >= 2])
    if not keep:
        return _constant_scorer(y_train)
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, random_state=seed)
    model.fit(train[keep], y_train)
    columns = list(X_train.columns)

    def scorer(X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(apply_encoder(levels, X[columns])[keep])[:, 1]

    return scorer


def logistic_baseline(X_train: pd.DataFrame, y_train: np.ndarray,
                      X_test: pd.DataFrame) -> np.ndarray:
    """:func:`logistic_fit` as a ``predict_fn``."""
    return logistic_fit(X_train, y_train)(X_test)


def gbdt_baseline(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame,
                  max_iter: int = 150, learning_rate: float = 0.1,
                  seed: int = 42) -> np.ndarray:
    """:func:`gbdt_fit` as a ``predict_fn``."""
    return gbdt_fit(X_train, y_train, max_iter=max_iter,
                    learning_rate=learning_rate, seed=seed)(X_test)
