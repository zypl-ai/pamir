"""Failure accounting for the evaluation protocols.

A benchmark that drops the datasets a model failed on reports an average over
a subset the model itself selected.  That flatters a broken model over a
working one, so every call into user code is counted here and every failure is
kept rather than discarded.

The policy is chosen by the caller:

``raise``
    Propagate the original exception immediately.  Use while developing a model.
``warn`` (default)
    Record the failure, carry on, and emit one warning per dataset naming how
    many calls raised and the first error.
``ignore``
    Record the failure silently.  Still counted in ``n_failures``; only the
    warning is suppressed.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional

ON_ERROR_POLICIES = ("raise", "warn", "ignore")

# Distinct messages kept per run.  A model that fails the same way 200 times
# needs one line, not two hundred; a model that fails several ways needs them.
MAX_RECORDED_ERRORS = 5


def check_on_error(on_error: str) -> str:
    """Validate an ``on_error`` policy name, returning it unchanged."""
    if on_error not in ON_ERROR_POLICIES:
        raise ValueError(
            f"on_error must be one of {ON_ERROR_POLICIES}, got {on_error!r}."
        )
    return on_error


class FailureLog:
    """Counts calls into a user-supplied ``predict_fn`` and the ones that raised.

    Parameters
    ----------
    on_error : str
        One of ``"raise"``, ``"warn"``, ``"ignore"``.
    context : str
        Prefix for the warning, e.g. ``"south_german"`` — so a fleet run says
        which dataset produced the failure.
    """

    def __init__(self, on_error: str = "warn", context: str = ""):
        self.on_error = check_on_error(on_error)
        self.context = context
        self.n_calls = 0
        self.n_failures = 0
        self.errors: List[str] = []

    def call(self, fn: Callable, *args: Any) -> Optional[Any]:
        """Invoke ``fn``, recording a failure instead of losing it.

        Returns ``None`` when the call raised and the policy allows continuing.
        A ``None`` return from ``fn`` itself is indistinguishable here, which is
        why callers wrap a function that commits its own result.
        """
        self.n_calls += 1
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            self.n_failures += 1
            self._record(exc)
            if self.on_error == "raise":
                raise
            return None

    def _record(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if message not in self.errors and len(self.errors) < MAX_RECORDED_ERRORS:
            self.errors.append(message)

    def warn_if_failed(self) -> None:
        """Emit one warning for the run, if the policy asks for it."""
        if not self.n_failures or self.on_error != "warn":
            return
        where = f"{self.context}: " if self.context else ""
        warnings.warn(
            f"{where}{self.n_failures} of {self.n_calls} predict_fn calls raised. "
            f"First error — {self.errors[0]}. "
            "Rows left unscored are excluded from the AUC, so this number is not "
            "comparable with a model that scored everything. Pass "
            "on_error='raise' to see the traceback.",
            UserWarning,
            stacklevel=3,
        )

    def as_dict(self) -> Dict[str, Any]:
        """The accounting fields to merge into a result dict."""
        return {
            "n_calls": self.n_calls,
            "n_failures": self.n_failures,
            "errors": list(self.errors),
        }
