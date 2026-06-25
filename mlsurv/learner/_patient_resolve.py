"""Patient-spec resolution for ``patient_report()`` and ``explain()``.

Accepts either an integer row index or a string risk-percentile spec
(``'low_risk'``, ``'high_risk'``, ``'median'``, ``'pNN'``) and returns
the integer row index of the patient in ``X``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Percentile-spec parsing is shared with PredictionResult.plot() via the
# package-root module so results/ need not import from learner/.
from .._patient_select import _parse_percentile


def resolve_patient_index(
    patient,
    X: pd.DataFrame,
    pipeline,
    *,
    arg_name: str = "patient",
) -> int:
    """Resolve ``patient`` (int or risk-percentile string) to a row index in ``X``.

    For string specs, runs ``pipeline.predict(X)`` and selects the row at the
    nearest-rank percentile of the resulting risk scores:
    ``order[int(pct/100 * (len(X) - 1))]``.

    Parameters
    ----------
    patient : int or str
        Integer row index, or one of ``'low_risk'`` (= ``'p10'``),
        ``'high_risk'`` (= ``'p90'``), ``'median'`` (= ``'p50'``), or
        ``'pNN'`` for NN in 0..100.
    X : pd.DataFrame
        Patient cohort. Must be non-empty.
    pipeline
        Fitted sklearn ``Pipeline`` with a ``.predict`` method returning
        risk scores (higher = worse prognosis).
    arg_name : str, default ``'patient'``
        Argument name to use in error messages. Both ``explain()`` and
        ``patient_report()`` use the default ``'patient'``.

    Returns
    -------
    int
        Row index into ``X``.

    Raises
    ------
    TypeError
        If ``patient`` is a bool or any non-int/non-str type.
    ValueError
        If ``patient`` is an out-of-range int, an unrecognized string, if
        ``X`` is empty, or if predicted risk scores contain non-finite values.
    """
    if len(X) == 0:
        raise ValueError(f"Cannot resolve {arg_name}: X is empty.")

    # bool is a subclass of int — reject explicitly.
    if isinstance(patient, bool):
        raise TypeError(
            f"{arg_name} must be int or str, got bool ({patient!r})."
        )

    if isinstance(patient, (int, np.integer)):
        idx = int(patient)
        if idx < 0 or idx >= len(X):
            raise ValueError(
                f"{arg_name}={idx} out of range for X with {len(X)} rows."
            )
        return idx

    if isinstance(patient, str):
        pct = _parse_percentile(patient, arg_name)
        risk = np.asarray(pipeline.predict(X))
        if not np.all(np.isfinite(risk)):
            raise ValueError(
                f"Cannot resolve {arg_name}={patient!r}: predicted risk "
                "scores contain non-finite values (NaN/Inf). The fitted "
                "model is returning invalid predictions on the provided "
                "data."
            )
        order = np.argsort(risk)
        return int(order[int(pct / 100 * (len(order) - 1))])

    raise TypeError(
        f"{arg_name} must be int or str, got {type(patient).__name__}."
    )
