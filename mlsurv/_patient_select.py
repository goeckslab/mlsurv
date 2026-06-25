"""Resolve a patient spec against a precomputed risk-score array.

Shared by ``learner`` (where the percentile vocabulary originates, via
``learner/_patient_resolve.py``) and ``results.PredictionResult.plot()``, which
resolves the same shorthand against the result's already-computed
``risk_scores`` with no fitted pipeline. Kept at the package root so that
``results/`` does not import from ``learner/``.

The percentile vocabulary: ``'low_risk'`` (= ``'p10'``), ``'high_risk'``
(= ``'p90'``), ``'median'`` (= ``'p50'``), or ``'pNN'`` for NN in 0..100.
"""

from __future__ import annotations

import re

import numpy as np

_ALIASES = {"low_risk": 10, "high_risk": 90, "median": 50}
_PNN_RE = re.compile(r"^p(\d{1,3})$")


def _parse_percentile(spec: str, arg_name: str) -> int:
    """Parse a risk-percentile spec string to an integer percentile in 0..100."""
    key = spec.lower()
    if key in _ALIASES:
        return _ALIASES[key]
    m = _PNN_RE.match(key)
    if m is not None:
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            return pct
    raise ValueError(
        f"{arg_name}={spec!r} not recognized. Use an integer row index "
        "or one of: 'low_risk', 'high_risk', 'median', 'pNN' "
        "(NN in 0..100, e.g. 'p25')."
    )


def resolve_index_from_scores(
    patient,
    risk_scores,
    *,
    arg_name: str = "patient",
) -> int:
    """Resolve ``patient`` (int or risk-percentile string) to an index into ``risk_scores``.

    For string specs, selects the entry at the nearest-rank percentile of
    ``risk_scores``: ``order[int(pct / 100 * (n - 1))]`` where
    ``order = argsort(risk_scores)`` (ascending; higher risk = worse prognosis).
    This mirrors the pipeline-based ``resolve_patient_index`` so a given
    shorthand picks the same patient whether resolved from a fitted model or
    from a ``PredictionResult``.

    Parameters
    ----------
    patient : int or str
        Integer index, or one of ``'low_risk'`` (= ``'p10'``), ``'high_risk'``
        (= ``'p90'``), ``'median'`` (= ``'p50'``), or ``'pNN'`` for NN in 0..100.
    risk_scores : array-like
        Per-patient risk scores (higher = worse prognosis). Must be non-empty.
    arg_name : str, default ``'patient'``
        Argument name to use in error messages.

    Returns
    -------
    int
        Index into ``risk_scores``.

    Raises
    ------
    TypeError
        If ``patient`` is a bool or any non-int/non-str type.
    ValueError
        If ``patient`` is an out-of-range int, an unrecognized string,
        ``risk_scores`` is empty, or the scores contain non-finite values.
    """
    scores = np.asarray(risk_scores)
    n = len(scores)
    if n == 0:
        raise ValueError(f"Cannot resolve {arg_name}: risk_scores is empty.")

    # bool is a subclass of int — reject explicitly.
    if isinstance(patient, bool):
        raise TypeError(
            f"{arg_name} must be int or str, got bool ({patient!r})."
        )

    if isinstance(patient, (int, np.integer)):
        idx = int(patient)
        if idx < 0 or idx >= n:
            raise ValueError(
                f"{arg_name}={idx} out of range for {n} patient(s)."
            )
        return idx

    if isinstance(patient, str):
        pct = _parse_percentile(patient, arg_name)
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"Cannot resolve {arg_name}={patient!r}: risk scores contain "
                "non-finite values (NaN/Inf)."
            )
        order = np.argsort(scores)
        return int(order[int(pct / 100 * (n - 1))])

    raise TypeError(
        f"{arg_name} must be int or str, got {type(patient).__name__}."
    )
