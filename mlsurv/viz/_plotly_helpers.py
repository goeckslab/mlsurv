"""Shared Plotly error-bar helpers for mlsurv visualizations.

Centralises the ``error_x`` / ``error_y`` dict construction that recurs
across feature-importance, hazard-ratio, coefficient, SHAP, calibration,
metrics, and report plots.  Also ensures numpy arrays are serialised to
plain lists (required for Plotly HTML export).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _to_list(arr):
    """Convert numpy arrays to Python lists for Plotly JSON serialization.

    Plotly silently drops numpy arrays from ``error_x`` / ``error_y``
    dicts during HTML export.  This helper normalises both scalar-like
    and array-like inputs.
    """
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return arr


def symmetric_error_bar(
    values,
    *,
    color: Optional[str] = None,
    visible: bool = True,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a symmetric Plotly ``error_x`` / ``error_y`` dict.

    Parameters
    ----------
    values : array-like
        Half-widths of the error bars.
    color : str, optional
        CSS colour for the error bars.
    visible : bool
        Whether the error bars are rendered.
    **extra
        Additional keys forwarded to the dict (e.g. ``thickness``, ``width``).

    Returns
    -------
    dict
        Ready for ``go.Bar(error_x=...)`` or ``go.Scatter(error_y=...)``.
    """
    d: Dict[str, Any] = {
        "type": "data",
        "array": _to_list(values),
        "visible": visible,
    }
    if color is not None:
        d["color"] = color
    d.update(extra)
    return d


def asymmetric_error_bar(
    upper_delta,
    lower_delta,
    *,
    color: Optional[str] = None,
    visible: bool = True,
    **extra: Any,
) -> Optional[Dict[str, Any]]:
    """Build an asymmetric Plotly ``error_x`` / ``error_y`` dict.

    Parameters
    ----------
    upper_delta : array-like
        Positive deltas from the central value to the upper bound.
    lower_delta : array-like
        Positive deltas from the central value to the lower bound.
    color : str, optional
        CSS colour for the error bars.
    visible : bool
        Whether the error bars are rendered.
    **extra
        Additional keys (e.g. ``thickness``, ``width``).

    Returns
    -------
    dict or None
        ``None`` when all upper/lower deltas are non-finite (nothing to draw).
    """
    upper_arr = np.asarray(upper_delta)
    lower_arr = np.asarray(lower_delta)
    if not np.isfinite(upper_arr).any() and not np.isfinite(lower_arr).any():
        return None
    d: Dict[str, Any] = {
        "type": "data",
        "symmetric": False,
        "array": _to_list(upper_arr),
        "arrayminus": _to_list(lower_arr),
        "visible": visible,
    }
    if color is not None:
        d["color"] = color
    d.update(extra)
    return d
