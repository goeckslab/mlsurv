"""Binary missingness indicator columns."""

from __future__ import annotations

import warnings

import pandas as pd

from mlsurv._warnings import SuggestionWarning


def add_missing_indicators(
    X: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Add binary missingness indicator columns to a DataFrame.

    For each specified column, appends a ``{col}_is_missing`` column
    containing 1 where the original value is NaN and 0 otherwise.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.
    columns : list of str
        Column names to create indicators for.

    Returns
    -------
    pd.DataFrame
        Copy of *X* with ``{col}_is_missing`` columns appended.

    Raises
    ------
    TypeError
        If *columns* is not a list or tuple.
    ValueError
        If any name in *columns* is not present in *X*.
    """
    if not isinstance(columns, (list, tuple)):
        raise TypeError(
            f"columns must be a list or tuple, got {type(columns).__name__}"
        )

    unknown = [c for c in columns if c not in X.columns]
    if unknown:
        raise ValueError(
            f"Columns not found in DataFrame: {unknown}"
        )

    for col in columns:
        if X[col].notna().all():
            warnings.warn(
                f"Column '{col}' has no missing values; "
                f"'{col}_is_missing' will be all zeros.",
                SuggestionWarning,
                stacklevel=2,
            )

    result = X.copy()
    for col in columns:
        result[f"{col}_is_missing"] = X[col].isna().astype(int)

    return result
