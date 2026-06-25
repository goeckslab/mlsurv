"""Lightweight column validation for low-level viz functions."""

from typing import Sequence

import pandas as pd


def validate_columns(
    df: pd.DataFrame, required: Sequence[str], func_name: str
) -> None:
    """Check that *df* contains all *required* columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns are validated.
    required : sequence of str
        Column names that must all be present in *df*.
    func_name : str
        Calling function's name, used to prefix error messages.

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.
    ValueError
        Lists missing columns and available columns for quick debugging.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{func_name}() expected a pandas DataFrame, "
            f"got {type(df).__name__}."
        )
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(
            f"{func_name}() received DataFrame with duplicate columns: "
            f"{sorted(set(dupes))}. De-duplicate before plotting."
        )
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{func_name}() missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )
