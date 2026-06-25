"""Shared formatting utilities for the validation section."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsurv.constants import (
    METRIC_REGISTRY,
    get_metric_short_name,
)
from mlsurv.reports._helpers import (  # noqa: F401 — re-exported
    collect_calibration_rows,
    empty_calibration_figure,
)

_CAL_COLUMNS = [
    'model_type', 'eval_time',
    'bin_predicted', 'bin_observed', 'bin_ci_lower', 'bin_ci_upper',
]


def cohort_key(name: str) -> str:
    """Derive a filesystem/dict-safe key from a cohort name."""
    return name.lower().replace(' ', '_')


def format_median_survival_table(median_survival_rows: list) -> list:
    """Convert median survival rows into display-ready table row dicts.

    Parameters
    ----------
    median_survival_rows : list[dict]
        Raw median survival rows from _build_stratification_km_data.

    Returns
    -------
    list[dict]
        Table rows with formatted 'Median Survival' strings.
    """
    table_data = []
    for row in median_survival_rows:
        median_str = (
            f"{row['median_survival']:.1f}"
            if row['median_survival'] is not None
            else "Not reached"
        )
        table_data.append({
            'Risk Group': row['risk_group'],
            'N': row['n_patients'],
            'Events': row['n_events'],
            'Median Survival': median_str,
        })
    return table_data


def find_best_in_column(
    values: pd.Series,
    is_lower_better: bool,
    mask: pd.Series = None,
) -> pd.Series:
    """Return a boolean mask of rows that hold the best value.

    Parameters
    ----------
    values : pd.Series
        Parsed numeric values (may contain NaN).
    is_lower_better : bool
        If True, the minimum is best; otherwise the maximum.
    mask : pd.Series, optional
        Boolean mask limiting the search to a subset of rows.

    Returns
    -------
    pd.Series
        Boolean mask with True at positions holding the best value.
    """
    subset = values[mask] if mask is not None else values
    if subset.isna().all():
        return pd.Series(False, index=values.index)
    best_val = subset.min() if is_lower_better else subset.max()
    best_mask = values == best_val
    if mask is not None:
        best_mask = mask & best_mask
    return best_mask


def apply_bold_formatting(
    df: pd.DataFrame,
    col: str,
    values: pd.Series,
    is_lower_better: bool,
    group_col: str = None,
) -> None:
    """Bold the best value(s) in *col*, optionally per group.

    Modifies *df* in place.
    """
    if group_col and group_col in df.columns:
        for group in df[group_col].unique():
            group_mask = df[group_col] == group
            best_mask = find_best_in_column(
                values, is_lower_better, mask=group_mask,
            )
            df.loc[best_mask, col] = df.loc[best_mask, col].apply(
                lambda x: f"**{x}**"
            )
    else:
        best_mask = find_best_in_column(values, is_lower_better)
        df.loc[best_mask, col] = df.loc[best_mask, col].apply(
            lambda x: f"**{x}**"
        )


def get_metric_column_sets() -> tuple:
    """Return (lower_better, all_metric_cols) sets from the registry.

    Returns
    -------
    tuple of (set[str], set[str])
        lower_better: column names where lower is better.
        all_metric_cols: all recognised metric column names.
    """
    lower_better = set()
    all_metric_cols = set()
    for info in METRIC_REGISTRY.values():
        col_name = get_metric_short_name(info.key, with_asterisk=False)
        if info.lower_is_better:
            suffixed = col_name + " *"
            lower_better.add(suffixed)
            all_metric_cols.add(suffixed)
        else:
            all_metric_cols.add(col_name)
    return lower_better, all_metric_cols


def bold_best_values(
    df: pd.DataFrame,
    group_col: str = None,
) -> pd.DataFrame:
    """Bold best values in numeric columns, optionally grouped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with string-formatted numeric values.
    group_col : str, optional
        Column to group by when determining best values.

    Returns
    -------
    pd.DataFrame
        DataFrame with best values bolded in markdown format.
    """
    lower_better, all_metric_cols = get_metric_column_sets()

    df = df.copy()
    numeric_cols = [c for c in df.columns if c in all_metric_cols]

    if not numeric_cols or len(df) <= 1:
        return df

    for col in numeric_cols:
        values = df[col].apply(
            lambda x: float(x)
            if isinstance(x, str)
            and x.replace('.', '').replace('-', '', 1).isdigit()
            else np.nan
        )
        if values.isna().all():
            continue

        apply_bold_formatting(
            df, col, values,
            is_lower_better=(col in lower_better),
            group_col=group_col,
        )

    return df
