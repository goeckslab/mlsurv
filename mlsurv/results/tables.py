"""Output table formatting utilities."""

import warnings

import pandas as pd
import numpy as np
from typing import Optional, List


def format_metrics_table(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    model_col: str = 'model_type',
    precision: int = 3,
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Format metrics DataFrame for display.

    Parameters
    ----------
    df : DataFrame
        Metrics DataFrame with model performance data.
    metrics : list, optional
        Columns to include. Auto-detected if None.
    model_col : str
        Column containing model names.
    precision : int
        Number of decimal places.
    sort_by : str, optional
        Column to sort by.
    ascending : bool
        Sort order.

    Returns
    -------
    DataFrame
        Formatted table.
    """
    df = df.copy()

    if metrics is None:
        # Auto-detect numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        metrics = [c for c in numeric_cols if c != model_col]

    # Select columns
    cols = [model_col] + [c for c in metrics if c in df.columns]
    result = df[cols].copy()

    # Round numeric columns
    for col in metrics:
        if col in result.columns:
            result[col] = result[col].round(precision)

    # Sort if requested
    if sort_by and sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=ascending)

    return result.reset_index(drop=True)


def format_importance_table(
    df: pd.DataFrame,
    n: int = 20,
    importance_col: str = 'perm_mean',
    std_col: Optional[str] = 'perm_std',
    feature_col: str = 'feature',
    precision: int = 4,
) -> pd.DataFrame:
    """
    Format feature importance DataFrame for display.

    Parameters
    ----------
    df : DataFrame
        Importance DataFrame (e.g., from EvaluationResult.importance).
    n : int
        Number of top features to display.
    importance_col : str
        Column with importance values.
    std_col : str, optional
        Column with standard deviation.
    feature_col : str
        Column with feature names.
    precision : int
        Number of decimal places.

    Returns
    -------
    DataFrame
        Formatted table with rank column.
    """
    # Validate required columns up front so callers get a clear error
    # (df.nlargest would raise a bare KeyError; result[cols] would KeyError on
    # a missing feature_col). std_col is optional and handled gracefully below.
    missing = [
        (name, col)
        for name, col in (("importance_col", importance_col), ("feature_col", feature_col))
        if col not in df.columns
    ]
    if missing:
        names = ", ".join(f"{name}={col!r}" for name, col in missing)
        raise ValueError(
            f"format_importance_table: column(s) not found ({names}). "
            f"Available columns: {list(df.columns)}"
        )

    # Sort and select top N
    result = df.nlargest(n, importance_col).copy()

    # Add rank
    result.insert(0, 'Rank', range(1, len(result) + 1))

    # Rename columns for display
    rename_map = {
        feature_col: 'Feature',
        importance_col: 'Importance',
    }
    if std_col and std_col in result.columns:
        rename_map[std_col] = 'Std Dev'

    result = result.rename(columns=rename_map)

    # Select columns
    cols = ['Rank', 'Feature', 'Importance']
    if 'Std Dev' in result.columns:
        cols.append('Std Dev')

    result = result[cols]

    # Round
    for col in ['Importance', 'Std Dev']:
        if col in result.columns:
            result[col] = result[col].round(precision)

    return result.reset_index(drop=True)


def format_bootstrap_summary(
    df: pd.DataFrame,
    ci_format: str = '[{lower:.3f}, {upper:.3f}]',
    precision: int = 3,
    ci_level: Optional[float] = None,
) -> pd.DataFrame:
    """
    Format bootstrap summary for display.

    Parameters
    ----------
    df : DataFrame
        Summary DataFrame (e.g., from BootstrapResult.summary).
    ci_format : str
        Format string for CI display.
    precision : int
        Number of decimal places.
    ci_level : float, optional
        Confidence level used to label the CI column (e.g. 0.80 ->
        "80% CI"). When None, falls back to ``df.attrs['ci_level']``
        (stamped by BootstrapResult), then to 0.95 (with a
        SuggestionWarning, since the true level is unknown).

    Returns
    -------
    DataFrame
        Formatted table with combined CI column.

    Raises
    ------
    ValueError
        If ``ci_level`` is not strictly between 0 and 1.
    """
    has_ci_cols = 'ci_lower' in df.columns and 'ci_upper' in df.columns
    if ci_level is None:
        if 'ci_level' not in df.attrs and has_ci_cols:
            from mlsurv._warnings import SuggestionWarning
            warnings.warn(
                "CI level unknown for this DataFrame (no ci_level argument "
                "and no df.attrs['ci_level']); labeling the column '95% CI'. "
                "Pass ci_level= if the intervals were computed at another "
                "level.",
                SuggestionWarning,
                stacklevel=2,
            )
        ci_level = df.attrs.get('ci_level', 0.95)
    if not 0 < ci_level < 1:
        raise ValueError(
            f"ci_level must be between 0 and 1 (exclusive), got {ci_level}. "
            f"Use ci_level=0.95 for a 95% interval, not ci_level=95."
        )
    result = df.copy()

    # Round numeric columns
    for col in ['point', 'mean', 'std', 'ci_lower', 'ci_upper']:
        if col in result.columns:
            result[col] = result[col].round(precision)

    # Create combined CI column
    if has_ci_cols:
        result[f'{ci_level * 100:g}% CI'] = result.apply(
            lambda row: ci_format.format(lower=row['ci_lower'], upper=row['ci_upper']),
            axis=1
        )
        result = result.drop(columns=['ci_lower', 'ci_upper'])

    # Rename columns
    rename_map = {
        'metric': 'Metric',
        'point': 'Point',
        'mean': 'Mean',
        'std': 'Std Dev',
        'model_type': 'Model',
    }
    result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})

    return result


def _warn_no_ci(model, cause: str) -> None:
    """Say why the ``95% CI`` column is ``'N/A'`` instead of returning it bare.

    ``'N/A'`` is honest but mute: a caller reading the table has no way to tell
    a model that cannot have intervals from a Cox fit whose Hessian turned out
    to be singular. CoxPH's recorded reason is a complete sentence and already
    says the hazard ratios are unaffected, so it replaces the whole tail rather
    than *cause* being spliced into it. ``covariance_reason`` reads it by
    attribute rather than by model type, so ``results/`` keeps no dependency on
    ``models/``, the same way the ``hazard_ratio_valid`` gate below does.

    *cause* follows a full stop, so callers pass it capitalized.
    """
    from mlsurv._shared import covariance_reason
    from mlsurv._warnings import SuggestionWarning

    detail = covariance_reason(model) or f"{cause} Hazard ratios are unaffected."
    warnings.warn(
        f"{model.__class__.__name__}: the 95% CI column is 'N/A'. {detail}",
        SuggestionWarning,
        stacklevel=3,
    )


def format_hazard_ratios(
    model,
    feature_names: Optional[List[str]] = None,
    n: int = 20,
    precision: int = 3,
) -> pd.DataFrame:
    """
    Format Cox model hazard ratios as a table.

    Parameters
    ----------
    model : fitted model with ``hazard_ratio_valid=True``
        A model whose ``exp(coef)`` is a genuine hazard ratio (currently only
        unpenalized Cox PH). Other models raise ``ValueError``.
    feature_names : list, optional
        Feature names.
    n : int
        Number of features to display.
    precision : int
        Number of decimal places.

    Returns
    -------
    DataFrame
        Table with Feature, HR, 95% CI columns. The ``95% CI`` cells are the
        string ``'N/A'`` when no usable covariance matrix is available (e.g. a
        singular Hessian on a rank-deficient design); a ``SuggestionWarning``
        says why. ``HR`` is unaffected.

    Raises
    ------
    ValueError
        If the model's coefficients are not hazard ratios
        (``hazard_ratio_valid`` is False), or if it does not expose
        coefficients.
    """
    # exp(coef) is a hazard ratio only for unpenalized Cox PH.
    # Inline getattr keeps results/ free of a viz dependency.
    if not getattr(model, "hazard_ratio_valid", False):
        raise ValueError(
            f"{model.__class__.__name__} coefficients are not hazard ratios: "
            f"exp(coef) is a valid hazard ratio only for unpenalized Cox PH "
            f"models (hazard_ratio_valid=True). Penalized (CoxNet) and ranking "
            f"(FSSVM) coefficients are not valid hazard ratios, and AFT "
            f"coefficients are not log-hazards. Use plot_coefficients() to "
            f"inspect them."
        )

    coef = model.get_coefficients()
    if coef is None:
        raise ValueError(f"{model.__class__.__name__} does not have coefficients")

    if feature_names is None:
        if hasattr(coef, 'index'):
            feature_names = coef.index.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(len(coef))]

    coef_values = coef.values if hasattr(coef, 'values') else np.array(coef)

    df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef_values,
        'HR': np.exp(coef_values)
    })

    # Try to get standard errors
    if hasattr(model, 'model_') and hasattr(model.model_, 'covariance_matrix_'):
        try:
            se = np.sqrt(np.diag(model.model_.covariance_matrix_))
            df['CI Lower'] = np.exp(df['Coefficient'] - 1.96 * se)
            df['CI Upper'] = np.exp(df['Coefficient'] + 1.96 * se)
            df['95% CI'] = df.apply(
                lambda row: f"[{row['CI Lower']:.{precision}f}, {row['CI Upper']:.{precision}f}]",
                axis=1
            )
        except (ValueError, IndexError, TypeError, np.linalg.LinAlgError) as exc:
            _warn_no_ci(model, f"The covariance matrix could not be used ({exc}).")
            df['95% CI'] = 'N/A'
    else:
        _warn_no_ci(model, "No covariance matrix was attached to the fit.")
        df['95% CI'] = 'N/A'

    # Sort by absolute coefficient
    df['abs_coef'] = df['Coefficient'].abs()
    df = df.nlargest(n, 'abs_coef')
    df = df.drop(columns=['abs_coef', 'CI Lower', 'CI Upper', 'Coefficient'], errors='ignore')

    # Round
    df['HR'] = df['HR'].round(precision)

    return df[['Feature', 'HR', '95% CI']].reset_index(drop=True)


def to_latex(
    df: pd.DataFrame,
    caption: str = '',
    label: str = '',
    precision: int = 3,
) -> str:
    """
    Convert DataFrame to LaTeX table.

    Parameters
    ----------
    df : DataFrame
        Table to convert.
    caption : str
        Table caption.
    label : str
        LaTeX label for referencing.
    precision : int
        Number of decimal places for floats.

    Returns
    -------
    str
        LaTeX table code.
    """
    return df.to_latex(
        index=False,
        caption=caption,
        label=label,
        float_format=f'%.{precision}f',
        escape=True,
    )


def to_markdown(df: pd.DataFrame, precision: int = 3) -> str:
    """
    Convert DataFrame to Markdown table.

    Parameters
    ----------
    df : DataFrame
        Table to convert.
    precision : int
        Number of decimal places for floats.

    Returns
    -------
    str
        Markdown table.
    """
    # Round numeric columns
    df = df.copy()
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].round(precision)

    return df.to_markdown(index=False)


def to_html(
    df: pd.DataFrame,
    precision: int = 3,
    classes: str = 'table table-striped',
) -> str:
    """
    Convert DataFrame to HTML table.

    Parameters
    ----------
    df : DataFrame
        Table to convert.
    precision : int
        Number of decimal places for floats.
    classes : str
        CSS classes for the table element.

    Returns
    -------
    str
        HTML table code.
    """
    df = df.copy()
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].round(precision)

    return df.to_html(index=False, classes=classes)



