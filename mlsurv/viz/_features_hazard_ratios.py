"""Hazard ratio visualizations.

Contains ``compute_hazard_ratios``, ``plot_hazard_ratios``,
``plot_hazard_ratios_comparison``, and supporting helpers.
"""

import math
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv._warnings import (
    ClampWarning, InternalWarning, MethodologicalWarning,
)
from mlsurv.viz._palettes import (
    FeatureGroupsType,
    MANUSCRIPT_DEFAULTS,
    _add_group_legend_traces,
    _add_group_separator_shapes,
    _make_group_strip_trace,
    _resolve_groups,
    _sort_features_by_group,
    get_model_color,
    hex_to_rgba,
)
from mlsurv.viz._features_helpers import _importance_full_title
from mlsurv.viz._plotly_helpers import asymmetric_error_bar
from mlsurv.viz._theme import (
    adaptive_size, apply_size_overrides, create_figure,
    margin_for_labels, truncate_labels,
)
from mlsurv.viz._validation import validate_columns

_EXP_CLIP = 700  # np.exp(709.78) overflows float64; 700 gives safe margin

# exp(coef) is a genuine hazard ratio only for models whose
# hazard_ratio_valid flag is True (unpenalized Cox PH).
HR_NOT_VALID_MSG = (
    "{name} coefficients are not hazard ratios: exp(coef) is a valid "
    "multivariable hazard ratio only for unpenalized Cox PH models "
    "(hazard_ratio_valid=True). Penalized (CoxNet) and ranking (FSSVM) "
    "coefficients are not valid hazard ratios, and AFT coefficients are not "
    "log-hazards. Use plot_coefficients() or plot_coefficient_comparison() "
    "to inspect coefficients instead."
)

# Emitted when a non-HR-valid model is dropped from a multi-model HR plot.
DROP_HR_MSG = (
    "Dropping '{name}' from the hazard-ratio plot: its coefficients are not "
    "hazard ratios (exp(coef) is a valid HR only for unpenalized Cox PH). "
    "Use plot_coefficients() to inspect its coefficients."
)


def _safe_exp(x):
    """Exponentiate with clipping to avoid float64 overflow."""
    arr = np.asarray(x, dtype=float)
    clipped = np.clip(arr, -_EXP_CLIP, _EXP_CLIP)
    n_clipped = np.count_nonzero(arr != clipped)
    if n_clipped:
        warnings.warn(
            f"{n_clipped} coefficient(s) clamped to [{-_EXP_CLIP}, {_EXP_CLIP}] "
            f"before exp() to prevent overflow. Displayed hazard ratios for "
            f"these features are capped, not exact. "
            f"Consider stronger regularization.",
            ClampWarning,
            stacklevel=2,
        )
    return np.exp(clipped)


def _extract_coefficients(model, feature_names):
    """Extract coefficient array and feature names from a Cox model."""
    if hasattr(model, 'get_coefficients'):
        coef = model.get_coefficients()
    elif hasattr(model, 'model_') and hasattr(model.model_, 'coef_'):
        coef = model.model_.coef_
    else:
        coef = None
    if coef is None:
        raise ValueError(f"{model.__class__.__name__} does not have coefficients")

    if feature_names is None:
        if hasattr(coef, "index"):
            feature_names = coef.index.tolist()
        elif hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_
        else:
            feature_names = [f"Feature {i}" for i in range(len(coef))]
    return coef, feature_names


def _compute_error_bounds(df, model, coef_values, ci_level, error_bars):
    """Add ci_lower, ci_upper, ci_available columns to *df* in place."""
    if hasattr(model, "model_") and hasattr(model.model_, "covariance_matrix_"):
        try:
            se = np.sqrt(np.diag(model.model_.covariance_matrix_))
            if error_bars == "se":
                multiplier = 1.0
            else:
                from scipy.stats import norm as _norm
                multiplier = _norm.ppf(1 - (1 - ci_level) / 2)
            df["ci_lower"] = _safe_exp(coef_values - multiplier * se)
            df["ci_upper"] = _safe_exp(coef_values + multiplier * se)
            df["ci_available"] = True
            return
        except (ValueError, IndexError, TypeError, np.linalg.LinAlgError) as e:
            warnings.warn(
                f"Could not compute error bounds from covariance matrix "
                f"for {model.__class__.__name__}: {e}. Showing point estimates only.",
                InternalWarning,
                stacklevel=3,
            )
    else:
        warnings.warn(
            f"{model.__class__.__name__} does not provide a covariance matrix; "
            f"confidence intervals are unavailable. "
            f"ci_lower and ci_upper are set equal to hr.",
            InternalWarning,
            stacklevel=3,
        )
    df["ci_lower"] = df["hr"]
    df["ci_upper"] = df["hr"]
    df["ci_available"] = False


def compute_hazard_ratios(
    model,
    feature_names: Optional[List[str]] = None,
    ci_level: float = 0.95,
    error_bars: str = "ci",
) -> pd.DataFrame:
    """Extract hazard ratios with error bounds from a fitted Cox model.

    Parameters
    ----------
    model : fitted model with ``hazard_ratio_valid=True``
        A model whose ``exp(coef)`` is a genuine hazard ratio (currently only
        unpenalized Cox PH). Other models raise ``ValueError`` — use
        ``plot_coefficients`` to inspect their coefficients instead.
    feature_names : list, optional
        Feature names. Uses ``model.feature_names_in_`` if available.
    ci_level : float
        Confidence level as a fraction in (0, 1). Default 0.95.
        Only used when ``error_bars="ci"``.
    error_bars : str, default ``"ci"``
        Type of error bars: ``"ci"`` for confidence intervals
        (``exp(coef ± z·SE)``), ``"se"`` for ±SE
        (``exp(coef ± SE)``).

    Returns
    -------
    DataFrame
        Columns: ``feature``, ``hr``, ``ci_lower``, ``ci_upper``.

    Raises
    ------
    ValueError
        If ``error_bars`` is not ``"ci"`` or ``"se"``, if ``ci_level`` is not
        strictly between 0 and 1, or if the model's coefficients are not
        hazard ratios (``hazard_ratio_valid`` is False).
    """
    if error_bars not in ("ci", "se"):
        raise ValueError(
            f"error_bars must be 'ci' or 'se', got {error_bars!r}."
        )
    if not 0 < ci_level < 1:
        raise ValueError(
            f"ci_level must be between 0 and 1 (exclusive), got {ci_level}. "
            f"Common values: 0.90, 0.95, 0.99."
        )

    # Gate before coefficient extraction so the message is about HR validity,
    # not "does not have coefficients".
    from mlsurv.viz._data import is_hazard_ratio_valid
    if not is_hazard_ratio_valid(model):
        raise ValueError(HR_NOT_VALID_MSG.format(name=model.__class__.__name__))

    coef, feature_names = _extract_coefficients(model, feature_names)
    coef_values = coef.values if hasattr(coef, "values") else np.array(coef)

    df = pd.DataFrame({
        "feature": feature_names,
        "hr": _safe_exp(coef_values),
    })

    _compute_error_bounds(df, model, coef_values, ci_level, error_bars)
    return df


def _error_bar_axis_label(has_ci: bool, error_bars: str, ci_level: float) -> str:
    """Build the x-axis label for the HR forest plot."""
    if not has_ci:
        return "Hazard Ratio"
    if error_bars == "se":
        return "Hazard Ratio (±SE)"
    return f"Hazard Ratio ({ci_level * 100:g}% CI)"


def _error_bar_band_label(has_ci, error_bars, ci_level):
    """Band string for the subtitle, e.g. '95% CI' / '±SE'; None when no band."""
    if not has_ci:
        return None
    if error_bars == "se":
        return "±SE"
    return f"{ci_level * 100:g}% CI"


def _hr_error_x(subset, hr_col, ci_lower_col, ci_upper_col, has_ci, color):
    """Build the ``error_x`` dict for a hazard-ratio forest-plot trace."""
    if not has_ci:
        return dict(visible=False)
    whisker_color = hex_to_rgba(color, 0.5)
    return asymmetric_error_bar(
        (subset[ci_upper_col] - subset[hr_col]).values,
        (subset[hr_col] - subset[ci_lower_col]).values,
        color=whisker_color,
    )


def _build_hr_hover_ci(has_ci, error_bars):
    """Build the hover CI template string for HR plots."""
    if not has_ci:
        return "CI: not available"
    if error_bars == "se":
        return "±SE: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
    return f"CI: [%{{customdata[0]:.2f}}, %{{customdata[1]:.2f}}]"


def _hr_data_range(df, hr_col="hr", ci_lower_col="ci_lower",
                   ci_upper_col="ci_upper"):
    """Extract min/max hazard ratio from *df*, including CI bounds if present."""
    hr_valid = df[hr_col].dropna()
    if hr_valid.empty:
        return 0.25, 4.0  # safe default for empty / all-NaN data
    lo = hr_valid.min()
    hi = hr_valid.max()
    if ci_lower_col in df.columns:
        ci_lo = df[ci_lower_col].dropna()
        if not ci_lo.empty:
            lo = min(lo, ci_lo.min())
    if ci_upper_col in df.columns:
        ci_hi = df[ci_upper_col].dropna()
        if not ci_hi.empty:
            hi = max(hi, ci_hi.max())
    # Safety: clamp to positive and ensure lo < hi
    lo = max(lo, 1e-6)
    hi = max(hi, lo * 2)
    return float(lo), float(hi)


def _hr_data_range_multi(hr_data, top_features):
    """Extract min/max HR across multiple model DataFrames.

    *hr_data* is ``{model_name: DataFrame}`` with ``feature``, ``hr``,
    ``ci_lower``, ``ci_upper`` columns.
    """
    lo, hi = float("inf"), 0.0
    for hr_df in hr_data.values():
        sub = hr_df[hr_df["feature"].isin(top_features)]
        if sub.empty:
            continue
        sub_lo, sub_hi = _hr_data_range(sub)
        lo = min(lo, sub_lo)
        hi = max(hi, sub_hi)
    if lo == float("inf"):
        warnings.warn(
            "No matching features found across models; using default "
            "tick range [0.25, 4.0].",
            InternalWarning,
            stacklevel=2,
        )
        lo, hi = 0.25, 4.0
    return lo, hi


def _format_hr_tick(val):
    """Format a hazard ratio tick value as a clean string.

    Examples: 0.1 → '0.1', 0.5 → '0.5', 1 → '1', 2 → '2', 10 → '10'.
    """
    if val == int(val):
        return str(int(val))
    # Use up to 3 decimal places, strip trailing zeros
    return f"{val:.3f}".rstrip("0").rstrip(".")


def _compute_hr_ticks(hr_min, hr_max):
    """Generate nice 1-2-5 log-scale tick values spanning *hr_min* to *hr_max*.

    Always includes 1.0 (the null-effect reference line).  The returned ticks
    bracket the data range: ``ticks[0] <= hr_min`` and ``ticks[-1] >= hr_max``.
    Returns a sorted list of tick values in linear (not log) space.
    """
    low_decade = math.floor(math.log10(hr_min)) - 1
    high_decade = math.ceil(math.log10(hr_max)) + 1

    candidates = sorted({float(mult * 10 ** exp)
                         for exp in range(low_decade, high_decade + 1)
                         for mult in (1, 2, 5)} | {1.0})

    # Tightest bracket: last candidate <= hr_min, first candidate >= hr_max
    lo_idx = max(i for i, v in enumerate(candidates) if v <= hr_min)
    hi_idx = min(i for i, v in enumerate(candidates) if v >= hr_max)
    result = candidates[lo_idx:hi_idx + 1]

    # Ensure at least 3 ticks
    if len(result) < 3:
        result = sorted({hr_min, 1.0, hr_max, *result})

    return result


def _add_hr_category_traces(fig, df, hr_col, ci_lower_col, ci_upper_col,
                            feature_col, has_ci, hover_ci, row=None, col=None):
    """Add category-coloured traces (Higher/N.S./Lower risk) to HR forest plot."""
    category_style = [
        ("Higher risk", "#d62728"),
        ("N.S.", "#999999"),
        ("Lower risk", "#1f77b4"),
    ]
    for label, color in category_style:
        subset = df[df["category"] == label]
        if subset.empty:
            continue
        error_x = _hr_error_x(subset, hr_col, ci_lower_col, ci_upper_col, has_ci, color)
        trace = go.Scatter(
            x=subset[hr_col], y=subset[feature_col],
            mode="markers",
            marker=dict(size=10, color=color),
            name=label, legendgroup=label,
            error_x=error_x,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "HR: %{x:.2f}<br>"
                f"{hover_ci}<extra></extra>"
            ),
            customdata=np.column_stack([subset[ci_lower_col], subset[ci_upper_col]]),
        )
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)


def _add_hr_risk_annotations(fig, ann_y, d, xref="x"):
    """Add 'Lower risk / Higher risk' annotations for HR forest plot.

    Uses domain-relative positioning so annotations stay at the 25% and 75%
    marks of the axis regardless of data range or asymmetry.
    """
    domain_ref = f"{xref} domain"
    fig.add_annotation(
        x=0.25, y=ann_y,
        xref=domain_ref, yref="paper", xanchor="center",
        text="\u2190 Lower risk",
        showarrow=False,
        font=dict(size=d["font_size_annotation"], color="gray"),
    )
    fig.add_annotation(
        x=0.75, y=ann_y,
        xref=domain_ref, yref="paper", xanchor="center",
        text="Higher risk \u2192",
        showarrow=False,
        font=dict(size=d["font_size_annotation"], color="gray"),
    )


def _create_hr_figure(title, feature_order, feat_to_grp, grp_to_color,
                      display_labels, full_labels, height, width,
                      subtitle=None):
    """Create HR figure with optional group-color strip column.

    Returns ``(fig, main_row, main_col, has_groups)``.
    """
    has_groups = bool(feat_to_grp and grp_to_color)
    if has_groups:
        from plotly.subplots import make_subplots
        strip_width = 0.03
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[strip_width, 1 - strip_width],
            horizontal_spacing=0.02,
        )
        fig.update_layout(
            template="mlsurv_minimal",
            title=dict(text=_importance_full_title(f"<b>{title}</b>", subtitle)),
            width=width, height=height,
        )
        fig.add_trace(
            _make_group_strip_trace(
                feature_order, feat_to_grp, grp_to_color,
                orientation="vertical",
            ),
            row=1, col=1,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_yaxes(
            showgrid=False, showticklabels=True, tickmode="array",
            categoryorder="array", categoryarray=full_labels,
            tickvals=full_labels, ticktext=display_labels,
            row=1, col=1,
        )
        fig.add_vline(x=1, line_dash="dash", line_color="gray", row=1, col=2)
        return fig, 1, 2, True

    fig = create_figure(title=_importance_full_title(title, subtitle),
                        width=width, height=height)
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    return fig, None, None, False


def _configure_hr_axes(fig, has_groups, has_ci, error_bars, ci_level,
                       display_labels, full_labels, grp_to_color,
                       main_row, main_col, hr_tickvals=None):
    """Configure axes and group legend for HR forest plot.

    When *has_groups* is True, updates subplot axes and adds group legend
    traces; otherwise updates the flat layout.

    *hr_tickvals* overrides the default x-axis ticks.  When ``None``, falls
    back to a hardcoded ``[0.25, 0.5, 1, 2, 4]`` sequence.
    """
    if hr_tickvals is None:
        hr_tickvals = [0.25, 0.5, 1, 2, 4]
    xaxis_kw = dict(
        title=dict(
            text=_error_bar_axis_label(has_ci, error_bars, ci_level),
            standoff=12,
        ),
        type="log",
        tickvals=hr_tickvals,
        ticktext=[_format_hr_tick(v) for v in hr_tickvals],
    )
    yaxis_kw = dict(
        title="",
        categoryorder="array", categoryarray=full_labels,
        tickmode="array", tickvals=full_labels, ticktext=display_labels,
    )
    if has_groups:
        fig.update_xaxes(xaxis_kw, row=1, col=2)
        fig.update_yaxes(yaxis_kw, showticklabels=False, row=1, col=2)
        _add_group_legend_traces(fig, grp_to_color, row=main_row, col=main_col)
    else:
        fig.update_layout(xaxis=xaxis_kw, yaxis=yaxis_kw)


def plot_hazard_ratios(
    df: pd.DataFrame,
    hr_col: str = "hr",
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    feature_col: str = "feature",
    n_features: int = 20,
    title: str = "Multivariate CoxPH — Hazard Ratios",
    ci_level: float = 0.95,
    error_bars: str = "ci",
    width: Optional[int] = None,
    height: Optional[int] = None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
    sort_by_group: bool = False,
) -> go.Figure:
    """Forest plot showing hazard ratios with error bars.

    Parameters
    ----------
    df : DataFrame
        Must contain ``hr_col``, ``ci_lower_col``, ``ci_upper_col``,
        and ``feature_col``.

        **DataFrame schema** (default column names)::

            feature   hr      ci_lower   ci_upper
            str       float   float      float
    hr_col : str
        Column with hazard ratio values.
    ci_lower_col : str
        Column with lower bound.
    ci_upper_col : str
        Column with upper bound.
    feature_col : str
        Column with feature names.
    n_features : int
        Maximum features to display.
    title : str
        Plot title.
    ci_level : float
        Confidence level as a fraction in (0, 1), used for axis label.
        Only used when ``error_bars="ci"``.
    error_bars : str, default ``"ci"``
        Type of error bars: ``"ci"`` for confidence intervals,
        ``"se"`` for ±SE.
    width : int, optional
        Override default figure width (pixels).
    height : int, optional
        Override default figure height (pixels).
    feature_groups : dict, list, Series, or None
        Feature-to-group mapping. See :func:`resolve_feature_group_colors`.
    group_colors : dict, optional
        Explicit ``{group: color}`` overrides.
    sort_by_group : bool, default=False
        When True and ``feature_groups`` is set, order features by group
        (primary) and by hazard-ratio magnitude within each group (secondary),
        with separators drawn between groups.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``df`` is missing any of the required columns, if ``error_bars``
        is not ``"ci"`` or ``"se"``, or if ``ci_level`` is not strictly
        between 0 and 1.
    """
    validate_columns(df, [feature_col, hr_col, ci_lower_col, ci_upper_col], "plot_hazard_ratios")
    if error_bars not in ("ci", "se"):
        raise ValueError(
            f"error_bars must be 'ci' or 'se', got {error_bars!r}."
        )
    if not 0 < ci_level < 1:
        raise ValueError(
            f"ci_level must be between 0 and 1 (exclusive), got {ci_level}. "
            f"Common values: 0.90, 0.95, 0.99."
        )
    df = df.copy()
    df["abs_log_hr"] = np.abs(np.log(df[hr_col]))
    df = df.nlargest(n_features, "abs_log_hr")
    df = df.sort_values(hr_col, ascending=True)

    # Detect whether real error bounds are available.
    if "ci_available" in df.columns:
        has_ci = df["ci_available"].any()
    else:
        # Legacy fallback for DataFrames not produced by compute_hazard_ratios
        ci_valid = df[ci_lower_col].notna() & df[ci_upper_col].notna()
        has_ci = ci_valid.any() and not (
            df.loc[ci_valid, ci_lower_col] == df.loc[ci_valid, hr_col]
        ).all()

    # Resolve feature groups
    all_features = df[feature_col].unique().tolist()
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, all_features, group_colors,
    )

    # Classify significance
    df["category"] = "N.S."
    if has_ci:
        df.loc[df[ci_lower_col] > 1, "category"] = "Higher risk"
        df.loc[df[ci_upper_col] < 1, "category"] = "Lower risk"

    # Name the error-bar band in a subtitle (in addition to the x-axis label),
    # only when error bars are actually drawn (has_ci).
    subtitle = _error_bar_band_label(has_ci, error_bars, ci_level)

    d = MANUSCRIPT_DEFAULTS
    margin_t = d["margin_t"] + 20 + (20 if subtitle else 0)
    margin_b = 70
    h = adaptive_size(len(df), per_item=30, base=120, min_size=d["height_small"])
    plot_h = h - margin_t - margin_b

    ann_y = 1.0 + 10 / plot_h
    legend_y = -(50 / plot_h)

    hover_ci = _build_hr_hover_ci(has_ci, error_bars)

    feature_order = df[feature_col].tolist()  # df already sorted by hr_col asc

    # sort_by_group: reorder features by group, then by HR within group
    if sort_by_group and feat_to_grp and grp_to_color:
        abs_log_hr = df[hr_col].apply(
            lambda x: abs(np.log(x)) if x > 0 else 0,
        )
        metric_vals = dict(zip(df[feature_col], abs_log_hr))
        feature_order = _sort_features_by_group(
            feature_order, feat_to_grp, grp_to_color, metric_values=metric_vals,
            reverse=True,
        )

    display_labels, full_labels = truncate_labels(feature_order)
    margin_l = margin_for_labels(display_labels)

    fig, main_row, main_col, has_groups = _create_hr_figure(
        title, feature_order, feat_to_grp, grp_to_color,
        display_labels, full_labels, h, d["width_single"], subtitle=subtitle,
    )

    # Always use risk-category coloring for markers (Higher/N.S./Lower);
    # group membership is shown via the color strip alongside.
    _add_hr_category_traces(
        fig, df, hr_col, ci_lower_col, ci_upper_col, feature_col,
        has_ci, hover_ci, row=main_row, col=main_col,
    )

    # Compute dynamic ticks from the actual data range
    hr_min, hr_max = _hr_data_range(df, hr_col, ci_lower_col, ci_upper_col)
    ticks = _compute_hr_ticks(hr_min, hr_max)

    _configure_hr_axes(
        fig, has_groups, has_ci, error_bars, ci_level,
        display_labels, full_labels, grp_to_color,
        main_row, main_col, hr_tickvals=ticks,
    )

    if has_groups:
        fig.update_layout(
            margin=dict(l=margin_l, t=margin_t, b=margin_b),
            legend=dict(
                x=1.02, xanchor="left", y=0.5, yanchor="middle",
            ),
        )
    else:
        fig.update_layout(
            margin=dict(l=margin_l, t=margin_t, b=margin_b),
            legend=dict(
                orientation="h", yanchor="top", y=legend_y,
                xanchor="center", x=0.5, itemwidth=30,
            ),
        )

    fig.update_traces(
        marker=dict(size=round(d["marker_size_main"] * 1.5)),
        selector=dict(type="scatter"),
    )
    _add_hr_risk_annotations(fig, ann_y, d, xref="x2" if has_groups else "x")

    if sort_by_group and feat_to_grp and grp_to_color:
        _add_group_separator_shapes(fig, feature_order, feat_to_grp, axis="y")

    apply_size_overrides(fig, width, height)
    return fig


def _safe_extract(models, extractor, label, stacklevel=2):
    """Call *extractor(model_name, model)* for each model, skipping failures.

    Returns ``{model_name: result}`` for models that succeeded.
    """
    results = {}
    for model_name, model in models.items():
        try:
            result = extractor(model_name, model)
        except ValueError as exc:
            warnings.warn(
                f"Model '{model_name}' could not be included in {label}: "
                f"{exc}",
                InternalWarning,
                stacklevel=stacklevel,
            )
            continue
        except AttributeError as exc:
            warnings.warn(
                f"Model '{model_name}' raised an unexpected AttributeError "
                f"during {label}: {exc}. This may indicate a bug in the "
                f"model implementation.",
                InternalWarning,
                stacklevel=stacklevel,
            )
            continue
        if result is None:
            warnings.warn(
                f"Model '{model_name}' returned no data and will be "
                f"excluded from {label}.",
                InternalWarning,
                stacklevel=stacklevel,
            )
            continue
        results[model_name] = result
    return results


def _collect_hr_data(models):
    """Compute HRs for each model (or accept pre-computed DataFrames).

    Returns ``(hr_data, all_features_set)``.
    """
    def _extract(_name, model_or_df):
        if isinstance(model_or_df, pd.DataFrame):
            return model_or_df
        return compute_hazard_ratios(model_or_df)

    hr_data = _safe_extract(models, _extract, "HR comparison", stacklevel=3)
    all_features_set: set = set()
    for hr_df in hr_data.values():
        all_features_set.update(hr_df["feature"].tolist())
    return hr_data, all_features_set


def _rank_features_by_abs_log_hr(hr_data, all_features_set):
    """Rank features by average |log(HR)| across models.

    Returns a dict mapping feature name to average absolute log HR.
    """
    avg_abs = {}
    for feat in all_features_set:
        vals = []
        for mdf in hr_data.values():
            if feat not in mdf["feature"].values:
                continue
            hr_val = mdf.loc[mdf["feature"] == feat, "hr"].iloc[0]
            if hr_val > 0 and np.isfinite(hr_val):
                vals.append(np.abs(np.log(hr_val)))
        avg_abs[feat] = np.mean(vals) if vals else 0
    return avg_abs


def _add_hr_comparison_traces(fig, hr_data, top_features, d, palette,
                              row=None, col=None):
    """Add per-model scatter traces to HR comparison figure."""
    y_positions = {feat: i for i, feat in enumerate(top_features[::-1])}
    n_models = len(hr_data)
    offset_step = 0.25

    for j, (model_name, hr_df) in enumerate(hr_data.items()):
        color = get_model_color(model_name, palette=palette)
        offset = (j - (n_models - 1) / 2) * offset_step
        plot_df = hr_df[hr_df["feature"].isin(top_features)].copy()
        y_vals = [y_positions[feat] + offset for feat in plot_df["feature"]]

        # Check if real CI bounds are available
        if "ci_available" in plot_df.columns:
            has_ci = plot_df["ci_available"].any()
        else:
            ci_valid = (
                plot_df["ci_lower"].notna()
                & plot_df["ci_upper"].notna()
                & (plot_df["ci_lower"] > 0)
                & (plot_df["ci_upper"] > 0)
                & ~(plot_df["ci_lower"] == plot_df["hr"])
            )
            has_ci = ci_valid.any()

        error_x = None
        if has_ci:
            error_x = asymmetric_error_bar(
                np.log10(plot_df["ci_upper"].values / plot_df["hr"].values),
                np.log10(plot_df["hr"].values / plot_df["ci_lower"].values),
                color=color, thickness=1.5,
            )

        hover_ci = (
            "CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
            if has_ci else "CI: not available"
        )

        trace = go.Scatter(
            x=plot_df["hr"], y=y_vals, mode="markers",
            marker=dict(size=round(d["marker_size_main"] * 1.5), color=color),
            name=model_name,
            error_x=error_x,
            customdata=np.column_stack([
                plot_df["ci_lower"].values, plot_df["ci_upper"].values,
            ]),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "%{text}<br>"
                f"HR: %{{x:.2f}}<br>{hover_ci}<extra></extra>"
            ),
            text=plot_df["feature"],
        )
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)


def plot_hazard_ratios_comparison(
    models: dict,
    feature_names: Optional[List[str]] = None,
    n_features: int = 15,
    title: str = "Hazard Ratios by Model",
    width: Optional[int] = None,
    height: Optional[int] = None,
    palette=None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
    sort_by_group: bool = False,
) -> go.Figure:
    """Compare hazard ratios across multiple sources.

    Parameters
    ----------
    models : dict
        ``{name: fitted_model_or_DataFrame}`` dictionary.  Values can be
        fitted Cox models (``compute_hazard_ratios`` is called) **or**
        pre-computed DataFrames with ``feature``, ``hr``, ``ci_lower``,
        ``ci_upper`` columns.
    feature_names : list, optional
        Feature names to display.
    n_features : int
        Maximum features per model.
    title : str
        Plot title.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    palette : dict, optional
        ``{model_name: hex_color}`` overrides.
    feature_groups : dict, list, or Series, optional
        Feature-to-group mapping.  When provided (and groups resolve), a
        colored group strip and legend are drawn.  To additionally reorder
        features so groups are contiguous and add separator lines between
        them, set ``sort_by_group=True``.
    group_colors : dict, optional
        ``{group_name: hex_color}`` overrides for group colours.
    sort_by_group : bool, default=False
        When True (and ``feature_groups``/``group_colors`` resolve), reorder
        features so groups are contiguous and draw separator lines between
        groups. The colored group strip and legend appear whenever feature
        groups resolve, regardless of this flag.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If none of the supplied models yield hazard-ratio coefficients.

    Warns
    -----
    MethodologicalWarning
        For each non-DataFrame model whose coefficients are not hazard
        ratios; such models are dropped from the plot.
    """
    from mlsurv.viz._data import is_hazard_ratio_valid

    # Drop fitted models whose exp(coef) is not a hazard ratio, with a clear
    # warning. DataFrame values are kept as-is (a precomputed frame can't be
    # inspected). The compute_hazard_ratios gate still backstops any leak.
    filtered = {}
    for name, value in models.items():
        if not isinstance(value, pd.DataFrame) and not is_hazard_ratio_valid(value):
            warnings.warn(
                DROP_HR_MSG.format(name=name),
                MethodologicalWarning,
                stacklevel=2,
            )
            continue
        filtered[name] = value

    hr_data, all_features_set = _collect_hr_data(filtered)
    if not hr_data:
        raise ValueError("No models with coefficients found")

    avg_abs = _rank_features_by_abs_log_hr(hr_data, all_features_set)
    top_features = sorted(avg_abs, key=avg_abs.get, reverse=True)[:n_features]

    # Resolve feature groups for ordering and visual separators
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, top_features, group_colors,
    )

    if sort_by_group and feat_to_grp and grp_to_color:
        # Re-order top features by group, preserving within-group ranking
        top_features = _sort_features_by_group(
            top_features, feat_to_grp, grp_to_color, metric_values=avg_abs,
        )

    d = MANUSCRIPT_DEFAULTS
    n_actual = len(top_features)
    h = adaptive_size(n_actual, per_item=28, base=80, min_size=d["height_medium"])

    display_labels, full_labels = truncate_labels(top_features[::-1])
    margin_l = margin_for_labels(display_labels)

    has_groups = bool(feat_to_grp and grp_to_color)
    if has_groups:
        from plotly.subplots import make_subplots
        strip_width = 0.03
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[strip_width, 1 - strip_width],
            horizontal_spacing=0.02,
        )
        fig.update_layout(
            template="mlsurv_minimal",
            title=dict(text=f"<b>{title}</b>"),
            width=d["width_double"], height=h,
        )
        fig.add_trace(
            _make_group_strip_trace(
                top_features[::-1], feat_to_grp, grp_to_color,
                orientation="vertical",
            ),
            row=1, col=1,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_yaxes(
            showgrid=False, showticklabels=True, tickmode="array",
            tickvals=list(range(n_actual)), ticktext=display_labels,
            row=1, col=1,
        )
        main_row, main_col = 1, 2
    else:
        fig = create_figure(title=title, width="medium", height=h)
        main_row, main_col = None, None

    if has_groups:
        fig.add_vline(x=1, line_dash="dash", line_color="gray", row=1, col=2)
    else:
        fig.add_vline(x=1, line_dash="dash", line_color="gray")

    _add_hr_comparison_traces(fig, hr_data, top_features, d, palette,
                              row=main_row, col=main_col)

    # Compute dynamic ticks from the actual data range
    hr_min, hr_max = _hr_data_range_multi(hr_data, top_features)
    ticks = _compute_hr_ticks(hr_min, hr_max)
    ticktext = [_format_hr_tick(v) for v in ticks]

    if has_groups:
        # Separators only make sense once features are reordered group-contiguous,
        # so gate them on sort_by_group; the strip/legend stay group-gated.
        if sort_by_group:
            _add_group_separator_shapes(
                fig, top_features[::-1], feat_to_grp, axis="y",
            )
        _add_group_legend_traces(fig, grp_to_color, row=main_row, col=main_col)
        fig.update_xaxes(
            title="Hazard Ratio", type="log",
            tickvals=ticks, ticktext=ticktext,
            row=1, col=2,
        )
        fig.update_yaxes(
            showticklabels=False,
            tickmode="array",
            tickvals=list(range(n_actual)),
            ticktext=display_labels,
            row=1, col=2,
        )
        fig.update_layout(
            legend=dict(
                x=1.02, xanchor="left", y=0.5, yanchor="middle",
            ),
        )
    else:
        fig.update_layout(
            xaxis_title="Hazard Ratio", xaxis_type="log",
            xaxis=dict(tickvals=ticks, ticktext=ticktext),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(n_actual)),
                ticktext=display_labels,
            ),
            yaxis_title="",
            legend=dict(title=dict(text="Model")),
        )

    fig.update_layout(margin=dict(l=margin_l, t=d["margin_t"] + 20))
    margin_t = d["margin_t"] + 20
    margin_b = fig.layout.margin.b or 50
    plot_h = h - margin_t - margin_b
    ann_y = 1.0 + 10 / plot_h
    _add_hr_risk_annotations(fig, ann_y, d, xref="x2" if has_groups else "x")

    apply_size_overrides(fig, width, height)
    return fig
