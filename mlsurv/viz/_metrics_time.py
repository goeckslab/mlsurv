"""Time-dependent metric renderers: line plots, timepoint forest, multi-panel.

Contains ``plot_metric_time()`` public dispatcher and its internal renderers.
"""

from math import ceil
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from mlsurv.constants import LOWER_IS_BETTER, get_metric_display_name
from mlsurv.viz._palettes import (
    MANUSCRIPT_DEFAULTS,
    get_model_color,
    get_model_display_name as _model_display,
)
from mlsurv.viz._theme import (
    add_ci_ribbon,
    add_reference_hline,
    apply_size_overrides,
    create_figure,
)

from mlsurv.viz._metrics_helpers import (
    _is_lower_better,
    _metric_reference,
    _resolve_model_style,
    normalize_metric_param,
)
from mlsurv.viz._metrics_renderers import _plot_metric_forest

from typing import TYPE_CHECKING, Set
if TYPE_CHECKING:
    from mlsurv.viz._context import SubpopulationAxis


def _eval_time_ticks(time_series: pd.Series) -> dict:
    """Build x-axis tick kwargs for eval-time columns.

    Uses explicit ticks when there are 20 or fewer unique times, otherwise
    falls back to evenly spaced ticks covering the range.  Tick labels are
    rounded to avoid overly long decimal strings from percentile-based times.
    """
    all_times = sorted(time_series.unique())
    if len(all_times) <= 20:
        # Round labels: integers if range > 10, else 1 decimal place
        span = all_times[-1] - all_times[0] if len(all_times) >= 2 else 0
        ndigits = 0 if span > 10 else 1
        tick_text = [str(int(round(t))) if ndigits == 0
                     else str(round(t, ndigits)) for t in all_times]
        return dict(tickvals=all_times, ticktext=tick_text, tickmode="array")
    return dict(dtick=(all_times[-1] - all_times[0]) / 10)


# =============================================================================
# Internal style renderers — time-dependent line
# =============================================================================

def _resolve_line_columns(df, metric, value_col, std_col, std_df):
    """Resolve value and std column names for line plots."""
    if value_col is None:
        if metric and metric in LOWER_IS_BETTER:
            value_col = "brier_score"
        elif metric in ("uno", "ipcw"):
            value_col = "auc"
        else:
            value_col = "cindex" if "cindex" in df.columns else "auc"
    if std_col is None and std_df is not None:
        std_col = f"{value_col}_std"
    return value_col, std_col


def _data_driven_y_min(df, std_df, value_col, std_col, model_col, time_col):
    """Compute y-axis lower bound that accommodates all data and CI ribbons.

    NaN-safe: drops NaN values before computing the minimum. Falls back to
    0.4 (the legacy default) when all values are NaN.
    """
    vals = df[value_col].dropna()
    if vals.empty:
        return 0.4
    y_min = float(vals.min())
    if std_df is not None and std_col and std_col in std_df.columns:
        merged = df.merge(
            std_df[[model_col, time_col, std_col]],
            on=[model_col, time_col], how='outer',
        )
        if len(merged) > 0:
            lower = (merged[value_col] - merged[std_col]).dropna()
            if not lower.empty:
                y_min = min(y_min, float(lower.min()))
    # Round down to nearest 0.05 with padding, floor at 0
    return max(0.0, np.floor(y_min * 20) / 20 - 0.05)


def _data_driven_y_max(df, std_df, value_col, std_col, model_col, time_col):
    """Compute y-axis upper bound that accommodates all data and CI ribbons.

    Mirror of ``_data_driven_y_min`` for lower-is-better metrics (e.g. Brier).
    NaN-safe: falls back to 0.35 (the legacy default) when all values are NaN.
    """
    vals = df[value_col].dropna()
    if vals.empty:
        return 0.35
    y_max = float(vals.max())
    if std_df is not None and std_col and std_col in std_df.columns:
        merged = df.merge(
            std_df[[model_col, time_col, std_col]],
            on=[model_col, time_col], how='outer',
        )
        if len(merged) > 0:
            upper = (merged[value_col] + merged[std_col]).dropna()
            if not upper.empty:
                y_max = max(y_max, float(upper.max()))
    # Round up to nearest 0.05 with padding, cap at 1.0
    return min(1.0, np.ceil(y_max * 20) / 20 + 0.05)


def _add_line_model_trace(fig, model_df, model, std_df, time_col, value_col,
                          std_col, model_col, show_ci, metric_display, color,
                          row=None, col=None, showlegend=True):
    """Add line + optional SD ribbon for one model."""
    display_name = _model_display(model)
    add_kw = {}
    if row is not None and col is not None:
        add_kw = dict(row=row, col=col)

    model_std = None
    if std_df is not None:
        candidate = std_df[std_df[model_col] == model].sort_values(time_col)
        if len(candidate) > 0 and std_col in candidate.columns:
            model_std = candidate

    hover = f"<b>{display_name}</b><br>Time: %{{x}}<br>{metric_display}: %{{y:.3f}}"
    customdata = None
    if model_std is not None:
        hover += "<br>\u00b1 %{customdata:.3f}"
        customdata = model_std[std_col].values
    hover += "<extra></extra>"

    # Draw line FIRST, then ribbon on top — the semi-transparent ribbon
    # covers the line's anti-aliased edges, preventing white fringe artifacts.
    d = MANUSCRIPT_DEFAULTS
    fig.add_trace(go.Scatter(
        x=model_df[time_col], y=model_df[value_col],
        mode='lines+markers', name=display_name,
        line=dict(color=color, width=d["line_width_main"]),
        marker=dict(size=d["marker_size_main"], line=dict(width=0)),
        hovertemplate=hover, customdata=customdata, legendgroup=model,
        showlegend=showlegend,
    ), **add_kw)

    if model_std is not None and show_ci:
        # Outer merge (not inner) so a point present on the curve but missing a
        # CI is NOT silently bridged over: the band breaks at that time (NaN
        # upper/lower -> tonexty gap) instead of drawing a straight ribbon
        # across it. Sort by time so the ribbon polygon stays ordered.
        merged = model_df.merge(
            model_std[[time_col, std_col]], on=time_col, how='outer',
        ).sort_values(time_col)
        if len(merged) > 0:
            upper = merged[value_col].values + merged[std_col].values
            lower = merged[value_col].values - merged[std_col].values
            add_ci_ribbon(
                fig, merged[time_col].values, upper, lower, color,
                name=f'{display_name} \u00b11 SD', legendgroup=model,
                **add_kw,
            )


def _plot_metric_line(
    df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    *,
    metric: Optional[str],
    time_col: str,
    value_col: str,
    std_col: Optional[str],
    model_col: str,
    show_ci: bool,
    title: str,
    time_unit: Optional[str],
    palette=None,
) -> go.Figure:
    """Line plot of time-dependent metric per model."""
    metric_display = get_metric_display_name(metric) if metric else "Value"
    value_col, std_col = _resolve_line_columns(df, metric, value_col, std_col, std_df)
    ref_val, ref_label = _metric_reference(metric) if metric else (None, None)

    fig = create_figure(title=title, width="single", height="small")
    if ref_val is not None:
        add_reference_hline(fig, y=ref_val, label=ref_label)

    for model in df[model_col].unique():
        model_df = df[df[model_col] == model].sort_values(time_col)
        color = get_model_color(model, palette=palette)
        _add_line_model_trace(fig, model_df, model, std_df, time_col,
                              value_col, std_col, model_col, show_ci,
                              metric_display, color)

    xaxis_kw = _eval_time_ticks(df[time_col])

    time_label = f'Time ({time_unit})' if time_unit else 'Time'
    if _is_lower_better(metric):
        y_max = _data_driven_y_max(
            df, std_df, value_col, std_col, model_col, time_col,
        )
        y_range = [0, y_max]
    else:
        y_min = _data_driven_y_min(
            df, std_df, value_col, std_col, model_col, time_col,
        )
        y_range = [y_min, 1.0]

    fig.update_layout(
        xaxis_title=time_label, xaxis=xaxis_kw,
        yaxis_title=f'{metric_display}(t)', yaxis=dict(range=y_range),
        hovermode='x unified',
        legend=dict(title=dict(text=''), orientation="h",
                    yanchor="top", y=-0.35, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    return fig


def _plot_metric_timepoint_forest(
    df: pd.DataFrame,
    *,
    eval_time: Optional[float],
    metric: Optional[str],
    time_col: str,
    value_col: str,
    model_col: str,
    std_col: Optional[str],
    n_bootstraps: Optional[int],
    ci_level: float,
    title: Optional[str],
    orientation: str,
    palette=None,
) -> go.Figure:
    """Dot-and-whisker showing metric at a single timepoint per model."""
    # Select single timepoint
    if eval_time is None:
        eval_time = sorted(df[time_col].unique())[0]
    df = df[df[time_col] == eval_time]

    metric_display = get_metric_display_name(metric) if metric else "Value"
    if title is None:
        title = f"{metric_display} Forest Plot (t = {eval_time})"

    z = norm.ppf(1 - (1 - ci_level) / 2)
    forest_df = pd.DataFrame({
        'model': df[model_col].values,
        'point': df[value_col].values,
    })

    if std_col and std_col in df.columns:
        forest_df['ci_lower'] = df[value_col].values - z * df[std_col].values
        forest_df['ci_upper'] = df[value_col].values + z * df[std_col].values
    else:
        forest_df['ci_lower'] = forest_df['point']
        forest_df['ci_upper'] = forest_df['point']

    fig = _plot_metric_forest(
        forest_df,
        model_col="model",
        point_col="point",
        ci_lower_col="ci_lower",
        ci_upper_col="ci_upper",
        metric=metric,
        title=title,
        sort="alpha",
        ci_level=ci_level,
        show_labels=True,
        show_reference_line=True,
        orientation=orientation,
        palette=palette,
    )

    axis_label = f"{metric_display} at t = {eval_time}"
    if std_col:
        axis_label += f" ({ci_level * 100:g}% CI)"
    fig.update_layout(xaxis_title=axis_label)

    if n_bootstraps is not None:
        fig.add_annotation(
            text=f"Bootstrap iterations: {n_bootstraps}",
            xref="paper", yref="paper",
            x=1.0, y=-0.12,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="right",
        )

    return fig


# =============================================================================
# Internal style renderer — multi-panel time-dependent line
# =============================================================================

def _render_time_panel(fig, df, std_df, metric, models, metric_col, model_col,
                       time_col, value_col, std_col, show_ci, palette,
                       legend_shown, r, c):
    """Render one panel in a multi-panel time-dependent grid."""
    panel_df = df[df[metric_col] == metric]
    panel_std = std_df[std_df[metric_col] == metric] if std_df is not None else None

    metric_display = get_metric_display_name(metric)
    resolved_val, resolved_std = _resolve_line_columns(
        panel_df, metric, value_col, std_col, panel_std,
    )

    for model in models:
        model_df = panel_df[panel_df[model_col] == model].sort_values(time_col)
        if model_df.empty:
            continue
        color = get_model_color(model, palette=palette)
        show = model not in legend_shown
        _add_line_model_trace(
            fig, model_df, model, panel_std, time_col,
            resolved_val, resolved_std, model_col, show_ci,
            metric_display, color, row=r, col=c, showlegend=show,
        )
        legend_shown.add(model)

    ref_val, _ref_label = _metric_reference(metric)
    if ref_val is not None:
        fig.add_hline(
            y=ref_val, line_dash="dash", line_color="gray",
            line_width=1.5, row=r, col=c,
        )

    if _is_lower_better(metric):
        y_max = _data_driven_y_max(
            panel_df, panel_std, resolved_val, resolved_std,
            model_col, time_col,
        )
        y_range = [0, y_max]
    else:
        y_min = _data_driven_y_min(
            panel_df, panel_std, resolved_val, resolved_std,
            model_col, time_col,
        )
        y_range = [y_min, 1.0]
    fig.update_yaxes(range=y_range, row=r, col=c)


def _plot_metric_time_multi_panel(
    df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    metrics: List[str],
    *,
    time_col: str,
    value_col: str,
    std_col: Optional[str],
    model_col: str,
    metric_col: str,
    show_ci: bool,
    title: str,
    time_unit: Optional[str],
    palette=None,
) -> go.Figure:
    """Multi-panel grid of time-dependent metric curves.

    Each panel renders one metric's line traces for all models,
    following the same grid layout pattern as ``_plot_metric_multi_panel``.
    """
    n_metrics = len(metrics)
    n_cols = min(n_metrics, 3)
    n_rows = ceil(n_metrics / n_cols)

    subplot_labels = [f"{get_metric_display_name(m)}(t)" for m in metrics]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_labels,
        shared_xaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.12 if n_rows > 1 else 0.08,
    )

    models = df[model_col].unique().tolist()
    # Track which models have already been added to the legend
    legend_shown = set()

    for panel_idx, metric in enumerate(metrics):
        r = panel_idx // n_cols + 1
        c = panel_idx % n_cols + 1
        _render_time_panel(
            fig, df, std_df, metric, models, metric_col, model_col,
            time_col, value_col, std_col, show_ci, palette,
            legend_shown, r, c,
        )

    xaxis_kw = _eval_time_ticks(df[time_col])

    time_label = f'Time ({time_unit})' if time_unit else 'Time'

    d = MANUSCRIPT_DEFAULTS
    grid_width = int(d["width_double"] * 1.12)
    fig.update_layout(
        template="mlsurv_minimal",
        width=grid_width,
        height=300 * n_rows,
        title=dict(text=f"<b>{title}</b>", y=0.98, yanchor="top"),
        showlegend=True,
        hovermode='x unified',
        legend=dict(title=dict(text=''), orientation="h",
                    yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=d["margin_l"] + 20, t=d["margin_t"] + 20, b=80),
    )

    # Set x-axis label on bottom row
    for c_idx in range(n_cols):
        bottom_panel = (n_rows - 1) * n_cols + c_idx
        if bottom_panel < n_metrics:
            x_key = f"xaxis{bottom_panel + 1}" if bottom_panel > 0 else "xaxis"
            fig.update_layout(**{x_key: dict(title=time_label, **xaxis_kw)})

    return fig


# =============================================================================
# Auto-resolution helper (extracted from plot_metric_time)
# =============================================================================

def _auto_resolve_time_value_col(df, metric, time_col, model_col):
    """Auto-resolve value_col from metric name and available columns.

    Returns the resolved value_col string.
    """
    if metric and metric in LOWER_IS_BETTER:
        return "brier_score"
    if "cindex" in df.columns:
        return "cindex"
    if "auc" in df.columns:
        return "auc"
    # Fallback: first numeric column that isn't time or model
    numeric = df.select_dtypes(include=['float64', 'int64']).columns
    candidates = [c for c in numeric if c not in [time_col, model_col]]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"Could not auto-resolve value column for metric '{metric}'. "
        f"No recognized columns (cindex, auc, brier_score) found. "
        f"Available columns: {sorted(df.columns)}. "
        f"Pass value_col explicitly."
    )


# =============================================================================
# Public API: plot_metric_time()
# =============================================================================

def plot_metric_time(
    df: pd.DataFrame,
    std_df: Optional[pd.DataFrame] = None,
    *,
    style: str = "line",
    metric: Optional[Union[str, List[str]]] = None,
    metric_col: str = "metric_name",
    eval_time: Optional[float] = None,
    time_col: str = "eval_time",
    value_col: Optional[str] = None,
    std_col: Optional[str] = None,
    model_col: str = "model",
    show_ci="auto",
    title: Optional[str] = None,
    time_unit: Optional[str] = None,
    n_bootstraps: Optional[int] = None,
    ci_level: float = 0.95,
    orientation: str = "horizontal",
    width: Optional[int] = None,
    height: Optional[int] = None,
    palette=None,
    return_data: bool = False,
) -> Union[go.Figure, dict]:
    """Unified dispatcher for time-dependent metric visualizations.

    Parameters
    ----------
    df : DataFrame
        Time-dependent metric data with *time_col*, *value_col*, *model_col*.
        For multi-panel mode, must include *metric_col* identifying each metric.
    std_df : DataFrame, optional
        Standard deviations for CI ribbons.
    style : {'line', 'forest'}
        ``'line'``: curves over time.  ``'forest'``: at single timepoint.
    metric : str or list of str, optional
        Metric key(s).  A single name drives labels, reference line, and auto
        value_col for a one-panel plot.  A list draws multi-panel mode (one
        panel per name); when a list is given, *df* must contain a *metric_col*
        column identifying each metric.  This low-level renderer treats any
        list as multi-panel.
    metric_col : str
        Column identifying metrics in multi-panel mode (default ``'metric_name'``).
    eval_time : float, optional
        Required for ``style='forest'``.  Ignored for ``style='line'``.
    time_col : str
        Column with evaluation times.
    value_col : str, optional
        Column with metric values.  Auto-resolved from *metric* if None.
    std_col : str, optional
        Column with standard deviations.
    model_col : str
        Column with model names.
    show_ci : {"auto", True, False}, default="auto"
        Show CI ribbons (line) or whiskers (forest). ``"auto"`` and ``True``
        both draw the band when ``std_df`` is present; ``False`` never draws
        it. (This low-level renderer does not warn; the learner-facing
        ``plot_metric_time`` warns on an explicit ``True`` with no data.)
    title : str, optional
        Figure title.
    time_unit : str, optional
        Unit label for time axis.
    n_bootstraps : int, optional
        Number of bootstrap iterations (annotated on forest).
    ci_level : float
        Confidence interval percentage.
    orientation : str
        Orientation for forest style.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    palette : dict, optional
        ``{model_name: hex_color}`` overrides for per-model colours.
    return_data : bool, default=False
        If True, return ``{"figure": fig, "data": df}`` instead of just the
        figure.

    Returns
    -------
    go.Figure or dict

    Raises
    ------
    ValueError
        If ``metric`` is an empty list, or if ``style`` is not ``"line"`` or
        ``"forest"``.
    TypeError
        If ``metric`` is not a str or list of str.
    """
    single_metric, metrics = normalize_metric_param(
        metric, collapse_singleton=False,
    )

    # A 1-element metric list is single-metric long-format data. Render it
    # through the flat single-panel path below so its dimensions and layout
    # match the scalar metric path — multi-panel grid sizing only applies to
    # 2+ metrics.
    if metrics is not None and len(metrics) == 1:
        single_metric = metrics[0]
        metrics = None
        if value_col is None:
            # long-format value column from the multi-metric builder
            value_col = "value"

    # Multi-panel dispatch: multiple metrics side-by-side
    if metrics is not None:
        if title is None:
            title = "Time-Dependent Metric Comparison"
        # value_col defaults to "value" for multi-panel long-format data
        if value_col is None:
            value_col = "value"
        fig = _plot_metric_time_multi_panel(
            df, std_df, metrics,
            time_col=time_col,
            value_col=value_col,
            std_col=std_col,
            model_col=model_col,
            metric_col=metric_col,
            show_ci=show_ci,
            title=title,
            time_unit=time_unit,
            palette=palette,
        )
        apply_size_overrides(fig, width, height)
        if return_data:
            return {"figure": fig, "data": df}
        return fig

    metric = single_metric
    metric_display = get_metric_display_name(metric) if metric else "Metric"

    # Auto-resolve value_col from metric
    if value_col is None:
        value_col = _auto_resolve_time_value_col(df, metric, time_col, model_col)

    if title is None:
        title = f"Time-Dependent {metric_display}"

    if style == "line":
        fig = _plot_metric_line(
            df, std_df,
            metric=metric,
            time_col=time_col,
            value_col=value_col,
            std_col=std_col,
            model_col=model_col,
            show_ci=show_ci,
            title=title,
            time_unit=time_unit,
            palette=palette,
        )
    elif style == "forest":
        fig = _plot_metric_timepoint_forest(
            df,
            eval_time=eval_time,
            metric=metric,
            time_col=time_col,
            value_col=value_col,
            model_col=model_col,
            std_col=std_col,
            n_bootstraps=n_bootstraps,
            ci_level=ci_level,
            title=title,
            orientation=orientation,
            palette=palette,
        )
    else:
        raise ValueError(
            f"Unknown style '{style}'. Must be one of: line, forest"
        )

    apply_size_overrides(fig, width, height)
    if return_data:
        return {"figure": fig, "data": df}
    return fig


# =============================================================================
# Grouped faceted metric-time plot (subpopulation / stratified)
# =============================================================================


def _grouped_metric_grid_dims(n_sg):
    """Grid layout for a single metric faceted by subpopulation level.

    Single source of truth for the subpopulation single-metric facet grid:
    subpopulations wrap into up to 2 columns, sized so AUC and any non-AUC
    single metric share the same dimensions.

    Parameters
    ----------
    n_sg : int
        Number of subpopulation levels (panels).

    Returns
    -------
    tuple
        ``(n_cols, n_rows, v_spacing, fig_width, fig_height)``.
    """
    n_cols = min(2, n_sg)
    n_rows = (n_sg + n_cols - 1) // n_cols
    v_spacing = min(0.10, 0.4 / max(n_rows, 1))
    fig_width = 900 if n_cols == 2 else 550
    fig_height = max(450, n_rows * 360 + 100)
    return n_cols, n_rows, v_spacing, fig_width, fig_height


#
# Refactored (not byte-identical) from the old
# ``SubpopulationAnalysisSection._build_auc_t_plot`` method: accepts a
# :class:`SubpopulationAxis` instead of reading ``self.learner._state`` /
# ``self._get_reference_names`` / a raw ``SubpopulationResult``, hoists the
# plotly imports to module-level, and drops the
# ``try/except IMPORT_EXCEPTIONS`` wrapper (error handling moved up to the
# section proxy). Layout formulas, panel assembly order, ribbon shading, and
# y-range computation are preserved byte-for-byte so that
# ``tests/test_subpopulation_analysis.py::TestSubpopulationAucTPlot`` and
# ``::TestSubpopulationAucTPlotSizing`` pass against the output unchanged.


def plot_grouped_metric_time(
    axis: "SubpopulationAxis",
    *,
    time_unit: Optional[str] = None,
    ref_names: Optional[Set[str]] = None,
    show_ci="auto",
):
    """Faceted AUC(t) grid: one panel per subpopulation level.

    Parameters
    ----------
    axis : SubpopulationAxis
        Pre-aggregated subpopulation frames. Requires ``axis.auc_t`` to be
        populated; returns ``None`` otherwise.
    time_unit : str, optional
        Display unit for the x-axis (e.g. ``"months"``). No unit shown if
        omitted.
    ref_names : set of str, optional
        Benchmark reference model names (rendered with dashed lines and
        reference palette). Defaults to an empty set.
    show_ci : {"auto", True, False}, default="auto"
        ``"auto"``/``True`` draw the bootstrap ribbon when
        ``axis.auc_t_bootstrap`` is present; ``False`` never draws it.

    Returns
    -------
    plotly.graph_objects.Figure or None
        ``None`` when ``axis.auc_t`` is empty or produces no traces.
    """
    if ref_names is None:
        ref_names = set()

    auc_df = axis.auc_t
    if auc_df is None or auc_df.empty:
        return None

    subpopulations = sorted(auc_df['subpopulation'].unique())
    model_names = sorted(auc_df['model_type'].unique())
    n_sg = len(subpopulations)

    # Bootstrap CI data (if available).
    #
    # If ribbons are missing but you expect them, the cause is
    # almost always upstream, not here: axis.auc_t_bootstrap is
    # populated by _attach_subpopulation_auc_bootstrap in
    # mlsurv/evaluation/_subpopulation.py, called from
    # _compute_and_store_subpopulation_metrics (for
    # learner.evaluate()) and _attach_subpopulation_results
    # (for learner.benchmark()). Both require the bootstrap
    # call to pass subpopulation_labels and eval_times.
    boot_auc_df = axis.auc_t_bootstrap
    has_ci = (
        bool(show_ci)
        and boot_auc_df is not None
        and not boot_auc_df.empty
    )
    if has_ci:
        ci_stats = (
            boot_auc_df.groupby(['subpopulation', 'model_type', 'eval_time'])['auc']
            .agg(['mean', 'std'])
            .reset_index()
        )

    # Grid layout: up to 2 columns, adaptive spacing (shared sizing helper)
    n_cols, n_rows, v_spacing, fig_width, fig_height = _grouped_metric_grid_dims(
        n_sg
    )
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subpopulations,
        horizontal_spacing=0.08,
        vertical_spacing=v_spacing,
    )

    any_ribbon = False
    legend_shown = set()
    ref_idx = 0
    for mname in model_names:
        display_name, color, is_ref, ref_idx = _resolve_model_style(
            mname, ref_names, ref_idx,
        )
        dash = 'dash' if is_ref else 'solid'

        for sg_idx, sg in enumerate(subpopulations):
            row = sg_idx // n_cols + 1
            col = sg_idx % n_cols + 1
            mask = (
                (auc_df['subpopulation'] == sg)
                & (auc_df['model_type'] == mname)
            )
            sub = auc_df[mask].sort_values('eval_time')
            if sub.empty:
                continue
            show = mname not in legend_shown
            fig.add_trace(
                go.Scatter(
                    x=sub['eval_time'],
                    y=sub['auc'],
                    mode='lines',
                    name=display_name,
                    line=dict(color=color, dash=dash, width=2),
                    legendgroup=display_name,
                    showlegend=show,
                ),
                row=row, col=col,
            )
            legend_shown.add(mname)

            # Add ±1 SD ribbon if bootstrap data available
            if has_ci:
                ci_mask = (
                    (ci_stats['subpopulation'] == str(sg))
                    & (ci_stats['model_type'] == mname)
                )
                ci_sub = ci_stats[ci_mask].sort_values('eval_time')
                if not ci_sub.empty:
                    means = ci_sub['mean'].values
                    stds = ci_sub['std'].values
                    if not (np.all(np.isnan(means)) or np.all(np.isnan(stds))):
                        upper = np.clip(means + stds, 0, 1)
                        lower = np.clip(means - stds, 0, 1)
                        add_ci_ribbon(
                            fig, ci_sub['eval_time'].values,
                            upper, lower, color,
                            row=row, col=col,
                            legendgroup=display_name,
                        )
                        any_ribbon = True

    x_label = f"Time ({time_unit})" if time_unit else "Time"
    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(text=(
            f"<b>Subpopulation AUC(t): {axis.group_variable}"
            f"{' (±1 SD)' if any_ribbon else ''}</b>"
        )),
        width=fig_width,
        height=fig_height,
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=1.02,
        ),
        margin=dict(l=60, r=140, t=80, b=50),
    )
    # Consistent rounded x-axis ticks across all panels
    tick_kw = _eval_time_ticks(auc_df['eval_time'])
    fig.update_xaxes(title_text=x_label, **tick_kw)
    # Data-driven y-range that accommodates ribbons (NaN-safe)
    auc_vals = auc_df['auc'].dropna()
    y_min = float(auc_vals.min()) if not auc_vals.empty else 0.4
    if has_ci:
        lower_vals = (ci_stats['mean'] - ci_stats['std']).dropna()
        if not lower_vals.empty:
            y_min = min(y_min, float(lower_vals.min()))
    y_min = max(0.0, np.floor(y_min * 20) / 20 - 0.05)
    fig.update_yaxes(title_text="AUC", range=[y_min, 1.0])

    if len(fig.data) == 0:
        return None

    return fig


def plot_grouped_single_metric_time(
    df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    metric: str,
    *,
    group_variable: str,
    subpopulations: Optional[List[str]] = None,
    time_unit: Optional[str] = None,
    ref_names: Optional[Set[str]] = None,
):
    """Faceted single-metric grid: one panel per subpopulation level.

    Metric-agnostic counterpart to :func:`plot_grouped_metric_time` (which is
    AUC-specific). Used for a single non-AUC metric (e.g. ``"brier"``) on the
    subpopulation surface so its grid and dimensions match the AUC layout
    instead of the wider 1-row multi-metric grid. Subpopulations wrap into up
    to 2 columns via :func:`_grouped_metric_grid_dims` (the same sizing the AUC
    renderer uses).

    Parameters
    ----------
    df : DataFrame
        Long-format with columns ``model_type``, ``subpopulation``,
        ``eval_time``, ``value`` (and ``metric_name``). Rows are filtered to
        *metric* before rendering.
    std_df : DataFrame, optional
        Same schema as *df* plus ``value_std``; matched on
        ``(subpopulation, model_type, eval_time, metric_name)`` for ribbons.
    metric : str
        Metric key driving labels, y-axis direction, and ribbon clipping.
    group_variable : str
        Name of the grouping variable (used in the figure title).
    subpopulations : list of str, optional
        Ordered subpopulation levels (one panel each). Defaults to the sorted
        unique values in ``df['subpopulation']``.
    time_unit : str, optional
        Unit label for the x-axis.
    ref_names : set of str, optional
        Benchmark reference model names (dashed lines).

    Returns
    -------
    plotly.graph_objects.Figure or None
        ``None`` when *df* is empty or produces no traces.
    """
    if ref_names is None:
        ref_names = set()
    if df is None or df.empty:
        return None

    if 'metric_name' in df.columns:
        df = df[df['metric_name'] == metric]
    if df.empty:
        return None

    # Ribbon data must carry the subpopulation axis to be matchable per cell.
    if std_df is not None and 'subpopulation' not in std_df.columns:
        std_df = None

    if subpopulations is None:
        subpopulations = sorted(df['subpopulation'].unique())
    model_names = sorted(df['model_type'].unique())
    n_sg = len(subpopulations)
    if n_sg == 0:
        return None

    n_cols, n_rows, v_spacing, fig_width, fig_height = _grouped_metric_grid_dims(
        n_sg
    )
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subpopulations,
        horizontal_spacing=0.08,
        vertical_spacing=v_spacing,
    )

    any_ribbon = False
    legend_shown = set()
    ref_idx = 0
    for mname in model_names:
        display_name, color, is_ref, ref_idx = _resolve_model_style(
            mname, ref_names, ref_idx,
        )
        dash = 'dash' if is_ref else 'solid'

        for sg_idx, sg in enumerate(subpopulations):
            row = sg_idx // n_cols + 1
            col = sg_idx % n_cols + 1
            mask = (
                (df['subpopulation'] == sg)
                & (df['model_type'] == mname)
            )
            sub = df[mask].sort_values('eval_time')
            if sub.empty:
                continue
            show = mname not in legend_shown
            fig.add_trace(
                go.Scatter(
                    x=sub['eval_time'],
                    y=sub['value'],
                    mode='lines',
                    name=display_name,
                    line=dict(color=color, dash=dash, width=2),
                    legendgroup=display_name,
                    showlegend=show,
                ),
                row=row, col=col,
            )
            legend_shown.add(mname)
            if _add_metric_time_cell_ribbon(
                fig, sub, std_df, sg, mname, metric, color,
                row, col, display_name,
            ):
                any_ribbon = True

    metric_label = get_metric_display_name(metric)
    x_label = f"Time ({time_unit})" if time_unit else "Time"
    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(text=(
            f"<b>Subpopulation {metric_label}(t): {group_variable}"
            f"{' (±1 SD)' if any_ribbon else ''}</b>"
        )),
        width=fig_width,
        height=fig_height,
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=1.02,
        ),
        margin=dict(l=60, r=140, t=80, b=50),
    )
    tick_kw = _eval_time_ticks(df['eval_time'])
    fig.update_xaxes(title_text=x_label, **tick_kw)

    # Single shared, data-driven y-range across every panel. Same rule as
    # _apply_grouped_metric_yaxes: lower-is-better floors at 0, else caps at 1.
    # The not-lower-better branch does not widen y_min to include CI lower
    # bounds (unlike the AUC renderer) because only lower-is-better metrics
    # route here today; revisit if a not-lower-better metric is added.
    vals = df['value'].dropna()
    if not vals.empty:
        if _is_lower_better(metric):
            y_max = min(1.0, np.ceil(float(vals.max()) * 20) / 20 + 0.05)
            y_range = [0.0, y_max]
        else:
            y_min = max(0.0, np.floor(float(vals.min()) * 20) / 20 - 0.05)
            y_range = [y_min, 1.0]
        fig.update_yaxes(title_text=metric_label, range=y_range)
    else:
        fig.update_yaxes(title_text=metric_label)

    if len(fig.data) == 0:
        return None

    return fig


# =============================================================================
# Grouped multi-metric faceted grid (subpopulation × metric)
# =============================================================================

def _apply_grouped_metric_yaxes(fig, df, metrics, n_cols) -> None:
    """Set a shared, data-driven y-range per metric row across all columns."""
    for r_idx, metric in enumerate(metrics, start=1):
        vals = df.loc[df['metric_name'] == metric, 'value'].dropna()
        if vals.empty:
            continue
        if _is_lower_better(metric):
            y_max = min(1.0, np.ceil(float(vals.max()) * 20) / 20 + 0.05)
            y_range = [0.0, y_max]
        else:
            y_min = max(0.0, np.floor(float(vals.min()) * 20) / 20 - 0.05)
            y_range = [y_min, 1.0]
        title = get_metric_display_name(metric)
        for c_idx in range(1, n_cols + 1):
            fig.update_yaxes(title_text=title, range=y_range, row=r_idx, col=c_idx)


def _add_metric_time_cell_ribbon(
    fig, sub, std_df, sg, mname, metric, color, row, col, display_name,
) -> bool:
    """Draw a per-cell ±1 SD ribbon; return True if one was added.

    Filters *std_df* on ``(subpopulation, model_type, metric_name)``, then
    outer-merges its ``value_std`` onto *sub* on ``eval_time`` before clipping,
    so a point lacking a CI breaks the band honestly rather than bridging it.
    """
    if std_df is None or std_df.empty:
        return False
    s_mask = (
        (std_df['subpopulation'] == sg)
        & (std_df['model_type'] == mname)
        & (std_df['metric_name'] == metric)
    )
    s_sub = std_df[s_mask].sort_values('eval_time')
    if s_sub.empty or 'value_std' not in s_sub.columns:
        return False
    merged = sub.merge(
        s_sub[['eval_time', 'value_std']], on='eval_time', how='outer',
    ).sort_values('eval_time')
    if merged.empty:
        return False
    means = merged['value'].values
    stds = merged['value_std'].values
    if np.all(np.isnan(means)) or np.all(np.isnan(stds)):
        return False
    # Brier-family is unbounded above; C-index-family caps at 1.
    hi = None if _is_lower_better(metric) else 1
    upper = np.clip(means + stds, 0, hi)
    lower = np.clip(means - stds, 0, hi)
    add_ci_ribbon(
        fig, merged['eval_time'].values, upper, lower, color,
        row=row, col=col, legendgroup=display_name,
    )
    return True


def plot_grouped_multi_metric_time(
    df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    metrics: List[str],
    *,
    group_variable: str,
    subpopulations: Optional[List[str]] = None,
    time_unit: Optional[str] = None,
    ref_names: Optional[Set[str]] = None,
    show_ci="auto",
):
    """2D faceted grid of time-dependent metric curves: rows = metrics, cols = subpopulations.

    Each cell renders one ``(metric, subpopulation)`` pair with one line
    per model. Ribbons are drawn per cell when *std_df* provides a
    matching ``value_std`` column.

    Parameters
    ----------
    df : DataFrame
        Long-format with columns ``model_type``, ``subpopulation``,
        ``eval_time``, ``value``, ``metric_name``.
    std_df : DataFrame, optional
        Same schema as *df* plus ``value_std``. Entries are matched on
        ``(subpopulation, model_type, eval_time, metric_name)``.
    metrics : list of str
        Ordered list of metric keys (one row per entry).
    group_variable : str
        Name of the grouping variable (used in the figure title).
    subpopulations : list of str, optional
        Ordered list of subpopulation levels (one column per entry). If
        omitted, uses the sorted unique values in ``df['subpopulation']``.
    time_unit : str, optional
        Unit label for the shared x-axis.
    ref_names : set of str, optional
        Benchmark reference model names (rendered with dashed lines).
    show_ci : {"auto", True, False}, default="auto"
        ``"auto"``/``True`` draw ribbons where *std_df* supplies them;
        ``False`` suppresses all ribbons.

    Returns
    -------
    plotly.graph_objects.Figure or None
        The faceted grid of time-dependent metric subplots, or ``None`` when
        *df* is empty, *metrics* or *subpopulations* is empty, or no rows
        match the requested metric/subpopulation/model combinations (no
        traces produced).
    """
    if ref_names is None:
        ref_names = set()
    if df is None or df.empty:
        return None

    # show_ci=False suppresses ribbons entirely; dropping std_df is the
    # simplest way to keep the per-cell ribbon logic ribbon-free.
    if not show_ci:
        std_df = None

    if subpopulations is None:
        subpopulations = sorted(df['subpopulation'].unique())
    model_names = sorted(df['model_type'].unique())

    # Ribbon data must be keyed by the same (subpop, model, metric, time)
    # tuple as the primary frame; drop it if a caller supplied a main-path
    # std frame that lacks the subpopulation axis.
    if std_df is not None and 'subpopulation' not in std_df.columns:
        std_df = None

    n_rows = len(metrics)
    n_cols = len(subpopulations)
    if n_rows == 0 or n_cols == 0:
        return None

    subplot_titles = [
        f"{get_metric_display_name(metric)} — {sg}"
        for metric in metrics for sg in subpopulations
    ]

    v_spacing = min(0.12, 0.4 / max(n_rows, 1))
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=v_spacing,
    )

    legend_shown = set()
    ref_idx = 0
    any_ribbon = False

    for mname in model_names:
        display_name, color, is_ref, ref_idx = _resolve_model_style(
            mname, ref_names, ref_idx,
        )
        dash = 'dash' if is_ref else 'solid'

        for r_idx, metric in enumerate(metrics, start=1):
            for c_idx, sg in enumerate(subpopulations, start=1):
                mask = (
                    (df['subpopulation'] == sg)
                    & (df['model_type'] == mname)
                    & (df['metric_name'] == metric)
                )
                sub = df[mask].sort_values('eval_time')
                if sub.empty:
                    continue
                show = mname not in legend_shown
                fig.add_trace(
                    go.Scatter(
                        x=sub['eval_time'],
                        y=sub['value'],
                        mode='lines',
                        name=display_name,
                        line=dict(color=color, dash=dash, width=2),
                        legendgroup=display_name,
                        showlegend=show,
                    ),
                    row=r_idx, col=c_idx,
                )
                legend_shown.add(mname)

                if _add_metric_time_cell_ribbon(
                    fig, sub, std_df, sg, mname, metric, color,
                    r_idx, c_idx, display_name,
                ):
                    any_ribbon = True

    x_label = f"Time ({time_unit})" if time_unit else "Time"
    panel_height = 340
    fig_height = max(450, n_rows * panel_height + 100)
    fig_width = max(550, 420 * n_cols)
    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(text=(
            f"<b>Subpopulation time-dependent metrics: {group_variable}"
            f"{' (±1 SD)' if any_ribbon else ''}</b>"
        )),
        width=fig_width,
        height=fig_height,
        hovermode='x unified',
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=1.02,
        ),
        margin=dict(l=60, r=140, t=80, b=50),
    )
    tick_kw = _eval_time_ticks(df['eval_time'])
    fig.update_xaxes(title_text=x_label, **tick_kw)

    _apply_grouped_metric_yaxes(fig, df, metrics, n_cols)

    if len(fig.data) == 0:
        return None

    return fig
