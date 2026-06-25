"""Multi-panel metric renderers: multi-forest and multi-panel grid.

Handles layouts where multiple metrics are displayed side-by-side
in a subplot grid.
"""

from math import ceil
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from mlsurv.viz._palettes import (
    MANUSCRIPT_DEFAULTS,
    get_model_color,
    get_model_display_name as _model_display,
)
from mlsurv.viz._theme import OrientedAxes, adaptive_size, margin_for_labels

from mlsurv.viz._metrics_helpers import (
    _PANEL_REFERENCE_VALUES,
    _add_pe_legend_entry,
    _add_point_estimate_marker,
    _is_lower_better,
    _metric_col_display,
    _prepare_model_styling,
    _unify_concordance_xranges,
)
from mlsurv.viz._metrics_renderers import (
    _add_boxplot_model_traces,
    _add_raincloud_model_traces,
)
from mlsurv.viz._plotly_helpers import asymmetric_error_bar, symmetric_error_bar


# =============================================================================
# Multi-panel grid constants
# =============================================================================

_MULTI_PANEL_PER_ITEM = {
    "forest": 40,
    "raincloud": 55,
    "boxplot": 45,
    "bar": 45,
}

# Extra px (added to both figure height and top margin) reserved when a
# multi-panel title carries a second line (header + subtitle), so both lines sit
# above the subplot titles instead of overlapping them or clipping at the top.
_SUBTITLE_TITLE_PAD = 40

# Headroom (px) kept above a two-line title so the bold header line is not
# clipped at the figure's top edge. Converted to a fraction of figure height for
# Plotly's container-referenced title.y.
_SUBTITLE_TITLE_HEADROOM = 18


def _title_layout(title, height):
    """Title dict + (height, top-margin) pads for a possibly two-line title.

    A multi-panel title with a ``<br>`` subtitle needs more vertical room: the
    header would otherwise clip at the top edge and the subtitle would crowd the
    subplot titles. Returns ``(title_dict, subtitle_pad, padded_height)`` where
    ``subtitle_pad`` is added to the top margin and is already folded into
    ``padded_height``. Single-line titles keep the original ``y=0.98`` anchor.
    """
    if "<br>" not in title:
        return (
            dict(text=f"<b>{title}</b>", y=0.98, yanchor="top"), 0, height,
        )
    padded_height = height + _SUBTITLE_TITLE_PAD
    title_y = 1 - _SUBTITLE_TITLE_HEADROOM / padded_height
    return (
        dict(text=f"<b>{title}</b>", y=title_y, yanchor="top"),
        _SUBTITLE_TITLE_PAD,
        padded_height,
    )


# =============================================================================
# Multi-panel helpers
# =============================================================================

def _collect_summary_panel_data(panel_df, models, colors, model_col,
                               point_col, ci_lower_col, ci_upper_col):
    """Extract point estimates, CI bounds, and colors for one summary panel.

    Returns (points, ci_lo, ci_hi, m_colors) as numpy arrays (or list for colors).
    """
    points = []
    ci_lo = []
    ci_hi = []
    m_colors = []
    for i, model in enumerate(models):
        row_data = panel_df[panel_df[model_col] == model]
        if row_data.empty:
            points.append(np.nan)
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
            m_colors.append(colors[i])
            continue
        points.append(row_data[point_col].values[0])
        ci_lo.append(row_data[ci_lower_col].values[0]
                     if ci_lower_col in row_data.columns else np.nan)
        ci_hi.append(row_data[ci_upper_col].values[0]
                     if ci_upper_col in row_data.columns else np.nan)
        m_colors.append(colors[i])
    return np.array(points), np.array(ci_lo), np.array(ci_hi), m_colors


def _build_error_x(points, ci_lo, ci_hi, **extra):
    """Build Plotly error_x dict from point estimates and CI bounds.

    Returns None if no finite CI bounds exist.
    """
    if not np.isfinite(ci_lo).any():
        return None
    return asymmetric_error_bar(ci_hi - points, points - ci_lo, **extra)


# =============================================================================
# Multi-forest renderer
# =============================================================================

def _plot_metric_multi_forest(
    df: pd.DataFrame,
    metrics: List[str],
    model_col: str,
    title: str,
    palette=None,
    ci_level: float = 0.95,
    uncertainty_label: Optional[str] = None,
) -> go.Figure:
    """Grouped forest plot with one subplot per metric.

    When *uncertainty_label* is ``"SD"`` the ``{m}_std`` band is drawn as
    ``point +/- std`` (z = 1) with an "± 1 SD" hover, matching the CV ± SD
    default. Otherwise it is the ``point +/- z*std`` CI band.
    """
    if df is None or df.empty:
        raise ValueError("data is empty or None -- nothing to plot.")
    if not 0 < ci_level < 1:
        raise ValueError(
            f"ci_level must be between 0 and 1 (exclusive), got {ci_level}. "
            f"Common values: 0.90, 0.95, 0.99."
        )
    is_sd = uncertainty_label == "SD"

    n_metrics = len(metrics)
    subplot_labels = [_metric_col_display(m) for m in metrics]
    fig = make_subplots(
        rows=1, cols=n_metrics,
        subplot_titles=subplot_labels,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    models = df[model_col].tolist()
    display_names = [_model_display(m) for m in models]
    model_colors = [get_model_color(m, palette=palette) for m in models]

    for col_idx, metric in enumerate(metrics, 1):
        values = df[metric].values
        metric_label = _metric_col_display(metric)

        ci_lower_col = f'{metric}_ci_lower'
        ci_upper_col = f'{metric}_ci_upper'
        std_metric = f'{metric}_std'
        error_x = None
        hover_kw = {}
        if not is_sd and ci_lower_col in df.columns and ci_upper_col in df.columns:
            # Prefer percentile CI bounds carried alongside the point — they
            # match the bootstrap CI used elsewhere in the report (e.g., the
            # performance table). Fall back to ±z*std only when bounds are
            # not provided (legacy CV / wide-input callers) or when an SD band
            # is explicitly requested (uncertainty="sd" → the std branch, z=1).
            ci_lo = df[ci_lower_col].values
            ci_hi = df[ci_upper_col].values
            error_x = dict(
                type="data", symmetric=False,
                array=(ci_hi - values).tolist(),
                arrayminus=(values - ci_lo).tolist(),
                color="rgba(0,0,0,0.4)",
            )
            hover_kw["customdata"] = np.column_stack([ci_lo, ci_hi])
            hover_kw["hovertemplate"] = (
                "<b>%{y}</b><br>"
                f"{metric_label}: " + "%{x:.3f}<br>"
                f"{ci_level * 100:g}% CI: "
                "[%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                "<extra></extra>"
            )
        elif std_metric in df.columns:
            z = 1.0 if is_sd else norm.ppf(1 - (1 - ci_level) / 2)
            half_w = z * df[std_metric].values
            error_x = symmetric_error_bar(half_w)
            ci_lo = values - half_w
            ci_hi = values + half_w
            band_label = "± 1 SD" if is_sd else f"{ci_level * 100:g}% CI"
            hover_kw["customdata"] = np.column_stack([ci_lo, ci_hi])
            hover_kw["hovertemplate"] = (
                "<b>%{y}</b><br>"
                f"{metric_label}: " + "%{x:.3f}<br>"
                f"{band_label}: "
                "[%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                "<extra></extra>"
            )
        else:
            hover_kw["hovertemplate"] = (
                "<b>%{y}</b><br>"
                f"{metric_label}: " + "%{x:.3f}<extra></extra>"
            )

        fig.add_trace(go.Scatter(
            x=values,
            y=display_names,
            mode='markers',
            marker=dict(size=10, color=model_colors),
            error_x=error_x,
            showlegend=False,
            **hover_kw,
        ), row=1, col=col_idx)

        ref = _PANEL_REFERENCE_VALUES.get(metric.lower())
        if ref is not None:
            fig.add_vline(
                x=ref, line_dash='dash', line_color='gray', line_width=1.5,
                row=1, col=col_idx,
            )

    d = MANUSCRIPT_DEFAULTS
    margin_l = margin_for_labels(display_names, min_margin=d["margin_l"])
    base_height = adaptive_size(len(models), per_item=40, base=120, min_size=200)
    title_dict, subtitle_pad, height = _title_layout(title, base_height)
    fig.update_layout(
        template="mlsurv_minimal",
        width=d["width_double"],
        height=height,
        title=title_dict,
        margin=dict(t=d["margin_t"] + 20 + subtitle_pad, l=margin_l),
    )

    _unify_concordance_xranges(fig, metrics)
    return fig


# =============================================================================
# Per-style panel renderers (extracted from _plot_metric_multi_panel loop)
# =============================================================================

def _render_multi_panel_raincloud(fig, panel_df, models, display_names, colors,
                                  model_col, value_col, axes, clip_violin,
                                  panel_pe, r, c):
    """Render raincloud traces for one panel in a multi-panel grid."""
    has_panel_pe = panel_pe is not None and len(panel_pe) > 0
    for i, model in enumerate(models):
        vals = panel_df[panel_df[model_col] == model][value_col].values
        if len(vals) == 0:
            continue
        _add_raincloud_model_traces(
            fig, vals, float(i), display_names[i], colors[i],
            axes, clip_violin, row=r, col=c,
        )
        if has_panel_pe and model in panel_pe:
            _add_point_estimate_marker(
                fig, panel_pe[model], display_names[i], colors[i],
                float(i) + 0.10, axes, row=r, col=c,
            )


def _render_multi_panel_boxplot(fig, panel_df, models, display_names, colors,
                                model_col, value_col, metric, axes,
                                panel_pe, r, c):
    """Render boxplot traces for one panel in a multi-panel grid."""
    metric_display = _metric_col_display(metric)
    has_panel_pe = panel_pe is not None and len(panel_pe) > 0
    for i, model in enumerate(models):
        vals = panel_df[panel_df[model_col] == model][value_col].values
        if len(vals) == 0:
            continue
        _add_boxplot_model_traces(
            fig, vals, i, display_names[i], colors[i],
            metric_display, axes, row=r, col=c,
        )
        if has_panel_pe and model in panel_pe:
            _add_point_estimate_marker(
                fig, panel_pe[model], display_names[i], colors[i],
                float(i) + 0.10, axes, row=r, col=c,
            )


def _render_multi_panel_summary(fig, panel_df, models, display_names, colors,
                                style, model_col, point_col, ci_lower_col,
                                ci_upper_col, metric, r, c, ci_level=0.95,
                                uncertainty_label=None):
    """Render forest or bar traces for one summary panel in a multi-panel grid.

    Bounds arrive already resolved (CV ± SD or ± z*std CI) from
    :func:`_wide_to_long_summary`; *uncertainty_label* only sets the hover
    wording so it matches the band.
    """
    points, ci_lo, ci_hi, m_colors = _collect_summary_panel_data(
        panel_df, models, colors, model_col,
        point_col, ci_lower_col, ci_upper_col,
    )
    band_label = (
        "± 1 SD" if uncertainty_label == "SD" else f"{ci_level * 100:g}% CI"
    )
    if style == "forest":
        error_x = _build_error_x(
            points, ci_lo, ci_hi, color="rgba(0,0,0,0.4)",
        )
        metric_label = _metric_col_display(metric)
        fig.add_trace(go.Scatter(
            x=points, y=display_names,
            mode="markers",
            marker=dict(size=10, color=m_colors),
            error_x=error_x,
            showlegend=False,
            customdata=np.column_stack([ci_lo, ci_hi]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{metric_label}: " + "%{x:.3f}<br>"
                f"{band_label}: "
                "[%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                "<extra></extra>"
            ),
        ), row=r, col=c)
    else:  # bar
        error_x = _build_error_x(
            points, ci_lo, ci_hi, visible=True,
        )
        bar_kw = {}
        if np.isfinite(ci_lo).all():
            # State the band in the hover only when every bar in the panel has
            # one. The test source carries no bounds (all-NaN), and a partial
            # bootstrap panel can carry a NaN bound for one model; either way we
            # keep the plain point hover rather than show "[nan, nan]".
            metric_label = _metric_col_display(metric)
            bar_kw["customdata"] = np.column_stack([ci_lo, ci_hi])
            bar_kw["hovertemplate"] = (
                "<b>%{y}</b><br>"
                f"{metric_label}: " + "%{x:.3f}<br>"
                f"{band_label}: "
                "[%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                "<extra></extra>"
            )
        fig.add_trace(go.Bar(
            x=points, y=display_names,
            orientation="h",
            marker_color=m_colors,
            error_x=error_x,
            showlegend=False,
            **bar_kw,
        ), row=r, col=c)


def _resolve_summary_models(first_df, model_col, point_col, sort, palette,
                            first_metric):
    """Resolve model ordering, display names, and colors for summary data."""
    models = first_df[model_col].tolist()
    if sort == "value":
        vals = {m: first_df.loc[first_df[model_col] == m, point_col].values[0]
                for m in models}
        lower_first = _is_lower_better(first_metric)
        models = sorted(models, key=lambda m: vals[m],
                        reverse=not lower_first)
    elif sort == "alpha":
        models = sorted(models, reverse=True)
    display_names = [_model_display(m) for m in models]
    colors = [get_model_color(m, palette=palette) for m in models]
    return models, display_names, colors


# =============================================================================
# Multi-panel dispatcher
# =============================================================================

def _resolve_multi_panel_models(df, metrics, model_col, value_col, point_col,
                                sort, palette, style, metric_col):
    """Resolve model ordering, display names, and colors for multi-panel grid.

    Returns ``(models, display_names, colors, is_iteration)``.
    """
    is_iteration = style in ("raincloud", "boxplot")
    first_df = df[df[metric_col] == metrics[0]]
    if is_iteration:
        models, display_names, colors = _prepare_model_styling(
            first_df, model_col, value_col, sort, palette,
        )
    else:
        models, display_names, colors = _resolve_summary_models(
            first_df, model_col, point_col, sort, palette, metrics[0],
        )
    return models, display_names, colors, is_iteration


def _plot_metric_multi_panel(
    df: pd.DataFrame,
    metrics: List[str],
    *,
    style: str,
    model_col: str,
    value_col: str = "value",
    metric_col: str = "metric_name",
    point_col: str = "point",
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    title: str = "Model Performance Comparison",
    sort: str = "alpha",
    clip_violin: bool = False,
    ci_level: float = 0.95,
    palette=None,
    point_estimates: Optional[Dict[str, Dict[str, float]]] = None,
    uncertainty_label: Optional[str] = None,
) -> go.Figure:
    """Multi-metric panel grid for any style.

    Parameters
    ----------
    df : DataFrame
        Long-format data with a *metric_col* column identifying the metric.
        Iteration styles expect ``[model_col, metric_col, value_col]``.
        Summary styles expect ``[model_col, metric_col, point_col, ci_lower_col, ci_upper_col]``.
    metrics : list of str
        Metric values to select from *metric_col*, one per panel.
    style : str
        One of ``'forest'``, ``'bar'``, ``'raincloud'``, ``'boxplot'``.
    point_estimates : dict, optional
        Nested dict ``{metric: {model: value}}``. When provided, triangle markers
        are added to raincloud/boxplot panels showing test-set point estimates.
    """
    if df is None or df.empty:
        raise ValueError("data is empty or None — nothing to plot.")

    n_metrics = len(metrics)
    n_cols = min(n_metrics, 2)
    n_rows = ceil(n_metrics / n_cols)

    subplot_labels = [_metric_col_display(m) for m in metrics]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_labels,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.22 if n_rows > 1 else 0.08,
    )

    # Resolve models + display names + colors once (sorted by first metric)
    models, display_names, colors, is_iteration = _resolve_multi_panel_models(
        df, metrics, model_col, value_col, point_col, sort, palette, style,
        metric_col,
    )

    axes = OrientedAxes("horizontal")
    n_models = len(models)

    for panel_idx, metric in enumerate(metrics):
        r = panel_idx // n_cols + 1
        c = panel_idx % n_cols + 1
        panel_df = df[df[metric_col] == metric]
        panel_pe = (point_estimates or {}).get(metric)

        if style == "raincloud":
            _render_multi_panel_raincloud(
                fig, panel_df, models, display_names, colors,
                model_col, value_col, axes, clip_violin, panel_pe, r, c,
            )
        elif style == "boxplot":
            _render_multi_panel_boxplot(
                fig, panel_df, models, display_names, colors,
                model_col, value_col, metric, axes, panel_pe, r, c,
            )
        elif style in ("forest", "bar"):
            _render_multi_panel_summary(
                fig, panel_df, models, display_names, colors,
                style, model_col, point_col, ci_lower_col,
                ci_upper_col, metric, r, c, ci_level=ci_level,
                uncertainty_label=uncertainty_label,
            )

        # Per-panel reference line
        ref = _PANEL_REFERENCE_VALUES.get(metric.lower())
        if ref is not None:
            fig.add_vline(
                x=ref, line_dash="dash", line_color="gray", line_width=1.5,
                row=r, col=c,
            )

    # Point-estimate legend entry (if any panel had point estimates)
    has_any_pe = point_estimates is not None and any(
        len(pe) > 0 for pe in point_estimates.values()
    )
    _add_pe_legend_entry(fig, has_any_pe)

    # Set shared y-axis tick labels on first column only
    tick_kw = dict(tickvals=list(range(n_models)), ticktext=display_names)
    fig.update_yaxes(**tick_kw, col=1)

    # Sizing — wider than standard double-column for grid readability
    per_item = _MULTI_PANEL_PER_ITEM.get(style, 45)
    d = MANUSCRIPT_DEFAULTS
    grid_width = int(d["width_double"] * 1.12)
    if has_any_pe:
        grid_width += 150
    show_legend = has_any_pe
    base_height = adaptive_size(n_models, per_item=per_item, base=120,
                                min_size=200) * n_rows
    title_dict, subtitle_pad, height = _title_layout(title, base_height)
    fig.update_layout(
        template="mlsurv_minimal",
        width=grid_width,
        height=height,
        title=title_dict,
        showlegend=show_legend,
        violinmode="overlay",
        margin=dict(
            l=margin_for_labels(display_names, min_margin=d["margin_l"] + 40),
            t=d["margin_t"] + 20 + subtitle_pad,
        ),
    )

    _unify_concordance_xranges(fig, metrics)
    return fig
