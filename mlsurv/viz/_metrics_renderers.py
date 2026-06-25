"""Single-panel metric renderers: forest, raincloud, bar, boxplot.

Each renderer produces a ``go.Figure`` for one metric across models.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv.constants import get_metric_display_name
from mlsurv.viz._palettes import (
    MANUSCRIPT_DEFAULTS,
    get_model_color,
    get_model_display_name as _model_display,
    hex_to_rgba,
)
from mlsurv.viz._theme import OrientedAxes, adaptive_size, create_figure

from mlsurv.viz._metrics_helpers import (
    _add_pe_legend_entry,
    _add_point_estimate_marker,
    _bar_value_range,
    _build_uncertainty_hover,
    _compute_value_range,
    _create_oriented_figure,
    _extract_ci,
    _filter_iterations_by_metric,
    _is_lower_better,
    _margin_for_labels,
    _maybe_add_reference,
    _metric_reference,
    _pe_legend_kw,
    _prepare_model_styling,
    _stacked_offsets,
)


# =============================================================================
# Internal style renderers — forest
# =============================================================================

def _build_forest_annotations(points, ci_lo, ci_hi, has_ci, is_sd):
    """Build annotation text strings for forest plot labels."""
    if has_ci:
        if is_sd:
            return [
                f"{p:.3f} \u00b1 {(hi - lo) / 2:.3f}"
                for p, lo, hi in zip(points, ci_lo, ci_hi)
            ]
        return [
            f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
            for p, lo, hi in zip(points, ci_lo, ci_hi)
        ]
    return [f"{p:.3f}" for p in points]


def _forest_label_kwargs(show_labels, points, ci_lo, ci_hi, has_ci, is_sd, axes):
    """Build text_kw and vert_annotations for forest label display."""
    if not show_labels:
        return dict(mode="markers"), []
    ann_texts = _build_forest_annotations(points, ci_lo, ci_hi, has_ci, is_sd)
    if axes.is_horiz:
        return dict(
            mode="markers+text", text=ann_texts,
            textposition="top right",
            textfont=dict(size=MANUSCRIPT_DEFAULTS["font_size_annotation"]),
        ), []
    return dict(mode="markers"), ann_texts


def _add_vert_annotations(fig, vert_annotations, display_names, points):
    """Add vertical annotation text for non-horizontal forest plots."""
    if not vert_annotations:
        return
    for name, pt, txt in zip(display_names, points, vert_annotations):
        fig.add_annotation(
            x=name, y=pt, text=txt, textangle=90, showarrow=False,
            xanchor="center", yanchor="bottom", xshift=18,
            font=dict(size=MANUSCRIPT_DEFAULTS["font_size_annotation"]),
        )


def _prepare_forest_data(data, model_col, point_col, ci_lower_col, ci_upper_col,
                         sort, palette, axes):
    """Sort data, extract model lists, points, CI arrays, colors, display names.

    Returns ``(df, models, display_names, points, ci_lo, ci_hi, colors)``.
    """
    df = data.copy()

    if sort == "alpha":
        df = df.sort_values(model_col, ascending=not axes.is_horiz)
    elif sort == "value":
        df = df.sort_values(point_col, ascending=axes.is_horiz)

    models = df[model_col].tolist()
    points = df[point_col].values
    ci_lo = df[ci_lower_col].values
    ci_hi = df[ci_upper_col].values
    colors = [get_model_color(m, palette=palette) for m in models]
    display_names = [_model_display(m) for m in models]

    return df, models, display_names, points, ci_lo, ci_hi, colors


def _plot_metric_forest(
    data: pd.DataFrame,
    *,
    model_col: str,
    point_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    metric: Optional[str],
    title: str,
    sort: str,
    ci_level: float,
    show_labels: bool,
    show_reference_line: bool,
    orientation: str,
    xlim: Optional[tuple] = None,
    uncertainty_label: Optional[str] = None,
    palette=None,
) -> go.Figure:
    """Dot-and-whisker plot of a metric with confidence intervals."""
    if data is None or data.empty:
        raise ValueError("data is empty or None — nothing to plot.")
    if not 0 < ci_level < 1:
        raise ValueError(
            f"ci_level must be between 0 and 1 (exclusive), got {ci_level}. "
            f"Common values: 0.90, 0.95, 0.99."
        )

    axes = OrientedAxes(orientation)
    df, models, display_names, points, ci_lo, ci_hi, colors = _prepare_forest_data(
        data, model_col, point_col, ci_lower_col, ci_upper_col,
        sort, palette, axes,
    )

    n_models = len(df)
    fig = _create_oriented_figure(axes, n_models, title)
    if axes.is_horiz:
        fig.update_layout(width=MANUSCRIPT_DEFAULTS["width_single"] + 50)

    _maybe_add_reference(fig, metric, show_reference_line, axes)
    ref_val = (_metric_reference(metric)[0] if metric else None)

    metric_display = get_metric_display_name(metric) if metric else "Value"

    has_ci = not np.allclose(ci_lo, ci_hi)
    is_sd = uncertainty_label == "SD"
    hover = _build_uncertainty_hover(metric_display, has_ci, is_sd,
                                     uncertainty_label, axes,
                                     ci_level=ci_level)

    text_kw, vert_annotations = _forest_label_kwargs(
        show_labels, points, ci_lo, ci_hi, has_ci, is_sd, axes,
    )

    error_kw = {}
    if has_ci:
        error_kw[axes.error_key] = dict(
            type="data", symmetric=False,
            array=(ci_hi - points).tolist(),
            arrayminus=(points - ci_lo).tolist(),
            color="rgba(0,0,0,0.4)",
        )

    fig.add_trace(go.Scatter(
        marker=dict(size=10, color=colors),
        hovertemplate=hover,
        customdata=np.column_stack([ci_lo, ci_hi, (ci_hi - ci_lo) / 2]),
        showlegend=False, cliponaxis=False,
        **axes.scatter_kw(points, display_names),
        **error_kw, **text_kw,
    ))

    _add_vert_annotations(fig, vert_annotations, display_names, points)

    label_pad = 0.05 if show_labels else 0
    val_range = _compute_value_range(ci_lo, ci_hi, xlim, ref_val,
                                     label_pad=label_pad)
    fig.update_layout(
        showlegend=False,
        **axes.value_axis_layout(metric_display, val_range),
        **axes.cat_axis_layout(""),
    )
    if axes.is_horiz:
        margin_l = _margin_for_labels(display_names, axes)
        fig.update_layout(margin=dict(l=margin_l, r=100 if show_labels else 15))
    elif show_labels:
        fig.update_layout(margin=dict(t=MANUSCRIPT_DEFAULTS["margin_t"] + 45))

    return fig


# =============================================================================
# Internal style renderers — raincloud
# =============================================================================

def _add_raincloud_model_traces(fig, values, base, display_name, color,
                                axes, clip_violin, row=None, col=None):
    """Add violin + hover overlay + boxplot + strip traces for one model."""
    add_kw = dict(row=row, col=col) if row is not None else {}
    orient = axes.bar_orientation()
    val_kw = {axes.val: values}
    base_kw = {f"{axes.cat}0": base}

    # Half-violin
    fig.add_trace(go.Violin(
        **val_kw, **base_kw, orientation=orient, side="positive",
        line_color=color, fillcolor=color, opacity=0.3,
        meanline_visible=True, showlegend=False, name=display_name,
        scalemode="width", width=0.5, hoverinfo="skip",
        spanmode="hard" if clip_violin else "soft",
    ), **add_kw)

    # Hover overlay
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    std_val = float(np.std(values, ddof=1))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    summary_hover = (
        f"<b>{display_name}</b><br>"
        f"Mean: {mean_val:.4f} \u00b1 {std_val:.4f}<br>"
        f"Median: {median_val:.4f}<br>"
        f"Q1: {q1:.4f}  Q3: {q3:.4f}<br>"
        f"n = {len(values)}<extra></extra>"
    )
    grid = np.linspace(float(values.min()), float(values.max()), 20)
    fig.add_trace(go.Scatter(
        **axes.scatter_kw(grid, [base + 0.10] * len(grid)),
        mode="markers", marker=dict(size=12, opacity=0, color=color),
        showlegend=False, name=display_name, hovertemplate=summary_hover,
    ), **add_kw)

    # Boxplot
    fig.add_trace(go.Box(
        **val_kw, **base_kw, orientation=orient, width=0.16,
        marker_color=color, line_color=color,
        fillcolor="rgba(255,255,255,0.8)",
        showlegend=False, name=display_name, boxpoints=False, boxmean=True,
    ), **add_kw)

    # Stacked strip
    swarm_offsets = _stacked_offsets(values)
    strip_hover = (
        f"<b>{display_name}</b><br>"
        f"Value: %{{{axes.val}:.4f}}<extra></extra>"
    )
    fig.add_trace(go.Scatter(
        **axes.scatter_kw(values, base - 0.12 + swarm_offsets),
        mode="markers", marker=dict(size=3, color=color, opacity=0.4),
        showlegend=False, name=display_name, hovertemplate=strip_hover,
    ), **add_kw)


def _build_raincloud_components(df, models, display_names, colors, model_col,
                                value_col, axes, clip_violin, point_estimates, fig):
    """Add raincloud traces and point estimate markers for all models.

    Modifies *fig* in place.
    """
    has_pe = point_estimates is not None and len(point_estimates) > 0
    for i, model in enumerate(models):
        values = df[df[model_col] == model][value_col].values
        _add_raincloud_model_traces(fig, values, float(i), display_names[i],
                                    colors[i], axes, clip_violin)
        if has_pe and model in point_estimates:
            _add_point_estimate_marker(fig, point_estimates[model],
                                       display_names[i], colors[i],
                                       float(i) + 0.10, axes)

    _add_pe_legend_entry(fig, has_pe)


def _plot_metric_raincloud(
    iterations: pd.DataFrame,
    *,
    model_col: str,
    metric_col: str,
    value_col: str,
    metric: Optional[str],
    title: str,
    sort: str,
    clip_violin: bool,
    show_reference_line: bool,
    orientation: str,
    xlim: Optional[tuple] = None,
    point_estimates: Optional[Dict[str, float]] = None,
    palette=None,
) -> go.Figure:
    """Half-violin + jittered strip + boxplot (raincloud) of bootstrap metrics."""
    _VALID_ORIENTATIONS = {"horizontal", "vertical"}
    if orientation not in _VALID_ORIENTATIONS:
        raise ValueError(
            f"Unknown orientation '{orientation}'. "
            f"Must be one of: {sorted(_VALID_ORIENTATIONS)}"
        )
    if iterations is None or iterations.empty:
        raise ValueError("iterations is empty or None — nothing to plot.")

    axes = OrientedAxes(orientation)
    df = _filter_iterations_by_metric(iterations, metric, metric_col)
    models, display_names, colors = _prepare_model_styling(
        df, model_col, value_col, sort, palette,
    )

    n_models = len(models)
    has_pe = point_estimates is not None and len(point_estimates) > 0
    if axes.is_horiz:
        rain_width = int(MANUSCRIPT_DEFAULTS["width_single"] * 0.75)
        h = adaptive_size(n_models, per_item=68, base=100, min_size=MANUSCRIPT_DEFAULTS["height_small"])
    else:
        rain_width = adaptive_size(n_models, per_item=85, base=70, min_size=260)
        h = MANUSCRIPT_DEFAULTS["height_medium"]
    if has_pe:
        rain_width += 150

    fig = create_figure(title=title, width=rain_width, height=h)

    _maybe_add_reference(fig, metric, show_reference_line, axes)
    ref_val = (_metric_reference(metric)[0] if metric else None)

    _build_raincloud_components(
        df, models, display_names, colors, model_col, value_col,
        axes, clip_violin, point_estimates, fig,
    )

    metric_display = get_metric_display_name(metric) if metric else "Value"
    all_values = df[value_col].values
    val_range = _compute_value_range(all_values, all_values, xlim, ref_val)

    legend_kw = _pe_legend_kw(has_pe)
    tick_kw = dict(tickvals=list(range(n_models)), ticktext=display_names)
    if n_models > 5:
        tick_kw["tickfont"] = dict(size=10)
    val_extra = {}
    if has_pe:
        val_extra["title_standoff"] = 15

    fig.update_layout(
        **axes.value_axis_layout(metric_display, val_range, **val_extra),
        **axes.cat_axis_layout("", **tick_kw),
        violinmode="overlay", **legend_kw,
    )
    if axes.is_horiz:
        margin_l = _margin_for_labels(display_names, axes)
        fig.update_layout(margin=dict(l=margin_l))
    if has_pe:
        fig.update_layout(margin=dict(r=100))

    return fig


# =============================================================================
# Internal style renderers — bar
# =============================================================================

def _plot_metric_bar(
    data: pd.DataFrame,
    *,
    model_col: str,
    point_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    metric: Optional[str],
    title: str,
    show_reference_line: bool,
    orientation: str,
    xlim: Optional[tuple] = None,
    uncertainty_label: Optional[str] = None,
    ci_level: float = 0.95,
    palette=None,
) -> go.Figure:
    """Vertical bar chart of point estimates with error bars from CI."""
    if data is None or data.empty:
        raise ValueError("data is empty or None — nothing to plot.")

    axes = OrientedAxes(orientation)
    df = data.copy()
    metric_display = get_metric_display_name(metric) if metric else "Value"
    lower_is_better = _is_lower_better(metric)
    df = df.sort_values(point_col, ascending=lower_is_better)

    models = df[model_col].tolist()
    points = df[point_col].values
    display_names = [_model_display(m) for m in models]
    colors = [get_model_color(m, palette=palette) for m in models]

    ci_lo, ci_hi, error_dict, has_ci = _extract_ci(
        df, ci_lower_col, ci_upper_col, points,
    )
    is_sd = uncertainty_label == "SD"
    hover = _build_uncertainty_hover(metric_display, has_ci, is_sd,
                                     uncertainty_label, axes,
                                     ci_level=ci_level)

    n_models = len(df)
    fig = _create_oriented_figure(
        axes, n_models, title,
        h_per_item=55, h_base=100,
        w_per_item=100, w_base=80, w_min=300,
    )
    if axes.is_horiz:
        bar_w = int(MANUSCRIPT_DEFAULTS["width_single"] * 0.85)
        fig.update_layout(width=bar_w)

    custom_kw = {}
    if has_ci:
        custom_kw["customdata"] = np.column_stack([ci_lo, ci_hi, (ci_hi - ci_lo) / 2])
    error_kw = {axes.error_key: error_dict} if error_dict else {}

    fig.add_trace(go.Bar(
        **axes.scatter_kw(points, display_names),
        orientation=axes.bar_orientation(),
        marker_color=colors,
        hovertemplate=hover,
        **custom_kw, **error_kw,
    ))

    val_range = _bar_value_range(xlim, points, lower_is_better)
    fig.update_layout(
        showlegend=False,
        **axes.value_axis_layout(metric_display, val_range),
        **axes.cat_axis_layout("Model" if not axes.is_horiz else ""),
    )
    if axes.is_horiz:
        margin_l = _margin_for_labels(display_names, axes)
        fig.update_layout(margin=dict(l=margin_l))

    _maybe_add_reference(fig, metric, show_reference_line, axes)

    return fig


# =============================================================================
# Internal style renderers — boxplot
# =============================================================================

def _add_boxplot_model_traces(fig, values, idx, display_name, color,
                              metric_display, axes, row=None, col=None):
    """Add jitter + box + hover traces for one model in a boxplot."""
    add_kw = dict(row=row, col=col) if row is not None else {}
    orient = axes.bar_orientation()
    val_kw = {axes.val: values}
    base_kw = {f"{axes.cat}0": idx}

    # Jittered points — local RNG so we don't disturb the caller's global NumPy state
    rng = np.random.RandomState(42 + idx)
    jitter = rng.uniform(-0.15, 0.15, len(values))
    cat_positions = [idx + j for j in jitter]
    fig.add_trace(go.Scatter(
        **axes.scatter_kw(values, cat_positions),
        mode='markers',
        marker=dict(size=5, color=color, opacity=0.7,
                    line=dict(width=0.5, color='white')),
        showlegend=False,
        hovertemplate=(
            f'<b>{display_name}</b><br>'
            f'{metric_display}: {axes.val_ref(".3f")}<extra></extra>'
        ),
    ), **add_kw)

    # Box
    fig.add_trace(go.Box(
        **val_kw, **base_kw, name=display_name, orientation=orient,
        marker_color=color, marker_size=0, line_color=color,
        fillcolor=hex_to_rgba(color, alpha=0.3), width=0.4,
        showlegend=False, boxmean=True, boxpoints=False, hoverinfo="skip",
    ), **add_kw)

    # Summary hover target
    mean_val = float(np.nanmean(values))
    median_val = float(np.nanmedian(values))
    std_val = float(np.nanstd(values, ddof=1))
    q1 = float(np.nanpercentile(values, 25))
    q3 = float(np.nanpercentile(values, 75))
    box_hover = (
        f"<b>{display_name}</b><br>"
        f"Mean: {mean_val:.4f} \u00b1 {std_val:.4f}<br>"
        f"Median: {median_val:.4f}<br>"
        f"Q1: {q1:.4f}  Q3: {q3:.4f}<br>"
        f"n = {len(values)}<extra></extra>"
    )
    fig.add_trace(go.Scatter(
        **axes.scatter_kw([median_val], [idx]),
        mode="markers", marker=dict(size=20, color=color, opacity=0),
        showlegend=False, hovertemplate=box_hover,
    ), **add_kw)


def _boxplot_value_range(df, value_col, metric, xlim):
    """Compute value-axis range for boxplot with metric-aware defaults."""
    if xlim is not None:
        return list(xlim)
    lower_is_better = _is_lower_better(metric)
    all_values = df[value_col].values
    data_min = np.nanmin(all_values)
    data_max = np.nanmax(all_values)
    data_range = data_max - data_min
    padding = max(data_range * 0.2, 0.02)
    if lower_is_better:
        val_min = max(0, data_min - padding)
        val_max = data_max + padding
        if data_max > 0.20:
            val_max = max(val_max, 0.27)
    else:
        val_min = data_min - padding
        val_max = min(1.0, data_max + padding)
        if data_min < 0.55:
            val_min = min(val_min, 0.48)
    return [val_min, val_max]


def _prepare_boxplot_data(iterations, model_col, value_col, metric_col, metric):
    """Filter by metric, sort models by median, build model/display/color lists.

    Returns ``(df, models, display_names)``.
    """
    df = _filter_iterations_by_metric(iterations, metric, metric_col)
    lower_is_better = _is_lower_better(metric)

    # Sort by median
    models = df[model_col].unique().tolist()
    medians = {m: df.loc[df[model_col] == m, value_col].median() for m in models}
    models = sorted(models, key=lambda m: medians[m], reverse=not lower_is_better)
    display_names = [_model_display(m) for m in models]

    return df, models, display_names


def _plot_metric_boxplot(
    iterations: pd.DataFrame,
    *,
    model_col: str,
    metric_col: str,
    value_col: str,
    metric: Optional[str],
    title: str,
    show_reference_line: bool,
    orientation: str,
    xlim: Optional[tuple] = None,
    point_estimates: Optional[Dict[str, float]] = None,
    palette=None,
) -> go.Figure:
    """Boxplot with jittered points of iteration-level metric data."""
    if iterations is None or iterations.empty:
        raise ValueError("iterations is empty or None — nothing to plot.")

    axes = OrientedAxes(orientation)
    df, models, display_names = _prepare_boxplot_data(
        iterations, model_col, value_col, metric_col, metric,
    )
    metric_display = get_metric_display_name(metric) if metric else "Value"

    n_models = len(models)
    has_pe = point_estimates is not None and len(point_estimates) > 0
    fig = _create_oriented_figure(
        axes, n_models, title,
        h_per_item=55, h_base=100,
        w_per_item=85, w_base=80, w_min=300,
    )

    for i, (model, display_name) in enumerate(zip(models, display_names)):
        values = df.loc[df[model_col] == model, value_col].values
        color = get_model_color(model, palette=palette)
        _add_boxplot_model_traces(fig, values, i, display_name, color,
                                  metric_display, axes)
        if has_pe and model in point_estimates:
            _add_point_estimate_marker(fig, point_estimates[model],
                                       display_name, color, i + 0.25, axes)

    _add_pe_legend_entry(fig, has_pe)

    val_range = _boxplot_value_range(df, value_col, metric, xlim)
    legend_kw = _pe_legend_kw(has_pe)
    tick_kw = dict(tickmode='array', tickvals=list(range(n_models)),
                   ticktext=display_names)
    fig.update_layout(
        **axes.value_axis_layout(metric_display, val_range),
        **axes.cat_axis_layout("Model" if not axes.is_horiz else "", **tick_kw),
        **legend_kw,
    )
    if axes.is_horiz:
        margin_l = _margin_for_labels(display_names, axes)
        horiz_w = int(MANUSCRIPT_DEFAULTS["width_single"] * 0.9)
        if has_pe:
            horiz_w += 150
        horiz_kw = dict(width=horiz_w, margin=dict(l=margin_l))
        if has_pe:
            horiz_kw["xaxis"] = dict(title_standoff=15)
        fig.update_layout(**horiz_kw)
    elif has_pe:
        fig.update_layout(width=fig.layout.width + 150)
    if has_pe:
        fig.update_layout(margin=dict(r=100))

    _maybe_add_reference(fig, metric, show_reference_line, axes)

    return fig
