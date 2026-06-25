"""Unified metric visualizations: plot_metric() and plot_metric_time().

This module is the public facade. Implementation lives in submodules:
- ``_metrics_helpers``: constants, data prep, shared renderer utilities
- ``_metrics_renderers``: single-panel renderers (forest, raincloud, bar, boxplot)
- ``_metrics_multi``: multi-panel renderers + grid dispatcher
- ``_metrics_time``: time-dependent renderers + ``plot_metric_time()``
- ``_metrics_roc``: ROC/Brier computation + ``plot_roc()``

All names are re-exported here so existing ``from mlsurv.viz.metrics import X``
statements continue to work unchanged.
"""

import warnings
from math import ceil
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv._warnings import InternalWarning
from mlsurv.constants import get_metric_display_name
from mlsurv.viz._palettes import (
    MANUSCRIPT_DEFAULTS,
    get_model_display_name as _model_display,
)
from mlsurv.viz._theme import apply_size_overrides
from mlsurv.viz._validation import validate_columns

# ── Re-exports: helpers ─────────────────────────────────────────────────────
from mlsurv.viz._metrics_helpers import (  # noqa: F401
    _PANEL_REFERENCE_VALUES,
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
    _metric_col_display,
    _metric_reference,
    _pe_legend_kw,
    _prepare_forest_data_from_bootstrap,
    _prepare_forest_data_from_comparison,
    _prepare_model_styling,
    _stacked_offsets,
    _wide_to_long_summary,
    normalize_metric_param,
)

# ── Re-exports: renderers ───────────────────────────────────────────────────
from mlsurv.viz._metrics_renderers import (  # noqa: F401
    _add_boxplot_model_traces,
    _add_raincloud_model_traces,
    _boxplot_value_range,
    _build_forest_annotations,
    _forest_label_kwargs,
    _add_vert_annotations,
    _plot_metric_bar,
    _plot_metric_boxplot,
    _plot_metric_forest,
    _plot_metric_raincloud,
)

# ── Re-exports: multi-panel ─────────────────────────────────────────────────
from mlsurv.viz._metrics_multi import (  # noqa: F401
    _MULTI_PANEL_PER_ITEM,
    _build_error_x,
    _collect_summary_panel_data,
    _plot_metric_multi_forest,
    _plot_metric_multi_panel,
    _render_multi_panel_boxplot,
    _render_multi_panel_raincloud,
    _render_multi_panel_summary,
)

# ── Re-exports: time ────────────────────────────────────────────────────────
from mlsurv.viz._metrics_time import (  # noqa: F401
    _add_line_model_trace,
    _auto_resolve_time_value_col,
    _plot_metric_line,
    _plot_metric_time_multi_panel,
    _plot_metric_timepoint_forest,
    _resolve_line_columns,
    plot_metric_time,
)

# ── Re-exports: ROC / Brier ─────────────────────────────────────────────────
from mlsurv.viz._metrics_roc import (  # noqa: F401
    RocAssembly,
    assemble_roc_curves_with_references,
    bootstrap_roc_curves,
    compute_brier_curves,
    compute_roc_curves,
    plot_roc,
    plot_roc_interactive,
)


# =============================================================================
# Dispatcher helpers (extracted from plot_metric)
# =============================================================================

def _detect_multi_data_layout(df, metrics, metric_col, value_col, point_col,
                              ci_lower_col):
    """Detect whether multi-panel data is wide, long-iteration, or long-summary."""
    is_wide = all(m in df.columns for m in metrics)
    is_long_iter = metric_col in df.columns and value_col in df.columns
    is_long_summary = (metric_col in df.columns and point_col in df.columns
                       and ci_lower_col in df.columns)
    return is_wide, is_long_iter, is_long_summary


_VALID_MULTI_STYLES = {"auto", "forest", "bar", "raincloud", "boxplot"}


def _resolve_multi_style(style, is_long_iter):
    """Resolve 'auto' style for multi-panel mode."""
    if style not in _VALID_MULTI_STYLES:
        raise ValueError(
            f"Unknown style '{style}'. "
            f"Must be one of: {sorted(_VALID_MULTI_STYLES)}"
        )
    if style != "auto":
        return style
    return "raincloud" if is_long_iter else "forest"


def _dispatch_multi_panel(
    df, metrics, *, style, model_col, value_col, metric_col, point_col,
    ci_lower_col, ci_upper_col, title, sort, clip_violin, ci_level, palette,
    point_estimates, uncertainty_label=None,
):
    """Handle multi-panel dispatch: validate, convert, and call renderer."""
    is_wide, is_long_iter, is_long_summary = _detect_multi_data_layout(
        df, metrics, metric_col, value_col, point_col, ci_lower_col,
    )
    effective_style = _resolve_multi_style(style, is_long_iter)

    if title is None:
        title = "Model Performance Comparison"

    # Wide-format forest → existing renderer (backward compat)
    if effective_style == "forest" and is_wide:
        validate_columns(df, [model_col] + list(metrics), "plot_metric")
        return _plot_metric_multi_forest(
            df, metrics, model_col, title,
            palette=palette, ci_level=ci_level,
            uncertainty_label=uncertainty_label,
        )

    # Convert wide-format to long for non-forest styles. Pass the label so the
    # SD/CI choice (z=1 vs z*std) is baked into ci_lower/ci_upper here.
    if is_wide and not is_long_iter and not is_long_summary:
        df = _wide_to_long_summary(
            df, metrics, model_col, ci_level=ci_level,
            uncertainty_label=uncertainty_label,
        )
        is_long_summary = True

    if is_long_iter:
        validate_columns(
            df, [model_col, value_col, metric_col], "plot_metric",
        )
    elif is_long_summary:
        validate_columns(
            df,
            [model_col, metric_col, point_col, ci_lower_col, ci_upper_col],
            "plot_metric",
        )
    else:
        validate_columns(
            df, [model_col, metric_col], "plot_metric",
        )
    return _plot_metric_multi_panel(
        df, metrics,
        style=effective_style,
        model_col=model_col,
        value_col=value_col,
        metric_col=metric_col,
        point_col=point_col,
        ci_lower_col=ci_lower_col,
        ci_upper_col=ci_upper_col,
        title=title,
        sort=sort,
        clip_violin=clip_violin,
        ci_level=ci_level,
        palette=palette,
        point_estimates=point_estimates,
        uncertainty_label=uncertainty_label,
    )


def _resolve_single_style(style, is_summary, df, model_col, value_col,
                          point_col, ci_lower_col, ci_upper_col):
    """Resolve and validate style for single-panel mode.

    Returns the effective style string. Raises ValueError on mismatch.
    """
    if style == "auto":
        return "forest" if is_summary else "raincloud"
    if style in ("forest", "bar") and not is_summary:
        raise ValueError(
            f"plot_metric() style={style!r} requires summary data with columns "
            f"[{model_col!r}, {point_col!r}, {ci_lower_col!r}, {ci_upper_col!r}], "
            f"but detected iteration data. Available columns: {sorted(df.columns)}"
        )
    if style in ("raincloud", "boxplot") and is_summary:
        raise ValueError(
            f"plot_metric() style={style!r} requires iteration data with columns "
            f"[{model_col!r}, {value_col!r}], but detected summary data."
        )
    return style


def _auto_title(style, metric_display, df):
    """Generate default title for single-panel plot_metric."""
    if style in ("forest", "bar"):
        return f"{metric_display} by Model"
    if "fold" in df.columns:
        return f"{metric_display} CV Fold Distribution"
    return f"{metric_display} Bootstrap Distribution"


_SINGLE_PANEL_DISPATCHERS = {
    "forest": lambda df, kw: _plot_metric_forest(
        df, model_col=kw["model_col"], point_col=kw["point_col"],
        ci_lower_col=kw["ci_lower_col"], ci_upper_col=kw["ci_upper_col"],
        metric=kw["metric"], title=kw["title"], sort=kw["sort"],
        ci_level=kw["ci_level"], show_labels=kw["show_labels"],
        show_reference_line=kw["show_reference_line"],
        orientation=kw["orientation"], xlim=kw["xlim"],
        uncertainty_label=kw["uncertainty_label"], palette=kw["palette"],
    ),
    "bar": lambda df, kw: _plot_metric_bar(
        df, model_col=kw["model_col"], point_col=kw["point_col"],
        ci_lower_col=kw["ci_lower_col"], ci_upper_col=kw["ci_upper_col"],
        metric=kw["metric"], title=kw["title"],
        show_reference_line=kw["show_reference_line"],
        orientation=kw["orientation"], xlim=kw["xlim"],
        uncertainty_label=kw["uncertainty_label"], ci_level=kw["ci_level"],
        palette=kw["palette"],
    ),
    "raincloud": lambda df, kw: _plot_metric_raincloud(
        df, model_col=kw["model_col"], metric_col=kw["metric_col"],
        value_col=kw["value_col"], metric=kw["metric"], title=kw["title"],
        sort=kw["sort"], clip_violin=kw["clip_violin"],
        show_reference_line=kw["show_reference_line"],
        orientation=kw["orientation"], xlim=kw["xlim"],
        point_estimates=kw["point_estimates"], palette=kw["palette"],
    ),
    "boxplot": lambda df, kw: _plot_metric_boxplot(
        df, model_col=kw["model_col"], metric_col=kw["metric_col"],
        value_col=kw["value_col"], metric=kw["metric"], title=kw["title"],
        show_reference_line=kw["show_reference_line"],
        orientation=kw["orientation"], xlim=kw["xlim"],
        point_estimates=kw["point_estimates"], palette=kw["palette"],
    ),
}


def _dispatch_single_panel(
    df, *, style, metric, metric_display, model_col, value_col, point_col,
    ci_lower_col, ci_upper_col, metric_col, sort, show_reference_line,
    show_labels, orientation, title, ci_level, clip_violin, xlim,
    uncertainty_label, point_estimates, palette,
):
    """Handle single-panel dispatch: detect data type, validate, and call renderer."""
    is_summary = point_col in df.columns and ci_lower_col in df.columns

    if is_summary:
        validate_columns(df, [model_col, point_col, ci_lower_col, ci_upper_col], "plot_metric")
    else:
        validate_columns(df, [model_col, value_col], "plot_metric")

    style = _resolve_single_style(
        style, is_summary, df, model_col, value_col,
        point_col, ci_lower_col, ci_upper_col,
    )
    if title is None:
        title = _auto_title(style, metric_display, df)

    dispatcher = _SINGLE_PANEL_DISPATCHERS.get(style)
    if dispatcher is None:
        raise ValueError(
            f"Unknown style '{style}'. "
            f"Must be one of: auto, forest, bar, raincloud, boxplot"
        )
    return dispatcher(df, dict(
        model_col=model_col, point_col=point_col, ci_lower_col=ci_lower_col,
        ci_upper_col=ci_upper_col, metric_col=metric_col, value_col=value_col,
        metric=metric, title=title, sort=sort, ci_level=ci_level,
        show_labels=show_labels, show_reference_line=show_reference_line,
        orientation=orientation, xlim=xlim,
        uncertainty_label=uncertainty_label, clip_violin=clip_violin,
        point_estimates=point_estimates, palette=palette,
    ))


# =============================================================================
# Public API: plot_metric()
# =============================================================================

def plot_metric(
    df: pd.DataFrame,
    *,
    style: str = "auto",
    metric: Optional[Union[str, List[str]]] = None,
    model_col: str = "model",
    value_col: str = "value",
    point_col: str = "point",
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    metric_col: str = "metric_name",
    sort: str = "alpha",
    show_reference_line: bool = True,
    show_labels: bool = True,
    orientation: str = "horizontal",
    title: Optional[str] = None,
    ci_level: float = 0.95,
    clip_violin: bool = False,
    xlim: Optional[tuple] = None,
    uncertainty_label: Optional[str] = None,
    point_estimates=None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    palette=None,
) -> go.Figure:
    """Unified dispatcher for time-invariant metric visualizations.

    Auto-detection:
    - *metric* as a list -> multi-panel forest (one panel per name)
    - *point_col* AND *ci_lower_col* in df -> summary data (forest/bar)
    - else -> iteration-level data (raincloud/boxplot)

    Parameters
    ----------
    df : DataFrame
        Summary or iteration-level metric data.

        **Summary schema** (forest/bar styles)::

            model     point    ci_lower   ci_upper
            str       float    float      float

        **Iteration schema** (raincloud/boxplot styles)::

            model       metric_name  value   iteration
            str         str          float   int

        Column names are configurable via parameters.  Bootstrap data
        uses ``model_type`` — pass ``model_col="model_type"``.

    style : {'auto', 'forest', 'bar', 'raincloud', 'boxplot'}
        Visualization style.  ``'auto'`` selects based on data shape.
    metric : str or list of str, optional
        Metric key(s).  A single name (e.g. ``'harrell'``,
        ``'integrated_brier'``) draws one panel and drives its reference line,
        labels, and axis ranges.  A list draws a multi-panel forest (one panel
        per metric column).  This low-level renderer treats any list as
        multi-panel; the public ``plot_metric`` collapses a 1-element list to
        the scalar before reaching here.
    model_col : str
        Column with model names.
    value_col : str
        Column with metric values (iteration data).
    point_col : str
        Column with point estimates (summary data).
    ci_lower_col : str
        Column with the CI lower bound (summary data).
    ci_upper_col : str
        Column with the CI upper bound (summary data).
    metric_col : str
        Column with metric names (iteration data, for filtering).
    sort : {'alpha', 'value', 'none'}
        Sort order for models.
    show_reference_line : bool
        Show dashed reference line at random performance level.
    show_labels : bool
        Show inline text labels on forest plots.
    orientation : {'horizontal', 'vertical'}
        Plot orientation (raincloud only).
    title : str, optional
        Figure title.  Auto-generated if ``None``.
    ci_level : float
        Confidence interval percentage for labels.
    xlim : tuple of (float, float), optional
        Explicit metric-axis limits ``(min, max)``.  Overrides auto-range.
        Controls x-axis for forest/horizontal styles, y-axis for
        bar/boxplot/vertical styles.
    clip_violin : bool
        If ``True``, clip violins to observed data range.
    uncertainty_label : str, optional
        Legend/annotation label for the uncertainty band or error bars
        (e.g. ``"95% CI"``, ``"bootstrap SD"``). Auto-derived when ``None``.
    point_estimates : dict, optional
        Test-set point estimates overlaid as triangle markers on
        raincloud/boxplot styles. ``{model: value}`` for a single metric, or
        nested ``{metric: {model: value}}`` for multi-panel raincloud/boxplot
        plots.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    palette : dict, optional
        ``{model_name: hex_color}`` overrides for per-model colours.

    Returns
    -------
    go.Figure
    """
    single_metric, metric_list = normalize_metric_param(
        metric, collapse_singleton=False,
    )
    metric_display = (
        get_metric_display_name(single_metric) if single_metric else "Metric"
    )

    if metric_list is not None:
        fig = _dispatch_multi_panel(
            df, metric_list,
            style=style, model_col=model_col, value_col=value_col,
            metric_col=metric_col, point_col=point_col,
            ci_lower_col=ci_lower_col, ci_upper_col=ci_upper_col,
            title=title, sort=sort, clip_violin=clip_violin,
            ci_level=ci_level, palette=palette,
            point_estimates=point_estimates,
            uncertainty_label=uncertainty_label,
        )
    else:
        fig = _dispatch_single_panel(
            df,
            style=style, metric=single_metric, metric_display=metric_display,
            model_col=model_col, value_col=value_col, point_col=point_col,
            ci_lower_col=ci_lower_col, ci_upper_col=ci_upper_col,
            metric_col=metric_col, sort=sort,
            show_reference_line=show_reference_line,
            show_labels=show_labels, orientation=orientation,
            title=title, ci_level=ci_level, clip_violin=clip_violin,
            xlim=xlim, uncertainty_label=uncertainty_label,
            point_estimates=point_estimates, palette=palette,
        )

    apply_size_overrides(fig, width, height)
    return fig


# =============================================================================
# Heatmap: model x metric summary
# =============================================================================

# Red (better) -> white (average) -> blue (worse)
_HEATMAP_COLORSCALE = [
    [0.0, "#2166ac"],   # blue (worst)
    [0.5, "#f7f7f7"],   # white (average)
    [1.0, "#b2182b"],   # red (best)
]

_HEATMAP_HOVER = (
    "<b>%{y}</b><br>%{x}: %{text:.3f}"
    "<br>z-score: %{z:.2f}<extra></extra>"
)


def _zscore_with_direction_flip(z, metrics, nan_mask, *, pooled_mean=None,
                                pooled_std=None):
    """Z-score columns; flip sign for lower-is-better so positive z = better.

    When *pooled_mean* and *pooled_std* are provided (as pd.Series keyed by
    metric), those values are used instead of per-column statistics.  This
    enables cross-cohort z-scoring with shared statistics.
    """
    zero_std_cols = []
    for col in metrics:
        col_data = z[col]
        std = pooled_std[col] if pooled_std is not None else col_data.std(ddof=0, skipna=True)
        mean = pooled_mean[col] if pooled_mean is not None else col_data.mean(skipna=True)
        if std <= 0:
            zero_std_cols.append(col)
        z[col] = (col_data - mean) / std if std > 0 else 0.0
        if _is_lower_better(col):
            z[col] = -z[col]
    # Warn only for the single-panel path; the faceted path warns in
    # _cross_cohort_zscore before calling us with pooled_std.
    if zero_std_cols and pooled_std is None:
        warnings.warn(
            f"plot_metric_heatmap: zero std in columns {zero_std_cols}; "
            f"z-scores set to 0.0 (no variation).",
            InternalWarning,
            stacklevel=3,
        )
    # Re-mask NaN positions so Plotly renders them as blank cells, not "average"
    z[nan_mask] = np.nan
    return z


def _resolve_heatmap_metrics(df, metrics, model_col):
    """Auto-detect or validate metric columns for heatmap."""
    if metrics is None:
        metrics = [
            col for col in df.columns
            if col != model_col and not col.endswith("_std")
            and pd.api.types.is_numeric_dtype(df[col])
        ]
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(
            f"plot_metric_heatmap() missing metric columns: {missing}. "
            f"Available: {sorted(df.columns)}"
        )
    if not metrics:
        raise ValueError(
            "plot_metric_heatmap() requires at least one numeric metric column."
        )
    return metrics


def _prepare_heatmap_data(df, metrics, model_col):
    """Prepare z-scored heatmap data from wide-format summary DataFrame.

    Resolves metrics, computes raw and z-scored matrices with NaN handling
    and direction flip.

    Returns ``(raw, z, nan_mask, ordered_labels, metric_labels)``.
    """
    metrics = _resolve_heatmap_metrics(df, metrics, model_col)

    ordered_models = list(dict.fromkeys(df[model_col]))
    ordered_labels = [_model_display(m) for m in ordered_models]
    raw = (
        df.assign(_display_model=df[model_col].map(_model_display))
        .set_index("_display_model")[metrics]
        .reindex(ordered_labels)
    )
    metric_labels = [get_metric_display_name(m) for m in metrics]

    z = raw.copy()
    nan_mask = raw.isna()
    nan_cols = [c for c in metrics if nan_mask[c].any()]
    if nan_cols:
        warnings.warn(
            f"plot_metric_heatmap: NaN values in columns {nan_cols}; "
            f"these cells will appear blank.",
            InternalWarning,
            stacklevel=3,
        )
    z = _zscore_with_direction_flip(z, metrics, nan_mask)

    return raw, z, nan_mask, ordered_labels, metric_labels


def _compute_heatmap_dimensions(ordered_labels, metric_labels):
    """Compute figure dimensions and margins for heatmap layout.

    Returns ``(fig_w, fig_h, margin_left, margin_bot, margin_top, margin_right)``.
    """
    n_rows = len(ordered_labels)
    n_cols = len(metric_labels)
    cell_h, cell_w = 32, 100
    max_model_len = max(len(lbl) for lbl in ordered_labels)
    max_metric_len = max(len(lbl) for lbl in metric_labels)
    margin_left = max(80, max_model_len * 7)
    margin_bot = max(80, int(max_metric_len * 5.5))
    margin_top, margin_right = 40, 80
    fig_h = max(200, n_rows * cell_h + margin_top + margin_bot)
    fig_w = n_cols * cell_w + margin_left + margin_right
    return fig_w, fig_h, margin_left, margin_bot, margin_top, margin_right


def plot_metric_heatmap(
    df: pd.DataFrame,
    *,
    metrics: Optional[List[str]] = None,
    model_col: str = "model_type",
    title: Optional[str] = None,
) -> go.Figure:
    """Plot a model x metric heatmap from wide-format summary data.

    Values are z-scored per metric (column-wise) so that metrics on different
    scales are comparable. The colorscale uses **red = better**: for
    higher-is-better metrics (e.g. C-index) higher raw values map to red; for
    lower-is-better metrics (e.g. IBS) lower raw values map to red. Direction
    is looked up from ``METRIC_REGISTRY``; unknown metrics default to
    higher-is-better.

    Cell annotations show the **original** (un-normalized) values.

    If any metric column contains NaN values, an ``InternalWarning`` is emitted.
    NaN cells are excluded from z-score computation (``skipna=True``) and
    render as blank cells in the heatmap.
    """
    validate_columns(df, [model_col], "plot_metric_heatmap")
    raw, z, nan_mask, ordered_labels, metric_labels = _prepare_heatmap_data(
        df, metrics, model_col,
    )

    if title is None:
        title = "Metric Summary Heatmap"

    z_arr = np.abs(z.to_numpy())
    z_abs_max = max(np.nanmax(z_arr), 1e-9) if not np.all(np.isnan(z_arr)) else 1e-9

    fig_w, fig_h, margin_left, margin_bot, margin_top, margin_right = (
        _compute_heatmap_dimensions(ordered_labels, metric_labels)
    )

    font_size = MANUSCRIPT_DEFAULTS["font_size_annotation"] - 1
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=metric_labels,
            y=ordered_labels,
            z=z.to_numpy(),
            text=np.where(nan_mask.to_numpy(), None, np.round(raw.to_numpy(), 3)),
            texttemplate="%{text:.3f}",
            textfont=dict(size=font_size),
            colorscale=_HEATMAP_COLORSCALE,
            zmid=0,
            zmin=-z_abs_max,
            zmax=z_abs_max,
            colorbar=dict(len=0.9, thickness=12),
            hovertemplate=_HEATMAP_HOVER,
        )
    )
    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center"),
        width=fig_w,
        height=fig_h,
        margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bot),
        xaxis=dict(side="bottom", tickangle=-45),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# =============================================================================
# Faceted heatmap: cohort × model × metric
# =============================================================================


def _cross_cohort_zscore(cohort_raws, metrics):
    """Z-score across all cohorts using pooled mean/std per metric.

    Parameters
    ----------
    cohort_raws : dict[str, pd.DataFrame]
        cohort_name -> raw DataFrame (model display names as index,
        metric keys as columns).
    metrics : list[str]
        Metric column keys.

    Returns
    -------
    cohort_zs : dict[str, pd.DataFrame]
        cohort_name -> z-scored DataFrame (same shape as input).
    cohort_masks : dict[str, pd.DataFrame]
        cohort_name -> NaN mask DataFrame.
    """
    all_raw = pd.concat(cohort_raws.values(), axis=0)
    pooled_mean = all_raw[metrics].mean(skipna=True)
    pooled_std = all_raw[metrics].std(ddof=0, skipna=True)

    zero_std_cols = [c for c in metrics if pooled_std[c] <= 0]
    if zero_std_cols:
        warnings.warn(
            f"plot_faceted_metric_heatmap: zero pooled std in columns "
            f"{zero_std_cols}; z-scores set to 0.0 (no cross-cohort variation).",
            InternalWarning,
            stacklevel=3,
        )

    cohort_zs = {}
    cohort_masks = {}
    for name, raw in cohort_raws.items():
        z = raw[metrics].copy()
        mask = raw[metrics].isna()
        _zscore_with_direction_flip(
            z, metrics, mask,
            pooled_mean=pooled_mean, pooled_std=pooled_std,
        )
        cohort_zs[name] = z
        cohort_masks[name] = mask
    return cohort_zs, cohort_masks


def plot_faceted_metric_heatmap(
    cohort_dfs: Dict[str, pd.DataFrame],
    *,
    metrics: Optional[List[str]] = None,
    model_col: str = "model_type",
    title: Optional[str] = None,
    recalibrated_models: Optional[set] = None,
) -> go.Figure:
    """Plot a faceted model x metric heatmap with one panel per cohort.

    Values are z-scored **across all cohorts** (pooled mean/std per metric)
    so the same shade of red always maps to the same metric value across
    panels. Direction-aware: red = better for both higher-is-better and
    lower-is-better metrics.

    Cell annotations show the **original** (un-normalized) values.

    Parameters
    ----------
    cohort_dfs : dict[str, pd.DataFrame]
        Mapping from cohort name to a wide-format DataFrame with a
        ``model_col`` column and one numeric column per metric.
    metrics : list[str], optional
        Metric columns to include. Auto-detected if ``None``.
    model_col : str
        Column name for model identifiers (default ``"model_type"``).
    title : str, optional
        Figure title (default ``"Validation Performance (z-scored, red = better)"``).
    recalibrated_models : set[str], optional
        Names (matching ``model_col`` values) for which a post-hoc isotonic
        recalibration of S(t|x) is applied — these rows get a ``" (recal)"``
        suffix on the y-axis label. Only set rows for models recalibrated in
        every cohort that includes them; for partial / mixed recalibration
        across cohorts the markdown fallback table covers per-cell labeling.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``cohort_dfs`` is empty, or if any cohort DataFrame is missing one
        of the requested metric columns.
    """
    from plotly.subplots import make_subplots

    if not cohort_dfs:
        raise ValueError("plot_faceted_metric_heatmap() requires at least one cohort.")

    cohort_names = list(cohort_dfs.keys())
    n_cohorts = len(cohort_names)

    # Validate all cohort DataFrames for model_col and metric columns
    first_df = next(iter(cohort_dfs.values()))
    for cname, df in cohort_dfs.items():
        validate_columns(df, [model_col], f"plot_faceted_metric_heatmap (cohort {cname!r})")
    metrics = _resolve_heatmap_metrics(first_df, metrics, model_col)
    for cname, df in cohort_dfs.items():
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            raise ValueError(
                f"plot_faceted_metric_heatmap() cohort {cname!r} missing "
                f"metric columns: {missing}. Available: {sorted(df.columns)}"
            )

    # Union of all models across cohorts, preserving first-seen order
    all_models = list(dict.fromkeys(
        m for df in cohort_dfs.values() for m in df[model_col]
    ))
    _recal = recalibrated_models or set()
    ordered_labels = [
        f"{_model_display(m)} (recal)" if m in _recal else _model_display(m)
        for m in all_models
    ]

    # Build per-cohort raw DataFrames (model display names as index)
    metric_labels = [get_metric_display_name(m) for m in metrics]

    def _display_with_recal(m):
        base = _model_display(m)
        return f"{base} (recal)" if m in _recal else base

    cohort_raws = {}
    for cname, df in cohort_dfs.items():
        raw = (
            df.assign(_display_model=df[model_col].map(_display_with_recal))
            .set_index("_display_model")[metrics]
            .reindex(ordered_labels)
        )
        cohort_raws[cname] = raw

    # Warn about NaN values
    for cname, raw in cohort_raws.items():
        nan_cols = [c for c in metrics if raw[c].isna().any()]
        if nan_cols:
            warnings.warn(
                f"plot_faceted_metric_heatmap: NaN values in cohort "
                f"{cname!r}, columns {nan_cols}; these cells will appear blank.",
                InternalWarning,
                stacklevel=2,
            )

    # Cross-cohort z-scoring
    cohort_zs, cohort_masks = _cross_cohort_zscore(cohort_raws, metrics)

    # Global z-abs-max for shared colorscale (NaN-safe)
    z_vals = []
    for z in cohort_zs.values():
        arr = np.abs(z.to_numpy())
        if not np.all(np.isnan(arr)):
            z_vals.append(np.nanmax(arr))
    z_abs_max = max(max(z_vals) if z_vals else 0.0, 1e-9)

    # Layout: max 2 columns (wider panels read better on most screens)
    n_cols = min(n_cohorts, 2)
    n_rows = ceil(n_cohorts / n_cols)

    # ── Dimensions (compute everything before make_subplots) ──────────
    _, _, ml, mb, mt, mr = _compute_heatmap_dimensions(
        ordered_labels, metric_labels,
    )
    n_model_rows = len(ordered_labels)
    n_metric_cols = len(metric_labels)
    cell_h, cell_w = 32, 100
    panel_content_w = n_metric_cols * cell_w
    panel_content_h = max(100, n_model_rows * cell_h)

    mr_total = mr + 30   # extra for colorbar
    mt_total = mt + 30   # extra for subplot titles + figure title

    h_spacing = 0.12

    # Plotly's v_spacing is a fraction of the plot area (region inside
    # outer margins).  We need enough pixel gap between rows for the
    # bottom x-labels of the upper panel plus the subplot title of the
    # lower panel.  Work backwards: pick a pixel gap, compute the total
    # plot-area height, then derive the fraction.
    row_gap_px = mb + 20  # angled labels + subtitle + breathing room
    if n_rows > 1:
        plot_area_h = (
            panel_content_h * n_rows + row_gap_px * (n_rows - 1)
        )
        v_spacing = row_gap_px / plot_area_h
    else:
        plot_area_h = panel_content_h
        v_spacing = 0.0

    fig_w = (
        ml + panel_content_w * n_cols
        + int(h_spacing * panel_content_w * max(n_cols - 1, 0))
        + mr_total
    )
    fig_h = mt_total + plot_area_h + mb

    # ── Create subplot grid ───────────────────────────────────────────
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=cohort_names,
        shared_yaxes=True,
        horizontal_spacing=h_spacing,
        vertical_spacing=v_spacing,
    )

    font_size = MANUSCRIPT_DEFAULTS["font_size_annotation"] - 1

    for idx, cname in enumerate(cohort_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        is_last = idx == n_cohorts - 1

        z_panel = cohort_zs[cname]
        raw_panel = cohort_raws[cname]
        mask_panel = cohort_masks[cname]

        fig.add_trace(
            go.Heatmap(
                x=metric_labels,
                y=ordered_labels,
                z=z_panel.to_numpy(),
                text=np.where(
                    mask_panel.to_numpy(), None,
                    np.round(raw_panel.to_numpy(), 3),
                ),
                texttemplate="%{text:.3f}",
                textfont=dict(size=font_size),
                colorscale=_HEATMAP_COLORSCALE,
                zmid=0,
                zmin=-z_abs_max,
                zmax=z_abs_max,
                showscale=is_last,
                colorbar=dict(
                    len=0.9, thickness=12, title=dict(text="z-score", side="right"),
                ) if is_last else None,
                hovertemplate=_HEATMAP_HOVER,
            ),
            row=row,
            col=col,
        )

    if title is None:
        title = "Validation Performance (z-scored, red = better)"

    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center"),
        width=fig_w,
        height=fig_h,
        margin=dict(l=ml, r=mr_total, t=mt_total, b=mb),
    )

    # Per-panel axis settings
    for idx in range(n_cohorts):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig.update_yaxes(autorange="reversed", row=row, col=col)
        fig.update_xaxes(tickangle=-45, side="bottom", row=row, col=col)

    return fig
