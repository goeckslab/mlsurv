"""Feature interpretation visualizations.

This module is the public facade. Implementation lives in submodules:
- ``_features_helpers``: shared data prep, group utilities, common helpers
- ``_features_importance``: single-model importance renderers (bar/forest/raincloud/boxplot)
- ``_features_importance_multi``: multi-model importance comparison + heatmap
- ``_features_coefficients``: plot_coefficients + plot_coefficient_comparison
- ``_features_hazard_ratios``: compute_hazard_ratios, plot_hazard_ratios, comparison
- ``_features_correlation``: plot_feature_correlation

All names are re-exported here so existing ``from mlsurv.viz.features import X``
statements continue to work unchanged.
"""

from typing import Dict, Optional, Union

import pandas as pd
import plotly.graph_objects as go

from mlsurv.viz._palettes import (
    FeatureGroupsType,
    MANUSCRIPT_DEFAULTS,
    _add_group_separator_shapes,
    _narrow_groups,
    _resolve_groups,
    _sort_features_by_group,
)
from mlsurv.viz._theme import adaptive_size, apply_size_overrides
from mlsurv.viz._validation import validate_columns

# ── Re-exports: helpers ─────────────────────────────────────────────────────
from mlsurv.viz._features_helpers import (  # noqa: F401
    _apply_group_layout,
    _assign_coef_sign,
    _auto_detect_feature_groups,
    _feature_color,
    _feature_summary_hover,
    _filter_numeric_for_corr,
    _importance_full_title,
)

# ── Re-exports: single-model importance ────────────────────────────────────
from mlsurv.viz._features_importance import (  # noqa: F401
    _build_importance_error_bars,
    _build_importance_subtitle,
    _create_bar_importance,
    _create_boxplot_importance,
    _create_forest_importance,
    _create_raincloud_importance,
    _dispatch_single_importance,
    _add_importance_forest_trace,
    _importance_legend_kw,
    _make_bar_trace,
    _plot_faceted_importance,
    _plot_group_agg_importance,
    _prepare_importance_data,
)

# ── Re-exports: multi-model importance ─────────────────────────────────────
from mlsurv.viz._features_importance_multi import (  # noqa: F401
    _add_heatmap_annotations,
    _bar_feature_order,
    _build_heatmap_matrix,
    _comparison_bar_traces,
    _comparison_forest_traces,
    _compute_ci_z,
    _format_importance_label,
    _plot_importance_comparison,
    _prepare_comparison_data,
    _render_heatmap_plain,
    _render_heatmap_with_groups,
    plot_importance_heatmap,
)

# ── Re-exports: coefficients ──────────────────────────────────────────────
from mlsurv.viz._features_coefficients import (  # noqa: F401
    _add_coef_comparison_annotations,
    _add_risk_direction_annotations,
    _coef_error_x,
    _plot_coef_multi_model,
    _plot_coef_single,
    _resolve_scatter_color,
    plot_coefficient_comparison,
    plot_coefficients,
)

# ── Re-exports: hazard ratios ─────────────────────────────────────────────
from mlsurv.viz._features_hazard_ratios import (  # noqa: F401
    _add_hr_category_traces,
    _add_hr_comparison_traces,
    _add_hr_risk_annotations,
    _build_hr_hover_ci,
    _collect_hr_data,
    _compute_hr_ticks,
    _error_bar_axis_label,
    _format_hr_tick,
    _hr_data_range,
    _hr_data_range_multi,
    _hr_error_x,
    _rank_features_by_abs_log_hr,
    _safe_extract,
    compute_hazard_ratios,
    plot_hazard_ratios,
    plot_hazard_ratios_comparison,
)

# ── Re-exports: correlation ───────────────────────────────────────────────
from mlsurv.viz._features_correlation import (  # noqa: F401
    _add_correlation_annotations,
    plot_feature_correlation,
)


# =============================================================================
# Dispatcher helpers (extracted from plot_importance)
# =============================================================================

def _plot_importance_multi_dispatch(
    df, n, importance_col, std_col, feature_col, model_col,
    show_error_bars, style, title, subtitle, sort, iterations_df,
    ci_level, xlim, error_bars, palette, feature_groups, group_colors,
    sort_by_group, width, height,
):
    """Handle multi-model importance dispatch."""
    if title is None:
        n_actual = df[feature_col].nunique()
        title = ("Feature Importance by Model" if n_actual <= n
                 else f"Top {n} Features: Importance by Model")
    fig = _plot_importance_comparison(
        df, models=None, n=n, importance_col=importance_col,
        std_col=std_col, show_error_bars=show_error_bars,
        model_col=model_col, feature_col=feature_col,
        title=_importance_full_title(title, subtitle),
        style=style, iterations_df=iterations_df, sort=sort,
        ci_level=ci_level, xlim=xlim, error_bars=error_bars,
        palette=palette,
        feature_groups=feature_groups, group_colors=group_colors,
        sort_by_group=sort_by_group,
    )
    apply_size_overrides(fig, width, height)
    return fig


def _plot_importance_single(
    df, n, importance_col, std_col, coef_col, feature_col,
    show_error_bars, style, title, subtitle, sort, has_iterations,
    iterations_df, error_bars, ci_level, xlim, show_labels,
    feat_to_grp, grp_to_color, facet_by_group, sort_by_group,
    width, height,
):
    """Handle single-model importance rendering."""
    top, top_feature_list, n_features = _prepare_importance_data(
        df, n, importance_col, feature_col, sort=sort,
    )

    # Narrow feat_to_grp and grp_to_color to the top features only
    feat_to_grp, grp_to_color = _narrow_groups(
        feat_to_grp, grp_to_color, top_feature_list,
    )

    # sort_by_group: reorder features by group, then by metric within group
    if sort_by_group and feat_to_grp and grp_to_color:
        metric_vals = dict(zip(
            top[feature_col], top[importance_col],
        ))
        top_feature_list = _sort_features_by_group(
            top_feature_list, feat_to_grp, grp_to_color, metric_values=metric_vals,
            reverse=True,
        )
        top = top.set_index(feature_col).reindex(top_feature_list).reset_index()

    # facet_by_group mode
    if facet_by_group and feat_to_grp and grp_to_color:
        return _plot_faceted_importance(
            df, n, importance_col, std_col, coef_col, feature_col,
            show_error_bars, style, title, subtitle, sort, error_bars, ci_level,
            xlim, show_labels, feat_to_grp, grp_to_color,
            iterations_df=iterations_df, width=width, height=height,
        )

    if title is None:
        title = ("Permutation Feature Importance" if n_features < n
                 else f"Top {n} Features by Permutation Importance")

    full_title = _importance_full_title(title, subtitle)
    d = MANUSCRIPT_DEFAULTS
    h_base = adaptive_size(n_features, per_item=30, base=60, min_size=d["height_small"])

    fig = _dispatch_single_importance(
        style, has_iterations, iterations_df, top_feature_list,
        top, importance_col, std_col, coef_col, feature_col,
        show_error_bars, d, h_base, full_title, xlim, show_labels,
        error_bars, ci_level, feat_to_grp, grp_to_color,
    )

    if sort_by_group and feat_to_grp and grp_to_color:
        _add_group_separator_shapes(fig, top_feature_list, feat_to_grp, axis="y")

    apply_size_overrides(fig, width, height)
    return fig


# =============================================================================
# Public API: plot_importance() — dispatcher
# =============================================================================

_VALID_IMPORTANCE_STYLES = {"auto", "bar", "forest", "raincloud", "boxplot"}


def _resolve_importance_style(style, is_multi, has_iterations):
    """Resolve ``"auto"`` importance style to a concrete style string.

    Returns the effective style string.
    """
    if style not in _VALID_IMPORTANCE_STYLES:
        raise ValueError(
            f"style={style!r} is not supported. "
            f"Must be one of: {', '.join(sorted(_VALID_IMPORTANCE_STYLES))}"
        )
    if style != "auto":
        return style
    if is_multi:
        return "bar"
    return "raincloud" if has_iterations else "bar"


def plot_importance(
    df: pd.DataFrame,
    style: str = "auto",
    n: int = 20,
    importance_col: str = "perm_mean",
    std_col: Optional[str] = "perm_std",
    coef_col: Optional[str] = None,
    feature_col: str = "feature",
    model_col: str = "model",
    title: Optional[str] = None,
    show_error_bars: bool = True,
    iterations_df: Optional[pd.DataFrame] = None,
    n_repeats: Optional[int] = None,
    sort: str = "value",
    orientation: str = "horizontal",
    error_bars: str = "sd",
    ci_level: float = 0.95,
    xlim: Optional[tuple] = None,
    show_labels: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
    palette=None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
    facet_by_group: bool = False,
    group_agg: bool = False,
    sort_by_group: bool = False,
    return_data: bool = False,
) -> Union[go.Figure, dict]:
    """Feature importance — single or multi-model, multiple display styles.

    Auto-detects multi-model when ``model_col`` has >1 unique value.

    Parameters
    ----------
    df : DataFrame
        Long-format importance data. Must contain ``feature_col`` and
        ``importance_col``. Include ``model_col`` for multi-model plots.

        **DataFrame schema** (default column names)::

            feature   perm_mean   perm_std (optional)   model (optional)
            str       float       float                str
    style : {"auto", "bar", "forest", "raincloud", "boxplot"}
        Display style.

        - ``"bar"`` — horizontal bars with error bars, solid blue by default.
        - ``"forest"`` — dot + error whiskers, one row per feature.
        - ``"raincloud"`` — half-violin + boxplot + jitter showing the
          per-iteration importance distribution. For multi-model data, one
          violin per model is grouped within each feature (the distribution
          analogue of the grouped bar). Needs ``iterations_df``; falls back to
          ``"bar"`` when per-iteration data is unavailable for the plotted models.
        - ``"boxplot"`` — box + jitter of the per-iteration distribution; for
          multi-model data, one box per model grouped within each feature.
          Needs ``iterations_df``; falls back to ``"bar"`` when per-iteration
          data is unavailable for the plotted models.
        - ``"auto"`` — for a single model, ``"raincloud"`` when ``iterations_df``
          is provided, otherwise ``"bar"``; for multiple models, always
          ``"bar"`` (request ``"raincloud"``/``"boxplot"`` explicitly for
          grouped per-model distributions).
    n : int
        Number of top features to display.
    importance_col : str
        Column with mean importance values.
    std_col : str, optional
        Column with standard deviations (error bars / CI half-width).
    coef_col : str, optional
        Column with coefficients for color-coding bars/dots by direction.
        Defaults to ``None`` (uniform blue). Pass ``"coefficient"`` to enable
        sign-based coloring when the data contains that column.
    feature_col : str
        Column with feature names.
    model_col : str
        Column with model names (multi-model detection).
    title : str, optional
        Plot title (auto-generated if ``None``).
    show_error_bars : bool
        Show error bars / whiskers where applicable.
    iterations_df : DataFrame, optional
        Per-iteration importance values for ``"raincloud"`` / ``"boxplot"``
        styles. Must contain ``"feature"`` and ``"importance"`` columns, plus a
        ``model_col`` column for multi-model plots (its absence triggers the
        ``"bar"`` fallback).
    n_repeats : int, optional
        Number of permutation repeats. Adds a subtitle showing the count.
    sort : {"value", "alpha"}
        Feature ordering. ``"value"`` — descending by importance (default).
        ``"alpha"`` — alphabetical A→Z from top to bottom.
    orientation : {"horizontal"}
        Plot orientation. Currently only horizontal is supported.
    error_bars : {"sd", "ci"}
        Error bar style for ``"bar"`` and ``"forest"`` styles.

        - ``"sd"`` (default) — ±1 SD whiskers; hover shows SD and ±SD range.
        - ``"ci"`` — ±z·SD whiskers at ``ci_level`` confidence; hover shows
          SD and CI bounds.
    ci_level : float
        Confidence level used when ``error_bars="ci"`` (default 0.95).
    xlim : tuple, optional
        ``(x_min, x_max)`` axis limits for the importance axis.
    show_labels : bool
        Show importance value labels on bars or dots.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    palette : dict, optional
        ``{model_name: hex_color}`` overrides for per-model colours.
    feature_groups : dict, list, Series, or None
        Feature-to-group mapping. See :func:`resolve_feature_group_colors`.
        Auto-detected from ``"group"``/``"feature_group"`` column if absent.
    group_colors : dict, optional
        Explicit ``{group: color}`` overrides.
    facet_by_group : bool
        When True and ``feature_groups`` is set, split into one subplot per
        group (``n`` applies per-group).
    group_agg : bool
        When True and ``feature_groups`` is set, aggregate importance by
        group (mean ± SD).  Requires ``feature_groups``.
    sort_by_group : bool
        When True and ``feature_groups`` is set, sort features by group
        (primary) and by importance within each group (secondary).
        Dotted separator lines are added between groups.
    return_data : bool, default=False
        If True, return ``{"figure": fig, "data": df}`` instead of just the
        figure.

    Returns
    -------
    go.Figure or dict

    Raises
    ------
    ValueError
        If ``group_agg=True`` is requested without ``feature_groups`` set.
    """
    validate_columns(df, [feature_col, importance_col], "plot_importance")

    # Auto-detect feature groups from column
    if feature_groups is None:
        feature_groups = _auto_detect_feature_groups(df, feature_col)

    # Resolve "auto" style
    has_iterations = iterations_df is not None and len(iterations_df) > 0
    # Multi-model detection (needed before style resolution)
    is_multi = model_col in df.columns and df[model_col].nunique() > 1

    style = _resolve_importance_style(style, is_multi, has_iterations)

    # Build subtitle
    has_std = bool(std_col and std_col in df.columns)
    subtitle = _build_importance_subtitle(n_repeats, has_std, error_bars=error_bars, ci_level=ci_level)

    # Validate group_agg
    if group_agg and feature_groups is None:
        raise ValueError("group_agg=True requires feature_groups to be set")
    if is_multi:
        fig = _plot_importance_multi_dispatch(
            df, n, importance_col, std_col, feature_col, model_col,
            show_error_bars, style, title, subtitle, sort, iterations_df,
            ci_level, xlim, error_bars, palette, feature_groups, group_colors,
            sort_by_group, width, height,
        )
        if return_data:
            return {"figure": fig, "data": df}
        return fig

    # --- Resolve groups once for the whole function ---
    all_features = df[feature_col].unique().tolist()
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, all_features, group_colors,
    )

    # --- group_agg mode: aggregate by group ---
    if group_agg:
        fig = _plot_group_agg_importance(
            df, feat_to_grp, grp_to_color, importance_col, feature_col,
            style, title, subtitle, xlim=xlim, width=width, height=height,
        )
        if return_data:
            return {"figure": fig, "data": df}
        return fig

    fig = _plot_importance_single(
        df, n, importance_col, std_col, coef_col, feature_col,
        show_error_bars, style, title, subtitle, sort, has_iterations,
        iterations_df, error_bars, ci_level, xlim, show_labels,
        feat_to_grp, grp_to_color, facet_by_group, sort_by_group,
        width, height,
    )
    if return_data:
        return {"figure": fig, "data": df}
    return fig
