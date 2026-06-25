"""Multi-model importance comparison and heatmap.

Contains ``_plot_importance_comparison``, ``plot_importance_heatmap``,
and supporting trace builders.
"""

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from mlsurv._warnings import InternalWarning
from mlsurv.viz._palettes import (
    FeatureGroupsType,
    MANUSCRIPT_DEFAULTS,
    _add_group_legend_traces,
    _add_group_separator_shapes,
    _make_group_strip_trace,
    _narrow_groups,
    _resolve_groups,
    _sort_features_by_group,
    get_model_color,
    hex_to_rgba,
)
from mlsurv.viz._plotly_helpers import symmetric_error_bar
from mlsurv.viz._theme import (
    adaptive_size, apply_size_overrides, create_figure,
    margin_for_labels, truncate_labels,
)
from mlsurv.viz._validation import validate_columns
from mlsurv.viz._features_helpers import _auto_detect_feature_groups


def _prepare_comparison_data(df, models, n, importance_col, feature_col, model_col,
                             sort, feature_groups, group_colors, sort_by_group):
    """Prepare top features and groups for multi-model importance comparison.

    Returns ``(models, mean_imp, top_features, top_features_sorted,
    feat_to_grp, grp_to_color)``.
    """
    if models is None:
        models = df[model_col].unique().tolist()

    all_features = df[feature_col].unique().tolist()
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, all_features, group_colors,
    )

    mean_imp = df.groupby(feature_col)[importance_col].mean()
    top_features = mean_imp.nlargest(n).index.tolist()

    feat_to_grp, grp_to_color = _narrow_groups(
        feat_to_grp, grp_to_color, top_features,
    )

    if sort == "alpha":
        top_features_sorted = sorted(top_features, reverse=True)
    else:
        top_features_sorted = mean_imp[top_features].sort_values(ascending=True).index.tolist()

    if sort_by_group and feat_to_grp and grp_to_color:
        metric_vals = mean_imp.to_dict()
        top_features_sorted = _sort_features_by_group(
            top_features_sorted, feat_to_grp, grp_to_color, metric_values=metric_vals,
            reverse=True,
        )

    return models, mean_imp, top_features, top_features_sorted, feat_to_grp, grp_to_color


def _compute_ci_z(error_bars, ci_level, has_std):
    """Return the z-multiplier for error bar width (1.0 for SD, z for CI)."""
    if error_bars == "ci" and has_std:
        from scipy.stats import norm as _norm
        return _norm.ppf((1 + ci_level) / 2)
    return 1.0


def _bar_feature_order(sort, sort_by_group, top_features, top_features_sorted,
                       mean_imp, feat_to_grp, grp_to_color):
    """Determine feature order for bar-style multi-model comparison."""
    has_groups = bool(sort_by_group and feat_to_grp and grp_to_color)
    if has_groups:
        return top_features_sorted[::-1]  # bar uses descending order
    if sort == "alpha":
        return sorted(top_features)
    return mean_imp[top_features].sort_values(ascending=False).index.tolist()


def _plot_importance_comparison(
    df, models, n, importance_col, std_col, show_error_bars,
    model_col, feature_col, title, style="bar", iterations_df=None,
    sort="value", ci_level=0.95, xlim=None, error_bars="sd",
    palette=None, feature_groups=None, group_colors=None,
    sort_by_group=False,
):
    """Multi-model importance comparison.

    Supports grouped ``bar``, grouped ``forest``, and the per-iteration
    distribution styles ``raincloud`` / ``boxplot`` (one violin/box per model
    grouped within each feature). Distribution styles need ``iterations_df``;
    when it is absent they fall back to ``bar``, matching the single-model path.
    """
    models, mean_imp, top_features, top_features_sorted, feat_to_grp, grp_to_color = (
        _prepare_comparison_data(
            df, models, n, importance_col, feature_col, model_col,
            sort, feature_groups, group_colors, sort_by_group,
        )
    )

    plot_df = df[df[feature_col].isin(top_features)].copy()

    d = MANUSCRIPT_DEFAULTS
    has_std = bool(std_col and std_col in plot_df.columns)
    h = adaptive_size(len(top_features), per_item=22, base=80, min_size=d["height_small"])
    margin_t = d["margin_t"] + (45 if "<br>" in (title or "") else 10)
    ci_pct = f"{ci_level * 100:g}"
    z = _compute_ci_z(error_bars, ci_level, has_std)
    has_groups = bool(sort_by_group and feat_to_grp and grp_to_color)

    if style == "forest":
        fig = _comparison_forest_traces(
            plot_df, models, top_features_sorted, model_col, feature_col,
            importance_col, std_col, show_error_bars, has_std, z, ci_pct,
            error_bars, d, h, margin_t, title, xlim, palette,
            feat_to_grp, grp_to_color,
        )
        if has_groups:
            _add_group_separator_shapes(fig, top_features_sorted, feat_to_grp, axis="y")
        return fig

    # --- raincloud / boxplot: per-model distributions; need iteration data ---
    if style in ("raincloud", "boxplot"):
        iter_top = _select_comparison_iterations(
            iterations_df, top_features, models, model_col, feature_col,
        )
        if iter_top is not None:
            h_dist = adaptive_size(
                len(top_features), per_item=34, base=90, min_size=d["height_small"],
            )
            fig = _comparison_distribution_traces(
                iter_top, models, top_features_sorted, model_col, feature_col,
                style, h_dist, margin_t, title, xlim, palette,
            )
            if has_groups:
                _add_group_separator_shapes(fig, top_features_sorted, feat_to_grp, axis="y")
            return fig
        warnings.warn(
            f"style='{style}' requires per-iteration importance data, but none "
            f"was available for the selected models (missing, or no matching "
            f"features/models). Falling back to 'bar' style.",
            InternalWarning,
            stacklevel=3,
        )
        style = "bar"  # fall through to bar

    # "bar" (default + distribution fallback) — reject any unknown style
    if style != "bar":
        raise ValueError(
            f"style='{style}' is not supported for multi-model comparison. "
            f"Supported styles: 'bar', 'forest', 'raincloud', 'boxplot'."
        )

    feature_order = _bar_feature_order(
        sort, sort_by_group, top_features, top_features_sorted,
        mean_imp, feat_to_grp, grp_to_color,
    )
    fig = _comparison_bar_traces(
        plot_df, feature_order, model_col, feature_col, importance_col,
        std_col, show_error_bars, has_std, z, ci_pct, error_bars,
        d, h, margin_t, title, xlim, palette, feat_to_grp, grp_to_color,
    )
    if has_groups:
        _add_group_separator_shapes(fig, feature_order, feat_to_grp, axis="x")
    return fig


def _select_comparison_iterations(iterations_df, top_features, models,
                                  model_col, feature_col, value_col="importance"):
    """Filter ``iterations_df`` to the plotted features and models.

    Returns the filtered frame, or ``None`` when per-iteration data is
    unavailable (missing frame, missing columns, or no matching rows) — the
    signal for the caller to fall back to ``bar``. The sole caller
    (:func:`_plot_importance_comparison`) MUST warn on a ``None`` return so the
    fallback is never silent.
    """
    if iterations_df is None or len(iterations_df) == 0:
        return None
    required = {model_col, feature_col, value_col}
    if not required.issubset(iterations_df.columns):
        return None
    sub = iterations_df[
        iterations_df[feature_col].isin(top_features)
        & iterations_df[model_col].isin(models)
    ]
    return sub.copy() if len(sub) > 0 else None


def _add_distribution_model_trace(fig, mdf, style, display_name, color,
                                  feature_col, value_col):
    """Add one model's grouped violin (raincloud) or box trace.

    The feature sits on the categorical (y) axis and importance on the value
    (x) axis; ``violinmode``/``boxmode`` ``"group"`` (set by the caller) offsets
    each model's trace within its feature band, the distribution analogue of the
    grouped bar.
    """
    hover = (
        f"<b>%{{y}}</b><br>Model: {display_name}<br>"
        f"Importance: %{{x:.4f}}<extra></extra>"
    )
    if style == "raincloud":
        fig.add_trace(go.Violin(
            y=mdf[feature_col], x=mdf[value_col], orientation="h",
            side="positive", scalemode="width", width=0.9,
            name=display_name, legendgroup=display_name,
            line_color=color, fillcolor=hex_to_rgba(color, 0.4),
            box_visible=True, meanline_visible=True,
            points="all", jitter=0.05, pointpos=-0.1,
            marker=dict(color=color, size=3, opacity=0.4),
            spanmode="hard", hovertemplate=hover,
        ))
    else:  # boxplot
        fig.add_trace(go.Box(
            y=mdf[feature_col], x=mdf[value_col], orientation="h",
            name=display_name, legendgroup=display_name,
            marker_color=color, line_color=color,
            fillcolor=hex_to_rgba(color, 0.3),
            boxmean=True, boxpoints="outliers", hovertemplate=hover,
        ))


def _warn_missing_distribution_data(missing_models, missing_features, style):
    """Warn (``InternalWarning``) about models/features dropped from a plot.

    The multi-model distribution renderer reaches here when iteration data
    covers only some of the requested models/features (``_select_comparison_iterations``
    gates on the union, so partial coverage passes). Mirrors the single-model
    raincloud/boxplot 'blank row' warnings so neither path drops data silently.
    """
    from mlsurv.viz._palettes import get_model_display_name

    if missing_models:
        names = ", ".join(get_model_display_name(m) for m in missing_models)
        warnings.warn(
            f"No per-iteration importance data for model(s) {names} among the "
            f"top features; they are omitted from the {style} comparison.",
            InternalWarning, stacklevel=3,
        )
    if missing_features:
        warnings.warn(
            f"{len(missing_features)} top feature(s) have no per-iteration data "
            f"for any model and appear as blank rows in the {style} comparison: "
            f"{missing_features[:5]}.",
            InternalWarning, stacklevel=3,
        )


def _comparison_distribution_traces(
    iter_df, models, top_features_sorted, model_col, feature_col,
    style, h, margin_t, title, xlim, palette,
):
    """Grouped per-model raincloud/boxplot of importance distributions.

    One violin/box per model, grouped within each feature band — the
    distribution counterpart of :func:`_comparison_bar_traces`. Features share
    a single categorical y-axis (no group color strip; the colour channel
    already encodes the model). ``feature_groups``/``sort_by_group`` still order
    the features (baked into ``top_features_sorted``) and draw dotted separators
    via the caller, so no group context is needed here.
    """
    from mlsurv.viz._palettes import get_model_display_name

    value_col = "importance"
    fig = create_figure(title=title, width="double", height=h)

    plotted_features = set()
    missing_models = []
    for model in models:
        mdf = iter_df[iter_df[model_col] == model]
        if mdf.empty:
            missing_models.append(model)
            continue
        plotted_features.update(mdf[feature_col].unique())
        _add_distribution_model_trace(
            fig, mdf, style, get_model_display_name(model),
            get_model_color(model, palette=palette), feature_col, value_col,
        )

    # Mirror the single-model 'blank row' warnings: any model or top feature
    # that contributes no trace is a silent data-loss path unless we signal it.
    missing_features = [f for f in top_features_sorted if f not in plotted_features]
    _warn_missing_distribution_data(missing_models, missing_features, style)

    display_labels, full_labels = truncate_labels(top_features_sorted)
    margin_l = margin_for_labels(display_labels)

    xaxis_kw = dict(title="Permutation Importance")
    if xlim is not None:
        xaxis_kw["range"] = list(xlim)

    fig.update_layout(
        violinmode="group", boxmode="group",
        margin=dict(l=margin_l, t=margin_t),
        xaxis=xaxis_kw, yaxis_title="",
        yaxis=dict(
            categoryorder="array", categoryarray=full_labels,
            tickmode="array", tickvals=full_labels, ticktext=display_labels,
        ),
        legend=dict(title=dict(text="Model")),
    )
    return fig


def _build_single_forest_trace(mdf, model, model_col, feature_col, importance_col,
                               std_col, show_error_bars, has_std, z, ci_pct,
                               error_bars, top_features_sorted, color,
                               n_models, mi, has_groups):
    """Build a single model's scatter trace kwargs for forest-style comparison.

    Returns ``(trace_kw, valid_feats)`` where *trace_kw* is a dict ready
    for ``go.Scatter(**trace_kw)`` and *valid_feats* is the list of features
    present in this model's data.
    """
    from mlsurv.viz._palettes import get_model_display_name

    display_name = get_model_display_name(model)
    jitter = (mi - (n_models - 1) / 2) * 0.25
    valid_feats = [f for f in mdf[feature_col] if f in top_features_sorted]
    mdf_idx = mdf.drop_duplicates(subset=feature_col).set_index(feature_col)
    y_vals = [top_features_sorted.index(f) + jitter for f in valid_feats]
    x_vals = mdf_idx.loc[valid_feats, importance_col].values

    error_x, customdata = None, None
    hover = "<b>%{text}</b><br>Importance: %{x:.4f}<extra></extra>"
    if show_error_bars and has_std:
        std_vals = mdf_idx.loc[valid_feats, std_col].values
        err_vals = std_vals * z
        error_x = symmetric_error_bar(err_vals, color=color)
        customdata = np.column_stack([std_vals, x_vals - err_vals, x_vals + err_vals])
        bounds_label = (f"{ci_pct}% CI: [%{{customdata[1]:.4f}}, %{{customdata[2]:.4f}}]"
                       if error_bars == "ci"
                       else "Range: [%{customdata[1]:.4f}, %{customdata[2]:.4f}]")
        hover = (
            f"<b>%{{text}}</b><br>Model: {display_name}<br>"
            f"Importance: %{{x:.4f}}<br>SD: %{{customdata[0]:.4f}}<br>"
            f"{bounds_label}<extra></extra>"
        )

    trace_kw = dict(
        x=x_vals, y=y_vals, mode="markers", name=display_name,
        marker=dict(color=color, size=8, symbol="circle"),
        error_x=error_x, hovertemplate=hover, text=valid_feats,
    )
    if customdata is not None:
        trace_kw["customdata"] = customdata
    if has_groups:
        trace_kw["legendgroup"] = "_models_"
        if mi == 0:
            trace_kw["legendgrouptitle"] = dict(text="Model")

    return trace_kw, valid_feats


def _comparison_forest_traces(
    plot_df, models, top_features_sorted, model_col, feature_col,
    importance_col, std_col, show_error_bars, has_std, z, ci_pct,
    error_bars, d, h, margin_t, title, xlim, palette,
    feat_to_grp, grp_to_color,
):
    """Build forest-style multi-model importance comparison."""
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
            _make_group_strip_trace(top_features_sorted, feat_to_grp, grp_to_color, orientation="vertical"),
            row=1, col=1,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        _forest_display, _ = truncate_labels(top_features_sorted)
        fig.update_yaxes(
            showgrid=False, showticklabels=True, tickmode="array",
            tickvals=list(range(len(top_features_sorted))),
            ticktext=_forest_display, row=1, col=1,
        )
        main_row, main_col = 1, 2
    else:
        fig = create_figure(title=title, width=d["width_double"], height=h)
        main_row, main_col = None, None

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5,
                   row=main_row, col=main_col)

    n_models = len(models)
    for mi, model in enumerate(models):
        mdf = plot_df[plot_df[model_col] == model]
        color = get_model_color(model, palette=palette)
        trace_kw, _valid = _build_single_forest_trace(
            mdf, model, model_col, feature_col, importance_col,
            std_col, show_error_bars, has_std, z, ci_pct,
            error_bars, top_features_sorted, color,
            n_models, mi, has_groups,
        )
        if has_groups:
            fig.add_trace(go.Scatter(**trace_kw), row=1, col=2)
        else:
            fig.add_trace(go.Scatter(**trace_kw))

    xaxis_kw = dict(title="Permutation Importance")
    if xlim is not None:
        xaxis_kw["range"] = list(xlim)
    display_labels, full_labels = truncate_labels(top_features_sorted)
    margin_l = margin_for_labels(display_labels)
    fig.update_layout(margin=dict(l=margin_l, t=margin_t))

    if has_groups:
        fig.update_xaxes(xaxis_kw, row=1, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=2)
        fig.update_layout(
            legend=dict(x=1.02, xanchor="left", y=0.5, yanchor="middle"),
        )
        _add_group_legend_traces(fig, grp_to_color, row=1, col=2)
    else:
        fig.update_layout(
            xaxis=xaxis_kw, yaxis_title="",
            yaxis=dict(tickmode="array",
                       tickvals=list(range(len(top_features_sorted))),
                       ticktext=display_labels),
            legend=dict(title=dict(text="Model")),
        )
    return fig


def _comparison_bar_traces(
    plot_df, feature_order, model_col, feature_col, importance_col,
    std_col, show_error_bars, has_std, z, ci_pct, error_bars,
    d, h, margin_t, title, xlim, palette, feat_to_grp, grp_to_color,
):
    """Build bar-style multi-model importance comparison."""
    # Truncate labels and compute bottom margin for -45° rotation
    display_labels, full_labels = truncate_labels(feature_order)
    margin_b = margin_for_labels(display_labels, per_char=5, max_margin=200)

    if has_std:
        plot_df = plot_df.copy()
        plot_df["_err"] = plot_df[std_col] * z
        plot_df["_lo"] = plot_df[importance_col] - plot_df["_err"]
        plot_df["_hi"] = plot_df[importance_col] + plot_df["_err"]
        error_y = "_err" if show_error_bars else None
        lo_label = f"{ci_pct}% CI lower" if error_bars == "ci" else "Lower"
        hi_label = f"{ci_pct}% CI upper" if error_bars == "ci" else "Upper"
        hover_data = {importance_col: ":.4f", std_col: ":.4f", "_lo": ":.4f", "_hi": ":.4f"}
        labels = {std_col: "SD", "_lo": lo_label, "_hi": hi_label, "_err": False}
    else:
        error_y = None
        hover_data = {importance_col: ":.4f"}
        labels = None

    model_names = plot_df[model_col].unique().tolist()
    color_map = {m: get_model_color(m, palette=palette) for m in model_names}
    has_groups = bool(feat_to_grp and grp_to_color)

    bar_fig = px.bar(
        plot_df,
        x=feature_col, y=importance_col, error_y=error_y, color=model_col,
        barmode="group", color_discrete_map=color_map,
        hover_data=hover_data, title=title,
        labels=labels,
        category_orders={feature_col: feature_order},
    )

    yaxis_kw = dict(title="Permutation Importance")
    if xlim is not None:
        yaxis_kw["range"] = list(xlim)

    if has_groups:
        from plotly.subplots import make_subplots
        strip_height = 0.04
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[1 - strip_height, strip_height],
            vertical_spacing=0.02,
        )
        for i, trace in enumerate(bar_fig.data):
            trace.legendgroup = "_models_"
            if i == 0:
                trace.legendgrouptitle = dict(text="Model")
            fig.add_trace(trace, row=1, col=1)
        fig.add_trace(
            _make_group_strip_trace(feature_order, feat_to_grp, grp_to_color, orientation="horizontal"),
            row=2, col=1,
        )
        # Sync category order: px.bar's category_orders is lost when
        # traces move to subplots, so set it explicitly on both axes.
        fig.update_xaxes(showticklabels=False, showgrid=False,
                         categoryorder="array", categoryarray=feature_order,
                         row=2, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
        fig.update_xaxes(
            tickangle=-45, tickmode="array",
            tickvals=full_labels, ticktext=display_labels,
            categoryorder="array", categoryarray=feature_order,
            row=1, col=1,
        )
        fig.update_yaxes(yaxis_kw, row=1, col=1)
        fig.update_layout(
            template="mlsurv_minimal", width=d["width_double"], height=h,
            margin=dict(b=margin_b, t=margin_t),
            title=dict(text=f"<b>{title}</b>"), xaxis_title="",
            legend=dict(x=1.02, xanchor="left", y=0.5, yanchor="middle"),
        )
        _add_group_legend_traces(fig, grp_to_color, row=1, col=1)
    else:
        fig = bar_fig
        fig.update_layout(
            template="mlsurv_minimal", width=d["width_double"], height=h,
            margin=dict(b=margin_b, t=margin_t),
            title=dict(text=f"<b>{title}</b>"), xaxis_title="",
            yaxis=yaxis_kw,
            xaxis=dict(
                tickangle=-45, tickmode="array",
                tickvals=full_labels, ticktext=display_labels,
            ),
            legend=dict(title=dict(text="Model")),
        )
    return fig


def _format_importance_label(value: float) -> str:
    """Format an importance value for heatmap cell annotation.

    Uses scientific notation for very small values, fewer decimals for
    large values, and three decimals for everything in between.
    """
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.1e}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _add_heatmap_annotations(fig, matrix, z, raw, d, xref=None, yref=None):
    """Add contrast-aware value annotations to a heatmap.

    Viridis luminance: dark purple/blue (low) -> teal (mid) -> yellow (high).
    White text is readable on the dark end (~0-55% of range),
    black text is readable on the bright end (~55-100%).
    """
    has_finite = np.any(np.isfinite(z))
    z_min = np.nanmin(z) if has_finite else 0
    z_max = np.nanmax(z) if has_finite else 1
    z_range = z_max - z_min if z_max != z_min else 1
    for i, feat in enumerate(matrix.index):
        for j, model in enumerate(matrix.columns):
            v = raw[i, j]
            if np.isnan(v):
                continue
            frac = (z[i, j] - z_min) / z_range
            ann_kw = dict(
                x=model, y=feat,
                text=_format_importance_label(v),
                showarrow=False,
                font=dict(
                    color="black" if frac > 0.55 else "white",
                    size=d["font_size_tick"] - 1,
                ),
            )
            if xref is not None:
                ann_kw["xref"] = xref
                ann_kw["yref"] = yref
            fig.add_annotation(**ann_kw)


def _build_heatmap_matrix(df, n, importance_col, feature_col, model_col,
                          sort, model_sort, feature_groups, group_colors,
                          sort_by_group):
    """Build the pivoted feature × model matrix with ordering applied.

    Returns ``(matrix, feature_order, top_features, raw, z_not_computed)``.
    """
    from mlsurv.viz._palettes import get_model_display_name

    mean_imp = df.groupby(feature_col)[importance_col].mean()
    top_features = mean_imp.nlargest(n).index.tolist()
    plot_df = df[df[feature_col].isin(top_features)].copy()

    matrix = plot_df.pivot_table(
        index=feature_col, columns=model_col,
        values=importance_col, aggfunc="mean",
    )

    # Feature ordering (rows)
    if sort == "alpha":
        feature_order = sorted(top_features, reverse=True)
    else:
        feature_order = mean_imp[top_features].sort_values(ascending=True).index.tolist()

    # sort_by_group overrides feature ordering
    if sort_by_group and feature_groups is not None:
        _ftg, _gtc = _resolve_groups(feature_groups, top_features, group_colors)
        if _ftg and _gtc:
            metric_vals = mean_imp[top_features].to_dict()
            feature_order = _sort_features_by_group(
                top_features, _ftg, _gtc, metric_values=metric_vals,
                reverse=True,
            )

    matrix = matrix.reindex(feature_order)

    # Model ordering (columns)
    if model_sort == "value":
        model_means = matrix.mean(axis=0).sort_values(ascending=False)
        matrix = matrix[model_means.index]
    matrix.columns = [get_model_display_name(m) for m in matrix.columns]
    if model_sort == "alpha":
        matrix = matrix[sorted(matrix.columns, key=str.lower)]

    return matrix, feature_order, top_features


def _render_heatmap_with_groups(matrix, feature_order, z, raw, colorbar_title,
                                annotations, title, w, h, margin_l, d,
                                feat_to_grp, grp_to_color, display_labels=None):
    """Render heatmap with a group color strip on the left."""
    from plotly.subplots import make_subplots

    strip_width = 0.05
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[strip_width, 1 - strip_width],
        horizontal_spacing=0.01,
    )
    fig.add_trace(
        _make_group_strip_trace(feature_order, feat_to_grp, grp_to_color, orientation="vertical"),
        row=1, col=1,
    )
    fig.add_trace(go.Heatmap(
        z=z, x=matrix.columns.tolist(), y=matrix.index.tolist(),
        customdata=raw, colorscale="Viridis",
        colorbar=dict(title=dict(text=colorbar_title, side="right"), x=1.02),
        hovertemplate=(
            "<b>%{y}</b><br>Model: %{x}<br>"
            "Importance: %{customdata:.4f}<extra></extra>"
        ),
        hoverongaps=False,
    ), row=1, col=2)

    if annotations:
        _add_heatmap_annotations(fig, matrix, z, raw, d, xref="x2", yref="y2")
    _add_group_legend_traces(fig, grp_to_color, row=1, col=2)

    fig.update_layout(
        template="mlsurv_minimal", title=dict(text=f"<b>{title}</b>"),
        width=w + 200, height=h,
        margin=dict(l=margin_l, b=d["margin_b"] + 30, t=d["margin_t"] + 10, r=250),
        showlegend=True,
        legend=dict(x=1.60, xanchor="left", y=0.5, yanchor="middle"),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
    _y1_kw = dict(showgrid=False)
    if display_labels is not None:
        _y1_kw.update(tickmode="array", tickvals=feature_order, ticktext=display_labels)
    fig.update_yaxes(_y1_kw, row=1, col=1)
    fig.update_xaxes(side="bottom", tickangle=-30, showgrid=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=2)
    return fig


def _render_heatmap_plain(matrix, z, raw, colorbar_title, annotations,
                          title, w, h, margin_l, d, display_labels=None):
    """Render heatmap without feature group coloring."""
    fig = go.Figure(go.Heatmap(
        z=z, x=matrix.columns.tolist(), y=matrix.index.tolist(),
        customdata=raw, colorscale="Viridis",
        colorbar=dict(title=dict(text=colorbar_title, side="right")),
        hovertemplate=(
            "<b>%{y}</b><br>Model: %{x}<br>"
            "Importance: %{customdata:.4f}<extra></extra>"
        ),
        hoverongaps=False,
    ))
    if annotations:
        _add_heatmap_annotations(fig, matrix, z, raw, d)

    fig.update_layout(
        template="mlsurv_minimal", title=dict(text=f"<b>{title}</b>"),
        width=w, height=h,
        margin=dict(l=margin_l, b=d["margin_b"] + 30, t=d["margin_t"] + 10),
        xaxis=dict(side="bottom", tickangle=-30, showgrid=False),
        yaxis=dict(
            showgrid=False,
            **(dict(tickmode="array",
                    tickvals=matrix.index.tolist(),
                    ticktext=display_labels) if display_labels is not None else {}),
        ),
    )
    return fig


def plot_importance_heatmap(
    df: pd.DataFrame,
    n: int = 20,
    importance_col: str = "perm_mean",
    feature_col: str = "feature",
    model_col: str = "model",
    log_scale: bool = True,
    sort: str = "value",
    model_sort: str = "alpha",
    annotations: bool = True,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
    sort_by_group: bool = False,
) -> go.Figure:
    """Feature × model importance heatmap.

    Parameters
    ----------
    df : DataFrame
        Long-format importance data with ``feature_col``, ``model_col``,
        and ``importance_col``.
    n : int
        Number of top features ranked by mean importance across all models
        (not per-model).
    importance_col : str
        Column with mean importance values.
    feature_col : str
        Column name for features.
    model_col : str
        Column name for models.
    log_scale : bool
        Apply ``log(1 + x)`` transform to the color scale. Useful when a
        few features dominate and compress the colour range.
    sort : ``"value"`` | ``"alpha"``
        Feature row ordering. ``"value"`` (default) sorts by mean importance
        (highest at top). ``"alpha"`` sorts alphabetically A–Z top-to-bottom.
    model_sort : ``"alpha"`` | ``"value"``
        Model column ordering. ``"alpha"`` (default) sorts by display name
        alphabetically. ``"value"`` sorts by descending mean importance across
        the top-N features (highest-contributing model on the left).
    annotations : bool
        Show formatted importance values on each cell. Adaptive formatting
        uses scientific notation for very small values and fewer decimals for
        large values.
    title : str, optional
        Plot title.
    width : int, optional
        Override default figure width (pixels).
    height : int, optional
        Override default figure height (pixels).
    feature_groups : dict, list, Series, or None
        Feature-to-group mapping. See :func:`resolve_feature_group_colors`.
    group_colors : dict, optional
        Explicit ``{group: color}`` overrides.
    sort_by_group : bool, default=False
        When True and ``feature_groups`` is set, order feature rows by group
        (primary) and by mean importance within each group (secondary).

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``df`` is missing any of the required columns, or if ``sort`` or
        ``model_sort`` is not ``"value"`` or ``"alpha"``.
    """
    validate_columns(df, [feature_col, model_col, importance_col], "plot_importance_heatmap")

    if feature_groups is None:
        feature_groups = _auto_detect_feature_groups(df, feature_col)

    valid_sorts = ("value", "alpha")
    if sort not in valid_sorts:
        raise ValueError(f"sort must be one of {valid_sorts}, got {sort!r}")
    if model_sort not in valid_sorts:
        raise ValueError(f"model_sort must be one of {valid_sorts}, got {model_sort!r}")

    matrix, feature_order, top_features = _build_heatmap_matrix(
        df, n, importance_col, feature_col, model_col,
        sort, model_sort, feature_groups, group_colors, sort_by_group,
    )

    raw = matrix.values
    z = np.log1p(np.maximum(raw, 0)) if log_scale else raw
    colorbar_title = "log(1 + importance)" if log_scale else "Importance"

    d = MANUSCRIPT_DEFAULTS
    display_feat_labels, _ = truncate_labels(feature_order)
    margin_l = margin_for_labels(display_feat_labels)
    n_actual = len(top_features)
    if title is None:
        title = ("Feature Importance by Model" if n_actual < n
                 else f"Top {n} Features: Importance by Model")
    n_models = len(matrix.columns)
    w = adaptive_size(n_models, per_item=60, base=margin_l + 80,
                      min_size=280, max_size=1600)
    h = adaptive_size(len(feature_order), per_item=25, base=80,
                      min_size=d["height_small"])

    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, top_features, group_colors,
    )

    if feat_to_grp and grp_to_color:
        fig = _render_heatmap_with_groups(
            matrix, feature_order, z, raw, colorbar_title,
            annotations, title, w, h, margin_l, d,
            feat_to_grp, grp_to_color,
            display_labels=display_feat_labels,
        )
    else:
        fig = _render_heatmap_plain(
            matrix, z, raw, colorbar_title, annotations,
            title, w, h, margin_l, d,
            display_labels=display_feat_labels,
        )

    apply_size_overrides(fig, width, height)
    return fig
