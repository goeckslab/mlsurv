"""Single-model importance renderers.

Bar, forest, raincloud, and boxplot styles for feature importance
from a single model. Also includes faceted and group-aggregated modes.
"""

import warnings

import numpy as np
import plotly.graph_objects as go

from mlsurv._warnings import InternalWarning
from mlsurv.viz._plotly_helpers import symmetric_error_bar
from mlsurv.viz._palettes import (
    MANUSCRIPT_DEFAULTS,
    _add_group_legend_traces,
    _iter_group_segments,
    hex_to_rgba,
)
from mlsurv.viz._theme import (
    adaptive_size, apply_size_overrides, create_figure,
    margin_for_labels, truncate_labels,
)
from mlsurv.viz._features_helpers import (
    _apply_group_layout,
    _assign_coef_sign,
    _feature_color,
    _feature_summary_hover,
    _importance_full_title,
)


def _prepare_importance_data(df, n, importance_col, feature_col, sort="value"):
    """Select and sort top N features, return (top_df, feature_list, n_features).

    Parameters
    ----------
    sort : {"value", "alpha"}
        ``"value"`` — descending by importance (highest feature at chart top).
        ``"alpha"`` — alphabetical A→Z from chart top to bottom.
    """
    top = df.nlargest(n, importance_col).copy()
    n_features = len(top)
    if sort == "alpha":
        # Descending name order → chart shows names A (top) … Z (bottom)
        top = top.sort_values(feature_col, ascending=False)
    else:
        # Ascending importance → chart shows highest importance at top
        top = top.sort_values(importance_col, ascending=True)
    return top, top[feature_col].tolist(), n_features


def _build_importance_subtitle(n_repeats, has_std, error_bars="sd", ci_level=0.95):
    """Build a subtitle string for importance plots, e.g. '50 permutation repeats · ±SD'."""
    parts = []
    if n_repeats is not None:
        parts.append(f"{n_repeats} permutation repeats")
    if has_std:
        if error_bars == "ci":
            parts.append(f"\u00b1{ci_level * 100:g}% CI")
        else:
            parts.append("\u00b1SD")
    return " \u00b7 ".join(parts) if parts else None


def _create_raincloud_importance(iter_data, top_feature_list, d, h, title, xlim=None,
                                  feat_to_grp=None, grp_to_color=None):
    """Raincloud (half-violin + boxplot + jitter) for importance distributions."""
    fig = create_figure(title=title, width="single", height=h)
    has_groups = bool(feat_to_grp and grp_to_color)
    plotted_features = []
    for feature in top_feature_list:
        feat_data = iter_data[iter_data["feature"] == feature]["importance"].values
        if len(feat_data) == 0:
            warnings.warn(
                f"Feature '{feature}' is in the top importance list but has "
                f"no iteration data. It will appear as a blank row.",
                InternalWarning,
                stacklevel=2,
            )
            continue

        plotted_features.append(feature)
        color = _feature_color(feature, feat_to_grp, grp_to_color)
        fill_color = hex_to_rgba(color, 0.3)
        fill_color_box = hex_to_rgba(color, 0.6)
        summary_hover = _feature_summary_hover(feature, feat_data)

        fig.add_trace(go.Violin(
            y=[feature] * len(feat_data), x=feat_data,
            orientation="h", side="positive", name=feature,
            line_color=color, fillcolor=fill_color,
            meanline_visible=False, showlegend=False,
            scalemode="width", width=0.8, hoverinfo="skip",
        ))

        # Transparent hover overlay across violin x-range
        grid = np.linspace(float(feat_data.min()), float(feat_data.max()), 20)
        fig.add_trace(go.Scatter(
            x=grid, y=[feature] * len(grid),
            mode="markers",
            marker=dict(size=12, opacity=0, color=color),
            showlegend=False, name=feature,
            hovertemplate=summary_hover,
        ))

        fig.add_trace(go.Box(
            y=[feature] * len(feat_data), x=feat_data,
            orientation="h", name=feature,
            marker_color=color, line_color=color,
            fillcolor=fill_color_box,
            boxmean=True, showlegend=False, width=0.3, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            y=[feature] * len(feat_data), x=feat_data,
            mode="markers", name=feature,
            marker=dict(color=color, size=4, opacity=0.6),
            showlegend=False,
            hovertemplate=f"<b>{feature}</b><br>Importance: %{{x:.4f}}<extra></extra>",
        ))

    _apply_group_layout(fig, has_groups, grp_to_color, d, plotted_features, xlim)
    return fig


def _create_boxplot_importance(iter_data, top_feature_list, d, h, title, xlim=None,
                                feat_to_grp=None, grp_to_color=None):
    """Boxplot + jitter for importance distributions."""
    fig = create_figure(title=title, width="single", height=h)
    has_groups = bool(feat_to_grp and grp_to_color)
    plotted_features = []
    for feature in top_feature_list:
        feat_data = iter_data[iter_data["feature"] == feature]["importance"].values
        if len(feat_data) == 0:
            warnings.warn(
                f"Feature '{feature}' is in the top importance list but has "
                f"no iteration data. It will appear as a blank row.",
                InternalWarning,
                stacklevel=2,
            )
            continue

        plotted_features.append(feature)
        color = _feature_color(feature, feat_to_grp, grp_to_color)
        fill_color = hex_to_rgba(color, 0.3)
        summary_hover = _feature_summary_hover(feature, feat_data)
        median_val = float(np.median(feat_data))

        # Box (no SD indicator -- avoids violin-like +/-SD lines)
        fig.add_trace(go.Box(
            y=[feature] * len(feat_data), x=feat_data,
            orientation="h", name=feature,
            marker_color=color, line_color=color,
            fillcolor=fill_color,
            boxmean=True, boxpoints=False,
            showlegend=False, width=0.4, hoverinfo="skip",
        ))

        # Summary hover overlay (invisible marker at median)
        fig.add_trace(go.Scatter(
            x=[median_val], y=[feature],
            mode="markers",
            marker=dict(size=20, opacity=0, color=color),
            showlegend=False, name=feature,
            hovertemplate=summary_hover,
        ))

        # Jitter strip
        fig.add_trace(go.Scatter(
            y=[feature] * len(feat_data), x=feat_data,
            mode="markers", name=feature,
            marker=dict(color=color, size=4, opacity=0.5),
            showlegend=False,
            hovertemplate=f"<b>{feature}</b><br>Importance: %{{x:.4f}}<extra></extra>",
        ))

    _apply_group_layout(fig, has_groups, grp_to_color, d, plotted_features, xlim)
    return fig


def _build_importance_error_bars(sub_df, importance_col, std_col, has_std,
                                 show_error_bars, z, ci_pct, error_bars, color):
    """Build error bar dict, customdata, and hover for importance traces.

    Returns (error_x, customdata, hover).
    """
    base_hover = "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    if not (show_error_bars and has_std):
        return None, None, base_hover

    std_vals = sub_df[std_col].values
    err_vals = std_vals * z
    lo = sub_df[importance_col].values - err_vals
    hi = sub_df[importance_col].values + err_vals
    error_x = symmetric_error_bar(err_vals, color=color)
    customdata = np.column_stack([std_vals, lo, hi])
    if error_bars == "ci":
        bounds_label = f"{ci_pct}% CI: [%{{customdata[1]:.4f}}, %{{customdata[2]:.4f}}]"
    else:
        bounds_label = "Range: [%{customdata[1]:.4f}, %{customdata[2]:.4f}]"
    hover = (
        "<b>%{y}</b><br>Importance: %{x:.4f}<br>"
        "SD: %{customdata[0]:.4f}<br>"
        f"{bounds_label}<extra></extra>"
    )
    return error_x, customdata, hover


def _add_importance_forest_trace(fig, sub_df, importance_col, feature_col,
                                 name, color, error_x, customdata, hover,
                                 show_labels):
    """Add a single forest scatter trace for importance."""
    text_vals = sub_df[importance_col].round(3).astype(str).tolist() if show_labels else None
    trace_kw = dict(
        x=sub_df[importance_col], y=sub_df[feature_col],
        mode="markers+text" if show_labels else "markers",
        name=name, marker=dict(color=color, size=9, symbol="circle"),
        error_x=error_x, text=text_vals, textposition="middle right",
        hovertemplate=hover,
    )
    if customdata is not None:
        trace_kw["customdata"] = customdata
    fig.add_trace(go.Scatter(**trace_kw))


def _importance_legend_kw(legend_title):
    """Return legend kwargs for importance forest/bar plots."""
    if legend_title == "Feature Group":
        return dict(
            title=dict(text=legend_title),
            x=1.02, xanchor="left", y=0.5, yanchor="middle",
        )
    return dict(
        title=dict(text=legend_title),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    )


def _create_forest_importance(top, importance_col, std_col, coef_col, feature_col,
                               show_error_bars, d, h, title, ci_level=0.95,
                               xlim=None, show_labels=False, error_bars="sd",
                               feat_to_grp=None, grp_to_color=None):
    """Forest-style dot + error whiskers for feature importance."""
    z = 1.0
    if error_bars == "ci":
        from scipy.stats import norm as _norm
        z = _norm.ppf((1 + ci_level) / 2)
    ci_pct = f"{ci_level * 100:g}"

    fig = create_figure(title=title, width="single", height=h)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)

    has_std = std_col and std_col in top.columns
    feature_list = top[feature_col].tolist()

    if feat_to_grp and grp_to_color:
        for grp, color, grp_feats in _iter_group_segments(
            feature_list, feat_to_grp, grp_to_color,
        ):
            grp_df = top[top[feature_col].isin(grp_feats)]
            if grp_df.empty:
                continue
            error_x, customdata, hover = _build_importance_error_bars(
                grp_df, importance_col, std_col, has_std,
                show_error_bars, z, ci_pct, error_bars, color,
            )
            _add_importance_forest_trace(fig, grp_df, importance_col, feature_col,
                                         grp, color, error_x, customdata, hover,
                                         show_labels)
        show_legend = True
        legend_title = "Feature Group"
    else:
        color_map = _assign_coef_sign(top, coef_col)
        n_groups = len(top["coef_sign"].unique())
        for sign in top["coef_sign"].unique():
            sign_df = top[top["coef_sign"] == sign]
            color = color_map.get(sign, "#7f7f7f")
            error_x, customdata, hover = _build_importance_error_bars(
                sign_df, importance_col, std_col, has_std,
                show_error_bars, z, ci_pct, error_bars, color,
            )
            _add_importance_forest_trace(fig, sign_df, importance_col, feature_col,
                                         sign, color, error_x, customdata, hover,
                                         show_labels)
        show_legend = (n_groups > 1)
        legend_title = "Effect"

    display_labels, full_labels = truncate_labels(feature_list)
    margin_l = margin_for_labels(display_labels)

    xaxis_kw = dict(title="Permutation Importance")
    if xlim is not None:
        xaxis_kw["range"] = list(xlim)

    fig.update_layout(
        margin=dict(l=margin_l, t=d["margin_t"] + 10),
        xaxis=xaxis_kw, yaxis_title="",
        yaxis=dict(
            categoryorder="array", categoryarray=full_labels,
            tickmode="array", tickvals=full_labels, ticktext=display_labels,
        ),
        showlegend=show_legend,
        legend=_importance_legend_kw(legend_title),
    )
    return fig


def _make_bar_trace(sign_df, sign, color, importance_col, std_col, coef_col,
                    feature_col, show_error_bars, error_bars="sd", ci_level=0.95, z=1.0):
    """Build a single go.Bar trace for one coefficient-sign group.

    Hover shows SD and ±1SD range (``error_bars="sd"``) or SD and CI bounds
    (``error_bars="ci"``). ``z`` must be pre-computed by the caller.
    """
    has_std = std_col and std_col in sign_df.columns
    has_coef = coef_col and coef_col in sign_df.columns

    error_x = None
    customdata = None
    hover = "<b>%{y}</b><br>Importance: %{x:.4f}"

    if show_error_bars and has_std:
        std_vals = sign_df[std_col].values
        err_vals = std_vals * z
        lo = sign_df[importance_col].values - err_vals
        hi = sign_df[importance_col].values + err_vals
        error_x = symmetric_error_bar(err_vals)
        cd_arrays = [std_vals, lo, hi]
        if has_coef:
            cd_arrays.append(sign_df[coef_col].values)
        customdata = np.column_stack(cd_arrays)

        hover += "<br>SD: %{customdata[0]:.4f}"
        if error_bars == "ci":
            ci_pct = f"{ci_level * 100:g}"
            hover += f"<br>{ci_pct}% CI: [%{{customdata[1]:.4f}}, %{{customdata[2]:.4f}}]"
        else:
            hover += "<br>Range: [%{customdata[1]:.4f}, %{customdata[2]:.4f}]"
        if has_coef:
            hover += "<br>Coefficient: %{customdata[3]:.4f}"
    elif has_coef:
        customdata = sign_df[[coef_col]].values
        hover += "<br>Coefficient: %{customdata[0]:.4f}"

    return go.Bar(
        x=sign_df[importance_col], y=sign_df[feature_col],
        orientation="h", name=sign, marker_color=color,
        error_x=error_x,
        hovertemplate=hover + "<extra></extra>",
        customdata=customdata,
    )


def _create_bar_importance(top, importance_col, std_col, coef_col, feature_col,
                           show_error_bars, d, h, title, xlim=None, show_labels=False,
                           error_bars="sd", ci_level=0.95,
                           feat_to_grp=None, grp_to_color=None):
    """Horizontal bar chart for feature importance."""
    if error_bars == "sd":
        z = 1.0
    else:
        from scipy.stats import norm as _norm
        z = _norm.ppf((1 + ci_level) / 2)

    fig = create_figure(title=title, width="single", height=h)
    feature_list = top[feature_col].tolist()

    if feat_to_grp and grp_to_color:
        # Group coloring: one trace per group for legend
        for grp, color, grp_feats in _iter_group_segments(
            feature_list, feat_to_grp, grp_to_color,
        ):
            grp_rows = top[top[feature_col].isin(grp_feats)]
            if grp_rows.empty:
                continue
            fig.add_trace(_make_bar_trace(
                grp_rows, grp, color, importance_col, std_col, coef_col,
                feature_col, show_error_bars, error_bars=error_bars, ci_level=ci_level, z=z,
            ))
        show_legend = True
        legend_title = "Feature Group"
    else:
        color_map = _assign_coef_sign(top, coef_col)
        n_groups = len(top["coef_sign"].unique())
        for sign in top["coef_sign"].unique():
            sign_df = top[top["coef_sign"] == sign]
            color = color_map.get(sign, "#7f7f7f")
            fig.add_trace(_make_bar_trace(
                sign_df, sign, color, importance_col, std_col, coef_col,
                feature_col, show_error_bars, error_bars=error_bars, ci_level=ci_level, z=z,
            ))
        show_legend = (n_groups > 1)
        legend_title = "Effect"

    if show_labels:
        fig.update_traces(texttemplate="%{x:.3f}", textposition="outside")

    display_labels, full_labels = truncate_labels(feature_list)
    margin_l = margin_for_labels(display_labels)

    xaxis_kw = dict(title="Permutation Importance")
    if xlim is not None:
        xaxis_kw["range"] = list(xlim)

    fig.update_layout(
        margin=dict(l=margin_l, t=d["margin_t"] + 10),
        xaxis=xaxis_kw, yaxis_title="",
        yaxis=dict(
            categoryorder="array", categoryarray=full_labels,
            tickmode="array", tickvals=full_labels, ticktext=display_labels,
        ),
        barmode="relative",
        showlegend=show_legend,
        legend=_importance_legend_kw(legend_title),
    )
    return fig


def _dispatch_single_importance(
    style, has_iterations, iterations_df, top_feature_list,
    top, importance_col, std_col, coef_col, feature_col,
    show_error_bars, d, h_base, full_title, xlim, show_labels,
    error_bars, ci_level, feat_to_grp, grp_to_color,
):
    """Dispatch single-model importance to the appropriate renderer.

    For raincloud/boxplot, validates iteration data availability and
    falls back to bar style if missing.
    """
    # --- raincloud / boxplot: require iteration data ---
    if style in ("raincloud", "boxplot"):
        iter_data = None
        if has_iterations:
            iter_data = iterations_df[
                iterations_df["feature"].isin(top_feature_list)
            ].copy()
        if iter_data is not None and len(iter_data) > 0:
            h = int(h_base * 1.1) if style == "raincloud" else h_base
            creator = (_create_raincloud_importance if style == "raincloud"
                       else _create_boxplot_importance)
            return creator(
                iter_data, top_feature_list, d, h, full_title, xlim=xlim,
                feat_to_grp=feat_to_grp, grp_to_color=grp_to_color,
            )

        if has_iterations:
            iter_features = set(iterations_df["feature"].unique())
            top_set = set(top_feature_list)
            if iter_features and top_set and not iter_features & top_set:
                reason = (
                    f"iterations_df was provided but none of its features "
                    f"match the top importance list. "
                    f"iterations_df features (sample): "
                    f"{sorted(iter_features)[:3]}, "
                    f"top features (sample): {list(top_feature_list)[:3]}. "
                    f"Check for name mismatches (casing, whitespace)"
                )
            else:
                reason = ("iterations_df was provided but contains no matching "
                          "features for the top importance list")
        else:
            reason = ("iterations_df with per-iteration importance values "
                      "was not provided")
        warnings.warn(
            f"style='{style}' requires per-iteration importance data, "
            f"but {reason}. Falling back to 'bar' style.",
            InternalWarning,
            stacklevel=3,
        )
        # Fall through to bar

    # --- forest ---
    if style == "forest":
        return _create_forest_importance(
            top, importance_col, std_col, coef_col, feature_col,
            show_error_bars, d, h_base, full_title, ci_level=ci_level,
            xlim=xlim, show_labels=show_labels, error_bars=error_bars,
            feat_to_grp=feat_to_grp, grp_to_color=grp_to_color,
        )

    # --- bar (default + fallback) ---
    return _create_bar_importance(
        top, importance_col, std_col, coef_col, feature_col,
        show_error_bars, d, h_base, full_title, xlim=xlim,
        show_labels=show_labels, error_bars=error_bars, ci_level=ci_level,
        feat_to_grp=feat_to_grp, grp_to_color=grp_to_color,
    )


def _plot_faceted_importance(
    df, n, importance_col, std_col, coef_col, feature_col,
    show_error_bars, style, title, subtitle, sort, error_bars, ci_level,
    xlim, show_labels, feat_to_grp, grp_to_color,
    iterations_df=None, width=None, height=None,
):
    """Faceted importance: one subplot row per group."""
    from plotly.subplots import make_subplots

    # Group the features
    groups_ordered = list(dict.fromkeys(feat_to_grp.values()))
    n_groups = len(groups_ordered)

    d = MANUSCRIPT_DEFAULTS
    fig = make_subplots(
        rows=n_groups, cols=1, shared_xaxes=True,
        subplot_titles=groups_ordered, vertical_spacing=0.08,
    )

    for row_idx, grp in enumerate(groups_ordered, 1):
        grp_feats = [f for f, g in feat_to_grp.items() if g == grp]
        grp_df = df[df[feature_col].isin(grp_feats)]
        if grp_df.empty:
            continue

        top, top_feats, _ = _prepare_importance_data(
            grp_df, n, importance_col, feature_col, sort=sort,
        )
        color = grp_to_color.get(grp, "#7f7f7f")

        for _, row in top.iterrows():
            fig.add_trace(go.Bar(
                x=[row[importance_col]], y=[row[feature_col]],
                orientation="h", marker_color=color,
                showlegend=False,
                hovertemplate=f"<b>{row[feature_col]}</b><br>Importance: {row[importance_col]:.4f}<extra></extra>",
            ), row=row_idx, col=1)

        grp_display, grp_full = truncate_labels(top_feats)
        fig.update_yaxes(
            categoryorder="array", categoryarray=grp_full,
            tickmode="array", tickvals=grp_full, ticktext=grp_display,
            row=row_idx, col=1,
        )

    if title is None:
        title = "Feature Importance by Group"
    full_title = _importance_full_title(title, subtitle)

    total_feats = sum(
        min(n, len([f for f, g in feat_to_grp.items() if g == grp]))
        for grp in groups_ordered
    )
    # Compute adaptive margin from all feature names across groups
    all_grp_feats = [f for f in feat_to_grp.keys()]
    facet_display, _ = truncate_labels(all_grp_feats)
    facet_margin_l = margin_for_labels(facet_display)

    h = adaptive_size(total_feats, per_item=25, base=80 + n_groups * 30, min_size=d["height_medium"])
    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(text=f"<b>{full_title}</b>"),
        height=h, width=d["width_single"],
        margin=dict(l=facet_margin_l, t=d["margin_t"] + 50),
    )
    _add_group_legend_traces(fig, grp_to_color, row=1, col=1)
    fig.update_layout(showlegend=True, legend=dict(x=1.02, xanchor="left", y=0.5, yanchor="middle"))
    apply_size_overrides(fig, width, height)
    return fig


def _plot_group_agg_importance(
    df, feat_to_grp, grp_to_color, importance_col, feature_col,
    style, title, subtitle, xlim=None, width=None, height=None,
):
    """Aggregate importance by group: one bar/box per group."""
    d = MANUSCRIPT_DEFAULTS

    # Map each feature's importance to its group
    df_mapped = df[[feature_col, importance_col]].copy()
    df_mapped["_group"] = df_mapped[feature_col].map(feat_to_grp)
    df_mapped = df_mapped.dropna(subset=["_group"])
    if df_mapped.empty:
        raise ValueError(
            "No features matched any group in feature_groups. "
            "Check that feature names in the DataFrame match the feature_groups keys."
        )

    # Summary per group
    grp_stats = df_mapped.groupby("_group")[importance_col].agg(["mean", "std"]).reset_index()
    grp_stats = grp_stats.sort_values("mean", ascending=True)
    group_order = grp_stats["_group"].tolist()

    if title is None:
        title = "Feature Importance by Group (Aggregated)"
    full_title = _importance_full_title(title, subtitle)

    n_groups = len(group_order)
    h = adaptive_size(n_groups, per_item=40, base=80, min_size=d["height_small"])
    fig = create_figure(title=full_title, width="single", height=h)

    if style == "boxplot":
        for grp in group_order:
            grp_vals = df_mapped[df_mapped["_group"] == grp][importance_col].values
            color = grp_to_color.get(grp, "#1f77b4")
            fig.add_trace(go.Box(
                y=[grp] * len(grp_vals), x=grp_vals,
                orientation="h", name=grp,
                marker_color=color, line_color=color,
                fillcolor=hex_to_rgba(color, 0.3),
                showlegend=False, hoverinfo="x",
            ))
    else:
        # Bar with error
        colors = [grp_to_color.get(g, "#1f77b4") for g in grp_stats["_group"]]
        fig.add_trace(go.Bar(
            x=grp_stats["mean"], y=grp_stats["_group"],
            orientation="h", marker_color=colors,
            error_x=symmetric_error_bar(grp_stats["std"].fillna(0).values),
            hovertemplate="<b>%{y}</b><br>Mean importance: %{x:.4f}<extra></extra>",
            showlegend=False,
        ))

    xaxis_kw = dict(title="Mean Permutation Importance")
    if xlim is not None:
        xaxis_kw["range"] = list(xlim)
    fig.update_layout(
        margin=dict(l=d["margin_l"] + 20, t=d["margin_t"] + 10),
        xaxis=xaxis_kw, yaxis_title="",
        yaxis=dict(categoryorder="array", categoryarray=group_order),
        showlegend=False,
    )
    apply_size_overrides(fig, width, height)
    return fig
