"""Correlation heatmap visualization.

Contains ``plot_feature_correlation`` and its annotation helper.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv._warnings import InternalWarning
from mlsurv.viz._palettes import (
    _add_group_legend_traces,
    _make_group_strip_trace,
    _resolve_groups,
    _sort_features_by_group,
)
from mlsurv.viz._theme import (
    adaptive_size, apply_size_overrides, create_figure,
    margin_for_labels, truncate_labels,
)
from mlsurv.viz._features_helpers import _filter_numeric_for_corr


def _add_correlation_annotations(fig, corr_masked, d, xref=None, yref=None):
    """Add correlation value annotations to a correlation heatmap."""
    for i, row_val in enumerate(corr_masked.index):
        for j, col_val in enumerate(corr_masked.columns):
            val = corr_masked.iloc[i, j]
            if pd.notna(val):
                ann_kw = dict(
                    x=col_val, y=row_val,
                    text=f"{val:.2f}", showarrow=False,
                    font=dict(
                        color="black" if -0.5 < val < 0.5 else "white",
                        size=d["font_size_tick"] - 1,
                    ),
                )
                if xref is not None:
                    ann_kw["xref"] = xref
                    ann_kw["yref"] = yref
                fig.add_annotation(**ann_kw)


def _build_correlation_matrix(X, max_features, sort_by_group, feature_groups,
                              group_colors):
    """Build correlation matrix with feature reduction and clustering.

    Filters to numeric columns, reduces to *max_features* by total correlation,
    applies hierarchical clustering for ordering, and optionally overrides with
    group-based ordering.  Returns lower-triangular masked matrix.

    Returns ``(corr, corr_masked, features_list)``.
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    # helper -> plot_feature_correlation (viz) -> impl -> wrapper -> user
    numeric_df = _filter_numeric_for_corr(X, stacklevel=5)

    # Reduce features if too many
    if len(numeric_df.columns) > max_features:
        corr_full = numeric_df.corr().abs()
        corr_vals = corr_full.to_numpy(copy=True)
        np.fill_diagonal(corr_vals, 0)
        corr_sums = pd.Series(corr_vals.sum(axis=0), index=corr_full.columns)
        top_features = corr_sums.nlargest(max_features).index.tolist()
        numeric_df = numeric_df[top_features]

    corr = numeric_df.corr()

    # Hierarchical clustering to order features
    try:
        dissimilarity = 1 - corr.abs()
        dissim_vals = dissimilarity.to_numpy(copy=True)
        np.fill_diagonal(dissim_vals, 0)
        linkage = hierarchy.linkage(squareform(dissim_vals), method="average")
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        order = dendro["leaves"]
        ordered_features = [corr.columns[i] for i in order]
        corr = corr.loc[ordered_features, ordered_features]
    except (ValueError, FloatingPointError) as exc:
        warnings.warn(
            f"Hierarchical clustering for feature ordering failed ({exc}); "
            "using original feature order.",
            InternalWarning,
            stacklevel=3,
        )

    # sort_by_group: override clustering order with group-based order
    if sort_by_group and feature_groups is not None:
        _ftg, _gtc = _resolve_groups(feature_groups, corr.columns.tolist(), group_colors)
        if _ftg and _gtc:
            ordered = _sort_features_by_group(
                corr.columns.tolist(), _ftg, _gtc,
            )
            corr = corr.loc[ordered, ordered]

    # Lower triangular mask
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    mask_arr = corr.to_numpy(copy=True)
    mask_arr[mask] = np.nan
    corr_masked = pd.DataFrame(mask_arr, index=corr.index, columns=corr.columns)
    features_list = corr.columns.tolist()

    return corr, corr_masked, features_list


def plot_feature_correlation(
    X: pd.DataFrame,
    threshold: float = 0.8,
    max_features: int = 20,
    title: str = "Feature Correlation Matrix",
    width=None,
    height=None,
    feature_groups=None,
    group_colors=None,
    sort_by_group: bool = False,
) -> go.Figure:
    """Triangular correlation heatmap for feature collinearity analysis.

    Features are ordered by hierarchical clustering to group correlated
    features together.

    Parameters
    ----------
    X : DataFrame
        Feature matrix (numeric columns only).
    threshold : float
        Threshold for flagging high correlations in hover text.
    max_features : int
        Maximum number of features to display.
    title : str
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
        When True and ``feature_groups`` is set, order features by group so
        within-group features cluster together along both axes.

    Returns
    -------
    go.Figure
    """
    from mlsurv.viz._palettes import MANUSCRIPT_DEFAULTS
    d = MANUSCRIPT_DEFAULTS

    corr, corr_masked, features_list = _build_correlation_matrix(
        X, max_features, sort_by_group, feature_groups, group_colors,
    )

    # Build hover text
    hover_text = []
    for i, row_feat in enumerate(corr.index):
        row = []
        for j, col_feat in enumerate(corr.columns):
            val = corr_masked.iloc[i, j]
            if pd.isna(val):
                row.append("")
            else:
                flag = " \u26a0\ufe0f HIGH" if abs(val) >= threshold else ""
                row.append(f"{row_feat} vs {col_feat}<br>r = {val:.3f}{flag}")
        hover_text.append(row)

    colorscale = [
        [0.0, "#2166ac"],
        [0.25, "#67a9cf"],
        [0.5, "#f7f7f7"],
        [0.75, "#ef8a62"],
        [1.0, "#b2182b"],
    ]

    n_feat = len(corr.columns)
    size = adaptive_size(n_feat, per_item=45, base=100, min_size=d["height_medium"])

    display_labels, full_labels = truncate_labels(features_list)
    margin_b = margin_for_labels(display_labels, per_char=5, max_margin=200)
    margin_l = margin_for_labels(display_labels)

    # Resolve feature groups
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, features_list, group_colors,
    )

    if feat_to_grp and grp_to_color:
        from plotly.subplots import make_subplots
        strip_frac = 0.03

        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[1 - strip_frac, strip_frac],
            column_widths=[strip_frac, 1 - strip_frac],
            horizontal_spacing=0.02, vertical_spacing=0.02,
        )

        # Left strip (vertical, row=1, col=1)
        left_strip = _make_group_strip_trace(
            features_list, feat_to_grp, grp_to_color, orientation="vertical",
        )
        left_strip.x = [""]  # no label on narrow strip
        fig.add_trace(left_strip, row=1, col=1)

        # Main heatmap (row=1, col=2)
        fig.add_trace(go.Heatmap(
            z=corr_masked.values,
            x=features_list, y=corr.index.tolist(),
            hoverinfo="text", text=hover_text,
            colorscale=colorscale, zmin=-1, zmax=1,
            colorbar=dict(title="Correlation",
                          len=0.5, y=0.75, yanchor="middle"),
        ), row=1, col=2)

        # Bottom strip (horizontal, row=2, col=2)
        bottom_strip = _make_group_strip_trace(
            features_list, feat_to_grp, grp_to_color, orientation="horizontal",
        )
        fig.add_trace(bottom_strip, row=2, col=2)

        _add_correlation_annotations(fig, corr_masked, d, xref="x2", yref="y2")
        _add_group_legend_traces(fig, grp_to_color, row=1, col=2)

        fig.update_layout(
            template="mlsurv_minimal",
            title=dict(text=f"<b>{title}</b>"),
            width=size, height=size,
            margin=dict(l=margin_l, r=d["margin_r"] + 40,
                        b=margin_b),
            showlegend=True,
            legend=dict(x=1.02, xanchor="left", y=0.2, yanchor="middle"),
        )
        # Left strip: show y-labels (truncated feature names), hide x-labels
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_yaxes(
            showgrid=False, autorange="reversed",
            tickmode="array", tickvals=full_labels, ticktext=display_labels,
            row=1, col=1,
        )
        # Empty corner: hide all labels
        fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
        # Bottom strip: show truncated feature names on x-axis, hide y-labels
        fig.update_xaxes(
            showticklabels=True, showgrid=False, tickangle=45,
            tickmode="array", tickvals=full_labels, ticktext=display_labels,
            row=2, col=2,
        )
        fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=2)
        # Main heatmap: hide tick labels (left strip has y-labels,
        # bottom strip has x-labels)
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, autorange="reversed",
                         showgrid=False, row=1, col=2)
    else:
        fig = create_figure(title=title, width=size, height=size)
        fig.add_trace(go.Heatmap(
            z=corr_masked.values,
            x=features_list, y=corr.index.tolist(),
            hoverinfo="text", text=hover_text,
            colorscale=colorscale, zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
        ))

        _add_correlation_annotations(fig, corr_masked, d)

        fig.update_layout(
            margin=dict(l=margin_l, r=d["margin_r"] + 40,
                        b=margin_b),
            xaxis=dict(
                tickangle=45, side="bottom",
                tickmode="array", tickvals=full_labels, ticktext=display_labels,
            ),
            yaxis=dict(
                autorange="reversed",
                tickmode="array", tickvals=full_labels, ticktext=display_labels,
            ),
        )

    apply_size_overrides(fig, width, height)
    return fig
