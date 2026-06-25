"""Coefficient visualizations.

Contains ``plot_coefficients`` and ``plot_coefficient_comparison``.
"""

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from mlsurv._warnings import InternalWarning, SuggestionWarning
from mlsurv.viz._palettes import (
    FeatureGroupsType,
    MANUSCRIPT_DEFAULTS,
    QUALITATIVE_PALETTE,
    _add_group_legend_traces,
    _add_group_separator_shapes,
    _iter_group_segments,
    _make_group_strip_trace,
    _resolve_groups,
    _sort_features_by_group,
    get_model_color,
)
from mlsurv.viz._plotly_helpers import symmetric_error_bar
from mlsurv.viz._theme import (
    adaptive_size, apply_size_overrides, create_figure,
    margin_for_labels, truncate_labels,
)
from mlsurv.viz._validation import validate_columns


def _coef_error_x(std_col, sub_df, color=None):
    """Build error_x dict for coefficient markers/bars (95% CI)."""
    if std_col and std_col in sub_df.columns and sub_df[std_col].notna().any():
        return symmetric_error_bar(1.96 * sub_df[std_col], color=color)
    return None


def _plot_coef_multi_model(fig, df, model_col, coef_col, std_col, feature_col,
                           feature_order, palette, style, row=None, col=None):
    """Add multi-model coefficient traces to fig (grouped bars or forest)."""
    models = df[model_col].unique().tolist()
    if style == "forest":
        _plot_coef_multi_forest(fig, df, models, model_col, coef_col, std_col,
                                feature_col, feature_order, palette,
                                row=row, col=col)
        return
    for model_name in models:
        mdf = df[df[model_col] == model_name].copy()
        mdf = mdf.set_index(feature_col).reindex(feature_order).reset_index()
        color = get_model_color(model_name, palette=palette)
        trace = go.Bar(
            x=mdf[coef_col], y=mdf[feature_col],
            orientation="h", name=model_name,
            marker=dict(color=color),
            error_x=_coef_error_x(std_col, mdf),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "%{y}<br>Coefficient: %{x:.3f}<extra></extra>"
            ),
        )
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
    fig.update_layout(barmode="group", legend=dict(title=dict(text="Model")))


def _plot_coef_multi_forest(fig, df, models, model_col, coef_col, std_col,
                            feature_col, feature_order, palette, row=None, col=None):
    """Add multi-model forest (dot-and-whisker) traces, one marker per model.

    Each model gets a fixed vertical offset within a feature row so models do
    not overlap, mirroring ``plot_importance``'s multi-model forest. Feature
    rows use numeric y-positions (``feature_order`` index plus the offset)
    because Plotly scatter has no ``barmode="group"`` auto-offset; the caller
    sets the y-axis tickvals to the integer positions. Each model is reindexed
    to the full ``feature_order`` (as the bar path does), so every model keeps
    a legend entry and a feature it lacks shows as a gap rather than shifting
    the other markers.
    """
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=row, col=col)
    n_models = len(models)
    y_base = list(range(len(feature_order)))
    for mi, model_name in enumerate(models):
        mdf = (df[df[model_col] == model_name]
               .drop_duplicates(subset=feature_col)
               .set_index(feature_col)
               .reindex(feature_order)
               .reset_index())
        offset = (mi - (n_models - 1) / 2) * 0.25
        color = get_model_color(model_name, palette=palette)
        trace = go.Scatter(
            x=mdf[coef_col], y=[i + offset for i in y_base],
            mode="markers", name=model_name,
            marker=dict(color=color, size=8, symbol="circle"),
            error_x=_coef_error_x(std_col, mdf, color=color),
            text=feature_order,
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "%{text}<br>Coefficient: %{x:.3f}<extra></extra>"
            ),
        )
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
    fig.update_layout(legend=dict(title=dict(text="Model")))


def _plot_coef_single(fig, df, coef_col, std_col, feature_col, feature_order, style,
                      row=None, col=None, feat_to_grp=None, grp_to_color=None):
    """Add single-model coefficient traces (bar or forest)."""
    top = df.set_index(feature_col).reindex(feature_order).reset_index()

    if feat_to_grp and grp_to_color:
        # Group coloring: one trace per group for legend
        if style != "bar":
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
        for grp, color, grp_feats in _iter_group_segments(
            feature_order, feat_to_grp, grp_to_color,
        ):
            grp_rows = top[top[feature_col].isin(grp_feats)]
            if grp_rows.empty:
                continue
            error_x = _coef_error_x(std_col, grp_rows)
            if style == "bar":
                trace = go.Bar(
                    x=grp_rows[coef_col], y=grp_rows[feature_col],
                    orientation="h",
                    marker_color=color, name=grp, showlegend=True,
                    error_x=error_x,
                    hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.3f}<extra></extra>",
                )
            else:
                trace = go.Scatter(
                    x=grp_rows[coef_col], y=grp_rows[feature_col],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=grp, showlegend=True,
                    error_x=error_x,
                    hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.3f}<extra></extra>",
                )
            if row is not None:
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)
        fig.update_layout(showlegend=True)
    else:
        # Default: risk-direction coloring (red = positive / blue = negative)
        colors = top[coef_col].apply(lambda x: "#d62728" if x > 0 else "#1f77b4")
        error_x = _coef_error_x(std_col, top)
        if style == "bar":
            trace = go.Bar(
                x=top[coef_col], y=top[feature_col],
                orientation="h",
                marker=dict(color=colors.tolist()),
                error_x=error_x,
                hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.3f}<extra></extra>",
            )
        else:
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            trace = go.Scatter(
                x=top[coef_col], y=top[feature_col],
                mode="markers",
                marker=dict(size=10, color=colors.tolist()),
                error_x=error_x,
                hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.3f}<extra></extra>",
            )
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
        fig.update_layout(showlegend=False)


def _add_risk_direction_annotations(fig, ann_y, d, xref="x"):
    """Add 'Lower risk / Higher risk' annotations above the plot.

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


def plot_coefficients(
    df: pd.DataFrame,
    n: int = 20,
    coef_col: str = "coefficient",
    std_col: Optional[str] = "coef_std",
    feature_col: str = "feature",
    model_col: str = "model",
    title: str = "Feature Coefficients",
    style: str = "bar",
    width: Optional[int] = None,
    height: Optional[int] = None,
    palette=None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
    sort_by_group: bool = False,
) -> go.Figure:
    """Plot feature coefficients as a forest plot or horizontal bar chart.

    Supports single-model (red/blue risk colouring) and multi-model
    (grouped bars with legend) modes.  Multi-model mode is activated
    when ``model_col`` exists in *df* and contains more than one value.

    Parameters
    ----------
    df : DataFrame
        Must contain ``feature_col`` and ``coef_col``.  Optionally
        ``model_col`` for multi-model grouped bars.
    n : int
        Number of features to display (ranked by max |coefficient|
        across models).
    coef_col : str
        Column with coefficient values.
    std_col : str, optional
        Column with standard errors. When present, drawn as 95% CI error bars
        in both bar and forest styles.
    feature_col : str
        Column with feature names.
    model_col : str
        Column identifying the model (used when multiple models present).
    title : str
        Plot title.
    style : str, default="bar"
        ``"bar"`` for horizontal bars, ``"forest"`` for a point-and-whisker
        plot. Applies to both single-model and multi-model modes; in
        multi-model forest each model gets its own marker per feature, offset
        vertically. An unrecognized value falls back to ``"bar"`` with a
        warning.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    palette : dict, optional
        ``{model_name: hex_color}`` overrides.
    feature_groups : dict, list, Series, or None
        Feature-to-group mapping. See :func:`resolve_feature_group_colors`.
        When provided, a colour strip indicating feature groups is added
        alongside the plot and bars/markers are coloured by group.
    group_colors : dict, optional
        Explicit ``{group: color}`` overrides.
    sort_by_group : bool, default=False
        When True and ``feature_groups`` is set, order features by group
        (primary) and by coefficient magnitude within each group (secondary).

    Returns
    -------
    go.Figure
    """
    validate_columns(df, [feature_col, coef_col], "plot_coefficients")

    if style not in ("bar", "forest"):
        warnings.warn(
            f"plot_coefficients: style={style!r} is not recognized; falling "
            f"back to 'bar'. Valid options are 'bar' and 'forest'.",
            SuggestionWarning, stacklevel=2,
        )
        style = "bar"

    multi_model = model_col in df.columns and df[model_col].nunique() > 1

    # Select top-N features by max |coef| across all models
    df = df.copy()
    max_abs = df.groupby(feature_col)[coef_col].apply(lambda s: s.abs().max())
    top_features = max_abs.nlargest(n).index.tolist()
    df = df[df[feature_col].isin(top_features)]

    # Sort features by mean coefficient (ascending so largest at top of bar chart)
    mean_coef = df.groupby(feature_col)[coef_col].mean()
    feature_order = mean_coef.reindex(top_features).sort_values().index.tolist()
    n_actual = len(feature_order)

    # Resolve feature groups
    all_features = df[feature_col].unique().tolist()
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, all_features, group_colors,
    )

    # sort_by_group: reorder features by group, then by |coefficient| within group
    if sort_by_group and feat_to_grp and grp_to_color:
        metric_vals = mean_coef.abs().to_dict()
        feature_order = _sort_features_by_group(
            feature_order, feat_to_grp, grp_to_color, metric_values=metric_vals,
            reverse=True,
        )

    d = MANUSCRIPT_DEFAULTS
    margin_t = d["margin_t"] + 40
    margin_b = 70
    per_item = 32 if multi_model else 28
    h = adaptive_size(n_actual, per_item=per_item, base=120, min_size=d["height_small"])
    plot_h = h - margin_t - margin_b
    ann_y = 1.0 + 25 / plot_h

    display_labels, full_labels = truncate_labels(feature_order)
    margin_l = margin_for_labels(display_labels)

    # Multi-model forest uses numeric y-positions (index + per-model offset),
    # so its axis ticks are placed at the integer feature indices rather than
    # the categorical feature names used by every other path (bars and
    # single-model forest).
    multi_forest = multi_model and style == "forest"
    y_tickvals = list(range(n_actual)) if multi_forest else full_labels

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
            width=d["width_single"], height=h,
        )
        fig.add_trace(
            _make_group_strip_trace(
                feature_order, feat_to_grp, grp_to_color, orientation="vertical",
            ),
            row=1, col=1,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_yaxes(
            showgrid=False, showticklabels=True, tickmode="array",
            tickvals=y_tickvals, ticktext=display_labels,
            row=1, col=1,
        )
        main_row, main_col = 1, 2
    else:
        fig = create_figure(title=title, width="single", height=h)
        main_row, main_col = None, None

    if multi_model:
        _plot_coef_multi_model(fig, df, model_col, coef_col, std_col, feature_col,
                               feature_order, palette, style,
                               row=main_row, col=main_col)
    else:
        _plot_coef_single(fig, df, coef_col, std_col, feature_col,
                          feature_order, style, row=main_row, col=main_col,
                          feat_to_grp=feat_to_grp, grp_to_color=grp_to_color)

    if has_groups:
        fig.update_xaxes(title="Coefficient", row=1, col=2)
        fig.update_yaxes(
            showticklabels=False, tickmode="array",
            tickvals=y_tickvals, ticktext=display_labels,
            row=1, col=2,
        )
        # Single-model: data traces already provide group legend entries.
        # Multi-model: data traces use model names, so add group legend separately.
        if multi_model:
            _add_group_legend_traces(fig, grp_to_color, row=main_row, col=main_col)
    else:
        fig.update_layout(
            xaxis_title="Coefficient", yaxis_title="",
            yaxis=dict(
                tickmode="array",
                tickvals=y_tickvals,
                ticktext=display_labels,
            ),
        )

    if multi_forest:
        # Match the numeric forest y-axis to the categorical group strip's
        # ``[-0.5, n-0.5]`` span so rows line up across the two subplots.
        fig.update_yaxes(range=[-0.5, n_actual - 0.5], row=main_row, col=main_col)

    fig.update_layout(
        margin=dict(l=margin_l, t=margin_t, b=margin_b),
        bargap=0.3,
    )

    _add_risk_direction_annotations(
        fig, ann_y, d, xref="x2" if has_groups else "x",
    )

    if sort_by_group and feat_to_grp and grp_to_color:
        _add_group_separator_shapes(fig, feature_order, feat_to_grp, axis="y")

    apply_size_overrides(fig, width, height)
    return fig


def _resolve_scatter_color(df, feature_col, color_col, feat_to_grp,
                           grp_to_color, hover_data):
    """Resolve colour column, color map, and legend title for scatter plot.

    Returns ``(df, effective_color_col, color_map, legend_title)``.
    """
    if feat_to_grp and grp_to_color:
        _grp_col = "_group_"
        df = df.copy()
        df[_grp_col] = df[feature_col].map(feat_to_grp).fillna("Other")
        hover_data[_grp_col] = True
        return df, _grp_col, grp_to_color, "Feature Group"
    if color_col and color_col in df.columns:
        hover_data[color_col] = True
        unique_vals = sorted(df[color_col].dropna().unique())
        color_map = {
            v: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
            for i, v in enumerate(unique_vals)
        }
        return df, color_col, color_map, "Feature Class"
    return df, None, None, None


def _select_divergent_features(df, x_col, y_col, feature_col, max_annotations):
    """Select top-N features by divergence from the identity line.

    Returns a set of feature names, or ``None`` to annotate all features.
    """
    if max_annotations is not None and len(df) > max_annotations:
        distance = (df[y_col] - df[x_col]).abs()
        top_idx = distance.nlargest(max_annotations).index
        return set(df.loc[top_idx, feature_col])
    return None


def _repel_labels(x_data, y_data, labels, font_size=9,
                  plot_width=350, plot_height=350,
                  pad_frac=0.015, max_iter=100):
    """Compute non-overlapping label positions via iterative repulsion.

    Mimics the ggrepel / adjustText approach: labels start near their data
    points, then overlapping bounding boxes push each other apart while a
    spring force pulls them back toward the data point.

    Parameters
    ----------
    x_data, y_data : array-like
        Data-point positions (data coordinates).
    labels : list of str
        Label text (used to estimate bounding-box width).
    font_size : int
        Label font size in pixels.
    plot_width, plot_height : int
        Approximate plot-area size in pixels (excluding margins).
    pad_frac : float
        Fractional padding around each label box (of data range).
    max_iter : int
        Maximum repulsion iterations.

    Returns
    -------
    xl, yl : np.ndarray
        Adjusted label-center positions in data coordinates.
    """
    n = len(x_data)
    if n == 0:
        return np.array([]), np.array([])

    xd = np.asarray(x_data, dtype=float)
    yd = np.asarray(y_data, dtype=float)

    x_range = np.ptp(xd) or 1.0
    y_range = np.ptp(yd) or 1.0

    # Label half-dimensions in data coordinates
    char_w = font_size * 0.55 * x_range / plot_width
    line_h = font_size * 1.5 * y_range / plot_height
    pad_x = pad_frac * x_range
    pad_y = pad_frac * y_range
    hw = np.array([len(str(l)) * char_w / 2 + pad_x for l in labels])
    hh = line_h / 2 + pad_y

    # Initialise: offset away from the identity line so labels don't
    # sit directly on their data points.
    min_dist_y = 5.0 * hh   # minimum vertical clearance from data point
    min_dist_x = hw * 0.6   # minimum horizontal clearance
    xl = xd.copy()
    yl = yd.copy()
    for i in range(n):
        if yd[i] > xd[i]:
            yl[i] += min_dist_y
            xl[i] -= min_dist_x[i]
        else:
            yl[i] -= min_dist_y
            xl[i] += min_dist_x[i]

    spring = 0.08          # pull toward data point (gentle)
    repel_strength = 0.45  # push away from overlapping labels
    converge = 0.001 * (x_range + y_range)

    for _ in range(max_iter):
        max_move = 0.0
        for i in range(n):
            fx, fy = 0.0, 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = xl[i] - xl[j]
                dy = yl[i] - yl[j]
                ov_x = (hw[i] + hw[j]) - abs(dx)
                ov_y = 2 * hh - abs(dy)
                if ov_x > 0 and ov_y > 0:
                    sign_x = 1.0 if dx >= 0 else -1.0
                    sign_y = 1.0 if dy >= 0 else -1.0
                    fx += ov_x * sign_x * repel_strength
                    fy += ov_y * sign_y * repel_strength

            # Spring toward original data point
            fx -= spring * (xl[i] - xd[i])
            fy -= spring * (yl[i] - yd[i])

            xl[i] += fx
            yl[i] += fy

            # Enforce minimum distance from data point
            dy_from_pt = yl[i] - yd[i]
            sign = 1.0 if dy_from_pt >= 0 else -1.0
            if abs(dy_from_pt) < min_dist_y:
                yl[i] = yd[i] + sign * min_dist_y

            max_move = max(max_move, abs(fx) + abs(fy))

        if max_move < converge:
            break

    return xl, yl


def _callout_line_segments(x_data, y_data, x_label, y_label,
                           min_dist_frac=0.02):
    """Build callout line arrays with ``None`` separators for Plotly.

    Only draws a line when the label has moved more than *min_dist_frac*
    of the data range from its point.

    Returns ``(xs, ys)`` lists suitable for a single ``go.Scatter``
    trace with ``mode="lines"``.
    """
    x_range = np.ptp(x_data) or 1.0
    y_range = np.ptp(y_data) or 1.0
    threshold = min_dist_frac * max(x_range, y_range)

    xs, ys = [], []
    for xd, yd, xl, yl in zip(x_data, y_data, x_label, y_label):
        dist = np.hypot(xl - xd, yl - yd)
        if dist > threshold:
            xs.extend([xd, xl, None])
            ys.extend([yd, yl, None])
    return xs, ys


def _warn_nan_coefficients(df, x_col, y_col, stacklevel=4):
    """Emit warning about NaN coefficients that will be skipped."""
    nan_x = df[x_col].isna().sum()
    nan_y = df[y_col].isna().sum()
    n_nan = (df[x_col].isna() | df[y_col].isna()).sum()
    if n_nan > 0:
        parts = []
        if nan_x:
            parts.append(f"{nan_x} in '{x_col}'")
        if nan_y:
            parts.append(f"{nan_y} in '{y_col}'")
        warnings.warn(
            f"{n_nan} features have NaN coefficients ({', '.join(parts)}) "
            f"and will not be annotated.",
            InternalWarning,
            stacklevel=stacklevel,
        )


def _add_coef_comparison_annotations(fig, df, x_col, y_col, feature_col,
                                     max_annotations):
    """Add feature name labels with repelled positions to a scatter figure."""
    _warn_nan_coefficients(df, x_col, y_col)

    top_features = _select_divergent_features(
        df, x_col, y_col, feature_col, max_annotations,
    )

    rows = []
    for _, row in df.iterrows():
        if top_features is not None and row[feature_col] not in top_features:
            continue
        x_val, y_val = row[x_col], row[y_col]
        if pd.isna(x_val) or pd.isna(y_val):
            continue
        rows.append((x_val, y_val, row[feature_col]))

    if not rows:
        return

    xd = [r[0] for r in rows]
    yd = [r[1] for r in rows]
    names = [r[2] for r in rows]

    xl, yl = _repel_labels(xd, yd, names)

    for i, (x_val, y_val, feat_name) in enumerate(rows):
        fig.add_annotation(
            x=x_val, y=y_val,
            ax=xl[i], ay=yl[i],
            xref="x", yref="y", axref="x", ayref="y",
            text=feat_name,
            showarrow=True,
            arrowhead=0, arrowwidth=0.8, arrowcolor="#999",
            font=dict(size=9, color="#555"),
            borderpad=2,
        )


def _add_coef_comparison_text_traces(fig, df, x_col, y_col, feature_col,
                                     max_annotations, feat_to_grp):
    """Add feature labels as text traces linked to legend groups.

    Uses repulsion to prevent overlap and callout lines to connect
    displaced labels back to their data points.  Text traces and
    callout lines share ``legendgroup`` with the scatter points so they
    toggle on/off together when clicking legend items.
    """
    _warn_nan_coefficients(df, x_col, y_col)

    top_features = _select_divergent_features(
        df, x_col, y_col, feature_col, max_annotations,
    )

    plot_df = df[df[x_col].notna() & df[y_col].notna()].copy()
    if top_features is not None:
        plot_df = plot_df[plot_df[feature_col].isin(top_features)]
    plot_df = plot_df.reset_index(drop=True)

    if plot_df.empty:
        return

    # Compute repelled positions for ALL labels at once
    xd = plot_df[x_col].values
    yd = plot_df[y_col].values
    names = plot_df[feature_col].tolist()
    xl, yl = _repel_labels(xd, yd, names)

    plot_df["_xl"] = xl
    plot_df["_yl"] = yl
    plot_df["_grp"] = plot_df[feature_col].map(feat_to_grp).fillna("Other")

    for grp in plot_df["_grp"].unique():
        gdf = plot_df[plot_df["_grp"] == grp]

        # Callout lines (data point → label)
        xs, ys = _callout_line_segments(
            gdf[x_col].values, gdf[y_col].values,
            gdf["_xl"].values, gdf["_yl"].values,
        )
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="#999", width=0.8),
                legendgroup=grp, showlegend=False, hoverinfo="skip",
            ))

        # Text labels at repelled positions
        fig.add_trace(go.Scatter(
            x=gdf["_xl"].tolist(),
            y=gdf["_yl"].tolist(),
            mode="text",
            text=gdf[feature_col].tolist(),
            textposition="middle center",
            textfont=dict(size=9, color="#555"),
            legendgroup=grp, showlegend=False, hoverinfo="skip",
        ))


def plot_coefficient_comparison(
    df: pd.DataFrame,
    x_col: str = "coxph_coef",
    y_col: str = "coxnet_coef",
    color_col: Optional[str] = "feature_class",
    importance_col: Optional[str] = "perm_mean",
    feature_col: str = "feature",
    title: str = "Coefficient Comparison",
    width: Optional[int] = None,
    height: Optional[int] = None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
    max_annotations: Optional[int] = 20,
) -> go.Figure:
    """Scatter comparing coefficients between two models.

    Parameters
    ----------
    df : DataFrame
        Must contain coefficient columns and feature names.
    x_col : str
        Column for x-axis coefficients.
    y_col : str
        Column for y-axis coefficients.
    color_col : str, optional
        Column for coloring points.  Ignored when *feature_groups* is
        provided.
    importance_col : str, optional
        Column for sizing points.
    feature_col : str
        Column with feature names.
    title : str
        Plot title.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    feature_groups : dict, list, or Series, optional
        Feature-to-group mapping.  When provided, points are coloured by
        group instead of *color_col*.
    group_colors : dict, optional
        ``{group_name: hex_color}`` overrides for group colours.
    max_annotations : int or None, default=20
        Maximum number of feature name annotations.  When the number of
        features exceeds this limit, only those with the largest coefficient
        divergence from the identity line are labeled.  ``None`` labels all.

    Returns
    -------
    go.Figure
    """
    validate_columns(df, [feature_col, x_col, y_col], "plot_coefficient_comparison")

    all_features = df[feature_col].tolist()
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, all_features, group_colors,
    )

    hover_data = {x_col: ":.3f", y_col: ":.3f"}
    if importance_col and importance_col in df.columns:
        hover_data[importance_col] = ":.4f"

    df, effective_color_col, color_map, legend_title = _resolve_scatter_color(
        df, feature_col, color_col, feat_to_grp, grp_to_color, hover_data,
    )

    fig = px.scatter(
        df, x=x_col, y=y_col, color=effective_color_col,
        color_discrete_map=color_map,
        hover_name=feature_col, hover_data=hover_data, title=title,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        max_val = np.nanmax([df[x_col].abs().max(), df[y_col].abs().max()])
    if pd.isna(max_val) or max_val == 0:
        max_val = 1.0
    max_val *= 1.1
    fig.add_trace(go.Scatter(
        x=[-max_val, max_val], y=[-max_val, max_val],
        mode="lines", line=dict(color="gray", dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    d = MANUSCRIPT_DEFAULTS
    fig.update_traces(marker=dict(size=10, opacity=0.8))

    if feat_to_grp and grp_to_color:
        _add_coef_comparison_text_traces(fig, df, x_col, y_col, feature_col,
                                         max_annotations, feat_to_grp)
    else:
        _add_coef_comparison_annotations(fig, df, x_col, y_col, feature_col,
                                         max_annotations)

    fig.update_layout(
        template="mlsurv_minimal",
        width=d["width_single"], height=d["height_medium"],
        title=dict(text=f"<b>{title}</b>"),
        xaxis=dict(title=x_col.replace("_", " ").title(), range=[-max_val, max_val]),
        yaxis=dict(title=y_col.replace("_", " ").title(), range=[-max_val, max_val]),
        legend=dict(
            title=dict(text=legend_title) if legend_title else None,
        ),
    )
    apply_size_overrides(fig, width, height)
    return fig
