"""SHAP (SHapley Additive exPlanations) visualizations."""

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv._warnings import InternalWarning
from mlsurv.viz._palettes import (
    FeatureGroupsType,
    MANUSCRIPT_DEFAULTS,
    _add_group_legend_traces,
    _iter_group_segments,
    _make_group_strip_trace,
    _narrow_groups,
    _resolve_groups,
    hex_to_rgba,
)
from mlsurv.viz._features_helpers import _importance_full_title
from mlsurv.viz._plotly_helpers import symmetric_error_bar
from mlsurv.viz._theme import adaptive_size, apply_size_overrides, create_figure
from mlsurv.viz._validation import validate_columns


def _get_feature_values(feat_shap, feature, X_values, sample_col):
    """Return raw feature values for a single feature's SHAP rows."""
    if 'feature_value' in feat_shap.columns:
        return feat_shap['feature_value'].values

    sample_indices = feat_shap[sample_col].values
    if X_values is not None and feature in X_values.columns:
        return X_values.loc[sample_indices, feature].values

    warnings.warn(
        f"Feature values unavailable for '{feature}': "
        f"no 'feature_value' column in shap_df and feature not found in X. "
        f"Color encoding will be uniform.",
        InternalWarning,
        stacklevel=3,
    )
    return np.full(len(feat_shap), np.nan)


def _normalize_feature_values(feat_values):
    """Normalize values to [0, 1], handling NaNs and constant features.

    The colour range is the 5th-95th percentile (as in the ``shap``
    library), and values outside it are clipped to the ends.  Plain
    min-max scaling would let a few extreme values (e.g. a very high
    TMB) push every other sample to the same colour.  If the
    percentile range is empty, the full min-max range is used instead.

    String values (from grouped categorical features) are treated as
    uniform 0.5 (neutral color).
    """
    # String feature values (grouped categoricals) → uniform neutral color
    try:
        numeric = np.asarray(feat_values, dtype=float)
    except (ValueError, TypeError):
        return np.full(len(feat_values), 0.5)

    if len(numeric) == 0 or np.all(np.isnan(numeric)):
        feat_min, feat_max = 0.0, 0.0
    else:
        feat_min, feat_max = np.nanpercentile(numeric, [5, 95])
        if not feat_max > feat_min:
            feat_min, feat_max = np.nanmin(numeric), np.nanmax(numeric)

    if feat_max > feat_min:
        normalized = np.clip(
            (numeric - feat_min) / (feat_max - feat_min), 0.0, 1.0,
        )
    else:
        normalized = np.full_like(numeric, 0.5)

    return np.nan_to_num(normalized, nan=0.5)


def _validate_beeswarm_inputs(shap_df, X):
    """Raise ValueError for invalid beeswarm inputs."""
    if shap_df is None or len(shap_df) == 0:
        raise ValueError("shap_df is empty or None")
    if X is not None and len(X) == 0:
        raise ValueError("X is empty or None")
    if X is None and 'feature_value' not in shap_df.columns:
        raise ValueError(
            "X is required when shap_df does not contain a 'feature_value' column"
        )


def _get_top_shap_features(shap_df, feature_col, shap_col, n):
    """Return top-N features sorted ascending by mean |SHAP|."""
    mean_abs = shap_df.groupby(feature_col)[shap_col].apply(
        lambda x: np.abs(x).mean()
    ).reset_index(name='mean_abs_shap')
    return mean_abs.nlargest(n, 'mean_abs_shap').sort_values(
        'mean_abs_shap'
    )[feature_col].tolist()


def _add_beeswarm_trace(fig, feature, feat_idx, shap_values, feat_values,
                        feat_normalized, rng, row=None, col=None):
    """Add a single beeswarm scatter trace for one feature."""
    jitter = rng.uniform(-0.3, 0.3, len(shap_values))
    y_positions = feat_idx + jitter

    # String feature values (grouped categoricals) need different formatting
    is_string = isinstance(feat_values[0], str) if len(feat_values) > 0 else False
    if is_string:
        hover = (
            f'<b>{feature}</b><br>'
            'SHAP: %{x:.4f}<br>'
            '%{customdata}<extra></extra>'
        )
    else:
        hover = (
            f'<b>{feature}</b><br>'
            'SHAP: %{x:.4f}<br>'
            f'Feature value: %{{customdata:.3f}}<extra></extra>'
        )

    trace = go.Scatter(
        x=shap_values,
        y=y_positions,
        mode='markers',
        name=feature,
        marker=dict(
            color=feat_normalized,
            colorscale='RdBu_r',
            size=6,
            opacity=0.7,
            showscale=False,
            line=dict(color='black', width=0.5),
        ),
        showlegend=False,
        hovertemplate=hover,
        customdata=feat_values,
    )
    if row is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def _add_shap_violin_traces(fig, top_features, plot_data, feat_to_grp,
                            grp_to_color, feature_col):
    """Add violin + box + jitter traces for each feature (show_violin=True path).

    Parameters
    ----------
    fig : go.Figure
        Target figure.
    top_features : list of str
        Features in display order.
    plot_data : DataFrame
        Filtered SHAP data with ``abs_shap`` column already computed.
    feat_to_grp, grp_to_color : dict or None
        Feature group mappings.
    feature_col : str
        Column name for features.
    """
    has_groups = bool(feat_to_grp and grp_to_color)
    default_color = '#1f77b4'

    for feature in top_features:
        feat_data = plot_data[plot_data[feature_col] == feature]['abs_shap'].values

        if len(feat_data) == 0:
            continue

        color = default_color
        if has_groups and feature in feat_to_grp:
            color = grp_to_color.get(feat_to_grp[feature], default_color)
        fill_color = hex_to_rgba(color, 0.3)
        fill_color_box = hex_to_rgba(color, 0.6)

        # Add violin trace
        fig.add_trace(go.Violin(
            y=[feature] * len(feat_data),
            x=feat_data,
            orientation='h',
            side='positive',
            name=feature,
            line_color=color,
            fillcolor=fill_color,
            meanline_visible=False,
            showlegend=False,
            scalemode='width',
            width=0.8,
            hoverinfo='skip',
        ))

        # Add box plot overlay
        fig.add_trace(go.Box(
            y=[feature] * len(feat_data),
            x=feat_data,
            orientation='h',
            name=feature,
            marker_color=color,
            line_color=color,
            fillcolor=fill_color_box,
            boxmean=True,
            showlegend=False,
            width=0.3,
            hoverinfo='skip',
        ))

        # Add jitter points
        fig.add_trace(go.Scatter(
            y=[feature] * len(feat_data),
            x=feat_data,
            mode='markers',
            name=feature,
            marker=dict(
                color=color,
                size=4,
                opacity=0.6,
            ),
            showlegend=False,
            hovertemplate=(
                f'<b>{feature}</b><br>'
                '|SHAP|: %{x:.4f}<extra></extra>'
            ),
        ))

    fig.update_layout(
        xaxis_title='Mean |SHAP| Value',
    )


def _add_shap_bar_traces(fig, top_stats, feat_to_grp, grp_to_color,
                          feature_col):
    """Add bar traces for SHAP importance (show_violin=False path).

    Parameters
    ----------
    fig : go.Figure
        Target figure.
    top_stats : DataFrame
        Must contain ``mean_abs_shap``, ``std_abs_shap``, and *feature_col*.
    feat_to_grp, grp_to_color : dict or None
        Feature group mappings.
    feature_col : str
        Column name for features.
    """
    has_groups = bool(feat_to_grp and grp_to_color)
    default_color = '#1f77b4'

    if has_groups:
        # One bar per group for legend + ungrouped
        for grp, color, grp_feats in _iter_group_segments(
            top_stats[feature_col].tolist(), feat_to_grp, grp_to_color,
        ):
            grp_stats = top_stats[top_stats[feature_col].isin(grp_feats)]
            if grp_stats.empty:
                continue
            fig.add_trace(go.Bar(
                x=grp_stats['mean_abs_shap'],
                y=grp_stats[feature_col],
                orientation='h',
                name=grp,
                marker_color=color,
                error_x=symmetric_error_bar(grp_stats['std_abs_shap']),
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    'Mean |SHAP|: %{x:.4f}<extra></extra>'
                ),
            ))
    else:
        fig.add_trace(go.Bar(
            x=top_stats['mean_abs_shap'],
            y=top_stats[feature_col],
            orientation='h',
            marker_color=default_color,
            error_x=symmetric_error_bar(top_stats['std_abs_shap']),
            hovertemplate=(
                '<b>%{y}</b><br>'
                'Mean |SHAP|: %{x:.4f}<extra></extra>'
            ),
        ))

    fig.update_layout(
        xaxis_title='Mean |SHAP| Value',
    )


def plot_shap_importance(
    shap_df: pd.DataFrame,
    n: int = 20,
    feature_col: str = 'feature',
    shap_col: str = 'shap_value',
    title: Optional[str] = None,
    show_violin: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Bar plot of mean |SHAP| with violin/boxplot showing variability.

    Creates a horizontal bar chart showing mean absolute SHAP values
    for each feature, with optional violin overlay showing the
    distribution across samples.

    Parameters
    ----------
    shap_df : DataFrame
        Long-format SHAP values.  Must contain ``feature`` and ``shap_value``
        columns; ``sample_idx`` is unused by this function.
        Typically from EvaluationResult.shap_values.

        **DataFrame schema** (default column names)::

            feature     shap_value
            str         float
    n : int, default=20
        Number of top features to display.
    feature_col : str, default='feature'
        Column containing feature names.
    shap_col : str, default='shap_value'
        Column containing SHAP values.
    title : str, optional
        Plot title. Auto-generated if None.
    show_violin : bool, default=True
        If True, show violin+boxplot+jitter overlay.
        If False, show simple bars with error bars.
    width : int, optional
        Override default figure width (pixels).
    height : int, optional
        Override default figure height (pixels).
    feature_groups : dict, list, Series, or None
        Feature-to-group mapping. See :func:`resolve_feature_group_colors`.
    group_colors : dict, optional
        Explicit ``{group: color}`` overrides.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive bar/violin plot.

    Raises
    ------
    ValueError
        If ``shap_df`` is ``None`` or contains no rows.

    Examples
    --------
    >>> result = learner.evaluate(model, include_shap=True)
    >>> fig = plot_shap_importance(result.shap_values, n=15)
    >>> fig.show()
    """
    if shap_df is None or len(shap_df) == 0:
        raise ValueError("shap_df is empty or None")
    validate_columns(shap_df, [feature_col, shap_col], "plot_shap_importance")

    # Compute mean |SHAP| per feature, sorted ascending (highest at chart top)
    mean_abs_shap = shap_df.groupby(feature_col)[shap_col].apply(
        lambda x: np.abs(x).mean()
    ).reset_index(name='mean_abs_shap')

    top_features = mean_abs_shap.nlargest(n, 'mean_abs_shap').sort_values(
        'mean_abs_shap'
    )[feature_col].tolist()
    n_features = len(top_features)

    if title is None:
        if n_features < n:
            title = 'SHAP Feature Importance'
        else:
            title = f'Top {n} Features by Mean |SHAP|'

    # The bar chart draws ±SD error bars, so name the band in a subtitle. The
    # violin path shows the full distribution instead, so it gets no band label.
    subtitle = "±SD" if not show_violin else None

    d = MANUSCRIPT_DEFAULTS
    h = adaptive_size(n_features, per_item=30, base=60, min_size=d["height_small"])

    fig = create_figure(
        title=_importance_full_title(title, subtitle), width="single", height=h,
    )

    # Resolve feature groups, narrowed to top features only
    all_features = shap_df[feature_col].unique().tolist()
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, all_features, group_colors,
    )
    feat_to_grp, grp_to_color = _narrow_groups(
        feat_to_grp, grp_to_color, top_features,
    )
    has_groups = bool(feat_to_grp and grp_to_color)

    if show_violin:
        # Filter to top features
        plot_data = shap_df[shap_df[feature_col].isin(top_features)].copy()

        # Use absolute SHAP values for importance visualization
        plot_data['abs_shap'] = plot_data[shap_col].abs()

        _add_shap_violin_traces(
            fig, top_features, plot_data, feat_to_grp, grp_to_color,
            feature_col,
        )

    else:
        # Simple bar chart with error bars
        top_stats = mean_abs_shap[mean_abs_shap[feature_col].isin(top_features)].copy()

        # Compute std for error bars
        std_abs_shap = shap_df.groupby(feature_col)[shap_col].apply(
            lambda x: np.abs(x).std()
        ).reset_index(name='std_abs_shap')

        top_stats = top_stats.merge(std_abs_shap, on=feature_col)
        top_stats = top_stats.sort_values('mean_abs_shap')

        _add_shap_bar_traces(
            fig, top_stats, feat_to_grp, grp_to_color, feature_col,
        )

    if has_groups and show_violin:
        _add_group_legend_traces(fig, grp_to_color)

    fig.update_layout(
        margin=dict(l=d["margin_l"] + 20,
                    t=d["margin_t"] + (30 if subtitle else 10)),
        yaxis_title='',
        yaxis=dict(categoryorder='array', categoryarray=top_features),
        showlegend=has_groups,
        legend=dict(x=1.02, xanchor="left", y=0.5, yanchor="middle") if has_groups else {},
    )

    apply_size_overrides(fig, width, height)
    return fig


def plot_shap_beeswarm(
    shap_df: pd.DataFrame,
    X: Optional[pd.DataFrame] = None,
    n: int = 15,
    feature_col: str = 'feature',
    shap_col: str = 'shap_value',
    sample_col: str = 'sample_idx',
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Beeswarm plot showing SHAP value distribution colored by feature value.

    Creates a scatter plot where each point is a sample, positioned by
    its SHAP value on the x-axis and colored by the original feature
    value (blue=low, red=high).

    Parameters
    ----------
    shap_df : DataFrame
        Long-format SHAP values with columns: feature, sample_idx, shap_value.
        May also contain a ``feature_value`` column with embedded feature
        values (produced by ``_compute_shap`` when ``max_explain`` subsamples).
        Typically from EvaluationResult.shap_values.

        **DataFrame schema** (default column names)::

            feature     shap_value   sample_idx
            str         float        int

        ``sample_idx`` is only required when the DataFrame does not
        contain an embedded ``feature_value`` column.
    X : DataFrame, optional
        Original feature values (for coloring). Index must align with
        sample_idx or be integer-indexed matching sample_idx values.
        Not required when ``shap_df`` contains a ``feature_value`` column.
    n : int, default=15
        Number of top features to display.
    feature_col : str, default='feature'
        Column containing feature names.
    shap_col : str, default='shap_value'
        Column containing SHAP values.
    sample_col : str, default='sample_idx'
        Column containing sample indices.
    title : str, optional
        Plot title. Auto-generated if None.
    width : int, optional
        Override default figure width (pixels).
    height : int, optional
        Override default figure height (pixels).
    feature_groups : dict, list, Series, or None
        Feature-to-group mapping. See :func:`resolve_feature_group_colors`.
    group_colors : dict, optional
        Explicit ``{group: color}`` overrides.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive beeswarm plot.

    Notes
    -----
    Visual elements:
    - Y-axis: features sorted by mean |SHAP|
    - X-axis: SHAP values (negative = decreases risk, positive = increases risk)
    - Color: feature value (blue=low, red=high)
    - Dots pile up vertically to show density

    Examples
    --------
    >>> result = learner.evaluate(model, include_shap=True)
    >>> fig = plot_shap_beeswarm(result.shap_values, X_test, n=15)
    >>> fig.show()
    """
    _validate_beeswarm_inputs(shap_df, X)
    required = [feature_col, shap_col]
    if 'feature_value' not in shap_df.columns:
        required.append(sample_col)
    validate_columns(shap_df, required, "plot_shap_beeswarm")

    top_features = _get_top_shap_features(shap_df, feature_col, shap_col, n)
    n_features = len(top_features)

    if title is None:
        title = (
            'SHAP Values (Effect Direction)' if n_features < n
            else f'Top {n} Features: SHAP Effect Direction'
        )

    d = MANUSCRIPT_DEFAULTS
    h = adaptive_size(n_features, per_item=35, base=60, min_size=d["height_medium"])

    # Resolve feature groups early — determines figure type
    all_features = shap_df[feature_col].unique().tolist()
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, all_features, group_colors,
    )
    feat_to_grp, grp_to_color = _narrow_groups(
        feat_to_grp, grp_to_color, top_features,
    )
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
            width=d["width_single"] + 50, height=h,
        )
        fig.add_trace(
            _make_group_strip_trace(top_features, feat_to_grp, grp_to_color, orientation="vertical"),
            row=1, col=1,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        # Show feature labels on strip y-axis (left of strip), hide on main
        fig.update_yaxes(
            showgrid=False, showticklabels=True,
            tickmode="array",
            tickvals=list(range(len(top_features))),
            ticktext=top_features, automargin=True,
            row=1, col=1,
        )
        main_row, main_col = 1, 2
    else:
        fig = create_figure(title=title, width=d["width_single"] + 50, height=h)
        main_row, main_col = None, None

    X_values = X.reset_index(drop=True) if X is not None else None
    rng = np.random.RandomState(42)

    for feat_idx, feature in enumerate(top_features):
        feat_shap = shap_df[shap_df[feature_col] == feature].copy()
        if len(feat_shap) == 0:
            continue

        shap_values = feat_shap[shap_col].values
        feat_values = _get_feature_values(feat_shap, feature, X_values, sample_col)
        feat_normalized = _normalize_feature_values(feat_values)
        _add_beeswarm_trace(
            fig, feature, feat_idx, shap_values, feat_values, feat_normalized, rng,
            row=main_row, col=main_col,
        )

    # Colorbar as a separate invisible trace
    colorbar_trace = go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            colorscale='RdBu_r', showscale=True, cmin=0, cmax=1,
            colorbar=dict(
                title=dict(text='Feature value', side='right'),
                tickvals=[0, 0.5, 1], ticktext=['Low', 'Mid', 'High'],
                len=0.5, y=0.75,
            ),
        ),
        showlegend=False, hoverinfo='skip',
    )
    if main_row is not None:
        fig.add_trace(colorbar_trace, row=main_row, col=main_col)
    else:
        fig.add_trace(colorbar_trace)

    fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5,
                  row=main_row, col=main_col)

    if has_groups:
        _add_group_legend_traces(fig, grp_to_color, row=1, col=2)
        fig.update_xaxes(
            title_text='SHAP Value (impact on model output)',
            zeroline=True, zerolinecolor='gray', zerolinewidth=1,
            row=1, col=2,
        )
        fig.update_yaxes(
            showticklabels=False, showgrid=False,
            row=1, col=2,
        )
    else:
        fig.update_layout(
            xaxis_title='SHAP Value (impact on model output)',
            xaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1),
            yaxis=dict(tickmode='array', tickvals=list(range(len(top_features))),
                       ticktext=top_features, automargin=True),
        )

    # Extra top margin holds the direction label between the title and
    # the plot, so it never overlaps the top row of dots.
    fig.update_layout(
        margin=dict(l=d["margin_l"] + 20, r=d["margin_r"] + 30, t=d["margin_t"] + 30),
        yaxis_title='',
        showlegend=has_groups,
        legend=dict(x=1.02, xanchor="left", y=0.2, yanchor="middle") if has_groups else {},
    )

    fig.add_annotation(
        x=0.5, y=1.0, xref='paper', yref='paper',
        xanchor='center', yanchor='bottom', yshift=4,
        text='<span style="color:#1f77b4">\u2190 Decreases Risk</span> | '
             '<span style="color:#d62728">Increases Risk \u2192</span>',
        showarrow=False, font=dict(size=d["font_size_annotation"]),
    )

    apply_size_overrides(fig, width, height)
    return fig
