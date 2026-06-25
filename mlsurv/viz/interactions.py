"""Feature interaction visualizations (H-statistic heatmap, SHAP heatmap, 2D PDP)."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import (
    create_figure, apply_size_overrides, adaptive_size,
    margin_for_labels, truncate_labels,
)


def _get_value_column(df):
    """Detect the interaction value column in a DataFrame.

    Returns (column_name, colorbar_title, hover_label).

    When both columns exist (mixed hstat + shap results), the column with
    non-null values is preferred.  This allows per-model subsets to
    resolve correctly in multi-method DataFrames.

    Raises
    ------
    ValueError
        If neither ``shap_interaction`` nor ``h_statistic`` column is found.
    """
    has_shap = 'shap_interaction' in df.columns and df['shap_interaction'].notna().any()
    has_hstat = 'h_statistic' in df.columns and df['h_statistic'].notna().any()
    if has_shap:
        return 'shap_interaction', 'mean |SHAP|', 'SHAP interaction'
    if has_hstat:
        return 'h_statistic', 'H\u00b2', 'H\u00b2'
    # Fall back to column existence (empty DataFrames, all-NaN edge cases)
    if 'shap_interaction' in df.columns:
        return 'shap_interaction', 'mean |SHAP|', 'SHAP interaction'
    if 'h_statistic' in df.columns:
        return 'h_statistic', 'H\u00b2', 'H\u00b2'
    raise ValueError(
        f"Interaction DataFrame must contain 'shap_interaction' or "
        f"'h_statistic' column. Found columns: {list(df.columns)}"
    )


def _is_missing_model_name(model_name) -> bool:
    """Return True when *model_name* is None / NaN / pd.NA.

    This exists because ``_method_label`` previously used ``if model_name``
    as a "has a name" check. That short-circuit silently accepted
    ``np.nan`` (NaN is truthy in bool context), which then rendered as the
    literal string ``"nan"`` in dropdown buttons and titles. ``pd.NA``
    raises when coerced to bool, and ``None`` was the only value the
    original check actually handled. Callers must guard against removing
    this normalization without understanding all three sentinels —
    deleting it will re-introduce the "nan"/"None" label bug.

    Uses ``model_name is None or pd.isna(model_name)`` to match the
    idiom used in ``mlsurv/results/containers.py``. ``pd.isna`` handles
    ``None``, ``np.nan``, ``pd.NA``, and ``pd.NaT`` uniformly but raises
    ``TypeError`` on arrays — here we only ever pass scalars from a
    ``unique()`` result or a single-entry ``[None]`` fallback.
    """
    return model_name is None or pd.isna(model_name)


def _method_label(model_name, method):
    """Build a dropdown label combining model name and interaction method."""
    suffix = 'SHAP' if method == 'shap' else 'H\u00b2'
    if _is_missing_model_name(model_name) or model_name == '':
        return suffix
    return f"{model_name} ({suffix})"


def _build_heatmap_for_model(df, n, colorscale):
    """Build a single heatmap trace + metadata from one model's interaction data.

    Returns (trace, features, matrix) or None if df is empty after filtering.
    """
    val_col, colorbar_title, hover_label = _get_value_column(df)

    df = df.sort_values(val_col, ascending=False).head(n)
    if df.empty:
        return None

    # Collect unique features in order of appearance
    all_feats = pd.concat([df['feature_1'], df['feature_2']])
    features = list(dict.fromkeys(all_feats))

    # Build symmetric matrix
    n_feat = len(features)
    feat_to_idx = {f: i for i, f in enumerate(features)}
    matrix = np.full((n_feat, n_feat), np.nan)
    np.fill_diagonal(matrix, 0.0)

    for f1, f2, val in zip(df['feature_1'], df['feature_2'], df[val_col]):
        i, j = feat_to_idx[f1], feat_to_idx[f2]
        matrix[i, j] = val
        matrix[j, i] = val

    # Lower-triangular mask (matches correlation heatmap style)
    upper_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    matrix[upper_mask] = np.nan

    # Format annotations — blank for NaN (masked) cells
    formatted = np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.3f}")
    text = formatted(matrix).tolist()

    trace = go.Heatmap(
        z=matrix,
        x=features,
        y=features,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=colorscale,
        zmin=0,
        zmax=max(0.01, np.nanmax(matrix)),
        colorbar=dict(title=colorbar_title, len=0.6),
        hovertemplate=(
            "%{y} \u00d7 %{x}<br>"
            + hover_label
            + " = %{z:.4f}<extra></extra>"
        ),
    )
    return trace, features, matrix


def plot_interaction_heatmap(
    interactions_df: pd.DataFrame,
    n: int = 10,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    colorscale: str = "YlOrRd",
    feature_groups=None,
    group_colors=None,
) -> go.Figure:
    """Plot heatmap of pairwise feature interaction strengths.

    When the DataFrame contains multiple models, a dropdown selector
    lets the user switch between model heatmaps interactively.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        Long-format DataFrame with columns
        ``[model_type, feature_1, feature_2, h_statistic]``.
    n : int, default=10
        Maximum number of top pairs to include per model.
    title : str, optional
        Figure title.
    width : int, optional
        Override figure width (pixels).
    height : int, optional
        Override figure height (pixels).
    colorscale : str, default="YlOrRd"
        Plotly colorscale for the heatmap.
    feature_groups
        Accepted for API consistency with other plot functions; unused.
    group_colors
        Accepted for API consistency with other plot functions; unused.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if interactions_df is None or interactions_df.empty:
        fig = create_figure(title=title or "Feature Interactions")
        fig.add_annotation(
            text="No interaction data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        return fig

    # Detect method for default title
    has_shap = 'shap_interaction' in interactions_df.columns and interactions_df['shap_interaction'].notna().any()
    has_hstat = 'h_statistic' in interactions_df.columns and interactions_df['h_statistic'].notna().any()
    if has_shap and has_hstat:
        method_label = "SHAP & H-statistic"
    elif has_shap:
        method_label = "SHAP"
    else:
        method_label = "H-statistic"

    # Identify models and split by method when a model has both
    if 'model_type' in interactions_df.columns:
        models = list(interactions_df['model_type'].unique())
    else:
        models = [None]

    has_method_col = 'method' in interactions_df.columns

    # Build per-model (per-method) heatmap traces.
    # Key: display label, Value: (trace, features, matrix)
    model_traces = {}
    max_n_feat = 0
    for model_name in models:
        # Normalize missing model names (None / NaN / pd.NA) to an empty
        # string so downstream label construction doesn't render literal
        # "None"/"nan"/"<NA>" in titles or dropdown buttons.
        # This matters when a caller passes a DataFrame built by
        # concatenating per-model frames where one leg is missing
        # ``model_type``, or when a ``string``-dtype column has ``pd.NA``.
        if _is_missing_model_name(model_name):
            display_name = ''
            mdf = interactions_df
        else:
            display_name = str(model_name)
            mdf = interactions_df[interactions_df['model_type'] == model_name]

        # Check if this model has both methods
        if has_method_col:
            methods_in_model = set(mdf['method'].unique())
        else:
            methods_in_model = set()

        if len(methods_in_model) > 1:
            # Split into separate entries per method
            for m in sorted(methods_in_model):
                mdf_method = mdf[mdf['method'] == m]
                result = _build_heatmap_for_model(mdf_method, n, colorscale)
                if result is not None:
                    label = _method_label(display_name, m)
                    model_traces[label] = result
                    max_n_feat = max(max_n_feat, len(result[1]))
        else:
            result = _build_heatmap_for_model(mdf, n, colorscale)
            if result is not None:
                if has_method_col and len(methods_in_model) == 1:
                    m = next(iter(methods_in_model))
                    label = _method_label(display_name, m)
                else:
                    label = display_name
                model_traces[label] = result
                max_n_feat = max(max_n_feat, len(result[1]))

    if not model_traces:
        fig = create_figure(title=title or "Feature Interactions")
        fig.add_annotation(
            text="No interaction pairs found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        return fig

    size = adaptive_size(max_n_feat, per_item=50, base=150, min_size=350, max_size=800)

    # Single model — simple case, no dropdown
    if len(model_traces) == 1:
        label_key = next(iter(model_traces))
        trace, features, matrix = model_traces[label_key]
        display_features, full_features = truncate_labels(features)
        margin_l = margin_for_labels(display_features)
        margin_b = margin_for_labels(display_features, per_char=5, max_margin=200)
        suffix = f" — {label_key}" if label_key else ""
        fig = create_figure(
            title=title or f"Feature Interactions{suffix}",
            width=size, height=size,
        )
        fig.add_trace(trace)
        fig.update_layout(
            margin=dict(l=margin_l, b=margin_b),
            xaxis=dict(
                tickangle=-45, side="bottom",
                tickmode="array", tickvals=full_features,
                ticktext=display_features,
                title=dict(standoff=15),
            ),
            yaxis=dict(
                autorange="reversed",
                tickmode="array", tickvals=full_features,
                ticktext=display_features,
                title=dict(standoff=15),
            ),
        )
        apply_size_overrides(fig, width, height)
        return fig

    # Multiple models — add all traces, use dropdown to toggle visibility
    def _entry_title(entry_label):
        """Build a per-entry title that names the specific method.

        Empty entry labels (anonymous models from missing/NaN ``model_type``)
        fall through to a title without the trailing parenthetical.
        """
        label_str = str(entry_label) if entry_label else ''
        if label_str and 'SHAP' in label_str:
            return title or f"Feature Interactions \u2014 SHAP ({label_str})"
        if label_str and 'H\u00b2' in label_str:
            return title or f"Feature Interactions \u2014 H-statistic ({label_str})"
        if label_str:
            return title or f"Feature Interactions \u2014 {method_label} ({label_str})"
        return title or f"Feature Interactions \u2014 {method_label}"

    # Compute truncated labels per model and global adaptive margins
    label_map = {}
    all_features = set()
    for entry_label, (trace, features, matrix) in model_traces.items():
        disp, full = truncate_labels(features)
        label_map[entry_label] = (disp, full)
        all_features.update(features)
    all_disp, _ = truncate_labels(sorted(all_features))
    margin_l = margin_for_labels(all_disp)
    margin_b = margin_for_labels(all_disp, per_char=5, max_margin=200)

    first_model = next(iter(model_traces))
    fig = create_figure(title=_entry_title(first_model), width=size, height=size)

    trace_indices = {}  # entry_label -> trace_index
    for i, (entry_label, (trace, features, matrix)) in enumerate(model_traces.items()):
        trace.visible = (entry_label == first_model)
        fig.add_trace(trace)
        trace_indices[entry_label] = i

    n_traces = len(fig.data)

    buttons = []
    for entry_label in model_traces:
        vis = [False] * n_traces
        vis[trace_indices[entry_label]] = True
        trace, features, matrix = model_traces[entry_label]
        disp, full = label_map[entry_label]
        # Fall back to the method label (e.g. "H-statistic") when the entry
        # is anonymous so buttons never read as literal "None"/"nan"/"".
        # Anonymous entries arise when ``model_type`` is
        # missing/NaN; see `_is_missing_model_name` in the main loop.
        button_label = str(entry_label) if entry_label else method_label
        buttons.append(dict(
            label=button_label,
            method="update",
            args=[
                {"visible": vis},
                {
                    "title.text": f"<b>{_entry_title(entry_label)}</b>",
                    "xaxis.tickvals": full,
                    "xaxis.ticktext": disp,
                    "yaxis.tickvals": full,
                    "yaxis.ticktext": disp,
                },
            ],
        ))

    first_disp, first_full = label_map[first_model]
    fig.update_layout(
        margin=dict(l=margin_l, b=margin_b),
        xaxis=dict(
            tickangle=-45, side="bottom",
            tickmode="array", tickvals=first_full,
            ticktext=first_disp,
            title=dict(standoff=15),
        ),
        yaxis=dict(
            autorange="reversed",
            tickmode="array", tickvals=first_full,
            ticktext=first_disp,
            title=dict(standoff=15),
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.32, xanchor="left",
            y=1.0, yanchor="top",
            buttons=buttons,
            active=0,
            showactive=True,
            bgcolor="#f5f7fa",
            bordercolor="#ddd",
            font=dict(size=11),
        )],
    )

    apply_size_overrides(fig, width, height)
    return fig


# ── 2D Partial Dependence Plots ────────────────────────────────────────


def _axis_coords(grid, scale: Optional[str]) -> List[float]:
    """Plot coordinates for a numeric grid under *scale*.

    Plotly has no native ``log1p`` axis, so ``'log1p'`` grids are pre-transformed
    into log1p space here (the matching real-unit ticks come from
    :func:`_numeric_axis_layout`).  ``'linear'`` and ``'log'`` keep raw
    coordinates — Plotly positions a ``type='log'`` axis on log10 itself.
    """
    if scale == 'log1p':
        return list(np.log1p(np.asarray(grid, dtype=float)))
    return list(grid)


def _format_log1p_label(v: float) -> str:
    """Format a real-unit log1p tick value (integers without a decimal point)."""
    return str(int(v)) if v == int(v) else f"{v:.3g}"


def _nice_log1p_ticks(hi: float) -> List[float]:
    """Real-unit 1-2-5 'nice' tick values in ``[0, hi]`` for a log1p axis.

    Always includes ``0`` (a log1p axis represents it) and the data max.  Tiny
    sub-unit decades are skipped when ``hi >= 1`` because a log1p axis piles
    them up at the origin.
    """
    if not np.isfinite(hi) or hi <= 0:
        return [0.0]
    ticks = [0.0]
    top = int(np.floor(np.log10(hi)))
    bottom = top - 2 if hi < 1 else 0
    for e in range(bottom, top + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if 0 < v <= hi:
                ticks.append(float(v))
    if ticks[-1] < hi:
        ticks.append(float(hi))
    return ticks


def _numeric_axis_layout(grid, scale: Optional[str]) -> Dict[str, Any]:
    """Plotly axis-layout kwargs for a numeric axis under *scale*.

    Returns ``type`` plus ``tickvals``/``ticktext`` (``None`` to clear, so a
    dropdown switching away from a log1p surface resets its manual ticks).  A
    ``'log1p'`` axis is ``type='linear'`` with ticks placed at log1p positions
    but labeled in real units; ``'log'`` uses Plotly's native log10 axis.
    """
    if scale == 'log':
        return {'type': 'log', 'tickvals': None, 'ticktext': None}
    if scale == 'log1p':
        nice = _nice_log1p_ticks(float(np.max(np.asarray(grid, dtype=float))))
        return {
            'type': 'linear',
            'tickvals': list(np.log1p(nice)),
            'ticktext': [_format_log1p_label(v) for v in nice],
        }
    return {'type': 'linear', 'tickvals': None, 'ticktext': None}


def _axis_title(feature: str, scale: Optional[str]) -> str:
    """Feature name with a scale suffix for log/log1p axes."""
    if scale == 'log':
        return f"{feature} (log scale)"
    if scale == 'log1p':
        return f"{feature} (log1p scale)"
    return feature


def _dotted_axis(axis: str, title: str, layout: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten an axis title + :func:`_numeric_axis_layout` dict to dotted keys.

    Used to build ``go.Figure.update``-style dropdown button args (e.g.
    ``{'xaxis.type': 'log'}``).
    """
    out = {f"{axis}.title.text": title}
    for key, value in layout.items():
        out[f"{axis}.{key}"] = value
    return out


def _nested_pair_axis(
    feature: str, grid, grid_type: Optional[str], scale: Optional[str],
) -> Dict[str, Any]:
    """Nested ``dict``-form axis layout for one contour/heatmap pair axis.

    Categorical axes use a ``'category'`` axis (no scale); numeric axes honor
    their resolved linear/log/log1p spacing (with a scale-annotated title).
    """
    if grid_type == 'categorical':
        return dict(title=dict(text=feature, standoff=15), type='category')
    kw = dict(title=dict(text=_axis_title(feature, scale), standoff=15))
    kw.update(_numeric_axis_layout(grid, scale))
    return kw


def _dotted_pair_axis(
    axis: str, feature: str, grid, grid_type: Optional[str], scale: Optional[str],
) -> Dict[str, Any]:
    """Dotted-key axis layout for one pair axis (dropdown-button form).

    Mirrors :func:`_nested_pair_axis` but emits ``go.Figure.update``-style dotted
    keys: categorical axes use a ``'category'`` axis (no scale); numeric axes
    honor their resolved linear/log/log1p spacing.
    """
    if grid_type == 'categorical':
        return {f"{axis}.title.text": feature, f"{axis}.type": 'category'}
    return _dotted_axis(axis, _axis_title(feature, scale), _numeric_axis_layout(grid, scale))


def _trunc(name, max_chars=15):
    """Shorten *name* for dropdown labels."""
    name = str(name)
    return (name[:max_chars - 3] + "...") if len(name) > max_chars else name


def _build_pdp_heatmap_trace(
    surface: Dict[str, Any],
    colorscale: str,
    visible: bool = True,
) -> go.Heatmap:
    """Build a Heatmap trace for a PDP surface involving categoricals."""
    pdp = surface['pdp_surface']
    if not isinstance(pdp, list):
        pdp = [list(row) for row in pdp]

    f1 = surface['feature_1']
    f2 = surface['feature_2']

    # Determine axis labels
    y_labels = surface.get('grid_1_labels')
    x_labels = surface.get('grid_2_labels')

    # For axes without labels, generate from grid values
    if y_labels is None:
        if surface.get('grid_1_type') == 'categorical':
            y_labels = [str(v) for v in surface['grid_1']]
        else:
            y_labels = [f"{v:.3g}" for v in surface['grid_1']]
    if x_labels is None:
        if surface.get('grid_2_type') == 'categorical':
            x_labels = [str(v) for v in surface['grid_2']]
        else:
            x_labels = [f"{v:.3g}" for v in surface['grid_2']]

    # Build formatted text for annotations
    text = [[f"{v:.4g}" for v in row] for row in pdp]

    return go.Heatmap(
        z=pdp,
        x=x_labels,
        y=y_labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=colorscale,
        colorbar=dict(title="Predicted<br>Risk"),
        hovertemplate=(
            f"{f1}=%{{y}}<br>"
            f"{f2}=%{{x}}<br>"
            "Risk=%{z:.4g}<extra></extra>"
        ),
        visible=visible,
    )


def _contour_trace(
    grid_1, grid_2, pdp, f1: str, f2: str,
    scale_1: Optional[str], scale_2: Optional[str],
    colorscale: str, visible: bool = True,
) -> go.Contour:
    """Build a numeric × numeric Contour trace, honoring per-axis scale.

    For a ``'log1p'`` axis the coordinates are pre-transformed (Plotly has no
    log1p axis), so the hover is fed real values via ``customdata`` to keep the
    displayed feature values in original units.
    """
    x = _axis_coords(grid_2, scale_2)
    y = _axis_coords(grid_1, scale_1)
    kw: Dict[str, Any] = dict(
        x=x, y=y, z=pdp,
        colorscale=colorscale,
        colorbar=dict(title="Predicted<br>Risk"),
        contours=dict(showlabels=True, labelfont=dict(size=10)),
        visible=visible,
    )
    if scale_1 == 'log1p' or scale_2 == 'log1p':
        # customdata[i][j] = [x_real, y_real] aligned to z (rows=y/grid_1).
        kw['customdata'] = [
            [[float(gx), float(gy)] for gx in grid_2] for gy in grid_1
        ]
        kw['hovertemplate'] = (
            f"{f1}=%{{customdata[1]:.3g}}<br>"
            f"{f2}=%{{customdata[0]:.3g}}<br>"
            "Risk=%{z:.4g}<extra></extra>"
        )
    else:
        kw['hovertemplate'] = (
            f"{f1}=%{{y:.3g}}<br>"
            f"{f2}=%{{x:.3g}}<br>"
            "Risk=%{z:.4g}<extra></extra>"
        )
    return go.Contour(**kw)


def _build_pdp_contour_trace(
    surface: Dict[str, Any],
    colorscale: str,
    visible: bool = True,
) -> go.Contour:
    """Build a Contour trace for a numeric × numeric PDP surface."""
    grid_1 = list(surface['grid_1']) if not isinstance(surface['grid_1'], list) else surface['grid_1']
    grid_2 = list(surface['grid_2']) if not isinstance(surface['grid_2'], list) else surface['grid_2']
    pdp = (
        [list(row) for row in surface['pdp_surface']]
        if not isinstance(surface['pdp_surface'], list)
        else surface['pdp_surface']
    )
    return _contour_trace(
        grid_1, grid_2, pdp, surface['feature_1'], surface['feature_2'],
        surface.get('grid_1_scale') or 'linear',
        surface.get('grid_2_scale') or 'linear',
        colorscale, visible,
    )


def _classify_pair(
    grid_1_type: Optional[str], grid_2_type: Optional[str],
) -> str:
    """Classify a feature pair for rendering.

    - both numeric -> ``'contour'``
    - both categorical -> ``'heatmap'``
    - exactly one categorical (continuous × categorical) -> ``'lines'``

    The ``'lines'`` case draws one partial-dependence line per category level
    instead of interpolating a contour/heatmap across discrete levels.
    """
    g1_cat = grid_1_type == 'categorical'
    g2_cat = grid_2_type == 'categorical'
    if g1_cat and g2_cat:
        return 'heatmap'
    if g1_cat or g2_cat:
        return 'lines'
    return 'contour'


def _pdp_line_colors(n: int) -> List[str]:
    """Return *n* distinct line colors, cycling a qualitative palette."""
    from plotly.colors import qualitative
    palette = qualitative.Plotly
    return [palette[i % len(palette)] for i in range(n)]


def _build_pdp_line_traces(surface: Dict[str, Any], visible: bool = True):
    """Build one ``go.Scatter`` line per category level for a mixed pair.

    The continuous feature is the x-axis; each line is one level of the
    categorical feature. Returns ``(traces, x_title, y_title, legend_title)``.
    """
    pdp = surface['pdp_surface']
    if not isinstance(pdp, list):
        pdp = [list(row) for row in pdp]
    grid_1 = list(surface['grid_1'])
    grid_2 = list(surface['grid_2'])
    f1, f2 = surface['feature_1'], surface['feature_2']
    g1_cat = surface.get('grid_1_type') == 'categorical'

    n_rows, n_cols = len(grid_1), len(grid_2)
    if len(pdp) != n_rows or any(len(row) != n_cols for row in pdp):
        raise ValueError(
            f"pdp_surface shape does not match grids for ({f1}, {f2}): "
            f"expected ({n_rows}, {n_cols})."
        )

    def _resolve_labels(key, grid, feat):
        # Explicit None check (not ``or``): an empty list is a producer bug,
        # not "no labels", so surface it as a length mismatch rather than
        # silently substituting stringified grid values.
        labels = surface.get(key)
        if labels is None:
            return [str(v) for v in grid]
        if len(labels) != len(grid):
            raise ValueError(
                f"{key} length ({len(labels)}) does not match its grid "
                f"({len(grid)}) for feature '{feat}'."
            )
        return labels

    if g1_cat:
        # feature_1 categorical (rows); feature_2 continuous (cols)
        x_values = grid_2
        cont_scale = surface.get('grid_2_scale') or 'linear'
        labels = _resolve_labels('grid_1_labels', grid_1, f1)
        series = [(labels[i], pdp[i]) for i in range(n_rows)]
        x_title, legend_title = f2, f1
    else:
        # feature_2 categorical (cols); feature_1 continuous (rows)
        x_values = grid_1
        cont_scale = surface.get('grid_1_scale') or 'linear'
        labels = _resolve_labels('grid_2_labels', grid_2, f2)
        series = [
            (labels[j], [pdp[i][j] for i in range(n_rows)])
            for j in range(n_cols)
        ]
        x_title, legend_title = f1, f2

    # Plot coords may be log1p-transformed; customdata keeps hover in real units.
    x_plot = _axis_coords(x_values, cont_scale)
    colors = _pdp_line_colors(len(series))
    traces = []
    for k, (label, y_vals) in enumerate(series):
        traces.append(go.Scatter(
            x=x_plot, y=y_vals, mode='lines+markers', name=str(label),
            line=dict(color=colors[k], width=2.5),
            marker=dict(size=5),
            legendgroup=str(label),
            showlegend=True,
            visible=visible,
            customdata=list(x_values),
            hovertemplate=(
                f"{x_title}=%{{customdata:.3g}}<br>"
                f"{legend_title}={label}<br>"
                "Risk=%{y:.4g}<extra></extra>"
            ),
        ))
    return traces, x_title, "Predicted Risk", legend_title


def _lines_axis_facts(surface: Dict[str, Any]) -> Tuple[str, Any, str]:
    """Return ``(x_axis_title, continuous_grid, continuous_scale)`` for a mixed pair.

    The continuous feature is the x-axis (its scale carries over); the
    categorical feature becomes the legend.
    """
    g1_cat = surface.get('grid_1_type') == 'categorical'
    if g1_cat:
        return surface['feature_2'], surface['grid_2'], (surface.get('grid_2_scale') or 'linear')
    return surface['feature_1'], surface['grid_1'], (surface.get('grid_1_scale') or 'linear')


def _gallery_layout_args(
    surface: Dict[str, Any], kind: str, legend_title: Optional[str],
    base_title: str, title_label: str,
) -> Dict[str, Any]:
    """Build a dropdown button's layout-update dict (dotted keys) for one surface."""
    args: Dict[str, Any] = {"title.text": f"<b>{base_title} — {title_label}</b>"}
    if kind == 'lines':
        x_title, x_grid, x_scale = _lines_axis_facts(surface)
        args.update(_dotted_axis(
            "xaxis", _axis_title(x_title, x_scale),
            _numeric_axis_layout(x_grid, x_scale),
        ))
        args["yaxis.title.text"] = "Predicted Risk"
        args["yaxis.type"] = 'linear'
        args["showlegend"] = True
        args["legend.title.text"] = legend_title or ""
    else:
        args.update(_dotted_pair_axis(
            "xaxis", surface['feature_2'], surface['grid_2'],
            surface.get('grid_2_type'), surface.get('grid_2_scale') or 'linear',
        ))
        args.update(_dotted_pair_axis(
            "yaxis", surface['feature_1'], surface['grid_1'],
            surface.get('grid_1_type'), surface.get('grid_1_scale') or 'linear',
        ))
        args["showlegend"] = False
    return args


def plot_interaction_pdp(
    grid_1,
    grid_2,
    pdp_surface,
    feature_1: str,
    feature_2: str,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    colorscale: str = "Viridis",
    grid_1_type: Optional[str] = None,
    grid_2_type: Optional[str] = None,
    grid_1_labels: Optional[List[str]] = None,
    grid_2_labels: Optional[List[str]] = None,
    grid_1_scale: str = 'linear',
    grid_2_scale: str = 'linear',
) -> go.Figure:
    """Plot 2D partial dependence for a single feature pair.

    Picks the rendering from the pair's grid types: a contour for
    numeric × numeric, a heatmap for categorical × categorical, and a
    line-per-category overlay (continuous feature on the x-axis) for a mixed
    continuous × categorical pair. The overlay avoids interpolating a surface
    across discrete category levels.

    Parameters
    ----------
    grid_1 : array-like
        Grid values for feature_1 (rows).
    grid_2 : array-like
        Grid values for feature_2 (columns).
    pdp_surface : array-like
        2D array of shape ``(len(grid_1), len(grid_2))`` with partial
        dependence values.
    feature_1 : str
        First feature name for axis labels.
    feature_2 : str
        Second feature name for axis labels.
    title : str, optional
        Figure title.
    width : int, optional
        Override figure width (pixels).
    height : int, optional
        Override figure height (pixels).
    colorscale : str, default="Viridis"
        Plotly colorscale.
    grid_1_type : str, optional
        ``'numeric'`` or ``'categorical'`` for grid_1.  Defaults to ``'numeric'``.
    grid_2_type : str, optional
        ``'numeric'`` or ``'categorical'`` for grid_2.  Defaults to ``'numeric'``.
    grid_1_labels : list of str, optional
        Human-readable labels for the grid_1 categorical axis.
    grid_2_labels : list of str, optional
        Human-readable labels for the grid_2 categorical axis.
    grid_1_scale : str, default='linear'
        ``'linear'``, ``'log'``, or ``'log1p'`` for the grid_1 (y) axis (numeric
        axes only).  ``'log'`` uses Plotly's native ``type='log'`` with grid
        values in original units; ``'log1p'`` plots log1p-transformed
        coordinates on a linear axis with real-unit tick labels.
    grid_2_scale : str, default='linear'
        ``'linear'``, ``'log'``, or ``'log1p'`` for the grid_2 (x) axis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    g1_type = grid_1_type or 'numeric'
    g2_type = grid_2_type or 'numeric'
    kind = _classify_pair(g1_type, g2_type)

    grid_1 = list(grid_1) if not isinstance(grid_1, list) else grid_1
    grid_2 = list(grid_2) if not isinstance(grid_2, list) else grid_2
    if not isinstance(pdp_surface, list):
        pdp_surface = [list(row) for row in pdp_surface]

    fig = create_figure(
        title=title or f"2D Partial Dependence — {feature_1} × {feature_2}",
        width=500, height=450,
    )

    if kind == 'lines':
        # Continuous × categorical: one line per category level with the
        # continuous feature on the x-axis (no interpolation across levels).
        surface = {
            'grid_1': grid_1, 'grid_2': grid_2, 'pdp_surface': pdp_surface,
            'feature_1': feature_1, 'feature_2': feature_2,
            'grid_1_type': g1_type, 'grid_2_type': g2_type,
            'grid_1_labels': grid_1_labels, 'grid_2_labels': grid_2_labels,
            'grid_1_scale': grid_1_scale, 'grid_2_scale': grid_2_scale,
        }
        traces, x_title, y_title, legend_title = _build_pdp_line_traces(surface)
        for tr in traces:
            fig.add_trace(tr)
        # The continuous axis may use a log/log1p scale; the categorical cannot.
        _, x_grid, x_scale = _lines_axis_facts(surface)
        fig.update_layout(
            xaxis=_nested_pair_axis(x_title, x_grid, 'numeric', x_scale),
            yaxis=dict(title=dict(text=y_title, standoff=15)),
            legend=dict(title=dict(text=legend_title)),
        )
        apply_size_overrides(fig, width, height)
        return fig

    if kind == 'heatmap':
        # Categorical × categorical
        y_labels = grid_1_labels if grid_1_labels is not None else [str(v) for v in grid_1]
        x_labels = grid_2_labels if grid_2_labels is not None else [str(v) for v in grid_2]
        text = [[f"{v:.4g}" for v in row] for row in pdp_surface]

        # Adaptive margins for categorical labels (matches gallery layout)
        margin_l = margin_for_labels(y_labels, min_margin=80, padding=20)
        margin_b = margin_for_labels(x_labels, per_char=5, min_margin=70)

        fig.add_trace(go.Heatmap(
            z=pdp_surface,
            x=x_labels,
            y=y_labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale=colorscale,
            colorbar=dict(title="Predicted<br>Risk"),
            hovertemplate=(
                f"{feature_1}=%{{y}}<br>"
                f"{feature_2}=%{{x}}<br>"
                "Risk=%{z:.4g}<extra></extra>"
            ),
        ))
        fig.update_layout(
            xaxis=dict(title=dict(text=feature_2, standoff=15)),
            yaxis=dict(title=dict(text=feature_1, standoff=15)),
            margin=dict(l=margin_l, b=margin_b),
        )
        apply_size_overrides(fig, width, height)
        return fig

    # Contour for numeric × numeric
    fig.add_trace(_contour_trace(
        grid_1, grid_2, pdp_surface, feature_1, feature_2,
        grid_1_scale, grid_2_scale, colorscale,
    ))
    # Plotly positions a 'log' axis on log10 (x/y stay in original units);
    # 'log1p' coordinates are pre-transformed with real-unit ticks instead.
    fig.update_layout(
        xaxis=_nested_pair_axis(feature_2, grid_2, 'numeric', grid_2_scale),
        yaxis=_nested_pair_axis(feature_1, grid_1, 'numeric', grid_1_scale),
    )

    apply_size_overrides(fig, width, height)
    return fig


def plot_interaction_pdp_gallery(
    surfaces: List[Dict[str, Any]],
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Multi-pair 2D PDP with dropdown selector.

    Picks the rendering per surface from its grid types: contour
    (numeric × numeric), heatmap (categorical × categorical), or a
    line-per-category overlay (continuous × categorical).

    Parameters
    ----------
    surfaces : list of dict
        Each dict has keys: ``feature_1``, ``feature_2``, ``grid_1``,
        ``grid_2``, ``pdp_surface``, and optionally ``model_type``,
        ``grid_1_type``, ``grid_2_type``, ``grid_1_labels``,
        ``grid_2_labels``.
    title : str, optional
        Base figure title.
    width : int, optional
        Override figure width (pixels).
    height : int, optional
        Override figure height (pixels).
    colorscale : str, default="Viridis"
        Plotly colorscale.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not surfaces:
        fig = create_figure(title=title or "2D Partial Dependence")
        fig.add_annotation(
            text="No PDP surface data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        return fig

    base_title = title or "2D Partial Dependence"

    # Single surface — delegate to plot_interaction_pdp
    if len(surfaces) == 1:
        s = surfaces[0]
        model_label = s.get('model_type', '')
        pair_label = f"{s['feature_1']} × {s['feature_2']}"
        full_label = f"{pair_label} ({model_label})" if model_label else pair_label
        return plot_interaction_pdp(
            s['grid_1'], s['grid_2'], s['pdp_surface'],
            s['feature_1'], s['feature_2'],
            title=f"{base_title} — {full_label}",
            width=width, height=height, colorscale=colorscale,
            grid_1_type=s.get('grid_1_type'),
            grid_2_type=s.get('grid_2_type'),
            grid_1_labels=s.get('grid_1_labels'),
            grid_2_labels=s.get('grid_2_labels'),
        )

    # Multiple surfaces — dropdown
    first = surfaces[0]
    first_f1, first_f2 = first['feature_1'], first['feature_2']
    first_model = first.get('model_type', '')
    first_label = f"{first_f1} × {first_f2}"
    if first_model:
        first_label = f"{first_label} ({first_model})"

    # Compute adaptive margins from the longest axis labels across surfaces
    y_labels_all: List[str] = []  # feature_1 labels → left margin
    x_labels_all: List[str] = []  # feature_2 labels → bottom margin
    for s in surfaces:
        if s.get('grid_1_labels'):
            y_labels_all.extend(s['grid_1_labels'])
        if s.get('grid_2_labels'):
            x_labels_all.extend(s['grid_2_labels'])
    margin_l = margin_for_labels(y_labels_all, min_margin=80, padding=20) if y_labels_all else 80
    margin_b = margin_for_labels(x_labels_all, per_char=5, min_margin=70) if x_labels_all else 70

    fig = create_figure(
        title=f"{base_title} — {first_label}",
        width=700, height=450,
    )

    # Add a contiguous block of traces per surface. Contour/heatmap surfaces
    # are one trace each; a continuous × categorical surface is a block of one
    # line per category level. Track each surface's (start, count) so the
    # dropdown can toggle whole blocks.
    blocks: List[Tuple[int, int]] = []
    kinds: List[str] = []
    legend_titles: List[Optional[str]] = []
    start = 0
    for i, s in enumerate(surfaces):
        kind = _classify_pair(s.get('grid_1_type'), s.get('grid_2_type'))
        legend_title = None
        if kind == 'heatmap':
            traces = [_build_pdp_heatmap_trace(s, colorscale, visible=(i == 0))]
        elif kind == 'lines':
            traces, _, _, legend_title = _build_pdp_line_traces(s, visible=(i == 0))
        else:
            traces = [_build_pdp_contour_trace(s, colorscale, visible=(i == 0))]
        for tr in traces:
            fig.add_trace(tr)
        blocks.append((start, len(traces)))
        kinds.append(kind)
        legend_titles.append(legend_title)
        start += len(traces)

    # Build dropdown buttons — each toggles its surface's whole trace block.
    n_traces = len(fig.data)
    buttons = []
    for i, s in enumerate(surfaces):
        vis = [False] * n_traces
        b_start, b_count = blocks[i]
        for t in range(b_start, b_start + b_count):
            vis[t] = True
        model = s.get('model_type', '')
        f1, f2 = s['feature_1'], s['feature_2']

        # Truncated label for dropdown button, full label for title
        pair_short = f"{_trunc(f1)} × {_trunc(f2)}"
        pair_full = f"{f1} × {f2}"
        button_label = f"{pair_short} ({model})" if model else pair_short
        title_label = f"{pair_full} ({model})" if model else pair_full

        layout_args = _gallery_layout_args(
            s, kinds[i], legend_titles[i], base_title, title_label,
        )
        buttons.append(dict(
            label=button_label,
            method="update",
            args=[{"visible": vis}, layout_args],
        ))

    # Initial layout reflects the first surface's render kind.
    first_kind = kinds[0]
    if first_kind == 'lines':
        x_title0, x_grid0, x_scale0 = _lines_axis_facts(first)
        init_xaxis = _nested_pair_axis(x_title0, x_grid0, 'numeric', x_scale0)
        init_yaxis = dict(title=dict(text="Predicted Risk", standoff=15))
        init_showlegend = True
        init_legend = dict(title=dict(text=legend_titles[0] or ""))
    else:
        init_xaxis = _nested_pair_axis(
            first_f2, first['grid_2'], first.get('grid_2_type', 'numeric'),
            first.get('grid_2_scale') or 'linear',
        )
        init_yaxis = _nested_pair_axis(
            first_f1, first['grid_1'], first.get('grid_1_type', 'numeric'),
            first.get('grid_1_scale') or 'linear',
        )
        init_showlegend = False
        init_legend = None

    fig.update_layout(
        margin=dict(l=margin_l, b=margin_b),
        xaxis=init_xaxis,
        yaxis=init_yaxis,
        showlegend=init_showlegend,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.32, xanchor="left",
            y=1.0, yanchor="top",
            buttons=buttons,
            active=0,
            showactive=True,
            bgcolor="#f5f7fa",
            bordercolor="#ddd",
            font=dict(size=11),
        )],
    )
    if init_legend is not None:
        fig.update_layout(legend=init_legend)

    apply_size_overrides(fig, width, height)
    return fig
