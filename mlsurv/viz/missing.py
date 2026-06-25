"""Missing data heatmap visualization."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import leaves_list, linkage

from mlsurv.logging_utils import get_logger, log_exception

_logger = get_logger("viz.missing")

# Present vs. missing must differ in luminance (not just hue) so the heatmap
# stays readable in grayscale print and under red-green color blindness.
_PRESENT_COLOR = "#ECECEC"  # light grey
_MISSING_COLOR = "#0072B2"  # Okabe-Ito blue: CVD-safe, dark for grayscale contrast


def plot_missing(
    X: pd.DataFrame,
    columns: list[str] | None = None,
    max_samples: int = 200,
    title: str = "Missing Data Pattern",
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Create a binary present/missing heatmap.

    Rows and columns are clustered using Ward's linkage on the
    missingness pattern so correlated missing values appear as
    contiguous blocks.

    Parameters
    ----------
    X : pd.DataFrame
        Input data (may contain NaN values).
    columns : list[str] or None
        Columns to include.  ``None`` (default) auto-detects columns
        with at least one missing value.
    max_samples : int, default=200
        Maximum number of rows to display (subsampled if exceeded).
    title : str
        Figure title.
    width : int or None
        Override the default auto-sized width (pixels).
    height : int or None
        Override the default auto-sized height (pixels).

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # Auto-detect columns with missing data
    if columns is None:
        columns = [c for c in X.columns if X[c].isnull().any()]

    if not columns:
        # No missing data — return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No missing data detected",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16),
            )],
        )
        return fig

    subset = X[columns]

    # Subsample for large datasets
    if len(subset) > max_samples:
        subset = subset.sample(max_samples, random_state=0)

    binary = subset.isnull().astype(int)

    # Ward's linkage needs at least 3 observations to be meaningful
    rows_clustered = True
    if len(binary) > 2:
        try:
            row_order = leaves_list(linkage(binary.values, method='ward'))
            binary = binary.iloc[row_order]
        except (ValueError, MemoryError) as exc:
            rows_clustered = False
            log_exception(
                _logger, exc, "missing heatmap: row clustering",
                warn_user=True,
            )

    # Cluster columns
    if len(binary.columns) > 2:
        try:
            col_order = leaves_list(linkage(binary.values.T, method='ward'))
            binary = binary.iloc[:, col_order]
        except (ValueError, MemoryError) as exc:
            log_exception(
                _logger, exc, "missing heatmap: column clustering",
                warn_user=True,
            )

    n_features = len(columns)
    # +80px gutter keeps the heatmap its usual size while leaving room for the
    # right-hand Present/Missing legend (see the wider right margin below).
    w = width if width is not None else max(350, min(700, 100 + n_features * 50)) + 80
    h = height if height is not None else max(300, min(600, len(binary) * 3))

    y_label = "Sample (clustered)" if rows_clustered else "Sample (original order)"

    fig = go.Figure(data=go.Heatmap(
        z=binary.values,
        x=list(binary.columns),
        y=list(range(len(binary))),
        colorscale=[[0, _PRESENT_COLOR], [1, _MISSING_COLOR]],
        # Pin the range so 0->present / 1->missing maps stably even when a
        # clustered subset happens to be all-present or all-missing.
        zmin=0,
        zmax=1,
        showscale=False,
    ))

    # Invisible marker traces carry the legend key — heatmaps don't produce
    # legend entries, so a reader can't otherwise tell which color is missing.
    for color, label in ((_PRESENT_COLOR, "Present"), (_MISSING_COLOR, "Missing")):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color, symbol="square",
                        line=dict(width=1, color="#888")),
            name=label, showlegend=True, hoverinfo="skip",
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Feature',
        yaxis_title=y_label,
        width=w,
        height=h,
        # Extra right margin holds the legend clear of the plot and title.
        margin=dict(l=50, r=110, t=40, b=60),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0,
                    xanchor="left", x=1.02),
    )
    return fig
