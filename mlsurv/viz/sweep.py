"""Parameter-sweep visualization."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go

from mlsurv.constants import get_metric_display_name
from mlsurv.viz._palettes import get_model_color
from mlsurv.viz._theme import create_figure, add_ci_ribbon


def plot_sweep(
    data: pd.DataFrame,
    param_name: str,
    *,
    metric: str = "harrell",
    phase: str = "cv",
    ribbon: Optional[str] = None,
    width: Union[str, int] = "double",
    height: Union[str, int] = "medium",
) -> go.Figure:
    """Line plot of metric vs. swept parameter value.

    Parameters
    ----------
    data : DataFrame
        Long-format sweep data with columns ``[param_value, model,
        phase, metric, mean, std, ci_lower, ci_upper]``.
    param_name : str
        Swept parameter name (used as x-axis label).
    metric : str
        Which metric to plot.
    phase : str
        ``'cv'`` for cross-validation means or ``'test'`` for held-out
        test-set point estimates.
    ribbon : str or None
        What the shaded ribbon shows: ``'std'`` for mean |plusmn| std,
        ``'ci'`` for 95 % confidence interval, or ``None`` to use the
        default (``'std'`` for CV, ``'ci'`` for test).
    width : str or int
        Figure width — named size or pixel value (see :func:`create_figure`).
    height : str or int
        Figure height — named size or pixel value (see :func:`create_figure`).

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``data`` is empty, or if no rows match the requested ``metric``
        and ``phase``.
    """
    if data.empty:
        raise ValueError("No sweep data to plot. The DataFrame is empty.")

    # Filter by phase if column exists (backward compat for old DataFrames)
    if "phase" in data.columns:
        subset = data.loc[
            (data["metric"] == metric) & (data["phase"] == phase)
        ].copy()
    else:
        subset = data.loc[data["metric"] == metric].copy()

    if subset.empty:
        available = sorted(data["metric"].unique())
        raise ValueError(
            f"No data for metric '{metric}' (phase='{phase}'). "
            f"Available metrics: {available}"
        )

    # Resolve ribbon mode
    if ribbon is None:
        ribbon = "std" if phase == "cv" else "ci"

    models = sorted(subset["model"].unique())
    metric_display = get_metric_display_name(metric)
    phase_label = "CV" if phase == "cv" else "Test"

    # Subtitle annotation
    has_std = subset["std"].notna().any()
    has_ci = subset["ci_upper"].notna().any()
    if ribbon == "ci" and has_ci:
        ribbon_note = ", 95% CI"
    elif ribbon == "std" and has_std:
        ribbon_note = ", \u00b1 std"
    else:
        ribbon_note = ""

    fig = create_figure(
        title=f"Parameter Sweep: {param_name}<br>"
              f"<span style='font-size:0.75em; font-weight:normal'>"
              f"{phase_label} {metric_display}{ribbon_note}</span>",
        width=width,
        height=height,
    )

    for model_name in models:
        mdata = subset.loc[subset["model"] == model_name].sort_values("param_value")
        color = get_model_color(model_name)

        # Ribbon
        if ribbon == "ci" and mdata["ci_upper"].notna().any():
            add_ci_ribbon(
                fig,
                x=mdata["param_value"],
                upper=mdata["ci_upper"],
                lower=mdata["ci_lower"],
                color=color,
                legendgroup=model_name,
            )
        elif ribbon == "std" and mdata["std"].notna().any():
            add_ci_ribbon(
                fig,
                x=mdata["param_value"],
                upper=mdata["mean"] + mdata["std"],
                lower=mdata["mean"] - mdata["std"],
                color=color,
                legendgroup=model_name,
            )

        # Mean line — show ±std on hover when available
        if has_std:
            customdata = mdata["std"].tolist()
            hover = (
                f"<b>{model_name}</b><br>"
                f"{param_name}=%{{x}}<br>"
                f"{metric_display}=%{{y:.4f}} "
                f"\u00b1 %{{customdata:.4f}}<extra></extra>"
            )
        else:
            customdata = None
            hover = (
                f"<b>{model_name}</b><br>"
                f"{param_name}=%{{x}}<br>"
                f"{metric_display}=%{{y:.4f}}<extra></extra>"
            )

        fig.add_trace(go.Scatter(
            x=mdata["param_value"].tolist(),
            y=mdata["mean"].tolist(),
            mode="lines+markers",
            name=model_name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            legendgroup=model_name,
            customdata=customdata,
            hovertemplate=hover,
        ))

    fig.update_layout(
        xaxis_title=param_name,
        yaxis_title=metric_display,
    )

    return fig
