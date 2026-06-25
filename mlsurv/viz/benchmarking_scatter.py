"""Scatter plot of model risk score vs. reference score.

One figure per benchmark, one subplot per reference, dropdown switches
the active model.

Pair this with the Spearman/Pearson correlation matrices already rendered
by the benchmarking report — the matrices summarise the relationship
with a single number; the scatter exposes the shape, leverage points,
and nonlinear disagreement between the two coefficients.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlsurv.viz._palettes import (
    build_benchmark_palette,
    get_model_display_name as _model_display,
)


_SUBPLOT_COLS = 2  # wrap after 2 references per row


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return "p<0.001"
    return f"p={p:.3f}"


def _ols_line(x: np.ndarray, y: np.ndarray):
    """Return (x_endpoints, y_endpoints) for the OLS best-fit line.

    Uses ``np.polyfit(deg=1)``. Returns None when fewer than 2 finite
    points or the inputs have no variance in x.
    """
    if len(x) < 2:
        return None
    if np.nanstd(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    return x_line, y_line


def plot_benchmark_scatter(
    model_risk_scores: Dict[str, np.ndarray],
    reference_arrays: Dict[str, np.ndarray],
    correlation: pd.DataFrame,
    *,
    palette: Optional[Dict[str, str]] = None,
    add_fit: bool = True,
    title: str = "Model vs. Reference Risk Scores",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """Scatter grid with a dropdown to pick the active model.

    Parameters
    ----------
    model_risk_scores : dict[str, np.ndarray]
        Test-aligned model predictions, one array per model.
    reference_arrays : dict[str, np.ndarray]
        Test-aligned reference scores, one array per reference. All
        arrays must share row order with the model-score arrays.
    correlation : pd.DataFrame
        Output of :func:`_compute_correlations` — columns
        ``model``, ``reference``, ``spearman_rho``, ``spearman_p``,
        ``pearson_r``, ``pearson_p``. Used to annotate each subplot.
    palette : dict, optional
        ``{name: color}``. If omitted, built via
        :func:`build_benchmark_palette`.
    add_fit : bool
        Overlay an OLS best-fit line per subplot.
    title : str
        Figure title.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``model_risk_scores`` or ``reference_arrays`` is empty, or if a
        model-score array and a reference array differ in length.
    """
    if not model_risk_scores:
        raise ValueError("model_risk_scores is empty — nothing to plot.")
    if not reference_arrays:
        raise ValueError("reference_arrays is empty — nothing to plot.")

    model_names = list(model_risk_scores.keys())
    reference_names = list(reference_arrays.keys())
    if palette is None:
        palette = build_benchmark_palette(model_names, reference_names)

    # Build an (model, reference) -> correlation-row lookup
    corr_lookup: Dict[tuple, pd.Series] = {}
    if correlation is not None and not correlation.empty:
        for _, row in correlation.iterrows():
            corr_lookup[(row["model"], row["reference"])] = row

    n_refs = len(reference_names)
    n_cols = min(_SUBPLOT_COLS, n_refs)
    n_rows = math.ceil(n_refs / n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=reference_names,
        horizontal_spacing=0.18,
        vertical_spacing=0.20 if n_rows > 1 else 0.14,
    )

    # Track trace indices per model so the dropdown can toggle visibility
    # in O(n) and annotations can be swapped in lockstep.
    model_trace_ranges: Dict[str, tuple] = {}
    # Per-model, per-reference annotation list (subplot-relative via xref/yref)
    model_annotations: Dict[str, list] = {m: [] for m in model_names}

    # Pre-compute row/col for each reference
    ref_layout: Dict[str, tuple] = {}
    for i, ref in enumerate(reference_names):
        ref_layout[ref] = (i // n_cols + 1, i % n_cols + 1)

    for mi, model in enumerate(model_names):
        start_idx = len(fig.data)
        m_arr = np.asarray(model_risk_scores[model], dtype=float)
        color = palette.get(model) or "#1f77b4"

        for ref in reference_names:
            r_arr = np.asarray(reference_arrays[ref], dtype=float)
            if len(r_arr) != len(m_arr):
                raise ValueError(
                    f"Length mismatch: model {model!r} has "
                    f"{len(m_arr)} rows but reference {ref!r} has "
                    f"{len(r_arr)} rows."
                )

            mask = np.isfinite(r_arr) & np.isfinite(m_arr)
            x = r_arr[mask]
            y = m_arr[mask]
            row, col = ref_layout[ref]

            visible = (mi == 0)
            display = _model_display(model)
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode="markers",
                    marker=dict(color=color, size=6, opacity=0.7),
                    name=f"{display} vs {ref}",
                    showlegend=False,
                    visible=visible,
                    hovertemplate=(
                        f"{display}<br>{ref}=%{{x:.3g}}"
                        f"<br>score=%{{y:.3g}}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

            if add_fit:
                line = _ols_line(x, y)
                if line is not None:
                    sx, sy = line
                    fig.add_trace(
                        go.Scatter(
                            x=sx, y=sy,
                            mode="lines",
                            line=dict(color=color, width=2, dash="solid"),
                            name=f"{display} OLS",
                            showlegend=False,
                            visible=visible,
                            hoverinfo="skip",
                        ),
                        row=row, col=col,
                    )

            # Axis titles on each subplot
            fig.update_xaxes(title_text=ref, row=row, col=col)
            fig.update_yaxes(title_text="Model risk", row=row, col=col)

            # Build annotation for this (model, reference)
            stats = corr_lookup.get((model, ref))
            if stats is not None:
                rho = stats.get("spearman_rho", np.nan)
                rho_p = stats.get("spearman_p", np.nan)
                r = stats.get("pearson_r", np.nan)
                r_p = stats.get("pearson_p", np.nan)
                text = (
                    f"ρ={rho:.2f} ({_format_p(rho_p)})"
                    f"<br>r={r:.2f} ({_format_p(r_p)})"
                )
            else:
                text = "(no correlation)"

            # Subplot-relative annotation: xref/yref anchored to the
            # subplot axes; top-left corner (paper=False).
            axis_idx = (row - 1) * n_cols + col
            xref = "x domain" if axis_idx == 1 else f"x{axis_idx} domain"
            yref = "y domain" if axis_idx == 1 else f"y{axis_idx} domain"
            model_annotations[model].append(dict(
                xref=xref, yref=yref,
                x=0.02, y=0.98, xanchor="left", yanchor="top",
                showarrow=False,
                text=text,
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="#ccc",
                borderwidth=1,
                font=dict(size=11),
            ))

        model_trace_ranges[model] = (start_idx, len(fig.data))

    # Preserve the subplot titles added by make_subplots and append the
    # per-model annotations for the initial (first-model) view.
    base_annotations = list(fig.layout.annotations)
    initial_annotations = base_annotations + model_annotations[model_names[0]]

    # Dropdown: for each model, build a visibility vector + annotation set
    n_traces = len(fig.data)
    buttons = []
    for model in model_names:
        vis = [False] * n_traces
        start, end = model_trace_ranges[model]
        for j in range(start, end):
            vis[j] = True
        buttons.append(dict(
            label=_model_display(model),
            method="update",
            args=[
                {"visible": vis},
                {"annotations": base_annotations + model_annotations[model]},
            ],
        ))

    default_width = width or (380 * n_cols + 120)
    default_height = height or (340 * n_rows + 140)

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>"),
        annotations=initial_annotations,
        margin=dict(l=70, r=40, t=110, b=60),
        width=default_width,
        height=default_height,
        showlegend=False,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.0, xanchor="right",
            y=1.12, yanchor="top",
            buttons=buttons,
            active=0,
            showactive=True,
            bgcolor="#f5f7fa",
            bordercolor="#ddd",
            font=dict(size=11),
        )],
    )

    return fig
