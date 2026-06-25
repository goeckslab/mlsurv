"""KM survival plot renderers (private implementation).

Public symbols:
    _plot_km_overall, _rmst_subtitle, _add_rmst_overlay,
    _plot_km_risk, _apply_rmst_if_requested, _annotate_pvalue,
    _plot_km_by_group, _add_by_group_risk_table,
    _plot_km_panel_grid, _resolve_panel_axes, _apply_panel_grid_layout,
    _make_km_panel_subplots, _add_maxstat_panel_annotation,
    _render_km_panel, _km_combined, _survival_grid
"""

import math
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlsurv._warnings import InternalWarning
from mlsurv.viz._palettes import (
    MANUSCRIPT_DEFAULTS,
    get_model_display_name,
    get_risk_colors,
    hex_to_rgba,
)
from mlsurv.viz._theme import create_figure
from mlsurv.viz._survival_figure import (
    _add_km_trace,
    _add_risk_table,
    _apply_km_risk_layout,
    _build_km_title_text,
    _fit_km_groups,
    _make_km_risk_figure,
)
from mlsurv.viz._survival_helpers import (
    _MAXSTAT_LABELS,
    _ci_band_subtitle,
    _create_risk_groups,
    _pct_range_labels,
    _resolve_axes_params,
    _resolve_risk_groups,
)
from mlsurv.viz._survival_logrank import (
    _add_logrank_annotation,
    _collect_logrank,
)
from mlsurv.viz._survival_maxstat import _resolve_maxstat


def _plot_km_overall(df, time_col, event_col, time_limit, show_ci, title,
                     time_unit=None, xlim=None, ylim=None, subtitle=None):
    """Single overall KM curve (no grouping)."""
    from lifelines import KaplanMeierFitter

    df = df.copy()
    time_limit, x_lo, x_hi, y_lo, y_hi, time_label = _resolve_axes_params(
        df, time_col, time_limit, xlim, ylim, time_unit,
    )

    subtitle = _ci_band_subtitle(subtitle, show_ci)
    # Build the title via update_layout (not create_figure) so the composed
    # "<b>..</b><br><sup>..</sup>" text is not wrapped in another <b>.
    fig = create_figure(title=None, width="single", height=330)

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col].clip(upper=time_limit), df[event_col])
    _add_km_trace(fig, kmf, df, "#1f77b4", "Overall", show_ci)

    layout_kwargs = dict(
        title=dict(text=_build_km_title_text(title, subtitle)),
        xaxis_title=time_label, yaxis_title="Survival Probability",
        xaxis=dict(range=[x_lo, x_hi]),
        yaxis=dict(range=[y_lo, y_hi]),
        hovermode="x unified",
    )
    if subtitle:
        layout_kwargs["margin"] = dict(t=MANUSCRIPT_DEFAULTS["margin_t"] + 30)
    fig.update_layout(**layout_kwargs)
    return fig


def _rmst_subtitle(rmst_result):
    """Build a subtitle string from an RMSTResult or PairwiseRMSTResult."""
    from mlsurv.evaluation.metrics import RMSTResult
    rmst_vals = rmst_result.rmst
    labels = list(rmst_vals.keys())
    if isinstance(rmst_result, RMSTResult):
        p_str = "<0.001" if rmst_result.p_value < 0.001 else f"{rmst_result.p_value:.3f}"
        return (
            f"RMST({labels[0]})={rmst_vals[labels[0]]:.1f}, "
            f"RMST({labels[1]})={rmst_vals[labels[1]]:.1f}, "
            f"\u0394={rmst_result.diff:.1f} "
            f"(95% CI [{rmst_result.ci_lower:.1f}, {rmst_result.ci_upper:.1f}], "
            f"p={p_str})"
        )
    # PairwiseRMSTResult: show per-group RMST values
    parts = [f"RMST({lbl})={rmst_vals[lbl]:.1f}" for lbl in labels]
    return ", ".join(parts) + " (pairwise comparisons via return_data)"


def _add_rmst_overlay(fig, df, time_col, event_col, colors, group_labels,
                      rmst_tau, row, time_limit=None,
                      group_col="risk_group"):
    """Add RMST shaded areas and vertical tau line to a KM figure.

    Returns the RMSTResult/PairwiseRMSTResult or None on failure.
    """
    from lifelines import KaplanMeierFitter
    from mlsurv.evaluation.metrics import (
        _km_step_points, rmst_test, rmst_test_pairwise,
    )

    groups_sorted = sorted(df[group_col].unique(), key=str)
    n_groups = len(groups_sorted)
    if n_groups < 2:
        warnings.warn(
            f"show_rmst=True requires >= 2 groups in column {group_col!r}, "
            f"found {n_groups}; RMST overlay skipped.",
            InternalWarning, stacklevel=5,
        )
        return None

    times = df[time_col].values.copy()
    if time_limit is not None:
        times = np.clip(times, None, time_limit)
    events = df[event_col].values
    group_arr = df[group_col].values

    try:
        if n_groups == 2:
            result = rmst_test(times, events, group_arr, tau=rmst_tau)
        else:
            result = rmst_test_pairwise(times, events, group_arr, tau=rmst_tau)
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        warnings.warn(
            f"RMST computation failed ({type(exc).__name__}: {exc}); "
            f"show_rmst=True ignored.",
            InternalWarning, stacklevel=5,
        )
        return None

    tau = result.tau
    add_kw = dict(row=row, col=1) if row else {}

    # Shaded area under each KM curve up to tau
    for i, group_val in enumerate(groups_sorted):
        group_df = df[df[group_col] == group_val]
        kmf = KaplanMeierFitter()
        fit_times = group_df[time_col]
        if time_limit is not None:
            fit_times = fit_times.clip(upper=time_limit)
        kmf.fit(fit_times, group_df[event_col])

        t_pts, s_pts = _km_step_points(kmf, tau)

        color = colors[i % len(colors)]
        label = group_labels[i] if i < len(group_labels) else str(group_val)

        fig.add_trace(
            go.Scatter(
                x=t_pts, y=s_pts,
                fill='tozeroy',
                fillcolor=hex_to_rgba(color, 0.15),
                line=dict(width=0, shape="hv"),
                showlegend=False,
                hoverinfo='skip',
                name=f"{label} RMST",
            ),
            **add_kw,
        )

    # Vertical dashed line at tau
    fig.add_trace(
        go.Scatter(
            x=[tau, tau], y=[0, 1],
            mode='lines',
            line=dict(color='gray', width=1.5, dash='dash'),
            showlegend=False,
            hoverinfo='skip',
            name=f'\u03c4={tau:.1f}',
        ),
        **add_kw,
    )

    return result


def _plot_km_risk(df, time_col, event_col, risk_col, n_groups, time_limit,
                  show_risk_table, show_ci, title, breaks=None, break_type="quantile",
                  correction="bonferroni",
                  time_unit=None, xlim=None, ylim=None, subtitle=None,
                  n_permutations=1000, random_state=None,
                  show_pvalue=False, p_adjust="holm", _logrank_out=None,
                  show_rmst=False, rmst_tau=None, _rmst_out=None):
    """KM curves grouped by risk scores (quantile, score, or maxstat) with optional risk table."""
    df = df.copy()
    time_limit, x_lo, x_hi, y_lo, y_hi, time_label = _resolve_axes_params(
        df, time_col, time_limit, xlim, ylim, time_unit,
    )
    risk_table_times = [round(t) for t in np.linspace(x_lo, x_hi, 5)]

    df, n_groups, group_labels, maxstat_info, subtitle, risk_colors = (
        _resolve_risk_groups(
            df, time_col, event_col, risk_col, n_groups, breaks, break_type,
            correction, subtitle, n_permutations, random_state,
        )
    )

    logrank = _collect_logrank(
        df, time_col, event_col, "risk_group", p_adjust, _logrank_out,
    )

    fig, total_h, row = _make_km_risk_figure(show_risk_table, n_groups)
    _fit_km_groups(fig, df, time_col, event_col, time_limit, risk_colors,
                   group_labels, show_ci, row)

    rmst_result = _apply_rmst_if_requested(
        fig, df, time_col, event_col, risk_colors, group_labels,
        show_rmst, rmst_tau, row, time_limit, "risk_group", _rmst_out,
    )

    if show_risk_table:
        _add_risk_table(fig, df, time_col, risk_table_times, group_labels,
                        time_limit, x_lo=x_lo, x_hi=x_hi, time_label=time_label,
                        km_row=1, table_row=2, col=1)
        fig.update_yaxes(range=[y_lo, y_hi], row=1, col=1)

    _annotate_pvalue(fig, logrank, show_pvalue, show_risk_table)

    if rmst_result is not None and subtitle is None:
        subtitle = _rmst_subtitle(rmst_result)
    subtitle = _ci_band_subtitle(subtitle, show_ci)

    _apply_km_risk_layout(fig, title, subtitle, total_h, show_risk_table,
                           time_label, x_lo, x_hi, y_lo, y_hi)
    return fig


def _apply_rmst_if_requested(fig, df, time_col, event_col, colors,
                              group_labels, show_rmst, rmst_tau, row,
                              time_limit, group_col, _rmst_out):
    """Conditionally add RMST overlay and collect result."""
    if not show_rmst:
        return None
    rmst_result = _add_rmst_overlay(
        fig, df, time_col, event_col, colors, group_labels,
        rmst_tau, row, time_limit=time_limit, group_col=group_col,
    )
    if rmst_result is not None and _rmst_out is not None:
        _rmst_out.append(rmst_result)
    return rmst_result


def _annotate_pvalue(fig, logrank, show_pvalue, show_risk_table):
    """Add log-rank p-value annotation if requested."""
    if show_pvalue and logrank is not None:
        subplot_kw = dict(row=1, col=1) if show_risk_table else {}
        _add_logrank_annotation(fig, logrank, **subplot_kw)


def _plot_km_by_group(df, time_col, event_col, group_col, time_limit,
                      show_ci, title, colors, time_unit=None, xlim=None, ylim=None,
                      show_pvalue=False, p_adjust="holm", _logrank_out=None,
                      show_rmst=False, rmst_tau=None, _rmst_out=None,
                      show_risk_table=False, subtitle=None):
    """KM curves for pre-defined groups."""
    from lifelines import KaplanMeierFitter

    df = df.copy()
    time_limit, x_lo, x_hi, y_lo, y_hi, time_label = _resolve_axes_params(
        df, time_col, time_limit, xlim, ylim, time_unit,
    )

    groups = sorted(df[group_col].unique(), key=str)
    n_groups = len(groups)
    default_colors = get_risk_colors(n_groups)
    group_labels = [str(g) for g in groups]

    fig, total_h, row = _make_km_risk_figure(show_risk_table, n_groups)

    color_map = colors or {}
    for i, group in enumerate(groups):
        group_df = df[df[group_col] == group]
        color = color_map.get(group, default_colors[i % len(default_colors)])
        kmf = KaplanMeierFitter()
        kmf.fit(group_df[time_col].clip(upper=time_limit), group_df[event_col])
        _add_km_trace(fig, kmf, group_df, color, str(group), show_ci, row=row)

    rmst_result = _apply_rmst_if_requested(
        fig, df, time_col, event_col, default_colors, group_labels,
        show_rmst, rmst_tau, row, time_limit, group_col, _rmst_out,
    )
    if rmst_result is not None and subtitle is None:
        subtitle = _rmst_subtitle(rmst_result)
    subtitle = _ci_band_subtitle(subtitle, show_ci)

    logrank = _collect_logrank(
        df, time_col, event_col, group_col, p_adjust, _logrank_out,
    )
    _annotate_pvalue(fig, logrank, show_pvalue, show_risk_table)

    if show_risk_table:
        _add_by_group_risk_table(
            fig, df, time_col, group_col, groups, group_labels,
            time_limit, x_lo, x_hi, y_lo, y_hi, time_label,
        )

    _apply_km_risk_layout(fig, title, subtitle, total_h, show_risk_table,
                           time_label, x_lo, x_hi, y_lo, y_hi,
                           legend_title="Group")
    return fig


def _add_by_group_risk_table(fig, df, time_col, group_col, groups,
                              group_labels, time_limit, x_lo, x_hi,
                              y_lo, y_hi, time_label):
    """Add risk table for by-group KM plots."""
    group_to_idx = {g: i for i, g in enumerate(groups)}
    df["risk_group"] = df[group_col].map(group_to_idx)
    risk_table_times = [round(t) for t in np.linspace(x_lo, x_hi, 5)]
    _add_risk_table(fig, df, time_col, risk_table_times, group_labels,
                    time_limit, x_lo=x_lo, x_hi=x_hi, time_label=time_label,
                    km_row=1, table_row=2, col=1)
    fig.update_yaxes(range=[y_lo, y_hi], row=1, col=1)


def _make_km_panel_subplots(n_models, n_cols, n_rows, km_titles,
                             show_risk_table):
    """Create panel grid subplots for KM panel grid.

    Returns the figure.
    """
    if not show_risk_table:
        return make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=km_titles,
            shared_yaxes=True,
            vertical_spacing=0.18, horizontal_spacing=0.08,
        )

    # Each logical row -> 2 physical rows: KM (75%) + risk table (25%)
    expanded_titles = []
    for lr in range(n_rows):
        for lc in range(n_cols):
            idx = lr * n_cols + lc
            expanded_titles.append(km_titles[idx] if idx < n_models else "")
        expanded_titles.extend([""] * n_cols)

    row_heights = [0.75, 0.25] * n_rows
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    return make_subplots(
        rows=n_rows * 2, cols=n_cols,
        subplot_titles=expanded_titles,
        shared_yaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.08, horizontal_spacing=0.08,
    )


def _add_maxstat_panel_annotation(fig, maxstat_info, km_row, col):
    """Add per-panel annotation for maxstat cutpoint(s)."""
    p = maxstat_info.p_value_corrected
    if len(maxstat_info.cutpoints) == 1:
        cut_text = f"cut={maxstat_info.cutpoints[0]:.3g}"
    else:
        cut_text = "cuts=" + ",".join(
            f"{c:.3g}" for c in maxstat_info.cutpoints
        )
    fig.add_annotation(
        text=f"{cut_text}, p={p:.2g}",
        x=0.98, y=0.05,
        xref="x domain", yref="y domain",
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=MANUSCRIPT_DEFAULTS["font_size_annotation"]),
        row=km_row, col=col,
    )


def _render_km_panel(fig, model_df, idx, n_cols, n_groups, breaks, break_type,
                     correction, time_col, event_col, risk_col, time_limit,
                     show_ci, show_risk_table, show_pvalue, p_adjust,
                     risk_table_times, x_lo, x_hi, y_lo, y_hi, time_label,
                     n_permutations, random_state, model_name, _logrank_out):
    """Render a single model panel (KM + optional risk table + annotations)."""
    lr = idx // n_cols
    col = idx % n_cols + 1
    km_row = lr * 2 + 1 if show_risk_table else lr + 1
    table_row = lr * 2 + 2 if show_risk_table else None

    model_df = model_df.copy()

    m_n_groups, m_breaks, m_break_type, m_maxstat = _resolve_maxstat(
        model_df, time_col, event_col, risk_col,
        n_groups, breaks, break_type, correction,
        n_permutations=n_permutations, random_state=random_state,
    )
    model_df, actual_groups, labels = _create_risk_groups(
        model_df, risk_col, m_n_groups, breaks=m_breaks, break_type=m_break_type,
    )
    # Use generic labels for maxstat -- each model has different cutpoints
    if m_maxstat is not None:
        labels = list(_MAXSTAT_LABELS.get(actual_groups, labels))
    risk_colors = get_risk_colors(actual_groups)

    _fit_km_groups(
        fig, model_df, time_col, event_col, time_limit,
        risk_colors, labels, show_ci, row=km_row, col=col,
        show_legend=(idx == 0),
    )

    if m_maxstat is not None:
        _add_maxstat_panel_annotation(fig, m_maxstat, km_row, col)

    m_logrank = _collect_logrank(
        model_df, time_col, event_col, "risk_group", p_adjust,
        _logrank_out, label=model_name,
    )
    if show_pvalue and m_logrank is not None:
        y_pos = 0.12 if m_maxstat is not None else 0.05
        _add_logrank_annotation(fig, m_logrank, y=y_pos, row=km_row, col=col)

    if show_risk_table:
        _add_risk_table(
            fig, model_df, time_col, risk_table_times, labels,
            time_limit, x_lo=x_lo, x_hi=x_hi, time_label=time_label,
            km_row=km_row, table_row=table_row, col=col,
        )
        fig.update_yaxes(range=[y_lo, y_hi], row=km_row, col=col)


def _plot_km_panel_grid(test_data_by_model, n_groups, time_col, event_col,
                        risk_col, time_limit, show_ci, title,
                        breaks=None, break_type="quantile",
                        correction="bonferroni",
                        time_unit=None, xlim=None, ylim=None,
                        show_risk_table=False, subtitle=None,
                        n_permutations=1000, random_state=None,
                        show_pvalue=False, p_adjust="holm", _logrank_out=None):
    """Panel grid of KM curves, one subplot per model."""
    if not test_data_by_model:
        raise ValueError("test_data_by_model is empty \u2014 nothing to plot.")

    model_names = list(test_data_by_model.keys())
    n_models = len(model_names)
    n_cols = min(2, n_models)
    n_rows = math.ceil(n_models / n_cols)

    time_limit, x_lo, x_hi, y_lo, y_hi, time_label = (
        _resolve_panel_axes(test_data_by_model, time_col, time_limit,
                            xlim, ylim, time_unit)
    )

    km_titles = [get_model_display_name(m) for m in model_names]
    fig = _make_km_panel_subplots(n_models, n_cols, n_rows, km_titles,
                                   show_risk_table)

    risk_table_times = [round(t) for t in np.linspace(x_lo, x_hi, 5)]

    for idx, (model_name, model_df) in enumerate(test_data_by_model.items()):
        _render_km_panel(
            fig, model_df, idx, n_cols, n_groups, breaks, break_type,
            correction, time_col, event_col, risk_col, time_limit,
            show_ci, show_risk_table, show_pvalue, p_adjust,
            risk_table_times, x_lo, x_hi, y_lo, y_hi, time_label,
            n_permutations, random_state, model_name, _logrank_out,
        )

    if break_type == "maxstat" and subtitle is None:
        subtitle = "Maxstat optimal risk stratification"
    subtitle = _ci_band_subtitle(subtitle, show_ci)

    _apply_panel_grid_layout(
        fig, title, subtitle, n_rows, n_cols, n_groups,
        show_risk_table, x_lo, x_hi, y_lo, y_hi, time_label,
    )
    return fig


def _resolve_panel_axes(test_data_by_model, time_col, time_limit,
                        xlim, ylim, time_unit):
    """Resolve axis parameters for panel grid (max across all models)."""
    if time_limit is None:
        time_limit = max(
            mdf[time_col].max() for mdf in test_data_by_model.values()
        )
    x_lo, x_hi = xlim if xlim is not None else (0, time_limit)
    y_lo, y_hi = ylim if ylim is not None else (0, 1.05)
    time_label = f"Time ({time_unit})" if time_unit else "Time"
    return time_limit, x_lo, x_hi, y_lo, y_hi, time_label


def _apply_panel_grid_layout(fig, title, subtitle, n_rows, n_cols,
                              n_groups, show_risk_table, x_lo, x_hi,
                              y_lo, y_hi, time_label):
    """Apply final layout to a panel grid figure."""
    d = MANUSCRIPT_DEFAULTS
    margin_t = d["margin_t"] + 50 if subtitle else d["margin_t"] + 20

    km_h = 290
    table_h = max(60, n_groups * 20) if show_risk_table else 0
    height_per_row = km_h + table_h
    total_height = height_per_row * n_rows + 80

    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(text=_build_km_title_text(title, subtitle)),
        width=int(d["width_double"] * 0.9),
        height=total_height,
        hovermode="x unified",
        margin=dict(l=d["margin_l"], r=d["margin_r"],
                    t=margin_t, b=d["margin_b"]),
    )

    if not show_risk_table:
        for i in range(1, n_rows * n_cols + 1):
            r, c = (i - 1) // n_cols + 1, (i - 1) % n_cols + 1
            fig.update_yaxes(range=[y_lo, y_hi], row=r, col=c)
            fig.update_xaxes(range=[x_lo, x_hi], title_text=time_label, row=r, col=c)


# ---- Internal helpers (used by reports) ------------------------------------


def _km_combined(
    fig: go.Figure,
    df: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    group_col: Optional[str] = None,
    time_limit: Optional[float] = None,
    show_ci: bool = True,
    row: Optional[int] = None,
    col: int = 1,
    colors: Optional[Dict[str, str]] = None,
    group_labels: Optional[Dict] = None,
    show_legend: bool = True,
) -> tuple:
    """Fit KM curves and add traces to *fig*, returning tidy data.

    Used by report stratification sections. Adds traces to *fig* and
    returns structured data for downstream tables.

    Parameters
    ----------
    group_labels : dict, optional
        Mapping from raw group keys (e.g. integer indices from ``pd.qcut``)
        to display labels (e.g. ``{0: "Q1", 1: "Q2"}``).  When provided,
        display labels are used for trace names, color lookup, and returned
        data.
    show_legend : bool
        Whether traces added by this call appear in the legend.

    Returns
    -------
    tuple of (list[dict], list[dict])
        ``(km_rows, median_rows)`` -- tidy records for KM survival curves
        and median survival, suitable for ``pd.DataFrame()``.
    """
    from lifelines import KaplanMeierFitter

    if time_limit is None:
        time_limit = df[time_col].max()

    km_rows: list = []
    median_rows: list = []
    color_map = colors or {}
    label_map = group_labels or {}

    if group_col is None:
        groups = {"Overall": df}
        default_colors = ["#1f77b4"]
    else:
        groups = {
            name: group_df
            for name, group_df in df.groupby(group_col, observed=True, sort=True)
        }
        default_colors = get_risk_colors(len(groups))

    for i, (raw_label, group_df) in enumerate(groups.items()):
        display_label = label_map.get(raw_label, str(raw_label))
        kmf = KaplanMeierFitter()
        kmf.fit(group_df[time_col].clip(upper=time_limit), group_df[event_col])

        color = color_map.get(
            display_label,
            color_map.get(raw_label, default_colors[i % len(default_colors)]),
        )

        _add_km_trace(
            fig, kmf, group_df, color, display_label, show_ci,
            row=row, col=col, show_legend=show_legend,
        )

        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        for t_idx, t in enumerate(sf.index):
            entry = {
                "group": display_label,
                "n_patients": len(group_df),
                "time": t,
                "survival_prob": sf.iloc[t_idx, 0],
            }
            if show_ci:
                entry["ci_lower"] = ci.iloc[t_idx, 0]
                entry["ci_upper"] = ci.iloc[t_idx, 1]
            km_rows.append(entry)

        median = kmf.median_survival_time_
        median_not_reached = np.isnan(median) or np.isinf(median)
        median_rows.append({
            "group": display_label,
            "n_patients": len(group_df),
            "n_events": int(group_df[event_col].sum()),
            "median_survival": median if not median_not_reached else None,
        })

    return km_rows, median_rows


def _survival_grid(
    data_dict: Dict,
    models: List[str],
    diseases: List[str],
    time_col: str = "time",
    event_col: str = "event",
    risk_col: str = "risk_score",
    n_groups: int = 4,
) -> go.Figure:
    """Grid of KM curves: rows=models, columns=diseases."""
    from lifelines import KaplanMeierFitter

    risk_colors = get_risk_colors(n_groups)
    pcts = [int(round(i * 100 / n_groups)) for i in range(1, n_groups)]
    group_labels = _pct_range_labels(pcts)

    fig = make_subplots(
        rows=len(models), cols=len(diseases),
        subplot_titles=[f"{d}" for d in diseases],
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.06, row_titles=models,
    )

    for i, model in enumerate(models, 1):
        for j, disease in enumerate(diseases, 1):
            df = data_dict.get((model, disease))
            if df is None:
                continue

            df = df.copy()
            try:
                df["risk_group"] = pd.qcut(
                    df[risk_col], q=n_groups, labels=False, duplicates="drop",
                )
            except ValueError:
                warnings.warn(
                    f"Skipping panel ({model}, {disease}): could not create "
                    f"risk groups from available scores.",
                    InternalWarning, stacklevel=2,
                )
                continue

            for q, (group_idx, gdf) in enumerate(
                df.groupby("risk_group", observed=True)
            ):
                kmf = KaplanMeierFitter()
                kmf.fit(gdf[time_col], gdf[event_col])

                fig.add_trace(
                    go.Scatter(
                        x=kmf.survival_function_.index,
                        y=kmf.survival_function_.iloc[:, 0],
                        mode="lines",
                        line=dict(color=risk_colors[q], shape="hv"),
                        name=group_labels[q],
                        legendgroup=group_labels[q],
                        showlegend=(i == 1 and j == 1),
                    ),
                    row=i, col=j,
                )

    fig.update_layout(
        template="mlsurv_minimal",
        height=200 * len(models),
        width=250 * len(diseases),
        title=dict(text="<b>Survival by Model and Disease</b>"),
    )
    return fig
