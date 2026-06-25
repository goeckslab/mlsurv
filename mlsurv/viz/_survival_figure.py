"""Figure building primitives for KM survival plots.

Public symbols:
    _make_km_risk_figure, _build_km_title_text, _apply_km_risk_layout,
    _add_km_trace, _add_risk_table, _fit_km_groups
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlsurv.viz._palettes import MANUSCRIPT_DEFAULTS
from mlsurv.viz._theme import add_ci_ribbon


def _make_km_risk_figure(show_risk_table, n_groups):
    """Create figure (with or without risk table subplot) for KM risk plots.

    Returns (fig, total_h, row) where row is 1 (subplots) or None (plain).
    """
    km_h = 290
    if show_risk_table:
        table_h = max(60, n_groups * 20)
        total_h = km_h + table_h
        km_frac = km_h / total_h
        fig = make_subplots(
            rows=2, cols=1, row_heights=[km_frac, 1 - km_frac],
            vertical_spacing=0.03, shared_xaxes=True,
        )
        return fig, total_h, 1
    return go.Figure(), km_h, None


def _build_km_title_text(title, subtitle):
    """Build title HTML with optional subtitle."""
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><sup>{subtitle}</sup>"
    return title_text


def _apply_km_risk_layout(fig, title, subtitle, total_h, show_risk_table,
                           time_label, x_lo, x_hi, y_lo, y_hi,
                           legend_title="Risk Group"):
    """Apply final layout to a single-model KM risk figure."""
    d = MANUSCRIPT_DEFAULTS
    margin_t = d["margin_t"] + 30 if subtitle else d["margin_t"]
    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(text=_build_km_title_text(title, subtitle)),
        xaxis_title=time_label if not show_risk_table else None,
        yaxis_title="Survival Probability" if not show_risk_table else None,
        legend=dict(title=legend_title),
        hovermode="x unified",
        width=d["width_single"],
        height=total_h,
        margin=dict(l=d["margin_l"] + 50, r=d["margin_r"],
                    t=margin_t, b=d["margin_b"]),
    )
    if not show_risk_table:
        fig.update_xaxes(range=[x_lo, x_hi])
        fig.update_yaxes(range=[y_lo, y_hi])


def _add_km_trace(fig, kmf, group_df, color, label, show_ci, row=None,
                  col=1, show_legend=True):
    """Add a single KM group's CI ribbon and step-function curve to *fig*.

    Parameters
    ----------
    row, col : int or None
        Subplot position. When *row* is ``None``, traces are added to a
        plain ``go.Figure`` (no subplots).
    show_legend : bool
        Whether to show the KM line in the legend.
    """
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_
    times = sf.index.values
    survival = sf.iloc[:, 0].values
    ci_lower = ci.iloc[:, 0].values
    ci_upper = ci.iloc[:, 1].values

    add_kw = dict(row=row, col=col) if row else {}
    legend_group = label if row else None

    if show_ci:
        add_ci_ribbon(
            fig, times, ci_upper, ci_lower, color,
            row=row if row else None,
            col=col if row else None,
            name=f"{label} CI",
            legendgroup=legend_group,
        )

    # Build hover template -- include CI details only for single-plot mode
    if row is None:
        hover = (
            f"<b>{label}</b><br>"
            "Time: %{x:.1f}<br>"
            "Survival: %{y:.1%}<br>"
            "95% CI: [%{customdata[0]:.1%}, %{customdata[1]:.1%}]<extra></extra>"
        )
        custom = np.column_stack([ci_lower, ci_upper])
    else:
        hover = (
            f"<b>{label}</b><br>"
            "Time: %{x:.1f}<br>"
            "Survival: %{y:.1%}<extra></extra>"
        )
        custom = None

    km_kw = {}
    if custom is not None:
        km_kw["customdata"] = custom

    km_trace = go.Scatter(
        x=times, y=survival,
        mode="lines",
        name=f"{label} (n={len(group_df)})",
        line=dict(color=color, width=MANUSCRIPT_DEFAULTS["line_width_main"], shape="hv"),
        legendgroup=legend_group,
        showlegend=show_legend,
        hovertemplate=hover,
        **km_kw,
    )
    fig.add_trace(km_trace, **add_kw)


def _add_risk_table(fig, df, time_col, risk_table_times, group_labels,
                    time_limit, x_lo=0, x_hi=None, time_label="Time",
                    km_row=1, table_row=2, col=1):
    """Add number-at-risk text traces and configure the risk-table subplot axes.

    Parameters
    ----------
    km_row, table_row, col : int
        Subplot positions for the KM curve and risk table rows.
        Defaults (1, 2, 1) work for the single-model layout; pass explicit
        values when reusing inside a panel grid.
    """
    d = MANUSCRIPT_DEFAULTS
    if x_hi is None:
        x_hi = time_limit
    x_span = x_hi - x_lo
    x_pad = x_span * 0.04  # small buffer so edge numbers aren't clipped

    # Use integer y-positions so we can set an explicit range with top padding
    # (categorical y-axes don't support range padding, causing top-row clipping)
    y_labels = []
    for i, group_idx in enumerate(sorted(df["risk_group"].unique())):
        group_df = df[df["risk_group"] == group_idx]
        label = group_labels[int(group_idx)]
        y_labels.append(label)
        n_at_risk = [(group_df[time_col] >= t).sum() for t in risk_table_times]

        fig.add_trace(
            go.Scatter(
                x=risk_table_times,
                y=[i] * len(risk_table_times),
                mode="text",
                text=[str(n) for n in n_at_risk],
                textposition="middle center",
                textfont=dict(size=d["font_size_annotation"]),
                showlegend=False, hoverinfo="skip",
                cliponaxis=False,
            ),
            row=table_row, col=col,
        )

    n_actual = len(y_labels)
    fig.update_yaxes(
        row=table_row, col=col,
        title=dict(text="At Risk", standoff=8),
        showgrid=False, showline=False, zeroline=False,
        tickmode="array",
        tickvals=list(range(n_actual)),
        ticktext=y_labels,
        # Extra top padding (1.1 units) prevents text clipping on topmost row
        range=[-0.7, n_actual - 1 + 1.1],
    )
    fig.update_xaxes(
        row=table_row, col=col,
        title_text=time_label,
        showgrid=False, showline=False, zeroline=False,
        range=[x_lo - x_pad, x_hi + x_pad],
    )
    fig.update_yaxes(title_text="Survival Probability", row=km_row, col=col)
    fig.update_xaxes(range=[x_lo - x_pad, x_hi + x_pad], row=km_row, col=col)


def _fit_km_groups(fig, df, time_col, event_col, time_limit, colors, group_labels,
                   show_ci, row, col=1, show_legend=True):
    """Fit KM per risk group and add traces to *fig*."""
    from lifelines import KaplanMeierFitter

    for group_idx, group_df in df.groupby("risk_group", observed=True, sort=True):
        kmf = KaplanMeierFitter()
        kmf.fit(group_df[time_col].clip(upper=time_limit), group_df[event_col])
        color = colors[int(group_idx) % len(colors)]
        label = group_labels[int(group_idx)]
        _add_km_trace(fig, kmf, group_df, color, label, show_ci,
                      row=row, col=col, show_legend=show_legend)
