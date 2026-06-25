"""Waterfall plot for individual patient explanations."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv.viz._theme import create_figure, apply_size_overrides
from mlsurv.viz._palettes import FeatureGroupsType, resolve_feature_group_colors


def plot_waterfall(
    contributions: pd.DataFrame,
    risk_score: float,
    base_value: float | None = None,
    n: int = 15,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Dict[str, str] | None = None,
) -> go.Figure:
    """Horizontal waterfall plot showing per-feature contributions to risk.

    Parameters
    ----------
    contributions : pd.DataFrame
        Must have columns ``feature``, ``feature_value``, ``shap_value``,
        ``abs_shap_value``.  Typically sorted by ``abs_shap_value`` descending.
    risk_score : float
        Total risk score for this patient, i.e. ``model.predict(X_patient)``.
        Used as the final-bar anchor so the chart ends at the true prediction.
    base_value : float, optional
        ``E[f(X)]`` — model's expected risk over the background.  When given,
        the chart starts with an "absolute" baseline anchor at this value and
        closes with a "total" bar at ``risk_score``; the stack in between
        shows how each feature moved risk from baseline to prediction.  When
        ``None``, the legacy all-relative layout is used (bars start at 0
        and end at ``sum(shap_values)``).
    n : int, default=15
        Maximum number of features to show.  Remaining features are
        aggregated into an "Other features" bar.
    title : str, optional
        Figure title.
    width : int, optional
        Override default figure width (pixels).
    height : int, optional
        Override default figure height (pixels).
    feature_groups : dict, optional
        Mapping of feature name → group name (``FeatureGroupsType``).
        When provided, group names are appended to bar labels.
    group_colors : dict, optional
        Mapping of group name → hex colour (unused for waterfall bar
        colours — kept for API consistency with other viz functions).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = contributions.copy()

    # Hide exactly-zero contributions (e.g. raw features the selector dropped,
    # which never reach the model so their SHAP value is exactly 0).  Keep them
    # in the source DataFrame for reconciliation with the cohort; just don't
    # plot them here or pad the "Other" bucket with them.  Fall back to the
    # full set if every contribution is zero, so the chart still renders.
    if "abs_shap_value" in df.columns:
        nonzero = df[df["abs_shap_value"] != 0]
        if not nonzero.empty:
            df = nonzero

    # Top N features + aggregate remainder
    top = df.head(n)
    remaining = df.iloc[n:]

    # Feature → group mapping (for label annotation)
    feat_to_grp = {}
    if feature_groups is not None:
        feat_to_grp, _ = resolve_feature_group_colors(
            feature_groups, df["feature"].tolist(), group_colors,
        )

    # Collect (label, shap_value) pairs ordered bottom-to-top (largest
    # |shap| at top) matching the canonical SHAP waterfall layout.
    bars = []
    for _, row in top.iloc[::-1].iterrows():
        feat = row["feature"]
        sv = float(row["shap_value"])
        fv = row["feature_value"]

        if np.isscalar(fv) and pd.isna(fv):
            label = f"{feat} = NaN (imputed)"
        elif isinstance(fv, float):
            label = f"{feat} = {fv:.3g}"
        else:
            label = f"{feat} = {fv}"

        if feat in feat_to_grp:
            label = f"{label}  [{feat_to_grp[feat]}]"

        bars.append((label, sv))

    if len(remaining) > 0:
        bars.append((f"Other ({len(remaining)} features)",
                     float(remaining["shap_value"].sum())))

    fig = create_figure(
        title=title or "Patient Risk Explanation",
        width="double",
        height="large",
    )

    if not bars:
        return fig

    # Build filled-polygon arrows — matches the canonical shap.plots.waterfall
    # rendering where each bar is drawn as an arrow (rectangle body + triangle
    # tip) in the direction of its contribution. Anchor the cascade at
    # base_value when known; otherwise start at 0 and drift from there.
    running = float(base_value) if base_value is not None else 0.0
    starts = []
    ends = []
    svs = [sv for _, sv in bars]
    for sv in svs:
        starts.append(running)
        running = running + sv
        ends.append(running)

    # Arrow tip width scaled to the total span of the bars (~2% of range)
    # so arrowheads are visible but barely pointy — closer to SHAP's style.
    min_x = min(starts + ends + ([float(base_value)] if base_value is not None else []))
    max_x = max(starts + ends + ([float(risk_score)] if base_value is not None else []))
    span = max(max_x - min_x, 1e-9)
    tip_w = 0.02 * span
    bar_h = 0.7  # bar half-height in categorical y units (row spacing = 1)

    pos_color = "rgba(239, 85, 59, 0.9)"
    neg_color = "rgba(99, 110, 250, 0.9)"

    bar_labels = [label for label, _ in bars]
    # Collect hover-marker data so a single overlay trace handles all bars.
    hover_xs, hover_ys, hover_texts = [], [], []
    for i, (sv, x_start, x_end, label) in enumerate(
        zip(svs, starts, ends, bar_labels)
    ):
        if sv == 0:
            continue
        y_bot = i - bar_h / 2
        y_top = i + bar_h / 2
        y_mid = i
        # Clamp the tip length so it never exceeds the bar's own length
        tip = min(tip_w, abs(sv) * 0.9)
        if sv > 0:
            body_end = x_end - tip
            xs = [x_start, body_end, x_end, body_end, x_start, x_start]
            ys = [y_bot, y_bot, y_mid, y_top, y_top, y_bot]
            color = pos_color
            direction = "↑ increases risk"
        else:
            body_end = x_end + tip
            xs = [x_start, body_end, x_end, body_end, x_start, x_start]
            ys = [y_bot, y_bot, y_mid, y_top, y_top, y_bot]
            color = neg_color
            direction = "↓ decreases risk"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor=color, line=dict(color=color, width=0),
            hoverinfo="skip", showlegend=False,
        ))
        # SHAP-value label: centered white inside wide bars; otherwise
        # anchored just outside the tip in dark text so short bars stay
        # legible without overflowing.
        text_fits_inside = abs(sv) >= 0.10 * span
        label_text = f"{sv:+.3f}"
        if text_fits_inside:
            fig.add_annotation(
                x=(x_start + x_end) / 2, y=y_mid,
                text=label_text,
                showarrow=False,
                font=dict(size=11, color="#ffffff"),
            )
        else:
            # Offset the label a small distance past the tip, toward the
            # side the bar points to.
            offset = tip_w * 0.6
            if sv > 0:
                label_x = x_end + offset
                xanchor = "left"
            else:
                label_x = x_end - offset
                xanchor = "right"
            fig.add_annotation(
                x=label_x, y=y_mid,
                text=label_text,
                showarrow=False,
                xanchor=xanchor,
                font=dict(size=11, color="#212529"),
            )
        # Wide invisible marker overlay for hover — each bar gets a marker
        # at its midpoint wide enough to cover the bar area.
        hover_xs.append((x_start + x_end) / 2)
        hover_ys.append(y_mid)
        hover_texts.append(
            f"<b>{label}</b><br>"
            f"SHAP contribution: {sv:+.4f}<br>"
            f"Running risk: {x_start:.3f} → {x_end:.3f}<br>"
            f"{direction}"
        )

    # Overlay invisible hover markers on every bar so hovertemplate fires
    # correctly (go.Scatter with fill='toself' has quirky hoveron="fills"
    # rendering in some plotly versions — markers are more reliable).
    if hover_xs:
        fig.add_trace(go.Scatter(
            x=hover_xs, y=hover_ys, mode="markers",
            marker=dict(size=40, color="rgba(0,0,0,0)"),
            text=hover_texts, hoverinfo="text",
            name="", showlegend=False,
        ))

    # Categorical-looking y-axis with feature labels as ticks
    y_positions = list(range(len(bars)))
    y_ticktext = [label for label, _ in bars]
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_ticktext,
        range=[-0.8, len(bars) - 0.2],
        showgrid=False,
        zeroline=False,
    )

    # Dashed reference lines at baseline and predicted risk, each labeled
    # with an arrow annotation pointing at the corresponding vline so the
    # reader sees which value belongs to which line.
    if base_value is not None:
        fig.add_vline(x=float(base_value), line_color="#868e96",
                      line_dash="dash", line_width=1)
        fig.add_vline(x=float(risk_score), line_color="#000000",
                      line_dash="dash", line_width=1)
        # Baseline annotation below the plot, arrow pointing up to the vline
        fig.add_annotation(
            x=float(base_value), y=0, yref="paper",
            text=f"Baseline risk E[f(X)] = {base_value:.3f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            arrowcolor="#868e96",
            ax=0, ay=40,
            font=dict(size=12, color="#495057"),
        )
        # Predicted risk annotation above the plot, arrow pointing down to the vline
        fig.add_annotation(
            x=float(risk_score), y=1, yref="paper",
            text=f"Final risk f(x) = {risk_score:.3f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            arrowcolor="#000000",
            ax=0, ay=-30,
            font=dict(size=12, color="#000000"),
        )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="← decreases risk     SHAP contribution     increases risk →",
                standoff=70,
            ),
        ),
        yaxis_title=None,
        showlegend=False,
        # Extra bottom margin so x-axis title sits below the baseline
        # annotation; extra top margin for the f(x) arrow annotation.
        margin=dict(l=220, t=80, b=140),
    )

    apply_size_overrides(fig, width, height)
    return fig
