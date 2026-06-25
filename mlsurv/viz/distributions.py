"""Feature distribution plots with interactive dropdown selector."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv.preprocessing.utils import detect_categorical_groups
from mlsurv.viz._palettes import MANUSCRIPT_DEFAULTS
from mlsurv.viz._theme import apply_size_overrides, truncate_labels


# ── Palette ───────────────────────────────────────────────────────────────────

_TRAIN_COLOR = "#4e79a7"   # Steel blue
_TEST_COLOR = "#e15759"    # Soft red
_BAR_COLORS = [
    "#4e79a7", "#e15759", "#76b7b2", "#f28e2b",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def _is_numeric_feature(series: pd.Series) -> bool:
    """True when a column should be treated as continuous."""
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10


# ── Trace builders ────────────────────────────────────────────────────────────


def _add_numeric_traces(
    fig: go.Figure,
    col: str,
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    visible: bool,
) -> List[int]:
    """Add violin + boxplot + jittered scatter traces for a numeric feature."""
    indices: List[int] = []
    rng = np.random.default_rng(42)

    # Use numeric x-positions so scatter points can be jittered
    positions = {"Train": 0, "Test": 1}

    for label, vals, color in [
        ("Train", train_vals, _TRAIN_COLOR),
        ("Test", test_vals, _TEST_COLOR),
    ]:
        clean = vals[np.isfinite(vals)] if len(vals) > 0 else vals
        pos = positions[label]

        # Violin
        fig.add_trace(go.Violin(
            x0=pos,
            y=clean.tolist(),
            name=label,
            side="both",
            line_color=color,
            fillcolor=color,
            opacity=0.25,
            meanline_visible=False,
            showlegend=False,
            visible=visible,
            scalemode="width",
            width=0.7,
            points=False,
        ))
        indices.append(len(fig.data) - 1)

        # Box (overlaid)
        fig.add_trace(go.Box(
            x0=pos,
            y=clean.tolist(),
            name=label,
            marker_color=color,
            line_color=color,
            fillcolor="rgba(255,255,255,0)",
            showlegend=False,
            visible=visible,
            width=0.15,
            boxmean=False,
        ))
        indices.append(len(fig.data) - 1)

        # Jittered scatter
        jitter = rng.uniform(-0.12, 0.12, size=len(clean))
        fig.add_trace(go.Scatter(
            x=(pos + jitter).tolist(),
            y=clean.tolist(),
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.35),
            showlegend=False,
            visible=visible,
            hovertemplate=f"<b>{col}</b> ({label})<br>%{{y:.3g}}<extra></extra>",
        ))
        indices.append(len(fig.data) - 1)

    return indices, True  # True = uses numeric x-axis


def _add_categorical_traces(
    fig: go.Figure,
    col: str,
    train_series: pd.Series,
    test_series: pd.Series,
    visible: bool,
) -> List[int]:
    """Add grouped bar chart traces for a categorical/low-cardinality feature."""
    indices: List[int] = []

    # Show bars for all unique values (not just 0/1)
    all_vals = sorted(
        set(train_series.dropna().unique().tolist())
        | set(test_series.dropna().unique().tolist())
    )

    for i, val in enumerate(all_vals):
        color = _BAR_COLORS[i % len(_BAR_COLORS)]
        train_count = int((train_series == val).sum())
        test_count = int((test_series == val).sum())
        fig.add_trace(go.Bar(
            x=["Train", "Test"],
            y=[train_count, test_count],
            name=str(val),
            marker_color=color,
            opacity=0.8,
            showlegend=False,
            visible=visible,
            hovertemplate=(
                f"<b>{col} = {val}</b><br>"
                "%{x}: %{y}<extra></extra>"
            ),
        ))
        indices.append(len(fig.data) - 1)

    return indices


def _add_group_traces(
    fig: go.Figure,
    group_name: str,
    member_cols: List[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    visible: bool,
) -> List[int]:
    """Add grouped bar chart traces for a reconstructed categorical group."""
    indices: List[int] = []
    sep = "_"

    # Filter to columns present in both train and test
    valid_cols = [c for c in member_cols if c in X_test.columns]
    if not valid_cols:
        return indices

    for i, col in enumerate(valid_cols):
        # Extract category label from column name
        if col.startswith(group_name + sep):
            cat_label = col[len(group_name) + len(sep):]
        else:
            cat_label = col

        color = _BAR_COLORS[i % len(_BAR_COLORS)]
        train_count = int((X_train[col] == 1).sum())
        test_count = int((X_test[col] == 1).sum())

        fig.add_trace(go.Bar(
            x=["Train", "Test"],
            y=[train_count, test_count],
            name=cat_label,
            marker_color=color,
            opacity=0.8,
            showlegend=False,
            visible=visible,
            hovertemplate=(
                f"<b>{group_name} = {cat_label}</b><br>"
                "%{x}: %{y}<extra></extra>"
            ),
        ))
        indices.append(len(fig.data) - 1)

    return indices


# ── Main function ─────────────────────────────────────────────────────────────


def plot_feature_distributions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_groups: Optional[Dict[str, List[str]]] = None,
    title: str = "Feature Distributions",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """Single-panel distribution plot with a dropdown to switch features.

    Shows the first feature by default.  The dropdown lets the user explore
    any individual feature or detected categorical group.

    Parameters
    ----------
    X_train : DataFrame
        Training feature matrix (post-split, pre- or post-imputation).
    X_test : DataFrame
        Test feature matrix (post-split, pre- or post-imputation).
    categorical_groups : dict, optional
        Mapping from group name to list of one-hot column names.
        Auto-detected via :func:`detect_categorical_groups` when *None*.
    title : str
        Figure title.
    width : int, optional
        Override default figure width (pixels).
    height : int, optional
        Override default figure height (pixels).

    Returns
    -------
    go.Figure
    """
    if X_train.empty or X_test.empty:
        fig = go.Figure()
        fig.update_layout(title=dict(text=f"<b>{title}</b> -- No data"))
        apply_size_overrides(fig, width, height)
        return fig

    if categorical_groups is None:
        categorical_groups = detect_categorical_groups(X_train)

    # Columns that belong to a detected group
    grouped_cols = set()
    for cols in categorical_groups.values():
        grouped_cols.update(cols)

    # Build ordered list of dropdown entries:
    #   (label, entry_type, payload)
    #   entry_type: "numeric", "binary", "group"
    entries: List[tuple] = []

    for col in X_train.columns:
        if col in grouped_cols:
            continue
        if _is_numeric_feature(X_train[col]):
            entries.append((col, "numeric", col))
        else:
            entries.append((col, "categorical", col))

    # Append categorical group entries
    for group_name, member_cols in sorted(categorical_groups.items()):
        entries.append((f"{group_name} (group)", "group", (group_name, member_cols)))

    if not entries:
        fig = go.Figure()
        fig.update_layout(title=dict(text=f"<b>{title}</b> -- No features"))
        apply_size_overrides(fig, width, height)
        return fig

    fig = go.Figure()
    feat_indices: Dict[str, List[int]] = {}
    entry_types: Dict[str, str] = {}

    for idx, (label, etype, payload) in enumerate(entries):
        is_first = idx == 0
        entry_types[label] = etype

        if etype == "numeric":
            col = payload
            indices, _numeric_axis = _add_numeric_traces(
                fig, col,
                X_train[col].values, X_test[col].values,
                visible=is_first,
            )
        elif etype == "categorical":
            col = payload
            indices = _add_categorical_traces(
                fig, col,
                X_train[col], X_test[col],
                visible=is_first,
            )
        else:  # group
            group_name, member_cols = payload
            indices = _add_group_traces(
                fig, group_name, member_cols,
                X_train, X_test,
                visible=is_first,
            )

        feat_indices[label] = indices

    n_traces = len(fig.data)

    # --- Dropdown buttons ---
    # Numeric features use positions 0/1 with tick labels;
    # categorical/group features use string x-values.
    _TICK_VALS = [0, 1]
    _TICK_TEXT = ["Train", "Test"]

    buttons = []
    feat_labels = list(feat_indices)
    display_labels, _ = truncate_labels(feat_labels, max_chars=30)
    for label, display in zip(feat_labels, display_labels):
        vis = [False] * n_traces
        for i in feat_indices[label]:
            vis[i] = True

        etype = entry_types[label]
        yaxis_title = label if etype == "numeric" else "Count"
        barmode = "group" if etype != "numeric" else "overlay"

        layout_update = {
            "title.text": f"<b>{title}</b> -- {label}",
            "yaxis.title.text": yaxis_title,
            "barmode": barmode,
        }
        if etype == "numeric":
            layout_update["xaxis.tickvals"] = _TICK_VALS
            layout_update["xaxis.ticktext"] = _TICK_TEXT
            layout_update["xaxis.type"] = "linear"
        else:
            layout_update["xaxis.tickvals"] = None
            layout_update["xaxis.ticktext"] = None
            layout_update["xaxis.type"] = "category"

        buttons.append(dict(
            label=display,
            method="update",
            args=[{"visible": vis}, layout_update],
        ))

    d = MANUSCRIPT_DEFAULTS
    first_label = entries[0][0]
    first_etype = entries[0][1]

    fig.update_layout(
        template="mlsurv_minimal",
        width=600,
        height=d["height_medium"],
        title=dict(text=f"<b>{title}</b> -- {first_label}"),
        xaxis_title="Split",
        yaxis_title=first_label if first_etype == "numeric" else "Count",
        showlegend=False,
        barmode="group" if first_etype != "numeric" else "overlay",
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.0, xanchor="right",
            y=1.05, yanchor="bottom",
            buttons=buttons,
            active=0,
            showactive=True,
            bgcolor="#f5f7fa",
            bordercolor="#ddd",
            font=dict(size=11),
        )],
    )

    # Set initial x-axis tick config for first feature
    if first_etype == "numeric":
        fig.update_layout(xaxis=dict(
            tickvals=_TICK_VALS, ticktext=_TICK_TEXT, type="linear",
        ))

    fig.update_layout(margin=dict(
        l=d["margin_l"] + 10,
        t=d["margin_t"] + 70,
        b=d["margin_b"],
    ))

    apply_size_overrides(fig, width, height)
    return fig
