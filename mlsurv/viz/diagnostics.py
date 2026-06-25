"""Diagnostic residual plots for survival model assessment.

- ``plot_schoenfeld``: scaled Schoenfeld residuals vs. time (PH assumption check)
- ``plot_schoenfeld_interactive``: single-panel Schoenfeld with feature dropdown
- ``plot_deviance``: deviance residuals vs. linear predictor (outlier detection)
- ``plot_martingale_interactive``: null-model martingale residuals vs. continuous
  covariate with feature dropdown (log-hazard linearity check)
The matching ``compute_schoenfeld_data`` / ``compute_deviance_data`` /
``compute_martingale_data`` functions now live in
:mod:`mlsurv.evaluation._diagnostics` (the "evaluation computes, viz plots"
layering convention) and are re-exported here for backward compatibility. The
plot functions accept either raw data (triggering the internal re-fit) or
pre-computed DataFrames.
"""

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlsurv.evaluation._diagnostics import (
    compute_deviance_data,
    compute_martingale_data,
    compute_schoenfeld_data,
)
from mlsurv.logging_utils import get_logger
from mlsurv.viz._palettes import MANUSCRIPT_DEFAULTS
from mlsurv.viz._theme import (
    adaptive_size,
    apply_size_overrides,
    create_figure,
    truncate_labels,
)

logger = get_logger("viz.diagnostics")

# Semantic colors (Okabe-Ito colorblind-friendly)
_SCATTER_COLOR = "#0072B2"   # Blue — residual points
_LOWESS_COLOR = "#D55E00"    # Vermillion — trend line
_EVENT_COLOR = "#D55E00"     # Vermillion — observed events
_CENSORED_COLOR = "#0072B2"  # Blue — censored observations

# Required columns for pre-computed DataFrames
_SCHOENFELD_COLS = {"feature", "time", "residual"}
_DEVIANCE_COLS = {"linear_predictor", "deviance", "event"}
_MARTINGALE_COLS = {"feature", "x_value", "residual"}


def _smooth_trend(x: np.ndarray, y: np.ndarray, frac: float = 0.3):
    """Compute a smooth trend line using LOWESS (statsmodels) or rolling mean.

    Falls back to a uniform moving average via scipy when statsmodels is not
    installed, so no additional dependency is required.

    Returns (x_smooth, y_smooth) arrays sorted by x, or ``None`` on failure.
    """
    import warnings

    from mlsurv._warnings import InternalWarning

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(y_sorted, x_sorted, frac=frac)
        return smooth[:, 0], smooth[:, 1]
    except ImportError:
        pass  # statsmodels not installed; fall through to scipy
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        warnings.warn(
            f"LOWESS smoothing failed ({type(e).__name__}: {e}); "
            "trying scipy fallback.",
            InternalWarning,
            stacklevel=2,
        )

    # Fallback: rolling mean with scipy uniform_filter1d
    try:
        from scipy.ndimage import uniform_filter1d
        window = max(3, int(len(x_sorted) * frac))
        if window % 2 == 0:
            window += 1
        y_smooth = uniform_filter1d(y_sorted.astype(float), size=window, mode='nearest')
        return x_sorted, y_smooth
    except ImportError:
        return None
    except (ValueError, FloatingPointError) as e:
        warnings.warn(
            f"Smoothing fallback also failed ({type(e).__name__}: {e}); "
            "trend line will be omitted.",
            InternalWarning,
            stacklevel=2,
        )
        return None


def _empty_figure(
    title: str,
    width: Optional[int],
    height: Optional[int],
) -> go.Figure:
    """Return a blank figure with a "No data" title annotation."""
    fig = go.Figure()
    fig.update_layout(title=dict(text=f"<b>{title}</b> — No data"))
    apply_size_overrides(fig, width, height)
    return fig


# ── Plot functions ─────────────────────────────────────────────────────────
#
# ``compute_schoenfeld_data`` / ``compute_deviance_data`` /
# ``compute_martingale_data`` now live in
# :mod:`mlsurv.evaluation._diagnostics` and are imported at the top of this
# module (re-exported for backward compatibility).


def _resolve_schoenfeld_data(X_train, y_train, features, n_features):
    """Dispatch between pre-computed DataFrame and raw data requiring CoxPH fit.

    Returns the resolved Schoenfeld residual DataFrame.
    """
    if y_train is None:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                "plot_schoenfeld() requires y_train when X_train is not a "
                "pre-computed DataFrame."
            )
        missing = _SCHOENFELD_COLS - set(X_train.columns)
        if missing:
            raise ValueError(
                f"plot_schoenfeld() DataFrame missing required columns: "
                f"{sorted(missing)}. Available: {sorted(X_train.columns)}"
            )
        return X_train
    return compute_schoenfeld_data(
        X_train, y_train, features=features, n_features=n_features,
    )


def plot_schoenfeld(
    X_train,
    y_train=None,
    features: Optional[List[str]] = None,
    n_features: int = 6,
    title: str = 'Scaled Schoenfeld Residuals',
    width: Optional[int] = None,
    height: Optional[int] = None,
    time_unit: Optional[str] = None,
) -> go.Figure:
    """Panel grid of scaled Schoenfeld residuals vs. time per covariate.

    Accepts **either** raw training data (``X_train``, ``y_train``) **or**
    a pre-computed :class:`~pandas.DataFrame` from
    :func:`compute_schoenfeld_data`.

    Parameters
    ----------
    X_train : DataFrame
        Training features, **or** a pre-computed DataFrame with columns
        ``feature``, ``time``, ``residual``.
    y_train : structured array, optional
        Training outcomes (required when *X_train* is raw data).
    features : list of str, optional
        Specific features to plot. If ``None``, selects the top
        *n_features* by absolute coefficient magnitude (raw data mode)
        or all features present (DataFrame mode).
    n_features : int
        Number of features to plot (ignored if *features* is provided).
    title : str
        Figure title.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    time_unit : str, optional
        Unit label appended to the x-axis title
        (e.g. ``"months"`` -> ``"Time (months)"``).

    Returns
    -------
    go.Figure

    Notes
    -----
    Coefficients may differ slightly from sksurv models — this is acceptable
    for diagnostic purposes.  Optional LOWESS smoothing via ``statsmodels``
    (gracefully skipped if unavailable).
    """
    data = _resolve_schoenfeld_data(X_train, y_train, features, n_features)

    # Determine features to plot
    all_features = data["feature"].unique().tolist()
    if not all_features:
        return _empty_figure(title, width, height)

    if features is not None:
        plot_features = [f for f in features if f in all_features]
    else:
        plot_features = all_features[:n_features]

    if not plot_features:
        return _empty_figure(title, width, height)

    n_feat = len(plot_features)
    n_cols = min(3, n_feat)
    n_rows = math.ceil(n_feat / n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=plot_features,
        vertical_spacing=0.18,
        horizontal_spacing=0.14,
    )

    for idx, feat in enumerate(plot_features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        feat_data = data[data["feature"] == feat]
        event_times = feat_data["time"].values
        resid_values = feat_data["residual"].values

        # Scatter
        fig.add_trace(go.Scatter(
            x=event_times, y=resid_values,
            mode='markers',
            marker=dict(size=3, color=_SCATTER_COLOR, opacity=0.5),
            showlegend=False,
            hovertemplate=f'<b>{feat}</b><br>Time: %{{x:.1f}}<br>Residual: %{{y:.3f}}<extra></extra>',
        ), row=row, col=col)

        # Reference line at 0
        fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1.5,
                      row=row, col=col)

        # Trend line
        smooth = _smooth_trend(event_times, resid_values)
        if smooth is not None:
            fig.add_trace(go.Scatter(
                x=smooth[0], y=smooth[1],
                mode='lines',
                line=dict(color=_LOWESS_COLOR, width=2),
                showlegend=False,
                hoverinfo='skip',
            ), row=row, col=col)

    d = MANUSCRIPT_DEFAULTS
    fig.update_layout(
        template="mlsurv_minimal",
        width=d["width_double"],
        height=adaptive_size(n_rows, per_item=280, base=60, min_size=d["height_small"]),
        title=dict(text=f"<b>{title}</b>"),
        showlegend=False,
    )

    time_label = f"Time ({time_unit})" if time_unit else "Time"
    for col_idx in range(1, n_cols + 1):
        fig.update_xaxes(title_text=time_label, row=n_rows, col=col_idx)
    for row_idx in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Residual", row=row_idx, col=1)
    fig.update_layout(margin=dict(
        l=d["margin_l"] + 30,   # room for "Residual" label
        t=d["margin_t"] + 20,   # room between title and subplot titles
        b=d["margin_b"] + 30,   # room for "Time" label
    ))

    apply_size_overrides(fig, width, height)
    return fig


def plot_schoenfeld_interactive(
    X_train,
    y_train=None,
    title: str = 'Scaled Schoenfeld Residuals',
    width: Optional[int] = None,
    height: Optional[int] = None,
    time_unit: Optional[str] = None,
) -> go.Figure:
    """Single-panel Schoenfeld residual plot with a dropdown to switch features.

    Shows the top feature (by absolute coefficient) by default.  The dropdown
    lets the user explore any individual feature.

    Parameters
    ----------
    X_train : DataFrame
        Training features **or** a pre-computed DataFrame (feature, time, residual).
    y_train : structured array, optional
        Required when *X_train* is raw data.
    title : str
        Figure title, passed through to layout.
    width : int, optional
        Figure width in pixels, passed through to layout.
    height : int, optional
        Figure height in pixels, passed through to layout.
    time_unit : str, optional
        Unit label appended to the x-axis title (e.g. ``"months"``).

    Returns
    -------
    go.Figure

    Raises
    ------
    TypeError
        If ``y_train`` is omitted and ``X_train`` is not a pre-computed
        DataFrame.
    ValueError
        If ``X_train`` is a pre-computed DataFrame missing the required
        columns.
    """
    if y_train is None:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                "plot_schoenfeld_interactive() requires y_train when X_train "
                "is not a pre-computed DataFrame."
            )
        missing = _SCHOENFELD_COLS - set(X_train.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {sorted(missing)}"
            )
        data = X_train
    else:
        data = compute_schoenfeld_data(
            X_train, y_train, n_features=X_train.shape[1],
        )

    all_features = data["feature"].unique().tolist()
    if not all_features:
        return _empty_figure(title, width, height)

    fig = go.Figure()
    feat_indices = {}  # feature -> list of trace indices

    for feat in all_features:
        feat_data = data[data["feature"] == feat]
        times = feat_data["time"].values
        resids = feat_data["residual"].values
        indices = []

        # First feature visible by default, rest hidden
        is_first = (feat == all_features[0])

        fig.add_trace(go.Scatter(
            x=times, y=resids, mode='markers',
            marker=dict(size=4, color=_SCATTER_COLOR, opacity=0.5),
            showlegend=False,
            visible=is_first,
            hovertemplate=(
                f'<b>{feat}</b><br>Time: %{{x:.1f}}<br>'
                f'Residual: %{{y:.3f}}<extra></extra>'
            ),
        ))
        indices.append(len(fig.data) - 1)

        smooth = _smooth_trend(times, resids)
        if smooth is not None:
            fig.add_trace(go.Scatter(
                x=smooth[0], y=smooth[1], mode='lines',
                line=dict(color=_LOWESS_COLOR, width=2),
                showlegend=False,
                visible=is_first,
                hoverinfo='skip',
            ))
            indices.append(len(fig.data) - 1)

        feat_indices[feat] = indices

    n_traces = len(fig.data)

    # --- Dropdown buttons ---
    display_features, _ = truncate_labels(all_features, max_chars=30)
    buttons = []
    for feat, display in zip(all_features, display_features):
        vis = [False] * n_traces
        for i in feat_indices[feat]:
            vis[i] = True
        buttons.append(dict(
            label=display,
            method="update",
            args=[
                {"visible": vis},
                {"title.text": f"<b>{title}</b> — {feat}"},
            ],
        ))

    d = MANUSCRIPT_DEFAULTS
    fig.update_layout(
        template="mlsurv_minimal",
        width=d["width_double"],
        height=d["height_small"],
        title=dict(text=f"<b>{title}</b> — {all_features[0]}"),
        xaxis_title=f"Time ({time_unit})" if time_unit else "Time",
        yaxis_title="Scaled Schoenfeld Residual",
        showlegend=False,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.0, xanchor="right",
            y=1.15, yanchor="top",
            buttons=buttons,
            active=0,
            showactive=True,
            bgcolor="#f5f7fa",
            bordercolor="#ddd",
            font=dict(size=11),
        )],
    )

    # Zero reference line
    fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1.5)

    fig.update_layout(margin=dict(
        l=d["margin_l"] + 10,
        t=d["margin_t"] + 30,
        b=d["margin_b"],
    ))

    apply_size_overrides(fig, width, height)
    return fig


def plot_deviance(
    X_train,
    y_train=None,
    title: str = 'Deviance Residuals',
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """Deviance residuals vs. linear predictor, color-coded by event status.

    Accepts **either** raw training data (``X_train``, ``y_train``) **or**
    a pre-computed :class:`~pandas.DataFrame` from
    :func:`compute_deviance_data`.

    Parameters
    ----------
    X_train : DataFrame
        Training features, **or** a pre-computed DataFrame with columns
        ``linear_predictor``, ``deviance``, ``event``.
    y_train : structured array, optional
        Training outcomes (required when *X_train* is raw data).
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
    TypeError
        If ``y_train`` is omitted and ``X_train`` is not a pre-computed
        DataFrame.
    ValueError
        If ``X_train`` is a pre-computed DataFrame missing the required
        columns.
    """
    # Dual-input dispatch: pre-computed DataFrame when y_train is omitted
    if y_train is None:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                "plot_deviance() requires y_train when X_train is not a "
                "pre-computed DataFrame."
            )
        missing = _DEVIANCE_COLS - set(X_train.columns)
        if missing:
            raise ValueError(
                f"plot_deviance() DataFrame missing required columns: "
                f"{sorted(missing)}. Available: {sorted(X_train.columns)}"
            )
        data = X_train
    else:
        data = compute_deviance_data(X_train, y_train)

    if len(data) == 0:
        return _empty_figure(title, width, height)

    linear_predictor = data["linear_predictor"].values
    deviance = data["deviance"].values
    event_status = data["event"].values.astype(bool)

    fig = create_figure(title=title, width="single", height="small")

    # Events
    fig.add_trace(go.Scatter(
        x=linear_predictor[event_status],
        y=deviance[event_status],
        mode='markers',
        name='Event',
        marker=dict(size=5, color=_EVENT_COLOR, opacity=0.6),
        hovertemplate='LP: %{x:.2f}<br>Deviance: %{y:.3f}<extra></extra>',
    ))

    # Censored
    fig.add_trace(go.Scatter(
        x=linear_predictor[~event_status],
        y=deviance[~event_status],
        mode='markers',
        name='Censored',
        marker=dict(size=5, color=_CENSORED_COLOR, opacity=0.6),
        hovertemplate='LP: %{x:.2f}<br>Deviance: %{y:.3f}<extra></extra>',
    ))

    # Reference lines at +/-2
    fig.add_hline(y=2, line_dash='dash', line_color='gray', line_width=1.5,
                  annotation_text='+2', annotation_position='top right')
    fig.add_hline(y=-2, line_dash='dash', line_color='gray', line_width=1.5,
                  annotation_text='-2', annotation_position='bottom right')
    fig.add_hline(y=0, line_dash='solid', line_color='gray', line_width=1.5)

    fig.update_layout(
        xaxis_title='Linear Predictor',
        yaxis_title='Deviance Residual',
        legend=dict(title=dict(text='Status')),
    )

    apply_size_overrides(fig, width, height)
    return fig


# ── Martingale residuals for linearity diagnostics ───────────────────────
#
# ``compute_martingale_data`` (and its ``_identify_continuous_for_martingale``
# helper) now live in :mod:`mlsurv.evaluation._diagnostics`; it is imported at
# the top of this module and re-exported for backward compatibility.


def _resolve_martingale_data(X_train, y_train, features, min_unique):
    """Dispatch between pre-computed DataFrame and raw data.

    Mirrors ``_resolve_schoenfeld_data``.  When ``y_train`` is omitted we
    treat ``X_train`` as a pre-computed DataFrame and validate columns.
    """
    if y_train is None:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                "plot_martingale_interactive() requires y_train when "
                "X_train is not a pre-computed DataFrame."
            )
        missing = _MARTINGALE_COLS - set(X_train.columns)
        if missing:
            raise ValueError(
                f"plot_martingale_interactive() DataFrame missing required "
                f"columns: {sorted(missing)}. Available: {sorted(X_train.columns)}"
            )
        return X_train
    return compute_martingale_data(
        X_train, y_train, features=features, min_unique=min_unique,
    )


def plot_martingale_interactive(
    X_train,
    y_train=None,
    features: Optional[List[str]] = None,
    min_unique: int = 6,
    title: str = 'Martingale Residuals vs. Covariate',
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """Interactive dropdown plot of martingale residuals vs. each continuous covariate.

    Shows one continuous covariate at a time, with scatter points (one per
    observation) and a LOWESS trend line.  A **straight** LOWESS trend
    (of any slope) is consistent with a linear log-hazard effect for that
    covariate.  A **curved** trend (U-shape, monotone curve, step, etc.)
    reveals the functional form of the covariate's effect on the hazard
    and indicates a violation of the linearity assumption — typical
    remedies are a non-linear transformation of the feature or switching
    to a flexible model (RSF, GBSA, DeepSurv).

    Note: a flat trend near zero means the covariate has *no* association
    with hazard — which is also consistent with linearity (slope = 0),
    but is not a positive confirmation of linearity.  Linearity is
    assessed by the absence of *curvature*, not by a flat line.

    Accepts **either** raw training data (``X_train``, ``y_train``) **or**
    a pre-computed :class:`~pandas.DataFrame` from
    :func:`compute_martingale_data`.

    Parameters
    ----------
    X_train : DataFrame
        Training features, **or** a pre-computed DataFrame with columns
        ``feature``, ``x_value``, ``residual``.
    y_train : structured array, optional
        Training outcomes (required when ``X_train`` is raw data).
    features : list of str, optional
        Restrict to these features.  If ``None``, use every continuous
        column (raw-data mode) or every feature present (DataFrame mode).
    min_unique : int, default=6
        Minimum distinct values for a numeric column to count as continuous
        (raw-data mode only).
    title : str
        Figure title, passed through to layout.
    width : int, optional
        Figure width in pixels, passed through to layout.
    height : int, optional
        Figure height in pixels, passed through to layout.

    Returns
    -------
    go.Figure

    References
    ----------
    Therneau, Grambsch & Fleming (1990), "Martingale-based residuals for
    survival models", *Biometrika* 77(1):147-160.
    """
    data = _resolve_martingale_data(X_train, y_train, features, min_unique)

    all_features = data["feature"].unique().tolist() if len(data) else []
    if not all_features:
        return _empty_figure(title, width, height)

    fig = go.Figure()
    feat_indices = {}  # feature -> list of trace indices

    for feat in all_features:
        feat_data = data[data["feature"] == feat]
        x_vals = feat_data["x_value"].to_numpy(dtype=float)
        resids = feat_data["residual"].to_numpy(dtype=float)
        indices = []

        is_first = (feat == all_features[0])

        fig.add_trace(go.Scatter(
            x=x_vals, y=resids, mode='markers',
            marker=dict(size=4, color=_SCATTER_COLOR, opacity=0.5),
            showlegend=False,
            visible=is_first,
            hovertemplate=(
                f'<b>{feat}</b><br>Value: %{{x:.3f}}<br>'
                f'Residual: %{{y:.3f}}<extra></extra>'
            ),
        ))
        indices.append(len(fig.data) - 1)

        smooth = _smooth_trend(x_vals, resids)
        if smooth is not None:
            fig.add_trace(go.Scatter(
                x=smooth[0], y=smooth[1], mode='lines',
                line=dict(color=_LOWESS_COLOR, width=2),
                showlegend=False,
                visible=is_first,
                hoverinfo='skip',
            ))
            indices.append(len(fig.data) - 1)

        feat_indices[feat] = indices

    n_traces = len(fig.data)

    display_features, _ = truncate_labels(all_features, max_chars=30)
    buttons = []
    for feat, display in zip(all_features, display_features):
        vis = [False] * n_traces
        for i in feat_indices[feat]:
            vis[i] = True
        buttons.append(dict(
            label=display,
            method="update",
            args=[
                {"visible": vis},
                {"title.text": f"<b>{title}</b> — {feat}"},
            ],
        ))

    d = MANUSCRIPT_DEFAULTS
    fig.update_layout(
        template="mlsurv_minimal",
        width=d["width_double"],
        height=d["height_small"],
        title=dict(text=f"<b>{title}</b> — {all_features[0]}"),
        xaxis_title="Covariate value",
        yaxis_title="Martingale residual",
        showlegend=False,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.0, xanchor="right",
            y=1.15, yanchor="top",
            buttons=buttons,
            active=0,
            showactive=True,
            bgcolor="#f5f7fa",
            bordercolor="#ddd",
            font=dict(size=11),
        )],
    )

    # Zero reference line
    fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1.5)

    fig.update_layout(margin=dict(
        l=d["margin_l"] + 10,
        t=d["margin_t"] + 30,
        b=d["margin_b"],
    ))

    apply_size_overrides(fig, width, height)
    return fig
