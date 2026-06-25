"""Shared constants, data-prep utilities, and renderer helpers for metrics viz.

All functions here are internal (``_``-prefixed) and shared across the
``_metrics_renderers``, ``_metrics_multi``, ``_metrics_time``, and
``_metrics_roc`` submodules.
"""

from typing import TYPE_CHECKING, Dict, Optional, Set

import numpy as np
import pandas as pd
from scipy.stats import norm

from mlsurv.constants import LOWER_IS_BETTER, get_metric_display_name
from mlsurv.viz._palettes import (
    MANUSCRIPT_DEFAULTS,
    get_model_color,
    get_model_display_name,
    get_model_display_name as _model_display,
    get_reference_color,
)
from mlsurv.viz._plotly_helpers import asymmetric_error_bar
from mlsurv.viz._theme import (
    adaptive_size, create_figure,
    margin_for_labels as _theme_margin_for_labels,
)

import plotly.graph_objects as go

if TYPE_CHECKING:
    from mlsurv.viz._context import SubpopulationAxis


def normalize_metric_param(metric, *, collapse_singleton=True):
    """Normalize the public ``metric: str | list[str]`` parameter.

    Returns a ``(single_metric, metric_list)`` pair where exactly one slot is
    non-``None`` (or both ``None`` for the default case)::

        None        -> (None, None)         caller applies its own default
        "auc"       -> ("auc", None)        single panel
        ["auc"]     -> ("auc", None)        1-element list collapses to scalar
        ["a", "b"]  -> (None, ["a", "b"])   multi-panel facet

    When ``collapse_singleton`` is ``False`` a 1-element list stays a list
    (``["auc"] -> (None, ["auc"])``). Public entry points collapse (so a
    1-element list behaves like the scalar); the low-level viz renderers do not
    (they receive a pre-shaped frame and treat any list as multi-panel).

    Tuples are accepted like lists. An empty list/tuple raises ``ValueError``; a
    non-string/non-sequence value, or a sequence holding a non-string element,
    raises ``TypeError``.
    """
    if metric is None:
        return None, None
    if isinstance(metric, str):
        return metric, None
    if isinstance(metric, (list, tuple)):
        if len(metric) == 0:
            raise ValueError(
                "metric=[] is empty; pass a metric name or a list of "
                "metric names."
            )
        for i, m in enumerate(metric):
            if not isinstance(m, str):
                raise TypeError(
                    "metric= list entries must be strings; got "
                    f"{type(m).__name__} at index {i}."
                )
        if collapse_singleton and len(metric) == 1:
            return metric[0], None
        return None, list(metric)
    raise TypeError(
        f"metric= must be a str or list of str, got {type(metric).__name__}."
    )


# =============================================================================
# Metric reference helpers
# =============================================================================

# Multi-panel reference values keyed by metric/column name (lowercased).
# Used by _plot_metric_multi_forest and _plot_metric_multi_panel to draw
# dashed reference lines on each subplot.
_PANEL_REFERENCE_VALUES: Dict[str, float] = {
    "harrell": 0.5, "harrell_cindex": 0.5,
    "ipcw": 0.5, "ipcw_cindex": 0.5,
    "ctd": 0.5, "ctd_cindex": 0.5,
    "c_index": 0.5, "cv_metric_mean": 0.5,
    "integrated_brier": 0.25, "brier": 0.25,
}

# Concordance-family metric keys -- panels sharing these keys should
# use a unified x-axis range for visual comparability.
_CONCORDANCE_FAMILY: frozenset[str] = frozenset({
    "harrell", "harrell_cindex",
    "ipcw", "ipcw_cindex",
    "ctd", "ctd_cindex",
    "c_index", "cv_metric_mean",
})


def _metric_col_display(col: str) -> str:
    """Convert a DataFrame column name to a human-readable metric label.

    Handles evaluation result columns like ``harrell_cindex``, ``ipcw_cindex``,
    and ``integrated_brier`` by mapping through the constants registry.
    """
    # Direct registry hit (e.g. "integrated_brier")
    display = get_metric_display_name(col)
    if display != col.title():
        return display
    # Strip common suffixes and retry (e.g. "harrell_cindex" → "harrell")
    for suffix in ("_cindex", "_brier", "_score"):
        if col.endswith(suffix):
            base = col[:-len(suffix)]
            d = get_metric_display_name(base)
            if d != base.title():
                return d
    # Fallback: replace underscores with spaces and title-case
    return col.replace("_", " ").title()


def _is_lower_better(metric) -> bool:
    """True for metrics where lower is better.

    Single source of truth for metric direction across the viz package:
    time-series reference lines / axis ranges, the heatmap z-score sign flip
    in :mod:`mlsurv.viz.metrics`, the bar/boxplot/raincloud sorting and range
    computation in :mod:`mlsurv.viz._metrics_renderers`, and the multi-panel
    forest sorting in :mod:`mlsurv.viz._metrics_multi`. Covers
    :data:`LOWER_IS_BETTER` registry keys plus the non-registry short names
    ``brier``/``brier_score``; falsy/``None`` metrics return ``False``.
    """
    return bool(metric) and (
        metric in LOWER_IS_BETTER or metric in ("brier", "brier_score")
    )


def _metric_reference(metric: str):
    """Return (ref_value, ref_label) for a metric's reference line.

    C-index-family: 0.5 (chance), Brier-family: 0.25 (uninformative).
    Unknown metrics: no reference line (returns None, None).
    """
    if _is_lower_better(metric):
        return 0.25, "Uninformative"
    if metric in ("harrell", "ipcw", "auc", "cindex", "uno", "ctd"):
        return 0.5, "Random Chance"
    return None, None


def _maybe_add_reference(fig, metric, show_reference_line, axes):
    """Add a reference line if applicable for the given metric."""
    if not (show_reference_line and metric):
        return
    ref_val, ref_label = _metric_reference(metric)
    if ref_val is not None:
        axes.add_reference(fig, ref_val, ref_label)


# =============================================================================
# Data preparation helpers (from discrimination.py)
# =============================================================================

def _wide_to_long_summary(df, metrics, model_col, ci_level=0.95,
                          uncertainty_label=None):
    """Convert wide-format multi-metric data to long summary format.

    Wide format: ``[model_col, m1, m1_std, m2, m2_std, ...]`` and, for
    bootstrap, also ``m1_ci_lower``/``m1_ci_upper`` percentile bounds.
    Long format: ``[model_col, metric_name, point, ci_lower, ci_upper]``

    Mirrors the wide-forest renderer's column preference so the bar path draws
    the same band as the forest:

    1. When *uncertainty_label* is ``"SD"`` the bounds are ``point +/- std``
       (z = 1) -- the CV ± SD band, or an explicit SD override for bootstrap.
    2. Else prefer the ``{m}_ci_lower``/``{m}_ci_upper`` percentile bounds when
       the columns are present (bootstrap) -- they match the CI the forest
       draws, and a NaN bound there yields no band, exactly like the forest.
    3. Else fall back to ``point +/- z*std`` reconstructed from ``{m}_std``
       (CV with an explicit float, or normal-approx CI).
    """
    is_sd = uncertainty_label == "SD"
    z = 1.0 if is_sd else norm.ppf(1 - (1 - ci_level) / 2)
    rows = []
    for _, row in df.iterrows():
        model = row[model_col]
        for m in metrics:
            point = row[m]
            ci_lo_col = f"{m}_ci_lower"
            ci_hi_col = f"{m}_ci_upper"
            std_col = f"{m}_std"
            if not is_sd and ci_lo_col in df.columns and ci_hi_col in df.columns:
                # Bootstrap percentile CI carried alongside the point. Mirror
                # the wide-forest renderer's column-presence test exactly: when
                # the bound columns exist we use them as-is, so a NaN bound (a
                # bootstrap pair with missing iterations) yields no band -- the
                # same as the forest -- rather than falling back to a z*std band
                # the forest would never draw.
                ci_lo = row[ci_lo_col]
                ci_hi = row[ci_hi_col]
            elif std_col in df.columns and pd.notna(row[std_col]):
                std = row[std_col]
                ci_lo = point - z * std
                ci_hi = point + z * std
            else:
                ci_lo = np.nan
                ci_hi = np.nan
            rows.append({
                model_col: model,
                "metric_name": m,
                "point": point,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            })
    return pd.DataFrame(rows)


def _prepare_forest_data_from_comparison(
    metrics_df: pd.DataFrame,
    metric_col: str = "cv_metric_mean",
    std_col: str = "cv_metric_std",
    model_col: str = "model",
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Convert CV metrics DataFrame to forest-plot DataFrame.

    Derives CI as point +/- z * std, where *z* is the normal quantile
    for the requested *ci_level*.
    """
    if not 0 < ci_level < 1:
        raise ValueError(
            f"ci_level must be between 0 and 1 (exclusive), got {ci_level}. "
            "Use 0.90, 0.95, or 0.99 — not a percentage like 90 or 95."
        )
    z = norm.ppf(1 - (1 - ci_level) / 2)
    df = metrics_df[[model_col, metric_col, std_col]].copy()
    df = df.rename(columns={metric_col: "point", model_col: "model"})
    df["ci_lower"] = df["point"] - z * df[std_col]
    df["ci_upper"] = df["point"] + z * df[std_col]
    return df[["model", "point", "ci_lower", "ci_upper"]]


def _prepare_forest_data_from_bootstrap(
    summary_df: pd.DataFrame,
    metric: str = "harrell",
    ci_level: float = 0.95,
    iterations: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Convert ``BootstrapResult`` data to forest-plot DataFrame.

    When *iterations* is provided, CIs are recomputed as percentiles at
    the requested *ci_level* directly from the raw bootstrap distribution.
    """
    if not 0 < ci_level < 1:
        raise ValueError(
            f"ci_level must be between 0 and 1 (exclusive), got {ci_level}. "
            "Use 0.90, 0.95, or 0.99 — not a percentage like 90 or 95."
        )
    df = summary_df[summary_df["metric"] == metric].copy()
    if df.empty:
        raise ValueError(
            f"Metric '{metric}' not found in bootstrap summary. "
            f"Available: {summary_df['metric'].unique().tolist()}"
        )

    point_vals = df["point"].values if "point" in df.columns else df["mean"].values

    if iterations is not None and not iterations.empty:
        alpha = 1 - ci_level
        lo_pct = 100 * alpha / 2
        hi_pct = 100 * (1 - alpha / 2)

        iter_metric = iterations[iterations["metric_name"] == metric]
        ci_lowers, ci_uppers = [], []
        for model_type in df["model_type"].values:
            vals = iter_metric.loc[
                iter_metric["model_type"] == model_type, "value"
            ].dropna().values.astype(float)
            if len(vals) > 0:
                ci_lowers.append(np.percentile(vals, lo_pct))
                ci_uppers.append(np.percentile(vals, hi_pct))
            else:
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)

        return pd.DataFrame({
            "model": df["model_type"].values,
            "point": point_vals,
            "ci_lower": ci_lowers,
            "ci_upper": ci_uppers,
        })

    return pd.DataFrame({
        "model": df["model_type"].values,
        "point": point_vals,
        "ci_lower": df["ci_lower"].values,
        "ci_upper": df["ci_upper"].values,
    })


# =============================================================================
# Stacked offset helper (for raincloud strip points)
# =============================================================================

def _stacked_offsets(
    values: np.ndarray, *, step: float = 0.015, max_offset: float = 0.18,
    n_bins: int = 40,
) -> np.ndarray:
    """Compute y-offsets that stack points downward within bins (dot-plot style)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])

    x_range = x.max() - x.min()
    if x_range == 0:
        return np.zeros(n)

    bins = np.clip(
        ((x - x.min()) / x_range * (n_bins - 1)).astype(int), 0, n_bins - 1,
    )
    offsets = np.zeros(n)

    for b in range(n_bins):
        mask = bins == b
        count = mask.sum()
        if count <= 1:
            continue
        positions = -np.arange(count) * step
        positions = np.clip(positions, -max_offset, 0)
        offsets[mask] = positions

    return offsets


# =============================================================================
# Shared metric-renderer helpers
# =============================================================================

def _create_oriented_figure(axes, n_items, title,
                            h_per_item=40, h_base=120, h_min=None,
                            w_per_item=85, w_base=70, w_min=260):
    """Create figure with dimensions adapted to orientation and item count."""
    if h_min is None:
        h_min = MANUSCRIPT_DEFAULTS["height_small"]
    if axes.is_horiz:
        h = adaptive_size(n_items, per_item=h_per_item, base=h_base, min_size=h_min)
        return create_figure(title=title, width="single", height=h)
    w = adaptive_size(n_items, per_item=w_per_item, base=w_base, min_size=w_min)
    return create_figure(title=title, width=w, height="medium")


def _extract_ci(df, ci_lower_col, ci_upper_col, points):
    """Extract CI bounds and error dict from data.

    Returns (ci_lo, ci_hi, error_dict, has_ci).
    """
    ci_lo = ci_hi = None
    error_dict = None
    if ci_lower_col in df.columns and ci_upper_col in df.columns:
        ci_lo = df[ci_lower_col].values
        ci_hi = df[ci_upper_col].values
        if not np.allclose(ci_lo, ci_hi):
            error_dict = dict(
                type='data', symmetric=False,
                array=(ci_hi - points).tolist(), arrayminus=(points - ci_lo).tolist(), visible=True,
            )
    return ci_lo, ci_hi, error_dict, error_dict is not None


def _bar_value_range(xlim, points, lower_is_better):
    """Compute value-axis range for bar charts."""
    if xlim is not None:
        return list(xlim)
    if lower_is_better:
        return [0, (max(points) * 1.2) if len(points) > 0 else 0.5]
    return [0.4, (max(points) + 0.1) if len(points) > 0 else 1.0]


def _compute_value_range(values_lo, values_hi, xlim, ref_val=None,
                         pad_frac=0.15, label_pad=0.0):
    """Compute padded value-axis range from lo/hi arrays."""
    if xlim is not None:
        return list(xlim)
    v_min = float(np.nanmin(values_lo))
    v_max = float(np.nanmax(values_hi))
    pad = (v_max - v_min) * pad_frac
    range_lo = v_min - pad
    range_hi = v_max + pad + label_pad
    if ref_val is not None and range_lo < ref_val + 0.05:
        range_lo = min(range_lo, ref_val - 0.02)
    return [range_lo, range_hi]


def _build_uncertainty_hover(metric_display, has_ci, is_sd, u_label, axes,
                             ci_level=0.95):
    """Build hover template for forest/bar/boxplot renderers."""
    model_ref = axes.model_ref()
    val_ref = axes.val_ref()
    if has_ci:
        if is_sd:
            return (
                f"<b>{model_ref}</b><br>"
                f"{metric_display}: {val_ref}<br>"
                "SD: %{customdata[2]:.3f}<br>"
                "\u00b1 1 SD: "
                "[%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
            )
        u_label = u_label or f"{ci_level * 100:g}% CI"
        return (
            f"<b>{model_ref}</b><br>"
            f"{metric_display}: {val_ref}<br>"
            f"{u_label}: "
            "[%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
        )
    return (
        f"<b>{model_ref}</b><br>"
        f"{metric_display}: {val_ref}<extra></extra>"
    )


def _add_point_estimate_marker(fig, pe_val, display_name, color, position,
                               axes, row=None, col=None):
    """Add a triangle point-estimate marker to a raincloud/boxplot."""
    pe_hover = (
        f"<b>{display_name}</b><br>"
        f"Test-set estimate: {pe_val:.4f}"
        "<extra></extra>"
    )
    pe_symbol = "triangle-down" if axes.is_horiz else "triangle-left"
    pe_kw = dict(
        mode="markers",
        marker=dict(
            symbol=pe_symbol, size=10, color=color,
            line=dict(width=1.5, color="white"),
        ),
        showlegend=False,
        legendgroup="point_estimate",
        hovertemplate=pe_hover,
        **axes.scatter_kw([pe_val], [position]),
    )
    trace_kw = {}
    if row is not None and col is not None:
        trace_kw = dict(row=row, col=col)
    fig.add_trace(go.Scatter(**pe_kw), **trace_kw)


def _add_pe_legend_entry(fig, has_pe):
    """Add a single gray legend entry for point-estimate markers."""
    if has_pe:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="gray"),
            name="Test-set estimate",
            legendgroup="point_estimate",
            showlegend=True,
        ))


def _filter_iterations_by_metric(iterations, metric, metric_col):
    """Filter iteration data to a single metric and return a copy."""
    if metric is not None and metric_col in iterations.columns:
        df = iterations[iterations[metric_col] == metric].copy()
        if df.empty:
            raise ValueError(
                f"Metric '{metric}' not found in iterations. "
                f"Available: {iterations[metric_col].unique().tolist()}"
            )
    else:
        df = iterations.copy()
    return df


def _prepare_model_styling(df, model_col, value_col, sort, palette):
    """Sort models, compute display names and colors."""
    models = df[model_col].unique().tolist()
    if sort == "alpha":
        models = sorted(models, reverse=True)
    elif sort == "value":
        means = {m: df.loc[df[model_col] == m, value_col].mean() for m in models}
        models = sorted(models, key=lambda m: means[m])
    display_names = [_model_display(m) for m in models]
    colors = [get_model_color(m, palette=palette) for m in models]
    return models, display_names, colors


def _margin_for_labels(display_names, axes):
    """Compute left margin for horizontal label display."""
    if axes.is_horiz and display_names:
        return _theme_margin_for_labels(
            display_names, min_margin=MANUSCRIPT_DEFAULTS["margin_l"],
        )
    return MANUSCRIPT_DEFAULTS["margin_l"]


def _pe_legend_kw(has_pe):
    """Return legend kwargs for point-estimate-bearing figures."""
    if has_pe:
        return dict(
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="left", x=1.02,
            ),
        )
    return dict(showlegend=False)


def _unify_concordance_xranges(fig, metrics) -> None:
    """Unify x-axis ranges across concordance-family panels in a subplot figure.

    Concordance metrics (Harrell, Uno, Antolini) share the same scale
    (0.5 = chance, 1.0 = perfect), so they should display on the same
    x-axis range for visual comparability.
    """
    conc_indices = [
        i for i, m in enumerate(metrics)
        if m.lower() in _CONCORDANCE_FAMILY
    ]
    if len(conc_indices) < 2:
        return

    # Collect data extents from concordance-panel traces
    x_vals: list[float] = []
    for idx in conc_indices:
        axis_name = "x" if idx == 0 else f"x{idx + 1}"
        for trace in fig.data:
            if getattr(trace, "xaxis", "x") != axis_name or trace.x is None:
                continue
            vals = [v for v in trace.x if v is not None and not np.isnan(v)]
            x_vals.extend(vals)
            # Include error bar extents (arrayminus for asymmetric CIs)
            err = getattr(trace, "error_x", None)
            if err is not None:
                arr_plus = getattr(err, "array", None)
                arr_minus = getattr(err, "arrayminus", None) or arr_plus
                if arr_plus is not None:
                    for v, e in zip(trace.x, arr_plus):
                        if v is not None and e is not None:
                            x_vals.append(v + e)
                if arr_minus is not None:
                    for v, e in zip(trace.x, arr_minus):
                        if v is not None and e is not None:
                            x_vals.append(v - e)

    if not x_vals:
        return

    v_min = min(x_vals)
    v_max = max(x_vals)
    # Ensure reference line (0.5) is visible
    v_min = min(v_min, 0.48)
    pad = (v_max - v_min) * 0.15
    unified_range = [v_min - pad, v_max + pad]

    for idx in conc_indices:
        axis_key = "xaxis" if idx == 0 else f"xaxis{idx + 1}"
        fig.layout[axis_key].range = unified_range


# =============================================================================
# Grouped forest plot helpers (subpopulation / stratified)
# =============================================================================
#
# The five helpers below (``_compute_forest_summary``, ``_compute_y_positions``,
# ``_resolve_model_style``, ``_add_forest_traces``, ``_add_alternating_bands``)
# were moved **byte-identical** from
# ``mlsurv.reports.sections.subpopulation_analysis``. Preserving byte identity
# is load-bearing: ``tests/test_subpopulation_analysis.py::TestSubpopulationAucTPlot``
# pins the ribbon-fill / per-panel trace invariants against the output.
#
# ``plot_grouped_forest`` below is NOT byte-identical: it refactors the old
# ``SubpopulationAnalysisSection._build_forest_plot`` method body to accept a
# :class:`SubpopulationAxis` (instead of reading ``self.learner`` /
# ``self._get_reference_names`` / ``self._forest_hover``) and drops the
# ``try/except IMPORT_EXCEPTIONS`` wrapper — error handling moved up to the
# section proxy. Layout formulas, trace assembly order, and hover formatting
# are preserved so the test assertions still hold semantically.


def _compute_forest_summary(metrics_df, boot_df, model_names, subpopulations, metric):
    """Build a summary DataFrame with point estimates and CI bounds."""
    rows = []
    for sg in subpopulations:
        for mname in model_names:
            mask = (
                (metrics_df['subpopulation'] == sg)
                & (metrics_df['model_type'] == mname)
                & (metrics_df['metric'] == metric)
            )
            vals = metrics_df.loc[mask, 'value']
            pt = float(vals.iloc[0]) if len(vals) > 0 else np.nan

            cl, ch = np.nan, np.nan
            if boot_df is not None:
                ci_mask = (
                    (boot_df['subpopulation'] == sg)
                    & (boot_df['model_type'] == mname)
                    & (boot_df['metric'] == metric)
                )
                ci_rows = boot_df[ci_mask]
                if len(ci_rows) > 0:
                    cl = float(ci_rows.iloc[0].get('ci_lower', np.nan))
                    ch = float(ci_rows.iloc[0].get('ci_upper', np.nan))

            rows.append({
                'model': mname,
                'subpopulation': str(sg),
                'point': pt,
                'ci_lower': cl,
                'ci_upper': ch,
            })
    return pd.DataFrame(rows)


def _compute_y_positions(n_subpopulations, n_models, gap=0.6):
    """Compute y-axis positions for a grouped forest plot.

    Returns ``(y_map, sg_label_y)`` where *y_map* maps
    ``(subpopulation_idx, model_idx)`` to a y coordinate and *sg_label_y*
    lists the centre y for each subpopulation block.
    """
    y_map = {}
    sg_label_y = []
    y = 0
    for sg_idx in range(n_subpopulations):
        if sg_idx > 0:
            y += gap
        block_top = y
        for m_idx in range(n_models):
            y_map[(sg_idx, m_idx)] = y
            y += 1
        block_bot = y - 1  # last position assigned
        sg_label_y.append((block_top + block_bot) / 2)

    return y_map, sg_label_y


def _resolve_model_style(mname, ref_names, ref_idx):
    """Return ``(display_name, color, is_ref, ref_idx)`` for a model."""
    display_name = get_model_display_name(mname)
    is_ref = mname in ref_names
    if is_ref:
        color = get_reference_color(mname, ref_idx)
        return display_name, color, True, ref_idx + 1
    color = get_model_color(mname)
    return display_name, color, False, ref_idx


def _add_forest_traces(fig, summary_df, model_names, subpopulations, y_map, ref_names,
                       hover_fn, go):
    """Add per-model scatter traces with error bars to a forest plot figure."""
    ref_idx = 0
    for m_idx, mname in enumerate(model_names):
        model_df = summary_df[summary_df['model'] == mname]
        display_name, color, is_ref, ref_idx = _resolve_model_style(
            mname, ref_names, ref_idx,
        )
        marker_symbol = 'diamond' if is_ref else 'circle'

        pts = model_df['point'].values
        ci_lo = model_df['ci_lower'].values
        ci_hi = model_df['ci_upper'].values
        labels = model_df['subpopulation'].tolist()

        y_positions = [
            y_map[(subpopulations.index(sg), m_idx)] for sg in labels
        ]

        error_kw = {}
        if not np.all(np.isnan(ci_lo)):
            error_kw['error_x'] = asymmetric_error_bar(
                ci_hi - pts, pts - ci_lo,
                thickness=1.5, width=4, color=color,
            )

        hover = [
            hover_fn(display_name, sg, p, lo, hi)
            for sg, p, lo, hi in zip(labels, pts, ci_lo, ci_hi)
        ]

        fig.add_trace(go.Scatter(
            x=pts.tolist(),
            y=y_positions,
            mode='markers',
            name=display_name,
            marker=dict(
                size=9, color=color, symbol=marker_symbol,
                line=dict(width=1, color='white'),
            ),
            hovertemplate=hover,
            cliponaxis=False,
            legendgroup=display_name,
            **error_kw,
        ))


def _add_alternating_bands(fig, n_subpopulations, n_models, y_map):
    """Add alternating background shading bands for subpopulation blocks."""
    for sg_idx in range(n_subpopulations):
        if sg_idx % 2 == 0:
            band_top = y_map[(sg_idx, 0)] - 0.5
            band_bot = y_map[(sg_idx, n_models - 1)] + 0.5
            fig.add_shape(
                type='rect',
                x0=0, x1=1, xref='paper',
                y0=band_top, y1=band_bot, yref='y',
                fillcolor='rgba(0,0,0,0.04)',
                line_width=0, layer='below',
            )


def _default_forest_hover(display_name, sg, pt, lo, hi):
    """Default hover template for grouped forest traces."""
    if np.isnan(lo) or np.isnan(hi):
        return f"<b>{display_name}</b><br>{sg}: {pt:.3f}<extra></extra>"
    return (
        f"<b>{display_name}</b><br>"
        f"{sg}: {pt:.3f} [{lo:.3f}, {hi:.3f}]<extra></extra>"
    )


def _resolve_grouped_forest_metric(metrics_df, scoring):
    """Resolve which metric a grouped forest plots.

    Returns *scoring* when it names a metric present in ``metrics_df['metric']``,
    otherwise falls back to ``'harrell'``. Shared by ``plot_grouped_forest`` (the
    figure path) and the ``SubpopulationView.plot_metric(return_data=True)`` path
    so the returned DataFrame is filtered to exactly the metric the figure plots.
    """
    available = set(metrics_df['metric'].unique())
    return scoring if (scoring and scoring in available) else 'harrell'


def plot_grouped_forest(
    axis: "SubpopulationAxis",
    *,
    scoring: Optional[str] = None,
    ref_names: Optional[Set[str]] = None,
    ci_pct: str = "95",
    hover_fn=None,
):
    """Render a grouped forest plot: one row per (model, subpopulation level).

    Parameters
    ----------
    axis : SubpopulationAxis
        Pre-aggregated metric frames for the subpopulation variable.
    scoring : str, optional
        Preferred metric name. Falls back to ``'harrell'`` if not present
        in ``axis.metrics_long``.
    ref_names : set of str, optional
        Benchmark reference model names (rendered with diamond markers and
        reference palette). Defaults to an empty set.
    ci_pct : str, default "95"
        CI percentage to display in the subtitle (only shown when
        ``axis.bootstrap_summary`` is populated).
    hover_fn : callable, optional
        Hover template formatter with signature
        ``(display_name, sg, pt, lo, hi) -> str``. Defaults to
        :func:`_default_forest_hover`.

    Returns
    -------
    plotly.graph_objects.Figure or None
        ``None`` when the axis has no usable point estimates for the
        selected metric (all-NaN after filtering).
    """
    if ref_names is None:
        ref_names = set()
    if hover_fn is None:
        hover_fn = _default_forest_hover

    metrics_df = axis.metrics_long
    boot_df = axis.bootstrap_summary
    if metrics_df is None or metrics_df.empty:
        return None

    plot_metric_name = _resolve_grouped_forest_metric(metrics_df, scoring)
    metric_display = get_metric_display_name(plot_metric_name)

    model_names = list(metrics_df['model_type'].unique())
    subpopulations = sorted(str(s) for s in metrics_df['subpopulation'].unique())

    summary_df = _compute_forest_summary(
        metrics_df, boot_df, model_names, subpopulations, plot_metric_name,
    )
    if summary_df.empty or summary_df['point'].isna().all():
        return None

    # Drop subpopulations where ALL models have NaN points (e.g. too few
    # events to compute the metric).  Keeps the plot clean.
    valid_sgs = set(
        summary_df.loc[summary_df['point'].notna(), 'subpopulation']
    )
    subpopulations = [sg for sg in subpopulations if sg in valid_sgs]
    summary_df = summary_df[summary_df['subpopulation'].isin(valid_sgs)]
    if not subpopulations:
        return None

    fig = go.Figure()
    n_models = len(model_names)
    n_subpopulations = len(subpopulations)

    y_map, sg_label_y = _compute_y_positions(n_subpopulations, n_models)

    _add_forest_traces(
        fig, summary_df, model_names, subpopulations, y_map, ref_names,
        hover_fn, go,
    )
    _add_alternating_bands(fig, n_subpopulations, n_models, y_map)

    all_y = list(y_map.values())
    y_lo, y_hi = min(all_y) - 0.5, max(all_y) + 0.5
    total_rows = n_models * n_subpopulations
    fig_height = max(350, total_rows * 22 + 140)
    has_ci = boot_df is not None and not boot_df.empty
    ci_subtitle = (
        f"<br><sup>{ci_pct}% Bootstrap CI</sup>" if has_ci else ""
    )
    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(text=(
            f"<b>Subpopulation {metric_display}: {axis.group_variable}</b>"
            f"{ci_subtitle}"
        )),
        xaxis_title=metric_display,
        yaxis=dict(
            tickvals=sg_label_y,
            ticktext=subpopulations,
            range=[y_hi, y_lo],
            showgrid=False,
        ),
        height=fig_height,
        margin=dict(l=120, r=150, t=60, b=50),
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=1.02,
        ),
        showlegend=True,
    )

    return fig
