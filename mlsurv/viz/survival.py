"""Kaplan-Meier survival curve visualizations.

This module is the public facade. Implementation lives in submodules:
- ``_survival_maxstat``: optimal cutpoint search (MaxstatResult, etc.)
- ``_survival_helpers``: constants, validation, risk-group creation
- ``_survival_figure``: figure building primitives (traces, risk tables)
- ``_survival_logrank``: log-rank test computation and annotation
- ``_survival_renderers``: renderers (overall, risk, by-group, panel grid)

Public API:
    plot_km(data)  -- unified KM plot (single, by-group, by-risk, or panel grid)
    plot_patient_survival_curve(pred, patient)  -- individualized S(t|x) curve
        for one patient (shared builder behind PredictionResult.plot())

Internal helpers kept for report subsections:
    _km_combined   -- fit KM + return tidy data (used by report stratification)
    _survival_grid -- model x disease grid (niche, kept internal)

All names are re-exported here so existing ``from mlsurv.viz.survival import X``
statements continue to work unchanged.
"""

import warnings
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go

from mlsurv._warnings import SuggestionWarning
from mlsurv.viz._theme import apply_size_overrides, create_figure
from mlsurv.viz._time_format import (
    _axis_title_time,
    _canonicalize_time_unit,
    _format_time_with_unit,
)

# ── Submodule imports (used by the dispatcher below) ────────────────────────
from mlsurv.viz._survival_helpers import (
    _resolve_km_defaults,
    _validate_breaks,
    _validate_km_inputs,
    _warn_breaks_precedence,
)

# ── Re-exports: maxstat ────────────────────────────────────────────────────
from mlsurv.viz._survival_maxstat import (  # noqa: F401
    DegenerateSubpopulationError,
    MaxstatResult,
    _VALID_CORRECTIONS,
    _evaluate_cutpoint_candidates,
    _fast_logrank_chi2,
    _find_maxstat_cutpoint,
    _find_maxstat_cutpoints_recursive,
    _resolve_maxstat,
)

# ── Re-exports: helpers ───────────────────────────────────────────────────
from mlsurv.viz._survival_helpers import (  # noqa: F401
    _MAXSTAT_LABELS,
    _VALID_BREAK_TYPES,
    _VALID_P_ADJUST,
    _create_risk_groups,
    _maxstat_subtitle,
    _pct_range_labels,
    _resolve_axes_params,
    _resolve_risk_groups,
)

# ── Re-exports: figure ────────────────────────────────────────────────────
from mlsurv.viz._survival_figure import (  # noqa: F401
    _add_km_trace,
    _add_risk_table,
    _apply_km_risk_layout,
    _build_km_title_text,
    _fit_km_groups,
    _make_km_risk_figure,
)

# ── Re-exports: logrank ───────────────────────────────────────────────────
from mlsurv.viz._survival_logrank import (  # noqa: F401
    _add_logrank_annotation,
    _collect_logrank,
    _compute_logrank,
    _format_logrank_annotation,
)

# ── Re-exports: renderers ─────────────────────────────────────────────────
from mlsurv.viz._survival_renderers import (  # noqa: F401
    _add_by_group_risk_table,
    _add_maxstat_panel_annotation,
    _add_rmst_overlay,
    _annotate_pvalue,
    _apply_panel_grid_layout,
    _apply_rmst_if_requested,
    _km_combined,
    _make_km_panel_subplots,
    _plot_km_by_group,
    _plot_km_overall,
    _plot_km_panel_grid,
    _plot_km_risk,
    _render_km_panel,
    _rmst_subtitle,
    _survival_grid,
)


# =============================================================================
# Public API: plot_patient_survival_curve()
# =============================================================================

def plot_patient_survival_curve(
    pred,
    patient: int,
    *,
    time_unit: Optional[str] = None,
    width: Union[str, int] = "single",
    height: Union[str, int] = "medium",
) -> go.Figure:
    """Individualized predicted survival curve S(t|x) for one patient.

    Renders whatever ``pred`` holds: the point-estimate S(t|x) line always; a
    shaded bootstrap CI band when ``pred`` was produced with
    ``predict(ci=True, ci_band=True)``; a median-survival marker at S=0.5 when
    the median is defined; and milestone annotations ``P(t > t_i)`` when
    ``times`` were passed to ``predict()``.

    This is the shared figure builder behind both
    :meth:`mlsurv.results.PredictionResult.plot` and the survival-curve section
    of ``learner.patient_report()``.

    Parameters
    ----------
    pred : PredictionResult
        Result from ``learner.predict()``. Must have a survival function
        (``pred.survival_function is not None``).
    patient : int
        Row index into the predicted patients (0-based).
    time_unit : str, optional
        Time unit for the x-axis label and milestone captions. Falls back to
        ``pred.time_unit``; an unrecognized or absent unit renders unit-free.
    width, height : str or int
        Figure size, forwarded to :func:`create_figure` (named size or pixels).

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If ``pred`` has no survival function.
    """
    sf = pred.survival_function
    if sf is None:
        raise ValueError(
            "plot_patient_survival_curve needs a survival function, but "
            f"model '{pred.model_name}' did not produce one. Use a model that "
            "supports predict_survival_function()."
        )

    canonical_unit = _canonicalize_time_unit(
        time_unit if time_unit is not None else pred.time_unit
    )
    col = f"patient_{patient}"
    if col not in sf.columns:
        raise ValueError(
            f"patient={patient} is out of range: the survival function has "
            f"{sf.shape[1]} patient column(s) (0..{sf.shape[1] - 1})."
        )
    fig = create_figure(
        title=f"Predicted survival S(t|x) — patient {patient}",
        width=width, height=height,
    )

    # Bootstrap CI band (if available): shaded region between lower and upper
    # percentile curves. Drawn first so the point-estimate line renders on top.
    sf_ci = pred.survival_function_ci
    if sf_ci is not None and col in sf_ci["lower"].columns:
        lo = sf_ci["lower"][col]
        hi = sf_ci["upper"][col]
        ci_pct = pred._ci_pct
        # Upper boundary (no fill)
        fig.add_trace(go.Scatter(
            x=hi.index, y=hi.values, mode="lines",
            line=dict(width=0, color="rgba(30, 100, 180, 0.0)"),
            showlegend=False, hoverinfo="skip",
        ))
        # Lower boundary filled to upper
        fig.add_trace(go.Scatter(
            x=lo.index, y=lo.values, mode="lines",
            line=dict(width=0, color="rgba(30, 100, 180, 0.0)"),
            fill="tonexty", fillcolor="rgba(30, 100, 180, 0.18)",
            name=f"{ci_pct}% CI", hoverinfo="skip",
        ))

    y_point = sf[col]
    if not np.all(np.isfinite(y_point.to_numpy())):
        warnings.warn(
            f"Survival curve for patient {patient} contains non-finite "
            "values; the plotted line will have gaps.",
            SuggestionWarning, stacklevel=2,
        )
    fig.add_trace(go.Scatter(
        x=sf.index, y=y_point,
        mode="lines", name=f"Patient {patient}",
        line=dict(width=2.5),
    ))

    # Median-survival marker: horizontal reference line at S(t)=0.5 and a
    # marker dot at (median, 0.5). Skipped when median is undefined (S(t) never
    # drops below 0.5 within the observed horizon).
    if pred._median_survival is not None:
        med = float(pred._median_survival[patient])
        if np.isfinite(med):
            fig.add_hline(y=0.5, line_dash="dot",
                          line_color="#868e96", line_width=1)
            med_label = _format_time_with_unit(med, canonical_unit)
            fig.add_trace(go.Scatter(
                x=[med], y=[0.5], mode="markers",
                marker=dict(size=10, color="#000000",
                            line=dict(color="#ffffff", width=1.5)),
                name=f"Median = {med_label}",
                hovertemplate=f"Median survival: {med_label}<extra></extra>",
            ))

    # Milestone annotations from user-specified times
    if pred._user_times is not None and pred._surv_at_user_times is not None:
        for j, t in enumerate(pred._user_times):
            prob = float(pred._surv_at_user_times[patient, j])
            if not np.isfinite(prob):
                # Undefined S(t) at this horizon — skip rather than render a
                # misleading "P(t > ...) = nan%" annotation.
                continue
            fig.add_vline(x=float(t), line_dash="dash",
                          line_color="#868e96", line_width=1)
            label = _format_time_with_unit(t, canonical_unit)
            fig.add_annotation(
                x=float(t), y=prob,
                text=f"P(t > {label}) = {prob * 100:.0f}%",
                showarrow=True, arrowhead=2, ax=30, ay=-30,
            )

    fig.update_xaxes(title=_axis_title_time(canonical_unit))
    fig.update_yaxes(title="Survival probability S(t)", range=[0, 1.02])
    return fig


# =============================================================================
# Public API: plot_km()
# =============================================================================

def plot_km(
    data,
    time_col: str = "time",
    event_col: str = "event",
    group_col: Optional[str] = None,
    risk_col: Optional[str] = None,
    n_groups: Optional[int] = None,
    breaks: Optional[list] = None,
    break_type: str = "quantile",
    correction: str = "bonferroni",
    show_risk_table: Optional[bool] = None,
    show_ci: bool = True,
    time_limit: Optional[float] = None,
    time_unit: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    title: str = "Kaplan-Meier Survival",
    subtitle: Optional[str] = None,
    colors: Optional[dict] = None,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    show_pvalue: bool = False,
    return_data: bool = False,
    p_adjust: str = "holm",
    show_rmst: bool = False,
    rmst_tau: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Union[go.Figure, dict]:
    """Unified Kaplan-Meier survival visualization.

    Dispatches to the appropriate renderer based on input type:

    - ``data`` is ``DataFrame`` -> single KM plot

      - With ``group_col`` -> pre-defined groups
      - With ``risk_col`` -> quantile or custom-break risk groups
      - Neither -> overall KM

    - ``data`` is ``dict[str, DataFrame]`` -> panel grid (one subplot per model)

    Parameters
    ----------
    data : DataFrame or dict[str, DataFrame]
        Survival data. DataFrame must contain ``time_col`` and ``event_col``.
        Dict maps model names to DataFrames; each sub-DataFrame must also
        contain a risk column (defaults to ``"risk_score"``).

        **DataFrame schema** (default column names)::

            time     event      risk_score (optional for DataFrame, required for dict)
            float    bool/int   float
    time_col : str
        Column name for survival time.
    event_col : str
        Column name for event indicator (1=event, 0=censored).
    group_col : str, optional
        Column with pre-defined group labels.
    risk_col : str, optional
        Column with risk scores for grouping.
    n_groups : int, optional
        Number of risk groups. Defaults to 4 for quantile/score and 2 for
        maxstat.  For ``break_type="maxstat"``, supports 2, 3, or 4 groups
        via recursive binary splitting.
    breaks : list, optional
        Custom group boundaries. Overrides *n_groups*.

        - ``break_type="quantile"`` (default): values in (0, 1).
          E.g. ``[0.25, 0.5, 0.75]`` -> four groups at 0-25%, 25-50%,
          50-75%, 75-100% of the risk-score distribution.
        - ``break_type="score"``: raw risk-score cut-points.
          E.g. ``[3, 20, 35]`` -> four groups ``< 3``, ``3-20``,
          ``20-35``, ``>= 35``.
    break_type : {"quantile", "score", "maxstat"}
        Interpretation of *breaks*.  ``"maxstat"`` finds the cutpoint that
        maximally separates survival curves via the log-rank test (ignores
        *breaks*).  With ``n_groups > 2``, recursively splits the subpopulation
        with the highest separation.
    correction : {"bonferroni", "lausen-schumacher", "permutation"}
        Multiple-testing correction for maxstat p-values.
        Only used when ``break_type="maxstat"``.
    show_risk_table : bool, optional
        Show number-at-risk table below each KM curve.
        Defaults to ``True`` for single-model plots and ``False`` for panel grids.
    show_ci : bool
        Show 95% confidence intervals.
    time_limit : float, optional
        Clips the KM fitting at this time (default: max observed time).
        Patients with time > ``time_limit`` are censored at ``time_limit``.
    time_unit : str, optional
        Unit label appended to the x-axis title.
        E.g. ``"months"`` -> ``"Time (months)"``.
    xlim : tuple of (float, float), optional
        Display range for the x-axis as ``(xmin, xmax)``.
        Does **not** affect KM fitting -- use ``time_limit`` for that.
        E.g. ``xlim=(0, 12)`` shows only the first 12 time units.
        Risk-table timepoints are sampled within this range when set.
    ylim : tuple of (float, float), optional
        Display range for the y-axis. Defaults to ``(0, 1.05)``.
    title : str
        Plot title.
    subtitle : str, optional
        Secondary line shown below the title (rendered as smaller text).
        Defaults to a description of the stratification method for
        risk-stratified plots.  Pass ``""`` to suppress.
    colors : dict, optional
        ``{group_name: color}`` mapping.
    n_permutations : int
        Number of permutations when ``correction="permutation"``.
        Must be >= 100.
    random_state : int, optional
        Seed for permutation reproducibility.
    show_pvalue : bool
        When ``True``, annotate the plot with the omnibus log-rank p-value
        comparing survival across groups.  Silently ignored for overall KM
        (no groups to compare).
    return_data : bool
        When ``True``, return a dict with ``"figure"``, ``"logrank"``,
        and (when ``show_rmst=True``) ``"rmst"`` keys instead of the
        bare figure.  For dict input (panel grid), ``"logrank"`` maps
        model names to per-model log-rank results.
    p_adjust : str
        Multiplicity correction for pairwise log-rank tests when there
        are 3+ groups.  Options: ``"holm"``, ``"bh"``, ``"bonferroni"``,
        ``"none"``.
    show_rmst : bool, default=False
        When ``True`` and ``risk_col`` or ``group_col`` is set, compute
        Restricted Mean Survival Time (RMST) per group, shade the area
        under each KM curve up to ``rmst_tau``, and draw a vertical line
        at tau. For 2 groups, annotates the difference + p-value.
        For 3+ groups, shows per-group RMST values (pairwise comparisons
        available via ``return_data=True``).
    rmst_tau : float, optional
        Time horizon for RMST. Defaults to the minimum of maximum
        observed times across groups.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.

    Returns
    -------
    go.Figure or dict
        Figure when ``return_data=False``; dict with ``"figure"``,
        ``"logrank"``, and optionally ``"rmst"`` keys when
        ``return_data=True``.
    """
    # Flag any user parameter discarded by the grouping precedence.
    # stacklevel=3: warn -> helper -> plot_km -> user call site.
    _warn_breaks_precedence(n_groups, breaks, break_type, stacklevel=3)

    # Resolve n_groups default based on break_type
    if n_groups is None:
        n_groups = 2 if break_type == "maxstat" else 4

    _validate_km_inputs(data, time_col, event_col, group_col, risk_col,
                        break_type, correction, n_groups, n_permutations,
                        p_adjust)
    _validate_breaks(breaks, break_type)

    _rmst_active, subtitle = _resolve_km_defaults(
        data, risk_col, group_col, break_type, show_rmst, subtitle,
    )

    _logrank_out = [] if (show_pvalue or return_data) else None
    _rmst_out = [] if (_rmst_active or return_data) else None

    fig = _dispatch_km(
        data, time_col, event_col, group_col, risk_col, n_groups, breaks,
        break_type, correction, show_ci, time_limit, time_unit, xlim, ylim,
        title, subtitle, colors, n_permutations, random_state, show_pvalue,
        p_adjust, _rmst_active, rmst_tau, _logrank_out, _rmst_out,
        show_risk_table,
    )

    apply_size_overrides(fig, width, height)
    return _assemble_km_result(
        fig, data, _logrank_out, _rmst_out, return_data,
    )


def _assemble_km_result(fig, data, _logrank_out, _rmst_out, return_data):
    """Extract log-rank/RMST results and build the return value."""
    if not return_data:
        return fig

    logrank_result = None
    if _logrank_out:
        logrank_result = (dict(_logrank_out) if isinstance(data, dict)
                          else _logrank_out[0][1])

    rmst_result = _rmst_out[0] if _rmst_out else None
    return {"figure": fig, "logrank": logrank_result, "rmst": rmst_result}


def _dispatch_km(
    data, time_col, event_col, group_col, risk_col, n_groups, breaks,
    break_type, correction, show_ci, time_limit, time_unit, xlim, ylim,
    title, subtitle, colors, n_permutations, random_state, show_pvalue,
    p_adjust, _rmst_active, rmst_tau, _logrank_out, _rmst_out,
    show_risk_table,
):
    """Dispatch to the appropriate KM renderer based on input type."""
    if isinstance(data, dict):
        _show_rt = show_risk_table if show_risk_table is not None else False
        return _plot_km_panel_grid(
            data, n_groups=n_groups, time_col=time_col,
            event_col=event_col, risk_col=risk_col or "risk_score",
            time_limit=time_limit, show_ci=show_ci, title=title,
            breaks=breaks, break_type=break_type, correction=correction,
            time_unit=time_unit, xlim=xlim, ylim=ylim,
            show_risk_table=_show_rt, subtitle=subtitle,
            n_permutations=n_permutations, random_state=random_state,
            show_pvalue=show_pvalue, p_adjust=p_adjust,
            _logrank_out=_logrank_out,
        )

    if risk_col is not None:
        _show_rt = show_risk_table if show_risk_table is not None else True
        return _plot_km_risk(
            data, time_col=time_col, event_col=event_col,
            risk_col=risk_col, n_groups=n_groups, time_limit=time_limit,
            show_risk_table=_show_rt, show_ci=show_ci, title=title,
            breaks=breaks, break_type=break_type, correction=correction,
            time_unit=time_unit, xlim=xlim, ylim=ylim, subtitle=subtitle,
            n_permutations=n_permutations, random_state=random_state,
            show_pvalue=show_pvalue, p_adjust=p_adjust,
            _logrank_out=_logrank_out,
            show_rmst=_rmst_active, rmst_tau=rmst_tau,
            _rmst_out=_rmst_out,
        )

    if group_col is not None:
        _show_rt = show_risk_table if show_risk_table is not None else False
        return _plot_km_by_group(
            data, time_col=time_col, event_col=event_col,
            group_col=group_col, time_limit=time_limit,
            show_ci=show_ci, title=title, colors=colors,
            time_unit=time_unit, xlim=xlim, ylim=ylim,
            show_pvalue=show_pvalue, p_adjust=p_adjust,
            _logrank_out=_logrank_out,
            show_rmst=_rmst_active, rmst_tau=rmst_tau,
            _rmst_out=_rmst_out,
            show_risk_table=_show_rt, subtitle=subtitle,
        )

    if show_pvalue:
        warnings.warn(
            "show_pvalue=True has no effect for single-group KM plots. "
            "Provide group_col or risk_col to enable log-rank testing.",
            SuggestionWarning,
            stacklevel=3,
        )
    return _plot_km_overall(
        data, time_col=time_col, event_col=event_col,
        time_limit=time_limit, show_ci=show_ci, title=title,
        time_unit=time_unit, xlim=xlim, ylim=ylim, subtitle=subtitle,
    )
