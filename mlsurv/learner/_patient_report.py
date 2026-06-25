"""PatientReportMixin — single-patient HTML report.

Produces a standalone, interactive-Plotly HTML page summarizing one patient's
survival prediction: narrative prose, individualized S(t) curve, risk-cohort
context, SHAP waterfall, feature table with imputation flags, and a
similar-risk training-cohort anchor.

The report is descriptive of the model's output, not prescriptive — no
treatment suggestions or causal claims.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from ..viz._time_format import _axis_title_time, _canonicalize_time_unit

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


class PatientReportMixin:
    """Mixin providing ``patient_report()`` for single-patient HTML reports."""

    def patient_report(
        self: SurvivalLearner,
        X: pd.DataFrame,
        patient: Union[int, str] = 0,
        model: Optional[str] = None,
        output_path: Optional[str] = None,
        times: Optional[Union[list, np.ndarray]] = None,
        time_unit: Optional[str] = None,
        n_neighbors: int = 10,
        ci: bool = True,
        ci_band: bool = True,
        n_bootstrap: int = 200,
        offline: bool = False,
        random_state: Optional[int] = None,
        use_recalibrated: bool = True,
    ) -> str:
        """Generate an HTML report for a single patient's prediction.

        Parameters
        ----------
        X : DataFrame
            Feature matrix containing one or more patients.  The report
            focuses on row ``patient``; other rows are used only to make
            the ``predict()`` call batchable.
        patient : int or str, default 0
            Row index in ``X`` to report on, or a risk-percentile spec
            evaluated against the resolved model's predictions on ``X``:
            ``'low_risk'`` (= ``'p10'``), ``'high_risk'`` (= ``'p90'``),
            ``'median'`` (= ``'p50'``), or ``'pNN'`` for any NN in 0..100.
        model : str, optional
            Name of the model to use.  Required only when multiple models
            are trained.
        output_path : str, optional
            If given, writes the HTML to this path.  The full HTML string
            is always returned.
        times : list, optional
            Milestone times for P(survive > t) annotations. If ``None``,
            defaults are derived from the resolved time unit: years →
            ``[1, 3, 5]``; months → ``[12, 36, 60]``; weeks →
            ``[52, 156, 260]``; days → ``[365, 1095, 1825]``. If the unit is
            unknown (neither *time_unit* nor ``learner.setup(time_unit=...)``
            was set), milestone markers are omitted and the survival curve
            still renders.
        time_unit : str, optional
            Time unit of the outcome (``'years'``, ``'months'``, ``'weeks'``,
            ``'days'``; plural/singular accepted). Used to auto-default
            *times*, label the x-axis, and pluralize milestone captions.
            Falls back to ``learner._state.time_unit`` (from ``setup()``);
            if neither is set, the axis is labeled ``"Time"`` and no
            milestones are drawn.
        n_neighbors : int, default 10
            K for the nearest-neighbor training-cohort anchor.
        ci : bool, default True
            If True, compute bootstrap CIs on median survival / RMST /
            survival at ``times``.  Set False for faster runs.
        ci_band : bool, default True
            If True (and ``ci=True``), also shade a bootstrap CI band
            around the individualized S(t) curve. Adds
            ``n_bootstrap × n_times`` floats to the result; trivial for a
            single-patient report, hence on by default. When the model has a
            calibrator (and ``use_recalibrated=True``) the band is on the
            calibrated scale; see :meth:`predict` for the warper-vs-breslow
            band-uncertainty caveat.
        n_bootstrap : int, default 200
            Bootstrap iterations when ``ci=True``.
        offline : bool, default False
            If True, embed plotly.js into every figure div (larger file,
            works without internet).  If False (default), load plotly.js
            once from CDN.
        random_state : int, optional
            Seed for bootstrap reproducibility.
        use_recalibrated : bool, default True
            When ``True`` and a calibrator has been fit for the selected
            model via ``train(calibrate=...)``, all ``S(t|x)`` consumers
            (survival curve, CI band, milestone probabilities, KNN tables)
            display the recalibrated curve and an info banner is rendered.
            Risk score and feature attributions are unchanged. Pass
            ``False`` to inspect the raw model output.

        Returns
        -------
        str
            Full HTML document.
        """
        from ._patient_report_template import render_html
        from ._patient_resolve import resolve_patient_index

        self._ensure_setup()
        # Resolve the model up front so that the same pipeline is used for
        # both the risk-percentile lookup (if `patient` is a string) and the
        # downstream predict/explain calls below. Passing the resolved
        # `model_name` through keeps the storage-key collision rules
        # honored.
        resolved_model = self._resolve_single_model(model)
        model = resolved_model.name
        patient = resolve_patient_index(patient, X, resolved_model.model)

        # Resolve time unit (call-site arg > learner setup > None)
        resolved_unit = _canonicalize_time_unit(
            time_unit if time_unit is not None else self._state.time_unit
        )
        if times is None:
            times = _default_clinical_times(resolved_unit)

        # -- Core predictions ------------------------------------------------
        pred = self.predict(
            X, models=model, times=times, ci=ci, ci_band=ci_band and ci,
            n_bootstrap=n_bootstrap, random_state=random_state,
            verbose=False, use_recalibrated=use_recalibrated,
        )
        # If the caller passed time_unit= override, propagate the canonical
        # form onto pred so summary() uses it too (predict() attached the
        # learner-level value, which may be None or non-canonical).
        pred.time_unit = resolved_unit or pred.time_unit
        model_name = pred.model_name

        # explain() requires exactly one row
        expl = self.explain(
            X=X.iloc[[patient]], model=model_name, plot=True,
        )

        # -- Build figures ---------------------------------------------------
        include_js = _plotly_include(offline, is_first=True)
        subsequent_js = _plotly_include(offline, is_first=False)

        surv_html = _survival_curve_html(
            pred, patient, times, include_js, resolved_unit,
        ) + _ci_methodology_html(pred)
        hist_html = _risk_histogram_html(pred, patient, subsequent_js)
        km_html = _km_overlay_html(
            pred, patient, subsequent_js, resolved_unit,
        )
        shap_html = _shap_waterfall_html(expl, subsequent_js)

        # -- Build tables + narrative pieces ---------------------------------
        narrative = pred.summary(patient)
        stat_grid = _stat_grid_items(pred, patient, resolved_unit)
        feature_table = _feature_table_html(pred, expl, patient)
        knn_table = _knn_table_html(pred, patient, n_neighbors)
        alerts = _alerts_list(pred, patient)
        recal_notices = _recalibration_notices(
            self, model_name, use_recalibrated,
        )
        calibration_panel = _calibration_panel_html(
            self, model_name, times, X, patient,
        )

        provenance = pred._provenance_line() or f"Model: {model_name}"
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = f"Patient {patient} — Survival Prediction Report"

        footer_meta = _footer_meta_html(pred)
        disclaimer = (
            "This report is model output intended to support clinical "
            "interpretation. It is not a substitute for clinical judgment, "
            "does not estimate intervention effects, and does not predict "
            "individual outcomes with certainty."
        )

        html = render_html(
            title=title,
            provenance=provenance,
            date=date_str,
            narrative=narrative,
            stat_grid_items=stat_grid,
            survival_curve_html=surv_html,
            risk_histogram_html=hist_html,
            km_overlay_html=km_html,
            shap_waterfall_html=shap_html,
            feature_table_html=feature_table,
            knn_table_html=knn_table,
            alerts=alerts,
            footer_meta=footer_meta,
            disclaimer=disclaimer,
            recalibration_notices=recal_notices,
            calibration_panel_html=calibration_panel,
            plotlyjs_include="inline" if offline else "cdn",
        )

        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html


# =============================================================================
# Module-level helpers (figure + HTML builders)
# =============================================================================

_DEFAULT_CLINICAL_TIMES = {
    "years": [1, 3, 5],
    "months": [12, 36, 60],
    "weeks": [52, 156, 260],
    "days": [365, 1095, 1825],
}


def _default_clinical_times(canonical_unit):
    """Clinically relevant default milestone times for the canonical unit.

    Returns ``None`` when the unit is unknown — callers should omit
    milestone annotations rather than invent time points of unclear meaning.
    """
    if canonical_unit is None:
        return None
    return list(_DEFAULT_CLINICAL_TIMES[canonical_unit])


def _plotly_include(offline: bool, is_first: bool):
    """Plotly.js include flag to pass to ``fig.to_html``.

    - offline=True: embed the full JS into the first figure only, use
      ``False`` thereafter so it isn't duplicated across sections.
    - offline=False: rely on a single CDN ``<script>`` tag in the page
      template; every figure uses ``False`` to avoid embedding JS.
    """
    if not offline:
        return False
    return True if is_first else False


def _aria_figure(fig_html: str, label: str) -> str:
    """Wrap a patient-report figure with an accessible name for screen readers.

    ``role="img"`` plus a non-empty ``aria-label`` lets assistive tech announce
    the chart (which is otherwise an unlabeled ``<canvas>``/``<svg>``). The
    patient-report figures are all static, so ``role="img"`` is correct here.
    """
    return f'<div role="img" aria-label="{escape(label)}">{fig_html}</div>'


def _survival_curve_html(
    pred, patient: int, times, include_js, canonical_unit,
) -> str:
    """Individualized S(t|x) curve with milestone markers (HTML section).

    Delegates figure-building to
    :func:`mlsurv.viz.survival.plot_patient_survival_curve` (shared with
    ``PredictionResult.plot()``) and wraps the result as an ARIA-labeled,
    embeddable HTML div. The report uses the wider ``"double"`` layout.
    """
    from ..viz.survival import plot_patient_survival_curve

    if pred.survival_function is None:
        return "<p><em>Survival function not available for this model.</em></p>"

    fig = plot_patient_survival_curve(
        pred, patient, time_unit=canonical_unit,
        width="double", height="medium",
    )
    return _aria_figure(
        fig.to_html(full_html=False, include_plotlyjs=include_js),
        "Individualized predicted survival curve",
    )


def _n_bootstrap_iters(pred) -> Optional[int]:
    """Iteration count inferred from any stored bootstrap-draw array."""
    for arr in (pred._bootstrap_medians,
                pred._bootstrap_surv_probs,
                pred._bootstrap_rmst):
        if arr is not None:
            return int(arr.shape[0])
    return None


def _ci_methodology_html(pred) -> str:
    """One-paragraph explanation of how the report's CIs are produced.

    Returns an empty string when no bootstrap was run (``ci=False``), so the
    section stays clean for fast non-bootstrap reports.
    """
    if pred._bootstrap_medians is None:
        return ""
    ci_pct = pred._ci_pct
    n_boot = _n_bootstrap_iters(pred)
    n_str = (
        f"{n_boot} bootstrap iterations"
        if n_boot is not None else "bootstrap resampling"
    )
    alpha = (1 - pred._ci_level) / 2
    # The shaded curve band only renders when ci_band=True (predict()
    # populates _bootstrap_surv_probs, surfaced via survival_function_ci).
    # The [lo-hi] brackets render whenever ci=True. Adjust the lead so
    # the prose doesn't reference a band that isn't on the plot.
    has_band = pred.survival_function_ci is not None
    lead = (
        "Shaded band and [lo&ndash;hi] brackets show"
        if has_band else "[lo&ndash;hi] brackets show"
    )
    return (
        '<p class="methodology"><em><small>'
        f'{lead} {ci_pct}% '
        f'percentile-bootstrap confidence intervals from {n_str}: the '
        'training set is resampled with replacement, the full preprocessing '
        "pipeline and model are refit on each resample, and the patient's "
        'predictions are recomputed. Bounds are the empirical '
        f'{alpha:.3f} and {1 - alpha:.3f} quantiles across resamples. '
        'These reflect uncertainty from training-data variation, not '
        'patient-specific (aleatoric) noise.'
        '</small></em></p>'
    )


def _risk_histogram_html(pred, patient: int, include_js) -> str:
    """Training-cohort risk-score histogram with patient marker."""
    from ..viz._theme import create_figure
    import plotly.graph_objects as go

    if pred.training_risk_scores is None:
        return "<p><em>Training cohort risk distribution unavailable.</em></p>"

    patient_score = float(pred.risk_scores[patient])
    pct = pred.risk_percentile(patient)
    pct_label = f"{pct:.0f}th percentile" if pct is not None else ""

    fig = create_figure(
        title="Patient risk vs training cohort",
        width="single", height="medium",
    )
    fig.add_trace(go.Histogram(
        x=pred.training_risk_scores, name="Training cohort",
        nbinsx=30, marker_color="#adb5bd",
    ))
    fig.add_vline(x=patient_score, line_color="#d6336c", line_width=3)
    # Anchor the label to the top-left or top-right of the plot (in paper
    # coordinates) so it never collides with the histogram bars. Pick the
    # corner opposite to where the patient sits in the training distribution:
    # left-half patient -> top-right box; right-half patient -> top-left box.
    lo, hi = float(np.min(pred.training_risk_scores)), float(np.max(pred.training_risk_scores))
    midpoint = (lo + hi) / 2 if hi > lo else patient_score
    if patient_score <= midpoint:
        box_x, xanchor = 0.98, "right"
    else:
        box_x, xanchor = 0.02, "left"
    fig.add_annotation(
        x=box_x, y=0.98, xref="paper", yref="paper",
        xanchor=xanchor, yanchor="top",
        text=f"Patient {patient}<br>{pct_label}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#d6336c",
        borderwidth=1, borderpad=6,
    )
    fig.update_xaxes(title="Risk score (higher = higher risk)")
    fig.update_yaxes(title="Count")
    fig.update_layout(margin=dict(t=70))
    return _aria_figure(
        fig.to_html(full_html=False, include_plotlyjs=include_js),
        "Cohort risk-score distribution",
    )


def _km_overlay_html(pred, patient: int, include_js, canonical_unit) -> str:
    """KM curves by training risk-tertile, with patient's S(t) overlaid."""
    from ..viz._theme import create_figure
    import plotly.graph_objects as go

    if (pred.training_risk_scores is None or pred.training_y is None):
        return "<p><em>KM overlay unavailable (no training cohort data).</em></p>"

    try:
        y = pred.training_y
        times_arr = np.asarray(y["time"])
        events_arr = np.asarray(y["event"]).astype(bool)
    except (ValueError, KeyError, IndexError):
        return "<p><em>KM overlay unavailable (y_train not parseable).</em></p>"

    fig = create_figure(
        title="Training cohort by risk quartile (patient curve overlaid)",
        width="single", height="medium",
    )

    # Partition training patients into risk quartiles and plot each KM.
    # Matches the risk stratification convention used by the main report and
    # learner.plot_survival (via get_risk_colors(n_groups=4)).
    from ..viz._palettes import get_risk_colors
    risk = pred.training_risk_scores
    q1, q2, q3 = np.quantile(risk, [0.25, 0.5, 0.75])
    group_names = [
        "Q1: low risk (bottom 25%)",
        "Q2: 25–50%",
        "Q3: 50–75%",
        "Q4: high risk (top 25%)",
    ]
    masks = [
        risk <= q1,
        (risk > q1) & (risk <= q2),
        (risk > q2) & (risk <= q3),
        risk > q3,
    ]
    colors = get_risk_colors(n_groups=4)

    for mask, name, color in zip(masks, group_names, colors):
        if not mask.any():
            continue
        t_km, s_km = _km_estimate(times_arr[mask], events_arr[mask])
        fig.add_trace(go.Scatter(
            x=t_km, y=s_km, mode="lines",
            name=name, line=dict(color=color, width=1.8, shape="hv"),
        ))

    # Overlay the patient's predicted S(t) if available
    sf = pred.survival_function
    if sf is not None:
        col = f"patient_{patient}"
        fig.add_trace(go.Scatter(
            x=sf.index, y=sf[col], mode="lines",
            name=f"Patient {patient} (predicted)",
            line=dict(color="#000000", width=3, dash="dot"),
        ))

    fig.update_xaxes(title=_axis_title_time(canonical_unit))
    fig.update_yaxes(title="Survival probability", range=[0, 1.02])
    return _aria_figure(
        fig.to_html(full_html=False, include_plotlyjs=include_js),
        "Kaplan-Meier survival overlay",
    )


def _km_estimate(times: np.ndarray, events: np.ndarray):
    """Minimal Kaplan-Meier step function.

    Returns ``(time_grid, survival)`` arrays suitable for ``go.Scatter``
    with ``shape="hv"``.  Uses scikit-survival when available; falls back
    to a pure-numpy implementation.
    """
    try:
        from sksurv.nonparametric import kaplan_meier_estimator
        t, s = kaplan_meier_estimator(events, times)
        return t, s
    except ImportError:
        # Pure-numpy fallback
        idx = np.argsort(times, kind="mergesort")
        t_sorted = times[idx]
        e_sorted = events[idx]
        unique_times, inverse = np.unique(t_sorted, return_inverse=True)
        n_at_risk = len(t_sorted) - np.searchsorted(t_sorted, unique_times)
        events_at_t = np.zeros_like(unique_times, dtype=float)
        for k, e in zip(inverse, e_sorted):
            if e:
                events_at_t[k] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            surv_factors = 1.0 - events_at_t / n_at_risk
        surv = np.cumprod(np.where(np.isfinite(surv_factors), surv_factors, 1.0))
        return unique_times, surv


def _shap_waterfall_html(expl, include_js) -> str:
    """Embed SHAP waterfall figure from ``learner.explain()``."""
    fig = getattr(expl, "figure", None)
    if fig is None:
        return "<p><em>SHAP waterfall unavailable.</em></p>"
    return _aria_figure(
        fig.to_html(full_html=False, include_plotlyjs=include_js),
        "SHAP feature contribution waterfall",
    )


def _footer_meta_html(pred) -> str:
    """Full model metadata block for the footer."""
    parts = [f"<strong>Model:</strong> {escape(pred.model_name)}"]
    if pred.training_cohort_size is not None:
        parts.append(f"Training cohort: N={pred.training_cohort_size}")
    if pred.test_c_index is not None:
        parts.append(f"Test C-index: {pred.test_c_index:.3f}")
    if pred.feature_names is not None:
        parts.append(f"Features: {len(pred.feature_names)}")
    return "<div>" + " &nbsp;·&nbsp; ".join(parts) + "</div>"


# Re-export pure HTML/data builders from the split module so that
# patient_report() can call them by bare name.
from ._patient_report_html import (  # noqa: E402, F401
    _alerts_list,
    _calibration_panel_html,
    _feature_table_html,
    _knn_table_html,
    _recalibration_notices,
    _stat_grid_items,
)
