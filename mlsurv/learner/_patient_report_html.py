"""Pure HTML/data builders used by ``PatientReportMixin.patient_report``.

These are stateless module-level helpers — no ``self``, no figure rendering.
They were split out of ``_patient_report.py`` to isolate the high-CC
``_feature_table_html`` and shrink the mixin file.
"""

from __future__ import annotations

from html import escape
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from ..viz._time_format import _format_time_with_unit


def _safe_float_str(value) -> tuple[Optional[float], str]:
    """Coerce ``value`` to float for both numeric use and display.

    Returns ``(numeric, display)``: ``numeric`` is the float when the
    coercion succeeds (``None`` otherwise), and ``display`` is either
    ``f"{numeric:g}"`` or the HTML-escaped string repr on failure.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None, escape(str(value))
    if not np.isfinite(num):
        # NaN/inf coerce to float but render meaninglessly ("nan", "inf").
        # Treat as missing: downstream OOD/percentile branches already handle
        # None, and the em dash matches the calibration table's missing-cell
        # convention.
        return None, "—"
    return num, f"{num:g}"


def _resolve_patient_value(
    feat, feat_val, input_X, patient: int,
) -> tuple[Optional[float], str]:
    """Three-way fallback for a single feature's patient value.

    Priority: raw ``input_X`` row -> SHAP-frame ``feature_value`` -> empty.
    When the ``input_X`` cell is non-coercible (e.g. a category string),
    the display falls back to ``feat_val`` from the SHAP frame to preserve
    pre-refactor semantics — the numeric component stays ``None`` either
    way.
    """
    if input_X is not None and feat in input_X.columns:
        raw = input_X.iloc[patient][feat]
        num, disp = _safe_float_str(raw)
        # Non-coercible raw (e.g. a category string) falls back to the SHAP
        # frame's ``feat_val``. Skip when raw is itself missing — otherwise a
        # NaN feat_val would re-render as the literal "nan". Routing feat_val
        # through ``_safe_float_str`` also normalizes a NaN feat_val to the
        # em-dash sentinel.
        if num is None and not pd.isna(raw) and feat_val is not None:
            _, disp = _safe_float_str(feat_val)
        return num, disp
    if feat_val is not None:
        return _safe_float_str(feat_val)
    return None, ""


def _train_percentile_cell(
    pval_num: Optional[float], lo, hi, training_X, feat,
) -> str:
    """Render the ``Train percentile [range]`` cell."""
    if lo is None or hi is None:
        return ""
    range_part = f"[{lo:g}, {hi:g}]"
    if (
        pval_num is None
        or not np.isfinite(pval_num)
        or training_X is None
        or feat not in training_X.columns
    ):
        return range_part
    try:
        col = training_X[feat].to_numpy(dtype=float)
        pct = float(percentileofscore(col, pval_num, kind="mean"))
    except (TypeError, ValueError):
        return range_part
    if not np.isfinite(pct):
        return range_part
    return f"{pct:.0f}% {range_part}"


def _direction_html(shap: float) -> str:
    """SHAP direction marker. Explicit ``> 0`` / ``< 0`` checks (never
    truthy) so a SHAP value of ``0.0`` renders as the empty marker, not
    accidentally as a downward arrow."""
    if shap > 0:
        return '<span class="direction-up">↑ risk</span>'
    if shap < 0:
        return '<span class="direction-down">↓ risk</span>'
    return ""


def _is_ood(pval_num: Optional[float], lo, hi) -> bool:
    """OOD only when value AND both bounds are known; missing data is
    treated as in-range (returns ``False``)."""
    if pval_num is None or lo is None or hi is None:
        return False
    return pval_num < lo or pval_num > hi


def _imputed_for(impute_mask, feat, patient: int) -> bool:
    """True when ``feat`` was imputed for ``patient`` (mask absent → False)."""
    if impute_mask is None or feat not in impute_mask.columns:
        return False
    return bool(impute_mask.iloc[patient][feat])


def _stat_grid_items(pred, patient: int, canonical_unit) -> list:
    """Key numeric predictions formatted for the stat grid."""
    unit_suffix = f" {canonical_unit}" if canonical_unit else ""
    risk_val = f"{pred.risk_scores[patient]:.3f}"
    if pred.training_risk_scores is not None:
        lo = float(np.min(pred.training_risk_scores))
        hi = float(np.max(pred.training_risk_scores))
        risk_val = f"{risk_val} (training range: {lo:.2f} to {hi:.2f})"
    items = [("Risk score", risk_val)]

    pct = pred.risk_percentile(patient)
    if pct is not None:
        items.append(("Risk percentile",
                      f"{int(round(pct))}% of training cohort"))

    if pred._median_survival is not None:
        med = float(pred._median_survival[patient])
        if np.isnan(med):
            items.append(("Median survival",
                          "Not reached (S(t) > 0.5 throughout follow-up)"))
        else:
            ci = pred.median_survival_ci
            if ci is not None:
                lo = ci["ci_lower"].iloc[patient]
                hi = ci["ci_upper"].iloc[patient]
                items.append(("Median survival",
                              f"{med:.1f} [{lo:.1f}–{hi:.1f}]{unit_suffix}"))
            else:
                items.append(("Median survival", f"{med:.1f}{unit_suffix}"))

    if pred._user_times is not None and pred._surv_at_user_times is not None:
        from ..results._prediction import _time_col
        sat_ci = pred.survival_at_times_ci
        for j, t in enumerate(pred._user_times):
            prob = pred._surv_at_user_times[patient, j]
            label = _format_time_with_unit(t, canonical_unit)
            value = f"{prob * 100:.0f}%"
            if sat_ci is not None:
                col = _time_col("t", t)
                lo = sat_ci[f"{col}_ci_lower"].iloc[patient]
                hi = sat_ci[f"{col}_ci_upper"].iloc[patient]
                value = f"{value} [{lo * 100:.0f}%–{hi * 100:.0f}%]"
            items.append((f"P(survive > {label})", value))

    return items


def _feature_table_html(pred, expl, patient: int) -> str:
    """Feature contribution table: patient value, train percentile + range,
    SHAP magnitude, direction, imputation yes/no."""
    shap_df = expl.shap_values
    if shap_df is None or shap_df.empty:
        return ""

    shap_df = shap_df.copy()
    if "abs_shap_value" in shap_df.columns:
        # Hide exactly-zero contributions (e.g. selector-dropped raw features)
        # so they don't crowd out real contributors in the top-15 table.  They
        # stay in expl.shap_values for cohort reconciliation; only the display
        # drops them.
        nonzero = shap_df[shap_df["abs_shap_value"] != 0]
        if not nonzero.empty:
            shap_df = nonzero
        shap_df = shap_df.sort_values("abs_shap_value", ascending=False)
    top = shap_df.head(15)

    bounds = pred.feature_bounds or {}
    input_X = pred.input_X
    training_X = pred.training_X
    impute_mask = pred.imputation_mask

    rows = ["<thead><tr>"
            "<th>Feature</th>"
            "<th>Patient value</th>"
            "<th>Train percentile [range]</th>"
            "<th>SHAP</th>"
            "<th>Direction</th>"
            "<th>Imputed</th>"
            "</tr></thead><tbody>"]

    for _, row in top.iterrows():
        feat = row.get("feature")
        shap = float(row.get("shap_value", 0.0))
        pval_num, feat_val_str = _resolve_patient_value(
            feat, row.get("feature_value"), input_X, patient,
        )
        lo, hi = bounds.get(feat, (None, None))
        pct_range_str = _train_percentile_cell(
            pval_num, lo, hi, training_X, feat,
        )
        cls = ' class="ood"' if _is_ood(pval_num, lo, hi) else ""
        imputed = _imputed_for(impute_mask, feat, patient)
        rows.append(
            f"<tr{cls}>"
            f"<td>{escape(str(feat))}</td>"
            f"<td>{feat_val_str}</td>"
            f"<td>{pct_range_str}</td>"
            f"<td>{shap:+.3f}</td>"
            f"<td>{_direction_html(shap)}</td>"
            f"<td>{'yes' if imputed else 'no'}</td>"
            "</tr>"
        )
    rows.append("</tbody>")
    return "<table>" + "".join(rows) + "</table>"


def _knn_table_html(pred, patient: int, k: int) -> Optional[str]:
    """Two tables of nearest training patients: by predicted risk score
    (model-output space) and by raw feature distance (feature space)."""
    risk_knn = pred.nearest_neighbors(patient, k=k)
    feat_knn = pred.nearest_neighbors_feature_space(patient, k=k)
    if (risk_knn is None or risk_knn.empty) and (
        feat_knn is None or feat_knn.empty
    ):
        return None

    html_parts = []

    def _time_cell(row) -> str:
        time_val = row.get("time")
        if time_val is None or pd.isna(time_val):
            return ""
        s = f"{float(time_val):.1f}"
        event_val = row.get("event")
        if event_val is not None and not pd.isna(event_val) and not bool(event_val):
            s = f"{s} (censored)"
        return s

    if risk_knn is not None and not risk_knn.empty:
        header = (
            "<thead><tr>"
            "<th>Training patient</th>"
            "<th>Risk score</th>"
            "<th>Observed time</th>"
            "</tr></thead><tbody>"
        )
        body = []
        for _, row in risk_knn.iterrows():
            idx = int(row["train_index"])
            risk = float(row["risk_score"])
            body.append(
                f"<tr><td>#{idx}</td><td>{risk:.3f}</td>"
                f"<td>{_time_cell(row)}</td></tr>"
            )
        body.append("</tbody>")
        html_parts.append(
            "<h3>Similar predicted risk</h3>"
            "<p><em>Training patients with the closest predicted risk "
            "score.</em></p>"
            "<table>" + header + "".join(body) + "</table>"
        )

    if feat_knn is not None and not feat_knn.empty:
        header = (
            "<thead><tr>"
            "<th>Training patient</th>"
            "<th>Distance (z-score)</th>"
            "<th>Risk score</th>"
            "<th>Observed time</th>"
            "</tr></thead><tbody>"
        )
        body = []
        for _, row in feat_knn.iterrows():
            idx = int(row["train_index"])
            dist = float(row["distance"])
            risk_val = row.get("risk_score")
            risk_str = (
                f"{float(risk_val):.3f}"
                if risk_val is not None and not pd.isna(risk_val) else ""
            )
            body.append(
                f"<tr><td>#{idx}</td><td>{dist:.2f}</td>"
                f"<td>{risk_str}</td>"
                f"<td>{_time_cell(row)}</td></tr>"
            )
        body.append("</tbody>")
        html_parts.append(
            "<h3>Similar feature values</h3>"
            "<p><em>Training patients with the closest feature values "
            "(Euclidean distance in z-scored feature space).</em></p>"
            "<table>" + header + "".join(body) + "</table>"
        )

    if not html_parts:
        return None
    footnote = (
        "<p><em><small>Censored: the patient did not experience the event "
        "during follow-up; the time shown is the last observed time, not a "
        "time-to-event.</small></em></p>"
    )
    html_parts.append(footnote)
    return "".join(html_parts)


def _recalibration_notices(
    learner, model_name: str, use_recalibrated: bool,
) -> list:
    """Banner notices describing the report's calibration status.

    Returns 0 or 1 notices:

    - Banner A: selected model is shown with its calibrated S(t|x).
    - Banner B: selected model is shown raw, but other models on the
      learner have a calibrator attached.

    Both banners are about the report's *content*, not warnings per se,
    so the template renders them above (not inside) the amber alerts box.
    """
    any_calibrated = any(
        learner._has_calibrator(n) for n in learner._state.models
    )
    if not any_calibrated:
        return []

    has_for_selected = learner._has_calibrator(model_name)
    if has_for_selected and use_recalibrated:
        result = learner._state.models.get(model_name)
        method = getattr(getattr(result, "calibrator", None), "method", "isotonic")
        return [(
            f"Survival probabilities shown have been calibrated using "
            f"{method} on the training cohort. Risk score and feature "
            f"attributions are unchanged."
        )]
    if not has_for_selected:
        return [(
            f"Predictions for {model_name} are uncalibrated; other models on "
            f"this learner have a calibrator attached."
        )]
    return []


_BAND_HALF_WIDTH_PCT = 5.0


def _calibration_panel_html(
    learner, model_name: str, horizons, X, patient_idx: int,
) -> str:
    """Patient-centered calibration: a sliding ±5 percentile band of the
    test cohort, centered on this patient's predicted ``S(t|x)`` at each
    horizon.

    For each horizon, the cohort patients whose predicted survival is
    within 5 percentile points of the focal patient's predicted survival
    form the comparison band. The observed survival rate in that band is
    compared to the focal patient's predicted survival. The same logic
    runs twice: once on the raw model output, once on the recalibrated
    output (when a recalibrator exists).

    Returns an empty string when no calibrator is attached for
    *model_name* or no internal test set is available; warns via
    ``mlsurv._warnings`` on actionable skip paths so the silent-failure
    rule is respected.
    """
    import warnings
    from mlsurv._warnings import SuggestionWarning

    if not learner._has_calibrator(model_name):
        # Expected when the selected model has no calibrator. Banner B in
        # _recalibration_notices already directs the reader, so no warning here.
        return ""
    calibrator = learner._state.models[model_name].calibrator
    X_test = learner._state.X_test
    y_test = learner._state.y_test
    if X_test is None or y_test is None:
        warnings.warn(
            f"Patient report calibration panel skipped for '{model_name}': "
            f"no internal test set on the learner. To inspect calibration, "
            f"call validate(X_external, y_external) for an external view.",
            SuggestionWarning, stacklevel=2,
        )
        return ""
    if horizons is None or len(horizons) == 0:
        # Fall back to the model's calibrator horizons
        horizons = sorted(calibrator.horizons)

    horizons_arr = np.asarray(list(horizons), dtype=float)
    pipeline = learner._state.models[model_name].model

    try:
        from ..evaluation.metrics import (
            predict_survival_functions, surv_funcs_to_probs,
        )
        # Cohort raw S(t|x) at each horizon: [n_test, n_horizons]
        cohort_surv = predict_survival_functions(
            pipeline, X_test, horizons_arr,
        )
        cohort_raw = surv_funcs_to_probs(cohort_surv, horizons_arr)
        # Patient raw S(t|x) at each horizon: [n_horizons]
        patient_surv = predict_survival_functions(
            pipeline, X.iloc[[patient_idx]], horizons_arr,
        )
        patient_raw = surv_funcs_to_probs(
            patient_surv, horizons_arr,
        ).ravel()
        # Calibrated counterparts (the deployed model's S(t|x))
        cohort_recal = learner._apply_calibrator(
            model_name, cohort_raw, horizons_arr, X=X_test,
        )
        patient_recal = learner._apply_calibrator(
            model_name, patient_raw[None, :], horizons_arr,
            X=X.iloc[[patient_idx]],
        ).ravel()
    except (ValueError, RuntimeError) as exc:
        # The panel is supplementary, but dropping it is user-facing content
        # loss — surface it at a tier the user actually sees, not InternalWarning.
        warnings.warn(
            f"Patient report calibration panel skipped for '{model_name}': "
            f"{type(exc).__name__}: {exc}",
            SuggestionWarning, stacklevel=2,
        )
        return ""

    t_test = np.asarray(y_test["time"], dtype=float)
    e_test = np.asarray(y_test["event"]).astype(bool)

    def _band_observed(cohort_at_h, patient_at_h, h):
        """Return (n_in_band, observed_rate) for the ±5 pct band around
        the patient's predicted S(h|x). ``observed_rate`` is the share of
        band patients whose follow-up exceeds *h*; band patients censored
        before *h* are excluded (same convention as the cohort calibration
        helper)."""
        finite = np.isfinite(cohort_at_h)
        if not finite.any():
            return 0, float("nan"), float("nan")
        sorted_vals = np.sort(cohort_at_h[finite])
        n = sorted_vals.size
        # Patient's percentile within the cohort (% of cohort with strictly
        # lower predicted survival).
        pct = float((sorted_vals < patient_at_h).sum()) / n * 100.0
        lo_pct = max(0.0, pct - _BAND_HALF_WIDTH_PCT)
        hi_pct = min(100.0, pct + _BAND_HALF_WIDTH_PCT)
        lo_val = float(np.quantile(sorted_vals, lo_pct / 100.0))
        hi_val = float(np.quantile(sorted_vals, hi_pct / 100.0))
        in_band = finite & (cohort_at_h >= lo_val) & (cohort_at_h <= hi_val)
        # Drop censored-before-h: their event status at h is unknown.
        usable = in_band & ~((t_test <= h) & ~e_test)
        n_used = int(usable.sum())
        if n_used == 0:
            return 0, float("nan"), pct
        observed = float((t_test[usable] > h).mean())
        return n_used, observed, pct

    rows = ["<thead><tr>"
            "<th rowspan=\"2\">Horizon</th>"
            "<th rowspan=\"2\">Patient pct</th>"
            "<th colspan=\"3\">Raw S(t|x)</th>"
            "<th colspan=\"3\">Recalibrated S(t|x)</th>"
            "<th rowspan=\"2\">n in band</th>"
            "</tr><tr>"
            "<th>Predicted</th><th>Band observed</th><th>Bias</th>"
            "<th>Predicted</th><th>Band observed</th><th>Bias</th>"
            "</tr></thead><tbody>"]

    def _fmt(value):
        if value is None or not np.isfinite(value):
            return "—"
        return f"{value:.3f}"

    def _fmt_bias(value):
        if value is None or not np.isfinite(value):
            return "<td>—</td>"
        # Threshold below which we render as zero (avoids visual noise from
        # tiny float-noise values around perfect calibration).
        if abs(value) < 5e-4:
            return f'<td class="bias-zero">{value:+.3f}</td>'
        cls = "bias-positive" if value > 0 else "bias-negative"
        return f'<td class="{cls}">{value:+.3f}</td>'

    for j, h in enumerate(horizons_arr):
        h_key = float(h)
        n_raw, obs_raw, pct_raw = _band_observed(
            cohort_raw[:, j], float(patient_raw[j]), h_key,
        )
        n_rec, obs_rec, _pct_rec = _band_observed(
            cohort_recal[:, j], float(patient_recal[j]), h_key,
        )
        bias_raw = obs_raw - float(patient_raw[j]) if np.isfinite(obs_raw) else float("nan")
        bias_rec = obs_rec - float(patient_recal[j]) if np.isfinite(obs_rec) else float("nan")
        n_display = n_raw if np.isfinite(pct_raw) else 0
        pct_display = f"{pct_raw:.0f}%" if np.isfinite(pct_raw) else "—"

        rows.append(
            "<tr>"
            f"<td>{h_key:g}</td>"
            f"<td>{pct_display}</td>"
            f"<td>{_fmt(float(patient_raw[j]))}</td>"
            f"<td>{_fmt(obs_raw)}</td>"
            f"{_fmt_bias(bias_raw)}"
            f"<td>{_fmt(float(patient_recal[j]))}</td>"
            f"<td>{_fmt(obs_rec)}</td>"
            f"{_fmt_bias(bias_rec)}"
            f"<td>{n_display}</td>"
            "</tr>"
        )
    rows.append("</tbody>")
    legend = (
        "<dl class=\"calibration-legend\">"
        "<dt>Patient pct</dt>"
        "<dd>The patient's percentile of predicted survival within the "
        "internal test cohort at this horizon (raw model). 87% means the "
        "patient's predicted survival is higher than 87&#37; of the "
        "test cohort.</dd>"
        "<dt>Band observed</dt>"
        "<dd>The actual survival rate among test-cohort patients whose "
        "predicted survival falls within "
        f"&plusmn;{_BAND_HALF_WIDTH_PCT:.0f}&nbsp;percentile points of "
        "this patient &mdash; i.e. \"patients like this one\" by "
        "predicted risk.</dd>"
        "<dt>Bias</dt>"
        "<dd>Band observed &minus; predicted. "
        "<span class=\"bias-positive\">Positive bias</span>: model "
        "under-predicts survival "
        "(<span class=\"bias-positive\">pessimistic</span>) for similar "
        "patients. <span class=\"bias-negative\">Negative bias</span>: "
        "model over-predicts survival "
        "(<span class=\"bias-negative\">optimistic</span>). "
        "<em>Example:</em> bias "
        "<code>= <span class=\"bias-positive\">+0.03</span></code> "
        "means similar patients survived past this horizon at a rate "
        "3&nbsp;percentage points higher than the model predicted for "
        "this patient (e.g. 73&#37; observed vs. 70&#37; predicted).</dd>"
        "</dl>"
    )
    return (
        "<p><em><small>Calibration on the internal test cohort, "
        f"localized to a &plusmn;{_BAND_HALF_WIDTH_PCT:.0f}&nbsp;percentile "
        "band around this patient's predicted survival at each horizon. "
        "All values are survival probabilities (0&ndash;1)."
        "</small></em></p>"
        "<table>" + "".join(rows) + "</table>"
        + legend
    )


def _alerts_list(pred, patient: int) -> list:
    """Amber-banner alerts, only populated when warranted."""
    alerts = []

    # OOD feature values
    ood = pred.ood_features(patient)
    for name, val, lo, hi in ood:
        alerts.append(
            f"Feature '{name}' = {val:g} is outside the training range "
            f"[{lo:g}, {hi:g}] — predictions involve extrapolation."
        )

    # Imputed inputs on the reported patient
    if pred.imputation_mask is not None:
        row = pred.imputation_mask.iloc[patient]
        imputed = [c for c in row.index if bool(row[c])]
        if imputed:
            alerts.append(
                "Input values were missing for and will be imputed: "
                + ", ".join(imputed)
            )

    # Wide median-survival CI (>2x the point estimate)
    ci = pred.median_survival_ci
    if ci is not None and pred._median_survival is not None:
        med = float(pred._median_survival[patient])
        if not np.isnan(med) and med > 0:
            lo = float(ci["ci_lower"].iloc[patient])
            hi = float(ci["ci_upper"].iloc[patient])
            if np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > 2 * med:
                alerts.append(
                    "Median survival confidence interval is wide "
                    f"({lo:.1f}–{hi:.1f}); treat the point estimate with "
                    "caution."
                )

    return alerts
