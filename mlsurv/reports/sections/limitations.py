"""Limitations & Caveats section for report generation.

Auto-detects computable limitations from data already computed during the
pipeline, presents each with what was detected, why it matters, and a
suggested action.  Aligned with TRIPOD+AI Item 26 (limitations).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlsurv.constants import MODEL_DISPLAY_NAMES, get_metric_info
from mlsurv.evaluation.metrics import HIGH_CENSORING_THRESHOLD
from mlsurv.reports._constants import (
    CAL_SLOPE_HIGH,
    CAL_SLOPE_LOW,
    CORRELATION_THRESHOLD,
    DISCRIMINATION_THRESHOLD,
    MAX_PH_VIOLATION_DISPLAY,
    P_THRESHOLD_SIGNIFICANT,
    format_pvalue,
)
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, BaseSection


def _is_valid(value: float | None) -> bool:
    """Return True if *value* is a usable finite number (not None, not NaN)."""
    return value is not None and np.isfinite(value)


# Flag definitions: (flag_id, flag_name)
_FLAG_DEFS: list[tuple[str, str]] = [
    ("small_sample", "Small Sample Size"),
    ("low_epv", "Low Events Per Variable (EPV)"),
    ("high_censoring", "High Censoring Rate"),
    ("missing_data", "Extensive Missing Data"),
    ("ph_violations", "Proportional Hazards Violations"),
    ("linearity_violation", "Log-Hazard Linearity Violations"),
    ("overfitting_gap", "Train/Test Performance Gap"),
    ("low_discrimination", "Low Discrimination"),
    ("poor_calibration", "Poor Calibration"),
    ("multicollinearity", "Substantial Multicollinearity"),
    ("no_external_validation", "No External Validation"),
    ("small_subpopulations", "Small Subpopulations"),
    ("model_appropriateness", "Model Appropriateness"),
    ("ipcw_support", "IPCW Training-Support Violation"),
]


# Shared Flag 14 (IPCW support) wording, used for one cohort or several.
_IPCW_WHY = (
    "Every IPCW-weighted metric (Uno's C-index, AUC(t), Brier, IBS) "
    "reweights each test observation by 1/G_hat(t) (or 1/G_hat(t)^2 for "
    "Uno's C-index), where G_hat is the Kaplan-Meier of the "
    "{g_label} censoring distribution. Beyond that follow-up "
    "(or where G_hat is very small), these weights explode and a "
    "handful of late-time observations dominate both the point estimate "
    "and the bootstrap CI width, producing numbers that no longer behave "
    "like bias-corrected discrimination/calibration estimates."
)
_IPCW_COMPOSITION_WHY = (
    " Separately, when the G_hat-fitting cohort has too few events or "
    "too few censored observations, the censoring distribution cannot "
    "be estimated at all: G_hat stays flat near 1.0, the IPCW weights "
    "collapse to 1, and Uno's C-index, AUC(t), Brier, and IBS silently "
    "reduce to their unweighted forms (no NaN, no censoring correction)."
)
_IPCW_ADVICE = (
    "Compare Uno's IPCW against Harrell/Antolini before drawing "
    "conclusions about discrimination, and prefer Harrell/Antolini "
    "when their CIs are much tighter. Brier, IBS, and AUC(t) lack "
    "non-IPCW counterparts, so widened bootstrap CIs (relative to the "
    "Harrell/Antolini gap) are themselves the symptom — interpret those "
    "metrics with caution. Pass ipcw_tau=<float> to evaluate()/bootstrap() "
    "to override the default, or ipcw_tau=None to disable truncation "
    "(legacy behavior)."
)
_IPCW_COMPOSITION_ACTION = (
    " Because the G_hat-fitting cohort is too thin to estimate the "
    "censoring distribution, the IPCW correction is effectively off — "
    "read these IPCW rows as their unweighted counterparts, prefer "
    "Harrell/Antolini, and verify the cohort composition reflects true "
    "follow-up rather than dropped or missing survivors."
)


def _ipcw_composition_thin(flag: dict) -> bool:
    """True when the G_hat-fitting cohort has too few events or censored obs."""
    n_events = flag.get("n_g_events")
    n_censored = flag.get("n_g_censored")
    floor = flag.get("min_ipcw_support", 10)
    return (
        n_events is not None
        and n_censored is not None
        and min(n_events, n_censored) < floor
    )


class LimitationsSection(BaseSection):
    """Limitations & Caveats section.

    Auto-detects 14 computable limitation signals from pipeline data and
    presents each with what was detected, why it matters, and a suggested
    action.  Always available — if nothing triggers, a short reminder is
    shown instead.
    """

    name: str = "limitations"
    title: str = "Limitations & Caveats"

    def is_available(self) -> bool:
        """Always available."""
        return True

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_triggered_flags(self) -> list[tuple[str, str]]:
        """Return ``(flag_id, flag_name)`` for each triggered limitation.

        Runs all 14 checkers but skips markdown generation and data storage,
        making this suitable for cross-section summaries (e.g. the executive
        summary's limitations line).
        """
        triggered: list[tuple[str, str]] = []
        for flag_id, flag_name in _FLAG_DEFS:
            checker = getattr(self, f"_check_{flag_id}", None)
            if checker is None:
                continue
            try:
                result = checker()
            except (ImportError,) + ALL_EXCEPTIONS as exc:
                self._logger.debug(
                    "%s: flag %s skipped (%s)", self.name, flag_id, exc,
                )
                continue
            if result is not None:
                triggered.append((flag_id, flag_name))
        return triggered

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(self) -> dict[str, Any]:
        """Generate limitations section content."""
        md_lines: list[str] = [
            "This section highlights computable limitations detected during "
            "the analysis pipeline.  Each flag describes **what was detected**, "
            "**why it matters**, and a **suggested action**.  These are "
            "automatically derived — study-specific limitations must be added "
            "by the researcher.",
            "",
        ]

        # Run each checker once, keeping results for formatting
        all_flags: list[dict] = []
        triggered: list[tuple[str, str, dict]] = []  # (flag_id, flag_name, result)

        for flag_id, flag_name in _FLAG_DEFS:
            checker = getattr(self, f"_check_{flag_id}", None)
            if checker is None:
                continue
            try:
                result = checker()
            except (ImportError,) + ALL_EXCEPTIONS as exc:
                self._log_recoverable_exception(
                    exc, f"{self.name}: flag {flag_id}"
                )
                result = None

            is_triggered = result is not None
            all_flags.append({
                "flag_id": flag_id,
                "flag_name": flag_name,
                "triggered": is_triggered,
                "detail": result["detail"] if is_triggered else "",
            })
            if is_triggered:
                triggered.append((flag_id, flag_name, result))

        self.data["flags"] = pd.DataFrame(all_flags)

        if triggered:
            entries = [
                (name, self._flag_anchor(fid)) for fid, name, _ in triggered
            ]
            md_lines.extend(self._build_section_outline(entries, min_entries=0))

            for _flag_id, flag_name, result in triggered:
                md_lines.extend(self._format_flag(flag_name, result))
        else:
            md_lines.append(
                "No computable limitations were detected for this analysis. "
                "This does **not** mean the analysis is free of limitations "
                "— many important concerns (e.g., selection bias, "
                "generalizability, clinical relevance) cannot be assessed "
                "automatically."
            )
            md_lines.append("")

        return {
            "markdown": "\n".join(md_lines),
            "figures": [],
        }

    # ------------------------------------------------------------------
    # Flag formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _flag_anchor(flag_id: str) -> str:
        return flag_id.replace("_", "-")

    @staticmethod
    def _format_flag(flag_name: str, result: dict) -> list[str]:
        import html as html_module

        detected = html_module.escape(result.get("detected", "(details unavailable)"))
        why = html_module.escape(result.get("why", "(explanation unavailable)"))
        action = html_module.escape(result.get("action", "(no action suggested)"))
        return [
            f"### {flag_name}",
            "",
            "<!-- MLSURV_RAW_HTML -->",
            '<div class="mlsurv-callout-warning">',
            f"  <strong>&#9888; What was detected:</strong> {detected}",
            "  <br><br>",
            f"  <strong>Why it matters:</strong> {why}",
            "  <br><br>",
            f"  <strong>Suggested action:</strong> {action}",
            "</div>",
            "<!-- /MLSURV_RAW_HTML -->",
            "",
        ]

    # ------------------------------------------------------------------
    # Flag checkers — each returns dict or None
    # ------------------------------------------------------------------

    def _check_small_sample(self) -> dict | None:
        """Flag 1: Small sample size (n < 200)."""
        X_train = getattr(self.learner, "X_train", None)
        X_test = getattr(self.learner, "X_test", None)
        if X_train is None:
            return None
        n_train = len(X_train)
        n_test = len(X_test) if X_test is not None else 0
        n_total = n_train + n_test
        if n_total >= 200:
            return None
        return {
            "detected": f"Total sample size is {n_total} (train={n_train}, test={n_test}).",
            "why": (
                "Small sample sizes increase the risk of overfitting and "
                "reduce statistical power.  Model performance estimates may "
                "be unstable and confidence intervals wide."
            ),
            "action": (
                "Consider collecting more data, using simpler models, or "
                "applying regularization.  Interpret performance metrics "
                "with caution and report confidence intervals."
            ),
            "detail": f"n_total={n_total}",
        }

    def _check_low_epv(self) -> dict | None:
        """Flag 2: Low events per variable (EPV < 10)."""
        X_train = getattr(self.learner, "X_train", None)
        y_train = getattr(self.learner, "y_train", None)
        if X_train is None or y_train is None:
            return None
        n_events = int(y_train["event"].sum())
        n_features = X_train.shape[1]
        if n_features == 0:
            return None
        epv = n_events / n_features
        if epv >= 10:
            return None
        return {
            "detected": (
                f"Events per variable (EPV) = {epv:.1f} "
                f"({n_events} events / {n_features} features)."
            ),
            "why": (
                "An EPV below 10 is a widely cited rule of thumb indicating "
                "that Cox regression coefficients may be biased and unstable.  "
                "Regularized and tree-based models are more tolerant but can "
                "still overfit."
            ),
            "action": (
                "Reduce the number of features via domain-informed selection "
                "or regularization (e.g., CoxNet).  Alternatively, use "
                "penalized models or report results as exploratory."
            ),
            "detail": f"epv={epv:.1f}",
        }

    def _check_high_censoring(self) -> dict | None:
        """Flag 3: High censoring (above HIGH_CENSORING_THRESHOLD)."""
        y_train = getattr(self.learner, "y_train", None)
        y_test = getattr(self.learner, "y_test", None)
        if y_train is None:
            return None
        events = pd.Series(y_train["event"], dtype=float)
        if y_test is not None:
            events = pd.concat(
                [events, pd.Series(y_test["event"], dtype=float)],
                ignore_index=True,
            )
        event_rate = events.mean()
        censoring_rate = 1.0 - event_rate
        if censoring_rate <= HIGH_CENSORING_THRESHOLD:
            return None
        return {
            "detected": (
                f"Censoring rate is {censoring_rate:.1%} "
                f"(event rate {event_rate:.1%})."
            ),
            "why": (
                "High censoring reduces the effective sample size for "
                "event-based metrics (C-index, Brier score) and can bias "
                "survival estimates toward longer follow-up times."
            ),
            "action": (
                "Consider whether the follow-up period is sufficient.  "
                "Use IPCW-based metrics that account for censoring and "
                "report the censoring rate alongside all results."
            ),
            "detail": f"censoring_rate={censoring_rate:.1%}",
        }

    def _check_missing_data(self) -> dict | None:
        """Flag 4: Extensive missing data (>20% overall or >50% any variable)."""
        X_train = getattr(self.learner, "X_train", None)
        X_test = getattr(self.learner, "X_test", None)
        if X_train is None:
            return None
        frames = [X_train]
        if X_test is not None:
            frames.append(X_test)
        combined = pd.concat(frames, ignore_index=True)
        overall_pct = combined.isna().mean().mean() * 100
        per_var_pct = combined.isna().mean() * 100
        max_var_pct = per_var_pct.max()
        worst_var = per_var_pct.idxmax() if max_var_pct > 0 else None

        if overall_pct <= 20 and max_var_pct <= 50:
            return None

        parts = []
        if overall_pct > 20:
            parts.append(f"overall missingness is {overall_pct:.1f}%")
        if max_var_pct > 50:
            parts.append(
                f"variable '{worst_var}' has {max_var_pct:.1f}% missing"
            )
        detected = "; ".join(parts).capitalize() + "."

        return {
            "detected": detected,
            "why": (
                "Extensive missingness can introduce bias if data are not "
                "missing completely at random (MCAR).  Imputation assumptions "
                "may not hold, and model performance can degrade."
            ),
            "action": (
                "Investigate the missingness mechanism.  Consider sensitivity "
                "analyses or excluding variables with >50% missing.  For "
                "advanced handling, external multiple imputation tools can be "
                "applied before loading data into mlsurv."
            ),
            "detail": f"overall={overall_pct:.1f}%, max_var={max_var_pct:.1f}%",
        }

    def _affected_models_sentence(self, assumption_models: frozenset) -> str:
        """Return the "Affected trained models" sentence for an assumption flag.

        If any trained model is in ``assumption_models``, list them.  Otherwise
        derive the fallback model list from ``assumption_models`` itself (via
        ``MODEL_DISPLAY_NAMES``) so adding a new member to the set updates
        every flag's wording without touching this file.
        """
        affected = sorted(
            m for m in self._get_all_models() if m in assumption_models
        )
        if affected:
            return f"Affected trained models: {', '.join(affected)}."

        names = [MODEL_DISPLAY_NAMES.get(m, m) for m in sorted(assumption_models)]
        return (
            f"No affected models ({'/'.join(names)}) are in this "
            f"analysis, but this would matter if you add one."
        )

    def _check_ph_violations(self) -> dict | None:
        """Flag 5: Proportional hazards violations (FDR-adjusted p < 0.05).

        Reads the cached PH test result populated by ``setup()``.
        Fires whenever a violation is detected, regardless of which models
        were trained; the ``detected`` text names the affected trained
        models (if any) so the user knows whose results to interpret
        cautiously.  Clean results are not surfaced here — positive
        confirmation lives in the Dataset section.
        """
        from mlsurv.constants import PH_ASSUMPTION_MODELS

        result_df = self.learner.ph_test_result
        if result_df is None or result_df.empty:
            return None

        violations = result_df[
            (result_df["significant"]) & (result_df["feature"] != "GLOBAL")
        ]
        if violations.empty:
            return None

        n_violated = len(violations)
        features = ", ".join(
            violations["feature"].tolist()[:MAX_PH_VIOLATION_DISPLAY]
        )
        suffix = (
            f" (and {n_violated - MAX_PH_VIOLATION_DISPLAY} more)"
            if n_violated > MAX_PH_VIOLATION_DISPLAY else ""
        )

        models_sentence = self._affected_models_sentence(PH_ASSUMPTION_MODELS)

        return {
            "detected": (
                f"{n_violated} feature(s) violate the proportional hazards "
                f"assumption (Grambsch-Therneau FDR-adjusted p < "
                f"{P_THRESHOLD_SIGNIFICANT}): {features}{suffix}.  "
                f"{models_sentence}"
            ),
            "why": (
                "Cox proportional hazards models assume that hazard ratios "
                "are constant over time.  Violations mean that the effect "
                "of these features changes with time, potentially biasing "
                "coefficient estimates and survival predictions."
            ),
            "action": (
                "Consider non-PH alternatives available in mlsurv (e.g., "
                "RSF, DeepHit, XGBoost-AFT).  For Cox-family models, "
                "time-varying coefficients or stratified Cox may also help "
                "(not currently supported in mlsurv).  Inspect Schoenfeld "
                "residual plots for the flagged features."
            ),
            "detail": f"n_violations={n_violated}",
        }

    def _check_linearity_violation(self) -> dict | None:
        """Flag 6: Log-hazard linearity violations.

        Reads the cached linearity test result populated by ``setup()``.
        Fires whenever a violation is detected, regardless of which models
        were trained; the ``detected`` text names the affected trained
        models (if any) so the user knows whose results to interpret
        cautiously.
        """
        from mlsurv.constants import LINEARITY_ASSUMPTION_MODELS

        result_df = self.learner.linearity_test_result
        if result_df is None or result_df.empty:
            return None

        violations = result_df[
            (result_df["significant"]) & (result_df["feature"] != "GLOBAL")
        ]
        if violations.empty:
            return None

        n_violated = len(violations)
        features = ", ".join(
            violations["feature"].tolist()[:MAX_PH_VIOLATION_DISPLAY]
        )
        suffix = (
            f" (and {n_violated - MAX_PH_VIOLATION_DISPLAY} more)"
            if n_violated > MAX_PH_VIOLATION_DISPLAY else ""
        )

        models_sentence = self._affected_models_sentence(
            LINEARITY_ASSUMPTION_MODELS
        )

        return {
            "detected": (
                f"{n_violated} continuous feature(s) violate the log-hazard "
                f"linearity assumption (Harrell/rms Wald FDR-adjusted p < "
                f"{P_THRESHOLD_SIGNIFICANT}): {features}{suffix}.  "
                f"{models_sentence}"
            ),
            "why": (
                "Cox proportional hazards models assume the log-hazard is "
                "linear in each continuous covariate (Harrell 2015 §4.6, §5.5).  "
                "Violations produce biased coefficient estimates and "
                "miscalibrated risk predictions even when the PH assumption "
                "itself holds."
            ),
            "action": (
                "Consider a non-linear transformation of the flagged "
                "feature(s) (log, Box-Cox, square-root) before refitting, "
                "or switch to a flexible model that can capture non-linear "
                "effects natively (RSF, GBSA, DeepSurv).  Inspect the "
                "Linearity Assumption Assessment section's martingale "
                "residual plot to see the functional form of each violation."
            ),
            "detail": f"n_violations={n_violated}",
        }

    @staticmethod
    def _resolve_test_value(
        evaluation: object, info: object,
    ) -> tuple[float | None, object]:
        """Return (test_value, metric_info) for the overfitting gap check.

        Tries the scoring-metric column from the concordance DataFrame first,
        then falls back to ``c_index_test`` (Harrell).  Returns ``(None, ...)``
        when no valid test metric is available.
        """
        concordance = getattr(evaluation, "concordance", None)
        if concordance is not None and info.concordance_column in concordance.columns:
            val = concordance[info.concordance_column].iloc[0]
            if _is_valid(val):
                return float(val), info

        # Fall back to Harrell's C-index
        c_test = getattr(evaluation, "c_index_test", None)
        if _is_valid(c_test):
            return c_test, get_metric_info('harrell')

        return None, info

    def _check_overfitting_gap(self) -> dict | None:
        """Flag 7: Train/test performance gap (> 0.05 in scoring metric)."""
        models = self._get_all_models()
        if not models:
            return None

        # Resolve the scoring metric and its test-set column
        scoring = getattr(self.learner, 'scoring', 'harrell')
        info = get_metric_info(scoring) if isinstance(scoring, str) else None
        if info is None or info.diagnostic_only:
            info = get_metric_info('harrell')

        worst_gap = -np.inf
        worst_model = None
        worst_info = info
        for name, result in models.items():
            cv_mean = getattr(result, "cv_metric_mean", None)
            evaluation = getattr(result, "evaluation", None)
            if cv_mean is None or evaluation is None:
                continue
            if not _is_valid(cv_mean):
                continue

            test_val, active_info = self._resolve_test_value(evaluation, info)
            if test_val is None:
                continue

            # Direction-aware gap: positive means CV looks better than test
            if active_info.lower_is_better:
                gap = test_val - cv_mean  # overfit -> cv low (good), test high
            else:
                gap = cv_mean - test_val  # overfit -> cv high (good), test low

            if gap > worst_gap:
                worst_gap = gap
                worst_model = name
                worst_info = active_info

        if worst_gap <= 0.05 or worst_model is None:
            return None

        return {
            "detected": (
                f"Model '{worst_model}' shows a CV-to-test performance gap "
                f"of {worst_gap:.3f} (CV mean vs. test {worst_info.short_name})."
            ),
            "why": (
                "A gap exceeding 0.05 suggests potential overfitting \u2014 the "
                "model learned patterns in the training data that do not "
                "generalize to unseen data."
            ),
            "action": (
                "Increase regularization, reduce model complexity, or "
                "reduce the number of tuning trials (e.g., ``n_trials`` or "
                "``tune=False``)."
            ),
            "detail": f"model={worst_model}, gap={worst_gap:.3f}",
        }

    def _check_low_discrimination(self) -> dict | None:
        """Flag 8: Low discrimination (test C-index < 0.6 for any model)."""
        models = self._get_all_models()
        if not models:
            return None

        low_models: list[tuple[str, float]] = []
        for name, result in models.items():
            evaluation = getattr(result, "evaluation", None)
            if evaluation is None:
                continue
            c_test = getattr(evaluation, "c_index_test", None)
            if not _is_valid(c_test):
                continue
            if c_test < DISCRIMINATION_THRESHOLD:
                low_models.append((name, c_test))

        if not low_models:
            return None

        parts = [f"'{n}' (C-index={c:.3f})" for n, c in low_models]
        return {
            "detected": (
                f"{len(low_models)} model(s) have test C-index below {DISCRIMINATION_THRESHOLD}: "
                + ", ".join(parts) + "."
            ),
            "why": (
                f"A C-index below {DISCRIMINATION_THRESHOLD} indicates near-random discrimination — "
                "the model cannot reliably distinguish patients at higher vs. "
                "lower risk."
            ),
            "action": (
                "Review feature engineering and data quality.  Consider "
                "alternative models or additional predictors.  A C-index "
                f"< {DISCRIMINATION_THRESHOLD} may indicate that the available features do not "
                "carry sufficient prognostic information."
            ),
            "detail": f"n_low={len(low_models)}",
        }

    def _classify_model_calibration(
        self, name: str, result: Any
    ) -> tuple[bool, str] | None:
        """Classify one model's calibration.

        Returns ``(is_calibrated, entry)`` where *entry* is the display string
        and *is_calibrated* is True when the model already has a calibrator
        attached (so the drift is residual). Returns ``None`` when the model
        has no evaluation or an in-band slope. The calibration **slope** is the
        sole trigger (outside ``[CAL_SLOPE_LOW, CAL_SLOPE_HIGH]``); the GND
        p-value is shown as context only.
        """
        evaluation = getattr(result, "evaluation", None)
        if evaluation is None:
            return None

        slope = getattr(evaluation, "cal_slope_test", None)
        gnd_p = getattr(evaluation, "gnd_test", None)

        if not _is_valid(slope) or (CAL_SLOPE_LOW <= slope <= CAL_SLOPE_HIGH):
            return None

        is_calibrated = getattr(result, "calibrator", None) is not None
        if is_calibrated:
            pre = getattr(result, "precalibration_slope", None)
            if _is_valid(pre):
                detail = f"slope {pre:.3f} -> {slope:.3f} after calibration"
            else:
                detail = f"slope={slope:.3f} after calibration"
        else:
            detail = f"slope={slope:.3f}"
        if _is_valid(gnd_p):
            detail += f", GND p={format_pvalue(gnd_p)}"
        return is_calibrated, f"'{name}' ({detail})"

    def _check_poor_calibration(self) -> dict | None:
        """Flag 9: Poor calibration. Flags models whose test calibration slope
        falls outside the acceptable band; suggests calibrating S(t|x) via
        ``train(calibrate=...)``. Models that already carry a calibrator and
        still drift get the "try a different method" suggestion."""
        models = self._get_all_models()
        if not models:
            return None

        fresh: list[str] = []        # no calibrator yet
        residual: list[str] = []     # has a calibrator, still drifts
        ordered_entries: list[str] = []
        for name, result in models.items():
            classified = self._classify_model_calibration(name, result)
            if classified is None:
                continue
            is_calibrated, entry = classified
            (residual if is_calibrated else fresh).append(name)
            ordered_entries.append(entry)

        n_total = len(fresh) + len(residual)
        if n_total == 0:
            return None

        action_sentences = []
        if fresh:
            names = ", ".join(f"'{n}'" for n in fresh)
            action_sentences.append(
                f"Consider calibrating S(t|x) for {names} via "
                f"`train(calibrate=...)` (isotonic / lowdof / breslow_slope)."
            )
        if residual:
            names = ", ".join(f"'{n}'" for n in residual)
            action_sentences.append(
                f"{names} already have a calibrator but show residual drift — "
                f"consider a different `train(calibrate=...)` method."
            )

        return {
            "detected": (
                f"Calibration could be improved for {n_total} model(s): "
                + "; ".join(ordered_entries) + "."
            ),
            "why": (
                "A calibration slope far from 1.0 indicates systematic "
                "over- or under-prediction of risk. The GND p-value is shown "
                "for context (it over-rejects on well-powered cohorts, so it "
                "is not used as a standalone trigger)."
            ),
            "action": " ".join(action_sentences),
            "detail": (
                f"n_models={n_total}, n_uncalibrated={len(fresh)}, "
                f"n_residual_after_calibration={len(residual)}"
            ),
        }

    def _check_multicollinearity(self) -> dict | None:
        """Flag 10: Substantial multicollinearity."""
        X_train = getattr(self.learner, "X_train", None)
        if X_train is None:
            return None

        from mlsurv.evaluation.importance import detect_correlated_features
        from mlsurv.reports.sections.feature_analysis import (
            _assess_correlation_severity,
        )

        pairs = detect_correlated_features(X_train, threshold=CORRELATION_THRESHOLD)
        if not pairs:
            return None

        n_pairs = len(pairs)
        involved = set()
        max_corr = 0.0
        for f1, f2, corr in pairs:
            involved.update([f1, f2])
            max_corr = max(max_corr, abs(corr))

        n_features = X_train.shape[1]
        pct_involved = len(involved) / n_features * 100 if n_features else 0

        severity, recommendation = _assess_correlation_severity(
            n_pairs, pct_involved, max_corr
        )

        if severity == "minor":
            return None

        return {
            "detected": (
                f"{n_pairs} correlated feature pair(s) detected "
                f"(|R| >= {CORRELATION_THRESHOLD}, max |R| = {max_corr:.2f}, "
                f"{len(involved)} features involved = {pct_involved:.0f}% "
                f"of all features).  Severity: {severity}."
            ),
            "why": (
                "Multicollinearity inflates variance of coefficient "
                "estimates in Cox models and can distort permutation "
                "importance scores.  Feature effects become harder to "
                "disentangle."
            ),
            "action": recommendation,
            "detail": f"severity={severity}, n_pairs={n_pairs}",
        }

    def _check_no_external_validation(self) -> dict | None:
        """Flag 11: No external validation performed."""
        try:
            validation_results = self.learner.validation_results
        except AttributeError:
            validation_results = None

        if validation_results:
            return None

        return {
            "detected": "No external validation cohort was provided.",
            "why": (
                "Internal validation (train/test split, cross-validation) "
                "may overestimate model performance.  External validation on "
                "an independent cohort is the gold standard for assessing "
                "generalizability."
            ),
            "action": (
                "Validate the model on an independent dataset from a "
                "different institution or time period.  If unavailable, "
                "clearly state this limitation when reporting results."
            ),
            "detail": "no external cohort",
        }

    def _check_small_subpopulations(self) -> dict | None:
        """Flag 12: Subpopulations with too few events for reliable estimation."""
        sg_results = self.learner.subpopulation_results
        if not sg_results:
            return None

        small = []
        for group_var, sg_result in sg_results.items():
            for _, row in sg_result.summary.iterrows():
                if row['n_events'] < 30:
                    small.append(
                        f"{row['subpopulation']} ({group_var}: "
                        f"{row['n_events']} events, n={row['n_samples']})"
                    )

        if not small:
            return None

        return {
            "detected": (
                f"{len(small)} subpopulation(s) have fewer than 30 events: "
                f"{', '.join(small)}."
            ),
            "why": (
                "Concordance index estimates require sufficient events to be "
                "statistically reliable. With fewer than 30 events, confidence "
                "intervals are wide and point estimates may be unstable across "
                "resamples."
            ),
            "action": (
                "Interpret C-index estimates for small subpopulations with caution. "
                "Consider pooling small subpopulations, collecting more data, or "
                "reporting only point estimates without formal inference."
            ),
            "detail": f"{len(small)} small subpopulation(s)",
        }

    def _check_ipcw_support(self) -> dict | None:
        """Flag 14: Uno's IPCW concordance evaluated beyond training support.

        Covers the test set and every external cohort. Reads the
        ``ipcw_support_flag`` (test set) and ``ipcw_support_flags`` (one entry
        per cohort, keyed like the cohort names passed to ``validate()``)
        state fields — exempt getattr reads, no public accessor. The
        test-set flag is updated by each evaluate()/bootstrap() call on the
        test set, the per-cohort record by every evaluate()/bootstrap()/
        validate() call. We deliberately do NOT fall back to per-ModelResult flag
        attributes — those aren't cleared when a later in-support call
        replaces the most-recent evaluation, so a fallback would resurrect
        stale flags from prior violating runs.
        """
        flagged = []
        main_flag = getattr(self.learner._state, 'ipcw_support_flag', None)
        if main_flag is not None:
            flagged.append(("Test set", main_flag))
        by_cohort = getattr(self.learner._state, 'ipcw_support_flags', {})
        for cohort, flag in by_cohort.items():
            if cohort != "main" and flag is not None:
                flagged.append((f"Validation cohort {cohort}", flag))
        if not flagged:
            return None

        messages = [(label, self._ipcw_support_message(flag))
                    for label, flag in flagged]
        if len(messages) == 1 and flagged[0][0] == "Test set":
            return messages[0][1]
        # Several cohorts (or an external one): one line per cohort. Each
        # cohort has its own tau and G_hat source (in its detail), so the
        # shared advice must not quote one cohort's values for all of them.
        any_thin = any(_ipcw_composition_thin(flag) for _, flag in flagged)
        return {
            "detected": " ".join(f"{label}: {m['detected']}"
                                 for label, m in messages),
            "why": (_IPCW_WHY.format(g_label="corresponding")
                    + (_IPCW_COMPOSITION_WHY if any_thin else "")),
            "action": (
                "The IPCW-weighted rows for each cohort listed above use the "
                "tau and G_hat source given in that cohort's detail. "
                + _IPCW_ADVICE
                + (_IPCW_COMPOSITION_ACTION if any_thin else "")
            ),
            "detail": "; ".join(f"{label}: {m['detail']}"
                                for label, m in messages),
        }

    @staticmethod
    def _ipcw_support_message(flag: dict) -> dict:
        """Detected / why / action / detail text for one cohort's IPCW flag."""
        n_beyond = flag.get("n_beyond", 0)
        min_g = flag.get("min_g_hat", float("nan"))
        t_max = flag.get("t_max_train", float("nan"))
        tau = flag.get("tau_used", flag.get("tau"))
        g_source = flag.get("g_hat_source", "train")
        g_label = "validation-cohort" if g_source == "test" else "training-cohort"
        t_max_label = (
            "validation t_max" if g_source == "test" else "training t_max"
        )

        n_g_events = flag.get("n_g_events")
        n_g_censored = flag.get("n_g_censored")
        floor = flag.get("min_ipcw_support", 10)
        composition_thin = _ipcw_composition_thin(flag)

        detected_parts = []
        if n_beyond > 0:
            detected_parts.append(
                f"{n_beyond} test obs extend beyond {t_max_label}={t_max:.1f}"
            )
        if np.isfinite(min_g) and min_g < 0.05:
            detected_parts.append(f"min(G_hat) on evaluated range = {min_g:.3f}")
        if composition_thin:
            thin, count = (
                ("censored", n_g_censored)
                if n_g_censored <= n_g_events
                else ("events", n_g_events)
            )
            detected_parts.append(
                f"only {count} {thin} obs in the {g_label} G_hat set "
                f"(< {floor}); G_hat not estimable"
            )
        detected = (
            "IPCW-weighted metrics (Uno's C-index, AUC(t), Brier, IBS) "
            f"evaluated outside reliable {g_label} censoring support: "
            + "; ".join(detected_parts) + "."
        )

        tau_str = f"{tau:.1f}" if isinstance(tau, (int, float)) and tau is not None else "None (unbounded)"

        return {
            "detected": detected,
            "why": (
                _IPCW_WHY.format(g_label=g_label)
                + (_IPCW_COMPOSITION_WHY if composition_thin else "")
            ),
            "action": (
                f"All IPCW-weighted rows in the evaluation table (Uno's C-index, "
                f"AUC(t), Brier, IBS) use tau={tau_str} with the {g_label} G_hat. "
                + _IPCW_ADVICE
                + (_IPCW_COMPOSITION_ACTION if composition_thin else "")
            ),
            "detail": (
                f"n_beyond={n_beyond}, min_G={min_g:.3f}, tau={tau_str}, "
                f"g_source={g_source}, n_g_events={n_g_events}, "
                f"n_g_censored={n_g_censored}"
            ),
        }

    def _check_model_appropriateness(self) -> dict | None:
        """Flag 13: trained models the recommendation engine demoted to
        *not recommended* -- for any reason, not just dataset size.

        ``recommend_models`` demotes a model below the recommended tier for
        low events-per-variable, high dimensionality, or proportional-hazards
        / log-hazard-linearity violations. This flag reports each flagged
        model's *actual* reason (the same per-model ``details`` text the Model
        Recommendations table prints) and frames the explanation by cause, so
        it never attributes an assumption-driven demotion to dataset size and
        only cites the sample-size literature when the demotion is genuinely
        size-driven.
        """
        models = self._get_all_models()
        if not models:
            return None

        X_train = getattr(self.learner, "X_train", None)
        y_train = getattr(self.learner, "y_train", None)
        if X_train is None or y_train is None:
            return None

        # Reuse cached result from setup() when available
        rec = self.learner.model_recommendations
        if rec is None:
            from mlsurv.guidance import recommend_models
            # Report rendering must not re-run PH/linearity tests;
            # we read cached state only and let the violation kwargs stay
            # ``None`` so the table renders 'test unavailable' rather than
            # silently recomputing.
            rec = recommend_models(
                X_train, y_train,
                run_assumption_tests=False,
                verbose=False,
            )
        flagged = [name for name in models if name in rec.not_recommended]
        if not flagged:
            return None

        # State each flagged model's real reason -- identical to the per-model
        # text in the Model Recommendations table, so the two sections agree.
        reason_parts = [
            f"{name} ({rec.details.get(name, 'not recommended')})"
            for name in flagged
        ]
        detected = (
            "Model(s) trained but flagged as not recommended: "
            + "; ".join(reason_parts) + "."
        )

        # Group flagged models by demotion cause so each clause cites the
        # right literature (sample-size papers only for size-driven demotions).
        cause_models: dict[str, list[str]] = {
            "size": [], "high_dim": [], "ph": [], "linearity": [],
        }
        for name in flagged:
            for cause in rec.demotion_causes(name):
                cause_models[cause].append(name)

        why_clauses: list[str] = []
        if cause_models["size"]:
            names = ", ".join(cause_models["size"])
            why_clauses.append(
                f"{names}: too few events per variable for reliable training "
                f"(N={rec.n_samples}, EPV={rec.epv:.1f}), so these models have "
                "insufficient data to learn robust representations and are at "
                "high risk of overfitting (Riley et al. 2019; "
                "van der Ploeg et al. 2014)."
            )
        if cause_models["high_dim"]:
            names = ", ".join(cause_models["high_dim"])
            why_clauses.append(
                f"{names}: more features than events "
                f"(n_features={rec.n_features} >= n_events={rec.n_events}), "
                "which makes the Cox partial likelihood ill-conditioned "
                "(Hu et al. 2023)."
            )
        if cause_models["ph"]:
            names = ", ".join(cause_models["ph"])
            why_clauses.append(
                f"{names}: violate the proportional-hazards assumption -- see "
                "the 'Proportional Hazards Violations' flag above and the "
                "Proportional Hazards Assessment section."
            )
        if cause_models["linearity"]:
            names = ", ".join(cause_models["linearity"])
            why_clauses.append(
                f"{names}: violate the log-hazard linearity assumption -- see "
                "the 'Log-Hazard Linearity Violations' flag above and the "
                "Linearity Assumption Assessment section."
            )
        if not why_clauses:
            # Defensive: a flagged model with no identified cause should not
            # occur when ``rec`` is internally consistent, but never assert a
            # false (size) cause if it does. Log it so a rec/causes mismatch
            # is visible to maintainers rather than silently generic.
            self._logger.debug(
                "%s: flagged models %s have no demotion cause; rec and "
                "demotion_causes() disagree", self.name, flagged,
            )
            why_clauses.append(
                "These models were demoted below the recommended tier; see the "
                "Model Recommendations table for the per-model reason."
            )
        why = " ".join(why_clauses)

        if rec.recommended:
            action = (
                f"Consider the recommended model(s) "
                f"({', '.join(rec.recommended)}). "
            )
        else:
            action = "Consider collecting more data or reducing features. "
        if cause_models["size"] or cause_models["high_dim"]:
            action += "More events (or fewer features) would raise EPV. "
        if cause_models["ph"] or cause_models["linearity"]:
            action += (
                "For assumption violations, prefer flexible non-Cox models "
                "(e.g. RSF, GBSA, DeepHit) or transform the flagged features. "
            )
        action += "Run recommend_models(X, y) for full guidance."

        return {
            "detected": detected,
            "why": why,
            "action": action,
            "detail": f"flagged={', '.join(flagged)}",
        }
