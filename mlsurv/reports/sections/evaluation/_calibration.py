"""5.3 Calibration subsection."""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd

from mlsurv._warnings import InternalWarning, SuggestionWarning
from mlsurv.reports._constants import CALIBRATION_PERCENTILES, P_THRESHOLD_SIGNIFICANT
from mlsurv.reports._helpers import (
    collect_calibration_rows,
    empty_calibration_figure,
    eval_grid_with_percentile_fallback,
)
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, DATA_EXCEPTIONS, FigureInfo
from mlsurv.viz._palettes import get_model_display_name

from mlsurv.reports.sections.evaluation._base import _BaseSubsection


class _CalibrationSubsection(_BaseSubsection):
    """Calibration plots and metric interpretation."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def _calibrator_for(self, model_name: str):
        """Return the attached S(t|x) calibrator for *model_name*, or ``None``.

        Calibration figures show the calibrated curve by default, matching the
        deployed model's evaluation metrics (the report has no raw opt-out).
        """
        result = self._s._get_all_models().get(model_name)
        return getattr(result, "calibrator", None) if result is not None else None

    def is_available(self) -> bool:
        if not self._s.learner.is_setup:
            return False
        if self._s.learner.X_test is None or self._s.learner.y_test is None:
            return False
        return any(
            hasattr(result.model, 'predict_survival_function')
            for result in self._s._get_all_models().values()
        )

    def add(self, md_lines: list, figures: list, figure_info: list) -> None:
        md_lines.append(
            "Calibration plots compare predicted survival probabilities against observed "
            "(Kaplan-Meier estimated) survival at the evaluation timepoints listed above. "
            "Points on the diagonal indicate perfect calibration."
        )
        md_lines.append("")

        calibrated = sorted(
            name for name in self._s._get_all_models()
            if self._s.learner._has_calibrator(name)
        )
        if calibrated:
            md_lines.append(
                "> **Note:** "
                + ", ".join(f"`{m}`" for m in calibrated)
                + " have an attached calibrator (`train(calibrate=...)`); the "
                "calibration curves below are on the calibrated S(t|x) scale, "
                "matching their evaluation metrics."
            )
            md_lines.append("")

        models = {
            name: result.model
            for name, result in self._s._get_all_models().items()
            if hasattr(result.model, 'predict_survival_function')
        }

        if not models:
            md_lines.append("*No models available for calibration assessment.*")
            md_lines.append("")
            return

        md_lines.append(
            "**Interpretation:** Points that fall close to the diagonal line "
            "indicate good calibration \u2014 the model's predicted survival "
            "probabilities closely match the actual observed survival rates. "
            "Points above the diagonal mean the model underestimates survival "
            "(predicts worse outcomes than observed); points below mean the "
            "model overestimates survival."
        )
        md_lines.append("")

        # --- Calibration plot (interactive dropdown) ---
        eval_times = self._get_eval_times()
        self._add_calibration_plot(models, eval_times, md_lines, figures, figure_info)

        # --- Calibration metrics table (below the plot) ---
        try:
            self._add_calibration_metric_interpretation(md_lines)
        except DATA_EXCEPTIONS as exc:
            self._log_error(
                exc, "evaluation: calibration metric interpretation",
                warn_user=True,
                user_message=f"Could not generate calibration metrics: {exc}",
            )
            md_lines.append(f"*[Calibration metrics unavailable: {exc}]*")
        md_lines.append("")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_eval_times(self) -> List[float]:
        """Extract evaluation timepoints, preferring the realized auc_t grid.

        Tries the shared EvaluationResult's auc_t, then any individual model's
        auc_t, then falls back to percentiles of test-set event times (with a
        warning).
        """
        candidates = [getattr(self._s.eval_result, 'auc_t', None)]
        candidates += [
            getattr(r.evaluation, 'auc_t', None)
            for r in self._s._get_all_models().values() if r.evaluation
        ]
        grid, from_auc_t = eval_grid_with_percentile_fallback(
            candidates, self._s.learner.y_test, CALIBRATION_PERCENTILES,
        )
        if not from_auc_t:
            warnings.warn(
                "Calibration: eval_times not available from evaluation results. "
                "Using percentile-based fallback (25th/50th/75th of event times).",
                InternalWarning, stacklevel=1,
            )
        return grid

    @staticmethod
    def _fmt_or_dash(val) -> str:
        """Format a float to 3 decimals, or return a dash for None/NaN."""
        if pd.notna(val):
            return f"{val:.3f}"
        return "\u2014"

    @staticmethod
    def _fmt_gnd_significance(gnd_p) -> str:
        """Format GND significance: red 'Yes' if significant (bad), else 'No'."""
        if gnd_p is None or (isinstance(gnd_p, float) and np.isnan(gnd_p)):
            return "—"
        if gnd_p <= P_THRESHOLD_SIGNIFICANT:
            return '<span style="color: red; font-weight: bold;">Yes</span>'
        return "No"

    @staticmethod
    def _calibration_table_html(cal_df: pd.DataFrame) -> str:
        """Build an HTML table for calibration metrics.

        The 'Sig.' column may contain raw HTML (``<span>`` tags for
        significant p-values), so the table must bypass markdown rendering.
        """
        import html as html_module

        raw_html_col = 'Sig.'
        header_cells = "".join(
            f"<th>{html_module.escape(str(col))}</th>"
            for col in cal_df.columns
        )
        body_rows = []
        for _, row in cal_df.iterrows():
            cells = []
            for col in cal_df.columns:
                val = str(row[col])
                if col == raw_html_col:
                    cells.append(f"<td>{val}</td>")
                else:
                    cells.append(f"<td>{html_module.escape(val)}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        return (
            f"<table><thead><tr>{header_cells}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
        )

    def _add_calibration_metric_interpretation(self, md_lines: list) -> None:
        trained_models = self._s._get_all_models()
        rows = []

        for name, result in trained_models.items():
            if not result.evaluation:
                continue
            eval_r = result.evaluation
            slope = getattr(eval_r, 'cal_slope_test', None)
            intercept = getattr(eval_r, 'cal_intercept_test', None)
            gnd = getattr(eval_r, 'gnd_test', None)
            if all(pd.isna(v) for v in (slope, intercept, gnd)):
                continue
            rows.append({
                'Model': get_model_display_name(name),
                'Slope': self._fmt_or_dash(slope),
                'Intercept': self._fmt_or_dash(intercept),
                'GND p': self._fmt_or_dash(gnd),
                'Sig.': self._fmt_gnd_significance(gnd),
            })

        if not rows:
            return

        md_lines.append("#### Calibration Metrics")
        md_lines.append("")
        # Raw HTML — the <span> tag must bypass mistune's escape=True
        md_lines.append("<!-- MLSURV_RAW_HTML -->")
        md_lines.append(
            "<p>"
            "<strong>Slope</strong> measures the spread of predictions (ideal = 1; "
            "&lt;1 indicates underfitting, &gt;1 overfitting). "
            "<strong>Intercept</strong> measures systematic bias (ideal = 0; "
            "&gt;0 means the model underestimates risk, &lt;0 overestimates). "
            "<strong>GND p</strong> is the p-value from the Greenwood\u2013Nam\u2013D\u2019Agostino "
            "goodness-of-fit test (Demler et al., 2015), which compares predicted "
            "and observed survival using Kaplan\u2013Meier estimates with Greenwood\u2019s "
            "variance to account for censoring. A non-significant result "
            f"(p &gt; {P_THRESHOLD_SIGNIFICANT}) indicates no evidence of "
            "miscalibration; a "
            f'<span style="color: red;">significant result '
            f"(p \u2264 {P_THRESHOLD_SIGNIFICANT})</span> "
            "flags potential miscalibration. "
            "Note that <strong>Slope</strong> and <strong>Intercept</strong> use a "
            "complete-case approximation: subjects censored before the evaluation "
            "horizon are dropped and the rest treated as a binary outcome, so they can "
            "be biased under heavy or informative censoring. The ICI/E50/E90 indices, "
            "the GND test, and the Hosmer-Lemeshow test are censoring-aware: ICI/E50/E90 "
            "use inverse-probability-of-censoring (IPCW) weighting, and GND/Hosmer-"
            "Lemeshow are Kaplan-Meier based."
            "</p>"
        )
        md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")
        cal_df = pd.DataFrame(rows)
        # Raw HTML table — Sig. column contains <span> tags
        md_lines.append("<!-- MLSURV_RAW_HTML -->")
        md_lines.append(self._calibration_table_html(cal_df))
        md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")

    def _add_calibration_plot(
        self, models: dict, eval_times: List[float],
        md_lines: list, figures: list, figure_info: list,
    ) -> None:
        """Add a single interactive calibration figure with time dropdown."""
        from mlsurv.viz.calibration import (
            compute_calibration,
            _compute_multi_model_calibration_data,
            plot_calibration_interactive,
        )

        data_key = 'calibration'
        self._ensure_calibration_data_key(data_key)
        is_single = len(models) == 1

        try:
            if is_single:
                model_name, model = next(iter(models.items()))
                display_name = get_model_display_name(model_name)
                title = f"Calibration ({display_name})"

                calibration_data = compute_calibration(
                    model=model,
                    X_test=self._s.learner.X_test,
                    y_test=self._s.learner.y_test,
                    eval_times=eval_times,
                    n_bins=self._s.n_bins,
                    y_train=self._s.learner.y_train,
                    calibrator=self._calibrator_for(model_name),
                    model_name=model_name,
                )

                if not calibration_data:
                    warnings.warn(
                        f"Calibration: no bins had enough samples at any "
                        f"timepoint (n_bins={self._s.n_bins}).",
                        SuggestionWarning, stacklevel=1,
                    )
                    fig = empty_calibration_figure(title)
                else:
                    fig = plot_calibration_interactive(
                        calibration_data, title=title,
                    )
                    for item in calibration_data:
                        self._store_calibration_data(
                            model_name=model_name,
                            eval_time=item['eval_time'],
                            data_key=data_key,
                            predicted=item['predicted'],
                            observed=item['observed'],
                            ci_lower=item['ci_lower'],
                            ci_upper=item['ci_upper'],
                        )
            else:
                title = "Calibration Comparison"
                data_by_time = {}

                for t in eval_times:
                    cal_at_t = _compute_multi_model_calibration_data(
                        models=models,
                        X_test=self._s.learner.X_test,
                        y_test=self._s.learner.y_test,
                        eval_time=t,
                        n_bins=self._s.n_bins,
                        calibrators={
                            name: cal
                            for name in models
                            if (cal := self._calibrator_for(name)) is not None
                        },
                    )
                    if cal_at_t:
                        data_by_time[t] = cal_at_t
                        for m_name, item in cal_at_t.items():
                            self._store_calibration_data(
                                model_name=m_name,
                                eval_time=item['eval_time'],
                                data_key=data_key,
                                predicted=item['predicted'],
                                observed=item['observed'],
                                ci_lower=item.get('ci_lower', []),
                                ci_upper=item.get('ci_upper', []),
                            )

                if not data_by_time:
                    warnings.warn(
                        f"Calibration comparison: no bins had enough samples "
                        f"at any timepoint (n_bins={self._s.n_bins}).",
                        SuggestionWarning, stacklevel=1,
                    )
                    fig = empty_calibration_figure(title)
                else:
                    fig = plot_calibration_interactive(
                        data_by_time, title=title,
                    )

            fig_filename = 'calibration.png'
            figures.append((fig_filename, fig))
            md_lines.append('<!-- MLSURV_RAW_HTML -->')
            md_lines.append(f'<div class="figure-placeholder" data-figure="{fig_filename}"></div>')
            md_lines.append('<!-- /MLSURV_RAW_HTML -->')
            md_lines.append("")

            figure_info.append(FigureInfo(
                figure_id='calibration',
                title=title,
                data_description=(
                    'Tidy DataFrame with binned calibration data '
                    '(model_type, eval_time, bin_predicted, bin_observed, '
                    'bin_ci_lower, bin_ci_upper)'
                ),
                data_key=data_key,
                learner_code="fig = learner.plot_calibration()",
            ))
        except ALL_EXCEPTIONS as e:
            self._log_error(
                e,
                "evaluation: calibration plot generation",
                warn_user=True,
                user_message=f"Could not generate calibration plot: {e}",
            )
            md_lines.append(f"*[Calibration unavailable: {e}]*")
            md_lines.append("")

    def _ensure_calibration_data_key(self, data_key: str) -> None:
        if data_key not in self._s.data:
            self._s.data[data_key] = pd.DataFrame(columns=self._s._CALIBRATION_COLUMNS)

    def _store_calibration_data(
        self, model_name: str, eval_time: float, data_key: str,
        predicted: list, observed: list, ci_lower: list, ci_upper: list,
    ) -> None:
        try:
            lengths = {
                'predicted': len(predicted), 'observed': len(observed),
                'ci_lower': len(ci_lower), 'ci_upper': len(ci_upper),
            }
            if len(set(lengths.values())) > 1:
                raise ValueError(
                    f"Calibration data length mismatch for {model_name} at "
                    f"t={eval_time}: {lengths}"
                )

            item = {
                'eval_time': eval_time,
                'predicted': predicted,
                'observed': observed,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            }
            rows = collect_calibration_rows([(model_name, item)])

            if not rows:
                return

            rows_df = pd.DataFrame(rows, columns=self._s._CALIBRATION_COLUMNS)
            if self._s.data[data_key].empty:
                self._s.data[data_key] = rows_df
            else:
                self._s.data[data_key] = pd.concat(
                    [self._s.data[data_key], rows_df], ignore_index=True,
                )
        except DATA_EXCEPTIONS as e:
            display_name = get_model_display_name(model_name)
            self._log_error(
                e,
                f"evaluation: calibration data extraction for {display_name}",
                warn_user=True,
                user_message=(
                    f"Calibration data for {display_name} at "
                    f"t={eval_time} could not be stored: {e}"
                ),
            )
