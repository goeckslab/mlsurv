"""ROC Curves subsection (between Discrimination and Calibration)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsurv.reports.sections.base import ALL_EXCEPTIONS, FigureInfo
from mlsurv.reports.sections.evaluation._base import _BaseSubsection
from mlsurv.reports._helpers import extract_eval_times_from_auc_t


class _ROCSubsection(_BaseSubsection):
    """Time-dependent ROC curves with eval-time dropdown."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        er = self._s.eval_result
        if er is None:
            return False
        if er.auc_t is None or len(er.auc_t) == 0:
            return False
        test_set = getattr(er, "test_set", None)
        if test_set is None or len(test_set) == 0:
            return False
        if self._s.learner.y_train is None or self._s.learner.y_test is None:
            return False
        return True

    def add(self, md_lines: list, figures: list, figure_info: list) -> None:
        md_lines.append(
            "The time-dependent ROC curve plots sensitivity versus "
            "1\u2009\u2212\u2009specificity at horizon *t*, sweeping the "
            "risk-score threshold; its area is the **AUC(*t*) reported "
            "above**. Curves bowing toward the upper-left indicate "
            "better discrimination. Censored subjects are reweighted by "
            "IPCW (same pseudopopulation construction as AUC(*t*)) "
            "rather than dropped. Use the dropdown to inspect different "
            "evaluation times."
        )
        md_lines.append("")

        try:
            self._add_roc_figure(md_lines, figures, figure_info)
        except ALL_EXCEPTIONS as e:
            self._log_error(
                e,
                "evaluation: ROC curve generation",
                warn_user=True,
                user_message=f"Could not generate ROC curves: {e}",
            )
            md_lines.append(f"*[ROC curves unavailable: {e}]*")
            md_lines.append("")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_roc_figure(
        self, md_lines: list, figures: list, figure_info: list,
    ) -> None:
        from mlsurv.viz._metrics_roc import (
            assemble_roc_curves_with_references,
            bootstrap_roc_curves,
            plot_roc_interactive,
        )

        eval_times = extract_eval_times_from_auc_t(self._s.eval_result.auc_t)
        if not eval_times:
            md_lines.append("*[ROC curves unavailable: no valid evaluation times]*")
            md_lines.append("")
            return

        risk_scores = self._extract_risk_scores()
        if not risk_scores:
            md_lines.append(
                "*[ROC curves unavailable: no risk scores in evaluation results]*"
            )
            md_lines.append("")
            return

        y_train = self._s.learner.y_train
        y_test = self._s.learner.y_test

        # Shared per-horizon ROC assembly (no references on the evaluation
        # path, so the helper runs the model-only branch).
        res = assemble_roc_curves_with_references(
            y_train, y_test, risk_scores, eval_times,
            catch=(ValueError, RuntimeError),
            on_error=lambda exc, kind, name, t: self._log_error(
                exc, f"evaluation: ROC at t={t}",
            ),
        )
        roc_data = res.roc_data
        all_roc_dfs = res.all_roc_dfs

        if not roc_data:
            md_lines.append(
                "*[ROC curves unavailable: computation failed at all time points]*"
            )
            md_lines.append("")
            return

        self._s.data["roc"] = pd.concat(all_roc_dfs, ignore_index=True)

        # Compute bootstrap CI bands when predictions are available
        ci_data = None
        n_boot_iters = 0
        ci_attempted = bool(self._s.bootstrap_results)
        if ci_attempted:
            ci_data, n_boot_iters = self._compute_roc_ci(
                bootstrap_roc_curves, y_train, y_test, eval_times,
            )

        if ci_data:
            md_lines.append(
                "Shaded bands show 95\u2009% pointwise bootstrap "
                "confidence intervals."
            )
            if n_boot_iters < 20:
                md_lines.append(
                    f" *Note: only {n_boot_iters} bootstrap iterations "
                    f"were used — CI bands may not fully enclose the ROC "
                    f"curves. Increase `n_bootstrap` for tighter bands.*"
                )
            md_lines.append("")
        elif ci_attempted:
            md_lines.append(
                "*[Bootstrap CI bands could not be computed because "
                "per-iteration predictions were not stored on the "
                "bootstrap result. Re-run with "
                "`learner.bootstrap(include_predictions=True)` to "
                "enable ROC CI bands; showing plain ROC curves.]*"
            )
            md_lines.append("")

        fig = plot_roc_interactive(roc_data, ci_data=ci_data)
        fig_filename = "roc_curves.png"
        figures.append((fig_filename, fig))
        md_lines.append("<!-- MLSURV_RAW_HTML -->")
        md_lines.append(
            f'<div class="figure-placeholder" data-figure="{fig_filename}"></div>'
        )
        md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id="roc_curves",
            title="Time-Dependent ROC Curves",
            data_description="Tidy DataFrame with ROC curve data (model, fpr, tpr, auc, eval_time)",
            data_key="roc",
            learner_code="fig = learner.plot_roc()",
        ))

    def _compute_roc_ci(self, bootstrap_roc_curves_fn, y_train, y_test, eval_times):
        """Extract bootstrap predictions and compute ROC CI bands.

        Returns ``(ci_data, n_iterations)`` where *ci_data* is the CI
        dict (or ``None``) and *n_iterations* is the bootstrap iteration
        count (0 when unavailable).
        """
        if not self._s.bootstrap_results:
            return None, 0

        pred_dfs = []
        for boot_result in self._s.bootstrap_results.values():
            preds = getattr(boot_result, "predictions", None)
            if preds is not None and len(preds) > 0:
                pred_dfs.append(preds)

        if not pred_dfs:
            return None, 0

        try:
            all_predictions = pd.concat(pred_dfs, ignore_index=True)
            n_iters = int(all_predictions["iteration"].nunique())
            X_test_index = self._s.learner.X_test.index
            ci_data = bootstrap_roc_curves_fn(
                y_train, y_test, all_predictions, X_test_index, eval_times,
            )
            return ci_data, n_iters
        except ALL_EXCEPTIONS as exc:
            self._log_error(
                exc,
                "evaluation: ROC bootstrap CI bands",
                warn_user=True,
                user_message=(
                    f"Could not compute ROC confidence intervals: {exc}. "
                    f"Falling back to plain ROC curves."
                ),
            )
            return None, 0

    def _extract_risk_scores(self) -> dict:
        return _extract_risk_scores(self._s.eval_result.test_set)


def _extract_risk_scores(test_set: pd.DataFrame) -> dict:
    """Extract risk scores from a test_set DataFrame, keyed by model_type.

    Models with non-finite scores are skipped with a warning.
    """
    import warnings
    from mlsurv._warnings import InternalWarning

    risk_scores = {}
    for model_type in test_set["model_type"].unique():
        model_df = test_set[test_set["model_type"] == model_type]
        scores = model_df["risk_score"].values
        if len(scores) == 0:
            continue
        if not np.all(np.isfinite(scores)):
            n_bad = int((~np.isfinite(scores)).sum())
            warnings.warn(
                f"ROC: model '{model_type}' has {n_bad} non-finite risk "
                f"score(s); skipping.",
                InternalWarning, stacklevel=2,
            )
            continue
        risk_scores[model_type] = scores
    return risk_scores
