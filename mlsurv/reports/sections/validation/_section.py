"""ValidationSection orchestrator — delegates to subsection classes and helpers."""

from typing import Any, Dict, Optional

import pandas as pd

from mlsurv.reports._text import title_if_lower
from mlsurv.reports.sections.base import (
    ALL_EXCEPTIONS,
    BaseSection,
    DATA_EXCEPTIONS,
    DataInfo,
    FigureInfo,
)
from mlsurv.reports.sections.validation._helpers import (
    bold_best_values,
    cohort_key as _cohort_key,
)


def _has_auc_t(val_result) -> bool:
    """Check if a validation result has non-empty AUC(t) data."""
    return (
        val_result.evaluation is not None
        and val_result.evaluation.auc_t is not None
        and len(val_result.evaluation.auc_t) > 0
    )


def _has_risk_scores(val_result) -> bool:
    """Check if a validation result has risk scores for stratification."""
    return (
        val_result.test_set is not None
        and 'risk_score' in val_result.test_set.columns
    )


class ValidationSection(BaseSection):
    """External validation section showing model performance on external cohorts.

    Shows:
    - Performance metrics on external validation datasets
    - Cohort summary (n_samples, n_events, event_rate)
    - Distribution comparison (Table 1) between dev and validation cohorts
    - Internal vs. external performance drop analysis
    - Bootstrap confidence intervals (if computed)
    - Time-dependent AUC curves
    - Calibration plots
    - Survival stratification with RMST on external cohorts

    Maps to API method: validate()

    Supports multiple external cohorts - each validate() call accumulates
    results in the learner.validation_results dict.

    This section only appears if validate() has been called with external data.
    """

    name = "validation"
    title = "External Validation"

    @staticmethod
    def _cohort_key(name: str) -> str:
        """Derive a filesystem/dict-safe key from a cohort name."""
        return _cohort_key(name)

    def __init__(
        self,
        learner,
        validation_results: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(learner)
        self.validation_results = validation_results

    def is_available(self) -> bool:
        """Check if external validation has been performed."""
        if self.validation_results:
            return True

        if self.learner.validation_results:
            self.validation_results = self.learner.validation_results
            return True

        return False

    def generate(self) -> Dict[str, Any]:
        """Generate external validation section content."""
        from mlsurv.reports.sections.validation._bootstrap import add_combined_bootstrap_summary
        from mlsurv.reports.sections.validation._calibration import _CalibrationSubsection
        from mlsurv.reports.sections.validation._discrimination import _DiscriminationSubsection
        from mlsurv.reports.sections.validation._performance_drop import add_performance_drop
        from mlsurv.reports.sections.validation._stratification import _StratificationSubsection
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        md_lines = []
        figures = []
        figure_info = []
        data_info: list[DataInfo] = []

        if not self.is_available():
            return {'markdown': '', 'figures': [], 'figure_info': [], 'data_info': []}

        # Section description
        md_lines.append(
            "External validation assesses model performance on independent cohorts "
            "that were not used during model development or internal evaluation. "
            "This is the gold standard for evaluating model generalizability and "
            "is required by TRIPOD+AI guidelines for clinical prediction models. "
            "Performance on external data often differs from internal test set results "
            "due to differences in patient populations, treatment protocols, or data collection."
        )
        md_lines.append("")

        # Methodology note when any model carries an attached calibrator.
        if any(
            self.learner._has_calibrator(n) for n in (self.learner.models or {})
        ):
            md_lines.append(
                "**S(t\\|x) source.** Models with an attached calibrator "
                "(`train(calibrate=...)`) report external metrics from the "
                "calibrated S(t\\|x) — the deployed model. The calibrator is a "
                "fixed transform fit on the development cohort, applied here to "
                "the external rows it never saw (leakage-free), so these metrics "
                "show whether the development-cohort calibration transports. Pass "
                "``validate(use_recalibrated=False)`` for raw external metrics."
            )
            md_lines.append("")

        # Cohort listing
        cohort_names = list(self.validation_results.keys())
        n_cohorts = len(cohort_names)

        if n_cohorts == 1:
            md_lines.append(f"**External cohort:** {cohort_names[0]}")
        else:
            md_lines.append(f"**External cohorts ({n_cohorts}):** {', '.join(cohort_names)}")
        md_lines.append("")

        # Pre-compute flags
        has_bootstrap, has_xval, auct_cohorts, cal_cohorts, strat_cohorts = (
            self._compute_flags()
        )

        # Section outline (TOC)
        toc_entries = self._build_toc(
            has_bootstrap, has_xval, auct_cohorts, cal_cohorts, strat_cohorts,
        )
        md_lines.extend(self._build_section_outline(toc_entries))

        # 1. Cohort summary table
        md_lines.append("### Cohort Summary")
        md_lines.append("")
        self._add_cohort_summary_table(md_lines)

        # 2. Distribution comparison (Table 1)
        add_distribution_comparison(self, md_lines)

        # 3. Performance metrics heatmap (replaces markdown table)
        md_lines.append("### Performance Metrics")
        md_lines.append("")
        self._add_performance_heatmap(md_lines, figures, figure_info)

        # 4. Internal vs. External performance drop
        add_performance_drop(self, md_lines)

        # 5. Bootstrap confidence intervals
        if has_bootstrap:
            md_lines.append("### Bootstrap Confidence Intervals")
            md_lines.append("")
            add_combined_bootstrap_summary(self, md_lines, figures, figure_info)

        # 6–8. Per-cohort subsections
        disc_sub = _DiscriminationSubsection(self)
        cal_sub = _CalibrationSubsection(self)
        strat_sub = _StratificationSubsection(self)

        for name in auct_cohorts:
            val_result = self.validation_results[name]
            md_lines.append(f"### Time-Dependent Discrimination: {name}")
            md_lines.append("")
            try:
                disc_sub.add(
                    md_lines, figures, figure_info,
                    val_result, cohort_name=name, cohort_key=self._cohort_key(name),
                )
            except ALL_EXCEPTIONS as e:
                self._log_recoverable_exception(
                    e, f"validation: discrimination for {name}",
                    warn_user=True,
                    user_message=f"Discrimination analysis omitted for {name}: {e}",
                )

        for name in cal_cohorts:
            val_result = self.validation_results[name]
            md_lines.append(f"### Calibration: {name}")
            md_lines.append("")
            try:
                cal_sub.add(
                    md_lines, figures, figure_info,
                    val_result, cohort_name=name, cohort_key=self._cohort_key(name),
                )
            except ALL_EXCEPTIONS as e:
                self._log_recoverable_exception(
                    e, f"validation: calibration for {name}",
                    warn_user=True,
                    user_message=f"Calibration analysis omitted for {name}: {e}",
                )

        for name in strat_cohorts:
            val_result = self.validation_results[name]
            md_lines.append(f"### Risk Stratification: {name}")
            md_lines.append("")
            try:
                strat_sub.add(
                    md_lines, figures, figure_info,
                    val_result.test_set, cohort_name=name,
                    cohort_key=self._cohort_key(name),
                )
            except ALL_EXCEPTIONS as e:
                self._log_recoverable_exception(
                    e, f"validation: stratification for {name}",
                    warn_user=True,
                    user_message=f"Stratification analysis omitted for {name}: {e}",
                )

        data_info.extend(
            self._build_cohort_data_info(auct_cohorts, cal_cohorts, strat_cohorts)
        )

        return {
            'markdown': '\n'.join(md_lines),
            'figures': figures,
            'figure_info': figure_info,
            'data_info': data_info,
        }

    # ------------------------------------------------------------------
    # DataInfo helpers
    # ------------------------------------------------------------------

    def _build_cohort_data_info(
        self,
        auct_cohorts: list,
        cal_cohorts: list,
        strat_cohorts: list,
    ) -> list:
        """Build DataInfo entries pointing at learner.validation_results.

        External validation has no dedicated ``learner.plot_validation(...)``
        wrapper, so the programmatic-access dropdown points users at the
        raw ``ValidationResult`` objects for each cohort via
        ``results_code``.
        """
        entries: list[DataInfo] = []

        # Guard every branch with a ``data_key in self.data`` check so
        # subsection-level failures (caught in ``generate()``'s per-cohort
        # try/except) don't leave dangling DataInfo entries whose
        # ``data_key`` points at a non-existent ``section.data`` row.
        #
        # Use repr(name) to embed cohort names as valid Python string
        # literals — survives cohort names containing quotes/apostrophes.
        for name in auct_cohorts:
            cohort_key = self._cohort_key(name)
            data_key = f'auct_{cohort_key}'
            if data_key not in self.data:
                continue
            entries.append(DataInfo(
                data_key=data_key,
                description=f'Time-dependent AUC for cohort "{name}"',
                results_code=(
                    f"auc_df = learner.validation_results"
                    f"[{name!r}].evaluation.auc_t"
                ),
            ))

        for name in cal_cohorts:
            cohort_key = self._cohort_key(name)
            data_key = f'calibration_{cohort_key}'
            if data_key not in self.data:
                continue
            entries.append(DataInfo(
                data_key=data_key,
                description=f'Calibration bins for cohort "{name}"',
                results_code=(
                    f"val_result = learner.validation_results[{name!r}]\n"
                    f"# X_val, y_val available on val_result for recomputing calibration"
                ),
            ))

        for name in strat_cohorts:
            cohort_key = self._cohort_key(name)
            for key_stem, label in [
                ('km', 'KM survival curves'),
                ('median', 'median survival'),
                ('rmst', 'RMST per risk group'),
                ('rmst_pairwise', 'pairwise RMST differences'),
            ]:
                data_key = f'{key_stem}_{cohort_key}'
                if data_key not in self.data:
                    continue
                entries.append(DataInfo(
                    data_key=data_key,
                    description=f'{label} for cohort "{name}"',
                    results_code=(
                        f"test_set = learner.validation_results"
                        f"[{name!r}].test_set  # raw risk groups"
                    ),
                ))

        return entries

    # ------------------------------------------------------------------
    # Flags & TOC
    # ------------------------------------------------------------------

    def _compute_flags(self) -> tuple:
        """Pre-compute availability flags for TOC and section gating.

        Returns
        -------
        tuple of (has_bootstrap, has_xval, auct_cohorts, cal_cohorts, strat_cohorts)
        """
        has_bootstrap = any(
            val_result.bootstrap is not None
            for val_result in self.validation_results.values()
        )
        has_xval = any(
            val_result.X_val is not None
            for val_result in self.validation_results.values()
        )
        auct_cohorts = [
            name for name, vr in self.validation_results.items()
            if _has_auc_t(vr)
        ]
        cal_cohorts = [
            name for name, vr in self.validation_results.items()
            if vr.X_val is not None
        ]
        strat_cohorts = [
            name for name, vr in self.validation_results.items()
            if _has_risk_scores(vr)
        ]
        return has_bootstrap, has_xval, auct_cohorts, cal_cohorts, strat_cohorts

    def _build_toc(
        self,
        has_bootstrap: bool,
        has_xval: bool,
        auct_cohorts: list,
        cal_cohorts: list,
        strat_cohorts: list,
    ) -> list:
        """Build TOC entries list from pre-computed flags."""
        toc_entries = [('Cohort Summary', 'cohort-summary')]
        if has_xval:
            toc_entries.append(('Cohort Characteristics', 'cohort-characteristics'))
        toc_entries.append(('Performance Metrics', 'performance-metrics'))
        if self.learner.evaluation_result is not None:
            toc_entries.append(
                ('Internal vs. External Performance', 'internal-vs-external-performance')
            )
        if has_bootstrap:
            toc_entries.append(
                ('Bootstrap Confidence Intervals', 'bootstrap-confidence-intervals')
            )
        for name in auct_cohorts:
            anchor = f"time-dependent-discrimination-{name.lower().replace(' ', '-')}"
            toc_entries.append((f'Time-Dependent Discrimination: {name}', anchor))
        for name in cal_cohorts:
            anchor = f"calibration-{name.lower().replace(' ', '-')}"
            toc_entries.append((f'Calibration: {name}', anchor))
        for name in strat_cohorts:
            anchor = f"risk-stratification-{name.lower().replace(' ', '-')}"
            toc_entries.append((f'Risk Stratification: {name}', anchor))
        return toc_entries

    # ------------------------------------------------------------------
    # Section-level helpers (kept in orchestrator)
    # ------------------------------------------------------------------

    def _add_cohort_summary_table(self, md_lines: list) -> None:
        """Add summary table with cohort characteristics."""
        try:
            table_data = []
            for name, val_result in self.validation_results.items():
                table_data.append({
                    'Cohort': name,
                    'N': val_result.n_samples,
                    'Events': val_result.n_events,
                    f'{title_if_lower(self.outcome_name)} Rate': f"{val_result.event_rate:.1%}",
                })

            if table_data:
                summary_df = pd.DataFrame(table_data)
                md_lines.append(self._format_table(summary_df))
                md_lines.append("")
                self.data['cohort_summary'] = summary_df
        except DATA_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "validation: cohort summary table",
                warn_user=True,
                user_message=f"Could not generate cohort summary table: {e}",
            )

    def _add_performance_heatmap(
        self, md_lines: list, figures: list, figure_info: list,
    ) -> None:
        """Add faceted metric heatmap for all validation cohorts.

        Builds one panel per cohort showing a model x metric heatmap with
        cross-cohort z-scoring (shared colorbar, red = better). Also stores
        tidy data in ``self.data['metrics']`` for programmatic access.
        """
        from mlsurv.viz._data_metrics import build_validation_heatmap_data
        from mlsurv.viz.metrics import plot_faceted_metric_heatmap

        try:
            # Assemble per-cohort wide frames + tidy data via the shared
            # builder (also backs learner.plot_validation_heatmap).
            cohort_dfs, metric_cols, heatmap_recal, tidy_df = (
                build_validation_heatmap_data(self.validation_results)
            )

            # Always store tidy data for programmatic access
            if tidy_df is not None:
                self.data['metrics'] = tidy_df

            if not cohort_dfs:
                md_lines.append(
                    "*No performance metrics available. Ensure evaluate() "
                    "was called on each validation result.*"
                )
                md_lines.append("")
                return

            # Heatmap needs >= 2 metrics AND >= 2 cohorts for meaningful
            # cross-cohort z-score comparison. Fall back to a markdown
            # metrics table in either single-metric or single-cohort case.
            # (The builder carries per-row "(recal)" labeling in tidy_df for
            # the mixed-cohort case the fallback table renders.)
            if len(metric_cols) < 2 or len(cohort_dfs) < 2:
                if tidy_df is not None:
                    self._emit_validation_metrics_table(md_lines)
                return

            fig = plot_faceted_metric_heatmap(
                cohort_dfs,
                metrics=metric_cols,
                title="Validation Performance (z-scored, red = better)",
                recalibrated_models=None,
            )
            figures.append(('validation_metrics_heatmap.png', fig))
            md_lines.append(
                "![Validation Metrics Heatmap](images/validation_metrics_heatmap.png)"
            )
            md_lines.append("")

            figure_info.append(FigureInfo(
                figure_id='validation_metrics_heatmap',
                title='Validation Performance Heatmap',
                data_description=(
                    'Cross-cohort z-scored model x metric heatmap '
                    'for external validation cohorts'
                ),
                data_key='metrics',
                learner_code="fig = learner.plot_validation_heatmap()",
            ))
        except ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "validation: performance metrics heatmap",
                warn_user=True,
                user_message=f"Could not generate performance heatmap: {e}",
            )
            md_lines.append(f"*[Performance heatmap unavailable: {e}]*")
            md_lines.append("")

    def _emit_validation_metrics_table(self, md_lines: list) -> None:
        """Emit a markdown metrics table in place of the heatmap."""
        tidy_df = self.data['metrics']
        display_df = self._bold_best_values(tidy_df, group_col='Cohort')
        md_lines.append(self._format_table(display_df))
        md_lines.append("")
        md_lines.append("*Asterisk (*) indicates lower is better.*")
        md_lines.append("")

    def _bold_best_values(
        self,
        df: pd.DataFrame,
        group_col: str = None,
    ) -> pd.DataFrame:
        """Bold best values in numeric columns, optionally grouped."""
        return bold_best_values(df, group_col=group_col)

