"""Subpopulation Analysis report section.

The grouped forest / faceted AUC(t) plots are rendered via
``learner.subpopulation(group).plot_metric(style='forest')`` and
``.plot_metric_time()``. Renderer implementations live in
``mlsurv.viz._metrics_helpers.plot_grouped_forest`` and
``mlsurv.viz._metrics_time.plot_grouped_metric_time``. This section
is responsible for markdown assembly, cohort summary tables, and
routing the returned figures through the report's ``FigureInfo``
machinery — no plot assembly happens here.
"""

import html as html_module
from typing import Any, Dict, List, Optional, Tuple

from mlsurv.reports.sections.base import (
    ALL_EXCEPTIONS, BaseSection, DataInfo, FigureInfo,
)


class SubpopulationAnalysisSection(BaseSection):
    """Per-subpopulation performance breakdown with forest plot."""

    name = "subpopulation_analysis"
    title = "Subpopulation Analysis"

    def is_available(self) -> bool:
        return bool(self.learner.subpopulation_results)

    def _get_bootstrap_info(self) -> Tuple[Optional[int], Optional[float]]:
        """Return (n_bootstrap, ci_level) from bootstrap result, or (None, None)."""
        try:
            boot = self.learner.bootstrap_result
            if boot is None:
                bench = self.learner.benchmark_result
                if bench is not None:
                    eval_obj = getattr(bench, 'evaluation', None)
                    if eval_obj is not None:
                        boot = getattr(eval_obj, 'bootstrap', None)
            if boot is None:
                return None, None
            n_iter = None
            if hasattr(boot, 'iterations') and boot.iterations is not None:
                iters = boot.iterations
                if 'iteration' in iters.columns:
                    n_iter = iters['iteration'].nunique()
            return n_iter, boot.ci_level
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(exc, f"{self.name}: bootstrap info extraction")
            return None, None

    def generate(self) -> Dict[str, Any]:
        sg_results = self.learner.subpopulation_results
        md_lines: List[str] = []
        figures: List[tuple] = []
        figure_infos: List[FigureInfo] = []
        data_infos: List[DataInfo] = []

        n_boot, ci_level = self._get_bootstrap_info()
        # None when extraction failed: downstream prose then says "bootstrap
        # confidence intervals" without asserting a level we don't know.
        ci_pct = f"{ci_level * 100:g}" if ci_level is not None else None

        md_lines.append(
            "This section evaluates model discrimination within patient "
            "subpopulations defined by categorical variables. Per-subpopulation "
            "metrics are computed from the same test-set risk scores "
            "used in the main evaluation — no re-prediction or re-fitting "
            "is performed. This isolates the effect of patient subpopulation "
            "composition on model performance."
        )
        md_lines.append("")
        if n_boot:
            md_lines.append(
                f"Bootstrap confidence intervals are computed from "
                f"**{n_boot}** resampled iterations within each subpopulation "
                f"and displayed as point estimate ({ci_pct}% CI). "
                f"Subpopulations with fewer than 30 events may produce "
                f"unstable estimates."
            )
            md_lines.append("")

        for group_var, sg_result in sg_results.items():
            self._render_subpopulation(
                group_var, sg_result, ci_pct,
                md_lines, figures, figure_infos, data_infos,
            )

        return {
            'markdown': "\n".join(md_lines),
            'figures': figures,
            'figure_info': figure_infos,
            'data_info': data_infos,
        }

    def _render_subpopulation(self, group_var, sg_result, ci_pct,
                         md_lines, figures, figure_infos, data_infos):
        """Render one subpopulation variable into the report accumulators."""
        md_lines.append(f"### Subpopulation: {group_var}")
        md_lines.append("")

        # Warnings — consolidated into a single callout block
        if sg_result.warnings:
            md_lines.append('<!-- MLSURV_RAW_HTML -->')
            md_lines.append('<div class="mlsurv-callout-info">')
            md_lines.append(
                '  <strong>&#9888; Small Subpopulation Warnings:</strong>'
            )
            md_lines.append('  <ul>')
            for w in sg_result.warnings:
                md_lines.append(
                    f'    <li>{html_module.escape(w)}</li>'
                )
            md_lines.append('  </ul>')
            md_lines.append('</div>')
            md_lines.append('<!-- /MLSURV_RAW_HTML -->')
            md_lines.append('')

        # Cohort summary table (compact, centered — reuses feature-summary CSS)
        md_lines.append("#### Cohort Summary")
        md_lines.append("")
        md_lines.append(
            "The table below summarizes the test-set composition for each "
            f"*{group_var}* subpopulation, including the number of observations "
            "and events."
        )
        md_lines.append("")
        summary_display = sg_result.summary.copy()
        summary_display.columns = [
            'Subpopulation', 'N', 'Events', 'Event Rate'
        ]
        summary_display['Event Rate'] = summary_display['Event Rate'].map(
            lambda x: f"{x:.1%}"
        )
        md_lines.append('<!-- MLSURV_RAW_HTML -->')
        header = ''.join(
            f'<th>{html_module.escape(str(c))}</th>'
            for c in summary_display.columns
        )
        rows_html = []
        for _, row in summary_display.iterrows():
            cells = ''.join(
                f'<td>{html_module.escape(str(v))}</td>' for v in row
            )
            rows_html.append(f'<tr>{cells}</tr>')
        md_lines.append(
            f'<table class="feature-summary">'
            f'<thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody></table>'
        )
        md_lines.append('<!-- /MLSURV_RAW_HTML -->')
        md_lines.append("")

        # Store summary data
        self.data[f'subpopulation_summary_{group_var}'] = sg_result.summary
        data_infos.append(DataInfo(
            data_key=f'subpopulation_summary_{group_var}',
            description=f"Subpopulation cohort summary for {group_var}",
        ))

        # Store metrics data (table removed — forest plot shows the same info)
        metrics_df = sg_result.metrics.copy()
        self.data[f'subpopulation_metrics_{group_var}'] = metrics_df
        data_infos.append(DataInfo(
            data_key=f'subpopulation_metrics_{group_var}',
            description=f"Per-subpopulation metrics for {group_var}",
        ))

        # Forest plot — delegated to SubpopulationView.plot_metric
        fig = self._build_forest_plot(group_var)
        if fig is not None:
            md_lines.append("#### Performance Comparison")
            md_lines.append("")
            ci_phrase = f"{ci_pct}% bootstrap" if ci_pct else "bootstrap"
            md_lines.append(
                "The forest plot below compares concordance across "
                f"*{group_var}* subpopulations. Each point represents a model's "
                f"concordance within that subpopulation, with {ci_phrase} "
                "confidence intervals where available. Wider intervals "
                "indicate less certainty, often due to smaller subpopulation sizes."
            )
            md_lines.append("")
            fname = f"subpopulation_forest_{group_var}.png"
            md_lines.append(f"![Subpopulation Forest Plot](images/{fname})")
            md_lines.append("")
            figures.append((fname, fig))
            figure_infos.append(FigureInfo(
                figure_id=f"subpopulation_forest_{group_var}",
                title=f"Subpopulation Forest Plot ({group_var})",
                data_description=(
                    f"Per-subpopulation concordance with {ci_phrase} CIs"
                ),
                learner_code=(
                    f"learner.subpopulation({group_var!r})"
                    f".plot_metric(style='forest')"
                ),
            ))

        # AUC(t) line plot per subpopulation — delegated to
        # SubpopulationView.plot_metric_time
        auc_fig = self._build_auc_t_plot(group_var)
        if auc_fig is not None:
            md_lines.append("#### Time-Dependent Discrimination")
            md_lines.append("")
            has_boot_auc = (
                sg_result.auc_t_bootstrap is not None
                and not sg_result.auc_t_bootstrap.empty
            )
            ribbon_note = (
                " Shaded ribbons show ±1 standard deviation computed "
                "from bootstrap resampling."
                if has_boot_auc else ""
            )
            md_lines.append(
                "Time-dependent AUC curves show how each model's "
                "discriminative ability varies over time within each "
                f"*{group_var}* subpopulation. Higher curves indicate better "
                f"discrimination at that timepoint.{ribbon_note}"
            )
            md_lines.append("")
            fname = f"subpopulation_auc_t_{group_var}.png"
            md_lines.append(f"![Subpopulation AUC(t)](images/{fname})")
            md_lines.append("")
            figures.append((fname, auc_fig))
            figure_infos.append(FigureInfo(
                figure_id=f"subpopulation_auc_t_{group_var}",
                title=f"Subpopulation AUC(t) ({group_var})",
                data_description="Per-subpopulation time-dependent AUC",
                learner_code=(
                    f"learner.subpopulation({group_var!r}).plot_metric_time()"
                ),
            ))

        if sg_result.auc_t is not None and len(sg_result.auc_t) > 0:
            self.data[f'subpopulation_auc_t_{group_var}'] = sg_result.auc_t
            data_infos.append(DataInfo(
                data_key=f'subpopulation_auc_t_{group_var}',
                description=f"Per-subpopulation AUC(t) for {group_var}",
            ))

    def _build_forest_plot(self, group_var) -> Optional[Any]:
        """Render the grouped forest plot via :class:`SubpopulationView`.

        Thin proxy around
        ``learner.subpopulation(group_var).plot_metric(style='forest')``
        — kept as a method so the Programmatic Access snippet
        (``learner_code``) is the obvious single-line form while the
        section-level error handling (``ALL_EXCEPTIONS``) still wraps
        the call.
        """
        try:
            return self.learner.subpopulation(group_var).plot_metric(
                style="forest",
            )
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, f"{self.name}: forest plot for {group_var}",
            )
            return None

    def _build_auc_t_plot(self, group_var) -> Optional[Any]:
        """Render the faceted AUC(t) plot via :class:`SubpopulationView`.

        Thin proxy around
        ``learner.subpopulation(group_var).plot_metric_time()``.

        Short-circuits to ``None`` when the stored
        ``SubpopulationResult.auc_t`` is missing or empty — the view
        would raise ``ValueError`` in that case, which
        ``_log_recoverable_exception`` would write to the log on every
        report build (noisy but harmless). A ``None`` return is the
        canonical "nothing to plot, skip the figure" signal.
        """
        sg_result = self.learner.subpopulation_results.get(group_var)
        if (
            sg_result is None
            or sg_result.auc_t is None
            or sg_result.auc_t.empty
        ):
            return None
        try:
            return self.learner.subpopulation(group_var).plot_metric_time()
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, f"{self.name}: AUC(t) plot for {group_var}",
            )
            return None

