"""5.2 Overview subsection (test-set metrics, plots, heatmaps)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsurv.constants import (
    ALL_METRICS,
    LOWER_IS_BETTER,
    get_concordance_column,
    get_metric_display_name,
)
from mlsurv.reports.sections._metric_glossary import metric_glossary_block
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, FigureInfo
from mlsurv.viz.metrics import plot_metric_heatmap

from mlsurv.reports.sections.evaluation._base import _BaseSubsection
from mlsurv.reports.sections.evaluation._helpers import format_metric_with_ci


def _build_heatmap_dataframe(trained_models, get_metric_value_fn):
    """Build wide-format heatmap DataFrame from model evaluations.

    Returns None if fewer than 2 models have evaluation results.
    """
    rows = []
    for name, result in trained_models.items():
        if not result.evaluation:
            continue
        concordance = getattr(result.evaluation, 'concordance', None)
        if concordance is None:
            continue
        row = {'model_type': name}
        for metric_name in ALL_METRICS:
            col = get_concordance_column(metric_name)
            val = get_metric_value_fn(concordance, col)
            if val is not None:
                row[metric_name] = val
        rows.append(row)

    if len(rows) < 2:
        return None
    return pd.DataFrame(rows)


class _PerformanceSubsection(_BaseSubsection):
    """Test-set performance table with optional bootstrap CIs."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return True

    def add(self, md_lines: list, figures: list, figure_info: list) -> None:
        trained_models = self._s._get_all_models()

        md_lines.append(
            "The table below shows model performance on the hold-out test set. "
            "Models are shown with their tuning status (Baseline = default parameters, "
            "Tuned = optimized hyperparameters)."
        )
        md_lines.append("")

        md_lines.append(metric_glossary_block())
        md_lines.append("")

        ci_pct = "95"
        if self._s.bootstrap_results:
            first_result = next(iter(self._s.bootstrap_results.values()))
            ci_pct = f"{first_result.ci_level * 100:g}"
            if first_result.iterations is not None and not first_result.iterations.empty:
                n_bootstrap = first_result.iterations['iteration'].nunique()
                md_lines.append(
                    f"Each cell shows the full-test-set point estimate followed by a "
                    f"{ci_pct}% bootstrap percentile confidence interval ({n_bootstrap} iterations), "
                    f"formatted as `point [CI_lower, CI_upper]`."
                )
        md_lines.append("")

        table_data = []
        for name, result in trained_models.items():
            if not result.evaluation:
                continue
            table_data.append(self._build_performance_row(name, result))

        self._store_bootstrap_tidy_data()

        if table_data:
            perf_df = pd.DataFrame(table_data)
            perf_df = self._sort_performance_table(perf_df)
            perf_df = self._s._bold_best_values(perf_df)

            md_lines.append(self._format_table(perf_df))
            md_lines.append("")
            md_lines.append(
                "<!-- MLSURV_RAW_HTML -->"
            )
            md_lines.append(
                f"<p><em>Asterisk (*) indicates lower is better. "
                f"Values with brackets show {ci_pct}% CI.</em></p>"
            )
            md_lines.append(
                "<!-- /MLSURV_RAW_HTML -->"
            )
            md_lines.append("")

            self._add_ipcw_support_caveat(md_lines)
            self._add_truncation_footnote(md_lines, trained_models)

            self._s.data['test_performance'] = pd.DataFrame(table_data)

        # Multi-metric forest plot (optional — warn on failure, never crash report)
        self._add_forest_plots(md_lines, figures, figure_info)

        # Metric summary heatmap (z-scored, direction-aware; >=2 models only)
        self._add_metrics_heatmap(md_lines, figures, figure_info)

        # Bootstrap metric distribution rainclouds (selected CV metric only)
        self._add_bootstrap_rainclouds(md_lines, figures, figure_info)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_ipcw_support_caveat(self, md_lines) -> None:
        """Append a caveat beneath the table when the IPCW support-violation
        flag fired at evaluate()/bootstrap() time.

        Reads the ``ipcw_support_flag`` field via the exempt defensive getattr
        (authoritative; no public accessor); we do not fall
        back to per-ModelResult attributes because those would resurrect stale
        flags from prior violating runs.
        """
        learner = getattr(self._s, 'learner', None)
        flag = (getattr(learner._state, 'ipcw_support_flag', None)
                if learner is not None else None)
        if flag is None:
            return
        tau = flag.get('tau_used', flag.get('tau'))
        tau_str = f"{tau:.1f}" if isinstance(tau, (int, float)) and tau is not None else "None"
        g_source = flag.get('g_hat_source', 'train')
        g_label = "validation-cohort" if g_source == 'test' else "training-cohort"

        # Distinguish the two failure modes. When the G_hat-fitting cohort is
        # too thin to estimate the censoring distribution, the IPCW correction
        # is effectively off (weights collapse to 1) rather than unstable, so
        # the "wide CIs = instability" framing would mislabel it.
        n_ev = flag.get('n_g_events')
        n_cen = flag.get('n_g_censored')
        floor = flag.get('min_ipcw_support', 10)
        composition_thin = (
            n_ev is not None and n_cen is not None and min(n_ev, n_cen) < floor
        )
        closing = (
            'The G_hat-fitting cohort is too thin to estimate the censoring '
            'distribution, so the IPCW correction is effectively off and these '
            'rows reduce toward their unweighted forms; prefer Harrell/Antolini '
            'on this cohort.'
            if composition_thin else
            'Wide IPCW CIs reflect weight instability rather than model '
            'discrimination; prefer Harrell/Antolini on this cohort.'
        )
        md_lines.append("<!-- MLSURV_RAW_HTML -->")
        md_lines.append(
            '<p class="mlsurv-metric-caveat"><strong>&#9888; IPCW caveat:</strong> '
            f'on this cohort the IPCW-weighted rows (Uno C-index, AUC(t), Brier, IBS) '
            f'were evaluated with tau={tau_str} using the {g_label} G_hat because '
            f'the censoring distribution support was violated '
            f'(see Limitations &rarr; IPCW Training-Support Violation). '
            + closing + '</p>'
        )
        md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")

    def _add_truncation_footnote(self, md_lines, trained_models) -> None:
        """Append IPCW training-range cap footnote when truncation applied.

        All models for a given cohort share the same y_test, so the
        truncation diagnostic on the first model carrying it is canonical.
        """
        t_max = None
        n_beyond = 0
        for result in trained_models.values():
            ev = getattr(result, 'evaluation', None)
            if ev is None:
                continue
            cand = getattr(ev, 'truncation_t_max', None)
            if cand is not None:
                t_max = cand
                n_beyond = getattr(ev, 'truncation_n_beyond', 0)
                break
        if t_max is None or n_beyond <= 0:
            return
        md_lines.append("<!-- MLSURV_RAW_HTML -->")
        md_lines.append(
            '<p class="mlsurv-metric-caveat"><em>Uno’s C-index, AUC(t), '
            f'and IBS computed on <code>y_test</code> truncated to '
            f'<code>t_max={t_max:.1f}</code> ({n_beyond} obs censored). '
            'Harrell’s C uses the full test set.</em></p>'
        )
        md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")

    def _add_forest_plots(self, md_lines, figures, figure_info):
        """Add multi-metric forest plot via learner.plot_metric."""
        if self._s.bootstrap_results:
            self._add_forest_from_bootstrap(md_lines, figures, figure_info)
        else:
            self._add_forest_from_cv(md_lines, figures, figure_info)

    def _get_bootstrap_plot_metrics(self) -> list:
        """Return ordered list of metrics available in bootstrap summary, or []."""
        first_result = next(iter(self._s.bootstrap_results.values()))
        if first_result.summary is None or first_result.summary.empty:
            return []
        available = set(first_result.summary['metric'].unique())
        return [m for m in ALL_METRICS if m in available]

    def _add_forest_from_bootstrap(self, md_lines, figures, figure_info):
        """Build forest panel from bootstrap data."""
        plot_metrics = self._get_bootstrap_plot_metrics()
        if not plot_metrics:
            return

        try:
            if len(plot_metrics) == 1:
                fig = self._s.learner.plot_metric(
                    style="forest", source="bootstrap",
                    metric=plot_metrics[0],
                    title="Test-Set Metric Estimate",
                )
            else:
                fig = self._s.learner.plot_metric(
                    style="forest", source="bootstrap",
                    metric=plot_metrics,
                    title="Test-Set Metric Estimates",
                )
            if fig is None:
                return

            figures.append(('metric_forest.png', fig))
            md_lines.append("![Metric Forest Plot](images/metric_forest.png)")
            md_lines.append("")

            metric_list = ', '.join(
                get_metric_display_name(m) for m in plot_metrics
            )
            figure_info.append(FigureInfo(
                figure_id='metric_forest',
                title='Test-Set Metric Estimates',
                data_description=f'Bootstrap summary data for {metric_list}',
                data_key='test_performance',
                learner_code=(
                    "fig = learner.plot_metric(\n"
                    "    style='forest', source='bootstrap',\n"
                    f"    metric={plot_metrics!r},\n"
                    ")"
                ),
            ))
        except ALL_EXCEPTIONS as exc:
            self._log_error(
                exc,
                "evaluation: metric forest plot generation",
                warn_user=True,
                user_message=f"Could not generate metric forest plot: {exc}",
            )
            md_lines.append(f"*[Metric forest plot unavailable: {exc}]*")
            md_lines.append("")

    def _add_forest_from_cv(self, md_lines, figures, figure_info):
        """Fallback: single-metric forest from CV data."""
        try:
            fig = self._s.learner.plot_metric(
                style="forest", source="cv",
                title="CV Metric Estimate",
            )
            if fig is None:
                return

            figures.append(('metric_forest.png', fig))
            md_lines.append("![Metric Forest Plot](images/metric_forest.png)")
            md_lines.append("")
        except ALL_EXCEPTIONS as exc:
            self._log_error(
                exc,
                "evaluation: CV metric forest plot generation",
                warn_user=True,
                user_message=f"Could not generate CV metric forest plot: {exc}",
            )
            md_lines.append(f"*[CV metric forest plot unavailable: {exc}]*")
            md_lines.append("")

    def _add_metrics_heatmap(self, md_lines, figures, figure_info):
        """Add model × metric heatmap (z-scored, direction-aware)."""
        heatmap_df = _build_heatmap_dataframe(self._s._get_all_models(), self._get_metric_value)
        if heatmap_df is None:
            return

        metric_cols = [c for c in heatmap_df.columns if c != 'model_type']
        if len(metric_cols) < 2:
            return

        try:
            fig = plot_metric_heatmap(
                heatmap_df,
                metrics=metric_cols,
                title="Metric Summary (z-scored, red = better)",
            )
            figures.append(('metrics_heatmap.png', fig))
            md_lines.append("![Metrics Heatmap](images/metrics_heatmap.png)")
            md_lines.append("")

            self._s.data['metrics_heatmap'] = heatmap_df
            figure_info.append(FigureInfo(
                figure_id='metrics_heatmap',
                title='Metric Summary Heatmap',
                data_description='Wide-format model × metric values',
                data_key='metrics_heatmap',
                learner_code="fig = learner.plot_metric_heatmap()",
            ))
        except ALL_EXCEPTIONS as exc:
            self._log_error(
                exc,
                "evaluation: metrics heatmap generation",
                warn_user=True,
                user_message=f"Could not generate metrics heatmap: {exc}",
            )
            md_lines.append(f"*[Metrics heatmap unavailable: {exc}]*")
            md_lines.append("")

    def _build_performance_row(self, name: str, result) -> dict:
        # Indicator column showing whether a calibrator is attached for this
        # model (the reported metrics already reflect the calibrated S(t|x)).
        any_calibrated = any(
            self._s.learner._has_calibrator(n)
            for n in self._s._get_all_models()
        )
        row = {
            'Model': self._s._model_label(name),
            'Status': "Tuned" if result.is_tuned else "Baseline",
        }
        if any_calibrated:
            row['Calibrated'] = (
                'Yes' if self._s.learner._has_calibrator(name) else 'No'
            )

        boot_summary = None
        if self._s.bootstrap_results and name in self._s.bootstrap_results:
            boot_summary = self._s.bootstrap_results[name].summary

        concordance = getattr(result.evaluation, 'concordance', None)
        for metric_name in ALL_METRICS:
            col_name_conc = get_concordance_column(metric_name)
            test_val = self._get_metric_value(concordance, col_name_conc)
            if test_val is None:
                continue

            col_name = get_metric_display_name(metric_name)
            if metric_name in LOWER_IS_BETTER:
                col_name += " *"

            row[col_name] = format_metric_with_ci(test_val, boot_summary, metric_name)

        return row

    def _store_bootstrap_tidy_data(self) -> None:
        if not self._s.bootstrap_results:
            return

        bootstrap_rows = []
        for model_name, boot_result in self._s.bootstrap_results.items():
            display_name = self._s._model_label(model_name)
            if boot_result.iterations is not None:
                for _, iter_row in boot_result.iterations.iterrows():
                    metric_col = 'metric_name' if 'metric_name' in iter_row.index else 'metric'
                    bootstrap_rows.append({
                        'model_type': iter_row.get('model_type', model_name),
                        'model': display_name,
                        'iteration': iter_row['iteration'],
                        'metric': iter_row[metric_col],
                        'value': iter_row['value'],
                    })

        if bootstrap_rows:
            self._s.data['bootstrap'] = pd.DataFrame(bootstrap_rows)

    @staticmethod
    def _get_metric_value(concordance, col_name):
        """Extract a metric value from the concordance DataFrame."""
        if concordance is None or col_name not in concordance.columns:
            return None
        val = concordance[col_name].iloc[0]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)

    def _sort_performance_table(self, perf_df: pd.DataFrame) -> pd.DataFrame:
        sort_col = get_metric_display_name(self._s.metric)
        if self._s.metric in LOWER_IS_BETTER:
            sort_col += " *"
        if sort_col in perf_df.columns:
            perf_df['_sort'] = perf_df[sort_col].str.extract(r'^(-?[\d.]+)')[0].astype(float)
            ascending = self._s.metric in LOWER_IS_BETTER
            perf_df = perf_df.sort_values('_sort', ascending=ascending)
            perf_df = perf_df.drop(columns=['_sort'])
        return perf_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Bootstrap rainclouds
    # ------------------------------------------------------------------

    def _add_bootstrap_rainclouds(self, md_lines, figures, figure_info):
        """Add bootstrap metric distribution rainclouds via learner.plot_metric."""
        if not self._s.bootstrap_results:
            return

        plot_metrics = self._get_bootstrap_plot_metrics()
        if not plot_metrics:
            return

        try:
            if len(plot_metrics) == 1:
                fig = self._s.learner.plot_metric(
                    style="raincloud", source="bootstrap",
                    metric=plot_metrics[0],
                    title="Bootstrap Metric Distribution (Test Set)",
                )
            else:
                fig = self._s.learner.plot_metric(
                    style="raincloud", source="bootstrap",
                    metric=plot_metrics,
                    title="Bootstrap Metric Distributions (Test Set)",
                )
            if fig is None:
                return

            figures.append(('bootstrap_metrics.png', fig))
            md_lines.append("![Bootstrap Metrics](images/bootstrap_metrics.png)")
            md_lines.append("")

            metric_list = ', '.join(
                get_metric_display_name(m) for m in plot_metrics
            )
            figure_info.append(FigureInfo(
                figure_id='bootstrap_metrics',
                title='Bootstrap Metric Distributions',
                data_description=f'Bootstrap iteration data for {metric_list}',
                data_key='bootstrap',
                learner_code=(
                    "fig = learner.plot_metric(\n"
                    "    style='raincloud', source='bootstrap',\n"
                    f"    metric={plot_metrics!r},\n"
                    ")"
                ),
            ))
        except ALL_EXCEPTIONS as e:
            self._log_error(
                e,
                "evaluation: bootstrap raincloud generation",
                warn_user=True,
                user_message=f"Could not generate bootstrap rainclouds: {e}",
            )
            md_lines.append(f"*[Bootstrap distribution plot unavailable: {e}]*")
            md_lines.append("")

