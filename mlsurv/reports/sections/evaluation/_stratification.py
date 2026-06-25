"""5.5 Survival Stratification (KM curves) subsection."""

from __future__ import annotations


import numpy as np
import pandas as pd

from mlsurv.reports._constants import (
    P_THRESHOLD_EXCELLENT,
    P_THRESHOLD_GOOD,
    P_THRESHOLD_SIGNIFICANT,
    format_pvalue,
)
from mlsurv.reports.sections.base import DATA_EXCEPTIONS, FigureInfo
from mlsurv.viz._palettes import get_model_display_name, MANUSCRIPT_DEFAULTS, get_risk_colors
from mlsurv.viz.survival import _km_combined

from mlsurv.reports.sections.evaluation._base import _BaseSubsection
from mlsurv.reports.sections.evaluation._helpers import compute_separation_quality


def _build_rmst_group_table(result, group_keys, group_map, test_data):
    """Build a DataFrame with per-group RMST rows (with low/high risk labels)."""
    rows = []
    for g in group_keys:
        label = group_map[g]
        if g == group_keys[0]:
            label += " (low risk)"
        elif g == group_keys[-1]:
            label += " (high risk)"
        mask = test_data['risk_group'].astype(str) == g
        rows.append({
            'Risk Group': label,
            'N': int(mask.sum()),
            'RMST': f"{result.rmst[g]:.2f}",
            'SE': f"{result.se[g]:.2f}",
        })
    return pd.DataFrame(rows)


def _build_rmst_narrative(md_lines, extreme, group_map, tau, time_unit) -> None:
    """Append RMST narrative comparing the extreme risk groups.

    Sign convention: ``diff = rmst(high_risk) - rmst(low_risk)``, so
    negative means the low-risk group survived longer (expected).
    """
    diff_abs = abs(extreme['diff'])
    g0, g1 = extreme['groups']
    if diff_abs < 1e-10:
        md_lines.append(
            f"The low-risk and high-risk groups have equivalent "
            f"RMST up to \u03c4 = {tau:.1f}."
        )
    else:
        low_risk_label = group_map[g0]
        high_risk_label = group_map[g1]
        sig = ("statistically significant"
               if extreme['adjusted_p'] < P_THRESHOLD_SIGNIFICANT
               else "not statistically significant")
        if extreme['diff'] < 0:
            md_lines.append(
                f"The low-risk group ({low_risk_label}) survived on average "
                f"**{diff_abs:.1f} {time_unit}** longer than the "
                f"high-risk group ({high_risk_label}) up to "
                f"\u03c4 = {tau:.1f} "
                f"(p = {format_pvalue(extreme['adjusted_p'])}, {sig})."
            )
        else:
            md_lines.append(
                f"Unexpectedly, the high-risk group ({high_risk_label}) "
                f"survived on average "
                f"**{diff_abs:.1f} {time_unit}** longer than the "
                f"low-risk group ({low_risk_label}) up to "
                f"\u03c4 = {tau:.1f} "
                f"(p = {format_pvalue(extreme['adjusted_p'])}, {sig}). "
                f"This may indicate model miscalibration."
            )
    md_lines.append("")


class _StratificationSubsection(_BaseSubsection):
    """Risk stratification with Kaplan-Meier curves."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        test_set = getattr(self._s.eval_result, 'test_set', None)
        if test_set is not None:
            return 'risk_score' in test_set.columns

        return bool(self._s._get_trained_models()) and self._s.learner.X_test is not None

    def add(self, md_lines: list, figures: list, figure_info: list) -> None:
        md_lines.append(
            f"Risk stratification divides patients into {self._s.n_groups} equal-sized "
            f"quantile groups based on predicted risk scores (Q1 = lowest predicted "
            f"risk, Q{self._s.n_groups} = highest). Well-separated Kaplan-Meier curves "
            f"indicate the model effectively identifies high- and low-risk patients."
        )
        md_lines.append("")
        md_lines.append(
            "You can re-stratify programmatically to explore alternative "
            "cutpoints. Change the number of equal-sized quantile groups "
            "with the `n_groups` argument, or use `break_type=\"maxstat\"` "
            "for data-driven optimal cutpoints (log-rank maximization) "
            "instead of fixed quantiles:"
        )
        md_lines.append("")
        md_lines.append("```python")
        md_lines.append("learner.plot_survival(n_groups=3)              # tertiles")
        md_lines.append("learner.plot_survival(break_type=\"maxstat\")   # optimal cutpoint(s)")
        md_lines.append("```")
        md_lines.append("")

        test_data_by_model = self._get_test_data_with_risk()
        n_models = len(test_data_by_model)

        if n_models == 0:
            md_lines.append("*Risk scores unavailable for stratification.*")
            md_lines.append("")
            return
        elif n_models == 1:
            model_type = next(iter(test_data_by_model))
            test_data = test_data_by_model[model_type]
            try:
                self._add_single_model_stratification(test_data, model_type, md_lines, figures, figure_info)
            except DATA_EXCEPTIONS as exc:
                self._log_error(
                    exc, "evaluation: single-model stratification",
                    warn_user=True,
                    user_message=f"Could not generate risk stratification: {exc}",
                )
                md_lines.append(f"*[Risk stratification unavailable: {exc}]*")
                md_lines.append("")
        else:
            try:
                self._add_multi_model_stratification(test_data_by_model, md_lines, figures, figure_info)
            except DATA_EXCEPTIONS as exc:
                self._log_error(
                    exc, "evaluation: multi-model stratification",
                    warn_user=True,
                    user_message=f"Could not generate risk stratification: {exc}",
                )
                md_lines.append(f"*[Risk stratification unavailable: {exc}]*")
                md_lines.append("")

    # ------------------------------------------------------------------
    # Risk score extraction
    # ------------------------------------------------------------------

    def _get_test_data_with_risk(self) -> dict[str, pd.DataFrame]:
        result = self._get_risk_from_eval_result()
        if result is not None:
            return result
        return self._get_risk_from_learner()

    def _get_risk_from_eval_result(self) -> dict[str, pd.DataFrame] | None:
        test_set = getattr(self._s.eval_result, 'test_set', None)
        if test_set is None or 'risk_score' not in test_set.columns:
            return None

        if 'model_type' in test_set.columns:
            return {
                model_type: test_set[test_set['model_type'] == model_type][
                    ['time', 'event', 'risk_score']
                ].copy()
                for model_type in test_set['model_type'].unique()
            }

        trained_models = self._s._get_trained_models()
        model_type = next(iter(trained_models)) if trained_models else 'model'
        return {model_type: test_set[['time', 'event', 'risk_score']].copy()}

    def _get_risk_from_learner(self) -> dict[str, pd.DataFrame]:
        trained_models = self._s._get_trained_models()
        if not trained_models or self._s.learner.X_test is None:
            return {}

        result = {}
        for model_name, model_result in trained_models.items():
            try:
                risk_scores = model_result.model.predict(self._s.learner.X_test)
                result[model_name] = pd.DataFrame({
                    'time': self._s.learner.y_test['time'].values,
                    'event': self._s.learner.y_test['event'].values,
                    'risk_score': risk_scores,
                })
            except (AttributeError, NotImplementedError, ValueError, RuntimeError) as e:
                self._log_error(
                    e,
                    f"evaluation: risk score extraction for {model_name}",
                    warn_user=True,
                    user_message=f"Risk scores unavailable for {model_name}: {e}",
                )

        return result

    # ------------------------------------------------------------------
    # Single-model stratification
    # ------------------------------------------------------------------

    def _add_single_model_stratification(self, test_data, model_type, md_lines, figures, figure_info):
        from plotly.subplots import make_subplots

        d = MANUSCRIPT_DEFAULTS

        if test_data is None or len(test_data) == 0:
            md_lines.append("*Risk stratification unavailable: no risk scores.*")
            md_lines.append("")
            return

        test_data, actual_groups = self._s._create_risk_groups(test_data, self._s.n_groups)
        if test_data is None:
            md_lines.append("*Risk stratification unavailable: insufficient unique risk scores.*")
            md_lines.append("")
            return

        group_labels = [f'Q{i+1}' for i in range(actual_groups)]
        colors_list = get_risk_colors(actual_groups)
        color_map = dict(zip(group_labels, colors_list))
        label_map = dict(enumerate(group_labels))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Overall Cohort', 'By Predicted Risk Group'],
            horizontal_spacing=0.12,
        )

        # Left panel: overall KM
        _km_combined(fig, test_data, row=1, col=1, show_ci=True)

        # Right panel: by risk group
        km_rows, median_survival_rows = _km_combined(
            fig, test_data, group_col='risk_group',
            row=1, col=2, show_ci=True, colors=color_map,
            group_labels=label_map,
        )
        # Enrich rows with model_type and relabel group → risk_group
        for r in km_rows:
            r['model_type'] = model_type
            r['risk_group'] = r.pop('group')
        for r in median_survival_rows:
            r['model_type'] = model_type
            r['risk_group'] = r.pop('group')

        if km_rows:
            self._s.data['km_by_risk'] = pd.DataFrame(km_rows)
        if median_survival_rows:
            self._s.data['median_survival'] = pd.DataFrame(median_survival_rows)

        fig.update_layout(
            template="mlsurv_minimal",
            width=d["width_double"],
            height=int(d["width_single"] * 0.8),
            margin=dict(t=60),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5,
            ),
        )

        fig.update_xaxes(title_text=self._s.time_label, row=1, col=1)
        fig.update_xaxes(title_text=self._s.time_label, row=1, col=2)
        fig.update_yaxes(title_text='Survival Probability', range=[0, 1.05], row=1, col=1)
        fig.update_yaxes(title_text='Survival Probability', range=[0, 1.05], row=1, col=2)

        figures.append(('survival_curves.png', fig))

        self._append_logrank_result(test_data, md_lines)

        md_lines.append("![Survival Curves](images/survival_curves.png)")
        md_lines.append("")

        self._append_median_survival_table(median_survival_rows, md_lines)
        self._append_rmst_table(test_data, actual_groups, md_lines, model_type=model_type)

        md_lines.append(
            f"**Interpretation:** A log-rank p-value < {P_THRESHOLD_SIGNIFICANT} indicates that "
            "the predicted risk groups have significantly different survival "
            "experiences, confirming the model's ability to stratify patients. "
            "RMST differences quantify the average survival time gained or lost "
            "between risk groups over the follow-up period."
        )
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='survival_curves',
            title='Survival Curves (Overall + Risk Stratification)',
            data_description='Tidy DataFrames with KM curves, median survival, and RMST',
            data_key='km_by_risk',
            learner_code="fig = learner.plot_survival()",
        ))

    # ------------------------------------------------------------------
    # Multi-model stratification
    # ------------------------------------------------------------------

    def _add_multi_model_stratification(self, test_data_by_model, md_lines, figures, figure_info):
        from plotly.subplots import make_subplots

        d = MANUSCRIPT_DEFAULTS
        model_types = list(test_data_by_model.keys())
        n_models = len(model_types)

        # Pre-compute risk groups once per model
        grouped_by_model: dict[str, tuple[pd.DataFrame, int]] = {}
        for model_type, test_data in test_data_by_model.items():
            grouped_data, actual_groups = self._s._create_risk_groups(
                test_data, self._s.n_groups,
            )
            if grouped_data is not None:
                grouped_by_model[model_type] = (grouped_data, actual_groups)

        if not grouped_by_model:
            md_lines.append("*Risk stratification unavailable: insufficient unique risk scores.*")
            md_lines.append("")
            return

        self._append_stratification_comparison_table(grouped_by_model, md_lines)

        # Survival curves grid — placed after log-rank table (both use
        # all quartile groups), before RMST (which focuses on extremes)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        subplot_titles = [get_model_display_name(m) for m in model_types]

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.12,
            vertical_spacing=0.15,
        )

        km_by_risk_rows = []

        for idx, model_type in enumerate(model_types):
            if model_type not in grouped_by_model:
                continue
            test_data, actual_groups = grouped_by_model[model_type]

            r = idx // n_cols + 1
            c = idx % n_cols + 1

            group_labels = [f'Q{i+1}' for i in range(actual_groups)]
            colors_list = get_risk_colors(actual_groups)
            color_map = dict(zip(group_labels, colors_list))
            label_map = dict(enumerate(group_labels))

            group_rows, _ = _km_combined(
                fig, test_data, group_col='risk_group',
                row=r, col=c, show_ci=True, colors=color_map,
                group_labels=label_map, show_legend=(idx == 0),
            )
            for row_data in group_rows:
                row_data['model_type'] = model_type
                row_data['risk_group'] = row_data.pop('group')
            km_by_risk_rows.extend(group_rows)

        if km_by_risk_rows:
            self._s.data['km_by_risk'] = pd.DataFrame(km_by_risk_rows)

        fig.update_layout(
            template="mlsurv_minimal",
            width=d["width_double"],
            height=int(d["width_single"] * 0.8 * n_rows),
            margin=dict(t=60),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15 / n_rows,
                xanchor='center',
                x=0.5,
            ),
        )

        for r in range(1, n_rows + 1):
            for c in range(1, n_cols + 1):
                fig.update_xaxes(title_text=self._s.time_label, row=r, col=c)
                fig.update_yaxes(title_text='Survival Prob.', range=[0, 1.05], row=r, col=c)

        figures.append(('survival_curves_grid.png', fig))
        md_lines.append("![Survival Curves Grid](images/survival_curves_grid.png)")
        md_lines.append("")

        # RMST comparison — after survival curves
        self._append_rmst_comparison_multi(grouped_by_model, md_lines)

        md_lines.append(
            f"**Interpretation:** A log-rank p-value < {P_THRESHOLD_SIGNIFICANT} indicates that "
            "the predicted risk groups have significantly different survival "
            "experiences, confirming the model's ability to stratify patients. "
            "RMST differences quantify the average survival time gained or lost "
            "between risk groups over the follow-up period."
        )
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='survival_curves_grid',
            title='Multi-Model Survival Curves Grid',
            data_description='Tidy DataFrame with KM curves by model and RMST',
            data_key='km_by_risk',
            learner_code="fig = learner.plot_survival()",
        ))

    # ------------------------------------------------------------------
    # KM curve helpers
    # ------------------------------------------------------------------

    def _append_logrank_result(self, test_data, md_lines):
        try:
            from lifelines.statistics import multivariate_logrank_test
            result = multivariate_logrank_test(
                test_data['time'],
                test_data['risk_group'],
                test_data['event'],
            )
            p_value = result.p_value
            significance = '(significant separation)' if p_value < P_THRESHOLD_SIGNIFICANT else ''
            md_lines.append(f"**Log-rank test p-value:** *{format_pvalue(p_value)}* {significance}")
            md_lines.append("")
        except (ValueError, ZeroDivisionError) as e:
            self._log_error(
                e,
                "evaluation: log-rank computation",
                warn_user=True,
                user_message=f"Could not compute log-rank test: {e}",
            )

    def _append_median_survival_table(self, median_survival_rows, md_lines):
        md_lines.append("#### Median Survival by Risk Group")
        md_lines.append("")

        table_data = []
        for row in median_survival_rows:
            median_str = f"{row['median_survival']:.1f}" if row['median_survival'] is not None else "Not reached"
            table_data.append({
                'Risk Group': row['risk_group'],
                'N': row['n_patients'],
                'Events': row['n_events'],
                'Median Survival': median_str,
            })

        table_df = pd.DataFrame(table_data)
        md_lines.append(self._format_table(table_df))
        md_lines.append("")

    def _append_stratification_comparison_table(self, grouped_by_model, md_lines):
        from lifelines.statistics import multivariate_logrank_test

        md_lines.append("#### Risk Stratification Comparison")
        md_lines.append("")

        comparison_rows = []
        for model_type, (test_data, _) in grouped_by_model.items():
            try:
                result = multivariate_logrank_test(
                    test_data['time'],
                    test_data['risk_group'],
                    test_data['event'],
                )
                p_value = result.p_value
            except (ValueError, ZeroDivisionError) as e:
                self._log_error(
                    e,
                    f"evaluation: log-rank comparison for {model_type}",
                    warn_user=True,
                    user_message=f"Log-rank test failed for {model_type}: {e}",
                )
                p_value = np.nan

            quality = compute_separation_quality(p_value)
            comparison_rows.append({
                'Model': get_model_display_name(model_type),
                'Log-rank p': format_pvalue(p_value) if not np.isnan(p_value) else "N/A",
                'Quality': quality,
            })

        if not comparison_rows:
            return

        comp_df = pd.DataFrame(comparison_rows)
        md_lines.append(self._format_table(comp_df))
        md_lines.append("")
        md_lines.append(
            f"*Quality: Excellent (p<{P_THRESHOLD_EXCELLENT}), "
            f"Good (p<{P_THRESHOLD_GOOD}), "
            f"Fair (p<{P_THRESHOLD_SIGNIFICANT}), "
            f"Poor (p>={P_THRESHOLD_SIGNIFICANT})*"
        )
        md_lines.append("")

    # ------------------------------------------------------------------
    # RMST helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_group_map(group_keys):
        """Map raw group keys (e.g. '0', '1') to display labels ('Q1', 'Q2')."""
        return {k: f'Q{int(k) + 1}' for k in group_keys}

    @staticmethod
    def _find_extreme_pair(pairwise, first_key, last_key):
        """Find the pairwise comparison between lowest and highest risk groups."""
        return next(
            (p for p in pairwise if p['groups'] == (first_key, last_key)),
            pairwise[0],
        )

    @staticmethod
    def _pairwise_to_records(pairwise, group_map, model_type=None):
        """Convert pairwise results to flat records for DataFrame storage."""
        records = []
        for p in pairwise:
            row = {
                'group_0': group_map[p['groups'][0]],
                'group_1': group_map[p['groups'][1]],
                'diff': p['diff'],
                'ci_lower': p['ci_lower'],
                'ci_upper': p['ci_upper'],
                'raw_p': p['raw_p'],
                'adjusted_p': p['adjusted_p'],
            }
            if model_type is not None:
                row['model_type'] = model_type
            records.append(row)
        return records

    def _append_rmst_table(self, test_data, actual_groups, md_lines, model_type=None):
        """Append per-group RMST table and pairwise differences (single model)."""
        try:
            from mlsurv.evaluation.metrics import rmst_test_pairwise

            result = rmst_test_pairwise(
                test_data['time'],
                test_data['event'],
                test_data['risk_group'],
            )

            group_keys = sorted(result.rmst.keys(), key=lambda x: int(x))
            group_map = self._build_group_map(group_keys)

            # Table 1: Per-group RMST
            md_lines.append(f"#### RMST by Risk Group (\u03c4 = {result.tau:.1f})")
            md_lines.append("")

            group_table_df = _build_rmst_group_table(
                result, group_keys, group_map, test_data,
            )
            md_lines.append(self._format_table(group_table_df))
            md_lines.append("")

            self._s.data['rmst_comparison'] = pd.DataFrame([
                {'risk_group': group_map[g], 'rmst': result.rmst[g],
                 'se': result.se[g], 'tau': result.tau}
                for g in group_keys
            ])

            # Table 2: Pairwise differences
            if result.pairwise:
                md_lines.append("#### Pairwise RMST Differences")
                md_lines.append("")
                md_lines.append(
                    f"The pairwise comparison focuses on the difference "
                    f"between the lowest-risk ({group_map[group_keys[0]]}) and "
                    f"highest-risk ({group_map[group_keys[-1]]}) groups."
                )
                md_lines.append("")

                pair_rows = []
                for p in result.pairwise:
                    g0, g1 = p['groups']
                    pair_rows.append({
                        'Comparison': f"{group_map[g0]} vs {group_map[g1]}",
                        'Difference': f"{p['diff']:.2f}",
                        '95% CI': f"[{p['ci_lower']:.2f}, {p['ci_upper']:.2f}]",
                        'p-value (adj.)': format_pvalue(p['adjusted_p']),
                    })

                md_lines.append(self._format_table(pd.DataFrame(pair_rows)))
                md_lines.append("")

                self._s.data['rmst_pairwise'] = pd.DataFrame(
                    self._pairwise_to_records(result.pairwise, group_map)
                )

                # Narrative: extreme pair
                first_key, last_key = group_keys[0], group_keys[-1]
                extreme = self._find_extreme_pair(
                    result.pairwise, first_key, last_key,
                )
                time_unit = self._s.time_unit or "time units"
                _build_rmst_narrative(md_lines, extreme, group_map, result.tau, time_unit)

        except (ValueError, ZeroDivisionError, TypeError) as e:
            self._log_error(
                e,
                f"evaluation: RMST computation"
                f"{' for ' + model_type if model_type else ''}",
                warn_user=True,
                user_message=f"RMST table omitted: {e}",
            )

    def _append_rmst_comparison_multi(self, grouped_by_model, md_lines):
        """Append compact RMST summary table for multi-model comparison."""
        summary_rows = []
        pairwise_all = []

        for model_type, (test_data, actual_groups) in grouped_by_model.items():
            try:
                from mlsurv.evaluation.metrics import rmst_test_pairwise

                result = rmst_test_pairwise(
                    test_data['time'],
                    test_data['event'],
                    test_data['risk_group'],
                )

                group_keys = sorted(result.rmst.keys(), key=lambda x: int(x))
                group_map = self._build_group_map(group_keys)
                first_key, last_key = group_keys[0], group_keys[-1]

                extreme = self._find_extreme_pair(
                    result.pairwise, first_key, last_key,
                )

                summary_rows.append({
                    'Model': get_model_display_name(model_type),
                    '\u03c4': f"{result.tau:.1f}",
                    'Low-risk RMST': f"{result.rmst[first_key]:.2f}",
                    'High-risk RMST': f"{result.rmst[last_key]:.2f}",
                    'Difference': f"{extreme['diff']:.2f}",
                    '95% CI': f"[{extreme['ci_lower']:.2f}, {extreme['ci_upper']:.2f}]",
                    'p-value': format_pvalue(extreme['adjusted_p']),
                })

                pairwise_all.extend(
                    self._pairwise_to_records(
                        result.pairwise, group_map, model_type=model_type,
                    )
                )

            except (ValueError, ZeroDivisionError, TypeError) as e:
                self._log_error(
                    e,
                    f"evaluation: RMST computation for {model_type}",
                    warn_user=True,
                    user_message=f"RMST omitted for {model_type}: {e}",
                )

        if not summary_rows:
            return

        first_actual_groups = next(iter(grouped_by_model.values()))[1]
        md_lines.append(f"#### RMST Comparison (Q1 vs Q{first_actual_groups})")
        md_lines.append("")
        md_lines.append(
            f"The RMST comparison below quantifies the average survival time "
            f"difference between the lowest-risk (Q1) and highest-risk "
            f"(Q{first_actual_groups}) quartile groups identified above."
        )
        md_lines.append("")
        md_lines.append(self._format_table(pd.DataFrame(summary_rows)))
        md_lines.append("")
        md_lines.append(
            "τ is the longest follow-up at which RMST is comparable across "
            "all risk groups for that model; it varies between models because "
            "risk-group assignments differ."
        )
        md_lines.append("")

        if pairwise_all:
            self._s.data['rmst_pairwise'] = pd.DataFrame(pairwise_all)
