"""Risk stratification + KM curves + RMST per validation cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsurv.reports._constants import P_THRESHOLD_SIGNIFICANT, format_pvalue
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, FigureInfo
from mlsurv.reports.sections.validation._base import _BaseValidationSubsection
from mlsurv.reports.sections.validation._helpers import (
    format_median_survival_table,
)


class _StratificationSubsection(_BaseValidationSubsection):
    """Risk stratification with KM curves, log-rank test, and RMST.

    Figures are built by :meth:`ValidationView.plot_survival` so users
    can reproduce them via the learner API. The inline
    per-group ``KaplanMeierFitter`` loop is kept because it feeds the
    tidy ``km_{cohort_key}`` / ``median_{cohort_key}`` exports — the
    proxy's ``return_data`` path returns time/event/risk_score rows,
    not KM curve points.
    """

    def add(
        self,
        md_lines: list,
        figures: list,
        figure_info: list,
        val_result,
        cohort_name: str,
        cohort_key: str,
        n_groups: int = 4,
    ) -> None:
        """Add risk stratification analysis for one external cohort.

        The orchestrator passes ``val_result.test_set`` directly as
        *val_result* for stratification.
        """
        from lifelines import KaplanMeierFitter

        test_set = val_result
        test_data, actual_groups = self._s._create_risk_groups(test_set, n_groups)
        if test_data is None:
            md_lines.append("*Risk stratification unavailable: insufficient unique risk scores.*")
            md_lines.append("")
            return

        group_labels = [f'Q{i+1}' for i in range(actual_groups)]

        _add_logrank_test(self._s, md_lines, test_data, cohort_name)

        # Build tidy KM export via per-group KaplanMeierFitter. Tidy
        # data + log-rank still depend on the inline KM fits, so a
        # failure here aborts the whole subsection (no figure to show
        # without underlying KM data).
        km_by_risk_rows = []
        median_survival_rows = []
        try:
            for group_idx, group_df in test_data.groupby(
                'risk_group', observed=True, sort=True,
            ):
                kmf = KaplanMeierFitter()
                kmf.fit(group_df['time'], group_df['event'])

                label = group_labels[int(group_idx)]

                km_rows, median_row = _build_stratification_km_data(
                    kmf, group_df, cohort_name, label,
                )
                km_by_risk_rows.extend(km_rows)
                median_survival_rows.append(median_row)
        except ALL_EXCEPTIONS as e:
            self._log_error(
                e,
                f"validation: KM tidy data build for {cohort_name}",
                warn_user=True,
                user_message=f"KM stratification plot omitted for {cohort_name}: {e}",
            )
            md_lines.append(f"*[KM stratification plot unavailable: {e}]*")
            md_lines.append("")
            # RMST still uses test_data, fall through to it below.
            _add_rmst(self._s, md_lines, test_data, cohort_name, cohort_key)
            return

        # Store tidy DataFrames (with cohort-specific keys)
        if km_by_risk_rows:
            self._s.data[f'km_{cohort_key}'] = pd.DataFrame(km_by_risk_rows)
        if median_survival_rows:
            self._s.data[f'median_{cohort_key}'] = pd.DataFrame(median_survival_rows)

        # Delegate figure assembly to the validation view proxy.
        # For single-model cohorts (the common case) this reproduces
        # the previous single-plot output. For multi-model cohorts
        # (where the previous inline assembly produced a single plot
        # from quantile-bucketed combined risk scores — a dubious
        # mash-up across different models' risk-score scales) the
        # proxy produces a panel grid, one per model, which is the
        # more honest representation.
        try:
            fig = self._s.learner.validation(cohort_name).plot_survival(
                n_groups=actual_groups,
                title=f'Risk Stratification: {cohort_name}',
            )
        except (NotImplementedError, TypeError):
            # Developer bugs should crash the report build, not be
            # silently degraded to a missing-figure warning. The view
            # proxy raises NotImplementedError for unsupported kwargs
            # and TypeError for wrong models= type — both indicate
            # programming errors, not data issues.
            raise
        except ALL_EXCEPTIONS as e:
            self._log_error(
                e,
                f"validation: KM figure proxy for {cohort_name}",
                warn_user=True,
                user_message=f"KM stratification plot omitted for {cohort_name}: {e}",
            )
            # Tidy exports (km_/median_) succeeded — spell that out so
            # the reader isn't misled into thinking the data is also
            # missing. Only the figure is unavailable here.
            md_lines.append(
                f"*[KM stratification figure unavailable "
                f"(tidy data still exported): {e}]*"
            )
            md_lines.append("")
        else:
            figure_filename = f'validation_km_{cohort_key}.png'
            figures.append((figure_filename, fig))
            md_lines.append(f"![{cohort_name} KM Curves](images/{figure_filename})")
            md_lines.append("")

            figure_info.append(FigureInfo(
                figure_id=f'validation_km_{cohort_key}',
                title=f'{cohort_name} Validation KM Curves',
                data_description=(
                    f'Tidy DataFrames: validation_km_{cohort_key} (KM curves), '
                    f'validation_median_{cohort_key} (summary stats)'
                ),
                data_key=f'km_{cohort_key}',
                learner_code=(
                    f"fig = learner.validation({cohort_name!r}).plot_survival(\n"
                    f"    n_groups={actual_groups},\n"
                    f")"
                ),
            ))

        # Median survival table — independent of figure, runs whether
        # the proxy succeeded or fell through to the placeholder.
        md_lines.append(f"#### Median Survival by Risk Group: {cohort_name}")
        md_lines.append("")

        table_data = format_median_survival_table(median_survival_rows)
        table_df = pd.DataFrame(table_data)
        md_lines.append(self._format_table(table_df))
        md_lines.append("")

        # RMST analysis
        _add_rmst(self._s, md_lines, test_data, cohort_name, cohort_key)


# ------------------------------------------------------------------
# Standalone helpers (no class state needed)
# ------------------------------------------------------------------


def _add_logrank_test(section, md_lines: list, test_data: pd.DataFrame, cohort_name: str) -> None:
    """Run multivariate log-rank test and append result to *md_lines*."""
    from lifelines.statistics import multivariate_logrank_test

    try:
        result = multivariate_logrank_test(
            test_data['time'],
            test_data['risk_group'],
            test_data['event'],
        )
        p_value = result.p_value
        significance = '(significant separation)' if p_value < P_THRESHOLD_SIGNIFICANT else ''
        md_lines.append(
            f"**Log-rank test p-value ({cohort_name}):** *{format_pvalue(p_value)}* "
            f"{significance}"
        )
        md_lines.append("")
    except ALL_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            f"validation: log-rank computation for cohort {cohort_name}",
            warn_user=True,
            user_message=f"Could not compute log-rank test for {cohort_name}: {e}",
        )


def _build_stratification_km_data(
    kmf,
    group_df: pd.DataFrame,
    cohort_name: str,
    label: str,
) -> tuple:
    """Build tidy KM rows and median survival row for one risk group."""
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_

    km_rows = []
    for t_idx, t in enumerate(sf.index):
        km_rows.append({
            'cohort': cohort_name,
            'risk_group': label,
            'time': t,
            'survival_prob': sf.iloc[t_idx, 0],
            'ci_lower': ci.iloc[t_idx, 0],
            'ci_upper': ci.iloc[t_idx, 1],
            'n_patients': len(group_df),
        })

    median = kmf.median_survival_time_
    median_not_reached = np.isnan(median) or np.isinf(median)
    median_row = {
        'cohort': cohort_name,
        'risk_group': label,
        'n_patients': len(group_df),
        'n_events': int(group_df['event'].sum()),
        'median_survival': median if not median_not_reached else None,
    }
    return km_rows, median_row


def _build_rmst_group_table(result, group_keys, group_map, test_data) -> list:
    """Build RMST per-group table rows."""
    rmst_rows = []
    for g in group_keys:
        label = group_map[g]
        if g == group_keys[0]:
            label += " (low risk)"
        elif g == group_keys[-1]:
            label += " (high risk)"
        mask = test_data['risk_group'].astype(str) == g
        rmst_rows.append({
            'Risk Group': label,
            'N': int(mask.sum()),
            'RMST': f"{result.rmst[g]:.2f}",
            'SE': f"{result.se[g]:.2f}",
        })
    return rmst_rows


def _build_rmst_pairwise_table(result, group_map) -> tuple:
    """Build pairwise RMST difference table rows and tidy records."""
    pair_rows = []
    pairwise_records = []
    for p in result.pairwise:
        g0, g1 = p['groups']
        pair_rows.append({
            'Comparison': f"{group_map[g0]} vs {group_map[g1]}",
            'Difference': f"{p['diff']:.2f}",
            '95% CI': f"[{p['ci_lower']:.2f}, {p['ci_upper']:.2f}]",
            'p-value (adj.)': format_pvalue(p['adjusted_p']),
        })
        pairwise_records.append({
            'group_0': group_map[g0],
            'group_1': group_map[g1],
            'diff': p['diff'],
            'ci_lower': p['ci_lower'],
            'ci_upper': p['ci_upper'],
            'raw_p': p['raw_p'],
            'adjusted_p': p['adjusted_p'],
        })
    return pair_rows, pairwise_records


def _build_rmst_narrative(result, group_keys, group_map, time_unit="time units") -> list:
    """Build RMST narrative lines for the extreme pair.

    Sign convention: ``diff = rmst(high_risk) - rmst(low_risk)``, so
    negative means the low-risk group survived longer (expected).
    """
    first_key, last_key = group_keys[0], group_keys[-1]
    extreme = next(
        (p for p in result.pairwise if p['groups'] == (first_key, last_key)),
        result.pairwise[0],
    )
    diff_abs = abs(extreme['diff'])

    lines = []
    if diff_abs < 1e-10:
        lines.append(
            f"The low-risk and high-risk groups have equivalent "
            f"RMST up to \u03c4 = {result.tau:.1f}."
        )
    else:
        low_risk_label = group_map[first_key]
        high_risk_label = group_map[last_key]
        sig = (
            "statistically significant"
            if extreme['adjusted_p'] < P_THRESHOLD_SIGNIFICANT
            else "not statistically significant"
        )
        if extreme['diff'] < 0:
            lines.append(
                f"The low-risk group ({low_risk_label}) survived "
                f"on average **{diff_abs:.1f} {time_unit}** longer "
                f"than the high-risk group ({high_risk_label}) up "
                f"to \u03c4 = {result.tau:.1f} "
                f"(p = {format_pvalue(extreme['adjusted_p'])}, {sig})."
            )
        else:
            lines.append(
                f"Unexpectedly, the high-risk group "
                f"({high_risk_label}) survived on average "
                f"**{diff_abs:.1f} {time_unit}** longer than the "
                f"low-risk group ({low_risk_label}) up to "
                f"\u03c4 = {result.tau:.1f} "
                f"(p = {format_pvalue(extreme['adjusted_p'])}, {sig}). "
                f"This may indicate model miscalibration on this "
                f"cohort."
            )
    return lines


def _add_rmst(
    section,
    md_lines: list,
    test_data: pd.DataFrame,
    cohort_name: str,
    cohort_key: str,
) -> None:
    """Add RMST analysis for a stratification cohort."""
    try:
        from mlsurv.evaluation.metrics import rmst_test_pairwise

        result = rmst_test_pairwise(
            test_data['time'],
            test_data['event'],
            test_data['risk_group'],
        )

        group_keys = sorted(result.rmst.keys(), key=lambda x: int(x))
        group_map = {k: f'Q{int(k) + 1}' for k in group_keys}

        # Table: Per-group RMST
        md_lines.append(f"#### RMST by Risk Group (\u03c4 = {result.tau:.1f})")
        md_lines.append("")

        rmst_rows = _build_rmst_group_table(result, group_keys, group_map, test_data)
        md_lines.append(section._format_table(pd.DataFrame(rmst_rows)))
        md_lines.append("")

        section.data[f'rmst_{cohort_key}'] = pd.DataFrame([
            {'risk_group': group_map[g], 'rmst': result.rmst[g],
             'se': result.se[g], 'tau': result.tau}
            for g in group_keys
        ])

        # Table: Pairwise differences
        if result.pairwise:
            md_lines.append("#### Pairwise RMST Differences")
            md_lines.append("")

            pair_rows, pairwise_records = _build_rmst_pairwise_table(result, group_map)
            md_lines.append(section._format_table(pd.DataFrame(pair_rows)))
            md_lines.append("")
            section.data[f'rmst_pairwise_{cohort_key}'] = pd.DataFrame(pairwise_records)

            # Narrative: extreme pair
            time_unit = section.time_unit or "time units"
            narrative_lines = _build_rmst_narrative(result, group_keys, group_map, time_unit)
            md_lines.extend(narrative_lines)
            md_lines.append("")

    except (ALL_EXCEPTIONS + (IndexError,)) as e:
        section._log_recoverable_exception(
            e,
            f"validation: RMST computation for {cohort_name}",
            warn_user=True,
            user_message=f"RMST table omitted for {cohort_name}: {e}",
        )
