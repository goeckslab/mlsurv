"""5.2 Discrimination (AUC(t)) subsection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsurv.reports._helpers import (
    build_auc_std_lookup,
    build_auct_summary_rows,
    build_auct_tidy_data,
)
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, FigureInfo
from mlsurv.reports.sections.evaluation._base import _BaseSubsection
from mlsurv.reports.sections.evaluation._helpers import (
    interpret_auct_quality,
    warn_eval_times_mismatch,
)


class _DiscriminationSubsection(_BaseSubsection):
    """Time-dependent AUC curves and summary table."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return (
            self._s.eval_result is not None
            and self._s.eval_result.auc_t is not None
            and len(self._s.eval_result.auc_t) > 0
        )

    def add(self, md_lines: list, figures: list, figure_info: list) -> None:
        md_lines.append(
            "Time-dependent AUC is the area under the ROC curve at each "
            f"horizon *t*, discriminating cases ({self._s.outcome_name} "
            "before *t*) from controls (event-free at *t*). Unlike the "
            "overall C-index, it shows how discrimination varies across "
            "follow-up."
        )
        md_lines.append("")
        md_lines.append(
            "Computed via "
            "[`sksurv.metrics.cumulative_dynamic_auc`](https://"
            "scikit-survival.readthedocs.io/en/stable/api/generated/"
            "sksurv.metrics.cumulative_dynamic_auc.html) (Uno 2007; "
            "Heagerty & Saha 2005). Rather than dropping censored "
            "subjects, each retained subject is reweighted by the "
            "inverse of their estimated censoring-survival probability "
            "at *t* (IPCW), forming a **pseudopopulation** whose AUC is "
            "an unbiased estimate of the AUC under complete follow-up "
            "(assuming independent censoring)."
        )
        md_lines.append("")

        auc_df = self._s.eval_result.auc_t
        boot_auc = self._get_merged_bootstrap_auc()

        self._s.data['auct'] = build_auct_tidy_data(auc_df, boot_auc)

        std_lookup = build_auc_std_lookup(boot_auc)
        summary_df, has_std = self._build_auct_summary_table(auc_df, std_lookup)
        md_lines.append(self._format_table(summary_df))
        md_lines.append("")
        if has_std:
            md_lines.append("*Values shown as: AUC (std) from bootstrap.*")
            md_lines.append("")

        self._add_auct_figure(auc_df, boot_auc, md_lines, figures, figure_info)

        mean_auc = auc_df['auc'].mean()
        if not np.isnan(mean_auc):
            quality = interpret_auct_quality(mean_auc)
            md_lines.append(
                f"**Interpretation:** The mean time-dependent AUC across all "
                f"timepoints is {mean_auc:.2f}, indicating {quality} discrimination. "
                f"An AUC above 0.7 at a given timepoint means the model can "
                f"distinguish patients who will experience {self._s.outcome_name} before that "
                f"time from those who will not with reasonable accuracy."
            )
            md_lines.append("")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_auct_summary_table(self, auc_df: pd.DataFrame, std_lookup: dict):
        warn_eval_times_mismatch(auc_df)
        rows, n_model_rows, eval_times, time_labels = build_auct_summary_rows(
            auc_df, std_lookup,
        )

        # Evaluation-only: append events/at-risk rows
        if 'n_events' in auc_df.columns and 'n_at_risk' in auc_df.columns:
            models = auc_df['model_type'].unique()
            first_model_data = auc_df[auc_df['model_type'] == models[0]]
            events_row = {'': '**Events**'}
            at_risk_row = {'': '**At Risk**'}
            for t, label in zip(eval_times, time_labels):
                time_data = first_model_data[first_model_data['eval_time'] == t]
                if len(time_data) > 0:
                    events_row[label] = str(int(time_data['n_events'].values[0]))
                    at_risk_row[label] = str(int(time_data['n_at_risk'].values[0]))
                else:
                    events_row[label] = "N/A"
                    at_risk_row[label] = "N/A"
            rows.append(events_row)
            rows.append(at_risk_row)

        summary_df = pd.DataFrame(rows)
        has_std = bool(std_lookup)
        return self._s._bold_best_auct_values(summary_df, n_model_rows), has_std

    def _add_auct_figure(self, auc_df, boot_auc, md_lines, figures, figure_info):
        try:
            from mlsurv.viz.metrics import plot_metric_time

            std_df = None
            if boot_auc is not None:
                std_df = (
                    boot_auc
                    .groupby(['model_type', 'eval_time'])['auc']
                    .std()
                    .reset_index()
                    .rename(columns={'auc': 'auc_std'})
                )

            fig = plot_metric_time(auc_df, std_df=std_df, metric="auc", model_col="model_type", time_unit=self._s.time_unit)
            figures.append(('auct_curves.png', fig))
            md_lines.append("![Time-Dependent AUC](images/auct_curves.png)")
            md_lines.append("")

            figure_info.append(FigureInfo(
                figure_id='auct_curves',
                title='Time-Dependent AUC Curves',
                data_description='Tidy DataFrame with AUC(t) values',
                data_key='auct',
                learner_code="fig = learner.plot_metric_time()",
            ))
        except ALL_EXCEPTIONS as e:
            self._log_error(
                e,
                "evaluation: AUC(t) figure generation",
                warn_user=True,
                user_message=f"Could not generate AUC(t) plot: {e}",
            )
            md_lines.append(f"*[AUC(t) plot unavailable: {e}]*")
            md_lines.append("")

    def _get_merged_bootstrap_auc(self):
        if not self._s.bootstrap_results:
            return None

        auc_dfs = [
            boot_result.auc_t
            for boot_result in self._s.bootstrap_results.values()
            if boot_result.auc_t is not None and len(boot_result.auc_t) > 0
        ]

        return pd.concat(auc_dfs, ignore_index=True) if auc_dfs else None
