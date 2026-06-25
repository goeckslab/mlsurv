"""Time-dependent discrimination (AUC(t)) per validation cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsurv.reports._helpers import (
    build_auc_std_lookup,
    build_auct_summary_rows,
    build_auct_tidy_data,
)
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, FigureInfo
from mlsurv.reports.sections.evaluation._helpers import interpret_auct_quality
from mlsurv.reports.sections.validation._base import _BaseValidationSubsection


class _DiscriminationSubsection(_BaseValidationSubsection):
    """AUC(t) analysis for one external validation cohort."""

    def add(
        self,
        md_lines: list,
        figures: list,
        figure_info: list,
        val_result,
        cohort_name: str,
        cohort_key: str,
    ) -> None:
        auc_df = val_result.evaluation.auc_t
        boot_auc = val_result.bootstrap.auc_t if val_result.bootstrap else None

        md_lines.append(
            "Time-dependent AUC measures discrimination at specific timepoints, "
            "showing how well the model separates subjects by outcome over time."
        )
        md_lines.append("")

        # Build summary table
        std_lookup = build_auc_std_lookup(boot_auc)
        rows, n_model_rows, _, _ = build_auct_summary_rows(auc_df, std_lookup)
        has_std = bool(std_lookup)
        summary_df = pd.DataFrame(rows)
        summary_df = self._s._bold_best_auct_values(summary_df, n_model_rows)
        md_lines.append(self._format_table(summary_df))
        md_lines.append("")
        if has_std:
            md_lines.append("*Values shown as: AUC (std) from bootstrap.*")
            md_lines.append("")

        # Generate AUC(t) figure
        self._add_auct_figure(
            auc_df, boot_auc, cohort_name, cohort_key,
            md_lines, figures, figure_info,
        )

        # Interpretation
        mean_auc = auc_df['auc'].mean()
        if not np.isnan(mean_auc):
            quality = interpret_auct_quality(mean_auc)
            md_lines.append(
                f"**Interpretation:** Mean AUC(t) on {cohort_name} is {mean_auc:.2f}, "
                f"indicating {quality} discrimination."
            )
            md_lines.append("")

        # Store tidy data
        self._s.data[f'auct_{cohort_key}'] = build_auct_tidy_data(
            auc_df, boot_auc,
        )

    def _add_auct_figure(
        self,
        auc_df,
        boot_auc,
        cohort_name: str,
        cohort_key: str,
        md_lines: list,
        figures: list,
        figure_info: list,
    ) -> None:
        """Generate and append AUC(t) figure.

        Delegates to :meth:`ValidationView.plot_metric_time` so users can
        reproduce the figure via the learner API. The proxy internally
        calls the same ``plot_metric_time`` viz helper with the cohort's
        ``auc_t`` and bootstrap std — no duplicate inline assembly here.
        """
        try:
            fig = self._s.learner.validation(cohort_name).plot_metric_time()
            fig_filename = f'validation_auct_{cohort_key}.png'
            figures.append((fig_filename, fig))
            md_lines.append(f"![{cohort_name} AUC(t)](images/{fig_filename})")
            md_lines.append("")

            figure_info.append(FigureInfo(
                figure_id=f'validation_auct_{cohort_key}',
                title=f'{cohort_name} Time-Dependent AUC',
                data_description=f'Tidy DataFrame with AUC(t) for {cohort_name}',
                data_key=f'auct_{cohort_key}',
                learner_code=(
                    f"fig = learner.validation({cohort_name!r}).plot_metric_time()"
                ),
            ))
        except (NotImplementedError, TypeError):
            # Developer bugs should crash the report build, not be
            # silently degraded to a missing-figure warning. The view
            # proxy raises NotImplementedError for unsupported kwargs
            # (references=, metrics=) and TypeError for wrong models=
            # type — both indicate programming errors, not data issues.
            raise
        except ALL_EXCEPTIONS as e:
            self._log_error(
                e,
                f"validation: AUC(t) figure for {cohort_name}",
                warn_user=True,
                user_message=f"Could not generate AUC(t) plot for {cohort_name}: {e}",
            )
            md_lines.append(f"*[AUC(t) plot unavailable: {e}]*")
            md_lines.append("")
