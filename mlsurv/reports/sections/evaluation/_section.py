"""EvaluationSection orchestrator — delegates to per-subsection classes."""

from typing import Any, Dict, Optional

import pandas as pd

from mlsurv.constants import DEFAULT_SCORING_METRIC
from mlsurv.reports._constants import DEFAULT_CALIBRATION_BINS, DEFAULT_RISK_GROUPS
from mlsurv.reports.sections.base import BaseSection, DATA_EXCEPTIONS

from mlsurv.reports._helpers import extract_eval_times_from_auc_t, format_eval_time
from mlsurv.reports.sections.evaluation._methodology import _MethodologySubsection
from mlsurv.reports.sections.evaluation._performance import _PerformanceSubsection
from mlsurv.reports.sections.evaluation._discrimination import _DiscriminationSubsection
from mlsurv.reports.sections.evaluation._roc import _ROCSubsection
from mlsurv.reports.sections.evaluation._calibration import _CalibrationSubsection
from mlsurv.reports.sections.evaluation._stratification import _StratificationSubsection
from mlsurv.reports.sections.evaluation._paired_comparison import _PairedComparisonSubsection


class EvaluationSection(BaseSection):
    """Comprehensive model evaluation section.

    Combines multiple evaluation aspects into a single section with subsections:
    - TOC
    - 5.1 Methodology (bootstrap config, interpretation guidance)
    - 5.2 Overview (metrics table, forest plots, heatmap, rainclouds)
    - 5.3 Discrimination (AUC(t) + ROC curves, merged)
    - 5.4 Calibration
    - 5.5 Risk Stratification (KM curves, RMST)
    - 5.6 Paired Model Comparison (statistical tests)

    Maps to API methods: evaluate(), bootstrap()
    """

    name = "evaluation"
    title = "Model Evaluation"
    _CALIBRATION_COLUMNS = [
        'model_type',
        'eval_time',
        'bin_predicted',
        'bin_observed',
        'bin_ci_lower',
        'bin_ci_upper',
    ]

    def __init__(
        self,
        learner,
        eval_result: Optional[Any] = None,
        bootstrap_results: Optional[Dict] = None,
        metric: Optional[str] = None,
        n_groups: int = DEFAULT_RISK_GROUPS,
        n_bins: int = DEFAULT_CALIBRATION_BINS,
    ):
        super().__init__(learner)
        self.eval_result = eval_result
        self.bootstrap_results = bootstrap_results
        self.metric = metric if metric is not None else getattr(learner, 'scoring', DEFAULT_SCORING_METRIC)
        self.n_groups = n_groups
        self.n_bins = n_bins

    def is_available(self) -> bool:
        if self.bootstrap_results or self.eval_result is not None:
            return True
        return any(result.evaluation for result in self._get_all_models().values())

    def generate(self) -> Dict[str, Any]:
        md_lines = []
        figures = []
        figure_info = []

        # Section introduction
        md_lines.append(
            "This section presents model evaluation results on the hold-out test set, "
            "which was not used during training or hyperparameter tuning. "
            "Evaluation encompasses discrimination (ability to rank patients by risk), "
            "calibration (accuracy of predicted probabilities), and risk stratification "
            "(clinical utility for patient grouping)."
        )
        md_lines.append("")

        # Eval_times note (brief inline — full detail in Environment section)
        eval_times_note = self._build_eval_times_note()
        if eval_times_note:
            md_lines.append(eval_times_note)
            md_lines.append("")

        # Instantiate subsections
        disc = _DiscriminationSubsection(self)
        roc = _ROCSubsection(self)
        cal = _CalibrationSubsection(self)
        strat = _StratificationSubsection(self)
        paired = _PairedComparisonSubsection(self)

        self._add_outline(md_lines, disc, roc, cal, strat, paired)

        # === 5.1 Methodology ===
        md_lines.append("### Methodology")
        md_lines.append("")
        _MethodologySubsection(self).add(md_lines, figures, figure_info)

        # === 5.2 Overview ===
        md_lines.append("### Overview")
        md_lines.append("")
        _PerformanceSubsection(self).add(md_lines, figures, figure_info)

        # === 5.3 Discrimination (merged AUC(t) + ROC) ===
        disc_available = disc.is_available()
        roc_available = roc.is_available()
        if disc_available or roc_available:
            md_lines.append("### Discrimination")
            md_lines.append("")
            if disc_available:
                disc.add(md_lines, figures, figure_info)
            if roc_available:
                md_lines.append("#### ROC Curves")
                md_lines.append("")
                roc.add(md_lines, figures, figure_info)

        # Remaining conditional subsections
        conditional_subsections = [
            (cal, "Calibration"),
            (strat, "Risk Stratification"),
            (paired, "Paired Model Comparison"),
        ]
        for subsection, heading in conditional_subsections:
            if subsection.is_available():
                md_lines.append(f"### {heading}")
                md_lines.append("")
                subsection.add(md_lines, figures, figure_info)

        # Store prediction data for programmatic access
        self.data['train_predictions'] = self._build_train_predictions_df()
        self.data['test_predictions'] = self._build_test_predictions_df()

        return {
            'markdown': '\n'.join(md_lines),
            'figures': figures,
            'figure_info': figure_info,
            'tabs': None,
        }

    # ------------------------------------------------------------------
    # Eval_times note
    # ------------------------------------------------------------------

    def _build_eval_times_note(self) -> str:
        """Build a one-line note about the evaluation timepoints used."""
        try:
            if self.eval_result is None:
                return ""
            eval_times = extract_eval_times_from_auc_t(self.eval_result.auc_t)
            if not eval_times:
                return ""

            is_auto = getattr(self.learner._state, 'user_eval_times', None) is None
            source = (
                "auto-generated from 10th\u201390th percentile of training event times"
                if is_auto
                else "user-specified"
            )

            times_str = ", ".join(format_eval_time(t) for t in eval_times)
            unit = self.time_unit
            unit_suffix = f" {unit}" if unit else ""
            n = len(eval_times)

            return (
                f"**Evaluation Timepoints:** {times_str}{unit_suffix} "
                f"({n} points, {source})"
            )
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc,
                "evaluation: building eval_times note",
            )
            return ""

    # ------------------------------------------------------------------
    # TOC
    # ------------------------------------------------------------------

    def _add_outline(
        self,
        md_lines: list,
        disc,
        roc,
        cal,
        strat,
        paired,
    ) -> None:
        """Add a section TOC for the available subsections."""
        toc_entries = [
            ('Methodology', 'methodology'),
            ('Overview', 'overview'),
        ]
        if disc.is_available() or roc.is_available():
            toc_entries.append(('Discrimination', 'discrimination'))
        conditional_entries = [
            (cal, 'Calibration', 'calibration'),
            (strat, 'Risk Stratification', 'risk-stratification'),
            (paired, 'Paired Model Comparison', 'paired-model-comparison'),
        ]
        for subsection, label, anchor in conditional_entries:
            if subsection.is_available():
                toc_entries.append((label, anchor))

        md_lines.extend(self._build_section_outline(toc_entries))

    # ------------------------------------------------------------------
    # Prediction data (kept for programmatic access)
    # ------------------------------------------------------------------

    def _build_test_predictions_df(self) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["model", "sample_index", "time", "event", "risk_score"])
        if self.eval_result is None:
            return empty
        test_set = getattr(self.eval_result, "test_set", None)
        if test_set is None or len(test_set) == 0:
            return empty
        return self._format_prediction_rows(test_set)

    # ------------------------------------------------------------------
    # Delegation helpers — preserve the method API that existing tests
    # call directly
    # ------------------------------------------------------------------

    def _has_risk_scores(self) -> bool:
        return _StratificationSubsection(self).is_available()

    def _get_test_data_with_risk(self) -> Dict[str, pd.DataFrame]:
        return _StratificationSubsection(self)._get_test_data_with_risk()
