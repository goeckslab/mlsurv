"""ReportMixin - Report generation methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


def _split_bootstrap_to_dict(combined):
    """Split a combined BootstrapResult into a per-model dict.

    Parameters
    ----------
    combined : BootstrapResult
        Combined multi-model (or single-model) bootstrap result.

    Returns
    -------
    dict of {str: BootstrapResult}
        Per-model bootstrap results keyed by model_type name.
    """
    from mlsurv.results.containers import _extract_model_bootstrap, _get_model_names

    model_names = _get_model_names(combined.summary)
    if not model_names:
        return {"unknown": combined}
    if len(model_names) == 1:
        return {model_names[0]: combined}
    return {
        name: _extract_model_bootstrap(combined, name)
        for name in model_names
    }


def _validate_report_overrides(outcome_name, time_unit, feature_groups, group_colors):
    """Validate user-supplied report override parameters."""
    if outcome_name is not None:
        if not isinstance(outcome_name, str) or not outcome_name.strip():
            raise ValueError(
                f"outcome_name must be a non-empty string, got {outcome_name!r}"
            )
    if time_unit is not None:
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError(
                f"time_unit must be a non-empty string or None, got {time_unit!r}"
            )
    if feature_groups is not None:
        import pandas as pd
        if not isinstance(feature_groups, (dict, list, tuple, pd.Series)):
            raise TypeError(
                f"feature_groups must be a dict, list, tuple, Series, or None, "
                f"got {type(feature_groups).__name__}"
            )
    if group_colors is not None:
        if not isinstance(group_colors, dict):
            raise TypeError(
                f"group_colors must be a dict or None, "
                f"got {type(group_colors).__name__}"
            )


class ReportMixin:
    """Mixin for report generation."""

    def generate_report(
        self: SurvivalLearner,
        output_path=None,
        title="Survival Analysis Report",
        sections=None,
        include_optuna=True,
        eval_result=None,
        bootstrap_results=None,
        outcome_name=None,
        time_unit=None,
        feature_groups=None,
        group_colors=None,
        manuscript=False,
    ):
        """
        Generate comprehensive analysis report.

        Creates a report with embedded visualizations showing analysis results.
        Sections adapt to available data (only shows sections for which data exists).

        NOTE: This method does NOT auto-run evaluate() or bootstrap(). It only
        uses existing data from run() or explicitly provided results.
        If you need evaluation/bootstrap data in the report, run those methods
        first or use run() which runs the full pipeline by default.

        Parameters
        ----------
        output_path : str, optional
            Path to output report file. Format is inferred from file extension:
            '.html' for HTML, '.md' for markdown.
            If None, generates 'report.html' in current directory.
            Use '.html' for interactive human viewing (figures are embedded as
            interactive Plotly charts in one standalone file). Prefer '.md' for
            programmatic or AI-agent consumption: it is plain text, much smaller,
            and easy to read or grep, with figures saved as PNGs in an adjacent
            images/ folder (and it is pandoc-ready for PDF/Word).
        title : str, default='Survival Analysis Report'
            Report title.
        sections : list, optional
            Specific sections to include. Options: 'summary',
            'data_preprocessing', 'model_development',
            'evaluation', 'feature_analysis',
            'subpopulation_analysis', 'validation', 'benchmarking',
            'limitations', 'environment', 'references',
            'tripod_checklist'.
            ('tuning' is accepted as a legacy alias for 'model_development'.)
            If None, includes all 12 default sections.
        include_optuna : bool, default=True
            If True and tuning was done, include Optuna visualizations.
        eval_result : EvaluationResult, optional
            Evaluation result for AUC(t) and importance sections.
            If None, uses comparison.evaluation if available.
        bootstrap_results : dict, optional
            {model_name: BootstrapResult} for bootstrap raincloud and CIs.
            If None, uses comparison.bootstrap_result if available.
        outcome_name : str, optional
            Override the outcome name for display in the report (e.g., "death",
            "progression"). If None, uses the value from setup().
        time_unit : str, optional
            Override the time unit for display in the report (e.g., "months",
            "years"). If None, uses the value from setup().
        feature_groups : dict, sequence, or Series, optional
            Override the feature-to-group mapping for grouped importance plots.
            If None, uses the value from setup().
        group_colors : dict, optional
            Override the group-to-color mapping. If None, uses the value from
            setup().
        manuscript : bool, default=False
            If True and output format is HTML, also generate a companion
            ``_manuscript.md`` with pandoc-compatible markdown for PDF/Word
            manuscript preparation. If output is ``.md``, this flag is ignored
            (manuscript format is always used for ``.md``).

        Returns
        -------
        str
            Path to generated report file.

        Examples
        --------
        >>> # Full pipeline with run() (recommended)
        >>> learner = SurvivalLearner(X, y)
        >>> learner.setup()
        >>> comparison = learner.run()  # Runs tune, evaluate, bootstrap
        >>> learner.generate_report('report.html')  # Uses data from run()
        >>>
        >>> # Granular workflow (no auto-analysis)
        >>> learner.train('coxph')
        >>> learner.generate_report('report.html')  # Limited report - no eval/bootstrap
        >>>
        >>> # Granular workflow with explicit analyses
        >>> learner.train('rsf')
        >>> eval_result = learner.evaluate('rsf', bootstrap=True)
        >>> learner.generate_report('report.html', eval_result=eval_result)
        >>>
        >>> # Custom outcome naming
        >>> learner.generate_report('report.html', outcome_name='death', time_unit='months')
        >>>
        >>> # Generate HTML + companion manuscript markdown
        >>> learner.generate_report('report.html', manuscript=True)
        """
        from mlsurv.reports import ReportGenerator

        self._ensure_setup()

        if output_path is None:
            output_path = 'report.html'

        _validate_report_overrides(outcome_name, time_unit, feature_groups, group_colors)

        # Temporarily override display labels if provided, restoring on exit
        overrides = {
            'outcome_name': outcome_name,
            'time_unit': time_unit,
            'feature_groups': feature_groups,
            'group_colors': group_colors,
        }
        saved = {}
        for attr, value in overrides.items():
            if value is not None:
                saved[attr] = getattr(self._state, attr)
                setattr(self._state, attr, value)

        try:
            # Gather all report data from stored results
            eval_result, bootstrap_results = self._gather_report_data(
                eval_result, bootstrap_results
            )

            # Create generator and configure
            generator = ReportGenerator(self)
            if eval_result is not None:
                generator.set_evaluation_result(eval_result)
            if bootstrap_results is not None:
                generator.set_bootstrap_results(bootstrap_results)

            return generator.generate(
                output_path=output_path,
                title=title,
                sections=sections,
                include_optuna=include_optuna,
                manuscript=manuscript,
            )
        finally:
            for attr, value in saved.items():
                setattr(self._state, attr, value)

    def _gather_report_data(self: SurvivalLearner, eval_result=None, bootstrap_results=None):
        """Gather evaluation and bootstrap data for report.

        Uses computed properties that assemble multi-model views from
        per-model ModelResult data.

        Returns
        -------
        tuple of (eval_result, bootstrap_results)
        """
        if eval_result is None:
            eval_result = self.evaluation_result

        if bootstrap_results is None:
            has_embedded = (
                eval_result is not None
                and eval_result.bootstrap is not None
            )
            if not has_embedded:
                combined = self.bootstrap_result
                if combined is not None:
                    bootstrap_results = _split_bootstrap_to_dict(combined)

        return eval_result, bootstrap_results
