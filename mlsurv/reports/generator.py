"""Report generator for mlsurv survival analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mlsurv.reports.sections import (
    BaseSection,
    SummarySection,
    DataPreprocessingSection,
    ModelDevelopmentSection,
    EvaluationSection,
    FeatureAnalysisSection,
    SubpopulationAnalysisSection,
    ValidationSection,
    BenchmarkSection,
    LimitationsSection,
    EnvironmentSection,
    ReferencesSection,
    TripodChecklistSection,
)
from mlsurv.reports.renderers import ManuscriptRenderer, HTMLRenderer

if TYPE_CHECKING:
    import pandas as pd
    from mlsurv.learner import SurvivalLearner
    from mlsurv.results.containers import EvaluationResult, BootstrapResult


# Section registry with default order.
# 13-section registry (all default) aligned with API methods and
# TRIPOD+AI guidelines.
# 1. Executive Summary [all methods]
# 2. Data & Preprocessing [setup()]
# 3. Model Development [train()/run()/tune() — includes tuning + trained details]
# 4. Model Evaluation [evaluate() + bootstrap()]
# 5. Split Results [tabbed Train/Validation/Test summaries]
# 6. Feature Analysis [evaluate() importance]
# 7. Subpopulation Analysis [subpopulation-level metrics — conditional]
# 8. External Validation [validate() - conditional]
# 9. Benchmarking [conditional on reference targets]
# 10. Limitations & Caveats [always — TRIPOD+AI Item 26]
# 11. Environment & Reproducibility [always]
# 12. References [combined acknowledgments, citation, bibliography]
# 13. TRIPOD+AI Checklist [conditional on setup()]
SECTION_REGISTRY = {
    'summary': SummarySection,
    'data_preprocessing': DataPreprocessingSection,
    'model_development': ModelDevelopmentSection,
    'evaluation': EvaluationSection,
    'feature_analysis': FeatureAnalysisSection,
    'subpopulation_analysis': SubpopulationAnalysisSection,
    'validation': ValidationSection,
    'benchmarking': BenchmarkSection,
    'limitations': LimitationsSection,
    'environment': EnvironmentSection,
    'references': ReferencesSection,
    'tripod_checklist': TripodChecklistSection,
}

DEFAULT_SECTION_ORDER = [
    'summary', 'data_preprocessing', 'model_development',
    'evaluation',
    'feature_analysis', 'subpopulation_analysis',
    'validation', 'benchmarking',
    'limitations', 'environment', 'references', 'tripod_checklist'
]

# Legacy alias: 'tuning' resolves to 'model_development'
_SECTION_ALIASES = {'tuning': 'model_development'}


class ReportGenerator:
    """
    Orchestrate report generation for survival analysis.

    Detects available data and generates appropriate sections.
    Supports markdown and HTML output formats.

    Parameters
    ----------
    learner : SurvivalLearner
        The learner instance containing all analysis data.

    Attributes
    ----------
    data : dict
        Access all figure data as tidy DataFrames. Keys are formatted as
        '{section_name}_{data_key}', e.g., 'feature_analysis_correlation',
        'model_development_cv_scores'. Available after generate() is called.
    figures : dict
        Access all figures as Plotly Figure objects. Keys are formatted as
        '{section_name}_{figure_id}', e.g., 'feature_analysis_correlation_heatmap'.
        Available after generate() is called.

    Examples
    --------
    >>> generator = ReportGenerator(learner)
    >>> generator.generate('report.md', title='My Analysis')
    >>>
    >>> # Access tidy DataFrames
    >>> print(generator.data.keys())
    >>> corr_df = generator.data['feature_analysis_correlation']
    """

    def __init__(self, learner: 'SurvivalLearner'):
        self.learner = learner
        self._eval_result: Optional['EvaluationResult'] = None
        self._bootstrap_results: Optional[Dict[str, 'BootstrapResult']] = None
        self._sections: List[BaseSection] = []  # Store generated section instances
        self._data: Optional[Dict[str, 'pd.DataFrame']] = None  # Cached aggregated data
        self._figures: Optional[Dict[str, Any]] = None  # Cached figures

    @property
    def data(self) -> Dict[str, 'pd.DataFrame']:
        """
        Access all figure data as tidy DataFrames.

        Returns
        -------
        dict
            Dictionary mapping '{section_name}_{data_key}' to pandas DataFrame.
            Only populated after generate() is called.

        Examples
        --------
        >>> report.data['feature_analysis_correlation']  # Correlation matrix as tidy DataFrame
        >>> report.data['model_development_cv_scores']  # CV scores by fold as tidy DataFrame
        >>> report.data['evaluation_bootstrap']  # Bootstrap iterations as tidy DataFrame
        """
        if self._data is None:
            self._data = {}
            for section in self._sections:
                for key, df in section.data.items():
                    self._data[f"{section.name}_{key}"] = df
        return self._data

    @property
    def figures(self) -> Dict[str, Any]:
        """
        Access all figures as Plotly Figure objects.

        Returns
        -------
        dict
            Dictionary mapping '{section_name}_{figure_id}' to Plotly Figure.
            Only populated after generate() is called.

        Examples
        --------
        >>> report.figures['feature_analysis_correlation_heatmap']  # Correlation heatmap
        >>> report.figures['model_development_model_comparison_harrell']  # CV model comparison raincloud
        """
        if self._figures is None:
            self._figures = {}
        return self._figures

    def set_evaluation_result(self, result: 'EvaluationResult') -> 'ReportGenerator':
        """
        Set evaluation result for AUC(t) and importance sections.

        Also extracts embedded bootstrap results if present.

        Parameters
        ----------
        result : EvaluationResult
            Result from evaluate().

        Returns
        -------
        self
            For method chaining.
        """
        self._eval_result = result

        # Extract embedded bootstrap results if present and not already set
        if self._bootstrap_results is None and result.bootstrap is not None:
            # Get model names from the bootstrap summary DataFrame
            model_names = result.bootstrap.summary['model_type'].unique()
            if len(model_names) == 1:
                self._bootstrap_results = {model_names[0]: result.bootstrap}
            else:
                # For multi-model eval, split bootstrap into per-model results
                self._bootstrap_results = self._split_bootstrap_by_model(
                    result.bootstrap, model_names
                )

        return self

    def set_bootstrap_results(
        self,
        results: Dict[str, 'BootstrapResult']
    ) -> 'ReportGenerator':
        """
        Set bootstrap results for confidence intervals and raincloud.

        Parameters
        ----------
        results : dict
            {model_name: BootstrapResult} dictionary.

        Returns
        -------
        self
            For method chaining.
        """
        self._bootstrap_results = results
        return self

    # Map file extensions to format names
    _FORMAT_BY_EXT: Dict[str, str] = {'.html': 'html', '.md': 'markdown'}

    def generate(
        self,
        output_path: str,
        title: str = "Survival Analysis Report",
        sections: Optional[List[str]] = None,
        include_optuna: bool = True,
        metric: Optional[str] = None,
        manuscript: bool = False,
    ) -> str:
        """
        Generate comprehensive analysis report.

        Parameters
        ----------
        output_path : str
            Path to output file. Format is inferred from extension:
            '.html' for HTML, '.md' for markdown.
        title : str
            Report title.
        sections : list, optional
            Specific sections to include, or None for all available.
        include_optuna : bool
            If True, include Optuna visualizations in model development section.
        metric : str, optional
            Metric to display in results table. If None, uses learner.scoring.
            Options: 'harrell', 'ipcw', 'integrated_brier', 'ctd'
        manuscript : bool
            If True and format is HTML, also generate a companion
            ``_manuscript.md`` file with pandoc-compatible markdown.

        Returns
        -------
        str
            Path to generated report file.

        Raises
        ------
        ValueError
            If no report sections are available (setup, run, or evaluate
            has not been called yet).
        """
        fmt = self._detect_format(output_path)

        # Auto-wire evaluation/bootstrap from the learner's canonical state for
        # any slot the caller didn't inject. learner.generate_report() sets both
        # via set_evaluation_result()/set_bootstrap_results(); a bare
        # ReportGenerator(learner).generate() sets neither, and without them the
        # evaluation sections degrade silently to a partial report. Fill
        # each slot independently so a caller who set only one still gets the
        # other auto-wired instead of silently dropping its figures.
        if self._eval_result is None or self._bootstrap_results is None:
            eval_result, bootstrap_results = self.learner._gather_report_data()
            if self._eval_result is None and eval_result is not None:
                self.set_evaluation_result(eval_result)
            if self._bootstrap_results is None and bootstrap_results is not None:
                self.set_bootstrap_results(bootstrap_results)

        print("Generating report...", end="", flush=True)

        sections = self._resolve_sections(sections)

        # Create section instances and filter to available
        section_instances = self._create_sections(sections, include_optuna, metric)
        available_sections = [s for s in section_instances if s.is_available()]

        # Store sections for data access and clear cached data/figures
        self._sections = available_sections
        self._data = None  # Clear cache so it's rebuilt on next access
        self._figures = {}  # Initialize for this generation

        if not available_sections:
            raise ValueError(
                "No sections available for report. "
                "Run setup(), run(), or evaluate() first."
            )

        # Generate section content
        section_content, all_figures = self._generate_section_content(
            available_sections
        )

        # Build metadata for header bar
        metadata = self._build_report_metadata()

        # Select renderer and produce output
        renderers = {'markdown': ManuscriptRenderer, 'html': HTMLRenderer}
        renderer = renderers[fmt](output_path, title=title)

        content = renderer.render(section_content, metadata=metadata)
        failed = renderer.save_figures(all_figures)
        content = renderer.strip_missing_figures(content, failed)
        output_file = renderer.write(content)

        print(f" done. Report saved to: {output_file}")

        # Optionally produce a companion manuscript markdown alongside HTML
        if manuscript and fmt == 'html':
            try:
                out = Path(output_path)
                ms_path = out.with_name(f"{out.stem}_manuscript.md")
                ms_renderer = ManuscriptRenderer(str(ms_path), title=title)
                ms_content = ms_renderer.render(section_content, metadata=metadata)
                ms_failed = ms_renderer.save_figures(all_figures)
                ms_content = ms_renderer.strip_missing_figures(ms_content, ms_failed)
                ms_renderer.write(ms_content)
                print(f"Manuscript markdown saved to: {ms_path}")
            except (OSError, ValueError) as exc:
                import warnings
                from mlsurv._warnings import InternalWarning
                warnings.warn(
                    f"HTML report saved successfully, but manuscript "
                    f"companion failed: {exc}",
                    InternalWarning,
                )

        return str(output_file)

    def _build_report_metadata(self) -> dict:
        """Build metadata dict for the report header bar."""
        import datetime
        import mlsurv
        from mlsurv.logging_utils import get_logger, log_exception

        _logger = get_logger("reports.generator")

        meta = {
            'version': mlsurv.__version__,
            'timestamp': datetime.datetime.now(datetime.timezone.utc).strftime(
                '%Y-%m-%d %H:%M:%S UTC'
            ),
        }

        state = getattr(self.learner, '_state', None)
        if state is None or not getattr(state, 'is_setup', False):
            return meta

        # Block 1: dimensions (n_samples, n_features)
        try:
            n_train = len(state.X_train) if state.X_train is not None else 0
            n_test = len(state.X_test) if state.X_test is not None else 0
            meta['n_samples'] = int(n_train + n_test)
            meta['n_features'] = int(state.X_train.shape[1]) if state.X_train is not None else 0
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            log_exception(_logger, exc, "report metadata: dataset dimensions")

        # Block 2: events (n_events)
        try:
            n_events = 0
            if state.y_train is not None:
                n_events += int(state.y_train['event'].sum())
            if state.y_test is not None:
                n_events += int(state.y_test['event'].sum())
            meta['n_events'] = n_events
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            log_exception(_logger, exc, "report metadata: event count")

        # Block 3: labels (outcome_name, time_unit)
        try:
            if state.outcome_name != "event":
                meta['outcome_name'] = state.outcome_name
            if state.time_unit:
                meta['time_unit'] = state.time_unit
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            log_exception(_logger, exc, "report metadata: outcome labels")

        return meta

    def _detect_format(self, output_path: str) -> str:
        """Infer report format from file extension.

        Returns
        -------
        str
            'html' or 'markdown'.

        Raises
        ------
        ValueError
            If the extension is not '.html' or '.md'.
        """
        ext = Path(output_path).suffix.lower()
        fmt = self._FORMAT_BY_EXT.get(ext)
        if fmt is None:
            raise ValueError(
                f"Unknown file extension '{ext}'. Use '.html' or '.md'."
            )
        return fmt

    @staticmethod
    def _resolve_sections(sections: Optional[List[str]]) -> List[str]:
        """Return validated section names, defaulting to all sections.

        Resolves legacy aliases (e.g. 'tuning' → 'model_development') and
        deduplicates so a section is never generated twice.

        Parameters
        ----------
        sections : list or None
            Requested section names, or None for all.

        Returns
        -------
        list
            Validated section name list.

        Raises
        ------
        ValueError
            If any section name is not in the registry (after alias resolution).
        """
        if sections is None:
            return DEFAULT_SECTION_ORDER

        resolved = []
        seen = set()
        for name in sections:
            canonical = _SECTION_ALIASES.get(name, name)
            if canonical not in SECTION_REGISTRY:
                available = list(SECTION_REGISTRY.keys()) + list(_SECTION_ALIASES.keys())
                raise ValueError(
                    f"Unknown section: '{name}'. "
                    f"Available: {available}"
                )
            if canonical not in seen:
                resolved.append(canonical)
                seen.add(canonical)
        return resolved

    def _generate_section_content(
        self, available_sections: List[BaseSection]
    ) -> tuple:
        """Generate content for each available section.

        Returns
        -------
        tuple of (section_content, all_figures)
            section_content is a list of dicts; all_figures is a flat list of
            (filename, figure) tuples.
        """
        from mlsurv.logging_utils import get_logger, log_exception

        section_content = []
        all_figures = []

        for section in available_sections:
            try:
                result = section.generate()
            except (MemoryError, RecursionError):
                raise
            except Exception as exc:
                log_exception(
                    get_logger("reports.generator"), exc,
                    f"section generation: {section.title}",
                    warn_user=True,
                    user_message=f"Section '{section.title}' could not be generated",
                )
                section_content.append({
                    'title': section.title,
                    'markdown': f"*[Section unavailable: {type(exc).__name__}: {exc}]*",
                    'figures': [],
                    'figure_info': [],
                    'tabs': None,
                    'section_name': section.name,
                    'data_keys': [],
                    'data_info': [],
                })
                continue

            section_content.append({
                'title': section.title,
                'markdown': result['markdown'],
                'figures': result['figures'],
                'figure_info': result.get('figure_info', []),
                'tabs': result.get('tabs'),
                'section_name': section.name,
                'data_keys': list(section.data.keys()),
                'data_info': result.get('data_info', []),
            })
            all_figures.extend(result['figures'])

            # Store figures with section-prefixed keys
            for filename, fig in result['figures']:
                figure_id = filename.rsplit('.', 1)[0]  # Remove extension
                self._figures[f"{section.name}_{figure_id}"] = fig

        return section_content, all_figures

    def _split_bootstrap_by_model(
        self,
        bootstrap: 'BootstrapResult',
        model_names: list,
    ) -> Dict[str, 'BootstrapResult']:
        """Split a multi-model BootstrapResult into per-model results."""
        from mlsurv.results.containers import BootstrapResult

        result = {}
        for model_name in model_names:
            # Filter each DataFrame by model_type
            summary = bootstrap.summary[
                bootstrap.summary['model_type'] == model_name
            ].copy()
            iterations = bootstrap.iterations[
                bootstrap.iterations['model_type'] == model_name
            ].copy()

            # Optional DataFrames
            predictions = None
            if bootstrap.predictions is not None:
                predictions = bootstrap.predictions[
                    bootstrap.predictions['model_type'] == model_name
                ].copy()

            auc_t = None
            if bootstrap.auc_t is not None:
                auc_t = bootstrap.auc_t[
                    bootstrap.auc_t['model_type'] == model_name
                ].copy()

            brier_t = None
            if bootstrap.brier_t is not None:
                brier_t = bootstrap.brier_t[
                    bootstrap.brier_t['model_type'] == model_name
                ].copy()

            importance = None
            if bootstrap.importance is not None:
                importance = bootstrap.importance[
                    bootstrap.importance['model_type'] == model_name
                ].copy()

            result[model_name] = BootstrapResult(
                summary=summary,
                iterations=iterations,
                predictions=predictions,
                auc_t=auc_t,
                brier_t=brier_t,
                importance=importance,
                lower_is_better=bootstrap.lower_is_better,
                ci_level=bootstrap.ci_level,
            )

        return result

    # Supported export formats for figures and data
    _FIGURE_FORMATS = {'.html', '.svg', '.png', '.pdf', '.jpeg', '.webp'}
    _DATA_WRITERS = {
        '.csv': lambda df, path, **kw: df.to_csv(str(path), index=False, **kw),
        '.tsv': lambda df, path, **kw: df.to_csv(str(path), index=False, sep='\t', **kw),
        '.xlsx': lambda df, path, **kw: df.to_excel(
            str(path), index=False, **kw,
        ),
    }

    def export_figure(self, figure_key: str, output_path: str, **kwargs) -> str:
        """
        Export a single figure to file.

        Parameters
        ----------
        figure_key : str
            Key in self.figures, e.g. 'evaluation_auct_curves'.
        output_path : str
            Destination path. Format inferred from extension:
            .html -> write_html, .svg/.png/.pdf/.jpeg/.webp -> write_image.
        **kwargs
            Forwarded to write_html() or write_image().

        Returns
        -------
        str
            Absolute path to the written file.

        Raises
        ------
        KeyError
            If *figure_key* is not found.
        ValueError
            If the file extension is not supported.
        FigureExportTimeout
            If the static-image export (non-HTML) exceeds its timeout because
            kaleido/Chrome is unresponsive.
        """
        if figure_key not in self.figures:
            available = ', '.join(sorted(self.figures.keys()))
            raise KeyError(
                f"Figure '{figure_key}' not found. Available: {available}"
            )

        ext = Path(output_path).suffix.lower()
        if ext not in self._FIGURE_FORMATS:
            raise ValueError(
                f"Unsupported figure format '{ext}'. "
                f"Use one of: {', '.join(sorted(self._FIGURE_FORMATS))}"
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fig = self.figures[figure_key]
        if ext == '.html':
            fig.write_html(str(out), **kwargs)
        else:
            # Bounded so a stalled kaleido/Chrome raises FigureExportTimeout
            # instead of hanging the caller indefinitely.
            from mlsurv.reports._figure_export import write_image_safe
            write_image_safe(fig, out, **kwargs)

        return str(out.resolve())

    def export_data(self, data_key: str, output_path: str, **kwargs) -> str:
        """
        Export a single tidy DataFrame to file.

        Parameters
        ----------
        data_key : str
            Key in self.data, e.g. 'evaluation_auct'.
        output_path : str
            Destination path. Format inferred from extension:
            .csv, .tsv, .xlsx.
        **kwargs
            Forwarded to the pandas writer.

        Returns
        -------
        str
            Absolute path to the written file.

        Raises
        ------
        KeyError
            If *data_key* is not found.
        ValueError
            If the file extension is not supported.
        """
        if data_key not in self.data:
            available = ', '.join(sorted(self.data.keys()))
            raise KeyError(
                f"Data '{data_key}' not found. Available: {available}"
            )

        ext = Path(output_path).suffix.lower()
        writer = self._DATA_WRITERS.get(ext)
        if writer is None:
            raise ValueError(
                f"Unsupported data format '{ext}'. "
                f"Use one of: {', '.join(sorted(self._DATA_WRITERS))}"
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer(self.data[data_key], out, **kwargs)

        return str(out.resolve())

    def _create_sections(
        self,
        section_names: List[str],
        include_optuna: bool,
        metric: Optional[str] = None,
    ) -> List[BaseSection]:
        """Create section instances with appropriate data."""
        # Map section names to their extra constructor kwargs (beyond learner).
        # Sections not listed here receive only the learner.
        extra_kwargs = {
            'summary': dict(
                bootstrap_results=self._bootstrap_results,
                eval_result=self._eval_result,
            ),
            'data_preprocessing': dict(eval_result=self._eval_result),
            'model_development': dict(include_optuna_plots=include_optuna),
            'evaluation': dict(
                eval_result=self._eval_result,
                bootstrap_results=self._bootstrap_results,
                metric=metric,
            ),
            'feature_analysis': dict(eval_result=self._eval_result),
        }

        sections = []
        for name in section_names:
            section_class = SECTION_REGISTRY[name]
            kwargs = extra_kwargs.get(name, {})
            sections.append(section_class(self.learner, **kwargs))

        return sections
