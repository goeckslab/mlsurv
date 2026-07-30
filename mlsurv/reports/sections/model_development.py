"""Model development section for report generation.

Combines cross-validation results, hyperparameter tuning, and trained model
details into a single section (formerly split across Model Development and
Hyperparameter Tuning tabs).
"""

import html as html_module
from typing import Any, Dict
import pandas as pd

from mlsurv.reports.sections._metric_glossary import metric_glossary_block
from mlsurv.reports.sections.base import BaseSection, DataInfo, FigureInfo, ALL_EXCEPTIONS, DATA_EXCEPTIONS
from mlsurv.tuning.search_spaces import DEFAULT_SPACES
from mlsurv.viz._palettes import get_model_display_name
from mlsurv.constants import get_metric_display_name, get_metric_table_name, ALL_METRICS, DEFAULT_SCORING_METRIC


def _fold_entries(scores, display_name, name, status, metric):
    """Build fold-level tidy-data rows from a list of CV scores."""
    return [
        {
            'model': display_name, 'model_type': name,
            'status': status, 'fold': fold_idx,
            'metric': metric, 'score': score,
        }
        for fold_idx, score in enumerate(scores)
    ]


# --- Optuna plot helpers (moved from tuning.py) ---

def _copy_bar_trace(trace):
    """Copy a Plotly bar trace for use in subplots."""
    import plotly.graph_objects as go
    return go.Bar(
        x=getattr(trace, 'x', None),
        y=getattr(trace, 'y', None),
        orientation=getattr(trace, 'orientation', None),
        marker=getattr(trace, 'marker', None),
        name=getattr(trace, 'name', None),
        showlegend=False,
    )


def _copy_scatter_trace(trace):
    """Copy a Plotly scatter trace for use in subplots."""
    import plotly.graph_objects as go
    kwargs = dict(
        x=getattr(trace, 'x', None),
        y=getattr(trace, 'y', None),
        marker=getattr(trace, 'marker', None),
        name=getattr(trace, 'name', None),
        showlegend=False,
    )
    mode = getattr(trace, 'mode', None)
    if mode:
        kwargs['mode'] = mode
        kwargs['line'] = getattr(trace, 'line', None)
    return go.Scatter(**kwargs)


def _format_range_tuple(value: tuple) -> str:
    """Format a (low, high[, scale]) range tuple for display."""
    if len(value) == 3:
        low, high, _scale = value
        return f"{low} - {high} (log scale)"

    if len(value) == 2:
        low, high = value
        low_display = 0 if low is None else low
        high_display = "n_features" if high is None else high
        return f"{low_display} - {high_display}"

    return str(value)


def _add_optuna_traces(fig, param_imp_fig, opt_hist_fig):
    """Add parameter importance and optimization history traces to a subplot figure."""
    if param_imp_fig is not None:
        for trace in param_imp_fig.data:
            fig.add_trace(_copy_bar_trace(trace), row=1, col=1)

    if opt_hist_fig is not None:
        for trace in opt_hist_fig.data:
            fig.add_trace(_copy_scatter_trace(trace), row=1, col=2)


class ModelDevelopmentSection(BaseSection):
    """
    Model development section showing cross-validation results, hyperparameter
    tuning, and trained model details.

    Shows:
    - CV configuration (folds, strategy)
    - Model performance table (tuned or untuned metrics)
    - CV raincloud plots
    - Tuning subsection (conditional): intro, config, selected params, Optuna plots
    - Trained model details: coefficients, convergence, feature selection

    Maps to API methods: train(), run(), tune()
    """

    name = "model_development"
    title = "Model Development"

    def __init__(self, learner, include_optuna_plots: bool = True):
        super().__init__(learner)
        self.include_optuna_plots = include_optuna_plots

    def is_available(self) -> bool:
        """Check if any models have been trained via run() or train()."""
        return len(self._get_all_models()) > 0

    def generate(self) -> Dict[str, Any]:
        """Generate model development section content."""
        md_lines = []
        figures = []
        figure_info = []
        data_info: list[DataInfo] = []

        trained_models = self._get_all_models()
        tuning_during_compare = any(
            getattr(result, 'optuna_study', None) is not None
            for result in trained_models.values()
        )

        # Summary + table of contents
        models_with_studies = self._collect_models_with_studies()
        has_tuning = bool(models_with_studies)

        from mlsurv.reports.sections._trained_details import _TrainedDetailsSubsection
        details = _TrainedDetailsSubsection(self)
        has_details = details.is_available()

        n_models = len(trained_models)
        md_lines.append(
            f"This section covers cross-validation performance for {n_models} "
            f"model{'s' if n_models != 1 else ''}"
            f"{', hyperparameter tuning' if has_tuning else ''}"
            f"{', and trained model details' if has_details else ''}."
        )
        md_lines.append("")

        # Table of contents (dynamic — anchors match heading slugs)
        toc_entries = [
            ('Configuration', 'configuration'),
            ('Cross-Validation Performance', 'cross-validation-performance'),
        ]
        if has_tuning:
            toc_entries.append(('Hyperparameter Tuning', 'hyperparameter-tuning'))
            toc_entries.append(('Selected Hyperparameters', 'selected-hyperparameters'))
        if has_details:
            toc_entries.append(('Trained Model Details', 'trained-model-details'))

        md_lines.extend(self._build_section_outline(toc_entries))

        # --- Configuration ---
        try:
            self._add_cv_config_section(md_lines, trained_models, tuning_during_compare)
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "model_development: CV configuration",
                warn_user=True,
                user_message=f"Could not generate CV configuration: {exc}",
            )
            md_lines.append(f"*[CV configuration unavailable: {exc}]*")
            md_lines.append("")

        # --- CV Performance ---
        try:
            available_metrics = self._add_performance_section(
                md_lines, trained_models, tuning_during_compare
            )
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "model_development: CV performance",
                warn_user=True,
                user_message=f"Could not generate CV performance table: {exc}",
            )
            md_lines.append(f"*[CV performance table unavailable: {exc}]*")
            md_lines.append("")
            available_metrics = []
        self._add_comparison_figures(md_lines, figures, figure_info, trained_models, available_metrics)

        # --- Tuning subsection (conditional) ---
        if has_tuning:
            md_lines.append("### Hyperparameter Tuning")
            md_lines.append("")
            self._store_tuning_tidy_data(models_with_studies)
            try:
                self._add_tuning_intro(md_lines)
            except DATA_EXCEPTIONS as exc:
                self._log_recoverable_exception(
                    exc, "model_development: tuning intro",
                    warn_user=True,
                    user_message=f"Could not generate tuning intro: {exc}",
                )
                md_lines.append(f"*[Tuning intro unavailable: {exc}]*")
                md_lines.append("")
            self._add_tuning_config_section(md_lines, models_with_studies)

            md_lines.append("### Selected Hyperparameters")
            md_lines.append("")
            md_lines.append(self._build_compact_params_table(models_with_studies))
            md_lines.append("")

            if self.include_optuna_plots:
                self._add_optuna_plots(md_lines, figures, figure_info, models_with_studies)

            # Expose tuning_param_importance via DataInfo since the
            # matching FigureInfo.data_key only links to tuning_history.
            # The key is present iff ``_store_tuning_tidy_data`` succeeded
            # in computing importances (fANOVA can fail for tiny studies
            # or return an empty dict — both are logged upstream, lines
            # 458-471). If the key is absent there is no snippet to
            # render and we skip silently, matching the upstream logging.
            if 'tuning_param_importance' in self.data:
                data_info.append(DataInfo(
                    data_key='tuning_param_importance',
                    description='Per-model Optuna parameter importance scores',
                    learner_code=(
                        "data = learner.plot_optuna(return_data=True)\n"
                        "param_importance_df = data['param_importance']"
                    ),
                ))

        # --- Trained Model Details subsection ---
        if has_details:
            md_lines.append("### Trained Model Details")
            md_lines.append("")
            details.add(md_lines, figures, figure_info)

        return {
            'markdown': '\n'.join(md_lines),
            'figures': figures,
            'figure_info': figure_info,
            'data_info': data_info,
        }

    # ------------------------------------------------------------------
    # CV configuration
    # ------------------------------------------------------------------

    def _add_cv_config_section(self, md_lines, trained_models, tuning_during_compare):
        """Add CV configuration table subsection."""
        md_lines.append("### Configuration")
        md_lines.append("")

        cv_strategy = self.learner.cv_strategy
        cv_folds = self.learner.cv_folds
        cv_repeats = self.learner.cv_repeats
        random_state = self.learner.random_state
        scoring = self.learner.scoring

        stratify_label = self._get_stratification_label(cv_strategy)
        stratified_desc = (
            f"Stratified K-Fold (maintains {stratify_label} composition in each fold)"
            if stratify_label is not None else "Stratified K-Fold"
        )

        strategy_descriptions = {
            'stratifiedkfold': stratified_desc,
            'kfold': "Standard K-Fold",
            'groupkfold': "Group K-Fold (splits by group membership)",
            'timeseries': "Time-Series Split (respects temporal ordering)",
        }
        strategy_desc = strategy_descriptions.get(cv_strategy, cv_strategy)
        scoring_display = "Custom scorer" if callable(scoring) else get_metric_display_name(scoring)
        total_splits = cv_folds * cv_repeats

        config_data = [
            ["CV Strategy", strategy_desc],
            ["Number of Folds", str(cv_folds)],
            ["CV Repeats", str(cv_repeats)],
            ["Total Splits", str(total_splits)],
            ["Scoring Metric", scoring_display],
            ["Random State", str(random_state) if random_state is not None else "None"],
            ["Tuning During CV", "Yes" if tuning_during_compare else "No"],
        ]
        if stratify_label is not None:
            config_data.append(["Stratification", stratify_label])
        config_df = pd.DataFrame(config_data, columns=["Parameter", "Value"])
        md_lines.append(self._format_table(config_df))
        md_lines.append("")

        cv_context = self._get_cv_context(
            cv_strategy, cv_folds, cv_repeats, stratify_label=stratify_label,
        )
        md_lines.append(cv_context)
        md_lines.append("")

    def _get_stratification_label(self, cv_strategy):
        """Build a human-readable stratification label from learner state.

        Returns None when the strategy is not stratified K-Fold (so the
        caller skips adding a Stratification row and falls back to generic
        prose).
        """
        if cv_strategy != 'stratifiedkfold':
            return None
        # cv_stratify_spec is recorded by setup() after any subpopulation
        # augmentation or rare-cell fallback, so it names what the folds
        # actually balance on rather than what was requested.
        spec = getattr(self.learner._state, 'cv_stratify_spec', None)
        if spec is None:
            spec = getattr(self.learner._state, 'stratify_spec', None) or 'event'
        return spec

    # ------------------------------------------------------------------
    # CV performance
    # ------------------------------------------------------------------

    def _add_performance_section(self, md_lines, trained_models, tuning_during_compare):
        """Add cross-validation performance table subsection. Returns available metrics."""
        md_lines.append("### Cross-Validation Performance")
        md_lines.append("")

        if tuning_during_compare:
            md_lines.append(
                "The table below shows model performance across cross-validation folds. "
                "Models marked as 'Tuned' underwent hyperparameter optimization. "
                "The reported metrics reflect cross-validation using the final "
                "optimized hyperparameters (i.e., the best trial's parameters "
                "retrained and evaluated across all CV folds). "
                "For a tuned model this CV score is selection-optimistic: it is the "
                "objective that tuning maximized over many trials, on folds that by "
                "default share the reporting split, so it tends to read higher than a "
                "true out-of-sample estimate. This is not leakage (the held-out test "
                "set is never used during tuning); the held-out test metric, or nested "
                "cross-validation, is the honest out-of-sample estimate."
            )
        else:
            md_lines.append(
                "The table below shows baseline model performance with default parameters."
            )
        md_lines.append("")

        md_lines.append(metric_glossary_block())
        md_lines.append("")

        available_metrics = self._get_available_cv_metrics(trained_models)

        model_data = []
        cv_scores_rows = []
        for name, result in trained_models.items():
            row, fold_rows = self._build_model_cv_row(name, result, available_metrics)
            model_data.append(row)
            cv_scores_rows.extend(fold_rows)

        if cv_scores_rows:
            self.data['cv_scores'] = pd.DataFrame(cv_scores_rows)

        if model_data:
            comparison_df = pd.DataFrame(model_data)
            sort_col = 'Harrell C' if 'Harrell C' in comparison_df.columns else comparison_df.columns[2]
            comparison_df['_sort_val'] = comparison_df[sort_col].str.extract(r'^([\d.]+)')[0].astype(float)
            comparison_df = comparison_df.sort_values('_sort_val', ascending=False)
            comparison_df = comparison_df.drop(columns=['_sort_val']).reset_index(drop=True)
            comparison_df = self._bold_best_values(comparison_df)

            md_lines.append(self._format_table(comparison_df))
            md_lines.append("")
            md_lines.append("*Values shown as mean (std). Lower is better for columns marked with an asterisk.*")
            md_lines.append("")

        return available_metrics

    def _build_model_cv_row(self, name, result, available_metrics):
        """Build a single model's CV table row and fold-level tidy data."""
        display_name = get_model_display_name(name)
        status = "Tuned" if result.is_tuned else "Baseline"
        row = {'Model': display_name, 'Status': status}

        if result.cv_metrics and available_metrics:
            fold_rows = self._populate_multi_metric_row(
                row, result, available_metrics, display_name, name, status,
            )
        else:
            fold_rows = self._populate_single_metric_row(
                row, result, display_name, name, status,
            )

        return row, fold_rows

    @staticmethod
    def _populate_multi_metric_row(row, result, available_metrics,
                                   display_name, name, status):
        """Populate table row and fold data from multi-metric CV results."""
        fold_rows = []
        for metric in available_metrics:
            if metric not in result.cv_metrics:
                continue
            m = result.cv_metrics[metric]
            row[get_metric_table_name(metric)] = f"{m['mean']:.3f} ({m['std']:.3f})"
            fold_rows.extend(
                _fold_entries(m.get('scores', []), display_name, name, status, metric)
            )
        return fold_rows

    def _populate_single_metric_row(self, row, result,
                                    display_name, name, status):
        """Populate table row and fold data from legacy single-metric CV results."""
        scoring_metric = self.learner.scoring
        if callable(scoring_metric):
            col_name, metric_name = "Score", 'custom'
        else:
            col_name, metric_name = get_metric_table_name(scoring_metric), scoring_metric

        row[col_name] = f"{result.cv_metric_mean:.3f} ({result.cv_metric_std:.3f})"

        scores = result.cv_metric_scores if result.cv_metric_scores is not None else []
        return _fold_entries(scores, display_name, name, status, metric_name)

    # ------------------------------------------------------------------
    # CV comparison figures (rainclouds)
    # ------------------------------------------------------------------

    def _add_comparison_figures(self, md_lines, figures, figure_info, trained_models, available_metrics):
        """Generate model comparison rainclouds as a multi-panel grid via learner API."""
        try:
            fig = self.learner.plot_metric(
                style="raincloud", source="cv",
                metric=available_metrics,
            )
            if fig is not None:
                filename = 'model_comparison_cv.png'
                figures.append((filename, fig))
                md_lines.append(f"![CV Model Comparison](images/{filename})")
                md_lines.append("")

                figure_info.append(FigureInfo(
                    figure_id='model_comparison_cv',
                    title='Cross-Validation Model Comparison',
                    data_description='Tidy DataFrame with CV fold scores (model_type, fold, metric_name, value)',
                    data_key='cv_scores',
                    learner_code=(
                        "fig = learner.plot_metric(\n"
                        "    style='raincloud', source='cv',\n"
                        f"    metric={available_metrics!r},\n"
                        ")"
                    ),
                ))
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "model_development: CV comparison raincloud grid",
                warn_user=True,
                user_message=f"Could not generate CV comparison rainclouds: {e}",
            )
            md_lines.append(f"*[CV comparison chart unavailable: {e}]*")
            md_lines.append("")

    # ------------------------------------------------------------------
    # Tuning methods (moved from tuning.py)
    # ------------------------------------------------------------------

    def _collect_models_with_studies(self) -> Dict:
        """Collect all models with Optuna studies."""
        return {
            name: result
            for name, result in self._get_all_models().items()
            if getattr(result, 'optuna_study', None) is not None
        }

    def _store_tuning_tidy_data(self, models_with_studies: Dict):
        """Build and store tidy history and param_importance DataFrames."""
        tuning_history_rows = []
        param_importance_rows = []

        for model_name, result in models_with_studies.items():
            study = result.optuna_study
            for trial in study.trials:
                row = {
                    'model_type': model_name,
                    'trial': trial.number,
                    'value': trial.value,
                    'state': str(trial.state.name),
                }
                for param_name, param_value in trial.params.items():
                    row[param_name] = param_value
                tuning_history_rows.append(row)

            try:
                from optuna.importance import get_param_importances
                importances = get_param_importances(study)
                for param_name, importance in importances.items():
                    param_importance_rows.append({
                        'model_type': model_name,
                        'parameter': param_name,
                        'importance': importance,
                    })
            except ImportError as e:
                self._log_recoverable_exception(
                    e,
                    f"tuning: parameter importance extraction for {model_name} — optuna not available",
                    warn_user=True,
                    user_message="optuna not available — skipping parameter importance",
                )
            except ALL_EXCEPTIONS as e:
                self._log_recoverable_exception(
                    e,
                    f"tuning: parameter importance extraction for {model_name}",
                    warn_user=True,
                    user_message=f"Could not compute parameter importance for {model_name}: {e}",
                )

        if tuning_history_rows:
            self.data['tuning_history'] = pd.DataFrame(tuning_history_rows)
        if param_importance_rows:
            self.data['tuning_param_importance'] = pd.DataFrame(param_importance_rows)

    def _add_tuning_intro(self, md_lines):
        """Add tuning section introduction and experiment path."""
        md_lines.append(
            "Hyperparameter tuning was performed using Optuna's Bayesian optimization "
            "with Tree-structured Parzen Estimator (TPE). This approach efficiently explores "
            "the hyperparameter space by learning from previous trials. The parameter importance "
            "plot shows which hyperparameters have the greatest impact on model performance."
        )
        md_lines.append("")

        experiment_path = self.learner.experiment_path
        if experiment_path is not None:
            experiment_path_str = str(experiment_path).replace('\\', '/')
            md_lines.append(
                f"Optuna studies are stored in `{experiment_path_str}/`. "
                "You can explore the optimization history interactively using "
                "[Optuna Dashboard](https://github.com/optuna/optuna-dashboard): "
                f"`optuna-dashboard sqlite:///{experiment_path_str}/<model>_study.db`"
            )
            md_lines.append("")

    def _add_tuning_config_section(self, md_lines, models_with_studies: Dict):
        """Add tuning configuration table (Models Tuned, Metric, Trials).

        When models have different trial counts, each model's count is listed
        individually; otherwise a single shared count is shown.
        """
        trial_counts = {
            name: len(result.optuna_study.trials)
            for name, result in models_with_studies.items()
        }
        unique_counts = set(trial_counts.values())
        if len(unique_counts) == 1:
            trials_display = str(unique_counts.pop())
        else:
            trials_display = ", ".join(
                f"{get_model_display_name(name)}: {count}"
                for name, count in trial_counts.items()
            )

        metrics_used = set()
        for result in models_with_studies.values():
            metrics_used.add(result.scoring_metric or DEFAULT_SCORING_METRIC)
        metrics_display = ", ".join(get_metric_display_name(m) for m in sorted(metrics_used))

        tuned_model_names = [get_model_display_name(name) for name in models_with_studies.keys()]

        # Compact config — CV Strategy/Folds already shown in main config table
        lines = []
        lines.append("### Tuning Configuration")
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        lines.append(f"| Models Tuned | {', '.join(tuned_model_names)} |")
        lines.append(f"| Metric Optimized | {metrics_display} |")
        lines.append(f"| Tuning Trials | {trials_display} |")

        md_lines.append("\n".join(lines))
        md_lines.append("")

    def _add_optuna_plots(self, md_lines, figures, figure_info, models_with_studies: Dict):
        """Generate Optuna visualization plots for each model."""
        md_lines.append("### Optimization")
        md_lines.append("")

        for model_name, result in models_with_studies.items():
            combined_fig, error = self._generate_combined_optuna_plot(model_name, result.optuna_study)
            if combined_fig is not None:
                figures.append((f'{model_name}_optuna_combined.png', combined_fig))
                md_lines.append(
                    f"![{model_name} Optimization](images/{model_name}_optuna_combined.png)"
                )
                md_lines.append("")

                display_name = get_model_display_name(model_name)
                figure_info.append(FigureInfo(
                    figure_id=f'{model_name}_optuna_combined',
                    title=f'{display_name} Hyperparameter Optimization',
                    data_description='Tidy DataFrames: history (trials), param_importance (parameter importances)',
                    data_key='tuning_history',
                    learner_code=f"fig = learner.plot_optuna(models=[{model_name!r}])",
                ))
            elif error:
                md_lines.append(f"*[Optimization plots unavailable: {error}]*")
                md_lines.append("")

    # ------------------------------------------------------------------
    # Compact params table + helpers
    # ------------------------------------------------------------------

    def _parse_param_name(self, param_name: str) -> tuple:
        """Parse a parameter name that may have a component prefix."""
        prefix_mapping = {
            'feature_selector__': 'selector',
            'imputer__': 'imputer',
            'scaler__': 'scaler',
            'model__': 'model',
        }

        for prefix, component_type in prefix_mapping.items():
            if param_name.startswith(prefix):
                base_name = param_name[len(prefix):]
                return base_name, component_type

        # Unprefixed names belong to the model; the selector's k is namespaced
        # as feature_selector__k.
        return param_name, 'model'

    def _build_compact_params_table(self, models_with_studies: Dict) -> str:
        """Build a compact parameters table with one row per model.

        Returns raw HTML wrapped in MLSURV_RAW_HTML markers so that
        ``<br>`` tags inside cells survive mistune's HTML escaping.
        """
        header = (
            '<table class="hp-params-table">'
            "<thead><tr>"
            "<th>Model</th>"
            "<th>Best Trial Score</th>"
            "<th>Selected Hyperparameters and Search Spaces</th>"
            "</tr></thead><tbody>"
        )

        row_parts = []
        for name, result in models_with_studies.items():
            study = result.optuna_study
            best_params = study.best_params
            best_value = study.best_trial.value

            search_space = DEFAULT_SPACES.get(name, {})
            metric_name = result.scoring_metric or DEFAULT_SCORING_METRIC
            metric_display = html_module.escape(f"{metric_name}={best_value:.4f}")

            param_strs = []
            for k, v in sorted(best_params.items()):
                base_name, param_type = self._parse_param_name(k)
                formatted_value = self._format_param_value(v)

                if base_name in search_space:
                    search_range = self._format_search_space_value(search_space[base_name])
                    param_strs.append(
                        html_module.escape(f"{base_name}={formatted_value} ({param_type}; {search_range})")
                    )
                else:
                    param_strs.append(
                        html_module.escape(f"{base_name}={formatted_value} ({param_type})")
                    )

            params_cell = "<br>".join(param_strs) if param_strs else "(no parameters tuned)"
            model_display = html_module.escape(get_model_display_name(name))
            row_parts.append(
                f"<tr><td>{model_display}</td>"
                f"<td>{metric_display}</td>"
                f"<td>{params_cell}</td></tr>"
            )

        table_html = header + "".join(row_parts) + "</tbody></table>"
        return (
            "<!-- MLSURV_RAW_HTML -->\n"
            + table_html + "\n"
            + "<!-- /MLSURV_RAW_HTML -->"
        )

    def _format_param_value(self, value) -> str:
        """Format parameter value for display."""
        if isinstance(value, float):
            if abs(value) < 0.001 or abs(value) > 1000:
                return f"{value:.2e}"
            return f"{value:.4f}"
        return str(value)

    def _format_search_space_value(self, value) -> str:
        """Format search space specification for display."""
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        if isinstance(value, tuple):
            return _format_range_tuple(value)
        return str(value)

    # ------------------------------------------------------------------
    # Optuna combined plots
    # ------------------------------------------------------------------

    def _generate_combined_optuna_plot(self, model_name: str, study) -> tuple:
        """Generate combined Optuna visualization with param importance and history."""
        try:
            from plotly.subplots import make_subplots

            param_imp_fig, opt_hist_fig, errors = self._collect_optuna_plots(study)

            if param_imp_fig is None and opt_hist_fig is None:
                return None, "; ".join(errors) if errors else "No plots generated"

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Parameter Importance", "Optimization History"],
                horizontal_spacing=0.12,
            )

            _add_optuna_traces(fig, param_imp_fig, opt_hist_fig)

            fig.update_layout(
                title=f"{get_model_display_name(model_name)} Hyperparameter Optimization",
                height=400, width=1000, showlegend=False,
            )
            fig.update_xaxes(title_text="Importance", row=1, col=1)
            fig.update_yaxes(title_text="Hyperparameter", row=1, col=1)
            fig.update_xaxes(title_text="Trial", row=1, col=2)
            fig.update_yaxes(title_text="Objective Value", row=1, col=2)

            return fig, None

        except ImportError as e:
            self._log_recoverable_exception(
                e,
                f"tuning: optuna plotting dependencies unavailable for {model_name}",
                warn_user=True,
                user_message=(
                    "optuna.visualization or plotly.subplots is not available. "
                    "Optuna plots will be skipped in the report."
                ),
            )
            return None, "optuna.visualization or plotly not available"
        except ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                f"tuning: combined optuna plot generation for {model_name}",
                warn_user=True,
                user_message=f"Could not generate combined Optuna plot for {model_name}: {e}",
            )
            return None, str(e)

    def _collect_optuna_plots(self, study):
        """Generate individual Optuna plots, collecting errors for failures."""
        import optuna.visualization as vis

        param_imp_fig = None
        opt_hist_fig = None
        errors = []

        try:
            param_imp_fig = vis.plot_param_importances(study)
        except ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "tuning: optuna parameter importance plot generation",
                warn_user=True,
                user_message=f"Optuna parameter importance plot failed: {e}",
            )
            errors.append(f"param importance: {e}")

        try:
            opt_hist_fig = vis.plot_optimization_history(study)
        except ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "tuning: optuna optimization history plot generation",
                warn_user=True,
                user_message=f"Optuna optimization history plot failed: {e}",
            )
            errors.append(f"optimization history: {e}")

        return param_imp_fig, opt_hist_fig, errors

    # ------------------------------------------------------------------
    # CV helpers
    # ------------------------------------------------------------------

    def _get_available_cv_metrics(self, trained_models):
        """Get metrics that are available across all models."""
        if not trained_models:
            return []

        all_metrics = None
        for result in trained_models.values():
            if result.cv_metrics:
                model_metrics = set(result.cv_metrics.keys())
                if all_metrics is None:
                    all_metrics = model_metrics
                else:
                    all_metrics = all_metrics.intersection(model_metrics)
            else:
                return []

        if not all_metrics:
            return []

        return [m for m in ALL_METRICS if m in all_metrics]

    def _get_cv_context(self, cv_strategy: str, cv_folds: int, cv_repeats: int = 1,
                        stratify_label: str = None) -> str:
        """Get contextual explanation of mean values based on CV strategy."""
        total_splits = cv_folds * cv_repeats

        if cv_repeats > 1:
            strategy_name = "Stratified" if cv_strategy == 'stratifiedkfold' else "Standard"
            strat_clause = (
                f" Each fold maintains the same {stratify_label} composition as the full dataset."
                if cv_strategy == 'stratifiedkfold' and stratify_label
                else ""
            )
            return (
                f"**Repeated {strategy_name} K-Fold:** Mean values are computed across *{cv_repeats}* repeats x *{cv_folds}* folds = "
                f"*{total_splits}* total validation splits, each with a different random "
                f"train/validation partition.{strat_clause} This provides more stable performance estimates "
                f"than single-run CV."
            )

        if cv_strategy == 'stratifiedkfold':
            label = stratify_label or "event"
            return (
                f"**Stratified K-Fold:** Mean values are computed across *{cv_folds}* folds of the training data, "
                f"where each fold maintains the same {label} composition as the full dataset. "
                f"Standard deviations indicate model stability across different train/validation splits."
            )
        elif cv_strategy == 'timeseries':
            return (
                f"**Time-Series Split:** Mean values are computed across *{cv_folds}* temporal splits where training always "
                f"precedes validation in time, simulating prospective prediction. "
                f"This prevents future data leakage for time-ordered datasets."
            )
        elif cv_strategy == 'groupkfold':
            return (
                f"**Group K-Fold:** Mean values are computed across *{cv_folds}* folds where observations are split "
                f"by group membership, ensuring no group appears in both train and validation. "
                f"Groups are also kept intact across the initial train/test split, so no group leaks "
                f"between the held-out test set and training data. "
                f"This prevents data leakage when observations within groups are correlated."
            )
        else:  # kfold
            return (
                f"**Standard K-Fold:** Mean values are computed across *{cv_folds}* random folds of the training data. "
                f"Standard deviations reflect performance variability across different data partitions."
            )
