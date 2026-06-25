"""Executive summary section for report generation."""

from typing import Any, Dict, Optional, TYPE_CHECKING
import warnings

import numpy as np

from mlsurv._shared import _display_cv_score
from mlsurv._warnings import InternalWarning
from mlsurv.constants import DEFAULT_SCORING_METRIC, get_metric_display_name
from mlsurv.results.printing import _has_valid_point_column
from mlsurv.reports._constants import (
    DEFAULT_RISK_GROUPS,
    P_THRESHOLD_EXCELLENT,
    P_THRESHOLD_GOOD,
    P_THRESHOLD_SIGNIFICANT,
    format_pvalue,
)
from mlsurv.reports._text import pluralize, title_if_lower
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, BaseSection, DATA_EXCEPTIONS, FigureInfo
from mlsurv.viz._palettes import get_model_display_name

if TYPE_CHECKING:
    from mlsurv.results.containers import EvaluationResult, ModelResult


# ------------------------------------------------------------------
# Summary helpers (extracted from SummarySection methods)
# ------------------------------------------------------------------

def _format_model_listing(model_names):
    """Format model names into a sentence fragment."""
    n_models = len(model_names)
    if n_models == 1:
        return f"using *{model_names[0]}*"
    if n_models == 2:
        return f"compared *{model_names[0]}* and *{model_names[1]}*"
    names_em = ', '.join(f"*{n}*" for n in model_names[:-1])
    return f"compared *{n_models}* survival models — {names_em}, and *{model_names[-1]}* —"


def _format_methods_list(cv_desc, tuned_models, has_bootstrap):
    """Format the methods list (CV, tuning, bootstrap) into a sentence fragment."""
    methods = [cv_desc]
    if tuned_models:
        methods.append("hyperparameter tuning")
    if has_bootstrap:
        methods.append("bootstrap evaluation")
    if len(methods) == 1:
        return methods[0]
    if len(methods) == 2:
        return f"{methods[0]} and {methods[1]}"
    return f"{', '.join(methods[:-1])}, and {methods[-1]}"


def _classify_p_value(p_value):
    """Classify a p-value into quality category and formatted string.

    Returns ``(quality, p_str)`` tuple. NaN / None / non-finite inputs
    return ``("unknown", "p = *—*")`` so the downstream comparisons
    don't fall through to ``"poor"`` silently, and so ``None`` doesn't
    raise ``TypeError`` in the comparisons.
    """
    p_str = f"p = *{format_pvalue(p_value)}*"
    if p_value is None or not (
        isinstance(p_value, (int, float)) and np.isfinite(p_value)
    ):
        return "unknown", p_str
    if p_value < P_THRESHOLD_EXCELLENT:
        quality = "excellent"
    elif p_value < P_THRESHOLD_GOOD:
        quality = "good"
    elif p_value < P_THRESHOLD_SIGNIFICANT:
        quality = "fair"
    else:
        quality = "poor"
    return quality, p_str


class SummarySection(BaseSection):
    """
    Executive summary section providing a high-level overview of the analysis.

    Shows:
    - Analysis date and configuration
    - Best performing model and key metrics
    - Brief interpretation of results
    """

    name = "summary"
    title = "Executive Summary"

    def __init__(
        self,
        learner,
        bootstrap_results: Dict = None,
        eval_result: Optional['EvaluationResult'] = None,
    ):
        """
        Initialize summary section.

        Parameters
        ----------
        learner : SurvivalLearner
            The learner instance.
        bootstrap_results : dict, optional
            {model_name: BootstrapResult} for test set results narrative.
        eval_result : EvaluationResult, optional
            Evaluation result with importance data for feature summary.
        """
        super().__init__(learner)
        self.bootstrap_results = bootstrap_results or {}
        self.eval_result = eval_result

    def is_available(self) -> bool:
        """Summary is always available if setup has been called."""
        return self.learner.is_setup

    def generate(self) -> Dict[str, Any]:
        """Generate executive summary content."""
        md_lines = []
        figures = []

        trained_models = self._get_all_models()

        # Pre-compute shared values used across multiple paragraphs
        try:
            n_total = len(self.learner.X_train) + len(self.learner.X_test)
            n_train = len(self.learner.X_train)
            n_test = len(self.learner.X_test)
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(exc, "summary: could not access train/test data")
            n_total = n_train = n_test = 0

        tuned_models = self._get_tuned_models() if trained_models else {}
        cv_desc = self._build_cv_description() if trained_models else ""

        # Key findings box first
        if trained_models and n_total > 0:
            try:
                self._add_key_findings_box(md_lines, trained_models, n_total)
            except ALL_EXCEPTIONS as exc:
                self._log_recoverable_exception(exc, "summary: key findings box")

        # Paragraph 1: what was done + best result
        try:
            self._add_methods_paragraph(
                md_lines, trained_models, n_total, n_train, n_test, cv_desc, tuned_models
            )
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(exc, "summary: methods paragraph")

        # Paragraph 2: tuning (conditional)
        if trained_models:
            try:
                tuning_sentence = self._generate_tuning_narrative(
                    trained_models, tuned_models, cv_desc
                )
                if tuning_sentence:
                    md_lines.append("")
                    md_lines.append(tuning_sentence)
            except ALL_EXCEPTIONS as exc:
                self._log_recoverable_exception(exc, "summary: tuning narrative")

        # Paragraph 3: model quality + features
        try:
            self._add_quality_paragraph(md_lines, trained_models)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(exc, "summary: quality paragraph")

        # Paragraph 4: external validation (conditional)
        validation_summary = self._get_validation_summary()
        if validation_summary:
            md_lines.append("")
            md_lines.append(validation_summary)

        # Limitations cross-reference
        try:
            self._add_limitations_line(md_lines)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(exc, "summary: limitations cross-reference")

        md_lines.append("")

        # Key figures for visual assessment
        figure_info = []

        has_benchmark = self.learner.benchmark_result is not None
        references = "all" if has_benchmark else None

        if trained_models:
            self._add_cv_raincloud(md_lines, figures, figure_info)

        if self.bootstrap_results:
            self._add_bootstrap_raincloud(md_lines, figures, figure_info,
                                          references=references)

        self._add_auct_figure(md_lines, figures, figure_info,
                              references=references)

        self._add_importance_figure(md_lines, figures, figure_info)

        return {
            'markdown': '\n'.join(md_lines),
            'figures': figures,
            'figure_info': figure_info,
        }

    # ------------------------------------------------------------------
    # Executive summary figures
    # ------------------------------------------------------------------

    def _add_cv_raincloud(self, md_lines, figures, figure_info):
        """Add CV scoring metric raincloud to the summary."""
        try:
            scoring = self.learner.scoring
            fig = self.learner.plot_metric(
                style='raincloud', source='cv', metric=scoring,
                title='Cross-Validation Performance',
                width=550,
            )
            if fig is None:
                return

            figures.append(('cv_raincloud.png', fig))
            md_lines.append("![CV Performance](images/cv_raincloud.png)")
            md_lines.append("")

            figure_info.append(FigureInfo(
                figure_id='cv_raincloud',
                title='Cross-Validation Performance',
                data_description='CV fold scores per model',
                learner_code=(
                    "fig = learner.plot_metric(\n"
                    "    style='raincloud', source='cv',\n"
                    f"    metric='{scoring}',\n"
                    ")"
                ),
            ))
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "summary: CV raincloud",
                warn_user=True,
                user_message=f"Could not generate CV performance plot: {exc}",
            )
            md_lines.append(f"*[CV performance plot unavailable: {exc}]*")
            md_lines.append("")

    def _add_bootstrap_raincloud(self, md_lines, figures, figure_info,
                                 references=None):
        """Add bootstrap scoring metric raincloud to the summary."""
        if not self.bootstrap_results:
            return

        try:
            scoring = self.learner.scoring
            fig = self.learner.plot_metric(
                style='raincloud', source='bootstrap', metric=scoring,
                title='Bootstrap Performance (Test Set)',
                references=references,
                width=550,
            )
            if fig is None:
                return

            figures.append(('bootstrap_raincloud.png', fig))
            md_lines.append("![Bootstrap Performance](images/bootstrap_raincloud.png)")
            md_lines.append("")

            ref_arg = "\n    references='all'," if references else ""
            figure_info.append(FigureInfo(
                figure_id='bootstrap_raincloud',
                title='Bootstrap Performance (Test Set)',
                data_description='Bootstrap iteration scores per model',
                learner_code=(
                    "fig = learner.plot_metric(\n"
                    "    style='raincloud', source='bootstrap',\n"
                    f"    metric='{scoring}',{ref_arg}\n"
                    ")"
                ),
            ))
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "summary: bootstrap raincloud",
                warn_user=True,
                user_message=f"Could not generate bootstrap performance plot: {exc}",
            )
            md_lines.append(f"*[Bootstrap performance plot unavailable: {exc}]*")
            md_lines.append("")

    def _add_auct_figure(self, md_lines, figures, figure_info,
                         references=None):
        """Add AUC(t) curves to the summary."""
        if self.eval_result is None or self.eval_result.auc_t is None:
            return
        if self.eval_result.auc_t.empty:
            return

        try:
            fig = self.learner.plot_metric_time(
                references=references,
                time_unit=self.time_unit,
                metric="auc",
            )
            if fig is None:
                return

            figures.append(('summary_auct.png', fig))
            md_lines.append("![Time-Dependent AUC](images/summary_auct.png)")
            md_lines.append("")

            ref_snippet = "references='all', " if references else ""
            tu_snippet = f"time_unit='{self.time_unit}', " if self.time_unit else ""
            args_snippet = f"{ref_snippet}{tu_snippet}".rstrip(", ")

            figure_info.append(FigureInfo(
                figure_id='summary_auct',
                title='Time-Dependent AUC Curves',
                data_description='AUC(t) values at each evaluation time',
                learner_code=(
                    f"fig = learner.plot_metric_time({args_snippet})"
                ),
            ))
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "summary: AUC(t) figure",
                warn_user=True,
                user_message=f"Could not generate AUC(t) plot: {exc}",
            )
            md_lines.append(f"*[AUC(t) plot unavailable: {exc}]*")
            md_lines.append("")

    def _add_importance_figure(self, md_lines, figures, figure_info):
        """Add feature importance figure to the summary.

        Uses a heatmap for multi-model (>=2) and a bar chart for single model.
        """
        if self.eval_result is None or self.eval_result.importance is None:
            return
        if len(self.eval_result.importance) == 0:
            return

        try:
            n_models = self.eval_result.importance['model_type'].nunique()
            if n_models >= 2:
                self._add_importance_heatmap(md_lines, figures, figure_info)
            else:
                self._add_importance_bar(md_lines, figures, figure_info)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "summary: importance figure",
                warn_user=True,
                user_message=f"Could not generate importance plot: {exc}",
            )
            md_lines.append(f"*[Importance plot unavailable: {exc}]*")
            md_lines.append("")

    def _add_importance_heatmap(self, md_lines, figures, figure_info):
        """Add multi-model importance heatmap."""
        fig = self.learner.plot_importance(
            style="heatmap", n=10, title='Feature Importance',
            **self._group_plot_kwargs(),
        )
        if fig is None:
            return

        figures.append(('importance_heatmap.png', fig))
        md_lines.append("![Feature Importance](images/importance_heatmap.png)")
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='importance_heatmap',
            title='Feature Importance Heatmap',
            data_description='Feature importance per model (top features)',
            learner_code="fig = learner.plot_importance(style='heatmap', n=10)",
        ))

    def _add_importance_bar(self, md_lines, figures, figure_info):
        """Add single-model importance bar chart."""
        fig = self.learner.plot_importance(
            n=10, title='Feature Importance',
            **self._group_plot_kwargs(),
        )
        if fig is None:
            return

        figures.append(('importance.png', fig))
        md_lines.append("![Feature Importance](images/importance.png)")
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='importance',
            title='Feature Importance',
            data_description='Feature importance (top features)',
            learner_code="fig = learner.plot_importance(n=10)",
        ))

    def _add_methods_paragraph(
        self, md_lines: list, trained_models: Dict,
        n_total: int, n_train: int, n_test: int,
        cv_desc: str, tuned_models: Dict,
    ) -> None:
        """Build the main paragraph: data + models + best result."""
        if n_total == 0 or not trained_models:
            return
        model_names = [get_model_display_name(name) for name in trained_models.keys()]
        has_bootstrap = bool(self.bootstrap_results)
        n_events = int(
            self.learner.y_train['event'].sum() + self.learner.y_test['event'].sum()
        )
        event_rate = n_events / n_total * 100
        n_features = self.learner.X.shape[1]
        outcome = self.outcome_name

        model_part = _format_model_listing(model_names)
        methods_str = _format_methods_list(cv_desc, tuned_models, has_bootstrap)

        train_pct = n_train / n_total * 100
        test_pct = n_test / n_total * 100

        sentence = (
            f"This analysis {model_part} on *{n_total:,}* observations "
            f"(*{n_train:,}* training [{train_pct:.0f}%], *{n_test:,}* test [{test_pct:.0f}%]; "
            f"*{n_features}* features; {outcome} rate *{event_rate:.1f}%*) "
            f"using {methods_str}."
        )

        # Best model result
        scoring_metric = self.learner.scoring
        cv_metric_display = get_metric_display_name(scoring_metric)
        from mlsurv._shared import cv_sorted_model_keys
        best_name = cv_sorted_model_keys(trained_models)[0]
        best_score = _display_cv_score(trained_models[best_name].cv_metric_mean, scoring_metric)
        best_display = get_model_display_name(best_name)

        sentence += (
            f" *{best_display}* achieved the best {cv_desc} mean "
            f"{cv_metric_display} of *{best_score:.3f}*"
        )

        if 'harrell' in scoring_metric:
            pct = best_score * 100
            quality = self._interpret_quality(best_score)
            sentence += (
                f", correctly ranking risk in {pct:.0f}% of comparable pairs "
                f"({quality} discrimination)"
            )
        sentence += "."

        md_lines.append(sentence)

    def _add_quality_paragraph(self, md_lines: list, trained_models: Dict) -> None:
        """Build paragraph 3: model quality metrics + feature importance."""
        parts = []
        best_name = self._select_best_model_name(trained_models)

        # Test set bootstrap results
        if self.bootstrap_results:
            test_part = self._build_test_set_sentence(trained_models, best_name)
            if test_part:
                parts.append(test_part)

        # AUC(t) and risk stratification — scope to best model when available
        disc = self._get_discrimination_quality_text(best_name)
        if disc:
            parts.append(disc)
        strat = self._get_risk_stratification_quality_text(best_name)
        if strat:
            parts.append(strat)

        # Feature importance — scope to best model when available
        feat = self._build_features_sentence(best_name)
        if feat:
            parts.append(feat)

        if parts:
            md_lines.append("")
            md_lines.append('. '.join(parts) + '.')

    def _select_best_model_name(self, trained_models: Dict) -> Optional[str]:
        """Return the CV-best model name, or None when not resolvable."""
        if not trained_models:
            return None
        from mlsurv._shared import cv_sorted_model_keys
        try:
            ordered = cv_sorted_model_keys(trained_models)
        except ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "summary: could not resolve best model"
            )
            return None
        return ordered[0] if ordered else None

    def _build_test_set_sentence(
        self,
        trained_models: Dict,
        best_name: Optional[str] = None,
    ) -> Optional[str]:
        """One-sentence test set result for the quality paragraph."""
        if not self.bootstrap_results:
            return None

        first_result = next(iter(self.bootstrap_results.values()))
        n_bootstrap = len(first_result.iterations['iteration'].unique())
        first_row = first_result.summary.iloc[0]
        metric_name = get_metric_display_name(first_row['metric'])
        metric_key = first_row['metric']

        # Prefer the full-test-set point estimate (matches the Model Evaluation
        # table); fall back to bootstrap mean if 'point' is missing or all-NaN.
        # The label changes alongside the value so the prose never claims a
        # bootstrap mean is a point estimate, and we also raise a warning so
        # programmatic consumers see the degraded output (project rule:
        # No Silent Data Loss).
        use_point = _has_valid_point_column(first_result.summary)
        point_key = 'point' if use_point else 'mean'
        value_label = metric_name if use_point else f"{metric_name} (bootstrap mean)"
        if not use_point:
            warnings.warn(
                "Bootstrap summary lacks a valid 'point' column; "
                "Executive Summary falls back to the bootstrap mean for the "
                "test-set sentence.",
                InternalWarning,
                stacklevel=2,
            )

        n_models = len(self.bootstrap_results)
        if n_models == 1:
            model_name = next(iter(self.bootstrap_results.keys()))
            point_val = first_row[point_key]
            ci_part = ""
            if 'ci_lower' in first_result.summary.columns:
                ci_part = (
                    f" [{first_result.ci_level * 100:g}% CI: "
                    f"{first_row['ci_lower']:.3f}-{first_row['ci_upper']:.3f}]"
                )
            return (
                f"On the hold-out test set (*{n_bootstrap}* bootstrap iterations), "
                f"*{get_model_display_name(model_name)}* achieved "
                f"*{value_label}* = *{point_val:.3f}*{ci_part}"
            )

        # Multi-model: report CV-selected best model's test-set performance
        if best_name is None:
            best_name = self._select_best_model_name(trained_models)
        if best_name is None or best_name not in self.bootstrap_results:
            return None

        boot_result = self.bootstrap_results[best_name]
        metric_row = boot_result.summary[boot_result.summary['metric'] == metric_key]
        if len(metric_row) == 0:
            return None

        best_row = metric_row.iloc[0]
        ci_part = ""
        if 'ci_lower' in best_row.index:
            ci_part = (
                f" [{boot_result.ci_level * 100:g}% CI: "
                f"{best_row['ci_lower']:.3f}-{best_row['ci_upper']:.3f}]"
            )
        return (
            f"On the hold-out test set (*{n_bootstrap}* bootstrap iterations), "
            f"*{get_model_display_name(best_name)}* achieved "
            f"*{value_label}* = *{best_row[point_key]:.3f}*{ci_part}"
        )

    def _build_features_sentence(self, best_name: Optional[str] = None) -> Optional[str]:
        """One-sentence feature importance for the quality paragraph."""
        top_features = self._get_top_features_for_box(model_name=best_name)
        if not top_features:
            return None
        features_str = ', '.join(f"*{f}*" for f in top_features)
        return f"The top predictive features were {features_str}"

    def _add_key_findings_box(self, md_lines: list, trained_models: Dict, n_total: int) -> None:
        """Add a key findings highlight box using blockquote markdown."""
        scoring_metric = self.learner.scoring
        cv_metric_display = get_metric_display_name(scoring_metric)

        # Find best model
        from mlsurv._shared import cv_sorted_model_keys
        best_name = cv_sorted_model_keys(trained_models)[0]
        best_result = trained_models[best_name]
        best_display = get_model_display_name(best_name)
        best_score = _display_cv_score(best_result.cv_metric_mean, scoring_metric)

        # Sample size and event rate
        n_events = int(self.learner.y_train['event'].sum() + self.learner.y_test['event'].sum())
        event_rate = n_events / n_total * 100
        censoring_rate = 100 - event_rate

        # Build items
        cv_desc = self._build_cv_description()
        best_line = (
            f"**Best model:** *{best_display}* "
            f"({cv_desc} mean {cv_metric_display} = *{best_score:.3f}*"
        )

        # Append ± std from cross-validation folds
        cv_std = best_result.cv_metric_std
        if cv_std is not None and np.isfinite(cv_std):
            best_line += f" \u00b1 {cv_std:.3f}"

        best_line += ")"

        items = [best_line]
        outcome = self.outcome_name
        items.append(f"**Sample size:** {n_total:,} observations ({n_events:,} {pluralize(outcome)}, {censoring_rate:.0f}% censored)")

        # Top features — best-model-only, to match the body sentence
        top_features = self._get_top_features_for_box(model_name=best_name)
        if top_features:
            features_str = ', '.join(f"*{f}*" for f in top_features)
            items.append(
                f"**Top features ({best_display}):** {features_str}"
            )

        # Extended analyses (conditional one-liners)
        items.extend(self._build_extended_analysis_lines())

        md_lines.append("> **Key Findings**")
        md_lines.append(">")
        for item in items:
            md_lines.append(f"> - {item}")
        md_lines.append("")

    def _build_extended_analysis_lines(self) -> list[str]:
        """Build conditional one-liners for extended analyses in key findings box.

        Returns up to four lines describing: subpopulation analysis, external
        validation, benchmarking, and feature interactions.  Each line
        appears only when the corresponding analysis was performed.
        """
        lines: list[str] = []
        for builder in (
            self._build_subpopulation_summary_line,
            self._build_validation_summary_line,
            self._build_benchmark_summary_line,
            self._build_interactions_summary_line,
        ):
            try:
                line = builder()
                if line:
                    lines.append(line)
            except ALL_EXCEPTIONS as exc:
                self._log_recoverable_exception(
                    exc, f"summary: {builder.__name__} failed",
                )
        return lines

    def _build_subpopulation_summary_line(self) -> Optional[str]:
        """Build a concise subpopulation analysis one-liner for the key findings box."""
        sg_results = self.learner.subpopulation_results
        if not sg_results:
            return None
        try:
            parts = []
            for group_var, sg_result in sg_results.items():
                n_groups = len(sg_result.summary)
                display_var = title_if_lower(group_var.replace('_', ' '))
                parts.append(f"{n_groups} {display_var} groups")
            return f"**Subpopulation analysis:** {'; '.join(parts)}"
        except (AttributeError, KeyError, TypeError) as exc:
            self._log_recoverable_exception(exc, "summary: subpopulation summary line")
            return None

    def _build_validation_summary_line(self) -> Optional[str]:
        """Build a concise external validation one-liner for the key findings box."""
        validation_results = self.learner.validation_results
        if not validation_results:
            return None
        try:
            n = len(validation_results)
            cohort_word = "cohort" if n == 1 else "cohorts"
            return f"**External validation:** {n} {cohort_word}"
        except (AttributeError, TypeError) as exc:
            self._log_recoverable_exception(exc, "summary: validation summary line")
            return None

    def _build_benchmark_summary_line(self) -> Optional[str]:
        """Build a concise benchmarking one-liner for the key findings box."""
        bench = self.learner.benchmark_result
        if bench is None:
            return None
        try:
            ref_names = bench.reference_names
            if not ref_names:
                return None
            n = len(ref_names)
            names_str = ', '.join(ref_names)
            score_word = "score" if n == 1 else "scores"
            return f"**Benchmarking:** compared against {n} reference {score_word} ({names_str})"
        except (AttributeError, KeyError, TypeError) as exc:
            self._log_recoverable_exception(exc, "summary: benchmark summary line")
            return None

    def _build_interactions_summary_line(self) -> Optional[str]:
        """Build a concise feature interactions one-liner for the key findings box."""
        if self.eval_result is None or self.eval_result.interactions is None:
            return None
        try:
            interactions = self.eval_result.interactions
            n_pairs = len(interactions)
            if n_pairs == 0:
                return None
            method_display = self._resolve_interaction_method(interactions)
            pair_word = "pair" if n_pairs == 1 else "pairs"
            return f"**Feature interactions:** {n_pairs} {pair_word} analyzed via {method_display}"
        except (AttributeError, KeyError, TypeError) as exc:
            self._log_recoverable_exception(exc, "summary: interactions summary line")
            return None

    @staticmethod
    def _resolve_interaction_method(interactions) -> str:
        """Determine the display name for the interaction method(s) used."""
        if 'method' not in interactions.columns:
            return 'H-statistic'
        methods = set(interactions['method'].dropna().unique())
        if methods == {'hstat', 'shap'}:
            return 'H-statistic & SHAP'
        if 'shap' in methods:
            return 'SHAP interaction'
        return 'H-statistic'

    def _get_top_features_for_box(
        self,
        n_features: int = 5,
        model_name: Optional[str] = None,
    ) -> list[str]:
        """Get top unique feature names for the key findings box."""
        if self.eval_result is None or self.eval_result.importance is None:
            return []
        importance_df = self.eval_result.importance
        if len(importance_df) == 0:
            return []
        if model_name is not None and 'model_type' in importance_df.columns:
            filtered = importance_df[
                (importance_df['model_type'] == model_name)
                & importance_df['perm_mean'].notna()
            ]
            if len(filtered) > 0:
                importance_df = filtered
        # Average across models to get unique features ranked by overall importance
        avg_imp = importance_df.groupby('feature')['perm_mean'].mean()
        return avg_imp.nlargest(n_features).index.tolist()

    def _build_cv_description(self) -> str:
        """Build a human-readable cross-validation description string."""
        cv_strategy = self.learner.cv_strategy
        cv_folds = self.learner.cv_folds
        cv_repeats = getattr(self.learner, 'cv_repeats', 1)

        cv_strategy_names = {
            'stratifiedkfold': 'stratified',
            'stratified': 'stratified',
            'kfold': '',
            'standard': '',
            'groupkfold': 'group',
        }
        strategy_adj = cv_strategy_names.get(cv_strategy, '')
        repeat_prefix = f"{cv_repeats}\u00d7 " if isinstance(cv_repeats, int) and cv_repeats > 1 else ""
        if strategy_adj:
            return f"{repeat_prefix}{cv_folds}-fold {strategy_adj} cross-validation"
        return f"{repeat_prefix}{cv_folds}-fold cross-validation"

    def _generate_tuning_narrative(self, trained_models, tuned_models, cv_desc):
        """Build narrative sentence about hyperparameter tuning.

        Returns
        -------
        str or None
            Tuning narrative sentence, or None if no tuning was performed.
        """
        models_with_studies = self._collect_models_with_studies(tuned_models, trained_models)
        if not models_with_studies:
            return None

        tuned_names = [f"*{get_model_display_name(name)}*" for name in models_with_studies.keys()]

        # Get n_trials and metric from the first study
        first_result = next(iter(models_with_studies.values()))
        n_trials = len(first_result.optuna_study.trials)
        tuning_metric = first_result.scoring_metric or DEFAULT_SCORING_METRIC
        metric_display = get_metric_display_name(tuning_metric)

        sentence = (
            f"Hyperparameter tuning via [Optuna](https://optuna.org) "
            f"(*{n_trials}* trials, *{cv_desc}*, optimizing *{metric_display}*) "
            f"was applied to {', '.join(tuned_names)}"
        )

        tuning_details = self._get_tuning_improvement_details(models_with_studies, tuning_metric)
        if tuning_details:
            sentence += f". {tuning_details}"

        best_name = self._select_best_model_name(trained_models)
        sentence += self._format_influential_params(
            models_with_studies, sentence, best_name=best_name
        )
        return sentence

    @staticmethod
    def _has_optuna_study(result) -> bool:
        """Check whether a model result has an attached Optuna study."""
        return hasattr(result, 'optuna_study') and result.optuna_study is not None

    def _collect_models_with_studies(self, tuned_models, trained_models) -> Dict:
        """Collect all models that have an Optuna study, preferring tuned_models."""
        models_with_studies = {}
        for name, result in tuned_models.items():
            if self._has_optuna_study(result):
                models_with_studies[name] = result
        for name, result in trained_models.items():
            if name not in models_with_studies and self._has_optuna_study(result):
                models_with_studies[name] = result
        return models_with_studies

    def _format_influential_params(
        self,
        models_with_studies: Dict,
        sentence: str,
        best_name: Optional[str] = None,
    ) -> str:
        """Format the most influential parameters suffix for the tuning narrative.

        Restricts output to the CV-best tuned model when identifiable, so the
        executive summary highlights one representative model instead of
        dumping top parameters for every tuned model.
        """
        focus_models = models_with_studies
        if best_name is not None and best_name in models_with_studies:
            focus_models = {best_name: models_with_studies[best_name]}
        n_params = 3 if len(focus_models) == 1 else 2
        important_params = self._get_top_params_per_model(focus_models, n_params=n_params)
        if important_params:
            if len(important_params) == 1:
                name, params_list = next(iter(important_params.items()))
                params_formatted = ', '.join([f"*{p}*" for p in params_list])
                return (
                    f" Most influential parameters for "
                    f"*{get_model_display_name(name)}*: {params_formatted}."
                )
            param_strs = []
            for name, params_list in important_params.items():
                params_formatted = ', '.join([f"*{p}*" for p in params_list])
                param_strs.append(f"{params_formatted} ({get_model_display_name(name)})")
            return f" Most influential parameters: {', '.join(param_strs)}."
        if not sentence.endswith('.'):
            return "."
        return ""

    def _get_top_params_per_model(
        self,
        models_with_studies: Dict,
        n_params: int = 2
    ) -> Dict[str, list]:
        """
        Get the most important parameters for each tuned model.

        Parameters
        ----------
        models_with_studies : dict
            Dictionary mapping model names to results with optuna_study.
        n_params : int
            Number of top parameters to return per model. Default is 2.

        Returns
        -------
        dict
            Dictionary mapping model names to a list of their most important
            parameter names (up to n_params).
        """
        try:
            from optuna.importance import get_param_importances
        except ImportError as e:
            self._log_recoverable_exception(
                e,
                "summary: optuna not available — skipping parameter importance",
                warn_user=True,
                user_message="optuna not available — skipping parameter importance",
            )
            return {}

        top_params = {}
        for name, result in models_with_studies.items():
            try:
                study = result.optuna_study
                importances = get_param_importances(study)
                if importances:
                    sorted_params = sorted(
                        importances.keys(),
                        key=lambda p: importances[p],
                        reverse=True
                    )
                    top_params[name] = sorted_params[:n_params]
            except ALL_EXCEPTIONS as e:
                self._log_recoverable_exception(
                    e,
                    f"summary: parameter importance extraction for {name}",
                )
                continue

        return top_params

    def _get_tuning_improvement_details(
        self,
        models_with_studies: Dict,
        tuning_metric: str
    ) -> Optional[str]:
        """Build narrative about best trial values for tuned models.

        Tuning overwrites the untuned model in place, so no untuned sibling is
        retained to compare against; the narrative reports the best trial value
        without an improvement delta.
        """
        metric_display = get_metric_display_name(tuning_metric)
        single_model = len(models_with_studies) == 1

        details_parts = []
        for name, result in models_with_studies.items():
            best_value = abs(result.optuna_study.best_value)
            if single_model:
                detail = f"Best trial achieved {metric_display} = *{best_value:.3f}*"
            else:
                detail = f"{get_model_display_name(name)}: *{best_value:.3f}*"
            details_parts.append(detail)

        if not details_parts:
            return None
        if single_model:
            return details_parts[0]
        return f"Best trial {metric_display}: {'; '.join(details_parts)}"

    def _get_discrimination_quality_text(
        self, best_name: Optional[str] = None
    ) -> Optional[str]:
        """Get discrimination quality narrative from mean AUC(t)."""
        if self.eval_result is None or self.eval_result.auc_t is None:
            return None
        try:
            auc_df = self.eval_result.auc_t
            if len(auc_df) == 0:
                return None
            if best_name is not None and 'model_type' in auc_df.columns:
                filtered = auc_df[
                    (auc_df['model_type'] == best_name) & auc_df['auc'].notna()
                ]
                if len(filtered) > 0:
                    auc_df = filtered
            mean_auc = auc_df['auc'].mean()
            if np.isnan(mean_auc):
                return None
            quality = self._interpret_quality(mean_auc)
            return (
                f"Discrimination was {quality} across all timepoints "
                f"(mean AUC(t) = *{mean_auc:.2f}*)"
            )
        except ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "summary: discrimination quality text unavailable",
            )
            return None

    def _get_risk_stratification_quality_text(
        self, best_name: Optional[str] = None
    ) -> Optional[str]:
        """Get risk stratification quality narrative from log-rank p-value."""
        if self.eval_result is None or self.eval_result.test_set is None:
            return None
        try:
            test_set = self.eval_result.test_set
            required_cols = {'risk_score', 'time', 'event'}
            if not required_cols.issubset(test_set.columns):
                return None

            from lifelines.statistics import multivariate_logrank_test

            if 'model_type' in test_set.columns:
                models = test_set['model_type'].unique()
                chosen = best_name if (best_name is not None and best_name in models) else models[0]
                model_data = test_set[test_set['model_type'] == chosen].copy()
            else:
                model_data = test_set.copy()

            model_data, _ = self._create_risk_groups(model_data, DEFAULT_RISK_GROUPS)
            if model_data is None:
                return None

            result = multivariate_logrank_test(
                model_data['time'], model_data['risk_group'], model_data['event']
            )
            p_value = result.p_value

            quality, p_str = _classify_p_value(p_value)

            return (
                f"Risk stratification showed {quality} separation between "
                f"risk groups (log-rank {p_str})"
            )
        except ImportError as e:
            self._log_recoverable_exception(
                e,
                "summary: lifelines not available — skipping log-rank test",
                warn_user=True,
                user_message="lifelines not available — skipping log-rank test",
            )
            return None
        except ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "summary: risk stratification quality text unavailable",
            )
            return None

    def _get_validation_summary(self) -> Optional[str]:
        """
        Get a summary of external validation results.

        Returns
        -------
        str or None
            Narrative about external validation cohort(s) and performance, or None if not available.
        """
        validation_results = self.learner.validation_results
        if not validation_results:
            return None

        try:
            if len(validation_results) == 1:
                return self._get_single_cohort_validation_text(validation_results)
            return self._get_multi_cohort_validation_text(validation_results)
        except ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "summary: validation summary unavailable",
            )
            return None

    @staticmethod
    def _extract_c_index(val_result) -> Optional[float]:
        """Extract C-index from a validation result, or None if unavailable."""
        if val_result.concordance is None:
            return None
        if 'c_index_test' not in val_result.concordance.columns:
            return None
        c_index = val_result.concordance['c_index_test'].iloc[0]
        if np.isnan(c_index):
            return None
        return c_index

    def _get_single_cohort_validation_text(self, validation_results: Dict) -> str:
        """Build validation narrative for a single external cohort."""
        name, val_result = next(iter(validation_results.items()))
        parts = []
        outcome = self.outcome_name

        # External cohort size and event rate
        if hasattr(val_result, 'n_samples'):
            parts.append(
                f"external cohort '{name}' (*{val_result.n_samples}* observations, "
                f"*{val_result.n_events}* {pluralize(outcome)}, *{val_result.event_rate:.1%}* {outcome} rate)"
            )

        # Performance metrics
        c_index = self._extract_c_index(val_result)
        if c_index is not None:
            parts.append(f"{get_metric_display_name('harrell')} = *{c_index:.3f}*")

        if not parts:
            return f"The model was externally validated on cohort '{name}'."
        if len(parts) == 1:
            return f"The model was externally validated on {parts[0]}."
        return f"The model was externally validated on {parts[0]} ({parts[1]})."

    def _add_limitations_line(self, md_lines: list) -> None:
        """Add a one-line cross-reference to triggered limitation flags."""
        from mlsurv.reports.sections.limitations import LimitationsSection

        lim = LimitationsSection(self.learner)
        triggered = lim.get_triggered_flags()
        if not triggered:
            return
        names = ", ".join(name for _, name in triggered)
        n = len(triggered)
        label = "limitation" if n == 1 else "limitations"
        md_lines.append("")
        md_lines.append(
            f"**{n} {label} flagged:** {names}. "
            "See *Limitations & Caveats* for details."
        )

    def _get_multi_cohort_validation_text(self, validation_results: Dict) -> str:
        """Build validation narrative for multiple external cohorts."""
        n_cohorts = len(validation_results)
        cohort_names = list(validation_results.keys())
        total_samples = sum(getattr(v, 'n_samples', 0) for v in validation_results.values())

        # Get C-index range across cohorts
        c_indices = [
            c for val_result in validation_results.values()
            if (c := self._extract_c_index(val_result)) is not None
        ]

        detail = (
            f"*{n_cohorts}* external cohorts "
            f"({', '.join(cohort_names)}; *{total_samples:,}* total observations)"
        )

        if c_indices and len(c_indices) == n_cohorts:
            c_range = f"C-index range *{min(c_indices):.3f}*–*{max(c_indices):.3f}*"
            return f"The model was externally validated on {detail} ({c_range})."

        return f"The model was externally validated on {detail}."
