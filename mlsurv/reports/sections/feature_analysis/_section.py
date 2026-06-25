"""Feature analysis section orchestrator.

Delegates to focused helper modules for each subsection while keeping
state management and the ``generate()`` pipeline in one place.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mlsurv.constants import (
    DEFAULT_SCORING_METRIC, LOWER_IS_BETTER, get_metric_display_name,
)
from mlsurv.reports._constants import CORRELATION_THRESHOLD
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, BaseSection, FigureInfo
from mlsurv.results.containers import EvaluationResult
from mlsurv.viz._palettes import get_model_display_name

from ._audit import generate_feature_selection_audit
from ._coefficient_analysis import generate_coefficient_analysis
from ._coefficients import (
    get_baseline_c_index,
    get_univariate_cox,
)
from ._correlation import (
    generate_correlation_analysis,
    generate_correlation_warning_callout,
)
from ._hazard_ratios import generate_hazard_ratio_analysis
from ._interactions import generate_interaction_analysis, generate_interaction_pdp_analysis
from mlsurv.reports._constants import format_pvalue as _format_pvalue

from ._helpers import (
    CHART_EXCEPTIONS,
    _derive_effect_from_hr,
    _format_adjusted_pvalue,
    _format_importance_value,
    _format_significance,
    _lookup_adjusted_pvalue,
    _missing_data_note,
    _short_column_name,
)
from ._importance import (
    _compute_n_obs,
    _format_univar_fields,
    _lookup_univar_cox,
    append_coef_columns,
    build_feature_row,
    build_multi_model_feature_row,
    generate_importance_significance_narrative,
    _section_bootstrap_result,
    get_importance_pvalues,
    get_sorted_top_features,
    per_model_coef_blocks,
    sig_column_intro,
    sig_footnote,
    split_summary_blocks,
)
from ._interactive_table import (
    _classify_column_type,
    build_multi_model_column_groups,
    build_single_model_column_groups,
    format_interactive_table,
)
from ._shap import generate_shap_analysis


class FeatureAnalysisSection(BaseSection):
    """
    Feature analysis section.

    Shows:
    - Feature importance bar chart (top n_features, colored by direction)
    - Interactive importance table with all features (expandable beyond n_features)
    - Per-model coefficients and hazard ratios (when Cox/linear models available)
    - Multi-model comparison: How feature importance varies across models
    - Feature correlation analysis

    Maps to API method: evaluate() importance output
    """

    name = "feature_analysis"
    title = "Feature Analysis"

    def __init__(
        self,
        learner,
        eval_result: Optional[EvaluationResult] = None,
        n_features: int = 10,
    ):
        """
        Initialize feature analysis section.

        Parameters
        ----------
        learner : SurvivalLearner
            The learner instance.
        eval_result : EvaluationResult, optional
            Evaluation result with importance data.
        n_features : int
            Number of features visible by default in interactive tables.
            All features are included; rows beyond this are collapsed
            behind an expand button.
        """
        super().__init__(learner)
        self.eval_result = eval_result
        self.n_features = n_features
        self._pvalue_source: Optional[str] = None  # "pimp" or "bootstrap"
        self._pvalue_method: str = "bh"  # p_adjust method actually used
        self._feature_summary_error: Optional[str] = None

    def is_available(self) -> bool:
        """Check if evaluation data is available for feature analysis."""
        return self.eval_result is not None

    # ── Static method delegates (tests access via FeatureAnalysisSection._method) ──
    _format_significance = staticmethod(_format_significance)
    _format_adjusted_pvalue = staticmethod(_format_adjusted_pvalue)
    _short_column_name = staticmethod(_short_column_name)
    _format_importance_value = staticmethod(_format_importance_value)
    _format_pvalue = staticmethod(_format_pvalue)
    _derive_effect_from_hr = staticmethod(_derive_effect_from_hr)
    _lookup_adjusted_pvalue = staticmethod(_lookup_adjusted_pvalue)
    _classify_column_type = staticmethod(_classify_column_type)
    _append_coef_columns = staticmethod(append_coef_columns)

    def _get_feature_summary(self, correlated_features) -> Optional[pd.DataFrame]:
        """Assemble the feature summary from the section's own frames.

        Calls the same ``compute_feature_summary`` assembly that backs
        ``learner.feature_summary()``, but feeds it the section's
        ``eval_result`` importance/SHAP/bootstrap frames — so when a
        caller supplies an explicit eval result to ``generate_report``,
        the combined table shows the same evaluation as the charts and
        exports instead of live learner state. The univariate and
        coefficient pieces always come from the live learner (models
        only exist there). *correlated_features* is the set the
        correlation subsection already computed; reusing it avoids a
        second O(p^2) scan.

        Returns None when assembly fails. A failure must not take down
        the whole section — the charts, correlation analysis, and
        forest plots do not depend on the summary and still render;
        only the combined table is replaced by an unavailable note.
        """
        from mlsurv.evaluation._feature_summary import compute_feature_summary

        try:
            return compute_feature_summary(
                list(self.learner.models),
                coefficients=self.learner.coefficients,
                univariate_cox=self.learner.univariate_cox,
                importance=self.eval_result.importance,
                shap_values=self.eval_result.shap_values,
                pimp_results=self.learner.pimp_importance_test_results,
                bootstrap_result=_section_bootstrap_result(self),
                correlated_features=correlated_features,
                correlation_threshold=CORRELATION_THRESHOLD,
            )
        except ALL_EXCEPTIONS as e:
            self._feature_summary_error = str(e)
            self._log_recoverable_exception(
                e,
                "feature_analysis: feature summary assembly failed",
                warn_user=True,
                user_message=f"Could not assemble the combined feature table: {e}",
            )
            return None

    def _feature_summary_unavailable_note(self, missing_model=None) -> str:
        """Markdown note shown in place of the combined feature table.

        *missing_model* covers the non-exception degrade: the summary
        was assembled but holds no block for that model (the learner's
        trained models diverged from this evaluation result).
        """
        if self._feature_summary_error:
            return f"*[Combined feature table unavailable: {self._feature_summary_error}]*"
        if missing_model is not None:
            self._logger.warning(
                "feature_analysis: model %r is missing from the feature "
                "summary; the learner's trained models may have changed "
                "since this evaluation result was made.",
                missing_model,
            )
            return (
                f"*[Combined feature table unavailable: model "
                f"'{missing_model}' is not among the learner's trained "
                f"models — they may have changed since this "
                f"evaluation.]*"
            )
        return "*[Combined feature table unavailable.]*"

    def _categorical_groups_note(self) -> List[str]:
        """Return markdown lines describing categorical group aggregation, if any."""
        cat_groups = getattr(self.learner._state, 'categorical_groups', None)
        if not cat_groups:
            return []
        lines = []
        group_descriptions = []
        for name, cols in cat_groups.items():
            group_descriptions.append(
                f"**{name}** ({', '.join(cols)})"
            )
        lines.append(
            "**Grouped features**: The following one-hot encoded features were "
            "permuted jointly (group permutation importance) and reported as a "
            "single feature: "
            + "; ".join(group_descriptions)
            + ". Individual dummy columns (↳) are shown below each group with "
            "their univariate Cox associations and per-column PFI."
        )
        lines.append("")
        lines.append(
            "*Caution*: Per-column PFI for one-hot encoded dummies is "
            "upward-biased because shuffling one column independently "
            "breaks the mutual exclusivity of the encoding, creating "
            "impossible feature combinations. Use the **group row** "
            "as the correct measure of overall feature importance; "
            "per-column values (↳) are useful only for identifying "
            "which category level drives the effect."
        )
        lines.append("")
        return lines

    def _pfi_description(self) -> str:
        """Build PFI description with the actual scoring metric and sign convention."""
        scoring = self.learner.scoring or DEFAULT_SCORING_METRIC
        metric_label = get_metric_display_name(scoring)
        direction = "increase" if scoring in LOWER_IS_BETTER else "drop"
        return (
            f"**PFI**: {metric_label} {direction} when feature is shuffled; "
            f"percentage shows relative impact."
        )

    def _combined_table_intro(self, heading: str, univar_cox) -> List[str]:
        """Intro lines shared by the single- and multi-model feature tables."""
        cox_desc = (
            "**Univariate Cox**: Each feature is tested independently in a univariate Cox "
            "regression model. The p-value tests if HR differs from 1.0; HR > 1 indicates "
            "increased hazard (risk factor), HR < 1 indicates decreased hazard (protective)."
        )
        cox_desc += _missing_data_note(univar_cox)
        lines = [heading, "", cox_desc, "", self._pfi_description(), ""]
        lines.extend(self._categorical_groups_note())
        return lines

    def _build_categorical_child_rows(
        self, feat, models, univar_cox, baseline_c_index,
        per_model_coefs, total_samples,
    ) -> List[dict]:
        """Build indented child rows for individual dummy columns of a grouped feature."""
        cat_groups = getattr(self.learner._state, 'categorical_groups', None)
        if not cat_groups or feat not in cat_groups:
            return []

        per_col = getattr(self.eval_result, 'importance_per_column', None)
        cols = cat_groups[feat]
        rows = []
        for col in cols:
            row = {'Feature': f"  \u21b3 {col}"}

            # Per-feature sample size
            n_obs_str, show_n = _compute_n_obs(col, univar_cox, total_samples)
            if show_n:
                row['n'] = n_obs_str

            # Univariate Cox (available per-column)
            pval, hr_val = _lookup_univar_cox(col, univar_cox)
            pval_str, hr_str, effect = _format_univar_fields(pval, hr_val)
            row['p-value'] = pval_str
            row['HR'] = hr_str
            row['Effect'] = effect

            # Per-column PFI from each model
            for model_name in models:
                display = get_model_display_name(model_name)
                col_name = f"PFI ({display})"
                if per_col is not None:
                    model_pc = per_col[per_col['model_type'] == model_name]
                    feat_data = model_pc[model_pc['feature'] == col]
                    if len(feat_data) > 0:
                        imp_val = feat_data['perm_mean'].values[0]
                        model_c_index = baseline_c_index.get(model_name)
                        row[col_name] = _format_importance_value(
                            imp_val, model_c_index,
                        )
                    else:
                        row[col_name] = "-"
                else:
                    row[col_name] = "-"

                # Per-column coefficients
                if per_model_coefs and model_name in per_model_coefs:
                    append_coef_columns(
                        row, col, display, per_model_coefs[model_name],
                    )

            rows.append(row)
        return rows

    def generate(self) -> Dict[str, Any]:
        """Generate feature analysis section content."""
        md_lines: List[str] = []
        figures: List[Tuple] = []
        figure_info: List[FigureInfo] = []

        if not self.is_available():
            return {'markdown': '', 'figures': [], 'figure_info': []}

        # --- 0. Preamble placeholder (summary + TOC inserted after generation) ---
        preamble_idx = len(md_lines)

        importance_df = self.eval_result.importance
        has_importance = importance_df is not None and len(importance_df) > 0
        if has_importance:
            importance_df = importance_df.copy()  # avoid mutating shared state

        # Derive model list: importance > model_order > concordance
        if has_importance:
            models = importance_df['model_type'].unique()
        elif self.eval_result.model_order is not None:
            models = np.array(self.eval_result.model_order)
        else:
            models = self.eval_result.concordance['model_type'].unique()

        # Get baseline C-index for percentage calculation
        baseline_c_index = get_baseline_c_index(self)

        # Get univariate Cox statistics from learner
        univar_cox = get_univariate_cox(self)

        if has_importance:
            # Merge log(HR) onto importance data for direction-colored charts.
            # _assign_coef_sign checks sign: log(HR) > 0 = Risk, < 0 = Protective.
            if univar_cox is not None and '_log_hr' not in importance_df.columns:
                hr_map = univar_cox.set_index('feature')['hazard_ratio']
                importance_df['_log_hr'] = importance_df['feature'].map(hr_map).apply(
                    lambda hr: np.log(hr) if pd.notna(hr) and np.isfinite(hr) and hr > 0 else np.nan
                )

        # Compute bootstrap importance p-values once for all models
        if has_importance:
            imp_pvals_by_model = self._compute_importance_pvalues(models)
        else:
            imp_pvals_by_model = None

        # --- 2. Feature selection audit (conditional) ---
        audit_md = generate_feature_selection_audit(self)
        if audit_md:
            md_lines.append(audit_md)
            md_lines.append("")

        # --- 3. Feature correlation (moved up before importance) ---
        corr_result = generate_correlation_analysis(self)
        correlated_features = corr_result.get('_correlated_features', set())
        severity = corr_result.get('_severity', 'none')
        recommendation = corr_result.get('_recommendation', '')
        if corr_result['markdown']:
            md_lines.append(corr_result['markdown'])
            figures.extend(corr_result['figures'])
            figure_info.extend(corr_result['figure_info'])

        # --- 4. Correlation warning callout (conditional) ---
        callout_md = generate_correlation_warning_callout(severity, recommendation)
        if callout_md:
            md_lines.append(callout_md)
            md_lines.append("")

        # --- 5. Importance tables + charts (with correlated-feature markers) ---
        if has_importance:
            importance_tidy = self._build_importance_tidy(
                importance_df, univar_cox,
            )
            self.data['importance'] = importance_tidy

            # The combined table renders the same compute_feature_summary
            # assembly that backs learner.feature_summary(), fed
            # the section's own eval_result frames and the correlated set
            # computed above (no second O(p^2) scan).
            summary = self._get_feature_summary(correlated_features)

            if len(models) == 1:
                sub_lines, sub_figs, sub_info = self._generate_single_model_content(
                    models[0], importance_df, univar_cox, baseline_c_index,
                    summary,
                    imp_pvals_by_model=imp_pvals_by_model,
                )
            else:
                sub_lines, sub_figs, sub_info = self._generate_multi_model_content(
                    models, importance_df, univar_cox, baseline_c_index,
                    summary,
                    imp_pvals_by_model=imp_pvals_by_model,
                )
            md_lines.extend(sub_lines)

            scoring_metric = self.learner.scoring
            metric_label = get_metric_display_name(scoring_metric)
            md_lines.append(
                f"**Interpretation:** Permutation feature importance measures how much "
                f"the model's predictive accuracy ({metric_label}) drops when a "
                f"feature's values are randomly shuffled. A large drop means the model "
                f"relies heavily on that feature; a near-zero drop means the feature "
                f"contributes little to predictions. Features with high importance are "
                f"the strongest drivers of the model's risk predictions, though "
                f"importance does not imply causation."
            )
            md_lines.append("")

            # --- 6. PFI interpretation + bootstrap significance narratives ---
            if imp_pvals_by_model is not None:
                sig_narrative = generate_importance_significance_narrative(
                    self, models, imp_pvals_by_model,
                )
                if sig_narrative:
                    md_lines.append(sig_narrative)
                    md_lines.append("")

            figures.extend(sub_figs)
            figure_info.extend(sub_info)

            # Add correlated-feature footnote after importance tables if any markers used
            if correlated_features:
                md_lines.append(
                    f"*† Feature involved in a high-correlation pair (|R| > {CORRELATION_THRESHOLD}); "
                    "importance may be shared with correlated partner(s).*"
                )
                md_lines.append("")
        else:
            # Importance not computed — add placeholder note
            self.data['importance'] = pd.DataFrame(
                columns=['model_type', 'feature', 'perm_mean', 'perm_std', 'direction', 'rank']
            )
            md_lines.append("### Permutation Feature Importance")
            md_lines.append("")
            md_lines.append(
                "*Permutation feature importance was not computed for this evaluation "
                "(n\\_importance\\_repeats=0). To include importance analysis, re-run "
                "`learner.evaluate()` with `n_importance_repeats > 0`.*"
            )
            md_lines.append("")

        # --- 7. Hazard Ratios — univariate + multivariate ---
        hr_result = generate_hazard_ratio_analysis(self, univar_cox)
        if hr_result['markdown']:
            md_lines.append(hr_result['markdown'])
            figures.extend(hr_result['figures'])
            figure_info.extend(hr_result['figure_info'])

        # --- 8. Coefficient analysis (barplot + scatter comparison) ---
        coef_result = generate_coefficient_analysis(self)
        if coef_result['markdown']:
            md_lines.append(coef_result['markdown'])
            figures.extend(coef_result['figures'])
            figure_info.extend(coef_result['figure_info'])

        # --- 9. SHAP analysis ---
        shap_result = generate_shap_analysis(self)
        if shap_result['markdown']:
            md_lines.append(shap_result['markdown'])
            figures.extend(shap_result['figures'])
            figure_info.extend(shap_result['figure_info'])

        # --- 10. Feature interactions ---
        interactions_result = generate_interaction_analysis(self)
        if interactions_result['markdown']:
            md_lines.append(interactions_result['markdown'])
            figures.extend(interactions_result['figures'])
            figure_info.extend(interactions_result['figure_info'])

        # --- 10b. Interaction PDP ---
        pdp_result = generate_interaction_pdp_analysis(self)
        if pdp_result['markdown']:
            md_lines.append(pdp_result['markdown'])
            figures.extend(pdp_result['figures'])
            figure_info.extend(pdp_result['figure_info'])

        # --- Build summary + TOC and insert at the top ---
        preamble = self._build_preamble(
            imp_pvals_by_model, models,
            has_importance=has_importance,
            audit_md=audit_md, corr_result=corr_result,
            hr_result=hr_result, coef_result=coef_result,
            shap_result=shap_result,
            interactions_result=interactions_result,
        )
        md_lines[preamble_idx:preamble_idx] = preamble

        # Clean up temporary column added for chart coloring
        if has_importance and '_log_hr' in importance_df.columns:
            importance_df = importance_df.drop(columns='_log_hr')

        return {
            'markdown': '\n'.join(md_lines),
            'figures': figures,
            'figure_info': figure_info,
        }

    # ── Extracted helpers for generate() ──────────────────────────────────────

    def _compute_importance_pvalues(self, models) -> Optional[Dict[str, pd.DataFrame]]:
        """Compute importance p-values for all models."""
        imp_pvals_by_model: Dict[str, pd.DataFrame] = {}
        imp_summary_frames: List[pd.DataFrame] = []
        for model_name in models:
            pvals = get_importance_pvalues(self, model_name)
            if pvals is not None:
                imp_pvals_by_model[model_name] = pvals
                pvals_copy = pvals.copy()
                pvals_copy['model_type'] = model_name
                imp_summary_frames.append(pvals_copy)
        if imp_summary_frames:
            self.data['importance_pvalues'] = pd.concat(imp_summary_frames, ignore_index=True)
        return imp_pvals_by_model or None

    def _build_importance_tidy(
        self, importance_df, univar_cox,
    ) -> pd.DataFrame:
        """Build the tidy importance DataFrame with merged univariate Cox data."""
        importance_rows = []
        for _, row in importance_df.iterrows():
            imp_row = {
                'model_type': row['model_type'],
                'feature': row['feature'],
                'perm_mean': row['perm_mean'],
                'perm_std': row.get('perm_std', None),
            }

            if univar_cox is not None:
                cox_match = univar_cox[univar_cox['feature'] == row['feature']]
                if len(cox_match) > 0:
                    imp_row['p_value'] = cox_match['p_value'].values[0]
                    imp_row['hazard_ratio'] = cox_match['hazard_ratio'].values[0]
                    imp_row['direction'] = _derive_effect_from_hr(
                        cox_match['hazard_ratio'].values[0]
                    )
                else:
                    imp_row['direction'] = 'Unknown'
            else:
                imp_row['direction'] = 'Unknown'

            importance_rows.append(imp_row)

        importance_tidy = pd.DataFrame(importance_rows)
        importance_tidy['rank'] = importance_tidy.groupby('model_type')['perm_mean'].rank(
            ascending=False, method='dense', na_option='bottom'
        ).fillna(0).astype(int)
        return importance_tidy

    def _build_preamble(
        self, imp_pvals_by_model, models, *,
        has_importance: bool,
        audit_md, corr_result, hr_result, coef_result, shap_result,
        interactions_result=None,
    ) -> List[str]:
        """Build the summary + TOC preamble lines."""
        preamble: List[str] = []
        if has_importance:
            preamble.append(
                "Examines which features drive model predictions through correlation context, "
                "univariate Cox associations, permutation importance, and (when available) "
                "SHAP values and hazard ratio forest plots."
            )
        else:
            preamble.append(
                "Examines which features drive model predictions through correlation context, "
                "univariate Cox associations, and (when available) SHAP values, hazard ratio "
                "forest plots, and model coefficients."
            )
        preamble.append("")

        if has_importance:
            has_pimp = bool(self.learner.pimp_importance_test_results)
            if imp_pvals_by_model is None:
                preamble.append(
                    "*To add statistical significance testing for feature importance, call "
                    "`learner.importance_test()` (recommended — uses the PIMP permutation "
                    "test with a proper null distribution) or "
                    "`learner.bootstrap(include_importance=True)` (faster but less rigorous — "
                    "tests whether bootstrap importance CIs exclude zero, which assesses "
                    "estimate stability rather than true feature significance) "
                    "before generating the report.*"
                )
                preamble.append("")
            elif not has_pimp:
                preamble.append(
                    "*The Sig. column uses bootstrap-derived p-values, which test whether "
                    "importance estimates are stable across resamples (CI excludes zero). "
                    "For a more rigorous test of true feature significance against a null "
                    "distribution, call `learner.importance_test()` (PIMP; Altmann et al. 2010).*"
                )
                preamble.append("")

        # Detect which subsections were generated
        toc_entries = []
        if audit_md:
            toc_entries.append(('Feature Selection Audit', 'feature-selection-audit'))
        if corr_result['markdown']:
            toc_entries.append(('Feature Correlation Analysis', 'feature-correlation-analysis'))
        if has_importance:
            if len(models) == 1:
                toc_entries.append(('Univariate Cox and Permutation Importance', 'univariate-cox-and-permutation-importance'))
            else:
                toc_entries.append(('Univariate Cox and Permutation Importance by Model', 'univariate-cox-and-permutation-importance-by-model'))
        else:
            toc_entries.append(('Permutation Feature Importance', 'permutation-feature-importance'))
        if hr_result['markdown']:
            toc_entries.append(('Hazard Ratios', 'hazard-ratios'))
        if coef_result['markdown']:
            toc_entries.append(('Feature Coefficients', 'feature-coefficients'))
        if shap_result['markdown']:
            toc_entries.append(('SHAP-Based Feature Analysis', 'shap-based-feature-analysis'))
        if interactions_result and interactions_result['markdown']:
            toc_entries.append(('Feature Interactions', 'feature-interactions'))

        preamble.extend(self._build_section_outline(toc_entries))

        return preamble

    # ── Single / multi model content ──────────────────────────────────────────

    @property
    def _group_code_hint(self) -> str:
        """Code-snippet suffix for feature_groups in FigureInfo examples."""
        if self.feature_groups is not None:
            return ", feature_groups=feature_groups"
        return ""

    def _generate_single_model_content(
        self, model_name, importance_df, univar_cox,
        baseline_c_index, summary, imp_pvals_by_model=None,
    ):
        """Generate feature analysis content for a single model.

        *summary* is the ``learner.feature_summary()`` DataFrame (or
        None when it could not be assembled); the combined table renders
        its rows.
        """
        md_lines: List[str] = []
        figures: List[Tuple] = []
        figure_info: List[FigureInfo] = []

        md_lines.extend(self._combined_table_intro(
            "### Univariate Cox and Permutation Importance", univar_cox,
        ))

        imp_pvals = imp_pvals_by_model.get(model_name) if imp_pvals_by_model else None
        has_sig = imp_pvals is not None
        if has_sig:
            md_lines.append(sig_column_intro(self._pvalue_source, self._pvalue_method))
            md_lines.append("")

        block = split_summary_blocks(summary).get(model_name)
        if block is None:
            md_lines.append(self._feature_summary_unavailable_note(
                missing_model=None if summary is None else model_name,
            ))
            md_lines.append("")
        else:
            model_c_index = baseline_c_index.get(model_name)
            per_model_coefs = per_model_coef_blocks(summary)
            show_shap = block['mean_abs_shap'].notna().any()
            total_samples = len(self.learner.y_train) if self.learner.y_train is not None else None

            # Display order: importance-backed rows, highest PFI first.
            # Derived from importance_df (not the block) because the
            # block is re-sorted by uni_p_value; this keeps the table's
            # row and tie order tied to the importance results.
            model_imp = importance_df[importance_df['model_type'] == model_name]
            order = model_imp.sort_values('perm_mean', ascending=False)['feature']
            # Tolerate divergence between the section's eval_result and
            # the learner state behind feature_summary() (e.g. a
            # caller-supplied eval result): skip features the block
            # lacks instead of crashing the whole section.
            order = order[order.isin(block['feature'])]
            display_rows = (
                block.set_index('feature').loc[order].reset_index()
            )

            table_data = []
            for _, summary_row in display_rows.iterrows():
                table_data.append(
                    build_feature_row(
                        summary_row, model_name, model_c_index,
                        has_sig=has_sig, per_model_coefs=per_model_coefs,
                        univar_cox=univar_cox, total_samples=total_samples,
                        show_shap=show_shap,
                    )
                )
                # Insert child rows for grouped categorical features
                table_data.extend(
                    self._build_categorical_child_rows(
                        summary_row['feature'], [model_name], univar_cox,
                        baseline_c_index, per_model_coefs, total_samples,
                    )
                )

            table_df = pd.DataFrame(table_data)
            table_df = self._drop_all_empty_columns(table_df)
            column_groups = build_single_model_column_groups(
                table_df, model_name, coef_models=list(per_model_coefs),
            )
            md_lines.append(format_interactive_table(
                table_df, column_groups, n_visible=self.n_features,
            ))
            md_lines.append("")

        if has_sig:
            md_lines.append(sig_footnote(self._pvalue_source, self._pvalue_method))
            md_lines.append("")

        # Generate importance visualization
        _hr_kwargs = {'coef_col': '_log_hr'} if '_log_hr' in importance_df.columns else {}
        try:
            fig = self.learner.plot_importance(n=self.n_features, **_hr_kwargs, **self._group_plot_kwargs())
            figures.append((f'{model_name}_importance.png', fig))
            md_lines.append(f"![Feature Importance](images/{model_name}_importance.png)")
            md_lines.append("")

            display_name = get_model_display_name(model_name)
            gh = self._group_code_hint
            figure_info.append(FigureInfo(
                figure_id=f'{model_name}_importance',
                title=f'{display_name} Feature Importance',
                data_description='Tidy DataFrame with feature importance (model, feature, perm_mean, perm_std, direction, p_value, hazard_ratio, rank)',
                data_key='importance',
                learner_code=f"fig = learner.plot_importance(n={self.n_features}{gh})",
            ))
        except CHART_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                f"feature_analysis: importance chart generation for {model_name}",
                warn_user=True,
                user_message=f"Could not generate importance chart: {e}",
            )
            md_lines.append(f"*[Figure unavailable: {e}]*")
            md_lines.append("")

        return md_lines, figures, figure_info

    def _generate_multi_model_content(
        self, models, importance_df, univar_cox, baseline_c_index,
        summary, imp_pvals_by_model=None,
    ):
        """Generate feature analysis content for multiple models.

        *summary* is the ``learner.feature_summary()`` DataFrame (or
        None when it could not be assembled); the combined table renders
        its rows.
        """
        md_lines: List[str] = []
        figures: List[Tuple] = []
        figure_info: List[FigureInfo] = []

        md_lines.extend(self._combined_table_intro(
            "### Univariate Cox and Permutation Importance by Model", univar_cox,
        ))

        has_pvals = imp_pvals_by_model is not None and len(imp_pvals_by_model) > 0
        if has_pvals:
            md_lines.append(sig_column_intro(self._pvalue_source, self._pvalue_method))
            md_lines.append("")

        model_blocks = split_summary_blocks(summary)
        if not model_blocks:
            md_lines.append(self._feature_summary_unavailable_note())
            md_lines.append("")
        else:
            sorted_features = get_sorted_top_features(models, importance_df)
            per_model_coefs = per_model_coef_blocks(summary)
            sig_models = set(imp_pvals_by_model) if has_pvals else None
            total_samples = len(self.learner.y_train) if self.learner.y_train is not None else None

            table_data = []
            for feat in sorted_features:
                table_data.append(
                    build_multi_model_feature_row(
                        feat, models, model_blocks, baseline_c_index,
                        sig_models=sig_models,
                        per_model_coefs=per_model_coefs,
                        univar_cox=univar_cox, total_samples=total_samples,
                    )
                )
                # Insert child rows for grouped categorical features
                table_data.extend(
                    self._build_categorical_child_rows(
                        feat, models, univar_cox, baseline_c_index,
                        per_model_coefs, total_samples,
                    )
                )

            table_df = pd.DataFrame(table_data)
            table_df = self._drop_all_empty_columns(table_df)
            column_groups = build_multi_model_column_groups(
                table_df, models, coef_models=set(per_model_coefs),
            )
            md_lines.append(format_interactive_table(
                table_df, column_groups, n_visible=self.n_features,
            ))
            md_lines.append("")

        if has_pvals:
            md_lines.append(sig_footnote(self._pvalue_source, self._pvalue_method))
            md_lines.append("")

        # Multi-model comparison chart
        _hr_kwargs = {'coef_col': '_log_hr'} if '_log_hr' in importance_df.columns else {}
        try:
            fig = self.learner.plot_importance(n=self.n_features, **_hr_kwargs, **self._group_plot_kwargs())
            figures.append(('importance_comparison.png', fig))
            md_lines.append("![Feature Importance Comparison](images/importance_comparison.png)")
            md_lines.append("")

            gh = self._group_code_hint
            figure_info.append(FigureInfo(
                figure_id='importance_comparison',
                title='Multi-Model Feature Importance Comparison',
                data_description='Tidy DataFrame with feature importance (model, feature, perm_mean, perm_std, direction, p_value, hazard_ratio, rank)',
                data_key='importance',
                learner_code=f"fig = learner.plot_importance(n={self.n_features}{gh})",
            ))
        except CHART_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "feature_analysis: importance comparison chart generation",
                warn_user=True,
                user_message=f"Could not generate importance comparison chart: {e}",
            )
            md_lines.append(f"*[Comparison chart unavailable: {e}]*")
            md_lines.append("")

        # Importance heatmap (multi-model only)
        try:
            fig_hm = self.learner.plot_importance(style="heatmap", n=self.n_features, **self._group_plot_kwargs())
            figures.append(('importance_heatmap.png', fig_hm))
            md_lines.append("![Feature Importance Heatmap](images/importance_heatmap.png)")
            md_lines.append("")

            gh = self._group_code_hint
            figure_info.append(FigureInfo(
                figure_id='importance_heatmap',
                title='Feature × Model Importance Heatmap',
                data_description='Tidy DataFrame with feature importance (model, feature, perm_mean, perm_std, direction, p_value, hazard_ratio, rank)',
                data_key='importance',
                learner_code=f"fig = learner.plot_importance(style='heatmap', n=20{gh})",
            ))
        except CHART_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                "feature_analysis: importance heatmap generation",
                warn_user=True,
                user_message=f"Could not generate importance heatmap: {e}",
            )
            md_lines.append(f"*[Importance heatmap unavailable: {e}]*")
            md_lines.append("")

        return md_lines, figures, figure_info
