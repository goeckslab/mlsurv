"""SHAP-based feature analysis for feature analysis section."""

from typing import Any, Dict, List, Tuple

from mlsurv.reports.sections.base import FigureInfo

from ._helpers import CHART_EXCEPTIONS


def generate_shap_analysis(section) -> Dict[str, Any]:
    """Generate SHAP-based feature analysis content.

    Returns dict with 'markdown', 'figures', and 'figure_info' keys.
    """
    md_lines: List[str] = []
    figures: List[Tuple] = []
    figure_info: List[FigureInfo] = []

    if section.eval_result is None or section.eval_result.shap_values is None:
        return {'markdown': '', 'figures': [], 'figure_info': []}

    shap_df = section.eval_result.shap_values

    if len(shap_df) == 0:
        return {'markdown': '', 'figures': [], 'figure_info': []}

    md_lines.append("### SHAP-Based Feature Analysis")
    md_lines.append("")
    md_lines.append(
        "SHAP (SHapley Additive exPlanations) values provide a unified measure of feature "
        "importance that shows how each feature contributes to individual predictions. "
        "While permutation feature importance (PFI) measures global model reliance by "
        "shuffling each feature and observing the drop in performance, SHAP provides "
        "per-sample attributions that reveal both the magnitude and direction of each "
        "feature's effect on each prediction."
    )
    md_lines.append("")

    section.data['shap'] = shap_df

    try:
        fig_bar = section.learner.plot_shap(kind='importance', n=section.n_features, **section._group_plot_kwargs())
        figures.append(('shap_importance.png', fig_bar))
        md_lines.append("![SHAP Feature Importance](images/shap_importance.png)")
        md_lines.append("")
        md_lines.append(
            "*The bar plot shows mean |SHAP| with violin overlay showing variability across samples.*"
        )
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='shap_importance',
            title='SHAP Feature Importance',
            data_description='Long-format DataFrame with SHAP values (model_type, feature, sample_idx, shap_value)',
            data_key='shap',
            learner_code="fig = learner.plot_shap(kind='importance')",
        ))
    except CHART_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: shap bar plot generation",
            warn_user=True,
            user_message=f"Could not generate SHAP bar plot: {e}",
        )
        md_lines.append(f"*[SHAP bar plot unavailable: {e}]*")
        md_lines.append("")

    # Generate SHAP beeswarm plot
    try:
        fig_bee = section.learner.plot_shap(kind='beeswarm', n=section.n_features, **section._group_plot_kwargs())
        figures.append(('shap_beeswarm.png', fig_bee))
        md_lines.append("![SHAP Beeswarm](images/shap_beeswarm.png)")
        md_lines.append("")
        md_lines.append(
            "*The beeswarm plot shows effect directions: points to the right (positive SHAP) "
            "increase predicted risk, points to the left (negative SHAP) decrease predicted risk. "
            "Color indicates feature value (blue=low, red=high).*"
        )
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='shap_beeswarm',
            title='SHAP Beeswarm Plot',
            data_description='Long-format DataFrame with SHAP values (model_type, feature, sample_idx, shap_value)',
            data_key='shap',
            learner_code="fig = learner.plot_shap(kind='beeswarm')",
        ))
    except CHART_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: shap beeswarm plot generation",
            warn_user=True,
            user_message=f"Could not generate SHAP beeswarm plot: {e}",
        )
        md_lines.append(f"*[SHAP beeswarm plot unavailable: {e}]*")
        md_lines.append("")

    return {
        'markdown': '\n'.join(md_lines),
        'figures': figures,
        'figure_info': figure_info,
    }
