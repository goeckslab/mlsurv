"""Coefficient barplot and scatter comparison for feature analysis section."""

from typing import Any, Dict, List, Tuple

from mlsurv.reports.sections.base import FigureInfo

from ._helpers import CHART_EXCEPTIONS


def generate_coefficient_analysis(section) -> Dict[str, Any]:
    """Generate coefficient barplot and scatter comparison for Cox models.

    Returns dict with 'markdown', 'figures', and 'figure_info' keys.
    """
    md_lines: List[str] = []
    figures: List[Tuple] = []
    figure_info: List[FigureInfo] = []

    trained = section._get_all_models()
    coef_models = [name for name in trained if name in ('coxph', 'coxnet')]
    has_univar = getattr(section.learner, 'univariate_cox', None) is not None
    if not coef_models and not has_univar:
        return {'markdown': '', 'figures': [], 'figure_info': []}

    plot_models = coef_models.copy()
    if has_univar:
        plot_models.append('univariate')

    md_lines.append("### Feature Coefficients")
    md_lines.append("")
    md_lines.append(
        "Cox model coefficients are log hazard ratios: positive values "
        "indicate increased risk (higher hazard), negative values indicate "
        "a protective effect (lower hazard)."
    )
    md_lines.append("")

    # Combined coefficient barplot (trained Cox models + univariate)
    try:
        fig_coef = section.learner.plot_coefficients(models=plot_models, **section._group_plot_kwargs())
        figures.append(('coefficients.png', fig_coef))
        md_lines.append("![Feature Coefficients](images/coefficients.png)")
        md_lines.append("")
        figure_info.append(FigureInfo(
            figure_id='coefficients',
            title='Feature Coefficients',
            data_description='DataFrame with coefficients (feature, coefficient, model)',
            # data_key=None: this section does not populate
            # self.data['coefficients']; users access the tidy DataFrame
            # via ``learner.plot_coefficients(return_data=True)`` — which
            # the rendered figure-level ``learner_code`` already shows.
            data_key=None,
            learner_code="fig = learner.plot_coefficients()",
        ))
    except CHART_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e, "feature_analysis: coefficient plot",
            warn_user=True,
            user_message=f"Could not generate coefficient plot: {e}",
        )
        md_lines.append(f"*[Coefficient plot unavailable: {e}]*")
        md_lines.append("")

    # Scatter comparison (2+ Cox models only)
    if len(coef_models) >= 2:
        try:
            fig = section.learner.plot_coefficient_comparison(**section._group_plot_kwargs())
            figures.append(('coefficient_comparison.png', fig))
            md_lines.append("![Coefficient Comparison](images/coefficient_comparison.png)")
            md_lines.append("")
            figure_info.append(FigureInfo(
                figure_id='coefficient_comparison',
                title='Coefficient Comparison Across Cox Models',
                data_description='Coefficient comparison across Cox models',
                # data_key=None: see coefficients FigureInfo above.
                data_key=None,
                learner_code="fig = learner.plot_coefficient_comparison()",
            ))
        except CHART_EXCEPTIONS as e:
            section._log_recoverable_exception(
                e, "feature_analysis: coefficient comparison plot",
                warn_user=True,
                user_message=f"Could not generate coefficient comparison: {e}",
            )
            md_lines.append(f"*[Coefficient comparison unavailable: {e}]*")
            md_lines.append("")

    md_lines.append(
        "**Interpretation:** Features with the largest absolute coefficients "
        "have the strongest linear associations with the outcome. Regularized "
        "models (CoxNet) shrink weak coefficients toward zero, so differences "
        "between CoxPH and CoxNet highlight which features survive penalization."
    )
    md_lines.append("")

    return {
        'markdown': '\n'.join(md_lines),
        'figures': figures,
        'figure_info': figure_info,
    }
