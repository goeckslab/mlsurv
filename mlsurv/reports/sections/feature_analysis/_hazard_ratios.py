"""Hazard ratio forest plots for feature analysis section."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from mlsurv.reports.sections.base import FigureInfo

from ._coefficients import (
    find_covariance_gap_reason,
    find_hr_suppression_reason,
    find_hr_valid_model,
)
from ._helpers import (
    CHART_EXCEPTIONS, _missing_data_note, _unplottable_hr_note,
)


def generate_hazard_ratio_analysis(
    section, univar_cox: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate hazard ratio forest plots (univariate + multivariate).

    Returns dict with 'markdown', 'figures', and 'figure_info' keys.
    """
    md_lines: List[str] = []
    figures: List[Tuple] = []
    figure_info: List[FigureInfo] = []

    # Multivariate HR forest requires a model whose exp(coef) is a genuine
    # hazard ratio (only unpenalized Cox PH). The univariate path is genuine
    # per-feature Cox and stays valid for any cohort.
    cox_model = find_hr_valid_model(section)
    has_univar = univar_cox is not None and not univar_cox.empty
    has_multivar = cox_model is not None
    # Set only when a model was trained and its hazard ratios were deliberately
    # withheld. It is rendered here, where the forest plot would have been:
    # the same explanation in Model Development is no help to someone looking
    # at the gap in this section.
    suppressed_reason = None if has_multivar else find_hr_suppression_reason(section)

    if not has_univar and not has_multivar and not suppressed_reason:
        return {'markdown': '', 'figures': [], 'figure_info': []}

    md_lines.append("### Hazard Ratios")
    md_lines.append("")
    md_lines.append(
        "A hazard ratio (HR) quantifies the multiplicative effect of a "
        f"one-unit increase in a feature on the instantaneous {section.outcome_name} rate. "
        "HR = 2 means the hazard is doubled; HR = 0.5 means it is halved."
    )
    md_lines.append("")

    if has_univar:
        _render_univariate_hr(section, univar_cox, md_lines, figures, figure_info)

    if has_multivar:
        _render_multivariate_hr(section, md_lines, figures, figure_info)
    elif suppressed_reason:
        md_lines.append("#### Multivariate Hazard Ratios")
        md_lines.append("")
        md_lines.append(f"*Not reported for this model. {suppressed_reason}*")
        md_lines.append("")

    if has_univar and has_multivar:
        _render_hr_comparison(section, md_lines, figures, figure_info)

    return {
        'markdown': '\n'.join(md_lines),
        'figures': figures,
        'figure_info': figure_info,
    }


def _render_univariate_hr(section, univar_cox, md_lines, figures, figure_info):
    """Render univariate HR forest plot."""
    if 'hr_lower_95' not in univar_cox.columns:
        return

    try:
        from mlsurv.viz.features import plot_hazard_ratios as _plot_hr

        hr_data = section.learner.plot_hazard_ratios(
            kind='univariate', n_features=section.n_features, return_data=True,
        )
        section.data['univariate_hazard_ratios'] = hr_data

        fig_uni = _plot_hr(
            hr_data, n_features=section.n_features,
            title='Univariate Cox — Hazard Ratios',
            **section._group_plot_kwargs(),
        )

        md_lines.append("#### Univariate Hazard Ratios")
        md_lines.append("")
        hr_desc = (
            "Each feature is tested independently via univariate Cox regression. "
            "HR > 1 indicates increased hazard (risk factor); HR < 1 indicates "
            f"decreased hazard (protective). Shows up to {section.n_features} features "
            "ranked by absolute deviation from HR = 1."
        )
        hr_desc += _missing_data_note(univar_cox)
        # Describe the rows the figure draws, not the whole table behind it:
        # plot_hazard_ratios(return_data=True) hands back the untruncated frame.
        from mlsurv.viz.features import select_plotted_hr_features
        hr_desc += _unplottable_hr_note(
            select_plotted_hr_features(hr_data, section.n_features)
        )
        md_lines.append(hr_desc)
        md_lines.append("")

        figures.append(('univariate_hazard_ratios.png', fig_uni))
        md_lines.append("![Univariate Hazard Ratios](images/univariate_hazard_ratios.png)")
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='univariate_hazard_ratios',
            title='Univariate Cox — Hazard Ratios',
            data_description='DataFrame with univariate hazard ratios (feature, hr, ci_lower, ci_upper)',
            data_key='univariate_hazard_ratios',
            learner_code="fig = learner.plot_hazard_ratios(kind='univariate')",
        ))
    except CHART_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: univariate hazard ratio forest plot",
            warn_user=True,
            user_message=f"Could not generate univariate HR plot: {e}",
        )
        md_lines.append(f"*[Univariate HR plot unavailable: {e}]*")
        md_lines.append("")


def _render_multivariate_hr(section, md_lines, figures, figure_info):
    """Render multivariate HR forest plot."""
    try:
        from mlsurv.viz.features import plot_hazard_ratios as _plot_hr_mv

        hr_data = section.learner.plot_hazard_ratios(
            kind='multivariate', n_features=section.n_features, return_data=True,
        )
        section.data['hazard_ratios'] = hr_data

        fig = _plot_hr_mv(
            hr_data, n_features=section.n_features,
            title='Multivariate CoxPH — Hazard Ratios',
            **section._group_plot_kwargs(),
        )

        # Only promise intervals the plot actually drew. When the covariance
        # matrix could not be built, compute_hazard_ratios sets ci_available
        # False and the forest plot hides its whiskers, so an unconditional
        # "with 95% confidence intervals" would describe a chart that has none.
        # None (no such column) is a third state: say nothing either way, the
        # same way the convergence line treats an unrecorded outcome.
        has_ci = (bool(hr_data["ci_available"].any())
                  if "ci_available" in hr_data.columns else None)
        interval_clause = " with 95% confidence intervals" if has_ci else ""
        md_lines.append("#### Multivariate Hazard Ratios")
        md_lines.append("")
        md_lines.append(
            "Adjusted hazard ratios from the multivariate Cox model"
            f"{interval_clause}. Each HR represents the independent effect of "
            "a feature after adjustment for all other features in the model. "
            f"Shows up to {section.n_features} features "
            "ranked by absolute deviation from HR = 1."
        )
        md_lines.append("")
        # `is False`, not `not has_ci`: None means the frame did not say, so
        # asserting the intervals are missing would be as much of a guess as
        # asserting they are there.
        if has_ci is False:
            reason = (find_covariance_gap_reason(section)
                      or "No covariance matrix was available for this fit.")
            md_lines.append(f"*Confidence intervals are not shown. {reason}*")
            md_lines.append("")

        figures.append(('hazard_ratios.png', fig))
        md_lines.append("![Hazard Ratios](images/hazard_ratios.png)")
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='hazard_ratios',
            title='Multivariate Cox — Hazard Ratios',
            data_description='DataFrame with hazard ratios (feature, hr, ci_lower, ci_upper)',
            data_key='hazard_ratios',
            learner_code="fig = learner.plot_hazard_ratios(kind='multivariate')",
        ))
    except CHART_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: hazard ratio forest plot generation",
            warn_user=True,
            user_message=f"Could not generate hazard ratio plot: {e}",
        )
        md_lines.append(f"*[Hazard ratio plot unavailable: {e}]*")
        md_lines.append("")


def _render_hr_comparison(section, md_lines, figures, figure_info):
    """Render univariate vs multivariate HR comparison plot."""
    try:
        fig_cmp = section.learner.plot_hazard_ratios_comparison(
            n_features=section.n_features,
            **section._group_plot_kwargs(),
        )
        figures.append(('hazard_ratios_comparison.png', fig_cmp))
        md_lines.append("#### Univariate vs Multivariate Hazard Ratios")
        md_lines.append("")
        md_lines.append(
            "Side-by-side comparison of univariate and multivariate hazard ratios. "
            "Features whose HR changes substantially between the two analyses may be "
            "confounded by other variables."
        )
        md_lines.append("")
        md_lines.append("![HR Comparison](images/hazard_ratios_comparison.png)")
        md_lines.append("")
        figure_info.append(FigureInfo(
            figure_id='hazard_ratios_comparison',
            title='Univariate vs Multivariate Hazard Ratios',
            data_description='Merged DataFrame with univariate and multivariate HRs',
            # data_key=None: this section does not populate
            # self.data['hazard_ratios_comparison']; users access the
            # tidy DataFrame via
            # ``learner.plot_hazard_ratios_comparison(return_data=True)``.
            data_key=None,
            learner_code="fig = learner.plot_hazard_ratios_comparison()",
        ))
    except CHART_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: HR comparison plot",
            warn_user=True,
            user_message=f"Could not generate HR comparison plot: {e}",
        )
        md_lines.append(f"*[HR comparison plot unavailable: {e}]*")
        md_lines.append("")
