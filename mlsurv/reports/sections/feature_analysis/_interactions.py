"""Feature interaction analysis for the feature analysis report section."""

from typing import Any, Dict, List, Tuple

from mlsurv.reports.sections.base import FigureInfo

from ._helpers import CHART_EXCEPTIONS


def generate_interaction_analysis(section) -> Dict[str, Any]:
    """Generate feature interaction analysis content.

    Returns dict with 'markdown', 'figures', and 'figure_info' keys.
    """
    md_lines: List[str] = []
    figures: List[Tuple] = []
    figure_info: List[FigureInfo] = []

    if section.eval_result is None or section.eval_result.interactions is None:
        return {'markdown': '', 'figures': [], 'figure_info': []}

    interactions_df = section.eval_result.interactions
    if len(interactions_df) == 0:
        return {'markdown': '', 'figures': [], 'figure_info': []}

    md_lines.append("### Feature Interactions")
    md_lines.append("")

    has_shap = (
        'method' in interactions_df.columns
        and (interactions_df['method'] == 'shap').any()
    )
    has_hstat = (
        'method' in interactions_df.columns
        and (interactions_df['method'] == 'hstat').any()
    )
    intro = (
        "Feature interactions identify pairs of features whose joint effect on "
        "predictions differs from the sum of their individual effects. "
    )
    # SHAP interaction values use a path-dependent reference, unlike the
    # interventional cohort SHAP beeswarm shown elsewhere in the report; the
    # two can diverge when features are correlated.
    shap_reference_note = (
        " Note: these SHAP interaction values use a path-dependent "
        "(true-to-the-data) reference, which differs from the interventional "
        "reference used by the cohort SHAP beeswarm; the two can diverge when "
        "features are correlated."
    )
    if has_shap and has_hstat:
        md_lines.append(
            intro
            + "Two methods are shown: **SHAP interaction values** decompose each "
            "prediction into main effects and pairwise interactions using "
            "TreeSHAP (tree-based models), while Friedman's **H-statistic** "
            "measures the proportion of joint partial dependence variance "
            "attributable to the interaction (model-agnostic)."
            + shap_reference_note
        )
    elif has_shap:
        md_lines.append(
            intro
            + "SHAP interaction values decompose each prediction into main "
            "effects and pairwise interactions using TreeSHAP \u2014 values "
            "represent the mean absolute interaction strength across test "
            "samples."
            + shap_reference_note
        )
    else:
        md_lines.append(
            intro
            + "The Friedman H-statistic measures the proportion of joint partial "
            "dependence variance attributable to the interaction \u2014 values "
            "near 0 indicate no interaction, while values near 1 indicate "
            "strong interaction."
        )
    md_lines.append("")

    # Count features and pairs for context
    all_features = sorted(
        set(interactions_df['feature_1']) | set(interactions_df['feature_2'])
    )
    n_features = len(all_features)
    n_pairs = len(interactions_df)
    md_lines.append(
        f"*{n_pairs} feature pairs evaluated across {n_features} features "
        f"(top features by permutation importance).*"
    )
    md_lines.append("")

    # Note categorical groups when present
    cat_groups = getattr(section.learner._state, 'categorical_groups', None)
    if cat_groups:
        group_features = [f for f in all_features if f in cat_groups]
        if group_features:
            group_details = ", ".join(
                f"**{g}** ({len(cat_groups[g])} levels)"
                for g in group_features
            )
            md_lines.append(
                f"Grouped categorical features: {group_details}. "
                "These features were treated as single units during "
                "interaction detection — all dummy columns were permuted "
                "together in the partial dependence computation."
            )
            md_lines.append("")

    section.data['interactions'] = interactions_df

    try:
        fig = section.learner.plot_interactions(
            n=n_pairs,
            **section._group_plot_kwargs(),
        )
        figures.append(('interactions_heatmap.png', fig))
        md_lines.append("![Feature Interactions](images/interactions_heatmap.png)")
        md_lines.append("")
        if has_shap and has_hstat:
            md_lines.append(
                "*Heatmap of pairwise interaction values for the top "
                "features by permutation importance. Darker cells indicate "
                "stronger interactions. Models use SHAP or H-statistic "
                "depending on model type.*"
            )
        elif has_shap:
            md_lines.append(
                "*Heatmap of pairwise SHAP interaction values for the top "
                "features by permutation importance. Darker cells indicate "
                "stronger interactions.*"
            )
        else:
            md_lines.append(
                "*Heatmap of pairwise H-statistics for the top features by "
                "permutation importance. Darker cells indicate stronger "
                "interactions.*"
            )
        md_lines.append("")

        if has_shap and has_hstat:
            value_col_desc = 'h_statistic, shap_interaction, method'
        elif has_shap:
            value_col_desc = 'shap_interaction'
        else:
            value_col_desc = 'h_statistic'
        figure_info.append(FigureInfo(
            figure_id='interactions_heatmap',
            title='Feature Interaction Heatmap',
            data_description=(
                'Tidy DataFrame with interaction strengths '
                f'(model_type, feature_1, feature_2, {value_col_desc})'
            ),
            data_key='interactions',
            learner_code="fig = learner.plot_interactions()",
        ))
    except CHART_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: interactions heatmap generation",
            warn_user=True,
            user_message=f"Could not generate interactions heatmap: {e}",
        )
        md_lines.append(f"*[Interactions heatmap unavailable: {e}]*")
        md_lines.append("")

    return {
        'markdown': '\n'.join(md_lines),
        'figures': figures,
        'figure_info': figure_info,
    }


def generate_interaction_pdp_analysis(section) -> Dict[str, Any]:
    """Generate 2D partial dependence plot analysis for top interaction pairs.

    Returns dict with 'markdown', 'figures', and 'figure_info' keys.
    """
    md_lines: List[str] = []
    figures: List[Tuple] = []
    figure_info: List[FigureInfo] = []

    if section.eval_result is None or not section.eval_result.interaction_surfaces:
        return {'markdown': '', 'figures': [], 'figure_info': []}

    surfaces = section.eval_result.interaction_surfaces
    if len(surfaces) == 0:
        return {'markdown': '', 'figures': [], 'figure_info': []}

    md_lines.append("### 2D Partial Dependence")
    md_lines.append("")

    # Detect whether any surfaces involve categorical features
    has_categorical = any(
        s.get('grid_1_type') == 'categorical' or s.get('grid_2_type') == 'categorical'
        for s in surfaces
    )
    if has_categorical:
        md_lines.append(
            "The plots below show how predicted risk varies as two features "
            "change jointly. Contour plots are used for numeric × numeric "
            "pairs (non-parallel contours indicate interaction). For a numeric "
            "× categorical pair, one risk curve is drawn per category level "
            "(diverging curves indicate interaction). Heatmaps are used for "
            "categorical × categorical pairs, where each cell shows the "
            "average predicted risk for a category level combination."
        )
    else:
        md_lines.append(
            "The contour plots below show how predicted risk varies as two "
            "features change jointly. Contour lines connect points with equal "
            "predicted risk. Non-parallel contours indicate that the features "
            "interact — one feature's effect depends on the other's value."
        )
    md_lines.append("")

    try:
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        fig = plot_interaction_pdp_gallery(surfaces)
        figures.append(('interactions_pdp.png', fig))
        md_lines.append("![2D Partial Dependence](images/interactions_pdp.png)")
        md_lines.append("")
        md_lines.append(
            "*2D partial dependence plots for the top feature pairs by "
            "H-statistic. Use the dropdown to switch between pairs.*"
        )
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='interactions_pdp',
            title='2D Partial Dependence Plot',
            data_description=(
                'Tidy DataFrame with columns [feature_1, feature_2, prediction] '
                '— one row per grid cell'
            ),
            data_key='interaction_surfaces',
            learner_code=(
                "fig = learner.plot_interaction_pdp('feature_1', 'feature_2')"
            ),
        ))
    except CHART_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: interaction PDP generation",
            warn_user=True,
            user_message=f"Could not generate interaction PDP: {e}",
        )
        md_lines.append(f"*[Interaction PDP unavailable: {e}]*")
        md_lines.append("")

    return {
        'markdown': '\n'.join(md_lines),
        'figures': figures,
        'figure_info': figure_info,
    }
