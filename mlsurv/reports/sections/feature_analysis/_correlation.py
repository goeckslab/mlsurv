"""Correlation analysis for feature analysis section.

Generates the correlation heatmap, interpretation, and warning callout.
"""

import html as html_module
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from mlsurv.evaluation.importance import detect_correlated_features
from mlsurv.reports._constants import CORRELATION_THRESHOLD, MAX_CORRELATION_FEATURES
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, FigureInfo

from ._helpers import CHART_EXCEPTIONS, _assess_correlation_severity, _md_escape


def generate_correlation_analysis(section) -> Dict[str, Any]:
    """Generate feature correlation analysis content.

    Returns dict with 'markdown', 'figures', 'figure_info',
    '_correlated_features', '_severity' (str), and '_recommendation'
    keys. ``_correlated_features`` is the set of features in any
    high-correlation pair, or ``None`` when the correlation could not
    be computed ("unknown", not "none correlated" — the feature
    summary renders None as an all-NA ``correlated`` column).
    """
    md_lines: List[str] = []
    figures: List[Tuple] = []
    figure_info: List[FigureInfo] = []
    correlated_features: set = set()
    severity = 'none'

    md_lines.append("### Feature Correlation Analysis")
    md_lines.append("")

    md_lines.append(
        "Feature correlation analysis identifies linear relationships between predictor "
        f"variables. High correlations (|R| > {CORRELATION_THRESHOLD}) can indicate multicollinearity, which "
        "may affect model interpretation and coefficient stability in linear models like "
        "CoxPH. Tree-based models (RSF, GBSA) are generally robust to correlated features, "
        "but permutation importance values may be distributed across correlated variables, "
        "making individual feature contributions appear smaller than their combined effect. "
        "Correlations are computed on the full dataset (train + test) for a complete picture. "
        f"The heatmap shows up to {MAX_CORRELATION_FEATURES} features selected by highest total absolute correlation."
    )
    md_lines.append("")

    # Use full dataset (train + test) for correlation analysis
    try:
        X_full = pd.concat([section.learner.X_train, section.learner.X_test], ignore_index=True)
        corr_pairs = detect_correlated_features(X_full, threshold=CORRELATION_THRESHOLD)
        n_features = len(X_full.columns)
    except ALL_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: correlation data processing",
            warn_user=True,
            user_message=f"Could not compute feature correlations: {e}",
        )
        md_lines.append(f"*[Correlation analysis unavailable: {e}]*")
        md_lines.append("")
        section.data['correlation'] = pd.DataFrame(
            columns=['feature_1', 'feature_2', 'correlation']
        )
        return {
            'markdown': '\n'.join(md_lines),
            'figures': figures,
            'figure_info': figure_info,
            # None = "could not be determined", not "no correlated pairs".
            '_correlated_features': None,
            '_severity': severity,
            '_recommendation': '',
        }

    # Compute severity metadata for downstream consumers
    recommendation = ''
    if corr_pairs and n_features > 0:
        max_corr = max(abs(c) for _, _, c in corr_pairs)
        correlated_features = {f for f1, f2, _ in corr_pairs for f in (f1, f2)}
        pct_involved = (len(correlated_features) / n_features) * 100
        severity, recommendation = _assess_correlation_severity(
            len(corr_pairs), pct_involved, max_corr,
        )
    section.data['correlation_severity'] = severity

    # Build tidy correlation DataFrame
    corr_matrix = _compute_correlation_data(section, X_full)
    if corr_matrix is not None:
        corr_rows = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_rows.append({
                        'feature_1': col1,
                        'feature_2': col2,
                        'correlation': corr_matrix.loc[col1, col2],
                    })
        section.data['correlation'] = pd.DataFrame(corr_rows)
    else:
        section.data['correlation'] = pd.DataFrame(
            columns=['feature_1', 'feature_2', 'correlation']
        )

    # Generate correlation heatmap
    _render_correlation_chart(section, X_full, md_lines, figures, figure_info)

    # Generate interpretation statement
    interpretation = generate_correlation_interpretation(
        corr_pairs, n_features, correlated_features, severity, recommendation,
    )
    md_lines.append(interpretation)
    md_lines.append("")

    # Show highly correlated pairs table if any exist
    _render_correlated_pairs_table(section, corr_pairs, md_lines)

    return {
        'markdown': '\n'.join(md_lines),
        'figures': figures,
        'figure_info': figure_info,
        '_correlated_features': correlated_features,
        '_severity': severity,
        '_recommendation': recommendation,
    }


def _compute_correlation_data(section, X_full) -> Optional[pd.DataFrame]:
    """Compute the correlation matrix, returning None on failure."""
    try:
        return X_full.corr()
    except ALL_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: correlation matrix computation",
            warn_user=True,
            user_message=f"Could not compute correlation matrix: {e}",
        )
        return None


def _render_correlation_chart(section, X_full, md_lines, figures, figure_info):
    """Generate and append the correlation heatmap figure."""
    try:
        from mlsurv.viz.features import plot_feature_correlation

        fig = plot_feature_correlation(
            X_full,
            threshold=CORRELATION_THRESHOLD,
            max_features=MAX_CORRELATION_FEATURES,
            title='Feature Correlation Matrix',
            **section._group_plot_kwargs(),
        )
        figures.append(('correlation_heatmap.png', fig))
        md_lines.append("![Feature Correlation Matrix](images/correlation_heatmap.png)")
        md_lines.append("")

        figure_info.append(FigureInfo(
            figure_id='correlation_heatmap',
            title='Feature Correlation Matrix',
            data_description='Tidy DataFrame with feature pair correlations (feature_1, feature_2, correlation)',
            data_key='correlation',
            learner_code="fig = learner.plot_feature_correlation()",
        ))
    except CHART_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "feature_analysis: correlation heatmap generation",
            warn_user=True,
            user_message=f"Could not generate correlation heatmap: {e}",
        )
        md_lines.append(f"*[Correlation heatmap unavailable: {e}]*")
        md_lines.append("")


def _render_correlated_pairs_table(section, corr_pairs, md_lines):
    """Append the highly correlated pairs markdown table."""
    if not corr_pairs:
        return

    md_lines.append(f"#### Highly Correlated Feature Pairs (|R| > {CORRELATION_THRESHOLD})")
    md_lines.append("")

    corr_data = [
        [f1, f2, f"{corr:.3f}"]
        for f1, f2, corr in corr_pairs[:10]
    ]
    corr_df = pd.DataFrame(
        corr_data,
        columns=["Feature 1", "Feature 2", "Correlation"]
    )
    md_lines.append(section._format_table(corr_df))

    if len(corr_pairs) > 10:
        md_lines.append(f"\n*... and {len(corr_pairs) - 10} more pairs*")
    md_lines.append("")


def generate_correlation_interpretation(
    corr_pairs: list,
    n_features: int,
    correlated_features: set,
    severity: str,
    recommendation: str,
) -> str:
    """Generate an interpretation statement about feature correlations."""
    if not corr_pairs:
        return (
            "**Interpretation:** No feature pairs exceed the high correlation threshold "
            f"(|R| > {CORRELATION_THRESHOLD}). The feature set shows acceptable independence for stable "
            "coefficient estimation and interpretable permutation importance values."
        )

    if n_features == 0:
        return "**Interpretation:** No features available for correlation analysis."

    n_pairs = len(corr_pairs)
    max_pair = max(corr_pairs, key=lambda t: abs(t[2]))
    n_involved = len(correlated_features)
    pct_involved = (n_involved / n_features) * 100

    return (
        f"**Interpretation:** The dataset shows **{severity} multicollinearity** with "
        f"{n_pairs} feature pair{'s' if n_pairs > 1 else ''} exceeding |R| > {CORRELATION_THRESHOLD}. "
        f"The strongest correlation is between *{_md_escape(max_pair[0])}* and *{_md_escape(max_pair[1])}* "
        f"(r = {max_pair[2]:.3f}). "
        f"{n_involved} of {n_features} features ({pct_involved:.0f}%) are involved in "
        f"high correlations. {recommendation}"
    )


def generate_correlation_warning_callout(severity: str, recommendation: str = '') -> str:
    """Generate HTML warning callout for moderate/substantial multicollinearity.

    Returns empty string for minor or no correlation severity.
    """
    if severity not in ('moderate', 'substantial'):
        return ''

    sev_escaped = html_module.escape(severity.capitalize())
    rec_escaped = html_module.escape(recommendation)
    return (
        '<!-- MLSURV_RAW_HTML -->\n'
        '<div class="mlsurv-callout-warning">\n'
        f'  <strong>&#9888; Multicollinearity Warning:</strong>\n'
        f'  {sev_escaped} multicollinearity detected. Permutation importance scores may be\n'
        f'  split across correlated features, causing individual features to appear less\n'
        f'  important than their true effect. {rec_escaped}\n'
        '</div>\n'
        '<!-- /MLSURV_RAW_HTML -->'
    )
