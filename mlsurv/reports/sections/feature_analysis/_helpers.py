"""Pure formatting utilities for feature analysis.

All functions in this module are stateless — they take data in and return
formatted strings or classification labels without accessing any section state.
"""

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from mlsurv.reports._constants import P_STARS, format_pvalue
from mlsurv.reports.sections.base import ALL_EXCEPTIONS

# Chart blocks can fail on lazy viz imports (ImportError) in addition to data/runtime errors
CHART_EXCEPTIONS = (ImportError,) + ALL_EXCEPTIONS


def _md_escape(text: str) -> str:
    """Escape markdown special characters in feature names."""
    return text.replace('*', r'\*').replace('_', r'\_')


def _assess_correlation_severity(
    n_pairs: int, pct_involved: float, max_corr: float,
) -> Tuple[str, str]:
    """Return (severity_label, recommendation_text) for correlation findings."""
    if n_pairs >= 10 or pct_involved > 50:
        return (
            "substantial",
            "Consider feature selection or dimensionality reduction (e.g., PCA) "
            "before modeling. For interpretability, examine variable inflation factors "
            "(VIF) or use regularized models like CoxNet.",
        )

    if n_pairs >= 5 or max_corr > 0.9:
        return (
            "moderate",
            "Review correlated feature pairs and consider removing redundant variables "
            "or combining them. Tree-based models may distribute importance across "
            "correlated features.",
        )

    return (
        "minor",
        "These correlations are unlikely to substantially impact model performance, "
        "but be cautious when interpreting individual coefficients or importance "
        "scores for the affected features.",
    )


def _format_importance_value(
    importance: float, baseline_c_index: Optional[float],
) -> str:
    """Format importance as 'value (% change)' relative to the baseline metric."""
    if baseline_c_index is not None and baseline_c_index > 0:
        pct_change = (importance / baseline_c_index) * 100
        return f"{importance:.4f} ({pct_change:.1f}%)"
    return f"{importance:.4f}"


def _format_significance(p_adjusted: float) -> str:
    """Format adjusted p-value as star notation for table display."""
    if pd.isna(p_adjusted) or not np.isfinite(p_adjusted) or p_adjusted < 0:
        return ""
    for threshold, stars in P_STARS:
        if p_adjusted < threshold:
            return stars
    return ""


def _format_adjusted_pvalue(p: float) -> str:
    """Format adjusted p-value for display.

    Empty string (rather than em dash) for missing/invalid values so
    feature-analysis tables render blank cells for unrecomputed rows.
    """
    if pd.isna(p) or not np.isfinite(p) or p < 0:
        return ""
    return format_pvalue(p)


def _derive_effect_from_hr(hr: float) -> str:
    """Derive effect direction from univariate Cox hazard ratio."""
    if pd.isna(hr) or not np.isfinite(hr):
        return "Unknown"
    if hr > 1:
        return "Risk (+)"
    if hr < 1:
        return "Protective (-)"
    return "Neutral"


def _lookup_adjusted_pvalue(pvals_df: pd.DataFrame, feature: str) -> float:
    """Look up a feature's adjusted p-value, returning NaN if not found."""
    match = pvals_df[pvals_df['feature'] == feature]
    return match['p_adjusted'].values[0] if len(match) > 0 else np.nan


def _missing_data_note(univar_cox: Optional[pd.DataFrame]) -> str:
    """Return a markdown note about missing data in univariate Cox, or empty string."""
    if univar_cox is None or 'n_missing' not in univar_cox.columns:
        return ""
    missing = univar_cox[univar_cox['n_missing'] > 0]
    if missing.empty:
        return ""
    n_total = len(univar_cox)
    n_affected = len(missing)
    max_missing = int(missing['n_missing'].dropna().max())
    if n_affected == n_total:
        return (
            f"\n\n*Note: Missing data detected in all {n_total} features "
            f"(up to {max_missing} observations per feature). "
            "Univariate Cox results use per-feature complete-case analysis — "
            "each feature is fitted on all observations where that feature is non-missing.*"
        )
    return (
        f"\n\n*Note: Missing data detected in {n_affected} of {n_total} features "
        f"(up to {max_missing} observations per feature). "
        "Affected features use per-feature complete-case analysis.*"
    )


def _short_column_name(col_name: str) -> str:
    """Strip last parenthetical model suffix: 'PFI (CoxPH)' -> 'PFI'."""
    m = re.match(r'^(.+)\s+\([^)]+\)$', col_name)
    return m.group(1) if m else col_name
