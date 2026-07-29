"""Pure formatting utilities for feature analysis.

All functions in this module are stateless — they take data in and return
formatted strings or classification labels without accessing any section state.
"""

import re
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from mlsurv._warnings import InternalWarning
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
    # dropna() leaves +/-inf in place, and int(inf) raises OverflowError. Drop
    # the count from the note rather than the note (or the section) entirely.
    max_missing = missing['n_missing'].dropna().max()
    if np.isfinite(max_missing):
        extent = f" (up to {int(max_missing)} observations per feature)"
    else:
        extent = ""
        # A per-feature missing count is a non-negative integer, so this means
        # the univariate table is corrupt. Say so rather than just shortening
        # the sentence and leaving the reader none the wiser.
        warnings.warn(
            f"Univariate Cox 'n_missing' holds a non-finite value "
            f"({max_missing}); omitting the count from the missing-data note.",
            InternalWarning,
            stacklevel=2,
        )
    if n_affected == n_total:
        return (
            f"\n\n*Note: Missing data detected in all {n_total} features"
            f"{extent}. "
            "Univariate Cox results use per-feature complete-case analysis — "
            "each feature is fitted on all observations where that feature is non-missing.*"
        )
    return (
        f"\n\n*Note: Missing data detected in {n_affected} of {n_total} features"
        f"{extent}. "
        "Affected features use per-feature complete-case analysis.*"
    )


def _unplottable_hr_note(hr_data: Optional[pd.DataFrame]) -> str:
    """Markdown note naming features a log-scaled forest plot cannot draw.

    The plotting layer warns the person who ran the code. That warning never
    reaches the person reading the report, who otherwise sees a labelled row
    with no marker and no way to tell whether it is a bug or the data.
    """
    if hr_data is None or 'feature' not in hr_data.columns:
        return ""
    cols = [c for c in ('hr', 'ci_lower', 'ci_upper') if c in hr_data.columns]
    if not cols:
        return ""
    values = hr_data[cols].apply(pd.to_numeric, errors='coerce')
    usable = np.isfinite(values) & (values > 0)
    # Present-but-unusable, tested against the original frame: a value that was
    # already missing is ordinary missing data, not an overflow, while one that
    # coercion turned into NaN is a corrupt entry worth reporting.
    bad = hr_data[cols].notna() & ~usable
    affected = hr_data.loc[bad.any(axis=1), 'feature'].astype(str).tolist()
    if not affected:
        return ""
    names = ", ".join(_md_escape(f) for f in affected)
    return (
        f"\n\n*Note: {names} could not be drawn in full. A hazard ratio or "
        "confidence bound overflowed to infinity or underflowed to zero, and a "
        "log-scaled axis can place neither, so the affected marker or whisker "
        "is omitted. That usually means separation or bad scaling.*"
    )


def _short_column_name(col_name: str) -> str:
    """Strip last parenthetical model suffix: 'PFI (CoxPH)' -> 'PFI'."""
    m = re.match(r'^(.+)\s+\([^)]+\)$', col_name)
    return m.group(1) if m else col_name
