"""Shared helpers for feature visualization submodules.

Data prep, feature selection, group utilities, and common layout patterns
used by importance, coefficient, hazard ratio, and correlation renderers.
"""

import warnings

import numpy as np
import pandas as pd

from mlsurv._warnings import InternalWarning
from mlsurv.viz._palettes import _add_group_legend_traces
from mlsurv.viz._theme import truncate_labels, margin_for_labels


def _feature_color(feature, feat_to_grp, grp_to_color, default="#1f77b4"):
    """Return the group color for *feature*, or *default* if ungrouped."""
    if feat_to_grp and feature in feat_to_grp:
        return grp_to_color.get(feat_to_grp[feature], default)
    return default


def _feature_summary_hover(feature, feat_data):
    """Build a rich hover string summarising one feature's distribution."""
    mean_val = float(np.mean(feat_data))
    median_val = float(np.median(feat_data))
    q1 = float(np.percentile(feat_data, 25))
    q3 = float(np.percentile(feat_data, 75))
    std_val = float(np.std(feat_data, ddof=1))
    return (
        f"<b>{feature}</b><br>"
        f"Mean: {mean_val:.4f} \u00b1 {std_val:.4f}<br>"
        f"Median: {median_val:.4f}<br>"
        f"Q1: {q1:.4f}  Q3: {q3:.4f}<br>"
        f"n = {len(feat_data)}"
        "<extra></extra>"
    )


def _apply_group_layout(fig, has_groups, grp_to_color, d, top_feature_list,
                         xlim=None):
    """Apply shared layout epilogue for raincloud / boxplot importance plots."""
    if has_groups:
        _add_group_legend_traces(fig, grp_to_color)

    display_labels, full_labels = truncate_labels(top_feature_list)
    margin_l = margin_for_labels(display_labels)

    xaxis_kw = {}
    if xlim is not None:
        xaxis_kw["range"] = list(xlim)
    fig.update_layout(
        margin=dict(l=margin_l, t=d["margin_t"] + 10),
        xaxis_title="Permutation Importance", yaxis_title="",
        xaxis=xaxis_kw,
        yaxis=dict(
            categoryorder="array", categoryarray=full_labels,
            tickmode="array", tickvals=full_labels, ticktext=display_labels,
        ),
        showlegend=has_groups,
        legend=(
            dict(
                x=1.02, xanchor="left", y=0.5, yanchor="middle",
            )
            if has_groups
            else {}
        ),
    )


def _auto_detect_feature_groups(df, feature_col):
    """Auto-detect feature_groups from a 'group' or 'feature_group' column.

    NaN values are skipped and non-string values are coerced to strings
    (with a warning).  Returns ``None`` if no matching column exists or
    all values are NaN.
    """
    for col in ("group", "feature_group"):
        if col in df.columns:
            mapping = {}
            coerced = 0
            for feat, grp in zip(df[feature_col], df[col]):
                if pd.notna(grp):
                    if not isinstance(grp, str):
                        coerced += 1
                    mapping[feat] = str(grp)
            if coerced:
                warnings.warn(
                    f"Auto-detected '{col}' column has {coerced} non-string "
                    f"value(s) that were coerced to strings.",
                    InternalWarning,
                    stacklevel=3,
                )
            return mapping if mapping else None
    return None


def _assign_coef_sign(top, coef_col):
    """Add a 'coef_sign' column and return the associated color map."""
    if coef_col and coef_col in top.columns:
        def _label(x):
            if pd.isna(x):
                return "Zero"
            if x > 0:
                return "Positive (HR>1)"
            if x < 0:
                return "Negative (HR<1)"
            return "Zero"

        top["coef_sign"] = top[coef_col].apply(_label)
        return {
            "Positive (HR>1)": "#d62728",
            "Negative (HR<1)": "#1f77b4",
            "Zero": "#7f7f7f",
        }

    top["coef_sign"] = "Feature"
    return {"Feature": "#1f77b4"}


def _importance_full_title(title, subtitle):
    """Build importance title with optional subtitle."""
    if subtitle:
        return (f"{title}<br>"
                f"<span style='font-size:12px;font-weight:normal'>{subtitle}</span>")
    return title


def _filter_numeric_for_corr(
    X: pd.DataFrame, *, stacklevel: int = 2,
) -> pd.DataFrame:
    """Select numeric columns, warn if any were dropped, raise if none remain."""
    numeric = X.select_dtypes(include=[np.number, bool])
    n_dropped = X.shape[1] - numeric.shape[1]
    if numeric.empty:
        raise ValueError("No numeric columns found for correlation matrix.")
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} non-numeric column(s) excluded from correlation matrix.",
            InternalWarning,
            stacklevel=stacklevel,
        )
    return numeric
