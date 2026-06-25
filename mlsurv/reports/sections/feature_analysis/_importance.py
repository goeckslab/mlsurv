"""Importance row building, p-values, and significance narratives.

Functions build table rows and significance summaries for the feature
importance section of the report.
"""

from typing import Optional

import numpy as np
import pandas as pd

from mlsurv.reports._constants import (
    P_THRESHOLD_EXCELLENT,
    P_THRESHOLD_GOOD,
    P_THRESHOLD_SIGNIFICANT,
    format_pvalue,
)
from mlsurv.viz._palettes import get_model_display_name

from ._helpers import (
    _derive_effect_from_hr,
    _format_adjusted_pvalue,
    _format_importance_value,
    _format_significance,
    _md_escape,
)

_P_ADJUST_LABELS = {
    "bh": ("BH", "Benjamini-Hochberg", "FDR control"),
    "holm": ("Holm", "Holm", "FWER control"),
    "bonferroni": ("Bonferroni", "Bonferroni", "FWER control"),
    "none": ("unadjusted", "unadjusted", "no correction"),
}


def sig_footnote(pvalue_source: Optional[str], pvalue_method: str) -> str:
    """Return the appropriate significance footnote based on p-value source and method."""
    short, full, control = _P_ADJUST_LABELS.get(
        pvalue_method, ("adjusted", "adjusted", "multiple testing")
    )
    if pvalue_source == "pimp":
        return (
            f"*Sig.: PIMP importance significance ({short}-adjusted; Altmann et al. 2010). "
            f"\\*\\*\\* p < {P_THRESHOLD_EXCELLENT}, \\*\\* p < {P_THRESHOLD_GOOD}, \\* p < {P_THRESHOLD_SIGNIFICANT}. "
            f"p (adj.) shows the {full} adjusted p-value ({control}). "
            f"Tests whether observed importance exceeds a null distribution "
            f"built from models trained on permuted response data.*"
        )
    return (
        f"*Sig.: bootstrap importance significance ({short}-adjusted). "
        f"\\*\\*\\* p < {P_THRESHOLD_EXCELLENT}, \\*\\* p < {P_THRESHOLD_GOOD}, \\* p < {P_THRESHOLD_SIGNIFICANT}. "
        f"p (adj.) shows the {full} adjusted p-value ({control}). "
        f"Tests whether permutation importance > 0 across bootstrap iterations.*"
    )


def split_summary_blocks(summary: Optional[pd.DataFrame]) -> dict:
    """Split a feature summary into per-model blocks.

    Returns ``{model_type: block}`` in block order; ``{}`` for
    None/empty summaries. Rows with NaN ``model_type`` (the no-models
    univariate-only form) are dropped by the groupby.
    """
    if summary is None or summary.empty:
        return {}
    # Not dict(groupby): dict() would treat the GroupBy as a mapping and
    # call its .keys -- which is the grouping key string, not a method.
    return {
        name: block
        for name, block in summary.groupby('model_type', sort=False)
    }


def per_model_coef_blocks(summary: Optional[pd.DataFrame]) -> dict:
    """Per-model summary blocks that carry fitted coefficients.

    Replaces the old report-side coefficient extraction: the ``mv_*``
    columns of ``learner.feature_summary()`` already hold the gated
    coefficient table, so the report only selects which models have
    any coefficients at all.
    """
    return {
        name: block
        for name, block in split_summary_blocks(summary).items()
        if block['mv_coefficient'].notna().any()
    }


def append_coef_columns(
    row_data: dict, feature: str, display: str, coef_block: pd.DataFrame,
) -> None:
    """Append Coef., HR (mv), and p (mv) columns for *display* model.

    *coef_block* is that model's feature-summary block (the ``mv_*``
    columns). ``HR (mv)`` shows the hazard ratio with its 95% CI inline
    when available (``"1.234 (0.987-1.543)"``). The ``p (mv)`` column is
    added only when the block has at least one finite ``mv_p_value`` —
    in practice a ``hazard_ratio_valid`` model (currently CoxPH) with a
    non-singular Hessian. The column set is per-model, so it stays
    identical across rows. NaN values (e.g. the gated HR for
    CoxNet/FSSVM, or selector-dropped features) render as ``"-"``.
    """
    has_inference = np.isfinite(
        coef_block['mv_p_value'].astype(float)
    ).any()
    feat_coef = coef_block[coef_block['feature'] == feature]
    if len(feat_coef) == 0:
        row_data[f'Coef. ({display})'] = "-"
        row_data[f'HR (mv) ({display})'] = "-"
        if has_inference:
            row_data[f'p (mv) ({display})'] = "-"
        return

    coef_val = feat_coef['mv_coefficient'].values[0]
    row_data[f'Coef. ({display})'] = (
        f"{coef_val:.3f}" if np.isfinite(coef_val) else "-"
    )
    row_data[f'HR (mv) ({display})'] = _format_hr_with_ci(feat_coef)
    if has_inference:
        p_val = feat_coef['mv_p_value'].values[0]
        row_data[f'p (mv) ({display})'] = (
            format_pvalue(p_val) if np.isfinite(p_val) else "-"
        )


def _format_hr_with_ci(feat_coef: pd.DataFrame) -> str:
    """Format a one-row HR as ``"hr (lower-upper)"``, ``"hr"``, or ``"-"``."""
    hr = feat_coef['mv_hazard_ratio'].values[0]
    if not np.isfinite(hr):
        return "-"
    lo = feat_coef['mv_hr_lower_95'].values[0]
    hi = feat_coef['mv_hr_upper_95'].values[0]
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{hr:.3f} ({lo:.3f}-{hi:.3f})"
    return f"{hr:.3f}"


def _compute_n_obs(feature, univar_cox, total_samples):
    """Return ``(n_obs_str, show)`` for the *n* column.

    *show* is ``True`` when at least one feature in *univar_cox* has
    ``n_missing > 0``, indicating per-feature sample sizes vary.
    """
    if univar_cox is None or 'n_missing' not in univar_cox.columns:
        return None, False
    if total_samples is None:
        return None, False
    if not univar_cox['n_missing'].gt(0).any():
        return None, False
    feat_row = univar_cox[univar_cox['feature'] == feature]
    if len(feat_row) == 0:
        return str(total_samples), True
    n_miss_val = feat_row['n_missing'].values[0]
    if not np.isfinite(n_miss_val):
        return str(total_samples), True
    n_miss = int(n_miss_val)
    return str(total_samples - n_miss), True


def _lookup_univar_cox(feature, univar_cox):
    """Return ``(pval, hr)`` from univariate Cox results, both NaN if missing."""
    pval = np.nan
    hr = np.nan
    if univar_cox is not None:
        feat_cox = univar_cox[univar_cox['feature'] == feature]
        if len(feat_cox) > 0:
            if 'p_value' in feat_cox.columns:
                pval = feat_cox['p_value'].values[0]
            if 'hazard_ratio' in feat_cox.columns:
                hr = feat_cox['hazard_ratio'].values[0]
    return pval, hr


def _format_univar_fields(pval, hr):
    """Return ``(pval_str, hr_str, effect)`` formatted for table display."""
    pval_str = format_pvalue(pval) if np.isfinite(pval) else "N/A"
    hr_str = f"{hr:.3f}" if np.isfinite(hr) else "N/A"
    effect = _derive_effect_from_hr(hr)
    return pval_str, hr_str, effect


def _is_flagged_correlated(summary_row) -> bool:
    """True when the summary row's ``correlated`` flag is set (NA-safe)."""
    flag = summary_row.get('correlated')
    return pd.notna(flag) and bool(flag)


def build_feature_row(
    summary_row, model_name, model_c_index,
    has_sig=False, per_model_coefs=None,
    univar_cox=None, total_samples=None, show_shap=False,
):
    """Build a table row for a single feature in the single-model case.

    *summary_row* is one row of the model's ``learner.feature_summary()``
    block; the data columns come straight from it, while this function
    only formats them for display.

    When *per_model_coefs* is provided (per-model summary blocks from
    :func:`per_model_coef_blocks`), appends Coef. and HR (mv) columns
    for each model with fitted coefficients.

    When *univar_cox* has an ``n_missing`` column with any non-zero
    values, an ``n`` column is added showing the per-feature sample
    size used for univariate Cox fitting.
    """
    feature = summary_row['feature']
    imp_str = _format_importance_value(summary_row['perm_mean'], model_c_index)
    pval_str, hr_str, effect = _format_univar_fields(
        summary_row['uni_p_value'], summary_row['uni_hazard_ratio'],
    )

    feature_label = feature
    if _is_flagged_correlated(summary_row):
        feature_label += " †"

    row_data = {
        'Feature': feature_label,
    }

    # Per-feature sample size (only when some features have missing data)
    n_obs_str, show_n = _compute_n_obs(feature, univar_cox, total_samples)
    if show_n:
        row_data['n'] = n_obs_str

    row_data['p-value'] = pval_str
    row_data['HR'] = hr_str
    row_data['Effect'] = effect
    row_data[f'PFI ({get_model_display_name(model_name)})'] = imp_str

    if has_sig:
        p_adj = summary_row['p_adjusted']
        row_data['p (adj.)'] = _format_adjusted_pvalue(p_adj)
        row_data['Sig.'] = _format_significance(p_adj)

    if per_model_coefs:
        for m_name, coef_block in per_model_coefs.items():
            append_coef_columns(
                row_data, feature,
                get_model_display_name(m_name), coef_block,
            )

    if show_shap:
        shap_val = summary_row['mean_abs_shap']
        row_data['Mean |SHAP|'] = (
            f"{shap_val:.4f}" if pd.notna(shap_val) else "N/A"
        )

    return row_data


def get_sorted_top_features(models, importance_df):
    """Get all features across all models, sorted by average importance."""
    avg_imp = importance_df.groupby('feature')['perm_mean'].mean()
    return avg_imp.sort_values(ascending=False).index.tolist()


def _summary_feature_row(block, feat):
    """Return *feat*'s row from a summary block, or None."""
    if block is None:
        return None
    match = block[block['feature'] == feat]
    return match.iloc[0] if len(match) > 0 else None


def build_multi_model_feature_row(
    feat, models, model_blocks, baseline_c_index,
    sig_models=None, per_model_coefs=None,
    univar_cox=None, total_samples=None,
):
    """Build a table row for a feature in the multi-model case.

    *model_blocks* maps model name to its ``learner.feature_summary()``
    block (from :func:`split_summary_blocks`); the data columns come
    straight from it, while this function only formats them.

    *sig_models* is the set of models with significance data — None
    omits the p (adj.)/Sig. columns entirely; models outside the set
    get blank cells (so the column set stays identical across rows).

    When *per_model_coefs* is provided (per-model summary blocks from
    :func:`per_model_coef_blocks`), appends Coef. and HR (mv) columns
    for each model with fitted coefficients.

    When *univar_cox* has an ``n_missing`` column with any non-zero
    values, an ``n`` column is added showing the per-feature sample
    size used for univariate Cox fitting.
    """
    # Univariate columns are identical across blocks; take the first
    # block that has this feature.
    uni_row = None
    for model_name in models:
        uni_row = _summary_feature_row(model_blocks.get(model_name), feat)
        if uni_row is not None:
            break

    feature_label = feat
    if uni_row is not None and _is_flagged_correlated(uni_row):
        feature_label += " †"
    row = {'Feature': feature_label}

    # Per-feature sample size
    n_obs_str, show_n = _compute_n_obs(feat, univar_cox, total_samples)
    if show_n:
        row['n'] = n_obs_str

    uni_p = uni_row['uni_p_value'] if uni_row is not None else np.nan
    uni_hr = uni_row['uni_hazard_ratio'] if uni_row is not None else np.nan
    pval_str, hr_str, effect = _format_univar_fields(uni_p, uni_hr)
    row['p-value'] = pval_str
    row['HR'] = hr_str
    row['Effect'] = effect

    for model_name in models:
        display = get_model_display_name(model_name)
        feat_row = _summary_feature_row(model_blocks.get(model_name), feat)
        col_name = f"PFI ({display})"
        if feat_row is not None and pd.notna(feat_row['perm_mean']):
            model_c_index = baseline_c_index.get(model_name)
            row[col_name] = _format_importance_value(
                feat_row['perm_mean'], model_c_index,
            )
        else:
            row[col_name] = "-"

        if sig_models is not None:
            padj_col = f"p (adj.) ({display})"
            sig_col = f"Sig. ({display})"
            if model_name in sig_models and feat_row is not None:
                p_adj = feat_row['p_adjusted']
                row[padj_col] = _format_adjusted_pvalue(p_adj)
                row[sig_col] = _format_significance(p_adj)
            else:
                row[padj_col] = ""
                row[sig_col] = ""

        if per_model_coefs and model_name in per_model_coefs:
            append_coef_columns(
                row, feat, display, per_model_coefs[model_name],
            )

    return row


def _section_bootstrap_result(section):
    """Pick the bootstrap result the section should use for p-values.

    Prefers the bootstrap attached to ``eval_result``; falls back to the
    learner's standalone combined result when the attached one has no
    importance data.
    """
    bootstrap = getattr(section.eval_result, 'bootstrap', None)
    _imp = getattr(bootstrap, 'importance', None)
    if bootstrap is None or _imp is None or _imp.empty:
        # getattr: duck-typed learners (test mocks) may lack the property.
        standalone = getattr(section.learner, 'bootstrap_result', None)
        if standalone is not None:
            bootstrap = standalone
    return bootstrap


def get_importance_pvalues(section, model_type: str) -> Optional[pd.DataFrame]:
    """Get importance p-values for a model (PIMP preferred, bootstrap fallback).

    Thin wrapper over the shared
    :func:`mlsurv.evaluation.resolve_importance_pvalues` resolver — the
    same one ``learner.feature_summary()`` uses, so the report's
    significance values come from the same place as the API's. Sets
    ``section._pvalue_source`` and ``section._pvalue_method`` when
    resolution succeeds (the last successful model wins — the footnote
    describes that model's source/method).
    """
    from mlsurv.evaluation._feature_summary import resolve_importance_pvalues

    result, source, method = resolve_importance_pvalues(
        model_type,
        pimp_results=section.learner.pimp_importance_test_results,
        bootstrap_result=_section_bootstrap_result(section),
    )
    if source is not None:
        section._pvalue_source = source
        section._pvalue_method = method
    return result


def sig_column_intro(pvalue_source: Optional[str], pvalue_method: str) -> str:
    """Return the Sig. column intro text, adaptive to source and method."""
    short = _P_ADJUST_LABELS.get(pvalue_method, ("adjusted",))[0]
    if pvalue_source == "pimp":
        return (
            f"The **Sig.** column shows PIMP significance (Altmann et al. 2010), "
            f"testing each feature against a null distribution from permuted-response models "
            f"({short}-adjusted for multiple comparisons)."
        )
    return (
        f"When bootstrap results include importance data, the **Sig.** column indicates statistical "
        f"significance (empirical one-tailed test, {short}-adjusted for multiple comparisons)."
    )


def generate_importance_significance_narrative(
    section, models, imp_pvals_by_model,
) -> str:
    """Generate narrative about importance significance (PIMP or bootstrap)."""
    if section._pvalue_source == "pimp":
        lines = [
            "**Importance Significance (PIMP):** The Sig. column tests whether each "
            "feature's observed permutation importance exceeds what would be expected "
            "under the null hypothesis that the feature has no association with the "
            "response (Altmann et al. 2010). Null models are trained on permuted "
            "response data to build a proper null distribution. "
            "This is a *different* question from the univariate Cox p-value column, "
            "which tests feature-survival association independently of the model."
        ]
    else:
        lines = [
            "**Importance Significance (Bootstrap):** The Sig. column tests whether each "
            "feature's permutation importance is significantly greater than zero across bootstrap "
            "iterations. This is a *different* question from the univariate Cox p-value column, "
            "which tests feature-survival association independently of the model."
        ]

    for model_name in models:
        if model_name not in imp_pvals_by_model:
            continue
        pvals = imp_pvals_by_model[model_name]
        n_total = len(pvals)
        sig_mask = pvals['p_adjusted'] < P_THRESHOLD_SIGNIFICANT
        n_sig = sig_mask.sum()
        display_name = get_model_display_name(model_name)
        if n_sig > 0:
            sig_features = pvals.loc[sig_mask, 'feature'].tolist()
            feat_list = ', '.join(f'*{_md_escape(f)}*' for f in sig_features[:5])
            suffix = f', and {n_sig - 5} more' if n_sig > 5 else ''
            lines.append(
                f"**{display_name}:** {n_sig} of {n_total} features show statistically "
                f"significant importance: {feat_list}{suffix}."
            )
        else:
            lines.append(
                f"**{display_name}:** No features reached statistical significance "
                f"(p_adjusted < {P_THRESHOLD_SIGNIFICANT}) out of {n_total} tested."
            )

    return '\n\n'.join(lines)
