"""Compute the combined per-feature summary table.

This is the compute layer for ``SurvivalLearner.feature_summary()``. The
report-side renderer (the interactive feature table in the feature-analysis
section) consumes this same assembly, so the report's values come from the
same place as the programmatic accessor. The report shows a
reordered subset (importance-backed rows, highest PFI first).

The table joins, per feature:

- univariate Cox statistics (model-free, full cohort) — ``uni_*`` columns
- multivariable coefficients with gated inference (per model, training
  split) — ``mv_*`` columns, populated per the ``hazard_ratio_valid``
  gating in ``_coefficients.py``
- permutation feature importance — ``perm_mean`` / ``perm_std``
- mean absolute SHAP — ``mean_abs_shap``
- importance significance (PIMP preferred, bootstrap fallback) —
  ``p_adjusted`` / ``sig_stars``
- the correlated-feature flag — ``correlated``
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._warnings import InternalWarning
from ..constants import P_STARS
from ..logging_utils import get_logger, log_exception

logger = get_logger("evaluation.feature_summary")

#: Column schema of ``compute_feature_summary`` /
#: ``SurvivalLearner.feature_summary``. The ``uni_*`` and ``mv_*``
#: prefixes disambiguate the univariate and multivariable statistics,
#: which otherwise share names (``p_value``, ``hazard_ratio``, ...).
FEATURE_SUMMARY_COLUMNS = [
    'model_type', 'feature',
    # Univariate Cox (model-free; repeated identically in every model block)
    'uni_p_value', 'uni_hazard_ratio', 'uni_hr_lower_95', 'uni_hr_upper_95',
    # Multivariable, per model (hazard_ratio_valid-gated: currently CoxPH)
    'mv_coefficient', 'mv_hazard_ratio', 'mv_hr_lower_95', 'mv_hr_upper_95',
    'mv_p_value',
    # Permutation feature importance
    'perm_mean', 'perm_std',
    # SHAP
    'mean_abs_shap',
    # Importance significance (PIMP or bootstrap)
    'p_adjusted', 'sig_stars',
    # Correlated-feature flag
    'correlated',
]

_UNI_RENAMES = {
    'p_value': 'uni_p_value',
    'hazard_ratio': 'uni_hazard_ratio',
    'hr_lower_95': 'uni_hr_lower_95',
    'hr_upper_95': 'uni_hr_upper_95',
}

_MV_RENAMES = {
    'coefficient': 'mv_coefficient',
    'hazard_ratio': 'mv_hazard_ratio',
    'hr_lower_95': 'mv_hr_lower_95',
    'hr_upper_95': 'mv_hr_upper_95',
    'p_value': 'mv_p_value',
}


def _empty_feature_summary():
    """Return the 0-row sentinel DataFrame (nothing to summarize)."""
    df = pd.DataFrame(columns=FEATURE_SUMMARY_COLUMNS)
    return df.astype({'correlated': 'boolean'})


def _p_to_stars(p) -> Optional[str]:
    """Map an adjusted p-value to star notation.

    Returns ``'***'`` / ``'**'`` / ``'*'`` per ``P_STARS``, ``''`` for a
    tested-but-not-significant value, and ``None`` for NaN/invalid (no
    test result for this feature).
    """
    if p is None or pd.isna(p) or not np.isfinite(p) or p < 0:
        return None
    for threshold, stars in P_STARS:
        if p < threshold:
            return stars
    return ''


def compute_mean_abs_shap(shap_values) -> Optional[pd.DataFrame]:
    """Aggregate long-format SHAP values to mean |SHAP| per feature.

    Parameters
    ----------
    shap_values : pd.DataFrame or None
        Long format with at least ``model_type``, ``feature``,
        ``shap_value`` columns (``EvaluationResult.shap_values``).

    Returns
    -------
    pd.DataFrame or None
        Columns ``[model_type, feature, mean_abs_shap]``; None when
        *shap_values* is None or empty.
    """
    if shap_values is None or len(shap_values) == 0:
        return None
    return shap_values.groupby(['model_type', 'feature'])['shap_value'].apply(
        lambda x: np.abs(x).mean()
    ).reset_index(name='mean_abs_shap')


def resolve_importance_pvalues(
    model_name: str,
    *,
    pimp_results: Optional[Dict[str, pd.DataFrame]] = None,
    bootstrap_result=None,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """Resolve importance significance p-values for one model.

    PIMP results are preferred (stronger null hypothesis — Altmann et
    al. 2010); bootstrap importance is the fallback. This is the shared
    resolver behind both ``feature_summary`` and the report's
    significance columns; it mutates no state (it logs and warns when
    the bootstrap fallback fails).

    Parameters
    ----------
    model_name : str
        Model storage key to resolve p-values for.
    pimp_results : dict, optional
        Per-model PIMP tables
        (``learner.pimp_importance_test_results``). The dict is the
        only PIMP source — every result is keyed into it.
    bootstrap_result : BootstrapResult, optional
        Bootstrap result whose ``importance_summary`` provides the
        fallback empirical p-values.

    Returns
    -------
    tuple
        ``(pvals_df, source, method)`` where *source* is ``'pimp'`` or
        ``'bootstrap'`` and *method* the p-adjustment key (e.g.
        ``'bh'``); all three None when no significance data exists.
    """
    pimp_results = pimp_results or {}
    if model_name in pimp_results:
        result = pimp_results[model_name]
        if result is not None and not result.empty:
            return result, 'pimp', result.attrs.get('p_adjust') or 'bh'

    if bootstrap_result is not None:
        importance = getattr(bootstrap_result, 'importance', None)
        if importance is not None and not importance.empty:
            try:
                result = bootstrap_result.importance_summary(
                    model_type=model_name, p_adjust='bh',
                )
                return result, 'bootstrap', 'bh'
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                # The importance data exists, so a summarization failure
                # is real data loss — log visibly (InternalWarning is
                # filtered by default) in addition to warning.
                log_exception(
                    logger, exc,
                    f"bootstrap importance summary for {model_name!r}",
                    warn_user=True,
                    user_message=(
                        f"Could not compute bootstrap importance p-values "
                        f"for {model_name!r}: {exc}. Significance columns "
                        f"will be empty for this model."
                    ),
                    category=InternalWarning,
                )

    return None, None, None


def _model_rows(df, model_name):
    """Return *df*'s rows for one model, or None when *df* is None or empty."""
    if df is None or len(df) == 0:
        return None
    return df[df['model_type'] == model_name]


def _block_features(model_name, univariate_cox, coefficients,
                    importance, mean_shap):
    """Return the ordered feature universe for one model block.

    Order: univariate features first (already p-sorted), then
    coefficient features (pipeline order), then importance features,
    then SHAP features — each contributing only names not seen yet.
    """
    features = []
    seen = set()

    def _extend(values):
        for f in values:
            if f not in seen:
                seen.add(f)
                features.append(f)

    if univariate_cox is not None and len(univariate_cox) > 0:
        _extend(univariate_cox['feature'])
    for piece in (coefficients, importance, mean_shap):
        rows = _model_rows(piece, model_name)
        if rows is not None:
            _extend(rows['feature'])
    return features


def _merge_piece(block, piece, columns):
    """Left-merge *columns* of *piece* onto *block* by feature.

    Duplicate features in *piece* keep the first row.
    """
    if piece is None or len(piece) == 0:
        return block
    keep = piece[['feature', *columns]].drop_duplicates(
        subset='feature', keep='first',
    )
    return block.merge(keep, on='feature', how='left')


def _assemble_block(
    model_name, *, univariate_cox, coefficients, importance, mean_shap,
    sig_pvals, correlated_features,
):
    """Assemble one model block of the summary (``model_type`` not yet set)."""
    features = _block_features(
        model_name, univariate_cox, coefficients, importance, mean_shap,
    )
    block = pd.DataFrame({'feature': features})

    uni = None
    if univariate_cox is not None and len(univariate_cox) > 0:
        uni = univariate_cox.rename(columns=_UNI_RENAMES)
    block = _merge_piece(block, uni, list(_UNI_RENAMES.values()))

    mv = _model_rows(coefficients, model_name)
    if mv is not None:
        mv = mv.rename(columns=_MV_RENAMES)
    block = _merge_piece(block, mv, list(_MV_RENAMES.values()))

    block = _merge_piece(
        block, _model_rows(importance, model_name), ['perm_mean', 'perm_std'],
    )
    block = _merge_piece(
        block, _model_rows(mean_shap, model_name), ['mean_abs_shap'],
    )

    block = _merge_piece(block, sig_pvals, ['p_adjusted'])
    if 'p_adjusted' not in block.columns:
        block['p_adjusted'] = np.nan
    # astype(object): keep the dtype stable whether the block holds
    # strings, None, or a mix (string inference would otherwise make
    # single-block and concatenated frames disagree).
    block['sig_stars'] = block['p_adjusted'].map(_p_to_stars).astype(object)

    if correlated_features is None:
        block['correlated'] = pd.array([pd.NA] * len(block), dtype='boolean')
    else:
        block['correlated'] = pd.array(
            [f in correlated_features for f in block['feature']],
            dtype='boolean',
        )

    # Columns absent from every source piece still need to exist.
    for col in FEATURE_SUMMARY_COLUMNS:
        if col != 'model_type' and col not in block.columns:
            block[col] = np.nan

    block = block.sort_values(
        by=['uni_p_value', 'perm_mean', 'feature'],
        ascending=[True, False, True],
        na_position='last',
        kind='stable',
    ).reset_index(drop=True)
    return block


def compute_feature_summary(
    model_names: Sequence[str],
    *,
    coefficients: Optional[pd.DataFrame] = None,
    univariate_cox: Optional[pd.DataFrame] = None,
    importance: Optional[pd.DataFrame] = None,
    shap_values: Optional[pd.DataFrame] = None,
    pimp_results: Optional[Dict[str, pd.DataFrame]] = None,
    bootstrap_result=None,
    correlated_features=None,
    correlation_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Assemble the combined per-feature summary table.

    Pure assembly over pre-computed pieces — no models are touched, so
    this is unit-testable with fabricated frames. The public entry point
    that gathers the pieces from live state is
    ``SurvivalLearner.feature_summary``.

    Parameters
    ----------
    model_names : sequence of str
        Models to emit blocks for, in order. Empty sequence emits a
        single univariate-only block with ``model_type`` NaN (when
        *univariate_cox* has rows), else the 0-row sentinel.
    coefficients : pd.DataFrame, optional
        Long-format coefficients (``SurvivalLearner.coefficients``
        shape: ``model_type`` plus the ``COEFFICIENT_COLUMNS``).
    univariate_cox : pd.DataFrame, optional
        Univariate Cox table (``SurvivalLearner.univariate_cox``).
    importance : pd.DataFrame, optional
        Permutation importance (``EvaluationResult.importance``).
    shap_values : pd.DataFrame, optional
        Long-format SHAP values (``EvaluationResult.shap_values``);
        aggregated here via :func:`compute_mean_abs_shap`.
    pimp_results, bootstrap_result
        Significance sources, resolved per model by
        :func:`resolve_importance_pvalues`.
    correlated_features : set of str, optional
        Features involved in at least one high-correlation pair. None
        means "could not be determined" — the ``correlated`` column is
        all ``pd.NA`` then.
    correlation_threshold : float, optional
        Recorded in ``attrs['correlation_threshold']`` only.

    Returns
    -------
    pd.DataFrame
        ``FEATURE_SUMMARY_COLUMNS`` schema; one block of rows per model,
        each sorted by ``uni_p_value`` (NaN last), then ``perm_mean``
        descending, then feature name. ``attrs`` carries ``significance_source`` and
        ``significance_method`` dicts (only models with significance
        data) and ``correlation_threshold``.
    """
    mean_shap = compute_mean_abs_shap(shap_values)

    sig_source: Dict[str, str] = {}
    sig_method: Dict[str, str] = {}
    blocks = []

    for name in model_names:
        sig_pvals, source, method = resolve_importance_pvalues(
            name,
            pimp_results=pimp_results,
            bootstrap_result=bootstrap_result,
        )
        if source is not None:
            sig_source[name] = source
            sig_method[name] = method
        block = _assemble_block(
            name,
            univariate_cox=univariate_cox,
            coefficients=coefficients,
            importance=importance,
            mean_shap=mean_shap,
            sig_pvals=sig_pvals,
            correlated_features=correlated_features,
        )
        block.insert(0, 'model_type', name)
        blocks.append(block)

    if not blocks:
        if univariate_cox is not None and len(univariate_cox) > 0:
            # Model-free summary: univariate stats stand on their own.
            block = _assemble_block(
                None,
                univariate_cox=univariate_cox,
                coefficients=None,
                importance=None,
                mean_shap=None,
                sig_pvals=None,
                correlated_features=correlated_features,
            )
            block.insert(0, 'model_type', np.nan)
            blocks.append(block)
        else:
            result = _empty_feature_summary()
            result.attrs['significance_source'] = {}
            result.attrs['significance_method'] = {}
            result.attrs['correlation_threshold'] = correlation_threshold
            return result

    result = pd.concat(blocks, ignore_index=True)[FEATURE_SUMMARY_COLUMNS]
    result.attrs['significance_source'] = sig_source
    result.attrs['significance_method'] = sig_method
    result.attrs['correlation_threshold'] = correlation_threshold
    return result
