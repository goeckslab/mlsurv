"""Shared univariate Cox fitting helpers.

Single source of truth for the penalizer-fallback loop, NaN casewise
deletion, and unexpected-error handling used by both
``preprocessing/selection.py`` (TopKCoxSelector ranking) and
``evaluation/importance.py`` (univariate Cox table for explain()).

Two output shapes are supported via the ``extract`` parameter:

- ``extract='pval'`` returns just the p-value. Used by selection ranking.
  ConvergenceWarnings are elevated to exceptions so the fallback loop can
  try a more aggressive strategy.
- ``extract='full'`` returns p-value plus hazard ratio, 95% CI, and
  coefficient. Used by importance for descriptive output.
  ConvergenceWarnings are accepted (results returned with a verbose note).
"""

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

from ..utils import SurvivalArray


# Sentinel for features that failed Cox fitting (sorts last, never a real p-value)
FAILED_PVAL = float('inf')

# Minimum non-NaN rows per feature for reliable univariate Cox fitting
MIN_COX_ROWS = 10


def check_feature_viable(feature_series, col_name, verbose):
    """Check whether *feature_series* has enough variance for Cox fitting.

    Parameters
    ----------
    feature_series : pandas.Series
        Single feature column to assess.
    col_name : str
        Column name, used in the skip message when *verbose* is True.
    verbose : bool
        If True, print a message when the feature is skipped.

    Returns
    -------
    str or None
        Reason string if not viable, ``None`` if OK.
    """
    if feature_series.nunique() <= 1:
        if verbose:
            print(f"  {col_name}: Skipped (constant column)")
        return "constant"
    if feature_series.std() < 1e-10:
        if verbose:
            print(f"  {col_name}: Skipped (near-zero variance)")
        return "near-zero variance"
    return None


def build_cox_strategies(penalizer):
    """Build ordered list of CoxPHFitter parameter dicts to try in fallback.

    Order: unpenalized → penalized (if ``penalizer > 0``) → reduced step size.
    """
    strategies = [dict(penalizer=0.0)]
    if penalizer > 0:
        strategies.append(dict(penalizer=penalizer))
    strategies.append(dict(penalizer=penalizer, step_size=0.25))
    return strategies


def _default_full_result(col_name):
    """Failed-fit sentinel for ``extract='full'`` callers."""
    return {
        'feature': col_name,
        'p_value': FAILED_PVAL,
        'hazard_ratio': np.nan,
        'hr_lower_95': np.nan,
        'hr_upper_95': np.nan,
        'coefficient': np.nan,
    }


def _do_cox_fit(df, col_name, penalizer, step_size, verbose, *, extract, caught):
    """Inner Cox-fit body — caller owns the warnings context."""
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    fit_kwargs = {'duration_col': 'T', 'event_col': 'E'}
    if step_size is not None:
        fit_kwargs['fit_options'] = {'step_size': step_size}
    cph.fit(df, **fit_kwargs)

    summary = cph.summary
    pval = summary.loc[col_name, 'p']
    coef = summary.loc[col_name, 'coef']

    if np.isnan(pval) or (extract == 'full' and np.isnan(coef)):
        if verbose:
            print(f"  {col_name}: NaN in results, trying next strategy")
        return None

    if extract == 'pval':
        if abs(coef) > 20 and verbose:
            print(f"  {col_name}: Extreme coefficient ({coef:.2f}), "
                  "may indicate separation")
        return {'p_value': pval}

    if caught and verbose and any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    ):
        print(f"  {col_name}: Converged with warning (results accepted)")
    return {
        'feature': col_name,
        'p_value': pval,
        'hazard_ratio': summary.loc[col_name, 'exp(coef)'],
        'hr_lower_95': summary.loc[col_name, 'exp(coef) lower 95%'],
        'hr_upper_95': summary.loc[col_name, 'exp(coef) upper 95%'],
        'coefficient': coef,
    }


def try_cox_fit(df, col_name, *, penalizer=0.0, step_size=None, verbose=False,
                extract='pval'):
    """Attempt a single Cox fit with the given parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``'T'``, ``'E'``, and the feature column.
    col_name : str
        Feature column name (used for messages and summary lookups).
    penalizer : float
        L2 regularization passed to ``CoxPHFitter``.
    step_size : float or None
        Optimizer step size; passed via ``fit_options`` when not None.
    verbose : bool
        Print debug info on expected failures.
    extract : {'pval', 'full'}
        ``'pval'`` returns ``{'p_value': float}`` on success and elevates
        ConvergenceWarning to a retry trigger. ``'full'`` returns the full
        stats dict and accepts ConvergenceWarning (fit accepted, warning is
        informational only).

    Returns
    -------
    dict or None or str
        On success: ``{'p_value': float}`` for ``extract='pval'``, or a
        dict with ``feature``, ``p_value``, ``hazard_ratio``, ``hr_lower_95``,
        ``hr_upper_95``, ``coefficient`` for ``extract='full'``.
        ``None`` on expected failure (convergence/numerical/NaN);
        error-message string for unexpected errors (callers surface these in
        the main process — loky workers can't deliver warnings).
    """
    try:
        if extract == 'pval':
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=ConvergenceWarning)
                return _do_cox_fit(df, col_name, penalizer, step_size,
                                   verbose, extract='pval', caught=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            return _do_cox_fit(df, col_name, penalizer, step_size,
                               verbose, extract='full', caught=caught)
    except (ConvergenceWarning, ValueError, TypeError,
            np.linalg.LinAlgError, KeyError) as e:
        # lifelines raises TypeError on NaN/degenerate data.
        if verbose:
            print(f"  {col_name}: {type(e).__name__} - {str(e)[:50]}...")
        return None
    except Exception as e:
        if isinstance(e, (MemoryError, RecursionError)):
            raise
        return (
            f"Unexpected error fitting Cox model for '{col_name}': "
            f"{type(e).__name__}: {str(e)}"
        )


def fit_univariate_cox(feature_series, y: SurvivalArray, col_name, *,
                       penalizer=0.01, verbose=False, extract='pval',
                       logger=None):
    """Fit a univariate Cox model with NaN casewise deletion + penalizer fallback.

    Parameters
    ----------
    feature_series : pd.Series
        Single feature column (may contain NaN).
    y : SurvivalArray
        Survival outcome.
    col_name : str
        Feature name.
    penalizer : float
        L2 regularization for fallback fits.
    verbose : bool
        Print per-feature progress (NaN drops, retries, failures).
    extract : {'pval', 'full'}
        Output shape — see ``try_cox_fit``.
    logger : logging.Logger or None
        When provided, NaN-related messages also route through
        ``logger.debug`` / ``logger.warning`` (used by ``importance.py``).

    Returns
    -------
    dict
        Always includes ``feature``, ``p_value``, ``status``, ``error_detail``.
        ``status`` is one of ``'ok'``, ``'skipped'`` (constant/near-zero
        variance/all NaN), ``'failed'`` (all strategies exhausted),
        ``'error'`` (unexpected exception). When ``extract='full'``, also
        includes ``hazard_ratio``, ``hr_lower_95``, ``hr_upper_95``,
        ``coefficient``, and ``n_missing`` (only when any rows were dropped).
    """
    if extract == 'full':
        base = _default_full_result(col_name)
    else:
        base = {'feature': col_name, 'p_value': FAILED_PVAL}
    base['status'] = 'skipped'
    base['error_detail'] = None

    mask = feature_series.notna().values
    n_total = len(feature_series)
    n_dropped = int((~mask).sum())
    n_valid = n_total - n_dropped
    if extract == 'full' and n_dropped > 0:
        base['n_missing'] = n_dropped

    if n_valid == 0:
        if verbose:
            print(f"  {col_name}: Skipped (all NaN)")
        if logger is not None:
            logger.warning("%s: all values are NaN — skipping", col_name)
        return base

    if n_dropped > 0:
        feature_series = feature_series[mask].reset_index(drop=True)
        y = y[mask]
        if verbose:
            print(f"  {col_name}: Dropped {n_dropped}/{n_total} "
                  f"NaN rows (casewise deletion)")
        if logger is not None:
            pct = n_dropped / n_total * 100
            logger.debug("%s: dropping %d/%d NaN rows (%.0f%%)",
                         col_name, n_dropped, n_total, pct)
            if pct > 50:
                logger.warning(
                    "%s: %.0f%% missing values — univariate Cox fit uses "
                    "only %d of %d observations",
                    col_name, pct, n_valid, n_total,
                )

    if n_valid < MIN_COX_ROWS and verbose:
        print(f"  {col_name}: Only {n_valid}/{n_total} non-NaN rows — "
              f"Cox p-value may be unreliable")

    if check_feature_viable(feature_series, col_name, verbose) is not None:
        return base

    df = pd.DataFrame({
        'T': y['time'],
        'E': y['event'].astype(bool),
        col_name: feature_series.values,
    })

    error_detail = None
    for i, params in enumerate(build_cox_strategies(penalizer)):
        if i > 0 and verbose:
            desc = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"  {col_name}: Retrying with {desc}")
        result = try_cox_fit(df, col_name, verbose=verbose, extract=extract,
                             **params)
        if isinstance(result, dict):
            out = dict(base)
            out.update(result)
            out['status'] = 'ok'
            out['error_detail'] = None
            return out
        if isinstance(result, str):
            error_detail = result

    if verbose:
        print(f"  {col_name}: All fit strategies failed, assigning p=inf")
    base['status'] = 'error' if error_detail is not None else 'failed'
    base['error_detail'] = error_detail
    return base
