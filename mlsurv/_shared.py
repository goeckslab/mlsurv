"""Shared utilities to reduce cross-module complexity.

Consolidates recurring patterns (exception guards, subpopulation iteration,
display-name resolution, model-column tidying, verbose-print guards)
into tested, single-source helpers.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from ._warnings import InternalWarning, SuggestionWarning
from .constants import LOWER_IS_BETTER
from .logging_utils import get_logger
from .utils import SurvivalArray

logger = get_logger("shared")


# ---------------------------------------------------------------------------
# safe_call — exception guard with re-raise for critical errors
# ---------------------------------------------------------------------------

def safe_call(fn, *args, default=np.nan, reraise=(MemoryError, RecursionError),
              context="", log=None, warn_user=False, warn_category=SuggestionWarning,
              **kwargs):
    """Call *fn*, re-raise critical errors, log + return *default* for others.

    Parameters
    ----------
    fn : callable
        Function to call.
    *args
        Positional arguments forwarded to *fn*.
    default
        Value returned when *fn* raises a non-critical exception.
    reraise : tuple of type
        Exception types that are always re-raised.
    context : str
        Free-form label included in log/warning messages.
    log : logging.Logger, optional
        Logger instance.  When ``None``, uses the module-level logger.
    warn_user : bool
        If ``True``, also emit ``warnings.warn``.
    warn_category : type, default SuggestionWarning
        Warning category for the ``warnings.warn`` call.
    **kwargs
        Keyword arguments forwarded to *fn*.

    Returns
    -------
    object
        Return value of *fn*, or *default* on failure.

    Notes
    -----
    ``**kwargs`` are forwarded to *fn*.  Do not pass ``default``,
    ``reraise``, ``context``, ``log``, ``warn_user``, or
    ``warn_category`` as keyword arguments intended for *fn* — they
    will be consumed by ``safe_call`` instead.  Use a zero-arg closure
    when *fn* needs keyword arguments with colliding names.
    """
    _log = log or logger
    try:
        return fn(*args, **kwargs)
    except reraise:
        raise
    except Exception as exc:
        _log.warning("%s failed: %s", context or fn.__name__, exc)
        if warn_user:
            warnings.warn(
                f"{context or fn.__name__} failed: {exc}",
                warn_category,
                stacklevel=3,
            )
        return default


# ---------------------------------------------------------------------------
# iter_subpopulations — unified subpopulation loop
# ---------------------------------------------------------------------------

def iter_subpopulations(
    subpopulations,
    y: SurvivalArray,
    min_events: int = 2,
) -> Iterator[Tuple[str, object, np.ndarray, SurvivalArray]]:
    """Yield ``(group_var, label, mask, y_sub)`` for each subpopulation level.

    Parameters
    ----------
    subpopulations : dict[str, ndarray] or ndarray
        Dict of ``{group_var: labels_array}``, or a single flat labels
        array (wrapped internally as ``{"_bootstrap": array}``).
    y : SurvivalArray
        Survival outcomes aligned with the labels.
    min_events : int
        Skip subpopulation levels with fewer uncensored events.

    Yields
    ------
    group_var : str
        Name of the grouping variable.
    label : object
        The subpopulation level value.
    mask : ndarray of bool
        Boolean mask selecting this subpopulation from the full arrays.
    y_sub : SurvivalArray
        Sliced survival outcomes for this subpopulation.
    """
    if not isinstance(subpopulations, dict):
        subpopulations = {"_bootstrap": np.asarray(subpopulations)}

    events = np.asarray(y["event"])

    for group_var, labels in subpopulations.items():
        labels = np.asarray(labels)
        unique_labels = sorted(set(labels[~pd.isna(labels)]))
        for label in unique_labels:
            mask = labels == label
            n_events = int(events[mask].sum())
            if n_events < min_events:
                continue
            yield group_var, label, mask, y[mask]


# ---------------------------------------------------------------------------
# try_verbose_print — guard terminal/pipe errors on print calls
# ---------------------------------------------------------------------------

def try_verbose_print(fn, *args, context="Summary", warn=True,
                      detail="", **kwargs):
    """Call *fn*, catching terminal/pipe errors gracefully.

    Parameters
    ----------
    fn : callable
        Printing function to call.
    *args
        Positional arguments forwarded to *fn*.
    context : str
        Label for the warning message (e.g. ``"Evaluate summary"``).
    warn : bool
        If ``True``, emit ``warnings.warn`` on failure.  If ``False``,
        silently swallow the error (only ``logger.debug``).
    detail : str
        Extra reassurance appended to the warning (e.g.
        ``"All configuration is active."``).
    **kwargs
        Keyword arguments forwarded to *fn*.

    Notes
    -----
    ``stacklevel=3`` assumes the caller is two frames from user code
    (user → public method → ``try_verbose_print``).
    """
    try:
        fn(*args, **kwargs)
    except (BrokenPipeError, UnicodeEncodeError, OSError) as exc:
        logger.debug("Failed to print %s: %s", context, exc)
        if warn:
            msg = (
                f"{context} could not be printed (terminal/pipe error)."
            )
            if detail:
                msg = f"{msg} {detail}"
            warnings.warn(msg, InternalWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# tidy_model_col — rename model_type → model for return_data
# ---------------------------------------------------------------------------

def tidy_model_col(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ``model_type`` → ``model`` for user-facing ``return_data`` output."""
    if "model_type" in df.columns and "model" not in df.columns:
        return df.rename(columns={"model_type": "model"})
    return df


# ---------------------------------------------------------------------------
# display_name — resolve model key to display name
# ---------------------------------------------------------------------------

def display_name(key: str) -> str:
    """Resolve a model key to its human-readable display name.

    Convenience wrapper around the two-step
    ``resolve_display_name(key, get_model_display_names())`` pattern.
    Lazy-imports to avoid circular dependencies.
    """
    from .models import get_model_display_names
    from .results.printing import resolve_display_name

    return resolve_display_name(key, get_model_display_names())


# ---------------------------------------------------------------------------
# percentile_ci — bootstrap percentile confidence interval
# ---------------------------------------------------------------------------

def percentile_ci(values, ci_level=0.95, *, axis=0):
    """Compute percentile confidence interval from bootstrap samples.

    Parameters
    ----------
    values : array-like
        Bootstrap distribution.  Axis 0 is bootstrap iterations.
    ci_level : float
        Confidence level, e.g. 0.95 for 95 % CI.
    axis : int or None
        Axis along which to compute percentiles (default 0).

    Returns
    -------
    tuple of (ndarray or float, ndarray or float)
        ``(lower, upper)`` bounds.
    """
    alpha = (1 - ci_level) / 2
    lower = np.nanpercentile(values, 100 * alpha, axis=axis)
    upper = np.nanpercentile(values, 100 * (1 - alpha), axis=axis)
    return lower, upper


# ---------------------------------------------------------------------------
# cv_sorted_model_keys — best-first ordering (cv_metric_mean is
# pre-normalized to higher-is-better; sort is unconditionally descending)
# ---------------------------------------------------------------------------

def cv_sorted_model_keys(model_results, model_keys=None):
    """Return model keys ordered best-first by ``cv_metric_mean``.

    ``cv_metric_mean`` is always stored in higher-is-better space:
    ``score_survival_model`` negates lower-is-better scorers (e.g.
    integrated Brier returns ``-ibs``; see
    ``mlsurv/evaluation/metrics.py::_score_brier``) before fold scores
    are aggregated. Sorting descending is therefore best-first
    regardless of the configured scoring metric — the
    ``lower_is_better`` flag in ``METRIC_REGISTRY`` describes the
    raw-metric direction shown in the table cells, not the
    ``cv_metric_mean`` direction.

    Models with missing or non-finite ``cv_metric_mean`` are placed
    last. Stable on ties.

    Parameters
    ----------
    model_results : dict[str, ModelResult]
        Trained model results keyed by model name.
    model_keys : list[str], optional
        Subset / ordering of keys to sort. Defaults to
        ``list(model_results)``.

    Returns
    -------
    list of str
        Model keys ordered best-first by ``cv_metric_mean``. Models with
        missing or non-finite scores are placed last; stable on ties.
    """
    keys = list(model_results) if model_keys is None else list(model_keys)

    def sort_key(idx_name):
        idx, name = idx_name
        mr = model_results.get(name)
        cv = getattr(mr, "cv_metric_mean", None) if mr is not None else None
        if cv is None or not np.isfinite(cv):
            return (1, idx, 0.0)
        return (0, -cv, idx)

    return [name for _, name in sorted(enumerate(keys), key=sort_key)]


def _distinct_scoring_metrics(real):
    """Distinct non-None ``scoring_metric`` values across *real* models."""
    return {
        getattr(mr, "scoring_metric", None)
        for mr in real.values()
        if getattr(mr, "scoring_metric", None) is not None
    }


def _harrell_cv_mean(mr):
    """Harrell C CV mean read directly from ``cv_metrics['harrell']['mean']``.

    Returns a finite float, or ``None`` when ``cv_metrics`` is absent or has
    no finite Harrell entry. Deliberately bypasses
    ``ModelResult.get_metric('harrell', ...)``: that accessor falls back to
    ``cv_metric_mean`` when ``cv_metrics`` is absent, which mislabels a
    negated lower-is-better primary score (e.g. ``-IBS``) as Harrell C. Here
    a model lacking a true Harrell CV mean must be skipped, not given a
    wrong-scale value.
    """
    cv_metrics = getattr(mr, "cv_metrics", None)
    if not cv_metrics:
        return None
    entry = cv_metrics.get("harrell")
    if not isinstance(entry, dict):
        return None
    val = entry.get("mean")
    # bool is a subclass of int — exclude it so a stray True/False is not
    # coerced to a 1.0/0.0 Harrell mean.
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val) if np.isfinite(val) else None


def _best_by_harrell(real):
    """Best real-model key by Harrell-C CV mean (best-first).

    Tie/missing behavior mirrors :func:`cv_sorted_model_keys`: enumerate-index
    tie-break, models without a finite Harrell mean pushed last.

    Returns
    -------
    tuple[str or None, list[str]]
        ``(key, skipped)`` where *key* is the winning name, or ``None`` when
        no real model has a finite Harrell mean, and *skipped* lists the names
        dropped for lacking a finite Harrell mean.
    """
    keys = list(real)
    harrell = {name: _harrell_cv_mean(real[name]) for name in keys}
    skipped = [name for name in keys if harrell[name] is None]

    def sort_key(idx_name):
        idx, name = idx_name
        h = harrell[name]
        if h is None:
            return (1, idx, 0.0)
        return (0, -h, idx)

    ranked = [name for _, name in sorted(enumerate(keys), key=sort_key)]
    best = ranked[0] if ranked and harrell[ranked[0]] is not None else None
    return best, skipped


def best_model_key(model_results):
    """Return the name of the best non-synthetic model, or ``None``.

    Filters out synthetic models, then picks the top key by
    ``cv_metric_mean`` via :func:`cv_sorted_model_keys` (always
    higher-is-better space, so best-first). Single source of truth for
    both the ``best_model`` property and the ``models='best'`` plot
    sentinel, so the two can never drift.

    When the real models were tuned on **different** scoring metrics, their
    stored ``cv_metric_mean`` values live on different scales (lower-is-better
    primaries are negated), so comparing their max is meaningless. In that
    case this emits a :class:`SuggestionWarning` and ranks by the common
    Harrell C-index instead (``cv_metrics['harrell']['mean']``, computed for
    every model on every fold). Models lacking a Harrell CV mean are skipped;
    if none expose one, it falls back to the raw ordering so a real model is
    always returned.

    Parameters
    ----------
    model_results : dict[str, ModelResult]
        Trained model results keyed by model name.

    Returns
    -------
    str or None
        Name of the best non-synthetic model, or ``None`` when no
        non-synthetic models are present.
    """
    real = {
        name: mr for name, mr in model_results.items()
        if not getattr(mr, "is_synthetic", False)
    }
    if not real:
        return None

    metrics = _distinct_scoring_metrics(real)
    if len(metrics) <= 1:
        # At most one distinct *known* scoring metric. Models with an unknown
        # (None) scoring_metric — e.g. loaded from an older artifact — are
        # assumed to share that one scale, so cv_metric_mean stays comparable.
        return cv_sorted_model_keys(real)[0]

    # Heterogeneous scoring: stored cv_metric_mean values live on different
    # scales (lower-is-better primaries are negated). Comparing their max is
    # meaningless, so rank by the common Harrell C instead. Compute the Harrell
    # ranking first so the user-facing message states what actually happened
    # rather than promising a Harrell ranking that may not be possible.
    best, skipped = _best_by_harrell(real)

    if best is not None:
        warnings.warn(
            "best_model is comparing models tuned on different scoring metrics "
            f"({', '.join(sorted(metrics))}); their cross-validated scores live "
            "on different scales. Ranking by the common Harrell C-index instead. "
            "Re-tune with a single scoring metric for a like-for-like comparison.",
            SuggestionWarning,
            stacklevel=2,
        )
        if skipped:
            warnings.warn(
                "Excluded from the Harrell-C fallback ranking (no Harrell CV "
                f"mean available): {', '.join(sorted(skipped))}.",
                InternalWarning,
                stacklevel=2,
            )
        return best

    # No real model exposes a finite Harrell mean, so a Harrell ranking is not
    # possible. Fall back to the raw cv_metric_mean ordering — a comparison
    # across different scales, so the chosen best_model may not be a like-for-like
    # winner. User-visible (SuggestionWarning) because the result should not be
    # trusted as a fair comparison. Never return None when real models exist
    # (callers assume a name).
    warnings.warn(
        "best_model could not rank by Harrell C (no real model exposes a "
        "Harrell CV mean), so it fell back to the raw cross-validated ordering "
        "across different scoring metrics. The chosen model may not be a "
        "like-for-like winner; re-tune with a single scoring metric.",
        SuggestionWarning,
        stacklevel=2,
    )
    return cv_sorted_model_keys(real)[0]


# ---------------------------------------------------------------------------
# _display_cv_score — un-negate cv_metric_mean for user-facing display
# ---------------------------------------------------------------------------

def _display_cv_score(cv_metric_mean, scoring_metric):
    """Convert ``cv_metric_mean`` to its raw, user-facing direction.

    ``cv_metric_mean`` is stored in higher-is-better space —
    lower-is-better scorers (e.g. integrated Brier) are negated by
    ``score_survival_model`` before aggregation. For display next to
    a metric label like "Integrated Brier Score", negate back so the
    user sees the raw IBS rather than ``-IBS``.
    """
    if cv_metric_mean is None or not np.isfinite(cv_metric_mean):
        return cv_metric_mean
    return -cv_metric_mean if scoring_metric in LOWER_IS_BETTER else cv_metric_mean


# ---------------------------------------------------------------------------
# nan_summary — NaN sentinel dict for bootstrap/CI summary rows
# ---------------------------------------------------------------------------

def nan_summary(**extra) -> Dict[str, float]:
    """Return a dict of NaN values for the standard summary keys.

    The base dict contains ``{'mean': nan, 'std': nan, 'ci_lower': nan,
    'ci_upper': nan}``.  Pass keyword arguments to extend it
    (e.g., ``nan_summary(p_value=np.nan)``).

    Returns a fresh dict on every call.
    """
    return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan,
            "ci_upper": np.nan, **extra}


# ---------------------------------------------------------------------------
# select_alpha_by_cv_concordance — shared K-fold alpha selection
# ---------------------------------------------------------------------------

def select_alpha_by_cv_concordance(
        X, y, alphas, *, l1_ratio, cv_folds, random_state,
        max_iter=None, tol=None, no_valid_scores_stacklevel=3):
    """Select a penalty alpha by K-fold cross-validated concordance.

    Shared core of ``CoxNetModel._select_alpha_cv`` (models/coxnet.py) and
    ``_PenalizedCoxSelector._select_alpha_cv`` (preprocessing/selection.py):
    per fold, fit a ``CoxnetSurvivalAnalysis`` path on the training split and
    score every candidate alpha on the held-out split's concordance index,
    then pick the alpha with the highest mean C-index.

    Returns ``(selected_alpha, n_failed_folds)`` — the smallest alpha is the
    fallback when no fold/alpha combination scored. This helper does not
    validate ``cv_folds`` or ``n_samples``; that is the caller's job (the
    selector path pre-validates and falls back to the smallest alpha, while
    the model path lets ``KFold`` / sksurv raise). Any failed-fold warning is
    also the caller's responsibility — this helper emits only the "no valid
    scores" ``SuggestionWarning``, at *no_valid_scores_stacklevel*. Both
    callers pass ``4`` (they delegate from one method frame); the parameter is
    exposed so a caller can adjust if its call depth changes.
    ``max_iter`` / ``tol`` reach ``CoxnetSurvivalAnalysis`` only when not
    ``None``, so callers relying on sksurv defaults keep them.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = np.zeros(len(alphas))
    counts = np.zeros(len(alphas))  # how many folds scored each alpha
    failed_folds = 0

    fold_kwargs = {"l1_ratio": l1_ratio, "alphas": alphas,
                   "fit_baseline_model": False}
    if max_iter is not None:  # is not None — NOT truthiness
        fold_kwargs["max_iter"] = max_iter
    if tol is not None:
        fold_kwargs["tol"] = tol

    for train_idx, val_idx in kf.split(X_arr):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        fold_model = CoxnetSurvivalAnalysis(**fold_kwargs)
        try:
            fold_model.fit(X_train, y_train)
        except (MemoryError, RecursionError):
            raise
        except Exception as exc:
            failed_folds += 1
            logger.debug(
                "Penalized Cox CV fold fit failed (%s: %s). Skipping fold.",
                type(exc).__name__, exc,
            )
            continue

        # Path may stop early, fitting fewer alphas than requested.
        coef_2d = fold_model.coef_.ndim == 2
        n_fitted = fold_model.coef_.shape[1] if coef_2d else 1

        for i in range(min(len(alphas), n_fitted)):
            coef = fold_model.coef_[:, i] if coef_2d else fold_model.coef_
            risk_scores = X_val @ coef
            try:
                c_index = concordance_index_censored(
                    y_val['event'], y_val['time'], risk_scores
                )[0]
                scores[i] += c_index
                counts[i] += 1
            except (MemoryError, RecursionError):
                raise
            except Exception as exc:
                # Scoring can fail for valid reasons (all-tied predictions,
                # single-class fold). Skip this alpha/fold combination —
                # valid_mask handles alphas with zero valid folds.
                logger.debug(
                    "C-index scoring failed for alpha[%d] (%s: %s). "
                    "Skipping alpha/fold combination.",
                    i, type(exc).__name__, exc,
                )

    valid_mask = counts > 0
    if not valid_mask.any():
        warnings.warn(
            "Alpha selection: no valid C-index scores across any "
            "fold/alpha combination. Falling back to smallest alpha.",
            SuggestionWarning,
            stacklevel=no_valid_scores_stacklevel,
        )
        return alphas[-1], failed_folds

    mean_scores = np.where(valid_mask, scores / counts, -np.inf)
    return alphas[np.argmax(mean_scores)], failed_folds


# ---------------------------------------------------------------------------
# _warn_unused_dict_keys — per-model dict key validation
# ---------------------------------------------------------------------------

def _warn_unused_dict_keys(param_dict, param_name, used_keys):
    """Warn if a per-model dict contains keys not matching any model.

    Parameters
    ----------
    param_dict : dict, scalar, or None
        Per-model parameter dict, or a scalar (no warning needed).
    param_name : str
        Parameter name for the warning message (e.g., ``"n_trials"``).
    used_keys : list or set
        Model names that were actually used.
    """
    if not isinstance(param_dict, dict):
        return
    unused = set(param_dict.keys()) - set(used_keys)
    if unused:
        warnings.warn(
            f"{param_name} dict contains unused keys that were "
            f"ignored: {sorted(unused)}. Models used: {sorted(used_keys)}",
            SuggestionWarning,
        )
