"""Feature importance utilities for survival models."""

import contextlib
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

from .._warnings import (
    InternalWarning,
    MethodologicalWarning,
    MetricFailureWarning,
    SuggestionWarning,
)
from ..logging_utils import get_logger, log_metric_failure
from ..preprocessing._cox_fit import fit_univariate_cox
from ..utils import (
    SurvivalArray, MIN_FEATURES_FOR_PARALLEL, resolve_parallel_n_jobs,
    unwrap_pipeline, propagate_feature_names, joblib_parallel_backend,
    resolve_joblib_backend,
)
from .helpers import _format_elapsed
from ._paired_comparison import _adjust_p_values

logger = get_logger("importance")


# ---------------------------------------------------------------------------
# Grouped permutation importance (categorical feature support)
# ---------------------------------------------------------------------------


def _validate_categorical_groups(categorical_groups, available_columns):
    """Validate categorical_groups against available columns.

    Parameters
    ----------
    categorical_groups : dict of {str: list of str}
        Mapping from group name to encoded column names.
    available_columns : list of str
        Columns present in the data.

    Returns
    -------
    dict of {str: list of str}
        Filtered groups (partially-present groups warn and keep available
        columns; fully-absent groups are dropped).

    Raises
    ------
    TypeError
        If categorical_groups is not a dict.
    ValueError
        If a column appears in multiple groups, or a group name collides
        with an ungrouped column.
    """
    if not isinstance(categorical_groups, dict):
        raise TypeError(
            f"categorical_groups must be a dict, got {type(categorical_groups).__name__}"
        )

    available_set = set(available_columns)
    grouped_cols = set()
    filtered = {}

    for group_name, cols in categorical_groups.items():
        if not isinstance(cols, (list, tuple)):
            raise TypeError(
                f"categorical_groups['{group_name}'] must be a list of column "
                f"names, got {type(cols).__name__}"
            )
        # Check for duplicate column assignments
        overlap = grouped_cols & set(cols)
        if overlap:
            raise ValueError(
                f"Column(s) {sorted(overlap)} appear in multiple "
                f"categorical_groups entries."
            )

        present = [c for c in cols if c in available_set]
        missing = [c for c in cols if c not in available_set]

        if not present:
            warnings.warn(
                f"categorical_groups['{group_name}']: all columns "
                f"{cols} are absent from the data; dropping group.",
                SuggestionWarning,
                stacklevel=3,
            )
            continue
        if missing:
            warnings.warn(
                f"categorical_groups['{group_name}']: columns "
                f"{missing} are absent (possibly removed by feature "
                f"selection); using {present} only.",
                SuggestionWarning,
                stacklevel=3,
            )
        if len(present) < 2:
            warnings.warn(
                f"categorical_groups['{group_name}']: only 1 column "
                f"remains ({present}); treating as ungrouped.",
                SuggestionWarning,
                stacklevel=3,
            )
            continue

        grouped_cols.update(present)
        filtered[group_name] = present

    # Check group names don't collide with ungrouped column names
    ungrouped = available_set - grouped_cols
    collisions = set(filtered.keys()) & ungrouped
    if collisions:
        raise ValueError(
            f"Group name(s) {sorted(collisions)} collide with ungrouped "
            f"column names in the data. Rename the groups or the columns."
        )

    return filtered


def _build_permutation_units(columns, categorical_groups):
    """Build ordered list of (name, column_indices) for permutation.

    Each group's unit is inserted at the position of its first member
    column. Ungrouped columns each get their own single-column unit.

    Returns
    -------
    list of (str, list of int)
        Each entry is (result_feature_name, list_of_column_indices).
    """
    col_to_idx = {c: i for i, c in enumerate(columns)}
    grouped_cols = set()
    group_first_idx = {}

    for group_name, cols in categorical_groups.items():
        idxs = [col_to_idx[c] for c in cols if c in col_to_idx]
        if idxs:
            grouped_cols.update(cols)
            group_first_idx[group_name] = (min(idxs), idxs)

    units = []
    used_groups = set()
    for i, col in enumerate(columns):
        if col in grouped_cols:
            # Insert group unit at the position of its first member
            for gname, (first_idx, idxs) in group_first_idx.items():
                if first_idx == i and gname not in used_groups:
                    units.append((gname, idxs))
                    used_groups.add(gname)
        else:
            units.append((col, [i]))

    return units


def _grouped_permutation_importance(
    model, X, y, categorical_groups, n_repeats, random_state,
    scoring=None, n_jobs=1,
):
    """Compute permutation importance with grouped feature permutation.

    Features listed in *categorical_groups* are permuted together (same
    row shuffle applied to all columns in the group). Other features are
    permuted individually, matching sklearn's standard behavior.

    Parameters
    ----------
    model : estimator
        Fitted model with a ``score`` method.
    X : DataFrame
        Test features.
    y : SurvivalArray
        Test outcomes.
    categorical_groups : dict of {str: list of str}
        Mapping from group name to encoded column names.
    n_repeats : int
        Number of permutation repeats.
    random_state : int or RandomState
        Random seed for reproducibility.
    scoring : callable, optional
        ``scoring(model, X, y) -> float``. Defaults to ``model.score``.
    n_jobs : int, optional
        Unused (reserved for API compatibility). Permutation is always
        sequential to avoid data-copy overhead on small datasets.

    Returns
    -------
    sklearn.utils.Bunch
        With keys ``importances_mean``, ``importances_std``,
        ``importances`` (shape ``(n_units, n_repeats)``), and
        ``feature_names`` (list of result feature names).
    """
    from sklearn.utils import Bunch

    if scoring is None:
        scoring = lambda m, X_, y_: m.score(X_, y_)

    rng = (random_state if isinstance(random_state, np.random.RandomState)
           else np.random.RandomState(random_state))

    columns = X.columns.tolist()
    units = _build_permutation_units(columns, categorical_groups)
    n_units = len(units)

    baseline = scoring(model, X, y)
    importances = np.zeros((n_units, n_repeats))

    # Work on underlying numpy array for fast column shuffle/restore
    X_arr = X.values.copy()
    X_df = pd.DataFrame(X_arr, columns=columns, index=X.index)

    for r in range(n_repeats):
        for u, (name, col_idxs) in enumerate(units):
            # Save original columns
            saved = X_arr[:, col_idxs].copy()
            # Shuffle rows identically across all columns in the unit
            perm = rng.permutation(X_arr.shape[0])
            X_arr[:, col_idxs] = X_arr[perm][:, col_idxs]
            # Update the DataFrame (pandas 3.0 CoW makes .values read-only)
            X_df = pd.DataFrame(X_arr, columns=columns, index=X.index)
            # Score
            score = scoring(model, X_df, y)
            importances[u, r] = baseline - score
            # Restore
            X_arr[:, col_idxs] = saved

    feature_names = [name for name, _ in units]
    return Bunch(
        importances_mean=importances.mean(axis=1),
        importances_std=importances.std(axis=1, ddof=0),
        importances=importances,
        feature_names=feature_names,
    )


def detect_correlated_features(X, threshold=0.8):
    """
    Detect highly correlated feature pairs.

    Permutation importance can be misleading for correlated features.
    When two features are correlated and one is permuted, the model
    still accesses the information through its correlated partner.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    threshold : float, default=0.8
        Correlation threshold (absolute value).

    Returns
    -------
    pairs : list of tuple
        List of (feature1, feature2, correlation) tuples for pairs
        with |correlation| >= threshold.
    """
    corr_matrix = X.corr()

    # Vectorized upper-triangle scan: the old per-pair Python loop took
    # seconds at p ~ 1000 while X.corr() itself is fast C code. Only the
    # (few) above-threshold pairs are materialized as tuples. NaN
    # correlations compare False, as before. Pairs are built in row-major
    # i < j order, so the stable sort keeps the same tie order as the loop.
    features = corr_matrix.columns
    vals = corr_matrix.values
    iu, ju = np.triu_indices(len(features), k=1)
    with np.errstate(invalid='ignore'):
        keep = np.abs(vals[iu, ju]) >= threshold
    pairs = [
        (features[i], features[j], vals[i, j])
        for i, j in zip(iu[keep], ju[keep])
    ]

    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


def compute_univariate_cox(X, y: SurvivalArray, penalizer=0.01, verbose=False, n_jobs=1, force_parallel=False):
    """
    Compute univariate Cox regression statistics for all features.

    Fits a separate univariate Cox proportional hazards model for each feature
    and extracts p-value, hazard ratio, and confidence intervals.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : SurvivalArray
        Survival outcome with 'event' and 'time' fields.
    penalizer : float, default=0.01
        L2 regularization to improve convergence. Applied when initial
        fit fails.
    verbose : bool, default=False
        If True, print warnings for features that fail to fit.
    n_jobs : int, default=1
        Number of parallel jobs for fitting univariate Cox models.
        - 1: Sequential execution (default)
        - -1: Use all available CPU cores
        - n > 1: Use n CPU cores
        Note: Parallelization is auto-disabled when n_features < 15
        (threshold where sequential is faster).
    force_parallel : bool, default=False
        If True, skip the auto-disable threshold check and use the
        specified n_jobs regardless of feature count. Useful for
        benchmarking or when user knows better (e.g., slow custom metrics).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - feature: feature name
        - p_value: Wald test p-value
        - hazard_ratio: exp(coefficient)
        - hr_lower_95: 95% CI lower bound for hazard ratio
        - hr_upper_95: 95% CI upper bound for hazard ratio
        - coefficient: raw log hazard ratio
        Sorted by ascending p-value (most significant first).

    Examples
    --------
    >>> from mlsurv.evaluation.importance import compute_univariate_cox
    >>> univar_cox = compute_univariate_cox(X, y)
    >>> print(univar_cox.head())
          feature  p_value  hazard_ratio  hr_lower_95  hr_upper_95  coefficient
    0        age    0.001         1.045        1.019        1.071        0.044
    1      stage    0.003         2.134        1.287        3.539        0.758
    ...

    Notes
    -----
    Features that fail to fit are assigned ``p_value=inf`` (matching the
    ``FAILED_PVAL`` sentinel used by ``selection.py``), ``hazard_ratio=NaN``,
    ``coefficient=NaN``, and NaN confidence intervals.  Using ``inf``
    ensures failed features always sort last regardless of context.
    Common failure reasons include:
    - Constant or near-constant columns
    - Complete separation
    - Convergence issues
    """
    columns = list(X.columns)
    n_features = len(columns)

    # Auto-disable parallelization if below threshold (unless forced)
    effective_n_jobs = resolve_parallel_n_jobs(
        n_jobs, n_features, MIN_FEATURES_FOR_PARALLEL, force_parallel, verbose,
    )

    def _fit_one(col):
        return fit_univariate_cox(
            X[col], y, col,
            penalizer=penalizer,
            verbose=verbose if effective_n_jobs == 1 else False,
            extract='full',
            logger=logger,
        )

    if effective_n_jobs == 1:
        results = [_fit_one(col) for col in columns]
    else:
        backend = resolve_joblib_backend("loky")
        results = Parallel(n_jobs=effective_n_jobs, backend=backend)(
            delayed(_fit_one)(col) for col in columns
        )

    # Surface unexpected errors from workers in the main process
    errors = [r['error_detail'] for r in results if r.get('error_detail')]
    if errors:
        summary = "; ".join(errors)
        warnings.warn(
            f"{len(errors)} unexpected Cox fitting error(s): {summary}",
            InternalWarning,
            stacklevel=2,
        )

    drop = {'status', 'error_detail'}
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in drop}
                       for r in results])
    return df.sort_values('p_value').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation importance helpers (used by evaluate.py)
# ---------------------------------------------------------------------------


def _build_importance_summary(result, features, coefficients):
    """Build importance summary DataFrame from permutation importance result."""
    return pd.DataFrame({
        'feature': features,
        'coefficient': coefficients,
        'perm_mean': result.importances_mean,
        'perm_std': result.importances_std,
    }).sort_values('perm_mean', ascending=False)


def _build_iterations_dataframe(result, features, n_repeats):
    """Build long-format iteration-level importance DataFrame."""
    iterations_data = []
    for feat_idx, feat in enumerate(features):
        for j, imp in enumerate(result.importances[feat_idx]):
            iterations_data.append({'feature': feat, 'iteration': j, 'importance': imp})

    return pd.DataFrame(iterations_data)


def _selected_from_step(step):
    """Return a selector step's ``selected_features_``, or None.

    A ``feature_selector=[...]`` list is wrapped into a nested Pipeline
    (see ``preprocessing/pipelines.py``), which carries no
    ``selected_features_`` of its own even though its inner steps do. For
    such a chain the *last* inner step that exposes the attribute wins: it
    ran last, so its output is closest to what the model saw. Callers should
    ask the wrapper Pipeline for ``get_feature_names_out()`` first, which
    accounts for every step; this is the fallback for when that fails.
    """
    selected = getattr(step, 'selected_features_', None)
    if selected is not None:
        return selected
    result = None
    for _, inner in getattr(step, 'steps', []):
        inner_selected = getattr(inner, 'selected_features_', None)
        if inner_selected is not None:
            result = inner_selected
    return result


def _names_out(step):
    """Return a step's post-transform feature names, or None.

    This is the authoritative answer for a chained ``feature_selector=[...]``
    wrapper Pipeline, whose own names-out accounts for every step. It also
    covers third-party selectors (``SelectKBest``, ``VarianceThreshold``,
    ``RFE``), which have no ``selected_features_`` at all.
    """
    getter = getattr(step, 'get_feature_names_out', None)
    if getter is None:
        return None
    try:
        return list(getter())
    except (AttributeError, ValueError, TypeError, KeyError, NotFittedError) as e:
        logger.debug(
            "get_feature_names_out() on %s raised %s: %s",
            type(step).__name__, type(e).__name__, e,
        )
        return None


def _get_selected_features(model, all_features):
    """Return selected feature names from a pipeline's selector step, or None.

    Three rungs, in order: a chained selector's own ``get_feature_names_out()``
    (it accounts for every step), then ``selected_features_``, then a plain
    step's names-out. When the selector received integer column indices (e.g.
    after a scaler that strips names), the indices are mapped back to
    *all_features*.

    ``None`` means the selection is unknown, which covers both "no selector in
    the pipeline" and "a selector that reports nothing usable". Callers that
    branch on it should treat those the same way.
    """
    if not (hasattr(model, 'named_steps') and 'selector' in model.named_steps):
        return None
    selector = model.named_steps['selector']
    # For a chain of selectors, the wrapper Pipeline's own get_feature_names_out()
    # is authoritative: it reflects every step, including a final third-party
    # one that narrows further without exposing `selected_features_`. Asking
    # the inner steps first would stop at a stale, too-long list.
    selected = _names_out(selector) if hasattr(selector, 'steps') else None
    if selected is None:
        selected = _selected_from_step(selector)
    if selected is None:
        # Non-Pipeline selectors only: a Pipeline already tried this above.
        selected = _names_out(selector)
    if selected is None:
        logger.debug(
            "Pipeline 'selector' step (%s) reports neither selected_features_ "
            "nor usable get_feature_names_out().",
            type(selector).__name__,
        )
        return None
    selected = list(selected)
    # If the selector saw integer column indices, map them to real names
    if selected and isinstance(selected[0], (int, np.integer)):
        if any(not 0 <= i < len(all_features) for i in selected):
            logger.debug(
                "Selector indices out of range for %d features: %s",
                len(all_features), selected,
            )
            return None
        return [all_features[i] for i in selected]
    return selected


def _align_coefficients(coef_values, selected_features, all_features):
    """Align coefficients from selected features to the full feature list.

    *selected_features* holds the resolved labels that key *coef_values*, one
    per value. Labels that are not in *all_features* (an intercept, a scale
    term) drop out; features with no label get NaN.

    Always returns exactly ``len(all_features)`` values. Raises ``ValueError``
    on a length mismatch or a duplicated label, rather than padding or
    truncating to fit.
    """
    values = coef_values.values if isinstance(coef_values, pd.Series) else coef_values
    labels = list(selected_features)
    if len(values) != len(labels):
        raise ValueError(
            f"Cannot align coefficients: {len(values)} values but "
            f"{len(labels)} feature labels."
        )
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"Cannot align coefficients: duplicate feature labels in {labels}."
        )
    coef_series = pd.Series(values, index=labels)
    return coef_series.reindex(all_features).values.tolist()


def _string_coef_index(coef_series):
    """Return *coef_series*' index as a list when every label is a string.

    ``None`` otherwise, which covers a missing or empty index, an integer
    index (lifelines and FSSVM both fall back to one when a model is fit on
    an unnamed array), and a mixed index. Every element has to be a string:
    checking only the first would call a half-named index usable and then
    align the rest by a label that does not exist.
    """
    index = getattr(coef_series, 'index', None)
    # isinstance, not a None check: `list` and `tuple` have an `index` METHOD,
    # so a wrapper returning a bare list would hand back a bound method here
    # and raise on len(). Both surfaces accept whatever get_coefficients()
    # returns, so this has to reject a non-index rather than trust the name.
    if not isinstance(index, (pd.Index, np.ndarray, list, tuple)):
        return None
    if len(index) == 0:
        return None
    if not all(isinstance(i, str) for i in index):
        return None
    return list(index)


def _resolve_coef_labels(coef_series, selected_features, features):
    """Return the labels that key *coef_series*, or None if unresolvable.

    *features* is the target feature space to align onto. Both coefficient
    surfaces call this, and they name that space differently: the importance
    table passes the test-set columns, ``_coefficients.py`` passes the
    pipeline's pre-selection names. Keep the wording here neutral so it stays
    true for both.

    Priority matters:

    1. The model's own coefficient index, when every label is a string and at
       least one of them is in *features*. Some models return coefficients in
       a different order than the selector emitted them (lifelines'
       Weibull-AFT sorts its covariates), so keying by the selector's order
       would silently attach each value to the wrong feature.
    2. The selector's post-selection names, correct when the model has no
       name index, because then no name-based reordering can have happened.
    3. *features* itself, when the vector is already full length.

    ``None`` means no rung applied. The caller has to decide what to report,
    because it cannot say which feature any value belongs to.
    """
    index = _string_coef_index(coef_series)
    if index is not None:
        if set(index) & set(features):
            # Keying by name is correct per label, so labels *features* does
            # not contain (an intercept, a scale term) simply drop out.
            return index
        # Names that match nothing (a pipeline fit on unnamed arrays gets
        # sklearn's x0, x1, ...). Those are positional by construction, so
        # fall through to positional alignment.
        logger.debug(
            "Coefficient index %s matches no entry in the target feature "
            "list; aligning by position instead.", index,
        )
    if selected_features is not None:
        return list(selected_features)
    if len(coef_series) == len(features):
        return list(features)
    return None


# Returned when a model legitimately has no coefficients (trees, deep
# models). Distinct from None, which means extraction was attempted and
# failed. That one has to reach the user; this one is the normal case.
_NO_COEFFICIENTS = object()

# Errors a model wrapper can raise while producing its coefficients. Anything
# outside this set (a NameError in a wrapper, say) is a bug and should escape
# rather than turn into a quiet NaN column.
_COEF_EXTRACTION_ERRORS = (
    ValueError, TypeError, AttributeError, KeyError, IndexError, NotFittedError,
)


def _warn_coefficients_unresolved(actual_model, reason, model_name=None):
    """Report that the coefficient column had to fall back to NaN.

    Permutation importance itself is unaffected, so say so: the user should
    not read this as a lost importance result. Routed through
    ``log_metric_failure`` so the logger summary drops to DEBUG whenever a
    visible (non-``InternalWarning``) warning is emitted for the same failure,
    which keeps it off stderr twice (see ``logging_utils``).
    """
    exc = reason if isinstance(reason, BaseException) else ValueError(str(reason))
    # Name the model, not just its class: MetricFailureWarning is an
    # IterativeWarning, deduped once per unique message, so a class-only
    # message would silence every model after the first.
    label = model_name or type(actual_model).__name__
    log_metric_failure(
        logger, "importance_coefficients", exc,
        model_name=label, warn_user=True,
        warn_category=MetricFailureWarning,
        user_message=(
            f"Could not resolve coefficients for {label} "
            f"({type(actual_model).__name__}): {reason}. The coefficient "
            f"column will be NaN; permutation importance is unaffected."
        ),
    )


def _extract_raw_coefficients(actual_model, model_name=None):
    """Return the model's raw coefficients, or a sentinel.

    Tries mlsurv's ``get_coefficients()`` first, then a direct ``coef_`` for
    plain sklearn estimators. Every mlsurv wrapper defines
    ``get_coefficients()``; the ones without coefficients return ``None``.

    Returns ``_NO_COEFFICIENTS`` when the model simply has none (the normal
    case for trees and deep models), and ``None`` when extraction was
    attempted and failed, which is reported to the user before returning.
    """
    if hasattr(actual_model, 'get_coefficients'):
        try:
            coef = actual_model.get_coefficients()
        except (MemoryError, RecursionError):
            raise
        except _COEF_EXTRACTION_ERRORS as e:
            _warn_coefficients_unresolved(actual_model, e, model_name)
            return None
        return _NO_COEFFICIENTS if coef is None else coef

    if hasattr(actual_model, 'coef_'):
        try:
            coef = actual_model.coef_
            if coef.ndim == 2:  # CoxNet: (n_features, n_alphas)
                coef = coef[:, 0]
            return coef
        except (MemoryError, RecursionError):
            raise
        except _COEF_EXTRACTION_ERRORS as e:
            _warn_coefficients_unresolved(actual_model, e, model_name)
            return None

    return _NO_COEFFICIENTS


def _get_coefficients(model, features, model_name=None):
    """Extract model coefficients, aligned to feature order.

    Always returns a list of exactly ``len(features)`` values; features with
    no resolvable coefficient are NaN. Every coefficient problem it can
    recognize degrades to that NaN column rather than raising, because a
    coefficient problem must not discard a permutation-importance result that
    was computed successfully. Only an unexpected exception type (a bug in a
    wrapper) propagates. Models without coefficients (trees, deep models) get
    all NaN, which is the normal case rather than a failure.
    """
    nan_result = [np.nan] * len(features)
    actual_model = unwrap_pipeline(model)
    selected_features = _get_selected_features(model, features)

    # Propagate names only when the selector told us nothing, either because
    # there is no selector or because it reports no usable selection. When we
    # do know the selection, the inner model's own post-selection names are
    # the ones that matter.
    if selected_features is None:
        propagate_feature_names(actual_model, model)

    coef_series = _extract_raw_coefficients(actual_model, model_name)
    if coef_series is _NO_COEFFICIENTS or coef_series is None:
        # No coefficients to have (trees, deep models), or extraction failed
        # and already reported. Either way the column is NaN.
        return nan_result
    if len(coef_series) == 0:
        _warn_coefficients_unresolved(
            actual_model, "the model returned 0 coefficients", model_name,
        )
        return nan_result

    labels = _resolve_coef_labels(coef_series, selected_features, features)
    if labels is None:
        _warn_coefficients_unresolved(
            actual_model,
            f"{len(coef_series)} coefficients could not be matched to "
            f"{len(features)} features",
            model_name,
        )
        return nan_result

    try:
        aligned = _align_coefficients(coef_series, labels, features)
        # Inside the try: a non-numeric coefficient makes this cast raise,
        # and that is a recognizable coefficient problem, not a wrapper bug.
        has_finite = bool(np.isfinite(np.asarray(aligned, dtype=float)).any())
    except (ValueError, TypeError) as e:
        _warn_coefficients_unresolved(actual_model, e, model_name)
        return nan_result

    # The model had coefficients but not one usable value reached a test
    # feature, either because the labels describe a different feature space or
    # because the fit produced NaNs. Same NaN column either way, but the user
    # should hear about it rather than read it as "this model has no
    # coefficients".
    if not has_finite:
        _warn_coefficients_unresolved(
            actual_model,
            f"no finite coefficient out of {len(coef_series)} landed on a "
            f"test-set feature",
            model_name,
        )
    return aligned


def _empty_importance_results():
    """Return empty importance DataFrames when importance is skipped or fails."""
    summary_df = pd.DataFrame(
        columns=['feature', 'coefficient', 'perm_mean', 'perm_std']
    )
    iterations_df = pd.DataFrame(
        columns=['feature', 'iteration', 'importance']
    )
    return summary_df, iterations_df


def _subsample_for_importance(X_test, y_test: SurvivalArray, max_samples,
                              random_state):
    """Return *X_test*, *y_test* capped at *max_samples* rows.

    Permutation importance costs ``n_features x n_repeats x (predict +
    C-index over n_test rows)``, and the C-index term is superlinear in
    ``n_test``, so on large cohorts importance can out-cost model fitting.
    Capping the rows attacks that term directly.

    With an int *random_state* the draw depends only on that seed, the cohort
    size, and *max_samples*, so every model in an evaluate() call sees the
    same rows and their importances stay comparable. A shared
    ``RandomState`` instance advances between calls, so each model draws
    different rows; that case is not warned about. With ``None`` the caller
    is warned. ``max_samples`` of ``None``, or a value at or above the cohort
    size, returns the inputs untouched.

    Raises
    ------
    ValueError
        If *max_samples* is not an int, or is below 2. This is the only
        validation on the ``learner.importance()`` path, which does not build
        an ``EvalConfig``.
    """
    if max_samples is None:
        return X_test, y_test
    if isinstance(max_samples, bool) or not isinstance(
            max_samples, (int, np.integer)):
        raise ValueError(
            f"importance_max_samples must be an int or None, got "
            f"{type(max_samples).__name__}."
        )
    # 2, not 1: the C-index needs a comparable pair, so a single row makes
    # every permutation score undefined and costs the model its whole
    # importance table. Mirrors interaction_max_samples.
    if max_samples < 2:
        raise ValueError(
            f"importance_max_samples must be >= 2 or None, got {max_samples}."
        )
    if max_samples >= len(X_test):
        return X_test, y_test
    if random_state is None:
        warnings.warn(
            "importance_max_samples was set but random_state is None, so "
            "each model draws a different subsample and importances are not "
            "comparable across models. Pass setup(random_state=...) or "
            "evaluate(random_state=...) to make the draw reproducible.",
            MethodologicalWarning, stacklevel=2,
        )
    # A RandomState instance is a supported random_state throughout this
    # module (see _grouped_permutation_importance); passing one to the
    # RandomState constructor would raise. Note that a shared instance
    # advances between models, so their subsamples differ.
    rng = (random_state if isinstance(random_state, np.random.RandomState)
           else np.random.RandomState(random_state))
    idx = rng.choice(len(X_test), size=max_samples, replace=False)
    idx.sort()  # keep cohort order, so the subsample reads like the cohort
    return X_test.iloc[idx], y_test[idx]


def _compute_permutation_importance(model, X_test, y_test: SurvivalArray, n_repeats, random_state,
                                    show_progress=True, model_name=None, n_jobs=1,
                                    categorical_groups=None, replacing=False,
                                    max_samples=None):
    """Compute feature importance with iteration-level detail.

    Parallelism is applied across features via sklearn's
    ``permutation_importance``.  For small feature sets (<~15 features)
    the joblib overhead may negate speedup gains with n_jobs>1.

    When *replacing* is True, the progress line notes that previously computed
    importance results are being replaced.

    *max_samples* caps the test rows used (see
    :func:`_subsample_for_importance`); ``None`` uses all of them.

    Returns
    -------
    summary_df : DataFrame
        Grouped importance (or standard if no groups).
    iterations_df : DataFrame
        Iteration-level detail for the grouped importance.
    per_column_df : DataFrame or None
        Per-column (ungrouped) importance for individual dummy columns.
        Only populated when *categorical_groups* is used; ``None`` otherwise.
    """
    features = X_test.columns.tolist()
    X_test, y_test = _subsample_for_importance(
        X_test, y_test, max_samples, random_state,
    )

    if show_progress:
        display_name = model_name if model_name else "model"
        replace_note = " (replacing previous results)" if replacing else ""
        print(
            f"Computing permutation importance for {display_name} "
            f"({n_repeats} repeats){replace_note}...",
            end="", flush=True,
        )

    per_column_df = None
    t0 = time.perf_counter()

    # Phase 1: the permutation itself. Only a failure here means there is no
    # importance to report, so only this phase returns the empty sentinel.
    try:
        use_grouped = bool(categorical_groups)
        if use_grouped:
            valid_groups = _validate_categorical_groups(
                categorical_groups, features
            )
            use_grouped = bool(valid_groups)

        # Always compute standard per-column importance
        scoring_fn = lambda m, X, y: m.score(X, y)
        backend_ctx = (
            joblib_parallel_backend(n_jobs, backend="loky")
            if n_jobs != 1 else contextlib.nullcontext()
        )
        with backend_ctx:
            result_std = permutation_importance(
                model, X_test, y_test,
                n_repeats=n_repeats, random_state=random_state,
                scoring=scoring_fn, n_jobs=n_jobs,
            )

        if use_grouped:
            # Also compute grouped importance
            result = _grouped_permutation_importance(
                model, X_test, y_test, valid_groups,
                n_repeats=n_repeats, random_state=random_state,
                scoring=scoring_fn,
            )
            result_features = result.feature_names
        else:
            result = result_std
            result_features = features

        if show_progress:
            print(f" done ({_format_elapsed(time.perf_counter() - t0)}).")

    except (MemoryError, RecursionError):
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError,
            RuntimeError) as e:
        log_metric_failure(
            logger, "permutation_importance", e,
            model_name=model_name, warn_user=True,
            warn_category=MetricFailureWarning,
            user_message=(
                f"Permutation importance failed for "
                f"{model_name or 'model'} ({type(e).__name__}: {e}). "
                f"Importance is unavailable for this model."
            ),
        )
        if show_progress:
            print(f" failed ({type(e).__name__}, "
                  f"{_format_elapsed(time.perf_counter() - t0)}).")
        summary_df, iterations_df = _empty_importance_results()
        return summary_df, iterations_df, None

    # Phase 2: attach coefficients. A problem here degrades the coefficient
    # column to NaN. It must never throw away the importance just computed.
    try:
        coefficients_std = _get_coefficients(model, features, model_name)
        if use_grouped:
            per_column_df = _build_importance_summary(
                result_std, features, coefficients_std,
            )
            # Grouped features get NaN coefficients (not aggregatable). The
            # rest are ungrouped column names, so each one indexes back into
            # `features` to find its own coefficient.
            coefficients = [
                np.nan if fname in valid_groups
                else coefficients_std[features.index(fname)]
                for fname in result_features
            ]
        else:
            coefficients = coefficients_std
        # The summary builders below sit outside this guard, so check the
        # length contract here: a violation becomes the handled NaN column
        # rather than a pandas error that aborts the whole evaluate() run.
        if len(coefficients) != len(result_features):
            raise ValueError(
                f"coefficient/feature length mismatch: {len(coefficients)} "
                f"vs {len(result_features)}"
            )
    except (MemoryError, RecursionError):
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError,
            RuntimeError) as e:
        log_metric_failure(
            logger, "importance_coefficients", e,
            model_name=model_name, warn_user=True,
            warn_category=MetricFailureWarning,
            user_message=(
                f"Could not attach coefficients for "
                f"{model_name or 'model'} ({type(e).__name__}: {e}). "
                f"The coefficient column will be NaN; permutation "
                f"importance is unaffected."
            ),
        )
        coefficients = [np.nan] * len(result_features)
        per_column_df = None

    summary_df = _build_importance_summary(result, result_features, coefficients)
    iterations_df = _build_iterations_dataframe(result, result_features, n_repeats)

    return summary_df, iterations_df, per_column_df


# ---------------------------------------------------------------------------
# PIMP — Permutation Importance with Permuted response (Altmann et al. 2010)
# ---------------------------------------------------------------------------


def _permute_y(y: SurvivalArray, rng: np.random.RandomState) -> SurvivalArray:
    """Shuffle (event, time) pairs together, breaking X-y association."""
    idx = rng.permutation(len(y))
    return y[idx]


def _fit_null_importance(pipeline, X_train, y_perm, X_test, y_test, n_repeats, seed,
                         categorical_groups=None):
    """Clone pipeline, fit on permuted y, return importances_mean array.

    Returns None if the null model fails to fit or score (e.g., convergence
    failure on permuted data). The caller filters None entries.
    """
    try:
        model = clone(pipeline)
        model.fit(X_train, y_perm)
        if categorical_groups:
            result = _grouped_permutation_importance(
                model, X_test, y_test, categorical_groups,
                n_repeats=n_repeats, random_state=seed,
            )
        else:
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=n_repeats, random_state=seed, n_jobs=1,
            )
        return result.importances_mean
    except (MemoryError, RecursionError):
        raise
    except Exception as e:
        logger.debug("Null model fitting failed (seed=%d): %s: %s",
                     seed, type(e).__name__, e)
        return None



def _validate_pimp_inputs(pipeline_or_model, n_permutations, n_importance_repeats):
    """Validate PIMP parameters and verify clone pre-flight."""
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    if n_importance_repeats < 1:
        raise ValueError(
            f"n_importance_repeats must be >= 1, got {n_importance_repeats}"
        )
    try:
        clone(pipeline_or_model)
    except Exception as e:
        raise TypeError(
            f"Cannot clone the model for PIMP null fitting. "
            f"Ensure the model follows sklearn's clone() contract. "
            f"Original error: {e}"
        ) from e


def _fit_null_models(
    pipeline_or_model, X_train, perm_ys, X_test, y_test,
    n_importance_repeats, importance_seeds, n_jobs, n_permutations,
    show_progress, categorical_groups=None,
):
    """Fit null models on permuted data and return null importance matrix.

    Returns
    -------
    np.ndarray of shape (n_valid, n_features)
    """
    if show_progress:
        print(f"PIMP: fitting {n_permutations} null models (n_jobs={n_jobs})...")

    t0 = time.perf_counter()
    if n_jobs == 1:
        null_importances = []
        progress_interval = max(1, n_permutations // 10)
        for i, (y_perm, seed) in enumerate(zip(perm_ys, importance_seeds)):
            null_imp = _fit_null_importance(
                pipeline_or_model, X_train, y_perm,
                X_test, y_test, n_importance_repeats, seed,
                categorical_groups=categorical_groups,
            )
            null_importances.append(null_imp)
            if show_progress and (i + 1) % progress_interval == 0:
                elapsed = _format_elapsed(time.perf_counter() - t0)
                print(f"  [{i + 1}/{n_permutations}] ({elapsed})")
    else:
        backend = resolve_joblib_backend("loky")
        null_importances = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_fit_null_importance)(
                pipeline_or_model, X_train, y_perm,
                X_test, y_test, n_importance_repeats, seed,
                categorical_groups=categorical_groups,
            )
            for y_perm, seed in zip(perm_ys, importance_seeds)
        )

    # Filter out failed null iterations (None entries)
    valid_importances = [imp for imp in null_importances if imp is not None]
    n_failed = len(null_importances) - len(valid_importances)
    n_valid = len(valid_importances)
    if n_valid == 0:
        raise RuntimeError(
            "All null models failed to fit. Cannot compute PIMP p-values. "
            "This typically indicates the model cannot be cloned and refitted "
            "on permuted data."
        )
    _MIN_NULL_PERMUTATIONS = 10
    if n_valid < _MIN_NULL_PERMUTATIONS and n_failed > 0:
        raise RuntimeError(
            f"Only {n_valid}/{n_permutations} null models succeeded. "
            f"PIMP requires at least {_MIN_NULL_PERMUTATIONS} valid "
            f"permutations for meaningful p-values. Check model "
            f"compatibility with sklearn's clone()."
        )
    if n_failed > 0:
        warnings.warn(
            f"{n_failed}/{n_permutations} null models failed to fit. "
            f"P-values are based on {n_valid} permutations.",
            MetricFailureWarning, stacklevel=3,
        )

    return np.array(valid_importances)  # (n_valid, n_features)


def _compute_pimp_pvalues(observed, null_matrix, features, p_adjust):
    """Compute empirical p-values and apply multiplicity correction.

    Returns
    -------
    pd.DataFrame
        Columns: feature, observed_importance, null_mean, null_std,
        p_value, p_adjusted. Sorted by p_value ascending.
    """
    n_valid = null_matrix.shape[0]
    p_values = (np.sum(null_matrix >= observed, axis=0) + 1) / (n_valid + 1)
    p_adjusted = _adjust_p_values(p_values, method=p_adjust)

    result = pd.DataFrame({
        "feature": features,
        "observed_importance": observed,
        "null_mean": null_matrix.mean(axis=0),
        "null_std": null_matrix.std(axis=0),
        "p_value": p_values,
        "p_adjusted": p_adjusted,
    })
    return result.sort_values("p_value", ignore_index=True)


def compute_pimp(
    pipeline_or_model,
    X_train: pd.DataFrame,
    y_train: SurvivalArray,
    X_test: pd.DataFrame,
    y_test: SurvivalArray,
    n_permutations: int = 100,
    n_importance_repeats: int = 10,
    p_adjust: str = "bh",
    random_state: int = 42,
    n_jobs: int = 1,
    show_progress: bool = True,
    categorical_groups=None,
) -> pd.DataFrame:
    """Compute PIMP feature importance p-values (Altmann et al. 2010).

    Tests H₀: feature has no association with the survival response, by
    building a null distribution of importance scores from models trained
    on permuted response data.

    Parameters
    ----------
    pipeline_or_model : Pipeline or estimator
        Fitted pipeline or model. Will be cloned for each null iteration.
    X_train : DataFrame
        Training features.
    y_train : SurvivalArray
        Training survival outcomes.
    X_test : DataFrame
        Test features for importance evaluation.
    y_test : SurvivalArray
        Test survival outcomes.
    n_permutations : int, default=100
        Number of response permutations (null models to fit).
    n_importance_repeats : int, default=10
        Permutation importance repeats per model evaluation.
    p_adjust : str, default="bh"
        Multiplicity correction: "bh" (Benjamini-Hochberg FDR),
        "holm" (FWER), "bonferroni", or "none".
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs for null model fitting.
    show_progress : bool, default=True
        Print progress updates.
    categorical_groups : dict of {str: list of str}, optional
        Mapping from original categorical feature name to its encoded
        column names.  When provided, columns in each group are permuted
        jointly and a single importance score is reported per group.

    Returns
    -------
    DataFrame
        Columns: feature, observed_importance, null_mean, null_std,
        p_value, p_adjusted. Sorted by p_value ascending.
    """
    _validate_pimp_inputs(pipeline_or_model, n_permutations, n_importance_repeats)

    rng = np.random.RandomState(random_state)

    # Validate and resolve categorical groups
    valid_groups = None
    if categorical_groups:
        valid_groups = _validate_categorical_groups(
            categorical_groups, X_test.columns.tolist()
        )
        if not valid_groups:
            valid_groups = None

    # 1. Observed importance
    if show_progress:
        print("PIMP: computing observed importance...", end="", flush=True)
    t0 = time.perf_counter()
    if valid_groups:
        obs_result = _grouped_permutation_importance(
            pipeline_or_model, X_test, y_test, valid_groups,
            n_repeats=n_importance_repeats, random_state=random_state,
        )
        features = obs_result.feature_names
    else:
        obs_result = permutation_importance(
            pipeline_or_model, X_test, y_test,
            n_repeats=n_importance_repeats, random_state=random_state, n_jobs=1,
        )
        features = X_test.columns.tolist()
    observed = obs_result.importances_mean
    if show_progress:
        print(f" done ({_format_elapsed(time.perf_counter() - t0)}).")

    # 2. Null distribution: permute y_train, refit, compute importance
    shuffle_seeds = rng.randint(0, 2**31, size=n_permutations)
    perm_ys = [_permute_y(y_train, np.random.RandomState(s)) for s in shuffle_seeds]
    importance_seeds = rng.randint(0, 2**31, size=n_permutations).tolist()

    t0_null = time.perf_counter()
    null_matrix = _fit_null_models(
        pipeline_or_model, X_train, perm_ys, X_test, y_test,
        n_importance_repeats, importance_seeds, n_jobs, n_permutations,
        show_progress, categorical_groups=valid_groups,
    )

    if show_progress:
        print(f"PIMP: null models done ({_format_elapsed(time.perf_counter() - t0_null)}).")

    # 3. Empirical p-values + multiplicity correction
    return _compute_pimp_pvalues(observed, null_matrix, features, p_adjust)
