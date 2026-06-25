"""Feature importance utilities for survival models."""

import contextlib
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

from .._warnings import (
    InternalWarning,
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


def _get_selected_features(model, all_features):
    """Return selected feature names from a pipeline's selector step, or None.

    When the selector received integer column indices (e.g. after a scaler
    that strips names), the indices are mapped back to *all_features*.
    """
    if not (hasattr(model, 'named_steps') and 'selector' in model.named_steps):
        return None
    selector = model.named_steps['selector']
    selected = getattr(selector, 'selected_features_', None)
    if selected is None:
        logger.debug(
            "Pipeline has 'selector' step (%s) but no selected_features_ attribute.",
            type(selector).__name__,
        )
        return None
    selected = list(selected)
    # If the selector saw integer column indices, map them to real names
    if selected and isinstance(selected[0], (int, np.integer)):
        return [all_features[i] for i in selected]
    return selected


def _align_coefficients(coef_values, selected_features, all_features):
    """Align coefficients from selected features to the full feature list.

    Uses NaN for features not in *selected_features* (no coefficient exists).
    *selected_features* must already contain string names matching *all_features*.
    """
    # Always build a fresh Series keyed by the resolved string names
    values = coef_values.values if isinstance(coef_values, pd.Series) else coef_values
    coef_series = pd.Series(values, index=selected_features)
    return coef_series.reindex(all_features).values.tolist()


def _get_coefficients(model, features):
    """Extract model coefficients, aligned to feature order."""
    actual_model = unwrap_pipeline(model)
    selected_features = _get_selected_features(model, features)

    # Only propagate pipeline feature names when there is no selector.
    # With a selector, the inner model already has the correct post-selection
    # feature names; propagating the full set would cause a length mismatch
    # inside get_coefficients().
    if selected_features is None:
        propagate_feature_names(actual_model, model)

    # get_coefficients() is the mlsurv standard (CoxPH, CoxNet, FSSVM)
    if hasattr(actual_model, 'get_coefficients'):
        try:
            coef_series = actual_model.get_coefficients()
        except Exception as e:
            if isinstance(e, (MemoryError, RecursionError)):
                raise
            logger.warning(
                "Failed to extract coefficients from %s: %s",
                type(actual_model).__name__, e,
            )
            return [np.nan] * len(features)
        if coef_series is None:
            return [np.nan] * len(features)
        if len(coef_series) != len(features) and selected_features is not None:
            return _align_coefficients(coef_series, selected_features, features)
        return coef_series.values.tolist()

    # Fallback: direct coef_ access (non-mlsurv sklearn estimators)
    if hasattr(actual_model, 'coef_'):
        try:
            coef = actual_model.coef_
            if coef.ndim == 2:  # CoxNet: (n_features, n_alphas)
                coef = coef[:, 0]
            if len(coef) != len(features) and selected_features is not None:
                return _align_coefficients(coef, selected_features, features)
            return coef.tolist()
        except Exception as e:
            if isinstance(e, (MemoryError, RecursionError)):
                raise
            logger.warning(
                "Failed to extract coefficients from %s: %s",
                type(actual_model).__name__, e,
            )
            return [np.nan] * len(features)

    return [np.nan] * len(features)


def _empty_importance_results():
    """Return empty importance DataFrames when importance is skipped or fails."""
    summary_df = pd.DataFrame(
        columns=['feature', 'coefficient', 'perm_mean', 'perm_std']
    )
    iterations_df = pd.DataFrame(
        columns=['feature', 'iteration', 'importance']
    )
    return summary_df, iterations_df


def _compute_permutation_importance(model, X_test, y_test: SurvivalArray, n_repeats, random_state,
                                    show_progress=True, model_name=None, n_jobs=1,
                                    categorical_groups=None, replacing=False):
    """Compute feature importance with iteration-level detail.

    Parallelism is applied across features via sklearn's
    ``permutation_importance``.  For small feature sets (<~15 features)
    the joblib overhead may negate speedup gains with n_jobs>1.

    When *replacing* is True, the progress line notes that previously computed
    importance results are being replaced.

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

            # Build per-column summary from the standard result
            coefficients_std = _get_coefficients(model, features)
            per_column_df = _build_importance_summary(
                result_std, features, coefficients_std,
            )
        else:
            result = result_std
            result_features = features

        if show_progress:
            print(f" done ({_format_elapsed(time.perf_counter() - t0)}).")

        if use_grouped:
            # Grouped features get NaN coefficients (not aggregatable); reuse
            # `coefficients_std` from the per-column summary above (same model,
            # same features -> identical extraction).
            coefficients = []
            for fname in result_features:
                if fname in valid_groups:
                    coefficients.append(np.nan)
                else:
                    idx = features.index(fname)
                    coefficients.append(coefficients_std[idx])
        else:
            coefficients = _get_coefficients(model, features)

        summary_df = _build_importance_summary(result, result_features, coefficients)
        iterations_df = _build_iterations_dataframe(result, result_features, n_repeats)

    except (MemoryError, RecursionError):
        raise
    except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
        log_metric_failure(logger, "permutation_importance", e,
                           model_name=model_name, warn_user=True)
        if show_progress:
            print(f" failed ({_format_elapsed(time.perf_counter() - t0)}).")
        summary_df, iterations_df = _empty_importance_results()
        per_column_df = None

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
