"""Feature selection for survival analysis.

Includes univariate Cox ranking (TopKCoxSelector), multicollinearity
removal (VIFSelector, CorrelationSelector), and penalized Cox selectors
(CoxLassoSelector, CoxNetSelector).
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sksurv.linear_model import CoxnetSurvivalAnalysis
from joblib import Parallel, delayed

from mlsurv._shared import select_alpha_by_cv_concordance
from mlsurv._warnings import (
    FitFailureWarning, InternalWarning, MethodologicalWarning, SuggestionWarning,
)
from ..utils import (
    SurvivalArray,
    MIN_FEATURES_FOR_PARALLEL,
    resolve_joblib_backend,
    resolve_parallel_n_jobs,
)
from ._cox_fit import (
    MIN_COX_ROWS as _MIN_COX_ROWS,
    fit_univariate_cox,
)

logger = logging.getLogger(__name__)


class FeatureSelectorMixin:
    """Shared transform/get_support/get_feature_names_out for feature selectors.

    Subclasses must set ``selected_features_`` and ``feature_names_in_``
    during ``fit()``.
    """

    def transform(self, X):
        """Select fitted features from *X*."""
        check_is_fitted(self, ['selected_features_', 'feature_names_in_'])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X[self.selected_features_]

    def get_support(self, indices=False):
        """Boolean mask or index list of selected features."""
        check_is_fitted(self, ['selected_features_', 'feature_names_in_'])
        if indices:
            return [
                list(self.feature_names_in_).index(f)
                for f in self.selected_features_
            ]
        return np.array([f in self.selected_features_ for f in self.feature_names_in_])

    def get_feature_names_out(self, input_features=None):
        """Selected feature names (sklearn 1.0+ compatibility)."""
        check_is_fitted(self, ['selected_features_'])
        return self.selected_features_


def _is_cox_based(selector):
    """Check if a selector uses univariate Cox fitting.

    Cox-based selectors should be pre-fit on raw data (before imputation)
    so that per-feature casewise deletion preserves the true signal.
    """
    if isinstance(selector, TopKCoxSelector):
        return True
    if isinstance(selector, CorrelationSelector) and selector.strategy == 'cox':
        return True
    return False


class TopKCoxSelector(FeatureSelectorMixin, BaseEstimator, TransformerMixin):
    """
    Select top K features by univariate Cox p-value.

    Computes feature rankings during fit() using only the training data,
    preventing data leakage when used inside cross-validation.

    For each feature, fits a univariate Cox proportional hazards model and
    extracts the p-value. Features are ranked by ascending p-value (most
    significant first) and the top K are selected.

    Parameters
    ----------
    k : int, 'auto', or 'tune', default='auto'
        Number of features to select.

        - ``'auto'``: Harrell's rule, ``min(n_features, max(1, n_events // 15))``.
          Note that on high-event cohorts this can equal ``n_features`` (no
          selection occurs).
        - an ``int``: keep exactly that many top-ranked features.
        - ``'tune'``: let Optuna search the feature count during
          ``learner.tune(...)`` / ``learner.run(..., tune=True)``. The tuner
          applies the default range ``(5, n_features)`` (clamped) unless you
          supply an explicit ``search_spaces={'feature_selector': {'k': (lo, hi)}}``.
          Fitting a selector with ``k='tune'`` outside of tuning raises
          ``ValueError``.

        Harrell, F.E. (2015). *Regression Modeling Strategies*, 2nd ed.,
        Springer. Chapter 4 recommends n_predictors < m/15 where m is the
        effective sample size (number of events for survival/time-to-event data).
    penalizer : float, default=0.01
        L2 regularization for Cox fitting. Applied when initial fit fails
        to improve convergence.
    verbose : bool, default=False
        If True, print per-feature fitting details to stdout.
        A summary warning is always issued via ``warnings.warn``
        when any features fail, regardless of this setting.
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

    Attributes
    ----------
    effective_k_ : int
        Actual number of features selected after fit(). For ``k='auto'``,
        this is ``min(n_features, max(1, n_events // 15))``.
        For explicit ``k``, this is ``min(k, n_features)``.
    sorted_features_ : list of tuples
        Feature rankings after fit(): [(name, pval), ...] sorted by p-value.
    selected_features_ : list
        Names of selected features after fit().
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray
        Feature names seen during fit.

    Examples
    --------
    >>> from mlsurv.preprocessing import TopKCoxSelector
    >>> selector = TopKCoxSelector()  # k='auto' by default
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>> print(selector.effective_k_)
    13
    >>> print(selector.selected_features_)
    ['age', 'stage', 'grade', ...]

    Notes
    -----
    The selector handles common CoxPH convergence issues by trying
    up to three fit strategies in sequence (unpenalized, penalized,
    reduced step size). Any failure -- convergence warning, numerical
    error, or other exception -- triggers the next strategy.
    Constant or near-constant columns are skipped before fitting.
    Features that exhaust all strategies are assigned p=inf.

    References
    ----------
    - Harrell, F.E. (2015). *Regression Modeling Strategies*, 2nd ed.,
      Springer. Chapter 4: n_predictors < m/15 rule.
    - lifelines convergence troubleshooting:
      https://lifelines.readthedocs.io/en/stable/Examples.html
    """

    def __init__(self, k='auto', penalizer=0.01, verbose=False, n_jobs=1, force_parallel=False):
        if k not in ('auto', 'tune'):
            if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
                raise ValueError(
                    f"k must be 'auto', 'tune', or an integer >= 1, got {k!r}"
                )
        self.k = k
        self.penalizer = penalizer
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.force_parallel = force_parallel

    def fit(self, X, y: SurvivalArray):
        """
        Rank features by Cox p-value and select top K.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : SurvivalArray
            Survival outcomes with 'event' and 'time' fields.

        Returns
        -------
        self : TopKCoxSelector
            Fitted selector.

        Notes
        -----
        CRITICAL: This computes feature rankings on the provided data only.
        When used inside a CV loop or Pipeline, rankings are computed
        separately for each fold, preventing data leakage.

        NaN values are handled via per-feature casewise deletion: for each
        feature, rows where that feature is NaN are dropped before fitting
        the univariate Cox model.
        """
        # Validate input
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_features = X.shape[1]
        self.n_features_in_ = n_features
        self.feature_names_in_ = np.array(X.columns)

        # Resolve effective_k_
        if self.k == 'tune':
            raise ValueError(
                "k='tune' means the number of features must be searched by the "
                "tuner, but this selector was fit with k still set to 'tune'. "
                "Run under tuning (learner.tune(...) or "
                "learner.run(..., tune=True)), or set k to 'auto' or an integer."
            )
        if self.k == 'auto':
            n_events = int(y['event'].sum())
            # Harrell's rule: n_predictors < m/15 (Regression Modeling Strategies, 2nd ed.)
            # where m = number of events for survival data
            harrell_k = n_events // 15
            self.effective_k_ = min(n_features, max(1, harrell_k))
            if harrell_k < 1:
                warnings.warn(
                    f"k='auto' resolved to effective_k={self.effective_k_} "
                    f"({n_events} events, Harrell m/15={harrell_k}). "
                    f"With < 15 events, feature selection is highly constrained.",
                    MethodologicalWarning,
                )
            if self.verbose:
                print(f"  k='auto' resolved to {self.effective_k_} "
                      f"(Harrell m/15: {n_events} events // 15 = {n_events // 15}, "
                      f"n_features = {n_features})")
        else:
            self.effective_k_ = min(self.k, n_features)
            if self.verbose and self.k > n_features:
                print(f"  k={self.k} clamped to {n_features} (n_features)")
            elif self.verbose and self.effective_k_ == n_features:
                print(f"  k={self.k} == n_features={n_features}: "
                      f"selector is a no-op")

        # Compute rankings on training data only
        self.sorted_features_ = _sort_features(
            X, y,
            penalizer=self.penalizer,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            force_parallel=self.force_parallel
        )

        # Select top K features
        feature_names = [f[0] for f in self.sorted_features_]
        self.selected_features_ = feature_names[:self.effective_k_]

        return self

    def __repr__(self):
        # Always show k
        parts = [f"k={self.k!r}"]

        # After fit, show effective_k
        if hasattr(self, 'effective_k_'):
            parts.append(f"effective_k={self.effective_k_}")

        # Show other params only when non-default
        if self.penalizer != 0.01:
            parts.append(f"penalizer={self.penalizer!r}")
        if self.verbose:
            parts.append(f"verbose={self.verbose!r}")
        if self.n_jobs != 1:
            parts.append(f"n_jobs={self.n_jobs!r}")
        if self.force_parallel:
            parts.append(f"force_parallel={self.force_parallel!r}")

        return f"TopKCoxSelector({', '.join(parts)})"

def _resolve_n_jobs(n_jobs, n_features, force_parallel, verbose):
    """Resolve effective n_jobs using ``MIN_FEATURES_FOR_PARALLEL`` as threshold."""
    return resolve_parallel_n_jobs(
        n_jobs, n_features, MIN_FEATURES_FOR_PARALLEL, force_parallel, verbose,
    )


def _format_failure_counts(n_skipped, n_failed):
    """Build a human-readable summary of skipped/failed feature counts.

    Returns
    -------
    str
        e.g. ``"3 skipped (constant/near-zero variance) and 2 failed Cox fitting"``
    """
    parts = []
    if n_skipped:
        parts.append(f"{n_skipped} skipped (constant/near-zero variance)")
    if n_failed:
        parts.append(f"{n_failed} failed Cox fitting")
    return " and ".join(parts)


def _warn_cox_errors(errors, stacklevel=3):
    """Emit a single summary warning for unexpected Cox fitting errors.

    Parameters
    ----------
    errors : list of (col, detail) tuples
        Features that raised unexpected errors during Cox fitting.
    stacklevel : int
        Stack depth for the warning.
    """
    if not errors:
        return
    error_msgs = [detail for _, detail in errors]
    warnings.warn(
        f"{len(errors)} feature(s) encountered unexpected errors "
        f"during Cox fitting:\n" + "\n".join(f"  - {m}" for m in error_msgs),
        SuggestionWarning,
        stacklevel=stacklevel,
    )


def _check_and_warn_failures(results, n_total, verbose, *,
                              all_failed_msg, partial_fail_suffix,
                              warn_stacklevel):
    """Tally skipped/failed/error results; raise if all failed, warn if some.

    Parameters
    ----------
    results : list of (col, pval, status, error_detail)
        Per-feature fit results.
    n_total : int
        Total number of features attempted.
    verbose : bool
        If True, print verbose details for failed features.
    all_failed_msg : str
        RuntimeError message when every feature fails.
    partial_fail_suffix : str
        Appended to the warning when some (not all) features fail.
    warn_stacklevel : int
        ``stacklevel`` for ``warnings.warn`` calls.
    """
    skipped = [col for col, _, status, _ in results if status == 'skipped']
    failed = [col for col, _, status, _ in results if status == 'failed']
    errors = [(col, detail) for col, _, status, detail in results if status == 'error']
    n_bad = len(skipped) + len(failed) + len(errors)

    if n_bad == n_total:
        raise RuntimeError(all_failed_msg)

    # Surface unexpected errors from workers in the main process
    _warn_cox_errors(errors, stacklevel=warn_stacklevel + 1)

    n_skip_fail = len(skipped) + len(failed)
    if n_skip_fail:
        summary = _format_failure_counts(len(skipped), len(failed))
        msg = (
            f"{summary}, out of {n_total} features. "
            f"{partial_fail_suffix}"
        )
        if not verbose:
            msg += " Use verbose=True for details."
        warnings.warn(msg, InternalWarning, stacklevel=warn_stacklevel)
        if verbose and failed:
            print(f"  Failed features: "
                  f"{failed[:5]}{'...' if len(failed) > 5 else ''}")


def _fit_cox_features(X, y, columns, penalizer, verbose, n_jobs, force_parallel,
                       *, all_failed_msg, partial_fail_suffix, warn_stacklevel):
    """Fit univariate Cox models for *columns* and validate results.

    Resolves parallelism, dispatches fitting, and calls
    ``_check_and_warn_failures`` to raise/warn as needed.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (all columns available; only *columns* are fitted).
    y : SurvivalArray
        Survival outcome.
    columns : list of str
        Feature names to fit.
    penalizer : float
        L2 regularization for fallback fit.
    verbose : bool
        Print per-feature fitting details (suppressed in parallel).
    n_jobs : int
        Number of parallel jobs.
    force_parallel : bool
        Skip auto-disable threshold check.
    all_failed_msg : str
        RuntimeError message when every feature fails.
    partial_fail_suffix : str
        Warning tail when some features fail.
    warn_stacklevel : int
        ``stacklevel`` for warnings (accounts for caller depth).

    Returns
    -------
    list of (col, pval, status, error_detail)
    """
    effective_n_jobs = _resolve_n_jobs(n_jobs, len(columns), force_parallel, verbose)

    def _fit_single(col):
        # Suppress per-feature verbose in parallel (output interleaves)
        v = verbose and (effective_n_jobs == 1)
        r = fit_univariate_cox(X[col], y, col, penalizer=penalizer, verbose=v)
        return (col, r['p_value'], r['status'], r['error_detail'])

    if effective_n_jobs == 1:
        results = [_fit_single(col) for col in columns]
    else:
        backend = resolve_joblib_backend("loky")
        results = Parallel(n_jobs=effective_n_jobs, backend=backend)(
            delayed(_fit_single)(col) for col in columns
        )

    _check_and_warn_failures(
        results, len(columns), verbose,
        all_failed_msg=all_failed_msg,
        partial_fail_suffix=partial_fail_suffix,
        warn_stacklevel=warn_stacklevel,
    )

    # Aggregate casewise deletion summary in main process.
    # Per-feature warnings inside _fit_univariate_cox would be lost in
    # parallel workers (loky), so we emit one summary here instead.
    nan_counts = {col: int(X[col].isna().sum()) for col in columns}
    features_with_nan = {col: n for col, n in nan_counts.items() if n > 0}
    if features_with_nan:
        n_rows = len(X)
        max_dropped = max(features_with_nan.values())
        min_valid = n_rows - max_dropped
        low_row_cols = [col for col, n in features_with_nan.items()
                        if n_rows - n < _MIN_COX_ROWS]

        msg = (
            f"Casewise deletion: {len(features_with_nan)}/{len(columns)} features "
            f"had NaN rows dropped before univariate Cox fitting "
            f"(max {max_dropped} rows dropped, smallest n={min_valid})."
        )
        if low_row_cols:
            names = ", ".join(low_row_cols[:5])
            suffix = f" (and {len(low_row_cols) - 5} more)" if len(low_row_cols) > 5 else ""
            msg += (
                f" {len(low_row_cols)} feature(s) had fewer than {_MIN_COX_ROWS} "
                f"non-NaN rows — Cox p-values may be unreliable: {names}{suffix}."
            )
        warnings.warn(msg, MethodologicalWarning, stacklevel=warn_stacklevel)

    return results


def _sort_features(X, y: SurvivalArray, penalizer=0.01, verbose=False, n_jobs=1, force_parallel=False):
    """
    Rank features by univariate Cox p-value with robust error handling.

    Fits a separate CoxPH model for each feature and extracts the p-value.
    Features that fail to fit are assigned p=inf (worst rank).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : SurvivalArray
        Survival outcome with 'event' and 'time' fields.
    penalizer : float, default=0.01
        L2 regularization to improve convergence. Applied when initial
        fit fails. Set to 0 to disable penalized fallback; step-size reduction still applies.
    verbose : bool, default=False
        If True, print per-feature fitting details to stdout.
        A summary warning is always issued via ``warnings.warn``
        when any features fail, regardless of this setting.
    n_jobs : int, default=1
        Number of parallel jobs for fitting.
        - 1: Sequential execution (default)
        - -1: Use all available CPU cores
        - n > 1: Use n CPU cores
        Note: Parallelization is auto-disabled when n_features < 15.
    force_parallel : bool, default=False
        If True, skip the auto-disable threshold check and use the
        specified n_jobs regardless of feature count.

    Returns
    -------
    list of tuples
        [(feature_name, p_value), ...] sorted by ascending p-value.
    """
    columns = list(X.columns)
    results = _fit_cox_features(
        X, y, columns, penalizer, verbose, n_jobs, force_parallel,
        all_failed_msg=(
            f"All {len(columns)} features failed Cox fitting. "
            f"Cannot rank features. Check data quality "
            f"(constant columns, extreme values, too few events)."
        ),
        partial_fail_suffix="These received p=inf (worst rank).",
        # warn -> _check_and_warn_failures -> _fit_cox_features -> here -> fit -> user
        warn_stacklevel=5,
    )
    pairs = [(col, pval) for col, pval, _, _ in results]
    return sorted(pairs, key=lambda x: x[1])


class VIFSelector(FeatureSelectorMixin, BaseEstimator, TransformerMixin):
    """
    Select features by removing those with high Variance Inflation Factor (VIF).

    VIF measures multicollinearity between features. Features with VIF > threshold
    indicate high correlation with other features and are iteratively removed.

    Parameters
    ----------
    threshold : float, default=10.0
        Maximum VIF allowed. Features with VIF above this are removed.
        Common thresholds: 5 (strict), 10 (moderate), 20 (lenient).
    min_features : int, default=2
        Minimum number of features to keep. Stops removing even if
        some features still exceed threshold.
    verbose : bool, default=False
        If True, print removal progress.

    Attributes
    ----------
    removed_features_ : list
        Features removed (in order of removal).
    vif_scores_ : dict
        Final VIF scores for remaining features.
    selected_features_ : list
        Features kept after selection.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray
        Feature names seen during fit.

    Examples
    --------
    >>> from mlsurv.preprocessing import VIFSelector
    >>> selector = VIFSelector(threshold=10, verbose=True)
    >>> X_selected = selector.fit_transform(X)
    >>> print(f"Removed: {selector.removed_features_}")

    Notes
    -----
    VIF is computed as 1 / (1 - R²) where R² is the coefficient of determination
    from regressing each feature on all other features. VIF = 1 means no
    correlation, VIF > 10 suggests high multicollinearity.
    """

    def __init__(self, threshold=10.0, min_features=2, verbose=False):
        self.threshold = threshold
        self.min_features = min_features
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Identify features to remove based on VIF.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : ignored
            Not used, present for API compatibility.

        Returns
        -------
        self : VIFSelector
            Fitted selector.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)

        # Iteratively remove high-VIF features
        remaining_features = list(X.columns)
        self.removed_features_ = []

        while len(remaining_features) > self.min_features:
            vif_scores = self._compute_vif(X[remaining_features])

            # Find max VIF
            max_vif_feature = max(vif_scores, key=vif_scores.get)
            max_vif = vif_scores[max_vif_feature]

            if max_vif <= self.threshold:
                break  # All remaining features pass threshold

            # Remove the feature with highest VIF
            remaining_features.remove(max_vif_feature)
            self.removed_features_.append(max_vif_feature)

            if self.verbose:
                print(f"Removed '{max_vif_feature}' (VIF={max_vif:.2f})")

        # Store final VIF scores and selected features
        self.vif_scores_ = self._compute_vif(X[remaining_features])
        self.selected_features_ = remaining_features

        if self.verbose:
            print(f"Final: kept {len(self.selected_features_)} features, "
                  f"removed {len(self.removed_features_)}")

        return self

    def _compute_vif(self, X):
        """
        Compute VIF for all features.

        VIF = 1 / (1 - R²) where R² is from regressing each feature
        on all other features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        dict
            {feature_name: vif_score} for all features.
        """
        from sklearn.linear_model import LinearRegression

        vif_scores = {}
        features = list(X.columns)

        for i, feature in enumerate(features):
            # Regress this feature on all other features
            other_features = [f for j, f in enumerate(features) if j != i]

            if len(other_features) == 0:
                vif_scores[feature] = 1.0  # No other features to compare
                continue

            y_feat = X[feature].values
            X_others = X[other_features].values

            # Handle constant features
            if np.std(y_feat) < 1e-10:
                vif_scores[feature] = float('inf')
                continue

            lr = LinearRegression()
            lr.fit(X_others, y_feat)
            r_squared = lr.score(X_others, y_feat)

            # VIF = 1 / (1 - R²); R²=1 means perfect collinearity
            if r_squared >= 1.0:
                vif_scores[feature] = float('inf')
            else:
                vif_scores[feature] = 1.0 / (1.0 - r_squared)

        return vif_scores

class CorrelationSelector(FeatureSelectorMixin, BaseEstimator, TransformerMixin):
    """
    Select features by removing one from each highly correlated pair.

    When two features are highly correlated, keeping both adds little
    information but increases model complexity and can cause issues
    with feature importance interpretation.

    Parameters
    ----------
    threshold : float, default=0.8
        Correlation threshold (absolute value). Pairs with |corr| >= threshold
        are considered highly correlated.
    strategy : str, default='frequency'
        How to decide which feature to keep from a correlated pair:
        - 'frequency': Keep the feature involved in fewer correlated pairs
          (faster, no survival data needed)
        - 'cox': Keep the feature with lower univariate Cox p-value
          (requires y, keeps more survival-relevant features)
    penalizer : float, default=0.01
        L2 regularization for Cox fitting (only used with strategy='cox').
    verbose : bool, default=False
        If True, print removal progress.
    n_jobs : int, default=1
        Number of parallel jobs for Cox fitting (only used with strategy='cox').
    force_parallel : bool, default=False
        If True, skip the auto-disable threshold check and use the
        specified n_jobs regardless of feature count.

    Attributes
    ----------
    correlation_pairs_ : list of tuple
        Detected correlated pairs: [(feature1, feature2, correlation), ...].
    removed_features_ : list
        Features removed.
    selected_features_ : list
        Features kept after selection.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray
        Feature names seen during fit.

    Examples
    --------
    >>> from mlsurv.preprocessing import CorrelationSelector
    >>> # Remove one from each correlated pair (keep less frequent)
    >>> selector = CorrelationSelector(threshold=0.8, strategy='frequency')
    >>> X_selected = selector.fit_transform(X)
    >>>
    >>> # Keep more survival-relevant feature from each pair
    >>> selector = CorrelationSelector(threshold=0.8, strategy='cox')
    >>> X_selected = selector.fit_transform(X, y)

    Notes
    -----
    With strategy='frequency', the feature appearing in more correlated pairs
    is removed first. This tends to remove "hub" features that correlate with
    many others.

    With strategy='cox', the feature with higher p-value (less survival-relevant)
    is removed from each pair. This requires survival outcomes (y) during fit.
    """

    def __init__(self, threshold=0.8, strategy='frequency', penalizer=0.01,
                 verbose=False, n_jobs=1, force_parallel=False):
        self.threshold = threshold
        self.strategy = strategy
        self.penalizer = penalizer
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.force_parallel = force_parallel

    def fit(self, X, y=None):
        """
        Identify features to remove based on correlation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : SurvivalArray, optional
            Survival outcomes (required for strategy='cox').

        Returns
        -------
        self : CorrelationSelector
            Fitted selector.

        Raises
        ------
        ValueError
            If ``strategy='cox'`` but no survival outcomes ``y`` are
            supplied, or if ``strategy`` is not a recognized option.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)

        # Detect correlated pairs (reuse from evaluation.importance)
        from mlsurv.evaluation.importance import detect_correlated_features
        self.correlation_pairs_ = detect_correlated_features(X, self.threshold)

        if not self.correlation_pairs_:
            # No correlated pairs found
            self.removed_features_ = []
            self.selected_features_ = list(X.columns)
            if self.verbose:
                print("No correlated pairs found above threshold")
            return self

        # Decide which features to remove
        if self.strategy == 'frequency':
            self.removed_features_ = self._remove_by_frequency(X)
        elif self.strategy == 'cox':
            if y is None:
                raise ValueError("strategy='cox' requires y (survival outcomes)")
            self.removed_features_ = self._remove_by_cox(X, y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. "
                             f"Options: 'frequency', 'cox'")

        self.selected_features_ = [
            f for f in X.columns if f not in self.removed_features_
        ]

        if self.verbose:
            print(f"Removed {len(self.removed_features_)} features, "
                  f"kept {len(self.selected_features_)}")

        return self

    def _remove_by_frequency(self, X):
        """Remove features that appear in more correlated pairs."""
        from collections import Counter

        # Count how many pairs each feature appears in
        feature_counts = Counter()
        for f1, f2, _ in self.correlation_pairs_:
            feature_counts[f1] += 1
            feature_counts[f2] += 1

        removed = set()
        for f1, f2, corr in self.correlation_pairs_:
            # Skip if either already removed
            if f1 in removed or f2 in removed:
                continue

            # Remove the one that appears in more pairs
            if feature_counts[f1] > feature_counts[f2]:
                removed.add(f1)
                if self.verbose:
                    print(f"Removed '{f1}' (corr={corr:.3f} with '{f2}', "
                          f"appears in {feature_counts[f1]} pairs)")
            else:
                removed.add(f2)
                if self.verbose:
                    print(f"Removed '{f2}' (corr={corr:.3f} with '{f1}', "
                          f"appears in {feature_counts[f2]} pairs)")

        return list(removed)

    def _remove_by_cox(self, X, y):
        """Remove the less survival-relevant feature from each pair."""
        features_in_pairs = {f for f1, f2, _ in self.correlation_pairs_ for f in (f1, f2)}
        results = _fit_cox_features(
            X, y, list(features_in_pairs), self.penalizer, self.verbose,
            self.n_jobs, self.force_parallel,
            all_failed_msg=(
                f"All {len(features_in_pairs)} features in correlated pairs failed Cox "
                f"fitting. Cannot make survival-relevant removal decisions. "
                f"Consider using strategy='frequency' instead, or check "
                f"data quality (constant columns, extreme values, too few events)."
            ),
            partial_fail_suffix="These are features in correlated pairs; removal decisions for them may be arbitrary.",
            # warn -> _check_and_warn_failures -> _fit_cox_features -> here -> fit -> user
            warn_stacklevel=5,
        )
        pvalues = {col: pval for col, pval, _, _ in results}

        removed = set()
        for f1, f2, corr in self.correlation_pairs_:
            if f1 in removed or f2 in removed:
                continue
            if pvalues[f1] > pvalues[f2]:
                to_remove, to_keep = f1, f2
            else:
                to_remove, to_keep = f2, f1
            removed.add(to_remove)
            if self.verbose:
                print(f"Removed '{to_remove}' (p={pvalues[to_remove]:.4f}) over "
                      f"'{to_keep}' (p={pvalues[to_keep]:.4f}), corr={corr:.3f}")
        return list(removed)


# ============================================================================
# Penalized Cox Feature Selectors
# ============================================================================


class _PenalizedCoxSelector(FeatureSelectorMixin, BaseEstimator, TransformerMixin):
    """Base class for penalized Cox feature selectors.

    Fits a CoxNet (elastic net penalized Cox) model on training data and
    selects features with non-zero coefficients.  L1 penalization drives
    redundant or irrelevant coefficients to zero, performing multivariate
    feature selection that accounts for inter-feature relationships.

    This is the survival analog of sklearn's ``SelectFromModel(Lasso())``.

    Parameters
    ----------
    alpha : {'auto', 'min'} or float, default='auto'
        Regularization strength selection strategy:

        - ``'auto'``: Use K-fold CV to select the alpha with the highest
          mean concordance index on held-out folds.
        - ``'min'``: Use the smallest alpha from the regularization path
          (least regularization, most features retained).
        - ``float``: Use a specific alpha value.
    n_alphas : int, default=100
        Number of alphas in the regularization path.
    alpha_min_ratio : float, default=0.01
        Ratio of the smallest to the largest alpha in the path.
    cv_folds : int, default=5
        Number of CV folds for alpha selection when ``alpha='auto'``.
    verbose : bool, default=False
        Print fitting details to stdout.
    random_state : int or None, default=None
        Seed for the K-fold split used when ``alpha='auto'``. Only the fold
        assignment is randomized (shuffle=True); with a fixed seed the selected
        alpha and features are reproducible. ``None`` gives genuinely random
        folds (the sklearn default). SurvivalLearner passes the configured
        ``setup(random_state=...)`` here automatically.

    Attributes
    ----------
    selected_features_ : list
        Feature names with non-zero coefficients after fit.
    feature_names_in_ : ndarray
        All input feature names.
    n_features_in_ : int
        Number of input features.
    selected_alpha_ : float
        Alpha value used for feature selection.
    coefficients_ : pd.Series
        All coefficients indexed by feature name.
    n_selected_ : int
        Number of selected features.

    Notes
    -----
    Subclasses must set ``self._l1_ratio`` before calling ``super().fit()``.

    References
    ----------
    - Tibshirani, R. (1997). The lasso method for variable selection in the
      Cox model. *Statistics in Medicine*, 16(4), 385-395.
    - Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2011).
      Regularization Paths for Cox's Proportional Hazards Model via
      Coordinate Descent. *Journal of Statistical Software*, 39(5), 1-13.
    """

    def __init__(self, alpha='auto', n_alphas=100, alpha_min_ratio=0.01,
                 cv_folds=5, verbose=False, random_state=None):
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y: SurvivalArray):
        """Fit penalized Cox model and select features with non-zero coefficients.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Feature matrix.
        y : SurvivalArray
            Survival outcomes with 'event' and 'time' fields.

        Returns
        -------
        self
            Fitted selector.

        Raises
        ------
        ValueError
            If the regularization path produces no alphas, or if ``alpha``
            is not ``'auto'``, ``'min'``, or a float.
        """
        # Sync _l1_ratio for sklearn set_params() compatibility
        if hasattr(self, 'l1_ratio'):
            self._l1_ratio = self.l1_ratio

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)

        # Fit regularization path
        alphas = self._fit_regularization_path(X, y)

        if len(alphas) == 0:
            raise ValueError(
                f"Regularization path produced no alphas for "
                f"l1_ratio={self._l1_ratio}. Check that the data is valid "
                f"(sufficient events, non-constant features)."
            )

        # Select alpha
        if isinstance(self.alpha, float) and not isinstance(self.alpha, bool):
            self.selected_alpha_ = self.alpha
        elif self.alpha == 'auto':
            self.selected_alpha_ = self._select_alpha_cv(X, y, alphas)
        elif self.alpha == 'min':
            self.selected_alpha_ = alphas[-1]
        else:
            raise ValueError(
                f"alpha must be 'auto', 'min', or a float, got {self.alpha!r}"
            )

        # Fit final model with selected alpha and extract coefficients
        coef = self._fit_and_extract_coef(X, y)

        # Select features with non-zero coefficients
        nonzero_mask = np.abs(coef) > 0
        selected = list(self.feature_names_in_[nonzero_mask])

        # Handle all-zero edge case: fall back to alpha='min'
        if len(selected) == 0 and self.alpha != 'min':
            warnings.warn(
                f"All coefficients are zero at alpha={self.selected_alpha_:.6f} "
                f"(over-regularization). Falling back to alpha='min' "
                f"(least regularization).",
                SuggestionWarning,
                stacklevel=2,
            )
            self.selected_alpha_ = alphas[-1]
            coef = self._fit_and_extract_coef(X, y)
            nonzero_mask = np.abs(coef) > 0
            selected = list(self.feature_names_in_[nonzero_mask])

        # If still all zeros even at alpha='min', keep all features
        if len(selected) == 0:
            warnings.warn(
                "All coefficients are zero even at minimum alpha. "
                "Selecting all features (no reduction). This may indicate "
                "the data has too few events or features lack variance.",
                FitFailureWarning,
                stacklevel=2,
            )
            selected = list(self.feature_names_in_)

        self.coefficients_ = pd.Series(coef, index=self.feature_names_in_)
        self.selected_features_ = selected
        self.n_selected_ = len(selected)

        if self.verbose:
            print(f"  Selected {self.n_selected_}/{self.n_features_in_} features "
                  f"at alpha={self.selected_alpha_:.6f}")

        return self

    def _fit_regularization_path(self, X, y):
        """Fit the regularization path and return the alpha array."""
        path_model = CoxnetSurvivalAnalysis(
            l1_ratio=self._l1_ratio,
            alpha_min_ratio=self.alpha_min_ratio,
            n_alphas=self.n_alphas,
            fit_baseline_model=False,
        )
        path_model.fit(X, y)
        return path_model.alphas_

    def _fit_and_extract_coef(self, X, y):
        """Fit with selected_alpha_ and return coefficient array."""
        model = CoxnetSurvivalAnalysis(
            l1_ratio=self._l1_ratio,
            alphas=[self.selected_alpha_],
            fit_baseline_model=False,
        )
        model.fit(X, y)
        coef = model.coef_
        if coef.ndim == 2:
            coef = coef[:, -1]
        return coef

    def _select_alpha_cv(self, X, y, alphas):
        """Select the best alpha using K-fold CV on concordance index.

        Pre-validates the fold count, then delegates the fold-scoring loop to
        the shared :func:`mlsurv._shared.select_alpha_by_cv_concordance`
        (mirrored by CoxNetModel._select_alpha_cv in models/coxnet.py). Unlike
        the model path, this selector surfaces a failed-fold warning.
        """
        n_samples = X.shape[0]

        if self.cv_folds < 2:
            warnings.warn(
                f"cv_folds={self.cv_folds} is invalid (must be >= 2). "
                f"Falling back to alpha='min'.",
                FitFailureWarning,
                stacklevel=3,
            )
            return alphas[-1]

        if n_samples < self.cv_folds:
            warnings.warn(
                f"Dataset has only {n_samples} samples, less than "
                f"cv_folds={self.cv_folds}. Falling back to alpha='min'.",
                FitFailureWarning,
                stacklevel=3,
            )
            return alphas[-1]

        alpha, failed_folds = select_alpha_by_cv_concordance(
            X, y, alphas,
            l1_ratio=self._l1_ratio,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            no_valid_scores_stacklevel=4,
        )

        if failed_folds > 0:
            warnings.warn(
                f"Alpha CV: {failed_folds}/{self.cv_folds} folds failed "
                f"during fitting. Alpha selected from "
                f"{self.cv_folds - failed_folds} fold(s).",
                FitFailureWarning,
                stacklevel=3,
            )

        return alpha

    def _repr_parts(self):
        """Build repr parameter list. Subclasses prepend extra parts."""
        parts = [f"alpha={self.alpha!r}"]
        if hasattr(self, 'n_selected_'):
            parts.append(f"n_selected={self.n_selected_}")
        if self.n_alphas != 100:
            parts.append(f"n_alphas={self.n_alphas}")
        if self.cv_folds != 5:
            parts.append(f"cv_folds={self.cv_folds}")
        if self.verbose:
            parts.append(f"verbose={self.verbose!r}")
        return parts

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(self._repr_parts())})"


class CoxLassoSelector(_PenalizedCoxSelector):
    """Select features using L1-penalized Cox regression (Cox-LASSO).

    Fits a LASSO-penalized Cox proportional hazards model and retains
    features with non-zero coefficients.  Pure L1 penalization produces
    sparse solutions, aggressively zeroing out redundant features and
    typically selecting one representative from each correlated group.

    Parameters
    ----------
    alpha : {'auto', 'min'} or float, default='auto'
        Regularization strength selection strategy.
        See :class:`_PenalizedCoxSelector` for details.
    n_alphas : int, default=100
        Number of alphas in the regularization path.
    alpha_min_ratio : float, default=0.01
        Ratio of the smallest to the largest alpha in the path.
    cv_folds : int, default=5
        Number of CV folds for alpha selection when ``alpha='auto'``.
    verbose : bool, default=False
        Print fitting details to stdout.
    random_state : int or None, default=None
        Seed for the K-fold split when ``alpha='auto'``.
        See :class:`_PenalizedCoxSelector` for details.

    Examples
    --------
    >>> from mlsurv.preprocessing import CoxLassoSelector
    >>> selector = CoxLassoSelector()
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>> print(f"Selected {selector.n_selected_}/{selector.n_features_in_} features")

    See Also
    --------
    CoxNetSelector : Elastic net variant that retains correlated feature groups.
    TopKCoxSelector : Univariate Cox p-value ranking (selects top K features).
    """

    _l1_ratio = 1.0


class CoxNetSelector(_PenalizedCoxSelector):
    """Select features using elastic-net-penalized Cox regression.

    Fits an elastic-net-penalized Cox proportional hazards model and retains
    features with non-zero coefficients.  Unlike :class:`CoxLassoSelector`
    (pure L1), the L2 component encourages correlated features to share
    similar coefficients rather than arbitrarily zeroing one out, resulting
    in grouped selection.

    Parameters
    ----------
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter.  ``1.0`` = pure L1 (LASSO),
        ``0.0`` = pure L2 (Ridge, no feature selection).
        Values in ``(0, 1)`` blend both penalties.
    alpha : {'auto', 'min'} or float, default='auto'
        Regularization strength selection strategy.
        See :class:`_PenalizedCoxSelector` for details.
    n_alphas : int, default=100
        Number of alphas in the regularization path.
    alpha_min_ratio : float, default=0.01
        Ratio of the smallest to the largest alpha in the path.
    cv_folds : int, default=5
        Number of CV folds for alpha selection when ``alpha='auto'``.
    verbose : bool, default=False
        Print fitting details to stdout.
    random_state : int or None, default=None
        Seed for the K-fold split when ``alpha='auto'``.
        See :class:`_PenalizedCoxSelector` for details.

    Examples
    --------
    >>> from mlsurv.preprocessing import CoxNetSelector
    >>> selector = CoxNetSelector(l1_ratio=0.8)
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>> print(f"Selected {selector.n_selected_}/{selector.n_features_in_} features")

    See Also
    --------
    CoxLassoSelector : Pure L1 variant for aggressive sparsity.
    TopKCoxSelector : Univariate Cox p-value ranking (selects top K features).
    """

    def __init__(self, l1_ratio=0.5, alpha='auto', n_alphas=100,
                 alpha_min_ratio=0.01, cv_folds=5, verbose=False,
                 random_state=None):
        if not (0 < l1_ratio <= 1.0):
            raise ValueError(
                f"l1_ratio must be in (0, 1] for feature selection "
                f"(l1_ratio=0 is pure Ridge with no sparsity), got {l1_ratio!r}"
            )
        super().__init__(
            alpha=alpha, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio,
            cv_folds=cv_folds, verbose=verbose, random_state=random_state,
        )
        self.l1_ratio = l1_ratio
        self._l1_ratio = l1_ratio

    def _repr_parts(self):
        return [f"l1_ratio={self.l1_ratio!r}"] + super()._repr_parts()
