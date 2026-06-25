"""Evaluation metrics for survival analysis."""

import inspect
import logging
import warnings
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    brier_score,
    integrated_brier_score,
)

from ..constants import DEFAULT_SCORING_METRIC
from ..logging_utils import (
    get_logger,
    log_metric_failure,
    _is_fold_boundary_error,
    _log_once_level,
    _fold_boundary_message,
)
from .._warnings import MethodologicalWarning, SuggestionWarning, _warn_eval_times_failure
from ..utils import SurvivalArray, auto_eval_times, clip_eval_times, clip_eval_times_to_data, drop_nan_rows, truncate_y_to_train_range, unwrap_pipeline

logger = get_logger("metrics")

# Censoring fraction above which Harrell's C-index is materially upward-biased
# and IPCW (Uno's) metrics should be preferred. Shared by the runtime
# heavy-censoring warning (evaluate path) and the report's high-censoring flag.
HIGH_CENSORING_THRESHOLD = 0.80

# Manual dedup for calibration NaN-return warnings — Python's default dedup
# is overridden by simplefilter("always") in the CV/evaluation pipeline.
# Keyed by (metric, model_name) so each model's failure surfaces once
# instead of being silenced by the first failing model. ``model_name=None``
# (legacy/direct calls outside _evaluate_single) preserves the per-metric
# coarse-key behavior.
_calibration_warned: set = set()

# Manual dedup for _resolve_eval_times failures during CV/tuning scoring.
# Same reason as _calibration_warned: simplefilter("always") in the CV
# pipeline disables Python's per-(message, location) dedup, so without
# this set every fold/trial fires a SuggestionWarning. Keyed by
# (exception_type_name, truncated_message) so distinct root causes each
# surface once instead of being collapsed into a single warning. Reset
# by tests via _reset_eval_times_dedup().
_eval_times_warned: set = set()


def _reset_eval_times_dedup() -> None:
    """Clear the _resolve_eval_times warning dedup set (test helper)."""
    _eval_times_warned.clear()


def _emit_or_collect(metric_key, model_name, msg, warnings_out):
    """Emit a calibration warning, or collect it for parent-side emission.

    When *warnings_out* is provided (non-None), append ``(metric_key,
    model_name, msg)`` for the parent process to emit later — this is the
    path used by ``_add_calibration_metrics`` so that warnings escape any
    joblib/loky worker that silences ``warnings.warn``.

    When *warnings_out* is None (direct callers, tests), fall through to
    the existing in-process ``warnings.warn`` + per-(metric, model)
    dedup. Tests in ``test_calibration_metrics.py`` rely on this path.
    """
    if warnings_out is not None:
        warnings_out.append((metric_key, model_name, msg))
        return
    key = (metric_key, model_name)
    if key in _calibration_warned:
        return
    _calibration_warned.add(key)
    warnings.warn(msg, MethodologicalWarning, stacklevel=3)

def _supports_predict_times(predict_fn):
    """Return True if predict_survival_function accepts a ``times`` kwarg."""
    try:
        signature = inspect.signature(predict_fn)
    except (TypeError, ValueError):
        return False
    return 'times' in signature.parameters


def predict_survival_functions(model, X, eval_times=None):
    """Predict survival functions with optional explicit evaluation times.

    Calls ``model.predict_survival_function(X)`` and, when *eval_times* is
    given and the model supports a ``times`` keyword, aligns output to that
    time grid.  Otherwise falls back to the model's default behaviour.

    Parameters
    ----------
    model : BaseSurvivalModel or fitted estimator
        A fitted model with a ``predict_survival_function`` method.
    X : DataFrame
        Feature matrix for prediction.
    eval_times : array-like, optional
        Evaluation time points.  When provided **and** the model's
        ``predict_survival_function`` accepts a ``times`` keyword, the
        output is aligned to these times.  When ``None`` or the model
        does not support ``times``, the model's native time grid is used.

    Returns
    -------
    ndarray, DataFrame, or array of StepFunction
        The return type depends on the underlying model implementation.
        Use :func:`surv_funcs_to_probs` to convert to a uniform
        ``(n_samples, n_times)`` ndarray.
    """
    predict_fn = model.predict_survival_function

    if eval_times is None:
        return predict_fn(X)

    if _supports_predict_times(predict_fn):
        return predict_fn(X, times=np.asarray(eval_times))

    return predict_fn(X)


def compute_safe_ipcw_tau(
    y_train: SurvivalArray,
    y_test: SurvivalArray,
) -> Optional[float]:
    """Return a conservative default tau for Uno's IPCW concordance.

    The returned tau is ``min(train_t_max_event, percentile(test_event_times, 90))``:

    - ``train_t_max_event`` caps evaluation at the right edge of training
      censoring support, beyond which the Kaplan-Meier estimate of G(t) must
      be extrapolated and ``1/G(t)^2`` explodes.
    - The 90th-percentile test-event cap trims the far right tail where
      few informative pairs exist but IPCW weights are largest.

    If either component cannot be computed (e.g., no training events,
    no test events), the other is used. If neither is available, returns
    ``None`` (caller should pass ``tau=None`` through to sksurv, which
    defaults to the full test range).
    """
    if y_train is None or y_test is None:
        return None
    train_events = y_train[y_train['event']]
    train_cap = float(train_events['time'].max()) if len(train_events) > 0 else None
    test_events = y_test[y_test['event']]
    test_cap = float(np.percentile(test_events['time'], 90)) if len(test_events) > 0 else None
    if train_cap is None and test_cap is None:
        return None
    if train_cap is None:
        return test_cap
    if test_cap is None:
        return train_cap
    return min(train_cap, test_cap)


# Default cap on the number of points in the dense IBS integration grid.
# IBS is a smooth trapezoid integral, so ~100 points is effectively converged
# for any realistic cohort; the cap bounds bootstrap cost (N_iters x N_times).
IBS_GRID_CAP = 100


def ibs_integration_times(
    y_test: SurvivalArray,
    y_train: SurvivalArray,
    cap: int = IBS_GRID_CAP,
) -> np.ndarray:
    """Dense event-time grid for the integrated Brier score (IBS).

    IBS must be integrated over a *dense* grid so the number does not depend on
    the coarse display grid used for AUC(t)/Brier(t) curves.  sksurv's
    :func:`integrated_brier_score` trapezoid-integrates over exactly the supplied
    times, so a coarse grid makes IBS grid-dependent and non-comparable across
    datasets/splits.  This builds the integration grid separately.

    The grid is anchored to **training** event times (constant across
    splits/resamples/subpopulations within a run, so IBS stays comparable across
    cohorts), restricted to the censoring-stable window ``(t_min, upper)`` where
    ``upper = min(tau, test_t_max)``:

    - ``tau`` is the censoring-stable upper bound from
      :func:`compute_safe_ipcw_tau` (``min(train_t_max_event,
      percentile(test_event_times, 90))``), beyond which IPCW weights
      ``1/G(t)^2`` are unreliable.
    - ``test_t_max`` is the max follow-up of the test cohort after truncation to
      the training range (:func:`truncate_y_to_train_range`), so the window never
      exceeds training support.
    - ``t_min`` is the min follow-up time of the (truncated) test cohort.

    Points are unique training event times within ``(t_min, upper)``.  When more
    than ``cap`` points fall in the window they are subsampled to ``cap`` actual
    event times spaced evenly by rank (this keeps the first and last event time
    in the window, so the trapezoid still spans it); when ``<= cap`` all points
    are kept.

    Returns the grid *points* only.  Callers pass the result through
    :func:`clip_eval_times_to_data` (as they do for the display grid) and keep
    their existing ``len(...) < 2 -> NaN`` guard.  On degenerate input (no
    training events, an empty window, fewer than 2 points) this returns an array
    of length < 2 rather than raising, so the caller NaNs out IBS.

    Parameters
    ----------
    y_test : SurvivalArray
        Test (or evaluated-cohort) survival outcomes; defines the window bounds.
    y_train : SurvivalArray
        Training survival outcomes; the source of the grid points.
    cap : int, default 100
        Maximum number of grid points (bounds bootstrap cost).

    Returns
    -------
    np.ndarray
        Dense, sorted, unique event-time grid for IBS integration (possibly
        length < 2 on degenerate input).

    See Also
    --------
    compute_safe_ipcw_tau : Censoring-stable upper bound (tau).
    clip_eval_times_to_data : Per-cohort clip applied by callers afterward.
    auto_eval_times : The coarse *display* grid (kept separate from this).
    """
    if y_train is None or y_test is None:
        return np.empty(0, dtype=float)

    # Window comes from the test cohort, truncated to the training range so the
    # upper bound never exceeds training censoring support.
    y_test_trunc = truncate_y_to_train_range(y_test, y_train, warn=False)
    test_times = np.asarray(y_test_trunc['time'], dtype=float)
    if test_times.size == 0:
        return np.empty(0, dtype=float)
    t_min = float(test_times.min())
    test_t_max = float(test_times.max())

    tau = compute_safe_ipcw_tau(y_train, y_test_trunc)
    upper = test_t_max if tau is None else min(float(tau), test_t_max)

    events = np.asarray(y_train['event']).astype(bool)
    train_event_times = np.unique(np.asarray(y_train['time'], dtype=float)[events])
    grid = train_event_times[(train_event_times > t_min) & (train_event_times < upper)]

    if grid.size > cap:
        # Subsample to ``cap`` actual event times, evenly spaced by rank. Keeps
        # the first/last event time in the window so the trapezoid spans it; at
        # cap=100 the exact subsample is immaterial to the (smooth) integral.
        idx = np.unique(np.linspace(0, grid.size - 1, cap).round().astype(int))
        grid = grid[idx]

    return grid


def check_ipcw_support(
    y_train: SurvivalArray,
    y_test: SurvivalArray,
    tau: Optional[float] = None,
    g_hat_floor: float = 0.05,
    min_ipcw_support: int = 10,
) -> Optional[Dict[str, float]]:
    """Detect training-support violations for Uno's IPCW concordance.

    Here ``y_train`` is the cohort whose censoring distribution feeds the
    G_hat estimator (the resolved Ĝ source: the training split for
    ``evaluate()``/``bootstrap()``, the validation cohort for ``validate()``).

    Fires when **any** of:

    - Any test follow-up time exceeds the maximum training time
      (G(t) would be extrapolated beyond training support), or
    - The Kaplan-Meier estimate of the training censoring distribution
      ``G_hat(t)`` dips below ``g_hat_floor`` anywhere in ``[0, tau]``
      (weights ``1/G_hat(t)^2`` are unreliably large), or
    - The G_hat-fitting cohort has too few events or too few censored
      observations to estimate the censoring distribution at all, i.e.
      ``min(n_events, n_censored) < min_ipcw_support``. With ~0 censored,
      ``G_hat`` never steps down (stays ~1.0) and IPCW silently reduces to
      its unweighted form; with ~0 events concordance is degenerate.

    Parameters
    ----------
    y_train : SurvivalArray
        Outcomes of the cohort that fits G_hat.
    y_test : SurvivalArray
        Test outcomes.
    tau : float or None, default None
        Evaluation horizon for the G_hat check. When ``None``, the full
        test range is used.
    g_hat_floor : float, default 0.05
        Threshold below which ``min(G_hat)`` is considered a violation.
    min_ipcw_support : int, default 10
        Minimum number of both events and censored observations the
        G_hat-fitting cohort must have. A violation fires when
        ``min(n_events, n_censored)`` in ``y_train`` is below this. Count,
        not percentage: G_hat estimability scales with the number of
        minority-class observations, not their share.

    Returns
    -------
    dict or None
        ``{"n_beyond": int, "t_max_train": float, "min_g_hat": float,
        "tau": float or None, "n_g_events": int, "n_g_censored": int,
        "min_ipcw_support": int}`` when a violation fires, else ``None``.
        ``n_g_events``/``n_g_censored`` are the event/censored counts of the
        G_hat-fitting cohort; ``min_ipcw_support`` echoes the floor used.
    """
    if y_train is None or y_test is None:
        return None
    t_max_train = float(y_train['time'].max())
    n_beyond = int((y_test['time'] > t_max_train).sum())

    # Composition of the G_hat-fitting cohort: G_hat estimability scales with
    # the *number* of minority-class observations, so key on the count, not a
    # percentage. min(events, censored) is symmetric — it catches both the
    # too-little-censoring end (G_hat stuck at ~1.0) and the too-few-events end.
    n_g_events = int(np.asarray(y_train['event']).astype(bool).sum())
    n_g_censored = int(len(y_train) - n_g_events)
    min_support_count = min(n_g_events, n_g_censored)

    # Estimate G_hat on the training censoring distribution and probe it
    # at the largest relevant time (tau or t_max_train, whichever is smaller).
    min_g_hat = np.nan
    try:
        from sksurv.nonparametric import CensoringDistributionEstimator
        cde = CensoringDistributionEstimator()
        cde.fit(y_train)
        probe_max = float(tau) if tau is not None else float(y_test['time'].max())
        probe_max = min(probe_max, t_max_train)
        probe_times = np.linspace(
            max(float(y_train['time'].min()), 1e-6),
            max(probe_max, float(y_train['time'].min()) + 1e-6),
            num=50,
        )
        g_hat = cde.predict_proba(probe_times)
        min_g_hat = float(np.min(g_hat))
    except (ValueError, AttributeError, ImportError):
        min_g_hat = np.nan

    composition_unstable = min_support_count < min_ipcw_support
    fires = (
        (n_beyond > 0)
        or (np.isfinite(min_g_hat) and min_g_hat < g_hat_floor)
        or composition_unstable
    )
    if not fires:
        return None
    return {
        "n_beyond": n_beyond,
        "t_max_train": t_max_train,
        "min_g_hat": min_g_hat,
        "tau": float(tau) if tau is not None else None,
        "n_g_events": n_g_events,
        "n_g_censored": n_g_censored,
        "min_ipcw_support": min_ipcw_support,
    }


def c_index(
    y_true: SurvivalArray,
    risk_scores,
    method='harrell',
    y_train: Optional[SurvivalArray] = None,
    tau: Optional[float] = None,
):
    """
    Compute concordance index.

    Parameters
    ----------
    y_true : SurvivalArray
        True survival outcomes with 'event' and 'time' fields.
    risk_scores : array-like
        Predicted risk scores (higher = higher risk).
    method : str, default='harrell'
        'harrell' - standard C-index
        'ipcw' - inverse probability of censoring weighted
    y_train : SurvivalArray, optional
        Training outcomes (required for 'ipcw' method).
    tau : float, optional
        For ``method='ipcw'``, evaluation horizon passed through to
        :func:`sksurv.metrics.concordance_index_ipcw`. When ``None``
        (default), sksurv evaluates across the full test range, which
        can produce unstable estimates when test follow-up extends
        beyond training censoring support. See
        :func:`compute_safe_ipcw_tau` for a conservative default.

    Returns
    -------
    float
        C-index score (0.5 = random, 1.0 = perfect).

    Raises
    ------
    ValueError
        If ``method='ipcw'`` is requested without supplying ``y_train``,
        or if an unrecognized ``method`` is passed.

    Examples
    --------
    >>> from mlsurv.evaluation import c_index
    >>> c = c_index(y_test, model.predict(X_test))
    >>> print(f"C-index: {c:.3f}")
    """
    if method == 'harrell':
        result = concordance_index_censored(
            y_true['event'], y_true['time'], risk_scores
        )
        return result[0]
    elif method == 'ipcw':
        if y_train is None:
            raise ValueError("y_train required for IPCW method")
        y_true = truncate_y_to_train_range(y_true, y_train)
        return concordance_index_ipcw(y_train, y_true, risk_scores, tau=tau)[0]
    else:
        raise ValueError(f"Unknown method: {method}. Options: 'harrell', 'ipcw'")


def auc_t(y_train: SurvivalArray, y_test: SurvivalArray, risk_scores, eval_times):
    """
    Compute time-dependent AUC at specified timepoints.

    Parameters
    ----------
    y_train : SurvivalArray
        Training survival outcomes (for IPCW estimation).
    y_test : SurvivalArray
        Test survival outcomes.
    risk_scores : array-like
        Predicted risk scores.
    eval_times : array-like
        Evaluation timepoints (distinct from y['time'] survival times).

    Returns
    -------
    tuple
        (auc_values, mean_auc) where auc_values is an array of AUC values
        at each eval_time and mean_auc is the integrated AUC.

    Examples
    --------
    >>> from mlsurv.evaluation import auc_t
    >>> eval_times = np.array([1, 2, 3, 5])
    >>> auc_values, mean_auc = auc_t(y_train, y_test, risk_scores, eval_times)
    """
    eval_times = np.asarray(eval_times)
    valid_times, valid_mask, y_test_ipcw = clip_eval_times_to_data(
        eval_times, y_test, y_train,
    )
    if len(valid_times) < 2:
        return np.full(len(eval_times), np.nan), np.nan

    auc_valid, mean_auc = cumulative_dynamic_auc(
        y_train, y_test_ipcw, risk_scores, valid_times,
    )
    auc_all = np.full(len(eval_times), np.nan)
    auc_all[valid_mask] = auc_valid
    return auc_all, mean_auc


def _auc_values_and_nan_mask(y_data, y_train, eval_times, risk):
    """Compute AUC(t) at eval_times, aligned with NaN where unevaluable.

    Shared pure core for :func:`mlsurv.evaluation._concordance._compute_auc_t`
    (the evaluate path) and :func:`mlsurv.evaluation._subpopulation._safe_auc_t`
    (the subpopulation path). Clips ``eval_times`` to ``y_data``'s follow-up
    range, runs ``cumulative_dynamic_auc`` on the >= 2 valid times (suppressing
    the IPCW divide-instability RuntimeWarning), and reports which requested
    times came back NaN -- whether out-of-range OR from divide instability.

    Parameters
    ----------
    y_data : SurvivalArray
        Test (or subpopulation) outcomes whose follow-up range constrains
        ``eval_times``.
    y_train : SurvivalArray
        Training outcomes (for IPCW estimation).
    eval_times : array-like
        Requested evaluation time points.
    risk : array-like
        Predicted risk scores aligned with ``y_data``.

    Returns
    -------
    auc_all : ndarray
        AUC values aligned to ``eval_times``, NaN where unevaluable.
    nan_mask : ndarray of bool
        ``np.isnan(auc_all)`` -- True at every requested time that came back
        NaN (out-of-range or divide-instability).
    valid_times : ndarray
        ``eval_times`` falling within the open interval ``(t_min, t_max)`` of
        the data after truncation to the training range.

    Notes
    -----
    May raise sksurv's documented failure modes (``ValueError`` etc.); callers
    keep their own ``try``/``except`` so each routes the failure through its own
    notice mechanism.
    """
    eval_times_arr = np.asarray(eval_times, dtype=float)
    valid_times, valid_mask, y_trunc = clip_eval_times_to_data(
        eval_times_arr, y_data, y_train,
    )
    auc_all = np.full(len(eval_times_arr), np.nan)
    if len(valid_times) >= 2:
        with warnings.catch_warnings():
            # sksurv's cumulative_dynamic_auc weights by IPCW (1/G), which
            # becomes unstable as the censoring survival G approaches zero at
            # the late-time edge; numpy then emits "invalid value encountered
            # in divide" and yields NaN AUC there. Suppress the numpy spam --
            # the resulting NaN is detected via nan_mask below.
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning,
                message="invalid value encountered in divide",
            )
            auc_valid, _ = cumulative_dynamic_auc(
                y_train, y_trunc, risk, valid_times,
            )
        auc_all[valid_mask] = auc_valid
    return auc_all, np.isnan(auc_all), valid_times


def _df_to_probs(df, eval_times):
    """Convert a DataFrame (time-as-index, samples-as-columns) to survival probabilities.

    Uses step interpolation: for times between observed values, the last known
    survival probability is used (survival is constant between events).
    For times before the first observed value, the first value is returned;
    for times after the last, the last value is returned.

    Returns ndarray of shape (n_samples, n_times).
    """
    index_values = df.index.values
    surv_probs = []
    for t in eval_times:
        # side='right' so that exact matches map to idx-1 = the matching row
        idx = np.searchsorted(index_values, t, side='right')
        if idx == 0:
            surv_probs.append(df.iloc[0].values)
        else:
            surv_probs.append(df.iloc[idx - 1].values)
    return np.array(surv_probs).T


def _array_to_probs(arr, eval_times):
    """Convert an ndarray of survival probabilities, handling shape alignment.

    Accepts:
    - 1D array matching n_times (single sample) or n_samples when n_times == 1
    - 2D array with shape (n_samples, n_times) or (n_times, n_samples);
      when both dimensions equal n_times (square), assumes (n_samples, n_times)
    - Object array of callables (StepFunction objects)

    Returns ndarray of shape (n_samples, n_times).
    """
    n_times = len(eval_times)

    if arr.dtype == object and arr.size > 0 and callable(arr.flat[0]):
        return _callable_to_probs(arr, eval_times)

    if arr.ndim == 1:
        if arr.shape[0] == n_times:
            return arr.reshape(1, n_times)
        if n_times == 1:
            return arr.reshape(-1, 1)
        raise ValueError(
            f"1D survival array has {arr.shape[0]} values but {n_times} eval times."
        )

    if arr.ndim == 2:
        if arr.shape[1] == n_times:
            return arr
        if arr.shape[0] == n_times:
            return arr.T
        raise ValueError(
            f"Survival array shape {arr.shape} is incompatible with {n_times} eval times."
        )

    raise ValueError(
        f"Survival array must be 1D or 2D, got shape {arr.shape}."
    )


def _callable_to_probs(funcs, eval_times):
    """Convert an iterable of callable survival functions to probabilities.

    Each callable ``fn(t)`` returns the survival probability at time *t*
    (e.g. sksurv StepFunction objects).

    Returns ndarray of shape (n_samples, n_times).
    """
    return np.array([[fn(t) for t in eval_times] for fn in funcs])


def _surv_funcs_to_probs(surv_funcs, eval_times):
    """
    Convert survival functions to a 2D array of survival probabilities.

    Dispatches to type-specific helpers based on the input format:
    - DataFrame with time as index (from DeepSurv/pycox models)
    - ndarray of precomputed probabilities or callable objects
    - list/tuple of numeric values or callable objects

    Parameters
    ----------
    surv_funcs : list, ndarray, or DataFrame
        Survival functions from model.predict_survival_function().
    eval_times : array-like
        Evaluation timepoints.

    Returns
    -------
    ndarray
        Shape (n_samples, n_times) - each row is a sample, each column is a time.
    """
    import pandas as pd

    eval_times = np.asarray(eval_times)

    if isinstance(surv_funcs, pd.DataFrame):
        return _df_to_probs(surv_funcs, eval_times)

    if isinstance(surv_funcs, np.ndarray):
        return _array_to_probs(surv_funcs, eval_times)

    # list/tuple of numeric values → convert to ndarray and re-dispatch
    if isinstance(surv_funcs, (list, tuple)):
        arr = np.asarray(surv_funcs)
        if arr.dtype != object:
            return _array_to_probs(arr, eval_times)

    # Fallback: iterable of callables (StepFunction objects)
    return _callable_to_probs(surv_funcs, eval_times)


def surv_funcs_to_probs(surv_funcs, eval_times):
    """Convert survival functions to a 2D array of survival probabilities.

    Dispatches to type-specific helpers based on the input format:
    - DataFrame with time as index (from DeepSurv/pycox models)
    - ndarray of precomputed probabilities or callable objects
    - list/tuple of numeric values or callable objects

    Parameters
    ----------
    surv_funcs : list, ndarray, or DataFrame
        Survival functions from ``model.predict_survival_function()``.
    eval_times : array-like
        Evaluation timepoints.

    Returns
    -------
    ndarray
        Shape ``(n_samples, n_times)`` — each row is a sample, each
        column is a time point.

    Raises
    ------
    TypeError
        If ``surv_funcs`` or ``eval_times`` is ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsurv.evaluation import surv_funcs_to_probs
    >>> surv_arr = np.random.rand(100, 3)  # 100 samples, 3 time points
    >>> probs = surv_funcs_to_probs(surv_arr, np.array([1, 2, 5]))
    >>> probs.shape
    (100, 3)
    """
    if surv_funcs is None:
        raise TypeError(
            "surv_funcs must not be None. Pass an array, DataFrame, "
            "or list of callables."
        )
    if eval_times is None:
        raise TypeError(
            "eval_times must not be None. Pass an array of evaluation "
            "time points."
        )
    return _surv_funcs_to_probs(surv_funcs, eval_times)


def _brier_score_at_times(y_train: SurvivalArray, y_test: SurvivalArray, surv_funcs, eval_times):
    """
    Compute Brier score at specified timepoints.

    Requires survival functions from models that support predict_survival_function().

    Parameters
    ----------
    y_train : SurvivalArray
        Training survival outcomes.
    y_test : SurvivalArray
        Test survival outcomes.
    surv_funcs : list, ndarray, or DataFrame
        Survival functions from model.predict_survival_function().
        Can be StepFunction objects or DataFrame with time as index.
    eval_times : array-like
        Evaluation timepoints.

    Returns
    -------
    list
        Brier scores at each timepoint.

    Notes
    -----
    For CoxNet, the model must be fitted with `fit_baseline_model=True`
    to support predict_survival_function().
    """
    # Convert to 2D array of survival probabilities
    surv_probs = _surv_funcs_to_probs(surv_funcs, eval_times)

    # Compute Brier score at each timepoint individually
    # sksurv.brier_score expects 1D preds array with scalar time
    scores = []
    for i, t in enumerate(eval_times):
        preds = surv_probs[:, i]  # Shape: (n_samples,)
        bs = brier_score(y_train, y_test, preds, t)
        scores.append(float(bs[1][0]))

    return scores


def _integrated_brier(y_train: SurvivalArray, y_test: SurvivalArray, surv_funcs, eval_times):
    """
    Compute integrated Brier score over the supplied ``eval_times``.

    This is the **scoring / CV-tuning** IBS path: it integrates over exactly the
    grid it is handed (the auto/display ``eval_times``), giving a relative
    ranking signal across hyperparameters and folds. It is intentionally
    distinct from the headline IBS reported by ``evaluate()``/``bootstrap()``,
    which integrates over the dense, train-anchored grid (see
    :func:`_integrated_brier_dense`). Do not unify the two — doing so would
    reintroduce display-grid dependence into the reported IBS.

    Parameters
    ----------
    y_train : SurvivalArray
        Training survival outcomes.
    y_test : SurvivalArray
        Test survival outcomes.
    surv_funcs : list, ndarray, or DataFrame
        Survival functions from model.predict_survival_function().
        Can be StepFunction objects or DataFrame with time as index.
    eval_times : array-like
        Evaluation timepoints.

    Returns
    -------
    float
        Integrated Brier score.
    """
    # Convert to 2D array of survival probabilities
    surv_probs = _surv_funcs_to_probs(surv_funcs, eval_times)
    # sksurv uses np.trapz internally, which emits a DeprecationWarning on NumPy 2.x.
    # Filter here so it's suppressed even in joblib worker processes.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`trapz` is deprecated",
                                category=DeprecationWarning)
        return integrated_brier_score(y_train, y_test, surv_probs, eval_times)


def _integrated_brier_from_probs(surv_probs, ibs_times, y_data, y_train):
    """IBS from a precomputed dense S(t|x) matrix aligned to ``ibs_times``.

    Clips ``ibs_times`` to the cohort's follow-up range (truncated to the
    training range) and trapezoid-integrates over the surviving points.
    Returns ``np.nan`` when fewer than 2 points survive — callers keep their
    own NaN handling.  ``surv_probs`` must have one column per entry of
    ``ibs_times``.
    """
    valid_times, valid_mask, y_ipcw = clip_eval_times_to_data(
        ibs_times, y_data, y_train,
    )
    if len(valid_times) < 2:
        return np.nan
    # sksurv uses np.trapz internally (DeprecationWarning on NumPy 2.x); filter
    # here so it stays suppressed inside joblib worker processes.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`trapz` is deprecated",
                                category=DeprecationWarning)
        return float(integrated_brier_score(
            y_train, y_ipcw, surv_probs[:, valid_mask], valid_times,
        ))


def _predict_dense_ibs_probs(model, X, y_data, y_train, *,
                             surv_probs_transform=None, cap=IBS_GRID_CAP):
    """Predict ``S(t|x)`` on the dense, train-anchored IBS grid.

    Builds the dense integration grid (:func:`ibs_integration_times`, separate
    from the display grid used for Brier(t)/AUC(t)) and predicts ``S(t|x)`` on
    it, applying ``surv_probs_transform`` if given (e.g. isotonic
    recalibration).

    Returns ``(surv_probs, ibs_times)``.  ``surv_probs`` is ``None`` when the
    grid has fewer than 2 points (caller NaNs out IBS); ``ibs_times`` is always
    returned so callers can cache the grid.  Prediction errors propagate.
    """
    ibs_times = ibs_integration_times(y_data, y_train, cap=cap)
    if len(ibs_times) < 2:
        return None, ibs_times
    surv_funcs = predict_survival_functions(model, X, ibs_times)
    surv_probs = _surv_funcs_to_probs(surv_funcs, ibs_times)
    if surv_probs_transform is not None:
        surv_probs = surv_probs_transform(surv_probs, ibs_times)
    return surv_probs, ibs_times


def _integrated_brier_dense(model, X, y_data, y_train, *,
                            surv_probs_transform=None, cap=IBS_GRID_CAP):
    """IBS over the dense train-anchored grid, predicting ``S(t|x)`` from ``model``.

    Convenience wrapper around :func:`_predict_dense_ibs_probs` +
    :func:`_integrated_brier_from_probs`.  Returns ``np.nan`` when the grid has
    fewer than 2 points.  Prediction/integration errors propagate so the caller
    can log them the same way it logs display-grid failures.
    """
    surv_probs, ibs_times = _predict_dense_ibs_probs(
        model, X, y_data, y_train,
        surv_probs_transform=surv_probs_transform, cap=cap,
    )
    if surv_probs is None:
        return np.nan
    return _integrated_brier_from_probs(surv_probs, ibs_times, y_data, y_train)


def _resolve_eval_times(y: SurvivalArray, y_train: Optional[SurvivalArray], eval_times):
    """Determine eval_times from data if not provided.

    On `auto_eval_times`/`clip_eval_times` failure (e.g. degenerate y_train
    with <2 unique event times), emits a deduped user-facing
    `SuggestionWarning` and returns `None`. Callers (`_score_brier`,
    `_score_ctd`) short-circuit to NaN when the return is `None`,
    replacing per-fold `log_metric_failure` ERROR spam with a single
    user-facing warning per distinct root cause.
    """
    if eval_times is not None:
        return eval_times
    # auto_eval_times derives times from y_train's event distribution, so
    # clip_eval_times is usually a no-op. It is kept as a safety net for
    # the edge case where event times cluster at t_max.
    try:
        eval_times = auto_eval_times(y, y_train=y_train, _quiet=True)
        return clip_eval_times(eval_times, y_train)
    except (ValueError, TypeError) as exc:
        # Dedup key collapses identical failures across folds/trials but
        # surfaces distinct root causes (e.g. ValueError vs TypeError, or
        # different messages) so genuinely new failures aren't hidden.
        dedup_key = (type(exc).__name__, str(exc)[:200])
        if dedup_key not in _eval_times_warned:
            _eval_times_warned.add(dedup_key)
            _warn_eval_times_failure(
                exc, "during scoring",
                "Time-dependent metrics (Brier, IBS, AUC(t), C^td, "
                "calibration) will be NaN.",
            )
        return None


def _require_surv_func(estimator, scoring):
    """Validate that estimator supports predict_survival_function.

    Unwraps sklearn Pipelines to check the final model step.

    Raises
    ------
    ValueError
        If the underlying model lacks ``predict_survival_function``.
    """
    model = unwrap_pipeline(estimator)
    if not hasattr(model, 'predict_survival_function'):
        raise ValueError(
            f"Model does not support predict_survival_function() "
            f"required for '{scoring}' scoring. Use 'harrell' or 'ipcw' instead."
        )


def _require_y_train(y_train, scoring):
    """Raise ValueError if y_train is None."""
    if y_train is None:
        raise ValueError(f"y_train required for '{scoring}' scoring")


def _score_brier(estimator, X, y: SurvivalArray, scoring, y_train: Optional[SurvivalArray], eval_times):
    """Compute Brier-based scores (negated so higher is better for optimization)."""
    _require_y_train(y_train, scoring)
    _require_surv_func(estimator, scoring)

    # Setup: truncation is a prerequisite, not a metric-computation edge
    # case — let failures propagate (consistent with _add_brier_metrics
    # in _concordance.py which keeps setup outside try/except).
    y_ipcw = truncate_y_to_train_range(y, y_train)

    try:
        eval_times = _resolve_eval_times(y, y_train, eval_times)
        # _resolve_eval_times returns None on degenerate inputs after
        # emitting a single SuggestionWarning. Short-circuit to NaN here so
        # CV/tuning sees one user-facing warning instead of one
        # log_metric_failure ERROR per fold × repeat × Optuna trial.
        if eval_times is None:
            return np.nan
        surv_funcs = predict_survival_functions(estimator, X, eval_times)
        if scoring == 'brier':
            median_time = np.median(eval_times)
            scores = _brier_score_at_times(y_train, y_ipcw, surv_funcs, [median_time])
            return -scores[0]
        # Scoring path: integrate over the supplied CV/display grid (a relative
        # ranking signal for tuning), NOT the dense headline grid — dense
        # prediction per Optuna trial x fold is needlessly costly here. See
        # _integrated_brier's docstring.
        ibs = _integrated_brier(y_train, y_ipcw, surv_funcs, eval_times)
        return -ibs
    except (MemoryError, RecursionError):
        raise
    except (ValueError, ZeroDivisionError, NotImplementedError, RuntimeError, TypeError) as e:
        if _is_fold_boundary_error(e):
            log_metric_failure(
                logger, scoring, e,
                level=_log_once_level("fold_boundary"),
                warn_once_key="fold_boundary_warn",
                user_message=_fold_boundary_message(scoring),
            )
        else:
            log_metric_failure(logger, scoring, e, level=logging.ERROR)
        return np.nan
    except Exception as e:
        log_metric_failure(logger, scoring, e, warn_user=True)
        return np.nan


def _score_ctd(estimator, X, y: SurvivalArray, y_train: Optional[SurvivalArray], eval_times):
    """Compute Antolini's C^td for scoring (higher is better)."""
    from ._ctd import concordance_td

    _require_y_train(y_train, 'ctd')
    _require_surv_func(estimator, 'ctd')

    try:
        eval_times = _resolve_eval_times(y, y_train, eval_times)
        # See _score_brier for short-circuit rationale.
        if eval_times is None:
            return np.nan
        surv_funcs = predict_survival_functions(estimator, X, eval_times)
        surv_probs = _surv_funcs_to_probs(surv_funcs, eval_times)
        return concordance_td(y['time'], y['event'], surv_probs, eval_times)
    except (MemoryError, RecursionError):
        raise
    except (ValueError, ZeroDivisionError, NotImplementedError, RuntimeError, TypeError) as e:
        log_metric_failure(logger, 'ctd', e)
        return np.nan
    except Exception as e:
        log_metric_failure(logger, 'ctd', e, warn_user=True)
        return np.nan


def calibration_slope_intercept(y_test: SurvivalArray, pred_probs, eval_time,
                                 model_name: Optional[str] = None,
                                 warnings_out: Optional[list] = None):
    """Compute calibration slope and intercept via logistic recalibration.

    Fits logistic regression of binary outcome (survived past eval_time)
    on logit(predicted survival probability). A perfectly calibrated model
    produces slope=1.0 and intercept=0.0.

    Parameters
    ----------
    y_test : SurvivalArray
        Test survival outcomes.
    pred_probs : array-like
        Predicted survival probabilities at ``eval_time``.
    eval_time : float
        Time point at which to evaluate calibration.
    model_name : str, optional
        Model name used to prefix any warning/diagnostic messages.
    warnings_out : list, optional
        If provided, NaN-return warning details are appended here for
        parent-side emission instead of being passed to
        ``warnings.warn`` directly. The latter is silently dropped in
        joblib/loky workers.

    Returns
    -------
    tuple of (float, float)
        (slope, intercept). Returns (NaN, NaN) on failure.

    Raises
    ------
    MemoryError
        Re-raised if the computation exhausts available memory; resource
        errors are never swallowed by the broad failure handler.
    RecursionError
        Re-raised if the computation exceeds the recursion limit; resource
        errors are never swallowed by the broad failure handler.
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.special import logit

    events = y_test['event'].astype(bool)
    times = y_test['time']
    pred_probs = np.asarray(pred_probs, dtype=float)

    prefix = f"{model_name}: " if model_name else ""

    # Binary outcome: 1 = survived past eval_time, 0 = event before eval_time
    # Exclude censored-before-eval_time (uninformative)
    informative = events | (times >= eval_time)
    if informative.sum() < 10:
        n_info = int(informative.sum())
        _emit_or_collect(
            "cal_slope", model_name,
            f"{prefix}Calibration slope/intercept: only {n_info} "
            f"informative samples (need >= 10); returning NaN",
            warnings_out,
        )
        logger.debug(
            "calibration_slope_intercept: %d informative samples < 10; "
            "returning NaN", n_info,
        )
        return (np.nan, np.nan)

    y_binary = (times[informative] >= eval_time).astype(int)
    preds = pred_probs[informative]

    # Clip to avoid logit(0) or logit(1)
    preds = np.clip(preds, 1e-7, 1 - 1e-7)
    logit_preds = logit(preds).reshape(-1, 1)

    try:
        # sklearn 1.8 deprecated penalty=None; the documented replacement
        # C=np.inf still emits a UserWarning ("penalty=None will ignore C
        # and l1_ratio"), so use a large finite C for effectively
        # unpenalized fitting. l1_ratio=0 (vs the default None) avoids the
        # l1_ratio=None FutureWarning sklearn 1.8 emits on the default
        # path.
        lr = LogisticRegression(
            C=1e10, l1_ratio=0, solver='lbfgs', max_iter=1000,
        )
        lr.fit(logit_preds, y_binary)
        return (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
    except (MemoryError, RecursionError):
        raise
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        log_metric_failure(logger, "cal_slope", e)
        _emit_or_collect(
            "cal_slope", model_name,
            f"{prefix}Calibration slope/intercept returned NaN "
            f"(enable debug logging for details)",
            warnings_out,
        )
        return (np.nan, np.nan)
    except Exception as e:
        log_metric_failure(logger, "cal_slope", e)
        _emit_or_collect(
            "cal_slope", model_name,
            f"{prefix}Calibration slope/intercept returned NaN "
            f"(enable debug logging for details)",
            warnings_out,
        )
        return (np.nan, np.nan)


def _assign_bins(pred_probs, n_bins, metric_name):
    """Assign equal-frequency bins via quantile cut.

    Parameters
    ----------
    pred_probs : ndarray
        Predicted survival probabilities (already converted to float).
    n_bins : int
        Number of equal-frequency bins.
    metric_name : str
        Name of the calling metric (for log messages on failure).

    Returns
    -------
    ndarray or None
        Integer bin labels for each sample, or None on failure.
    """
    if len(pred_probs) < n_bins:
        logger.debug("%s: only %d samples for %d bins; skipping",
                     metric_name, len(pred_probs), n_bins)
        return None
    try:
        import pandas as pd
        return pd.qcut(pred_probs, q=n_bins, labels=False, duplicates='drop')
    except ValueError as e:
        log_metric_failure(logger, metric_name, e)
        return None


def _km_survival_at(times, events, eval_time):
    """Fit Kaplan-Meier and return S(eval_time) with the fitted model.

    Parameters
    ----------
    times : array-like
        Observed event/censoring times for the group.
    events : array-like of bool
        Event indicators for the group.
    eval_time : float
        Time point at which to evaluate survival.

    Returns
    -------
    tuple of (float, KaplanMeierFitter) or None
        (survival_estimate, fitted_kmf), or None on failure.
    """
    from lifelines import KaplanMeierFitter

    try:
        kmf = KaplanMeierFitter()
        kmf.fit(times, events)
        sf = kmf.survival_function_at_times(eval_time)
        return float(sf.iloc[0]), kmf
    except (ValueError, IndexError) as e:
        logger.debug("KM fit/eval failed (n=%d, n_events=%d): %s",
                     len(times), int(events.sum()), e)
        return None


def hosmer_lemeshow(y_test: SurvivalArray, pred_probs, eval_time, n_bins=10,
                    model_name: Optional[str] = None,
                    warnings_out: Optional[list] = None):
    """Compute Hosmer-Lemeshow goodness-of-fit test for survival calibration.

    Bins predicted survival probabilities, compares expected vs. observed
    survival (Kaplan-Meier) per bin, and returns a chi-squared statistic
    with p-value.

    Parameters
    ----------
    y_test : SurvivalArray
        Test survival outcomes.
    pred_probs : array-like
        Predicted survival probabilities at ``eval_time``.
    eval_time : float
        Time point at which to evaluate calibration.
    n_bins : int, default=10
        Number of equal-frequency bins.
    model_name : str, optional
        Model name used to prefix any warning/diagnostic messages.
    warnings_out : list, optional
        If provided, NaN-return warning details are appended here for
        parent-side emission instead of being passed to
        ``warnings.warn`` directly (silently dropped in joblib/loky
        workers).

    Returns
    -------
    tuple of (float, float)
        (chi2_statistic, p_value). Returns (NaN, NaN) on failure.
    """
    from scipy.stats import chi2 as chi2_dist

    events = y_test['event'].astype(bool)
    times = y_test['time']
    pred_probs = np.asarray(pred_probs, dtype=float)

    prefix = f"{model_name}: " if model_name else ""

    # Need >= 5 samples per bin; bail early if impossible
    min_per_bin = 5
    min_samples = n_bins * min_per_bin
    if len(pred_probs) < min_samples:
        _emit_or_collect(
            "hl", model_name,
            f"{prefix}Hosmer-Lemeshow calibration: {len(pred_probs)} "
            f"samples < {min_samples} required for {n_bins} bins of "
            f"{min_per_bin}; returning NaN",
            warnings_out,
        )
        logger.debug(
            "hosmer_lemeshow: %d samples < %d required for %d bins; "
            "returning NaN",
            len(pred_probs), min_samples, n_bins,
        )
        return (np.nan, np.nan)

    bin_labels = _assign_bins(pred_probs, n_bins, "hosmer_lemeshow")
    if bin_labels is None:
        _emit_or_collect(
            "hl", model_name,
            f"{prefix}Hosmer-Lemeshow returned NaN "
            f"(enable debug logging for details)",
            warnings_out,
        )
        return (np.nan, np.nan)

    unique_bins = np.unique(bin_labels)

    chi2_val = 0.0
    valid_bins = 0
    for b in unique_bins:
        mask = bin_labels == b
        n_k = mask.sum()
        if n_k < 5:
            continue

        E_k = pred_probs[mask].mean()

        result = _km_survival_at(times[mask], events[mask], eval_time)
        if result is None:
            continue
        O_k, _kmf = result

        # Binomial variance approximation; skip near-zero denominators
        denom = E_k * (1 - E_k) / n_k
        if denom < 1e-12:
            continue

        chi2_val += (O_k - E_k) ** 2 / denom
        valid_bins += 1

    skipped = len(unique_bins) - valid_bins
    if skipped > 0:
        logger.debug("hosmer_lemeshow: %d/%d bins skipped", skipped, len(unique_bins))

    df = valid_bins - 2
    if df < 1:
        _emit_or_collect(
            "hl", model_name,
            f"{prefix}Hosmer-Lemeshow returned NaN "
            f"(enable debug logging for details)",
            warnings_out,
        )
        logger.debug(
            "hosmer_lemeshow: only %d valid bins (< 3 required for "
            "df >= 1); returning NaN",
            valid_bins,
        )
        return (np.nan, np.nan)

    p_value = float(chi2_dist.sf(chi2_val, df=df))
    return (float(chi2_val), p_value)


def greenwood_nam_dagostino(y_test: SurvivalArray, pred_probs, eval_time,
                             n_bins=10, model_name: Optional[str] = None,
                             warnings_out: Optional[list] = None):
    """Greenwood-Nam-D'Agostino calibration test (Demler et al. 2015).

    Uses Kaplan-Meier estimates for observed survival and Greenwood's
    variance formula to properly account for censoring in the variance
    estimate, rather than the binomial variance approximation used by
    the Hosmer-Lemeshow test.

    Parameters
    ----------
    y_test : SurvivalArray
        Test survival outcomes.
    pred_probs : array-like
        Predicted survival probabilities at ``eval_time``.
    eval_time : float
        Time point at which to evaluate calibration.
    n_bins : int, default=10
        Number of equal-frequency bins.
    model_name : str, optional
        Model name used to prefix any warning/diagnostic messages.
    warnings_out : list, optional
        If provided, NaN-return warning details are appended here for
        parent-side emission instead of being passed to
        ``warnings.warn`` directly (silently dropped in joblib/loky
        workers).

    Returns
    -------
    tuple of (float, float, int)
        (chi2_statistic, p_value, df). Returns (NaN, NaN, 0) on failure.
    """
    from scipy.stats import chi2 as chi2_dist

    events = y_test['event'].astype(bool)
    times = y_test['time']
    pred_probs = np.asarray(pred_probs, dtype=float)

    prefix = f"{model_name}: " if model_name else ""

    # Need >= 5 samples per bin; bail early if impossible
    min_per_bin = 5
    min_samples = n_bins * min_per_bin
    if len(pred_probs) < min_samples:
        # Distinct metric_key from the generic "gnd" path: the sample-size
        # shortage is a cohort property (identical math across every model
        # in a multi-model evaluate), so the parent-side coalescer groups
        # these into one warning naming all affected models. See
        # _format_coalesced_calibration_nan_warning.
        _emit_or_collect(
            "gnd_sample_size", model_name,
            f"{prefix}GND calibration: {len(pred_probs)} samples < "
            f"{min_samples} required for {n_bins} bins of "
            f"{min_per_bin}; returning NaN",
            warnings_out,
        )
        return (np.nan, np.nan, 0)

    bin_labels = _assign_bins(pred_probs, n_bins, "gnd")
    if bin_labels is None:
        _emit_or_collect(
            "gnd", model_name,
            f"{prefix}GND calibration returned NaN "
            f"(enable debug logging for details)",
            warnings_out,
        )
        return (np.nan, np.nan, 0)

    unique_bins = np.unique(bin_labels)

    chi2_val = 0.0
    valid_bins = 0
    for b in unique_bins:
        mask = bin_labels == b
        n_k = mask.sum()
        if n_k < 5:
            continue

        E_k = pred_probs[mask].mean()

        result = _km_survival_at(times[mask], events[mask], eval_time)
        if result is None:
            continue
        O_k, kmf = result

        # Greenwood variance: S(t)^2 * sum(d_j / (n_j * (n_j - d_j)))
        # for event times <= eval_time
        et = kmf.event_table
        et_before = et.loc[et.index <= eval_time]
        if et_before.empty:
            continue
        d = et_before['observed']
        n_at_risk = et_before['at_risk']
        denom = n_at_risk * (n_at_risk - d)
        valid = denom > 0
        if not valid.any():
            continue
        V_k = O_k ** 2 * float((d[valid] / denom[valid]).sum())

        if V_k < 1e-12:
            continue

        chi2_val += (O_k - E_k) ** 2 / V_k
        valid_bins += 1

    skipped = len(unique_bins) - valid_bins
    if skipped > 0:
        logger.debug("gnd: %d/%d bins skipped", skipped, len(unique_bins))

    if valid_bins < 2:
        _emit_or_collect(
            "gnd", model_name,
            f"{prefix}GND calibration returned NaN "
            f"(enable debug logging for details)",
            warnings_out,
        )
        logger.debug(
            "gnd: only %d valid bins (< 2 required); returning NaN",
            valid_bins,
        )
        return (np.nan, np.nan, 0)

    # GND uses df = G-1 (Demler et al. 2015; reference SAS macro by Cook)
    df = valid_bins - 1
    p_value = float(chi2_dist.sf(chi2_val, df=df))
    return (float(chi2_val), p_value, df)


def calibration_test(y_test: SurvivalArray, pred_probs, eval_time, n_bins=10,
                     method='gnd', model_name: Optional[str] = None,
                     warnings_out: Optional[list] = None):
    """Dispatch calibration goodness-of-fit test.

    Parameters
    ----------
    y_test : SurvivalArray
        Test survival outcomes.
    pred_probs : array-like
        Predicted survival probabilities at ``eval_time``.
    eval_time : float
        Time point at which to evaluate calibration.
    n_bins : int, default=10
        Number of equal-frequency bins.
    method : str, default='gnd'
        ``'gnd'`` for Greenwood-Nam-D'Agostino (recommended) or
        ``'hl'`` for Hosmer-Lemeshow.
    model_name : str, optional
        Model name used to prefix any warning/diagnostic messages.
    warnings_out : list, optional
        If provided, NaN-return warning details are appended here for
        parent-side emission instead of via ``warnings.warn``.

    Returns
    -------
    tuple of (float, float)
        (chi2_statistic, p_value). Returns (NaN, NaN) on failure.

    Raises
    ------
    ValueError
        If ``method`` is not ``'gnd'`` or ``'hl'``.
    """
    if method == 'gnd':
        chi2, p_value, _ = greenwood_nam_dagostino(
            y_test, pred_probs, eval_time, n_bins, model_name=model_name,
            warnings_out=warnings_out,
        )
        return (chi2, p_value)
    elif method == 'hl':
        return hosmer_lemeshow(
            y_test, pred_probs, eval_time, n_bins, model_name=model_name,
            warnings_out=warnings_out,
        )
    else:
        raise ValueError(f"Unknown calibration test method: {method!r}. Options: 'gnd', 'hl'.")


_PH_COLUMNS = ['feature', 'test_statistic', 'p_value', 'p_adjusted', 'significant']


def _build_per_feature_rows(summary):
    """Extract per-feature test statistic rows from proportional_hazard_test summary.

    Filters out features with NaN results.  Does NOT set ``p_adjusted`` or
    ``significant`` — those are added by ``ph_test()`` after FDR correction.

    Returns
    -------
    tuple of (list of dict, list of str)
        Rows of valid features (keys: feature, test_statistic, p_value) and
        the names of features excluded for producing NaN. The caller emits
        a single summary warning; per-feature names are logged at DEBUG.
    """
    rows = []
    excluded = []
    for feat in summary.index:
        ts = float(summary.loc[feat, 'test_statistic'])
        pv = float(summary.loc[feat, 'p'])
        if np.isnan(ts) or np.isnan(pv):
            logger.debug("ph_test: NaN result for feature '%s' — excluded", feat)
            excluded.append(feat)
            continue
        rows.append({
            'feature': feat,
            'test_statistic': ts,
            'p_value': pv,
        })
    return rows, excluded


def _suppress_residual_overflow(fn, *args, **kwargs):
    """Run a lifelines residual / PH computation with numpy overflow suppressed.

    lifelines exponentiates the linear predictor when forming Schoenfeld
    residuals (``exp(dot(X, params))``); on a large-magnitude / collinear design
    the numpy ufunc raises a raw ``RuntimeWarning: overflow encountered in
    exp``. The numerical failure is already reported through the tiered PH
    warnings, so suppress the raw copy at the source.

    ``np.errstate`` is thread-local and governs *only* numpy floating-point
    errors, so it drops the overflow without disturbing Python's warning state.
    Genuine Python warnings (e.g. lifelines ``ConvergenceWarning``) are left
    untouched and surface normally under the caller's filters — using
    ``catch_warnings`` + re-emit here instead would repeatedly bump the global
    filter version and defeat the per-message deduplication of those
    convergence warnings, flooding the output.
    """
    with np.errstate(over="ignore"):
        return fn(*args, **kwargs)


def _safe_scaled_schoenfeld(cph, df):
    """Scaled Schoenfeld residuals from a fitted Cox model, or ``None`` on failure.

    Computed once and shared by the joint global test and the tidy residual
    frame (so ``ph_test_with_schoenfeld`` fits and computes residuals once).
    Returns ``None`` rather than raising so the joint test degrades and the
    Schoenfeld plot is simply omitted.
    """
    try:
        return _suppress_residual_overflow(
            cph.compute_residuals, df, 'scaled_schoenfeld'
        )
    except (ValueError, KeyError, RuntimeError, np.linalg.LinAlgError) as exc:
        logger.debug(
            "ph_test: scaled Schoenfeld residual computation failed (%s: %s)",
            type(exc).__name__, exc,
        )
        return None


def _compute_joint_global_test(cph, df, rows, time_transform, scaled_resids=None):
    """Compute Grambsch-Therneau joint global test statistic.

    Per-feature: ``T_j = U_j^2 / (n * se_j^2 * S_gg)``
    Joint:       ``T = U' @ (S_gg * n * Sigma_beta)^{-1} @ U``
    where ``Sigma_beta = cph.variance_matrix_`` (covariance of coefficients).

    Falls back to sum-of-chi2 if the covariance is singular or computation
    fails.

    Parameters
    ----------
    scaled_resids : DataFrame, optional
        Pre-computed scaled Schoenfeld residuals. When ``None`` (the default,
        preserving standalone behaviour), they are computed here from *cph*.

    Returns
    -------
    tuple of (total_chi2: float, joint_df: int, degraded: bool)
        ``degraded`` is ``True`` when the joint covariance was singular or
        failed and the returned statistic is the cruder sum-of-per-feature
        chi2 (which ignores between-feature covariance). The caller emits a
        :class:`MethodologicalWarning` so the user knows the GLOBAL row is an
        independence-assuming approximation.
    """
    try:
        from lifelines.statistics import TimeTransformers
        if scaled_resids is None:
            scaled_resids = _suppress_residual_overflow(
                cph.compute_residuals, df, 'scaled_schoenfeld'
            )
        n_deaths = int(df['E'].sum())

        time_transformer = TimeTransformers().get(time_transform)
        times_g = time_transformer(
            cph.durations, cph.event_observed, cph.weights
        )[cph.event_observed.values]
        g = times_g - times_g.mean()
        S_gg = float((g ** 2).sum())

        valid_feats = [r['feature'] for r in rows]
        resid_cols = [c for c in scaled_resids.columns if c in valid_feats]
        U = (g.values[:, None] * scaled_resids[resid_cols].values).sum(axis=0)

        Sigma_beta = cph.variance_matrix_.loc[resid_cols, resid_cols].values
        Var_U = S_gg * n_deaths * Sigma_beta
        Var_U_inv = np.linalg.inv(Var_U)
        total_chi2 = float(U @ Var_U_inv @ U)
        if np.isnan(total_chi2):
            raise ValueError("joint chi2 is NaN (residuals may contain NaN)")
        return total_chi2, len(resid_cols), False
    except np.linalg.LinAlgError as exc:
        logger.warning("ph_test: joint covariance singular (%s), "
                       "falling back to sum-of-chi2", exc)
    except (ValueError, KeyError) as exc:
        logger.warning("ph_test: joint covariance failed (%s: %s), "
                       "falling back to sum-of-chi2",
                       type(exc).__name__, exc)
    return sum(r['test_statistic'] for r in rows), len(rows), True


def _ph_test_core(X_train, y_train, time_transform='rank'):
    """Shared core for ``ph_test`` / ``ph_test_with_schoenfeld``: one Cox fit.

    Returns
    -------
    tuple of (ph_df, cph, df, scaled_resids)
        ``ph_df`` is the PH-test DataFrame (see :func:`ph_test`). The other
        three are the fitted model, the design it was fit on (carrying
        ``T``/``E``), and the scaled Schoenfeld residuals — reused by
        :func:`ph_test_with_schoenfeld` to build the residual plot from the
        same fit. They are ``None`` whenever ``ph_df`` is the empty sentinel,
        or ``scaled_resids`` is ``None`` if the residual computation failed.

    Warnings use ``stacklevel=3`` so they resolve to the public caller
    (user -> ph_test/ph_test_with_schoenfeld -> _ph_test_core -> warn).
    """
    import pandas as pd

    empty = pd.DataFrame(columns=_PH_COLUMNS)

    # NaN handling outside try/except so bugs propagate with clear tracebacks
    df = X_train.copy()
    df, y_train = drop_nan_rows(df, y_train, "ph_test")
    if len(df) == 0:
        msg = "ph_test: no rows remain after NaN removal; returning empty"
        warnings.warn(msg, MethodologicalWarning, stacklevel=3)
        return empty, None, None, None

    try:
        from lifelines.statistics import proportional_hazard_test
        from scipy.stats import chi2, false_discovery_control

        df['T'] = y_train['time']
        df['E'] = y_train['event'].astype(int)

        cph, _info = _fit_cox_with_fallback(
            df,
            duration_col='T',
            event_col='E',
            context='ph_test',
            penalizers=(0.01, 0.1, 1.0),
            warn_stacklevel=4,
        )

        result = _suppress_residual_overflow(
            proportional_hazard_test, cph, df, time_transform=time_transform
        )
        summary = result.summary

        if summary.empty:
            msg = (
                "ph_test: no features in proportional_hazard_test summary; "
                "returning empty"
            )
            warnings.warn(msg, MethodologicalWarning, stacklevel=3)
            return empty, None, None, None

        rows, excluded = _build_per_feature_rows(summary)

        if not rows:
            msg = (
                f"ph_test: all {len(excluded)} features produced NaN "
                f"(likely Cox fit convergence failure); PH diagnostic "
                f"unavailable"
            )
            warnings.warn(msg, MethodologicalWarning, stacklevel=3)
            return empty, None, None, None
        if excluded:
            logger.warning(
                "ph_test: %d/%d features produced NaN and were excluded "
                "from the result",
                len(excluded), len(excluded) + len(rows),
            )
            # Surface dropped features so users notice missing rows. Tier is
            # SuggestionWarning (not MethodologicalWarning): the surviving rows
            # are correct and visible in the result; only the count is reduced.
            warnings.warn(
                f"ph_test: {len(excluded)} feature(s) produced NaN and were "
                f"excluded from the result: "
                f"{', '.join(map(str, excluded[:5]))}"
                f"{' (and more)' if len(excluded) > 5 else ''}.",
                SuggestionWarning,
                stacklevel=3,
            )

        # Benjamini-Hochberg FDR correction on per-feature p-values
        raw_pvals = np.array([r['p_value'] for r in rows])
        adj_pvals = false_discovery_control(raw_pvals, method='bh')
        for row, adj_p in zip(rows, adj_pvals):
            row['p_adjusted'] = float(adj_p)
            row['significant'] = adj_p < 0.05

        # Compute residuals once; the joint test reuses them and so does the
        # tidy Schoenfeld frame in ph_test_with_schoenfeld.
        scaled_resids = _safe_scaled_schoenfeld(cph, df)

        total_chi2, joint_df, degraded = _compute_joint_global_test(
            cph, df, rows, time_transform, scaled_resids=scaled_resids,
        )

        global_p = float(chi2.sf(total_chi2, df=joint_df))
        rows.append({
            'feature': 'GLOBAL',
            'test_statistic': total_chi2,
            'p_value': global_p,
            'p_adjusted': np.nan,
            'significant': global_p < 0.05,
        })

        # Warn from the parent (not the helper) so stacklevel resolves to the
        # user's call. The GLOBAL chi-square degraded to a sum-of-per-feature
        # approximation that assumes independence between features.
        if degraded:
            warnings.warn(
                "ph_test: joint covariance was singular/unavailable; the "
                "GLOBAL chi-square is a sum-of-per-feature approximation that "
                "assumes independence between features (ignores between-feature "
                "covariance). Per-feature rows are unaffected.",
                MethodologicalWarning,
                stacklevel=3,
            )

        return pd.DataFrame(rows, columns=_PH_COLUMNS), cph, df, scaled_resids

    except (MemoryError, RecursionError):
        raise
    except (ValueError, KeyError, RuntimeError, np.linalg.LinAlgError) as e:
        log_metric_failure(logger, "ph_test", e)
        return empty, None, None, None
    except (ImportError, TypeError, AttributeError, IndexError) as e:
        log_metric_failure(logger, "ph_test", e, warn_user=True)
        return empty, None, None, None


def ph_test(X_train, y_train, time_transform='rank'):
    """Grambsch-Therneau test for the proportional hazards assumption.

    Re-fits a lifelines CoxPHFitter internally (via the shared robust
    fallback ladder, the same fit :func:`ph_test_with_schoenfeld` and
    ``compute_schoenfeld_data`` use).

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : structured array
        Training outcomes with 'time' and 'event' fields.
    time_transform : str, default='rank'
        Time transform for the test: 'rank', 'km', 'identity', 'log'.

    Returns
    -------
    pd.DataFrame
        Per-feature rows + a 'GLOBAL' row. Columns:
        'feature', 'test_statistic', 'p_value', 'p_adjusted', 'significant'.

        Per-feature ``p_adjusted`` values use Benjamini-Hochberg FDR correction;
        ``significant`` is ``p_adjusted < 0.05``.  The GLOBAL row retains the
        raw joint-test p-value (``p_adjusted`` = NaN, ``significant`` based on
        raw p).

    Raises
    ------
    MemoryError
        Re-raised if the computation exhausts available memory; resource
        errors are never swallowed by the broad failure handler.
    RecursionError
        Re-raised if the computation exceeds the recursion limit; resource
        errors are never swallowed by the broad failure handler.
    """
    ph_df, _cph, _df, _resids = _ph_test_core(X_train, y_train, time_transform)
    return ph_df


def ph_test_with_schoenfeld(
    X_train, y_train, time_transform='rank', n_features=None,
):
    """PH test **and** tidy Schoenfeld residuals from a single Cox fit.

    The report's PH-assessment section used to fit the same Cox model twice —
    once for the Schoenfeld scatter (``compute_schoenfeld_data``) and once for
    this p-value table (``ph_test``). This entry point fits once and returns
    both, so the plot and the table are guaranteed mutually consistent and the
    plot inherits the robust fallback fit.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : structured array
        Training outcomes with 'time' and 'event' fields.
    time_transform : str, default='rank'
        Time transform for the test: 'rank', 'km', 'identity', 'log'.
    n_features : int, optional
        Number of features in the Schoenfeld frame (top by absolute
        coefficient). ``None`` keeps ``compute_schoenfeld_data``'s default of 6.

    Returns
    -------
    tuple of (ph_df, schoenfeld_df)
        ``ph_df`` is identical to :func:`ph_test`. ``schoenfeld_df`` has
        columns ``feature``, ``time``, ``residual`` (empty when the fit or
        residual computation was unavailable).
    """
    import pandas as pd

    ph_df, cph, df, scaled_resids = _ph_test_core(X_train, y_train, time_transform)

    schoenfeld_empty = pd.DataFrame(columns=["feature", "time", "residual"])
    if cph is None or scaled_resids is None:
        return ph_df, schoenfeld_empty

    from mlsurv.evaluation._diagnostics import _schoenfeld_tidy_from_cph

    schoenfeld_df = _schoenfeld_tidy_from_cph(
        cph, df, scaled_resids=scaled_resids,
        n_features=6 if n_features is None else n_features,
    )
    return ph_df, schoenfeld_df


# ---------------------------------------------------------------------------
# Linearity assumption test
# ---------------------------------------------------------------------------

_LINEARITY_COLUMNS = ['feature', 'test_statistic', 'p_value', 'p_adjusted', 'significant']

# Harrell 2015 Table 2.3 — default knot quantiles by ``n_knots``.
_RCS_KNOT_PERCENTILES = {
    3: [10, 50, 90],
    4: [5, 35, 65, 95],
    5: [5, 27.5, 50, 72.5, 95],
    6: [5, 23, 41, 59, 77, 95],
    7: [2.5, 18.33, 34.17, 50, 65.83, 81.67, 97.5],
}


def _identify_continuous_features(
    X: "pd.DataFrame", min_unique: int = 6,
) -> List[str]:
    """Numeric columns with at least ``min_unique`` distinct values.

    Harrell's ``rms`` package uses 6 as the default cutoff for treating a
    covariate as continuous when deciding whether to apply an RCS
    expansion.  Binary indicators, one-hot columns, and low-cardinality
    ordinals are automatically skipped.
    """
    continuous = []
    for col in X.columns:
        series = X[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        # nunique counts non-NaN by default
        if series.nunique() >= min_unique:
            continuous.append(col)
    return continuous


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline nonlinear basis (Harrell 2015, eq. 2.25).

    For ``n_knots = k``, returns an ``(n, k - 2)`` matrix of **nonlinear**
    basis columns — the spline basis *beyond* the linear term.  The caller
    is expected to supply the raw (z-scored) linear term separately so the
    design matrix can place each feature's nonlinear block in a known,
    contiguous column range for later covariance-submatrix extraction.

    Formula (Harrell 2015 eq. 2.25)::

        X_j(x) = ((x - t_j)_+^3
                  - (x - t_{k-1})_+^3 * (t_k - t_j) / (t_k - t_{k-1})
                  + (x - t_k)_+^3     * (t_{k-1} - t_j) / (t_k - t_{k-1}))
                 / (t_k - t_1)**2

    for ``j = 1, ..., k - 2`` where ``(u)_+ = max(u, 0)``.

    Parameters
    ----------
    x : (n,) ndarray
        Raw feature values (NaN-free).
    knots : (k,) ndarray
        Sorted knot locations (must be strictly increasing).

    Returns
    -------
    basis : (n, k - 2) ndarray
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    k = len(knots)
    if k < 3:
        raise ValueError(f"RCS basis requires >= 3 knots, got {k}")
    t1, tk = knots[0], knots[-1]
    tkm1 = knots[-2]
    denom = tk - tkm1
    scale = (tk - t1) ** 2  # Harrell's per-Shields scaling
    # Reusable positive-part cube helpers
    def cubed_plus(u):
        return np.where(u > 0, u ** 3, 0.0)

    basis = np.empty((x.shape[0], k - 2), dtype=float)
    for j in range(k - 2):
        tj = knots[j]
        term1 = cubed_plus(x - tj)
        term2 = cubed_plus(x - tkm1) * (tk - tj) / denom
        term3 = cubed_plus(x - tk) * (tkm1 - tj) / denom
        basis[:, j] = (term1 - term2 + term3) / scale
    return basis


def _zscore_column(values: np.ndarray) -> np.ndarray:
    """Z-score a 1-D array with std=0 protection."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return values - mean
    return (values - mean) / std


def _build_linearity_design(
    X: "pd.DataFrame",
    continuous: List[str],
    n_knots: int,
) -> tuple:
    """Build the stacked (linear + RCS) design DataFrame for ``linearity_test``.

    Returns
    -------
    design : DataFrame
        Columns are (in order) the z-scored linear term for every input
        column, followed by the nonlinear RCS basis columns for each
        continuous feature.  Nonlinear column names follow the convention
        ``f"{feature}__rcs{j+1}"`` for ``j = 0, ..., n_knots - 3``.
    nonlinear_cols : dict[str, list[str]]
        Map from each continuous feature to its list of nonlinear column
        names (length ``n_knots - 2``).  Features whose knots collapse
        are omitted from this map (and from the design matrix).
    skipped : list[str]
        Continuous features that were skipped because their knot quantiles
        collapsed due to ties / mass points.
    """
    percentiles = _RCS_KNOT_PERCENTILES[n_knots]
    design_cols: dict[str, np.ndarray] = {}
    nonlinear_cols: dict[str, list[str]] = {}
    skipped: list[str] = []

    # Linear (z-scored) terms for every column — continuous features will
    # also get an RCS nonlinear block appended below.
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        design_cols[col] = _zscore_column(values) if col in continuous else values

    # Nonlinear RCS expansions per continuous feature.
    # Per-feature skips are recorded in ``skipped`` and surfaced to the
    # user as a single aggregated SuggestionWarning by the caller — the
    # ``logger.info`` lines below are only for debug-level traces of
    # which specific knot values caused the collapse.
    for feat in continuous:
        raw = X[feat].to_numpy(dtype=float)
        knots = np.percentile(raw, percentiles)
        if np.unique(knots).size != n_knots:
            logger.info(
                "linearity_test: feature '%s' skipped — knot quantiles "
                "collapsed due to ties/mass points (knots=%s)",
                feat, knots.tolist(),
            )
            skipped.append(feat)
            continue
        # Build basis from z-scored values so the nonlinear columns are on
        # the same scale as the linear term (keeps the Wald covariance
        # well-conditioned).
        z = design_cols[feat]
        # Re-derive knots on the z-scored scale (equivalent to z-scoring
        # the knots, since z-scoring is affine).
        z_knots = np.percentile(z, percentiles)
        if np.unique(z_knots).size != n_knots:
            # Can happen for heavy-tied features whose raw knots survive
            # quantile uniqueness but collapse after z-scaling roundoff.
            logger.info(
                "linearity_test: feature '%s' skipped — z-scored knot "
                "quantiles collapsed (z_knots=%s)",
                feat, z_knots.tolist(),
            )
            skipped.append(feat)
            continue
        basis = _rcs_basis(z, z_knots)
        feat_cols = []
        for j in range(n_knots - 2):
            col_name = f"{feat}__rcs{j + 1}"
            design_cols[col_name] = basis[:, j]
            feat_cols.append(col_name)
        nonlinear_cols[feat] = feat_cols

    design = pd.DataFrame(design_cols, index=X.index)
    return design, nonlinear_cols, skipped


def _wald_chi2(beta: np.ndarray, V: np.ndarray) -> float:
    """Compute β̂ᵀ V⁻¹ β̂ with ``solve`` first and ``pinv`` fallback.

    Returns ``np.nan`` on total failure.  The caller is responsible for
    surfacing a non-finite result to the user via ``warnings.warn``.  A
    `MethodologicalWarning` is emitted here if the pseudoinverse path
    runs, because a singular covariance changes the statistical
    interpretation of the Wald chi-square.
    """
    try:
        solved = np.linalg.solve(V, beta)
    except np.linalg.LinAlgError:
        try:
            V_pinv = np.linalg.pinv(V)
            solved = V_pinv @ beta
            warnings.warn(
                "linearity_test: Wald covariance is singular — falling back "
                "to a pseudoinverse. The resulting chi-square is an "
                "approximation; the effective degrees of freedom may be "
                "lower than reported, so p-values should be interpreted "
                "as diagnostic.",
                MethodologicalWarning,
                stacklevel=3,
            )
        except np.linalg.LinAlgError as exc:
            # Diagnostic-only detail: the caller (_compute_per_feature_wald /
            # _compute_global_wald) already emits the user-facing MethodologicalWarning
            # about the non-finite feature, so this raw-exception line is DEBUG, not WARNING.
            logger.debug(
                "linearity_test: Wald covariance pseudoinverse failed (%s)", exc,
            )
            return float('nan')
    return float(beta @ solved)


def _linearity_preflight(
    X_train: "pd.DataFrame",
    y_train,
) -> Optional[tuple]:
    """Validate inputs and identify continuous features for linearity_test.

    Returns ``(df, y_train, continuous)`` if validation passes, or
    ``None`` if the test should abort (an appropriate warning has already
    been emitted and the caller should return the empty sentinel).
    """
    df = X_train.copy()
    df, y_train = drop_nan_rows(df, y_train, "linearity_test")
    n_effective = len(df)
    if n_effective == 0:
        warnings.warn(
            "linearity_test: no rows remain after NaN removal; returning "
            "empty result. Check X_train for all-NaN columns or rows.",
            MethodologicalWarning,
            stacklevel=3,
        )
        return None

    continuous = _identify_continuous_features(df)
    if not continuous:
        warnings.warn(
            "linearity_test: no continuous features found (numeric columns "
            "with >=6 unique values); nothing to test. The log-hazard "
            "linearity assumption applies to continuous covariates only.",
            SuggestionWarning,
            stacklevel=3,
        )
        return None

    # Sample-size guards
    n_continuous = len(continuous)
    hard_floor = max(20, 10 * n_continuous)
    if n_effective < hard_floor:
        warnings.warn(
            f"linearity_test: n={n_effective} is below the hard floor of "
            f"{hard_floor} (max(20, 10 * n_continuous={n_continuous})). "
            f"The joint Wald has essentially no power at this sample size; "
            f"returning empty result rather than a misleading null.",
            MethodologicalWarning,
            stacklevel=3,
        )
        return None
    if n_effective < 100:
        warnings.warn(
            f"linearity_test: n={n_effective} is below 100; the test is "
            f"underpowered and null results should not be interpreted as "
            f"'assumption holds'.",
            SuggestionWarning,
            stacklevel=3,
        )

    return (df, y_train, continuous)


def _fit_cox_with_fallback(
    design: "pd.DataFrame",
    duration_col: str,
    event_col: str,
    context: str,
    penalizers: tuple = (0.01, 0.1, 1.0),
    warn_stacklevel: int = 3,
):
    """Fit CoxPH with zero-variance pre-drop + escalating-penalizer ladder.

    Drops zero-variance feature columns from *design* (in place) before
    fitting — they carry no hazard information and lifelines' internal
    standardization divides by std, producing NaN gradients that no
    penalty can recover from. Then tries each penalty in *penalizers*
    until one converges. Emits a :class:`SuggestionWarning` naming the
    dropped columns and the penalty used (when > the smallest).

    Parameters
    ----------
    design : DataFrame
        Feature columns plus ``duration_col`` and ``event_col``. Mutated
        in place: zero-variance columns are removed.
    duration_col, event_col : str
        Column names for time and event.
    context : str
        Diagnostic name (e.g. ``"ph_test"``) used in warning messages.
    penalizers : tuple of float
        Penalty ladder, ascending. The first value is the "default"
        penalty — using it does not emit a ridge warning.
    warn_stacklevel : int
        ``stacklevel`` passed to :func:`warnings.warn` so the message
        is attributed to the user's call site.

    Returns
    -------
    (cph, info) : (CoxPHFitter, dict)
        ``info`` contains ``penalizer_used`` (float) and
        ``dropped_zero_var`` (list[str]).

    Raises
    ------
    lifelines.exceptions.ConvergenceError
        If every penalty in the ladder fails.
    """
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError, ConvergenceWarning
    from scipy.linalg import LinAlgWarning

    if not penalizers:
        raise ValueError(f"{context}: penalizers ladder must be non-empty")

    feature_cols = [c for c in design.columns if c not in (duration_col, event_col)]
    variances = design[feature_cols].var()
    dropped_zero_var = variances[variances == 0].index.tolist()
    if dropped_zero_var:
        design.drop(columns=dropped_zero_var, inplace=True)
        warnings.warn(
            f"{context}: dropped {len(dropped_zero_var)} zero-variance "
            f"column(s) before Cox fit (no information about hazard): "
            f"{dropped_zero_var}",
            SuggestionWarning,
            stacklevel=warn_stacklevel,
        )

    remaining_features = [c for c in design.columns if c not in (duration_col, event_col)]
    if not remaining_features:
        warnings.warn(
            f"{context}: no features remain after dropping zero-variance "
            f"columns; cannot fit Cox model.",
            MethodologicalWarning,
            stacklevel=warn_stacklevel,
        )
        raise ConvergenceError(
            f"{context}: no features remain after zero-variance pre-drop"
        )

    # Same overflow guard used at model-fit/predict sites: lifelines exponentiates
    # the linear predictor during its fit, so a large-magnitude / collinear design
    # raises a raw ``RuntimeWarning: overflow encountered in exp``. That signal is
    # already reported through the clean tiered warnings below, so drop the raw copy.
    from ..models._overflow import reemit_non_overflow

    last_err: Optional[ConvergenceError] = None
    for penalty in penalizers:
        try:
            # Record warnings rather than ignoring them: only attempts
            # that are subsequently discarded by the retry loop should
            # be silenced. Warnings from the kept fit are replayed below
            # so soft numerical diagnostics on the returned model still
            # reach the user.
            with np.errstate(over="ignore"), \
                    warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", LinAlgWarning)
                warnings.simplefilter("always", ConvergenceWarning)
                cph = CoxPHFitter(penalizer=penalty)
                cph.fit(design, duration_col=duration_col, event_col=event_col)
        except ConvergenceError as exc:
            last_err = exc
            # Per-rung convergence detail is diagnostic, not actionable: the
            # user-facing signal comes from the SuggestionWarning (a higher
            # rung succeeded) or the MethodologicalWarning (whole ladder
            # failed) below. Keep this at debug to avoid duplicating that
            # signal in the default stderr stream.
            logger.debug(
                "%s: Cox fit did not converge at penalizer=%s (%s)",
                context, penalty, exc,
            )
            continue

        # Kept attempt: replay soft diagnostics from the recorded warnings so
        # numerical-conditioning signal on the returned fit is not lost, but
        # route them through the shared filter that drops overflow /
        # ill-conditioned ``RuntimeWarning``s (the same class of issue the
        # tiered warnings below already report).
        reemit_non_overflow(caught)

        if penalty > penalizers[0]:
            warnings.warn(
                f"{context}: Cox fit required ridge penalty "
                f"penalizer={penalty} to converge (first successful in "
                f"ladder {list(penalizers)}); the Wald chi-square is now "
                f"an approximation rather than a strict test. Interpret "
                f"p-values as diagnostic.",
                SuggestionWarning,
                stacklevel=warn_stacklevel,
            )
        return cph, {"penalizer_used": penalty, "dropped_zero_var": dropped_zero_var}

    warnings.warn(
        f"{context}: Cox fit failed to converge across the entire "
        f"penalizer ladder {list(penalizers)} (last error: {last_err}); "
        f"the diagnostic cannot be computed for this design. The data "
        f"likely contains severe collinearity or quasi-separation "
        f"beyond what ridge regularization can recover.",
        MethodologicalWarning,
        stacklevel=warn_stacklevel,
    )
    raise last_err


def _fit_cox_robust(
    design: "pd.DataFrame",
    y_train,
):
    """Fit CoxPH for ``linearity_test`` via the shared fallback ladder.

    Appends ``T``/``E`` columns to *design* in place, then delegates to
    :func:`_fit_cox_with_fallback` which drops zero-variance columns
    and tries an escalating ridge penalty ladder. The ladder begins at
    ``penalizer=0`` (true unpenalized fit) so the linearity Wald test
    remains a strict test when the data permits.
    """
    design['T'] = y_train['time']
    design['E'] = y_train['event'].astype(int)
    cph, _info = _fit_cox_with_fallback(
        design,
        duration_col='T',
        event_col='E',
        context='linearity_test',
        penalizers=(0, 0.01, 0.1, 1.0),
        warn_stacklevel=4,
    )
    return cph


def _compute_per_feature_wald(
    params: "pd.Series",
    var_matrix: "pd.DataFrame",
    continuous: List[str],
    nonlinear_cols: Dict[str, List[str]],
) -> tuple:
    """Per-feature Wald chi-square on nonlinear blocks.

    Returns ``(rows, dropped_features)`` where *rows* is a list of
    per-feature result dicts and *dropped_features* lists features
    excluded due to non-finite Wald statistics.  Emits appropriate
    warnings for dropped features.
    """
    from scipy.stats import chi2

    rows: list[dict] = []
    dropped_features: list[str] = []
    for feat in continuous:
        if feat not in nonlinear_cols:
            continue  # dropped due to knot collapse (already warned)
        nl = nonlinear_cols[feat]
        beta = params.loc[nl].to_numpy(dtype=float)
        V = var_matrix.loc[nl, nl].to_numpy(dtype=float)
        W = _wald_chi2(beta, V)
        if not np.isfinite(W):
            dropped_features.append(feat)
            continue
        df_feat = len(nl)
        p_value = float(chi2.sf(W, df=df_feat))
        rows.append({
            'feature': feat,
            'test_statistic': float(W),
            'p_value': p_value,
        })

    # Surface dropped features so users notice missing rows.
    if dropped_features:
        warnings.warn(
            f"linearity_test: {len(dropped_features)} feature(s) "
            f"produced a non-finite Wald statistic and were excluded "
            f"from the result: {', '.join(dropped_features[:5])}"
            f"{' (and more)' if len(dropped_features) > 5 else ''}.",
            MethodologicalWarning,
            stacklevel=3,
        )

    if not rows:
        warnings.warn(
            "linearity_test: all continuous features produced "
            "non-finite Wald statistics; returning empty result. "
            "This usually indicates a singular covariance matrix — "
            "check for perfect collinearity among the RCS basis "
            "columns.",
            MethodologicalWarning,
            stacklevel=3,
        )

    return (rows, dropped_features)


def _compute_global_wald(
    params: "pd.Series",
    var_matrix: "pd.DataFrame",
    continuous: List[str],
    nonlinear_cols: Dict[str, List[str]],
    dropped_features: List[str],
) -> Optional[dict]:
    """Joint Wald chi-square across all tested features' nonlinear blocks.

    Returns a result dict for the ``'GLOBAL'`` row, or ``None`` if the
    joint Wald is non-finite (with a :class:`MethodologicalWarning`
    emitted).
    """
    from scipy.stats import chi2

    tested = [f for f in continuous if f not in dropped_features]
    all_nl_cols = [
        c for feat in tested for c in nonlinear_cols.get(feat, [])
    ]
    beta_all = params.loc[all_nl_cols].to_numpy(dtype=float)
    V_all = var_matrix.loc[all_nl_cols, all_nl_cols].to_numpy(dtype=float)
    W_all = _wald_chi2(beta_all, V_all)
    if np.isfinite(W_all):
        global_df = len(all_nl_cols)
        global_p = float(chi2.sf(W_all, df=global_df))
        return {
            'feature': 'GLOBAL',
            'test_statistic': float(W_all),
            'p_value': global_p,
            'p_adjusted': np.nan,
            'significant': bool(global_p < 0.05),
        }

    warnings.warn(
        "linearity_test: GLOBAL joint Wald statistic is non-finite "
        "(singular covariance block); the GLOBAL row is omitted. "
        "Per-feature p-values are still valid but the joint test "
        "is unavailable for this dataset.",
        MethodologicalWarning,
        stacklevel=3,
    )
    return None


def linearity_test(
    X_train,
    y_train,
    n_knots: int = 4,
) -> "pd.DataFrame":
    """Test log-hazard linearity for continuous covariates via single-fit Wald.

    Follows Harrell 2015 §5.5 and R's ``rms::anova.cph``: fits **one**
    unpenalized Cox model whose design matrix stacks, for every continuous
    covariate, a z-scored linear term plus a restricted-cubic-spline
    basis (``n_knots`` knots at Harrell's default quantiles; Harrell 2015
    Table 2.3).  For each continuous feature the joint nonlinearity Wald
    chi-square with ``df = n_knots - 2`` is extracted from the feature's
    nonlinear coefficient vector and its covariance submatrix.  A GLOBAL
    joint Wald is also computed across the nonlinear blocks of every
    feature that was successfully tested, with
    ``df = (n_knots - 2) * n_tested``, where ``n_tested`` is the number
    of continuous features not skipped due to knot collapse or singular
    covariance.

    Only numeric columns with ``nunique() >= 6`` are treated as continuous
    (Harrell's default cutoff).  Binary / one-hot / low-cardinality ordinal
    columns are passed through as linear terms and are excluded from the
    linearity test.  Features whose knot quantiles collapse due to ties
    (e.g. variables with a large mass point at zero) are dropped with a
    logged warning and excluded from both the per-feature rows and the
    GLOBAL block.

    Parameters
    ----------
    X_train : DataFrame
        Training feature matrix.
    y_train : structured array
        Training outcomes with 'time' and 'event' fields.
    n_knots : int, default=4
        Number of RCS knots (supported: 3, 4, 5, 6, 7).  Harrell recommends
        4 for most datasets.

    Returns
    -------
    DataFrame
        Per-continuous-feature rows + a ``'GLOBAL'`` row.  Columns:
        ``feature``, ``test_statistic``, ``p_value``, ``p_adjusted``,
        ``significant``.  Per-feature ``p_adjusted`` uses Benjamini-Hochberg
        FDR correction; ``significant`` is ``p_adjusted < 0.05``.  The
        GLOBAL row retains the raw joint-Wald p-value (``p_adjusted = NaN``,
        ``significant`` based on raw p).  Returns an empty DataFrame when
        the data has no continuous features, too few rows, or when lifelines
        fails to converge under both unpenalized and fallback penalties.

    Raises
    ------
    MemoryError
        Re-raised if the computation exhausts available memory; resource
        errors are never swallowed by the broad failure handler.
    RecursionError
        Re-raised if the computation exceeds the recursion limit; resource
        errors are never swallowed by the broad failure handler.

    Notes
    -----
    The Wald test matches ``rms::anova.cph`` output (Harrell 2015, *Regression
    Modeling Strategies*, 2nd ed., §5.5).  The fit is unpenalized so the
    chi-square reference distribution is valid; if the unpenalized fit
    raises a ``ConvergenceError``, we retry once with ``penalizer=0.01``
    and emit a :class:`~mlsurv._warnings.SuggestionWarning` noting that the
    diagnostic is now an approximation rather than a strict test.

    References
    ----------
    Harrell, F. E. (2015). *Regression Modeling Strategies*, 2nd ed.,
    §2.4.4 (RCS basis), §4.6 (assessment of model fit), §5.5 (Cox model).
    Springer.  ``rms::anova.cph`` is the canonical R implementation this
    function mirrors.

    Therneau, T. M., Grambsch, P. M., & Fleming, T. R. (1990).
    "Martingale-based residuals for survival models." *Biometrika*
    77(1):147–160.  The companion martingale-residual plot
    (``compute_martingale_data`` / ``plot_martingale_interactive``) is
    from this paper.

    Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false
    discovery rate: a practical and powerful approach to multiple
    testing." *JRSS B* 57(1):289–300.  Used for the per-feature
    ``p_adjusted`` column.
    """
    empty = pd.DataFrame(columns=_LINEARITY_COLUMNS)

    # Preflight (outside try/except so genuine bugs surface immediately)
    preflight = _linearity_preflight(X_train, y_train)
    if preflight is None:
        return empty
    df, y_train, continuous = preflight

    try:
        from scipy.stats import false_discovery_control

        # Design build
        design, nonlinear_cols, skipped = _build_linearity_design(
            df, continuous, n_knots,
        )

        if skipped:
            # Aggregated user-visible warning — the individual skip log
            # lines inside _build_linearity_design are info-only; users
            # need to see a single consolidated notice.
            warnings.warn(
                f"linearity_test: {len(skipped)} continuous feature(s) "
                f"skipped because their knot quantiles collapsed due to "
                f"ties / mass points: "
                f"{', '.join(skipped[:5])}"
                f"{' (and more)' if len(skipped) > 5 else ''}. "
                f"Consider transforming or excluding these features.",
                SuggestionWarning,
                stacklevel=2,
            )

        if not nonlinear_cols:
            warnings.warn(
                "linearity_test: all continuous features had collapsed "
                "knot quantiles (heavy ties or mass points); nothing to "
                "test. Consider transforming the affected features or "
                "removing mass-point observations before re-running.",
                MethodologicalWarning,
                stacklevel=2,
            )
            return empty

        # Cox fit
        cph = _fit_cox_robust(design, y_train)
        params = cph.params_
        var_matrix = cph.variance_matrix_

        # Per-feature Wald
        rows, dropped_features = _compute_per_feature_wald(
            params, var_matrix, continuous, nonlinear_cols,
        )
        if not rows:
            return empty

        # Benjamini-Hochberg FDR correction on per-feature p-values
        raw_pvals = np.array([r['p_value'] for r in rows])
        adj_pvals = false_discovery_control(raw_pvals, method='bh')
        for row, adj_p in zip(rows, adj_pvals):
            row['p_adjusted'] = float(adj_p)
            row['significant'] = bool(adj_p < 0.05)

        # Global Wald
        global_row = _compute_global_wald(
            params, var_matrix, continuous, nonlinear_cols,
            dropped_features,
        )
        if global_row is not None:
            rows.append(global_row)

        return pd.DataFrame(rows, columns=_LINEARITY_COLUMNS)

    except (MemoryError, RecursionError):
        raise
    except (ValueError, KeyError, RuntimeError, np.linalg.LinAlgError,
            ImportError, TypeError, AttributeError, IndexError) as e:
        log_metric_failure(logger, "linearity_test", e, warn_user=True)
        return empty


class RMSTResult(NamedTuple):
    """Result of an RMST comparison between two groups."""

    rmst: Dict[str, float]
    """RMST estimate per group label."""
    se: Dict[str, float]
    """Standard error per group."""
    diff: float
    """RMST difference (group 1 - group 0)."""
    diff_se: float
    """SE of the difference."""
    ci_lower: float
    """Lower 95% CI for the difference."""
    ci_upper: float
    """Upper 95% CI for the difference."""
    z_stat: float
    """z-statistic for the difference."""
    p_value: float
    """Two-sided p-value for the difference."""
    tau: float
    """Time horizon used."""


class PairwiseRMSTResult(NamedTuple):
    """Result of pairwise RMST comparisons across 2+ groups."""

    rmst: Dict[str, float]
    """RMST estimate per group label."""
    se: Dict[str, float]
    """Standard error per group."""
    pairwise: List[Dict]
    """Per-pair dicts with keys: groups, diff, diff_se, ci_lower, ci_upper, raw_p, adjusted_p."""
    tau: float
    """Time horizon used."""
    p_adjust_method: str
    """Multiplicity correction method applied."""


def _step_area(t_pts, s_pts):
    """Area under a left-continuous step function via vectorized multiply.

    Parameters
    ----------
    t_pts, s_pts : ndarray
        Time points and corresponding survival values (same length).

    Returns
    -------
    float
    """
    dt = np.diff(t_pts)
    return float(np.dot(s_pts[:-1], dt))


def _km_step_points(kmf, tau):
    """Extract deduplicated KM step-function coordinates on [0, tau].

    Returns arrays ``(t_pts, s_pts)`` with the origin prepended and *tau*
    appended, duplicate time points resolved to their last (lowest survival)
    value.
    """
    sf = kmf.survival_function_
    km_times = sf.index.values
    km_surv = sf.iloc[:, 0].values

    mask = km_times <= tau
    t_pts = np.concatenate([[0.0], km_times[mask], [tau]])
    last_surv = km_surv[mask][-1] if mask.any() else 1.0
    s_pts = np.concatenate([[1.0], km_surv[mask], [last_surv]])

    # Keep last occurrence at each duplicate time (lowest survival value)
    _, last_idx = np.unique(t_pts[::-1], return_index=True)
    last_idx = np.sort(len(t_pts) - 1 - last_idx)
    t_pts = t_pts[last_idx]
    s_pts = s_pts[last_idx]

    return t_pts, s_pts


def rmst(times, events, tau=None):
    """Compute Restricted Mean Survival Time (area under KM curve up to tau).

    Parameters
    ----------
    times : array-like
        Observed event/censoring times.
    events : array-like
        Event indicators (1=event, 0=censored).
    tau : float, optional
        Time horizon. Defaults to the maximum observed time.

    Returns
    -------
    tuple of (float, float)
        (rmst_value, se) — RMST estimate and its standard error.

    Raises
    ------
    ValueError
        If no observations are supplied, if ``times`` or ``events`` contain
        NaN values, or if ``tau`` is not positive.

    References
    ----------
    Tian L, Zhao L, Wei LJ. Predicting the restricted mean event time with
    the subject's baseline covariates in survival analysis. Biostatistics.
    2014;15(2):222-233.
    """
    from lifelines import KaplanMeierFitter

    times = np.asarray(times, dtype=float)
    events = np.asarray(events)

    if len(times) == 0:
        raise ValueError("rmst() requires at least one observation.")
    if np.any(np.isnan(times)):
        raise ValueError("times contains NaN values; remove or impute before computing RMST.")
    if np.issubdtype(events.dtype, np.floating) and np.any(np.isnan(events)):
        raise ValueError("events contains NaN values; remove or impute before computing RMST.")
    if tau is not None and tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")

    if tau is None:
        tau = float(times.max())

    kmf = KaplanMeierFitter()
    kmf.fit(times, events)

    t_pts, s_pts = _km_step_points(kmf, tau)
    rmst_val = _step_area(t_pts, s_pts)

    # Greenwood-based variance for RMST (Tian et al. 2014)
    # Var(RMST) = sum over event times t_j <= tau of:
    #   d_j / (n_j * (n_j - d_j)) * [integral from t_j to tau of S(t) dt]^2
    event_table = kmf.event_table
    table_tau = event_table.loc[event_table.index <= tau]

    event_times = table_tau.index.values
    deaths = table_tau['observed'].values
    at_risk = table_tau['at_risk'].values

    # Filter to times with valid denominators (n_j > d_j and n_j > 0)
    valid = (at_risk > deaths) & (at_risk > 0)
    event_times = event_times[valid]
    deaths = deaths[valid]
    at_risk = at_risk[valid]

    # Compute tail areas: integral of S(t) from each t_j to tau
    tail_areas = np.empty(len(event_times))
    for k, t_j in enumerate(event_times):
        tail_mask = (t_pts >= t_j) & (t_pts <= tau)
        t_tail = t_pts[tail_mask]
        s_tail = s_pts[tail_mask]
        tail_areas[k] = _step_area(t_tail, s_tail) if len(t_tail) >= 2 else 0.0

    var_rmst = float(np.sum(deaths / (at_risk * (at_risk - deaths)) * tail_areas ** 2))

    se_rmst = float(np.sqrt(var_rmst))
    return float(rmst_val), se_rmst


def rmst_test(times, events, groups, tau=None):
    """Test RMST difference between two risk groups.

    Parameters
    ----------
    times : array-like
        Observed event/censoring times.
    events : array-like
        Event indicators (1=event, 0=censored).
    groups : array-like
        Group labels (must have exactly 2 unique values).
    tau : float, optional
        Time horizon. Defaults to the minimum of maximum observed times
        across groups (standard convention).

    Returns
    -------
    RMSTResult
        RMST per group, difference with CI and p-value.

    Raises
    ------
    ValueError
        If groups does not have exactly 2 unique values.
    """
    from scipy.stats import norm

    times = np.asarray(times, dtype=float)
    events = np.asarray(events)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        raise ValueError(
            f"rmst_test requires exactly 2 groups, got {len(unique_groups)}: "
            f"{unique_groups.tolist()}"
        )

    g0, g1 = unique_groups
    mask0 = groups == g0
    mask1 = groups == g1

    # Default tau: minimum of max observed times per group
    if tau is None:
        tau = float(min(times[mask0].max(), times[mask1].max()))

    rmst0, se0 = rmst(times[mask0], events[mask0], tau=tau)
    rmst1, se1 = rmst(times[mask1], events[mask1], tau=tau)

    diff = rmst1 - rmst0
    diff_se = float(np.sqrt(se0 ** 2 + se1 ** 2))

    if diff_se > 0:
        z_stat = diff / diff_se
        p_value = float(2 * norm.sf(abs(z_stat)))
    else:
        z_stat = 0.0
        p_value = 1.0

    ci_lower = diff - 1.96 * diff_se
    ci_upper = diff + 1.96 * diff_se

    rmst_dict = {str(g0): rmst0, str(g1): rmst1}
    se_dict = {str(g0): se0, str(g1): se1}

    return RMSTResult(
        rmst=rmst_dict, se=se_dict,
        diff=diff, diff_se=diff_se,
        ci_lower=ci_lower, ci_upper=ci_upper,
        z_stat=z_stat, p_value=p_value, tau=tau,
    )


def rmst_test_pairwise(times, events, groups, tau=None, p_adjust="holm"):
    """Pairwise RMST comparisons across 2+ groups.

    Parameters
    ----------
    times : array-like
        Observed event/censoring times.
    events : array-like
        Event indicators (1=event, 0=censored).
    groups : array-like
        Group labels (must have >= 2 unique values).
    tau : float, optional
        Time horizon. Defaults to the minimum of maximum observed times
        across all groups.
    p_adjust : str
        Multiplicity correction: ``"holm"``, ``"bh"``, ``"bonferroni"``,
        ``"none"``.

    Returns
    -------
    PairwiseRMSTResult

    Raises
    ------
    ValueError
        If ``p_adjust`` is not one of the supported correction methods,
        or if fewer than 2 distinct groups are present.
    """
    from itertools import combinations

    from mlsurv.evaluation._paired_comparison import _adjust_p_values

    _valid = {"holm", "bh", "bonferroni", "none"}
    if p_adjust not in _valid:
        raise ValueError(
            f"Unknown p_adjust {p_adjust!r}. Choose from {sorted(_valid)}."
        )

    times = np.asarray(times, dtype=float)
    events = np.asarray(events)
    groups = np.asarray(groups)

    unique_groups = sorted(np.unique(groups), key=str)
    n = len(unique_groups)
    if n < 2:
        raise ValueError(
            f"rmst_test_pairwise requires >= 2 groups, got {n}: "
            f"{[str(g) for g in unique_groups]}"
        )

    # Default tau: minimum of per-group max observed times
    if tau is None:
        tau = float(min(times[groups == g].max() for g in unique_groups))

    # Per-group RMST
    rmst_dict = {}
    se_dict = {}
    for g in unique_groups:
        mask = groups == g
        r, s = rmst(times[mask], events[mask], tau=tau)
        rmst_dict[str(g)] = r
        se_dict[str(g)] = s

    # Pairwise comparisons
    pairs = list(combinations(unique_groups, 2))
    pairwise = []
    raw_ps = []
    for g0, g1 in pairs:
        if n == 2:
            result = rmst_test(times, events, groups, tau=tau)
        else:
            result = _rmst_test_pair(times, events, groups, g0, g1, tau)
        pairwise.append({
            "groups": (str(g0), str(g1)),
            "diff": result.diff,
            "diff_se": result.diff_se,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "raw_p": result.p_value,
            "adjusted_p": result.p_value,  # placeholder; overwritten below
        })
        raw_ps.append(result.p_value)

    # Apply multiplicity correction
    adjusted = _adjust_p_values(np.array(raw_ps), method=p_adjust)
    for i, adj_p in enumerate(adjusted):
        pairwise[i]["adjusted_p"] = float(adj_p)

    return PairwiseRMSTResult(
        rmst=rmst_dict, se=se_dict, pairwise=pairwise,
        tau=tau, p_adjust_method=p_adjust,
    )


def _rmst_test_pair(times, events, groups, g0, g1, tau):
    """RMST test for a specific pair of groups within a multi-group array."""
    mask = (groups == g0) | (groups == g1)
    return rmst_test(times[mask], events[mask], groups[mask], tau=tau)


def score_survival_model(estimator, X, y: SurvivalArray, scoring=DEFAULT_SCORING_METRIC, y_train: Optional[SurvivalArray] = None, eval_times=None):
    """
    Score a survival model using the specified metric.

    Parameters
    ----------
    estimator : fitted model or Pipeline
        Model with predict() method.
    X : DataFrame
        Features to predict on.
    y : SurvivalArray
        True survival outcomes.
    scoring : str or callable, default=DEFAULT_SCORING_METRIC
        Scoring metric:
        - 'harrell': Harrell's C-index
        - 'ipcw': Uno's C-index (IPCW-weighted)
        - 'integrated_brier': Integrated Brier score (lower is better)
        - 'ctd': Antolini's time-dependent C-index
        - callable: Custom scorer with signature (estimator, X, y) -> float
    y_train : SurvivalArray, optional
        Training outcomes (required for 'ipcw', 'integrated_brier', and 'ctd').
    eval_times : array-like, optional
        Evaluation timepoints for Brier scores. If None, uses percentiles
        of observed times.

    Returns
    -------
    float
        Score value. For C-index metrics, higher is better.
        For Brier scores, lower is better (negated for optimization).

    Raises
    ------
    ValueError
        If ``scoring`` names a diagnostic-only metric that cannot be used
        for optimization, or if ``scoring`` is an unrecognized metric name.
    """
    # Handle custom callable scorer
    if callable(scoring):
        return scoring(estimator, X, y)

    # Guard: diagnostic metrics cannot be used for scoring/optimization
    if isinstance(scoring, str):
        from ..constants import METRIC_REGISTRY
        info = METRIC_REGISTRY.get(scoring)
        if info and info.diagnostic_only:
            raise ValueError(
                f"'{scoring}' is a diagnostic metric and cannot be used for "
                f"scoring or optimization. Use 'harrell', 'ipcw', "
                f"'integrated_brier', or 'ctd' instead."
            )

    # C-index metrics
    if scoring in ('harrell', 'ipcw'):
        risk_scores = estimator.predict(X)
        return c_index(y, risk_scores, method=scoring, y_train=y_train)

    # Brier score metrics (require survival functions).
    # 'brier' retained for internal use (_bootstrap_iteration.py); not in public METRIC_REGISTRY.
    if scoring in ('brier', 'integrated_brier'):
        return _score_brier(estimator, X, y, scoring, y_train, eval_times)

    # Antolini's time-dependent C-index (requires survival functions)
    if scoring == 'ctd':
        return _score_ctd(estimator, X, y, y_train, eval_times)

    raise ValueError(
        f"Unknown scoring: {scoring}. "
        f"Options: 'harrell', 'ipcw', 'integrated_brier', 'ctd', or callable."
    )
