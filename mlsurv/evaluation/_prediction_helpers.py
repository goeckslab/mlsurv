"""Pure helper functions for per-patient survival predictions.

These functions convert heterogeneous model outputs to uniform arrays
and compute per-patient survival statistics (median, RMST, interpolation).
"""

from __future__ import annotations

from typing import Optional, Union

import warnings

import numpy as np
import pandas as pd

from .._shared import percentile_ci
from .._warnings import MethodologicalWarning, SuggestionWarning
from .metrics import _step_area


# ---------------------------------------------------------------------------
# Survival function extraction
# ---------------------------------------------------------------------------

def extract_time_grid_and_probs(surv_funcs):
    """Convert heterogeneous survival function output to uniform arrays.

    Parameters
    ----------
    surv_funcs : list of StepFunction, DataFrame, or ndarray
        Raw output from ``model.predict_survival_function()``.

    Returns
    -------
    tuple of (ndarray, ndarray)
        ``(time_grid, surv_probs)`` where ``time_grid`` has shape
        ``(n_times,)`` and ``surv_probs`` has shape ``(n_patients, n_times)``.

    Raises
    ------
    TypeError
        If ``surv_funcs`` is an ndarray without a recoverable time grid.
    ValueError
        If ``surv_funcs`` is an empty list of survival functions.
    """
    if isinstance(surv_funcs, pd.DataFrame):
        return _extract_from_dataframe(surv_funcs)

    if isinstance(surv_funcs, np.ndarray):
        # Object arrays of callables (StepFunction stored as ndarray)
        if surv_funcs.dtype == object and callable(surv_funcs.flat[0]):
            return _extract_from_step_functions(list(surv_funcs))
        raise TypeError(
            "Cannot extract time grid from a plain ndarray. "
            "Use predict_survival_function() without return_array=True, "
            "or provide explicit times= to learner.predict()."
        )

    # Assume list/tuple of StepFunction-like callables
    if hasattr(surv_funcs, '__iter__'):
        items = list(surv_funcs)
        if len(items) == 0:
            raise ValueError("Empty survival function list.")
        if hasattr(items[0], 'x') and hasattr(items[0], 'y'):
            return _extract_from_step_functions(items)
        raise TypeError(
            f"Unsupported survival function type: {type(items[0]).__name__}. "
            "Expected StepFunction, DataFrame, or ndarray."
        )

    raise TypeError(
        f"Unsupported survival function type: {type(surv_funcs).__name__}."
    )


def _extract_from_dataframe(df):
    """Extract from pycox-style DataFrame (times as index, samples as columns)."""
    time_grid = np.asarray(df.index, dtype=float)
    surv_probs = df.values.T  # (n_samples, n_times)
    return time_grid, np.ascontiguousarray(surv_probs)


def _step_interpolate(source_times, source_values, query_times):
    """Step-interpolate survival values onto a new time grid.

    Left-continuous: S(t) = value at the most recent source_time <= t.
    Before the first source time: 1.0.  After the last: last value.
    """
    idx = np.searchsorted(source_times, query_times, side='right') - 1
    result = np.empty(len(query_times))
    for q, j in enumerate(idx):
        if j < 0:
            result[q] = 1.0
        elif j >= len(source_values):
            result[q] = source_values[-1]
        else:
            result[q] = source_values[j]
    return result


def _extract_from_step_functions(funcs):
    """Extract from sksurv StepFunction list.

    If all functions share the same time grid, use it directly.
    Otherwise, take the union and evaluate each function on the shared grid.
    """
    grids = [np.asarray(f.x, dtype=float) for f in funcs]
    values = [np.asarray(f.y, dtype=float) for f in funcs]

    # Check if all grids are identical (common case for sksurv)
    all_same = all(
        g.shape == grids[0].shape and np.array_equal(g, grids[0])
        for g in grids[1:]
    )

    if all_same:
        time_grid = grids[0]
        surv_probs = np.vstack(values)
    else:
        # Union of all time grids; step-interpolate each function
        # onto the shared grid (StepFunction.domain may not cover all
        # times, so we interpolate manually instead of calling f(t))
        time_grid = np.unique(np.concatenate(grids))
        surv_probs = np.empty((len(funcs), len(time_grid)))
        for i in range(len(funcs)):
            surv_probs[i] = _step_interpolate(grids[i], values[i], time_grid)

    return time_grid, surv_probs


# ---------------------------------------------------------------------------
# Per-patient median survival
# ---------------------------------------------------------------------------

def compute_patient_median_survival(time_grid, surv_probs, *, warn_undefined=True):
    """Find time t where S(t) crosses 0.5 for each patient.

    Uses linear interpolation between the last time where S(t) > 0.5
    and the first time where S(t) <= 0.5.

    Parameters
    ----------
    time_grid : ndarray, shape (n_times,)
    surv_probs : ndarray, shape (n_patients, n_times)
    warn_undefined : bool, default True
        If ``True`` (default) and any patient has S(t) > 0.5 across the full
        grid (i.e., NaN median), emit a :class:`MethodologicalWarning`
        recommending RMST as a bounded alternative. Set to ``False`` when
        the count is internal bookkeeping (e.g., raw pre-recalibration
        prediction whose user-facing equivalent is the recalibrated count).

    Returns
    -------
    ndarray, shape (n_patients,)
        Median survival time per patient. ``NaN`` if S(t) > 0.5 at all times.
    """
    n_patients = surv_probs.shape[0]
    medians = np.full(n_patients, np.nan)

    for i in range(n_patients):
        s = surv_probs[i]
        # Find first index where S(t) <= 0.5
        below = np.where(s <= 0.5)[0]
        if len(below) == 0:
            continue  # S(t) > 0.5 everywhere -> NaN

        j = below[0]
        if j == 0:
            # S(t) <= 0.5 at the very first time point
            medians[i] = time_grid[0]
        else:
            # Linear interpolation between bracket points
            s_hi, s_lo = s[j - 1], s[j]
            t_hi, t_lo = time_grid[j - 1], time_grid[j]
            denom = s_hi - s_lo
            if denom == 0:
                medians[i] = t_hi
            else:
                medians[i] = t_hi + (s_hi - 0.5) / denom * (t_lo - t_hi)

    n_undefined = int(np.isnan(medians).sum())
    if n_undefined > 0 and warn_undefined:
        warnings.warn(
            f"Median survival undefined for {n_undefined} patient(s): "
            "S(t) > 0.5 at all observed time points. These patients are "
            "reported as NaN. Consider using RMST for a bounded summary.",
            MethodologicalWarning,
            stacklevel=3,
        )

    return medians


# ---------------------------------------------------------------------------
# Per-patient RMST
# ---------------------------------------------------------------------------

def compute_patient_rmst(time_grid, surv_probs, tau):
    """Compute per-patient RMST: area under S(t|x) from 0 to tau.

    Prepends t=0, S=1.0 if not present, truncates at tau.
    Uses left-continuous step function area (same as ``_step_area``).

    Parameters
    ----------
    time_grid : ndarray, shape (n_times,)
    surv_probs : ndarray, shape (n_patients, n_times)
    tau : float
        Time horizon for RMST.

    Returns
    -------
    ndarray, shape (n_patients,)
    """
    n_patients = surv_probs.shape[0]
    rmst_values = np.empty(n_patients)

    for i in range(n_patients):
        s = surv_probs[i]

        # Build (t, s) pairs with t=0 prepended and truncated at tau
        mask = time_grid <= tau
        t_pts = time_grid[mask]
        s_pts = s[mask]

        # Prepend origin (t=0, S=1.0) if not present
        if len(t_pts) == 0 or t_pts[0] > 0:
            t_pts = np.concatenate([[0.0], t_pts])
            s_pts = np.concatenate([[1.0], s_pts])

        # Append tau with last known survival
        if t_pts[-1] < tau:
            t_pts = np.append(t_pts, tau)
            s_pts = np.append(s_pts, s_pts[-1])

        rmst_values[i] = _step_area(t_pts, s_pts)

    return rmst_values


# ---------------------------------------------------------------------------
# Step interpolation at specific times
# ---------------------------------------------------------------------------

def interpolate_surv_at_times(time_grid, surv_probs, query_times):
    """Evaluate S(t|x) at specific times via step interpolation.

    Uses left-continuous step function semantics: S(t) is the value
    at the most recent grid point <= t.

    Parameters
    ----------
    time_grid : ndarray, shape (n_times,)
    surv_probs : ndarray, shape (n_patients, n_times)
    query_times : ndarray, shape (n_query,)

    Returns
    -------
    ndarray, shape (n_patients, n_query)
    """
    query_times = np.asarray(query_times, dtype=float)
    n_patients, n_grid = surv_probs.shape
    n_query = len(query_times)

    result = np.empty((n_patients, n_query))

    # np.searchsorted with side='right' gives index of first grid > query
    # Subtract 1 to get last grid <= query (left-continuous)
    indices = np.searchsorted(time_grid, query_times, side='right') - 1

    for q, idx in enumerate(indices):
        if idx < 0:
            # Query time is before the first grid point -> S = 1.0
            result[:, q] = 1.0
        elif idx >= n_grid:
            # Query time is after the last grid point -> use last value
            result[:, q] = surv_probs[:, -1]
        else:
            result[:, q] = surv_probs[:, idx]

    return result


# ---------------------------------------------------------------------------
# Percentile CI computation
# ---------------------------------------------------------------------------

def compute_percentile_ci(bootstrap_values, ci_level):
    """Compute percentile confidence intervals from bootstrap samples.

    Parameters
    ----------
    bootstrap_values : ndarray, shape (B, ...)
        Bootstrap samples along axis 0.
    ci_level : float
        Confidence level, e.g. 0.95 for 95% CIs.

    Returns
    -------
    tuple of (ndarray, ndarray)
        ``(lower, upper)`` with shape matching ``bootstrap_values[0]``.
        NaN bootstrap values are ignored (nanpercentile).
    """
    return percentile_ci(bootstrap_values, ci_level)


# ---------------------------------------------------------------------------
# Standalone pipeline prediction (no learner state required)
# ---------------------------------------------------------------------------

def predict_from_pipeline(
    pipeline,
    X: pd.DataFrame,
    *,
    model_name: str,
    times: Optional[Union[list, np.ndarray]] = None,
    rmst_default_tau: Optional[float] = None,
    warn_undefined_median: bool = True,
):
    """Predict survival for new patients using a fitted pipeline.

    This is the core prediction logic shared by ``learner.predict()``
    (which layers on bootstrap CIs) and ``cmd_predict`` in the CLI
    (which operates on a bare artifact without training data).

    Parameters
    ----------
    pipeline : fitted estimator
        Sklearn-compatible pipeline with ``.predict()`` and optionally
        ``.predict_survival_function()``.
    X : DataFrame
        Feature matrix for new patients.
    model_name : str
        User-facing model identifier stored in the result.
    times : array-like, optional
        Specific time points at which to evaluate S(t) and RMST.
    rmst_default_tau : float, optional
        RMST horizon. If ``None``, defaults to ``time_grid.max()``
        (the last event time in the model's survival function output).
        When called from ``learner.predict()``, this is the exact
        ``y_train['time'].max()``.
    warn_undefined_median : bool, default True
        Forwarded to :func:`compute_patient_median_survival`. Pass ``False``
        when the caller will re-derive the median from a recalibrated S(t|x)
        and emit the actionable warning at that downstream site instead
        (avoids duplicate MethodologicalWarning emissions per ``predict()``).

    Returns
    -------
    PredictionResult
        Container with ``risk_scores``, ``survival_function``,
        ``median_survival``, ``rmst``, and time-specific outputs.
        Bootstrap CI fields are always ``None`` (CIs require training
        data available only through the learner).

    Raises
    ------
    ValueError
        If ``times`` is not a non-empty 1-D array, contains non-finite or
        negative values, or contains duplicate time points.
    MemoryError
        Re-raised if the computation exhausts available memory; resource
        errors are never swallowed by the broad failure handler.
    RecursionError
        Re-raised if the computation exceeds the recursion limit; resource
        errors are never swallowed by the broad failure handler.
    """
    from .metrics import predict_survival_functions
    from ..results._prediction import PredictionResult

    # -- Validate times ------------------------------------------------------
    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.ndim == 0:
            times = times.reshape(1)
        if times.ndim != 1 or len(times) == 0:
            raise ValueError(
                "times must be a non-empty 1-D array of time points."
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("times must contain only finite numeric values.")
        if np.any(times < 0):
            raise ValueError("times must be >= 0.")
        if len(np.unique(times)) != len(times):
            raise ValueError("times must not contain duplicates.")

    n_patients = len(X)

    # -- Risk scores (always available) --------------------------------------
    risk_scores = np.asarray(pipeline.predict(X), dtype=float).reshape(-1)

    # -- Survival functions --------------------------------------------------
    time_grid = None
    surv_probs = None
    median_surv = None
    rmst_values = None
    effective_tau = None
    surv_at_user_times = None
    rmst_at_user_times = None

    has_surv_func = hasattr(pipeline, 'predict_survival_function')
    if has_surv_func:
        try:
            surv_funcs = predict_survival_functions(pipeline, X)
            time_grid, surv_probs = extract_time_grid_and_probs(surv_funcs)
        except (MemoryError, RecursionError):
            raise
        except (ValueError, NotImplementedError, RuntimeError,
                TypeError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"Model '{model_name}' does not support survival "
                f"function prediction: {exc}. Only risk_scores are "
                "available.",
                SuggestionWarning,
                stacklevel=2,
            )
            has_surv_func = False

    if not has_surv_func and times is not None:
        warnings.warn(
            f"Model '{model_name}' does not support "
            "predict_survival_function(). Only risk_scores are available. "
            "times parameter is ignored.",
            SuggestionWarning,
            stacklevel=2,
        )

    if surv_probs is not None:
        # Median survival
        median_surv = compute_patient_median_survival(
            time_grid, surv_probs, warn_undefined=warn_undefined_median,
        )

        # RMST at default horizon
        if rmst_default_tau is not None:
            effective_tau = float(rmst_default_tau)
        else:
            effective_tau = float(time_grid.max())
        rmst_values = compute_patient_rmst(
            time_grid, surv_probs, effective_tau,
        )

        # Time-specific outputs
        if times is not None:
            # Warn about extrapolation beyond training range
            beyond = times[times > effective_tau]
            if len(beyond) > 0:
                warnings.warn(
                    f"Requested times {beyond.tolist()} exceed the "
                    f"maximum training time ({effective_tau:.1f}). "
                    "Survival estimates beyond this point are "
                    "extrapolations and may be unreliable.",
                    MethodologicalWarning,
                    stacklevel=2,
                )

            surv_at_user_times = interpolate_surv_at_times(
                time_grid, surv_probs, times,
            )
            rmst_at_user_times = np.column_stack([
                compute_patient_rmst(time_grid, surv_probs, tau)
                for tau in times
            ])

    return PredictionResult(
        risk_scores=risk_scores,
        model_name=model_name,
        n_patients=n_patients,
        _time_grid=time_grid,
        _surv_probs=surv_probs,
        _median_survival=median_surv,
        _rmst_values=rmst_values,
        _rmst_default_tau=effective_tau,
        _user_times=times,
        _surv_at_user_times=surv_at_user_times,
        _rmst_at_user_times=rmst_at_user_times,
        _bootstrap_medians=None,
        _bootstrap_surv_at_times=None,
        _bootstrap_rmst=None,
        _bootstrap_rmst_at_times=None,
        _ci_level=0.95,
    )
