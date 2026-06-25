"""Shared overflow protection for Cox-family Breslow estimators."""

import re
import warnings

import numpy as np
from sksurv.functions import StepFunction

from .._warnings import ClampWarning, MethodologicalWarning  # noqa: F401 — ClampWarning re-exported for backward compat

_LP_CLIP = 20  # exp(20) ~ 4.9e8; keeps Breslow denominators in safe float64 range
# Match overflow only. "divide by zero" and "invalid value" produce
# Inf/NaN — those signal corruption that LP clamping does NOT fix, so they
# must propagate to the user instead of being swallowed.
_OVERFLOW_RE = re.compile(r"overflow")
# scipy's solve() / inv() emit ``LinAlgWarning`` (subclass of
# ``RuntimeWarning``) with message "An ill-conditioned matrix detected: ..."
# from inside sksurv's CoxPH optimizer on rank-deficient / collinear designs.
# Same root cause class as overflow ``RuntimeWarning``s, so route both
# through the same CoxPHModel fallback path. Match against
# ``RuntimeWarning`` — the issue text mentioned ``UserWarning`` but the
# actual category is ``scipy.linalg.LinAlgWarning(RuntimeWarning)``.
_ILL_CONDITIONED_RE = re.compile(r"ill[-\s]?conditioned matrix", re.IGNORECASE)
# sksurv's CoxPH IRLS optimizer emits ``ConvergenceWarning`` (subclass of
# ``UserWarning``, not ``RuntimeWarning``) when it hits ``n_iter`` without
# converging — one warning per fit, multiplied by N folds * M tuning trials
# on hard datasets. CoxPHModel surfaces one wrapper ``FitFailureWarning``
# with actionable guidance and the raw per-fold copies are suppressed here
# here.
_CONVERGENCE_RE = re.compile(r"Optimization did not converge")

# Module-level dedup for non-finite predict warnings, keyed by
# (model_display_name, func_name). Without this, bootstrap (200+ iters)
# and CV folds would emit one warning per call when a model has a
# degenerate Breslow intermediate, drowning the signal in noise. Matches
# the per-(metric, model) pattern in metrics.py::_calibration_warned.
_nonfinite_warned: set = set()


def reemit_non_overflow(caught):
    """Re-emit captured warnings except numerical-instability ones.

    Suppresses overflow ``RuntimeWarning``s and ill-conditioned-matrix
    ``RuntimeWarning``s. Both signal the same class of
    rank-deficient / collinear design issue that CoxPHModel handles via
    its fallback path; the raw library warnings would otherwise leak one
    per fold per CV iteration.

    Also suppresses sksurv ``ConvergenceWarning`` (``UserWarning``
    subclass) — CoxPHModel emits one wrapper warning per fit instead
    of the raw per-fold messages.
    """
    for w in caught:
        msg = str(w.message)
        if _CONVERGENCE_RE.search(msg):
            continue
        if issubclass(w.category, RuntimeWarning) and (
            _OVERFLOW_RE.search(msg) or _ILL_CONDITIONED_RE.search(msg)
        ):
            continue
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def _count_nonfinite(result):
    """Count non-finite (NaN/Inf) values in *result*.

    Handles ndarrays directly and ndarrays-of-StepFunctions by inspecting
    each StepFunction's ``.y`` array. Returns 0 for unrecognized types.
    """
    if not isinstance(result, np.ndarray):
        return 0
    if result.dtype == object:
        n = 0
        for item in result.ravel():
            if isinstance(item, StepFunction):
                y = np.asarray(item.y, dtype=float)
                n += int(np.sum(~np.isfinite(y)))
        return n
    return int(np.sum(~np.isfinite(result)))


def _warn_nonfinite(model_display_name, func_name, n_bad, stacklevel):
    """Emit a MethodologicalWarning for non-finite predict output.

    LP clamping prevents overflow but does NOT fix divide-by-zero or 0/0
    in Breslow / recalibration intermediates. When NaN/Inf appear in
    survival probabilities or cumulative hazards, we surface that here
    so it doesn't silently corrupt downstream Brier / AUC / patient
    reports.
    """
    if n_bad <= 0:
        return
    key = (model_display_name, func_name)
    if key in _nonfinite_warned:
        return
    _nonfinite_warned.add(key)
    warnings.warn(
        f"{model_display_name}: {n_bad} non-finite value(s) in {func_name} "
        f"output after overflow clamping. Survival estimates may be "
        f"unreliable — likely cause is divide-by-zero or 0/0 in a Breslow "
        f"or recalibration intermediate.",
        MethodologicalWarning,
        stacklevel=stacklevel,
    )


def predict_with_clamp(model, func_name, X, return_array, model_display_name):
    """Call *func_name* on *model* with clamped linear predictors.

    Temporarily monkey-patches ``model.predict`` to clip LP values to
    ``[-20, 20]`` so that ``np.exp(LP)`` stays within float64 range
    inside the Breslow estimator.

    Overflow-related ``RuntimeWarning``s are captured and suppressed;
    all other warnings are re-emitted.

    Parameters
    ----------
    model : sksurv estimator
        Fitted sksurv model (e.g., ``CoxPHSurvivalAnalysis``).
    func_name : str
        Method name to call (``"predict_survival_function"`` or
        ``"predict_cumulative_hazard_function"``).
    X : array-like
        Feature matrix.
    return_array : bool
        Passed through to the sksurv method.
    model_display_name : str
        Human-readable name for warning messages (e.g., ``"CoxPH"``).

    Returns
    -------
    result
        Output of ``model.<func_name>(X, return_array=...)``.

    Notes
    -----
    This function temporarily mutates ``model.predict`` and is **not**
    thread-safe with respect to the monkey-patch.  Both CoxPH and CoxNet
    set ``prefer_sequential_cv = True`` to avoid parallel CV, and bootstrap
    workers precompute predictions before forking.

    Overflow suppression itself *is* thread-safe: ``np.errstate`` uses
    context-local state (NumPy >= 2.0) to tell NumPy's ufunc layer to
    silently ignore overflow conditions, so no ``RuntimeWarning`` is
    generated in the first place.  This works even when
    ``warnings.catch_warnings`` (which manipulates global state) is
    ineffective in threaded contexts.  ``warnings.catch_warnings`` is
    retained to capture non-overflow warnings emitted by scikit-survival
    internals (e.g., ``ConvergenceWarning``) for selective re-emission
    via :func:`reemit_non_overflow`.
    """
    original_predict = model.predict

    def _clamped_predict(X_inner, **kwargs):
        raw = original_predict(X_inner, **kwargs)
        clipped = np.clip(raw, -_LP_CLIP, _LP_CLIP)
        n_clipped = np.count_nonzero(raw != clipped)
        if n_clipped:
            warnings.warn(
                f"{model_display_name}: {n_clipped} linear predictor(s) "
                f"clamped to [{-_LP_CLIP}, {_LP_CLIP}] to prevent overflow. "
                f"Survival estimates for these samples may be extreme "
                f"(near 0 or 1). Consider stronger regularization.",
                ClampWarning,
                stacklevel=4,
            )
        return clipped

    model.predict = _clamped_predict
    try:
        # np.errstate: suppresses numpy-level overflow at the ufunc layer
        # (thread-safe).  catch_warnings: captures any remaining Python
        # warnings (e.g., from sksurv code) for reemit_non_overflow.
        with np.errstate(over="ignore"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = getattr(model, func_name)(X, return_array=return_array)
    except Exception:
        reemit_non_overflow(caught)
        raise
    finally:
        model.predict = original_predict

    reemit_non_overflow(caught)
    _warn_nonfinite(model_display_name, func_name, _count_nonfinite(result),
                    stacklevel=4)
    return result


def breslow_call_with_clamp(
    risk_scores, breslow_fn, *args, model_name="Model", stacklevel=2, **kwargs
):
    """Clamp risk scores before a BreslowEstimator call, suppress overflow warnings.

    Used by models that call ``BreslowEstimator`` directly (XGBCox, FSSVM)
    rather than through an sksurv model's ``predict_survival_function``.

    Parameters
    ----------
    risk_scores : ndarray
        Raw risk scores from the model.
    breslow_fn : callable
        The Breslow method to call (e.g., ``BreslowEstimator().fit`` or
        ``breslow_.get_survival_function``).
    *args
        Positional arguments forwarded to *breslow_fn* after *risk_scores*.
    model_name : str
        Human-readable name for warning messages.
    stacklevel : int
        Stack level for the clamping ``UserWarning``.
    **kwargs
        Keyword arguments forwarded to *breslow_fn*.

    Returns
    -------
    result
        Output of ``breslow_fn(clipped_scores, *args, **kwargs)``.

    See Also
    --------
    predict_with_clamp : Notes section for thread-safety discussion of the
        ``np.errstate`` / ``catch_warnings`` layering.
    """
    clipped = np.clip(risk_scores, -_LP_CLIP, _LP_CLIP)
    n_clipped = np.count_nonzero(risk_scores != clipped)
    if n_clipped:
        warnings.warn(
            f"{model_name}: {n_clipped} risk score(s) clamped to "
            f"[{-_LP_CLIP}, {_LP_CLIP}] to prevent overflow. "
            f"Survival estimates for these samples may be extreme "
            f"(near 0 or 1). Consider stronger regularization.",
            ClampWarning,
            stacklevel=stacklevel,
        )
    try:
        # np.errstate: suppresses numpy-level overflow at the ufunc layer
        # (thread-safe).  catch_warnings: captures any remaining Python
        # warnings (e.g., from sksurv code) for reemit_non_overflow.
        with np.errstate(over="ignore"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = breslow_fn(clipped, *args, **kwargs)
    except Exception:
        reemit_non_overflow(caught)
        raise
    reemit_non_overflow(caught)
    _warn_nonfinite(model_name, "Breslow", _count_nonfinite(result),
                    stacklevel=stacklevel)
    return result
