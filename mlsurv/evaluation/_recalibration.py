"""Recalibration compute layer: calibration indices and isotonic helpers.

Pure computation for post-hoc ``S(t|x)`` recalibration — collecting
out-of-fold predictions, fitting per-horizon isotonic mappings, applying
them to survival-probability arrays, and summarizing calibration with
ICI/E50/E90 indices. The ``train(calibrate=...)`` orchestration and
learner-state wiring live in ``learner/_recalibration.py`` (the facade); this
module holds the numbers so that ``evaluation/`` and ``reports/`` can use
them without importing from ``mlsurv.learner``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import isnan
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression

from .._warnings import (
    FitFailureWarning,
    MethodologicalWarning,
    _dedup_warning_entries,
    _reemit_deduped_warnings,
)
from ..utils import adjust_model_njobs
from .metrics import predict_survival_functions, surv_funcs_to_probs

_MIN_OOF_SAMPLES_PER_HORIZON = 10
_DEFAULT_CALIBRATION_BINS = 10
# Floor on the censoring-survival G(t) when forming IPCW weights, so a
# near-zero G in the tail cannot blow one subject up to a dominating weight
# (weights are capped at 1 / _MIN_G_HAT). Matches the g_hat floor used by the
# IPCW-violation probe in ``evaluation.metrics``.
_MIN_G_HAT = 1e-3


@dataclass(frozen=True)
class CalibrationIndices:
    """Per-horizon calibration summary: ICI / E50 / E90 / bias / n_used.

    ``ici``, ``e50``, ``e90`` are unsigned magnitudes (mean / median / 90th
    percentile of per-bin absolute residuals); they say nothing about the
    direction of miscalibration. ``bias`` is the signed mean residual
    ``mean(observed - predicted)`` across bins: ``> 0`` means the model
    under-predicts survival (pessimistic, observed > predicted), ``< 0``
    means it over-predicts (optimistic). Numeric fields are ``NaN`` when
    fewer than ``n_bins`` usable samples remained after censoring
    filtering; ``n_used`` is always a finite int.
    """
    ici: float
    e50: float
    e90: float
    n_used: int
    bias: float = float("nan")

    @property
    def is_valid(self) -> bool:
        return not isnan(self.ici)


def compute_calibration_indices(
    y: Any,
    pred_probs: np.ndarray,
    horizon: float,
    n_bins: int = _DEFAULT_CALIBRATION_BINS,
) -> CalibrationIndices:
    """Compute (ICI, E50, E90) at a single horizon from binned calibration.

    Each sample is assigned to one of *n_bins* equal-frequency bins by its
    predicted survival probability at *horizon*. Within each bin the observed
    survival (``T > horizon``) is compared with the predicted probability;
    the per-bin absolute differences yield ICI (mean), E50 (median), and E90
    (90th percentile).

    The observed-survival target is estimated with **IPCW (inverse probability
    of censoring) weights** (Graf/Brier scheme — see
    :func:`_ipcw_calibration_weights`), the same censoring correction the
    isotonic recalibration target uses, so the reported indices are unbiased
    under right-censoring. Subjects censored before *horizon* carry zero weight
    and drop out; the retained subjects are reweighted by ``1 / G(time)``. Both
    the observed and the predicted within-bin means are computed with these
    weights, so each residual is centered on the same reweighted subpopulation.

    Parameters
    ----------
    y : SurvivalArray
        Outcomes with ``time`` and ``event`` fields, aligned with *pred_probs*.
    pred_probs : np.ndarray
        Predicted survival probabilities at *horizon*, one per sample.
    horizon : float
        Time horizon at which calibration is assessed.
    n_bins : int, default=10
        Number of equal-frequency bins of predicted probability.

    Returns
    -------
    CalibrationIndices
        Numeric fields are ``NaN`` when fewer than *n_bins* usable samples
        remain after censoring filtering. ``n_used`` is always a finite int.

    Raises
    ------
    ValueError
        If ``pred_probs`` does not have the same shape as the outcome
        arrays in ``y``.
    """
    t = np.asarray(y["time"])
    e = np.asarray(y["event"]).astype(bool)
    p = np.asarray(pred_probs, dtype=float)
    if p.shape != t.shape:
        raise ValueError(
            f"pred_probs shape {p.shape} != y shape {t.shape}"
        )
    finite = np.isfinite(p)
    # IPCW weights; censored-before-h subjects get weight 0 (uninformative) and
    # so are dropped exactly as in the isotonic fit, leaving indices and fit on
    # the same retained subjects. Only the censoring-distribution FIT is guarded:
    # if G(t) cannot be estimated (canonically a degenerate all-censored cohort
    # with no events, so IPCW is undefined — but this catches any "G unavailable"
    # case), fall back to unweighted complete-case (keep everyone except
    # censored-before-h, identical to uniform weights) and tell the user the
    # indices are no longer IPCW-corrected. The weight computation stays OUTSIDE
    # the try so a genuine bug there surfaces instead of silently degrading.
    try:
        cde = _fit_censoring_distribution(y)
    except ValueError:
        cde = None
    if cde is None:
        w_all = (~((t <= float(horizon)) & ~e)).astype(float)
        warnings.warn(
            "Calibration indices: the censoring distribution G(t) could not be "
            "estimated (degenerate cohort); falling back to unweighted "
            "complete-case. These indices are NOT IPCW-corrected and may be "
            "biased under heavy censoring.",
            MethodologicalWarning, stacklevel=2,
        )
    else:
        w_all = _ipcw_calibration_weights(t, e, float(horizon), cde)
    keep = finite & (w_all > 0)
    pk = p[keep]
    obs = (t[keep] > float(horizon)).astype(float)
    w = w_all[keep]
    if pk.size < n_bins:
        return CalibrationIndices(np.nan, np.nan, np.nan, int(pk.size))

    edges = np.quantile(pk, np.linspace(0.0, 1.0, n_bins + 1))
    bin_idx = np.clip(np.digitize(pk, edges[1:-1]), 0, n_bins - 1)
    signed = []
    for b in range(n_bins):
        mask_b = bin_idx == b
        # Skip an empty bin or one whose IPCW weights sum to zero (no
        # informative mass) so the weighted average never divides by zero.
        if not mask_b.any() or w[mask_b].sum() <= 0:
            continue
        obs_bar = float(np.average(obs[mask_b], weights=w[mask_b]))
        pred_bar = float(np.average(pk[mask_b], weights=w[mask_b]))
        signed.append(obs_bar - pred_bar)
    if not signed:
        return CalibrationIndices(np.nan, np.nan, np.nan, int(pk.size))
    signed_arr = np.asarray(signed, dtype=float)
    diffs_arr = np.abs(signed_arr)
    return CalibrationIndices(
        ici=float(diffs_arr.mean()),
        e50=float(np.median(diffs_arr)),
        e90=float(np.percentile(diffs_arr, 90)),
        n_used=int(pk.size),
        bias=float(signed_arr.mean()),
    )


def _collect_oof_survival(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: Any,
    splitter: Any,
    horizons: np.ndarray,
    *,
    stratify_array: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    seen_messages: Optional[set] = None,
    want_scores: bool = False,
) -> Any:
    """Collect out-of-fold ``S(t|x)`` predictions across the training cohort.

    Re-fits *pipeline* on each CV fold's training portion and predicts at
    *horizons* on the held-out portion. The returned matrix gives each
    training sample exactly one OOF prediction at each horizon — no sample
    is ever predicted by a fold that saw it during fit.

    When ``want_scores`` is True, also collects each fold's held-out risk
    score (``fold_pipeline.predict``) into a parallel ``(n,)`` array and
    returns ``(oof_S, oof_scores)`` instead of just ``oof_S``. The scores
    feed the ``breslow_slope`` calibrator (van Houwelingen refit on OOF
    scores); they are only meaningful for score-based models.

    Module-level for joblib pickling (matches ``_train_cv_fold`` in
    ``learner/_cv_training.py``).
    """
    horizons = np.asarray(horizons, dtype=float)
    n = len(X_train)
    out = np.full((n, len(horizons)), np.nan, dtype=float)
    scores = np.full(n, np.nan, dtype=float) if want_scores else None
    seen = np.zeros(n, dtype=bool)

    split_y = stratify_array if stratify_array is not None else y_train["event"]
    n_folds = 0
    # Capture per-fold warnings (e.g. CoxPH alpha=0 numerical-instability refit)
    # and re-emit them once with an aggregated count instead of one copy per
    # fold. Mirrors the CV-fold dedup in
    # ``_cv_training._deduplicate_worker_warnings`` / ``_emit_cv_warnings``.
    with warnings.catch_warnings(record=True) as caught:
        # "always" so every per-fold emission is captured; we count occurrences
        # ourselves via _dedup_warning_entries and emit a single aggregated copy.
        warnings.simplefilter("always")
        for train_idx, val_idx in splitter.split(X_train, split_y, groups=groups):
            n_folds += 1
            fold_pipeline = clone(pipeline)
            fold_pipeline = adjust_model_njobs(fold_pipeline, outer_n_jobs=1)
            fold_pipeline.fit(X_train.iloc[train_idx], y_train[train_idx])

            if not hasattr(fold_pipeline, "predict_survival_function"):
                raise ValueError(
                    f"Pipeline does not expose predict_survival_function; "
                    f"isotonic recalibration of S(t|x) is not supported for this model."
                )
            surv = predict_survival_functions(
                fold_pipeline, X_train.iloc[val_idx], horizons,
            )
            probs = surv_funcs_to_probs(surv, horizons)
            out[val_idx, :] = probs
            if want_scores:
                scores[val_idx] = np.asarray(
                    fold_pipeline.predict(X_train.iloc[val_idx]), dtype=float,
                )
            seen[val_idx] = True

    # Only the per-fold refit FitFailureWarning is the one repeating itself
    # 5x; other warnings (methodological, prediction-path, sklearn convergence,
    # etc.) carry distinct semantics per fold and must surface verbatim with
    # their original category. Aggregating them would silently merge unrelated
    # signals and downgrade unknown categories via the
    # ``mlsurv._warnings._WARNING_CLASSES`` map.
    dedup_entries = []
    for w in caught:
        if issubclass(w.category, FitFailureWarning):
            dedup_entries.append(
                (w.category.__name__, str(w.message), w.filename, w.lineno)
            )
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    _reemit_deduped_warnings(
        _dedup_warning_entries(dedup_entries),
        n_folds,
        scope_label="recalibration folds",
        seen_messages=seen_messages,
    )

    if not seen.all():
        n_missed = int((~seen).sum())
        warnings.warn(
            f"Recalibration: {n_missed}/{n} training samples never appeared "
            f"in a validation fold; their OOF predictions are NaN and will "
            f"be dropped from isotonic fits.",
            MethodologicalWarning, stacklevel=2,
        )
    if want_scores:
        return out, scores
    return out


def _fit_censoring_distribution(y_train: Any) -> Any:
    """Fit ``G(t)``, the KM estimate of the censoring survival, on *y_train*.

    Returns a fitted ``sksurv.nonparametric.CensoringDistributionEstimator``
    (the same estimator the IPCW Brier path uses), so the recalibration target
    and the metrics it is judged against share one censoring model.
    """
    from sksurv.nonparametric import CensoringDistributionEstimator

    cde = CensoringDistributionEstimator()
    cde.fit(y_train)
    return cde


def _ipcw_calibration_weights(
    t: np.ndarray,
    e: np.ndarray,
    horizon: float,
    cde: Any,
) -> np.ndarray:
    """Graf/Brier IPCW weights for calibration at a single horizon.

    Returns one weight per subject for estimating ``P(T > horizon)`` from
    right-censored data without bias:

    - subject with an event at or before ``horizon`` -> ``1 / G(T_i)``
      (observed indicator ``I[T>h] = 0``),
    - subject still at risk past ``horizon`` -> ``1 / G(horizon)``
      (observed indicator ``I[T>h] = 1``),
    - subject censored before ``horizon`` -> ``0`` (uninformative).

    ``G`` is the Kaplan-Meier estimate of the censoring survival function
    (``cde``, a fitted ``CensoringDistributionEstimator``), evaluated
    right-continuously at the event/horizon time (the same ``predict_proba``
    convention as the package's IPCW Brier path). The textbook Graf weight uses
    ``G(T_i-)``; the two differ only at exact event/censoring ties. The
    complete-case
    target this replaces simply dropped the censored-before-h subjects and
    left the rest unweighted, which biases the observed-survival target low
    under heavy late censoring; the inverse-censoring weights undo that bias.
    Weights are floored at ``1 / _MIN_G_HAT`` so a near-zero ``G`` in the tail
    cannot blow a single subject up to a dominating weight.
    """
    t = np.asarray(t, dtype=float)
    e = np.asarray(e).astype(bool)
    h = float(horizon)
    w = np.zeros(t.shape[0], dtype=float)

    event_before = (t <= h) & e
    at_risk = t > h
    if event_before.any():
        g_event = np.clip(cde.predict_proba(t[event_before]), _MIN_G_HAT, None)
        w[event_before] = 1.0 / g_event
    if at_risk.any():
        g_h = np.clip(
            cde.predict_proba(np.full(int(at_risk.sum()), h)), _MIN_G_HAT, None,
        )
        w[at_risk] = 1.0 / g_h
    return w


def _fit_isotonic_recalibrators(
    oof_S: np.ndarray,
    horizons: np.ndarray,
    y_train: Any,
) -> tuple:
    """Fit one IsotonicRegression per horizon on (S_pred, I[T>h]) pairs.

    The isotonic target ``I[T>h]`` is fit with **IPCW (inverse probability of
    censoring) sample weights** (Graf/Brier scheme — see
    :func:`_ipcw_calibration_weights`) so right-censoring does not bias the
    observed-survival target. Subjects censored before a horizon carry zero
    weight and drop out; the retained subjects are reweighted by
    ``1 / G(time)``. Horizons with fewer than ``_MIN_OOF_SAMPLES_PER_HORIZON``
    nonzero-weight samples are skipped, leaving them uncalibrated.

    Returns
    -------
    (per_horizon, dropped) : tuple
        ``per_horizon``: ``{horizon: IsotonicRegression}`` for every fit.
        ``dropped``: ``[(horizon, n_used), ...]`` for every horizon that
        was skipped due to insufficient samples — the caller should
        surface this list so the user is aware of partial coverage.
    """
    per_horizon: Dict[float, IsotonicRegression] = {}
    dropped: List[tuple] = []
    t = np.asarray(y_train["time"])
    e = np.asarray(y_train["event"]).astype(bool)
    cde = _fit_censoring_distribution(y_train)

    for h_idx, h in enumerate(horizons):
        pred = oof_S[:, h_idx]
        finite = np.isfinite(pred)
        # IPCW weights; censored-before-h subjects get weight 0 (uninformative).
        w = _ipcw_calibration_weights(t, e, float(h), cde)
        mask = finite & (w > 0)
        n_used = int(mask.sum())
        if n_used < _MIN_OOF_SAMPLES_PER_HORIZON:
            dropped.append((float(h), n_used))
            continue
        obs = (t[mask] > float(h)).astype(float)
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(pred[mask], obs, sample_weight=w[mask])
        per_horizon[float(h)] = iso
    return per_horizon, dropped


def _iso_predict_anchored(iso: IsotonicRegression, x: np.ndarray) -> np.ndarray:
    """Apply *iso* with linear extrapolation to (0, 0) and (1, 1).

    sklearn's ``IsotonicRegression(out_of_bounds="clip")`` saturates inputs
    outside the training input range to a single output value, which collapses
    bootstrap CI bands at every grid cell where the raw ``S(t|x)`` lies near 0
    or 1. Replacing that with linear interpolation between the iso's
    boundary point and the natural endpoints (0, 0) / (1, 1) keeps the mapping
    monotone and preserves between-iteration variance at the boundaries.

    The endpoints are principled: a model predicting ``S = 0`` at horizon ``h``
    has the patient certain to die by ``h`` (observed indicator ``I[T>h] = 0``),
    and ``S = 1`` correspondingly maps to ``I[T>h] = 1``.
    """
    x = np.asarray(x, dtype=float)
    x_min = float(iso.X_min_)
    x_max = float(iso.X_max_)
    y_min = float(iso.predict(np.asarray([x_min]))[0])
    y_max = float(iso.predict(np.asarray([x_max]))[0])

    out = np.empty_like(x)
    below = x < x_min
    above = x > x_max
    inside = ~below & ~above
    if inside.any():
        out[inside] = iso.predict(x[inside])
    if below.any():
        # Linear from (0, 0) to (x_min, y_min)
        if x_min > 0.0:
            out[below] = x[below] * (y_min / x_min)
        else:
            out[below] = y_min
    if above.any():
        # Linear from (x_max, y_max) to (1, 1)
        if x_max < 1.0:
            out[above] = y_max + (x[above] - x_max) * (
                (1.0 - y_max) / (1.0 - x_max)
            )
        else:
            out[above] = y_max
    return out


def _apply_recalibrators_to_array(
    recal: Dict[float, IsotonicRegression],
    surv_probs: np.ndarray,
    eval_times: Iterable[float],
) -> np.ndarray:
    """Apply per-horizon isotonic recalibrators to a 2D ``S(t|x)`` array.

    For each column (eval time ``t``), the recalibrated value is a linear
    blend between the two nearest fitted-horizon isotonic mappings — using
    nearest-neighbour assignment instead would create sharp jumps at horizon
    boundaries that ``minimum.accumulate`` then absorbs asymmetrically between
    the central row and individual bootstrap-iteration rows, visibly
    compressing the CI band into the point estimate. Within each iso
    we use anchored extrapolation (see :func:`_iso_predict_anchored`) so
    inputs near 0 or 1 don't all clip to the same boundary value, which would
    otherwise collapse the band at very early / very late horizons.

    The result is clipped to ``[0, 1]`` and ``minimum.accumulate`` is applied
    row-wise (in time-ascending order) to guarantee a valid non-increasing
    survival function.

    Module-level (no ``self``) so it is joblib-pickleable and reusable inside
    bootstrap workers.
    """
    surv_probs = np.asarray(surv_probs, dtype=float)
    eval_times = np.asarray(list(eval_times), dtype=float)
    if surv_probs.ndim != 2 or surv_probs.shape[1] != eval_times.size:
        raise ValueError(
            f"surv_probs shape {surv_probs.shape} incompatible with "
            f"{eval_times.size} eval_times."
        )

    horizons = np.asarray(sorted(recal.keys()), dtype=float)
    if horizons.size == 0:
        raise ValueError(
            "No fitted recalibrators to apply (recal is empty): every horizon "
            "was dropped for insufficient usable samples, so S(t|x) cannot be "
            "recalibrated. Callers that may pass an empty mapping should skip "
            "recalibration instead."
        )
    out = np.empty_like(surv_probs)
    n_h = horizons.size
    for j, t in enumerate(eval_times):
        t_f = float(t)
        idx = int(np.searchsorted(horizons, t_f))
        if n_h == 1 or idx == 0:
            iso_a = recal[float(horizons[0])]
            out[:, j] = _iso_predict_anchored(iso_a, surv_probs[:, j])
        elif idx >= n_h:
            iso_b = recal[float(horizons[-1])]
            out[:, j] = _iso_predict_anchored(iso_b, surv_probs[:, j])
        else:
            h_lo = float(horizons[idx - 1])
            h_hi = float(horizons[idx])
            w_hi = (t_f - h_lo) / (h_hi - h_lo)
            w_lo = 1.0 - w_hi
            iso_lo = recal[h_lo]
            iso_hi = recal[h_hi]
            out[:, j] = (
                w_lo * _iso_predict_anchored(iso_lo, surv_probs[:, j])
                + w_hi * _iso_predict_anchored(iso_hi, surv_probs[:, j])
            )

    np.clip(out, 0.0, 1.0, out=out)
    order = np.argsort(eval_times, kind="stable")
    if not np.array_equal(order, np.arange(eval_times.size)):
        sorted_out = out[:, order]
        np.minimum.accumulate(sorted_out, axis=1, out=sorted_out)
        inverse = np.argsort(order, kind="stable")
        out = sorted_out[:, inverse]
    else:
        np.minimum.accumulate(out, axis=1, out=out)
    return out
