"""Concordance, AUC(t), Brier score, and calibration metric computation for evaluation."""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sksurv.metrics import (
    brier_score, concordance_index_censored, concordance_index_ipcw,
)

from ._ctd import concordance_td
from .metrics import _surv_funcs_to_probs, predict_survival_functions, calibration_slope_intercept, calibration_test, _auc_values_and_nan_mask, _predict_dense_ibs_probs, _integrated_brier_from_probs
from ..logging_utils import (
    get_logger,
    log_metric_failure,
    _is_fold_boundary_error,
    _log_once_level,
    _fold_boundary_message,
)
from ..utils import (
    SurvivalArray, clip_eval_times_to_data, supports_survival_function,
    truncate_y_to_train_range,
)
from .._warnings import MethodologicalWarning

logger = get_logger("evaluate")


def _add_brier_metrics(data, model, X_test: pd.DataFrame,
                       y_test: SurvivalArray, y_train: Optional[SurvivalArray],
                       eval_times, surv_probs_transform=None):
    """Add Brier score and IBS to data dict if model supports survival functions.

    Parameters
    ----------
    surv_probs_transform : callable, optional
        ``f(surv_probs, eval_times) -> ndarray`` applied to the survival
        probability matrix immediately after it is produced from the model
        and before any metric is computed. Used by ``validate()`` to apply
        post-hoc recalibration. When ``None``, raw probabilities flow
        through unchanged.

    Returns
    -------
    tuple of (surv_probs, eval_times, ibs_surv_probs, ibs_times)
        The display-grid survival probability array and deduplicated eval_times
        (for reuse by downstream metrics, e.g. calibration), plus the dense
        train-anchored IBS probability matrix and its grid (for reuse by
        subpopulation IBS). Any element is ``None`` when not computable.
    """
    if y_train is None or eval_times is None:
        return None, None, None, None
    # Skip silently for entities that structurally cannot produce a survival
    # curve (e.g. ReferenceAdapter raw-score wrappers).  Attempting + catching
    # here would emit ERROR log lines per model per iteration.
    if not supports_survival_function(model):
        data['brier_score_test'] = [np.nan]
        data['integrated_brier_test'] = [np.nan]
        return None, None, None, None

    # Deduplicate eval_times BEFORE passing to the dispatcher so models that
    # accept a `times` parameter return an ndarray aligned to the same grid
    # used by downstream Brier/IBS computation.
    eval_times = np.unique(np.asarray(eval_times))

    # Guard: model may define predict_survival_function (via base class)
    # but raise NotImplementedError at runtime.
    # Use dispatcher to pass eval_times for models that support it (e.g. XGBoost),
    # producing an aligned ndarray directly instead of StepFunction objects.
    try:
        surv_funcs = predict_survival_functions(model, X_test, eval_times)
    except NotImplementedError as e:
        log_metric_failure(
            logger, "brier/ibs", e,
            model_name=getattr(model, 'name', type(model).__name__),
        )
        data['brier_score_test'] = [np.nan]
        data['integrated_brier_test'] = [np.nan]
        return None, None, None, None

    # Convert survival functions to probability array once, reused by both
    # brier_score and integrated_brier_score (avoids redundant StepFunction
    # evaluation that the old wrapper functions each performed independently).
    surv_probs = _surv_funcs_to_probs(surv_funcs, eval_times)

    # Optional post-hoc transform (e.g. isotonic recalibration) applied
    # before any metric is computed so Brier/IBS/C^td/calibration all reflect
    # the same recalibrated S(t|x).
    if surv_probs_transform is not None:
        surv_probs = surv_probs_transform(surv_probs, eval_times)

    # Clip eval_times to the truncated test data's actual follow-up range
    # (bootstrap resamples / small test sets may not cover the full grid).
    valid_times, valid_mask, y_brier_ipcw = clip_eval_times_to_data(
        eval_times, y_test, y_train,
    )

    # Brier score at the nearest eval_time to the median (snaps to grid
    # rather than interpolating at an off-grid time point).
    try:
        if len(valid_times) == 0:
            data['brier_score_test'] = [np.nan]
        else:
            median_idx = np.argmin(np.abs(valid_times - np.median(valid_times)))
            _, scores = brier_score(y_train, y_brier_ipcw,
                                    surv_probs[:, valid_mask][:, median_idx],
                                    valid_times[median_idx])
            data['brier_score_test'] = [float(scores[0])]
    except (MemoryError, RecursionError):
        raise
    except (ValueError, ZeroDivisionError) as e:
        if _is_fold_boundary_error(e):
            log_metric_failure(
                logger, "brier", e,
                level=_log_once_level("fold_boundary"),
                warn_once_key="fold_boundary_warn",
                user_message=_fold_boundary_message("brier"),
            )
        else:
            log_metric_failure(logger, "brier", e, level=logging.ERROR)
        data['brier_score_test'] = [np.nan]

    # Integrated Brier score — computed over a DENSE, train-anchored event-time
    # grid inside the censoring-stable window, decoupled from the display grid
    # above. This makes IBS independent of how many display points are used for
    # Brier(t)/AUC(t) curves. Uses a separate S(t|x) prediction on the dense
    # grid (the display surv_probs above are kept for Brier(median)/C^td/
    # calibration, which must stay aligned to the display times). The dense
    # matrix is also returned so subpopulation IBS reuses it (same grid).
    ibs_surv_probs = None
    ibs_times = None
    try:
        ibs_surv_probs, ibs_times = _predict_dense_ibs_probs(
            model, X_test, y_test, y_train,
            surv_probs_transform=surv_probs_transform,
        )
        if ibs_surv_probs is None:
            data['integrated_brier_test'] = [np.nan]
        else:
            data['integrated_brier_test'] = [_integrated_brier_from_probs(
                ibs_surv_probs, ibs_times, y_test, y_train,
            )]
    except (MemoryError, RecursionError):
        raise
    except (ValueError, ZeroDivisionError) as e:
        if _is_fold_boundary_error(e):
            log_metric_failure(
                logger, "ibs", e,
                level=_log_once_level("fold_boundary"),
                warn_once_key="fold_boundary_warn",
                user_message=_fold_boundary_message("integrated_brier"),
            )
        else:
            log_metric_failure(logger, "ibs", e, level=logging.ERROR)
        data['integrated_brier_test'] = [np.nan]

    # Return display surv_probs/eval_times for downstream callers (e.g. C^td)
    # plus the dense IBS matrix/grid for subpopulation IBS reuse.
    return surv_probs, eval_times, ibs_surv_probs, ibs_times


def _add_ctd_metric(data, y_test: SurvivalArray, surv_probs, eval_times):
    """Add Antolini's time-dependent C-index to data dict.

    Reuses the survival probability matrix already computed by
    ``_add_brier_metrics``, so there is zero redundant computation.
    """
    if surv_probs is None or eval_times is None:
        data['c_index_td_test'] = [np.nan]
        return

    try:
        ctd_val = concordance_td(
            y_test['time'], y_test['event'], surv_probs, eval_times,
        )
        data['c_index_td_test'] = [ctd_val]
    except (MemoryError, RecursionError):
        raise
    except Exception as e:
        log_metric_failure(logger, "ctd", e)
        data['c_index_td_test'] = [np.nan]


def _add_calibration_metrics(data, y_test: SurvivalArray, surv_probs, eval_times,
                              model_name: Optional[str] = None,
                              warnings_out: Optional[list] = None,
                              gnd_n_bins: int = 10):
    """Add calibration slope, intercept, and GND test to data dict.

    ``model_name`` is forwarded to the metric helpers so their NaN-return
    warnings dedupe per (metric, model) instead of per metric.

    ``warnings_out``: when provided, calibration NaN-return warning
    details are appended here for parent-side emission instead of being
    passed to ``warnings.warn`` (silenced in joblib/loky workers).

    ``gnd_n_bins``: number of equal-frequency bins for the GND test.
    Lowering it lets small cohorts (N < 50) clear the >=5-samples-per-bin
    floor and compute a GND statistic instead of returning NaN.
    """
    if surv_probs is None or eval_times is None:
        data['cal_slope_test'] = [np.nan]
        data['cal_intercept_test'] = [np.nan]
        data['gnd_test'] = [np.nan]
        return

    # Use the eval_time nearest the statistical median (consistent with Brier score)
    median_idx = np.argmin(np.abs(eval_times - np.median(eval_times)))
    median_time = eval_times[median_idx]
    pred_probs = surv_probs[:, median_idx]

    try:
        slope, intercept = calibration_slope_intercept(
            y_test, pred_probs, median_time, model_name=model_name,
            warnings_out=warnings_out,
        )
        data['cal_slope_test'] = [slope]
        data['cal_intercept_test'] = [intercept]
        # Note: the poor-calibration MethodologicalWarning is emitted
        # in the parent process (mlsurv.evaluation.evaluate._warn_poor_calibration)
        # because joblib/loky workers silently drop warnings.warn.
    except (MemoryError, RecursionError):
        raise
    except (ValueError, ZeroDivisionError) as e:
        log_metric_failure(logger, "cal_slope", e)
        data['cal_slope_test'] = [np.nan]
        data['cal_intercept_test'] = [np.nan]
    except Exception as e:
        log_metric_failure(logger, "cal_slope", e, warn_user=True)
        data['cal_slope_test'] = [np.nan]
        data['cal_intercept_test'] = [np.nan]

    try:
        _, p_value = calibration_test(
            y_test, pred_probs, median_time, n_bins=gnd_n_bins,
            model_name=model_name, warnings_out=warnings_out,
        )
        data['gnd_test'] = [p_value]
    except (MemoryError, RecursionError):
        raise
    except (ValueError, ZeroDivisionError) as e:
        log_metric_failure(logger, "gnd", e)
        data['gnd_test'] = [np.nan]
    except Exception as e:
        log_metric_failure(logger, "gnd", e, warn_user=True)
        data['gnd_test'] = [np.nan]


def _add_train_metrics(data, model, X_train, y_train, eval_times, ipcw_tau):
    """Add the four apparent (resubstitution) train-set metrics to ``data``.

    Computes the standard four on ``(X_train, y_train)`` by resubstitution:
    Harrell's C (``c_index_train``), Uno's IPCW C (``c_index_ipcw_train``),
    Antolini's C^td (``c_index_td_train``), and IBS
    (``integrated_brier_train``) -- the same scalar set CV produces per fold.
    These are apparent-fit optimism diagnostics, so survival probabilities are
    predicted raw (no recalibration transform). Each metric degrades to
    ``np.nan`` independently on failure (e.g. ReferenceAdapter, which has no
    train-set reference scores). C^td and IBS require survival functions and
    are NaN for models that do not provide them.
    """
    # Harrell's C (rank-based; always attempted).
    risk_train = None
    try:
        risk_train = model.predict(X_train)
        c_train = concordance_index_censored(
            y_train['event'], y_train['time'], risk_train
        )[0]
    except (MemoryError, RecursionError):
        raise
    except Exception as e:
        # ReferenceAdapter (and similar) may return NaN for training data
        # since reference scores are only available for the test set.
        log_metric_failure(logger, 'c_index_train', e)
        c_train = np.nan
    data['c_index_train'] = [c_train]

    # Uno's IPCW C (resubstitution: G_hat fit on y_train, evaluated on y_train,
    # so no truncation is needed -- train-vs-train is a no-op).
    if risk_train is not None:
        try:
            c_ipcw_train = concordance_index_ipcw(
                y_train, y_train, risk_train, tau=ipcw_tau
            )[0]
        except (MemoryError, RecursionError):
            raise
        except Exception as e:
            log_metric_failure(logger, 'c_index_ipcw_train', e)
            c_ipcw_train = np.nan
    else:
        c_ipcw_train = np.nan
    data['c_index_ipcw_train'] = [c_ipcw_train]

    # Antolini's C^td and IBS require survival functions.
    if not supports_survival_function(model) or eval_times is None:
        data['c_index_td_train'] = [np.nan]
        data['integrated_brier_train'] = [np.nan]
        return

    # Antolini's C^td on the display grid (raw S(t|x), no recalibration).
    td_times = np.unique(np.asarray(eval_times))
    try:
        surv_funcs_train = predict_survival_functions(model, X_train, td_times)
        surv_probs_train = _surv_funcs_to_probs(surv_funcs_train, td_times)
        data['c_index_td_train'] = [concordance_td(
            y_train['time'], y_train['event'], surv_probs_train, td_times,
        )]
    except (MemoryError, RecursionError):
        raise
    except Exception as e:
        log_metric_failure(logger, 'c_index_td_train', e)
        data['c_index_td_train'] = [np.nan]

    # IBS on the dense, train-anchored grid (resubstitution: y_train is both
    # the evaluated cohort and the censoring cohort).
    try:
        ibs_probs_train, ibs_times_train = _predict_dense_ibs_probs(
            model, X_train, y_train, y_train,
        )
        if ibs_probs_train is None:
            data['integrated_brier_train'] = [np.nan]
        else:
            data['integrated_brier_train'] = [_integrated_brier_from_probs(
                ibs_probs_train, ibs_times_train, y_train, y_train,
            )]
    except (MemoryError, RecursionError):
        raise
    except Exception as e:
        log_metric_failure(logger, 'integrated_brier_train', e)
        data['integrated_brier_train'] = [np.nan]


def _compute_concordance(model, X_test: pd.DataFrame,
                         y_test: SurvivalArray, X_train=None,
                         y_train: Optional[SurvivalArray] = None,
                         eval_times=None, custom_metrics=None, risk_scores=None,
                         ipcw_tau: Optional[float] = None,
                         ipcw_y_train: Optional[SurvivalArray] = None,
                         model_name: Optional[str] = None,
                         pending_warnings: Optional[list] = None,
                         surv_probs_transform=None,
                         gnd_n_bins: int = 10):
    """Compute train/test concordance indices and calibration metrics.

    When ``X_train``/``y_train`` are provided, the four apparent
    (resubstitution) train-set metrics are added: ``c_index_train`` (Harrell),
    ``c_index_ipcw_train`` (Uno), ``c_index_td_train`` (Antolini), and
    ``integrated_brier_train`` (IBS). These are point values on the training
    set itself -- an optimism diagnostic -- and are deliberately left out of
    the metric registry, so they never appear in printed output or the HTML
    report (they do flow into ``export_results`` and ``plot_metric(source=
    "train")``).

    Returns
    -------
    tuple of (DataFrame, ndarray or None, ndarray or None, ndarray or None, ndarray or None)
        (concordance_df, surv_probs, clean_eval_times, ibs_surv_probs,
        ibs_times). The last two are the dense train-anchored IBS probability
        matrix and grid (for subpopulation IBS reuse). Calibration NaN-
        return warnings (cal_slope, HL, GND) are appended to
        ``pending_warnings`` (if provided) for parent-side emission.
    """
    risk_test = risk_scores if risk_scores is not None else model.predict(X_test)

    # Compute test C-index
    try:
        c_test = concordance_index_censored(
            y_test['event'], y_test['time'], risk_test
        )[0]
    except (MemoryError, RecursionError):
        raise
    except Exception as e:
        log_metric_failure(logger, 'c_index_test', e)
        c_test = np.nan

    data = {'c_index_test': [c_test]}

    # Compute Uno's C-index (IPCW) if training data available.
    # `ipcw_y_train` overrides the G_hat fitting cohort when set
    # (e.g. validate() routes the validation cohort here); otherwise we fall
    # back to the training cohort.
    y_for_g = ipcw_y_train if ipcw_y_train is not None else y_train
    if y_for_g is not None:
        try:
            y_test_trunc = truncate_y_to_train_range(y_test, y_for_g)
            c_ipcw = concordance_index_ipcw(y_for_g, y_test_trunc, risk_test, tau=ipcw_tau)[0]
            data['c_index_ipcw_test'] = [c_ipcw]
        except (MemoryError, RecursionError):
            raise
        except Exception as e:
            log_metric_failure(logger, 'c_index_ipcw_test', e)
            data['c_index_ipcw_test'] = [np.nan]

    # Compute the apparent (resubstitution) train-set metrics if training data
    # is provided. Guarded so bootstrap resamples (which omit X_train) never
    # recompute them -- the train fit is the same across all resamples.
    if X_train is not None and y_train is not None:
        _add_train_metrics(data, model, X_train, y_train, eval_times, ipcw_tau)

    # Compute Brier score and IBS if model supports survival functions.
    # ibs_surv_probs/ibs_times are the dense train-anchored IBS matrix + grid.
    surv_probs, clean_eval_times, ibs_surv_probs, ibs_times = _add_brier_metrics(
        data, model, X_test, y_test, y_train, eval_times,
        surv_probs_transform=surv_probs_transform,
    )

    # Compute Antolini's C^td (reuses surv_probs from Brier computation)
    _add_ctd_metric(data, y_test, surv_probs, clean_eval_times)

    # Compute calibration metrics (reuses surv_probs from Brier computation)
    _add_calibration_metrics(
        data, y_test, surv_probs, clean_eval_times, model_name=model_name,
        warnings_out=pending_warnings, gnd_n_bins=gnd_n_bins,
    )

    # Add custom metrics
    if custom_metrics is not None:
        for name, scorer in custom_metrics.items():
            try:
                data[name] = [scorer(model, X_test, y_test)]
            except (MemoryError, RecursionError):
                raise
            except Exception as e:
                log_metric_failure(logger, name, e, warn_user=True)
                data[name] = [np.nan]

    return pd.DataFrame(data), surv_probs, clean_eval_times, ibs_surv_probs, ibs_times


def _build_test_set_df(X_test: pd.DataFrame, y_test: SurvivalArray, risk_scores):
    """Build test set DataFrame with predictions."""
    return pd.DataFrame({
        'index': X_test.index,
        'risk_score': risk_scores,
        'time': y_test['time'],
        'event': y_test['event'],
    })


def _compute_auc_t(y_test: SurvivalArray, y_train: Optional[SurvivalArray],
                   eval_times, risk_scores, model_name: str = "",
                   auc_nan_collector: Optional[list] = None,
                   _warn_stacklevel: int = 3):
    """Compute time-dependent AUC at specified timepoints.

    Every path that returns NaN at a requested ``eval_time`` surfaces a
    user-visible :class:`MethodologicalWarning` (per the "No Silent Data Loss"
    rule), with one documented exception: when ``y_train`` is None, AUC(t) is
    undefined by contract (no IPCW possible) and the all-NaN return stays
    unwarned (see below). When ``auc_nan_collector`` is supplied, the NaN notice is recorded
    into it as an ``("auc_t", model_name, msg)`` tuple for parent-side
    coalescing instead of warning inline (mirrors the calibration-NaN collector
    in ``evaluate.py``); direct/single-model callers leave it ``None`` and get
    the inline warning.
    """
    # When upstream auto_eval_times falls back to None on degenerate
    # data, return an empty AUC(t) frame rather than crashing on iteration.
    if eval_times is None or len(eval_times) == 0:
        return pd.DataFrame({
            'eval_time': pd.Series([], dtype=float),
            'auc': pd.Series([], dtype=float),
            'n_events': pd.Series([], dtype=int),
            'n_at_risk': pd.Series([], dtype=int),
        })
    # Compute events and at-risk counts for each eval time
    times = y_test['time']
    events = y_test['event']
    n_events = [int(((times <= t) & events).sum()) for t in eval_times]
    n_at_risk = [int((times >= t).sum()) for t in eval_times]
    eval_times_arr = np.asarray(eval_times, dtype=float)

    def _build_result(auc_values):
        return pd.DataFrame({
            'eval_time': eval_times,
            'auc': auc_values,
            'n_events': n_events,
            'n_at_risk': n_at_risk,
        })

    def _record(reason: str, nan_times) -> None:
        """Surface a NaN-fallback notice. With a collector, record into it for
        coalesced parent-side emission; otherwise emit a standalone
        MethodologicalWarning so "No Silent Data Loss" holds for direct callers.
        """
        nan_list = [float(t) for t in np.asarray(nan_times, dtype=float)]
        who = f" for {model_name}" if model_name else ""
        msg = (
            f"AUC(t){who}: {reason}. Reporting NaN at {len(nan_list)} "
            f"requested time point(s): {nan_list}."
        )
        if auc_nan_collector is not None:
            auc_nan_collector.append(("auc_t", model_name, msg))
        else:
            warnings.warn(msg, MethodologicalWarning, stacklevel=_warn_stacklevel)

    # y_train is required for IPCW; without it AUC(t) is undefined by contract,
    # so the all-NaN return is expected (not a silent-loss surprise) and stays
    # unwarned.
    if y_train is None:
        return _build_result([np.nan] * len(eval_times))

    try:
        # Shared clip -> compute -> divide-filter -> NaN-detect core
        # (mirrored by _safe_auc_t in _subpopulation.py).
        auc_all, nan_mask, valid_times = _auc_values_and_nan_mask(
            y_test, y_train, eval_times_arr, risk_scores,
        )
        if len(valid_times) < 2:
            _record(
                f"only {len(valid_times)}/{len(eval_times_arr)} requested "
                f"time(s) fall in the test follow-up range",
                eval_times_arr,
            )
        elif nan_mask.any():
            # nan_mask covers both NaN sources: out-of-range times and
            # divide-instability NaN within the valid times.
            _record(
                "undefined at some requested times; IPCW weights are "
                "unreliable near the end of follow-up, and times outside the "
                "test range cannot be evaluated",
                eval_times_arr[nan_mask],
            )
        return _build_result(auc_all)
    except (MemoryError, RecursionError):
        raise
    except (ValueError, ZeroDivisionError, ArithmeticError) as e:
        # Same sksurv failure modes the shared core can raise; kept in sync
        # with _safe_auc_t so both AUC(t) paths fail identically.
        log_metric_failure(logger, "auc_t", e)
        _record(f"failed with {type(e).__name__} ({e})", eval_times_arr)
    return _build_result([np.nan] * len(eval_times))
