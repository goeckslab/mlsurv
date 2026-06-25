"""Single bootstrap iteration logic and metric computation."""

import warnings
from typing import Optional

import numpy as np
from sklearn.inspection import permutation_importance

from .importance import _grouped_permutation_importance
from sksurv.metrics import cumulative_dynamic_auc

from .._warnings import MethodologicalWarning
from ..logging_utils import get_logger, log_metric_failure
from ..utils import SurvivalArray, clip_eval_times_to_data, truncate_y_to_train_range
from .metrics import (
    c_index, _brier_score_at_times, _surv_funcs_to_probs,
    predict_survival_functions, calibration_slope_intercept, calibration_test,
)

logger = get_logger("bootstrap")


def _record_failure(failures, metric, e, iteration, model_name, *, warn_user=False):
    """Record a metric failure, either accumulating for summary or logging immediately.

    When *failures* is a list, appends a ``(metric, exc_type, exc_msg)`` tuple
    and emits a DEBUG log.  When *failures* is ``None``, delegates to
    ``log_metric_failure()`` for immediate per-iteration logging.
    """
    if failures is not None:
        failures.append((metric, type(e).__name__, str(e)))
        logger.debug("%s: %s failed in iteration %s: %s", model_name, metric, iteration, e, exc_info=True)
    else:
        log_metric_failure(logger, metric, e, iteration=iteration, model_name=model_name, warn_user=warn_user)


def _record_low_coverage(notices, metric, iteration, model_name,
                         n_valid, n_total, *, subpopulation=None):
    """Record a per-iteration low-coverage notice for AUC(t) and friends.

    Mirrors ``_record_failure`` but for the non-exception silent-NaN path
    (``len(valid_times) < 2`` or partial out-of-range coverage). Tuples
    shape ``(metric, subpopulation, n_valid, n_total)`` so
    :func:`_emit_low_coverage_summary` can group by (metric, subpopulation).

    When ``notices`` is ``None`` (a direct caller that didn't allocate a
    collector), emit a standalone :class:`MethodologicalWarning` so the
    "No Silent Data Loss" rule is honored — matches the subpop-side
    pattern in :func:`mlsurv.evaluation._subpopulation._safe_auc_t`.
    """
    if notices is None:
        sub_label = f" subpopulation '{subpopulation}'" if subpopulation else ""
        warnings.warn(
            f"{model_name}{sub_label}: {metric} eval_times coverage "
            f"{n_valid}/{n_total} (iteration {iteration}); reporting NaN.",
            MethodologicalWarning,
            stacklevel=3,
        )
        return
    notices.append((metric, subpopulation, n_valid, n_total))
    logger.debug(
        "%s: %s low coverage in iteration %s (subpop=%s, %d/%d)",
        model_name, metric, iteration, subpopulation, n_valid, n_total,
    )


def _safe_compute_metric(y_boot: SurvivalArray, risk_scores, metric, y_train: Optional[SurvivalArray],
                         model, X_boot, eval_times, surv_probs, i, model_name,
                         failures=None, ipcw_tau: Optional[float] = None,
                         ipcw_y_train: Optional[SurvivalArray] = None, cal_cache=None,
                         ibs_surv_probs=None, ibs_times=None):
    """Compute a single metric with error handling. Returns the metric value, or NaN on failure.

    Parameters
    ----------
    failures : list or None
        When provided, append ``(metric, type(e).__name__, str(e))`` instead
        of calling ``log_metric_failure()``.  The caller aggregates failures
        into a summary after all iterations complete.
    cal_cache : dict or None
        Per-iteration single-entry cache shared between cal_slope and
        cal_intercept so they fit the logistic recalibration only once.
    """
    try:
        value = _compute_metric(
            y_boot, risk_scores, metric,
            y_train=y_train, model=model, X=X_boot,
            eval_times=eval_times, surv_probs=surv_probs,
            ipcw_tau=ipcw_tau, ipcw_y_train=ipcw_y_train, cal_cache=cal_cache,
            ibs_surv_probs=ibs_surv_probs, ibs_times=ibs_times,
        )
    except (MemoryError, RecursionError):
        raise
    except (ValueError, ZeroDivisionError) as e:
        _record_failure(failures, metric, e, i, model_name)
        value = np.nan
    except Exception as e:
        _record_failure(failures, metric, e, i, model_name, warn_user=True)
        value = np.nan
    return value


def _compute_all_metrics(i, model_name, metrics, custom_metrics, y_boot: SurvivalArray,
                         risk_scores, X_boot, get_X_boot, y_train: Optional[SurvivalArray],
                         model, eval_times, surv_probs_boot, failures=None,
                         ipcw_tau: Optional[float] = None,
                         ipcw_y_train: Optional[SurvivalArray] = None, cal_cache=None,
                         ibs_surv_probs_boot=None, ibs_times=None):
    """Compute all built-in and custom metrics for one bootstrap iteration."""
    data = []
    for metric in metrics:
        value = _safe_compute_metric(
            y_boot, risk_scores, metric, y_train, model, X_boot,
            eval_times, surv_probs_boot, i, model_name,
            failures=failures, ipcw_tau=ipcw_tau, ipcw_y_train=ipcw_y_train,
            cal_cache=cal_cache,
            ibs_surv_probs=ibs_surv_probs_boot, ibs_times=ibs_times,
        )
        data.append({'model_type': model_name, 'iteration': i,
                     'metric_name': metric, 'value': value})

    for metric_name, scorer in custom_metrics.items():
        try:
            value = scorer(model, get_X_boot(), y_boot)
        except (MemoryError, RecursionError):
            raise
        except Exception as e:
            _record_failure(failures, metric_name, e, i, model_name, warn_user=True)
            value = np.nan
        data.append({'model_type': model_name, 'iteration': i,
                     'metric_name': metric_name, 'value': value})
    return data


def _collect_predictions(include, model_name, i, get_X_boot, risk_scores):
    """Collect prediction data if requested."""
    if not include:
        return None
    X_boot = get_X_boot()
    return [{'model_type': model_name, 'iteration': i,
             'index': orig_idx, 'risk_score': score}
            for orig_idx, score in zip(X_boot.index, risk_scores)]


def _collect_auc(include, y_train: Optional[SurvivalArray], y_boot: SurvivalArray,
                 risk_scores, eval_times, model_name, i, failures=None,
                 notices=None):
    """Collect AUC(t) data if requested."""
    if not include:
        return None
    results = _compute_auc_for_iteration(
        y_train, y_boot, risk_scores, eval_times, model_name, i,
        failures=failures, notices=notices,
    )
    return results if results else []


def _collect_brier_t(include, y_train: Optional[SurvivalArray], y_boot: SurvivalArray,
                     surv_probs, eval_times, model_name, i, failures=None,
                     notices=None):
    """Collect per-time Brier(t) data if requested.

    The Brier counterpart of :func:`_collect_auc`. Unlike AUC (rank-based,
    risk_scores), Brier needs ``S(t|x)``: pass the precomputed ``surv_probs``
    for this resample, which already carries any recalibration transform
    applied upstream. ``surv_probs`` is precomputed whenever ``include_brier_t``
    is set (see :func:`_precompute_predictions`).
    """
    if not include:
        return None
    results = _compute_brier_t_for_iteration(
        y_train, y_boot, surv_probs, eval_times, model_name, i,
        failures=failures, notices=notices,
    )
    return results if results else []


def _collect_importance(include, model, get_X_boot, y_boot: SurvivalArray, feature_names,
                        n_importance_repeats, rng, model_name, i, failures=None,
                        categorical_groups=None):
    """Collect permutation importance data if requested."""
    if not include:
        return None
    return _compute_importance_for_iteration(
        model, get_X_boot(), y_boot, feature_names, n_importance_repeats,
        rng, model_name, i, failures=failures,
        categorical_groups=categorical_groups)


def _compute_auc_for_iteration(y_train: Optional[SurvivalArray], y_boot: SurvivalArray,
                               risk_scores, eval_times, model_name, iteration,
                               failures=None, notices=None):
    """Compute AUC(t) values for one bootstrap iteration."""
    if y_train is None:
        return None
    # Upstream may have set eval_times=None on degenerate data — no
    # AUC(t) rows to emit, just return empty.
    if eval_times is None or len(eval_times) == 0:
        return []

    # Coerce up front so eval_times[~valid_mask] works when callers
    # pass a Python list. clip_eval_times_to_data's internal coercion does
    # not propagate back to this local.
    eval_times_arr = np.asarray(eval_times, dtype=float)

    try:
        valid_times, valid_mask, y_boot_ipcw = clip_eval_times_to_data(
            eval_times_arr, y_boot, y_train,
        )

        rows = []
        if len(valid_times) >= 2:
            # scikit-survival may emit RuntimeWarning for degenerate bootstrap
            # resamples; we convert those cases to NaN outputs below.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="invalid value encountered in divide",
                )
                auc_values, _ = cumulative_dynamic_auc(
                    y_train, y_boot_ipcw, risk_scores, valid_times,
                )
            for t, a in zip(valid_times, auc_values):
                rows.append({
                    'model_type': model_name, 'iteration': iteration,
                    'eval_time': float(t), 'auc': float(a),
                })

        # Surface both insufficient-coverage and
        # partial-out-of-range silent-NaN cases as per-iteration notices so
        # the bootstrap aggregator can coalesce them (one warning per metric,
        # collapsing a metric's subpopulation levels).
        n_out = int((~valid_mask).sum())
        if len(valid_times) < 2 or n_out > 0:
            _record_low_coverage(
                notices, "auc_t", iteration, model_name,
                n_valid=len(valid_times), n_total=len(eval_times_arr),
            )

        # Keep output shape invariant — len(rows) == len(eval_times).
        # When < 2 valid, no AUC rows were appended, so NaN-fill every
        # eval_time; otherwise NaN-fill only the out-of-range times. Filling
        # only ~valid_mask in the < 2 case silently drops valid-but-too-few
        # times, making the row count vary by branch.
        nan_times = (eval_times_arr if len(valid_times) < 2
                     else eval_times_arr[~valid_mask])
        for t in nan_times:
            rows.append({
                'model_type': model_name, 'iteration': iteration,
                'eval_time': float(t), 'auc': np.nan,
            })
        return rows
    except (MemoryError, RecursionError):
        raise
    except (ValueError, IndexError, TypeError, ZeroDivisionError,
            ArithmeticError) as e:
        # Narrowed from bare `except Exception` to match the
        # documented sksurv failure modes plus the indexing TypeErrors
        # the AUC(t) sites historically swallowed (CONTRIBUTING.md exception rule).
        _record_failure(failures, "auc_t", e, iteration, model_name,
                        warn_user=not isinstance(e, (ValueError, ZeroDivisionError)))
        return [{'model_type': model_name, 'iteration': iteration,
                 'eval_time': float(t), 'auc': np.nan} for t in eval_times_arr]


def _compute_brier_t_for_iteration(y_train: Optional[SurvivalArray], y_boot: SurvivalArray,
                                   surv_probs, eval_times, model_name, iteration,
                                   failures=None, notices=None):
    """Compute Brier(t) values for one bootstrap iteration.

    Shares the structure of :func:`_compute_auc_for_iteration` (shape-invariant
    NaN-fill and low-coverage notices keyed ``"brier_t"``), but scores
    ``S(t|x)`` with IPCW Brier instead of AUC and needs only one valid time
    (AUC needs two). ``surv_probs`` is the resampled survival-probability array
    aligned to ``eval_times``; it already carries any recalibration transform
    applied upstream at precompute time. When it is ``None`` (the model cannot
    produce survival curves) Brier(t) is uncomputable, so every eval_time is
    NaN-filled and a low-coverage notice fires — we never predict raw curves
    here, which would mislabel a recalibrated band as raw.
    """
    from sksurv.metrics import brier_score

    if y_train is None:
        return None
    # Upstream may have set eval_times=None on degenerate data — no
    # Brier(t) rows to emit, just return empty.
    if eval_times is None or len(eval_times) == 0:
        return []

    eval_times_arr = np.asarray(eval_times, dtype=float)

    try:
        valid_times, valid_mask, y_boot_ipcw = clip_eval_times_to_data(
            eval_times_arr, y_boot, y_train,
        )

        probs = surv_probs
        rows = []
        if probs is not None and len(valid_times) >= 1:
            valid_probs = probs[:, valid_mask]
            _, bs_scores = brier_score(
                y_train, y_boot_ipcw, valid_probs, valid_times,
            )
            for t, b in zip(valid_times, bs_scores):
                rows.append({
                    'model_type': model_name, 'iteration': iteration,
                    'eval_time': float(t), 'brier': float(b),
                })

        # Surface insufficient-coverage / partial-out-of-range cases as
        # per-iteration notices so the aggregator coalesces them (one warning
        # per metric). No usable probs counts as zero coverage.
        n_out = int((~valid_mask).sum())
        n_scored = len(valid_times) if probs is not None else 0
        if n_scored < 1 or n_out > 0:
            _record_low_coverage(
                notices, "brier_t", iteration, model_name,
                n_valid=n_scored, n_total=len(eval_times_arr),
            )

        # Keep output shape invariant — len(rows) == len(eval_times). NaN-fill
        # every eval_time when nothing was scored, else only out-of-range ones.
        nan_times = (eval_times_arr if n_scored < 1
                     else eval_times_arr[~valid_mask])
        for t in nan_times:
            rows.append({
                'model_type': model_name, 'iteration': iteration,
                'eval_time': float(t), 'brier': np.nan,
            })
        return rows
    except (MemoryError, RecursionError):
        raise
    except (ValueError, IndexError, TypeError, ZeroDivisionError,
            ArithmeticError) as e:
        # Narrowed to the documented sksurv brier_score failure modes plus the
        # indexing errors the surv_probs slicing can raise (CONTRIBUTING.md rule).
        _record_failure(failures, "brier_t", e, iteration, model_name,
                        warn_user=not isinstance(e, (ValueError, ZeroDivisionError)))
        return [{'model_type': model_name, 'iteration': iteration,
                 'eval_time': float(t), 'brier': np.nan} for t in eval_times_arr]


def _compute_subpopulation_auc_for_iter(sub_y, sub_risk, y_train, eval_times,
                                   group_name, model_name, iteration,
                                   failures=None, notices=None):
    """Compute AUC(t) for one subpopulation in one bootstrap iteration.

    Mirrors the ``_safe_auc_t`` pattern from ``_subpopulation.py``: truncates
    to the training time range, filters eval_times to the subpopulation's
    valid range, and emits NaN for out-of-range times.
    """
    # Upstream may have set eval_times=None on degenerate data.
    if eval_times is None or len(eval_times) == 0:
        return []
    # Coerce up front so eval_times[~valid_mask] works when callers
    # pass a Python list.
    eval_times_arr = np.asarray(eval_times, dtype=float)
    try:
        valid_times, valid_mask, sub_y_trunc = clip_eval_times_to_data(
            eval_times_arr, sub_y, y_train,
        )
        # Surface insufficient-coverage AND partial-coverage
        # silent-NaN cases per iter.
        n_out = int((~valid_mask).sum())
        if len(valid_times) < 2 or n_out > 0:
            _record_low_coverage(
                notices, "subpopulation_auc_t", iteration, model_name,
                n_valid=len(valid_times), n_total=len(eval_times_arr),
                subpopulation=str(group_name),
            )

        rows = []
        if len(valid_times) >= 2:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="invalid value encountered in divide",
                )
                auc_values, _ = cumulative_dynamic_auc(
                    y_train, sub_y_trunc, sub_risk, valid_times,
                )
            for t, a in zip(valid_times, auc_values):
                rows.append({
                    'subpopulation': str(group_name), 'model_type': model_name,
                    'iteration': iteration, 'eval_time': float(t), 'auc': float(a),
                })

        # Keep output shape invariant — len(rows) == len(eval_times).
        # When < 2 valid, no AUC rows were appended, so NaN-fill every
        # eval_time; otherwise NaN-fill only the out-of-range times.
        nan_times = (eval_times_arr if len(valid_times) < 2
                     else eval_times_arr[~valid_mask])
        for t in nan_times:
            rows.append({
                'subpopulation': str(group_name), 'model_type': model_name,
                'iteration': iteration, 'eval_time': float(t), 'auc': np.nan,
            })
        return rows
    except (MemoryError, RecursionError):
        raise
    except (ValueError, IndexError, TypeError, ZeroDivisionError,
            ArithmeticError) as e:
        # Narrowed from bare `except Exception` (CONTRIBUTING.md exception rule).
        _record_failure(
            failures, f"subpopulation_auc_t_{group_name}", e, iteration,
            model_name,
            warn_user=not isinstance(e, (ValueError, ZeroDivisionError)),
        )
        return [{
            'subpopulation': str(group_name), 'model_type': model_name,
            'iteration': iteration, 'eval_time': float(t), 'auc': np.nan,
        } for t in eval_times_arr]


def _nan_importance_rows(model_name, iteration, feature_names):
    """Return NaN importance rows for all features (fallback on error)."""
    return [{'model_type': model_name, 'iteration': iteration, 'feature': feat,
             'importance': np.nan}
            for feat in feature_names]


def _compute_importance_for_iteration(model, X_boot, y_boot: SurvivalArray, feature_names,
                                       n_importance_repeats, rng, model_name, iteration,
                                       failures=None, categorical_groups=None):
    """Compute permutation importance for one bootstrap iteration."""
    try:
        seed = rng.randint(0, 2**31)
        if categorical_groups:
            result = _grouped_permutation_importance(
                model, X_boot, y_boot, categorical_groups,
                n_repeats=n_importance_repeats, random_state=seed,
            )
            result_names = result.feature_names
        else:
            # Pass full X_boot — if the model is a Pipeline with a selector,
            # permuting non-selected features correctly yields 0 importance.
            result = permutation_importance(
                model, X_boot, y_boot,
                n_repeats=n_importance_repeats, random_state=seed,
            )
            result_names = feature_names
        return [{'model_type': model_name, 'iteration': iteration, 'feature': feat,
                 'importance': result.importances_mean[feat_idx]}
                for feat_idx, feat in enumerate(result_names)]
    except (MemoryError, RecursionError):
        raise
    except (ValueError, AttributeError, KeyError) as e:
        _record_failure(failures, "permutation_importance", e, iteration, model_name)
    except Exception as e:
        _record_failure(failures, "permutation_importance", e, iteration, model_name, warn_user=True)
    return _nan_importance_rows(model_name, iteration, feature_names)


def _get_bootstrap_indices(y: SurvivalArray, rng, stratify=False,
                           subpopulation_labels=None):
    """Generate bootstrap indices, optionally stratified by event status.

    When *subpopulation_labels* is provided and *stratify* is True, stratification
    uses (event x subpopulation) combinations to preserve subpopulation proportions.
    """
    n = len(y)

    if not stratify:
        return rng.choice(n, size=n, replace=True)

    if subpopulation_labels is not None:
        # Stratify by (event × subpopulation) combinations
        events = np.asarray(y['event'])
        combined = [f"{int(e)}_{s}" for e, s in zip(events, subpopulation_labels)]
        combined = np.asarray(combined)
        boot_indices = []
        for stratum in np.unique(combined):
            stratum_idx = np.where(combined == stratum)[0]
            if len(stratum_idx) == 0:
                continue
            boot_indices.append(
                rng.choice(stratum_idx, size=len(stratum_idx), replace=True)
            )
        return np.concatenate(boot_indices)

    # Stratified by event only: preserve event ratio
    # Use explicit comparison for dtype safety (avoids issues with float event dtype)
    event_idx = np.where(y['event'] == 1)[0]
    censored_idx = np.where(y['event'] == 0)[0]

    boot_event = rng.choice(event_idx, size=len(event_idx), replace=True)
    boot_censored = rng.choice(censored_idx, size=len(censored_idx), replace=True)

    return np.concatenate([boot_event, boot_censored])


def _nan_subpopulation_rows(model_name, i, metrics, group_name, eval_times, include_auc):
    """Return NaN iteration + AUC rows for an invalid subpopulation."""
    iteration_rows = [{
        'model_type': model_name, 'iteration': i,
        'metric_name': f'{metric}__subpopulation_{group_name}',
        'value': np.nan,
    } for metric in metrics]
    auc_rows = []
    if include_auc and eval_times is not None:
        auc_rows = [{
            'subpopulation': str(group_name), 'model_type': model_name,
            'iteration': i, 'eval_time': float(t), 'auc': np.nan,
        } for t in eval_times]
    return iteration_rows, auc_rows


def _compute_subpopulation_iteration_metrics(
    config, idx, model_name, i, y_boot, risk_scores, X_boot,
    surv_probs_boot, y_train, model, failures, notices=None, cal_cache=None,
    ibs_surv_probs_boot=None, ibs_times=None,
):
    """Compute per-subpopulation metrics for one bootstrap iteration.

    Returns
    -------
    tuple of (subpopulation_iteration_data: list, subpopulation_auc_data: list)
    """
    subpopulation_iteration_data = []
    subpopulation_auc_data = []
    if config.subpopulation_labels is None or config.subpopulation_unique is None:
        return subpopulation_iteration_data, subpopulation_auc_data

    include_subpopulation_auc = y_train is not None and config.eval_times is not None
    boot_subpopulations = config.subpopulation_labels[idx]

    for group_name in config.subpopulation_unique:
        mask = boot_subpopulations == group_name
        if mask.sum() < 2:
            rows, auc_rows = _nan_subpopulation_rows(
                model_name, i, config.metrics, group_name,
                config.eval_times, include_subpopulation_auc,
            )
            subpopulation_iteration_data.extend(rows)
            subpopulation_auc_data.extend(auc_rows)
            continue

        sub_y = y_boot[mask]
        sub_events = np.asarray(sub_y['event'])
        if sub_events.sum() < 1 or (1 - sub_events).sum() < 1:
            rows, auc_rows = _nan_subpopulation_rows(
                model_name, i, config.metrics, group_name,
                config.eval_times, include_subpopulation_auc,
            )
            subpopulation_iteration_data.extend(rows)
            subpopulation_auc_data.extend(auc_rows)
            continue

        sub_risk = risk_scores[mask]
        sub_surv = surv_probs_boot[mask, :] if surv_probs_boot is not None else None
        sub_ibs_surv = (ibs_surv_probs_boot[mask, :]
                        if ibs_surv_probs_boot is not None else None)
        # Use subpopulation-masked X to avoid size mismatch if metrics
        # fall back to model.predict_survival_function(X)
        sub_X = X_boot.iloc[mask] if X_boot is not None and sub_surv is None else X_boot

        for metric in config.metrics:
            metric_key = f'{metric}__subpopulation_{group_name}'
            value = _safe_compute_metric(
                sub_y, sub_risk, metric, y_train, model, sub_X,
                config.eval_times, sub_surv, i, model_name,
                failures=failures, ipcw_tau=getattr(config, 'ipcw_tau', None),
                ipcw_y_train=getattr(config, 'ipcw_y_train', None),
                cal_cache=cal_cache,
                ibs_surv_probs=sub_ibs_surv, ibs_times=ibs_times,
            )
            subpopulation_iteration_data.append({
                'model_type': model_name, 'iteration': i,
                'metric_name': metric_key, 'value': value,
            })

        if include_subpopulation_auc:
            auc_rows = _compute_subpopulation_auc_for_iter(
                sub_y, sub_risk, y_train, config.eval_times,
                group_name, model_name, i,
                failures=failures, notices=notices,
            )
            subpopulation_auc_data.extend(auc_rows)

    return subpopulation_iteration_data, subpopulation_auc_data


def _bootstrap_single_iteration(
    i,
    seed,
    model,
    X_test,
    y_test: SurvivalArray,
    y_train: Optional[SurvivalArray],
    config,
    model_name,
    feature_names,
    precomputed_risk=None,
    precomputed_surv_probs=None,
    precomputed_ibs_surv_probs=None,
    ibs_times=None,
):
    """
    Execute a single bootstrap iteration - worker function for parallelization.

    Parameters
    ----------
    i : int or str
        Iteration index (int) or label (e.g. ``'point'``).
    seed : int or None
        Random seed for this iteration.
    model : estimator
        Fitted model to evaluate.
    X_test : DataFrame
        Test features.
    y_test : SurvivalArray
        Test outcomes.
    y_train : SurvivalArray or None
        Training outcomes (for IPCW metrics).
    config : BootstrapConfig
        Bundled iteration configuration (metrics, eval_times, include_* flags, etc.).
    model_name : str
        Model name for results.
    feature_names : list
        Feature names for importance results.
    precomputed_risk : ndarray or None
        Precomputed risk scores for all samples.
    precomputed_surv_probs : ndarray or None
        Precomputed survival probabilities.

    Returns
    -------
    dict with keys: iteration_data, predictions_data, auc_data, brier_t_data, importance_data
    """
    # Suppress repeated iterative warnings per worker process — loky workers
    # don't inherit the parent's catch_warnings filter, so we apply it here
    # to ensure at most one per unique message even under process-based
    # parallelism.
    from .._warnings import IterativeWarning
    warnings.filterwarnings("once", category=IterativeWarning)

    rng = np.random.RandomState(seed)

    # Get bootstrap sample indices
    idx = _get_bootstrap_indices(y_test, rng, config.stratify,
                                 subpopulation_labels=config.subpopulation_labels)
    y_boot = y_test[idx]

    # Use precomputed if available, otherwise compute
    if precomputed_risk is not None:
        risk_scores = precomputed_risk[idx]
        X_boot = None  # Not needed when precomputing
    else:
        X_boot = X_test.iloc[idx]
        risk_scores = model.predict(X_boot)

    surv_probs_boot = precomputed_surv_probs[idx, :] if precomputed_surv_probs is not None else None
    ibs_surv_probs_boot = (precomputed_ibs_surv_probs[idx, :]
                           if precomputed_ibs_surv_probs is not None else None)

    # Lazily resolve X_boot for optional outputs that need it
    def get_X_boot():
        nonlocal X_boot
        if X_boot is None:
            X_boot = X_test.iloc[idx]
        return X_boot

    failures = []
    # Notices carry insufficient-coverage / silent-NaN
    # signals from AUC(t) sites; aggregated at the bootstrap top level.
    notices = []

    # Per-iteration single-entry cache: lets cal_slope and cal_intercept
    # share one logistic-recalibration fit on the same y array.  Created fresh
    # here and never shared across iterations or threads, so it is correct under
    # the threading backend (RSF/XGB) as well as loky.
    cal_cache = {}

    iteration_data = _compute_all_metrics(
        i, model_name, config.metrics, config.custom_metrics, y_boot, risk_scores,
        X_boot, get_X_boot, y_train, model, config.eval_times, surv_probs_boot,
        failures=failures, ipcw_tau=getattr(config, 'ipcw_tau', None),
        ipcw_y_train=getattr(config, 'ipcw_y_train', None), cal_cache=cal_cache,
        ibs_surv_probs_boot=ibs_surv_probs_boot, ibs_times=ibs_times,
    )

    # Per-subpopulation metrics (near-zero cost: reuse resampled risk scores)
    subpopulation_iteration_data, subpopulation_auc_data = _compute_subpopulation_iteration_metrics(
        config, idx, model_name, i, y_boot, risk_scores, X_boot,
        surv_probs_boot, y_train, model, failures, notices=notices,
        cal_cache=cal_cache,
        ibs_surv_probs_boot=ibs_surv_probs_boot, ibs_times=ibs_times,
    )
    iteration_data.extend(subpopulation_iteration_data)

    return {
        'iteration_data': iteration_data,
        'predictions_data': _collect_predictions(
            config.include_predictions, model_name, i, get_X_boot, risk_scores),
        'auc_data': _collect_auc(
            config.include_auc, y_train, y_boot, risk_scores, config.eval_times, model_name, i,
            failures=failures, notices=notices),
        'brier_t_data': _collect_brier_t(
            config.include_brier_t, y_train, y_boot, surv_probs_boot,
            config.eval_times, model_name, i,
            failures=failures, notices=notices),
        'importance_data': _collect_importance(
            config.include_importance, model, get_X_boot, y_boot, feature_names,
            config.n_importance_repeats, rng, model_name, i, failures=failures,
            categorical_groups=getattr(config, 'categorical_groups', None)),
        'subpopulation_auc_data': subpopulation_auc_data or None,
        'failures': failures,
        'notices': notices,
    }


# ---------------------------------------------------------------------------
# Metric dispatch (used by _safe_compute_metric)
# ---------------------------------------------------------------------------

def _compute_metric(y: SurvivalArray, risk_scores, metric, y_train: Optional[SurvivalArray] = None, model=None,
                    X=None, eval_times=None, surv_probs=None, ipcw_tau: Optional[float] = None,
                    ipcw_y_train: Optional[SurvivalArray] = None, cal_cache=None,
                    ibs_surv_probs=None, ibs_times=None):
    """Compute a specific metric using metrics.py functions."""
    fn = _METRIC_DISPATCH.get(metric)
    if fn is None:
        raise ValueError(
            f"Unknown metric: {metric}. Options: {list(_METRIC_DISPATCH.keys())}"
        )
    return fn(
        y, risk_scores, y_train=y_train, model=model,
        X=X, eval_times=eval_times, surv_probs=surv_probs,
        ipcw_tau=ipcw_tau, ipcw_y_train=ipcw_y_train, cal_cache=cal_cache,
        ibs_surv_probs=ibs_surv_probs, ibs_times=ibs_times,
    )


def _compute_harrell(y: SurvivalArray, risk_scores, **kwargs):
    """Compute Harrell's C-index."""
    return c_index(y, risk_scores, method='harrell')


def _compute_ipcw(y: SurvivalArray, risk_scores, y_train: Optional[SurvivalArray] = None,
                  ipcw_tau: Optional[float] = None,
                  ipcw_y_train: Optional[SurvivalArray] = None, **kwargs):
    """Compute IPCW C-index.

    ``ipcw_y_train`` overrides the G_hat fitting cohort when set
    (validate() routes the validation cohort here). Falls back to
    ``y_train`` for the in-distribution case.
    """
    from sksurv.metrics import concordance_index_ipcw

    y_for_g = ipcw_y_train if ipcw_y_train is not None else y_train
    if y_for_g is None:
        raise ValueError("y_train required for IPCW metric")
    y_ipcw = truncate_y_to_train_range(y, y_for_g, warn=False)
    return concordance_index_ipcw(y_for_g, y_ipcw, risk_scores, tau=ipcw_tau)[0]


def _compute_brier(y: SurvivalArray, risk_scores, y_train: Optional[SurvivalArray] = None, model=None, X=None,
                   eval_times=None, surv_probs=None, **kwargs):
    """Compute Brier score at median evaluation time."""
    from sksurv.metrics import brier_score

    if y_train is None or eval_times is None:
        raise ValueError("Brier score requires y_train and eval_times")

    valid_times, valid_mask, y_ipcw = clip_eval_times_to_data(
        eval_times, y, y_train,
    )
    if len(valid_times) == 0:
        return np.nan

    # Pick median of the *valid* times
    t_idx_valid = int(np.argmin(np.abs(valid_times - np.median(valid_times))))
    t = valid_times[t_idx_valid]

    if surv_probs is not None:
        # surv_probs columns are aligned to original eval_times; slice to valid
        valid_probs = surv_probs[:, valid_mask]
        preds = valid_probs[:, t_idx_valid]
        bs = brier_score(y_train, y_ipcw, preds, t)
        return float(bs[1][0])

    surv_funcs = _get_surv_funcs(model, X, "Brier score", eval_times=valid_times)
    # If dispatcher returned an ndarray aligned to valid_times, extract
    # the column at t_idx_valid directly.
    if isinstance(surv_funcs, np.ndarray) and surv_funcs.dtype != object:
        preds = surv_funcs[:, t_idx_valid]
        bs = brier_score(y_train, y_ipcw, preds, t)
        return float(bs[1][0])
    scores = _brier_score_at_times(y_train, y_ipcw, surv_funcs, [t])
    return scores[0]


def _compute_integrated_brier(y: SurvivalArray, risk_scores, y_train: Optional[SurvivalArray] = None, model=None, X=None,
                              eval_times=None, surv_probs=None,
                              ibs_surv_probs=None, ibs_times=None, **kwargs):
    """Compute Integrated Brier Score over the dense, train-anchored grid.

    IBS is integrated over the dense event-time grid (decoupled from the display
    ``eval_times``/``surv_probs`` used for Brier(t)/AUC(t)), so the bootstrap
    point estimate and CI match the headline IBS. When the precomputed dense
    matrix (``ibs_surv_probs``) and grid (``ibs_times``) are supplied (the
    precompute path), they are sliced/integrated directly; otherwise the dense
    grid is predicted from ``model``/``X`` (the no-precompute path, e.g. custom
    metrics / importance). The display ``surv_probs``/``eval_times`` are accepted
    for dispatch compatibility but are not used for IBS.
    """
    from .metrics import _integrated_brier_from_probs, _integrated_brier_dense

    if y_train is None:
        raise ValueError("Integrated Brier requires y_train")

    # Precompute path: dense matrix already built once on the full test set,
    # row-sliced for this resample.
    if ibs_surv_probs is not None and ibs_times is not None:
        return _integrated_brier_from_probs(ibs_surv_probs, ibs_times, y, y_train)

    # No precomputed dense matrix (precompute disabled, or it failed). Predict
    # the dense grid from the model: a broken predict_survival_function raises
    # here and is surfaced as a metric failure (matching the display Brier
    # path), while a degenerate <2-point grid NaNs out silently inside
    # ``_integrated_brier_dense``.
    return _integrated_brier_dense(model, X, y, y_train)


def _get_surv_probs_at_median(y, eval_times, surv_probs, model, X):
    """Get predicted survival probabilities at the eval time nearest the median."""
    if eval_times is None:
        raise ValueError("Calibration metrics require eval_times")

    # Clip eval_times to the data's follow-up range so the median reference
    # time is always within the observable window (no y_train needed here —
    # calibration doesn't use IPCW).
    valid_times, valid_mask, _ = clip_eval_times_to_data(eval_times, y)
    if len(valid_times) == 0:
        n_samples = len(y) if hasattr(y, '__len__') else X.shape[0]
        return np.full(n_samples, np.nan), np.nan

    # Pick the valid eval_time nearest the median of valid times
    med_idx_valid = int(np.argmin(np.abs(valid_times - np.median(valid_times))))
    median_time = valid_times[med_idx_valid]
    # Map back to original eval_times index for surv_probs column alignment
    orig_idx = np.where(valid_mask)[0][med_idx_valid]

    if surv_probs is not None:
        return surv_probs[:, orig_idx], median_time

    surv_funcs = _get_surv_funcs(model, X, "Calibration", eval_times=eval_times)
    probs = _surv_funcs_to_probs(surv_funcs, eval_times)
    return probs[:, orig_idx], median_time


def _get_cal_slope_intercept(y, eval_times, surv_probs, model, X, cache):
    """Compute slope & intercept once, returning cached result on repeat calls.

    ``cache`` is a per-iteration dict (created fresh in
    ``_bootstrap_single_iteration``) holding a single entry: the last ``y``
    array and its ``(slope, intercept)`` result.  cal_slope and cal_intercept
    are dispatched sequentially on the *same* ``y`` within one iteration, so
    the cache avoids a redundant logistic regression fit.  Object identity
    (``is``) — not ``id()`` — guards against CPython memory-address reuse and
    lets one dict correctly serve both the main ``y_boot`` and each
    subpopulation subset (a different array misses and recomputes).

    The cache is never shared across iterations or threads, so this is correct
    under any joblib backend (loky, threading, or future intra-iteration
    parallelism).  ``cache=None`` always recomputes.
    """
    if cache is not None and cache.get('ref') is y:
        return cache['result']

    pred_probs, median_time = _get_surv_probs_at_median(
        y, eval_times, surv_probs, model, X)
    result = calibration_slope_intercept(y, pred_probs, median_time)

    if cache is not None:
        cache['ref'] = y
        cache['result'] = result
    return result


def _compute_cal_slope(y: SurvivalArray, risk_scores, eval_times=None,
                       surv_probs=None, model=None, X=None, cal_cache=None,
                       **kwargs):
    """Compute calibration slope for bootstrap."""
    slope, _ = _get_cal_slope_intercept(y, eval_times, surv_probs, model, X, cal_cache)
    return slope


def _compute_cal_intercept(y: SurvivalArray, risk_scores, eval_times=None,
                           surv_probs=None, model=None, X=None, cal_cache=None,
                           **kwargs):
    """Compute calibration intercept for bootstrap."""
    _, intercept = _get_cal_slope_intercept(y, eval_times, surv_probs, model, X, cal_cache)
    return intercept


def _compute_gnd(y: SurvivalArray, risk_scores, eval_times=None,
                  surv_probs=None, model=None, X=None, **kwargs):
    """Compute GND calibration test p-value for bootstrap."""
    pred_probs, median_time = _get_surv_probs_at_median(
        y, eval_times, surv_probs, model, X)
    _, p_value = calibration_test(y, pred_probs, median_time)
    return p_value


def _compute_ctd(y: SurvivalArray, risk_scores, y_train: Optional[SurvivalArray] = None, model=None, X=None,
                  eval_times=None, surv_probs=None, **kwargs):
    """Compute Antolini's time-dependent C-index."""
    from ._ctd import concordance_td

    if eval_times is None:
        raise ValueError("C^td requires eval_times")

    if surv_probs is not None:
        return concordance_td(y['time'], y['event'], surv_probs, eval_times)

    surv_funcs = _get_surv_funcs(model, X, "C^td", eval_times=eval_times)
    probs = _surv_funcs_to_probs(surv_funcs, eval_times)
    return concordance_td(y['time'], y['event'], probs, eval_times)


def _get_surv_funcs(model, X, metric_name, eval_times=None):
    """Get survival functions from model, with validation.

    Parameters
    ----------
    model : estimator
        Fitted model.
    X : DataFrame or None
        Feature matrix.
    metric_name : str
        Name of the metric requesting survival functions (for error messages).
    eval_times : array-like or None
        When provided, passed to the dispatcher so models that accept a
        ``times`` parameter return an aligned ndarray directly.

    Raises
    ------
    ValueError
        If model/X are None, the model lacks predict_survival_function,
        or the model raises NotImplementedError at runtime.
    """
    if model is None or X is None:
        raise ValueError(f"{metric_name} requires model and X when surv_probs not provided")

    unsupported_msg = (
        f"{getattr(model, 'name', 'Model')} does not support "
        f"predict_survival_function(). {metric_name} unavailable."
    )

    if not hasattr(model, 'predict_survival_function'):
        raise ValueError(unsupported_msg)
    try:
        return predict_survival_functions(model, X, eval_times)
    except NotImplementedError as e:
        raise ValueError(unsupported_msg) from e


# Module-level dispatch table (built once after all _compute_* functions are defined)
_METRIC_DISPATCH = {
    'harrell': _compute_harrell,
    'ipcw': _compute_ipcw,
    'brier': _compute_brier,
    'integrated_brier': _compute_integrated_brier,
    'ctd': _compute_ctd,
    'cal_slope': _compute_cal_slope,
    'cal_intercept': _compute_cal_intercept,
    'gnd': _compute_gnd,
}
