"""Bootstrap retraining for per-patient prediction confidence intervals.

Unlike the existing bootstrap in ``bootstrap.py`` (which resamples the *test
set* with a fixed model), this module resamples the *training set* and
retrains the model on each bootstrap sample before predicting on new data.
This provides CIs that capture model uncertainty.

All functions are module-level for joblib pickling compatibility.
"""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .._iter_warn import IterativeFailure, coalesce_failures
from .._warnings import IterativeWarning
from ..models import get_model_metadata
from ..utils import resolve_joblib_backend
from .bootstrap import _generate_seeds
from ._prediction_helpers import (
    compute_patient_median_survival,
    compute_patient_rmst,
    extract_time_grid_and_probs,
    interpolate_surv_at_times,
)

# Prediction bootstrap retrains the full model each iteration.  Benchmarks
# (GBSG2, n_bootstrap=50) show loky (process-based) parallelism only helps
# for heavy models:
#   RSF  loky n_jobs=4: 2.37x speedup  (~350ms/iter, amortises startup)
#   CoxPH loky n_jobs=4: 0.35x          (~20ms/iter, startup dominates)
# Lightweight models never recoup the ~2-3s loky startup at typical bootstrap
# counts (50-200), so parallelism is restricted to heavy models.
MIN_ITERS_FOR_PREDICTION_PARALLEL = 20


def _resolve_prediction_parallel_config(n_jobs, n_bootstrap, model_name,
                                        verbose):
    """Determine effective n_jobs and joblib backend for prediction bootstrap.

    Only models with internal parallelism (RSF, XGBCox, XGBaft) benefit from
    parallel bootstrap retraining — their per-iteration cost (~350ms) amortises
    loky startup.  Lightweight models (~20ms/iter) run sequentially because
    loky overhead dominates at typical bootstrap counts.

    Returns
    -------
    tuple of (int, str)
        (effective_n_jobs, backend)
    """
    if n_jobs == 0:
        raise ValueError(
            "n_jobs=0 has no meaning; use 1 for sequential or -1 for all CPUs."
        )

    if n_jobs == 1:
        return 1, 'loky'

    meta = get_model_metadata(model_name)
    if not meta['has_internal_parallelism']:
        from .._warnings import SuggestionWarning
        warnings.warn(
            f"Parallel prediction bootstrap disabled for {model_name}: "
            "per-iteration cost is too low for loky overhead to pay off. "
            "Running sequentially.",
            SuggestionWarning,
            stacklevel=3,
        )
        return 1, 'loky'

    if n_bootstrap < MIN_ITERS_FOR_PREDICTION_PARALLEL:
        from .._warnings import SuggestionWarning
        warnings.warn(
            f"n_bootstrap={n_bootstrap} < "
            f"{MIN_ITERS_FOR_PREDICTION_PARALLEL} threshold for "
            f"{model_name}; running sequentially to avoid parallel overhead.",
            SuggestionWarning,
            stacklevel=3,
        )
        return 1, 'loky'

    backend = resolve_joblib_backend('loky')
    return n_jobs, backend


def _adjust_proto_njobs(model_proto, model_name, outer_n_jobs):
    """Force n_jobs=1 on a fresh clone when the outer parallel loop is active.

    Returns a clone with adjusted ``n_jobs`` when modification is needed;
    returns the original prototype otherwise (the per-iteration code clones
    again before fitting, so callers may share a single prototype across
    iterations).
    """
    if outer_n_jobs == 1:
        return model_proto

    meta = get_model_metadata(model_name)
    if meta['has_internal_parallelism'] \
            and getattr(model_proto, 'n_jobs', 1) != 1:
        adjusted = clone(model_proto)
        adjusted.set_params(n_jobs=1)
        return adjusted

    return model_proto


def _build_iter_calibrate(pipeline, X_new, X_boot, y_boot, calib_plan):
    """Build this resample's ``S(t|x)`` calibration callable, or ``None``.

    Returns ``calibrate(surv_2d, eval_times) -> surv_2d`` mapping a raw
    ``S(t|x)`` array onto the calibrated scale for this bootstrap iteration,
    or ``None`` when no calibration applies (the iteration stays raw). The
    callable is applied to every per-iteration ``S(t|x)`` array (the band's
    ``surv_on_ref`` and the user-times ``surv_at_times``), not just the band,
    so medians/RMST derived from it are calibrated too. This mirrors the
    ``train(calibrate=...)`` point-estimate path per resample:

    * Warper calibrators (``isotonic``, ``lowdof``) are a probs->probs map.
      A ``'fixed'`` plan (``{'method', 'calibrator'}``) reapplies the
      already-fitted map to each iteration -- no per-resample refit. A
      ``'refit'`` plan (``{'method', 'horizons', 'splitter', ...}``) instead
      refits the warper on this resample's own OOF ``S(t|x)`` (the literature-
      endorsed full-pipeline replication), so the band carries the calibration
      step's own sampling variability.
    * ``breslow_slope`` recomputes ``S(t|x)`` from ``pipeline.predict(X)``
      scaled by a baseline fit on ``beta * score`` (it ignores ``surv_probs``
      entirely), so a stored baseline can't be reused -- it needs this
      resample's own refitted scores. Refit a resample-specific Breslow
      baseline on this resample's OOF risk scores. Replicating the
      calibration step per resample is the literature-endorsed approach
      (Riley et al. 2024 / Steyerberg ABCD / rms ``calibrate``) and gives the
      band the calibration step's own sampling variability.

    A ``None`` return for a refit plan (the per-resample refit failed) is a
    *skip*, not "no calibration" -- the caller flags it so the parent can
    warn (see ``_prediction_bootstrap_iteration`` / ``run_prediction_bootstrap``).

    Module-level (not a nested ``def``) for readability; the returned closure
    is created and consumed inside one worker call, so it is never pickled.
    """
    if calib_plan is None:
        return None
    method = calib_plan['method']
    if method in ('isotonic', 'lowdof'):
        if 'calibrator' in calib_plan:                 # 'fixed': reapply map
            cal = calib_plan['calibrator']
            return lambda arr, et: cal.apply(
                pipeline=pipeline, X=X_new, surv_probs=arr, eval_times=et,
            )
        # 'refit': refit the warper on this resample's own OOF S(t|x), mirroring
        # the breslow_slope path below so the band carries calibration-fit
        # variability. want_scores=False -> _collect_oof_survival returns just
        # the (n, n_horizons) oof_S matrix the warper fit() consumes.
        from ._calibrators import IsotonicCalibrator, LowDoFCalibrator
        from ._recalibration import _collect_oof_survival
        horizons = np.asarray(calib_plan['horizons'], dtype=float)
        warper_cls = (
            IsotonicCalibrator if method == 'isotonic' else LowDoFCalibrator
        )
        try:
            oof_S = _collect_oof_survival(
                pipeline, X_boot, y_boot, calib_plan['splitter'], horizons,
                want_scores=False,
            )
            cal = warper_cls().fit(
                y_train=y_boot, horizons=horizons, oof_S=oof_S,
                model_name=calib_plan.get('model_name', 'model'),
            )
        except (ValueError, RuntimeError, TypeError, ArithmeticError,
                np.linalg.LinAlgError):
            # Rare: this resample is too small/degenerate to refit the warper.
            # Leave the iteration raw rather than crash the whole bootstrap;
            # the caller counts the skip and warns from the parent process.
            # (Matches the breslow_slope handler below.)
            return None
        # A degenerate resample can leave a warper an identity no-op without
        # raising: isotonic with no usable per-horizon maps (empty ``mappings``,
        # apply() returns S unchanged), or lowdof falling back to (a, b)=(0, 1).
        # That iteration is really raw, so route it to the skip count -- like a
        # refit failure -- instead of passing it off as calibrated (which would
        # leave the band quietly mixing raw and calibrated curves with no warn).
        identity = (
            (method == 'isotonic' and not cal.mappings)
            or (method == 'lowdof' and cal.a == 0.0 and cal.b == 1.0)
        )
        if identity:
            return None
        return lambda arr, et: cal.apply(
            pipeline=pipeline, X=X_new, surv_probs=arr, eval_times=et,
        )
    if method == 'breslow_slope':
        from ._calibrators import BreslowSlopeCalibrator
        from ._recalibration import _collect_oof_survival
        horizons = np.asarray(calib_plan['horizons'], dtype=float)
        try:
            _oof_S, oof_scores = _collect_oof_survival(
                pipeline, X_boot, y_boot, calib_plan['splitter'], horizons,
                want_scores=True,
            )
            cal = BreslowSlopeCalibrator().fit(
                y_train=y_boot, horizons=horizons, oof_scores=oof_scores,
            )
        except (ValueError, RuntimeError, TypeError, ArithmeticError,
                np.linalg.LinAlgError):
            # Rare: this resample is too small/degenerate to refit the slope.
            # Leave the iteration raw rather than crash the whole bootstrap;
            # the caller counts the skip and warns from the parent process.
            # (Matches the iteration-level handler below and the slope-refit
            # guard in BreslowSlopeCalibrator.fit.)
            return None
        return lambda arr, et: cal.apply(
            pipeline=pipeline, X=X_new, surv_probs=arr, eval_times=et,
        )
    return None


def _prediction_bootstrap_iteration(
    seed,
    X_train,
    y_train,
    X_new,
    model_proto,
    imputer,
    scaler,
    selector,
    time_grid,
    rmst_default_tau,
    user_times,
    n_patients,
    compute_surv_band=False,
    calib_plan=None,
):
    """Single bootstrap iteration: resample training data, refit, predict.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    X_train, y_train : DataFrame, SurvivalArray
        Original training data to resample from.
    X_new : DataFrame
        New patient data to predict on.
    model_proto : BaseSurvivalModel
        Unfitted model prototype carrying all hyperparameters (including
        any tuned values). Cloned per iteration before fit. Sourced from
        ``pipeline.named_steps['model']`` upstream — this avoids the
        registry / name round-trip that previously broke for tuned
        models.
    imputer, scaler, selector : transformer or None
        Unfitted preprocessing components (will be cloned and fitted).
    time_grid : ndarray
        Reference time grid from point-estimate predictions.
    rmst_default_tau : float
        Default RMST horizon (max training time).
    user_times : ndarray or None
        User-specified time points for S(t) and RMST.
    n_patients : int
        Number of new patients.
    calib_plan : dict or None
        Per-iteration ``S(t|x)`` calibration plan from
        ``train(calibrate=...)``; ``None`` leaves the iteration on the raw
        scale. See ``_build_iter_calibrate``.

    Returns
    -------
    dict or None
        Dict with keys ``medians``, ``rmst``, ``surv_at_times``,
        ``rmst_at_times``.  ``None`` if the iteration fails.
    """
    with warnings.catch_warnings():
        # Suppress per-iteration warnings (restored after context exits)
        warnings.filterwarnings("once", category=IterativeWarning)
        # Suppress per-patient median-undefined warnings during bootstrap
        # (the point-estimate call already warns; repeating per-iteration
        # would produce up to n_bootstrap warnings)
        from .._warnings import MethodologicalWarning
        warnings.filterwarnings("ignore", category=MethodologicalWarning)

        rng = np.random.RandomState(seed)

        # 1. Resample training data with replacement
        n_train = len(X_train)
        idx = rng.randint(0, n_train, size=n_train)
        X_boot = X_train.iloc[idx].reset_index(drop=True)
        y_boot = y_train[idx]

        try:
            # 2. Clone and fit preprocessing
            X_processed, fitted_imputer, fitted_scaler, fitted_selector = (
                _fit_preprocessing(X_boot, y_boot, imputer, scaler, selector)
            )

            # 3. Clone proto (carries tuned hyperparameters) and fit
            model = clone(model_proto)
            model.fit(X_processed, y_boot)

            # 4. Assemble pipeline and predict
            pipeline = _assemble_pipeline(
                model, fitted_imputer, fitted_scaler, fitted_selector,
            )
            surv_funcs = pipeline.predict_survival_function(X_new)
            boot_time_grid, boot_surv_probs = extract_time_grid_and_probs(
                surv_funcs,
            )

            # 5. Compute quantities on the reference time grid
            boot_surv_on_ref = interpolate_surv_at_times(
                boot_time_grid, boot_surv_probs, time_grid,
            )

            # 5a. Per-resample calibration. Calibrating each iteration's
            # S(t|x) here -- before deriving medians/RMST/S(t|x)-at-times --
            # keeps every CI quantity on the same scale as the calibrated
            # point estimate (calibrating only the assembled band would leave
            # the median/RMST CIs raw). Warpers reapply the fitted map;
            # breslow_slope refits a resample-specific Breslow baseline. See
            # _build_iter_calibrate.
            iter_calibrate = _build_iter_calibrate(
                pipeline, X_new, X_boot, y_boot, calib_plan,
            )
            # _build_iter_calibrate returns None when a breslow_slope plan's
            # per-resample refit failed (this iteration stays raw); a failed
            # calibrator *apply* below sets calib_skipped too, for any method.
            # Flag the skip so the parent can warn -- a warning from inside the
            # worker would not reach the user.
            calib_skipped = calib_plan is not None and iter_calibrate is None
            if iter_calibrate is not None:
                try:
                    boot_surv_on_ref = iter_calibrate(boot_surv_on_ref, time_grid)
                except (ValueError, RuntimeError, TypeError, ArithmeticError,
                        np.linalg.LinAlgError):
                    # Applying the calibrator failed for this resample. Leave
                    # the iteration on the raw scale (and skip the user-times
                    # apply below, since iter_calibrate is now None) rather
                    # than drop the whole iteration; the parent counts the
                    # skip. Keeps "calibration problems degrade to raw, never
                    # crash" uniform with the refit path.
                    iter_calibrate = None
                    calib_skipped = True

            medians = compute_patient_median_survival(
                time_grid, boot_surv_on_ref,
            )
            rmst_vals = compute_patient_rmst(
                time_grid, boot_surv_on_ref, rmst_default_tau,
            )

            surv_at_times = None
            rmst_at_times = None
            if user_times is not None:
                surv_at_times = interpolate_surv_at_times(
                    boot_time_grid, boot_surv_probs, user_times,
                )
                if iter_calibrate is not None:
                    surv_at_times = iter_calibrate(surv_at_times, user_times)
                rmst_at_times = np.column_stack([
                    compute_patient_rmst(time_grid, boot_surv_on_ref, tau)
                    for tau in user_times
                ])

            return {
                'medians': medians,
                'rmst': rmst_vals,
                'surv_at_times': surv_at_times,
                'rmst_at_times': rmst_at_times,
                # Full survival-function band (shape: n_patients x n_times)
                # is retained only when requested — skipping the copy keeps
                # batched predict(ci=True, ci_band=False) memory flat.
                'surv_on_ref': (
                    boot_surv_on_ref if compute_surv_band else None
                ),
                'calib_skipped': calib_skipped,
            }

        except (MemoryError, RecursionError):
            raise
        except (ValueError, RuntimeError, TypeError,
                np.linalg.LinAlgError) as exc:
            return {
                '__failure__': IterativeFailure(
                    exc_type_name=type(exc).__name__,
                    message=str(exc),
                    context_prefix=f"seed={seed}",
                ),
            }


def _fit_preprocessing(X, y, imputer, scaler, selector):
    """Clone and fit preprocessing components on data.

    Mirrors the logic of ``_prefit_preprocessing`` in ``_training.py``.
    """
    from mlsurv.preprocessing.selection import _is_cox_based

    fitted_imputer = fitted_scaler = fitted_selector = None

    # Pre-fit Cox-based selector on raw data before imputation
    cox_prefitted = False
    if selector is not None and _is_cox_based(selector):
        fitted_selector = clone(selector)
        if hasattr(fitted_selector, 'set_output'):
            fitted_selector.set_output(transform="pandas")
        fitted_selector.fit(X, y)
        cox_prefitted = True

    if imputer is not None:
        fitted_imputer = clone(imputer)
        if hasattr(fitted_imputer, 'set_output'):
            fitted_imputer.set_output(transform="pandas")
        X = fitted_imputer.fit_transform(X)

    if scaler is not None:
        fitted_scaler = clone(scaler)
        if hasattr(fitted_scaler, 'set_output'):
            fitted_scaler.set_output(transform="pandas")
        X = fitted_scaler.fit_transform(X)

    if selector is not None:
        if cox_prefitted:
            X = fitted_selector.transform(X)
        else:
            fitted_selector = clone(selector)
            if hasattr(fitted_selector, 'set_output'):
                fitted_selector.set_output(transform="pandas")
            X = fitted_selector.fit_transform(X, y)

    return X, fitted_imputer, fitted_scaler, fitted_selector


def _assemble_pipeline(model, fitted_imputer, fitted_scaler, fitted_selector):
    """Assemble Pipeline from pre-fitted components."""
    steps = []
    for name, comp in [('imputer', fitted_imputer),
                       ('scaler', fitted_scaler),
                       ('selector', fitted_selector)]:
        if comp is not None:
            steps.append((name, comp))
    steps.append(('model', model))
    pipeline = Pipeline(steps)
    pipeline.set_output(transform="pandas")
    return pipeline


def run_prediction_bootstrap(
    X_train,
    y_train,
    X_new,
    model_name,
    model_proto,
    imputer,
    scaler,
    selector,
    time_grid,
    rmst_default_tau,
    user_times,
    n_bootstrap=200,
    random_state=None,
    n_jobs=1,
    verbose=True,
    compute_surv_band=False,
    calib_plan=None,
):
    """Run bootstrap retraining loop for prediction CIs.

    Parameters
    ----------
    X_train : DataFrame
        Training features to resample from.
    y_train : SurvivalArray
        Training outcomes to resample from.
    X_new : DataFrame
        New patient data.
    model_name : str
        Algorithm identity (registry key, e.g. ``'rsf'``) — used only
        for metadata lookup (``has_internal_parallelism``) and progress
        labels. Pass ``ModelResult.base_name``, not ``ModelResult.name``.
    model_proto : BaseSurvivalModel
        Unfitted model prototype carrying all hyperparameters. Cloned
        per iteration before fit.
    imputer : transformer or None
        Unfitted imputation component.
    scaler : transformer or None
        Unfitted scaling component.
    selector : transformer or None
        Unfitted feature-selection component.
    time_grid : ndarray
        Time grid from point-estimate predictions.
    rmst_default_tau : float
        Default RMST horizon.
    user_times : ndarray or None
        User-specified time points (also used as RMST horizons).
    n_bootstrap : int
        Number of bootstrap iterations.
    random_state : int or None
        Base seed.
    n_jobs : int
        Parallel jobs (1 = sequential).
    verbose : bool
        Print progress.
    compute_surv_band : bool, default=False
        If True, also collect each iteration's survival curve on the
        reference grid to build a survival-probability CI band.
    calib_plan : dict, optional
        ``train(calibrate=...)`` internal-calibrator plan applied per
        iteration so the CI band sits on the same scale as the calibrated
        point estimate. A *fixed* plan carries ``{'method', 'calibrator'}``
        (the warper map reapplied unchanged); a *refit* plan carries
        ``{'method', 'horizons', 'splitter', ...}`` for a resample-specific
        refit (warpers under ``band_calibration='refit'``, and always
        ``breslow_slope``). See ``_build_iter_calibrate``.

    Returns
    -------
    dict
        Keys: ``medians``, ``rmst``, ``surv_at_times``, ``rmst_at_times``.
        Each is an ndarray with shape ``(n_valid, ...)`` or ``None``.
    """
    seeds = _generate_seeds(random_state, n_bootstrap)
    n_patients = len(X_new)

    effective_n_jobs, backend = _resolve_prediction_parallel_config(
        n_jobs, n_bootstrap, model_name, verbose,
    )
    # Freeze any hyperparameters the wrapper would otherwise re-select per
    # resample (e.g., CoxNet's alpha path / CV-based selection). The
    # default impl is just clone(self); only wrappers with internal selection
    # override it.
    frozen_proto = model_proto.freeze_for_resampling()
    adjusted_proto = _adjust_proto_njobs(
        frozen_proto, model_name, effective_n_jobs,
    )

    iter_args = dict(
        X_train=X_train,
        y_train=y_train,
        X_new=X_new,
        model_proto=adjusted_proto,
        imputer=imputer,
        scaler=scaler,
        selector=selector,
        time_grid=time_grid,
        rmst_default_tau=rmst_default_tau,
        user_times=user_times,
        n_patients=n_patients,
        compute_surv_band=compute_surv_band,
        calib_plan=calib_plan,
    )

    if verbose:
        print(f"  Bootstrap retraining: {n_bootstrap} iterations "
              f"(n_jobs={effective_n_jobs}, backend={backend!r})...")

    if effective_n_jobs == 1:
        results = [
            _prediction_bootstrap_iteration(seed=s, **iter_args)
            for s in seeds
        ]
    else:
        results = Parallel(n_jobs=effective_n_jobs, backend=backend)(
            delayed(_prediction_bootstrap_iteration)(seed=s, **iter_args)
            for s in seeds
        )

    # Split results: success dicts vs failure sentinels
    valid = [r for r in results if r is not None and '__failure__' not in r]
    failures = [
        r['__failure__'] for r in results
        if r is not None and '__failure__' in r
    ]

    if failures:
        coalesce_failures(
            failures,
            n_total=n_bootstrap,
            summary_label="bootstrap",
            category=IterativeWarning,
            stacklevel=4,
            tail_msg="CIs may be less reliable.",
        )

    if len(valid) == 0:
        from .._warnings import MethodologicalWarning
        # When every iteration died the same way, surface that exact
        # exception instead of generic "data instability" copy — a single
        # shared (type, message) almost always means a deterministic
        # programming error (e.g. feeding a non-existent model key to
        # get_model()), not statistical instability.
        unique = {(f.exc_type_name, f.message) for f in failures}
        if len(unique) == 1:
            exc_type_name, message = next(iter(unique))
            msg = (
                "All bootstrap iterations failed with the same error "
                f"({exc_type_name}: {message}). This is a deterministic "
                "failure, not data instability — fix the underlying error "
                "before interpreting CIs as missing for statistical reasons."
            )
        else:
            msg = (
                "All bootstrap iterations failed. CIs are unavailable. "
                "Inspect the IterativeWarning above for the underlying "
                "exception. If every iteration shares the same exception "
                "type and message, it is almost certainly a programming "
                "error (e.g., an unfittable preprocessing/model "
                "configuration), not data instability — fix the underlying "
                "error before interpreting CIs as missing for statistical "
                "reasons."
            )
        warnings.warn(msg, MethodologicalWarning, stacklevel=4)
        return {
            'medians': None,
            'rmst': None,
            'surv_at_times': None,
            'rmst_at_times': None,
            'surv_probs': None,
        }

    # A breslow_slope plan can fail to refit on a degenerate resample, leaving
    # that iteration's S(t|x) raw. Surface the count from the parent process —
    # a warning from inside the joblib worker would not reach the user.
    n_calib_skipped = sum(1 for r in valid if r.get('calib_skipped'))
    if n_calib_skipped:
        from .._warnings import SuggestionWarning
        warnings.warn(
            f"{n_calib_skipped}/{len(valid)} bootstrap iteration(s) could not "
            "apply the S(t|x) calibrator and were left on the raw scale; the "
            "CI band mixes calibrated and raw curves and is approximate.",
            SuggestionWarning, stacklevel=4,
        )

    # Stack results
    boot_medians = np.vstack([r['medians'] for r in valid])
    boot_rmst = np.vstack([r['rmst'] for r in valid])

    boot_surv_at_times = None
    boot_rmst_at_times = None
    if user_times is not None and valid[0]['surv_at_times'] is not None:
        boot_surv_at_times = np.stack([r['surv_at_times'] for r in valid])
        boot_rmst_at_times = np.stack([r['rmst_at_times'] for r in valid])

    boot_surv_probs = None
    if compute_surv_band and valid[0].get('surv_on_ref') is not None:
        # Shape: (n_valid, n_patients, n_times)
        boot_surv_probs = np.stack([r['surv_on_ref'] for r in valid])

    if verbose:
        print(f"  Bootstrap complete: {len(valid)}/{n_bootstrap} "
              "iterations succeeded.")

    return {
        'medians': boot_medians,
        'rmst': boot_rmst,
        'surv_at_times': boot_surv_at_times,
        'rmst_at_times': boot_rmst_at_times,
        'surv_probs': boot_surv_probs,
    }
