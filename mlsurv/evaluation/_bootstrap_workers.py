"""Bootstrap parallelization helpers: precomputation, n_jobs resolution, execution."""

import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from ..constants import SURV_FUNC_METRICS
from ..utils import resolve_joblib_backend, tqdm_joblib as _tqdm_joblib
from .metrics import _surv_funcs_to_probs, predict_survival_functions, _predict_dense_ibs_probs
from ..logging_utils import get_logger, log_metric_failure
from ._bootstrap_iteration import _bootstrap_single_iteration

logger = get_logger("bootstrap")

# Threshold constants for parallelization benefit
# Empirical basis: benchmarked on 16-core machine (Jan 2026) with CoxPH and RSF models.
# RSF models have high serialization overhead (~0.2s per worker), so parallelization
# rarely provides speedup for bootstrap. Use force_parallel=True to override if you know
# your model serializes efficiently.
# Run tests/benchmarks/bench_thresholds.py to re-validate on your hardware.
MIN_ITERATIONS_FOR_PARALLEL = 500  # Full model: sequential faster below this (parallelization overhead)
MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED = 50  # Precomputed arrays: much lower serialization cost


def _reuse_surv_probs_override(surv_probs_override, eval_times, X_test):
    """Return a reusable precomputed ``S(t|x)`` array, or None to fall back.

    ``surv_probs_override`` is an ``(probs, probs_eval_times)`` tuple — a raw
    survival-probability matrix already computed elsewhere (e.g. cached by
    ``evaluate()``). It is only safe to reuse when it was built on the **same**
    eval-times grid the bootstrap resolved (``_resolve_bootstrap_config`` clips
    /auto-derives the grid, so only the resolved ``eval_times`` is
    authoritative) and covers every test row. On any mismatch, return None so
    the caller predicts as usual — keeping output bit-identical.
    """
    if not surv_probs_override:
        return None
    probs, probs_eval_times = surv_probs_override
    if probs is None or probs_eval_times is None:
        return None
    probs = np.asarray(probs, dtype=float)
    probs_eval_times = np.asarray(probs_eval_times, dtype=float)
    # The array must be (n_test_rows, n_eval_times) on the resolved grid. The
    # column check is implied when the grids match and the producer aligned
    # ``probs`` to ``probs_eval_times``, but verify it locally so a malformed
    # override is rejected here rather than misaligning recalibrators downstream.
    if (np.array_equal(probs_eval_times, np.asarray(eval_times, dtype=float))
            and probs.ndim == 2
            and probs.shape[0] == len(X_test)
            and probs.shape[1] == probs_eval_times.shape[0]):
        return probs
    return None


def _precompute_predictions(model, X_test, metrics, eval_times, custom_metrics,
                            include_importance, model_name,
                            surv_probs_transform=None,
                            surv_probs_override=None, include_brier_t=False,
                            y_test=None, y_train=None):
    """
    Precompute risk scores and survival probabilities if beneficial.

    Precomputation is disabled when custom_metrics or include_importance
    are requested (these require fresh predictions per iteration).

    Parameters
    ----------
    surv_probs_transform : callable, optional
        ``f(surv_probs, eval_times) -> ndarray`` applied to the precomputed
        survival probability matrix immediately after it is built and before
        bootstrap iterations index into it. Used by
        ``validate(use_recalibrated=...)`` to apply post-hoc isotonic
        recalibration so bootstrap CIs reflect the same recalibrated S(t|x)
        as the headline point estimate. No-op when ``surv_probs is None``
        (i.e. when ``can_precompute=False`` or the model lacks survival
        functions) — bootstrap CIs revert to raw in that branch.
    surv_probs_override : tuple, optional
        ``(probs, probs_eval_times)`` raw survival probabilities to reuse
        instead of calling ``predict_survival_function`` again. Used by the
        recalibrated bootstrap replay to reuse the raw S(t|x) ``evaluate()``
        already computed, skipping a redundant full-test prediction. Reused
        only when ``probs_eval_times`` equals the resolved ``eval_times`` and
        the row count matches; otherwise the model is predicted as usual.
        The ``surv_probs_transform`` (recalibration) is still applied on top.
    include_brier_t : bool, default=False
        Whether per-time Brier(t) is being collected. Brier(t) needs
        ``S(t|x)`` too, so this forces survival-curve precomputation even when
        the scalar ``metrics`` list has no survival-function metric. Crucially,
        it ensures the ``surv_probs_transform`` (recalibration) is applied to
        the curves Brier(t) reads — without it a recalibrated Brier(t) band
        could silently fall back to raw probabilities.

    Returns
    -------
    tuple : (precomputed_risk, precomputed_surv_probs,
             precomputed_ibs_surv_probs, ibs_times, can_precompute)
        ``precomputed_ibs_surv_probs`` is the dense, train-anchored IBS matrix
        (predicted once on the full test set, with ``surv_probs_transform``
        applied) and ``ibs_times`` its grid; both ``None`` when IBS isn't
        requested or the dense grid has < 2 points. Bootstrap iterations slice
        the matrix by row, so IBS integrates over the same dense grid as the
        headline metric.
    """
    can_precompute = not custom_metrics and not include_importance
    if not can_precompute:
        return None, None, None, None, False

    precomputed_risk = model.predict(X_test)
    precomputed_surv_probs = None

    # Precompute survival curves for any metric that structurally needs them:
    # Brier-family plus ctd/cal_slope/cal_intercept/gnd. The precompute
    # path always sets precomputed_risk, so per-iteration X_boot is dropped to
    # None (see _bootstrap_iteration). These metrics fall back to
    # _get_surv_funcs(model, X) when surv_probs is None, which raises when X is
    # None — so without precomputing the curves they NaN out for *any* model on
    # the precompute path (loky CoxPH as well as threading RSF/XGB).
    needs_surv_funcs = include_brier_t or any(m in metrics for m in SURV_FUNC_METRICS)
    if needs_surv_funcs:
        # Reuse a matching raw S(t|x) override (e.g. evaluate()'s cache) to
        # skip the redundant full-test prediction; build identical numbers.
        precomputed_surv_probs = _reuse_surv_probs_override(
            surv_probs_override, eval_times, X_test,
        )
    if (needs_surv_funcs and precomputed_surv_probs is None
            and hasattr(model, 'predict_survival_function')):
        try:
            surv_funcs = predict_survival_functions(model, X_test, eval_times)
            # If dispatcher returned an ndarray (model supports times param),
            # use it directly; otherwise convert StepFunctions to probs.
            if isinstance(surv_funcs, np.ndarray) and surv_funcs.dtype != object:
                precomputed_surv_probs = surv_funcs
            else:
                precomputed_surv_probs = _surv_funcs_to_probs(surv_funcs, eval_times)
        except (MemoryError, RecursionError):
            raise
        except Exception as e:
            log_metric_failure(
                logger, "survival_precomputation", e,
                model_name=model_name,
                # Expected failure modes: ValueError/AttributeError = metric
                # preconditions not met, NotImplementedError = custom model
                # lacks survival-function support.
                warn_user=not isinstance(e, (ValueError, AttributeError, NotImplementedError)),
            )

    if precomputed_surv_probs is not None and surv_probs_transform is not None:
        precomputed_surv_probs = surv_probs_transform(
            precomputed_surv_probs, eval_times,
        )

    # Precompute the dense, train-anchored IBS matrix once on the full test set
    # (predicted on its own grid, decoupled from the display eval_times). Each
    # iteration slices it by row, so bootstrap IBS integrates over the same
    # dense grid as the headline metric without per-iteration prediction.
    precomputed_ibs_surv_probs = None
    ibs_times = None
    needs_ibs = 'integrated_brier' in metrics
    if (needs_ibs and y_test is not None and y_train is not None
            and hasattr(model, 'predict_survival_function')):
        try:
            precomputed_ibs_surv_probs, ibs_times = _predict_dense_ibs_probs(
                model, X_test, y_test, y_train,
                surv_probs_transform=surv_probs_transform,
            )
        except (MemoryError, RecursionError):
            raise
        except Exception as e:
            log_metric_failure(
                logger, "ibs_precomputation", e,
                model_name=model_name,
                warn_user=not isinstance(e, (ValueError, AttributeError, NotImplementedError)),
            )
            precomputed_ibs_surv_probs, ibs_times = None, None

    return (precomputed_risk, precomputed_surv_probs,
            precomputed_ibs_surv_probs, ibs_times, True)


def _resolve_effective_n_jobs(n_jobs, n_bootstrap, can_precompute, force_parallel,
                              show_progress, model=None):
    """
    Determine effective n_jobs with auto-disable for small workloads.

    Returns 1 (sequential) if parallelization overhead exceeds benefit.
    """
    if n_jobs == 1 or force_parallel:
        return n_jobs

    threshold = (MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED if can_precompute
                 else MIN_ITERATIONS_FOR_PARALLEL)

    # Heavy models (RSF, XGBCox) need more iterations to amortize threading overhead
    if can_precompute and model is not None and getattr(model, 'has_internal_parallelism', False):
        threshold = max(threshold, 100)

    if n_bootstrap >= threshold:
        return n_jobs

    if show_progress:
        print(f"Note: Parallelization disabled ({n_bootstrap} iterations < "
              f"{threshold} threshold). Use force_parallel=True to override.")
    return 1


def _execute_bootstrap_iterations(
    n_bootstrap, seeds, iter_kwargs, effective_n_jobs, show_progress, model_name,
    backend='loky',
):
    """Execute bootstrap iterations sequentially or in parallel.

    Returns
    -------
    list of dict, one per iteration
    """
    if effective_n_jobs == 1:
        iterations_range = range(n_bootstrap)
        if show_progress:
            iterations_range = tqdm(
                iterations_range, desc=f"Bootstrapping {model_name}",
                unit="iter", leave=True,
            )
        return [
            _bootstrap_single_iteration(i=i, seed=seeds[i], **iter_kwargs)
            for i in iterations_range
        ]

    # Parallel execution
    jobs = (
        delayed(_bootstrap_single_iteration)(i=i, seed=seeds[i], **iter_kwargs)
        for i in range(n_bootstrap)
    )
    backend = resolve_joblib_backend(backend)
    if show_progress:
        with _tqdm_joblib(tqdm(
            total=n_bootstrap, desc=f"Bootstrapping {model_name}",
            unit="iter", leave=True,
        )):
            return Parallel(n_jobs=effective_n_jobs, backend=backend)(jobs)
    return Parallel(n_jobs=effective_n_jobs, backend=backend)(jobs)
