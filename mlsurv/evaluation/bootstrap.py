"""Bootstrap evaluation for confidence interval estimation."""

import warnings
from collections import Counter, OrderedDict
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .._shared import nan_summary, percentile_ci
from .._warnings import (
    IterativeWarning,
    MethodologicalWarning,
    MetricFailureWarning,
    _warn_eval_times_failure,
)
from ..results.containers import BootstrapResult
from ..utils import (
    SurvivalArray,
    auto_eval_times,
    clip_eval_times,
    union_extra_eval_times,
    adjust_model_njobs,
    save_model_njobs,
    restore_model_njobs,
    resolve_joblib_backend,
)
from ..constants import ALL_METRICS, LOWER_IS_BETTER
from .helpers import resolve_n_bootstrap, resolve_model_name, _has_internal_parallelism
from ..logging_utils import get_logger, log_exception, log_metric_failure
from ._config import BootstrapConfig
from ._bootstrap_workers import (
    _precompute_predictions,
    _resolve_effective_n_jobs,
    _execute_bootstrap_iterations,
)
from ._bootstrap_iteration import _safe_compute_metric
from ._bootstrap_workers import MIN_ITERATIONS_FOR_PARALLEL  # noqa: F401 — re-exported
from ._bootstrap_workers import MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED  # noqa: F401

logger = get_logger("bootstrap")

# Threshold for outer (multi-model) parallelization
# Empirical basis: 2 models x 150 iterations = 300, crossover point where outer
# parallelism begins to outperform sequential (benchmarked Jan 2026, 16-core).
# Run tests/benchmarks/bench_thresholds.py to re-validate on your hardware.
MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL = 300  # n_models * n_bootstrap threshold


def _generate_seeds(base_seed, n_seeds):
    """Generate n independent seeds for reproducible parallel execution."""
    if base_seed is None:
        return [None] * n_seeds
    rng = np.random.RandomState(base_seed)
    return rng.randint(0, 2**31, size=n_seeds).tolist()


def _compute_point_estimates(model, X_test, y_test: SurvivalArray, y_train: Optional[SurvivalArray], metrics, custom_metrics,
                             eval_times, precomputed_risk=None, precomputed_surv_probs=None,
                             model_name=None, ipcw_tau: Optional[float] = None,
                             ipcw_y_train: Optional[SurvivalArray] = None,
                             precomputed_ibs_surv_probs=None, ibs_times=None):
    """Compute point estimates on the full (un-resampled) test set.

    Parameters
    ----------
    model : BaseSurvivalModel
        Fitted survival model.
    X_test : DataFrame
        Full test features (no resampling).
    y_test : SurvivalArray
        Full test outcomes.
    y_train : SurvivalArray or None
        Training outcomes (needed for IPCW metrics).
    metrics : list of str
        Built-in metric names to compute.
    custom_metrics : dict
        Custom scorer functions: {name: callable(model, X, y)}.
    eval_times : array-like
        Evaluation timepoints for Brier/AUC(t).
    precomputed_risk : array-like, optional
        Precomputed risk scores (avoids redundant predict call).
    precomputed_surv_probs : array-like, optional
        Precomputed survival probabilities.
    model_name : str, optional
        Model name for log messages.

    Returns
    -------
    dict
        {metric_name: float} point estimates for each metric.
    """
    risk_scores = precomputed_risk if precomputed_risk is not None else model.predict(X_test)
    estimates = {}

    for metric in metrics:
        value = _safe_compute_metric(
            y_test, risk_scores, metric, y_train, model, X_test,
            eval_times, precomputed_surv_probs, i='point', model_name=model_name,
            ipcw_tau=ipcw_tau, ipcw_y_train=ipcw_y_train,
            ibs_surv_probs=precomputed_ibs_surv_probs, ibs_times=ibs_times,
        )
        estimates[metric] = value

    for metric_name, scorer in custom_metrics.items():
        try:
            estimates[metric_name] = scorer(model, X_test, y_test)
        except (MemoryError, RecursionError):
            raise
        except Exception as e:
            log_metric_failure(logger, metric_name, e, iteration='point',
                               model_name=model_name, warn_user=True)
            estimates[metric_name] = np.nan

    return estimates


def _compute_summary_statistics(iterations_df, metrics, custom_metrics, model_name, ci,
                                point_estimates=None):
    """Compute mean, std, and CI for each metric from bootstrap iterations."""
    # Include all subpopulation-namespaced metrics (harrell__subpopulation_*, ipcw__subpopulation_*,
    # integrated_brier__subpopulation_*, etc.) found in iterations
    subpopulation_metrics = []
    if 'metric_name' in iterations_df.columns:
        all_iter_metrics = iterations_df['metric_name'].unique()
        subpopulation_metrics = [m for m in all_iter_metrics
                            if isinstance(m, str) and '__subpopulation_' in m]
    all_metric_names = list(metrics) + list(custom_metrics.keys()) + subpopulation_metrics
    summary_data = []

    for metric in all_metric_names:
        if len(iterations_df) > 0 and 'metric_name' in iterations_df.columns:
            mask = iterations_df['metric_name'] == metric
            dist = iterations_df.loc[mask, 'value'].values.astype(float)
        else:
            dist = np.array([])
        valid_dist = dist[~np.isnan(dist)]

        if len(valid_dist) > 0:
            lo, hi = percentile_ci(valid_dist, ci_level=ci)
            summary_data.append({
                'model_type': model_name,
                'metric': metric,
                'mean': np.mean(valid_dist),
                'std': np.std(valid_dist, ddof=1),
                'ci_lower': lo,
                'ci_upper': hi,
            })
        else:
            summary_data.append({
                'model_type': model_name,
                'metric': metric,
                **nan_summary(),
            })

    summary_df = pd.DataFrame(summary_data)

    if point_estimates is not None:
        summary_df['point'] = summary_df['metric'].map(point_estimates)
    else:
        summary_df['point'] = np.nan

    cols = ['model_type', 'metric', 'point', 'mean', 'std', 'ci_lower', 'ci_upper']
    return summary_df[cols]


def _aggregate_iteration_results(results, config):
    """Aggregate per-iteration bootstrap results into flat lists.

    Parameters
    ----------
    results : list of dict, one per iteration
    config : BootstrapConfig

    Returns
    -------
    tuple of (iteration_data, predictions_data, auc_data, brier_t_data,
              importance_data, subpopulation_auc_data, all_failures, all_notices)
    """
    iteration_data = []
    all_failures = []
    all_notices = []
    has_subpopulations = config.subpopulation_labels is not None
    optional_keys = [
        ('predictions_data', config.include_predictions),
        ('auc_data', config.include_auc),
        ('brier_t_data', config.include_brier_t),
        ('importance_data', config.include_importance),
        ('subpopulation_auc_data', has_subpopulations and config.eval_times is not None),
    ]
    aggregated = {key: [] if include else None for key, include in optional_keys}

    for result in results:
        iteration_data.extend(result['iteration_data'])
        all_failures.extend(result.get('failures', []))
        all_notices.extend(result.get('notices', []))
        for key, include in optional_keys:
            if include and result.get(key):
                aggregated[key].extend(result[key])

    return (iteration_data, aggregated['predictions_data'], aggregated['auc_data'],
            aggregated['brier_t_data'], aggregated['importance_data'],
            aggregated['subpopulation_auc_data'], all_failures, all_notices)


def _emit_failure_summary(failures, n_bootstrap, model_name, *,
                          _warn_stacklevel=2):
    """Emit a summary of metric failures across bootstrap iterations.

    Groups failures by (metric, exception_type) and emits one
    ``MetricFailureWarning`` per group.

    ``_warn_stacklevel`` is the ``stacklevel`` value passed to
    ``warnings.warn``. Callers thread it so ``MetricFailureWarning`` attributes
    to the user's frame instead of a helper inside ``bootstrap.py``.
    Each entry point (free ``bootstrap()``, ``EvaluationMixin.bootstrap``,
    ``evaluate(bootstrap=True)``, ``validate()``) counts the hops from the
    ``warn`` call to the user and passes the matching value.
    """
    if not failures:
        return

    counts = Counter()
    examples = {}
    for metric, exc_type, exc_msg in failures:
        key = (metric, exc_type)
        counts[key] += 1
        if key not in examples:
            examples[key] = exc_msg

    for (metric, exc_type), count in counts.items():
        msg = (
            f"{model_name}: {metric} failed in {count}/{n_bootstrap} "
            f"iterations ({exc_type}: {examples[(metric, exc_type)]})"
        )
        # Every failed iteration drops user-visible bootstrap data,
        # so warn unconditionally. IterativeWarning's "once" filter handles
        # dedup of repeated identical summaries.
        warnings.warn(msg, MetricFailureWarning, stacklevel=_warn_stacklevel)


def _emit_low_coverage_summary(notices, n_bootstrap, model_name, *,
                                _warn_stacklevel=2):
    """Emit a low-coverage summary, coalescing subpopulation levels.

    Notices are grouped by ``metric``. A metric whose notices span **two or
    more distinct subpopulation levels** (e.g. ``subpopulation_auc_t`` over a
    many-subgroup cohort) collapses into a single MethodologicalWarning that
    lists each level with its own coverage detail, rather than one warning per
    level. Cohort-level metrics (``subpop=None``) and metrics with a single
    level keep the original one-line-per-group format.

    Unlike :func:`_emit_failure_summary`, this always warns — insufficient
    eval_times coverage is a methodological concern the user must see per
    the "No Silent Data Loss" rule.

    Tracks min/max ``n_valid`` across iterations within each (metric, level)
    group so the summary reflects worst-case and best-case coverage rather
    than pinning to the first-seen iteration. ``n_total`` is invariant per
    metric site (always ``len(eval_times)``).
    """
    if not notices:
        return
    # metric -> {n_total, subpops: OrderedDict[subpop_or_None -> bucket]}
    grouped: "OrderedDict[str, dict]" = OrderedDict()
    for metric, subpop, n_valid, n_total in notices:
        mg = grouped.get(metric)
        if mg is None:
            mg = grouped[metric] = {"n_total": n_total, "subpops": OrderedDict()}
        bucket = mg["subpops"].get(subpop)
        if bucket is None:
            mg["subpops"][subpop] = {
                "count": 1,
                "min_valid": n_valid,
                "max_valid": n_valid,
            }
        else:
            bucket["count"] += 1
            bucket["min_valid"] = min(bucket["min_valid"], n_valid)
            bucket["max_valid"] = max(bucket["max_valid"], n_valid)
        # ``n_total`` is invariant per metric site; refresh defensively.
        mg["n_total"] = n_total

    def _emit_single_group(subpop, info):
        # Original one-line format for a cohort or single-level group.
        label = f" subpopulation '{subpop}'" if subpop is not None else ""
        coverage = _format_coverage(
            info["min_valid"], info["max_valid"], n_total
        )
        msg = (
            f"{model_name}{label}: {metric} insufficient eval_times "
            f"coverage ({coverage}) in {info['count']}/{n_bootstrap} "
            f"bootstrap iterations. Reporting NaN for affected rows. "
            f"Pass eval_times within the cohort's observed time window."
        )
        warnings.warn(msg, MethodologicalWarning, stacklevel=_warn_stacklevel)

    for metric, mg in grouped.items():
        subpops = mg["subpops"]
        n_total = mg["n_total"]
        # ``subpop is None`` is the cohort-level metric (``auc_t``); named
        # levels carry ``subpopulation_auc_t``. Use identity, not truthiness,
        # so an empty-string level is still treated as a named level.
        named = [(s, b) for s, b in subpops.items() if s is not None]
        cohort = [(s, b) for s, b in subpops.items() if s is None]
        if len(named) >= 2:
            # Collapse all subpopulation levels into a single line.
            frags = ", ".join(
                f"{subpop} "
                f"({_format_coverage(b['min_valid'], b['max_valid'], n_total)}, "
                f"{b['count']}/{n_bootstrap} iters)"
                for subpop, b in named
            )
            msg = (
                f"{model_name}: {metric} insufficient eval_times coverage in "
                f"{len(named)} subpopulation levels / {n_bootstrap} "
                f"bootstrap iterations. Reporting NaN for affected rows "
                f"({frags}). Pass eval_times within the cohort's observed time "
                f"window."
            )
            warnings.warn(msg, MethodologicalWarning, stacklevel=_warn_stacklevel)
            # A cohort bucket should not share a metric with named levels
            # (different metric names), but emit it separately if present so
            # no recorded notice is ever dropped.
            for subpop, info in cohort:
                _emit_single_group(subpop, info)
        else:
            # Cohort-level or single-level metric: original one-line format.
            for subpop, info in subpops.items():
                _emit_single_group(subpop, info)


def bootstrap(
    model: Union[object, Dict[str, object]],
    X_test,
    y_test: SurvivalArray,
    y_train: Optional[SurvivalArray] = None,
    n_bootstrap="auto",
    metrics=None,
    custom_metrics=None,
    lower_is_better=None,
    ci=0.95,
    eval_times=None,
    extra_eval_times=None,
    include_predictions=False,
    include_auc=False,
    include_brier_t=False,
    include_importance=False,
    n_importance_repeats=10,
    stratify=False,
    random_state=None,
    show_progress=True,
    model_name=None,
    n_jobs=1,
    n_jobs_models=1,
    force_parallel=False,
    subpopulation_labels=None,
    categorical_groups=None,
    ipcw_tau: Optional[float] = None,
    ipcw_y_train: Optional[SurvivalArray] = None,
    _warn_stacklevel=4,
    # Per-model survival-probability transforms applied to
    # ``precomputed_surv_probs`` once before iterations index into it. Used
    # by validate(use_recalibrated=...) so bootstrap CIs reflect the same
    # recalibrated S(t|x) as the headline point estimate.
    surv_probs_transforms: Optional[Dict[str, Callable]] = None,
    # Per-model ``(probs, eval_times)`` raw survival probabilities to reuse
    # instead of re-predicting (see ``_precompute_predictions``). Used by the
    # recalibrated bootstrap replay to reuse the raw S(t|x) ``evaluate()``
    # already cached, skipping a redundant full-test prediction.
    surv_probs_overrides: Optional[Dict[str, tuple]] = None,
    # External failure collector. When supplied (a dict), the
    # single-model branch suppresses its per-model emission and appends
    # failures under ``model_name``. The dict-of-models branch ignores it
    # because ``_bootstrap_multiple`` already aggregates internally via
    # ``_emit_multi_failure_summary``.
    _failure_collector: Optional[dict] = None,
    # External low-coverage notices collector, analogous to
    # ``_failure_collector``. When supplied (a dict), the single-model branch
    # suppresses its per-model emission and appends notices under
    # ``model_name`` so the caller can cross-coalesce via
    # ``_emit_multi_low_coverage_summary``. The dict-of-models branch ignores
    # it because ``_bootstrap_multiple`` already aggregates internally.
    _notices_collector: Optional[dict] = None,
):
    """
    Bootstrap evaluation with configurable outputs.

    Supports both single models and dictionaries of models. When a dict is
    provided, results are aggregated with model_type column identifying each.

    Parameters
    ----------
    model : BaseSurvivalModel or Dict[str, BaseSurvivalModel]
        Fitted model (not retrained during bootstrap), or dictionary of
        {model_name: model} pairs for multi-model bootstrap.
    X_test : DataFrame
        Test features.
    y_test : SurvivalArray
        Test outcomes.
    y_train : SurvivalArray, optional
        Training outcomes (required for IPCW metrics).
    n_bootstrap : int or "auto", default="auto"
        Number of bootstrap iterations. If "auto", computed as
        min(500, max(100, 10 * sqrt(n_samples))).
    metrics : list of str, optional
        Metrics to compute. Default: all metrics (harrell, ipcw, integrated_brier, ctd).
    custom_metrics : dict, optional
        Additional custom metrics to compute. Dict of {name: scorer_fn}.
    lower_is_better : list or set, optional
        Metric names where lower values indicate better performance.
    ci : float, default=0.95
        Confidence level for summary statistics.
    eval_times : array-like, optional
        Timepoints for AUC(t) and Brier score.
    include_predictions : bool, default=False
        Store risk scores for each bootstrap iteration.
    include_auc : bool, default=False
        Store AUC(t) values for each bootstrap iteration.
    include_brier_t : bool, default=False
        Store per-time Brier(t) values for each bootstrap iteration (the
        Brier counterpart of ``include_auc``). When ``surv_probs_transforms``
        is supplied, the stored Brier(t) reflects the transformed (e.g.
        recalibrated) ``S(t|x)``.
    include_importance : bool, default=False
        Store permutation importance for each bootstrap iteration.
    n_importance_repeats : int, default=10
        Permutation repeats per iteration (fewer than evaluate()).
    stratify : bool, default=False
        Stratify resampling by event status.
    random_state : int, optional
        Seed for reproducibility.
    show_progress : bool, default=True
        Show progress bar during bootstrapping.
    model_name : str, optional
        Name to display in progress bar and results.
    n_jobs : int, default=1
        Number of parallel jobs for bootstrap iterations (inner parallelization).
    n_jobs_models : int, default=1
        Number of parallel jobs for models (outer parallelization).
    force_parallel : bool, default=False
        Skip auto-disable threshold checks.
    subpopulation_labels : dict of {name: array}, optional
        Subpopulation labels aligned with ``X_test`` for per-subgroup
        bootstrap metrics.
    categorical_groups : dict of {str: list of str}, optional
        Mapping from an original categorical feature name to its encoded
        (e.g. one-hot) column names, so permutation importance permutes all
        encoded columns of a categorical feature together.
    ipcw_tau : float, optional
        Evaluation horizon τ for Uno's IPCW concordance, passed through to
        :func:`sksurv.metrics.concordance_index_ipcw`. When unset, a
        conservative default is auto-selected (see
        :func:`mlsurv.evaluation.metrics.compute_safe_ipcw_tau`).
    ipcw_y_train : SurvivalArray, optional
        Cohort whose censoring distribution fits Uno's IPCW ``G_hat``
        estimator; overrides ``y_train`` for that purpose when given.
    surv_probs_transforms : dict of {model: callable}, optional
        Per-model transforms ``f(surv_probs, eval_times) -> ndarray`` applied
        to the precomputed survival probabilities before iterations index into
        them, so CIs reflect the recalibrated S(t|x) used for the point
        estimate.
    surv_probs_overrides : dict of {model: (probs, eval_times)}, optional
        Per-model raw survival probabilities to reuse instead of re-predicting,
        each reused only when its ``eval_times`` matches the resolved bootstrap
        grid and its row count matches the test set (else the model is
        predicted as usual). Lets the recalibrated bootstrap replay reuse the
        raw S(t|x) ``evaluate()`` already computed, skipping a redundant
        full-test prediction. Any ``surv_probs_transforms`` entry still applies.

    Returns
    -------
    BootstrapResult
        Bootstrap results with summary and per-iteration data.

    Notes
    -----
    The model is NOT retrained during bootstrap. This estimates variance in
    the evaluation metric, not model generalization variance.

    Examples
    --------
    Single model:
    >>> result = bootstrap(model, X_test, y_test, y_train=y_train)
    >>> print(result.summary)

    Multiple models:
    >>> models = {'coxph': coxph_model, 'rsf': rsf_model}
    >>> result = bootstrap(models, X_test, y_test, y_train=y_train)
    >>> print(result.summary)  # Contains both models
    """
    # Handle dict of models - bootstrap each and aggregate
    if isinstance(model, dict):
        return _bootstrap_multiple(
            models=model,
            surv_probs_transforms=surv_probs_transforms,
            surv_probs_overrides=surv_probs_overrides,
            X_test=X_test,
            y_test=y_test,
            y_train=y_train,
            n_bootstrap=n_bootstrap,
            metrics=metrics,
            custom_metrics=custom_metrics,
            lower_is_better=lower_is_better,
            ci=ci,
            eval_times=eval_times,
            extra_eval_times=extra_eval_times,
            include_predictions=include_predictions,
            include_auc=include_auc,
            include_brier_t=include_brier_t,
            include_importance=include_importance,
            n_importance_repeats=n_importance_repeats,
            stratify=stratify,
            random_state=random_state,
            show_progress=show_progress,
            n_jobs=n_jobs,
            n_jobs_models=n_jobs_models,
            force_parallel=force_parallel,
            subpopulation_labels=subpopulation_labels,
            categorical_groups=categorical_groups,
            ipcw_tau=ipcw_tau,
            ipcw_y_train=ipcw_y_train,
            _warn_stacklevel=_warn_stacklevel + 1,
        )

    # Single model bootstrap
    _single_transform = (
        surv_probs_transforms.get(model_name)
        if surv_probs_transforms and model_name else None
    )
    _single_override = (
        surv_probs_overrides.get(model_name)
        if surv_probs_overrides and model_name else None
    )
    return _bootstrap_single(
        model=model,
        surv_probs_transform=_single_transform,
        surv_probs_override=_single_override,
        _failure_collector=_failure_collector,
        _notices_collector=_notices_collector,
        X_test=X_test,
        y_test=y_test,
        y_train=y_train,
        n_bootstrap=n_bootstrap,
        metrics=metrics,
        custom_metrics=custom_metrics,
        lower_is_better=lower_is_better,
        ci=ci,
        eval_times=eval_times,
        extra_eval_times=extra_eval_times,
        include_predictions=include_predictions,
        include_auc=include_auc,
        include_brier_t=include_brier_t,
        include_importance=include_importance,
        n_importance_repeats=n_importance_repeats,
        stratify=stratify,
        random_state=random_state,
        show_progress=show_progress,
        model_name=model_name,
        n_jobs=n_jobs,
        force_parallel=force_parallel,
        subpopulation_labels=subpopulation_labels,
        categorical_groups=categorical_groups,
        ipcw_tau=ipcw_tau,
        ipcw_y_train=ipcw_y_train,
        _warn_stacklevel=_warn_stacklevel,
    )


def _resolve_bootstrap_config(
    X_test, y_test, y_train, n_bootstrap, metrics, custom_metrics,
    lower_is_better, eval_times, random_state, show_progress, model_name, model,
    *, extra_eval_times=None,
):
    """Resolve bootstrap parameters into concrete values.

    Returns
    -------
    tuple of (n_bootstrap, metrics, custom_metrics, all_lower_is_better,
              eval_times, seeds, model_name)
    """
    n_samples = len(X_test)
    n_bootstrap = resolve_n_bootstrap(n_bootstrap, n_samples, show_progress)

    metrics = metrics if metrics is not None else ALL_METRICS
    custom_metrics = custom_metrics or {}

    all_lower_is_better = set(LOWER_IS_BETTER)
    if lower_is_better:
        all_lower_is_better.update(lower_is_better)
    if eval_times is None:
        # Top-level emits the IPCW truncation warning.
        # Degenerate inputs (e.g. <2 unique event times in
        # y_train) make IPCW grids impossible.  Fall back to None so
        # time-dependent metrics emit NaN instead of aborting bootstrap.
        try:
            eval_times = auto_eval_times(y_test, y_train=y_train, _quiet=True)
            if y_train is not None:
                eval_times = clip_eval_times(eval_times, y_train)
        except (ValueError, TypeError) as exc:
            _warn_eval_times_failure(
                exc, "in bootstrap()",
                "Time-dependent metrics will be NaN.",
            )
            eval_times = None
    elif y_train is not None:
        try:
            eval_times = clip_eval_times(eval_times, y_train)
        except (ValueError, TypeError) as exc:
            _warn_eval_times_failure(
                exc, "in bootstrap()",
                "Time-dependent metrics will be NaN.",
                verb="clip_eval_times",
            )
            eval_times = None

    # Union additive report horizons (add_eval_times) so the CI band lands on
    # the same grid as evaluate()'s point curve. In-window only; out-of-window
    # extras were warned at registration and render as honest NA.
    eval_times = union_extra_eval_times(eval_times, extra_eval_times, y_train)

    seeds = _generate_seeds(random_state, n_bootstrap)

    if model_name is None:
        model_name = resolve_model_name(model)

    return (n_bootstrap, metrics, custom_metrics, all_lower_is_better,
            eval_times, seeds, model_name)


def _finalize_bootstrap_result(
    iteration_data, predictions_data, auc_data, brier_t_data, importance_data,
    subpopulation_auc_data, model, X_test, y_test, y_train, metrics,
    custom_metrics, eval_times, precomputed_risk,
    precomputed_surv_probs, model_name, ci, all_lower_is_better,
    ipcw_tau: Optional[float] = None,
    ipcw_y_train: Optional[SurvivalArray] = None,
    precomputed_ibs_surv_probs=None, ibs_times=None,
):
    """Build DataFrames and BootstrapResult from aggregated iteration data."""
    iterations_df = pd.DataFrame(iteration_data)
    try:
        point_estimates = _compute_point_estimates(
            model, X_test, y_test, y_train, metrics, custom_metrics,
            eval_times, precomputed_risk, precomputed_surv_probs, model_name,
            ipcw_tau=ipcw_tau, ipcw_y_train=ipcw_y_train,
            precomputed_ibs_surv_probs=precomputed_ibs_surv_probs, ibs_times=ibs_times,
        )
    except (MemoryError, RecursionError):
        raise
    except Exception as e:
        log_exception(logger, e, f"Point estimate computation failed for {model_name}",
                      warn_user=True,
                      user_message=f"Point estimate computation failed: {e}. Bootstrap CIs are still valid.")
        point_estimates = None
    summary_df = _compute_summary_statistics(
        iterations_df, metrics, custom_metrics, model_name, ci,
        point_estimates=point_estimates,
    )

    return BootstrapResult(
        summary=summary_df,
        iterations=iterations_df,
        predictions=pd.DataFrame(predictions_data) if predictions_data else None,
        auc_t=pd.DataFrame(auc_data) if auc_data else None,
        brier_t=pd.DataFrame(brier_t_data) if brier_t_data else None,
        subpopulation_auc_t=pd.DataFrame(subpopulation_auc_data) if subpopulation_auc_data else None,
        importance=pd.DataFrame(importance_data) if importance_data else None,
        lower_is_better=all_lower_is_better,
        ci_level=ci,
    )


def _bootstrap_single(
    model,
    X_test,
    y_test: SurvivalArray,
    y_train: Optional[SurvivalArray] = None,
    n_bootstrap="auto",
    metrics=None,
    custom_metrics=None,
    lower_is_better=None,
    ci=0.95,
    eval_times=None,
    extra_eval_times=None,
    include_predictions=False,
    include_auc=False,
    include_brier_t=False,
    include_importance=False,
    n_importance_repeats=10,
    stratify=False,
    random_state=None,
    show_progress=True,
    model_name=None,
    n_jobs=1,
    force_parallel=False,
    subpopulation_labels=None,
    categorical_groups=None,
    ipcw_tau: Optional[float] = None,
    ipcw_y_train: Optional[SurvivalArray] = None,
    _warn_stacklevel=3,
    surv_probs_transform: Optional[Callable] = None,
    surv_probs_override: Optional[tuple] = None,
    _return_failures: bool = False,
    _failure_collector: Optional[dict] = None,
    _notices_collector: Optional[dict] = None,
):
    """Bootstrap evaluation for a single model.

    When ``_return_failures`` is True, both the per-model failure summary
    (:func:`_emit_failure_summary`) and the per-model low-coverage summary
    (:func:`_emit_low_coverage_summary`) are skipped, and the return value
    becomes ``(result, all_failures, all_notices)``. The caller is then
    responsible for emitting aggregated summaries (e.g. across multiple
    models in :func:`_bootstrap_multiple`).

    When ``_failure_collector`` is provided (a dict), failures are appended
    under ``model_name`` and the per-model summary is suppressed. Used by
    :func:`mlsurv.evaluation.evaluate._evaluate_multiple` to coalesce
    bootstrap failures across the per-model evaluate→bootstrap loop into
    a single :func:`_emit_multi_failure_summary` call. ``_notices_collector``
    is the analogous collector for low-coverage notices, feeding
    :func:`_emit_multi_low_coverage_summary`.
    """
    n_bootstrap, metrics, custom_metrics, all_lower_is_better, eval_times, \
        seeds, model_name = _resolve_bootstrap_config(
            X_test, y_test, y_train, n_bootstrap, metrics, custom_metrics,
            lower_is_better, eval_times, random_state, show_progress, model_name, model,
            extra_eval_times=extra_eval_times,
        )

    # Save original n_jobs to restore after bootstrap (avoid permanent mutation)
    saved_njobs = save_model_njobs(model)

    # Adjust model n_jobs to prevent CPU oversubscription
    model = adjust_model_njobs(model, n_jobs)

    try:
        # Get feature names for importance results
        feature_names = X_test.columns.tolist()

        # Build config for iteration workers
        config = BootstrapConfig(
            metrics=metrics,
            custom_metrics=custom_metrics,
            eval_times=eval_times,
            ci=ci,
            include_predictions=include_predictions,
            include_auc=include_auc,
            include_brier_t=include_brier_t,
            include_importance=include_importance,
            n_importance_repeats=n_importance_repeats,
            stratify=stratify,
            subpopulation_labels=(np.asarray(subpopulation_labels)
                             if subpopulation_labels is not None else None),
            categorical_groups=categorical_groups,
            ipcw_tau=ipcw_tau,
            ipcw_y_train=ipcw_y_train,
        )

        # Suppress repeated iterative warnings across bootstrap iterations —
        # at most one per unique message rather than one per iteration.
        with warnings.catch_warnings():
            warnings.filterwarnings("once", category=IterativeWarning)

            # Precompute predictions if beneficial. ``surv_probs_transform``
            # is applied to ``precomputed_surv_probs`` once
            # so iterations indexing into it AND _compute_point_estimates'
            # bootstrap-summary point all see recalibrated values.
            (precomputed_risk, precomputed_surv_probs,
             precomputed_ibs_surv_probs, ibs_times,
             can_precompute) = _precompute_predictions(
                model, X_test, metrics, eval_times, custom_metrics,
                include_importance, model_name,
                surv_probs_transform=surv_probs_transform,
                surv_probs_override=surv_probs_override,
                include_brier_t=include_brier_t,
                y_test=y_test, y_train=y_train,
            )

            # Determine effective parallelization
            effective_n_jobs = _resolve_effective_n_jobs(
                n_jobs, n_bootstrap, can_precompute, force_parallel, show_progress,
                model=model,
            )

            # Build shared iteration kwargs
            iter_kwargs = dict(
                model=model, X_test=X_test, y_test=y_test, y_train=y_train,
                config=config, model_name=model_name, feature_names=feature_names,
                precomputed_risk=precomputed_risk, precomputed_surv_probs=precomputed_surv_probs,
                precomputed_ibs_surv_probs=precomputed_ibs_surv_probs, ibs_times=ibs_times,
            )

            # Execute and aggregate
            # Threading for heavy models (RSF, XGBCox) with precomputed iterations:
            # avoids serializing large tree ensembles in iter_kwargs. Note: threading
            # provides no true parallelism for GIL-bound metric computation — the
            # benefit is reduced overhead, not parallel speedup. Lightweight models
            # (CoxPH, CoxNet) serialize cheaply and benefit from loky's process
            # isolation for CPU-bound work.
            use_threading = can_precompute and _has_internal_parallelism(model)
            backend = 'threading' if use_threading else 'loky'
            results = _execute_bootstrap_iterations(
                n_bootstrap, seeds, iter_kwargs, effective_n_jobs, show_progress, model_name,
                backend=backend,
            )
        (iteration_data, predictions_data, auc_data, brier_t_data,
         importance_data, subpopulation_auc_data, all_failures,
         all_notices) = _aggregate_iteration_results(
            results, config,
        )
        if _failure_collector is not None:
            if all_failures:
                _failure_collector[model_name] = all_failures
        elif not _return_failures:
            _emit_failure_summary(all_failures, n_bootstrap, model_name,
                                  _warn_stacklevel=_warn_stacklevel)
        # Coalesce per-iteration low-coverage notices into
        # MethodologicalWarnings, collapsing a metric's subpopulation levels
        # into one warning per metric.
        # When a collector is supplied (multi-model dispatch) or
        # notices are returned to the caller, defer emission so the caller
        # can cross-coalesce across models via _emit_multi_low_coverage_summary.
        if _notices_collector is not None:
            if all_notices:
                _notices_collector[model_name] = all_notices
        elif not _return_failures:
            _emit_low_coverage_summary(all_notices, n_bootstrap, model_name,
                                       _warn_stacklevel=_warn_stacklevel)

        result = _finalize_bootstrap_result(
            iteration_data, predictions_data, auc_data, brier_t_data, importance_data,
            subpopulation_auc_data, model, X_test, y_test, y_train, metrics,
            custom_metrics, eval_times, precomputed_risk,
            precomputed_surv_probs, model_name, ci, all_lower_is_better,
            ipcw_tau=ipcw_tau, ipcw_y_train=ipcw_y_train,
            precomputed_ibs_surv_probs=precomputed_ibs_surv_probs, ibs_times=ibs_times,
        )
        if _return_failures:
            return result, all_failures, all_notices
        return result
    finally:
        restore_model_njobs(model, saved_njobs)


def _emit_multi_failure_summary(model_failures, n_bootstrap, *,
                                _warn_stacklevel=2):
    """Emit a cross-model aggregated failure summary.

    ``model_failures`` is a mapping ``{model_name: failures}``, where each
    ``failures`` is the list collected by :func:`_bootstrap_single` (a list
    of ``(metric, exc_type, exc_msg)`` tuples). Failures with the same
    ``(metric, exc_type, exc_msg, count)`` across multiple models are
    collapsed into a single ``MetricFailureWarning`` that lists all affected
    models. This avoids N near-identical lines when shared bootstrap seeds
    hit the same edge-case resamples across every model.
    """
    if not model_failures:
        return

    # group_key -> {model_name: count}
    grouped: dict = {}
    examples: dict = {}
    for model_name, failures in model_failures.items():
        per_model_counts = Counter()
        per_model_examples = {}
        for metric, exc_type, exc_msg in failures:
            key = (metric, exc_type)
            per_model_counts[key] += 1
            per_model_examples.setdefault(key, exc_msg)
        for key, count in per_model_counts.items():
            grouped.setdefault(key, {})[model_name] = count
            examples.setdefault(key, per_model_examples[key])

    for (metric, exc_type), counts_by_model in grouped.items():
        unique_counts = set(counts_by_model.values())
        models = list(counts_by_model.keys())
        if len(unique_counts) == 1 and len(models) > 1:
            count = next(iter(unique_counts))
            model_list = ", ".join(models)
            msg = (
                f"{metric} failed in {count}/{n_bootstrap} iterations for "
                f"{len(models)} models ({model_list}) "
                f"({exc_type}: {examples[(metric, exc_type)]})"
            )
        else:
            # Mixed counts across models — list per-model counts.
            per_model = ", ".join(f"{m}={c}" for m, c in counts_by_model.items())
            total = sum(counts_by_model.values())
            msg = (
                f"{metric} failed in {total} iterations across "
                f"{len(models)} models [{per_model}] / {n_bootstrap} per model "
                f"({exc_type}: {examples[(metric, exc_type)]})"
            )
        # Every failed iteration drops user-visible bootstrap data
        # across the listed models, so warn unconditionally. The cross-model
        # coalescing above already collapses identical failures into one line,
        # and IterativeWarning's "once" filter handles dedup across calls.
        warnings.warn(msg, MetricFailureWarning, stacklevel=_warn_stacklevel)


def _format_coverage(min_valid, max_valid, n_total):
    """Format a coverage fraction, collapsing to ``x/total`` when min==max."""
    if min_valid == max_valid:
        return f"{min_valid}/{n_total}"
    return f"{min_valid}–{max_valid}/{n_total}"


def _render_multi_subpop_fragment(subpop, by_model, n_bootstrap):
    """Render one subpopulation level's coverage detail for a collapsed
    multi-model summary line.

    Reuses the cross-model coalescing decision: when every model shares the
    same coverage/count signature, a compact ``(coverage, iters, N models)``
    fragment is produced; otherwise a per-model breakdown is listed.
    """
    models = list(by_model.keys())
    signatures = {
        (b["min_valid"], b["max_valid"], b["count"]) for b in by_model.values()
    }
    n_total = next(iter(by_model.values()))["n_total"]
    if len(signatures) == 1 and len(models) > 1:
        min_valid, max_valid, count = next(iter(signatures))
        coverage = _format_coverage(min_valid, max_valid, n_total)
        return (
            f"{subpop} ({coverage}, {count}/{n_bootstrap} iters, "
            f"{len(models)} models)"
        )
    per_model = ", ".join(
        f"{m}={_format_coverage(b['min_valid'], b['max_valid'], b['n_total'])}"
        f" in {b['count']}"
        for m, b in by_model.items()
    )
    return f"{subpop} [{per_model}]"


def _build_multi_single_subpop_msg(metric, subpop, by_model, n_bootstrap):
    """Build the cross-model summary message for a single (metric, subpop)
    group — the original single-group format, preserved for cohort-level metrics and
    metrics with only one subpopulation level."""
    label = f" subpopulation '{subpop}'" if subpop is not None else ""
    models = list(by_model.keys())
    signatures = {
        (b["min_valid"], b["max_valid"], b["count"]) for b in by_model.values()
    }
    n_total = next(iter(by_model.values()))["n_total"]
    if len(signatures) == 1 and len(models) > 1:
        # Identical coverage and count across all models — one coalesced line.
        min_valid, max_valid, count = next(iter(signatures))
        coverage = _format_coverage(min_valid, max_valid, n_total)
        model_list = ", ".join(models)
        return (
            f"{metric}{label}: insufficient eval_times coverage "
            f"({coverage}) in {count}/{n_bootstrap} bootstrap iterations "
            f"for {len(models)} models ({model_list}). Reporting NaN for "
            f"affected rows. Pass eval_times within the cohort's observed "
            f"time window."
        )
    # Mixed coverage/counts across models — per-model breakdown.
    per_model = ", ".join(
        f"{m}={_format_coverage(b['min_valid'], b['max_valid'], b['n_total'])}"
        f" in {b['count']}"
        for m, b in by_model.items()
    )
    return (
        f"{metric}{label}: insufficient eval_times coverage across "
        f"{len(models)} models [{per_model}] / {n_bootstrap} bootstrap "
        f"iterations per model. Reporting NaN for affected rows. Pass "
        f"eval_times within the cohort's observed time window."
    )


def _emit_multi_low_coverage_summary(model_notices, n_bootstrap, *,
                                     _warn_stacklevel=2):
    """Emit a cross-model aggregated low-coverage summary, coalescing
    subpopulation levels.

    ``model_notices`` is a mapping ``{model_name: notices}``, where each
    ``notices`` is the list collected by :func:`_bootstrap_single` (a list
    of ``(metric, subpop, n_valid, n_total)`` tuples). Notices are grouped by
    ``metric``; within a metric, coverage is aggregated **across models** for
    each subpopulation level (the original cross-model coalescing).

    A metric spanning **two or more distinct subpopulation levels** collapses
    into one MethodologicalWarning listing every level; cohort-level
    metrics (``subpop=None``) and metrics with a single level keep the
    original one-line-per-group format. This avoids N near-identical lines
    when a many-subgroup cohort hits the same insufficient-coverage condition
    for each level.

    Like :func:`_emit_low_coverage_summary`, this always warns —
    insufficient eval_times coverage is a methodological concern the user
    must see per the "No Silent Data Loss" rule.
    """
    if not model_notices:
        return

    # metric -> OrderedDict[subpop -> OrderedDict[model -> bucket]]
    # (``n_total`` is tracked per bucket; readers use the per-bucket value.)
    grouped: "OrderedDict[str, OrderedDict]" = OrderedDict()
    for model_name, notices in model_notices.items():
        for metric, subpop, n_valid, n_total in notices:
            subpops = grouped.setdefault(metric, OrderedDict())
            per_model = subpops.setdefault(subpop, OrderedDict())
            bucket = per_model.get(model_name)
            if bucket is None:
                per_model[model_name] = {
                    "count": 1,
                    "min_valid": n_valid,
                    "max_valid": n_valid,
                    "n_total": n_total,
                }
            else:
                bucket["count"] += 1
                bucket["min_valid"] = min(bucket["min_valid"], n_valid)
                bucket["max_valid"] = max(bucket["max_valid"], n_valid)
                # ``n_total`` is invariant per metric site (always
                # ``len(eval_times)``); refresh defensively so a future change
                # that varies it per-iteration can't report a stale denominator.
                bucket["n_total"] = n_total

    for metric, subpops in grouped.items():
        # ``subpop is None`` is the cohort-level metric (``auc_t``); named
        # levels carry ``subpopulation_auc_t``. Use identity, not truthiness,
        # so an empty-string level is still treated as a named level.
        named = [(s, bm) for s, bm in subpops.items() if s is not None]
        cohort = [(s, bm) for s, bm in subpops.items() if s is None]
        if len(named) >= 2:
            # Collapse all subpopulation levels into a single line.
            frags = ", ".join(
                _render_multi_subpop_fragment(subpop, by_model, n_bootstrap)
                for subpop, by_model in named
            )
            msg = (
                f"{metric}: insufficient eval_times coverage across "
                f"{len(named)} subpopulation levels / {n_bootstrap} "
                f"bootstrap iterations per model. Reporting NaN for affected "
                f"rows ({frags}). Pass eval_times within the cohort's observed "
                f"time window."
            )
            warnings.warn(msg, MethodologicalWarning, stacklevel=_warn_stacklevel)
            # A cohort bucket should not share a metric with named levels
            # (different metric names), but emit it separately if present so
            # no recorded notice is ever dropped.
            for subpop, by_model in cohort:
                warnings.warn(
                    _build_multi_single_subpop_msg(
                        metric, subpop, by_model, n_bootstrap
                    ),
                    MethodologicalWarning, stacklevel=_warn_stacklevel,
                )
        else:
            # Cohort-level or single-level metric: original single-group format.
            for subpop, by_model in subpops.items():
                msg = _build_multi_single_subpop_msg(
                    metric, subpop, by_model, n_bootstrap
                )
                warnings.warn(
                    msg, MethodologicalWarning, stacklevel=_warn_stacklevel
                )


def _bootstrap_multiple(
    models,
    X_test,
    y_test: SurvivalArray,
    y_train: Optional[SurvivalArray] = None,
    n_bootstrap="auto",
    metrics=None,
    custom_metrics=None,
    lower_is_better=None,
    ci=0.95,
    eval_times=None,
    extra_eval_times=None,
    include_predictions=False,
    include_auc=False,
    include_brier_t=False,
    include_importance=False,
    n_importance_repeats=10,
    stratify=False,
    random_state=None,
    show_progress=True,
    n_jobs=1,
    n_jobs_models=1,
    force_parallel=False,
    subpopulation_labels=None,
    categorical_groups=None,
    ipcw_tau: Optional[float] = None,
    ipcw_y_train: Optional[SurvivalArray] = None,
    _warn_stacklevel=4,
    surv_probs_transforms: Optional[Dict[str, Callable]] = None,
    surv_probs_overrides: Optional[Dict[str, tuple]] = None,
):
    """Bootstrap evaluation for multiple models and aggregate results."""
    # Resolve n_bootstrap if "auto" (once for all models)
    n_samples = len(X_test)
    n_bootstrap = resolve_n_bootstrap(n_bootstrap, n_samples, show_progress)

    n_models = len(models)
    total_work = n_models * n_bootstrap

    # Auto-disable outer parallelization if below threshold (unless forced)
    effective_n_jobs_models = n_jobs_models
    if n_jobs_models != 1 and total_work < MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL and not force_parallel:
        effective_n_jobs_models = 1
        if show_progress:
            print(f"Note: Model parallelization disabled ({total_work} model-iterations < "
                  f"{MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL} threshold). Use force_parallel=True to override.")

    model_failures: dict = {}
    model_notices: dict = {}
    if effective_n_jobs_models == 1:
        # Sequential model execution
        results = []
        for name, model in models.items():
            boot_result, failures, notices = _bootstrap_single(
                model, X_test, y_test,
                y_train=y_train, n_bootstrap=n_bootstrap,
                metrics=metrics, custom_metrics=custom_metrics,
                lower_is_better=lower_is_better, ci=ci, eval_times=eval_times,
                extra_eval_times=extra_eval_times,
                include_predictions=include_predictions, include_auc=include_auc,
                include_brier_t=include_brier_t,
                include_importance=include_importance,
                n_importance_repeats=n_importance_repeats,
                stratify=stratify, random_state=random_state,
                show_progress=show_progress, model_name=name,
                n_jobs=n_jobs, force_parallel=force_parallel,
                subpopulation_labels=subpopulation_labels,
                categorical_groups=categorical_groups,
                ipcw_tau=ipcw_tau,
                ipcw_y_train=ipcw_y_train,
                _warn_stacklevel=_warn_stacklevel,
                surv_probs_transform=(
                    surv_probs_transforms.get(name)
                    if surv_probs_transforms else None
                ),
                surv_probs_override=(
                    surv_probs_overrides.get(name)
                    if surv_probs_overrides else None
                ),
                _return_failures=True,
            )
            results.append((name, boot_result))
            if failures:
                model_failures[name] = failures
            if notices:
                model_notices[name] = notices
    else:
        # Parallel model execution — no outer save/restore needed because:
        # (a) n_jobs=1 is passed to each _bootstrap_single, so adjust_model_njobs
        #     is a no-op inside each worker, and
        # (b) under the loky backend, models are serialized into isolated worker
        #     processes so internal mutations do not propagate back. Under the
        #     threading fallback, models share memory but n_jobs=1 inside each
        #     worker prevents concurrent mutation.
        def _bootstrap_model(name, model):
            boot_result, failures, notices = _bootstrap_single(
                model, X_test, y_test,
                y_train=y_train, n_bootstrap=n_bootstrap,
                metrics=metrics, custom_metrics=custom_metrics,
                lower_is_better=lower_is_better, ci=ci, eval_times=eval_times,
                extra_eval_times=extra_eval_times,
                include_predictions=include_predictions, include_auc=include_auc,
                include_brier_t=include_brier_t,
                include_importance=include_importance,
                n_importance_repeats=n_importance_repeats,
                stratify=stratify, random_state=random_state,
                show_progress=False, model_name=name,
                n_jobs=1, force_parallel=force_parallel,
                subpopulation_labels=subpopulation_labels,
                categorical_groups=categorical_groups,
                ipcw_tau=ipcw_tau,
                ipcw_y_train=ipcw_y_train,
                _warn_stacklevel=_warn_stacklevel,
                surv_probs_transform=(
                    surv_probs_transforms.get(name)
                    if surv_probs_transforms else None
                ),
                surv_probs_override=(
                    surv_probs_overrides.get(name)
                    if surv_probs_overrides else None
                ),
                _return_failures=True,
            )
            return (name, boot_result, failures, notices)

        if show_progress:
            print(f"Bootstrapping {len(models)} models in parallel...")

        backend = resolve_joblib_backend("loky")
        parallel_results = Parallel(
            n_jobs=effective_n_jobs_models, backend=backend,
        )(
            delayed(_bootstrap_model)(name, model)
            for name, model in models.items()
        )
        results = []
        for name, boot_result, failures, notices in parallel_results:
            results.append((name, boot_result))
            if failures:
                model_failures[name] = failures
            if notices:
                model_notices[name] = notices

    # ``_warn_stacklevel`` is calibrated for the per-model
    # ``_bootstrap_single`` → ``_emit_failure_summary`` path (free bootstrap
    # adds +1 for that path at the dict-dispatch site). Because deferred
    # per-model emission (``_return_failures=True``) surfaces failures
    # once here, the aggregate ``_emit_multi_*`` is called directly from
    # ``_bootstrap_multiple`` — one frame shallower than ``_bootstrap_single``
    # — so subtract that extra hop, otherwise the warning overshoots the
    # user's frame by one (landing in pytest / the caller's caller).
    _emit_multi_failure_summary(
        model_failures, n_bootstrap, _warn_stacklevel=_warn_stacklevel - 1,
    )
    _emit_multi_low_coverage_summary(
        model_notices, n_bootstrap, _warn_stacklevel=_warn_stacklevel - 1,
    )

    return _aggregate_multiple_bootstrap_results(results)


def _aggregate_multiple_bootstrap_results(results):
    """Aggregate bootstrap results from multiple models.

    Parameters
    ----------
    results : list of (name, BootstrapResult) tuples

    Returns
    -------
    BootstrapResult
    """
    summary = pd.concat([r.summary for _, r in results], ignore_index=True)
    iterations = pd.concat([r.iterations for _, r in results], ignore_index=True)

    predictions = _concat_optional_attr(results, 'predictions')
    auc_t = _concat_optional_attr(results, 'auc_t')
    brier_t = _concat_optional_attr(results, 'brier_t')
    subpopulation_auc_t = _concat_optional_attr(results, 'subpopulation_auc_t')
    importance = _concat_optional_attr(results, 'importance')

    all_lower_is_better = results[0][1].lower_is_better if results else set(LOWER_IS_BETTER)

    return BootstrapResult(
        summary=summary,
        iterations=iterations,
        predictions=predictions,
        auc_t=auc_t,
        brier_t=brier_t,
        subpopulation_auc_t=subpopulation_auc_t,
        importance=importance,
        lower_is_better=all_lower_is_better,
        ci_level=results[0][1].ci_level if results else 0.95,
    )


def _concat_optional_attr(results, attr):
    """Concatenate an optional DataFrame attribute across (name, result) tuples."""
    items = [getattr(r, attr) for _, r in results if getattr(r, attr, None) is not None]
    if items:
        return pd.concat(items, ignore_index=True)
    return None
