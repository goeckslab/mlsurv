"""CV fold execution, model comparison, and post-run evaluation helpers."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..evaluation import score_survival_model
from ..utils import adjust_model_njobs, resolve_joblib_backend, tqdm_joblib
from ..logging_utils import (
    get_logger,
    log_exception,
    log_metric_failure,
    _is_fold_boundary_error,
    _log_once_level,
    _fold_boundary_message,
)
from ..constants import IPCW_FAILURE_THRESHOLD
from .._warnings import (
    MethodologicalWarning,
    SuggestionWarning,
    _dedup_warning_entries,
    _reemit_deduped_warnings,
)

logger = get_logger("learner")

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner

# Local alias for the shared tqdm_joblib helper
_tqdm_joblib = tqdm_joblib

# Keywords in exception messages that indicate degenerate survival data in a fold.
# Kept as a module constant so it can be referenced from tests.
_DEGENERATE_KEYWORDS = ("identical", "degenerate", "t_min", "empty")


def _is_degenerate_error(exc):
    """Check if an exception indicates degenerate survival data."""
    return any(kw in str(exc).lower() for kw in _DEGENERATE_KEYWORDS)


def _handle_model_error(
    exc: BaseException, model_name: str, action: str, *, warn_user: bool = True,
) -> str:
    """Handle an exception from a model training/tuning attempt.

    Re-raises ``MemoryError`` and ``RecursionError``.  Logs
    ``ImportError`` with a user-facing install hint; logs all other
    exceptions with the *action* context (``"train"`` or ``"tune"``).

    Returns a concise ``"<ExcType>: <message>"`` detail string so the caller
    can fold it into the aggregate ``_raise_if_all_failed`` and
    ``_warn_dropped_models`` messages.

    When *warn_user* is ``False``, the failure is still logged (at ERROR) but
    no per-model ``warnings.warn`` is emitted. Multi-model loops use this so the
    single consolidated ``_warn_dropped_models`` summary is the only user-facing
    channel, instead of one scattered ``SuggestionWarning`` per dropped model.
    """
    if isinstance(exc, (MemoryError, RecursionError)):
        raise
    if not warn_user:
        log_exception(logger, exc, f"Failed to {action} {model_name}")
    elif isinstance(exc, ImportError):
        log_exception(
            logger, exc,
            f"Missing dependency for {model_name}",
            warn_user=True,
            user_message=(
                f"Missing dependency for {model_name}: {exc}. "
                f"Install the required package and retry."
            ),
            category=SuggestionWarning,
        )
    else:
        log_exception(
            logger, exc, f"Failed to {action} {model_name}",
            warn_user=True, category=SuggestionWarning,
        )
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ImportError):
        # Carry the install hint in the detail string so it survives in both
        # the all-failed RuntimeError and the consolidated drop warning, even
        # when the per-model SuggestionWarning is suppressed (warn_user=False).
        detail += " (install the missing dependency and retry)"
    return detail


def _warn_dropped_models(
    errors: dict, n_total: int, action: str, *, stacklevel: int = 3,
) -> None:
    """Emit one loud ``MethodologicalWarning`` when some models were dropped.

    Call this *after* ``_raise_if_all_failed`` so it is reached only on a
    partial failure (at least one model succeeded); the all-failed case raises
    instead. A no-op when *errors* is empty (a fully successful run).

    Dropping a model from a multi-model run is a methodological concern, not a
    mere suggestion: the comparison table and the best-model selection then
    reflect a silently reduced set of candidates. Unlike ``SuggestionWarning``,
    ``MethodologicalWarning`` is not registered for once-per-session dedup, so
    each distinct drop summary surfaces. The message lists every dropped model
    with its failure reason.
    """
    if not errors:
        return
    n_failed = len(errors)
    n_ok = n_total - n_failed
    detail = "; ".join(f"{name} ({err})" for name, err in errors.items())
    warnings.warn(
        f"{n_failed} of {n_total} models were dropped after failing to "
        f"{action}: {detail}. Results and the best-model selection reflect "
        f"only the {n_ok} remaining model(s).",
        MethodologicalWarning,
        stacklevel=stacklevel,
    )


def _raise_if_all_failed(
    results: dict, model_names, action: str, errors: dict | None = None,
) -> None:
    """Raise ``RuntimeError`` if no model succeeded.

    When *errors* (``{model_name: detail_string}``) is supplied, each model's
    underlying failure is listed so the message is self-explaining instead of
    pointing at "warnings above".
    """
    if not results:
        failed = ', '.join(model_names)
        msg = f"All models failed to {action}: {failed}."
        if errors:
            detail = '\n'.join(f"  - {name}: {err}" for name, err in errors.items())
            msg += f"\n{detail}"
        else:
            msg += " Check warnings above for details."
        raise RuntimeError(msg)


def _compute_ipcw_score(scoring_fn, pipeline, X_val, y_val, y_train, fold_idx):
    """Compute IPCW C-index for a single CV fold.

    Returns updated result dict entries for ipcw, ipcw_failed, ipcw_degenerate.
    """
    try:
        return {'ipcw': scoring_fn(pipeline, X_val, y_val, scoring='ipcw', y_train=y_train)}
    except (ValueError, ZeroDivisionError) as e:
        log_metric_failure(logger, "ipcw", e, fold=fold_idx)
        return {'ipcw': np.nan, 'ipcw_failed': True, 'ipcw_degenerate': _is_degenerate_error(e)}
    except Exception as e:
        if isinstance(e, (MemoryError, RecursionError)):
            raise
        log_metric_failure(logger, "ipcw", e, fold=fold_idx, warn_user=True)
        return {'ipcw': np.nan, 'ipcw_failed': True, 'ipcw_degenerate': _is_degenerate_error(e)}


def _compute_ibs_score(scoring_fn, pipeline, X_val, y_val, y_train, fold_idx):
    """Compute Integrated Brier Score for a single CV fold.

    Returns updated result dict entries for integrated_brier and ibs_degenerate.
    """
    try:
        ibs_score = scoring_fn(pipeline, X_val, y_val, scoring='integrated_brier', y_train=y_train)
        return {'integrated_brier': -ibs_score}  # Store as positive
    except (ValueError, ZeroDivisionError) as e:
        if _is_fold_boundary_error(e):
            log_metric_failure(
                logger, "ibs", e, fold=fold_idx,
                level=_log_once_level("fold_boundary"),
                warn_once_key="fold_boundary_warn",
                user_message=_fold_boundary_message("integrated_brier"),
            )
        else:
            log_metric_failure(logger, "ibs", e, fold=fold_idx, level=logging.ERROR)
        return {'ibs_degenerate': _is_degenerate_error(e)}
    except Exception as e:
        if isinstance(e, (MemoryError, RecursionError)):
            raise
        log_metric_failure(logger, "ibs", e, fold=fold_idx, warn_user=True)
        return {'ibs_degenerate': _is_degenerate_error(e)}


def _compute_ctd_score(scoring_fn, pipeline, X_val, y_val, y_train, fold_idx):
    """Compute Antolini's C^td for a single CV fold.

    Returns updated result dict entry for ctd.
    """
    try:
        ctd_score = scoring_fn(pipeline, X_val, y_val, scoring='ctd', y_train=y_train)
        return {'ctd': ctd_score}
    except (ValueError, ZeroDivisionError) as e:
        log_metric_failure(logger, "ctd", e, fold=fold_idx)
        return {}
    except Exception as e:
        if isinstance(e, (MemoryError, RecursionError)):
            raise
        log_metric_failure(logger, "ctd", e, fold=fold_idx, warn_user=True)
        return {}


def _train_cv_fold(
    fold_idx,
    train_idx,
    val_idx,
    X_train,
    y_train,
    pipeline,
    effective_scoring,
    score_survival_model_fn,
    outer_n_jobs=1,
):
    """
    Train and evaluate a single CV fold - worker function for parallelization.

    Parameters
    ----------
    fold_idx : int
        Fold index (for tracking).
    train_idx : array
        Training indices for this fold.
    val_idx : array
        Validation indices for this fold.
    X_train : DataFrame
        Full training features.
    y_train : SurvivalArray
        Full training outcomes.
    pipeline : Pipeline
        sklearn Pipeline to fit (will be cloned).
    effective_scoring : str
        Primary scoring metric.
    score_survival_model_fn : callable
        Function to compute scores.
    outer_n_jobs : int, default=1
        Number of jobs in the outer parallel loop. Used to prevent
        CPU oversubscription by setting model n_jobs=1 when outer_n_jobs > 1.

    Returns
    -------
    dict
        Results for this fold with keys:
        - 'score': primary metric score
        - 'harrell': Harrell C-index
        - 'ipcw': IPCW C-index (or np.nan if failed)
        - 'ipcw_failed': bool indicating IPCW failure
        - 'ipcw_degenerate': bool, True if IPCW failure was due to degenerate data
          (only present on IPCW failure)
        - 'integrated_brier': IBS (or None if not computed)
        - 'ibs_degenerate': bool, True if IBS failure was due to degenerate
          data (only present on IBS failure)
        - 'exception': exception message if fold failed, None otherwise
        - 'warnings': list of (category_name, message, filename, lineno) tuples
          captured during fold execution. category_name is a string
          (e.g., 'UserWarning') for picklability across joblib workers.
    """
    X_train_cv = X_train.iloc[train_idx]
    X_val_cv = X_train.iloc[val_idx]
    y_train_cv = y_train[train_idx]
    y_val_cv = y_train[val_idx]

    result = {
        'score': np.nan,
        'harrell': np.nan,
        'ipcw': np.nan,
        'ipcw_failed': False,
        'integrated_brier': None,
        'ctd': None,
        'exception': None,
        'warnings': [],
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Deduplicate iterative warnings (ClampWarning, FitFailureWarning, etc.)
        # so each unique message fires at most once per fold.
        from .._warnings import IterativeWarning
        warnings.filterwarnings("once", category=IterativeWarning)
        # Suppress sklearn convergence warnings per fold — the user sees the
        # final model's convergence status, not per-fold noise.
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # Suppress XGBoost reproducibility warning that clone() re-triggers
        # (it calls __init__ with the original params).  The user already saw
        # this warning at model construction time; repeating it per fold is noise.
        warnings.filterwarnings("ignore", message=r"^XGBoost with n_jobs != 1", category=UserWarning)
        # sksurv.metrics uses np.trapz which is deprecated in NumPy 2.x
        warnings.filterwarnings("ignore", message=r"`trapz` is deprecated", category=DeprecationWarning)
        # CoxNet "all coefficients are zero" is expected during CV when
        # regularization is too strong for a fold.  Suppress here; the final
        # trained model is checked separately in _training.py.
        warnings.filterwarnings("ignore", message=r"all coefficients are zero", category=UserWarning)
        # lifelines emits pandas-4.0 keyword-only deprecation warnings from
        # internal var/mean/std calls during fit.  See ``mlsurv/__init__.py``
        # for the version-portability rationale (Pandas4Warning vs
        # UserWarning); ``simplefilter('always')`` above wipes that
        # import-time filter, so re-add it here for the worker.
        warnings.filterwarnings(
            "ignore",
            message=r"Starting with pandas version 4\.0",
            category=Warning,
            module=r"lifelines\..*",
        )

        try:
            # Clone the pipeline so each fold gets a fresh copy
            fold_pipeline = clone(pipeline)

            # Adjust model n_jobs AFTER cloning to prevent CPU oversubscription
            # This must be done after cloning because sklearn's clone() may reset n_jobs
            fold_pipeline = adjust_model_njobs(fold_pipeline, outer_n_jobs)

            fold_pipeline.fit(X_train_cv, y_train_cv)

            # Compute primary scoring metric
            result['score'] = score_survival_model_fn(
                fold_pipeline, X_val_cv, y_val_cv,
                scoring=effective_scoring,
                y_train=y_train_cv,
            )

            # Compute harrell C-index
            result['harrell'] = score_survival_model_fn(
                fold_pipeline, X_val_cv, y_val_cv,
                scoring='harrell',
                y_train=y_train_cv,
            )

            # Compute ipcw C-index (may fail)
            result.update(_compute_ipcw_score(
                score_survival_model_fn, fold_pipeline, X_val_cv, y_val_cv,
                y_train_cv, fold_idx,
            ))

            # Try to compute IBS and C^td if model supports survival functions
            if hasattr(fold_pipeline, 'predict_survival_function'):
                result.update(_compute_ibs_score(
                    score_survival_model_fn, fold_pipeline, X_val_cv, y_val_cv,
                    y_train_cv, fold_idx,
                ))
                result.update(_compute_ctd_score(
                    score_survival_model_fn, fold_pipeline, X_val_cv, y_val_cv,
                    y_train_cv, fold_idx,
                ))

        except Exception as e:
            if isinstance(e, (MemoryError, RecursionError)):
                raise
            log_exception(logger, e, f"CV fold {fold_idx} failed")
            result['exception'] = f"{type(e).__name__}: {e}"

    # Serialize captured warnings to picklable tuples.
    # WarningMessage objects are not designed for pickling (they hold file
    # handles, source references, etc.), so we extract the essential fields
    # into plain tuples that survive joblib serialization reliably.
    result['warnings'] = [
        (w.category.__name__, str(w.message), w.filename, w.lineno)
        for w in caught
    ]

    return result


def _aggregate_fold_scores(fold_results):
    """Loop over folds, collect scores/metrics/failure counts.

    Returns
    -------
    tuple
        (scores, cv_metrics, ipcw_failed_folds, degenerate_folds)
    """
    scores = []
    cv_metrics = {'harrell': [], 'ipcw': []}
    ipcw_failed_folds = 0
    degenerate_folds = 0

    for result in fold_results:
        if result['exception']:
            warnings.warn(f"CV fold failed: {result['exception']}", MethodologicalWarning)
            scores.append(np.nan)
            cv_metrics['harrell'].append(np.nan)
            cv_metrics['ipcw'].append(np.nan)
            # Append NaN to optional metrics already initialized by earlier folds
            if 'integrated_brier' in cv_metrics:
                cv_metrics['integrated_brier'].append(np.nan)
            if 'ctd' in cv_metrics:
                cv_metrics['ctd'].append(np.nan)
            continue

        scores.append(result['score'])
        cv_metrics['harrell'].append(result['harrell'])
        cv_metrics['ipcw'].append(result['ipcw'])
        if result['ipcw_failed']:
            ipcw_failed_folds += 1
        if result.get('ipcw_degenerate') or result.get('ibs_degenerate'):
            degenerate_folds += 1

        # Optional metrics: append value or NaN (never skip) to keep lengths aligned
        ibs_val = result.get('integrated_brier')
        if ibs_val is not None:
            cv_metrics.setdefault('integrated_brier', []).append(ibs_val)
        elif 'integrated_brier' in cv_metrics:
            cv_metrics['integrated_brier'].append(np.nan)

        ctd_val = result.get('ctd')
        if ctd_val is not None:
            cv_metrics.setdefault('ctd', []).append(ctd_val)
        elif 'ctd' in cv_metrics:
            cv_metrics['ctd'].append(np.nan)

    # Backfill: if an optional metric key was first created on fold N > 0,
    # earlier folds have no entry.  Left-pad with NaN to align all lengths.
    n_total = len(cv_metrics['harrell'])
    for key in ('integrated_brier', 'ctd'):
        if key in cv_metrics:
            deficit = n_total - len(cv_metrics[key])
            if deficit > 0:
                logger.debug(
                    "Backfilling %d NaN entries for metric '%s' "
                    "(first appeared after fold 0)",
                    deficit, key,
                )
                cv_metrics[key] = [np.nan] * deficit + cv_metrics[key]

    # Postcondition: all metric arrays must have the same length
    for key, values in cv_metrics.items():
        if len(values) != n_total:
            logger.error(
                "BUG: cv_metrics['%s'] has %d entries but expected %d "
                "after backfill", key, len(values), n_total,
            )

    return scores, cv_metrics, ipcw_failed_folds, degenerate_folds


def _deduplicate_worker_warnings(fold_results):
    """Collect, dedup, and count warnings from fold results.

    Returns
    -------
    dict
        Mapping of (category, message) -> (count, filename, lineno).
    """
    return _dedup_warning_entries(
        entry for result in fold_results for entry in result.get('warnings', [])
    )


def _replay_captured_warnings(captured, seen_messages=None, dedup_category=None):
    """Replay captured warnings verbatim, optionally squelching repeats.

    For each captured warning whose category is a subclass of
    ``dedup_category`` and whose ``(category_name, message)`` is already in
    ``seen_messages``, the warning is dropped; otherwise the key is added
    to ``seen_messages`` and the warning is re-emitted via
    ``warn_explicit`` so its original class, file, and line are preserved.
    Warnings outside ``dedup_category`` always pass through verbatim.
    """
    for w in captured:
        msg_text = str(w.message)
        if (
            dedup_category is not None
            and seen_messages is not None
            and issubclass(w.category, dedup_category)
        ):
            key = (w.category.__name__, msg_text)
            if key in seen_messages:
                continue
            seen_messages.add(key)
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def _emit_cv_warnings(ipcw_failed_folds, degenerate_folds, actual_folds,
                      seen_warnings, seen_messages=None):
    """Emit degenerate, IPCW, and worker warnings.

    Parameters
    ----------
    ipcw_failed_folds : int
    degenerate_folds : int
    actual_folds : int
    seen_warnings : dict
        From ``_deduplicate_worker_warnings``.
    seen_messages : set, optional
        Cross-call dedup set (see ``_reemit_deduped_warnings``).
    """
    # Warn about degenerate data in folds
    if degenerate_folds > 0:
        warnings.warn(
            f"Degenerate survival data detected in {degenerate_folds}/{actual_folds} "
            f"CV fold(s) (identical times, empty arrays, etc.). "
            f"IPCW/IBS metrics may be unreliable.",
            MethodologicalWarning,
        )

    # Warn about IPCW failures if too many folds fail
    if actual_folds > 0 and ipcw_failed_folds / actual_folds > IPCW_FAILURE_THRESHOLD:
        successful = actual_folds - ipcw_failed_folds
        warnings.warn(
            f"IPCW metric unavailable in {ipcw_failed_folds}/{actual_folds} CV folds "
            f"(insufficient censored observations). IPCW computed from {successful}/{actual_folds} folds.",
            MethodologicalWarning
        )

    # Re-emit deduplicated warnings captured in worker processes.
    _reemit_deduped_warnings(seen_warnings, actual_folds, scope_label="CV folds",
                             seen_messages=seen_messages)


class CvTrainingMixin:
    """Mixin for CV fold execution, model comparison, and post-run evaluation."""

    def _get_cv_folds(self: SurvivalLearner):
        """Get CV folds as a list, handling GroupKFold groups parameter."""
        stratify_for_cv = (
            self._state.stratify_array if self._state.stratify_array is not None
            else self.y_train['event']
        )
        if isinstance(self._state.cv_splitter, GroupKFold):
            return list(self._state.cv_splitter.split(self.X_train, stratify_for_cv, self._state.groups))
        return list(self._state.cv_splitter.split(self.X_train, stratify_for_cv))

    def _run_cv_folds(self: SurvivalLearner, folds, pipeline, effective_scoring,
                      cv_n_jobs, model_name):
        """Execute CV folds sequentially or in parallel.

        Parameters
        ----------
        folds : list of (train_idx, val_idx) tuples
        pipeline : Pipeline to fit (will be cloned per fold)
        effective_scoring : str, scoring metric
        cv_n_jobs : int, number of parallel jobs
        model_name : str, for progress bar label

        Returns
        -------
        list of dict, one per fold
        """
        fold_args = [
            dict(
                fold_idx=fold_idx,
                train_idx=train_idx,
                val_idx=val_idx,
                X_train=self.X_train,
                y_train=self.y_train,
                pipeline=pipeline,
                effective_scoring=effective_scoring,
                score_survival_model_fn=score_survival_model,
                outer_n_jobs=cv_n_jobs,
            )
            for fold_idx, (train_idx, val_idx) in enumerate(folds)
        ]

        if cv_n_jobs == 1:
            return [_train_cv_fold(**args) for args in fold_args]

        # Parallel execution with joblib and progress bar
        jobs = (delayed(_train_cv_fold)(**args) for args in fold_args)
        backend = resolve_joblib_backend("loky")
        with _tqdm_joblib(tqdm(
            total=len(folds), desc=f"CV {model_name}", unit="fold", leave=False,
        )):
            return Parallel(n_jobs=cv_n_jobs, backend=backend)(jobs)

    def _aggregate_fold_results(self: SurvivalLearner, fold_results, actual_folds):
        """Aggregate per-fold CV results into scores and metrics dicts.

        Parameters
        ----------
        fold_results : list of dict
        actual_folds : int, total number of folds

        Returns
        -------
        tuple of (scores, cv_metrics)
            scores: list of primary metric scores
            cv_metrics: dict of {metric_name: list of scores}

        Notes
        -----
        Re-emits deduplicated warnings captured in worker processes.
        Warnings that appeared in multiple folds are consolidated into a
        single ``UserWarning`` with fold-count annotation.  Also emits
        warnings for degenerate fold data and IPCW metric failures.
        """
        scores, cv_metrics, ipcw_failed_folds, degenerate_folds = (
            _aggregate_fold_scores(fold_results)
        )
        seen_warnings = _deduplicate_worker_warnings(fold_results)
        _emit_cv_warnings(
            ipcw_failed_folds, degenerate_folds, actual_folds, seen_warnings,
            seen_messages=self._state.fit_failure_warned,
        )
        return scores, cv_metrics

    def _run_model_comparison(
        self: SurvivalLearner, models, tune, n_trials, search_spaces, experiment_name,
        optuna_verbose, model_params=None, resume=False,
        preprocessing=None, show_progress_bar=True,
    ):
        """Train/tune all models and return results dict.

        Parameters
        ----------
        model_params : dict, optional
            Per-model constructor kwargs: {model_name: {param: value}}.
            Only used when tune=False.
        preprocessing : dict, optional
            Resolved per-model preprocessing overrides from run().

        Returns
        -------
        dict of {str: ModelResult}
        """
        from ._training import (
            _resolve_per_model, _resolve_model_search_spaces,
            _warn_unused_model_params, _check_search_space_model_params,
        )
        from .._shared import _warn_unused_dict_keys

        action = "tune" if tune else "train"
        model_results = {}
        errors = {}
        resolved_search_spaces = (
            _resolve_model_search_spaces(search_spaces, models) if tune else {}
        )
        for model_name in models:
            try:
                if tune:
                    model_search_spaces = resolved_search_spaces[model_name]
                    model_n_trials = _resolve_per_model(n_trials, model_name, default='auto')
                    result = self._tune_and_train(
                        model_name, model_n_trials, model_search_spaces, experiment_name,
                        optuna_verbose=optuna_verbose,
                        resume=resume,
                        preprocessing=preprocessing,
                        show_progress_bar=show_progress_bar,
                    )
                else:
                    params = model_params.get(model_name, {}) if model_params else {}
                    _check_search_space_model_params(model_name, params)
                    result = self._train_model_internal(
                        model_name, _preprocessing=preprocessing, **params,
                    )
                model_results[model_name] = result
                # The tune=False and fallback paths already stored this
                # object (with the overwrite warning) via
                # _finalize_model_result, so `existing is result` there. The
                # successful-tuning path builds with _store=False, so this
                # assignment is its FIRST storage — warn here before replacing
                # an existing model. Replacing the stored ModelResult
                # drops the old one and everything riding on it.
                existing = self._state.models.get(model_name)
                if existing is not None and existing is not result:
                    self._warn_overwrite(model_name)
                self._state.models[model_name] = result
            except Exception as e:
                errors[model_name] = _handle_model_error(
                    e, model_name, action, warn_user=False,
                )

        _warn_unused_model_params(model_params, models)
        _warn_unused_dict_keys(n_trials, "n_trials", models)
        _warn_unused_dict_keys(preprocessing, "preprocessing", models)

        _raise_if_all_failed(model_results, models, action, errors=errors)
        _warn_dropped_models(errors, len(models), action)

        return model_results

    def _post_run_evaluation(
        self: SurvivalLearner, model_results, eval_times, n_importance_repeats, include_shap,
        shap_max_background, shap_max_explain,
        show_eval_progress, verbose, bootstrap, n_bootstrap, bootstrap_metrics,
        include_interactions=False, interaction_top_k=5, interaction_max_samples=100,
        interaction_method='hstat',
    ):
        """Evaluate all models on the test set.

        Per-model evaluations are stored on ModelResult by evaluate() via
        _store_per_model_evaluations().
        """
        self.evaluate(
            model_results,
            eval_times=eval_times,
            n_importance_repeats=n_importance_repeats,
            include_shap=include_shap,
            shap_max_background=shap_max_background,
            shap_max_explain=shap_max_explain,
            show_progress=show_eval_progress,
            verbose=verbose,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            bootstrap_metrics=bootstrap_metrics,
            include_predictions=bootstrap,
            include_interactions=include_interactions,
            interaction_top_k=interaction_top_k,
            interaction_max_samples=interaction_max_samples,
            interaction_method=interaction_method,
        )
        self._check_cv_vs_holdout_best(self._state.models, verbose)
