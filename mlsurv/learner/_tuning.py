"""TuningMixin - Hyperparameter tuning with Optuna."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace as _replace

import numpy as np
from typing import TYPE_CHECKING, Optional, Union

from sklearn.base import clone

from ..preprocessing.utils import extract_component_params, get_or_create_selector
from ..tuning import (
    get_tuner, auto_n_trials, NoTunerError,
    CustomModelTuner,
)
from ..tuning.base import _default_experiment_name
from ._format import (
    _print_param_box,
    _format_trials_display,
    _format_scoring_display,
    _format_search_display,
    _format_storage_display,
)
from ..results.printing import print_multi_model_cv_table, print_multi_model_best_params, resolve_display_name
from ..models import get_model_display_names
from ..models.base import BaseSurvivalModel
from ..logging_utils import get_logger
from ._cv_training import (
    _handle_model_error, _raise_if_all_failed, _warn_dropped_models,
)
from ._training import _resolve_per_model
from .._shared import _warn_unused_dict_keys
from ..constants import EPV_THRESHOLD_WARNING, EPV_THRESHOLD_INFO, MIN_SAMPLE_SIZE_WARNING
from .._warnings import InternalWarning, MethodologicalWarning, SuggestionWarning

logger = get_logger("learner")

# Sentinel for "not provided" preprocessing components (distinct from None = "skip")
_TUNER_NOT_SET = object()

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


@dataclass(frozen=True)
class TuneConfig:
    """Bundled config for a single tune() invocation (internal use)."""
    n_trials: Union[int, str, dict] = 'auto'
    timeout: Optional[float] = None
    storage: Optional[str] = None
    study_name: Optional[str] = None
    experiment_name: Optional[str] = None
    force: bool = False
    search_spaces: Optional[dict] = None
    resume: bool = False
    scoring: Optional[str] = None
    n_jobs: Optional[int] = None
    optuna_verbose: bool = False
    show_progress_bar: bool = True
    evaluate: bool = False
    eval_times: Optional[object] = None
    n_importance_repeats: Union[int, str] = "auto"
    importance_max_samples: Optional[int] = None
    include_shap: bool = False
    shap_max_background: int = 50
    shap_max_explain: int = 50
    show_eval_progress: bool = True
    bootstrap: bool = False
    n_bootstrap: Union[int, str] = "auto"
    bootstrap_metrics: Optional[list] = None
    include_interactions: bool = False
    interaction_top_k: int = 5
    interaction_max_samples: int = 100
    interaction_method: str = 'hstat'
    interaction_scale: str = 'auto'


def _validate_single_n_trials(value, label="n_trials", extra_types=""):
    """Validate a single n_trials value (int >= 0 or 'auto')."""
    if isinstance(value, bool):
        raise TypeError(
            f"{label} must be an int or 'auto', got {type(value).__name__}"
        )
    if value != 'auto' and (not isinstance(value, int) or value < 0):
        raise ValueError(
            f"{label} must be a non-negative integer"
            f", 'auto'{extra_types}, got {value!r}"
        )


def _validate_n_trials(n_trials, model_names=None):
    """Validate the n_trials parameter (dict, scalar, or 'auto')."""
    if isinstance(n_trials, dict):
        for model_key, val in n_trials.items():
            _validate_single_n_trials(
                val, label=f"n_trials[{model_key!r}]", extra_types=""
            )
    else:
        _validate_single_n_trials(
            n_trials, label="n_trials", extra_types=", or dict"
        )


def _validate_timeout(timeout):
    """Validate the timeout parameter."""
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ValueError(
            f"timeout must be a positive number (seconds), got {timeout!r}"
        )


def _resolve_search_space_compat(search_space, search_spaces):
    """Resolve backward-compat search_space alias to search_spaces."""
    if search_space is not None and search_spaces is None:
        return search_space
    return search_spaces


def _resolve_model_name(model):
    """Resolve model argument to a model name string.

    Parameters
    ----------
    model : str, ModelResult, or other
        The model specification.

    Returns
    -------
    str
        The resolved model name.

    Raises
    ------
    ValueError
        If model is not a string, ModelResult, or None.
    """
    if isinstance(model, str):
        return model
    if hasattr(model, 'name'):
        return model.name
    raise ValueError("model must be a string, ModelResult, or None")


def _translate_deep_learning_params(model_params, model_name):
    """Translate DeepSurv/DeepHit tuner params to model constructor params.

    Converts n_layers/n_nodes to layers list and lr to learning_rate.
    Mutates model_params in place.
    """
    if model_name not in ('deepsurv', 'deephit'):
        return
    n_layers = model_params.pop('n_layers', 2)
    n_nodes = model_params.pop('n_nodes', 64)
    model_params['layers'] = [n_nodes] * n_layers
    if 'lr' in model_params:
        model_params['learning_rate'] = model_params.pop('lr')


def _translate_coxnet_params(model_params):
    """Translate CoxNet alpha_index to alpha (integer path index).

    Mutates model_params in place.
    """
    if 'alpha_index' in model_params:
        model_params['alpha'] = int(model_params.pop('alpha_index'))


def _resolve_n_trials_for_model(config, model_name, n_features):
    """Resolve n_trials for a specific model, applying auto logic if needed.

    Parameters
    ----------
    config : TuneConfig
    model_name : str
    n_features : int

    Returns
    -------
    int
        Resolved number of trials.
    """
    n_trials = _resolve_per_model(config.n_trials, model_name, default='auto')
    if n_trials == 'auto':
        n_trials = auto_n_trials(model_name, n_features)
    return n_trials


class TuningMixin:
    """Mixin for hyperparameter tuning methods."""

    def _warn_epv_if_low(self: SurvivalLearner, stacklevel=2):
        """Warn if EPV is low for tuning.

        Skips the MethodologicalWarning if setup() already warned about
        low EPV, but always emits the info log about search space
        constraints since that is tuning-specific context.

        Parameters
        ----------
        stacklevel : int, default=2
            Warning stacklevel to attribute warnings to the caller.
        """
        n_events_train = int(self.y_train['event'].sum())
        n_features = len(self.X_train.columns)
        epv = n_events_train / n_features if n_features > 0 else float('inf')

        if epv < EPV_THRESHOLD_WARNING:
            if not self._state.epv_warned_at_setup:
                warnings.warn(
                    f"EPV={epv:.1f} is very low ({n_events_train} events, "
                    f"{n_features} features). Hyperparameter tuning is unlikely "
                    f"to generalize — CV scores will overestimate test "
                    f"performance. Consider using default hyperparameters "
                    f"(tune=False) or reducing features first.",
                    MethodologicalWarning,
                    stacklevel=stacklevel,
                )
            else:
                logger.debug(
                    "EPV warning skipped — already warned during setup()"
                )
        elif epv < EPV_THRESHOLD_INFO:
            logger.info(
                "EPV=%.1f is low — search spaces may be automatically "
                "constrained to reduce overfitting risk.", epv
            )

    def _warn_small_sample_tuning(self: SurvivalLearner, stacklevel=2):
        """Warn if training set is too small for stable tuning."""
        n_train = len(self.y_train)
        if n_train < MIN_SAMPLE_SIZE_WARNING:
            warnings.warn(
                f"Small training set (n={n_train}) may produce unstable tuning "
                f"results. CV scores on small folds are noisy — Optuna may "
                f"select hyperparameters that exploit fold-specific noise. "
                f"Consider using fixed hyperparameters (tune=False) or "
                f"increasing cv_repeats for more stable estimates.",
                MethodologicalWarning,
                stacklevel=stacklevel,
            )

    @staticmethod
    def _validate_tune_inputs(n_trials, timeout, search_space, search_spaces):
        """Validate tune() inputs and resolve backward-compat alias.

        Returns resolved search_spaces.
        """
        _validate_n_trials(n_trials)
        _validate_timeout(timeout)
        return _resolve_search_space_compat(search_space, search_spaces)

    def _resolve_tune_models_for_display(self: SurvivalLearner, model):
        """Resolve model argument to a list of names for the summary display.

        Filter mirrors _tune_all_models.trainable_models — keep in sync.
        """
        return self._resolve_model_names(
            model,
            filter_fn=lambda k, v: not v.is_tuned,
        )

    def tune(
        self: SurvivalLearner,
        models=None,
        n_trials='auto',
        timeout=None,
        storage=None,
        study_name=None,
        experiment_name=None,
        force=False,
        search_spaces=None,
        resume=False,
        scoring=None,
        search_space=None,  # Backward compat alias
        verbose=True,
        n_jobs=None,
        optuna_verbose=False,
        show_progress_bar=True,
        # Evaluation parameters
        evaluate=False,
        eval_times=None,
        n_importance_repeats="auto",
        importance_max_samples=None,
        include_shap=False,
        shap_max_background=50,
        shap_max_explain=50,
        show_eval_progress=True,
        # Bootstrap parameters
        bootstrap=False,
        n_bootstrap="auto",
        bootstrap_metrics=None,
        # Feature interactions
        include_interactions=False,
        interaction_top_k=5,
        interaction_max_samples=100,
        interaction_method='hstat',
        interaction_scale='auto',
    ):
        """
        Tune hyperparameters for model(s) using Optuna.

        The tuned model is stored under its plain algorithm name (e.g.
        ``'coxnet'``), overwriting any existing untuned model. To compare
        tuned against untuned, build a second learner with the same
        ``setup(random_state=...)`` (identical split, folds, and
        preprocessing) and compare on cross-validation.

        Parameters
        ----------
        models : str, ModelResult, BaseSurvivalModel, or None, default=None
            Model(s) to tune.
            - If None: tunes all trained models in _state.models
            - If str: tunes single model by name
            - If ModelResult: tunes the model from that result
            - If BaseSurvivalModel: tunes a custom instance (clones it per
              trial). Requires a search space via the instance's
              ``get_search_space()`` hook or a per-call ``search_spaces=``.
            Returns dict of {name: ModelResult} if tuning multiple models.
        n_trials : int, str, or dict, default='auto'
            Number of Optuna trials.
            - 'auto': convergence-based stopping (default).
            - 0: skip tuning and use default hyperparameters. With
              ``resume=True``, loads from an existing Optuna study instead.
            - dict: per-model trial counts, e.g. {'rsf': 50, 'gbsa': 'auto'}.
        timeout : float, optional
            Maximum tuning time in seconds.
        storage : str, optional
            Optuna storage URL (e.g., 'postgresql://user:pass@host/db').
            If not provided, uses SQLite at ./optuna_studies/{experiment_name}/{model}_study.db
        study_name : str, optional
            Optuna study name. Defaults to '{experiment_name}/{model}_tuning'.
        experiment_name : str, optional
            Experiment name for this study. Overrides the experiment_name set in setup().
            If neither is set, auto-generates a timestamped name.
        force : bool, default=False
            If True, re-tune even if already tuned.
        search_spaces : dict, optional
            Search spaces for model and/or pipeline components. Supports tuning
            any component's parameters with prefixed names in best_params_.
            Format: {
                'model': {'n_estimators': (100, 500), 'max_depth': (5, 15)},
                'feature_selector': {'k': (5, 20)},
                'imputer': {'n_neighbors': (3, 10)},
                'scaler': {}  # Omit or empty = no tuning
            }
            If only model params are needed, can pass flat dict for backward
            compatibility: {'n_estimators': (100, 500)} is equivalent to
            {'model': {'n_estimators': (100, 500)}}
            This is the single-model form: pass the space directly. The
            per-model wrapper ``{model_name: {...}}`` is only for the
            multi-model ``run()`` / ``train([...])`` calls. Passing it here
            reads the model name as a model hyperparameter (a
            ``SuggestionWarning`` fires and nothing is tuned), so unwrap it:
            use ``{'feature_selector': {'k': (...)}}``, not
            ``{'coxph': {'feature_selector': {'k': (...)}}}``.
            For a custom ``BaseSurvivalModel`` instance, the ``'model'`` space
            merges with (and overrides) the instance's declared
            ``get_search_space()``.
        search_space : dict, optional
            Backward-compatible singular alias for ``search_spaces``. If both
            are given, ``search_spaces`` takes precedence.
        resume : bool, default=False
            If True, continue from existing Optuna study trials.
            If False (default), silently deletes any existing study before starting fresh.
        scoring : str, optional
            Scoring metric for tuning optimization: 'harrell', 'ipcw', or 'ctd'.
            If None (default), uses the scoring metric from setup().
            This allows tuning with a different metric than run().
        verbose : bool, default=True
            If True, print CV metrics table after tuning.
        n_jobs : int, optional
            Number of parallel jobs. If None, uses the value from setup().
        optuna_verbose : bool, default=False
            If True, show trial-by-trial Optuna output (INFO level).
            If False, suppress Optuna logging to WARNING level only.
        show_progress_bar : bool, default=True
            Show the Optuna per-model tuning progress bar.
        evaluate : bool, default=False
            If True, automatically evaluate tuned model on test set.
        eval_times : array-like, optional
            Timepoints for AUC(t) calculation.
        n_importance_repeats : int or "auto", default="auto"
            Number of permutation importance repeats.  When ``"auto"``
            (the default), uses 10 repeats for all models.
        importance_max_samples : int, optional
            Cap on the test rows used for permutation importance in the
            post-tune evaluation, so it applies only when ``evaluate=True``.
            ``None`` (the default) uses every row; the minimum is 2. See
            :meth:`evaluate`.
        include_shap : bool, default=False
            If True, compute SHAP values.
        shap_max_background : int, default=50
            Maximum background samples for the SHAP masker/explainer.
        shap_max_explain : int, default=50
            Maximum test samples to explain with SHAP.
        show_eval_progress : bool, default=True
            Show progress messages during evaluation.
        bootstrap : bool, default=False
            If True, compute bootstrap confidence intervals.
        n_bootstrap : int or "auto", default="auto"
            Number of bootstrap iterations.
        bootstrap_metrics : list, optional
            Metrics for bootstrap CI. Default: all metrics.
        include_interactions : bool, default=False
            If True, compute pairwise feature interactions during the
            post-tune evaluation (only used when ``evaluate=True``).
        interaction_top_k : int, default=5
            Number of top features (by permutation importance) to test.
            Produces ``C(top_k, 2)`` pairs.
        interaction_max_samples : int, default=100
            Maximum samples for interaction computation.
        interaction_method : str, default='hstat'
            Interaction method: ``'hstat'`` (Friedman's H-statistic,
            model-agnostic) or ``'shap'`` (SHAP interaction values via
            TreeExplainer, XGBoost models only).
        interaction_scale : str, default='auto'
            Axis spacing for the report's 2D interaction PDP surfaces:
            ``'auto'`` picks ``'linear'``/``'log'``/``'log1p'`` per axis from
            each feature's right-skew; pass ``'linear'`` to keep all axes
            linear, or ``'log'``/``'log1p'`` to force a spacing everywhere.

        Returns
        -------
        ModelResult or dict
            Single ModelResult if a single model is specified, dict of
            {name: ModelResult} if models=None (tuning all trained models).
            Dict keys are the plain algorithm names (e.g. ``'rsf'``) — the
            tuned variant overwrites the untuned model at the same key.

        Examples
        --------
        >>> # Tune single model directly
        >>> tuned_rsf = learner.tune('rsf')
        >>>
        >>> # Tune after training (tuned overwrites the untuned 'rsf')
        >>> learner.train()
        >>> learner.tune('rsf')  # learner.models['rsf'] is now tuned
        >>>
        >>> # Tune all trained models
        >>> learner.train()
        >>> tuned = learner.tune()  # Tune all
        """
        self._ensure_setup()
        search_spaces = self._validate_tune_inputs(
            n_trials, timeout, search_space, search_spaces
        )

        # Pre-resolve experiment_name so the Storage row in the summary box
        # shows the actual auto-generated path instead of "(auto-generated)"
        # Pass via TuneConfig so the tuner uses the same value; the
        # existing post-tune writeback in _tune_single_model will persist it
        # to _state.experiment_name.
        if (storage is None
                and experiment_name is None
                and self._state.experiment_name is None):
            experiment_name = _default_experiment_name()

        # Print tuning summary box (cosmetic — never abort tuning on failure).
        if verbose:
            try:
                tune_models = self._resolve_tune_models_for_display(models)
                if tune_models:
                    self._print_tuning_summary(
                        tune_models, n_trials, scoring, search_spaces,
                        experiment_name, storage, optuna_verbose,
                    )
            except Exception as e:
                if isinstance(e, (MemoryError, RecursionError)):
                    raise
                logger.warning("Tuning summary display failed: %s: %s", type(e).__name__, e)

        config = TuneConfig(
            n_trials=n_trials, timeout=timeout, storage=storage,
            study_name=study_name, experiment_name=experiment_name,
            force=force,
            search_spaces=search_spaces, resume=resume,
            scoring=scoring,
            n_jobs=n_jobs, optuna_verbose=optuna_verbose,
            show_progress_bar=show_progress_bar,
            evaluate=evaluate, eval_times=eval_times,
            n_importance_repeats=n_importance_repeats,
            importance_max_samples=importance_max_samples,
            include_shap=include_shap,
            shap_max_background=shap_max_background,
            shap_max_explain=shap_max_explain,
            show_eval_progress=show_eval_progress,
            bootstrap=bootstrap, n_bootstrap=n_bootstrap,
            bootstrap_metrics=bootstrap_metrics,
            include_interactions=include_interactions,
            interaction_top_k=interaction_top_k,
            interaction_max_samples=interaction_max_samples,
            interaction_method=interaction_method,
            interaction_scale=interaction_scale,
        )

        # Multi-model: early return
        if models is None:
            results = self._tune_all_models(config)
            if verbose and results:
                dn = self._display_name_map()
                print_multi_model_best_params(results, display_names=dn)
                first_result = next(iter(results.values()))
                n_folds = len(first_result.cv_metric_scores)
                print_multi_model_cv_table(
                    results, n_folds, display_names=dn,
                    cv_repeats=self._state.cv_repeats,
                )
            return results

        # Resolve model name for single model tuning. A custom
        # BaseSurvivalModel instance is carried through so the tuner clones
        # it per trial (the registry has no entry for it). Fall back to
        # 'custom' for a nameless instance so the storage key is never None
        # (matches the train() custom route).
        custom_model = models if isinstance(models, BaseSurvivalModel) else None
        if custom_model is not None:
            model_name = getattr(custom_model, 'name', None) or 'custom'
        else:
            model_name = _resolve_model_name(models)

        result = self._tune_single_model(
            model_name, config, custom_model=custom_model,
        )

        _warn_unused_dict_keys(n_trials, "n_trials", [model_name])

        if verbose:
            self._print_cv_summary(result)

        return result

    def _print_tuning_summary(
        self: SurvivalLearner,
        models,
        n_trials,
        scoring,
        search_spaces,
        experiment_name,
        storage,
        optuna_verbose,
    ):
        """Print a boxed summary of tuning parameters.

        Parameters
        ----------
        models : iterable of str
            Model names being tuned.
        """
        models_list = list(models)
        dn = get_model_display_names()
        display_list = [resolve_display_name(m, dn) for m in models_list]
        if len(display_list) == 1:
            models_str = display_list[0]
        else:
            models_str = f"{', '.join(display_list)} ({len(display_list)} models)"

        rows = [
            ("Models:", models_str),
            ("Trials:", _format_trials_display(
                n_trials, models_list, len(self.X_train.columns), display_list)),
            ("Scoring:", _format_scoring_display(scoring, self.scoring)),
            ("Search:", _format_search_display(search_spaces)),
            ("Storage:", _format_storage_display(
                storage, experiment_name, self._state.experiment_name)),
            ("Optuna:", "verbose" if optuna_verbose else "quiet"),
        ]
        _print_param_box("Tuning", rows)

    def _tune_all_models(self: SurvivalLearner, config: TuneConfig):
        """Tune all trained (non-tuned) models.

        Parameters
        ----------
        config : TuneConfig
            Tuning configuration.

        Returns
        -------
        dict of {name: ModelResult}
            Keys are the plain algorithm names in ``self._state.models``
            (e.g. ``'rsf'``); the tuned variant overwrites the untuned model.
        """
        trainable_models = {
            k: v for k, v in self._state.models.items()
            if not v.is_tuned
        }
        if not trainable_models:
            raise ValueError(
                "No trained models found. Run train() first, "
                "or specify a model name."
            )
        results = {}
        succeeded = set()  # tracks base model_name (key in trainable_models)
        skipped = set()
        errors = {}
        for model_name in trainable_models:
            try:
                model_n_trials = _resolve_per_model(config.n_trials, model_name, default='auto')
                model_config = _replace(config, n_trials=model_n_trials)
                result = self._tune_single_model(
                    model_name, model_config
                )
                if result.is_tuned:
                    # Key by the plain model name so the returned dict matches
                    # _state.models keys for downstream evaluate/bootstrap/
                    # report consumers.
                    results[result.name] = result
                    succeeded.add(model_name)
                else:
                    skipped.add(model_name)
            except Exception as e:
                errors[model_name] = _handle_model_error(
                    e, model_name, "tune", warn_user=False,
                )

        _warn_unused_dict_keys(config.n_trials, "n_trials", trainable_models.keys())
        # Only raise if models actually failed (not just skipped for no params)
        failed = set(trainable_models) - succeeded - skipped
        if failed:
            failed_errors = {n: errors[n] for n in failed if n in errors}
            _raise_if_all_failed(results, failed, "tune", errors=failed_errors)
            _warn_dropped_models(failed_errors, len(trainable_models), "tune")
        return results

    def _tune_single_model(
        self: SurvivalLearner,
        model_name,
        config: TuneConfig,
        preprocessing=None,
        custom_model=None,
    ):
        """Internal method to tune a single model.

        Falls back to training with defaults if no tuner is registered.

        Parameters
        ----------
        preprocessing : dict, optional
            Resolved per-model preprocessing overrides from train().
        custom_model : BaseSurvivalModel, optional
            A user-provided custom model instance to tune. When set, a
            ``CustomModelTuner`` clones it per trial and the tuned model is
            rebuilt from the instance rather than the registry.
        """
        n_jobs = config.n_jobs if config.n_jobs is not None else self.n_jobs

        # Skip if already tuned, unless forcing re-tune.
        existing = self._state.models.get(model_name)
        if existing is not None and existing.is_tuned and not config.force:
            warnings.warn(
                f"Model '{model_name}' was already tuned. Skipping tuning. "
                f"Use force=True to re-tune.",
                SuggestionWarning,
                stacklevel=2,
            )
            return existing

        effective_scoring = config.scoring if config.scoring is not None else self.scoring

        # Resolve effective preprocessing for this model
        effective_imputer, effective_scaler, effective_selector = \
            self._resolve_model_preprocessing(model_name, preprocessing)

        tuner = self._maybe_create_tuner(
            model_name, config, effective_scoring,
            effective_imputer=effective_imputer,
            effective_scaler=effective_scaler,
            effective_selector=effective_selector,
            preprocessing=preprocessing,
            custom_model=custom_model,
        )
        if tuner is None:
            # No tuner or no searchable params — return existing untuned model
            if existing is not None:
                return existing
            return self._train_model_internal(
                model_name, _preprocessing=preprocessing,
            )

        # An existing result will be replaced by the tuned one. Warn now, before
        # the (possibly minute-long) Optuna run and the CV table, so the heads-up
        # leads the output instead of trailing it — matching train()/run(). Every
        # path from here installs the tuned result; _install_tuned_result does the
        # replacement silently.
        if existing is not None:
            self._warn_overwrite(model_name)

        n_trials = _resolve_n_trials_for_model(config, model_name, len(self.X_train.columns))

        # n_trials=0 without resume → skip tuning, use defaults
        if n_trials == 0 and not config.resume:
            return self._fallback_train_as_tuned(
                model_name, config, preprocessing=preprocessing,
            )

        # Skip EPV warning for n_trials=0 (loading existing study, no new tuning)
        if n_trials != 0:
            self._warn_epv_if_low(stacklevel=4)
            self._warn_small_sample_tuning(stacklevel=4)

        best_params, study, used_experiment_name = tuner.tune(
            n_trials=n_trials, timeout=config.timeout, storage=config.storage,
            study_name=config.study_name,
            experiment_name=(
                config.experiment_name
                if config.experiment_name is not None
                else self._state.experiment_name
            ),
            resume=config.resume, n_jobs=n_jobs,
            optuna_verbose=config.optuna_verbose,
            show_progress_bar=config.show_progress_bar,
        )
        if self._state.experiment_name is None:
            self._state.experiment_name = used_experiment_name

        result = self._train_with_tuned_params(
            model_name, best_params, effective_scoring, preprocessing=preprocessing,
            custom_model=custom_model,
        )
        result.best_params_ = best_params
        result.optuna_study = study
        result.scoring_metric = effective_scoring
        result.is_tuned = True

        self._install_tuned_result(model_name, result)

        if config.evaluate:
            self._safe_post_tune_evaluation(result, config, model_name)

        return result

    def _run_post_tune_evaluation(self: SurvivalLearner, result, config: TuneConfig):
        """Run post-tune evaluation on a single model result."""
        result.evaluation = self.evaluate(
            result, eval_times=config.eval_times,
            n_importance_repeats=config.n_importance_repeats,
            importance_max_samples=config.importance_max_samples,
            include_shap=config.include_shap,
            shap_max_background=config.shap_max_background,
            shap_max_explain=config.shap_max_explain,
            show_progress=config.show_eval_progress,
            bootstrap=config.bootstrap, n_bootstrap=config.n_bootstrap,
            bootstrap_metrics=config.bootstrap_metrics,
            include_predictions=config.bootstrap,
            include_interactions=config.include_interactions,
            interaction_top_k=config.interaction_top_k,
            interaction_max_samples=config.interaction_max_samples,
            interaction_method=config.interaction_method,
            interaction_scale=config.interaction_scale,
        )

    def _safe_post_tune_evaluation(
        self: SurvivalLearner, result, config: TuneConfig, model_name
    ):
        """Run post-tune evaluation, warning on failure instead of raising."""
        try:
            self._run_post_tune_evaluation(result, config)
        except (ValueError, TypeError, KeyError, AttributeError,
                RuntimeError, ZeroDivisionError, ImportError,
                np.linalg.LinAlgError) as e:
            warnings.warn(
                f"Post-tune evaluation failed for {model_name}: {e}. "
                "The tuned model is still available.",
                InternalWarning,
                stacklevel=2,
            )

    def _install_tuned_result(self: SurvivalLearner, model_name, result):
        """Store a freshly tuned *result* under its plain ``model_name``.

        Overwrites ``_state.models[model_name]``. Replacing the stored
        ``ModelResult`` drops the old one and everything riding on it
        (recalibrator, importance-test, and risk-group-test tables) — the tuned
        ``S(t|x)`` differs, so the old isotonic mapping would silently
        miscalibrate and old p-values describe the old pipeline.

        This is silent: the overwrite ``SuggestionWarning`` (same message as
        ``train()``/``run()``) is emitted up front in ``_tune_single_model``,
        before the Optuna run, so it leads the tune output rather than trailing
        the CV table.
        """
        result.name = model_name
        self._state.models[model_name] = result
        return result

    def _fallback_train_as_tuned(
        self: SurvivalLearner, model_name, config, preprocessing=None,
    ):
        """Train with defaults and mark as tuned (no tuner or no searchable params)."""
        result = self._train_model_internal(
            model_name, _store=False, _preprocessing=preprocessing,
        )
        result.is_tuned = True
        result.best_params_ = {}
        self._install_tuned_result(model_name, result)
        return result

    def _maybe_create_tuner(
        self: SurvivalLearner, model_name, config, effective_scoring,
        effective_imputer=_TUNER_NOT_SET, effective_scaler=_TUNER_NOT_SET,
        effective_selector=_TUNER_NOT_SET, preprocessing=None, custom_model=None,
    ):
        """Create a tuner if the model has searchable parameters.

        Returns
        -------
        tuner or None
            The tuner on success, or ``None`` if the model has no tuner or
            no searchable parameters.
        """
        try:
            tuner = self._create_tuner_for_model(
                model_name, config.search_spaces, effective_scoring,
                effective_imputer=effective_imputer,
                effective_scaler=effective_scaler,
                effective_selector=effective_selector,
                custom_model=custom_model,
            )
        except NoTunerError as e:
            warnings.warn(
                f"{e} Skipping tuning.",
                SuggestionWarning,
            )
            return None

        if not tuner.has_searchable_params:
            print(
                f"Skipping tuning for '{model_name}': no hyperparameters to "
                f"search. Training with default parameters."
            )
            return None
        return tuner

    def _create_tuner_for_model(
        self: SurvivalLearner, model_name, search_spaces=None,
        scoring=None,
        effective_imputer=_TUNER_NOT_SET, effective_scaler=_TUNER_NOT_SET,
        effective_selector=_TUNER_NOT_SET, custom_model=None,
    ):
        """Create and return the appropriate tuner for a model.

        Uses ``_state.cv_splitter_single`` for tuning (always single-run CV for efficiency).

        Parameters
        ----------
        scoring : str, optional
            If None, scoring is not passed to the tuner (uses tuner default).
        effective_imputer, effective_scaler, effective_selector : optional
            Pre-resolved per-model preprocessing components. When provided,
            these override the global ``_state`` defaults so the tuner uses
            the correct base components for this model. Use ``None`` to
            explicitly skip a component.
        custom_model : BaseSurvivalModel, optional
            A user-provided custom model instance. When set, a
            ``CustomModelTuner`` is built around it (clone-per-trial) instead
            of resolving ``model_name`` against the registry.

        Raises
        ------
        NoTunerError
            If ``model_name`` has no registered tuner (propagated from ``get_tuner``).
        ValueError
            If ``custom_model`` is set but has no searchable parameters: no
            model space (declared via ``get_search_space()`` or per-call
            ``search_spaces``) *and* no per-call component space
            (feature_selector / imputer / scaler).
        """
        imputer = self._state.imputer if effective_imputer is _TUNER_NOT_SET else effective_imputer
        scaler = self._state.scaler if effective_scaler is _TUNER_NOT_SET else effective_scaler
        selector = self._state.feature_selector if effective_selector is _TUNER_NOT_SET else effective_selector

        tuner_kwargs = {
            'X': self.X_train,
            'y': self.y_train,
            'cv_splitter': self._state.cv_splitter_single,
            'random_state': self.random_state,
            'search_spaces': search_spaces,
            'imputer': imputer,
            'scaler': scaler,
            'selector': selector,
            'stratify_array': self._state.stratify_array,
            'groups': self._state.groups,
        }
        if scoring is not None:
            tuner_kwargs['scoring'] = scoring

        if custom_model is not None:
            # Build a tuner around the instance. The declared get_search_space()
            # becomes the base; BaseTuner.__init__ merges search_spaces['model']
            # on top (user wins). model_name is used only for display/storage.
            tuner = CustomModelTuner(
                custom_model=custom_model,
                search_space=custom_model.get_search_space(),
                model_name=model_name,
                **tuner_kwargs,
            )
            if not tuner.has_searchable_params:
                raise ValueError(
                    f"Custom model '{model_name}' has no parameters to tune. "
                    f"Implement get_search_space() on the model to return a "
                    f"non-empty dict, or pass a search_spaces= dict naming a "
                    f"model space ({{'model': {{...}}}}) and/or a component "
                    f"space (e.g. {{'feature_selector': {{'k': (2, 4)}}}}) "
                    f"to train()/tune()."
                )
            # Guard against a set_params that silently ignores unknown keys
            # (a common custom-model pattern): if a tuned param is absent from
            # get_params(), every trial would train the default and best_params_
            # would be meaningless.
            unknown = set(tuner.search_space) - set(custom_model.get_params())
            if unknown:
                warnings.warn(
                    f"Custom model '{model_name}' search space includes "
                    f"parameter(s) {sorted(unknown)} not exposed by get_params(); "
                    f"set_params may silently ignore them, so Optuna would "
                    f"optimize over a no-op axis. Ensure the model's __init__ / "
                    f"get_params expose these parameters.",
                    SuggestionWarning,
                    stacklevel=2,
                )
            # Custom instances skip automatic n_jobs propagation; warn once here
            # (covers both train(tune=True) and tune(instance) entry points).
            self._warn_custom_njobs_mismatch(custom_model, model_name)
        else:
            tuner = get_tuner(model_name, **tuner_kwargs)

        # Cross-phase FitFailureWarning dedup. The tuner emits its own trial
        # warnings (BaseTuner._emit_trial_warnings, called from tune()), so it
        # needs a direct reference to the learner's set. Without it, the
        # tune-phase copy and the post-tune CV / final-fit copy of the same
        # instability message both reach the user.
        tuner._seen_warning_keys = self._state.fit_failure_warned
        return tuner

    def _extract_tuned_params(self: SurvivalLearner, best_params, model_name):
        """
        Extract component-specific params from tuner's best_params dict.

        Returns
        -------
        tuple
            (model_params, imputer_params, scaler_params, selector_params, tuned_k)
        """
        model_params = extract_component_params(best_params, 'model')
        selector_params = extract_component_params(best_params, 'feature_selector')
        imputer_params = extract_component_params(best_params, 'imputer')
        scaler_params = extract_component_params(best_params, 'scaler')

        # If no model__ prefixed params, unprefixed best_params are model params.
        # Bare names belong to the model: a 'k' here is the model's
        # own hyperparameter, NOT the selector's (which arrives as
        # feature_selector__k and is picked up by selector_params below).
        if not model_params:
            model_params = {k: v for k, v in best_params.items()
                          if not k.startswith(('imputer__', 'scaler__', 'feature_selector__', 'model__'))}

        # The selector's k is tuned only via feature_selector__k, so it lands in
        # selector_params. Use .get (not an `or`-chain) so a tuned k of 0 — a
        # falsy-but-valid value — is not silently dropped.
        tuned_k = selector_params.get('k')

        # Translate model-specific tuner params to constructor params
        if model_name == 'coxnet':
            _translate_coxnet_params(model_params)
        _translate_deep_learning_params(model_params, model_name)

        return model_params, imputer_params, scaler_params, selector_params, tuned_k

    def _apply_tuned_components(self: SurvivalLearner, imputer_params, scaler_params,
                                selector_params, tuned_k):
        """
        Apply tuned parameters to preprocessing components.

        Clones and modifies the learner's imputer, scaler, and feature selector
        based on tuned parameters.
        """
        # Apply imputer and scaler params (simple clone-and-set pattern)
        for params, attr in [(imputer_params, 'imputer'), (scaler_params, 'scaler')]:
            component = getattr(self._state, attr)
            if params and component is not None:
                component = clone(component)
                component.set_params(**params)
                setattr(self._state, attr, component)

        # Apply selector params (filter out k — handled via get_or_create_selector below)
        if selector_params and self._state.feature_selector is not None:
            self._state.feature_selector = clone(self._state.feature_selector)
            non_k_params = {k: v for k, v in selector_params.items() if k != 'k'}
            if non_k_params:
                self._state.feature_selector.set_params(**non_k_params)

        # If the selector's k was tuned (feature_selector__k), update it
        if tuned_k is not None:
            self._state.feature_selector = get_or_create_selector(
                self._state.feature_selector, k=tuned_k)

    def _tune_and_train(self: SurvivalLearner, model_name, n_trials, search_spaces,
                        experiment_name=None,
                        optuna_verbose=False, resume=False, preprocessing=None,
                        show_progress_bar=True):
        """Tune model and return trained result (used by run()).

        Falls back to training with defaults if no tuner is registered.
        When ``n_trials=0`` and ``resume=False``, skips tuning entirely.
        When ``resume=True``, threads through to ``tuner.tune()``.

        Parameters
        ----------
        preprocessing : dict, optional
            Resolved per-model preprocessing overrides from run().
        """
        # Resolve effective preprocessing for this model (for tuner base components)
        effective_imputer, effective_scaler, effective_selector = \
            self._resolve_model_preprocessing(model_name, preprocessing)

        try:
            tuner = self._create_tuner_for_model(
                model_name, search_spaces, scoring=self.scoring,
                effective_imputer=effective_imputer,
                effective_scaler=effective_scaler,
                effective_selector=effective_selector,
            )
        except NoTunerError as e:
            warnings.warn(
                f"{e} Training with default parameters.",
                SuggestionWarning,
            )
            return self._train_model_internal(model_name, _preprocessing=preprocessing)

        if not tuner.has_searchable_params:
            print(
                f"Skipping tuning for '{model_name}': no hyperparameters to search. "
                f"Training with default parameters."
            )
            return self._train_model_internal(model_name, _preprocessing=preprocessing)

        n_trials = _resolve_per_model(n_trials, model_name, default='auto')

        # n_trials=0 without resume → skip tuning, use defaults
        if n_trials == 0 and not resume:
            warnings.warn(
                f"n_trials=0 without resume=True for '{model_name}': "
                f"skipping tuning, training with default parameters. "
                f"Use resume=True to load an existing study, or set "
                f"n_trials > 0 to run new trials.",
                SuggestionWarning,
                stacklevel=4,
            )
            result = self._train_model_internal(model_name, _preprocessing=preprocessing)
            result.is_tuned = True
            result.best_params_ = {}
            return result

        # n_trials=0 here means resume=True (loading existing study) — skip warnings
        if n_trials != 0:
            self._warn_epv_if_low(stacklevel=5)
            self._warn_small_sample_tuning(stacklevel=5)

        if n_trials == 'auto':
            n_trials = auto_n_trials(model_name, len(self.X_train.columns))

        effective_experiment_name = (
            experiment_name if experiment_name is not None else self._state.experiment_name
        )

        best_params, study, used_experiment_name = tuner.tune(
            n_trials=n_trials,
            experiment_name=effective_experiment_name,
            n_jobs=self.n_jobs,
            optuna_verbose=optuna_verbose,
            resume=resume,
            show_progress_bar=show_progress_bar,
        )
        if self._state.experiment_name is None:
            self._state.experiment_name = used_experiment_name

        result = self._train_with_tuned_params(
            model_name, best_params, self.scoring, preprocessing=preprocessing,
        )
        result.best_params_ = best_params
        result.optuna_study = study
        result.scoring_metric = self.scoring
        result.is_tuned = True

        return result

    def _train_with_tuned_params(
        self: SurvivalLearner, model_name, best_params, scoring, preprocessing=None,
        custom_model=None,
    ):
        """Extract tuned params, temporarily apply component changes, and train.

        Parameters
        ----------
        preprocessing : dict, optional
            Resolved per-model preprocessing overrides (threaded to
            ``_train_model_internal`` so ``_build_pipeline`` uses them).
        custom_model : BaseSurvivalModel, optional
            The original custom instance. When set, the tuned model is rebuilt
            via ``clone(custom_model).set_params(**model_params)`` and trained
            through the custom-model CV path (the registry has no entry for it).

        Returns
        -------
        ModelResult
        """
        model_params, imputer_params, scaler_params, selector_params, tuned_k = \
            self._extract_tuned_params(best_params, model_name)

        original_imputer = self._state.imputer
        original_scaler = self._state.scaler
        original_selector = self._state.feature_selector

        try:
            self._apply_tuned_components(imputer_params, scaler_params, selector_params, tuned_k)
            if custom_model is not None:
                tuned = clone(custom_model)
                if model_params:
                    tuned.set_params(**model_params)
                # Inject the learner seed so the stored, prediction-serving
                # pipeline is refit with the same seed its tuning trials used
                # (CustomModelTuner.objective does the same). Guard on
                # get_params() so models without random_state are untouched.
                if "random_state" in tuned.get_params():
                    tuned.set_params(random_state=self.random_state)
                    # Read back: a custom set_params can list random_state in
                    # get_params() yet silently drop it on write. If so the
                    # seed never lands and tuning is non-reproducible despite
                    # looking seeded. Warn once here (the per-trial objective
                    # stays quiet to avoid spamming). Mirrors the no-op-axis
                    # SuggestionWarning emitted for ignored tuned params above.
                    if tuned.get_params().get("random_state") != self.random_state:
                        warnings.warn(
                            f"Custom model '{model_name}' lists random_state in "
                            f"get_params() but its set_params did not apply the "
                            f"learner seed ({self.random_state}); tuning may be "
                            f"non-reproducible despite appearing seeded. Make "
                            f"set_params write random_state.",
                            SuggestionWarning,
                            stacklevel=2,
                        )
                # _emit_warnings=False: the kwargs/n_jobs warnings already fired
                # (or are irrelevant) for the original train() call; don't repeat
                # them on every tuned rebuild.
                return self._train_custom_model_internal(
                    tuned, _store=False, _emit_warnings=False,
                )
            return self._train_model_internal(
                model_name, _scoring=scoring, _store=False, _is_tuned=True,
                _preprocessing=preprocessing, **model_params,
            )
        finally:
            self._state.imputer = original_imputer
            self._state.scaler = original_scaler
            self._state.feature_selector = original_selector
