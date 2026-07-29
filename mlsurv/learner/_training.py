"""TrainingMixin - Model training, pipeline building, and run()."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .._shared import _warn_unused_dict_keys, display_name as _display_name
from .._warnings import FitFailureWarning, MethodologicalWarning, SuggestionWarning
from ..models import (
    get_model, list_models, _AUTO_DEFAULTS_REGISTRY,
)
from ..models.base import BaseSurvivalModel
from ..evaluation._calibrators import _CALIBRATE_UNSET
from ..results.containers import ModelResult
from ..results.printing import (
    print_cv_metrics_table, print_multi_model_cv_table,
    print_best_params, print_multi_model_best_params,
)
from ..logging_utils import get_logger
from ._cv_training import (
    CvTrainingMixin,
    _handle_model_error,
    _raise_if_all_failed,
    _replay_captured_warnings,
    _warn_dropped_models,
)

logger = get_logger("learner")

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


def _finite_mean_std(values):
    """Return mean/std using only finite values; all-NaN inputs return NaN/NaN.

    Uses sample standard deviation (ddof=1), consistent with ``_compute_cv_ci``.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan, np.nan
    if finite.size == 1:
        return float(np.mean(finite)), 0.0
    return float(np.mean(finite)), float(np.std(finite, ddof=1))


def _compute_cv_ci(scores, ci_level):
    """Compute t-distribution CI from CV fold scores.

    Parameters
    ----------
    scores : array-like
        Per-fold scores (may contain NaN/inf).
    ci_level : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    tuple of (float, float)
        (ci_lower, ci_upper), or (NaN, NaN) if fewer than 2 finite scores.

    Notes
    -----
    Uses sample standard deviation (ddof=1) and t-distribution with k-1
    degrees of freedom, where k is the number of finite scores. Assumes
    fold scores are independently and identically distributed.
    """
    from scipy.stats import t as t_dist

    arr = np.asarray(scores, dtype=float)
    finite = arr[np.isfinite(arr)]
    k = finite.size
    if k < 2:
        return np.nan, np.nan
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1))
    alpha = 1 - ci_level
    t_crit = t_dist.ppf(1 - alpha / 2, df=k - 1)
    margin = t_crit * (std / np.sqrt(k))
    return float(mean - margin), float(mean + margin)


def _resolve_per_model(value, model_name, default):
    """Resolve a scalar-or-dict parameter to a per-model value.

    Parameters
    ----------
    value : scalar or dict
        If dict, look up model_name; missing keys fall back to *default*.
        If scalar, return as-is.
    model_name : str
        Model key to look up.
    default
        Fallback when *value* is a dict and *model_name* is absent.

    Returns
    -------
    The resolved value for the given model.
    """
    if isinstance(value, dict):
        return value.get(model_name, default)
    return value


# Top-level keys that mark a search_spaces dict as *component-structured* (one
# space describing the pipeline), mirroring _parse_search_spaces in tuning/base.py.
# No registered model shares a name with any of these, so the two forms below are
# never ambiguous.
_COMPONENT_SEARCH_SPACE_KEYS = frozenset(
    {'model', 'feature_selector', 'imputer', 'scaler'}
)


def _resolve_model_search_spaces(search_spaces, model_names):
    """Map a multi-model ``search_spaces`` to a per-model dict.

    ``run()`` / multi-model ``train()`` accept two forms:

    - **Component-structured** -- top-level keys are component names
      (``model``/``feature_selector``/``imputer``/``scaler``), e.g.
      ``{'feature_selector': {'k': (5, 20)}}``. The same space is applied to
      **every** model (broadcast), matching how single-model ``tune()`` reads it.
    - **Per-model** -- top-level keys are model names, e.g.
      ``{'coxph': {'feature_selector': {'k': (5, 20)}}}``. Each model gets its
      own space; keys matching no model in *model_names* are ignored with a
      ``SuggestionWarning`` (rather than dropped silently).

    Parameters
    ----------
    search_spaces : dict or None
        The user-supplied ``search_spaces``.
    model_names : list of str
        Models being trained.

    Returns
    -------
    dict of {str: dict or None}
        Per-model search space (``None`` when a model has none).

    Raises
    ------
    ValueError
        If the dict mixes component keys with model-name keys.
    """
    if not search_spaces:
        return {name: None for name in model_names}

    component_keys = set(search_spaces) & _COMPONENT_SEARCH_SPACE_KEYS
    model_keys = set(search_spaces) - _COMPONENT_SEARCH_SPACE_KEYS

    if component_keys and model_keys:
        raise ValueError(
            "search_spaces mixes component keys "
            f"{sorted(component_keys)} with non-component keys "
            f"{sorted(model_keys)}. Use EITHER a component-structured dict "
            "applied to all models (e.g. {'feature_selector': {'k': (5, 20)}}) "
            "OR a per-model dict (e.g. {'coxph': {'feature_selector': "
            "{'k': (5, 20)}}}), not both."
        )

    if component_keys:
        # Broadcast the one space to every model.
        return {name: search_spaces for name in model_names}

    # Per-model form: surface keys that match no model instead of dropping them.
    _warn_unused_dict_keys(search_spaces, "search_spaces", model_names)
    return {name: search_spaces.get(name) for name in model_names}


def _validate_model_params(model_params, kwargs):
    """Validate model_params structure and check for conflict with kwargs.

    Parameters
    ----------
    model_params : dict or None
        Per-model constructor kwargs: {model_name: {param: value}}.
    kwargs : dict
        Additional keyword arguments (from **kwargs).

    Raises
    ------
    ValueError
        If both model_params and kwargs are provided.
    TypeError
        If model_params or its values have incorrect types.
    """
    if model_params is not None and kwargs:
        raise ValueError(
            "Cannot specify both model_params and **kwargs. "
            "Use model_params for per-model configuration."
        )
    if model_params is not None:
        if not isinstance(model_params, dict):
            raise TypeError(
                f"model_params must be a dict, got {type(model_params).__name__}"
            )
        for key, val in model_params.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"model_params keys must be strings (model names), "
                    f"got {type(key).__name__}: {key!r}"
                )
            if not isinstance(val, dict):
                raise TypeError(
                    f"model_params[{key!r}] must be a dict of parameter "
                    f"names to values, got {type(val).__name__}"
                )


def _warn_unused_model_params(model_params, model_names):
    """Warn if model_params contains keys not in the models list.

    Parameters
    ----------
    model_params : dict or None
        Per-model constructor kwargs.
    model_names : list or set
        Model names that were actually trained.
    """
    _warn_unused_dict_keys(model_params, "model_params", model_names)


def _check_search_space_model_params(model_name, params):
    """Raise a clear ValueError if model_params holds Optuna search-space keys.

    The DeepSurv/DeepHit Optuna search space samples ``n_layers``, ``n_nodes``,
    and (DeepSurv only) ``lr``. The tuner translates these to constructor form
    before building the model (see ``_translate_deep_learning_params`` in
    ``_tuning.py``: ``layers=[n_nodes]*n_layers``, ``lr -> learning_rate``), but
    ``best_params_`` records the *sampled* search-space names. Replaying those
    names via ``model_params=`` would reach the constructor as unexpected kwargs
    and raise an opaque ``TypeError``. Detect them up front and explain instead.

    Parameters
    ----------
    model_name : str
        Model being trained.
    params : dict
        The per-model ``model_params`` entry about to be applied.
    """
    if model_name not in ('deepsurv', 'deephit'):
        return
    offending = [k for k in ('n_layers', 'n_nodes', 'lr') if k in params]
    if not offending:
        return
    hints = []
    if 'n_layers' in offending or 'n_nodes' in offending:
        hints.append('combine n_layers/n_nodes into layers=[n_nodes]*n_layers')
    if 'lr' in offending:
        hints.append('rename lr -> learning_rate')
    raise ValueError(
        f"model_params for '{model_name}' contains Optuna search-space keys "
        f"{offending} that the model constructor does not accept. These are the "
        f"names best_params_ records, not constructor arguments. Translate first: "
        f"{'; '.join(hints)}."
    )


def _estimate_post_selection_features(selector, n_features_in, n_events):
    """Estimate post-selection feature count without fitting the selector.

    For ``TopKCoxSelector`` the output dimensionality is deterministic given
    ``(k, n_features_in, n_events)``.  For data-dependent selectors (VIF,
    Correlation, penalized Cox) the output cannot be known pre-fit, so we
    fall back to *n_features_in* and emit a ``SuggestionWarning``.

    Parameters
    ----------
    selector : transformer, Pipeline, or None
        The resolved feature selector for this model.
    n_features_in : int
        Number of features before selection (``len(X_train.columns)``).
    n_events : int
        Number of events in ``y_train``.

    Returns
    -------
    int
        Estimated post-selection feature count.
    """
    if selector is None:
        return n_features_in

    # Lazy imports to avoid circular imports (mirrors _get_selector_presets).
    from mlsurv.preprocessing.selection import (
        TopKCoxSelector, VIFSelector, CorrelationSelector,
        _PenalizedCoxSelector,
    )

    # If the selector is a Pipeline (e.g., [VIF, TopK]), inspect the last step.
    terminal = selector[-1] if isinstance(selector, Pipeline) else selector

    if isinstance(terminal, TopKCoxSelector):
        if terminal.k == 'auto':
            # Mirror Harrell's rule from TopKCoxSelector.fit() (selection.py).
            # Uses full training-set n_events — an approximation of per-fold
            # counts, but sufficient for auto-default sizing.
            return min(n_features_in, max(1, n_events // 15))
        # Explicit integer k
        return min(terminal.k, n_features_in)

    if isinstance(terminal, (VIFSelector, CorrelationSelector,
                             _PenalizedCoxSelector)):
        selector_name = type(terminal).__name__
        warnings.warn(
            f"{selector_name} has data-dependent output dimensionality. "
            f"Auto-default model parameters use the pre-selection feature "
            f"count ({n_features_in}). If the selector significantly reduces "
            f"features, consider passing explicit model_params.",
            SuggestionWarning,
            stacklevel=3,
        )
        return n_features_in

    # Unknown / custom selector — cannot estimate dimensionality.
    selector_name = type(terminal).__name__
    warnings.warn(
        f"Unrecognized selector {selector_name}: cannot estimate "
        f"post-selection feature count. Auto-default model parameters use "
        f"the pre-selection feature count ({n_features_in}). If the selector "
        f"significantly reduces features, consider passing explicit "
        f"model_params.",
        SuggestionWarning,
        stacklevel=3,
    )
    return n_features_in


class TrainingMixin(CvTrainingMixin):
    """Mixin for model training, pipeline building, and the run() pipeline."""

    def train(
        self: SurvivalLearner,
        models: str | list[str] | BaseSurvivalModel | None = None,
        tune: bool = False,
        n_trials: int | str | dict = 'auto',
        experiment_name: str | None = None,
        search_spaces: dict | None = None,
        verbose: bool = True,
        model_params: dict | None = None,
        preprocessing: dict | None = None,
        calibrate=_CALIBRATE_UNSET,
        **kwargs
    ) -> ModelResult | dict[str, ModelResult]:
        """
        Train model(s) with default or tuned hyperparameters.

        When tune=False (default), trains with default hyperparameters.
        When tune=True, performs Optuna hyperparameter tuning before training.

        Training a model overwrites any existing model stored under the same
        name (baseline or tuned).

        Parameters
        ----------
        models : str, list of str, BaseSurvivalModel, or None, default=None
            Model(s) to train. Use `list_models()` to see available options.
            - If None: trains all available models
            - If str: trains single model (returns ModelResult)
            - If BaseSurvivalModel: trains custom model instance (returns ModelResult).
              Tunable with ``tune=True`` when the instance declares a
              ``get_search_space()`` hook or a per-call ``search_spaces=`` is given.
            - If list: trains specified models (returns dict)
        tune : bool, default=False
            If True, perform hyperparameter tuning using Optuna.
            Models without tuners fall back to default parameters. Custom
            model instances require a search space (declared via
            ``get_search_space()`` or passed via ``search_spaces=``);
            otherwise a clear ``ValueError`` is raised.
        n_trials : int, str, or dict, default='auto'
            Number of Optuna trials for tuning. Only used when tune=True.
            'auto' determines trials based on model complexity.
            Pass a dict of {model_name: int} for per-model trial counts;
            models not in the dict fall back to 'auto'.
        experiment_name : str, optional
            Experiment name for Optuna study persistence. Only used when tune=True.
            Overrides the experiment_name set in setup().
        search_spaces : dict, optional
            Custom search spaces for tuning. Only used when tune=True.
            For a single model: a flat ``{param: distribution}`` or the
            component-structured ``{'model': {...}, 'feature_selector': {...}}``.
            For multiple models: either a **component-structured** dict applied
            to every model (top-level keys ``model``/``feature_selector``/
            ``imputer``/``scaler``) or a **per-model** dict of
            ``{model_name: {...}}``; mixing the two raises ``ValueError`` (see
            ``run()`` for details). To search the feature count without a dict,
            set the selector's ``k='tune'`` (see ``TopKCoxSelector``).
            For a custom model instance, ``{'model': {param: distribution}}``
            (or a flat ``{param: distribution}``) merges with — and overrides —
            the model's declared ``get_search_space()``.
        verbose : bool, default=True
            If True, print CV metrics table after training.
        model_params : dict, optional
            Per-model constructor kwargs. Dict of {model_name: {param: value}}.
            Only used when tune=False. Cannot be combined with **kwargs.
        preprocessing : dict, optional
            Per-model preprocessing overrides for this call. Dict of
            ``{model_name: {component: value}}``. Overrides setup-level
            per-model preprocessing. See ``setup()`` for details.
        calibrate : unset, bool, str, or dict, default=unset
            Attach an internal ``S(t|x)`` calibrator (fit on held-out OOF of
            the training cohort) to each model. The calibrated probabilities
            flow to evaluate, bootstrap, and predict (and the report metric
            tables that read them). The calibration plots and the metric-time
            Brier(t) panel also show the calibrated curve by default (pass
            ``raw=True`` to those plots for the uncalibrated view).

            - unset (default): apply the per-model default registry
              (``fssvm``→``breslow_slope``, ``deephit``→``isotonic``); every
              other model stays raw (rsf can opt in with
              ``calibrate={'rsf': 'lowdof'}``).
            - ``False`` / ``None``: no calibration (all raw).
            - a method string (``'isotonic'``, ``'lowdof'``, ``'breslow_slope'``):
              apply it to every model being trained.
            - a dict ``{model: method}``: per-model control; models not listed
              stay raw.

            ``isotonic`` and ``lowdof`` work for any model; ``breslow_slope`` is
            valid only for score-based models (an inapplicable pair raises
            ``ValueError``). The calibrators are monotone, so rank ordering and
            the C-index are preserved (``isotonic`` can introduce ties — a
            negligible drift; it is only a default for deephit).
        **kwargs
            Model-specific hyperparameters to override defaults.
            Only applied when training a single model with tune=False.
            Cannot be combined with model_params.

        Returns
        -------
        ModelResult or dict
            Single ModelResult if models is str, dict of {name: ModelResult}
            if models is None or list.

        Examples
        --------
        >>> # Train all models with defaults
        >>> results = learner.train()
        >>>
        >>> # Train specific models
        >>> results = learner.train(['coxph', 'rsf'])
        >>>
        >>> # Train single model
        >>> result = learner.train('coxph')
        >>>
        >>> # Train with tuning
        >>> result = learner.train('rsf', tune=True, n_trials=50)
        >>>
        >>> # Per-model hyperparameters
        >>> results = learner.train(['rsf', 'gbsa'], model_params={
        ...     'rsf': {'n_estimators': 100},
        ...     'gbsa': {'n_estimators': 50, 'max_depth': 3},
        ... })
        >>>
        >>> # train() overwrites a previously tuned model at the same key
        >>> learner.tune('rsf')       # learner.models['rsf'] is tuned
        >>> learner.train('rsf')      # overwrites with the default 'rsf'
        """
        self._ensure_setup()

        from ._setup import _validate_preprocessing, _resolve_per_model_preprocessing
        _validate_preprocessing(preprocessing, context='train')
        resolved_preprocessing = _resolve_per_model_preprocessing(
            preprocessing, random_state=self.random_state
        )

        model_params, kwargs = self._normalize_train_args(tune, model_params, kwargs)
        tuning_args = dict(
            n_trials=n_trials, experiment_name=experiment_name,
            search_spaces=search_spaces,
        )

        # Emit tuning summary for a single named model. Custom BaseSurvivalModel
        # instances are tunable too, but the summary box resolves registry
        # metadata by name, so it is skipped for instances.
        if tune and verbose and isinstance(models, str):
            self._emit_tuning_summary([models], n_trials, search_spaces, experiment_name)

        single_kwargs = self._resolve_single_model_kwargs(models, model_params, kwargs)
        result = self._try_train_single(
            models, tune, tuning_args, resolved_preprocessing, **single_kwargs,
        )
        if self._finalize_single_model_result(
            result, models, verbose, model_params, n_trials,
            preprocessing=resolved_preprocessing,
        ):
            self._attach_calibrators(result, calibrate)
            return result

        models_to_train = self._resolve_models_to_train(models)

        if tune and verbose:
            self._emit_tuning_summary(models_to_train, n_trials, search_spaces, experiment_name)

        if not tune and kwargs:
            warnings.warn(
                "**kwargs are ignored when training multiple models. "
                "Use model_params={model: {params}} for per-model configuration.",
                SuggestionWarning,
            )

        results = self._train_multiple_models(
            models_to_train, tune, n_trials, experiment_name,
            search_spaces,
            model_params=model_params,
            preprocessing=resolved_preprocessing,
        )
        self._attach_calibrators(results, calibrate)
        self._print_multi_train_verbose(verbose, results, tune)
        return results

    def _emit_tuning_summary(
        self: SurvivalLearner,
        models,
        n_trials,
        search_spaces,
        experiment_name,
        optuna_verbose: bool = False,
    ) -> None:
        """Print the tuning summary box, swallowing non-critical display errors."""
        try:
            self._print_tuning_summary(
                models, n_trials, None, search_spaces,
                experiment_name, None, optuna_verbose,
            )
        except Exception as e:
            if isinstance(e, (MemoryError, RecursionError)):
                raise
            logger.warning("Tuning summary display failed: %s: %s", type(e).__name__, e)

    def _normalize_train_args(
        self: SurvivalLearner,
        tune: bool,
        model_params,
        kwargs: dict,
    ) -> tuple:
        """Validate inputs, warn on misuse. Returns (model_params, kwargs)."""
        _validate_model_params(model_params, kwargs)
        if tune and (kwargs or model_params):
            warnings.warn(
                "model_params/**kwargs are ignored when tune=True. "
                "Use search_spaces to customize tuning.",
                SuggestionWarning,
            )
            model_params = None
            kwargs = {}
        return model_params, kwargs

    @staticmethod
    def _resolve_single_model_kwargs(model, model_params, kwargs) -> dict:
        """Merge per-model params into kwargs for single-model str path."""
        single_kwargs = dict(kwargs)
        if model_params and isinstance(model, str) and model in model_params:
            single_kwargs.update(model_params[model])
        return single_kwargs

    def _print_multi_train_verbose(
        self: SurvivalLearner,
        verbose: bool,
        results: dict,
        tune: bool,
    ) -> None:
        """Print multi-model training results (best params + CV table) when verbose."""
        if not (verbose and results):
            return
        dn = self._display_name_map()
        if tune:
            print_multi_model_best_params(results, display_names=dn)
        first_result = next(iter(results.values()))
        n_folds = len(first_result.cv_metric_scores)
        print_multi_model_cv_table(
            results, n_folds, display_names=dn, cv_repeats=self._state.cv_repeats,
        )

    @staticmethod
    def _resolve_run_model_params(models, tune: bool, model_params, kwargs) -> tuple:
        """Normalize run() kwargs: warn on misuse, merge single-model convenience kwargs.

        Returns (model_params, kwargs) after handling four cases:
        1. tune=True with kwargs/model_params → warn, discard (return empty kwargs)
        2. tune=False, kwargs, no models specified → warn, discard (return empty kwargs)
        3. tune=False, kwargs, multiple models → warn, discard (return empty kwargs)
        4. tune=False, kwargs, single model → merge into model_params, discard kwargs
        """
        _validate_model_params(model_params, kwargs)
        if tune and (kwargs or model_params):
            warnings.warn(
                "model_params/**kwargs are ignored when tune=True. "
                "Use search_spaces to customize tuning.",
                SuggestionWarning,
            )
            return None, {}
        if not tune and kwargs:
            if models is None:
                warnings.warn(
                    "**kwargs are ignored when models is not specified (defaults to all models). "
                    "Use model_params={model: {params}} for per-model configuration, "
                    "or pass a single model name: run('rsf', n_estimators=50).",
                    SuggestionWarning,
                )
                return model_params, {}
            if len(models) > 1:
                warnings.warn(
                    "**kwargs are ignored when training multiple models. "
                    "Use model_params={model: {params}} for per-model configuration.",
                    SuggestionWarning,
                )
                return model_params, {}
            model_params = {models[0]: kwargs}
            kwargs = {}
        return model_params, kwargs

    def _finalize_single_model_result(
        self: SurvivalLearner,
        result,
        model,
        verbose: bool,
        model_params,
        n_trials,
        preprocessing=None,
    ) -> bool:
        """Post-training steps for single-model path. Returns True if result was produced."""
        if result is None:
            return False
        if isinstance(model, str):
            _warn_unused_model_params(model_params, [model])
            _warn_unused_dict_keys(n_trials, "n_trials", [model])
            _warn_unused_dict_keys(preprocessing, "preprocessing", [model])
        if verbose:
            self._print_cv_summary(result)
        return True

    def _try_train_single(
        self: SurvivalLearner, model, tune, tuning_args, preprocessing=None, **kwargs,
    ):
        """Train a single model (str or BaseSurvivalModel). Returns None for multi-model."""
        if isinstance(model, str):
            if tune:
                return self._train_single_with_tuning(
                    model, preprocessing=preprocessing, **tuning_args,
                )
            return self._train_model_internal(
                model, _preprocessing=preprocessing, **kwargs,
            )
        if isinstance(model, BaseSurvivalModel):
            if tune:
                # Route custom instances through the tuning path, carrying the
                # instance itself (the registry knows nothing about it). The
                # name is only used for storage keys / display.
                model_name = getattr(model, 'name', None) or 'custom'
                return self._train_single_with_tuning(
                    model_name, custom_model=model,
                    preprocessing=preprocessing, **tuning_args,
                )
            return self._train_custom_model_internal(model, **kwargs)
        return None

    def _train_single_with_tuning(
        self: SurvivalLearner,
        model_name,
        n_trials='auto',
        experiment_name=None,
        search_spaces=None,
        preprocessing=None,
        custom_model=None,
    ):
        """
        Train a single model with tuning, falling back to defaults if no tuner.

        Parameters
        ----------
        model_name : str
            Model identifier.
        n_trials : int or 'auto'
            Number of Optuna trials.
        experiment_name : str, optional
            Experiment name for Optuna study persistence.
        search_spaces : dict, optional
            Custom search spaces for this model.
        preprocessing : dict, optional
            Resolved per-model preprocessing overrides from train().
        custom_model : BaseSurvivalModel, optional
            A user-provided custom model instance to tune. When set, the
            tuner clones this instance per trial instead of resolving
            ``model_name`` against the registry.

        Returns
        -------
        ModelResult
            Trained model result (tuned if tuner available, defaults otherwise).
        """
        from ._tuning import TuneConfig
        config = TuneConfig(
            n_trials=n_trials, experiment_name=experiment_name,
            search_spaces=search_spaces,
        )
        return self._tune_single_model(
            model_name, config, preprocessing=preprocessing,
            custom_model=custom_model,
        )

    @staticmethod
    def _resolve_models_to_train(model):
        """Resolve model argument to a list of model names."""
        if model is None:
            return list_models()
        if isinstance(model, list):
            return model
        raise ValueError(
            f"model must be str, list of str, BaseSurvivalModel instance, "
            f"or None. Got {type(model).__name__}"
        )

    def _train_multiple_models(
        self: SurvivalLearner, models_to_train, tune, n_trials, experiment_name,
        search_spaces, model_params=None,
        preprocessing=None,
    ):
        """Train or tune multiple models, collecting results.

        Parameters
        ----------
        model_params : dict, optional
            Per-model constructor kwargs: {model_name: {param: value}}.
        preprocessing : dict, optional
            Resolved per-model preprocessing overrides from run()/train().

        Returns
        -------
        dict of {name: ModelResult}
        """
        action = "tune" if tune else "train"
        results = {}
        errors = {}
        resolved_search_spaces = (
            _resolve_model_search_spaces(search_spaces, models_to_train)
            if tune else {}
        )
        for model_name in models_to_train:
            try:
                if tune:
                    model_search_spaces = resolved_search_spaces[model_name]
                    model_n_trials = _resolve_per_model(n_trials, model_name, default='auto')
                    results[model_name] = self._train_single_with_tuning(
                        model_name,
                        n_trials=model_n_trials,
                        experiment_name=experiment_name,
                        search_spaces=model_search_spaces,
                        preprocessing=preprocessing,
                    )
                else:
                    params = model_params.get(model_name, {}) if model_params else {}
                    _check_search_space_model_params(model_name, params)
                    results[model_name] = self._train_model_internal(
                        model_name, _preprocessing=preprocessing, **params,
                    )
            except Exception as e:
                errors[model_name] = _handle_model_error(
                    e, model_name, action, warn_user=False,
                )

        _warn_unused_model_params(model_params, models_to_train)
        _warn_unused_dict_keys(n_trials, "n_trials", models_to_train)
        _warn_unused_dict_keys(preprocessing, "preprocessing", models_to_train)
        _raise_if_all_failed(results, models_to_train, action, errors=errors)
        _warn_dropped_models(errors, len(models_to_train), action)
        return results

    def _print_cv_summary(self: SurvivalLearner, result):
        """Print best hyperparameters (if tuned) and CV metrics for a single model."""
        display_name = _display_name(result.name)
        if result.best_params_:
            print_best_params(result.best_params_, model_name=display_name)
        n_folds = len(result.cv_metric_scores)
        cv_metrics = result.cv_metrics or {
            'harrell': {
                'mean': result.cv_metric_mean,
                'std': result.cv_metric_std,
                'scores': result.cv_metric_scores,
                'ci_lower': result.cv_ci_lower,
                'ci_upper': result.cv_ci_upper,
            }
        }
        print_cv_metrics_table(
            cv_metrics, n_folds, display_name=display_name,
            cv_repeats=self._state.cv_repeats,
        )

    def _train_model_internal(self: SurvivalLearner, model_name, _scoring=None,
                              _store=True, _is_tuned=False, _preprocessing=None,
                              _cv_n_jobs=None, **kwargs):
        """Internal model creation with CV.

        Parameters
        ----------
        model_name : str
            Model identifier.
        _scoring : str, optional
            Scoring metric override. If None, uses self.scoring.
            Used internally by tune() to ensure tuned model CV scores
            match the metric used for optimization.
        _store : bool, default=True
            If True, store result in _state.models dict.
        _is_tuned : bool, default=False
            Whether this is a tuned model (from tune()).
        _cv_n_jobs : int, optional
            Override for CV fold parallelism.  When set, this value is
            used instead of the default ``self.n_jobs`` / sequential
            logic.  Used by ``parameter_sweep(n_jobs>1)`` to force
            sequential CV when the outer sweep loop is parallel.
        **kwargs :
            Model parameters.
        """
        from mlsurv.learner._core import (
            DEEP_LEARNING_MODELS,
            MODELS_WITH_INTERNAL_PARALLELISM,
            MODELS_WITH_RANDOM_STATE,
            SEQUENTIAL_CV_MODELS,
        )

        if model_name in MODELS_WITH_RANDOM_STATE and 'random_state' not in kwargs:
            kwargs['random_state'] = self.random_state

        # Propagate n_jobs to models with internal parallelism, e.g. RSF, XGBCox,
        # so their fit() and predict() calls run in parallel. This is especially
        # important for permutation importance, which calls predict() repeatedly.
        # The 'n_jobs' not in kwargs guard preserves any explicit override
        # passed by the caller (e.g., learner.train('rsf', n_jobs=2)).
        # When _cv_n_jobs is set the caller is running inside a parallel outer
        # loop (e.g. parameter_sweep) — suppress model-internal parallelism too.
        if model_name in MODELS_WITH_INTERNAL_PARALLELISM and 'n_jobs' not in kwargs:
            kwargs['n_jobs'] = 1 if _cv_n_jobs is not None else self.n_jobs

        # Propagate device to deep learning models (DeepSurv, DeepHit).
        # Guards: explicit kwargs override takes priority; None means
        # "use model's constructor default" so we skip propagation.
        if (model_name in DEEP_LEARNING_MODELS
                and 'device' not in kwargs
                and self._state.device is not None):
            kwargs['device'] = self._state.device

        effective_scoring = _scoring if _scoring is not None else self.scoring
        if _cv_n_jobs is not None:
            cv_n_jobs = _cv_n_jobs
        else:
            cv_n_jobs = 1 if model_name in SEQUENTIAL_CV_MODELS else self.n_jobs

        model = self._create_model_instance(
            model_name, _preprocessing=_preprocessing, **kwargs,
        )
        pipeline = self._build_pipeline(model, model_name, _preprocessing)

        folds = self._get_cv_folds()
        fold_results = self._run_cv_folds(
            folds, pipeline, effective_scoring, cv_n_jobs, model_name,
        )
        scores, cv_metrics = self._aggregate_fold_results(fold_results, len(folds))

        # Fit final pipeline on full training set — suppress upstream
        # CoxNet warning; we check ourselves below. Capture & replay
        # so a FitFailureWarning already shown during CV (e.g. CoxPH alpha=0
        # numerical-instability refit) does not surface again here.
        final_pipeline = self._build_pipeline(model, model_name, _preprocessing)
        with warnings.catch_warnings(record=True) as caught_final:
            warnings.simplefilter("always")
            warnings.filterwarnings(
                "ignore", message=r"all coefficients are zero",
                category=UserWarning,
            )
            final_pipeline.fit(self.X_train, self.y_train)
        _replay_captured_warnings(
            caught_final,
            seen_messages=self._state.fit_failure_warned,
            dedup_category=FitFailureWarning,
        )

        # Check if the final CoxNet model has all-zero coefficients.
        # This is worth warning about (unlike the per-fold case) because it
        # means the trained model has no discriminative power.
        if model_name == 'coxnet':
            final_model = final_pipeline[-1]
            coef = getattr(final_model, 'coef_', None)
            if coef is not None and np.all(coef == 0):
                warnings.warn(
                    "CoxNet: all coefficients are zero in the final trained "
                    "model. The model has no discriminative power. Consider "
                    "decreasing alpha or using a different model.",
                    FitFailureWarning,
                    stacklevel=2,
                )

        return self._finalize_model_result(
            model_name=model_name,
            final_pipeline=final_pipeline,
            scores=scores,
            cv_metrics=cv_metrics,
            effective_scoring=effective_scoring,
            is_tuned=_is_tuned,
            store=_store,
        )

    def _create_model_instance(self: SurvivalLearner, model_name,
                               _preprocessing=None, **kwargs):
        """Create a model instance, passing data characteristics if needed.

        Parameters
        ----------
        model_name : str
            Name of the model to create.
        _preprocessing : dict, optional
            Run-time per-model preprocessing overrides, used to resolve the
            effective feature selector for post-selection feature count
            estimation.
        **kwargs
            Forwarded to ``get_model()``.
        """
        if model_name in _AUTO_DEFAULTS_REGISTRY:
            _, _, selector = self._resolve_model_preprocessing(
                model_name, _preprocessing,
            )
            n_features_in = len(self.X_train.columns)
            n_events = int(self.y_train['event'].sum())
            n_features = _estimate_post_selection_features(
                selector, n_features_in, n_events,
            )
            return get_model(
                model_name,
                n_samples=len(self.X_train),
                n_features=n_features,
                n_events=n_events,
                **kwargs,
            )
        return get_model(model_name, **kwargs)

    def _warn_custom_njobs_mismatch(self: SurvivalLearner, model_instance, model_name):
        """Warn if a custom model's n_jobs differs from setup(n_jobs=...).

        Custom instances do not get automatic n_jobs propagation the way named
        models do, so a mismatch means the model runs with parallelism the user
        may not expect. Emitted once per train()/tune() call (both the plain
        custom path and the tuned path), not per CV fold or per trial.
        """
        if (getattr(model_instance, 'has_internal_parallelism', False)
                and hasattr(model_instance, 'n_jobs')
                and model_instance.n_jobs != self.n_jobs):
            warnings.warn(
                f"Custom {model_name} instance has n_jobs={model_instance.n_jobs}, "
                f"but setup(n_jobs={self.n_jobs}) was configured. "
                f"Pass n_jobs={self.n_jobs} to the model constructor, or use "
                f"train('{model_name}') for automatic propagation.",
                SuggestionWarning,
                stacklevel=2,
            )

    def _train_custom_model_internal(self: SurvivalLearner, model_instance,
                                     _store=True, _emit_warnings=True, **kwargs):
        """Train a custom BaseSurvivalModel instance with CV.

        Parameters
        ----------
        model_instance : BaseSurvivalModel
            A user-provided model instance.
        _store : bool, default=True
            If True, store result in _state.models dict.
        _emit_warnings : bool, default=True
            If True, emit the ignored-kwargs and n_jobs-mismatch suggestions.
            Set to False for internal tuned reconstruction, where these have
            already fired (or are irrelevant) for the original train() call.
        **kwargs :
            Ignored (model is already configured).
        """
        if kwargs and _emit_warnings:
            warnings.warn(
                f"**kwargs ({', '.join(kwargs.keys())}) are ignored when training "
                f"a custom model instance. Configure parameters on the model "
                f"instance directly before passing it to train().",
                SuggestionWarning,
                # stacklevel=4: user -> train() -> _try_train_single -> here
                stacklevel=4,
            )

        model_name = getattr(model_instance, 'name', None) or 'custom'

        if _emit_warnings:
            self._warn_custom_njobs_mismatch(model_instance, model_name)

        model = clone(model_instance)
        pipeline = self._build_pipeline(model)
        effective_scoring = self.scoring
        prefer_sequential = getattr(model_instance, 'prefer_sequential_cv', False)
        cv_n_jobs = 1 if prefer_sequential else self.n_jobs

        folds = self._get_cv_folds()
        fold_results = self._run_cv_folds(
            folds, pipeline, effective_scoring, cv_n_jobs, model_name,
        )
        scores, cv_metrics = self._aggregate_fold_results(fold_results, len(folds))

        final_pipeline = self._build_pipeline(clone(model_instance))
        with warnings.catch_warnings(record=True) as caught_final:
            warnings.simplefilter("always")
            final_pipeline.fit(self.X_train, self.y_train)
        _replay_captured_warnings(
            caught_final,
            seen_messages=self._state.fit_failure_warned,
            dedup_category=FitFailureWarning,
        )

        return self._finalize_model_result(
            model_name=model_name,
            final_pipeline=final_pipeline,
            scores=scores,
            cv_metrics=cv_metrics,
            effective_scoring=effective_scoring,
            is_tuned=False,
            store=_store,
        )

    def _finalize_model_result(
        self: SurvivalLearner,
        model_name,
        final_pipeline,
        scores,
        cv_metrics,
        effective_scoring,
        is_tuned,
        store,
    ):
        """Build ModelResult with CI statistics and optionally store it.

        Shared by _train_model_internal and _train_custom_model_internal to
        avoid duplicating the TimeSeriesSplit CI warning (fold scores are
        not i.i.d.), per-metric CI computation, ModelResult construction,
        and storage logic.

        Parameters
        ----------
        model_name : str
            Model identifier.
        final_pipeline : Pipeline
            Fitted pipeline for predictions.
        scores : list of float
            Primary metric scores per CV fold.
        cv_metrics : dict of {metric_name: list of float}
            Per-fold scores for each metric.
        effective_scoring : str
            Scoring metric used for CV evaluation.
        is_tuned : bool
            Whether this model was created via tune().
        store : bool
            If True, store result in _state.models dict.

        Returns
        -------
        ModelResult
        """
        from sklearn.model_selection import TimeSeriesSplit

        cv_ci_level = self.cv_ci

        # Warn once that t-distribution CI assumes independently and
        # identically distributed (i.i.d.) fold scores
        if isinstance(self._state.cv_splitter, TimeSeriesSplit):
            if not self._state.timeseries_ci_warned:
                warnings.warn(
                    "CV confidence intervals assume independently and "
                    "identically distributed fold scores. TimeSeriesSplit "
                    "uses an expanding training window, so fold scores are "
                    "not identically distributed — interpret CI with caution.",
                    MethodologicalWarning,
                    stacklevel=4,
                )
                self._state.timeseries_ci_warned = True

        # Build cv_metrics summary dict with CI for each metric
        cv_metrics_summary = {}
        for metric_name, metric_scores in cv_metrics.items():
            if len(metric_scores) > 0:
                arr = np.array(metric_scores)
                mean, std = _finite_mean_std(arr)
                ci_lo, ci_hi = _compute_cv_ci(arr, cv_ci_level)
                cv_metrics_summary[metric_name] = {
                    'scores': arr,
                    'mean': mean,
                    'std': std,
                    'ci_lower': ci_lo,
                    'ci_upper': ci_hi,
                }

        cv_metric_mean, cv_metric_std = _finite_mean_std(scores)
        cv_ci_lower, cv_ci_upper = _compute_cv_ci(scores, cv_ci_level)

        # Warn if non-finite fold scores were discarded from CI computation
        n_total = len(scores)
        n_finite = int(np.sum(np.isfinite(scores)))
        if n_finite < n_total:
            warnings.warn(
                f"{n_total - n_finite} of {n_total} CV fold scores for "
                f"'{model_name}' were non-finite and excluded from CI "
                f"computation (used {n_finite} scores).",
                MethodologicalWarning,
                stacklevel=4,
            )

        result = ModelResult(
            name=model_name,
            model=final_pipeline,
            cv_metric_scores=np.array(scores),
            cv_metric_mean=cv_metric_mean,
            cv_metric_std=cv_metric_std,
            cv_metrics=cv_metrics_summary if cv_metrics_summary else None,
            cv_ci_lower=cv_ci_lower,
            cv_ci_upper=cv_ci_upper,
            cv_ci_level=cv_ci_level,
            is_tuned=is_tuned,
        )
        result.scoring_metric = effective_scoring

        if store:
            if model_name in self._state.models:
                self._warn_overwrite(model_name)
            self._state.models[model_name] = result
        return result

    def _warn_overwrite(self: SurvivalLearner, model_name):
        """Warn that *model_name*'s stored result is being replaced.

        Single entry point for every path that overwrites
        ``_state.models[model_name]`` in place: ``train()``/``run()``
        re-train and ``tune()``'s ``_install_tuned_result``. Replacing the
        result drops the old ``ModelResult`` and everything riding on it
        (recalibrator, importance-test and risk-group-test tables), so no
        separate state-dropping is needed — the warning is all that remains.

        The replacement pipeline produces a different S(t|x), so the old
        recalibrator would now miscalibrate and the old p-values described the
        old pipeline; that is why all of it goes.
        """
        warnings.warn(
            f"Existing model '{model_name}' and all results associated with "
            "it will be overwritten.",
            SuggestionWarning,
        )

    def _resolve_model_preprocessing(
        self: SurvivalLearner, model_name, run_preprocessing=None,
    ):
        """Resolve effective preprocessing components for a model.

        Resolution order (highest to lowest priority):
        1. ``run_preprocessing[model_name]`` (run-time override)
        2. ``_state.model_preprocessing[model_name]`` (setup-time override)
        3. Global defaults (``_state.imputer``, ``_state.scaler``,
           ``_state.feature_selector``)

        Returns
        -------
        tuple of (imputer, scaler, feature_selector)
        """
        imputer = self._state.imputer
        scaler = self._state.scaler
        selector = self._state.feature_selector

        # Layer 1: setup() per-model overrides
        if model_name is not None and model_name in self._state.model_preprocessing:
            setup_overrides = self._state.model_preprocessing[model_name]
            if 'imputer' in setup_overrides:
                imputer = setup_overrides['imputer']
            if 'scaler' in setup_overrides:
                scaler = setup_overrides['scaler']
            if 'feature_selector' in setup_overrides:
                selector = setup_overrides['feature_selector']

        # Layer 2: run()/train() per-model overrides (highest priority)
        if run_preprocessing and model_name is not None and model_name in run_preprocessing:
            run_overrides = run_preprocessing[model_name]
            if 'imputer' in run_overrides:
                imputer = run_overrides['imputer']
            if 'scaler' in run_overrides:
                scaler = run_overrides['scaler']
            if 'feature_selector' in run_overrides:
                selector = run_overrides['feature_selector']

        return imputer, scaler, selector

    def _build_pipeline(
        self: SurvivalLearner, model, model_name=None, run_preprocessing=None,
    ):
        """Build sklearn Pipeline with per-model preprocessing resolution.

        Parameters
        ----------
        model : estimator
            The survival model to wrap.
        model_name : str, optional
            Model name for per-model preprocessing lookup.
        run_preprocessing : dict, optional
            Run-time per-model preprocessing overrides (resolved transformers).
        """
        imputer, scaler, selector = self._resolve_model_preprocessing(
            model_name, run_preprocessing,
        )
        optional = [
            ('imputer', imputer),
            ('scaler', scaler),
            ('selector', selector),
        ]
        steps = [(name, clone(comp)) for name, comp in optional if comp is not None]
        steps.append(('model', model))
        pipeline = Pipeline(steps)
        pipeline.set_output(transform="pandas")
        return pipeline

    def run(
        self: SurvivalLearner,
        models: str | list[str] | None = None,
        exclude: list[str] | None = None,
        tune: bool = True,
        n_trials: int | str | dict = 'auto',
        sort_by: str = "c_index",
        search_spaces: dict | None = None,
        experiment_name: str | None = None,
        verbose: bool = True,
        optuna_verbose: bool = False,
        resume: bool = False,
        # Evaluation parameters
        evaluate: bool = True,
        eval_times: np.ndarray | list | None = None,
        n_importance_repeats: int | str | dict = "auto",
        include_shap: bool = False,
        shap_max_background: int = 50,
        shap_max_explain: int = 50,
        show_eval_progress: bool = True,
        show_progress_bar: bool = True,
        # Feature interactions
        include_interactions: bool = False,
        interaction_top_k: int = 5,
        interaction_max_samples: int = 100,
        interaction_method: str = 'hstat',
        # Bootstrap parameters
        bootstrap: bool = True,
        n_bootstrap: int | str = "auto",
        bootstrap_metrics: list | None = None,
        # Per-model configuration
        model_params: dict | None = None,
        preprocessing: dict | None = None,
        calibrate=_CALIBRATE_UNSET,
        **kwargs
    ) -> dict[str, ModelResult]:
        """
        Run full survival analysis pipeline.

        Cross-validates and compares multiple models, optionally tunes them,
        evaluates on test set, and computes bootstrap confidence intervals.

        Parameters
        ----------
        models : str or list, optional
            Models to compare. Accepts a single model name (str) or a list
            of model names. Default: all available models.
        exclude : list, optional
            Models to exclude.
        tune : bool, default=True
            If True, run Optuna tuning for each model.
        n_trials : int, str, or dict, default='auto'
            Optuna trials per model. ``0`` skips tuning and uses default
            hyperparameters (with ``resume=True``, loads an existing study
            instead). Pass a dict of {model_name: int} for per-model trial
            counts; models not in the dict fall back to 'auto'.
        sort_by : str, default='c_index'
            Metric to sort results by.
        search_spaces : dict, optional
            Two accepted forms. A **component-structured** dict whose top-level
            keys are component names (``model``/``feature_selector``/``imputer``/
            ``scaler``), e.g. ``{'feature_selector': {'k': (5, 20)}}``, is applied
            to every model. A **per-model** dict whose top-level keys are model
            names, e.g. ``{'coxph': {'feature_selector': {'k': (5, 20)}}}``, gives
            each model its own space; keys matching no model in the run are
            reported with a warning. Mixing the two forms raises ``ValueError``.
            To search the feature count without a dict, set the selector's
            ``k='tune'`` (see ``TopKCoxSelector``).
        experiment_name : str, optional
            Experiment name for Optuna studies. Overrides setup() default.
        verbose : bool, default=True
            If True, print comparison table.
        optuna_verbose : bool, default=False
            If True, show trial-by-trial Optuna output (INFO level).
            If False, suppress Optuna logging to WARNING level only.
        resume : bool, default=False
            If True, continue from existing Optuna study trials instead of
            starting fresh. If False (default), any existing Optuna studies
            are deleted before starting fresh. Only meaningful when
            ``tune=True``. With ``n_trials=0`` and ``resume=True``, loads
            best parameters from an existing study without running new
            trials.
        evaluate : bool, default=True
            If True, automatically evaluate all models on test set after
            comparison. Set to False to skip evaluation.
        eval_times : array-like, optional
            Timepoints for AUC(t) calculation.
        n_importance_repeats : int, "auto", or dict, default="auto"
            Number of permutation importance repeats.  When ``"auto"``
            (the default), uses 10 repeats for all models.
            Pass a dict of {model_name: int} for per-model repeat counts;
            models not in the dict fall back to ``"auto"``.
        include_shap : bool, default=False
            If True, compute SHAP values.
        shap_max_background : int, default=50
            Maximum background samples for the SHAP masker/explainer.
        shap_max_explain : int, default=50
            Maximum test samples to explain with SHAP.
        show_eval_progress : bool, default=True
            Show progress messages during evaluation.
        show_progress_bar : bool, default=True
            Show the Optuna per-model tuning progress bar (only used when
            ``tune=True``).
        include_interactions : bool, default=False
            If True, compute pairwise feature interactions.
        interaction_top_k : int, default=5
            Number of top features (by permutation importance) to test.
            Produces ``C(top_k, 2)`` pairs.
        interaction_max_samples : int, default=100
            Maximum samples for interaction computation.
        interaction_method : str, default='hstat'
            Interaction method: ``'hstat'`` (Friedman's H-statistic,
            model-agnostic) or ``'shap'`` (SHAP interaction values via
            TreeExplainer, XGBoost models only).
        bootstrap : bool, default=True
            If True, compute bootstrap confidence intervals.
        n_bootstrap : int or "auto", default="auto"
            Number of bootstrap iterations. If "auto", computed as
            min(500, max(100, 10 * sqrt(n_samples))).
        bootstrap_metrics : list, optional
            Metrics for bootstrap CI. Default: all metrics.
        model_params : dict, optional
            Per-model constructor kwargs. Dict of {model_name: {param: value}}.
            Used when ``tune=False``, or when ``tune=True`` with ``n_trials=0``
            (which automatically infers ``tune=False``). Example:
            ``model_params={'rsf': {'n_estimators': 100}, 'gbsa': {'max_depth': 3}}``
        preprocessing : dict, optional
            Per-model preprocessing overrides for this call. Dict of
            ``{model_name: {component: value}}``. Overrides setup-level
            per-model preprocessing. See ``setup()`` for details.
        calibrate : unset, bool, str, or dict, default=unset
            Attach an internal ``S(t|x)`` calibrator to each model, fit on
            held-out OOF of the training cohort and applied to evaluate,
            bootstrap, and predict (the calibration plots and metric-time
            Brier(t) panel also show the calibrated curve by default; pass
            ``raw=True`` there for the uncalibrated view). Defaults to the
            per-model
            registry (``fssvm``→``breslow_slope``, ``deephit``→``isotonic``);
            pass ``False`` for all raw, a method
            string for all models, or ``{model: method}`` per model. See
            ``train()`` for the full grammar. The calibrators are monotone, so
            the C-index is preserved.
        **kwargs
            Model-specific hyperparameters for single-model convenience.
            Only applied when a single model is specified with ``tune=False``
            (or ``tune=True`` with ``n_trials=0``, which infers ``tune=False``).
            Cannot be combined with ``model_params``.

        Returns
        -------
        dict of {str: ModelResult}
            Trained/tuned models keyed by name.  Each ModelResult is
            progressively enriched with evaluation and bootstrap data
            when ``evaluate=True`` / ``bootstrap=True``.

        Examples
        --------
        >>> # Full pipeline with tuning
        >>> learner = SurvivalLearner(X, y)
        >>> results = learner.run(tune=True)
        >>>
        >>> # Quick comparison without tuning
        >>> results = learner.run(tune=False)
        >>>
        >>> # Single model (string convenience)
        >>> results = learner.run('rsf', tune=False, n_estimators=100)
        >>>
        >>> # Per-model hyperparameters
        >>> results = learner.run(['rsf', 'gbsa'], tune=False, model_params={
        ...     'rsf': {'n_estimators': 100},
        ...     'gbsa': {'n_estimators': 50, 'max_depth': 3},
        ... })
        >>>
        >>> # Per-model trial counts
        >>> results = learner.run(tune=True, n_trials={'rsf': 100, 'coxph': 20})
        >>>
        >>> # Resume tuning from a previous run
        >>> results = learner.run(tune=True, resume=True, experiment_name='my_exp')
        >>>
        >>> # Skip tuning but apply specific params
        >>> results = learner.run('deephit', n_trials=0,
        ...     model_params={'deephit': {'learning_rate': 0.01}})
        """
        self._ensure_setup()

        from ._setup import _validate_preprocessing, _resolve_per_model_preprocessing
        _validate_preprocessing(preprocessing, context='run')
        resolved_preprocessing = _resolve_per_model_preprocessing(
            preprocessing, random_state=self.random_state
        )

        # Normalize models: accept str for single-model convenience
        if isinstance(models, str):
            models = [models]

        # Infer tune=False when n_trials=0 + model_params/kwargs:
        # user's intent is "skip tuning, use these params directly"
        if tune and n_trials == 0 and not resume and (model_params or kwargs):
            tune = False
            warnings.warn(
                "Inferring tune=False because n_trials=0 and model_params were "
                "provided. Your model_params will be applied directly.",
                SuggestionWarning,
                stacklevel=2,
            )

        model_params, kwargs = self._resolve_run_model_params(models, tune, model_params, kwargs)

        # Default to all models
        if models is None:
            models = list_models(include_unavailable=False)

        # Apply exclusions
        if exclude is not None:
            models = [m for m in models if m not in exclude]

        if tune and verbose:
            self._emit_tuning_summary(models, n_trials, search_spaces, experiment_name, optuna_verbose)

        # Train/tune all models
        model_results = self._run_model_comparison(
            models, tune, n_trials, search_spaces, experiment_name,
            optuna_verbose,
            model_params=model_params,
            resume=resume,
            preprocessing=resolved_preprocessing,
            show_progress_bar=show_progress_bar,
        )

        # Attach internal calibrators before evaluate/bootstrap so the reported
        # metrics reflect the calibrated (deployed) S(t|x).
        self._attach_calibrators(model_results, calibrate)

        if verbose:
            self._print_comparison_table(model_results, tuned=tune)

        # Auto-evaluate if requested (with optional bootstrap)
        if evaluate:
            self._post_run_evaluation(
                model_results, eval_times, n_importance_repeats, include_shap,
                shap_max_background, shap_max_explain,
                show_eval_progress, verbose, bootstrap, n_bootstrap,
                bootstrap_metrics,
                include_interactions=include_interactions,
                interaction_top_k=interaction_top_k,
                interaction_max_samples=interaction_max_samples,
                interaction_method=interaction_method,
            )

        # Return shallow copy so callers can't corrupt internal state.
        # ModelResult objects are shared references (intentional — progressive
        # enrichment by evaluate/bootstrap still works).
        return dict(self._state.models)

    def _check_cv_vs_holdout_best(self: SurvivalLearner, models, verbose=True):
        """Check if CV-selected best model differs from holdout best and inform user.

        This prints an informational message when the model with the highest
        holdout C-index differs from the CV-selected best model. The CV-selected
        model remains the principled choice - the holdout is for unbiased
        estimation, not model selection.

        Parameters
        ----------
        models : dict of {str: ModelResult}
            Trained models with evaluation results.
        """
        if not verbose:
            return

        # Get holdout C-indices for each model
        holdout_scores = {}
        for model_name, model_result in models.items():
            if model_result.evaluation is not None:
                c_test = model_result.evaluation.c_index_test
                if c_test is not None:
                    holdout_scores[model_name] = c_test

        if not holdout_scores:
            return

        # Find holdout best and CV best
        holdout_best = max(holdout_scores, key=holdout_scores.get)

        def _cv_score(name):
            score = models[name].cv_metric_mean
            return score if score is not None else -float('inf')
        cv_best = max(models, key=_cv_score)

        if holdout_best != cv_best:
            cv_score = models[cv_best].cv_metric_mean
            holdout_best_score = holdout_scores[holdout_best]
            cv_best_holdout = holdout_scores.get(cv_best, float('nan'))
            print(
                f"\nNote: Holdout test shows '{holdout_best}' ({holdout_best_score:.3f}) "
                f"outperforming CV-selected '{cv_best}' (CV={cv_score:.3f}, "
                f"holdout={cv_best_holdout:.3f}).\n"
                f"The CV-selected model remains the principled choice - "
                f"holdout is for unbiased estimation, not model selection."
            )

    def _print_comparison_table(self: SurvivalLearner, model_results, tuned=False):
        """Print best hyperparameters (if tuned) and CV comparison table."""
        dn = self._display_name_map()
        if tuned:
            print_multi_model_best_params(model_results, display_names=dn)
        n_folds = (
            len(next(iter(model_results.values())).cv_metric_scores)
            if model_results else 0
        ) or self._state.cv_folds or 5
        # The header's tuning annotation derives from per-model is_tuned flags
        # (handles the mixed case where only some models were tuned).
        print_multi_model_cv_table(
            model_results, n_folds, display_names=dn,
            cv_repeats=self._state.cv_repeats,
        )
