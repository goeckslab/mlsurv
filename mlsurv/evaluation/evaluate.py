"""Evaluation functions for survival models."""

import contextlib
import dataclasses
import re
import time as _time
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .._calibration_thresholds import (
    CAL_SLOPE_HIGH,
    CAL_SLOPE_LOW,
)
from .._warnings import MethodologicalWarning, SuggestionWarning, _warn_eval_times_failure
from ..results.containers import EvaluationResult
from ..results._containers_helpers import _merge_optional_dicts, _resolve_interaction_method
from ..utils import (
    SurvivalArray, auto_eval_times, clip_eval_times, union_extra_eval_times,
    adjust_model_njobs, save_model_njobs, restore_model_njobs,
)
from .bootstrap import bootstrap as run_bootstrap
from .helpers import resolve_n_bootstrap, _has_internal_parallelism, _format_elapsed
from ..logging_utils import get_logger
from ._config import EvalConfig
from ._concordance import _compute_concordance, _build_test_set_df, _compute_auc_t
from ._interactions import (
    _compute_h_statistic, _compute_shap_interactions,
    _extract_tree_estimator, _is_linear_model, _select_interaction_features,
)
from ._shap import _compute_shap, _group_shap_values, _ShapDimensionError
from .importance import (
    _compute_permutation_importance, _empty_importance_results,
    _validate_categorical_groups,
)

logger = get_logger("evaluate")

DEFAULT_N_IMPORTANCE_REPEATS = 10

# ---------------------------------------------------------------------------
# SHAP / interaction exception tuple
# ---------------------------------------------------------------------------
# Progress-printing guard for SHAP and interaction computation.
# These exceptions are caught to print "failed" timing, then ALWAYS re-raised.
# Do NOT narrow to ``except Exception`` — see CONTRIBUTING.md.
_SHAP_EXCEPTIONS = (
    ImportError, ValueError, TypeError, np.linalg.LinAlgError,
    KeyError, NotImplementedError, RuntimeError, _ShapDimensionError,
)


def _validate_repeats_value(val, label="n_importance_repeats"):
    """Validate a single importance-repeats value (int or ``"auto"``).

    Parameters
    ----------
    val : int, str, or other
        Value to validate.
    label : str
        Context label for error messages (e.g., ``"n_importance_repeats['rsf']"``).

    Raises
    ------
    ValueError
        If *val* is a string other than ``"auto"`` or a negative integer.
    TypeError
        If *val* is a bool or an unsupported type.
    """
    if isinstance(val, str):
        if val.lower() != "auto":
            raise ValueError(
                f"{label} must be a non-negative integer or 'auto', got {val!r}"
            )
    elif isinstance(val, bool):
        raise TypeError(
            f"{label} must be an int or 'auto', got {type(val).__name__}"
        )
    elif isinstance(val, int):
        if val < 0:
            raise ValueError(f"{label} must be >= 0, got {val}")
    else:
        raise TypeError(
            f"{label} must be an int or 'auto', got {type(val).__name__}"
        )


def _resolve_repeats_for_model(repeats_dict, model_name):
    """Resolve a per-model entry from a dict of n_importance_repeats.

    Parameters
    ----------
    repeats_dict : dict
        Mapping of model names to repeat counts or ``"auto"``.
    model_name : str
        Model key to look up.

    Returns
    -------
    int
        Resolved repeat count.
    """
    raw = repeats_dict.get(model_name, "auto")
    if isinstance(raw, str) and raw.lower() == "auto":
        return DEFAULT_N_IMPORTANCE_REPEATS
    return raw


def _find_changed_params(actual: dict, defaults: dict) -> List[str]:
    """Return list of 'name=value' strings for params that differ from defaults."""
    return [
        f"{name}={actual[name]}"
        for name in defaults
        if actual[name] != defaults[name]
    ]


# Default values for bootstrap parameters (used to detect if user changed them).
# Must match evaluate() signature defaults.
_BOOTSTRAP_DEFAULTS = {
    'n_bootstrap': "auto",
    'bootstrap_ci': 0.95,
    'bootstrap_metrics': None,
    'bootstrap_stratify': False,
}


def _warn_unused_bootstrap_params(
    bootstrap, n_bootstrap, bootstrap_ci, bootstrap_metrics, bootstrap_stratify,
    stacklevel=3
):
    """Warn if bootstrap parameters were specified but bootstrap=False.

    Parameters
    ----------
    bootstrap : bool
        Whether bootstrapping is enabled.
    n_bootstrap, bootstrap_ci, bootstrap_metrics, bootstrap_stratify
        Bootstrap parameter values to check against defaults.
    stacklevel : int
        Stack level for warning.  Default 3 points to the public
        ``evaluate()`` boundary for direct calls; pass 5 from the
        learner mixin to point to ``EvaluationMixin.evaluate()``.
    """
    if bootstrap:
        return
    actual = {
        'n_bootstrap': n_bootstrap, 'bootstrap_ci': bootstrap_ci,
        'bootstrap_metrics': bootstrap_metrics, 'bootstrap_stratify': bootstrap_stratify,
    }
    changed = _find_changed_params(actual, _BOOTSTRAP_DEFAULTS)
    if changed:
        warnings.warn(
            f"Bootstrap parameters ({', '.join(changed)}) were specified "
            "but bootstrap=False. Set bootstrap=True to enable bootstrapping.",
            SuggestionWarning,
            stacklevel=stacklevel,
        )


# Derive SHAP defaults from EvalConfig to keep them in sync automatically.
_SHAP_DEFAULTS = {
    f.name: f.default
    for f in dataclasses.fields(EvalConfig)
    if f.name in ('shap_max_background', 'shap_max_explain')
}


def _warn_unused_shap_params(
    include_shap, shap_max_background, shap_max_explain, stacklevel=3
):
    """Warn if SHAP parameters were specified but include_shap=False.

    Parameters
    ----------
    include_shap : bool
        Whether SHAP computation is enabled.
    shap_max_background, shap_max_explain
        SHAP parameter values to check against defaults.
    stacklevel : int
        Stack level for warning.  Default 3 points to the public
        ``evaluate()`` boundary for direct calls; pass 5 from the
        learner mixin to point to ``EvaluationMixin.evaluate()``.
    """
    if include_shap:
        return
    actual = {
        'shap_max_background': shap_max_background,
        'shap_max_explain': shap_max_explain,
    }
    changed = _find_changed_params(actual, _SHAP_DEFAULTS)
    if changed:
        warnings.warn(
            f"SHAP parameters ({', '.join(changed)}) were specified "
            "but include_shap=False. Set include_shap=True to enable SHAP.",
            SuggestionWarning,
            stacklevel=stacklevel,
        )


# Derive interaction defaults from EvalConfig to keep them in sync.
_INTERACTION_DEFAULTS = {
    f.name: f.default
    for f in dataclasses.fields(EvalConfig)
    if f.name in ('interaction_top_k', 'interaction_max_samples',
                  'interaction_method', 'interaction_scale')
}


def _warn_unused_interaction_params(
    include_interactions, interaction_top_k, interaction_max_samples,
    interaction_method='hstat', interaction_scale='auto', stacklevel=3,
):
    """Warn if interaction parameters were specified but include_interactions=False."""
    if include_interactions:
        return
    actual = {
        'interaction_top_k': interaction_top_k,
        'interaction_max_samples': interaction_max_samples,
        'interaction_method': interaction_method,
        'interaction_scale': interaction_scale,
    }
    changed = _find_changed_params(actual, _INTERACTION_DEFAULTS)
    if changed:
        warnings.warn(
            f"Interaction parameters ({', '.join(changed)}) were specified "
            "but include_interactions=False. Set include_interactions=True "
            "to enable feature interaction detection.",
            SuggestionWarning,
            stacklevel=stacklevel,
        )


def evaluate(
    model,
    X_test,
    y_test: SurvivalArray,
    X_train=None,
    y_train: Optional[SurvivalArray] = None,
    eval_times=None,
    extra_eval_times=None,
    n_importance_repeats: Union[int, str] = "auto",
    importance_max_samples: Optional[int] = None,
    include_shap=False,
    shap_max_background=50,
    shap_max_explain=50,
    custom_metrics=None,
    random_state=None,
    model_name=None,
    show_progress=True,
    # Bootstrap parameters
    bootstrap=False,
    n_bootstrap="auto",
    bootstrap_ci=0.95,
    bootstrap_metrics=None,
    bootstrap_stratify=False,
    include_predictions=False,
    # Parallel computing
    n_jobs=1,
    # Internal: warning stacklevel (3 for direct calls, 5 for learner mixin)
    _warn_stacklevel=3,
    # Subpopulation analysis
    subpopulation_labels=None,
    # Categorical group importance
    categorical_groups=None,
    categorical_features=None,
    # Feature interactions
    include_interactions=False,
    interaction_top_k=5,
    interaction_max_samples=100,
    interaction_method='hstat',
    interaction_scale='auto',
    # IPCW concordance tau (evaluation horizon)
    ipcw_tau: Optional[float] = None,
    # IPCW G_hat fitting cohort override
    ipcw_y_train: Optional[SurvivalArray] = None,
    # Equal-frequency bin count for the GND calibration test.
    gnd_n_bins: int = 10,
    # Per-model survival-probability transforms (e.g. recalibration).
    # Dict keyed by model name; values are callables ``f(surv_probs, eval_times)
    # -> ndarray`` applied immediately after the model produces S(t|x). Used
    # by validate() to apply post-hoc isotonic recalibration before metrics.
    _surv_probs_transforms: Optional[dict] = None,
    # Names of models that have an internal train(calibrate=...) calibrator
    # attached. Routes their residual drift to the residual_after_calibration
    # branch (advise trying a different calibration method).
    _calibrated_models: Optional[frozenset] = None,
    # Skip calibration-drift assessment entirely.
    _skip_drift_warning: bool = False,
    # Skip the unused-parameter warnings (bootstrap/shap/interaction).
    _skip_param_warnings: bool = False,
):
    """
    Evaluate fitted model(s) on test data.

    Supports both single models and dictionaries of models. When a dict is
    provided, results are aggregated with model_type column identifying each.

    Parameters
    ----------
    model : BaseSurvivalModel or Dict[str, BaseSurvivalModel]
        Fitted model to evaluate, or dictionary of {model_name: model} pairs
        for multi-model evaluation.
    X_test : DataFrame
        Test features.
    y_test : SurvivalArray
        Test survival outcomes.
    X_train : DataFrame, optional
        Training features (for train C-index computation).
    y_train : SurvivalArray, optional
        Training outcomes (for IPCW metrics and train C-index).
    eval_times : array-like, optional
        Timepoints for AUC(t). Auto-computed as quantiles if None.
    n_importance_repeats : int, "auto", or dict, default="auto"
        Number of permutation importance repeats per feature.  When
        ``"auto"`` (the default), uses 10 repeats for all models.
        Pass a dict of {model_name: int} for per-model repeat counts;
        models not in the dict fall back to ``"auto"``.
    importance_max_samples : int, optional
        Cap on the test rows used for permutation importance. ``None``
        (default) uses every row. A value at or above the cohort size is a
        no-op; the minimum is 2. The draw is seeded from *random_state*, so
        with a seed set every model sees the same rows; without one, each
        model draws its own subsample and mlsurv warns.
    include_shap : bool, default=False
        If True, compute SHAP values (requires shap package).
    shap_max_background : int, default=50
        Maximum background samples for the SHAP masker/explainer.
    shap_max_explain : int, default=50
        Maximum test samples to explain with SHAP. When
        ``len(X_test) > shap_max_explain``, a random subset is drawn.
    custom_metrics : dict, optional
        Additional metrics to compute. Dict of {name: scorer_fn}.
        Scorer signature: scorer(estimator, X, y) -> float
    random_state : int, optional
        Seed for permutation importance reproducibility.
    model_name : str, optional
        Explicit model name to use. If None, inferred from model.
        Ignored when model is a dict (dict keys are used as names).
    show_progress : bool, default=True
        Show progress bar during permutation importance calculation.
    bootstrap : bool, default=False
        If True, run bootstrap evaluation to compute confidence intervals.
    n_bootstrap : int or "auto", default="auto"
        Number of bootstrap iterations. If "auto", computed as
        min(500, max(100, 10 * sqrt(n_samples))).
    bootstrap_ci : float, default=0.95
        Confidence level for bootstrap intervals (only used if bootstrap=True).
    bootstrap_metrics : list of str, optional
        Metrics to compute during bootstrap. Default: all available metrics.
        Options: 'harrell', 'ipcw', 'integrated_brier', 'ctd'.
    bootstrap_stratify : bool, default=False
        Stratify bootstrap resampling by event status (only used if bootstrap=True).
    include_predictions : bool, default=False
        Store per-iteration risk scores in ``BootstrapResult.predictions``.
        Required for ROC CI bands in reports. Only used when bootstrap=True.
    n_jobs : int, default=1
        Number of parallel jobs for permutation importance.
        - 1: Sequential execution (default, no overhead)
        - -1: Use all available CPU cores
        - n > 1: Use n CPU cores

        For permutation importance, parallelism is applied across features.
        Models with internal parallelism (RSF, XGBCox) always use n_jobs=1 for
        the outer importance loop to avoid nested parallelism; the model handles
        parallelism internally.  For other models (CoxPH, CoxNet, etc.), n_jobs>1
        can significantly speed up importance computation when the number of
        features is large (>~15) but may add overhead for small feature sets.
        Multi-model evaluation is always sequential to provide rich per-model
        progress output.
    subpopulation_labels : dict of {name: array}, optional
        Subpopulation labels aligned with ``X_test``. When provided,
        per-subgroup metrics are computed and attached to the result.
    categorical_groups : dict of {str: list of str}, optional
        Mapping from an original categorical feature name to its encoded
        (e.g. one-hot) column names, so permutation importance permutes all
        encoded columns of a categorical feature together.
    categorical_features : list of str, optional
        Single columns to treat as categorical/discrete in interaction PDP
        surfaces. None (default) uses a cardinality heuristic; a list disables
        it and marks exactly those columns categorical.
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
    interaction_scale : str, default='auto'
        Axis spacing for the report's 2D interaction PDP surfaces:
        ``'auto'`` picks ``'linear'``/``'log'``/``'log1p'`` per axis from each
        feature's right-skew (so skewed features read clearly instead of
        crushing into a corner); pass ``'linear'`` to keep all axes linear, or
        ``'log'``/``'log1p'`` to force a spacing everywhere. Negative-valued and
        categorical axes always stay linear.
    ipcw_tau : float, optional
        Evaluation horizon τ for Uno's IPCW concordance, passed through to
        :func:`sksurv.metrics.concordance_index_ipcw`. When unset, a
        conservative default is auto-selected (see
        :func:`mlsurv.evaluation.metrics.compute_safe_ipcw_tau`) to avoid
        unstable estimates when test follow-up extends beyond training
        censoring support.
    ipcw_y_train : SurvivalArray, optional
        Cohort whose censoring distribution fits Uno's IPCW ``G_hat``
        estimator; overrides ``y_train`` for that purpose when given.
    gnd_n_bins : int, default=10
        Number of equal-frequency bins for the Greenwood-Nam-D'Agostino (GND)
        calibration test. The test requires >=5 samples per bin, so it returns
        NaN when ``len(X_test) < 5 * gnd_n_bins`` (e.g. the default 10 bins
        needs N>=50). Lower this (keep it <= N/5) to allow GND to be computed
        on small cohorts at the cost of coarser calibration granularity; very
        small or tied-prediction cohorts may still return NaN if fewer than 2
        bins remain usable. The default of 10 follows Demler et al. 2015.

    Returns
    -------
    EvaluationResult
        Complete evaluation results with all DataFrames. If bootstrap=True,
        includes bootstrap attribute with BootstrapResult.

    Examples
    --------
    Single model:
    >>> result = evaluate(model, X_test, y_test)
    >>> print(result.concordance)

    Multiple models:
    >>> models = {'coxph': coxph_model, 'rsf': rsf_model}
    >>> result = evaluate(models, X_test, y_test)
    >>> print(result.concordance)  # Contains both models
    """
    # Validate and resolve "auto" importance repeats (dict deferred to _evaluate_multiple)
    if isinstance(n_importance_repeats, dict):
        for model_key, val in n_importance_repeats.items():
            _validate_repeats_value(val, label=f"n_importance_repeats[{model_key!r}]")
        resolved_repeats = n_importance_repeats  # pass dict through
    else:
        _validate_repeats_value(n_importance_repeats)
        if isinstance(n_importance_repeats, str):
            resolved_repeats = DEFAULT_N_IMPORTANCE_REPEATS
        else:
            resolved_repeats = n_importance_repeats

    # Handle dict of models - evaluate each and aggregate
    if isinstance(model, dict):
        return _evaluate_multiple(
            models=model,
            _surv_probs_transforms=_surv_probs_transforms,
            _calibrated_models=_calibrated_models,
            _skip_drift_warning=_skip_drift_warning,
            _skip_param_warnings=_skip_param_warnings,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            eval_times=eval_times,
            extra_eval_times=extra_eval_times,
            n_importance_repeats=resolved_repeats,
            importance_max_samples=importance_max_samples,
            include_shap=include_shap,
            shap_max_background=shap_max_background,
            shap_max_explain=shap_max_explain,
            random_state=random_state,
            show_progress=show_progress,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            bootstrap_ci=bootstrap_ci,
            bootstrap_metrics=bootstrap_metrics,
            bootstrap_stratify=bootstrap_stratify,
            include_predictions=include_predictions,
            n_jobs=n_jobs,
            _warn_stacklevel=_warn_stacklevel,
            subpopulation_labels=subpopulation_labels,
            categorical_groups=categorical_groups,
            categorical_features=categorical_features,
            include_interactions=include_interactions,
            interaction_top_k=interaction_top_k,
            interaction_max_samples=interaction_max_samples,
            interaction_method=interaction_method,
            interaction_scale=interaction_scale,
            ipcw_tau=ipcw_tau,
            ipcw_y_train=ipcw_y_train,
            gnd_n_bins=gnd_n_bins,
        )

    # Single model: resolve dict n_importance_repeats using model_name
    single_repeats = resolved_repeats
    if isinstance(resolved_repeats, dict):
        _name = model_name or getattr(model, 'name', None) or model.__class__.__name__
        single_repeats = _resolve_repeats_for_model(resolved_repeats, _name)

    # Resolve single-model transform from the dict using model_name.
    single_transform = None
    if _surv_probs_transforms:
        _name = model_name or getattr(model, 'name', None) or model.__class__.__name__
        single_transform = _surv_probs_transforms.get(_name)

    _name_for_cal = (
        model_name or getattr(model, 'name', None) or model.__class__.__name__
    )
    single_is_calibrated = bool(
        _calibrated_models and _name_for_cal in _calibrated_models
    )

    return _evaluate_single(
        model=model,
        surv_probs_transform=single_transform,
        _model_is_calibrated=single_is_calibrated,
        _skip_drift_warning=_skip_drift_warning,
        _skip_param_warnings=_skip_param_warnings,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        eval_times=eval_times,
        extra_eval_times=extra_eval_times,
        n_importance_repeats=single_repeats,
        importance_max_samples=importance_max_samples,
        include_shap=include_shap,
        shap_max_background=shap_max_background,
        shap_max_explain=shap_max_explain,
        custom_metrics=custom_metrics,
        random_state=random_state,
        model_name=model_name,
        show_progress=show_progress,
        bootstrap=bootstrap,
        n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci,
        bootstrap_metrics=bootstrap_metrics,
        bootstrap_stratify=bootstrap_stratify,
        include_predictions=include_predictions,
        n_jobs=n_jobs,
        _warn_stacklevel=_warn_stacklevel,
        subpopulation_labels=subpopulation_labels,
        categorical_groups=categorical_groups,
        categorical_features=categorical_features,
        include_interactions=include_interactions,
        interaction_top_k=interaction_top_k,
        interaction_max_samples=interaction_max_samples,
        interaction_method=interaction_method,
        ipcw_tau=ipcw_tau,
        ipcw_y_train=ipcw_y_train,
        gnd_n_bins=gnd_n_bins,
    )


def _evaluate_single(
    model,
    X_test,
    y_test: SurvivalArray,
    X_train=None,
    y_train: Optional[SurvivalArray] = None,
    eval_times=None,
    extra_eval_times=None,
    n_importance_repeats=DEFAULT_N_IMPORTANCE_REPEATS,
    importance_max_samples=None,
    include_shap=False,
    shap_max_background=50,
    shap_max_explain=50,
    custom_metrics=None,
    random_state=None,
    model_name=None,
    show_progress=True,
    bootstrap=False,
    n_bootstrap="auto",
    bootstrap_ci=0.95,
    bootstrap_metrics=None,
    bootstrap_stratify=False,
    include_predictions=False,
    _skip_param_warnings=False,
    n_jobs=1,
    _warn_stacklevel=3,
    subpopulation_labels=None,
    categorical_groups=None,
    categorical_features=None,
    include_interactions=False,
    interaction_top_k=5,
    interaction_max_samples=100,
    interaction_method='hstat',
    interaction_scale='auto',
    ipcw_tau: Optional[float] = None,
    ipcw_y_train: Optional[SurvivalArray] = None,
    surv_probs_transform=None,
    gnd_n_bins: int = 10,
    _drift_collector=None,
    _bootstrap_failure_collector=None,
    _bootstrap_notices_collector=None,
    _calibration_nan_collector=None,
    _auc_nan_collector=None,
    _model_is_calibrated: bool = False,
    _skip_drift_warning: bool = False,
):
    """Evaluate a single fitted model on test data.

    When ``_bootstrap_failure_collector`` is provided (a dict), bootstrap
    failures are appended under ``model_name`` instead of being emitted
    inline. Used by :func:`_evaluate_multiple` to coalesce per-model
    bootstrap-failure summaries into a single cross-model emission.
    ``_bootstrap_notices_collector`` is the analogous collector for
    low-coverage AUC(t) notices, coalesced via
    :func:`_emit_multi_low_coverage_summary`.
    """
    # Check if params were changed without enabling their feature
    # Skip if called from _evaluate_multiple (which already checked)
    if not _skip_param_warnings:
        _warn_unused_bootstrap_params(
            bootstrap, n_bootstrap, bootstrap_ci, bootstrap_metrics, bootstrap_stratify,
            stacklevel=_warn_stacklevel,
        )
        _warn_unused_shap_params(
            include_shap, shap_max_background, shap_max_explain,
            stacklevel=_warn_stacklevel,
        )
        _warn_unused_interaction_params(
            include_interactions, interaction_top_k, interaction_max_samples,
            interaction_method=interaction_method,
            interaction_scale=interaction_scale,
            stacklevel=_warn_stacklevel,
        )

    # Resolve n_bootstrap if "auto"
    n_samples = len(X_test)
    n_bootstrap = resolve_n_bootstrap(n_bootstrap, n_samples, show_progress, bootstrap)

    if eval_times is None:
        # Top-level evaluate/bootstrap/validate emit a single
        # IPCW truncation warning; suppress the deeper divergence warning.
        # Degenerate inputs (e.g. <2 unique event times in y_train,
        # or all training times identical) make IPCW grids impossible.  Fall
        # back to None so time-dependent consumers (Brier/IBS, AUC(t), C^td,
        # calibration) gracefully emit NaN instead of aborting the entire
        # evaluation.
        try:
            eval_times = auto_eval_times(y_test, y_train=y_train, _quiet=True)
            if y_train is not None:
                eval_times = clip_eval_times(eval_times, y_train)
        except (ValueError, TypeError) as exc:
            _warn_eval_times_failure(
                exc, "in evaluate()",
                "Time-dependent metrics (Brier, IBS, AUC(t), C^td, "
                "calibration) will be NaN.",
            )
            eval_times = None
    elif y_train is not None:
        try:
            eval_times = clip_eval_times(eval_times, y_train)
        except (ValueError, TypeError) as exc:
            _warn_eval_times_failure(
                exc, "in evaluate()",
                "Time-dependent metrics will be NaN.",
                verb="clip_eval_times",
            )
            eval_times = None

    # Union additive report horizons (add_eval_times) into the resolved grid so
    # FUTURE computations include them. In-window only; out-of-window extras
    # were warned at registration and render as honest NA. No-op when none are
    # registered, so the point curve and the embedded bootstrap CI band still
    # land on the same grid by construction.
    eval_times = union_extra_eval_times(eval_times, extra_eval_times, y_train)

    if model_name is None:
        model_name = getattr(model, 'name', model.__class__.__name__)

    # Bundle eval config for helper calls
    config = EvalConfig(
        n_importance_repeats=n_importance_repeats,
        importance_max_samples=importance_max_samples,
        eval_times=eval_times,
        random_state=random_state,
        include_shap=include_shap,
        shap_max_background=shap_max_background,
        shap_max_explain=shap_max_explain,
        show_progress=show_progress,
        n_jobs=n_jobs,
        categorical_groups=categorical_groups,
        categorical_features=categorical_features,
        include_interactions=include_interactions,
        interaction_top_k=interaction_top_k,
        interaction_max_samples=interaction_max_samples,
        interaction_method=interaction_method,
        interaction_scale=interaction_scale,
        ipcw_tau=ipcw_tau,
        ipcw_y_train=ipcw_y_train,
        gnd_n_bins=gnd_n_bins,
    )

    # Compute all DataFrames
    dfs = _compute_eval_dataframes(
        model, X_test, y_test, X_train, y_train, custom_metrics, model_name, config,
        _warn_stacklevel=_warn_stacklevel + 1,
        surv_probs_transform=surv_probs_transform,
        _drift_collector=_drift_collector,
        _calibration_nan_collector=_calibration_nan_collector,
        _auc_nan_collector=_auc_nan_collector,
        _model_is_calibrated=_model_is_calibrated,
        _skip_drift_warning=_skip_drift_warning,
    )

    # Bootstrap evaluation (optional)
    bootstrap_result = None
    if bootstrap:
        bootstrap_result = run_bootstrap(
            model,
            X_test,
            y_test,
            y_train=y_train,
            n_bootstrap=n_bootstrap,
            metrics=bootstrap_metrics,
            ci=bootstrap_ci,
            eval_times=eval_times,
            include_auc=True,           # AUC(t) for uncertainty ribbons
            include_brier_t=True,       # Brier(t) for uncertainty ribbons
            include_predictions=include_predictions,
            stratify=bootstrap_stratify,
            random_state=random_state,
            show_progress=show_progress,
            model_name=model_name,
            n_jobs=n_jobs,
            subpopulation_labels=subpopulation_labels,
            categorical_groups=categorical_groups,
            ipcw_tau=ipcw_tau,
            ipcw_y_train=ipcw_y_train,
            # run_bootstrap fires its MetricFailureWarning three frames deeper
            # than warnings emitted from this function's own body: the chain
            # warn → _emit_failure_summary → _bootstrap_single → free bootstrap
            # → _evaluate_single body adds 3 hops below _evaluate_single.
            # ``_warn_stacklevel`` counts from _evaluate_single's body to the
            # user, so offset by +3 to attribute the warning to the user's
            # frame. Pinned by ``tests/test_phase3_learner.py::TestValidateSubpopulationsSetupBootstrap::test_metric_failure_warning_stacklevel_on_*_path``.
            _warn_stacklevel=_warn_stacklevel + 3,
            # Pass the same transform used by the headline
            # path so bootstrap CIs reflect recalibrated S(t|x). Keyed by
            # model_name because run_bootstrap accepts a per-model dict.
            surv_probs_transforms=(
                {model_name: surv_probs_transform}
                if surv_probs_transform is not None else None
            ),
            _failure_collector=_bootstrap_failure_collector,
            _notices_collector=_bootstrap_notices_collector,
        )

    # Build surv_probs cache for the public survival_probabilities property
    surv_probs = dfs.get('_surv_probs')
    surv_eval_times = dfs.get('_surv_eval_times')
    surv_cache = (
        {model_name: (surv_probs, surv_eval_times)}
        if surv_probs is not None and surv_eval_times is not None
        else None
    )

    # Build the dense IBS surv_probs cache (train-anchored grid), reused by
    # subpopulation IBS so it integrates over the same dense grid as the
    # headline metric instead of the coarse display grid.
    ibs_surv_probs = dfs.get('_ibs_surv_probs')
    ibs_times = dfs.get('_ibs_times')
    ibs_surv_cache = (
        {model_name: (ibs_surv_probs, ibs_times)}
        if ibs_surv_probs is not None and ibs_times is not None
        else None
    )

    # Build interaction raw cache for multi-model aggregation
    interaction_raw = dfs.get('interaction_values_raw')
    interaction_feat = dfs.get('interaction_feature_names')
    interaction_raw_cache = (
        {model_name: (interaction_raw, interaction_feat)}
        if interaction_raw is not None and interaction_feat is not None
        else None
    )

    return EvaluationResult(
        concordance=dfs['concordance'],
        test_set=dfs['test_set'],
        auc_t=dfs['auc_t'],
        importance=dfs['importance'],
        importance_iterations=dfs['iterations'],
        importance_per_column=dfs.get('importance_per_column'),
        shap_values=dfs['shap'],
        shap_values_per_column=dfs.get('shap_per_column'),
        interactions=dfs.get('interactions'),
        interaction_surfaces=dfs.get('interaction_surfaces'),
        interaction_values_raw=dfs.get('interaction_values_raw'),
        interaction_feature_names=dfs.get('interaction_feature_names'),
        interaction_method=dfs.get('interaction_method'),
        bootstrap=bootstrap_result,
        _surv_probs_cache=surv_cache,
        _ibs_surv_probs_cache=ibs_surv_cache,
        _interaction_raw_cache=interaction_raw_cache,
    )


def _setup_importance_parallelism(model, config):
    """Determine importance parallelism strategy.

    Models with internal parallelism (RSF, XGBCox) already parallelize
    predict(), so outer importance parallelism would cause oversubscription.
    Other models (CoxPH, CoxNet) are lightweight and benefit from outer
    parallelism across features.

    Returns
    -------
    tuple of (importance_n_jobs: int, use_outer_parallel: bool)
    """
    use_outer_parallel = (
        not _has_internal_parallelism(model) and config.n_jobs != 1
    )
    importance_n_jobs = config.n_jobs if use_outer_parallel else 1
    return importance_n_jobs, use_outer_parallel


@contextlib.contextmanager
def _model_njobs_override(model, config_n_jobs, use_outer_parallel):
    """Temporarily set model n_jobs=1 for outer-parallel importance.

    When *use_outer_parallel* is True, saves the model's current n_jobs,
    sets it to 1 (so the outer joblib loop owns parallelism), and restores
    on exit.  When False, this is a no-op context manager.
    """
    if not use_outer_parallel:
        yield
        return
    saved = save_model_njobs(model)
    try:
        adjust_model_njobs(model, config_n_jobs)
        yield
    finally:
        restore_model_njobs(model, saved)


# ---------------------------------------------------------------------------
# Phase helpers for _compute_eval_dataframes
# ---------------------------------------------------------------------------

def _run_importance_phase(model, X_test, y_test, config, model_name,
                          importance_n_jobs):
    """Compute permutation importance.

    Parameters
    ----------
    model : BaseSurvivalModel
        Fitted model (Pipeline).
    X_test : pd.DataFrame
        Test features.
    y_test : SurvivalArray
        Test outcomes.
    config : EvalConfig
        Evaluation configuration.
    model_name : str
        Model identifier for tagging output DataFrames.
    importance_n_jobs : int
        Number of parallel jobs for importance computation.

    Returns
    -------
    tuple of (importance_df, iterations_df, per_column_importance_df)
        per_column_importance_df may be None when categorical_groups is unused.
    """
    effective_repeats = config.n_importance_repeats

    per_column_importance_df = None
    if effective_repeats > 0:
        importance_df, iterations_df, per_column_importance_df = _compute_permutation_importance(
            model, X_test, y_test, effective_repeats, config.random_state,
            show_progress=config.show_progress, model_name=model_name,
            n_jobs=importance_n_jobs,
            categorical_groups=config.categorical_groups,
            max_samples=config.importance_max_samples,
        )
    else:
        importance_df, iterations_df = _empty_importance_results()
    importance_df['model_type'] = model_name
    iterations_df['model_type'] = model_name
    if per_column_importance_df is not None:
        per_column_importance_df['model_type'] = model_name
    return importance_df, iterations_df, per_column_importance_df


def _run_shap_phase(model, X_test, config, model_name, display_name, X_train=None):
    """Compute SHAP values with optional categorical grouping.

    Parameters
    ----------
    model : BaseSurvivalModel
        Fitted model (Pipeline).
    X_test : pd.DataFrame
        Test features (the cohort explained in the beeswarm).
    config : EvalConfig
        Evaluation configuration.
    model_name : str
        Model identifier for tagging output DataFrames.
    display_name : str
        Human-readable model name for progress messages.
    X_train : pd.DataFrame, optional
        Training features used as the SHAP background.  When None, the
        background falls back to ``X_test`` inside ``_compute_shap``.

    Returns
    -------
    tuple of (shap_df, shap_per_column_df)
        Both may be None when include_shap is False or grouping is unused.
    """
    shap_df = None
    shap_per_column_df = None

    if not config.include_shap:
        return shap_df, shap_per_column_df

    if config.show_progress:
        print(f"Evaluating {display_name}: computing SHAP values...", end="", flush=True)
    t_shap = _time.perf_counter()
    try:
        shap_df = _compute_shap(
            model, X_test, X_background=X_train,
            max_background=config.shap_max_background,
            max_explain=config.shap_max_explain,
            random_state=config.random_state,
        )
        shap_df['model_type'] = model_name
        if config.show_progress:
            print(f" done ({_format_elapsed(_time.perf_counter() - t_shap)}).")
    except _SHAP_EXCEPTIONS:
        if config.show_progress:
            print(f" failed ({_format_elapsed(_time.perf_counter() - t_shap)}).")
        raise

    # Group SHAP values by categorical_groups (dual-storage pattern)
    if config.categorical_groups and shap_df is not None:
        shap_features = shap_df['feature'].unique().tolist()
        valid_groups = _validate_categorical_groups(
            config.categorical_groups, shap_features,
        )
        if valid_groups:
            shap_per_column_df = shap_df.copy()
            shap_df = _group_shap_values(shap_df, valid_groups)

    return shap_df, shap_per_column_df


def _run_interactions_phase(model, X_test, importance_df, config,
                            model_name, display_name, _warn_stacklevel=2):
    """Compute feature interactions via SHAP or H-statistic.

    Parameters
    ----------
    model : BaseSurvivalModel
        Fitted model (Pipeline).
    X_test : pd.DataFrame
        Test features.
    importance_df : pd.DataFrame
        Importance scores used to select top-k features.
    config : EvalConfig
        Evaluation configuration.
    model_name : str
        Model identifier for tagging output DataFrames.
    display_name : str
        Human-readable model name for progress messages.
    _warn_stacklevel : int, default 2
        Stack depth for warnings.warn() calls, so warnings point to the
        correct caller frame (not this helper).

    Returns
    -------
    dict with keys: interactions, interaction_surfaces,
        interaction_values_raw, interaction_feature_names, interaction_method.
        All values may be None when interactions are disabled or skipped.
    """
    _NONE_RESULT = {
        'interactions': None,
        'interaction_surfaces': None,
        'interaction_values_raw': None,
        'interaction_feature_names': None,
        'interaction_method': None,
    }

    if not config.include_interactions:
        return _NONE_RESULT

    # Skip: linear model + H-statistic (always zero)
    if config.interaction_method == 'hstat' and _is_linear_model(model):
        if config.show_progress:
            print(
                f"Evaluating {display_name}: skipping interactions "
                "(linear model — H-statistic is always zero)."
            )
        return _NONE_RESULT

    # Skip: non-tree model + SHAP interactions
    if config.interaction_method == 'shap' and _extract_tree_estimator(model)[0] is None:
        warnings.warn(
            f"Skipping SHAP interactions for '{display_name}': "
            f"SHAP interaction values require a tree-based model "
            f"(xgbcox, xgbaft, gbsa, or rsf). Use method='hstat' for "
            f"model-agnostic interaction detection.",
            SuggestionWarning,
            stacklevel=_warn_stacklevel,
        )
        if config.show_progress:
            print(
                f"Evaluating {display_name}: skipping SHAP interactions "
                "(requires tree-based model)."
            )
        return _NONE_RESULT

    top_features = _select_interaction_features(
        importance_df, config.interaction_top_k, X_test.columns,
        categorical_groups=config.categorical_groups,
    )
    if len(top_features) < 2:
        warnings.warn(
            f"include_interactions=True but only {len(top_features)} "
            "feature(s) available for interaction testing (need >= 2). "
            "Interaction analysis skipped.",
            SuggestionWarning,
            stacklevel=_warn_stacklevel,
        )
        return _NONE_RESULT

    method_label = (
        "SHAP interaction values" if config.interaction_method == 'shap'
        else "feature interactions"
    )
    if config.show_progress:
        print(
            f"Evaluating {display_name}: computing {method_label} "
            f"({len(top_features)} features)..."
        )
    t_int = _time.perf_counter()
    try:
        if config.interaction_method == 'shap':
            interactions_df, interaction_values_raw, interaction_feature_names = (
                _compute_shap_interactions(
                    model, X_test, top_features,
                    max_samples=config.interaction_max_samples,
                    random_state=config.random_state,
                    show_progress=config.show_progress,
                    model_name=model_name,
                    categorical_groups=config.categorical_groups,
                )
            )
            interaction_method = 'shap'
            interaction_surfaces = None
        else:
            interactions_df, interaction_surfaces = _compute_h_statistic(
                model, X_test, top_features,
                max_samples=config.interaction_max_samples,
                random_state=config.random_state,
                show_progress=config.show_progress,
                model_name=model_name,
                compute_surfaces=True,
                categorical_groups=config.categorical_groups,
                categorical_features=config.categorical_features,
                scale=config.interaction_scale,
            )
            interaction_method = 'hstat'
            interaction_values_raw = None
            interaction_feature_names = None
        interactions_df['model_type'] = model_name
        if interaction_surfaces:
            for s in interaction_surfaces:
                s['model_type'] = model_name
        if config.show_progress:
            print(
                f"  Interactions {display_name}: "
                f"done ({_format_elapsed(_time.perf_counter() - t_int)})."
            )
    except _SHAP_EXCEPTIONS:
        if config.show_progress:
            print(
                f"  Interactions {display_name}: "
                f"failed ({_format_elapsed(_time.perf_counter() - t_int)})."
            )
        raise

    return {
        'interactions': interactions_df,
        'interaction_surfaces': interaction_surfaces,
        'interaction_values_raw': interaction_values_raw,
        'interaction_feature_names': interaction_feature_names,
        'interaction_method': interaction_method,
    }


def _format_pvalue_inline(p: float) -> str:
    """Compact p-value formatter for warning text (avoids reports import)."""
    if not np.isfinite(p):
        return "—"
    return f"{p:.3f}" if p >= 0.01 else f"{p:.2e}"


def _emit_pending_calibration_warnings(pending, stacklevel, *, collector=None):
    """Emit calibration NaN warnings collected from worker code paths.

    The cal_slope / HL / GND helpers in ``mlsurv.evaluation.metrics`` no
    longer call ``warnings.warn`` directly when invoked from
    ``_compute_concordance`` — joblib/loky workers silence those calls.
    They append ``(metric_key, model_name, msg)`` tuples instead,
    which we re-emit here from the parent process.

    Dedup uses the same module-level ``_calibration_warned`` set as the
    legacy direct-call path so process-wide dedup is preserved across
    repeat ``evaluate()`` calls (parity with the earlier behavior that
    in-worker ``warnings.warn`` had via this set).

    When ``collector`` is a list (multi-model evaluate), the per-
    model NaN-return messages are appended to it for one coalesced summary
    per metric at the end of the run. Non-NaN-return messages (e.g. the
    very-rare "only N informative samples" path) are emitted inline so
    their per-model context isn't lost in a count-style summary.

    For ``gnd_sample_size`` shortages the actionable workaround
    hint is appended on both paths: the multi-model coalesced summary adds
    it in ``_format_coalesced_calibration_nan_warning``, and the
    single-model inline path adds it here right after the per-model
    message.
    """
    if not pending:
        return
    from .metrics import _calibration_warned
    for metric_key, model_name, msg in pending:
        dedup_key = (metric_key, model_name)
        if dedup_key in _calibration_warned:
            continue
        _calibration_warned.add(dedup_key)
        # Routing by message substring: the producer in
        # ``mlsurv.evaluation.metrics`` uses ``returned NaN`` for the
        # exception-branch wording and ``returning NaN`` for early-exit
        # paths whose per-model context (sample count) would be lost in
        # a count-style summary. The contract is locked by
        # ``TestUpstreamWordingContract`` in
        # ``tests/test_coalesced_calibration_nan.py`` so a
        # producer rewording breaks loud rather than silently disabling
        # coalescing.
        #
        # Carve-out: ``gnd_sample_size`` is a cohort-keyed
        # shortage — the sample/bin math is identical across every
        # model, so the formatter can emit the math once and list all
        # affected models. Route it to the collector even though the
        # wording is ``returning NaN``.
        if collector is not None and (
            "returned NaN" in msg or metric_key == "gnd_sample_size"
        ):
            collector.append((metric_key, model_name, msg))
            continue
        warnings.warn(msg, MethodologicalWarning, stacklevel=stacklevel)
        # The single-model (no-collector) inline path also gets the
        # actionable workaround hint that the multi-model coalesced path
        # appends. ``_gnd_workaround_hint`` parses the cohort math out of
        # ``msg`` (tolerant of the leading ``"<model>: "`` prefix) and
        # returns None when no useful bin count exists at this cohort size.
        if metric_key == "gnd_sample_size":
            hint = _gnd_workaround_hint(msg)
            if hint is not None:
                warnings.warn(hint, MethodologicalWarning, stacklevel=stacklevel)


_CALIBRATION_NAN_METRIC_LABELS = {
    "cal_slope": "Calibration slope/intercept",
    "gnd": "GND calibration",
}


def _format_coalesced_calibration_nan_warning(
    collected: list, n_total: int,
) -> list:
    """Return one summary line per metric for coalesced calibration-NaN warns.

    ``collected`` is a list of ``(metric_key, model_name, msg)`` tuples
    accumulated by ``_emit_pending_calibration_warnings`` when running with
    a collector. We group by ``metric_key`` and produce one line per metric
    naming the affected models, e.g.::

        Calibration slope/intercept returned NaN for 6/6 models:
        [xgbaft, fssvm, coxnet, coxph, xgbcox, rsf].
        Enable debug logging for details.

    The ``gnd_sample_size`` key uses a dedicated line shape that
    preserves the cohort math from the producer's message verbatim::

        GND calibration: 35 samples < 50 required for 10 bins of 5;
        returning NaN for all 6/6 models: [coxph, rsf, ...].

    Order preserved by first occurrence so the output is deterministic.
    """
    grouped: dict[str, list[str]] = {}
    # Capture the first message per metric_key — the GND sample-size path
    # needs the cohort math from the producer's message verbatim.
    first_msg: dict[str, str] = {}
    for metric_key, model_name, msg in collected:
        models = grouped.setdefault(metric_key, [])
        if model_name not in models:
            models.append(model_name)
        first_msg.setdefault(metric_key, msg)

    lines = []
    for metric_key, model_names in grouped.items():
        names = ", ".join(model_names)
        if metric_key == "gnd_sample_size":
            # Preserve cohort math from upstream message. The
            # producer formats it as ``[prefix]GND calibration: N samples
            # < M required for B bins of P; returning NaN`` — strip the
            # per-model prefix and rebuild with a single trailing model
            # list. All records share the math, so the first one suffices.
            cohort_math = _strip_model_prefix(first_msg[metric_key])
            lines.append(
                f"{cohort_math} for all {len(model_names)}/{n_total} "
                f"models: [{names}]."
            )
            # Append an actionable workaround derived from the cohort
            # math — the largest gnd_n_bins that clears the >=P-per-bin floor.
            hint = _gnd_workaround_hint(cohort_math)
            if hint is not None:
                lines.append(hint)
            continue
        label = _CALIBRATION_NAN_METRIC_LABELS.get(metric_key, metric_key)
        lines.append(
            f"{label} returned NaN for {len(model_names)}/{n_total} "
            f"models: [{names}]. Enable debug logging for details."
        )
    return lines


def _format_coalesced_auc_nan_warning(collected: list, n_total: int) -> list:
    """Return one summary line for coalesced AUC(t) NaN-return warnings.

    ``collected`` is a list of ``("auc_t", model_name, msg)`` tuples appended by
    :func:`_compute_auc_t` when run with a collector. All records share the one
    ``"auc_t"`` metric, so a single line names every model that returned NaN at
    one or more requested times, e.g.::

        AUC(t) returned NaN at one or more requested time points for 2/5
        models: [coxph, coxnet].

    Order preserved by first occurrence so the output is deterministic.
    """
    model_names: list = []
    for _metric_key, model_name, _msg in collected:
        if model_name not in model_names:
            model_names.append(model_name)
    if not model_names:
        return []
    names = ", ".join(model_names)
    return [
        f"AUC(t) returned NaN at one or more requested time points for "
        f"{len(model_names)}/{n_total} models: [{names}]."
    ]


def _emit_coalesced_auc_nan_warning(
    collected: list, n_total: int, stacklevel: int,
) -> None:
    """Emit one :class:`MethodologicalWarning` summarizing AUC(t) NaN returns.

    No-op when ``collected`` is empty. On formatter failure, falls back to
    emitting each original per-model message individually so no warning is
    silently dropped (matches the calibration-NaN summary fallback pattern).
    """
    if not collected:
        return
    try:
        lines = _format_coalesced_auc_nan_warning(collected, n_total)
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        warnings.warn(
            f"Failed to format coalesced AUC(t)-NaN summary "
            f"({type(exc).__name__}: {exc}); emitting per-model warnings "
            f"instead.",
            MethodologicalWarning, stacklevel=stacklevel,
        )
        for _mk, _mn, msg in collected:
            warnings.warn(msg, MethodologicalWarning, stacklevel=stacklevel)
        return
    for line in lines:
        warnings.warn(line, MethodologicalWarning, stacklevel=stacklevel)


def _gnd_workaround_hint(cohort_math: str) -> Optional[str]:
    """Build the GND workaround hint from a cohort-math message.

    ``cohort_math`` is the prefix-stripped producer string, e.g.
    ``"GND calibration: 35 samples < 50 required for 10 bins of 5; ..."``.
    We parse the sample count ``N`` and per-bin floor ``P`` and suggest the
    largest ``gnd_n_bins`` that clears the floor: ``N // P``. Returns ``None``
    (no hint) when the numbers cannot be parsed or the suggestion would be
    ``< 2`` — a single-bin GND test is degenerate (no calibration
    granularity), so for such tiny cohorts there is no useful workaround.
    """
    m = re.search(r"(\d+) samples < \d+ required for \d+ bins of (\d+)",
                  cohort_math)
    if m is None:
        # Producer wording drifted — skip the hint rather than emit a
        # garbled one. The cohort-math line itself is still informative.
        # Mirror _strip_model_prefix: surface the drift as an
        # InternalWarning so the silent loss of the hint is observable to
        # developers/CI (off by default for end users) rather than only
        # discoverable by noticing the advisory has vanished.
        from .._warnings import InternalWarning
        warnings.warn(
            "_gnd_workaround_hint could not parse cohort math from the GND "
            "coalesced message; producer wording may have drifted. The "
            "gnd_n_bins workaround hint will be omitted.",
            InternalWarning, stacklevel=2,
        )
        return None
    n_samples, per_bin = int(m.group(1)), int(m.group(2))
    suggested = n_samples // per_bin
    if suggested < 2:
        return None
    return (
        f"Pass evaluate(gnd_n_bins={suggested}) (or <= N/{per_bin}) to "
        f"compute GND at this cohort size."
    )


def _strip_model_prefix(msg: str) -> str:
    """Drop the leading ``"<model_name>: "`` prefix from a GND msg.

    Producer in ``mlsurv.evaluation.metrics.greenwood_nam_dagostino``
    formats messages as ``f"{model_name}: GND calibration: ..."`` (or
    just ``"GND calibration: ..."`` when model_name is None). The
    coalesced summary lists model names separately, so the per-record
    prefix is redundant. We locate ``"GND calibration:"`` directly
    rather than partition on ``": "`` to avoid clipping the real
    message when no prefix is present.

    On anchor miss (producer rewording), the original message is
    returned verbatim and an :class:`InternalWarning` is emitted so
    silent garbling (e.g. duplicated prefix in the coalesced line) is
    observable when ``InternalWarning`` is enabled — even if the
    upstream wording contract test in
    ``tests/test_gnd_sample_size_coalesce.py`` is skipped.
    """
    anchor = "GND calibration:"
    idx = msg.find(anchor)
    if idx == -1:
        from mlsurv._warnings import InternalWarning
        warnings.warn(
            f"_strip_model_prefix could not locate {anchor!r} anchor in "
            f"GND coalesced message; producer wording may have drifted. "
            f"Coalesced output may include a duplicated prefix.",
            InternalWarning, stacklevel=2,
        )
        return msg
    return msg[idx:]


def _emit_coalesced_calibration_nan_warning(
    collected: list, n_total: int, stacklevel: int,
) -> None:
    """Emit one :class:`MethodologicalWarning` per metric_key.

    No-op when ``collected`` is empty. On formatter failure, falls back to
    emitting each original per-model message individually so no warning is
    silently dropped (matches the drift-summary fallback pattern).
    """
    if not collected:
        return
    try:
        lines = _format_coalesced_calibration_nan_warning(collected, n_total)
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        warnings.warn(
            f"Failed to format coalesced calibration-NaN summary "
            f"({type(exc).__name__}: {exc}); emitting per-model warnings "
            f"instead.",
            MethodologicalWarning, stacklevel=stacklevel,
        )
        for _mk, _mn, msg in collected:
            warnings.warn(msg, MethodologicalWarning, stacklevel=stacklevel)
        return
    for line in lines:
        warnings.warn(line, MethodologicalWarning, stacklevel=stacklevel)


@dataclasses.dataclass(frozen=True)
class CalibrationDriftRecord:
    """Structured drift assessment, branch-tagged for downstream formatting.

    ``branch`` ∈ {"residual_after_calibration", "score_based", "default"}.
    """
    label: str
    slope: float
    gnd: float
    branch: str


def _assess_calibration_drift(model, concordance_df, model_name, is_calibrated=False):
    """Return a :class:`CalibrationDriftRecord` if the model drifted, else None.

    Pure: no warnings emitted. The calibration **slope** is the sole trigger:
    a model is flagged only when ``|slope - 1| > SEVERE_DRIFT_THRESHOLD``
    (slope outside ``[CAL_SLOPE_LOW, CAL_SLOPE_HIGH]``). A NaN slope stays
    silent. The GND p-value is carried for context in the message text only —
    it over-rejects on well-powered cohorts, so it never flags on its own.

    When ``is_calibrated`` (the model has an internal ``train(calibrate=...)``
    calibrator attached), the metrics are already post-calibration, so any
    residual drift routes to the ``residual_after_calibration`` branch (advise
    trying a different calibration method). Otherwise score-based models route
    to ``score_based`` (breslow_slope recommended) and everything else to
    ``default``.
    """
    inner = model
    if hasattr(inner, 'named_steps'):
        inner = inner.named_steps.get('model', inner)
    if concordance_df.empty:
        return None

    def _col(name: str) -> float:
        if name in concordance_df.columns:
            return float(concordance_df[name].iloc[0])
        return float('nan')

    slope = _col('cal_slope_test')
    gnd = _col('gnd_test')

    # Slope is the sole trigger; GND is context only.
    if not np.isfinite(slope) or (CAL_SLOPE_LOW <= slope <= CAL_SLOPE_HIGH):
        return None

    label = model_name or type(inner).__name__

    if is_calibrated:
        branch = "residual_after_calibration"
    elif bool(getattr(inner, 'score_based', False)):
        branch = "score_based"
    else:
        branch = "default"

    return CalibrationDriftRecord(label=label, slope=slope, gnd=gnd, branch=branch)


def _drift_recipe(branch: str) -> str:
    """The 'how to fix it' sentence for a drift branch.

    Every branch points at ``train(calibrate=...)`` — the single supported
    way to calibrate ``S(t|x)``. ``score_based`` models get the breslow_slope
    recommendation; calibrated models with residual drift are told to try a
    different method.
    """
    if branch == "residual_after_calibration":
        return (
            "Consider setting train(calibrate=...) to a different calibration "
            "method (isotonic / lowdof / breslow_slope)."
        )
    if branch == "score_based":
        return (
            "Consider calibrating S(t|x) via "
            "train(calibrate='breslow_slope') (van Houwelingen 2000)."
        )
    return (
        "Consider calibrating S(t|x) via train(calibrate=...) "
        "(isotonic / lowdof / breslow_slope)."
    )


def _format_single_drift_warning(record: CalibrationDriftRecord) -> str:
    """Concise single-model calibration-drift suggestion."""
    slope_str = f"{record.slope:.2f}" if np.isfinite(record.slope) else "NaN"
    gnd_str = _format_pvalue_inline(record.gnd)
    target = f"target [{CAL_SLOPE_LOW:.2f}, {CAL_SLOPE_HIGH:.2f}]"
    if record.branch == "residual_after_calibration":
        head = (
            f"{record.label} residual calibration drift after calibration: "
            f"slope={slope_str} ({target}), GND p={gnd_str}."
        )
    else:
        head = (
            f"{record.label} calibration drift: slope={slope_str} ({target}), "
            f"GND p={gnd_str}."
        )
    return f"{head} {_drift_recipe(record.branch)}"


def _format_coalesced_drift_warning(records: list, n_total: int) -> str:
    """Single summary listing every flagged model + the train(calibrate=...) fix."""
    n = len(records)
    lines = [
        f"{n} of {n_total} models show calibration drift "
        f"(slope outside [{CAL_SLOPE_LOW:.2f}, {CAL_SLOPE_HIGH:.2f}]):"
    ]
    for r in records:
        slope_str = f"{r.slope:.2f}" if np.isfinite(r.slope) else "NaN"
        gnd_str = _format_pvalue_inline(r.gnd)
        marker = (
            " [calibrator attached]"
            if r.branch == "residual_after_calibration" else ""
        )
        lines.append(
            f"  - {r.label}: slope={slope_str}, GND p={gnd_str}{marker}"
        )

    residual = [r for r in records if r.branch == "residual_after_calibration"]
    fresh = [r for r in records if r.branch != "residual_after_calibration"]
    if fresh:
        names = ", ".join(r.label for r in fresh)
        lines.append(
            f"Consider calibrating S(t|x) for {names} via train(calibrate=...) "
            f"(isotonic / lowdof / breslow_slope)."
        )
    if residual:
        names = ", ".join(r.label for r in residual)
        lines.append(
            f"{names}: residual drift after calibration — consider setting "
            f"train(calibrate=...) to a different method "
            f"(isotonic / lowdof / breslow_slope)."
        )
    return "\n".join(lines)


def _warn_poor_calibration(
    model, concordance_df, model_name, _warn_stacklevel, *, collector=None,
    is_calibrated: bool = False,
):
    """Assess calibration drift; emit immediately or defer to ``collector``.

    When ``collector`` is a list, the drift record (if any) is appended
    instead of emitted — multi-model ``evaluate()`` uses this to coalesce
    per-model warnings into a single summary. Single-model evaluate()
    calls leave ``collector=None`` and emit the concise per-model warning
    inline.

    Emitted from the parent process — joblib/loky workers silence
    ``warnings.warn``. Scoping: primary ``evaluate()``
    aggregation only; ``bootstrap`` does not call this path.
    """
    record = _assess_calibration_drift(
        model, concordance_df, model_name, is_calibrated=is_calibrated,
    )
    if record is None:
        return
    if collector is not None:
        collector.append(record)
        return
    warnings.warn(
        _format_single_drift_warning(record),
        MethodologicalWarning,
        stacklevel=_warn_stacklevel,
    )


def _emit_coalesced_calibration_warning(
    records: list, n_total: int, stacklevel: int,
) -> None:
    """Emit one coalesced :class:`MethodologicalWarning` summarizing drift.

    No-op when ``records`` is empty.
    """
    if not records:
        return
    warnings.warn(
        _format_coalesced_drift_warning(records, n_total),
        MethodologicalWarning,
        stacklevel=stacklevel,
    )


def _compute_eval_dataframes(model, X_test, y_test: SurvivalArray, X_train,
                              y_train: Optional[SurvivalArray],
                              custom_metrics, model_name, config,
                              _warn_stacklevel=2, surv_probs_transform=None,
                              _drift_collector=None,
                              _calibration_nan_collector=None,
                              _auc_nan_collector=None,
                              _model_is_calibrated: bool = False,
                              _skip_drift_warning: bool = False):
    """Compute all evaluation DataFrames for a single model.

    Orchestrates: predict → concordance/auc_t → importance → SHAP →
    interactions → reorder → pack.

    Parameters
    ----------
    config : EvalConfig
    _warn_stacklevel : int, default 2
        Stack depth for warnings.warn() calls inside this function.

    Returns
    -------
    dict of DataFrames with keys: concordance, test_set, auc_t,
        importance, iterations, importance_per_column, shap,
        shap_per_column, interactions, interaction_surfaces,
        interaction_values_raw, interaction_feature_names,
        interaction_method, _surv_probs, _surv_eval_times.
    """
    importance_n_jobs, use_outer_parallel = _setup_importance_parallelism(model, config)
    display_name = model_name or "model"

    if config.show_progress:
        print(f"Evaluating {display_name}: computing metrics and predictions...", end="", flush=True)

    _t0 = _time.perf_counter()
    risk_scores = model.predict(X_test)

    n_nonfinite = int(np.sum(~np.isfinite(risk_scores)))
    if n_nonfinite:
        warnings.warn(
            f"{display_name}: {n_nonfinite}/{len(risk_scores)} non-finite "
            f"risk scores from predict(). Affected metrics will be NaN. "
            f"Consider tuning hyperparameters or using a different model.",
            MethodologicalWarning,
            stacklevel=_warn_stacklevel,
        )

    with _model_njobs_override(model, config.n_jobs, use_outer_parallel):
        # Collect calibration NaN warnings for parent-side emission:
        # ``warnings.warn`` from inside _compute_concordance is silently
        # dropped when the metric helpers run in joblib/loky workers (e.g.
        # via internal model parallelism), so we surface them here instead.
        pending_calibration_warnings: list = []
        concordance_df, _surv_probs, _surv_eval_times, _ibs_surv_probs, _ibs_times = _compute_concordance(
            model, X_test, y_test, X_train, y_train, config.eval_times, custom_metrics,
            risk_scores=risk_scores, ipcw_tau=getattr(config, 'ipcw_tau', None),
            ipcw_y_train=getattr(config, 'ipcw_y_train', None),
            model_name=model_name,
            pending_warnings=pending_calibration_warnings,
            surv_probs_transform=surv_probs_transform,
            gnd_n_bins=getattr(config, 'gnd_n_bins', 10),
        )
        concordance_df['model_type'] = model_name

        _emit_pending_calibration_warnings(
            pending_calibration_warnings, _warn_stacklevel,
            collector=_calibration_nan_collector,
        )

        # Parent-side warning — workers silence warnings.warn.
        # When a collector is supplied (multi-model evaluate), the
        # drift record is deferred to a single coalesced summary; otherwise
        # the per-model warning fires inline here.
        if not _skip_drift_warning:
            _warn_poor_calibration(
                model, concordance_df, model_name, _warn_stacklevel,
                collector=_drift_collector,
                is_calibrated=_model_is_calibrated,
            )

        test_set_df = _build_test_set_df(X_test, y_test, risk_scores)
        test_set_df['model_type'] = model_name

        auc_df = _compute_auc_t(
            y_test, y_train, config.eval_times, risk_scores,
            model_name=model_name,
            auc_nan_collector=_auc_nan_collector,
            # +1: _compute_auc_t warns one frame deeper than the body-level
            # warnings here (via its inner _record helper), so bump to land the
            # inline single-model warning on the caller's frame, matching the
            # calibration sibling's stacklevel handling on this path.
            _warn_stacklevel=_warn_stacklevel + 1,
        )
        auc_df['model_type'] = model_name

        if config.show_progress:
            print(f" done ({_format_elapsed(_time.perf_counter() - _t0)}).")

        # Phase 1: Importance
        importance_df, iterations_df, per_column_importance_df = _run_importance_phase(
            model, X_test, y_test, config, model_name, importance_n_jobs,
        )

        # Phase 2: SHAP
        shap_df, shap_per_column_df = _run_shap_phase(
            model, X_test, config, model_name, display_name, X_train=X_train,
        )

        # Phase 3: Interactions
        interaction_result = _run_interactions_phase(
            model, X_test, importance_df, config, model_name, display_name,
            _warn_stacklevel=_warn_stacklevel + 1,
        )

    # Reorder 'model_type' first in all output DataFrames.
    result = {
        'concordance': concordance_df,
        'test_set': test_set_df,
        'auc_t': auc_df,
        'importance': importance_df,
        'iterations': iterations_df,
        'importance_per_column': per_column_importance_df,
        'shap': shap_df,
        'shap_per_column': shap_per_column_df,
        **interaction_result,
        '_surv_probs': _surv_probs,
        '_surv_eval_times': _surv_eval_times,
        '_ibs_surv_probs': _ibs_surv_probs,
        '_ibs_times': _ibs_times,
    }
    for key in ('concordance', 'test_set', 'auc_t', 'importance', 'iterations'):
        result[key] = _reorder_columns(result[key], 'model_type')
    for key in ('shap', 'shap_per_column', 'interactions'):
        if result[key] is not None:
            result[key] = _reorder_columns(result[key], 'model_type')
    return result


def _evaluate_multiple(
    models,
    X_test,
    y_test: SurvivalArray,
    X_train=None,
    y_train: Optional[SurvivalArray] = None,
    eval_times=None,
    extra_eval_times=None,
    n_importance_repeats=DEFAULT_N_IMPORTANCE_REPEATS,
    importance_max_samples=None,
    include_shap=False,
    shap_max_background=50,
    shap_max_explain=50,
    random_state=None,
    show_progress=True,
    # Bootstrap parameters
    bootstrap=False,
    n_bootstrap="auto",
    bootstrap_ci=0.95,
    bootstrap_metrics=None,
    bootstrap_stratify=False,
    include_predictions=False,
    # Parallel computing
    n_jobs=1,
    _warn_stacklevel=3,
    subpopulation_labels=None,
    categorical_groups=None,
    categorical_features=None,
    include_interactions=False,
    interaction_top_k=5,
    interaction_max_samples=100,
    interaction_method='hstat',
    interaction_scale='auto',
    ipcw_tau: Optional[float] = None,
    ipcw_y_train: Optional[SurvivalArray] = None,
    gnd_n_bins: int = 10,
    _surv_probs_transforms: Optional[dict] = None,
    _calibrated_models: Optional[frozenset] = None,
    _skip_drift_warning: bool = False,
    # Skip the unused-parameter warnings (bootstrap/shap/interaction).
    _skip_param_warnings: bool = False,
):
    """Evaluate multiple models and aggregate results (internal)."""
    # Check if params were changed without enabling their feature
    if not _skip_param_warnings:
        _warn_unused_bootstrap_params(
            bootstrap, n_bootstrap, bootstrap_ci, bootstrap_metrics, bootstrap_stratify,
            stacklevel=_warn_stacklevel,
        )
        _warn_unused_shap_params(
            include_shap, shap_max_background, shap_max_explain,
            stacklevel=_warn_stacklevel,
        )
        _warn_unused_interaction_params(
            include_interactions, interaction_top_k, interaction_max_samples,
            interaction_method=interaction_method,
            interaction_scale=interaction_scale,
            stacklevel=_warn_stacklevel,
        )

    # Resolve n_bootstrap if "auto"
    n_samples = len(X_test)
    n_bootstrap = resolve_n_bootstrap(n_bootstrap, n_samples, show_progress, bootstrap)

    # Sequential model loop with rich per-model progress.
    # n_jobs is passed through to _evaluate_single for inner parallelism
    # (permutation importance across features), where it actually helps.
    # Cross-model parallelism gives near-zero speedup when one model (e.g. RSF)
    # dominates wall-clock time.
    model_items = list(models.items())
    results = []

    # Coalesce per-model calibration-drift warnings into one summary.
    # Only when there are multiple models to compare — single-model multi
    # call still gets the inline per-model warning so the message is intact.
    # When validate()'s raw pass calls in with _skip_drift_warning,
    # leave the collector disabled so no summary is emitted at the end.
    drift_collector = (
        [] if (len(model_items) > 1 and not _skip_drift_warning) else None
    )

    # Coalesce per-model bootstrap-failure summaries into one summary.
    # Same multi-model-only guard: a single-model dispatch keeps its inline
    # per-model emission so the verbose message stays intact.
    bootstrap_failure_collector = (
        {} if (len(model_items) > 1 and bootstrap) else None
    )

    # Coalesce per-model bootstrap low-coverage AUC(t) summaries into one
    # summary per metric, collapsing both across models and across
    # subpopulation levels. Same multi-model-only guard: a single-model
    # dispatch keeps its inline per-model emission so the verbose message stays
    # intact.
    bootstrap_notices_collector = (
        {} if (len(model_items) > 1 and bootstrap) else None
    )

    # Coalesce per-model calibration NaN-return warnings into one summary
    # per metric. Multi-model-only guard mirrors the drift collector:
    # a single-model dispatch keeps the inline per-model emission.
    calibration_nan_collector = [] if len(model_items) > 1 else None

    # Coalesce per-model AUC(t) NaN-return warnings into one summary naming the
    # affected models. Multi-model-only guard mirrors calibration_nan_collector:
    # a single-model dispatch keeps the inline per-model emission.
    auc_nan_collector = [] if len(model_items) > 1 else None

    bar = None
    if show_progress and len(model_items) > 1:
        bar = tqdm(
            total=len(model_items),
            desc="Evaluating models",
            unit="model",
            leave=True,
        )

    try:
        for name, model in model_items:
            # Resolve per-model importance repeats when a dict is provided
            if isinstance(n_importance_repeats, dict):
                model_repeats = _resolve_repeats_for_model(
                    n_importance_repeats, name,
                )
            else:
                model_repeats = n_importance_repeats

            t_model = _time.perf_counter()
            eval_result = _evaluate_single(
                model,
                X_test,
                y_test,
                X_train=X_train,
                y_train=y_train,
                eval_times=eval_times,
                extra_eval_times=extra_eval_times,
                n_importance_repeats=model_repeats,
                importance_max_samples=importance_max_samples,
                include_shap=include_shap,
                shap_max_background=shap_max_background,
                shap_max_explain=shap_max_explain,
                random_state=random_state,
                model_name=name,
                show_progress=show_progress,
                bootstrap=bootstrap,
                n_bootstrap=n_bootstrap,
                bootstrap_ci=bootstrap_ci,
                bootstrap_metrics=bootstrap_metrics,
                bootstrap_stratify=bootstrap_stratify,
                include_predictions=include_predictions,
                _skip_param_warnings=True,
                n_jobs=n_jobs,
                # +1 accounts for this _evaluate_multiple body frame between
                # _evaluate_single and the free evaluate caller, so warnings
                # fired inside _evaluate_single still attribute to the user's
                # frame on the dict-of-models dispatch path.
                _warn_stacklevel=_warn_stacklevel + 1,
                subpopulation_labels=subpopulation_labels,
                categorical_groups=categorical_groups,
                categorical_features=categorical_features,
                include_interactions=include_interactions,
                interaction_top_k=interaction_top_k,
                interaction_max_samples=interaction_max_samples,
                interaction_method=interaction_method,
                interaction_scale=interaction_scale,
                ipcw_tau=ipcw_tau,
                ipcw_y_train=ipcw_y_train,
                gnd_n_bins=gnd_n_bins,
                surv_probs_transform=(
                    _surv_probs_transforms.get(name)
                    if _surv_probs_transforms else None
                ),
                _drift_collector=drift_collector,
                _bootstrap_failure_collector=bootstrap_failure_collector,
                _bootstrap_notices_collector=bootstrap_notices_collector,
                _calibration_nan_collector=calibration_nan_collector,
                _auc_nan_collector=auc_nan_collector,
                _model_is_calibrated=bool(
                    _calibrated_models and name in _calibrated_models
                ),
                _skip_drift_warning=_skip_drift_warning,
            )
            results.append((name, eval_result))
            if bar is not None:
                elapsed = _format_elapsed(_time.perf_counter() - t_model)
                bar.set_postfix_str(f"{name} {elapsed}")
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
        # Emit the coalesced calibration-drift summary even if a per-model
        # evaluation raised partway through — otherwise records accumulated
        # from completed models are silently dropped (the earlier inline path
        # fired per model, so partial info reached the user). A formatter failure
        # falls back to inline per-record emission so no record is lost.
        if drift_collector:
            try:
                _emit_coalesced_calibration_warning(
                    drift_collector, len(model_items), _warn_stacklevel,
                )
            except (TypeError, ValueError, AttributeError) as exc:
                warnings.warn(
                    f"Failed to format coalesced calibration drift summary "
                    f"({type(exc).__name__}: {exc}); emitting per-model "
                    f"warnings instead.",
                    MethodologicalWarning, stacklevel=_warn_stacklevel,
                )
                # Guard each per-record format independently. An exception
                # here inside `finally` would replace the in-flight original
                # exception (Python semantics), and a single bad record
                # shouldn't drop the rest.
                for rec in drift_collector:
                    try:
                        msg = _format_single_drift_warning(rec)
                    except (TypeError, ValueError, AttributeError,
                            KeyError) as rec_exc:
                        label = getattr(rec, 'label', '<unknown>')
                        warnings.warn(
                            f"Failed to format calibration drift record for "
                            f"{label} ({type(rec_exc).__name__}: {rec_exc}); "
                            f"record dropped.",
                            MethodologicalWarning, stacklevel=_warn_stacklevel,
                        )
                        continue
                    warnings.warn(
                        msg, MethodologicalWarning,
                        stacklevel=_warn_stacklevel,
                    )

        # Emit the coalesced bootstrap-failure summary. Mirrors the
        # drift block above: if the cross-model formatter fails, fall back
        # to per-model emission so no record is lost. Guarded inside the
        # outer try/finally so failures from completed models still surface
        # if a later model's evaluate raises.
        if bootstrap_failure_collector:
            from .bootstrap import (
                _emit_multi_failure_summary,
                _emit_failure_summary,
            )
            # Broad ``except Exception`` here is deliberate. This block runs
            # inside the outer ``finally:`` block. A narrower catch would let
            # unexpected exception types (e.g. logger ``OSError``, custom
            # warning-filter errors) propagate out and **replace the in-flight
            # exception** from the model loop above. Diagnostic-emission code
            # inside ``finally`` must never mask the original exception, so
            # the convention against bare ``except Exception`` does not apply
            # here. Same applies to the inner per-model loop below.
            try:
                _emit_multi_failure_summary(
                    bootstrap_failure_collector, n_bootstrap,
                    _warn_stacklevel=_warn_stacklevel,
                )
            except Exception as exc:  # noqa: BLE001 — see comment above
                warnings.warn(
                    f"Failed to format coalesced bootstrap-failure summary "
                    f"({type(exc).__name__}: {exc}); emitting per-model "
                    f"summaries instead.",
                    MethodologicalWarning, stacklevel=_warn_stacklevel,
                )
                for mname, mfails in bootstrap_failure_collector.items():
                    try:
                        _emit_failure_summary(
                            mfails, n_bootstrap, mname,
                            _warn_stacklevel=_warn_stacklevel,
                        )
                    except Exception as rec_exc:  # noqa: BLE001
                        warnings.warn(
                            f"Failed to emit bootstrap-failure summary for "
                            f"{mname} ({type(rec_exc).__name__}: {rec_exc}); "
                            f"record dropped.",
                            MethodologicalWarning, stacklevel=_warn_stacklevel,
                        )

        # Emit the coalesced bootstrap low-coverage AUC(t) summary.
        # Mirrors the bootstrap-failure block above: one warning per
        # (metric, subpopulation) group naming all affected models, with a
        # per-model fallback if the cross-model formatter fails. The broad
        # ``except Exception`` is justified by the same in-``finally``
        # rationale documented for the failure block above — a narrower
        # catch could let an unexpected type replace the in-flight loop
        # exception.
        if bootstrap_notices_collector:
            from .bootstrap import (
                _emit_multi_low_coverage_summary,
                _emit_low_coverage_summary,
            )
            try:
                _emit_multi_low_coverage_summary(
                    bootstrap_notices_collector, n_bootstrap,
                    _warn_stacklevel=_warn_stacklevel,
                )
            except Exception as exc:  # noqa: BLE001 — see comment above
                warnings.warn(
                    f"Failed to format coalesced bootstrap low-coverage "
                    f"summary ({type(exc).__name__}: {exc}); emitting "
                    f"per-model summaries instead.",
                    MethodologicalWarning, stacklevel=_warn_stacklevel,
                )
                for mname, mnotices in bootstrap_notices_collector.items():
                    try:
                        _emit_low_coverage_summary(
                            mnotices, n_bootstrap, mname,
                            _warn_stacklevel=_warn_stacklevel,
                        )
                    except Exception as rec_exc:  # noqa: BLE001
                        warnings.warn(
                            f"Failed to emit bootstrap low-coverage summary "
                            f"for {mname} ({type(rec_exc).__name__}: "
                            f"{rec_exc}); record dropped.",
                            MethodologicalWarning, stacklevel=_warn_stacklevel,
                        )

        # Emit the coalesced calibration NaN-return summary. One
        # warning per metric_key naming all NaN-returning models. Broad
        # ``except Exception`` here mirrors the bootstrap block above:
        # this code runs inside the outer ``finally:`` and any uncaught
        # exception type would replace the in-flight loop exception
        # (Python semantics). _emit_coalesced_calibration_nan_warning
        # has its own internal narrow catch for formatter failures, so
        # this is belt-and-suspenders for unexpected types.
        if calibration_nan_collector:
            try:
                _emit_coalesced_calibration_nan_warning(
                    calibration_nan_collector, len(model_items),
                    _warn_stacklevel,
                )
            except Exception as exc:  # noqa: BLE001 — runs inside finally
                warnings.warn(
                    f"Failed to emit coalesced calibration NaN-return "
                    f"summary ({type(exc).__name__}: {exc}); emitting "
                    f"per-model warnings instead.",
                    MethodologicalWarning, stacklevel=_warn_stacklevel,
                )
                for _mk, _mn, rec_msg in calibration_nan_collector:
                    try:
                        warnings.warn(
                            rec_msg, MethodologicalWarning,
                            stacklevel=_warn_stacklevel,
                        )
                    except Exception as rec_exc:  # noqa: BLE001
                        warnings.warn(
                            f"Failed to emit calibration NaN record "
                            f"({type(rec_exc).__name__}: {rec_exc}); "
                            f"record dropped.",
                            MethodologicalWarning,
                            stacklevel=_warn_stacklevel,
                        )

        # Emit the coalesced AUC(t) NaN-return summary: one warning naming all
        # models that returned NaN at any requested time. Broad
        # ``except Exception`` here mirrors the calibration-NaN block above:
        # this code runs inside the outer ``finally:`` and any uncaught
        # exception type would replace the in-flight loop exception (Python
        # semantics). The formatter has its own narrow catch for formatting
        # failures, so this is belt-and-suspenders for unexpected types.
        if auc_nan_collector:
            try:
                _emit_coalesced_auc_nan_warning(
                    auc_nan_collector, len(model_items), _warn_stacklevel,
                )
            except Exception as exc:  # noqa: BLE001 — runs inside finally
                warnings.warn(
                    f"Failed to emit coalesced AUC(t) NaN-return summary "
                    f"({type(exc).__name__}: {exc}); emitting per-model "
                    f"warnings instead.",
                    MethodologicalWarning, stacklevel=_warn_stacklevel,
                )
                for _mk, _mn, rec_msg in auc_nan_collector:
                    try:
                        warnings.warn(
                            rec_msg, MethodologicalWarning,
                            stacklevel=_warn_stacklevel,
                        )
                    except Exception as rec_exc:  # noqa: BLE001
                        warnings.warn(
                            f"Failed to emit AUC(t) NaN record "
                            f"({type(rec_exc).__name__}: {rec_exc}); "
                            f"record dropped.",
                            MethodologicalWarning,
                            stacklevel=_warn_stacklevel,
                        )

    # Warn about unused dict keys
    from mlsurv._shared import _warn_unused_dict_keys
    model_names = [name for name, _ in model_items]
    _warn_unused_dict_keys(n_importance_repeats, "n_importance_repeats", model_names)

    return _aggregate_evaluation_results(results)


def _aggregate_evaluation_results(results):
    """Aggregate per-model EvaluationResults into a single combined result.

    Parameters
    ----------
    results : list of (name, EvaluationResult) tuples

    Returns
    -------
    EvaluationResult
    """
    required_attrs = ['concordance', 'test_set', 'auc_t', 'importance', 'importance_iterations']
    frames = {
        attr: pd.concat([getattr(r, attr) for _, r in results], ignore_index=True)
        for attr in required_attrs
    }

    eval_objects = [r for _, r in results]
    shap_values = _concat_attr(eval_objects, 'shap_values')
    bootstrap_result = _aggregate_bootstrap_results(results)

    importance_per_column = _concat_attr(eval_objects, 'importance_per_column')
    shap_values_per_column = _concat_attr(eval_objects, 'shap_values_per_column')
    interactions = _concat_attr(eval_objects, 'interactions')

    merged_surfaces = [s for r in eval_objects for s in (r.interaction_surfaces or [])]

    return EvaluationResult(
        concordance=frames['concordance'],
        test_set=frames['test_set'],
        auc_t=frames['auc_t'],
        importance=frames['importance'],
        importance_iterations=frames['importance_iterations'],
        importance_per_column=importance_per_column,
        shap_values=shap_values,
        shap_values_per_column=shap_values_per_column,
        interactions=interactions,
        interaction_surfaces=merged_surfaces or None,
        interaction_method=_resolve_interaction_method(eval_objects),
        bootstrap=bootstrap_result,
        _surv_probs_cache=_merge_optional_dicts(eval_objects, '_surv_probs_cache'),
        _ibs_surv_probs_cache=_merge_optional_dicts(eval_objects, '_ibs_surv_probs_cache'),
        _interaction_raw_cache=_merge_optional_dicts(eval_objects, '_interaction_raw_cache'),
    )


def _concat_attr(objects, attr):
    """Concatenate an optional DataFrame attribute across objects.

    Parameters
    ----------
    objects : iterable
        Objects with the named attribute.
    attr : str
        Attribute name on each object.

    Returns
    -------
    DataFrame or None
    """
    items = [getattr(obj, attr) for obj in objects if getattr(obj, attr, None) is not None]
    if items:
        return pd.concat(items, ignore_index=True)
    return None


def _aggregate_bootstrap_results(results):
    """Aggregate bootstrap results from multiple model evaluations.

    Parameters
    ----------
    results : list of (name, EvaluationResult) tuples

    Returns
    -------
    BootstrapResult or None
    """
    boots = [r.bootstrap for _, r in results if r.bootstrap is not None]
    if not boots:
        return None

    from ..results.containers import BootstrapResult

    return BootstrapResult(
        summary=pd.concat([b.summary for b in boots], ignore_index=True),
        iterations=pd.concat([b.iterations for b in boots], ignore_index=True),
        predictions=_concat_attr(boots, 'predictions'),
        auc_t=_concat_attr(boots, 'auc_t'),
        brier_t=_concat_attr(boots, 'brier_t'),
        subpopulation_auc_t=_concat_attr(boots, 'subpopulation_auc_t'),
        importance=_concat_attr(boots, 'importance'),
        lower_is_better=boots[0].lower_is_better if boots else None,
        ci_level=boots[0].ci_level,
    )


def _reorder_columns(df, first_col):
    """Reorder DataFrame columns to have first_col first."""
    if first_col not in df.columns:
        return df
    cols = [first_col] + [c for c in df.columns if c != first_col]
    return df[cols]
