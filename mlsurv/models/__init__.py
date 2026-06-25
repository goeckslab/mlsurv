"""Survival model wrappers for mlsurv."""

import warnings

from mlsurv.models.base import BaseSurvivalModel
from mlsurv.models.coxph import CoxPHModel
from mlsurv.models.coxnet import CoxNetModel
from mlsurv.models.rsf import RSFModel
from mlsurv.models.gbsa import GBSAModel
from mlsurv.models.fssvm import FSSVMModel
from mlsurv.models.xgbcox import XGBCoxModel
from mlsurv.models.xgbaft import XGBAFTModel
from mlsurv.models.weibull_aft import WeibullAFTModel
from mlsurv.models._auto_defaults import (
    auto_deephit_defaults,
    auto_deepsurv_defaults,
    auto_gbsa_defaults,
    auto_rsf_defaults,
    auto_weibull_aft_defaults,
)

# Model registry for name-based lookup
MODEL_REGISTRY = {
    "coxph": CoxPHModel,
    "coxnet": CoxNetModel,
    "rsf": RSFModel,
    "gbsa": GBSAModel,
    "fssvm": FSSVMModel,
    "xgbcox": XGBCoxModel,
    "xgbaft": XGBAFTModel,
    "weibull_aft": WeibullAFTModel,
}

# Metadata for lazy-loaded models (deepsurv, deephit not in MODEL_REGISTRY).
# These models require optional dependencies (torch, pycox) so we can't import
# them eagerly. Fallback metadata lives in _metadata.py.
from mlsurv.models._metadata import DEEP_MODEL_METADATA as _LAZY_MODEL_METADATA


def get_model_metadata(model_name: str) -> dict:
    """
    Get metadata dict for a model.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'coxph', 'rsf', 'deepsurv').

    Returns
    -------
    dict
        Metadata dict with keys: display_name, color, supports_random_state,
        prefer_sequential_cv.
    """
    # Check registered models first (class metadata is authoritative)
    if model_name in MODEL_REGISTRY:
        model_cls = MODEL_REGISTRY[model_name]
        return {
            "display_name": model_cls.display_name or model_cls.name.upper(),
            "color": model_cls.color,
            "supports_random_state": model_cls.supports_random_state,
            "prefer_sequential_cv": model_cls.prefer_sequential_cv,
            "has_internal_parallelism": model_cls.has_internal_parallelism,
            "complexity": model_cls.complexity,
            "assumes_ph": model_cls.assumes_ph,
            "assumes_linearity": model_cls.assumes_linearity,
        }

    # Check lazy-loaded models (fallback when dependencies not installed)
    if model_name in _LAZY_MODEL_METADATA:
        return _LAZY_MODEL_METADATA[model_name].copy()

    # Unknown model - return defaults
    from mlsurv._warnings import InternalWarning
    warnings.warn(
        f"No metadata registered for model '{model_name}'; using defaults.",
        InternalWarning,
        stacklevel=2,
    )
    return {
        "display_name": model_name.upper(),
        "color": "#7f7f7f",
        "supports_random_state": False,
        "prefer_sequential_cv": False,
        "has_internal_parallelism": False,
        "complexity": "linear",
        "assumes_ph": False,
        "assumes_linearity": False,
    }


def get_models_with_random_state() -> set:
    """
    Build set of models that support random_state parameter.

    Returns
    -------
    set
        Set of model names that support random_state.
    """
    result = set()

    # Check registered models
    for name, model_cls in MODEL_REGISTRY.items():
        if model_cls.supports_random_state:
            result.add(name)

    # Add lazy-loaded models
    for name, meta in _LAZY_MODEL_METADATA.items():
        if meta["supports_random_state"]:
            result.add(name)

    return result


def get_sequential_cv_models() -> set:
    """
    Build set of models that prefer sequential CV (no joblib parallelization).

    Returns
    -------
    set
        Set of model names that prefer sequential CV.
    """
    result = set()

    # Check registered models
    for name, model_cls in MODEL_REGISTRY.items():
        if model_cls.prefer_sequential_cv:
            result.add(name)

    # Add lazy-loaded models
    for name, meta in _LAZY_MODEL_METADATA.items():
        if meta["prefer_sequential_cv"]:
            result.add(name)

    return result


def get_models_with_internal_parallelism() -> set:
    """
    Build set of models with internal parallelism (n_jobs).

    Returns
    -------
    set
        Set of model names that have internal parallelism.
    """
    result = set()

    # Check registered models
    for name, model_cls in MODEL_REGISTRY.items():
        if model_cls.has_internal_parallelism:
            result.add(name)

    # Add lazy-loaded models
    for name, meta in _LAZY_MODEL_METADATA.items():
        if meta.get("has_internal_parallelism", False):
            result.add(name)

    return result


def get_deep_learning_models() -> set:
    """Build set of deep learning model names (complexity == 'deep_learning').

    Returns
    -------
    set
        Model names that are deep learning models.
    """
    result = set()

    # Check registered models
    for name, model_cls in MODEL_REGISTRY.items():
        if model_cls.complexity == 'deep_learning':
            result.add(name)

    # Add lazy-loaded models
    for name, meta in _LAZY_MODEL_METADATA.items():
        if meta.get("complexity") == 'deep_learning':
            result.add(name)

    return result


def get_model_complexity(model_name: str) -> str:
    """Return complexity class for a model.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'coxph', 'rsf', 'deepsurv').

    Returns
    -------
    str
        One of 'linear', 'tree_ensemble', or 'deep_learning'.
    """
    return get_model_metadata(model_name).get("complexity", "linear")


def get_model_colors() -> dict:
    """
    Build dict of model colors for visualizations.

    Includes ensemble and baseline colors for completeness.

    Returns
    -------
    dict
        Dict mapping model name to hex color code.
    """
    result = {}

    # Add lazy-loaded models first (fallback values)
    for name, meta in _LAZY_MODEL_METADATA.items():
        result[name] = meta["color"]

    # Add registered models (class metadata overrides lazy)
    for name, model_cls in MODEL_REGISTRY.items():
        result[name] = model_cls.color

    # Add special colors for ensemble/baseline
    result["ensemble"] = "#44AA99"  # Teal
    result["baseline"] = "#999999"  # Gray

    return result


def get_model_display_names() -> dict:
    """
    Build dict of model display names for reports/visualizations.

    Returns
    -------
    dict
        Dict mapping model name to human-readable display name.
    """
    result = {}

    # Add lazy-loaded models first (fallback values)
    for name, meta in _LAZY_MODEL_METADATA.items():
        result[name] = meta["display_name"]

    # Add registered models (class metadata overrides lazy)
    for name, model_cls in MODEL_REGISTRY.items():
        result[name] = model_cls.display_name or model_cls.name.upper()

    return result


def get_ph_assumption_models() -> frozenset:
    """
    Build frozenset of model names that assume proportional hazards.

    Includes models that fit the Cox partial likelihood (or otherwise
    produce a single time-invariant risk score whose hazard ratios
    between subjects are constant by construction).

    Returns
    -------
    frozenset
        Model names with ``assumes_ph = True``.
    """
    result = set()

    for name, model_cls in MODEL_REGISTRY.items():
        if model_cls.assumes_ph:
            result.add(name)

    for name, meta in _LAZY_MODEL_METADATA.items():
        if meta.get("assumes_ph", False):
            result.add(name)

    return frozenset(result)


def get_linearity_assumption_models() -> frozenset:
    """
    Build frozenset of model names that assume log-hazard linearity
    in continuous covariates (Harrell 2015 §4.6, §5.5).

    Returns
    -------
    frozenset
        Model names with ``assumes_linearity = True``.
    """
    result = set()

    for name, model_cls in MODEL_REGISTRY.items():
        if model_cls.assumes_linearity:
            result.add(name)

    for name, meta in _LAZY_MODEL_METADATA.items():
        if meta.get("assumes_linearity", False):
            result.add(name)

    return frozenset(result)


# Registry of models with dynamic defaults (model_name -> defaults function)
_AUTO_DEFAULTS_REGISTRY = {
    "rsf": auto_rsf_defaults,
    "gbsa": auto_gbsa_defaults,
    "weibull_aft": auto_weibull_aft_defaults,
    "deepsurv": auto_deepsurv_defaults,
    "deephit": auto_deephit_defaults,
}

# Lazy-loaded model classes (require optional dependencies)
_LAZY_MODEL_IMPORTS = {
    "deepsurv": ("mlsurv.models.deepsurv", "DeepSurvModel"),
    "deephit": ("mlsurv.models.deephit", "DeepHitModel"),
}


def _apply_dynamic_defaults(name: str, n_samples, n_features, n_events, kwargs: dict) -> None:
    """Apply dynamic defaults for a model if data characteristics are provided."""
    if name not in _AUTO_DEFAULTS_REGISTRY:
        return
    if not all(v is not None for v in [n_samples, n_features, n_events]):
        return

    defaults_fn = _AUTO_DEFAULTS_REGISTRY[name]
    defaults = defaults_fn(n_samples, n_features, n_events)
    for key, value in defaults.items():
        kwargs.setdefault(key, value)


def get_model(name: str, n_samples=None, n_features=None, n_events=None, **kwargs):
    """
    Get a model instance by name.

    Parameters
    ----------
    name : str
        Model name: 'coxph', 'coxnet', 'rsf', 'gbsa', 'fssvm', 'xgbcox',
        'xgbaft', 'weibull_aft', 'deepsurv', or 'deephit'. Case-insensitive.
    n_samples : int, optional
        Number of training samples. Used for dynamic defaults (RSF, GBSA, DeepSurv).
    n_features : int, optional
        Number of features. Used for dynamic defaults (RSF, GBSA, DeepSurv).
    n_events : int, optional
        Number of events. Used for dynamic defaults (RSF, GBSA, DeepSurv).
    **kwargs : dict
        Model hyperparameters. These override any computed defaults.

    Returns
    -------
    BaseSurvivalModel
        Model instance.

    Raises
    ------
    ValueError
        If `name` does not match any registered or lazy-loaded model.

    Examples
    --------
    >>> from mlsurv.models import get_model
    >>> model = get_model('rsf', n_estimators=200)
    >>> model = get_model('RSF', n_estimators=200)  # Case-insensitive
    >>> # RSF with dynamic defaults based on data characteristics
    >>> model = get_model('rsf', n_samples=686, n_features=7, n_events=299)
    >>> # DeepSurv with dynamic defaults based on data characteristics
    >>> model = get_model('deepsurv', n_samples=686, n_features=7, n_events=299)
    """
    name = name.lower()

    # Validate model name
    all_models = set(MODEL_REGISTRY.keys()) | set(_LAZY_MODEL_IMPORTS.keys())
    if name not in all_models:
        raise ValueError(f"Unknown model: {name}. Available: {sorted(all_models)}")

    # Apply dynamic defaults if data characteristics are provided
    _apply_dynamic_defaults(name, n_samples, n_features, n_events, kwargs)

    # Handle lazy-loaded models (deep learning)
    if name in _LAZY_MODEL_IMPORTS:
        import importlib
        module_path, class_name = _LAZY_MODEL_IMPORTS[name]
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)
        return model_cls(**kwargs)

    # Handle registry models
    return MODEL_REGISTRY[name](**kwargs)


def _is_deep_learning_available():
    """Check if pycox is installed for deep learning model support."""
    try:
        import torch  # noqa: F401
        import pycox  # noqa: F401
        return True
    except ImportError:
        return False


def list_models(include_unavailable=False):
    """
    List available model names.

    Parameters
    ----------
    include_unavailable : bool, default=False
        If True, include models even if optional dependencies are missing.
        If False, include only models that can be imported in the current
        environment.

    Returns
    -------
    list
        Available model names.
    """
    models = list(MODEL_REGISTRY.keys())
    if include_unavailable or _is_deep_learning_available():
        models.extend(["deepsurv", "deephit"])
    return models


__all__ = [
    "BaseSurvivalModel",
    "CoxPHModel",
    "CoxNetModel",
    "RSFModel",
    "GBSAModel",
    "FSSVMModel",
    "XGBCoxModel",
    "XGBAFTModel",
    "WeibullAFTModel",
    "MODEL_REGISTRY",
    "get_model",
    "list_models",
    "get_model_metadata",
    "get_models_with_random_state",
    "get_sequential_cv_models",
    "get_models_with_internal_parallelism",
    "get_model_colors",
    "get_model_display_names",
    "get_model_complexity",
    "get_ph_assumption_models",
    "get_linearity_assumption_models",
]
