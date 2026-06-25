"""Hyperparameter tuning module for mlsurv."""

from .search_spaces import (
    DEFAULT_SPACES,
    PREPROCESSING_DEFAULTS,
    auto_n_trials,
    auto_n_bootstrap,
    auto_cv_folds,
    get_preprocessing_defaults,
)
from .exceptions import NoTunerError
from .base import BaseTuner
from .storage import (
    DEFAULT_STORAGE_BASE,
    OPTUNA_STORAGE_ENV_VAR,
    get_storage_url,
    validate_storage_connection,
)
from .studies import (
    list_studies,
    export_study,
    get_study_path,
)
from .tuners import (
    RSFTuner,
    GBSATuner,
    CoxNetTuner,
    FSSVMTuner,
    DeepSurvTuner,
    CoxPHTuner,
    CustomModelTuner,
    XGBCoxTuner,
    DeepHitTuner,
    WeibullAFTTuner,
    get_tuner,
    TUNER_REGISTRY,
)

__all__ = [
    # Search spaces
    "DEFAULT_SPACES",
    "PREPROCESSING_DEFAULTS",
    "auto_n_trials",
    "auto_n_bootstrap",
    "auto_cv_folds",
    "get_preprocessing_defaults",
    # Exceptions
    "NoTunerError",
    # Tuners
    "BaseTuner",
    "RSFTuner",
    "GBSATuner",
    "CoxNetTuner",
    "FSSVMTuner",
    "DeepSurvTuner",
    "CoxPHTuner",
    "CustomModelTuner",
    "XGBCoxTuner",
    "DeepHitTuner",
    "WeibullAFTTuner",
    "get_tuner",
    "TUNER_REGISTRY",
    # Study management
    "list_studies",
    "export_study",
    "get_study_path",
    "DEFAULT_STORAGE_BASE",
    # PostgreSQL storage helpers
    "OPTUNA_STORAGE_ENV_VAR",
    "get_storage_url",
    "validate_storage_connection",
]
