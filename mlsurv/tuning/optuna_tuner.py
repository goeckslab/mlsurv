"""Backward-compatibility shim for mlsurv.tuning.optuna_tuner.

All functionality has been split into focused modules:
- base.py: BaseTuner and helper functions
- storage.py: Storage constants and connection helpers
- studies.py: Study listing, export, and record extraction
- tuners/: Per-model tuner classes and registry

This module re-exports everything for code that imports directly from
``mlsurv.tuning.optuna_tuner``.
"""

# Base tuner and helpers
from .base import (
    BaseTuner,
    _compute_parallelism_split,
    _is_class_name,
    _parse_component_search_space,
    _get_step_class_mapping,
)

# Storage
from .storage import (
    DEFAULT_STORAGE_BASE,
    OPTUNA_STORAGE_ENV_VAR,
    get_storage_url,
    validate_storage_connection,
)

# Studies
from .studies import (
    _parse_study_summary,
    _list_studies_from_storage,
    _list_studies_from_files,
    list_studies,
    get_study_path,
    _study_to_records,
    export_study,
)

# Tuner classes and registry
from .tuners import (
    RSFTuner,
    GBSATuner,
    CoxNetTuner,
    FSSVMTuner,
    DeepSurvTuner,
    CoxPHTuner,
    XGBCoxTuner,
    DeepHitTuner,
    WeibullAFTTuner,
    TUNER_REGISTRY,
    get_tuner,
)

# Re-export everything that was importable from the old monolith
__all__ = [
    # Base
    "BaseTuner",
    "_compute_parallelism_split",
    "_is_class_name",
    "_parse_component_search_space",
    "_get_step_class_mapping",
    # Storage
    "DEFAULT_STORAGE_BASE",
    "OPTUNA_STORAGE_ENV_VAR",
    "get_storage_url",
    "validate_storage_connection",
    # Studies
    "_parse_study_summary",
    "_list_studies_from_storage",
    "_list_studies_from_files",
    "list_studies",
    "get_study_path",
    "_study_to_records",
    "export_study",
    # Tuners
    "RSFTuner",
    "GBSATuner",
    "CoxNetTuner",
    "FSSVMTuner",
    "DeepSurvTuner",
    "CoxPHTuner",
    "XGBCoxTuner",
    "DeepHitTuner",
    "WeibullAFTTuner",
    "TUNER_REGISTRY",
    "get_tuner",
]
