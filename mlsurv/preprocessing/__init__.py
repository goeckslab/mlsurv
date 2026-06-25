"""Preprocessing and feature selection for mlsurv."""

from mlsurv.preprocessing.pipelines import (
    # Resolver functions
    get_imputer,
    get_scaler,
    get_selector,
    warn_preprocessing_order,
)
from mlsurv.preprocessing.selection import (
    TopKCoxSelector,
    VIFSelector,
    CorrelationSelector,
    CoxLassoSelector,
    CoxNetSelector,
)
from mlsurv.preprocessing.indicators import add_missing_indicators
from mlsurv.preprocessing.utils import (
    get_default_imputer,
    get_default_scaler,
    extract_component_params,
    get_or_create_selector,
    detect_categorical_groups,
)

__all__ = [
    # Resolver functions
    "get_imputer",
    "get_scaler",
    "get_selector",
    "warn_preprocessing_order",
    # Feature selectors
    "TopKCoxSelector",
    "VIFSelector",
    "CorrelationSelector",
    "CoxLassoSelector",
    "CoxNetSelector",
    # Indicators
    "add_missing_indicators",
    # Utilities
    "get_default_imputer",
    "get_default_scaler",
    "extract_component_params",
    "get_or_create_selector",
    "detect_categorical_groups",
]
