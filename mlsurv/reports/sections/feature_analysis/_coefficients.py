"""Model inspection helpers for feature analysis.

Functions operate on a *section* instance (``FeatureAnalysisSection``) to
access the learner, evaluation result, and logging facilities.

The per-model coefficient *tables* used to be built here; that assembly
now lives in ``mlsurv.evaluation`` (``compute_model_coefficients`` /
``compute_feature_summary``) and the section consumes
``learner.feature_summary()`` instead. What remains: the
model-unwrapping helpers the hazard-ratio forest plot needs
(``find_coefficient_model``, used by ``_hazard_ratios.py``) and small
data-access helpers the section orchestrator uses
(``get_baseline_c_index``, ``get_univariate_cox``).
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mlsurv._shared import covariance_reason
from mlsurv.utils import unwrap_pipeline
from mlsurv.reports.sections.base import ALL_EXCEPTIONS

_COEF_MODEL_NAMES = ('coxph', 'coxnet', 'fssvm')



def unwrap_coefficient_model(section, name: str) -> Optional[Any]:
    """Unwrap a trained pipeline and return the inner model if it exposes coefficients.

    Returns the unwrapped model object, or None if the model is not
    trained or does not have ``get_coefficients``.
    """
    trained = section._get_all_models()
    if name not in trained:
        return None

    pipeline = trained[name].model
    actual_model = unwrap_pipeline(pipeline)

    if not hasattr(actual_model, 'get_coefficients'):
        return None

    ensure_feature_names_for_coefficients(section, actual_model, pipeline)
    return actual_model


def find_hr_valid_model(section) -> Optional[Any]:
    """Find a trained model whose ``exp(coef)`` is a genuine hazard ratio.

    Returns the first trained coefficient model that also has
    ``hazard_ratio_valid`` set (only unpenalized Cox PH qualifies). Used to
    gate the multivariate hazard-ratio forest plot so penalized (CoxNet) and
    ranking (FSSVM) coefficients are not mislabeled as HRs.
    """
    for name in _COEF_MODEL_NAMES:
        model = unwrap_coefficient_model(section, name)
        if model is not None and getattr(model, "hazard_ratio_valid", False):
            return model
    return None


def find_hr_suppression_reason(section) -> Optional[str]:
    """Why the multivariate hazard-ratio block was withheld, or None.

    ``find_hr_valid_model`` returns None both when no coefficient model was
    trained and when one was trained but its ``exp(coef)`` is not a hazard
    ratio. Only the second case has something to explain, and only the model
    knows the reason, so read it back off the model here rather than
    reconstructing it in the section.
    """
    for name in _COEF_MODEL_NAMES:
        model = unwrap_coefficient_model(section, name)
        if model is None:
            continue
        reason = getattr(model, "hr_suppressed_reason_", None)
        if reason:
            return reason
    return None


def find_covariance_gap_reason(section) -> Optional[str]:
    """Why confidence intervals are missing from an otherwise valid HR model.

    ``covariance_reason`` already answers None for a None model, so the "no
    HR-valid model was trained" case needs no separate branch.
    """
    return covariance_reason(find_hr_valid_model(section))


def get_all_coefficient_models(section) -> Dict[str, Any]:
    """Return all trained models that expose coefficients.

    Returns dict mapping model_name to unwrapped model object.
    Only includes models whose ``get_coefficients()`` returns
    a non-empty result.
    """
    result = {}
    for name in _COEF_MODEL_NAMES:
        model = unwrap_coefficient_model(section, name)
        if model is None:
            continue
        try:
            coefs = model.get_coefficients()
            if coefs is not None and len(coefs) > 0:
                result[name] = model
        except ALL_EXCEPTIONS as e:
            section._log_recoverable_exception(
                e,
                f"feature_analysis: coefficient extraction skipped for {name}",
                warn_user=True,
                user_message=f"Could not extract coefficients for {name}: {e}",
            )
    return result


def ensure_feature_names_for_coefficients(section, model, pipeline) -> None:
    """Populate ``feature_names_in_`` for fitted linear models when absent."""
    if hasattr(model, "feature_names_in_"):
        return

    coef = getattr(getattr(model, "model_", None), "coef_", None)
    if coef is None:
        return
    n_coef = int(np.asarray(coef).reshape(-1).shape[0])

    names = _names_from_selector(pipeline, n_coef)
    if names is None:
        names = _names_from_train_data(section, n_coef)

    if names is not None:
        model.feature_names_in_ = names


def _names_from_selector(pipeline, n_coef: int) -> Optional[np.ndarray]:
    """Extract feature names from a pipeline's selector step."""
    if hasattr(pipeline, "named_steps") and "selector" in pipeline.named_steps:
        selected = getattr(pipeline.named_steps["selector"], "selected_features_", None)
        if selected is not None and len(selected) == n_coef:
            return np.array(selected)
    return None


def _names_from_train_data(section, n_coef: int) -> Optional[np.ndarray]:
    """Extract feature names from the learner's training data."""
    X_train = getattr(section.learner, "X_train", None)
    if hasattr(X_train, "columns") and len(X_train.columns) == n_coef:
        return np.array(X_train.columns)
    return None



def get_baseline_c_index(section) -> Dict[str, float]:
    """Get baseline C-index for each model from concordance data.

    Returns mapping of model_type -> test C-index.
    """
    if section.eval_result is None or section.eval_result.concordance is None:
        return {}
    concordance_df = section.eval_result.concordance
    if 'c_index_test' not in concordance_df.columns:
        return {}
    return dict(zip(concordance_df['model_type'], concordance_df['c_index_test']))


def get_univariate_cox(section) -> Optional[pd.DataFrame]:
    """Get univariate Cox statistics from learner, or None if unavailable."""
    return getattr(section.learner, 'univariate_cox', None)
