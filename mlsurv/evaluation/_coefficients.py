"""Compute model coefficients with gated multivariable Cox inference.

This is the compute layer for coefficient accessors
(``ModelResult.coefficients`` and ``SurvivalLearner.coefficients``).
The report-side renderer lives in
``reports/sections/feature_analysis/_coefficients.py`` (same name,
different layer — intended).

Hazard ratios, confidence intervals, and Wald p-values are populated only
for models whose ``hazard_ratio_valid`` class flag is True (currently
CoxPH). Penalized (CoxNet) and ranking (FSSVM) coefficients are not
log-hazards with valid inference, and AFT coefficients are not log-hazards
at all, so those columns stay NaN for them.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .._shared import covariance_reason
from .._warnings import InternalWarning, SuggestionWarning
from ..utils import propagate_feature_names, unwrap_pipeline
from .importance import _align_coefficients, _get_selected_features

#: Column schema shared by ``ModelResult.coefficients`` and
#: ``SurvivalLearner.coefficients`` (mirrors the ``univariate_cox`` columns).
COEFFICIENT_COLUMNS = [
    'feature', 'coefficient', 'hazard_ratio',
    'hr_lower_95', 'hr_upper_95', 'p_value',
]


def _empty_coefficients():
    """Return the 0-row sentinel DataFrame (nothing computed)."""
    return pd.DataFrame(columns=COEFFICIENT_COLUMNS)


def compute_wald_inference(actual_model, coef_values, ci_level=0.95):
    """Compute Wald p-values and HR confidence bounds from a fitted model.

    Uses the covariance matrix that CoxPH attaches to its fitted sksurv
    model (``model_.covariance_matrix_``, the inverse observed Fisher
    information): ``SE = sqrt(diag(cov))``, ``z = coef / SE``,
    ``p = 2 * (1 - Phi(|z|))``, and ``CI = exp(coef +/- z_crit * SE)``.
    No refit is performed.

    Parameters
    ----------
    actual_model : object
        Unwrapped mlsurv model wrapper. Inference requires
        ``actual_model.model_.covariance_matrix_``; when absent (e.g.
        singular Hessian on a rank-deficient design) NaN arrays are
        returned with a ``SuggestionWarning`` instead of crashing.
    coef_values : array-like
        Coefficients in the model's (post-selection) feature order, which
        MUST match the covariance matrix diagonal order. This invariant
        holds because ``get_coefficients()`` returns ``model_.coef_`` and
        the covariance is the inverse Hessian built from that same
        ``coef_`` (see ``coxph.py::_attach_covariance``). If the lengths
        ever diverge, ``coef / se`` raises and we fail loud (NaN + warn).
    ci_level : float, default=0.95
        Two-sided confidence level for the HR bounds.

    Returns
    -------
    tuple of ndarray
        ``(p_value, hr_lower, hr_upper)``, each aligned to *coef_values*.
        All NaN when the covariance matrix is unavailable or unusable.
    """
    coef = np.asarray(coef_values, dtype=float)
    nan_result = (
        np.full(coef.shape, np.nan),
        np.full(coef.shape, np.nan),
        np.full(coef.shape, np.nan),
    )
    cov = getattr(getattr(actual_model, 'model_', None),
                  'covariance_matrix_', None)
    if cov is None:
        # SuggestionWarning, not InternalWarning: the default filters ignore
        # InternalWarning, and losing every confidence interval and p-value on
        # the coefficient table is not an internal detail. CoxPH's recorded
        # reason is already a complete explanation, so it replaces the generic
        # tail rather than being appended to it.
        detail = covariance_reason(actual_model) or (
            "Confidence intervals and p-values will be NaN; hazard ratios "
            "remain available."
        )
        warnings.warn(
            f"Covariance matrix unavailable for "
            f"{type(actual_model).__name__}. {detail}",
            SuggestionWarning,
            stacklevel=2,
        )
        return nan_result
    try:
        se = np.sqrt(np.diag(cov))
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            z = coef / se
            p_value = 2.0 * norm.sf(np.abs(z))
            multiplier = norm.ppf(1 - (1 - ci_level) / 2)
            hr_lower = np.exp(coef - multiplier * se)
            hr_upper = np.exp(coef + multiplier * se)
        return p_value, hr_lower, hr_upper
    except (ValueError, IndexError, TypeError, np.linalg.LinAlgError) as exc:
        warnings.warn(
            f"Could not compute Wald inference for "
            f"{type(actual_model).__name__}: {exc}. Confidence intervals "
            f"and p-values will be NaN.",
            SuggestionWarning,
            stacklevel=2,
        )
        return nan_result


def _resolve_all_features(pipeline):
    """Return the pre-selection feature names from a fitted pipeline.

    Iterates the pipeline steps in order and returns the first
    ``feature_names_in_`` found (the first step saw the full, pre-selection
    feature set). Returns None when no step carries names (e.g. the
    pipeline was fit on a bare numpy array).
    """
    for _, step in getattr(pipeline, 'steps', []):
        names = getattr(step, 'feature_names_in_', None)
        if names is not None:
            return list(names)
    return None


def compute_model_coefficients(pipeline, *, warn=True):
    """Extract model coefficients with gated multivariable inference.

    Parameters
    ----------
    pipeline : Pipeline or estimator
        Fitted model pipeline (or bare mlsurv model wrapper).
    warn : bool, default=True
        Emit a ``SuggestionWarning`` when the model exposes no
        coefficients (tree/neural models). Set False when sweeping many
        models and silently skipping non-coefficient ones is intended.

    Returns
    -------
    pd.DataFrame
        Columns ``[feature, coefficient, hazard_ratio, hr_lower_95,
        hr_upper_95, p_value]`` in pipeline feature order (not p-sorted,
        unlike ``univariate_cox``). Selector-dropped features appear with
        NaN in every value column. ``hazard_ratio``/CI/``p_value`` are NaN
        unless the model's ``hazard_ratio_valid`` flag is True (CoxPH).
        0-row DataFrame when the model has no coefficients.
    """
    actual_model = unwrap_pipeline(pipeline)

    def _no_coefficients():
        # Benign skip: this model type simply has no coefficients
        # (tree/neural). Suppressible via warn=False for the multi-model
        # sweep. stacklevel=3 points past this closure at the caller.
        if warn:
            warnings.warn(
                f"{type(actual_model).__name__} does not expose "
                f"coefficients; returning an empty DataFrame.",
                SuggestionWarning,
                stacklevel=3,
            )
        return _empty_coefficients()

    if not hasattr(actual_model, 'get_coefficients'):
        return _no_coefficients()

    all_features = _resolve_all_features(pipeline)
    selected = (
        _get_selected_features(pipeline, all_features)
        if all_features is not None else None
    )
    if selected is None:
        # With a selector, the inner model already carries post-selection
        # names; propagating the full set would cause a length mismatch.
        propagate_feature_names(actual_model, pipeline)

    try:
        coef_series = actual_model.get_coefficients()
    except (AttributeError, ValueError, TypeError, KeyError) as exc:
        # A genuine failure on a model that *advertised* get_coefficients()
        # (it passed the hasattr check). This is never an intended skip, so
        # it must surface even under warn=False (the multi-model sweep).
        warnings.warn(
            f"Could not extract coefficients from "
            f"{type(actual_model).__name__}: {exc}. Returning an "
            f"empty DataFrame.",
            InternalWarning,
            stacklevel=2,
        )
        return _empty_coefficients()
    if coef_series is None or len(coef_series) == 0:
        return _no_coefficients()

    coef = np.asarray(coef_series.values, dtype=float)

    # Gating: HR, CI, and p-value stand or fall together on the flag.
    if getattr(actual_model, 'hazard_ratio_valid', False):
        with np.errstate(over='ignore'):
            hazard_ratio = np.exp(coef)
        p_value, hr_lower, hr_upper = compute_wald_inference(
            actual_model, coef
        )
    else:
        hazard_ratio = np.full(coef.shape, np.nan)
        p_value = np.full(coef.shape, np.nan)
        hr_lower = np.full(coef.shape, np.nan)
        hr_upper = np.full(coef.shape, np.nan)

    values = {
        'coefficient': coef,
        'hazard_ratio': hazard_ratio,
        'hr_lower_95': hr_lower,
        'hr_upper_95': hr_upper,
        'p_value': p_value,
    }

    has_names = _has_string_index(coef_series)
    if all_features is not None and (has_names or len(coef) == len(all_features)):
        # Emit one row per pre-selection feature, in pipeline order, with
        # NaN for selector-dropped features. Key the value arrays by the
        # model's OWN coefficient labels (coef_series.index) when it has
        # them, NOT by the selector's selected_features_ order: every
        # value array is in coef_series order, and some models (e.g.
        # Weibull-AFT via lifelines) return coefficients in a different
        # order than the selector emitted them, so positional keying by
        # `selected` would silently put each value on the wrong feature.
        # Without string labels (names lost before fit), fall back to the
        # selector order / positional alignment, which is correct because
        # no name-based reordering can have occurred.
        if has_names:
            coef_labels = list(coef_series.index)
        elif selected is not None:
            coef_labels = selected
        else:
            coef_labels = all_features  # no selector, lengths match
        features = all_features
        values = {
            key: _align_coefficients(vals, coef_labels, all_features)
            for key, vals in values.items()
        }
    elif has_names:
        features = list(coef_series.index)
    else:
        if warn:
            warnings.warn(
                "Feature names unavailable (pipeline fit on unnamed "
                "arrays); using generic feature_{i} names.",
                InternalWarning,
                stacklevel=2,
            )
        features = [f"feature_{i}" for i in range(len(coef))]

    return pd.DataFrame({'feature': features, **values})


def _has_string_index(coef_series):
    """Return True when *coef_series* has a usable string feature index."""
    index = getattr(coef_series, 'index', None)
    if index is None or len(index) == 0:
        return False
    return isinstance(index[0], str)
