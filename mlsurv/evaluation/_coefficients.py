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

``importance.py::_resolve_coef_labels`` decides which feature each
coefficient belongs to, here and for ``EvaluationResult.importance``. Two
surfaces reporting the same fitted model must not disagree about that, and
some models (lifelines' Weibull-AFT) return coefficients in a different
order than the selector emitted them, so a second local implementation of
the rule would be a second chance to get it wrong. Sharing the resolver
settles the label question the same way on both surfaces. It does not make
the surfaces identical: they align onto different feature lists (test
columns there, pipeline pre-selection names here) and they degrade
differently, since a NaN coefficient column still leaves a usable
permutation-importance row while a coefficient table has nothing left to
report.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .._shared import covariance_reason
from .._warnings import InternalWarning, MetricFailureWarning, SuggestionWarning
from ..logging_utils import get_logger, log_metric_failure
from ..utils import propagate_feature_names, unwrap_pipeline
from .importance import (
    _COEF_EXTRACTION_ERRORS,
    _align_coefficients,
    _get_selected_features,
    _resolve_coef_labels,
    _string_coef_index,
)

logger = get_logger("coefficients")

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


def _warn_coefficients_failed(actual_model, verb, reason, consequence,
                              model_name=None):
    """Report that a model's coefficients could not be produced or placed.

    Fires as a ``MetricFailureWarning``, not an ``InternalWarning``: the
    latter is suppressed by mlsurv's default filters, so a failure reported
    that way leaves the user with an empty coefficient table and no
    explanation anywhere.

    Because ``MetricFailureWarning`` is an ``IterativeWarning``, deduplicated
    per unique message text, the guarantee is one report per distinct message
    per session, not one per failure. The message therefore has to identify
    the model, or the second model to fail the same way says nothing.

    Parameters
    ----------
    actual_model : object
        Unwrapped model wrapper, used for its class name.
    verb : str
        What could not be done, e.g. ``'extract'`` or ``'align'``. Keeps the
        two failure classes distinct in the message and in the dedup key.
    reason : BaseException or str
        What went wrong. A string is wrapped in a ``ValueError`` because
        ``log_metric_failure`` wants an exception; nothing was raised in that
        case, so the log record's type is a formality and its traceback is
        empty.
    consequence : str
        A complete sentence saying what the caller returns, appended to the
        message.
    model_name : str, optional
        Registry name (``'coxnet'``). Included when known so this surface
        names the model the way the user asked for it, matching
        ``importance.py::_warn_coefficients_unresolved``. Falling back to the
        class name alone still keeps messages distinct, because the registry
        maps one name per wrapper class.
    """
    exc = reason if isinstance(reason, BaseException) else ValueError(str(reason))
    cls = type(actual_model).__name__
    label = model_name or cls
    # Name the class alongside the registry name, but not twice when they are
    # the same string (no model_name passed).
    named = label if label == cls else f"{label} ({cls})"
    log_metric_failure(
        logger, "model_coefficients", exc,
        model_name=label, warn_user=True,
        warn_category=MetricFailureWarning,
        user_message=f"Could not {verb} coefficients for {named}: "
                     f"{reason}. {consequence}",
    )


def _warn_extraction_failed(actual_model, reason, model_name=None):
    """Report that ``get_coefficients()`` produced nothing usable."""
    _warn_coefficients_failed(
        actual_model, "extract", reason,
        "No coefficient rows are reported for this model.", model_name,
    )


def _warn_alignment_failed(actual_model, reason, consequence, model_name=None):
    """Report that coefficients could not be placed on their features."""
    _warn_coefficients_failed(
        actual_model, "align", reason, consequence, model_name,
    )


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


def compute_model_coefficients(pipeline, *, warn=True, model_name=None):
    """Extract model coefficients with gated multivariable inference.

    Parameters
    ----------
    pipeline : Pipeline or estimator
        Fitted model pipeline (or bare mlsurv model wrapper).
    warn : bool, default=True
        Gate on the two benign-skip notices, both ``SuggestionWarning``:
        the model exposes no coefficients (tree/neural), and nothing in the
        pipeline or the model carries feature names, so rows are numbered
        rather than named. Set False when sweeping many models and silently
        skipping non-coefficient ones is intended. It does not gate
        failures: those warn either way.
    model_name : str, optional
        Registry name (``'coxnet'``) used in failure messages, so they name
        the model the caller asked for rather than its wrapper class.

    Returns
    -------
    pd.DataFrame
        Columns ``[feature, coefficient, hazard_ratio, hr_lower_95,
        hr_upper_95, p_value]`` in pipeline feature order (not p-sorted,
        unlike ``univariate_cox``). Selector-dropped features appear with
        NaN in every value column. ``hazard_ratio``/CI/``p_value`` are NaN
        unless the model's ``hazard_ratio_valid`` flag is True (CoxPH).
        0-row DataFrame when the model has no coefficients, and also on
        every failure to produce a usable table: extraction raised, the
        model returned nothing, the labels do not align, or no value is
        finite. Every one of those warns with ``MetricFailureWarning``
        regardless of *warn*, since they are failures rather than skips. So
        a non-empty return always carries at least one real coefficient.

        Non-empty does not guarantee the feature column is *yours*. When
        the labels cannot be tied to the pipeline's features the rows are
        still returned, keyed by the model's own coefficient index or by
        position, and that also warns. Absence of a warning is what says
        the feature column is the pipeline's.
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
    except _COEF_EXTRACTION_ERRORS as exc:
        # A genuine failure on a model that *advertised* get_coefficients()
        # (it passed the hasattr check). This is never an intended skip, so
        # it surfaces even under warn=False (the multi-model sweep). Shares
        # importance.py's tuple so the two surfaces agree on what counts as a
        # recognizable coefficient problem, the same way they now agree on
        # label resolution.
        _warn_extraction_failed(actual_model, exc, model_name)
        return _empty_coefficients()
    if coef_series is None:
        # The documented "this model has no coefficients" return.
        return _no_coefficients()

    try:
        # getattr, not .values: the contract is a Series (models/base.py), but
        # the docstring accepts any wrapper, and a bare ndarray has no
        # .values. atleast_1d so a scalar return has a length instead of
        # raising on len(). All of it guarded, including the emptiness test,
        # because this runs inside the unguarded loop in
        # SurvivalLearner.coefficients, where a raise costs the other models
        # their rows.
        coef = np.atleast_1d(
            np.asarray(getattr(coef_series, 'values', coef_series),
                       dtype=float)
        )
        if coef.ndim != 1:
            raise ValueError(
                f"expected one coefficient per feature, got a "
                f"{coef.ndim}-dimensional array"
            )
    except (ValueError, TypeError) as exc:
        # An extraction problem, not an alignment one: the model handed back
        # something that is not a vector of numbers, so there is nothing to
        # place.
        _warn_extraction_failed(actual_model, exc, model_name)
        return _empty_coefficients()

    if coef.size == 0:
        # Not the same as returning None: the model advertised
        # get_coefficients(), was asked, and handed back nothing. A failure,
        # so it is reported rather than routed through the benign skip, which
        # warn=False silences.
        _warn_extraction_failed(
            actual_model, "the model returned 0 coefficients", model_name,
        )
        return _empty_coefficients()

    if not np.isfinite(coef).any():
        # The fit produced nothing usable, which is true before any question
        # of labeling arises. Reported here so the user gets one accurate
        # message rather than a labeling warning followed by a contradicting
        # one. isfinite rather than notna because an overflowed value is no
        # more usable than a NaN, while an exactly-zero L1 coefficient is.
        _warn_extraction_failed(
            actual_model,
            f"no finite value among {coef.size} coefficients; the fit "
            f"produced only NaN or Inf",
            model_name,
        )
        return _empty_coefficients()

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

    # importance.py::_resolve_coef_labels decides which label keys each value,
    # so both coefficient surfaces answer that the same way. Its docstring
    # spells out the rungs; the one worth knowing here is that the model's own
    # coefficient index wins over the selector's emission order, because
    # lifelines sorts Weibull-AFT covariates and positional keying would
    # silently put each value on the wrong feature.
    labels = None
    if all_features is not None:
        labels = _resolve_coef_labels(coef_series, selected, all_features)

    if labels is not None:
        # One row per pre-selection feature, in pipeline order, with NaN for
        # selector-dropped features.
        features = all_features
        try:
            values = {
                key: _align_coefficients(vals, labels, all_features)
                for key, vals in values.items()
            }
        except (ValueError, TypeError) as exc:
            # SurvivalLearner.coefficients loops every trained model with no
            # guard of its own, so raising here would cost the other models
            # their rows (and the report the whole feature table).
            _warn_alignment_failed(
                actual_model, exc,
                "No coefficient rows are reported for this model.",
                model_name,
            )
            return _empty_coefficients()
        # Only this branch can lose values: alignment drops any label that is
        # not a pipeline feature, so a vector that had finite values before
        # can arrive here with none of them landing. The other branches keep
        # `values` exactly as the finiteness check above already saw it.
        if not np.isfinite(np.asarray(values['coefficient'],
                                      dtype=float)).any():
            _warn_alignment_failed(
                actual_model,
                f"none of the {coef.size} coefficients landed on a pipeline "
                f"feature; the labels describe a different feature space",
                "No coefficient rows are reported for this model.",
                model_name,
            )
            # 0 rows, not the NaN skeleton: a NaN-filled frame passes the
            # `if not df.empty` check in SurvivalLearner.coefficients and
            # would leak into feature_summary() and the exports, carrying
            # none of the warning that explains it.
            return _empty_coefficients()
    elif (index := _string_coef_index(coef_series)) is not None:
        if all_features is not None:
            # The resolver gave up, so nothing ties these labels to the
            # pipeline's features. Reporting them is still the most useful
            # thing available, but the user has to know the feature column is
            # the model's own vocabulary rather than theirs.
            _warn_alignment_failed(
                actual_model,
                f"{len(coef)} coefficients could not be matched to "
                f"{len(all_features)} pipeline features",
                "Rows are labeled by the model's own coefficient index, "
                "which matches no pipeline feature.",
                model_name,
            )
        features = index
    elif all_features is not None:
        # The pipeline does have names; they just could not be matched to a
        # coefficient index that carries none. Saying "fit on a DataFrame"
        # here would be false advice, so this is reported as the placement
        # failure it is, and ungated: the rows below are labeled by position
        # and would otherwise enter feature_summary() looking like features.
        _warn_alignment_failed(
            actual_model,
            f"{len(coef)} coefficients carry no usable names and could not "
            f"be matched to {len(all_features)} pipeline features",
            "Rows are numbered by position rather than named.",
            model_name,
        )
        # Not `feature_{i}`: this branch runs precisely when the pipeline has
        # real feature names, and `feature_0` is a name a real feature can
        # have. feature_summary() joins on this column, so an ambiguous label
        # would silently join a positional row onto a real feature.
        features = [f"position_{i}" for i in range(len(coef))]
    else:
        # No names anywhere, so positional labels are the only option and
        # the advice is real. A benign skip, hence gated on warn, and a
        # SuggestionWarning rather than the default-filtered InternalWarning.
        if warn:
            warnings.warn(
                "Feature names unavailable (pipeline fit on unnamed "
                "arrays); using generic feature_{i} names. Fit on a "
                "DataFrame to keep the real names.",
                SuggestionWarning,
                stacklevel=2,
            )
        features = [f"feature_{i}" for i in range(len(coef))]

    # No finiteness gate here: the two above cover every path between them.
    # A fit with no usable value is caught before labeling, and alignment is
    # the only step that can drop the usable ones, so it checks its own
    # result. Anything reaching this line has at least one real coefficient.
    return pd.DataFrame({'feature': features, **values})
