"""Tests for opt-in CoxPH ridge-penalty (alpha) tuning.

CoxPH is the unpenalized baseline: ``DEFAULT_SPACES['coxph'] == {}`` and a
plain ``tune('coxph')`` stays preprocessing-only (alpha=0). Supplying an
explicit model space, e.g. ``search_spaces={'model': {'alpha': (1e-4, 10.0,
'log')}}``, opts into tuning alpha. A tuned ``alpha > 0`` regularizes the fit,
which makes the hazard ratios / Wald CIs / p-values conditional on the penalty,
so the model emits a ``MethodologicalWarning`` and reports
``hazard_ratio_valid=False`` (which suppresses HR output downstream).
"""

import warnings

import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning
from mlsurv.models import CoxPHModel
from mlsurv.viz._data import is_hazard_ratio_valid


def _best_alpha(result):
    """Tuned alpha regardless of component prefix."""
    return result.best_params_.get("model__alpha", result.best_params_.get("alpha"))


def _setup_learner(small_data):
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(
        test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1,
        feature_selector=None,
    )
    return learner


# ---------------------------------------------------------------------------
# hazard_ratio_valid property — keyed on the constructor alpha (no fit needed)
# ---------------------------------------------------------------------------

def test_hazard_ratio_valid_unpenalized():
    """alpha == 0 (the default) keeps exp(coef) a valid multivariable HR."""
    assert CoxPHModel().hazard_ratio_valid is True
    assert CoxPHModel(alpha=0).hazard_ratio_valid is True


def test_hazard_ratio_valid_penalized():
    """A user-specified ridge penalty flips the HR-validity flag off."""
    assert CoxPHModel(alpha=0.5).hazard_ratio_valid is False
    assert CoxPHModel(alpha=1e-4).hazard_ratio_valid is False


def test_penalized_coxph_excluded_from_hr_viz_helper():
    """The shared viz gate drops a penalized CoxPH and keeps an unpenalized one."""
    assert is_hazard_ratio_valid(CoxPHModel(alpha=0.5)) is False
    assert is_hazard_ratio_valid(CoxPHModel(alpha=0)) is True


# ---------------------------------------------------------------------------
# fit-time MethodologicalWarning
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_penalized_fit_warns_and_flags(small_data):
    """Fitting alpha>0 emits the inference caveat and reports HR-invalid."""
    X, y = small_data
    model = CoxPHModel(alpha=0.5)
    with pytest.warns(MethodologicalWarning, match="ridge penalty"):
        model.fit(X, y)
    assert model.hazard_ratio_valid is False


@pytest.mark.integration
def test_unpenalized_fit_emits_no_ridge_warning(small_data):
    """The default alpha=0 fit must not emit the ridge-penalty caveat."""
    X, y = small_data
    model = CoxPHModel(alpha=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    ridge = [
        w for w in caught
        if issubclass(w.category, MethodologicalWarning)
        and "ridge penalty" in str(w.message)
    ]
    assert ridge == []
    assert model.hazard_ratio_valid is True


@pytest.mark.integration
def test_negative_alpha_rejected_on_fit(small_data):
    """A negative ridge penalty must raise on fit, not silently fall back.

    Boundary case (CONTRIBUTING.md): sksurv rejects alpha<0 with an
    InvalidParameterError (a ValueError subclass), which coxph's
    _fit_and_check classifies as a caller error and re-raises instead of
    routing it into the escalating alpha fallback.
    The ridge caveat (gated on ``alpha > 0``) must not fire for a negative.
    """
    X, y = small_data
    model = CoxPHModel(alpha=-0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            model.fit(X, y)
    ridge = [
        w for w in caught
        if issubclass(w.category, MethodologicalWarning)
        and "ridge penalty" in str(w.message)
    ]
    assert ridge == []


# ---------------------------------------------------------------------------
# learner.tune — opt-in alpha search vs. unchanged default
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_tune_coxph_alpha_recorded_and_applied(small_data):
    """An explicit alpha space is searched, recorded, and applied to the fit."""
    learner = _setup_learner(small_data)
    result = learner.tune(
        "coxph", n_trials=4, verbose=False, evaluate=False,
        search_spaces={"model": {"alpha": (1e-4, 10.0, "log")}},
    )

    assert result.is_tuned is True
    alpha = _best_alpha(result)
    assert alpha is not None, "alpha must be recorded in best_params_"
    assert 1e-4 <= alpha <= 10.0, f"tuned alpha {alpha} outside declared range"

    fitted = result.model.named_steps["model"]
    assert fitted.alpha == alpha and fitted.alpha > 0
    assert fitted.hazard_ratio_valid is False


@pytest.mark.integration
def test_tune_alpha_surfaces_methodological_warning(small_data):
    """Tuning into alpha>0 surfaces the caveat on the tune() path, not just
    a direct fit().

    The CV path captures and re-emits fold warnings (serialized across joblib),
    and the final best-model re-fit emits in the parent process — this pins
    that the caveat reaches the user somewhere across the tune call.
    """
    learner = _setup_learner(small_data)
    with pytest.warns(MethodologicalWarning, match="ridge penalty"):
        learner.tune(
            "coxph", n_trials=4, verbose=False, evaluate=False,
            search_spaces={"model": {"alpha": (1e-4, 10.0, "log")}},
        )


@pytest.mark.integration
def test_tune_coxph_default_preprocessing_only(small_data):
    """Plain tune('coxph') with no model space stays alpha=0 / HR-valid."""
    learner = _setup_learner(small_data)
    result = learner.tune("coxph", n_trials=4, verbose=False, evaluate=False)

    # No searchable params -> tuning is skipped; no alpha is recorded.
    assert result.best_params_ in (None, {}), result.best_params_
    fitted = result.model.named_steps["model"]
    assert fitted.alpha == 0
    assert fitted.hazard_ratio_valid is True
