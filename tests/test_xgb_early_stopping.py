"""Validation-based early stopping for the XGBoost survival models.

Both XGBoost wrappers (``xgbcox``, ``xgbaft``) early-stop by default,
mirroring ``gbsa`` / ``deepsurv`` / ``deephit``. These tests check that:

- early stopping fires on an overfit-prone configuration (trimmed round count),
- ``early_stopping_rounds=None`` trains the full ``n_estimators`` rounds,
- tiny data auto-disables early stopping with a "too small" warning,
- an out-of-range ``validation_fraction`` raises ``ValueError``,
- the trimmed booster round-trips through pickle with matching predictions.
"""

import pickle
import warnings

import numpy as np
import pytest

from tests._helpers import make_feature_dataframe, make_random_survival_array


pytestmark = pytest.mark.integration


def _model(name, **kwargs):
    pytest.importorskip("xgboost")
    from mlsurv.models import XGBAFTModel, XGBCoxModel
    return {"xgbcox": XGBCoxModel, "xgbaft": XGBAFTModel}[name](**kwargs)


def _rounds_used(model):
    """Boosting rounds actually used by the production model."""
    if model.name == "xgbcox":
        best_iter = getattr(model.model_, "best_iteration", None)
        return None if best_iter is None else best_iter + 1
    return model.booster_.num_boosted_rounds()


@pytest.fixture(scope="module")
def overfit_data():
    """A dataset large enough to hold out a validation fold."""
    X = make_feature_dataframe(400, p=8)
    y = make_random_survival_array(400)
    return X, y


# Overfit-prone: deep trees, high learning rate, many estimators. Early
# stopping should truncate well before n_estimators.
_OVERFIT = dict(n_estimators=300, max_depth=8, learning_rate=0.3, random_state=42)


@pytest.mark.parametrize("name", ["xgbcox", "xgbaft"])
def test_early_stopping_fires(name, overfit_data):
    """Early stopping trims the round count below n_estimators."""
    X, y = overfit_data
    model = _model(name, early_stopping_rounds=10, validation_fraction=0.1,
                   **_OVERFIT)
    model.fit(X, y)

    rounds = _rounds_used(model)
    assert rounds is not None
    assert rounds < model.n_estimators, (
        f"expected early stopping to trim rounds, got {rounds} of "
        f"{model.n_estimators}"
    )
    preds = model.predict(X)
    assert preds.shape == (len(y),) and np.isfinite(preds).all()


@pytest.mark.parametrize("name", ["xgbcox", "xgbaft"])
def test_disabled_when_none(name, overfit_data):
    """early_stopping_rounds=None trains the full n_estimators rounds."""
    X, y = overfit_data
    model = _model(name, early_stopping_rounds=None, n_estimators=25,
                   random_state=42)
    model.fit(X, y)

    if name == "xgbcox":
        # best_iteration is only set when early stopping is active.
        assert getattr(model.model_, "best_iteration", None) is None
    else:
        assert _rounds_used(model) == 25
        assert model.n_rounds_ == 25


@pytest.mark.parametrize("name", ["xgbcox", "xgbaft"])
def test_small_data_disables_with_warning(name):
    """Tiny data disables early stopping (warns) and trains the full rounds."""
    X = make_feature_dataframe(12, p=4)
    y = make_random_survival_array(12)
    model = _model(name, early_stopping_rounds=5, n_estimators=15,
                   random_state=42)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
        # Both the sigma-calibration fallback (xgbaft) and the early-stopping
        # split (xgbcox) report "too small"; xgbaft additionally warns that
        # early stopping was disabled.
        too_small = [w for w in caught if "too small" in str(w.message)]
    assert len(too_small) >= 1, "expected a 'too small' warning"

    # Early stopping was disabled, so the full round count is used.
    if name == "xgbcox":
        assert getattr(model.model_, "best_iteration", None) is None
    else:
        assert model.n_rounds_ == model.n_estimators

    # Model still fits and predicts.
    preds = model.predict(X)
    assert preds.shape == (12,) and np.isfinite(preds).all()


def test_xgbaft_user_sigma_early_stopping_fires(overfit_data):
    """xgbaft user-sigma path: validation_fraction split trims rounds, sigma kept."""
    X, y = overfit_data
    model = _model("xgbaft", early_stopping_rounds=10, validation_fraction=0.1,
                   aft_loss_distribution_scale=1.5, **_OVERFIT)
    model.fit(X, y)

    assert model.sigma_ == 1.5, "user-supplied sigma must be preserved"
    assert model.n_rounds_ < model.n_estimators, "expected early stopping to trim"
    assert _rounds_used(model) == model.n_rounds_


def test_xgbaft_user_sigma_small_data_disables():
    """xgbaft user-sigma + tiny data: the early-stopping split disables and warns."""
    X = make_feature_dataframe(12, p=4)
    y = make_random_survival_array(12)
    model = _model("xgbaft", early_stopping_rounds=5,
                   aft_loss_distribution_scale=1.0, n_estimators=15,
                   random_state=42)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
        disabled = [w for w in caught
                    if "Disabling early stopping" in str(w.message)]
    assert len(disabled) >= 1, "expected a 'Disabling early stopping' warning"
    assert model.n_rounds_ == model.n_estimators


def test_invalid_validation_fraction_raises_xgbcox(overfit_data):
    """xgbcox: out-of-range validation_fraction raises ValueError."""
    X, y = overfit_data
    for bad in [0.0, 1.0, -0.1, 1.5]:
        model = _model("xgbcox", early_stopping_rounds=5,
                       validation_fraction=bad, n_estimators=10,
                       random_state=42)
        with pytest.raises(ValueError, match="validation_fraction must be in"):
            model.fit(X, y)


def test_invalid_validation_fraction_raises_xgbaft(overfit_data):
    """xgbaft: validation_fraction drives the user-sigma early-stopping split,
    so an out-of-range value raises there."""
    X, y = overfit_data
    for bad in [0.0, 1.0, -0.1, 1.5]:
        model = _model("xgbaft", early_stopping_rounds=5,
                       validation_fraction=bad, aft_loss_distribution_scale=1.0,
                       n_estimators=10, random_state=42)
        with pytest.raises(ValueError, match="validation_fraction must be in"):
            model.fit(X, y)


@pytest.mark.parametrize("name", ["xgbcox", "xgbaft"])
def test_serialization_roundtrips_trimmed_booster(name, overfit_data):
    """Pickle round-trips the early-stopped model with matching predictions."""
    X, y = overfit_data
    model = _model(name, early_stopping_rounds=10, validation_fraction=0.1,
                   **_OVERFIT)
    model.fit(X, y)

    before = model.predict(X)
    restored = pickle.loads(pickle.dumps(model))
    after = restored.predict(X)

    assert np.allclose(before, after)
    assert _rounds_used(restored) == _rounds_used(model)


def test_xgbaft_sigma_finite_and_survival_monotone(overfit_data):
    """xgbaft: early stopping leaves a finite positive sigma and a monotone S(t)."""
    X, y = overfit_data
    model = _model("xgbaft", early_stopping_rounds=10, validation_fraction=0.1,
                   **_OVERFIT)
    model.fit(X, y)

    assert np.isfinite(model.sigma_) and model.sigma_ > 0
    sf = model.predict_survival_function(X[:5], times=[1.0, 3.0, 6.0, 10.0])
    assert sf.shape == (5, 4)
    assert np.all((sf >= 0) & (sf <= 1))
    # Survival is non-increasing in time.
    assert np.all(np.diff(sf, axis=1) <= 1e-9)


# --------------------------------------------------------------------------
# Report rendering — fast tier (no model training).
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "actual, requested, expected",
    [
        (6, 300, "Early stopping: 6 of 300 boosting rounds used."),
        (300, 300, "300 boosting rounds."),
        (None, 300, "300 boosting rounds."),
        (50, None, "50 boosting rounds."),
    ],
)
def test_add_boosting_rounds_rendering(actual, requested, expected):
    """The report helper flags early stopping only when the count was trimmed."""
    from mlsurv.reports.sections._trained_details import _TrainedDetailsSubsection

    md = []
    _TrainedDetailsSubsection._add_boosting_rounds(md, actual, requested)
    assert any(expected in line for line in md), f"missing {expected!r} in {md}"
