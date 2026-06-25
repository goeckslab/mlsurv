"""Tests for XGBoost AFT model.

Covers:
- XGBAFTModel fit/predict roundtrip
- Risk score direction (higher = higher risk)
- Parametric survival function (shape, range, monotonicity)
- Cumulative hazard consistency (H = -log(S))
- All three AFT distributions
- Invalid distribution rejection
- Model serialization
- Registry integration
- Search space configuration
"""

import warnings

import numpy as np
import pandas as pd
import pytest


from tests._helpers import make_feature_dataframe, make_random_survival_array


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestXGBAFTModel:
    """XGBAFTModel core functionality."""

    @pytest.fixture()
    def fitted_model(self):
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        X = make_feature_dataframe(200, p=10)
        y = make_random_survival_array(200)
        model = XGBAFTModel(n_estimators=20, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_fit_predict(self, fitted_model):
        """Basic fit/predict roundtrip with finite results."""
        model, X, _y = fitted_model
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert np.isfinite(preds).all()

    def test_predict_nontrivial(self, fitted_model):
        """Predictions should have meaningful variance (not all identical)."""
        model, X, _y = fitted_model
        preds = model.predict(X)
        assert preds.std() > 0, "All predictions are identical"

    def test_score(self, fitted_model):
        """C-index should be in a reasonable range."""
        model, X, y = fitted_model
        c_index = model.score(X, y)
        assert 0.4 <= c_index <= 1.0

    def test_predict_survival_function_default_grid(self, fitted_model):
        """predict_survival_function() without times returns DataFrame."""
        model, X, _y = fitted_model
        surv = model.predict_survival_function(X[:5])
        assert isinstance(surv, pd.DataFrame)
        # Times as index, samples as columns
        assert surv.shape == (100, 5)
        # Values in [0, 1]
        assert (surv >= 0).all().all()
        assert (surv <= 1).all().all()

    def test_predict_survival_function_with_times(self, fitted_model):
        """predict_survival_function(times=...) returns ndarray."""
        model, X, _y = fitted_model
        times = np.array([1.0, 2.0, 5.0, 10.0])
        surv = model.predict_survival_function(X[:5], times=times)
        assert isinstance(surv, np.ndarray)
        assert surv.shape == (5, 4)
        assert np.all(surv >= 0) and np.all(surv <= 1)

    def test_survival_function_monotonic(self, fitted_model):
        """S(t) should be non-increasing over time."""
        model, X, _y = fitted_model
        times = np.linspace(0.1, 20.0, 50)
        surv = model.predict_survival_function(X[:5], times=times)
        # Check non-increasing for each sample (allow small floating-point tolerance)
        diffs = np.diff(surv, axis=1)
        assert np.all(diffs <= 1e-8)

    def test_predict_survival_function_return_array(self, fitted_model):
        """return_array=True returns ndarray from default grid."""
        model, X, _y = fitted_model
        surv = model.predict_survival_function(X[:5], return_array=True)
        assert isinstance(surv, np.ndarray)
        assert surv.shape == (5, 100)

    def test_cumulative_hazard_matches_survival(self, fitted_model):
        """H(t) should equal -log(S(t))."""
        model, X, _y = fitted_model
        times = np.array([1.0, 2.0, 5.0, 10.0])
        surv = model.predict_survival_function(X[:5], times=times)
        chf = model.predict_cumulative_hazard_function(X[:5], times=times)
        expected_chf = -np.log(np.clip(surv, 1e-15, 1.0))
        np.testing.assert_allclose(chf, expected_chf, rtol=1e-8)

    def test_cumulative_hazard_default_grid(self, fitted_model):
        """predict_cumulative_hazard_function() returns DataFrame by default."""
        model, X, _y = fitted_model
        chf = model.predict_cumulative_hazard_function(X[:5])
        assert isinstance(chf, pd.DataFrame)
        assert chf.shape == (100, 5)
        assert (chf >= 0).all().all()

    def test_all_distributions(self):
        """Fit/predict works for all three AFT distributions."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        X = make_feature_dataframe(100)
        y = make_random_survival_array(100)
        for dist in ('normal', 'logistic', 'extreme'):
            model = XGBAFTModel(
                n_estimators=10, aft_loss_distribution=dist, random_state=42,
            )
            model.fit(X, y)
            preds = model.predict(X)
            assert np.isfinite(preds).all(), f"Non-finite preds for {dist}"

    def test_invalid_distribution(self):
        """Invalid aft_loss_distribution raises ValueError on fit."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel(aft_loss_distribution='invalid')
        X = make_feature_dataframe(50)
        y = make_random_survival_array(50)
        with pytest.raises(ValueError, match="aft_loss_distribution must be one of"):
            model.fit(X, y)

    def test_zero_scale_raises(self):
        """aft_loss_distribution_scale=0 raises ValueError on fit."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel(aft_loss_distribution_scale=0.0)
        X = make_feature_dataframe(50)
        y = make_random_survival_array(50)
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(X, y)

    def test_negative_times_raises(self):
        """Non-positive survival times raise ValueError on fit."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel(n_estimators=10)
        X = make_feature_dataframe(50)
        # Create y with a zero time
        y = make_random_survival_array(50)
        y['time'][0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(X, y)

    def test_serialization(self, fitted_model):
        """Model survives joblib round-trip."""
        import joblib
        import tempfile
        import os

        model, X, _y = fitted_model
        preds_before = model.predict(X)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            joblib.dump(model, path)
            loaded = joblib.load(path)
            preds_after = loaded.predict(X)
            np.testing.assert_array_equal(preds_before, preds_after)
        finally:
            os.unlink(path)

    def test_get_params(self):
        """get_params() returns all constructor parameters."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel(aft_loss_distribution='logistic')
        params = model.get_params()
        assert params['aft_loss_distribution'] == 'logistic'
        assert params['aft_loss_distribution_scale'] is None
        assert 'max_delta_step' not in params

    def test_set_params(self):
        """set_params() updates parameters."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel()
        model.set_params(aft_loss_distribution='extreme', n_estimators=50)
        assert model.aft_loss_distribution == 'extreme'
        assert model.n_estimators == 50

    def test_unfitted_predict_raises(self):
        """predict() on unfitted model raises NotFittedError."""
        pytest.importorskip("xgboost")
        from sklearn.exceptions import NotFittedError
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel()
        with pytest.raises(NotFittedError):
            model.predict(make_feature_dataframe(10))

    def test_unfitted_survival_function_raises(self):
        """predict_survival_function() on unfitted model raises NotFittedError."""
        pytest.importorskip("xgboost")
        from sklearn.exceptions import NotFittedError
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel()
        with pytest.raises(NotFittedError):
            model.predict_survival_function(make_feature_dataframe(10))

    def test_survival_at_time_zero(self, fitted_model):
        """S(0) should be 1.0 for all samples."""
        model, X, _y = fitted_model
        surv = model.predict_survival_function(X[:5], times=np.array([0.0, 1.0, 5.0]))
        np.testing.assert_array_equal(surv[:, 0], 1.0)

    def test_distribution_cdf_correctness(self):
        """Verify CDF implementations against known values."""
        pytest.importorskip("xgboost")
        from scipy import stats
        from mlsurv.models.xgbaft import XGBAFTModel

        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        model = XGBAFTModel(aft_loss_distribution='normal')
        np.testing.assert_allclose(model._distribution_cdf(z), stats.norm.cdf(z))

        model = XGBAFTModel(aft_loss_distribution='logistic')
        np.testing.assert_allclose(model._distribution_cdf(z), stats.logistic.cdf(z))

        model = XGBAFTModel(aft_loss_distribution='extreme')
        expected = 1.0 - np.exp(-np.exp(z))
        np.testing.assert_allclose(model._distribution_cdf(z), expected)

    def test_single_sample_prediction(self, fitted_model):
        """Predict on a single sample should return correct shapes."""
        model, X, _y = fitted_model
        preds = model.predict(X[:1])
        assert preds.shape == (1,)
        surv = model.predict_survival_function(X[:1], times=np.array([1.0, 5.0]))
        assert surv.shape == (1, 2)

    def test_cumulative_hazard_return_array(self, fitted_model):
        """predict_cumulative_hazard_function(return_array=True) returns ndarray."""
        model, X, _y = fitted_model
        chf = model.predict_cumulative_hazard_function(X[:5], return_array=True)
        assert isinstance(chf, np.ndarray)
        assert chf.shape == (5, 100)
        assert np.all(np.isfinite(chf))
        assert np.all(chf >= 0)


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestXGBAFTCalibration:
    """Auto-MLE sigma calibration on training residuals."""

    def test_default_scale_is_none(self):
        """Default aft_loss_distribution_scale is None (auto-MLE)."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel()
        assert model.aft_loss_distribution_scale is None

    @pytest.mark.parametrize("dist", ["normal", "logistic", "extreme"])
    def test_sigma_attr_set_after_fit(self, dist):
        """fit() populates sigma_ for every supported distribution.

        Uses n=400 so the hold-out path (>=300 viable) actually fires
        rather than the in-sample fallback.
        """
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel(
            n_estimators=10, max_depth=2,
            aft_loss_distribution=dist, random_state=42,
        )
        model.fit(make_feature_dataframe(400, p=5), make_random_survival_array(400))
        assert isinstance(model.sigma_, float)
        assert np.isfinite(model.sigma_) and model.sigma_ > 0

    def test_holdout_path_no_internal_fallback(self):
        """On clearly-viable data the hold-out path runs without a
        methodological-fallback warning. Guards against silent regression
        into the in-sample fallback when the viability gate shifts.
        """
        pytest.importorskip("xgboost")
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.models.xgbaft import XGBAFTModel

        X = make_feature_dataframe(400, p=5)
        y = make_random_survival_array(400)
        model = XGBAFTModel(n_estimators=10, max_depth=2, random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)
        assert not any(
            issubclass(x.category, MethodologicalWarning) for x in w
        ), "Unexpected MethodologicalWarning on viable hold-out dataset"

    @pytest.mark.integration
    def test_sigma_responds_to_residual_scale(self):
        """MLE sigma scales with the noise level of the data.

        Generates two log-normal AFT datasets that differ only in
        sigma_true, fits a low-capacity XGBoost-AFT to each, and
        verifies that the recovered sigma values track the true scales
        (the dataset with larger noise has the larger sigma_).

        We don't assert close recovery to truth in absolute terms here —
        XGBoost-AFT will inevitably over-fit training to some degree,
        compressing sigma below the noise level — but the relative
        ordering is robust and that's what matters for calibration.
        """
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel

        def fit_with_truth(sigma_true):
            rng = np.random.RandomState(0)
            n, p = 600, 3
            X = pd.DataFrame(rng.randn(n, p), columns=[f'f{i}' for i in range(p)])
            beta = rng.randn(p) * 0.4
            log_t = X.values @ beta + sigma_true * rng.randn(n) + 2.0
            times = np.exp(log_t)
            events = rng.uniform(0, 1, n) > 0.2  # ~80% events
            y = np.array(
                list(zip(events.astype(bool), times)),
                dtype=[('event', bool), ('time', float)],
            )
            # Low-capacity ensemble: 10 stumps. Limits over-fitting so
            # residuals reflect mostly the true noise.
            model = XGBAFTModel(
                n_estimators=10, max_depth=2, learning_rate=0.3,
                random_state=0,
            )
            model.fit(X, y)
            return model.sigma_

        sigma_low = fit_with_truth(0.3)
        sigma_high = fit_with_truth(0.9)
        assert sigma_high > sigma_low, (
            f"MLE sigma did not respond to noise scale: "
            f"sigma_low={sigma_low:.3f}, sigma_high={sigma_high:.3f}"
        )
        # Sanity: the recovered values should be in a plausible range
        # (not collapsed to the optimizer floor, not blown up)
        assert 0.05 < sigma_low < 1.0
        assert 0.2 < sigma_high < 2.0

    def test_user_supplied_scale_skips_mle(self):
        """An explicit aft_loss_distribution_scale bypasses MLE."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel(
            n_estimators=10, aft_loss_distribution_scale=2.5, random_state=42,
        )
        model.fit(make_feature_dataframe(100), make_random_survival_array(100))
        assert model.sigma_ == 2.5

    def test_no_events_falls_back(self):
        """Zero events → MethodologicalWarning + sigma_=1.0 fallback.

        We expect the early-return path (no booster trained) — pin the
        warning message so future refactors that re-route through a
        different fallback are flagged.
        """
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        from mlsurv._warnings import MethodologicalWarning

        X = make_feature_dataframe(60)
        y = make_random_survival_array(60)
        y['event'] = False  # all censored

        model = XGBAFTModel(n_estimators=10, random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)

        assert model.sigma_ == 1.0
        msgs = [
            str(x.message) for x in w
            if issubclass(x.category, MethodologicalWarning)
        ]
        assert any("zero events" in m for m in msgs), (
            f"Expected zero-events MethodologicalWarning, got: {msgs}"
        )

    def test_optimizer_failure_falls_back(self, monkeypatch):
        """If minimize_scalar reports failure, sigma falls back to 1.0
        and a MethodologicalWarning fires (the returned sigma directly
        drives S(t|x), so silent fallback is a real risk).
        """
        pytest.importorskip("xgboost")

        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.models import xgbaft as xgbaft_mod
        from mlsurv.models.xgbaft import XGBAFTModel

        class _FailedResult:
            success = False
            x = float("nan")
            message = "forced failure"

        monkeypatch.setattr(
            xgbaft_mod.optimize, "minimize_scalar",
            lambda *a, **kw: _FailedResult(),
        )

        model = XGBAFTModel(n_estimators=10, random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(make_feature_dataframe(400, p=5), make_random_survival_array(400))

        assert model.sigma_ == 1.0
        assert any(
            issubclass(x.category, MethodologicalWarning)
            and "failed to converge" in str(x.message)
            for x in w
        )

    def test_negative_scale_raises(self):
        """Negative aft_loss_distribution_scale raises ValueError on fit."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel
        model = XGBAFTModel(aft_loss_distribution_scale=-0.5)
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(make_feature_dataframe(50), make_random_survival_array(50))

    def test_serialization_preserves_sigma(self):
        """sigma_ survives joblib round-trip and predictions match."""
        import joblib
        import os
        import tempfile

        pytest.importorskip("xgboost")
        from mlsurv.models.xgbaft import XGBAFTModel

        X = make_feature_dataframe(150, p=5)
        y = make_random_survival_array(150)
        model = XGBAFTModel(n_estimators=20, random_state=42).fit(X, y)
        sigma_before = model.sigma_
        preds_before = model.predict(X)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            joblib.dump(model, path)
            loaded = joblib.load(path)
            assert loaded.sigma_ == sigma_before
            np.testing.assert_array_equal(loaded.predict(X), preds_before)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestXGBAFTRegistry:
    """Registry integration for xgbaft."""

    def test_model_in_registry(self):
        from mlsurv.models import MODEL_REGISTRY
        assert 'xgbaft' in MODEL_REGISTRY

    def test_get_model(self):
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model
        from mlsurv.models.xgbaft import XGBAFTModel
        model = get_model('xgbaft')
        assert isinstance(model, XGBAFTModel)

    def test_get_model_case_insensitive(self):
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model
        model = get_model('XGBAFT')
        assert model.name == 'xgbaft'

    def test_list_models_includes_xgbaft(self):
        from mlsurv.models import list_models
        assert 'xgbaft' in list_models()

    def test_metadata(self):
        from mlsurv.models import get_model_metadata
        meta = get_model_metadata('xgbaft')
        assert meta['display_name'] == 'XGBoost-AFT'
        assert meta['color'] == '#AA4499'
        assert meta['complexity'] == 'tree_ensemble'
        assert meta['supports_random_state'] is True
        assert meta['has_internal_parallelism'] is True


# ---------------------------------------------------------------------------
# Search space tests
# ---------------------------------------------------------------------------

class TestXGBAFTSearchSpace:
    """Search space configuration for xgbaft."""

    def test_default_space_exists(self):
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES
        assert 'xgbaft' in DEFAULT_SPACES

    def test_calibration_params_not_searched(self):
        """Sigma + family are not in the default search space.

        Sigma is MLE-fit on training residuals inside fit(); family is
        fixed at the model default. Tuning these via a rank-based metric
        is noise (c-index is invariant to both), so they're excluded by
        default. Users can still override with a custom search space.
        """
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES
        space = DEFAULT_SPACES['xgbaft']
        assert 'aft_loss_distribution' not in space
        assert 'aft_loss_distribution_scale' not in space

    def test_no_max_delta_step(self):
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES
        assert 'max_delta_step' not in DEFAULT_SPACES['xgbaft']

    def test_auto_n_trials(self):
        from mlsurv.tuning.search_spaces import auto_n_trials
        trials = auto_n_trials('xgbaft')
        assert trials == 60

    def test_tuner_in_registry(self):
        from mlsurv.tuning.tuners import TUNER_REGISTRY
        assert 'xgbaft' in TUNER_REGISTRY

    def test_epv_constraints(self):
        from mlsurv.tuning.search_spaces import _EPV_CONSTRAINTS
        assert 'xgbaft' in _EPV_CONSTRAINTS
        constraints = _EPV_CONSTRAINTS['xgbaft']
        assert 'max_depth' in constraints
        assert 'min_child_weight' in constraints
