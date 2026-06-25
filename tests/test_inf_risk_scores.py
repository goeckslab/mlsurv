"""Tests for inf risk score handling.

Covers:
- XGBCox.predict() returns finite log-hazard values
- max_delta_step default and forwarding
- _compute_concordance graceful NaN on inf/NaN scores
- evaluate() warning for non-finite risk scores
"""

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests._helpers import make_feature_dataframe, make_random_survival_array


# ---------------------------------------------------------------------------
# XGBCox model tests
# ---------------------------------------------------------------------------

class TestXGBCoxPredict:
    """XGBCox.predict() should return finite log-hazard values."""

    def test_predict_returns_finite(self):
        """predict() returns all-finite values on a moderately-sized dataset."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbcox import XGBCoxModel

        X = make_feature_dataframe(200, p=10)
        y = make_random_survival_array(200)
        model = XGBCoxModel(n_estimators=50, random_state=42)
        model.fit(X, y)

        risk = model.predict(X)
        assert np.all(np.isfinite(risk)), "predict() returned non-finite values"

    def test_predict_returns_log_hazard(self):
        """predict() returns log-hazard (can be negative), not exp(margin)."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbcox import XGBCoxModel

        X = make_feature_dataframe(100, p=5)
        y = make_random_survival_array(100)
        model = XGBCoxModel(n_estimators=30, random_state=42)
        model.fit(X, y)

        risk = model.predict(X)
        # Log-hazard should have both positive and negative values
        # (centered near 0), unlike exp(margin) which is always positive
        assert risk.min() < 0, "Expected some negative log-hazard values"

    def test_max_delta_step_default(self):
        """Default max_delta_step is 0.7."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbcox import XGBCoxModel

        model = XGBCoxModel()
        assert model.max_delta_step == 0.7

    def test_max_delta_step_forwarded_to_xgb(self):
        """max_delta_step is forwarded to the XGBRegressor."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbcox import XGBCoxModel

        X = make_feature_dataframe(50, p=3)
        y = make_random_survival_array(50)
        model = XGBCoxModel(n_estimators=10, max_delta_step=1.5, random_state=42)
        model.fit(X, y)

        assert model.model_.max_delta_step == 1.5

    def test_max_delta_step_in_get_params(self):
        """max_delta_step appears in sklearn get_params()."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbcox import XGBCoxModel

        model = XGBCoxModel(max_delta_step=0.5)
        params = model.get_params()
        assert 'max_delta_step' in params
        assert params['max_delta_step'] == 0.5

    def test_survival_function_valid_after_fix(self):
        """predict_survival_function() returns valid curves (no double-exp)."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbcox import XGBCoxModel

        X = make_feature_dataframe(100, p=5)
        y = make_random_survival_array(100)
        model = XGBCoxModel(n_estimators=30, random_state=42)
        model.fit(X, y)

        times = np.array([1.0, 3.0, 5.0, 7.0])
        surv = model.predict_survival_function(X, times=times)

        # Valid survival curve: values in [0, 1], non-increasing over time
        assert surv.shape == (100, 4)
        assert np.all(surv >= 0.0) and np.all(surv <= 1.0), \
            "Survival probabilities outside [0, 1] — possible double-exponentiation"
        # Non-increasing: each column should be <= the previous
        diffs = np.diff(surv, axis=1)
        assert np.all(diffs <= 1e-8), "Survival curve not monotonically non-increasing"

    def test_predict_matches_output_margin(self):
        """predict() returns same values as model_.predict(output_margin=True)."""
        pytest.importorskip("xgboost")
        from mlsurv.models.xgbcox import XGBCoxModel

        X = make_feature_dataframe(50, p=3)
        y = make_random_survival_array(50)
        model = XGBCoxModel(n_estimators=20, random_state=42)
        model.fit(X, y)

        risk = model.predict(X)
        raw_margin = model.model_.predict(X, output_margin=True)
        np.testing.assert_array_equal(risk, raw_margin)


# ---------------------------------------------------------------------------
# _compute_concordance defense-in-depth tests
# ---------------------------------------------------------------------------

class TestConcordanceInfHandling:
    """_compute_concordance should return NaN (not crash) for bad scores."""

    def _run_concordance(self, risk_scores, y_train=None):
        """Helper: run _compute_concordance with a mock model returning given scores."""
        from mlsurv.evaluation._concordance import _compute_concordance

        n = len(risk_scores)
        X_test = make_feature_dataframe(n)
        y_test = make_random_survival_array(n)

        mock_model = MagicMock()
        mock_model.predict.return_value = risk_scores
        mock_model.predict_survival_function.side_effect = NotImplementedError

        df, _, _, _, _ = _compute_concordance(
            mock_model, X_test, y_test, y_train=y_train,
            risk_scores=risk_scores,
        )
        return df

    def test_inf_scores_return_nan(self):
        """Inf risk scores produce NaN c_index_test instead of crashing."""
        scores = np.array([np.inf, 1.0, 2.0, -np.inf, 3.0] * 10, dtype=float)
        df = self._run_concordance(scores)
        assert np.isnan(df['c_index_test'].iloc[0])

    def test_nan_scores_return_nan(self):
        """NaN risk scores produce NaN c_index_test instead of crashing."""
        scores = np.array([np.nan, 1.0, 2.0, np.nan, 3.0] * 10, dtype=float)
        df = self._run_concordance(scores)
        assert np.isnan(df['c_index_test'].iloc[0])

    def test_all_inf_scores_return_nan(self):
        """All-inf risk scores produce NaN c_index_test."""
        scores = np.full(50, np.inf)
        df = self._run_concordance(scores)
        assert np.isnan(df['c_index_test'].iloc[0])

    def test_all_nan_scores_return_nan(self):
        """All-NaN risk scores produce NaN c_index_test."""
        scores = np.full(50, np.nan)
        df = self._run_concordance(scores)
        assert np.isnan(df['c_index_test'].iloc[0])

    def test_ipcw_with_inf_scores_returns_nan(self):
        """IPCW concordance also returns NaN (not crash) for inf scores."""
        scores = np.array([np.inf, 1.0, 2.0, -np.inf, 3.0] * 10, dtype=float)
        y_train = make_random_survival_array(50, seed=99)

        df = self._run_concordance(scores, y_train=y_train)
        assert np.isnan(df['c_index_test'].iloc[0])
        assert np.isnan(df['c_index_ipcw_test'].iloc[0])

    def test_finite_scores_return_valid(self):
        """Finite risk scores produce a valid C-index (regression guard)."""
        scores = np.linspace(0.1, 5.0, 50)
        df = self._run_concordance(scores)
        c = df['c_index_test'].iloc[0]
        assert np.isfinite(c)
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# evaluate() warning tests
# ---------------------------------------------------------------------------

class TestEvaluateNonFiniteWarning:
    """evaluate() should warn when predict() returns non-finite scores."""

    def _collect_methodological_warnings(self, risk_scores, model_name):
        """Run _compute_eval_dataframes and return MethodologicalWarnings emitted.

        Downstream exceptions from mock limitations are suppressed --
        we only care about the warning side-effect from the predict() check.
        """
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.evaluate import _compute_eval_dataframes
        from mlsurv.evaluation._config import EvalConfig

        n = len(risk_scores)
        X = make_feature_dataframe(n, p=3)
        y = make_random_survival_array(n)
        y_train = make_random_survival_array(n, seed=99)

        mock_model = MagicMock()
        mock_model.predict.return_value = risk_scores
        mock_model.name = model_name
        mock_model.has_internal_parallelism = False
        mock_model.predict_survival_function.side_effect = NotImplementedError

        config = EvalConfig(
            eval_times=np.array([1.0, 3.0, 5.0]),
            n_importance_repeats=0,
            include_shap=False,
            show_progress=False,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                _compute_eval_dataframes(
                    mock_model, X, y, None, y_train,
                    custom_metrics=None, model_name=model_name, config=config,
                )
            except Exception:
                pass

        return [x for x in w if issubclass(x.category, MethodologicalWarning)]

    def test_warning_emitted_for_inf_scores(self):
        """MethodologicalWarning emitted when risk scores contain inf."""
        scores = np.full(50, np.inf)
        meth = self._collect_methodological_warnings(scores, "bad_model")

        assert len(meth) >= 1, "Expected MethodologicalWarning but got none"
        assert "non-finite" in str(meth[0].message)
        assert "bad_model" in str(meth[0].message)

    def test_no_warning_for_finite_scores(self):
        """No MethodologicalWarning when all scores are finite."""
        scores = np.linspace(-1, 1, 50)
        meth = self._collect_methodological_warnings(scores, "good_model")

        assert len(meth) == 0


# ---------------------------------------------------------------------------
# Search space test
# ---------------------------------------------------------------------------

class TestXGBCoxSearchSpace:
    """max_delta_step should be in the XGBCox search space."""

    def test_max_delta_step_in_default_space(self):
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES
        assert 'max_delta_step' in DEFAULT_SPACES['xgbcox']

    def test_max_delta_step_range(self):
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES
        lo, hi = DEFAULT_SPACES['xgbcox']['max_delta_step']
        assert lo == 0.0
        assert hi == 2.0


class TestXGBCoxTunerTunesAllParams:
    """Regression: the default tuning path (no user search space) must
    tune *every* parameter declared in DEFAULT_SPACES['xgbcox'].

    The bug: XGBCoxTuner.objective() used a hand-maintained fallback dict that
    listed 7 params and omitted max_delta_step, so it stayed pinned at the model
    default 0.7 and never appeared in study.best_params.
    """

    @pytest.mark.integration
    def test_max_delta_step_is_tuned_on_default_path(self):
        pytest.importorskip("xgboost")
        import optuna
        from mlsurv.tuning.optuna_tuner import XGBCoxTuner

        X = make_feature_dataframe(80, p=5)
        y = make_random_survival_array(80)
        tuner = XGBCoxTuner(X, y, model_name='xgbcox', random_state=42)

        # In-memory study (no SQLite side-effect) driving the real objective.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(tuner.objective, n_trials=5)

        # The omitted param must now be explored, and the explored set must match
        # the full declared search space — not a hand-maintained subset.
        assert 'max_delta_step' in study.best_params
        assert set(study.best_params) == set(tuner.search_space)
