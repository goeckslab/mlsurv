"""
FSSVM Validation Tests.

Tests for FSSVM parameter validation, convergence tracking, and tuner
behavior. All 5 optimizers remain functional at the model level. Tuning
uses avltree exclusively (all optimizers produce identical solutions for
the same convex problem).
"""

import sys

import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored


# =============================================================================
# Test 1: Parameter Validation (Unit Tests)
# =============================================================================

class TestFSSVMParameterValidation:
    """Test that invalid parameter combinations raise ValueError."""

    def test_prsvm_invalid_rank_ratio(self, train_test_data):
        """PRSVM with rank_ratio != 1.0 should raise ValueError."""
        from unittest.mock import patch
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(optimizer='PRSVM', rank_ratio=0.5)
        # Mock non-Windows so the platform check doesn't fire first
        with patch('mlsurv.models.fssvm.sys') as mock_sys:
            mock_sys.platform = 'linux'
            with pytest.raises(ValueError, match="rank_ratio=1.0"):
                model.fit(X_train, y_train)

    def test_simple_invalid_rank_ratio(self, train_test_data):
        """simple optimizer with rank_ratio != 1.0 should raise ValueError."""
        from unittest.mock import patch
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(optimizer='simple', rank_ratio=0.8)
        with patch('mlsurv.models.fssvm.sys') as mock_sys:
            mock_sys.platform = 'linux'
            with pytest.raises(ValueError, match="rank_ratio=1.0"):
                model.fit(X_train, y_train)

    def test_prsvm_invalid_fit_intercept(self, train_test_data):
        """PRSVM with fit_intercept=True should raise ValueError."""
        from unittest.mock import patch
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(optimizer='PRSVM', rank_ratio=1.0, fit_intercept=True)
        with patch('mlsurv.models.fssvm.sys') as mock_sys:
            mock_sys.platform = 'linux'
            with pytest.raises(ValueError, match="fit_intercept"):
                model.fit(X_train, y_train)

    def test_simple_invalid_fit_intercept(self, train_test_data):
        """simple optimizer with fit_intercept=True should raise ValueError."""
        from unittest.mock import patch
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(optimizer='simple', rank_ratio=1.0, fit_intercept=True)
        with patch('mlsurv.models.fssvm.sys') as mock_sys:
            mock_sys.platform = 'linux'
            with pytest.raises(ValueError, match="fit_intercept"):
                model.fit(X_train, y_train)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="PRSVM optimizer is not supported on Windows (sksurv C-extension segfault)",
    )
    @pytest.mark.integration
    def test_prsvm_valid_config(self, train_test_data):
        """PRSVM with valid config (rank_ratio=1.0, fit_intercept=False) should work."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(
            optimizer='PRSVM',
            rank_ratio=1.0,
            fit_intercept=False,
            max_iter=100
        )
        model.fit(X_train, y_train)

        assert model.is_fitted_
        pred = model.predict(X_test)
        assert len(pred) == len(X_test)
        assert not np.isnan(pred).any()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="simple optimizer is not supported on Windows (sksurv C-extension segfault)",
    )
    @pytest.mark.integration
    def test_simple_valid_config(self, train_test_data):
        """simple optimizer with valid config should work."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(
            optimizer='simple',
            rank_ratio=1.0,
            fit_intercept=False,
            max_iter=100
        )
        model.fit(X_train, y_train)

        assert model.is_fitted_
        pred = model.predict(X_test)
        assert len(pred) == len(X_test)

    @pytest.mark.integration
    def test_avltree_any_config(self, train_test_data):
        """avltree optimizer should work with any valid rank_ratio and fit_intercept."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        # Test with rank_ratio < 1 and fit_intercept=True
        model = FSSVMModel(
            optimizer='avltree',
            rank_ratio=0.5,
            fit_intercept=True,
            max_iter=100
        )
        model.fit(X_train, y_train)

        assert model.is_fitted_
        pred = model.predict(X_test)
        assert len(pred) == len(X_test)


# =============================================================================
# Test 2: All Optimizers Functional
# =============================================================================

@pytest.mark.integration
class TestFSSVMAllOptimizers:
    """Test that all 5 optimizers work with valid configurations.

    Priority: All optimizers must remain functional.
    """

    @pytest.mark.parametrize("optimizer", ['avltree', 'rbtree', 'direct-count'])
    def test_tree_optimizers_full_config(self, train_test_data, optimizer):
        """Tree-based optimizers should support all parameter combinations."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(
            optimizer=optimizer,
            rank_ratio=0.8,
            fit_intercept=True,
            max_iter=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        assert model.is_fitted_
        assert len(pred) == len(X_test)
        assert not np.isnan(pred).any()

        # Verify C-index is reasonable
        c_index = concordance_index_censored(
            y_test['event'], y_test['time'], pred
        )[0]
        assert 0.4 <= c_index <= 0.9, f"C-index {c_index} outside expected range"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="scikit-survival 0.26.0 segfaults (access violation) in the "
               "PRSVM/simple optimizer C extension on Windows — kills the "
               "entire pytest process (exit code 139)"
    )
    @pytest.mark.parametrize("optimizer", ['PRSVM', 'simple'])
    def test_ranking_optimizers_constrained_config(self, train_test_data, optimizer):
        """PRSVM and simple optimizers should work with constrained parameters."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(
            optimizer=optimizer,
            rank_ratio=1.0,
            fit_intercept=False,
            max_iter=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        assert model.is_fitted_
        assert len(pred) == len(X_test)
        assert not np.isnan(pred).any()

        # Verify C-index is reasonable
        c_index = concordance_index_censored(
            y_test['event'], y_test['time'], pred
        )[0]
        assert 0.4 <= c_index <= 0.9, f"C-index {c_index} outside expected range"


# =============================================================================
# Test 3: Tuner Constraint Enforcement
# =============================================================================

@pytest.mark.integration
class TestFSSVMTunerConstraints:
    """Test that FSSVMTuner uses simplified search space (no optimizer/fit_intercept)."""

    def test_tuner_does_not_suggest_optimizer(self, synthetic_data_clean):
        """Tuner should not include optimizer in suggested params."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42, n_jobs=1)

        result = learner.tune('fssvm', n_trials=3, verbose=False)

        assert result is not None
        best_params = result.optuna_study.best_params
        # optimizer is hardcoded to avltree, not in search space
        assert 'optimizer' not in best_params
        assert 'model__optimizer' not in best_params

    def test_tuner_does_not_suggest_fit_intercept(self, synthetic_data_clean):
        """Tuner should not include fit_intercept in suggested params."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42, n_jobs=1)

        result = learner.tune('fssvm', n_trials=3, verbose=False)

        assert result is not None
        best_params = result.optuna_study.best_params
        assert 'fit_intercept' not in best_params
        assert 'model__fit_intercept' not in best_params

    def test_tuner_suggests_rank_ratio(self, synthetic_data_clean):
        """Tuner should still suggest rank_ratio as a hyperparameter."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42, n_jobs=1)

        result = learner.tune('fssvm', n_trials=5, verbose=False)

        assert result is not None
        best_params = result.optuna_study.best_params
        assert 'rank_ratio' in best_params


# =============================================================================
# Test 4: FSSVM Tuning on Veterans Dataset (Critical Test)
# =============================================================================

@pytest.mark.integration
class TestFSSVMVeteransDataset:
    """Test FSSVM on the Veterans dataset that caused the original crash."""

    @pytest.fixture
    def veterans_prepared(self):
        """Load and prepare Veterans dataset."""
        from sksurv.datasets import load_veterans_lung_cancer
        from mlsurv.utils import to_structured_array

        X, y_raw = load_veterans_lung_cancer()
        X = X.copy()

        # Encode categorical columns
        for col in X.select_dtypes(include=['category', 'object']).columns:
            X[col] = X[col].astype('category').cat.codes

        y = to_structured_array(y_raw['Survival_in_days'], y_raw['Status'])
        return X, y

    def test_fssvm_fit_veterans(self, veterans_prepared):
        """FSSVM should fit on Veterans dataset without crashing."""
        from mlsurv.models import FSSVMModel

        X, y = veterans_prepared
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = FSSVMModel(
            optimizer='avltree',
            rank_ratio=0.8,
            fit_intercept=True,
            max_iter=100
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        assert model.is_fitted_
        assert len(pred) == len(X_test)

    def test_fssvm_tuning_veterans(self, veterans_prepared):
        """FSSVM tuning should complete on Veterans dataset without segfault."""
        from mlsurv import SurvivalLearner

        X, y = veterans_prepared

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

        # Run tuning with 10 trials
        result = learner.tune('fssvm', n_trials=10, verbose=False)

        assert result is not None
        assert 0.5 <= result.cv_metric_mean <= 0.85
        # optimizer is hardcoded (not tuned), so it shouldn't be in best_params
        assert 'optimizer' not in result.optuna_study.best_params

    @pytest.mark.skipif(sys.platform == "win32", reason="sksurv PRSVM/simple optimizer segfaults on Windows")
    def test_all_optimizers_veterans(self, veterans_prepared):
        """All 5 optimizers should work on Veterans dataset."""
        from mlsurv.models import FSSVMModel

        X, y = veterans_prepared
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        results = {}
        optimizers = ['avltree', 'rbtree', 'direct-count', 'PRSVM', 'simple']

        for opt in optimizers:
            try:
                if opt in ('PRSVM', 'simple'):
                    model = FSSVMModel(
                        optimizer=opt,
                        rank_ratio=1.0,
                        fit_intercept=False,
                        max_iter=100
                    )
                else:
                    model = FSSVMModel(
                        optimizer=opt,
                        rank_ratio=0.8,
                        fit_intercept=True,
                        max_iter=100
                    )

                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                c_index = concordance_index_censored(
                    y_test['event'], y_test['time'], pred
                )[0]
                results[opt] = c_index
            except Exception as e:
                pytest.fail(f"Optimizer {opt} failed: {e}")

        # All optimizers should produce reasonable C-indices
        for opt, c_index in results.items():
            assert 0.45 <= c_index <= 0.85, \
                f"Optimizer {opt} C-index {c_index} outside expected range"


# =============================================================================
# Test 5: GBSG2 Dataset Tests
# =============================================================================

@pytest.mark.integration
class TestFSSVMGBSG2:
    """Test FSSVM on GBSG2 dataset for broader validation."""

    @pytest.fixture
    def gbsg2_prepared(self):
        """Load and prepare GBSG2 dataset."""
        from sksurv.datasets import load_gbsg2
        from mlsurv.utils import to_structured_array

        X, y_raw = load_gbsg2()
        X = X.copy()

        # Encode categoricals
        X['horTh'] = (X['horTh'] == 'yes').astype(int)
        X['menostat'] = (X['menostat'] == 'Post').astype(int)
        X['tgrade'] = X['tgrade'].map({'I': 1, 'II': 2, 'III': 3}).astype(int)

        y = to_structured_array(y_raw['time'], y_raw['cens'])
        return X, y

    @pytest.mark.skipif(sys.platform == "win32", reason="sksurv PRSVM/simple optimizer segfaults on Windows")
    @pytest.mark.parametrize("optimizer", ['avltree', 'rbtree', 'direct-count', 'PRSVM', 'simple'])
    def test_optimizer_gbsg2(self, gbsg2_prepared, optimizer):
        """Each optimizer should work on GBSG2 dataset."""
        from mlsurv.models import FSSVMModel

        X, y = gbsg2_prepared
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if optimizer in ('PRSVM', 'simple'):
            model = FSSVMModel(
                optimizer=optimizer,
                rank_ratio=1.0,
                fit_intercept=False,
                max_iter=100,
                random_state=42
            )
        else:
            model = FSSVMModel(
                optimizer=optimizer,
                rank_ratio=0.8,
                fit_intercept=True,
                max_iter=100,
                random_state=42
            )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        assert model.is_fitted_
        assert len(pred) == len(X_test)
        assert not np.isnan(pred).any()

        c_index = concordance_index_censored(
            y_test['event'], y_test['time'], pred
        )[0]
        assert 0.5 <= c_index <= 0.8, \
            f"Optimizer {optimizer} C-index {c_index} outside expected range for GBSG2"


# =============================================================================
# Test 6: Survival Function Prediction
# =============================================================================

@pytest.mark.integration
class TestFSSVMSurvivalFunction:
    """Test FSSVM survival function prediction via Breslow estimation."""

    def test_predict_survival_function(self, train_test_data):
        """FSSVM should provide survival function predictions."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(optimizer='avltree', max_iter=100)
        model.fit(X_train, y_train)

        surv_funcs = model.predict_survival_function(X_test)

        assert len(surv_funcs) == len(X_test)
        # Check that survival functions are valid step functions
        for sf in surv_funcs[:5]:  # Check first 5
            # Evaluate at some time points
            times = y_train['time']
            mid_time = np.median(times)
            prob = sf(mid_time)
            assert 0 <= prob <= 1, f"Survival probability {prob} outside [0, 1]"

    def test_predict_survival_function_with_times(self, train_test_data):
        """FSSVM predict_survival_function(times=...) returns aligned ndarray."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(optimizer='avltree', max_iter=100)
        model.fit(X_train, y_train)

        times = np.array([1.0, 5.0, 10.0])
        result = model.predict_survival_function(X_test, times=times)

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(X_test), len(times))
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_predict_survival_function_times_matches_manual(self, train_test_data):
        """FSSVM times= output matches manual StepFunction evaluation."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(optimizer='avltree', max_iter=100)
        model.fit(X_train, y_train)

        times = np.array([1.0, 5.0, 10.0])

        result_times = model.predict_survival_function(X_test, times=times)

        surv_funcs = model.predict_survival_function(X_test)
        result_manual = np.array([[fn(t) for t in times] for fn in surv_funcs])

        np.testing.assert_allclose(result_times, result_manual)

    def test_predict_cumulative_hazard_function(self, train_test_data):
        """FSSVM should provide cumulative hazard function predictions."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(optimizer='avltree', max_iter=100)
        model.fit(X_train, y_train)

        cum_haz_funcs = model.predict_cumulative_hazard_function(X_test)

        assert len(cum_haz_funcs) == len(X_test)
        # Cumulative hazard should be non-negative
        for chf in cum_haz_funcs[:5]:
            times = y_train['time']
            mid_time = np.median(times)
            haz = chf(mid_time)
            assert haz >= 0, f"Cumulative hazard {haz} should be non-negative"


# =============================================================================
# Test 7: Windows Platform Filtering
# =============================================================================

class TestFSSVMWindowsFiltering:
    """Test that PRSVM/simple are excluded on Windows to prevent segfaults."""

    def test_model_raises_on_windows_prsvm(self, train_test_data):
        """FSSVMModel.fit() should raise ValueError for PRSVM on Windows."""
        from unittest.mock import patch
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data
        model = FSSVMModel(optimizer='PRSVM', rank_ratio=1.0, fit_intercept=False)

        with patch('mlsurv.models.fssvm.sys') as mock_sys:
            mock_sys.platform = 'win32'
            with pytest.raises(ValueError, match="not supported on Windows"):
                model.fit(X_train, y_train)

    def test_model_raises_on_windows_simple(self, train_test_data):
        """FSSVMModel.fit() should raise ValueError for simple on Windows."""
        from unittest.mock import patch
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data
        model = FSSVMModel(optimizer='simple', rank_ratio=1.0, fit_intercept=False)

        with patch('mlsurv.models.fssvm.sys') as mock_sys:
            mock_sys.platform = 'win32'
            with pytest.raises(ValueError, match="not supported on Windows"):
                model.fit(X_train, y_train)

    def test_model_allows_safe_optimizers_on_windows(self, train_test_data):
        """avltree/rbtree/direct-count should work on Windows (no raise)."""
        from unittest.mock import patch
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data
        with patch('mlsurv.models.fssvm.sys') as mock_sys:
            mock_sys.platform = 'win32'
            for opt in ['avltree', 'rbtree', 'direct-count']:
                model = FSSVMModel(optimizer=opt, max_iter=100)
                model.fit(X_train, y_train)
                assert model.is_fitted_

    def test_model_allows_prsvm_on_non_windows(self, train_test_data):
        """On non-Windows platforms, PRSVM should not raise."""
        from unittest.mock import patch
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data
        model = FSSVMModel(
            optimizer='PRSVM', rank_ratio=1.0, fit_intercept=False, max_iter=100
        )

        with patch('mlsurv.models.fssvm.sys') as mock_sys:
            mock_sys.platform = 'linux'
            # Should not raise — the platform check should pass
            model.fit(X_train, y_train)
            assert model.is_fitted_

    def test_model_allows_simple_on_non_windows(self, train_test_data):
        """On non-Windows platforms, simple should not raise."""
        from unittest.mock import patch
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data
        model = FSSVMModel(
            optimizer='simple', rank_ratio=1.0, fit_intercept=False, max_iter=100
        )

        with patch('mlsurv.models.fssvm.sys') as mock_sys:
            mock_sys.platform = 'linux'
            model.fit(X_train, y_train)
            assert model.is_fitted_


# =============================================================================
# Test 8: Convergence Tracking
# =============================================================================

class TestFSSVMConvergence:
    """Test convergence tracking attributes and warnings."""

    def test_convergence_tracking_converged(self, train_test_data):
        """Model with sufficient iterations should set converged_=True."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(optimizer='avltree', max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        assert model.converged_ is True
        assert model.actual_iterations_ is not None
        assert model.actual_iterations_ > 0

    def test_convergence_tracking_not_converged(self, train_test_data):
        """Model with max_iter=1 should set converged_=False and emit warning."""
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from mlsurv.models import FSSVMModel
        from mlsurv._warnings import SuggestionWarning

        X_train, X_test, y_train, y_test = train_test_data

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = FSSVMModel(optimizer='avltree', max_iter=1, random_state=42)
            model.fit(X_train, y_train)

        assert model.converged_ is False
        assert model.actual_iterations_ is not None

        suggestion_warnings = [w for w in caught
                               if issubclass(w.category, SuggestionWarning)]
        assert len(suggestion_warnings) >= 1
        assert "did not converge" in str(suggestion_warnings[0].message)

        # sksurv's ConvergenceWarning should be suppressed
        convergence_warnings = [w for w in caught
                                if issubclass(w.category, ConvergenceWarning)]
        assert len(convergence_warnings) == 0, (
            "sksurv ConvergenceWarning should be suppressed"
        )

    def test_convergence_attributes_before_fit(self):
        """Before fitting, convergence attributes should be None."""
        from mlsurv.models import FSSVMModel

        model = FSSVMModel()
        assert model.converged_ is None
        assert model.actual_iterations_ is None

    def test_default_max_iter_is_100(self):
        """Default max_iter should be 100 (not 20)."""
        from mlsurv.models import FSSVMModel

        model = FSSVMModel()
        assert model.max_iter == 100

    def test_default_tol_is_none(self):
        """Default tol should be None (scipy default, not 1e-3)."""
        from mlsurv.models import FSSVMModel

        model = FSSVMModel()
        assert model.tol is None


# =============================================================================
# Test 9: fit_intercept + rank_ratio Validation
# =============================================================================

class TestFSSVMFitInterceptValidation:
    """Test fit_intercept + rank_ratio=1.0 is rejected with clean error."""

    def test_fit_intercept_rank_ratio_1_raises(self, train_test_data):
        """fit_intercept=True + rank_ratio=1.0 should raise our ValueError."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(fit_intercept=True, rank_ratio=1.0)
        with pytest.raises(ValueError, match="not meaningful"):
            model.fit(X_train, y_train)

    def test_fit_intercept_with_hybrid_rank_ratio_works(self, train_test_data):
        """fit_intercept=True + rank_ratio < 1.0 should be allowed."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(
            fit_intercept=True, rank_ratio=0.5, max_iter=100, random_state=42,
        )
        model.fit(X_train, y_train)
        assert model.is_fitted_


# =============================================================================
# Test 10: Regression-mode caveat fires once per session, not per call
# =============================================================================

def test_regression_mode_warning_emitted_once(train_test_data):
    """rank_ratio < 1.0 warns once at fit and never floods predict loops.

    Guards the fix for the regression-mode MethodologicalWarning that used to
    fire on every predict_survival_function() call. Under simplefilter("always")
    (which defeats the once-filter), the module-level dedup must still collapse
    repeated fits and predict calls to a single emission, while rank_ratio=1.0
    stays silent.
    """
    import warnings
    from mlsurv.models.fssvm import FSSVMModel, _regression_mode_warned
    from mlsurv._warnings import RegressionModeWarning

    X_train, X_test, y_train, y_test = train_test_data
    # Module-level state: isolate from test order by clearing before and after.
    _regression_mode_warned.clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Two separate regression-mode fits + repeated predict calls on each.
            for _ in range(2):
                model = FSSVMModel(rank_ratio=0.5, max_iter=100, random_state=42)
                model.fit(X_train, y_train)
                for _ in range(3):
                    model.predict_survival_function(X_test)
            # rank_ratio=1.0 (pure ranking) must not warn at all.
            FSSVMModel(rank_ratio=1.0, max_iter=100, random_state=42).fit(X_train, y_train)

        regression_warnings = [w for w in caught
                               if issubclass(w.category, RegressionModeWarning)]
        assert len(regression_warnings) == 1
        assert "regression mode" in str(regression_warnings[0].message)
    finally:
        _regression_mode_warned.clear()
