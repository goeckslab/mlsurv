"""Test SurvivalLearner API for Phase 3."""

import copy
import warnings

import pytest
import numpy as np
import pandas as pd

from mlsurv._warnings import (
    MethodologicalWarning,
    MetricFailureWarning,
    SuggestionWarning,
)

from mlsurv import SurvivalLearner
from mlsurv.utils import SurvivalArray


# ---------------------------------------------------------------------------
# Module-scoped trained-learner templates.
#
# Several fixtures below build a trained ``SurvivalLearner`` (``setup()`` +
# ``train()``). Function-scoped, they retrained from scratch for every
# consuming test. These module-scoped templates train ONCE per module; each
# consuming fixture returns a ``copy.deepcopy`` so tests that mutate the learner
# (``validate()`` / ``evaluate()`` / ``bootstrap()``) stay isolated.
#
# Module-scoped fixtures cannot depend on the function-scoped conftest data
# fixtures, so the deterministic (seed=42) generation from ``conftest.py`` is
# reproduced verbatim here to keep test outcomes byte-for-byte identical.
# ---------------------------------------------------------------------------


def _make_small_data():
    """Reproduce conftest ``small_data`` (50 samples, deterministic seed=42)."""
    np.random.seed(42)
    n_samples = 50

    X = pd.DataFrame({
        'age': np.random.uniform(30, 80, n_samples),
        'stage': np.random.randint(1, 5, n_samples),
        'grade': np.random.uniform(1, 4, n_samples),
        'tumor_size': np.random.uniform(0.5, 10, n_samples),
        'biomarker': np.random.randn(n_samples),
    })

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y


def _make_synthetic_data_clean():
    """Reproduce conftest ``synthetic_data_clean`` (200x20, no NaN, seed=42)."""
    np.random.seed(42)
    n_samples, n_features = 200, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate survival times
    true_coef = np.zeros(n_features)
    true_coef[:5] = [0.5, -0.3, 0.4, -0.2, 0.3]
    linear_pred = X.values @ true_coef
    times = np.exp(-linear_pred + np.random.randn(n_samples) * 0.5)
    times = np.clip(times, 0.1, 10)

    # Generate events (~30% censored)
    events = np.random.binomial(1, 0.7, n_samples).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y


@pytest.fixture(scope="module")
def _trained_reorder_template():
    """Template for ``TestValidateColumnReorder.trained`` — coxph on small_data."""
    X, y = _make_small_data()
    learner = SurvivalLearner(X, y)
    learner.setup()
    learner.train('coxph')
    return learner, X, y


@pytest.fixture(scope="module")
def _trained_subpop_template():
    """Template for ``trained_with_subpopulations`` — coxph + validation slice."""
    X, y = _make_synthetic_data_clean()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, verbose=False)
    learner.train('coxph', verbose=False)

    # Use a subset as "external" validation (enough events per subpopulation)
    X_val = X.iloc[:60]
    y_val = y[:60]
    subpopulation_labels = np.array(['A'] * 30 + ['B'] * 30)
    return learner, X_val, y_val, subpopulation_labels


@pytest.fixture(scope="module")
def _setup_subpop_template():
    """Template for ``learner_with_setup_subpopulations`` — setup(subpopulations=)."""
    X, y = _make_synthetic_data_clean()
    training_labels = np.array(['A', 'B'] * (len(X) // 2) + ['A'] * (len(X) % 2))
    learner = SurvivalLearner(X, y)
    learner.setup(
        random_state=42,
        feature_selector='topk',
        subpopulations={'grp': training_labels},
        verbose=False,
    )
    learner.train('coxph', verbose=False)
    # Validation slice of deliberately different size than
    # state.subpopulations_test (size ~0.3 * 200 = 60) so the length-mismatch
    # safety net has something to detect.
    X_val = X.iloc[:80].copy()
    y_val = y[:80]
    return learner, X_val, y_val


class TestLearnerState:
    """Tests for LearnerState dataclass."""

    def test_defaults(self):
        """LearnerState() produces valid defaults for all fields."""
        from mlsurv.learner._state import LearnerState
        s = LearnerState()
        assert s.is_setup is False
        assert s.presplit is False
        assert s.models == {}
        assert s.validation_results == {}
        assert s.X is None

    def test_dicts_are_independent(self):
        """Each LearnerState gets its own mutable dicts (no shared default trap)."""
        from mlsurv.learner._state import LearnerState
        s1 = LearnerState()
        s2 = LearnerState()
        s1.models['coxph'] = 'fake'
        assert 'coxph' not in s2.models

    def test_state_property_roundtrip(self, small_data):
        """Public properties proxy to _state and back."""
        from mlsurv.learner import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=99, verbose=False)
        # Write via property (setup), read via _state
        assert learner._state.random_state == 99
        # Write via _state, read via property
        learner._state.random_state = 123
        assert learner.random_state == 123


class TestTrainingAggregationHelpers:
    """Test helpers used for robust CV metric aggregation."""

    def test_finite_mean_std_all_nan(self):
        """All-NaN inputs should return NaN summary without runtime warnings."""
        from mlsurv.learner._training import _finite_mean_std

        mean, std = _finite_mean_std([np.nan, np.nan, np.nan])
        assert np.isnan(mean)
        assert np.isnan(std)

    def test_finite_mean_std_filters_non_finite_values(self):
        """Finite mean/std should ignore NaN and infinite values."""
        from mlsurv.learner._training import _finite_mean_std

        mean, std = _finite_mean_std([0.7, np.nan, np.inf, 0.9])
        assert np.isclose(mean, 0.8)
        # ddof=1: std([0.7, 0.9]) = 0.1 * sqrt(2)
        assert np.isclose(std, 0.1 * np.sqrt(2))


class TestSurvivalLearnerSetup:
    """Test SurvivalLearner setup method."""

    def test_setup_default(self, small_data):
        """Verify default setup works."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        assert learner._state.is_setup
        assert hasattr(learner, 'X_train')
        assert hasattr(learner, 'X_test')
        assert hasattr(learner, 'y_train')
        assert hasattr(learner, 'y_test')

    def test_setup_test_fraction(self, small_data):
        """Verify test fraction is applied."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.2)

        expected_test = int(len(X) * 0.2)
        # Allow for rounding
        assert abs(len(learner.X_test) - expected_test) <= 1

    @pytest.mark.parametrize("bad_value", [
        True, False, 0, 1.0, -0.1, 1.5, "0.2", None,
    ])
    def test_setup_test_fraction_validation(self, small_data, bad_value):
        """Invalid test_fraction values raise ValueError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        with pytest.raises(ValueError, match="test_fraction must be a float in"):
            learner.setup(test_fraction=bad_value)

    def test_setup_test_fraction_accepts_numpy_float(self, small_data):
        """np.float64 test_fraction is accepted."""
        import numpy as np
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=np.float64(0.2))
        assert abs(len(learner.X_test) - int(len(X) * 0.2)) <= 1

    def test_setup_random_state(self, small_data):
        """Verify random state produces reproducible splits."""
        from mlsurv import SurvivalLearner

        X, y = small_data

        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42)

        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)

        assert learner1.X_train.index.equals(learner2.X_train.index)

    def test_setup_returns_self(self, small_data):
        """Verify setup returns self for chaining."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        result = learner.setup()

        assert result is learner

    def test_setup_survives_print_failure(self, small_data):
        """Verify _is_setup=True survives a print summary failure."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)

        original_print = type(learner)._print_setup_summary

        def exploding_print(self):
            raise BrokenPipeError("simulated pipe failure")

        type(learner)._print_setup_summary = exploding_print
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = learner.setup(verbose=True)

            # Setup state should be fully initialized despite print failure
            assert learner._state.is_setup is True
            assert hasattr(learner, 'X_train')
            assert hasattr(learner, 'y_train')
            assert result is learner

            # Should have emitted a UserWarning about the print failure
            print_warnings = [x for x in w if "could not be printed" in str(x.message)]
            assert len(print_warnings) == 1
        finally:
            type(learner)._print_setup_summary = original_print


class TestSurvivalLearnerCvCI:
    """Test cv_ci parameter in setup()."""

    def test_cv_ci_stored(self, small_data):
        """cv_ci should be stored on the learner."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_ci=0.90, verbose=False)
        assert learner.cv_ci == 0.90

    def test_cv_ci_default(self, small_data):
        """cv_ci should default to 0.95."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner.cv_ci == 0.95

    def test_cv_ci_invalid_raises(self, small_data):
        """cv_ci outside (0, 1) should raise ValueError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        with pytest.raises(ValueError, match="cv_ci must be between"):
            learner.setup(cv_ci=0.0, verbose=False)
        with pytest.raises(ValueError, match="cv_ci must be between"):
            learner.setup(cv_ci=1.0, verbose=False)
        with pytest.raises(ValueError, match="cv_ci must be between"):
            learner.setup(cv_ci=-0.5, verbose=False)

    def test_cv_ci_in_setup_box(self, small_data, capsys):
        """Setup box should display ci=X%."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_ci=0.90, verbose=True)
        captured = capsys.readouterr()
        assert "ci=90%" in captured.out

    def test_cv_ci_in_get_config(self, small_data):
        """config should include cv_ci."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_ci=0.90, verbose=False)
        config = learner.config
        assert config['cv_ci'] == 0.90

    def test_cv_ci_propagated_to_model_result(self, small_data):
        """Training should produce ModelResult with populated CI fields."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_ci=0.90, verbose=False)
        result = learner.train('coxph', verbose=False)
        assert not np.isnan(result.cv_ci_lower)
        assert not np.isnan(result.cv_ci_upper)
        assert result.cv_ci_level == 0.90
        assert result.cv_ci_lower < result.cv_metric_mean < result.cv_ci_upper

    def test_cv_ci_in_summary(self, small_data):
        """summary() should have non-NaN CI values for CV phase."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_ci=0.95, verbose=False)
        result = learner.train('coxph', verbose=False)
        df = result.summary(metrics=['harrell'])
        cv_row = df[df['phase'] == 'cv'].iloc[0]
        assert pd.notna(cv_row['ci_lower'])
        assert pd.notna(cv_row['ci_upper'])
        assert cv_row['ci_lower'] < cv_row['mean'] < cv_row['ci_upper']


class TestTimeSeriesCvCIWarning:
    """Test TimeSeriesSplit CI warning behavior."""

    def test_timeseries_cv_ci_warning(self, small_data):
        """TimeSeriesSplit should trigger a CI caveat warning."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_strategy='timeseries', stratify=None, verbose=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.train('coxph', verbose=False)
        ci_warnings = [x for x in w if "identically distributed" in str(x.message)]
        assert len(ci_warnings) >= 1

    def test_timeseries_cv_ci_warning_once(self, small_data):
        """TimeSeriesSplit CI warning should only fire once across models."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_strategy='timeseries', stratify=None, verbose=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.train('coxph', verbose=False)
            learner.train('rsf', verbose=False)
        ci_warnings = [x for x in w if "identically distributed" in str(x.message)]
        assert len(ci_warnings) == 1


class TestSurvivalLearnerPresplit:
    """Test SurvivalLearner with pre-split data."""

    def test_presplit_initialization(self, train_test_data):
        """Verify pre-split data initialization works."""
        from mlsurv import SurvivalLearner

        X_train, X_test, y_train, y_test = train_test_data
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )

        assert learner._state.presplit is True
        assert learner.X_train is X_train
        assert learner.X_test is X_test
        assert learner.y_train is y_train
        assert learner.y_test is y_test

    def test_presplit_setup_skips_splitting(self, train_test_data):
        """Verify setup() skips splitting with pre-split data."""
        from mlsurv import SurvivalLearner

        X_train, X_test, y_train, y_test = train_test_data
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        learner.setup()

        # Data should remain unchanged
        assert learner.X_train is X_train
        assert learner.X_test is X_test
        assert len(learner.X_train) == len(X_train)
        assert len(learner.X_test) == len(X_test)

    def test_presplit_warns_on_test_fraction(self, train_test_data):
        """Verify warning when test_fraction is set with pre-split data."""
        from mlsurv import SurvivalLearner
        import warnings

        X_train, X_test, y_train, y_test = train_test_data
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.setup(test_fraction=0.2)

            # Check for test_fraction warning
            test_frac_warnings = [
                x for x in w if "test_fraction" in str(x.message)
            ]
            assert len(test_frac_warnings) == 1

    def test_presplit_train_works(self, train_test_data):
        """Verify train() works with pre-split data."""
        from mlsurv import SurvivalLearner

        X_train, X_test, y_train, y_test = train_test_data
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        learner.setup()
        result = learner.train('coxph')

        assert result.name == 'coxph'
        assert hasattr(result, 'model')
        assert 0.0 <= result.cv_metric_mean <= 1.0

    def test_presplit_evaluate_uses_correct_test_set(self, train_test_data):
        """Verify evaluate() uses the provided test set."""
        from mlsurv import SurvivalLearner

        X_train, X_test, y_train, y_test = train_test_data
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        learner.setup()
        learner.train('coxph')
        eval_result = learner.evaluate()

        # Verify evaluation ran on the correct test set size
        assert eval_result is not None
        assert hasattr(eval_result, 'concordance')  # EvaluationResult attribute
        # Verify test set size matches
        assert len(eval_result.test_set) == len(X_test)

    def test_mixed_params_raises_error(self, synthetic_data_clean, train_test_data):
        """Verify mixing (X, y) with pre-split params raises error."""
        from mlsurv import SurvivalLearner
        import pytest

        X, y = synthetic_data_clean
        X_train, X_test, y_train, y_test = train_test_data

        with pytest.raises(ValueError, match="Cannot mix"):
            SurvivalLearner(X, y, X_train=X_train)

    def test_incomplete_presplit_raises_error(self, train_test_data):
        """Verify incomplete pre-split params raises error."""
        from mlsurv import SurvivalLearner
        import pytest

        X_train, X_test, y_train, y_test = train_test_data

        # Missing y_test
        with pytest.raises(ValueError, match="Provide either"):
            SurvivalLearner(X_train=X_train, X_test=X_test, y_train=y_train)

    def test_feature_mismatch_raises_error(self, synthetic_data_clean):
        """Verify feature mismatch between train/test raises error."""
        from mlsurv import SurvivalLearner
        import pytest

        X, y = synthetic_data_clean
        split_idx = int(len(X) * 0.7)

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:].drop(columns=['feature_0'])  # Remove a column
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        with pytest.raises(ValueError, match="Feature mismatch"):
            SurvivalLearner(
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test
            )

    def test_presplit_column_reorder(self, synthetic_data_clean):
        """Verify test columns are reordered to match train."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        split_idx = int(len(X) * 0.7)

        X_train = X.iloc[:split_idx]
        # Reverse column order for X_test
        X_test = X.iloc[split_idx:][X.columns[::-1]]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        learner.setup()

        # After setup, columns should match
        assert list(learner.X_test.columns) == list(learner.X_train.columns)


class TestSurvivalLearnerTrain:
    """Test SurvivalLearner train method."""

    def test_train_coxph(self, small_data):
        """Verify train for CoxPH."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        result = learner.train('coxph')

        assert hasattr(result, 'model')
        assert hasattr(result, 'cv_metric_scores')
        assert hasattr(result, 'cv_metric_mean')
        assert result.name == 'coxph'

    def test_train_rsf(self, small_data):
        """Verify train for RSF."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        result = learner.train('rsf', n_estimators=10, random_state=42)

        assert result.name == 'rsf'
        assert 0.0 <= result.cv_metric_mean <= 1.0

    def test_train_auto_setup(self, small_data):
        """Verify train auto-calls setup."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        # Don't call setup explicitly

        result = learner.train('coxph')

        assert learner._state.is_setup
        assert result is not None


class TestSurvivalLearnerRun:
    """Test SurvivalLearner run method."""

    def test_run_default(self, small_data):
        """Verify run with defaults returns dict of ModelResult."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        # Test with small subset of models
        result = learner.run(
            models=['coxph'],
            verbose=False
        )

        # run() now returns dict of {str: ModelResult}
        assert isinstance(result, dict)
        assert 'coxph' in result
        # Learner properties provide summary/best_model
        assert learner.best_model is not None
        summary = learner.summary()
        assert isinstance(summary, pd.DataFrame)

    def test_run_exclude(self, small_data):
        """Verify exclude parameter works."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        result = learner.run(
            models=['coxph', 'coxnet'],
            exclude=['coxnet'],
            verbose=False
        )

        # run() returns dict; excluded model should not be present
        assert isinstance(result, dict)
        assert 'coxnet' not in result

    def test_run_metrics_df(self, small_data):
        """Verify metrics DataFrame structure via learner.summary()."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        learner.run(
            models=['coxph'],
            verbose=False
        )

        # Metrics are accessed via learner.summary()
        metrics = learner.summary()
        assert isinstance(metrics, pd.DataFrame)
        assert 'model' in metrics.columns
        assert 'mean' in metrics.columns
        assert 'cv' in metrics['phase'].values

    def test_run_basic(self, small_data):
        """Verify run returns dict of ModelResult."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)

        result = learner.run(models=['coxph'], verbose=False, evaluate=False)

        # run() now returns dict of {str: ModelResult}
        assert isinstance(result, dict)
        assert 'coxph' in result
        assert learner.best_model is not None


class TestSurvivalLearnerEvaluate:
    """Test SurvivalLearner evaluate method."""

    def test_evaluate_basic(self, small_data):
        """Verify evaluate returns EvaluationResult."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        model_result = learner.train('coxph')
        eval_result = learner.evaluate(model_result, n_importance_repeats=0)

        assert hasattr(eval_result, 'concordance')
        assert hasattr(eval_result, 'test_set')
        assert hasattr(eval_result, 'importance')

    def test_evaluate_accepts_model_result(self, small_data):
        """Verify evaluate accepts ModelResult directly."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        model_result = learner.train('coxph')

        # Should work with ModelResult
        eval_result = learner.evaluate(model_result, n_importance_repeats=0)
        assert eval_result is not None


class TestSurvivalLearnerBootstrap:
    """Test SurvivalLearner bootstrap method."""

    def test_bootstrap_basic(self, small_data):
        """Verify bootstrap returns BootstrapResult."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        model_result = learner.train('coxph')
        boot_result = learner.bootstrap(model_result, n_bootstrap=2)

        assert hasattr(boot_result, 'summary')
        assert hasattr(boot_result, 'iterations')


class TestSurvivalLearnerUtils:
    """Test SurvivalLearner utility methods."""

    def test_summary(self, small_data):
        """Verify summary returns the CV metrics table as a DataFrame."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        learner.train('coxph')
        learner.run(models=['coxph'], verbose=False)

        df = learner.summary()
        assert df is not None
        assert isinstance(df, pd.DataFrame)

    def test_summary_integrated_brier_is_positive(self, small_data):
        """summary() surfaces raw (positive) IBS, not the negated scoring proxy."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, scoring='integrated_brier')

        learner.train('coxph')

        df = learner.summary()
        ibs_cv = df[(df['metric'] == 'integrated_brier')
                    & (df['phase'] == 'cv')]['mean']
        assert len(ibs_cv) > 0
        assert (ibs_cv > 0).all()

    def test_get_config(self, small_data):
        """Verify config returns configuration."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.25, random_state=123)

        config = learner.config

        assert config['test_fraction'] == 0.25
        assert config['random_state'] == 123
        assert 'n_train' in config
        assert 'n_test' in config

    def test_validate(self, small_data):
        """Verify validate on external data."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        model_result = learner.train('coxph')

        # Use a subset as "external" validation
        X_val = X.iloc[:10]
        y_val = y[:10]

        result = learner.validate(X_val, y_val, models=model_result)

        assert result.evaluation.c_index_test is not None
        assert len(result.test_set) == 10

    def test_validate_skips_importance_by_default(self, small_data):
        """validate() should skip importance (n_importance_repeats=0) without error."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()
        learner.train('coxph')

        # validate() defaults to n_importance_repeats=0; should not raise
        val_result = learner.validate(X, y, name='self-check')
        assert val_result is not None
        # importance should be an empty DataFrame (0 rows) since repeats=0
        assert val_result.evaluation.importance.empty


class TestValidateColumnReorder:
    """Tests for validate() auto-reordering columns."""

    @pytest.fixture()
    def trained(self, _trained_reorder_template):
        return copy.deepcopy(_trained_reorder_template)

    def test_reordered_columns_warns(self, trained, recwarn):
        """validate() warns when columns are in different order."""
        learner, X, y = trained

        # Reverse column order
        X_val = X.iloc[:10][X.columns[::-1]]
        y_val = y[:10]

        learner.validate(X_val, y_val, verbose=False, bootstrap=False)

        reorder_warnings = [
            w for w in recwarn.list
            if issubclass(w.category, UserWarning) and "reordered" in str(w.message).lower()
        ]
        assert len(reorder_warnings) == 1

    def test_matching_columns_no_warning(self, trained, recwarn):
        """validate() does not warn when columns match."""
        learner, X, y = trained

        X_val = X.iloc[:10]
        y_val = y[:10]

        learner.validate(X_val, y_val, verbose=False, bootstrap=False)

        reorder_warnings = [
            w for w in recwarn.list
            if issubclass(w.category, UserWarning) and "reordered" in str(w.message).lower()
        ]
        assert len(reorder_warnings) == 0

    def test_missing_columns_raises_value_error(self, trained):
        """validate() raises ValueError when validation data is missing training columns."""
        learner, X, y = trained

        X_val = X.iloc[:10].drop(columns=[X.columns[0]])
        y_val = y[:10]

        with pytest.raises(ValueError, match=r"missing.*1.*training column"):
            learner.validate(X_val, y_val, verbose=False, bootstrap=False)

    def test_extra_columns_warns_and_drops(self, trained, recwarn):
        """validate() warns about extra columns and drops them."""
        learner, X, y = trained

        X_val = X.iloc[:10].copy()
        X_val['extra_col'] = 99
        y_val = y[:10]

        result = learner.validate(X_val, y_val, verbose=False, bootstrap=False)

        extra_warnings = [
            w for w in recwarn.list
            if issubclass(w.category, UserWarning) and "extra" in str(w.message)
        ]
        assert len(extra_warnings) == 1
        assert "1 extra column" in str(extra_warnings[0].message)
        # Verify validation actually succeeded with the dropped column
        assert result is not None

    def test_missing_and_extra_columns_raises(self, trained):
        """validate() raises ValueError when both missing and extra columns present."""
        learner, X, y = trained

        X_val = X.iloc[:10].drop(columns=[X.columns[0]]).copy()
        X_val['extra_col'] = 99
        y_val = y[:10]

        with pytest.raises(ValueError, match=r"missing.*1.*training column"):
            learner.validate(X_val, y_val, verbose=False, bootstrap=False)

    def test_validate_override_warning(self, trained, recwarn):
        """validate() warns when overwriting existing validation result."""
        learner, X, y = trained

        X_val = X.iloc[:10]
        y_val = y[:10]

        # First validate - no override warning expected
        learner.validate(X_val, y_val, verbose=False, bootstrap=False)
        override_warnings = [
            w for w in recwarn.list
            if issubclass(w.category, UserWarning)
            and "overwriting" in str(w.message).lower()
        ]
        assert len(override_warnings) == 0

        # Second validate with same name - warning expected
        recwarn.clear()
        learner.validate(X_val, y_val, verbose=False, bootstrap=False)
        override_warnings = [
            w for w in recwarn.list
            if issubclass(w.category, UserWarning)
            and "overwriting" in str(w.message).lower()
        ]
        assert len(override_warnings) == 1


@pytest.mark.integration
class TestValidateSubpopulationEvalTimes:
    """Tests for validate() subpopulation metrics eval_times inheritance."""

    @pytest.fixture()
    def trained_with_subpopulations(self, _trained_subpop_template):
        """Train a learner and prepare subpopulation labels for validation data."""
        return copy.deepcopy(_trained_subpop_template)

    def test_validate_subpopulation_uses_explicit_eval_times(
        self, trained_with_subpopulations,
    ):
        """validate(eval_times=[...], subpopulations=...) uses explicit times."""
        learner, X_val, y_val, sg_labels = trained_with_subpopulations

        user_times = np.array([0.5, 1.0, 1.5, 2.0])
        result = learner.validate(
            X_val, y_val, name="ExplicitET",
            subpopulations={'grp': sg_labels},
            eval_times=user_times,
            bootstrap=False, verbose=False,
        )

        assert result.subpopulation_results is not None
        sg = result.subpopulation_results['grp']
        assert sg.auc_t is not None and not sg.auc_t.empty, (
            "AUC(t) should be computed when explicit eval_times are provided"
        )
        actual_times = sg.auc_t['eval_time'].unique()
        assert np.isin(actual_times, user_times).all(), (
            f"Subpopulation eval_times {actual_times} not all in {user_times}"
        )

    def test_validate_subpopulation_auto_computes_when_no_eval_times(
        self, trained_with_subpopulations,
    ):
        """Without explicit eval_times, subpopulations still compute via auto."""
        learner, X_val, y_val, sg_labels = trained_with_subpopulations

        result = learner.validate(
            X_val, y_val, name="AutoET",
            subpopulations={'grp': sg_labels},
            bootstrap=False, verbose=False,
        )

        assert result.subpopulation_results is not None
        assert 'grp' in result.subpopulation_results

    def test_validate_subpopulation_logs_warning_on_auto_failure(
        self, trained_with_subpopulations, caplog,
    ):
        """If auto_eval_times fails, a warning is logged (not silently swallowed)."""
        import logging
        from unittest.mock import patch

        learner, X_val, y_val, sg_labels = trained_with_subpopulations

        with patch(
            'mlsurv.utils.auto_eval_times',
            side_effect=ValueError("mocked: too few event times"),
        ), caplog.at_level(logging.WARNING):
            result = learner.validate(
                X_val, y_val, name="AutoFail",
                subpopulations={'grp': sg_labels},
                bootstrap=False, verbose=False,
            )

        assert result.subpopulation_results is not None
        assert "auto_eval_times failed" in caplog.text


@pytest.mark.integration
class TestValidateSubpopulationsSetupBootstrap:
    """setup(subpopulations=...) + validate() crashed with IndexError because
    training-scope labels leaked into the validation-cohort bootstrap config.
    """

    @pytest.fixture()
    def learner_with_setup_subpopulations(self, _setup_subpop_template):
        return copy.deepcopy(_setup_subpop_template)

    def test_validate_without_cohort_labels_does_not_crash(
        self, learner_with_setup_subpopulations,
    ):
        """Exact repro: validate() with bootstrap=True must not raise
        IndexError just because setup() had subpopulations=. Without cohort-scope
        labels the path is an explicit opt-out in validate() (not the safety
        net in _resolve_subpopulation_labels), and a SuggestionWarning must
        explain the skip per the 'No Silent Data Loss' rule."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = learner.validate(
                X_val, y_val, name='Holdout',
                bootstrap=True, n_bootstrap=2, verbose=False,
            )
        # No crash, bootstrap ran, subpop CIs correctly absent.
        assert result.evaluation is not None
        assert result.evaluation.bootstrap is not None
        assert result.evaluation.bootstrap.subpopulation_auc_t is None

        skip_warnings = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'Training-scope labels cannot be applied' in str(w.message)
        ]
        assert len(skip_warnings) == 1, (
            "Expected a single SuggestionWarning explaining the skip when "
            f"setup() had subpopulations= but validate() didn't; got {skip_warnings}"
        )
        # Distinguish the explicit-opt-out path from the safety-net path:
        # the length-mismatch warning must NOT fire here, because validate()
        # forwards override=None before the safety net sees state labels.
        mismatch = [
            w for w in caught
            if 'does not match evaluation data length' in str(w.message)
        ]
        assert not mismatch, (
            f"Length-mismatch safety net should not fire on the validate() "
            f"opt-out path; got {mismatch}"
        )

    def test_validate_subpopulations_multi_variable_warns(
        self, learner_with_setup_subpopulations,
    ):
        """validate(subpopulations={...multiple...}) warns that only the first
        variable gets bootstrap CIs — this is the only user-facing signal that
        the other variables are silently skipped."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        labels_a = np.array(['A'] * 40 + ['B'] * 40)
        labels_b = np.array(['X'] * 40 + ['Y'] * 40)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.validate(
                X_val, y_val, name='Multi',
                subpopulations={'grp_a': labels_a, 'grp_b': labels_b},
                bootstrap=True, n_bootstrap=2, verbose=False,
            )
        multi = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'grp_a' in str(w.message) and 'grp_b' in str(w.message)
        ]
        assert multi, "Expected multi-variable SuggestionWarning naming both keys"

    def test_validate_subpopulations_empty_dict_raises(
        self, learner_with_setup_subpopulations,
    ):
        """Empty dict is almost certainly user error → ValueError."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        with pytest.raises(ValueError, match='empty dict'):
            learner.validate(
                X_val, y_val, name='Empty',
                subpopulations={}, verbose=False,
            )

    def test_validate_subpopulations_wrong_length_raises(
        self, learner_with_setup_subpopulations,
    ):
        """Cohort-label length must match y_val; validate() raises upfront."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        bad = np.array(['A'] * 10)  # len(y_val) == 80
        with pytest.raises(ValueError, match='length 10 but y_val has length 80'):
            learner.validate(
                X_val, y_val, name='BadLen',
                subpopulations={'grp': bad},
                verbose=False,
            )

    def test_validate_subpopulations_ndim_raises(
        self, learner_with_setup_subpopulations,
    ):
        """2-D cohort labels caught before reaching the bootstrap path."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        two_d = np.array([['A'] * 40 + ['B'] * 40,
                          ['A'] * 40 + ['B'] * 40])
        with pytest.raises(ValueError, match='must be 1-D'):
            learner.validate(
                X_val, y_val, name='BadShape',
                subpopulations={'grp': two_d},
                verbose=False,
            )

    def test_bootstrap_with_mismatched_scope_subpop_labels_warns_and_skips(
        self, learner_with_setup_subpopulations,
    ):
        """Safety net also covers the sibling bootstrap() method — same
        root cause (training-scope labels applied to a different
        cohort). The resolver's length check catches it regardless of
        which entry point resolved the labels."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.bootstrap(
                X=X_val, y=y_val,
                n_bootstrap=2, verbose=False,
            )
        mismatch = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'does not match evaluation data length' in str(w.message)
        ]
        assert mismatch, (
            "bootstrap() must also surface the length-mismatch warning"
        )

    def test_validate_with_cohort_labels_propagates_to_bootstrap(
        self, learner_with_setup_subpopulations,
    ):
        """validate(subpopulations={name: val_labels}) routes cohort-scope labels
        into the bootstrap config so subpopulation_auc_t is populated."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        val_labels = np.array(['A'] * 40 + ['B'] * 40)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = learner.validate(
                X_val, y_val, name='Holdout',
                subpopulations={'grp': val_labels},
                bootstrap=True, n_bootstrap=2, verbose=False,
            )
        assert result.evaluation.bootstrap is not None
        sg_auc_t = result.evaluation.bootstrap.subpopulation_auc_t
        assert sg_auc_t is not None and not sg_auc_t.empty, (
            "Bootstrap subpopulation AUC(t) should be populated when "
            "cohort-scope labels are provided via validate(subpopulations=...)"
        )
        assert set(sg_auc_t['subpopulation'].unique()) <= {'A', 'B'}
        # Safety net must NOT fire on the happy path — correctly-sized cohort
        # labels should pass the resolver's length check silently.
        assert not any(
            'does not match evaluation data length' in str(w.message)
            for w in caught
        ), "Length-mismatch safety net should not fire when labels are aligned"

    def test_evaluate_with_mismatched_scope_subpop_labels_warns_and_skips(
        self, learner_with_setup_subpopulations,
    ):
        """Safety net: direct evaluate(X=smaller, y=smaller, bootstrap=True) on a
        learner whose state holds training-scope subpop labels must not crash —
        the length-mismatch check warns and skips bootstrap subpop CIs."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = learner.evaluate(
                X=X_val, y=y_val,
                bootstrap=True, n_bootstrap=2,
                n_importance_repeats=0,
                verbose=False,
                _store_result=False,
            )
        mismatch = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'does not match evaluation data length' in str(w.message)
        ]
        assert mismatch, "Expected length-mismatch SuggestionWarning"
        assert result.bootstrap is not None
        assert result.bootstrap.subpopulation_auc_t is None or \
            result.bootstrap.subpopulation_auc_t.empty

    def test_resolver_safety_net_stacklevel_on_evaluate_path(
        self, learner_with_setup_subpopulations,
    ):
        """The length-mismatch safety net must attribute its warning to
        the user's frame, not to the mixin internals. Default _warn_stacklevel=3
        is correct for the direct evaluate() path (warn → resolver → evaluate
        → user)."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.evaluate(
                X=X_val, y=y_val,
                bootstrap=True, n_bootstrap=2,
                n_importance_repeats=0,
                verbose=False,
                _store_result=False,
            )
        mismatch = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'does not match evaluation data length' in str(w.message)
        ]
        assert mismatch, "Expected length-mismatch SuggestionWarning"
        assert mismatch[0].filename == __file__, (
            f"Expected stacklevel=3 to attribute the warning to this test file "
            f"({__file__}); got {mismatch[0].filename}. A misattribution means "
            f"warnings.filterwarnings(module='user_code') filters won't match."
        )

    def test_resolver_safety_net_stacklevel_on_validate_path(
        self, learner_with_setup_subpopulations, monkeypatch,
    ):
        """When the safety net fires on the validate() chain (4 hops:
        warn → resolver → evaluate → validate → user), the warning must still
        attribute to the user's frame. Pre-validation normally blocks this —
        monkeypatch it out so the net actually fires, simulating a future
        bypass."""
        # Patch where validate() does the lookup. The historic
        # mlsurv.learner._evaluation module re-exports the helper for
        # backward-compat, but post-split, validate() lives in
        # _evaluation_core and resolves the symbol in that module's
        # namespace.
        import mlsurv.learner._evaluation_core as _eval_mod

        learner, X_val, y_val = learner_with_setup_subpopulations
        wrong_length = np.array(['A'] * (len(y_val) - 1))
        monkeypatch.setattr(
            _eval_mod, '_resolve_validate_subpopulation_override',
            lambda subpopulations, y_val_, state: wrong_length,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.validate(
                X_val, y_val, name='StacklevelProbe',
                subpopulations={'grp': np.array(['A'] * len(y_val))},
                bootstrap=True, n_bootstrap=2, verbose=False,
            )
        mismatch = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'does not match evaluation data length' in str(w.message)
        ]
        assert mismatch, (
            "Expected length-mismatch SuggestionWarning after bypassing "
            "pre-validation; safety net did not fire."
        )
        assert mismatch[0].filename == __file__, (
            f"Expected stacklevel=4 (threaded via validate → evaluate → "
            f"resolver) to attribute the warning to this test file "
            f"({__file__}); got {mismatch[0].filename}."
        )

    def test_resolver_safety_net_stacklevel_on_bootstrap_path(
        self, learner_with_setup_subpopulations,
    ):
        """`learner.bootstrap(X=..., y=...)` relies on the resolver's
        default `_warn_stacklevel=3` (chain: warn → _resolve_… → bootstrap →
        user). A future refactor that adds a helper frame between
        `bootstrap()` and `_resolve_subpopulation_labels` would silently
        misattribute; this test pins the contract."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.bootstrap(
                X=X_val, y=y_val,
                n_bootstrap=2, verbose=False,
            )
        mismatch = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'does not match evaluation data length' in str(w.message)
        ]
        assert mismatch, "Expected length-mismatch SuggestionWarning"
        assert mismatch[0].filename == __file__, (
            f"Expected stacklevel=3 (bootstrap → resolver default) to "
            f"attribute the warning to this test file ({__file__}); "
            f"got {mismatch[0].filename}."
        )

    def test_resolver_multivariable_warning_stacklevel_on_evaluate_path(
        self, synthetic_data_clean,
    ):
        """The resolver's multi-variable warning (line 64, fires when
        `state.subpopulations_test` has >1 key and bootstrap is enabled) is
        now threaded through `_warn_stacklevel`. The direct evaluate() path
        must attribute to the user frame with the default stacklevel=3.
        Fixture is local to this test because it needs aligned lengths
        (so only the multi-variable warning fires, not the safety net)."""
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        training_labels_a = np.array(
            ['A', 'B'] * (len(X) // 2) + ['A'] * (len(X) % 2)
        )
        training_labels_b = np.array(
            ['X', 'Y'] * (len(X) // 2) + ['X'] * (len(X) % 2)
        )
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42,
            feature_selector='topk',
            subpopulations={'grp_a': training_labels_a,
                            'grp_b': training_labels_b},
            verbose=False,
        )
        learner.train('coxph', verbose=False)
        # Evaluate on the in-state test split so state labels align with y_test
        # (no length mismatch) — only the multi-variable warning fires.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.evaluate(
                bootstrap=True, n_bootstrap=2,
                n_importance_repeats=0,
                verbose=False,
                _store_result=False,
            )
        multi = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'computed only for the first subpopulation variable' in str(w.message)
        ]
        assert multi, (
            "Expected multi-variable SuggestionWarning from "
            "_resolve_subpopulation_labels on direct evaluate() path"
        )
        assert multi[0].filename == __file__, (
            f"Expected stacklevel=3 to attribute the multi-variable warning "
            f"to this test file ({__file__}); got {multi[0].filename}. A "
            f"drift here means the two warn sites in the resolver are not "
            f"consistently threaded."
        )

    def test_inherited_eval_times_warning_stacklevel(
        self, learner_with_setup_subpopulations,
    ):
        """The inherited-eval_times warning in `EvaluationMixin.evaluate`
        uses `stacklevel=_warn_stacklevel - 1`. Fires on both the direct
        evaluate() path (stacklevel=2) and the validate() path (stacklevel=3)
        when a prior evaluate()/run() cached user eval_times and the caller
        passes explicit X/y. Pin both chains so a future refactor of
        `_warn_stacklevel` defaults can't silently mis-point this warning."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        # Seed user_eval_times so the inherited-times branch fires.
        user_times = np.array([0.5, 1.0, 1.5, 2.0])
        learner._state.user_eval_times = user_times

        # Direct evaluate() path — expect stacklevel=2 → this test file.
        with warnings.catch_warnings(record=True) as caught_direct:
            warnings.simplefilter('always')
            learner.evaluate(
                X=X_val, y=y_val,
                bootstrap=False, n_importance_repeats=0,
                verbose=False, _store_result=False,
            )
        inherited_direct = [
            w for w in caught_direct
            if issubclass(w.category, SuggestionWarning)
            and 'inherited eval_times' in str(w.message)
        ]
        assert inherited_direct, (
            "Expected inherited-eval_times SuggestionWarning on direct path"
        )
        assert inherited_direct[0].filename == __file__, (
            f"Direct path: expected stacklevel=2 to land in {__file__}; "
            f"got {inherited_direct[0].filename}."
        )

        # Reseed (validate() is a fresh call; cache is per-state field).
        learner._state.user_eval_times = user_times

        # validate() path — expect stacklevel=3 → this test file.
        with warnings.catch_warnings(record=True) as caught_val:
            warnings.simplefilter('always')
            learner.validate(
                X_val, y_val, name='EvalTimesStacklevel',
                bootstrap=False, verbose=False,
            )
        inherited_val = [
            w for w in caught_val
            if issubclass(w.category, SuggestionWarning)
            and 'inherited eval_times' in str(w.message)
        ]
        assert inherited_val, (
            "Expected inherited-eval_times SuggestionWarning on validate path"
        )
        assert inherited_val[0].filename == __file__, (
            f"Validate path: expected stacklevel=3 to land in {__file__}; "
            f"got {inherited_val[0].filename}."
        )

    def test_inherited_eval_times_no_warning_when_grid_fits(
        self, learner_with_setup_subpopulations,
    ):
        """``validate()`` must NOT warn about inherited eval_times when
        the inherited grid comfortably covers the validation cohort. Common
        path in vignette 07's multi-cohort external validation."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        # Pick a grid spanning the validation event-time range so that
        # <10% of events exceed grid.max() — the suppression heuristic
        # (`_inherited_grid_risky`) should return False.
        event_times = np.asarray(y_val['time'])[np.asarray(y_val['event']).astype(bool)]
        user_times = np.quantile(event_times, [0.1, 0.25, 0.5, 0.75, 0.95])
        learner._state.user_eval_times = user_times

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.validate(
                X_val, y_val, name='GridFits',
                bootstrap=False, verbose=False,
            )
        inherited = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'inherited eval_times' in str(w.message)
        ]
        assert not inherited, (
            f"Expected zero 'inherited eval_times' warnings when the inherited "
            f"grid {user_times.tolist()} covers y_val "
            f"(time range [{y_val['time'].min():.2f}, {y_val['time'].max():.2f}]); "
            f"got {len(inherited)}."
        )

    def test_small_subpopulation_warning_stacklevel(
        self, learner_with_setup_subpopulations,
    ):
        """SmallSubpopulationWarning must land on the user frame for
        both the evaluate-storage path (stacklevel=4) and the validate() path
        (stacklevel=3). Previously hardcoded stacklevel=2 pointed inside
        _subpopulation.py, so user filterwarnings(module=...) never matched."""
        from mlsurv._warnings import SmallSubpopulationWarning

        learner, X_val, y_val = learner_with_setup_subpopulations

        # Evaluate-storage path: subpopulations_test is already stored from
        # setup(), and the test subset has ~30 samples per group → well under
        # MIN_EVENTS_WARNING=30 events, so SmallSubpopulationWarning fires.
        #
        # setup() already emits this warning once and sets
        # state.subpop_test_warned, which suppresses the duplicate aggregate
        # warning on the evaluate-storage path. Clear the flag so this path
        # re-surfaces the warning (as it does on a re-evaluate) and we can pin
        # its stacklevel=4 contract.
        learner._state.subpop_test_warned = False
        with warnings.catch_warnings(record=True) as caught_eval:
            warnings.simplefilter('always')
            learner.evaluate(
                bootstrap=False, n_importance_repeats=0,
                verbose=False,
            )
        small_eval = [
            w for w in caught_eval
            if issubclass(w.category, SmallSubpopulationWarning)
        ]
        assert small_eval, (
            "Expected SmallSubpopulationWarning on evaluate-storage path"
        )
        assert small_eval[0].filename == __file__, (
            f"Evaluate path: expected stacklevel=4 to land in {__file__}; "
            f"got {small_eval[0].filename}."
        )

        # Validate path: pass a small cohort subpopulation directly.
        tiny_labels = np.array(['A'] * 40 + ['B'] * 40)
        with warnings.catch_warnings(record=True) as caught_val:
            warnings.simplefilter('always')
            learner.validate(
                X_val, y_val, name='SubpopStacklevel',
                subpopulations={'grp': tiny_labels},
                bootstrap=False, verbose=False,
            )
        small_val = [
            w for w in caught_val
            if issubclass(w.category, SmallSubpopulationWarning)
        ]
        assert small_val, (
            "Expected SmallSubpopulationWarning on validate path"
        )
        assert small_val[0].filename == __file__, (
            f"Validate path: expected stacklevel=3 to land in {__file__}; "
            f"got {small_val[0].filename}."
        )

    @pytest.fixture()
    def inject_bootstrap_failures(self, monkeypatch):
        """Force MetricFailureWarning by injecting a synthetic failure.

        Wraps ``_aggregate_iteration_results`` to append a synthetic failure
        per iteration so ``_emit_failure_summary`` fires (every
        failure raises a warning regardless of rate). The monkeypatch
        preserves the full call stack so the stacklevel assertion is
        meaningful."""
        import importlib
        bootstrap_mod = importlib.import_module('mlsurv.evaluation.bootstrap')
        real_aggregate = bootstrap_mod._aggregate_iteration_results

        def fake_aggregate(results, config):
            agg = real_aggregate(results, config)
            (iteration_data, predictions_data, auc_data, brier_t_data,
             importance_data, subpopulation_auc_data, all_failures,
             all_notices) = agg
            all_failures.append(('harrell', 'RuntimeError', 'injected failure'))
            return (iteration_data, predictions_data, auc_data, brier_t_data,
                    importance_data, subpopulation_auc_data, all_failures,
                    all_notices)

        monkeypatch.setattr(
            bootstrap_mod, '_aggregate_iteration_results', fake_aggregate,
        )

    def test_metric_failure_warning_stacklevel_on_bootstrap_path(
        self, learner_with_setup_subpopulations, inject_bootstrap_failures,
    ):
        """MetricFailureWarning from ``_emit_failure_summary`` must
        attribute to the user's frame on the direct ``learner.bootstrap()``
        chain (warn → _emit → _bootstrap_single → free bootstrap →
        mixin.bootstrap → user, stacklevel=5)."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.bootstrap(
                X=X_val, y=y_val, n_bootstrap=2, verbose=False,
            )
        metric_warns = [
            w for w in caught
            if issubclass(w.category, MetricFailureWarning)
        ]
        assert metric_warns, "Expected MetricFailureWarning"
        assert metric_warns[0].filename == __file__, (
            f"Expected stacklevel=5 (bootstrap path) to attribute the "
            f"warning to this test file ({__file__}); got "
            f"{metric_warns[0].filename}."
        )

    def test_metric_failure_warning_stacklevel_on_evaluate_path(
        self, learner_with_setup_subpopulations, inject_bootstrap_failures,
    ):
        """MetricFailureWarning must attribute to the user's frame on
        the ``learner.evaluate(bootstrap=True)`` chain (warn → _emit →
        _bootstrap_single → free bootstrap → _evaluate_single → free
        evaluate → _dispatch_evaluate → mixin.evaluate → user,
        stacklevel=8)."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.evaluate(
                X=X_val, y=y_val,
                bootstrap=True, n_bootstrap=2,
                n_importance_repeats=0,
                verbose=False, _store_result=False,
            )
        metric_warns = [
            w for w in caught
            if issubclass(w.category, MetricFailureWarning)
        ]
        assert metric_warns, "Expected MetricFailureWarning"
        assert metric_warns[0].filename == __file__, (
            f"Expected stacklevel=8 (evaluate path) to attribute the "
            f"warning to this test file ({__file__}); got "
            f"{metric_warns[0].filename}."
        )

    def test_metric_failure_warning_stacklevel_on_validate_path(
        self, learner_with_setup_subpopulations, inject_bootstrap_failures,
    ):
        """MetricFailureWarning must attribute to the user's frame on
        the ``learner.validate(bootstrap=True)`` chain — one extra hop
        beyond the evaluate path (stacklevel=9)."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.validate(
                X_val, y_val, name='MetricFailureProbe',
                bootstrap=True, n_bootstrap=2, verbose=False,
            )
        metric_warns = [
            w for w in caught
            if issubclass(w.category, MetricFailureWarning)
        ]
        assert metric_warns, "Expected MetricFailureWarning"
        assert metric_warns[0].filename == __file__, (
            f"Expected stacklevel=9 (validate path) to attribute the "
            f"warning to this test file ({__file__}); got "
            f"{metric_warns[0].filename}."
        )

    def test_metric_failure_warning_stacklevel_explicit_dict(
        self, learner_with_setup_subpopulations, inject_bootstrap_failures,
    ):
        """The explicit dict-of-models path through
        ``_bootstrap_multiple`` relies on free ``bootstrap()`` adding +1 to
        the forwarded stacklevel. Pinning the explicit-dict chain guards
        against a future refactor that drops the +1 offset — the default
        ``model=None`` path happens to dict-dispatch too, but the explicit
        form is what scripted callers use."""
        learner, X_val, y_val = learner_with_setup_subpopulations
        coxph = learner._state.models['coxph'].model
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.bootstrap(
                models={'coxph': coxph},
                X=X_val, y=y_val, n_bootstrap=2, verbose=False,
            )
        metric_warns = [
            w for w in caught
            if issubclass(w.category, MetricFailureWarning)
        ]
        assert metric_warns, "Expected MetricFailureWarning on explicit dict"
        assert metric_warns[0].filename == __file__, (
            f"Expected explicit dict-of-models path to attribute warning to "
            f"test file ({__file__}); got {metric_warns[0].filename}."
        )

    def test_low_coverage_warning_stacklevel_on_bootstrap_path(
        self, learner_with_setup_subpopulations, monkeypatch,
    ):
        """The multi-model low-coverage summary
        (``_emit_multi_low_coverage_summary``) shares the same dict-dispatch
        stacklevel arithmetic as the failure summary, so pin its attribution
        too — the ``_warn_stacklevel - 1`` offset in ``_bootstrap_multiple``
        must land the MethodologicalWarning on the user's frame, not pytest.
        Inject a synthetic low-coverage notice (``(metric, subpop, n_valid,
        n_total)``) so the summary fires regardless of real coverage."""
        import importlib
        bootstrap_mod = importlib.import_module('mlsurv.evaluation.bootstrap')
        real_aggregate = bootstrap_mod._aggregate_iteration_results

        def fake_aggregate(results, config):
            agg = real_aggregate(results, config)
            (iteration_data, predictions_data, auc_data, brier_t_data,
             importance_data, subpopulation_auc_data, all_failures,
             all_notices) = agg
            all_notices.append(('harrell', '', 1, 3))
            return (iteration_data, predictions_data, auc_data, brier_t_data,
                    importance_data, subpopulation_auc_data, all_failures,
                    all_notices)

        monkeypatch.setattr(
            bootstrap_mod, '_aggregate_iteration_results', fake_aggregate,
        )
        learner, X_val, y_val = learner_with_setup_subpopulations
        coxph = learner._state.models['coxph'].model
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.bootstrap(
                models={'coxph': coxph},
                X=X_val, y=y_val, n_bootstrap=2, verbose=False,
            )
        cov_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and 'insufficient eval_times coverage' in str(w.message)
        ]
        assert cov_warns, "Expected low-coverage MethodologicalWarning"
        assert cov_warns[0].filename == __file__, (
            f"Expected the multi-model low-coverage summary to attribute the "
            f"warning to this test file ({__file__}); got "
            f"{cov_warns[0].filename}."
        )

    def test_metric_failure_warning_expected_error_emits(
        self, learner_with_setup_subpopulations, monkeypatch,
    ):
        """Even "expected" statistical errors
        (ValueError/ZeroDivisionError) must surface a MetricFailureWarning,
        since every failed iteration drops user-visible bootstrap data.
        Inject 3/10 ValueError failures and assert the warning fires and
        attributes to the user frame."""
        import importlib
        bootstrap_mod = importlib.import_module('mlsurv.evaluation.bootstrap')
        real_aggregate = bootstrap_mod._aggregate_iteration_results

        def fake_aggregate(results, config):
            agg = real_aggregate(results, config)
            (iteration_data, predictions_data, auc_data, brier_t_data,
             importance_data, subpopulation_auc_data, all_failures,
             all_notices) = agg
            all_failures.extend([
                ('harrell', 'ValueError', 'boundary probe')
            ] * 3)
            return (iteration_data, predictions_data, auc_data, brier_t_data,
                    importance_data, subpopulation_auc_data, all_failures,
                    all_notices)

        monkeypatch.setattr(
            bootstrap_mod, '_aggregate_iteration_results', fake_aggregate,
        )
        learner, X_val, y_val = learner_with_setup_subpopulations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            # Inject 3/10 ValueError bootstrap failures.
            learner.bootstrap(
                X=X_val, y=y_val, n_bootstrap=10, verbose=False,
            )
        metric_warns = [
            w for w in caught
            if issubclass(w.category, MetricFailureWarning)
        ]
        assert metric_warns, (
            "Expected MetricFailureWarning for ValueError failures: "
            "any failure must surface."
        )
        assert metric_warns[0].filename == __file__

    def test_emit_failure_summary_default_stacklevel(self):
        """``_emit_failure_summary``'s default ``_warn_stacklevel=2``
        preserves pre-fix behavior for any direct caller. Pin it so a future
        refactor that drops the default or changes it is caught."""
        import importlib
        bootstrap_mod = importlib.import_module('mlsurv.evaluation.bootstrap')
        failures = [('harrell', 'RuntimeError', 'direct probe')]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            bootstrap_mod._emit_failure_summary(failures, 1, 'probe')
        metric_warns = [
            w for w in caught
            if issubclass(w.category, MetricFailureWarning)
        ]
        assert metric_warns, "Expected MetricFailureWarning"
        # Default stacklevel=2 attributes to the caller of
        # _emit_failure_summary, i.e., this test file.
        assert metric_warns[0].filename == __file__, (
            f"Expected default stacklevel=2 to attribute to caller file "
            f"({__file__}); got {metric_warns[0].filename}."
        )


class TestEvaluateColumnGuard:
    """Tests for evaluate() column alignment when X is passed explicitly."""

    def test_evaluate_missing_columns_raises(self, small_data):
        """evaluate(X=...) raises ValueError when custom X is missing columns."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()
        learner.train('coxph')

        X_bad = X.iloc[:10].drop(columns=[X.columns[0]])
        with pytest.raises(ValueError, match=r"missing.*1.*training column"):
            learner.evaluate('coxph', X=X_bad, y=y[:10])

    def test_evaluate_default_x_no_guard(self, small_data):
        """evaluate() with default X_test does not trigger column guard."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()
        learner.train('coxph')

        # Should succeed — default X_test always has correct columns
        result = learner.evaluate('coxph')
        assert result is not None


class TestSurvivalLearnerAdvanced:
    """Test advanced SurvivalLearner functionality."""

    def test_model_result_has_fitted_model(self, synthetic_data_clean):
        """Verify model_result.model is a fitted pipeline."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup()

        model_result = learner.train('coxph')

        # model_result.model is already fitted on full training set
        assert model_result.model is not None
        assert hasattr(model_result.model, 'predict')

    def test_setup_groupkfold(self, synthetic_data_clean):
        """Verify GroupKFold CV strategy."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        # Create group labels (simulate multi-site data)
        groups = np.repeat(np.arange(10), len(X) // 10 + 1)[:len(X)]

        learner = SurvivalLearner(X, y)
        learner.setup(
            cv_strategy='groupkfold',
            cv_folds=5,
            groups=groups
        )

        assert learner._state.is_setup
        # GroupKFold should be used
        from sklearn.model_selection import GroupKFold
        assert isinstance(learner._state.cv_splitter, GroupKFold)

    def test_setup_custom_cv_object(self, synthetic_data_clean):
        """Verify custom CV object passthrough."""
        from mlsurv import SurvivalLearner
        from sklearn.model_selection import RepeatedStratifiedKFold

        X, y = synthetic_data_clean

        custom_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)

        learner = SurvivalLearner(X, y)
        learner.setup(cv_strategy=custom_cv)

        assert learner._state.is_setup
        # Custom CV object should be used as-is
        assert learner._state.cv_splitter is custom_cv

    def test_tune_force(self, small_data):
        """Verify force re-tuning."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        # First tuning
        result1 = learner.tune(
            'rsf',
            n_trials=3,
            search_space={'n_estimators': (10, 20)}
        )
        assert result1 is not None

        # Second tuning with force=True should run
        result2 = learner.tune(
            'rsf',
            n_trials=3,
            force=True,
            search_space={'n_estimators': (10, 20)}
        )
        assert result2 is not None

    def test_run_with_tune(self, small_data):
        """Verify run with tune=True returns dict of ModelResult."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        # Tune CoxNet (doesn't have 'k' param issue in default space)
        result = learner.run(
            models=['coxnet'],
            tune=True,
            n_trials=3,
            search_spaces={'coxnet': {
                'l1_ratio': (0.0, 1.0),
                'n_alphas': (10, 50)
            }},
            verbose=False
        )

        # run() now returns dict of {str: ModelResult}
        assert isinstance(result, dict)
        assert 'coxnet' in result
        assert learner.best_model is not None

    def test_setup_kfold_strategy(self, synthetic_data_clean):
        """Verify KFold CV strategy."""
        from mlsurv import SurvivalLearner
        from sklearn.model_selection import KFold

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(cv_strategy='kfold', cv_folds=5)

        assert learner._state.is_setup
        assert isinstance(learner._state.cv_splitter, KFold)

    def test_setup_auto_cv_folds(self, small_data):
        """Verify auto CV folds selection."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds='auto')

        assert learner._state.is_setup
        # Auto should select based on number of events
        assert learner.cv_folds >= 3

    def test_get_config(self, small_data):
        """Verify config returns complete configuration."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.25, random_state=123, scoring='ipcw')

        config = learner.config

        assert config['test_fraction'] == 0.25
        assert config['random_state'] == 123
        assert config['scoring'] == 'ipcw'
        assert 'n_train' in config
        assert 'n_test' in config
        assert 'cv_folds' in config


class TestSurvivalLearnerStudyManagement:
    """Test Optuna study management."""

    def test_experiment_name_set_in_setup(self, small_data, tmp_path):
        """Verify experiment_name can be set in setup."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(experiment_name='test_experiment')

        assert learner._state.experiment_name == 'test_experiment'

    def test_tune_creates_study_files(self, small_data, tmp_path):
        """Verify tuning creates study files."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(experiment_name='test_tuning')

        # Tune with minimal trials
        result = learner.tune(
            'rsf',
            n_trials=2,
            search_space={'n_estimators': (10, 20)}
        )

        # Optuna study should be attached
        assert result.optuna_study is not None
        assert result.best_params_ is not None


class TestRepeatedCrossValidation:
    """Test repeated cross-validation functionality."""

    def test_cv_repeats_default(self, small_data):
        """Verify default cv_repeats=1."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        assert learner.cv_repeats == 1
        config = learner.config
        assert config['cv_repeats'] == 1

    def test_cv_repeats_multiple(self, small_data):
        """Verify cv_repeats > 1 creates repeated CV splitter."""
        from mlsurv import SurvivalLearner
        from sklearn.model_selection import RepeatedStratifiedKFold

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=5, cv_repeats=3)

        assert learner.cv_repeats == 3
        # Main splitter should be RepeatedStratifiedKFold
        assert isinstance(learner._state.cv_splitter, RepeatedStratifiedKFold)
        # Tuning splitter should be single-run (StratifiedKFold)
        from sklearn.model_selection import StratifiedKFold
        assert isinstance(learner._state.cv_splitter_single, StratifiedKFold)

    def test_cv_repeats_creates_more_scores(self, small_data):
        """Verify cv_repeats produces more CV scores than single-run."""
        from mlsurv import SurvivalLearner
        import warnings

        X, y = small_data
        cv_folds = 5

        # First, create model with single-run CV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner_single = SurvivalLearner(X, y)
            learner_single.setup(cv_folds=cv_folds, cv_repeats=1)
            result_single = learner_single.train('coxph')

        # Then, create model with repeated CV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner_repeat = SurvivalLearner(X, y)
            learner_repeat.setup(cv_folds=cv_folds, cv_repeats=3)
            result_repeat = learner_repeat.train('coxph')

        # Repeated CV should produce 3x as many scores
        assert len(result_repeat.cv_metric_scores) == 3 * len(result_single.cv_metric_scores), (
            f"Expected 3x scores: single={len(result_single.cv_metric_scores)}, "
            f"repeat={len(result_repeat.cv_metric_scores)}"
        )

    def test_cv_repeats_kfold(self, small_data):
        """Verify cv_repeats works with kfold strategy."""
        from mlsurv import SurvivalLearner
        from sklearn.model_selection import RepeatedKFold, KFold

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_strategy='kfold', cv_folds=5, cv_repeats=2)

        assert isinstance(learner._state.cv_splitter, RepeatedKFold)
        assert isinstance(learner._state.cv_splitter_single, KFold)

    def test_cv_repeats_groupkfold_warning(self, small_data):
        """Verify cv_repeats > 1 with GroupKFold issues warning."""
        import warnings
        from mlsurv import SurvivalLearner
        from sklearn.model_selection import GroupKFold

        X, y = small_data
        # Create group labels
        groups = np.repeat(np.arange(10), len(X) // 10 + 1)[:len(X)]

        learner = SurvivalLearner(X, y)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.setup(cv_strategy='groupkfold', cv_folds=5, cv_repeats=3, groups=groups)

            # Should have warning about cv_repeats not supported
            cv_warnings = [x for x in w if "cv_repeats > 1 is not supported" in str(x.message)]
            assert len(cv_warnings) > 0

        # Should still use GroupKFold (falls back to repeats=1)
        assert isinstance(learner._state.cv_splitter, GroupKFold)

    def test_cv_repeats_in_config(self, small_data):
        """Verify cv_repeats appears in config."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=5, cv_repeats=3)

        config = learner.config

        assert 'cv_repeats' in config
        assert config['cv_repeats'] == 3

    def test_cv_warning_small_sample(self, small_data):
        """Verify warning is issued for small sample size without cv_repeats."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = small_data  # small_data has ~50 samples
        learner = SurvivalLearner(X, y)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.setup(cv_repeats=1)

            # Should have warning about considering cv_repeats
            cv_warnings = [x for x in w if "cv_repeats > 1" in str(x.message)]
            assert len(cv_warnings) > 0

    def test_cv_warning_suppressed_with_repeats(self, small_data):
        """Verify warning is suppressed when using cv_repeats > 1."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.setup(cv_repeats=3)

            # Should NOT have warning about considering cv_repeats
            cv_warnings = [x for x in w if "Consider using cv_repeats" in str(x.message)]
            assert len(cv_warnings) == 0


@pytest.mark.integration
class TestSurvivalLearnerSerialization:
    """Tests for model serialization and deserialization."""

    def test_save_load_model(self, synthetic_data_clean, tmp_path):
        """Verify a trained model can be serialized and reloaded."""
        import joblib
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.train('coxph')
        original_preds = result.model.predict(learner.X_test)

        # Save model
        model_path = tmp_path / "model.joblib"
        joblib.dump(result.model, str(model_path))

        assert model_path.exists()

        # Load model
        loaded = joblib.load(model_path)
        loaded_preds = loaded.predict(learner.X_test)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_load_pipeline(self, synthetic_data_clean, tmp_path):
        """Verify pipeline with preprocessing can be serialized."""
        import joblib
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.train('rsf', n_estimators=10, random_state=42)
        original_preds = result.model.predict(learner.X_train)

        # Save
        model_path = tmp_path / "pipeline.joblib"
        joblib.dump(result.model, str(model_path))

        # Load
        loaded = joblib.load(model_path)
        loaded_preds = loaded.predict(learner.X_train)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_loaded_model_same_predictions(self, synthetic_data_clean, tmp_path):
        """Verify loaded model produces same predictions on new data."""
        import joblib
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.train('coxph')

        # Get predictions before saving
        train_preds = result.model.predict(learner.X_train)
        test_preds = result.model.predict(learner.X_test)

        # Save
        model_path = tmp_path / "coxph.joblib"
        joblib.dump(result.model, str(model_path))

        # Load
        loaded = joblib.load(model_path)

        # Verify on both train and test
        np.testing.assert_array_almost_equal(
            train_preds, loaded.predict(learner.X_train)
        )
        np.testing.assert_array_almost_equal(
            test_preds, loaded.predict(learner.X_test)
        )

    def test_save_load_with_feature_selector(self, synthetic_data_clean, tmp_path):
        """Verify pipeline with feature selector can be serialized."""
        import joblib
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        selector = TopKCoxSelector(k=10)
        learner.setup(random_state=42, feature_selector=selector)

        result = learner.train('coxph')

        # Verify selector is in pipeline
        assert 'selector' in result.model.named_steps

        original_preds = result.model.predict(learner.X_test)

        # Save
        model_path = tmp_path / "selected.joblib"
        joblib.dump(result.model, str(model_path))

        # Load
        loaded = joblib.load(model_path)
        loaded_preds = loaded.predict(learner.X_test)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_load_tuned_model(self, small_data, tmp_path):
        """Verify tuned model can be serialized."""
        import joblib
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.tune(
            'rsf',
            n_trials=3,
            search_space={'n_estimators': (10, 20)}
        )

        original_preds = result.model.predict(learner.X_test)

        # Save
        model_path = tmp_path / "tuned.joblib"
        joblib.dump(result.model, str(model_path))

        # Load
        loaded = joblib.load(model_path)
        loaded_preds = loaded.predict(learner.X_test)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_load_finalized_model(self, synthetic_data_clean, tmp_path):
        """Verify finalized model can be serialized."""
        import joblib
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.train('coxph')

        original_preds = result.model.predict(X)

        # Save
        model_path = tmp_path / "finalized.joblib"
        joblib.dump(result.model, model_path)

        # Load
        loaded = joblib.load(model_path)
        loaded_preds = loaded.predict(X)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_serialization_preserves_model_attributes(self, synthetic_data_clean, tmp_path):
        """Verify model attributes are preserved after serialization."""
        import joblib
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.train('coxph')

        # Save
        model_path = tmp_path / "attrs.joblib"
        joblib.dump(result.model, str(model_path))

        # Load
        loaded = joblib.load(model_path)

        # Check model step exists
        model_step = loaded.named_steps.get('model')
        if model_step is not None:
            assert hasattr(model_step, 'predict')
            assert hasattr(model_step, 'fit')

    def test_model_path_as_pathlib(self, synthetic_data_clean, tmp_path):
        """Verify serialization works with a pathlib.Path target."""
        import joblib
        from pathlib import Path
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.train('coxph')

        # Use pathlib.Path
        model_path = Path(tmp_path) / "pathlib_model.joblib"
        joblib.dump(result.model, model_path)

        assert model_path.exists()

        # Load and verify
        loaded = joblib.load(model_path)
        assert loaded is not None


@pytest.mark.integration
class TestOverrideWarnings:
    """Test that SurvivalLearner warns when results are overwritten."""

    def test_train_override_warning(self, synthetic_data_clean):
        """Verify train() warns when overwriting existing model."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        # First call - no override warning expected
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.train('coxph')
            override_warnings = [
                x for x in w if "overwritten" in str(x.message).lower()
            ]
            assert len(override_warnings) == 0

        # Second call - warning expected
        with pytest.warns(SuggestionWarning, match="will be overwritten"):
            learner.train('coxph')

    def test_run_override_warning(self, synthetic_data_clean):
        """Verify run() warns when overwriting existing run results."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        # First run - no override warning expected
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.run(models=['coxph'], tune=False)
            override_warnings = [
                x for x in w if "overwritten" in str(x.message).lower()
            ]
            assert len(override_warnings) == 0

        # Second run - warning expected
        with pytest.warns(SuggestionWarning, match="will be overwritten"):
            learner.run(models=['coxph'], tune=False)

    @pytest.mark.integration_heavy
    def test_run_tune_override_warning(self, small_data):
        """Verify run(tune=True) warns when overwriting existing model.

        The successful-tuning path stores its result in the run loop, not
        through _finalize_model_result, so the loop carries its own warning.
        """
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=1, verbose=False)
        learner.train('coxnet', verbose=False)

        with pytest.warns(SuggestionWarning, match="will be overwritten") as rec:
            learner.run(
                models=['coxnet'], n_trials=2, bootstrap=False,
                n_importance_repeats=0, show_eval_progress=False,
                verbose=False, show_progress_bar=False,
            )

        # Confirm the tuning path ran (not the fallback, which stores via
        # the already-warned _finalize_model_result).
        tuned = learner._lookup_model_by_name('coxnet')
        assert tuned.is_tuned and tuned.best_params_

        # No recalibrator/importance-test state existed, so the message is
        # the plain one - no dropped-state note.
        overwrite_msgs = [
            str(w.message) for w in rec
            if "will be overwritten" in str(w.message)
        ]
        assert overwrite_msgs and all(
            "re-run recalibrate" not in m for m in overwrite_msgs
        )

    @pytest.mark.integration_heavy
    def test_tune_overwrite_warning_precedes_install(self, small_data, monkeypatch):
        """The overwrite warning must fire before the tuned result is installed.

        It used to be emitted inside _install_tuned_result, the last step of the
        tune path (after the Optuna run and CV table), so it trailed the output.
        Now _tune_single_model warns up front, before the work, matching
        train()/run().
        """
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=1, verbose=False)
        learner.train('coxnet', verbose=False)

        order = []
        real_warn = learner._warn_overwrite
        real_install = learner._install_tuned_result
        monkeypatch.setattr(
            learner, '_warn_overwrite',
            lambda name: (order.append('warn'), real_warn(name))[-1],
        )
        monkeypatch.setattr(
            learner, '_install_tuned_result',
            lambda name, result: (
                order.append('install'), real_install(name, result)
            )[-1],
        )

        with pytest.warns(SuggestionWarning, match="will be overwritten"):
            learner.tune(
                'coxnet', n_trials=2, show_progress_bar=False, verbose=False,
            )

        # Old code warned inside _install_tuned_result → ['install', 'warn'].
        # The fix warns first → ['warn', 'install'].
        assert order == ['warn', 'install']

    @pytest.mark.integration
    def test_tune_zero_trials_overwrite_warning(self, small_data):
        """tune(n_trials=0) on an existing model still warns once.

        n_trials=0 (no resume) takes the _fallback_train_as_tuned route, which
        installs via the now-silent _install_tuned_result. The early warn in
        _tune_single_model is what keeps this path warning at all.
        """
        import warnings
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=1, verbose=False)
        learner.train('coxnet', verbose=False)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            learner.tune(
                'coxnet', n_trials=0, verbose=False, show_progress_bar=False,
            )
        overwrite_msgs = [
            w for w in rec if "will be overwritten" in str(w.message)
        ]
        assert len(overwrite_msgs) == 1

    @pytest.mark.integration_heavy
    def test_tune_fresh_model_no_overwrite_warning(self, small_data):
        """Tuning a never-trained model must not emit an overwrite warning.

        Guards the `existing is not None` gate in _tune_single_model: with
        nothing stored, there is nothing to overwrite.
        """
        import warnings
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=1, verbose=False)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            learner.tune(
                'coxnet', n_trials=2, verbose=False, show_progress_bar=False,
            )
        overwrite_msgs = [
            w for w in rec if "will be overwritten" in str(w.message)
        ]
        assert len(overwrite_msgs) == 0

    def test_evaluate_override_warning(self, synthetic_data_clean):
        """Verify evaluate() warns when overwriting existing evaluation."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        # First evaluate - no override warning expected
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.evaluate('coxph')
            override_warnings = [
                x for x in w if "overwritten" in str(x.message).lower()
            ]
            assert len(override_warnings) == 0

        # Second evaluate - warning expected
        with pytest.warns(SuggestionWarning, match="will be overwritten"):
            learner.evaluate('coxph')

    def test_bootstrap_override_warning(self, synthetic_data_clean):
        """Verify bootstrap() warns when overwriting existing bootstrap."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        # First bootstrap - no override warning expected
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.bootstrap('coxph', n_bootstrap=2)
            override_warnings = [
                x for x in w if "overwritten" in str(x.message).lower()
            ]
            assert len(override_warnings) == 0

        # Second bootstrap - warning expected
        with pytest.warns(SuggestionWarning, match="will be overwritten"):
            learner.bootstrap('coxph', n_bootstrap=2)

    def test_evaluate_disjoint_no_overwrite_warning(self, synthetic_data_clean):
        """evaluate() must not warn about models the call leaves untouched."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph', 'coxnet'])
        learner.evaluate('coxph')

        # Evaluating a different, not-yet-evaluated model must stay silent -
        # coxph's evaluation is preserved, not overwritten.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.evaluate('coxnet')
            override_warnings = [
                x for x in w if "will be overwritten" in str(x.message)
            ]
            assert len(override_warnings) == 0

        assert learner._state.models['coxph'].evaluation is not None
        assert learner._state.models['coxnet'].evaluation is not None

    def test_bootstrap_disjoint_no_overwrite_warning(self, synthetic_data_clean):
        """bootstrap() must not warn about models the call leaves untouched."""
        import warnings
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph', 'coxnet'])
        learner.bootstrap('coxph', n_bootstrap=2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.bootstrap('coxnet', n_bootstrap=2)
            override_warnings = [
                x for x in w if "will be overwritten" in str(x.message)
            ]
            assert len(override_warnings) == 0

        assert learner._state.models['coxph'].bootstrap is not None
        assert learner._state.models['coxnet'].bootstrap is not None

    def test_evaluate_all_models_still_warns(self, synthetic_data_clean):
        """evaluate(models=None) must still warn once any model is evaluated.

        Guards against over-narrowing: the all-models path resolves every
        trained model, so a re-run is a genuine overwrite and must warn.
        """
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph', 'coxnet'])
        learner.evaluate()  # first evaluate-all - nothing to overwrite yet

        with pytest.warns(SuggestionWarning, match="will be overwritten"):
            learner.evaluate()  # re-evaluating all models is a real overwrite

    def test_overwrite_guard_skips_unknown_names(self, synthetic_data_clean):
        """The scoped guard ignores target names absent from the namespace -
        no warning and no KeyError, never a fall-back to whole-namespace.
        """
        import warnings
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')
        learner.evaluate('coxph')  # coxph now carries an evaluation

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner._warn_if_overwriting_model_data(
                'evaluation', 'evaluation result', ['nonexistent_model'],
            )
            override_warnings = [
                x for x in w if "will be overwritten" in str(x.message)
            ]
            assert len(override_warnings) == 0



class TestCheckCvVsHoldoutBest:
    """Regression test for cv_score bug fix in _check_cv_vs_holdout_best."""

    def test_holdout_warning_includes_cv_and_holdout_scores(self, capsys):
        """Verify message contains both CV= and holdout= labels when models disagree."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class FakeEvaluation:
            c_index_test: Optional[float] = None

        @dataclass
        class FakeModelResult:
            cv_metric_mean: float = 0.0
            evaluation: Optional[FakeEvaluation] = None

        models = {
            'coxph': FakeModelResult(cv_metric_mean=0.75,
                                     evaluation=FakeEvaluation(c_index_test=0.70)),
            'rsf': FakeModelResult(cv_metric_mean=0.72,
                                   evaluation=FakeEvaluation(c_index_test=0.80)),
        }

        from mlsurv.learner._training import TrainingMixin
        # Call the method unbound — pass dict of models directly
        TrainingMixin._check_cv_vs_holdout_best(None, models, verbose=True)

        captured = capsys.readouterr()
        assert "CV=0.750" in captured.out
        assert "holdout=0.700" in captured.out
        assert "'rsf'" in captured.out
        assert "'coxph'" in captured.out


class TestResolveStratifyArray:
    """Tests for SetupMixin._resolve_stratify_array."""

    @staticmethod
    def _make_fixtures():
        """Create minimal X, y fixtures for _resolve_stratify_array."""
        X = pd.DataFrame({'age': [50, 60, 70], 'stage': [1, 2, 3]})
        y = np.array(
            [(True, 12.0), (False, 24.0), (True, 6.0)],
            dtype=[('event', bool), ('time', float)],
        )
        return X, y

    def test_event_string(self):
        """stratify='event' returns y['event']."""
        from mlsurv.learner._setup import SetupMixin
        X, y = self._make_fixtures()
        result = SetupMixin._resolve_stratify_array("event", X, y)
        np.testing.assert_array_equal(result, y['event'])

    def test_column_name_string(self):
        """stratify='stage' returns X['stage']."""
        from mlsurv.learner._setup import SetupMixin
        X, y = self._make_fixtures()
        result = SetupMixin._resolve_stratify_array("stage", X, y)
        pd.testing.assert_series_equal(result, X['stage'])

    def test_numpy_array_passthrough(self):
        """stratify=np.array passes through unchanged."""
        from mlsurv.learner._setup import SetupMixin
        X, y = self._make_fixtures()
        arr = np.array([0, 1, 0])
        result = SetupMixin._resolve_stratify_array(arr, X, y)
        assert result is arr

    def test_pandas_series_passthrough(self):
        """stratify=pd.Series passes through unchanged.

        Pre-fix: ``stratify == "event"`` on a Series raised
        ``ValueError: The truth value of a Series is ambiguous``.
        """
        from mlsurv.learner._setup import SetupMixin
        X, y = self._make_fixtures()
        series = pd.Series([0, 1, 0])
        result = SetupMixin._resolve_stratify_array(series, X, y)
        assert result is series

    def test_none_passthrough(self):
        """stratify=None returns None."""
        from mlsurv.learner._setup import SetupMixin
        X, y = self._make_fixtures()
        result = SetupMixin._resolve_stratify_array(None, X, y)
        assert result is None

    def test_presplit_array_stratify_no_crash(self):
        """Presplit path with array-like stratify doesn't crash."""
        import warnings
        from mlsurv.learner._setup import SetupMixin
        from mlsurv.learner._state import LearnerState

        X, y = self._make_fixtures()
        # Create a minimal mock learner with presplit state
        mock = type('MockLearner', (), {})()
        mock._state = LearnerState()
        mock._state.X_train = X
        mock._state.X_test = X.copy()
        mock._state.y_train = y
        mock.X_train = X
        mock.X_test = X.copy()
        mock.y_train = y

        arr = np.array([0, 1, 0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SetupMixin._setup_presplit_data(mock, 0.3, arr, None)
        # Should warn about ignored stratify, not crash
        msgs = [str(x.message) for x in w]
        assert any("ignored" in m for m in msgs)


class TestGetImputedX:
    """Unit tests for SurvivalLearner._get_imputed_X."""

    @pytest.fixture
    def learner_with_imputer(self, small_data):
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        return learner

    def test_none_input_returns_none(self, learner_with_imputer):
        result = learner_with_imputer._get_imputed_X(None)
        assert result is None

    def test_no_nan_returns_same_object(self, learner_with_imputer):
        X = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        result = learner_with_imputer._get_imputed_X(X)
        assert result is X  # identity — no copy

    def test_nan_imputed_with_configured_imputer(self, learner_with_imputer):
        X = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        result = learner_with_imputer._get_imputed_X(X)
        assert not result.isna().any().any()
        assert list(result.columns) == ['a', 'b']
        assert list(result.index) == [0, 1, 2]

    def test_fallback_median_when_no_imputer(self, small_data):
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, imputer=None)
        X_nan = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, 6.0]})
        result = learner._get_imputed_X(X_nan)
        assert not result.isna().any().any()
        # Median of [1.0, 3.0] is 2.0
        assert result.loc[1, 'a'] == pytest.approx(2.0)

    def test_raise_on_error_propagates(self, learner_with_imputer):
        """raise_on_error=True re-raises after logging."""
        from unittest.mock import patch

        X_nan = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        with patch(
            "sklearn.base.clone",
            side_effect=ValueError("all NaN column"),
        ):
            with pytest.raises(ValueError, match="all NaN column"):
                learner_with_imputer._get_imputed_X(X_nan, raise_on_error=True)

    def test_raise_on_error_false_returns_raw(self, learner_with_imputer):
        """raise_on_error=False (default) returns raw data on failure."""
        from unittest.mock import patch

        X_nan = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        with patch(
            "sklearn.base.clone",
            side_effect=ValueError("all NaN column"),
        ):
            result = learner_with_imputer._get_imputed_X(X_nan)
        assert result is X_nan  # returns same object unchanged

    def test_memory_error_always_propagates(self, learner_with_imputer):
        """MemoryError propagates regardless of raise_on_error."""
        from unittest.mock import patch

        X_nan = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        with patch("sklearn.base.clone", side_effect=MemoryError("oom")):
            with pytest.raises(MemoryError):
                learner_with_imputer._get_imputed_X(X_nan, raise_on_error=False)

    def test_uncaught_exception_type_propagates(self, learner_with_imputer):
        """Exception types outside the explicit list propagate unconditionally."""
        from unittest.mock import patch

        X_nan = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        with patch("sklearn.base.clone", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                learner_with_imputer._get_imputed_X(X_nan, raise_on_error=False)
