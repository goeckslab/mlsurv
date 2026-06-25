"""Tests for small-sample tuning instability warning."""

import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from mlsurv.learner import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning
from mlsurv.constants import MIN_SAMPLE_SIZE_WARNING
from mlsurv.utils import to_structured_array


_MODEL = "coxph"


def _make_data(n_samples, n_features=5, seed=42):
    """Create synthetic survival data with exact sample count."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    time = rng.exponential(10, n_samples).astype(np.float64)
    event = rng.binomial(1, 0.6, n_samples).astype(bool)
    y = to_structured_array(time, event)
    return X, y


class TestWarnSmallSampleTuningDirect:
    """Test _warn_small_sample_tuning() helper directly."""

    def test_fires_below_threshold(self):
        """Warning fires when n_train < MIN_SAMPLE_SIZE_WARNING."""
        X, y = _make_data(100)
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with pytest.warns(MethodologicalWarning, match="Small training set"):
            learner._warn_small_sample_tuning()

    def test_silent_at_threshold(self):
        """No warning when n_train >= MIN_SAMPLE_SIZE_WARNING."""
        # Default test_fraction=0.3, so need ~286 total to get 200 train
        X, y = _make_data(300)
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        assert len(learner.y_train) >= MIN_SAMPLE_SIZE_WARNING

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner._warn_small_sample_tuning()

        small_sample_warnings = [
            x for x in w if issubclass(x.category, MethodologicalWarning)
            and "Small training set" in str(x.message)
        ]
        assert len(small_sample_warnings) == 0

    def test_message_includes_n_train(self):
        """Warning message includes the actual training set size."""
        X, y = _make_data(80)
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        n_train = len(learner.y_train)

        with pytest.warns(MethodologicalWarning, match=f"n={n_train}"):
            learner._warn_small_sample_tuning()

    def test_warning_type_is_methodological(self):
        """Warning is MethodologicalWarning, not UserWarning."""
        X, y = _make_data(80)
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner._warn_small_sample_tuning()

        small_sample_warnings = [
            x for x in w if "Small training set" in str(x.message)
        ]
        assert len(small_sample_warnings) == 1
        assert issubclass(small_sample_warnings[0].category, MethodologicalWarning)


    def test_boundary_fires_at_199(self):
        """Warning fires when n_train is exactly 199 (one below threshold)."""
        X, y = _make_data(199 + 50)  # 50 for test set
        X_train, X_test = X.iloc[:199], X.iloc[199:]
        y_train, y_test = y[:199], y[199:]
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        )
        learner.setup(random_state=42)
        assert len(learner.y_train) == 199

        with pytest.warns(MethodologicalWarning, match="Small training set"):
            learner._warn_small_sample_tuning()

    def test_boundary_silent_at_200(self):
        """No warning when n_train is exactly 200 (at threshold)."""
        X, y = _make_data(200 + 50)
        X_train, X_test = X.iloc[:200], X.iloc[200:]
        y_train, y_test = y[:200], y[200:]
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        )
        learner.setup(random_state=42)
        assert len(learner.y_train) == 200

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner._warn_small_sample_tuning()

        small_sample_warnings = [
            x for x in w if issubclass(x.category, MethodologicalWarning)
            and "Small training set" in str(x.message)
        ]
        assert len(small_sample_warnings) == 0


class TestSmallSampleWarningInTuneAndTrain:
    """Test that _tune_and_train() fires the warning at the right times."""

    def _run_tune_and_train(self, learner, n_trials=5, resume=False):
        """Run _tune_and_train with a mocked tuner to avoid real tuning."""
        mock_tune = MagicMock(return_value=({}, MagicMock(), "test_exp"))
        with patch.object(type(learner), '_create_tuner_for_model') as mock_create:
            tuner = MagicMock()
            tuner.has_searchable_params = True
            tuner.tune = mock_tune
            mock_create.return_value = tuner

            learner.run(
                _MODEL, tune=True, n_trials=n_trials, resume=resume,
                evaluate=False, verbose=False,
            )

    def test_fires_during_tuning_small_data(self, small_data):
        """Warning fires when tuning on small dataset."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with pytest.warns(MethodologicalWarning, match="Small training set"):
            self._run_tune_and_train(learner)

    def test_silent_when_n_trials_zero_no_resume(self, small_data):
        """No small-sample warning when n_trials=0 (tuning skipped)."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._run_tune_and_train(learner, n_trials=0, resume=False)

        small_sample_warnings = [
            x for x in w if issubclass(x.category, MethodologicalWarning)
            and "Small training set" in str(x.message)
        ]
        assert len(small_sample_warnings) == 0

    def test_silent_when_tune_false(self, small_data):
        """No small-sample warning when tune=False."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.run(_MODEL, tune=False, evaluate=False, verbose=False)

        small_sample_warnings = [
            x for x in w if issubclass(x.category, MethodologicalWarning)
            and "Small training set" in str(x.message)
        ]
        assert len(small_sample_warnings) == 0

    def test_epv_and_small_sample_can_cofire(self):
        """Both EPV and small-sample warnings can fire on the same dataset.

        Setup fires the EPV warning (EPV < 10) and sets
        epv_warned_at_setup=True, so tuning skips its redundant EPV check.
        The total EPV warning count across setup + tuning is exactly 1.
        """
        # Many features + few samples = low EPV + small n
        X, y = _make_data(60, n_features=30)
        learner = SurvivalLearner(X, y)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.setup(random_state=42)
            self._run_tune_and_train(learner)

        epv_warnings = [x for x in w if "EPV" in str(x.message)]
        small_sample_warnings = [
            x for x in w if "Small training set" in str(x.message)
        ]
        assert len(epv_warnings) == 1, "EPV warning should fire exactly once (from setup)"
        assert len(small_sample_warnings) >= 1, "Small-sample warning should fire"

    def test_fires_via_tune_direct(self, small_data):
        """Warning fires through tune() -> _tune_single_model path."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        mock_tune = MagicMock(return_value=({}, MagicMock(), "test_exp"))
        with patch.object(type(learner), '_create_tuner_for_model') as mock_create:
            tuner = MagicMock()
            tuner.has_searchable_params = True
            tuner.tune = mock_tune
            mock_create.return_value = tuner

            with pytest.warns(MethodologicalWarning, match="Small training set"):
                learner.tune(_MODEL, n_trials=3, verbose=False)

    def test_silent_when_n_trials_zero_resume_true(self, small_data):
        """No warning when n_trials=0 + resume=True (loading existing study)."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        mock_tune = MagicMock(return_value=({}, MagicMock(), "test_exp"))
        with patch.object(type(learner), '_create_tuner_for_model') as mock_create:
            tuner = MagicMock()
            tuner.has_searchable_params = True
            tuner.tune = mock_tune
            mock_create.return_value = tuner

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                learner.tune(_MODEL, n_trials=0, resume=True, verbose=False)

        small_sample_warnings = [
            x for x in w if issubclass(x.category, MethodologicalWarning)
            and "Small training set" in str(x.message)
        ]
        assert len(small_sample_warnings) == 0
