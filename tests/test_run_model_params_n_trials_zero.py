"""Tests for run() model_params with n_trials=0."""

import warnings
from unittest.mock import patch, MagicMock

from mlsurv.learner import SurvivalLearner
from mlsurv._warnings import SuggestionWarning


_MODEL = "rsf"


class TestModelParamsWithNTrialsZero:
    """run(n_trials=0, model_params=...) should apply params, not discard them."""

    def test_model_params_applied_when_n_trials_zero(self, small_data):
        """model_params are applied (not silently discarded) when n_trials=0."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        results = learner.run(
            _MODEL, n_trials=0,
            model_params={_MODEL: {"n_estimators": 17}},
            evaluate=False, verbose=False,
        )
        assert _MODEL in results
        estimator = results[_MODEL].model.named_steps["model"]
        assert estimator.n_estimators == 17

    def test_kwargs_applied_when_n_trials_zero_single_model(self, small_data):
        """Single-model **kwargs convenience works with n_trials=0."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        results = learner.run(
            _MODEL, n_trials=0, n_estimators=17,
            evaluate=False, verbose=False,
        )
        assert _MODEL in results
        estimator = results[_MODEL].model.named_steps["model"]
        assert estimator.n_estimators == 17

    def test_inference_warning_emitted(self, small_data):
        """SuggestionWarning about inferring tune=False is emitted."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.run(
                _MODEL, n_trials=0,
                model_params={_MODEL: {"n_estimators": 17}},
                evaluate=False, verbose=False,
            )
            inference_warnings = [
                x for x in w
                if issubclass(x.category, SuggestionWarning)
                and "Inferring tune=False" in str(x.message)
            ]
            assert len(inference_warnings) == 1


class TestNTrialsZeroNoInference:
    """Cases where the n_trials=0 inference should NOT trigger."""

    def test_n_trials_zero_without_model_params_unchanged(self, small_data):
        """n_trials=0 without model_params keeps existing tuning behavior."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        results = learner.run(
            _MODEL, tune=True, n_trials=0,
            evaluate=False, verbose=False,
        )
        assert _MODEL in results
        # Existing behavior: is_tuned=True, empty best_params_ (defaults)
        assert results[_MODEL].is_tuned
        assert results[_MODEL].best_params_ == {}

    def test_resume_true_not_inferred(self, small_data):
        """resume=True bypasses inference even with model_params."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        # Mock the tuner so we don't need a real study on disk
        mock_tune = MagicMock(return_value=({}, MagicMock(), "test_exp"))
        with patch.object(type(learner), '_create_tuner_for_model') as mock_create:
            tuner = MagicMock()
            tuner.has_searchable_params = True
            tuner.tune = mock_tune
            mock_create.return_value = tuner

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                learner.run(
                    _MODEL, tune=True, n_trials=0, resume=True,
                    model_params={_MODEL: {"n_estimators": 17}},
                    evaluate=False, verbose=False,
                )
                # Should get "ignored when tune=True" warning, not our inference
                inference_warnings = [
                    x for x in w
                    if issubclass(x.category, SuggestionWarning)
                    and "Inferring tune=False" in str(x.message)
                ]
                assert len(inference_warnings) == 0

    def test_n_trials_auto_not_inferred(self, small_data):
        """n_trials='auto' does not trigger inference."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        # Mock the tuner to avoid running real Optuna trials
        mock_tune = MagicMock(return_value=({}, MagicMock(), "test_exp"))
        with patch.object(type(learner), '_create_tuner_for_model') as mock_create:
            tuner = MagicMock()
            tuner.has_searchable_params = True
            tuner.tune = mock_tune
            mock_create.return_value = tuner

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                learner.run(
                    _MODEL, tune=True, n_trials="auto",
                    model_params={_MODEL: {"n_estimators": 17}},
                    evaluate=False, verbose=False,
                )
                inference_warnings = [
                    x for x in w
                    if issubclass(x.category, SuggestionWarning)
                    and "Inferring tune=False" in str(x.message)
                ]
                assert len(inference_warnings) == 0

    def test_n_trials_dict_not_inferred(self, small_data):
        """n_trials={'rsf': 0} (dict) does not trigger inference."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.run(
                _MODEL, tune=True, n_trials={_MODEL: 0},
                model_params={_MODEL: {"n_estimators": 17}},
                evaluate=False, verbose=False,
            )
            inference_warnings = [
                x for x in w
                if issubclass(x.category, SuggestionWarning)
                and "Inferring tune=False" in str(x.message)
            ]
            assert len(inference_warnings) == 0

    def test_n_trials_positive_not_inferred(self, small_data):
        """n_trials=2 does not trigger inference (normal tuning path)."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        # Mock the tuner to avoid running real Optuna trials
        mock_tune = MagicMock(return_value=({}, MagicMock(), "test_exp"))
        with patch.object(type(learner), '_create_tuner_for_model') as mock_create:
            tuner = MagicMock()
            tuner.has_searchable_params = True
            tuner.tune = mock_tune
            mock_create.return_value = tuner

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                learner.run(
                    _MODEL, tune=True, n_trials=2,
                    model_params={_MODEL: {"n_estimators": 17}},
                    evaluate=False, verbose=False,
                )
                inference_warnings = [
                    x for x in w
                    if issubclass(x.category, SuggestionWarning)
                    and "Inferring tune=False" in str(x.message)
                ]
                assert len(inference_warnings) == 0
