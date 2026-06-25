"""Tests for run(resume=...) parameter threading."""

import warnings
from unittest.mock import patch, MagicMock

import pytest

from mlsurv.learner import SurvivalLearner
from mlsurv._warnings import SuggestionWarning


_MODEL = "rsf"


class TestRunResumeThreading:
    """Verify resume= is threaded from run() → _run_model_comparison → _tune_and_train → tuner.tune."""

    def test_run_resume_threads_to_tuner(self, small_data):
        """run(resume=True) passes resume=True to tuner.tune()."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        mock_tune = MagicMock(return_value=({}, MagicMock(), "test_exp"))
        with patch.object(type(learner), '_create_tuner_for_model') as mock_create:
            tuner = MagicMock()
            tuner.has_searchable_params = True
            tuner.tune = mock_tune
            mock_create.return_value = tuner

            learner.run(
                _MODEL, tune=True, n_trials=5, resume=True,
                evaluate=False, verbose=False,
            )

            mock_tune.assert_called_once()
            assert mock_tune.call_args.kwargs['resume'] is True

    def test_run_resume_false_by_default(self, small_data):
        """run() defaults to resume=False."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        mock_tune = MagicMock(return_value=({}, MagicMock(), "test_exp"))
        with patch.object(type(learner), '_create_tuner_for_model') as mock_create:
            tuner = MagicMock()
            tuner.has_searchable_params = True
            tuner.tune = mock_tune
            mock_create.return_value = tuner

            learner.run(
                _MODEL, tune=True, n_trials=5,
                evaluate=False, verbose=False,
            )

            mock_tune.assert_called_once()
            assert mock_tune.call_args.kwargs['resume'] is False

    def test_run_resume_threads_to_all_models(self, small_data):
        """resume= is threaded to each model in a multi-model run."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        mock_tune = MagicMock(return_value=({}, MagicMock(), "test_exp"))
        with patch.object(type(learner), '_create_tuner_for_model') as mock_create:
            tuner = MagicMock()
            tuner.has_searchable_params = True
            tuner.tune = mock_tune
            mock_create.return_value = tuner

            learner.run(
                ['rsf', 'gbsa'], tune=True, n_trials=5, resume=True,
                evaluate=False, verbose=False,
            )

            assert mock_tune.call_count == 2
            for call in mock_tune.call_args_list:
                assert call.kwargs['resume'] is True


@pytest.mark.integration
class TestRunNTrialsZeroResume:
    """n_trials=0 + resume interaction via run()."""

    def test_run_n_trials_zero_resume_loads_study(self, small_data):
        """run(n_trials=0, resume=True) loads existing study params."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        # First: tune normally to create a study
        learner.tune(_MODEL, n_trials=3, verbose=False)

        # Now: run with n_trials=0, resume=True to reload
        results = learner.run(
            _MODEL, tune=True, n_trials=0, resume=True,
            evaluate=False, verbose=False,
        )
        assert _MODEL in results
        assert results[_MODEL].is_tuned
        assert results[_MODEL].best_params_  # non-empty

    def test_run_n_trials_zero_no_resume_uses_defaults(self, small_data):
        """run(n_trials=0, resume=False) skips tuning and uses defaults."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        results = learner.run(
            _MODEL, tune=True, n_trials=0, resume=False,
            evaluate=False, verbose=False,
        )
        assert _MODEL in results
        assert results[_MODEL].is_tuned
        assert results[_MODEL].best_params_ == {}  # empty = defaults

    def test_run_n_trials_zero_no_resume_warns(self, small_data):
        """run(n_trials=0, resume=False) emits a SuggestionWarning."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.run(
                _MODEL, tune=True, n_trials=0, resume=False,
                evaluate=False, verbose=False,
            )
            suggestion_warnings = [x for x in w if issubclass(x.category, SuggestionWarning)]
            assert any("n_trials=0 without resume=True" in str(x.message) for x in suggestion_warnings)


class TestRunResumeIgnoredWhenTuneFalse:
    """resume= is harmlessly ignored when tune=False."""

    def test_run_resume_ignored_when_tune_false(self, small_data):
        """run(tune=False, resume=True) succeeds without error."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        results = learner.run(
            _MODEL, tune=False, resume=True,
            evaluate=False, verbose=False,
        )
        assert _MODEL in results


class TestCliRunResumeFlag:
    """CLI --resume flag is parsed correctly."""

    def test_cli_run_resume_flag(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['run', 'data.csv', '--resume'])
        assert args.resume is True

    def test_cli_run_resume_default_false(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['run', 'data.csv'])
        assert args.resume is False
