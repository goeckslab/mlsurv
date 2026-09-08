"""Tests for user-facing friendly error messages.

Each test verifies that common input errors produce clear, jargon-free
messages with corrective suggestions — not raw dependency tracebacks.
"""

import pytest
import numpy as np
import pandas as pd

from tests._helpers import make_survival_array


# ---------------------------------------------------------------------------
# Too few events
# ---------------------------------------------------------------------------
class TestTooFewEvents:
    """validate_survival_data raises on <5 events."""

    def test_zero_events(self):
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({'a': np.ones(20), 'b': np.ones(20)})
        y = make_survival_array(np.zeros(20, dtype=bool), np.ones(20))

        with pytest.raises(ValueError, match="Too few events.*0"):
            validate_survival_data(X, y)

    def test_three_events(self):
        from mlsurv.utils import validate_survival_data

        events = np.zeros(20, dtype=bool)
        events[:3] = True
        X = pd.DataFrame({'a': np.ones(20), 'b': np.ones(20)})
        y = make_survival_array(events, np.ones(20))

        with pytest.raises(ValueError, match="Too few events.*3"):
            validate_survival_data(X, y)

    def test_five_events_passes(self):
        from mlsurv.utils import validate_survival_data

        events = np.zeros(20, dtype=bool)
        events[:5] = True
        X = pd.DataFrame({'a': np.ones(20), 'b': np.ones(20)})
        y = make_survival_array(events, np.ones(20))

        validate_survival_data(X, y)  # Should not raise

    def test_min_events_zero_skips_check(self):
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({'a': np.ones(20), 'b': np.ones(20)})
        y = make_survival_array(np.zeros(20, dtype=bool), np.ones(20))

        # min_events=0 disables the check
        validate_survival_data(X, y, min_events=0)


# ---------------------------------------------------------------------------
# Non-numeric column detection
# ---------------------------------------------------------------------------
class TestDataTypeValidation:
    """validate_survival_data rejects object/category columns."""

    def test_object_columns(self):
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({
            'age': [30.0, 40.0, 50.0] * 5,
            'gender': ['M', 'F', 'M'] * 5,
        })
        y = make_survival_array(
            np.array([True, False] * 7 + [True]),
            np.ones(15),
        )

        with pytest.raises(ValueError, match="non-numeric columns.*gender"):
            validate_survival_data(X, y)

    def test_category_column(self):
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({
            'score': np.ones(15),
            'stage': pd.Categorical(['I', 'II', 'III'] * 5),
        })
        y = make_survival_array(
            np.array([True, False] * 7 + [True]),
            np.ones(15),
        )

        with pytest.raises(ValueError, match="non-numeric columns.*stage"):
            validate_survival_data(X, y)

    def test_suggests_encoding(self):
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({
            'txt': ['a'] * 15,
            'num': np.ones(15),
        })
        y = make_survival_array(
            np.array([True, False] * 7 + [True]),
            np.ones(15),
        )

        with pytest.raises(ValueError, match="pd.get_dummies"):
            validate_survival_data(X, y)

    def test_many_non_numeric_truncated(self):
        from mlsurv.utils import validate_survival_data

        cols = {f'cat_{i}': ['a'] * 15 for i in range(8)}
        cols['num'] = np.ones(15)
        X = pd.DataFrame(cols)
        y = make_survival_array(
            np.array([True, False] * 7 + [True]),
            np.ones(15),
        )

        with pytest.raises(ValueError, match="and 3 more"):
            validate_survival_data(X, y)


# ---------------------------------------------------------------------------
# Minimum sample size
# ---------------------------------------------------------------------------
class TestMinimumSampleSize:
    """validate_survival_data rejects datasets with <10 samples."""

    def test_five_samples_rejected(self):
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({'a': np.ones(5)})
        y = make_survival_array(
            np.array([True, True, True, True, True]),
            np.ones(5),
        )

        with pytest.raises(ValueError, match="only 5 samples"):
            validate_survival_data(X, y)

    def test_ten_samples_passes(self):
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({'a': np.ones(10)})
        events = np.zeros(10, dtype=bool)
        events[:5] = True
        y = make_survival_array(events, np.ones(10))

        validate_survival_data(X, y)  # Should not raise

    def test_min_samples_zero_skips(self):
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({'a': np.ones(3)})
        y = make_survival_array(
            np.array([True, True, True]),
            np.ones(3),
        )

        validate_survival_data(X, y, min_samples=0, min_events=0)


# ---------------------------------------------------------------------------
# Pre-split data validation
# ---------------------------------------------------------------------------
class TestPresplitValidation:
    """Pre-split data with small test sets should be accepted."""

    def test_small_test_set_accepted(self):
        from mlsurv.learner import SurvivalLearner

        rng = np.random.RandomState(42)
        X_train = pd.DataFrame(rng.randn(50, 3), columns=['a', 'b', 'c'])
        X_test = pd.DataFrame(rng.randn(5, 3), columns=['a', 'b', 'c'])
        events_train = np.array([True, False] * 25)
        y_train = make_survival_array(events_train, rng.uniform(1, 10, 50))
        events_test = np.array([True, False, True, False, True])
        y_test = make_survival_array(events_test, rng.uniform(1, 10, 5))

        # Small test set (5 samples, 3 events) should NOT raise
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(verbose=False)


# ---------------------------------------------------------------------------
# CV fold validation
# ---------------------------------------------------------------------------
class TestCvFoldValidation:
    """setup() rejects cv_folds > training events."""

    def test_folds_exceed_events(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        # With ~140 events total and 70% in training, training has ~98 events.
        # cv_folds=200 should exceed that.
        with pytest.raises(ValueError, match="cv_folds=200 exceeds training events"):
            learner.setup(cv_folds=200, verbose=False)


# ---------------------------------------------------------------------------
# Scoring metric validation
# ---------------------------------------------------------------------------
class TestScoringValidation:
    """setup() rejects unknown scoring metric strings."""

    def test_typo_in_metric(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        with pytest.raises(ValueError, match="Unknown scoring metric.*harrel"):
            learner.setup(scoring='harrel', verbose=False)  # Missing 'l'

    def test_valid_metric_accepted(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(scoring='ipcw', verbose=False)  # Should not raise


# ---------------------------------------------------------------------------
# Model not found
# ---------------------------------------------------------------------------
class TestModelNotFound:
    """_resolve_all_models and _lookup_model_by_name give helpful errors."""

    def test_no_trained_models(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with pytest.raises(ValueError, match="No trained models.*train\\(\\).*run\\(\\)"):
            learner.evaluate()

    def test_wrong_model_name(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner.train('coxph', verbose=False)

        with pytest.raises(ValueError, match="Model 'nonexistent' not found.*Trained.*All available"):
            learner.evaluate('nonexistent')


# ---------------------------------------------------------------------------
# Visualization errors
# ---------------------------------------------------------------------------
class TestVisualizationErrors:
    """plot_* methods give helpful errors when prerequisites are missing."""

    def test_plot_importance_no_models(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with pytest.raises(ValueError, match="No trained models.*train\\(\\).*run\\(\\)"):
            learner.plot_importance()

    def test_plot_hazard_ratios_multivariate_no_cox(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner.train('rsf', verbose=False)

        with pytest.raises(ValueError, match="valid hazard ratios"):
            learner.plot_hazard_ratios(kind='multivariate')


# ---------------------------------------------------------------------------
# CLI errors
# ---------------------------------------------------------------------------
class TestCLIErrors:
    """CLI _cli_load raises CLIDataError with helpful messages."""

    def test_missing_file(self, tmp_path):
        from argparse import Namespace
        from mlsurv.cli._data import _cli_load
        from mlsurv.cli._exit_codes import CLIDataError

        args = Namespace(data=str(tmp_path / "nonexistent.csv"),
                         time_col="time", event_col="event")
        with pytest.raises(CLIDataError, match="Check the path or use an absolute path"):
            _cli_load(args)

    def test_wrong_column_name(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("feat1,feat2,survival_time,status\n1,2,3,1\n")

        from argparse import Namespace
        from mlsurv.cli._data import _cli_load
        from mlsurv.cli._exit_codes import CLIDataError

        args = Namespace(data=str(csv), time_col="time", event_col="event")
        with pytest.raises(CLIDataError, match="Available columns:"):
            _cli_load(args)

    def test_empty_csv(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("")

        from argparse import Namespace
        from mlsurv.cli._data import _cli_load
        from mlsurv.cli._exit_codes import CLIDataError

        args = Namespace(data=str(csv), time_col="time", event_col="event")
        with pytest.raises(CLIDataError, match="at least a header row"):
            _cli_load(args)

    def test_malformed_csv(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("a,b,c\n1,2\n3,4,5,6\n")

        from argparse import Namespace
        from mlsurv.cli._data import _cli_load
        from mlsurv.cli._exit_codes import CLIDataError

        args = Namespace(data=str(csv), time_col="a", event_col="b")
        with pytest.raises(CLIDataError, match="Could not parse CSV"):
            _cli_load(args)

    def test_cli_import_error(self, capsys):
        """main() catches ImportError and returns 1."""
        from unittest.mock import patch
        from mlsurv.cli import main

        def raise_import(*a, **kw):
            raise ImportError("No module named 'missing_pkg'")

        with patch('mlsurv.cli.cmd_run', side_effect=raise_import):
            result = main(['run', 'dummy.csv'])
            assert result == 1
        captured = capsys.readouterr()
        assert "Missing dependency" in captured.err


# ---------------------------------------------------------------------------
# Training ImportError handling
# ---------------------------------------------------------------------------
class TestTrainingImportError:
    """train() catches ImportError for missing optional dependencies."""

    def test_train_catches_import_error(self, synthetic_data_clean):
        from unittest.mock import patch
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with patch.object(
            learner, '_train_model_internal',
            side_effect=ImportError("No module named 'pycox'"),
        ):
            with pytest.raises(RuntimeError, match="All models failed to train") as exc:
                learner._train_multiple_models(
                    ['coxph'], tune=False, n_trials=None,
                    experiment_name=None, search_spaces=None,
                )
        # The ImportError detail and its install hint are folded into the
        # all-failed RuntimeError (no separate per-model warning).
        msg = str(exc.value)
        assert "No module named 'pycox'" in msg
        assert "install the missing dependency" in msg


class TestRunModelComparisonImportError:
    """_run_model_comparison catches ImportError for missing dependencies."""

    def test_import_error_then_all_fail(self, synthetic_data_clean):
        from unittest.mock import patch
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with patch.object(
            learner, '_train_model_internal',
            side_effect=ImportError("No module named 'pycox'"),
        ):
            with pytest.raises(RuntimeError, match="All models failed to train") as exc:
                learner._run_model_comparison(
                    ['coxph'], tune=False, n_trials=None,
                    search_spaces=None, experiment_name=None,
                    optuna_verbose=False,
                )
        msg = str(exc.value)
        assert "No module named 'pycox'" in msg
        assert "install the missing dependency" in msg


class TestTuneAllModelsFailure:
    """_tune_all_models raises RuntimeError when all models fail."""

    def test_all_models_fail_to_tune(self, synthetic_data_clean):
        from unittest.mock import patch
        from mlsurv.learner import SurvivalLearner
        from mlsurv.learner._tuning import TuneConfig

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner.train('coxph', verbose=False)

        with patch.object(
            learner, '_tune_single_model',
            side_effect=RuntimeError("tuning failed"),
        ):
            with pytest.raises(RuntimeError) as exc:
                learner._tune_all_models(TuneConfig())
            msg = str(exc.value)
            assert "All models failed to tune" in msg
            # The underlying per-model cause is now folded into the aggregate.
            assert "tuning failed" in msg

    def test_import_error_raises_with_install_hint(self, synthetic_data_clean):
        """ImportError detail + install hint flow into the all-failed RuntimeError."""
        from unittest.mock import patch
        from mlsurv.learner import SurvivalLearner
        from mlsurv.learner._tuning import TuneConfig

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner.train('coxph', verbose=False)

        with patch.object(
            learner, '_tune_single_model',
            side_effect=ImportError("No module named 'optuna'"),
        ):
            with pytest.raises(RuntimeError, match="All models failed to tune") as exc:
                learner._tune_all_models(TuneConfig())
        msg = str(exc.value)
        assert "No module named 'optuna'" in msg
        assert "install the missing dependency" in msg


class TestPartialModelFailure:
    """When some models fail but others succeed, return successful results."""

    def test_train_partial_failure_returns_successes(self, synthetic_data_clean):
        import warnings
        from unittest.mock import patch
        from mlsurv.learner import SurvivalLearner
        from mlsurv._warnings import MethodologicalWarning

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        original = learner._train_model_internal

        def selective_fail(model_name, **kwargs):
            if model_name == 'rsf':
                raise ImportError("No module named 'fake_dep'")
            return original(model_name, **kwargs)

        with patch.object(learner, '_train_model_internal', side_effect=selective_fail):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = learner._train_multiple_models(
                    ['coxph', 'rsf'], tune=False, n_trials=None,
                    experiment_name=None, search_spaces=None,
                )
        assert 'coxph' in results
        assert 'rsf' not in results
        # One loud consolidated MethodologicalWarning names the dropped model and
        # its reason; the per-model "Missing dependency" SuggestionWarning is gone.
        drops = [x for x in w if issubclass(x.category, MethodologicalWarning)
                 and 'dropped' in str(x.message)]
        assert len(drops) == 1
        msg = str(drops[0].message)
        assert 'rsf' in msg
        assert 'ImportError' in msg
        assert 'install the missing dependency' in msg
        assert not any('Missing dependency' in str(x.message) for x in w)

    def test_run_model_comparison_partial_failure(self, synthetic_data_clean):
        import warnings
        from unittest.mock import patch
        from mlsurv.learner import SurvivalLearner
        from mlsurv._warnings import MethodologicalWarning

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        original = learner._train_model_internal

        def selective_fail(model_name, **kwargs):
            if model_name == 'rsf':
                raise ImportError("No module named 'fake_dep'")
            return original(model_name, **kwargs)

        with patch.object(learner, '_train_model_internal', side_effect=selective_fail):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = learner._run_model_comparison(
                    ['coxph', 'rsf'], tune=False, n_trials=None,
                    search_spaces=None, experiment_name=None,
                    optuna_verbose=False,
                )
        assert 'coxph' in result
        assert 'rsf' not in result
        drops = [x for x in w if issubclass(x.category, MethodologicalWarning)
                 and 'dropped' in str(x.message)]
        assert len(drops) == 1
        msg = str(drops[0].message)
        assert 'rsf' in msg
        assert '1 of 2 models were dropped' in msg
        assert not any('Missing dependency' in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# DeepSurv/DeepHit search-space model_params replay
# ---------------------------------------------------------------------------
class TestSearchSpaceModelParams:
    """best_params_ (search-space keys) replayed via model_params= is caught early."""

    def test_helper_raises_for_deepsurv(self):
        from mlsurv.learner._training import _check_search_space_model_params

        with pytest.raises(ValueError) as exc:
            _check_search_space_model_params(
                'deepsurv', {'n_layers': 2, 'n_nodes': 48, 'lr': 0.1, 'dropout': 0.3},
            )
        msg = str(exc.value)
        assert "search-space keys" in msg
        # All three offending keys are named; the passthrough key is not flagged.
        for key in ("n_layers", "n_nodes", "lr"):
            assert key in msg
        assert "layers=[n_nodes]*n_layers" in msg
        assert "rename lr -> learning_rate" in msg

    def test_helper_raises_for_deephit_without_lr_hint(self):
        from mlsurv.learner._training import _check_search_space_model_params

        with pytest.raises(ValueError) as exc:
            _check_search_space_model_params(
                'deephit', {'n_layers': 3, 'n_nodes': 64},
            )
        msg = str(exc.value)
        assert "layers=[n_nodes]*n_layers" in msg
        # DeepHit samples learning_rate directly, so the lr-rename hint is absent.
        assert "rename lr" not in msg

    def test_helper_noop_for_non_deep_model(self):
        from mlsurv.learner._training import _check_search_space_model_params

        # n_layers on a non-deep model is not a search-space key we own; pass through.
        assert _check_search_space_model_params(
            'coxph', {'n_layers': 2}) is None

    def test_helper_noop_for_translated_params(self):
        from mlsurv.learner._training import _check_search_space_model_params

        # Already-translated constructor form must pass through untouched.
        assert _check_search_space_model_params(
            'deepsurv', {'layers': [48, 48], 'learning_rate': 0.1}) is None

    def test_train_multiple_models_surfaces_detail(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with pytest.raises(RuntimeError) as exc:
            learner._train_multiple_models(
                ['deepsurv'], tune=False, n_trials=None,
                experiment_name=None, search_spaces=None,
                model_params={'deepsurv': {'n_layers': 2, 'n_nodes': 48, 'lr': 0.1}},
            )
        msg = str(exc.value)
        # Back-compat prefix preserved AND the underlying cause is now inline.
        assert "All models failed to train" in msg
        assert "deepsurv" in msg
        assert "search-space" in msg

    def test_run_model_comparison_surfaces_detail(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with pytest.raises(RuntimeError) as exc:
            learner._run_model_comparison(
                ['deephit'], tune=False, n_trials=None,
                search_spaces=None, experiment_name=None, optuna_verbose=False,
                model_params={'deephit': {'n_layers': 3, 'n_nodes': 64}},
            )
        msg = str(exc.value)
        assert "All models failed to train" in msg
        assert "deephit" in msg
        assert "search-space" in msg

    def test_partial_failure_keeps_good_model(self, synthetic_data_clean):
        import warnings
        from mlsurv.learner import SurvivalLearner
        from mlsurv._warnings import MethodologicalWarning

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = learner._train_multiple_models(
                ['coxph', 'deepsurv'], tune=False, n_trials=None,
                experiment_name=None, search_spaces=None,
                model_params={'deepsurv': {'n_layers': 2, 'n_nodes': 48, 'lr': 0.1}},
            )
        assert 'coxph' in results
        assert 'deepsurv' not in results
        # The dropped-model summary folds in the caught ValueError's detail.
        drops = [x for x in w if issubclass(x.category, MethodologicalWarning)
                 and 'dropped' in str(x.message)]
        assert len(drops) == 1
        assert 'search-space' in str(drops[0].message)

    def test_multiple_failures_list_every_model(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with pytest.raises(RuntimeError) as exc:
            learner._train_multiple_models(
                ['deepsurv', 'deephit'], tune=False, n_trials=None,
                experiment_name=None, search_spaces=None,
                model_params={
                    'deepsurv': {'lr': 0.1},
                    'deephit': {'n_layers': 3},
                },
            )
        msg = str(exc.value)
        # Both models get their own detail line in the joined aggregate message.
        assert "  - deepsurv:" in msg
        assert "  - deephit:" in msg


class TestRaiseIfAllFailed:
    """_raise_if_all_failed folds per-model detail in, and keeps the legacy fallback."""

    def test_errors_joined_into_message(self):
        from mlsurv.learner._cv_training import _raise_if_all_failed

        with pytest.raises(RuntimeError) as exc:
            _raise_if_all_failed(
                {}, ['coxph', 'rsf'], 'train',
                errors={'coxph': 'ValueError: bad', 'rsf': 'ImportError: nope'},
            )
        msg = str(exc.value)
        assert "All models failed to train: coxph, rsf." in msg
        assert "  - coxph: ValueError: bad" in msg
        assert "  - rsf: ImportError: nope" in msg
        assert "Check warnings above" not in msg

    def test_empty_errors_falls_back_to_legacy_message(self):
        from mlsurv.learner._cv_training import _raise_if_all_failed

        with pytest.raises(RuntimeError, match="Check warnings above for details"):
            _raise_if_all_failed({}, ['coxph', 'rsf'], 'train', errors={})

    def test_no_raise_when_some_succeeded(self):
        from mlsurv.learner._cv_training import _raise_if_all_failed

        # Non-empty results means at least one model worked: must not raise.
        _raise_if_all_failed({'coxph': object()}, ['coxph', 'rsf'], 'train',
                             errors={'rsf': 'ValueError: bad'})


class TestWarnDroppedModels:
    """_warn_dropped_models emits one loud MethodologicalWarning on partial failure."""

    def test_emits_methodological_warning(self):
        import warnings
        from mlsurv.learner._cv_training import _warn_dropped_models
        from mlsurv._warnings import MethodologicalWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_dropped_models(
                {'rsf': 'ImportError: nope', 'gbsa': 'ValueError: bad'},
                3, 'train',
            )
        assert len(w) == 1
        assert issubclass(w[0].category, MethodologicalWarning)
        msg = str(w[0].message)
        assert '2 of 3 models were dropped after failing to train' in msg
        assert 'rsf' in msg and 'gbsa' in msg
        assert '1 remaining model(s)' in msg

    def test_empty_errors_is_noop(self):
        import warnings
        from mlsurv.learner._cv_training import _warn_dropped_models

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_dropped_models({}, 3, 'train')
        assert len(w) == 0

    def test_tune_partial_failure_warns(self, synthetic_data_clean):
        """_tune_all_models warns loudly when one model fails but another tunes."""
        import warnings
        from types import SimpleNamespace
        from unittest.mock import patch
        from mlsurv.learner import SurvivalLearner
        from mlsurv.learner._tuning import TuneConfig
        from mlsurv._warnings import MethodologicalWarning

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        # Seed two untuned models directly (no real training needed).
        learner._state.models['coxph'] = SimpleNamespace(is_tuned=False, name='coxph')
        learner._state.models['rsf'] = SimpleNamespace(is_tuned=False, name='rsf')

        def selective(model_name, *a, **k):
            if model_name == 'rsf':
                raise RuntimeError("optuna blew up")
            return SimpleNamespace(is_tuned=True, name=model_name)

        with patch.object(learner, '_tune_single_model', side_effect=selective):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = learner._tune_all_models(TuneConfig())
        assert 'coxph' in results
        assert 'rsf' not in results
        drops = [x for x in w if issubclass(x.category, MethodologicalWarning)
                 and 'dropped' in str(x.message)]
        assert len(drops) == 1
        msg = str(drops[0].message)
        assert 'rsf' in msg
        assert 'optuna blew up' in msg
        assert '1 of 2 models were dropped after failing to tune' in msg
        # n_total is len(trainable_models); coxph tuned, so 1 remains.
        assert '1 remaining model(s)' in msg


# ---------------------------------------------------------------------------
# Tune parameter validation
# ---------------------------------------------------------------------------
class TestTuneParameterValidation:
    """tune() validates n_trials and timeout."""

    def test_negative_n_trials(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with pytest.raises(ValueError, match="n_trials must be a non-negative integer"):
            learner.tune('coxph', n_trials=-5)

    def test_zero_n_trials_without_resume(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        # n_trials=0 is valid now (requires resume=True for models with
        # searchable params). CoxPH has no searchable params so tuning is
        # skipped — returns baseline, not marked tuned.
        result = learner.tune('coxph', n_trials=0, verbose=False)
        assert not result.is_tuned

    def test_float_n_trials(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with pytest.raises(ValueError, match="n_trials must be a non-negative integer"):
            learner.tune('coxph', n_trials=5.5)

    def test_negative_timeout(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with pytest.raises(ValueError, match="timeout must be a positive number"):
            learner.tune('coxph', timeout=-10)

    def test_zero_timeout(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with pytest.raises(ValueError, match="timeout must be a positive number"):
            learner.tune('coxph', timeout=0)


# ---------------------------------------------------------------------------
# Unfitted model detection
# ---------------------------------------------------------------------------
class TestUnfittedModelErrors:
    """evaluate() wraps NotFittedError into a friendly message."""

    def test_unfitted_model_in_evaluate(self, synthetic_data_clean):
        from sklearn.exceptions import NotFittedError
        from mlsurv.learner import SurvivalLearner
        from mlsurv.models.coxph import CoxPHModel

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        unfitted = CoxPHModel()
        with pytest.raises(NotFittedError, match="Model not fitted.*train\\(\\)"):
            learner.evaluate(unfitted)


# ---------------------------------------------------------------------------
# Hand-built y: time-value and event-encoding validation
# ---------------------------------------------------------------------------
def _build_y(events, times, event_dtype=bool):
    """Build a canonical SurvivalArray directly, bypassing to_structured_array.

    Mirrors the convention shown in the project's API reference, which a user
    can follow without ever calling to_structured_array.
    """
    arr = np.empty(len(times), dtype=[('event', event_dtype), ('time', float)])
    arr['event'] = events
    arr['time'] = times
    return arr


def _valid_X(n):
    return pd.DataFrame({'a': np.arange(n, dtype=float), 'b': np.ones(n)})


class TestHandBuiltTimeValues:
    """validate_survival_data checks y['time'] values (hand-built y bypasses
    to_structured_array, so the validator is the only gate)."""

    def _ok_events_times(self, n=12):
        events = np.zeros(n, dtype=bool)
        events[:6] = True  # 6 events clears the min_events=5 check
        times = np.arange(1.0, n + 1.0)  # all strictly positive
        return events, times

    def test_nan_time(self):
        from mlsurv.utils import validate_survival_data

        events, times = self._ok_events_times()
        times[3] = np.nan
        y = _build_y(events, times)
        with pytest.raises(ValueError, match="NaN"):
            validate_survival_data(_valid_X(len(y)), y)

    def test_inf_time(self):
        from mlsurv.utils import validate_survival_data

        events, times = self._ok_events_times()
        times[3] = np.inf
        y = _build_y(events, times)
        with pytest.raises(ValueError, match="infinite"):
            validate_survival_data(_valid_X(len(y)), y)

    def test_negative_time(self):
        from mlsurv.utils import validate_survival_data

        events, times = self._ok_events_times()
        times[3] = -2.0
        y = _build_y(events, times)
        with pytest.raises(ValueError, match="negative survival time"):
            validate_survival_data(_valid_X(len(y)), y)

    def test_zero_time(self):
        from mlsurv.utils import validate_survival_data

        events, times = self._ok_events_times()
        times[3] = 0.0
        y = _build_y(events, times)
        with pytest.raises(ValueError, match="time=0"):
            validate_survival_data(_valid_X(len(y)), y)

    def test_zero_and_negative_messages_distinct(self):
        """Zero and negative times raise different, message-specific errors."""
        from mlsurv.utils import validate_survival_data

        events, times = self._ok_events_times()
        neg = times.copy()
        neg[3] = -1.0
        with pytest.raises(ValueError, match="negative survival time"):
            validate_survival_data(_valid_X(len(neg)), _build_y(events, neg))

        zero = times.copy()
        zero[3] = 0.0
        with pytest.raises(ValueError, match="time=0"):
            validate_survival_data(_valid_X(len(zero)), _build_y(events, zero))

    def test_negative_and_zero_both_present_mentions_zero(self):
        """Negative takes priority but the message also mentions the zeros —
        verified through the hand-built-y validator path, not just
        to_structured_array."""
        from mlsurv.utils import validate_survival_data

        events, times = self._ok_events_times()
        times[3] = -1.0
        times[7] = 0.0
        y = _build_y(events, times)
        with pytest.raises(ValueError, match="Additionally.*time=0"):
            validate_survival_data(_valid_X(len(y)), y)

    def test_clean_positive_times_pass(self):
        from mlsurv.utils import validate_survival_data

        events, times = self._ok_events_times()
        y = _build_y(events, times)
        validate_survival_data(_valid_X(len(y)), y)  # should not raise


class TestHandBuiltEventEncoding:
    """validate_survival_data checks the event encoding without requiring bool
    dtype (0/1-int and 0.0/1.0-float encodings stay valid)."""

    def test_nan_float_event(self):
        from mlsurv.utils import validate_survival_data

        events = np.array([1.0, 0.0, np.nan] + [1.0] * 9)
        times = np.arange(1.0, 13.0)
        y = _build_y(events, times, event_dtype=float)
        with pytest.raises(ValueError, match="Event array contains NaN"):
            validate_survival_data(_valid_X(len(y)), y, min_events=0)

    def test_ambiguous_binary_event_codes(self):
        from mlsurv.utils import validate_survival_data

        events = np.array([1, 2] * 6)  # two non-zero codes: ambiguous
        times = np.arange(1.0, 13.0)
        y = _build_y(events, times, event_dtype=int)
        with pytest.raises(ValueError, match="ambiguous binary numeric"):
            validate_survival_data(_valid_X(len(y)), y, min_events=0)

    def test_competing_risks_warns(self):
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import validate_survival_data

        events = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        times = np.arange(1.0, 13.0)
        y = _build_y(events, times, event_dtype=int)
        with pytest.warns(MethodologicalWarning, match="competing risks"):
            validate_survival_data(_valid_X(len(y)), y, min_events=0)

    def test_int_event_encoding_passes(self):
        from mlsurv.utils import validate_survival_data

        events = np.array([0, 1] * 6)
        times = np.arange(1.0, 13.0)
        y = _build_y(events, times, event_dtype=int)
        validate_survival_data(_valid_X(len(y)), y)  # should not raise

    def test_float_event_encoding_passes(self):
        from mlsurv.utils import validate_survival_data

        events = np.array([0.0, 1.0] * 6)
        times = np.arange(1.0, 13.0)
        y = _build_y(events, times, event_dtype=float)
        validate_survival_data(_valid_X(len(y)), y)  # should not raise


class TestHandBuiltYViaSetup:
    """The same checks fire through the real setup() path, not just the
    standalone validator."""

    def test_setup_rejects_zero_time(self):
        from mlsurv.learner import SurvivalLearner

        n = 30
        events = np.zeros(n, dtype=bool)
        events[:15] = True
        times = np.arange(1.0, n + 1.0)
        times[5] = 0.0
        y = _build_y(events, times)
        X = pd.DataFrame({
            'a': np.arange(n, dtype=float),
            'b': np.linspace(0.0, 1.0, n),
        })
        with pytest.raises(ValueError, match="time=0"):
            SurvivalLearner(X, y).setup(verbose=False)


# ---------------------------------------------------------------------------
# Features with no observed value
# ---------------------------------------------------------------------------
class TestEmptyFeatureColumns:
    """A 100%-missing feature is named, because nothing downstream uses it."""

    @staticmethod
    def _data(empty_names=("dz_stage",)):
        n = 30
        events = np.zeros(n, dtype=bool)
        events[:15] = True
        y = _build_y(events, np.arange(1.0, n + 1.0))
        X = pd.DataFrame({
            'age': np.linspace(40.0, 80.0, n),
            'marker': np.linspace(0.0, 1.0, n),
        })
        for name in empty_names:
            X[name] = np.nan
        return X, y

    def test_warns_naming_the_feature(self):
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import validate_survival_data

        X, y = self._data()
        with pytest.warns(MethodologicalWarning, match="dz_stage"):
            validate_survival_data(X, y)

    def test_warns_even_when_missing_warnings_are_off(self):
        """setup() passes warn_missing=False whenever an imputer is configured —
        which is exactly when the column vanishes without a trace."""
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import validate_survival_data

        X, y = self._data()
        with pytest.warns(MethodologicalWarning, match="no observed values"):
            validate_survival_data(X, y, warn_missing=False)

    def test_names_at_most_five(self):
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import validate_survival_data

        X, y = self._data(tuple(f"empty_{i}" for i in range(7)))
        with pytest.warns(MethodologicalWarning, match=r"and 2 more") as rec:
            validate_survival_data(X, y)
        assert "7 features" in str(rec[0].message)

    def test_silent_when_every_feature_has_data(self):
        import warnings as _warnings

        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import validate_survival_data

        X, y = self._data(empty_names=())
        X.loc[X.index[:5], 'age'] = np.nan  # partly missing is not empty
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            validate_survival_data(X, y)
        assert not [x for x in w
                    if issubclass(x.category, MethodologicalWarning)
                    and "no observed values" in str(x.message)]

    def test_warning_points_at_the_caller(self):
        """stacklevel must resolve to this test file, not utils.py."""
        import warnings as _warnings

        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import validate_survival_data

        X, y = self._data()
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            validate_survival_data(X, y)
        empty = [x for x in w
                 if issubclass(x.category, MethodologicalWarning)
                 and "no observed values" in str(x.message)]
        assert len(empty) == 1
        assert empty[0].filename.endswith("test_friendly_errors.py")

    def test_empty_feature_columns_helper(self):
        from mlsurv.utils import empty_feature_columns

        X, _ = self._data()
        assert empty_feature_columns(X) == ['dz_stage']
        assert empty_feature_columns(None) == []
        assert empty_feature_columns(pd.DataFrame(index=range(3))) == []

    def test_test_split_gets_its_own_wording(self):
        """A held-out cohort is not the fitting cohort, so the training-only
        consequence would be false there."""
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import validate_survival_data

        X, y = self._data()
        with pytest.warns(MethodologicalWarning) as rec:
            validate_survival_data(
                X, y, min_samples=0, min_events=0,
                label="X_test", fit_cohort=False,
            )
        msg = str(rec[0].message)
        assert "X_test" in msg
        assert "come entirely from the fit on the training data" in msg
        assert "dropped before model fitting" not in msg

    def test_presplit_setup_labels_each_frame(self):
        """The presplit path validates both splits; a reader has to be able
        to tell which one the warning is about."""
        import warnings as _warnings

        from mlsurv import SurvivalLearner
        from mlsurv._warnings import MethodologicalWarning

        X, y = self._data()
        n = len(X)
        cut = n // 2
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            SurvivalLearner(
                X_train=X.iloc[:cut], y_train=y[:cut],
                X_test=X.iloc[cut:], y_test=y[cut:],
            )
        msgs = [str(x.message) for x in w
                if issubclass(x.category, MethodologicalWarning)
                and "no observed values" in str(x.message)]
        assert any("X_train" in m for m in msgs)
        assert any("X_test" in m for m in msgs)
