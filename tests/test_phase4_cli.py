"""CLI contract and regression tests."""

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest


def _write_sample_csv(path: Path) -> Path:
    """Write a minimal survival CSV fixture."""
    df = pd.DataFrame(
        {
            'f1': [0.1, 0.2, 0.3, 0.4],
            'f2': [1.0, 0.0, 1.0, 0.0],
            'time': [10, 12, 8, 15],
            'event': [1, 0, 1, 0],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


class _FakeLearnerBase:
    """Base for inline FakeLearner stubs used by CLI tests.

    Provides a lazily-created ``_state`` backed by the real
    :class:`~mlsurv.learner._state.LearnerState` dataclass so that CLI
    helpers like ``_setup_learner`` and ``_save_artifact_if_requested``
    can access ``learner._state`` without crashing on test fakes.

    Public read-only properties that the real ``SurvivalLearner`` exposes
    as simple passthroughs to a ``_state`` field (``models``,
    ``benchmark_result``) are mirrored here, so CLI helpers that read them
    the public way work against the fake exactly as against
    a real learner.
    """

    # Public SurvivalLearner properties that just return a _state field and
    # are read by CLI helpers exercised against this fake. Kept minimal — add
    # a name here only when a fake-based CLI test actually reads it.
    _STATE_PASSTHROUGH = frozenset({'models', 'benchmark_result'})

    def __getattr__(self, name):
        if name == '_state':
            from mlsurv.learner._state import LearnerState
            self._state = LearnerState()
            return self._state
        if name in self._STATE_PASSTHROUGH:
            return getattr(self._state, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# Sentinel meaning "no behavior configured" — distinct from a configured ``None``
# so the factory can tell "don't attach this property" from "property returns None".
_UNSET = object()


class _Obj:
    """Flat attribute bag for ``FakeResult`` / ``FakeEvalResult`` / ``FakeBootstrapResult``
    shapes: ``_Obj(cv_metric_mean=0.7, cv_metric_std=0.05)`` builds an object whose
    attributes are the given keywords. Use small inline classes instead when the fake
    result needs a *method* (e.g. ``FakeBootstrapResult.compare``).
    """

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


@pytest.fixture
def make_fake_learner():
    """Factory for CLI ``SurvivalLearner`` stubs that capture forwarded call args.

    Replaces the per-test inline ``class FakeLearner(_FakeLearnerBase)`` stubs. A
    method is attached **only** when its ``*_return`` kwarg is supplied, so tests
    asserting the CLI must *not* call a method (e.g. must-not-touch ``best_model``)
    keep raising ``AttributeError``. Captured args land in the caller-owned ``seen``
    dict under the same keys the inline stubs used (``setup_kwargs``, ``train_model``,
    ``n_bootstrap``, ...), so existing assertions stay byte-for-byte identical.
    Named params land under their own key, NOT in the catch-all ``*_kwargs`` dict:
    ``evaluate``'s ``n_importance_repeats`` is captured as ``seen['n_importance_repeats']``,
    not ``seen['eval_kwargs']['n_importance_repeats']``.

    Returns the built ``FakeLearner`` class; instantiate it via the CLI by patching
    ``mlsurv.SurvivalLearner``.
    """

    def _build(seen, *, train_return=None, run_return=None, tune_return=None,
               evaluate_return=None, bootstrap_return=None, validate_return=None,
               best_model=_UNSET, evaluation_result=_UNSET, bootstrap_result=_UNSET,
               scoring=_UNSET, report_body='<html>ok</html>', train_records_n=False):
        ns = {}

        def __init__(self, X, y):
            if train_records_n:
                seen['train_n'] = len(X)

        ns['__init__'] = __init__
        ns['setup'] = lambda self, **kwargs: seen.__setitem__('setup_kwargs', kwargs)

        if train_return is not None:
            def train(self, model_name, **kwargs):
                seen['train_model'] = model_name
                return train_return
            ns['train'] = train

        if run_return is not None:
            def run(self, **kwargs):
                seen['run_kwargs'] = kwargs
                return run_return
            ns['run'] = run

        if tune_return is not None:
            def tune(self, model, n_trials=50, **kwargs):
                seen['tune_model'] = model
                seen['n_trials'] = n_trials
                seen['tune_kwargs'] = kwargs
                return tune_return
            ns['tune'] = tune

        if evaluate_return is not None:
            def evaluate(self, model_result, n_importance_repeats=100, **kwargs):
                seen['evaluate_model_result'] = model_result
                seen['n_importance_repeats'] = n_importance_repeats
                seen['eval_kwargs'] = kwargs
                return evaluate_return
            ns['evaluate'] = evaluate

        if bootstrap_return is not None:
            def bootstrap(self, model, n_bootstrap='auto', **kwargs):
                seen['bootstrap_model'] = model
                seen['n_bootstrap'] = n_bootstrap
                seen['bootstrap_kwargs'] = kwargs
                return bootstrap_return
            ns['bootstrap'] = bootstrap

        if validate_return is not None:
            def validate(self, X_val, y_val, models=None, name='External', **kwargs):
                seen['val_n'] = len(X_val)
                seen['validate_model'] = models
                seen['cohort_name'] = name
                seen['validate_kwargs'] = kwargs
                return validate_return
            ns['validate'] = validate

        def generate_report(self, output_path, eval_result=None,
                            bootstrap_results=None, **kwargs):
            seen['report_eval_result'] = eval_result
            seen['report_bootstrap_results'] = bootstrap_results
            seen['report_kwargs'] = kwargs
            Path(output_path).write_text(report_body)
        ns['generate_report'] = generate_report

        if best_model is not _UNSET:
            ns['best_model'] = property(lambda self: best_model)
        if evaluation_result is not _UNSET:
            ns['evaluation_result'] = property(lambda self: evaluation_result)
        if bootstrap_result is not _UNSET:
            ns['bootstrap_result'] = property(lambda self: bootstrap_result)
        if scoring is not _UNSET:
            ns['scoring'] = scoring

        return type('FakeLearner', (_FakeLearnerBase,), ns)

    return _build


class TestCLIParser:
    """Parser-level CLI tests."""

    def test_parser_creation(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        assert parser is not None

    def test_parser_version_from_package(self, capsys):
        """``--version`` reads from mlsurv.__version__, not a hardcoded string."""
        import mlsurv
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--version'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr().out
        assert mlsurv.__version__ in captured

    def test_parser_has_expected_subcommands(self):
        from mlsurv.cli import create_parser

        parser = create_parser()

        assert parser.parse_args(['run', 'test.csv']).command == 'run'
        assert parser.parse_args(['train', 'test.csv', '--model', 'coxph']).command == 'train'
        assert parser.parse_args(['tune', 'test.csv', '--model', 'rsf']).command == 'tune'
        assert parser.parse_args(['evaluate', 'test.csv', '--model', 'coxph']).command == 'evaluate'
        assert parser.parse_args(['bootstrap', 'test.csv', '--model', 'coxph']).command == 'bootstrap'
        assert parser.parse_args([
            'validate', 'train.csv', '--validation-data', 'external.csv', '--model', 'coxph'
        ]).command == 'validate'
        assert parser.parse_args([
            'predict', 'test.csv', '--load-artifact', '/tmp/artifact', '--times', '12',
        ]).command == 'predict'
        assert parser.parse_args([
            'benchmark', 'test.csv', '--reference-file', 'refs.csv',
        ]).command == 'benchmark'
        assert parser.parse_args([
            'sweep', 'test.csv', '--models', 'rsf',
            '--param', 'k', '--values', '5',
        ]).command == 'sweep'
        assert parser.parse_args(['setup', 'test.csv']).command == 'setup'
        assert parser.parse_args(['models']).command == 'models'
        assert parser.parse_args(['artifact-inspect', '/tmp/art']).command == 'artifact-inspect'
        assert parser.parse_args(['artifact-upgrade', '/tmp/art']).command == 'artifact-upgrade'
        assert parser.parse_args([
            'explain', '--load-learner', 'foo.learner', '--patient', '0',
        ]).command == 'explain'

    def test_parser_run_defaults(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['run', 'test.csv'])

        assert args.models == ['coxph', 'rsf']
        assert args.cv_folds == 5
        assert args.n_bootstrap == 'auto'
        assert args.compare is False
        assert args.compare_output is None
        assert args.tune is True
        assert args.n_trials == 'auto'

    def test_parser_run_no_tune_and_n_trials(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['run', 'test.csv', '--no-tune', '--n-trials', '7'])
        assert args.tune is False
        assert args.n_trials == 7

    @pytest.mark.parametrize('value,expected', [('1', 1), ('50', 50), ('auto', 'auto')])
    def test_parser_run_n_trials_valid(self, value, expected):
        # '1' is the minimum meaningful int; 'auto' is the default sentinel.
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['run', 'test.csv', '--n-trials', value])
        assert args.n_trials == expected

    @pytest.mark.parametrize('bad', ['0', '-1', 'notanumber'])
    def test_parser_run_n_trials_invalid(self, bad):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['run', 'test.csv', '--n-trials', bad])

    @pytest.mark.parametrize('value', ['auto', '5'])
    def test_validate_args_accepts_run_n_trials(self, value):
        """_validate_args tolerates 'auto' as well as ints for run --n-trials."""
        from mlsurv.cli import create_parser
        from mlsurv.cli._data import _validate_args

        parser = create_parser()
        args = parser.parse_args(['run', 'test.csv', '--n-trials', value])
        _validate_args(parser, args)  # must not raise

    def test_parser_run_compare_flags(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'run', 'test.csv', '--compare', '--compare-output', 'comp.csv',
        ])
        assert args.compare is True
        assert args.compare_output == 'comp.csv'

    def test_parser_train_requires_model(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['train', 'test.csv'])

    def test_parser_tune_requires_model(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['tune', 'test.csv'])

    def test_parser_validate_requires_validation_data(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['validate', 'train.csv', '--model', 'coxph'])

    def test_parser_predict_no_source_parses_ok(self):
        """Argparse no longer requires --load-artifact; validation catches it."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['predict', 'test.csv'])
        assert args.load_artifact is None
        assert getattr(args, 'load_learner', None) is None

    def test_parser_predict_times_optional(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['predict', 'test.csv', '--load-artifact', '/tmp/artifact'])
        assert args.load_artifact == '/tmp/artifact'
        assert args.times is None

    def test_parser_predict_strict_default_false(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['predict', 'test.csv', '--load-artifact', '/tmp/a'])
        assert args.strict is False

    def test_parser_validate_accepts_val_event_values(self):
        """validate parser should accept a separate validation event mapping."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'validate', 'train.csv',
            '--validation-data', 'external.csv',
            '--model', 'coxph',
            '--val-event-values', '2',
        ])
        assert args.val_event_values == ['2']

    def test_parser_validate_accepts_validation_date_cols(self):
        """validate parser should accept dedicated validation date columns."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'validate', 'train.csv',
            '--validation-data', 'external.csv',
            '--model', 'coxph',
            '--val-start-col', 'dx_date',
            '--val-end-col', 'followup_date',
            '--val-time-unit', 'months',
        ])
        assert args.val_start_col == 'dx_date'
        assert args.val_end_col == 'followup_date'
        assert args.val_time_unit == 'months'

    def test_parser_date_cols_parsed(self):
        """--start-col and --end-col are parsed correctly."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'run', 'test.csv',
            '--start-col', 'diag_date',
            '--end-col', 'followup_date',
            '--time-unit', 'months',
        ])
        assert args.start_col == 'diag_date'
        assert args.end_col == 'followup_date'
        assert args.time_unit == 'months'

    def test_parser_date_cols_defaults(self):
        """Date-column args default to None / 'days'."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['run', 'test.csv'])
        assert args.start_col is None
        assert args.end_col is None
        assert args.time_unit == 'days'

    def test_start_col_without_end_col_errors(self):
        """--start-col without --end-col fails cross-arg validation."""
        from mlsurv.cli import create_parser
        from mlsurv.cli._data import _validate_args

        parser = create_parser()
        args = parser.parse_args([
            'run', 'test.csv', '--start-col', 'diag_date',
        ])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)

    def test_date_cols_and_time_col_mutually_exclusive(self):
        """--start-col/--end-col with explicit --time-col fails validation."""
        from mlsurv.cli import create_parser
        from mlsurv.cli._data import _validate_args

        parser = create_parser()
        args = parser.parse_args([
            'run', 'test.csv',
            '--start-col', 'diag', '--end-col', 'followup',
            '--time-col', 'duration',
        ])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)

    def test_val_start_col_without_end_col_errors(self):
        """--val-start-col without --val-end-col fails cross-arg validation."""
        from mlsurv.cli import create_parser
        from mlsurv.cli._data import _validate_args

        parser = create_parser()
        args = parser.parse_args([
            'validate', 'train.csv',
            '--validation-data', 'external.csv',
            '--model', 'coxph',
            '--val-start-col', 'diag_date',
        ])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)

    def test_validation_date_cols_and_val_time_col_mutually_exclusive(self):
        """Validation date columns with explicit --val-time-col should fail validation."""
        from mlsurv.cli import create_parser
        from mlsurv.cli._data import _validate_args

        parser = create_parser()
        args = parser.parse_args([
            'validate', 'train.csv',
            '--validation-data', 'external.csv',
            '--model', 'coxph',
            '--val-start-col', 'diag', '--val-end-col', 'followup',
            '--val-time-col', 'duration',
        ])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)

    def test_parser_benchmark_requires_reference_file(self):
        """benchmark subcommand requires --reference-file."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['benchmark', 'test.csv'])

    def test_parser_benchmark_reference_cols_optional(self):
        """--reference-cols is optional, defaults to None."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'benchmark', 'test.csv', '--reference-file', 'refs.csv',
        ])
        assert args.reference_file == 'refs.csv'
        assert args.reference_cols is None

        args2 = parser.parse_args([
            'benchmark', 'test.csv', '--reference-file', 'refs.csv',
            '--reference-cols', 'score_a', 'score_b',
        ])
        assert args2.reference_cols == ['score_a', 'score_b']

    def test_parser_benchmark_defaults(self):
        """benchmark parser has expected defaults."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'benchmark', 'test.csv', '--reference-file', 'refs.csv',
        ])
        assert args.models == ['coxph', 'rsf']
        assert args.cv_folds == 5
        assert args.n_bootstrap == 200
        assert args.bootstrap is True


class TestCLIParserCvCI:
    """Test --cv-ci CLI argument."""

    def test_parser_cv_ci_default(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['train', 'test.csv', '--model', 'coxph'])
        assert args.cv_ci == 0.95

    def test_parser_cv_ci_custom(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'train', 'test.csv', '--model', 'coxph', '--cv-ci', '0.90'
        ])
        assert args.cv_ci == pytest.approx(0.90)

    def test_cv_ci_validation_rejects_out_of_range(self):
        from mlsurv.cli import create_parser, _validate_args

        parser = create_parser()
        args = parser.parse_args([
            'train', 'test.csv', '--model', 'coxph', '--cv-ci', '0.0'
        ])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)

        args = parser.parse_args([
            'train', 'test.csv', '--model', 'coxph', '--cv-ci', '1.0'
        ])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)

    def test_cv_ci_validation_rejects_negative(self):
        from mlsurv.cli import create_parser, _validate_args

        parser = create_parser()
        args = parser.parse_args([
            'train', 'test.csv', '--model', 'coxph', '--cv-ci', '-0.5'
        ])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)

    def test_cv_ci_setup_learner_passthrough(self, synthetic_data_clean):
        """_setup_learner should pass cv_ci through to the learner."""
        from types import SimpleNamespace
        from mlsurv import SurvivalLearner
        from mlsurv.cli import _setup_learner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        args = SimpleNamespace(
            test_fraction=0.2,
            random_state=42,
            cv_ci=0.90,
            cv_folds=None,
            cv_repeats=1,
            cv_strategy='stratifiedkfold',
            imputer='median',
            scaler='standard',
            feature_selector='none',
            scoring=None,
            n_jobs=1,
        )
        _setup_learner(learner, args)
        assert learner.cv_ci == 0.90

    def test_setup_learner_stores_col_names_in_state(self, synthetic_data_clean):
        """_setup_learner should persist time_col/event_col in learner state."""
        from types import SimpleNamespace
        from mlsurv import SurvivalLearner
        from mlsurv.cli import _setup_learner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        args = SimpleNamespace(
            test_fraction=0.2,
            random_state=42,
            cv_folds=None,
            cv_repeats=1,
            cv_strategy='stratifiedkfold',
            imputer='median',
            scaler='standard',
            feature_selector='none',
            scoring=None,
            n_jobs=1,
            time_col='OS_MONTHS',
            event_col='OS_STATUS',
        )
        _setup_learner(learner, args)
        assert learner._state.time_col == 'OS_MONTHS'
        assert learner._state.event_col == 'OS_STATUS'


class TestCLIAPIContracts:
    """Contract tests for CLI assumptions about API objects."""

    def test_survival_learner_signatures_support_cli_usage(self):
        from mlsurv import SurvivalLearner

        required = {
            'setup': {'test_fraction', 'cv_folds', 'cv_ci', 'random_state'},
            'run': {'models', 'verbose', 'evaluate', 'bootstrap', 'n_bootstrap'},
            'train': {'models'},
            'tune': {'models', 'n_trials', 'verbose'},
            'evaluate': {'n_importance_repeats'},
            'bootstrap': {'n_bootstrap'},
            'validate': {'X_val', 'y_val', 'models', 'name', 'n_bootstrap'},
        }

        for method_name, required_params in required.items():
            signature = inspect.signature(getattr(SurvivalLearner, method_name))
            actual_params = set(signature.parameters)
            missing = required_params - actual_params
            assert not missing, (
                f"SurvivalLearner.{method_name}{signature} is missing CLI-required "
                f"parameters: {sorted(missing)}"
            )


@pytest.mark.integration
class TestCLICommands:
    """Integration-style CLI command execution tests."""

    @pytest.fixture
    def sample_data_file(self, tmp_path, synthetic_data_clean):
        X, y = synthetic_data_clean
        df = X.copy()
        df['time'] = y['time']
        df['event'] = y['event'].astype(int)
        filepath = tmp_path / 'data.csv'
        df.to_csv(filepath, index=False)
        return filepath

    def test_main_no_command(self):
        from mlsurv.cli import main

        assert main([]) == 0

    def test_main_run(self, sample_data_file):
        from mlsurv.cli import main

        rc = main([
            'run',
            str(sample_data_file),
            '--models', 'coxph',
            '--cv-folds', '3',
            '--n-bootstrap', '10',
        ])
        assert rc == 0

    def test_main_train(self, sample_data_file):
        from mlsurv.cli import main

        rc = main([
            'train',
            str(sample_data_file),
            '--model', 'coxph',
            '--cv-folds', '3',
        ])
        assert rc == 0

    def test_main_evaluate(self, sample_data_file):
        from mlsurv.cli import main

        rc = main([
            'evaluate',
            str(sample_data_file),
            '--model', 'coxph',
            '--cv-folds', '3',
        ])
        assert rc == 0

    def test_main_models(self):
        from mlsurv.cli import main

        assert main(['models']) == 0

    def test_main_setup(self, sample_data_file):
        from mlsurv.cli import main

        rc = main([
            'setup',
            str(sample_data_file),
            '--cv-folds', '3',
        ])
        assert rc == 0


class TestCLIDataLoading:
    """Data loading utility tests."""

    def test_load_data(self, tmp_path, synthetic_data_clean):
        from argparse import Namespace
        from mlsurv.cli._data import _cli_load

        X, y = synthetic_data_clean
        df = X.copy()
        df['time'] = y['time']
        df['event'] = y['event'].astype(int)

        filepath = tmp_path / 'data.csv'
        df.to_csv(filepath, index=False)

        args = Namespace(data=str(filepath), time_col='time', event_col='event')
        X_loaded, y_loaded = _cli_load(args)

        assert len(X_loaded) == len(X)
        assert 'time' not in X_loaded.columns
        assert 'event' not in X_loaded.columns
        assert y_loaded.dtype.names == ('event', 'time')

    def test_load_data_date_mode(self, tmp_path):
        """_cli_load dispatches to date-based path when start_col/end_col present."""
        from argparse import Namespace
        from mlsurv.cli._data import _cli_load

        df = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0],
            'start': ['2020-01-01', '2020-06-01', '2020-03-01'],
            'end': ['2020-07-01', '2021-06-01', '2020-09-01'],
            'status': [1, 0, 1],
        })
        filepath = tmp_path / 'dates.csv'
        df.to_csv(filepath, index=False)

        args = Namespace(
            data=str(filepath), time_col='time', event_col='status',
            start_col='start', end_col='end', time_unit='days',
        )
        X, y = _cli_load(args)

        assert len(X) == 3
        assert 'feat' in X.columns
        assert 'start' not in X.columns
        assert 'end' not in X.columns
        assert y['event'].tolist() == [True, False, True]
        assert y['time'][0] == pytest.approx(182.0)

    def test_cli_load_date_mode_bad_path(self, tmp_path):
        """_cli_load exits with friendly error for missing file (date mode)."""
        from argparse import Namespace
        from mlsurv.cli._data import _cli_load

        args = Namespace(
            data=str(tmp_path / 'nonexistent.csv'),
            time_col='time', event_col='event',
            start_col='start', end_col='end', time_unit='days',
        )
        from mlsurv.cli._exit_codes import CLIDataError

        with pytest.raises(CLIDataError):
            _cli_load(args)

    # --event-values flag ------------------------------------------------

    def test_event_values_parser_flag(self):
        """--event-values parses to list and defaults to None."""
        from mlsurv.cli import create_parser

        parser = create_parser()

        args = parser.parse_args(['run', 'data.csv', '--event-values', 'deceased', 'progressed'])
        assert args.event_values == ['deceased', 'progressed']

        args_default = parser.parse_args(['run', 'data.csv'])
        assert args_default.event_values is None

    def test_event_values_standard_mode(self, tmp_path):
        """load_survival_data uses event_values for custom event mapping."""
        from mlsurv.utils import load_survival_data

        df = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0, 4.0],
            'time': [10, 20, 30, 40],
            'status': ['Progressed', 'Stable', 'Progressed', 'Stable'],
        })
        filepath = tmp_path / 'custom_events.csv'
        df.to_csv(filepath, index=False)

        X, y = load_survival_data(
            str(filepath), 'time', 'status',
            event_values=['Progressed'],
        )
        assert y['event'].tolist() == [True, False, True, False]

    def test_event_values_date_mode(self, tmp_path):
        """load_survival_data date-based mode forwards event_values."""
        from mlsurv.utils import load_survival_data

        df = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0],
            'start': ['2020-01-01', '2020-06-01', '2020-03-01'],
            'end': ['2020-07-01', '2021-06-01', '2020-09-01'],
            'outcome': ['Relapsed', 'Stable', 'Relapsed'],
        })
        filepath = tmp_path / 'dates_custom.csv'
        df.to_csv(filepath, index=False)

        X, y = load_survival_data(
            str(filepath), event_col='outcome',
            start_col='start', end_col='end',
            event_values=['Relapsed'],
        )
        assert y['event'].tolist() == [True, False, True]

    def test_event_values_load_data_dispatch(self, tmp_path):
        """_cli_load threads event_values through both standard and date paths."""
        from argparse import Namespace
        from mlsurv.cli._data import _cli_load

        # Standard path
        df1 = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0],
            'time': [10, 20, 30],
            'status': ['Dead', 'Alive', 'Dead'],
        })
        fp1 = tmp_path / 'standard.csv'
        df1.to_csv(fp1, index=False)

        args1 = Namespace(
            data=str(fp1), time_col='time', event_col='status',
            event_values=['Dead'],
        )
        _, y1 = _cli_load(args1)
        assert y1['event'].tolist() == [True, False, True]

        # Date-based path
        df2 = pd.DataFrame({
            'feat': [1.0, 2.0],
            'start': ['2020-01-01', '2020-06-01'],
            'end': ['2020-07-01', '2021-06-01'],
            'outcome': ['Progressed', 'Stable'],
        })
        fp2 = tmp_path / 'dates.csv'
        df2.to_csv(fp2, index=False)

        args2 = Namespace(
            data=str(fp2), time_col='time', event_col='outcome',
            start_col='start', end_col='end', time_unit='days',
            event_values=['Progressed'],
        )
        _, y2 = _cli_load(args2)
        assert y2['event'].tolist() == [True, False]

    def test_event_values_case_sensitive(self, tmp_path):
        """--event-values matching is case-sensitive."""
        from mlsurv.utils import load_survival_data

        df = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0],
            'time': [10, 20, 30],
            'status': ['Deceased', 'Alive', 'Deceased'],
        })
        filepath = tmp_path / 'case.csv'
        df.to_csv(filepath, index=False)

        # Lowercase 'deceased' does NOT match uppercase 'Deceased'
        _, y = load_survival_data(
            str(filepath), 'time', 'status',
            event_values=['deceased'],
        )
        assert y['event'].tolist() == [False, False, False]

    def test_event_values_no_match_warns(self, tmp_path):
        """--event-values with no matching values warns and produces all-censored."""
        import warnings
        from mlsurv.utils import load_survival_data

        df = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0],
            'time': [10, 20, 30],
            'status': ['Dead', 'Alive', 'Dead'],
        })
        filepath = tmp_path / 'nomatch.csv'
        df.to_csv(filepath, index=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, y = load_survival_data(
                str(filepath), 'time', 'status',
                event_values=['Typo'],
            )
        assert y['event'].tolist() == [False, False, False]
        assert any('No rows matched' in str(warning.message) for warning in w)

    def test_event_values_numeric_type_coercion(self, tmp_path):
        """--event-values coerces strings to column dtype for numeric columns."""
        from mlsurv.utils import load_survival_data

        df = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0],
            'time': [10, 20, 30],
            'status': [1, 0, 1],
        })
        filepath = tmp_path / 'numeric.csv'
        df.to_csv(filepath, index=False)

        _, y = load_survival_data(
            str(filepath), 'time', 'status',
            event_values=['1'],
        )
        assert y['event'].tolist() == [True, False, True]

    def test_event_values_empty_list_raises(self, tmp_path):
        """Empty event_values raises ValueError."""
        from mlsurv.utils import load_survival_data

        df = pd.DataFrame({
            'feat': [1.0, 2.0],
            'time': [10, 20],
            'status': ['Dead', 'Alive'],
        })
        filepath = tmp_path / 'empty.csv'
        df.to_csv(filepath, index=False)

        with pytest.raises(ValueError):
            load_survival_data(
                str(filepath), 'time', 'status',
                event_values=[],
            )

    def test_event_values_multiple_values(self, tmp_path):
        """--event-values with multiple values maps all to True."""
        from mlsurv.utils import load_survival_data

        df = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0, 4.0],
            'time': [10, 20, 30, 40],
            'status': ['Progressed', 'Stable', 'Relapsed', 'Stable'],
        })
        filepath = tmp_path / 'multi.csv'
        df.to_csv(filepath, index=False)

        _, y = load_survival_data(
            str(filepath), 'time', 'status',
            event_values=['Progressed', 'Relapsed'],
        )
        assert y['event'].tolist() == [True, False, True, False]


class TestCLIRegression:
    """Regression tests for prior CLI/API drift issues."""

    def test_cmd_run_generates_report_with_evaluation(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        report_path = tmp_path / 'report.html'
        seen = {}
        eval_result = _Obj(concordance=pd.DataFrame({'c_index_test': [0.712]}))
        # run() returns dict of {str: ModelResult}; share one eval object so the
        # identity check holds whichever source cmd_run threads into the report.
        model_result = _Obj(
            name='rsf', cv_metric_mean=0.74, cv_metric_std=0.03, evaluation=eval_result,
        )
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', make_fake_learner(
            seen,
            run_return={'rsf': model_result},
            best_model=model_result,
            evaluation_result=eval_result,
        ))

        rc = main([
            'run',
            str(data_path),
            '--cv-folds', '4',
            '--n-bootstrap', '11',
            '--output', str(report_path),
        ])

        assert rc == 0
        assert seen['setup_kwargs']['cv_folds'] == 4
        assert seen['run_kwargs']['n_bootstrap'] == 11
        assert seen['report_eval_result'] is eval_result
        assert seen['report_bootstrap_results'] is None
        assert report_path.exists()

    def test_cmd_tune_uses_best_params_underscore(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        output_path = tmp_path / 'tune.json'
        seen = {}

        # Kept inline: the best_params property guard asserts the CLI reads
        # best_params_ (underscore), not the property.
        class FakeTuneResult:
            name = 'rsf'
            cv_metric_mean = 0.741
            cv_metric_std = 0.031
            best_params_ = {'model__n_estimators': 200}

            @property
            def best_params(self):
                raise AssertionError('CLI must not access result.best_params')

        monkeypatch.setattr(mlsurv, 'SurvivalLearner',
                            make_fake_learner(seen, tune_return=FakeTuneResult()))

        rc = main([
            'tune',
            str(data_path),
            '--model', 'rsf',
            '--n-trials', '7',
            '--cv-folds', '3',
            '--output', str(output_path),
        ])

        assert rc == 0
        assert seen['setup_kwargs']['cv_folds'] == 3
        assert seen['tune_model'] == 'rsf'
        assert seen['n_trials'] == 7
        payload = json.loads(output_path.read_text())
        assert payload['best_params'] == {'model__n_estimators': 200}


# Sentinels asserting the trained model object is the one forwarded downstream to
# evaluate()/bootstrap() (identity check). Module-level so the parametrize CASES and
# their check helpers can share the same object.
_EVAL_MODEL_SENTINEL = object()
_BOOT_MODEL_SENTINEL = object()


def _check_train_calls(seen):
    assert seen['train_model'] == 'coxph'


def _check_train_output(df):
    assert set(df.columns) == {'model', 'cv_metric_mean', 'cv_metric_std'}


def _check_evaluate_calls(seen):
    assert seen['train_model'] == 'coxph'
    assert seen['evaluate_model_result'] is _EVAL_MODEL_SENTINEL
    assert seen['n_importance_repeats'] == 13


def _check_evaluate_output(df):
    assert df['c_index_test'].iloc[0] == pytest.approx(0.719)


def _check_bootstrap_calls(seen):
    assert seen['train_model'] == 'coxph'
    assert seen['bootstrap_model'] is _BOOT_MODEL_SENTINEL
    assert seen['n_bootstrap'] == 25


def _check_bootstrap_output(df):
    assert set(df.columns) == {'metric', 'mean'}


# (command, extra argv, factory kwargs, expected setup_kwargs subset,
#  call-capture check, output-DataFrame check). Each case mirrors the original
# per-command forwarding test; the trained-model sentinel verifies the result is
# threaded into the downstream evaluate()/bootstrap() call.
_CMD_FORWARD_CASES = [
    pytest.param(
        'train',
        ['--model', 'coxph', '--cv-folds', '4'],
        {'train_return': _Obj(cv_metric_mean=0.733, cv_metric_std=0.051)},
        {'cv_folds': 4},
        _check_train_calls,
        _check_train_output,
        id='train',
    ),
    pytest.param(
        'evaluate',
        ['--model', 'coxph', '--cv-folds', '6', '--n-importance-repeats', '13'],
        {'train_return': _EVAL_MODEL_SENTINEL,
         'evaluate_return': _Obj(concordance=pd.DataFrame({'c_index_test': [0.719]}))},
        {'cv_folds': 6},
        _check_evaluate_calls,
        _check_evaluate_output,
        id='evaluate',
    ),
    pytest.param(
        'bootstrap',
        ['--model', 'coxph', '--cv-folds', '4', '--n-bootstrap', '25'],
        {'train_return': _BOOT_MODEL_SENTINEL,
         'bootstrap_return': _Obj(
             summary=pd.DataFrame({'metric': ['harrell'], 'mean': [0.71]}))},
        {'cv_folds': 4},
        _check_bootstrap_calls,
        _check_bootstrap_output,
        id='bootstrap',
    ),
]


class TestCLICommandContracts:
    """Command-to-learner wiring tests."""

    @pytest.mark.parametrize(
        'command,argv_tail,build_kwargs,setup_checks,check_calls,check_output',
        _CMD_FORWARD_CASES,
    )
    def test_cmd_forwards_args_and_writes_output(
        self, monkeypatch, tmp_path, make_fake_learner,
        command, argv_tail, build_kwargs, setup_checks, check_calls, check_output,
    ):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        output_path = tmp_path / 'out.csv'
        seen = {}
        monkeypatch.setattr(
            mlsurv, 'SurvivalLearner', make_fake_learner(seen, **build_kwargs),
        )

        rc = main([command, str(data_path), *argv_tail, '--output', str(output_path)])

        assert rc == 0
        for key, value in setup_checks.items():
            assert seen['setup_kwargs'][key] == value
        check_calls(seen)
        check_output(pd.read_csv(output_path))

    def test_cmd_evaluate_forwards_shap_params(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        """--shap-max-background and --shap-max-explain reach evaluate()."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        output_path = tmp_path / 'eval_shap.csv'
        seen = {}
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', make_fake_learner(
            seen,
            train_return=object(),
            evaluate_return=_Obj(concordance=pd.DataFrame({'c_index_test': [0.700]})),
        ))

        rc = main([
            'evaluate',
            str(data_path),
            '--model', 'coxph',
            '--shap',
            '--shap-max-background', '25',
            '--shap-max-explain', '15',
            '--output', str(output_path),
        ])

        assert rc == 0
        assert seen['eval_kwargs']['include_shap'] is True
        assert seen['eval_kwargs']['shap_max_background'] == 25
        assert seen['eval_kwargs']['shap_max_explain'] == 15

    def test_cmd_evaluate_writes_report_when_requested(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        report_path = tmp_path / 'evaluate_report.html'
        seen = {}
        sentinel_model = object()
        eval_result = _Obj(concordance=pd.DataFrame({'c_index_test': [0.719]}))
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', make_fake_learner(
            seen, train_return=sentinel_model, evaluate_return=eval_result,
        ))

        rc = main([
            'evaluate',
            str(data_path),
            '--model', 'coxph',
            '--report', str(report_path),
        ])

        assert rc == 0
        assert seen['train_model'] == 'coxph'
        assert seen['evaluate_model_result'] is sentinel_model
        assert seen['report_eval_result'] is eval_result
        assert seen['report_bootstrap_results'] is None
        assert report_path.exists()

    def test_cmd_predict_loads_artifact_and_writes_output(
        self, monkeypatch, tmp_path, capsys,
    ):
        import warnings

        import numpy as np
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'predict.csv')
        output_path = tmp_path / 'predictions.csv'
        artifact_dir = tmp_path / 'artifact'
        artifact_dir.mkdir()
        seen = {}

        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2'], 'time_unit': 'months'},
        }

        class FakePredictModel:
            def predict(self, X):
                seen['predict_columns'] = X.columns.tolist()
                return np.array([0.1, 0.2, 0.3, 0.4])

            def predict_survival_function(self, X):
                seen['surv_columns'] = X.columns.tolist()
                # Return DataFrame: times as index, patients as columns
                times = np.array([6.0, 12.0, 24.0, 48.0])
                data = {}
                for i in range(len(X)):
                    data[i] = [1.0 - (t / 100.0) for t in times]
                return pd.DataFrame(data, index=times)

        def fake_load_artifact(path):
            seen['load_artifact_path'] = path
            return metadata, FakePredictModel(), None

        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rc = main([
                'predict',
                str(data_path),
                '--load-artifact', str(artifact_dir),
                '--times', '12', '24',
                '--output', str(output_path),
            ])

        assert rc == 0
        assert seen['predict_columns'] == ['f1', 'f2']
        assert seen['surv_columns'] == ['f1', 'f2']

        out = pd.read_csv(output_path)
        # New columns: median_survival, rmst_48 (default tau = max grid),
        # survival/risk at times, rmst at times
        assert 'sample_idx' in out.columns
        assert 'model_name' in out.columns
        assert 'risk_score' in out.columns
        assert 'median_survival' in out.columns
        assert 'rmst_48' in out.columns  # default tau = max grid time
        assert 'survival_12' in out.columns
        assert 'risk_12' in out.columns
        assert 'survival_24' in out.columns
        assert 'risk_24' in out.columns
        assert 'rmst_12' in out.columns  # RMST at user-specified times
        assert 'rmst_24' in out.columns
        assert out['model_name'].tolist() == ['coxph'] * 4
        assert out['survival_12'].iloc[0] == pytest.approx(0.88)
        assert out['risk_24'].iloc[0] == pytest.approx(0.24)

        # Extra-column warning is now a SuggestionWarning
        assert any("extra column" in str(wn.message).lower() for wn in w)
        stderr = capsys.readouterr().err
        assert "time unit (months)" in stderr

    def test_cmd_predict_without_times_prints_risk_scores_to_stdout(
        self, monkeypatch, tmp_path, capsys,
    ):
        import numpy as np
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'predict.csv')
        artifact_dir = tmp_path / 'artifact'
        artifact_dir.mkdir()

        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2']},
        }

        class FakePredictModel:
            def predict(self, X):
                return np.array([0.1, 0.2, 0.3, 0.4])

        def fake_load_artifact(path):
            return metadata, FakePredictModel(), None

        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'predict',
            str(data_path),
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 0
        captured = capsys.readouterr()
        assert 'risk_score' in captured.out
        assert 'model_name' in captured.out
        assert 'survival_12' not in captured.out

    def test_cmd_predict_setup_only_artifact_returns_1(
        self, monkeypatch, tmp_path, capsys,
    ):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'predict.csv')
        artifact_dir = tmp_path / 'artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'step': 'setup',
            'model_name': None,
            'data_contract': {'feature_names': ['f1', 'f2']},
        }

        def fake_load_artifact(path):
            return metadata, None, None

        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'predict',
            str(data_path),
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 3  # EXIT_DATA
        assert "does not contain a model" in capsys.readouterr().err

    def test_cmd_predict_missing_feature_names_returns_1(
        self, monkeypatch, tmp_path, capsys,
    ):
        import numpy as np
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'predict.csv')
        artifact_dir = tmp_path / 'artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {},
        }

        class FakePredictModel:
            def predict(self, X):
                return np.array([0.1, 0.2, 0.3, 0.4])

        def fake_load_artifact(path):
            return metadata, FakePredictModel(), None

        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'predict',
            str(data_path),
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 3  # EXIT_DATA
        assert "feature_names" in capsys.readouterr().err

    def test_cmd_predict_wraps_risk_score_failures(
        self, monkeypatch, tmp_path, capsys,
    ):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'predict.csv')
        artifact_dir = tmp_path / 'artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2']},
        }

        class BrokenRiskModel:
            def predict(self, X):
                raise RuntimeError("shape mismatch")

        def fake_load_artifact(path):
            return metadata, BrokenRiskModel(), None

        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'predict',
            str(data_path),
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 4  # EXIT_COMPUTATION
        assert "Could not compute predictions" in capsys.readouterr().err

    def test_cmd_predict_survival_failure_degrades_gracefully(
        self, monkeypatch, tmp_path, capsys,
    ):
        """When predict_survival_function fails, return risk scores + warning."""
        import warnings
        import numpy as np
        from mlsurv.cli import main
        from mlsurv._warnings import SuggestionWarning

        data_path = _write_sample_csv(tmp_path / 'predict.csv')
        artifact_dir = tmp_path / 'artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2']},
        }

        class BrokenSurvivalModel:
            def predict(self, X):
                return np.array([0.1, 0.2, 0.3, 0.4])

            def predict_survival_function(self, X, times=None):
                raise RuntimeError("bad survival function")

        def fake_load_artifact(path):
            return metadata, BrokenSurvivalModel(), None

        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rc = main([
                'predict',
                str(data_path),
                '--load-artifact', str(artifact_dir),
                '--times', '12',
            ])

        # Graceful degradation: risk scores returned, rc=0
        assert rc == 0
        out = capsys.readouterr().out
        assert 'risk_score' in out
        # SuggestionWarning emitted about survival function failure
        suggestion_warnings = [x for x in w if issubclass(x.category, SuggestionWarning)]
        assert len(suggestion_warnings) >= 1

    def test_cmd_predict_strict_returns_1_on_survival_failure(
        self, monkeypatch, tmp_path, capsys,
    ):
        """With --strict, return rc=1 when --times requested but survival fails."""
        import warnings
        import numpy as np
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'predict.csv')
        artifact_dir = tmp_path / 'artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2']},
        }

        class BrokenSurvivalModel:
            def predict(self, X):
                return np.array([0.1, 0.2, 0.3, 0.4])

            def predict_survival_function(self, X, times=None):
                raise RuntimeError("bad survival function")

        def fake_load_artifact(path):
            return metadata, BrokenSurvivalModel(), None

        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rc = main([
                'predict',
                str(data_path),
                '--load-artifact', str(artifact_dir),
                '--times', '12',
                '--strict',
            ])

        assert rc == 4  # EXIT_COMPUTATION
        captured = capsys.readouterr()
        assert 'Error: Survival function prediction failed' in captured.err
        assert 'risk_score' not in captured.out

    def test_cmd_predict_strict_without_failure_returns_0(
        self, monkeypatch, tmp_path, capsys,
    ):
        """With --strict, return rc=0 when survival prediction succeeds."""
        import warnings
        import numpy as np
        import pandas as pd
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'predict.csv')
        artifact_dir = tmp_path / 'artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2']},
        }

        class WorkingSurvivalModel:
            def predict(self, X):
                return np.array([0.1, 0.2, 0.3, 0.4])

            def predict_survival_function(self, X):
                times = np.array([6.0, 12.0, 24.0])
                data = {i: [1.0 - (t / 100.0) for t in times]
                        for i in range(len(X))}
                return pd.DataFrame(data, index=times)

        def fake_load_artifact(path):
            return metadata, WorkingSurvivalModel(), None

        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rc = main([
                'predict',
                str(data_path),
                '--load-artifact', str(artifact_dir),
                '--times', '12',
                '--strict',
            ])

        assert rc == 0

    def test_cmd_validate_loads_external_and_writes_output(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        import mlsurv
        from mlsurv.cli import main

        train_path = _write_sample_csv(tmp_path / 'sample.csv')
        val_path = _write_sample_csv(tmp_path / 'val' / 'sample.csv')
        output_path = tmp_path / 'validation.csv'
        seen = {}
        sentinel_model = object()
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', make_fake_learner(
            seen,
            train_return=sentinel_model,
            validate_return=_Obj(concordance=pd.DataFrame({'c_index_test': [0.702]})),
            train_records_n=True,
        ))

        rc = main([
            'validate',
            str(train_path),
            '--validation-data', str(val_path),
            '--model', 'coxph',
            '--cv-folds', '4',
            '--cohort-name', 'MSK',
            '--n-bootstrap', '30',
            '--output', str(output_path),
        ])

        assert rc == 0
        assert seen['setup_kwargs']['cv_folds'] == 4
        assert seen['train_model'] == 'coxph'
        assert seen['validate_model'] is sentinel_model
        assert seen['cohort_name'] == 'MSK'
        assert seen['validate_kwargs']['n_bootstrap'] == 30
        out = pd.read_csv(output_path)
        assert 'c_index_test' in out.columns

    def test_cmd_validate_uses_val_event_values(self, monkeypatch, tmp_path):
        """validate should support validation cohorts with different event coding."""
        import mlsurv
        from mlsurv.cli import main

        train_path = _write_sample_csv(tmp_path / 'train.csv')
        val_path = tmp_path / 'val.csv'
        pd.DataFrame(
            {
                'f1': [0.2, 0.4, 0.6, 0.8],
                'f2': [1.0, 0.0, 1.0, 0.0],
                'time': [10, 12, 8, 15],
                'event': [2, 1, 2, 1],
            }
        ).to_csv(val_path, index=False)
        seen = {}
        sentinel_model = object()

        class FakeValidationResult:
            concordance = pd.DataFrame({'c_index_test': [0.702]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                seen['train_n'] = len(X)

            def setup(self, **kwargs):
                seen['setup_kwargs'] = kwargs

            def train(self, model_name):
                return sentinel_model

            def validate(self, X_val, y_val, models=None, name='External', **kwargs):
                seen['validation_events'] = y_val['event'].tolist()
                return FakeValidationResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'validate',
            str(train_path),
            '--validation-data', str(val_path),
            '--model', 'coxph',
            '--val-event-values', '2',
        ])

        assert rc == 0
        assert seen['validation_events'] == [True, False, True, False]

    def test_cmd_validate_loads_external_dates(self, monkeypatch, tmp_path):
        """validate should support date-based loading for the validation cohort."""
        import mlsurv
        from mlsurv.cli import main

        train_path = _write_sample_csv(tmp_path / 'train.csv')
        val_path = tmp_path / 'val_dates.csv'
        pd.DataFrame(
            {
                'f1': [0.2, 0.4, 0.6],
                'f2': [1.0, 0.0, 1.0],
                'dx': ['2020-01-01', '2020-02-01', '2020-03-01'],
                'fu': ['2020-06-01', '2020-08-01', '2020-09-01'],
                'status': ['dead', 'alive', 'dead'],
            }
        ).to_csv(val_path, index=False)
        seen = {}
        sentinel_model = object()

        class FakeValidationResult:
            concordance = pd.DataFrame({'c_index_test': [0.702]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                return sentinel_model

            def validate(self, X_val, y_val, models=None, name='External', **kwargs):
                seen['val_columns'] = list(X_val.columns)
                seen['val_events'] = y_val['event'].tolist()
                assert all(t > 0 for t in y_val['time'])
                return FakeValidationResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'validate',
            str(train_path),
            '--validation-data', str(val_path),
            '--model', 'coxph',
            '--val-start-col', 'dx',
            '--val-end-col', 'fu',
            '--val-event-col', 'status',
            '--val-time-unit', 'months',
            '--val-event-values', 'dead',
        ])

        assert rc == 0
        assert 'dx' not in seen['val_columns']
        assert 'fu' not in seen['val_columns']
        assert seen['val_events'] == [True, False, True]

    def test_cmd_validate_event_values_falls_back_to_training(self, monkeypatch, tmp_path):
        """When --val-event-values is omitted, training --event-values should apply."""
        import mlsurv
        from mlsurv.cli import main

        train_path = _write_sample_csv(tmp_path / 'train.csv')
        val_path = tmp_path / 'val.csv'
        pd.DataFrame(
            {
                'f1': [0.2, 0.4, 0.6, 0.8],
                'f2': [1.0, 0.0, 1.0, 0.0],
                'time': [10, 12, 8, 15],
                'event': ['dead', 'alive', 'dead', 'alive'],
            }
        ).to_csv(val_path, index=False)
        seen = {}

        class FakeValidationResult:
            concordance = pd.DataFrame({'c_index_test': [0.702]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                return object()

            def validate(self, X_val, y_val, models=None, name='External', **kwargs):
                seen['val_events'] = y_val['event'].tolist()
                return FakeValidationResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'validate',
            str(train_path),
            '--validation-data', str(val_path),
            '--model', 'coxph',
            '--event-values', 'dead',
            # No --val-event-values: should inherit 'dead' from training
        ])

        assert rc == 0
        assert seen['val_events'] == [True, False, True, False]

    def test_cmd_setup_prints_diagnostics(self, monkeypatch, tmp_path, capsys):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                self.X_train = X.iloc[:3]
                self.X_test = X.iloc[3:]

            def setup(self, **kwargs):
                seen['setup_kwargs'] = kwargs

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main(['setup', str(data_path), '--cv-folds', '4'])

        assert rc == 0
        assert seen['setup_kwargs']['cv_folds'] == 4
        out = capsys.readouterr().out
        assert 'Setup Diagnostics:' in out
        assert 'n_samples:' in out
        assert 'n_train:' in out


class TestLoadReferenceCsv:
    """Unit tests for load_reference_csv()."""

    def test_all_columns(self, tmp_path):
        import numpy as np
        from mlsurv.cli._data import load_reference_csv

        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({'score_a': [1.0, 2.0], 'score_b': [3.0, 4.0]}).to_csv(
            ref_path, index=False,
        )
        result = load_reference_csv(str(ref_path))
        assert set(result.keys()) == {'score_a', 'score_b'}
        assert isinstance(result['score_a'], np.ndarray)
        assert result['score_a'].dtype == np.float64

    def test_selected_columns(self, tmp_path):
        from mlsurv.cli._data import load_reference_csv

        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]}).to_csv(
            ref_path, index=False,
        )
        result = load_reference_csv(str(ref_path), reference_cols=['a', 'c'])
        assert set(result.keys()) == {'a', 'c'}

    def test_missing_column_exits(self, tmp_path):
        from mlsurv.cli._data import load_reference_csv
        from mlsurv.cli._exit_codes import CLIDataError

        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({'a': [1.0]}).to_csv(ref_path, index=False)
        with pytest.raises(CLIDataError):
            load_reference_csv(str(ref_path), reference_cols=['nonexistent'])

    def test_row_mismatch_exits(self, tmp_path):
        from mlsurv.cli._data import load_reference_csv
        from mlsurv.cli._exit_codes import CLIDataError

        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({'a': [1.0, 2.0, 3.0]}).to_csv(ref_path, index=False)
        with pytest.raises(CLIDataError):
            load_reference_csv(str(ref_path), expected_n_rows=5)

    def test_non_numeric_column_exits(self, tmp_path):
        from mlsurv.cli._data import load_reference_csv
        from mlsurv.cli._exit_codes import CLIDataError

        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({'score': [1.0, 2.0], 'stage': ['I', 'II']}).to_csv(
            ref_path, index=False,
        )
        with pytest.raises(CLIDataError):
            load_reference_csv(str(ref_path))

    def test_empty_reference_file_exits(self, tmp_path):
        """Header-only CSV (no data rows) exits."""
        from mlsurv.cli._data import load_reference_csv
        from mlsurv.cli._exit_codes import CLIDataError

        ref_path = tmp_path / 'refs.csv'
        ref_path.write_text('score_a,score_b\n')
        with pytest.raises(CLIDataError):
            load_reference_csv(str(ref_path))

    def test_file_not_found_exits(self, tmp_path):
        from mlsurv.cli._data import load_reference_csv
        from mlsurv.cli._exit_codes import CLIDataError

        with pytest.raises(CLIDataError):
            load_reference_csv(str(tmp_path / 'missing.csv'))


class TestCLIBenchmarkCommand:
    """Integration test for cmd_benchmark with --reference-file."""

    def test_cmd_benchmark_loads_reference_file(self, monkeypatch, tmp_path):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({
            'psa': [0.5, 0.6, 0.7, 0.8],
            'nomogram': [0.3, 0.4, 0.5, 0.6],
        }).to_csv(ref_path, index=False)
        seen = {}

        class FakeBenchmarkResult:
            evaluation = type('E', (), {
                'concordance': pd.DataFrame({'c_index_test': [0.7]}),
            })()

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                seen['train_n'] = len(X)

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                seen['train_model'] = model_name

            def benchmark(self, references, **kwargs):
                seen['ref_keys'] = sorted(references.keys())
                seen['ref_len'] = len(next(iter(references.values())))
                return FakeBenchmarkResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'benchmark',
            str(data_path),
            '--reference-file', str(ref_path),
            '--no-bootstrap',
        ])

        assert rc == 0
        assert seen['ref_keys'] == ['nomogram', 'psa']
        assert seen['ref_len'] == 4

    def test_cmd_benchmark_reference_cols_filter(self, monkeypatch, tmp_path):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({
            'psa': [0.5, 0.6, 0.7, 0.8],
            'nomogram': [0.3, 0.4, 0.5, 0.6],
        }).to_csv(ref_path, index=False)
        seen = {}

        class FakeBenchmarkResult:
            evaluation = type('E', (), {
                'concordance': pd.DataFrame({'c_index_test': [0.7]}),
            })()

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                pass

            def benchmark(self, references, **kwargs):
                seen['ref_keys'] = sorted(references.keys())
                return FakeBenchmarkResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'benchmark',
            str(data_path),
            '--reference-file', str(ref_path),
            '--reference-cols', 'psa',
            '--no-bootstrap',
        ])

        assert rc == 0
        assert seen['ref_keys'] == ['psa']

    def test_cmd_benchmark_row_mismatch_exits(self, tmp_path, capsys):
        """Reference file with wrong row count exits before training."""
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')  # 4 rows
        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({'psa': [0.5, 0.6]}).to_csv(ref_path, index=False)  # 2 rows

        rc = main([
            'benchmark', str(data_path),
            '--reference-file', str(ref_path),
            '--no-bootstrap', '--verbose',
        ])

        assert rc == 3  # EXIT_DATA
        captured = capsys.readouterr()
        assert 'Loaded data' not in captured.out

    def test_cmd_benchmark_no_loaded_data_on_ref_error(self, tmp_path, capsys):
        """'Loaded data' must not appear when reference file is invalid."""
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        rc = main([
            'benchmark', str(data_path),
            '--reference-file', str(tmp_path / 'nonexistent.csv'),
            '--no-bootstrap', '--verbose',
        ])

        assert rc == 3  # EXIT_DATA
        captured = capsys.readouterr()
        assert 'Loaded data' not in captured.out


class TestCLIErrorHandling:
    """CLI error handling and arg validation tests."""

    @pytest.fixture
    def sample_data_file(self, tmp_path, synthetic_data_clean):
        X, y = synthetic_data_clean
        df = X.copy()
        df['time'] = y['time']
        df['event'] = y['event'].astype(int)
        filepath = tmp_path / 'data.csv'
        df.to_csv(filepath, index=False)
        return filepath

    def test_invalid_model_run_still_completes(self, sample_data_file):
        from mlsurv.cli import main

        result = main(['run', str(sample_data_file), '--models', 'invalid_model_xyz'])
        assert result in (0, 1, 3, 4)  # may succeed partially or fail with data/computation error

    def test_invalid_file_format(self, tmp_path):
        from mlsurv.cli import main

        bad_file = tmp_path / 'data.txt'
        bad_file.write_text('not,valid,csv\n')

        rc = main(['run', str(bad_file), '--models', 'coxph'])
        assert rc != 0

    @pytest.mark.parametrize(
        'argv',
        [
            ['run', 'test.csv', '--cv-folds', '1'],
            ['train', 'test.csv', '--model', 'coxph', '--cv-folds', '1'],
            ['tune', 'test.csv', '--model', 'rsf', '--n-trials', '0'],
            ['evaluate', 'test.csv', '--model', 'coxph', '--test-fraction', '0'],
            ['evaluate', 'test.csv', '--model', 'coxph', '--test-fraction', '1'],
            ['evaluate', 'test.csv', '--model', 'coxph', '--test-fraction', '1.5'],
            ['bootstrap', 'test.csv', '--model', 'coxph', '--n-bootstrap', '0'],
            ['setup', 'test.csv', '--n-jobs', '0'],
            ['run', 'test.csv', '--cv-strategy', 'groupkfold'],
            ['evaluate', 'test.csv', '--model', 'coxph', '--shap', '--shap-max-background', '0'],
            ['evaluate', 'test.csv', '--model', 'coxph', '--shap', '--shap-max-explain', '0'],
        ],
        ids=[
            'run_cv_folds_too_small',
            'train_cv_folds_too_small',
            'tune_n_trials_zero',
            'evaluate_test_fraction_zero',
            'evaluate_test_fraction_one',
            'evaluate_test_fraction_over_one',
            'bootstrap_n_bootstrap_zero',
            'setup_n_jobs_zero',
            'run_groupkfold_missing_group_col',
            'evaluate_shap_max_background_zero',
            'evaluate_shap_max_explain_zero',
        ],
    )
    def test_main_rejects_invalid_numeric_args(self, argv):
        from mlsurv.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(argv)

        assert exc_info.value.code == 2

    def test_missing_file(self):
        from mlsurv.cli import main

        rc = main(['run', 'nonexistent_file.csv', '--models', 'coxph'])
        assert rc == 3  # EXIT_DATA

    def test_missing_time_column(self, tmp_path):
        from mlsurv.cli import main

        csv = tmp_path / 'no_time.csv'
        df = pd.DataFrame({'f1': [1, 2], 'event': [1, 0]})
        df.to_csv(csv, index=False)

        rc = main(['run', str(csv), '--models', 'coxph'])
        assert rc == 3  # EXIT_DATA

    def test_missing_event_column(self, tmp_path):
        from mlsurv.cli import main

        csv = tmp_path / 'no_event.csv'
        df = pd.DataFrame({'f1': [1, 2], 'time': [10, 20]})
        df.to_csv(csv, index=False)

        rc = main(['run', str(csv), '--models', 'coxph'])
        assert rc == 3  # EXIT_DATA


class TestCLIArtifactArgs:
    """Parser-level tests for artifact arguments."""

    @pytest.mark.parametrize('subcommand,argv_extra', [
        ('train', ['test.csv', '--model', 'coxph', '--save-artifact', '/tmp/art']),
        ('tune', ['test.csv', '--model', 'rsf', '--save-artifact', '/tmp/art']),
        ('run', ['test.csv', '--save-artifact', '/tmp/art']),
        ('evaluate', ['test.csv', '--model', 'coxph', '--save-artifact', '/tmp/art']),
        ('evaluate', ['test.csv', '--model', 'coxph', '--load-artifact', '/tmp/art']),
        ('bootstrap', ['test.csv', '--model', 'coxph', '--save-artifact', '/tmp/art']),
        ('bootstrap', ['test.csv', '--model', 'coxph', '--load-artifact', '/tmp/art']),
        ('validate', [
            'test.csv', '--validation-data', 'ext.csv',
            '--model', 'coxph', '--load-artifact', '/tmp/art',
        ]),
        ('setup', ['test.csv', '--save-artifact', '/tmp/art']),
    ])
    def test_subcommand_accepts_artifact_flag(self, subcommand, argv_extra):
        """Each subcommand should accept its assigned artifact flags."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([subcommand] + argv_extra)
        # Just check it parsed without error
        assert args.command == subcommand

    def test_train_does_not_have_load_artifact(self):
        """train should not accept --load-artifact."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['train', 'test.csv', '--model', 'coxph', '--load-artifact', '/tmp'])

    def test_validate_does_not_have_save_artifact(self):
        """validate should not accept --save-artifact."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'validate', 'test.csv', '--validation-data', 'ext.csv',
                '--model', 'coxph', '--save-artifact', '/tmp',
            ])


class TestCLIArtifactContracts:
    """Monkeypatch contract tests for artifact wiring."""

    def test_train_save_artifact_creates_directory(self, monkeypatch, tmp_path):
        """train --save-artifact should create artifact directory."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        artifact_dir = tmp_path / 'my_artifact'
        seen = {}

        class FakeResult:
            cv_metric_mean = 0.733
            cv_metric_std = 0.051
            model = 'fake_model'
            name = 'coxph'
            is_tuned = False
            best_params_ = None

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                self.X = X

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                seen['train_model'] = model_name
                return FakeResult()

            def save_artifact(self, model, path, *, step, time_col, event_col):
                seen['save_artifact_called'] = True
                seen['save_artifact_path'] = path
                seen['save_artifact_step'] = step
                Path(path).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'train', str(data_path),
            '--model', 'coxph',
            '--save-artifact', str(artifact_dir),
        ])

        assert rc == 0
        assert seen['save_artifact_called']
        assert seen['save_artifact_step'] == 'train'
        assert artifact_dir.exists()

    def test_evaluate_load_artifact_uses_loaded_model(self, monkeypatch, tmp_path):
        """evaluate --load-artifact should use loaded model without training."""
        import json
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}

        # Create a fake artifact directory
        artifact_dir = tmp_path / 'fake_artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2']},
            'results': {
                'cv_metric_mean': 0.733,
                'cv_metric_std': 0.051,
            },
        }
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))

        class FakeEvalResult:
            concordance = pd.DataFrame({'c_index_test': [0.719]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                seen['train_called'] = True
                raise AssertionError("train() should not be called")

            def evaluate(self, model_result, **kwargs):
                seen['evaluate_model'] = model_result.model
                seen['model_name'] = model_result.name
                return FakeEvalResult()

        # Mock PersistenceMixin.load_artifact to return known values
        loaded_sentinel = object()

        def fake_load_artifact(path):
            seen['load_artifact_called'] = True
            return metadata, loaded_sentinel, None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'evaluate', str(data_path),
            '--model', 'coxph',
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 0
        assert seen['load_artifact_called']
        assert 'train_called' not in seen
        # The model on the result should be the loaded one
        assert seen['evaluate_model'] is loaded_sentinel
        assert seen['model_name'] == 'coxph'

    def test_feature_mismatch_missing_raises(self, monkeypatch, tmp_path, capsys):
        """Missing features should raise ValueError (fatal error)."""
        import json
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        # Artifact expects features that the data doesn't have
        artifact_dir = tmp_path / 'mismatch_artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['x1', 'x2', 'x3']},
            'results': {
                'cv_metric_mean': 0.700,
                'cv_metric_std': 0.050,
            },
        }
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

        loaded_sentinel = object()

        def fake_load_artifact(path):
            return metadata, loaded_sentinel, None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'evaluate', str(data_path),
            '--model', 'coxph',
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 3  # EXIT_DATA (ValueError from feature compatibility)
        stderr = capsys.readouterr().err
        assert "missing features" in stderr.lower()

    def test_feature_mismatch_extra_warns(self, monkeypatch, tmp_path, capsys):
        """Extra features in data should warn but not fail."""
        import json
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        # Artifact expects only f1 — data has f1 + f2 (extra)
        artifact_dir = tmp_path / 'extra_features_artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1']},
            'results': {
                'cv_metric_mean': 0.700,
                'cv_metric_std': 0.050,
            },
        }
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))

        class FakeEvalResult:
            concordance = pd.DataFrame({'c_index_test': [0.700]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def evaluate(self, model_result, **kwargs):
                return FakeEvalResult()

        loaded_sentinel = object()

        def fake_load_artifact(path):
            return metadata, loaded_sentinel, None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'evaluate', str(data_path),
            '--model', 'coxph',
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 0
        stderr = capsys.readouterr().err
        assert "extra features" in stderr.lower()

    def test_load_artifact_falls_back_to_model_arg(
        self, monkeypatch, tmp_path, capsys,
    ):
        """When artifact has no model_name, --model arg should be used in message."""
        import json
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        # Artifact with model_name=None (like a setup-only artifact that
        # shouldn't reach here, but tests the fallback)
        artifact_dir = tmp_path / 'no_name_artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': None,
            'data_contract': {'feature_names': ['f1', 'f2']},
            'results': {},
        }
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))

        class FakeEvalResult:
            concordance = pd.DataFrame({'c_index_test': [0.700]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                raise AssertionError("train() should not be called")

            def evaluate(self, model_result, **kwargs):
                return FakeEvalResult()

        loaded_sentinel = object()

        def fake_load_artifact(path):
            return metadata, loaded_sentinel, None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'evaluate', str(data_path),
            '--model', 'coxph',
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 0
        stderr = capsys.readouterr().err
        # Should fall back to --model arg for display since metadata model_name is None
        assert "coxph" in stderr
        assert "no CV metrics stored" in stderr

    def test_tune_save_artifact_calls_save(self, monkeypatch, tmp_path):
        """tune --save-artifact should call save_artifact with step='tune'."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        artifact_dir = tmp_path / 'tune_artifact'
        seen = {}

        class FakeTuneResult:
            name = 'rsf'
            cv_metric_mean = 0.741
            cv_metric_std = 0.031
            best_params_ = {'model__n_estimators': 200}

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                self.X = X

            def setup(self, **kwargs):
                pass

            def tune(self, model, n_trials=50, verbose=False):
                return FakeTuneResult()

            def save_artifact(self, model, path, *, step, time_col, event_col):
                seen['save_artifact_called'] = True
                seen['save_artifact_step'] = step
                Path(path).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'tune', str(data_path),
            '--model', 'rsf', '--n-trials', '5',
            '--save-artifact', str(artifact_dir),
        ])

        assert rc == 0
        assert seen['save_artifact_called']
        assert seen['save_artifact_step'] == 'tune'

    def test_run_save_artifact_calls_save(self, monkeypatch, tmp_path):
        """run --save-artifact should call save_artifact with step='run'."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        artifact_dir = tmp_path / 'run_artifact'
        seen = {}

        class FakeModelResult:
            name = 'coxph'
            cv_metric_mean = 0.74
            cv_metric_std = 0.03
            evaluation = None

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                self.X = X

            def setup(self, **kwargs):
                pass

            def run(self, **kwargs):
                # run() now returns dict of {str: ModelResult}
                return {'coxph': FakeModelResult()}

            @property
            def best_model(self):
                return FakeModelResult()

            def save_artifact(self, model, path, *, step, time_col, event_col):
                seen['save_artifact_called'] = True
                seen['save_artifact_step'] = step
                Path(path).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'run', str(data_path),
            '--save-artifact', str(artifact_dir),
        ])

        assert rc == 0
        assert seen['save_artifact_called']
        assert seen['save_artifact_step'] == 'run'

    def test_setup_save_artifact_calls_save(self, monkeypatch, tmp_path):
        """setup --save-artifact should call save_artifact with step='setup'."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        artifact_dir = tmp_path / 'setup_artifact'
        seen = {}

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                self.X = X
                self.X_train = X.iloc[:3]
                self.X_test = X.iloc[3:]

            def setup(self, **kwargs):
                pass

            def save_artifact(self, model, path, *, step, time_col, event_col):
                seen['save_artifact_called'] = True
                seen['save_artifact_step'] = step
                seen['save_artifact_model'] = model
                Path(path).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'setup', str(data_path),
            '--save-artifact', str(artifact_dir),
        ])

        assert rc == 0
        assert seen['save_artifact_called']
        assert seen['save_artifact_step'] == 'setup'
        assert seen['save_artifact_model'] is None  # setup passes None

    def test_save_artifact_oserror_degrades_gracefully(self, monkeypatch, tmp_path, capsys):
        """OSError during save_artifact should warn on stderr, not crash."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        class FakeResult:
            cv_metric_mean = 0.733
            cv_metric_std = 0.051

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                self.X = X

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                return FakeResult()

            def save_artifact(self, model, path, *, step, time_col, event_col):
                raise OSError("Permission denied")

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'train', str(data_path),
            '--model', 'coxph',
            '--save-artifact', str(tmp_path / 'blocked'),
        ])

        assert rc == 0  # Primary operation succeeded
        stderr = capsys.readouterr().err
        assert "Failed to save artifact" in stderr

    def test_save_artifact_overwrite_warns(self, monkeypatch, tmp_path, capsys):
        """Overwriting an existing artifact should warn on stderr."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        artifact_dir = tmp_path / 'existing_artifact'
        artifact_dir.mkdir()
        (artifact_dir / 'artifact.json').write_text('{}')  # pre-existing

        class FakeResult:
            cv_metric_mean = 0.733
            cv_metric_std = 0.051

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                self.X = X

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                return FakeResult()

            def save_artifact(self, model, path, *, step, time_col, event_col):
                pass

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'train', str(data_path),
            '--model', 'coxph',
            '--save-artifact', str(artifact_dir),
        ])

        assert rc == 0
        stderr = capsys.readouterr().err
        assert "Overwriting existing artifact" in stderr

    def test_check_feature_compatibility_missing_raises(self, capsys):
        """Missing features should raise ValueError."""
        from mlsurv.cli import _check_feature_compatibility

        metadata = {'data_contract': {'feature_names': ['a', 'b', 'c']}}
        df = pd.DataFrame({'b': [1], 'c': [2], 'd': [3]})
        with pytest.raises(ValueError, match="missing features"):
            _check_feature_compatibility(metadata, df)

    def test_check_feature_compatibility_extra_warns(self, capsys):
        """Extra features should warn but not raise."""
        from mlsurv.cli import _check_feature_compatibility

        metadata = {'data_contract': {'feature_names': ['b']}}
        df = pd.DataFrame({'b': [1], 'c': [2], 'd': [3]})
        _check_feature_compatibility(metadata, df)
        stderr = capsys.readouterr().err
        assert "extra features" in stderr.lower()

    def test_check_feature_compatibility_null_data_contract(self, capsys):
        """null data_contract should warn about missing contract."""
        from mlsurv.cli import _check_feature_compatibility

        _check_feature_compatibility(
            {'data_contract': None},
            pd.DataFrame({'f1': [1]}),
        )
        stderr = capsys.readouterr().err
        assert "data_contract" in stderr.lower()

    def test_load_setup_only_artifact_raises(self, monkeypatch, tmp_path):
        """Loading a setup-only artifact (no model) should raise ValueError."""
        import json
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        artifact_dir = tmp_path / 'setup_artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'step': 'setup',
            'model_name': None,
            'data_contract': {'feature_names': ['f1', 'f2']},
        }
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

        def fake_load_artifact(path):
            return metadata, None, None  # No model

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        # Should fail with exit code 1 (caught by main's except handler)
        rc = main([
            'evaluate', str(data_path),
            '--model', 'coxph',
            '--load-artifact', str(artifact_dir),
        ])
        assert rc == 3  # EXIT_DATA (ValueError: artifact has no model)

    def test_check_feature_compatibility_no_data_contract(self, capsys):
        """Missing data_contract key should warn."""
        from mlsurv.cli import _check_feature_compatibility

        _check_feature_compatibility({}, pd.DataFrame({'f1': [1], 'f2': [2]}))
        stderr = capsys.readouterr().err
        assert "data_contract" in stderr.lower()

    def test_check_feature_compatibility_no_feature_names_warns(self, capsys):
        """data_contract with no feature_names should warn."""
        from mlsurv.cli import _check_feature_compatibility

        _check_feature_compatibility(
            {'data_contract': {}},
            pd.DataFrame({'f1': [1]}),
        )
        stderr = capsys.readouterr().err
        assert "feature_names" in stderr.lower()

    def test_check_feature_compatibility_matching_features(self, capsys):
        """No warning when features match exactly (order-independent)."""
        from mlsurv.cli import _check_feature_compatibility

        metadata = {'data_contract': {'feature_names': ['f2', 'f1']}}
        _check_feature_compatibility(metadata, pd.DataFrame({'f1': [1], 'f2': [2]}))
        assert capsys.readouterr().err == ""

    # --- _extract_feature_names helper ---

    def test_extract_feature_names_valid(self):
        from mlsurv.cli._artifacts import _extract_feature_names

        metadata = {'data_contract': {'feature_names': ['a', 'b']}}
        names, error = _extract_feature_names(metadata)
        assert names == ['a', 'b']
        assert error is None

    def test_extract_feature_names_no_contract(self):
        from mlsurv.cli._artifacts import _extract_feature_names

        names, error = _extract_feature_names({})
        assert names is None
        assert 'data_contract' in error

    def test_extract_feature_names_no_feature_names(self):
        from mlsurv.cli._artifacts import _extract_feature_names

        names, error = _extract_feature_names({'data_contract': {}})
        assert names is None
        assert 'feature_names' in error

    def test_extract_feature_names_null_contract(self):
        from mlsurv.cli._artifacts import _extract_feature_names

        names, error = _extract_feature_names({'data_contract': None})
        assert names is None
        assert error is not None

    # --- cmd_predict artifact error handling ---

    def test_cmd_predict_missing_artifact_returns_1(self, tmp_path, capsys):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'data.csv')
        rc = main([
            'predict', str(data_path),
            '--load-artifact', str(tmp_path / 'nonexistent'),
        ])
        assert rc == 3  # EXIT_DATA
        assert 'Error' in capsys.readouterr().err

    def test_prediction_mode_missing_features_raises(self, tmp_path):
        from mlsurv.utils import load_survival_data

        data_path = _write_sample_csv(tmp_path / 'predict.csv')

        with pytest.raises(ValueError, match="missing required features"):
            load_survival_data(str(data_path), feature_names=['f1', 'missing'])

    def test_prediction_mode_empty_feature_names_raises(self, tmp_path):
        from mlsurv.utils import load_survival_data

        data_path = _write_sample_csv(tmp_path / 'predict.csv')

        with pytest.raises(ValueError, match="at least one feature column"):
            load_survival_data(str(data_path), feature_names=[])

    def test_load_artifact_prints_cv_metrics(self, monkeypatch, tmp_path, capsys):
        """Loading artifact should print stored CV metrics to stderr."""
        import json
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        artifact_dir = tmp_path / 'cv_note_artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2']},
            'results': {
                'cv_metric_mean': 0.700,
                'cv_metric_std': 0.050,
            },
        }
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))

        class FakeEvalResult:
            concordance = pd.DataFrame({'c_index_test': [0.700]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                raise AssertionError("train() should not be called")

            def evaluate(self, model_result, **kwargs):
                return FakeEvalResult()

        loaded_sentinel = object()

        def fake_load_artifact(path):
            return metadata, loaded_sentinel, None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'evaluate', str(data_path),
            '--model', 'coxph',
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 0
        stderr = capsys.readouterr().err
        assert "Loaded artifact 'coxph'" in stderr
        assert "CV 0.7000" in stderr

    def test_validate_load_artifact_uses_loaded_model(self, monkeypatch, tmp_path):
        """validate --load-artifact should use the loaded model without training."""
        import mlsurv
        from mlsurv.cli import main

        train_path = _write_sample_csv(tmp_path / 'train.csv')
        val_path = _write_sample_csv(tmp_path / 'val' / 'val.csv')
        seen = {}

        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2']},
            'results': {
                'cv_metric_mean': 0.700,
                'cv_metric_std': 0.050,
            },
        }

        class FakeValidationResult:
            concordance = pd.DataFrame({'c_index_test': [0.702]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                seen['train_called'] = True
                raise AssertionError("train() should not be called")

            def validate(self, X_val, y_val, models=None, name='External', **kwargs):
                seen['validate_model'] = models.model if models else None
                return FakeValidationResult()

        loaded_sentinel = object()

        def fake_load_artifact(path):
            seen['load_artifact_called'] = True
            return metadata, loaded_sentinel, None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'validate', str(train_path),
            '--validation-data', str(val_path),
            '--model', 'coxph',
            '--load-artifact', str(tmp_path / 'fake'),
        ])

        assert rc == 0
        assert seen['load_artifact_called']
        assert 'train_called' not in seen
        assert seen['validate_model'] is loaded_sentinel

    def test_load_artifact_skips_training(self, monkeypatch, tmp_path, capsys):
        """--load-artifact should never call learner.train()."""
        import json
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        artifact_dir = tmp_path / 'skip_train_artifact'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'rsf',
            'data_contract': {'feature_names': ['f1', 'f2']},
            'results': {
                'cv_metric_mean': 0.820,
                'cv_metric_std': 0.030,
                'is_tuned': True,
                'cv_metric_scores': [0.79, 0.82, 0.85],
            },
            'config': {'scoring': 'harrell'},
        }
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))

        class FakeBootstrapResult:
            summary = pd.DataFrame({'metric': ['harrell'], 'mean': [0.81]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name):
                raise AssertionError("train() must not be called with --load-artifact")

            def bootstrap(self, model_result, **kwargs):
                return FakeBootstrapResult()

        loaded_sentinel = object()

        def fake_load_artifact(path):
            return metadata, loaded_sentinel, None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'bootstrap', str(data_path),
            '--model', 'rsf',
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 0
        stderr = capsys.readouterr().err
        assert "Loaded artifact 'rsf'" in stderr
        assert "(tuned)" in stderr

    def test_save_artifact_uses_learner_state_col_names(self, tmp_path):
        """_save_artifact_if_requested should prefer learner._state column names."""
        from types import SimpleNamespace
        from mlsurv.cli._artifacts import _save_artifact_if_requested

        seen = {}

        class FakeState:
            time_col = 'OS_MONTHS'
            event_col = 'OS_STATUS'

        class FakeLearner(_FakeLearnerBase):
            _state = FakeState()

            def save_artifact(self, model, path, *, step, time_col, event_col):
                seen['time_col'] = time_col
                seen['event_col'] = event_col
                Path(path).mkdir(parents=True, exist_ok=True)

        args = SimpleNamespace(
            save_artifact=str(tmp_path / 'art'),
            time_col='time',
            event_col='event',
            quiet=False,
        )

        _save_artifact_if_requested(FakeLearner(), None, args, step='evaluate')

        assert seen['time_col'] == 'OS_MONTHS'
        assert seen['event_col'] == 'OS_STATUS'

    def test_save_artifact_falls_back_to_args_when_state_is_none(self, tmp_path):
        """When learner._state has no column names, fall back to args."""
        from types import SimpleNamespace
        from mlsurv.cli._artifacts import _save_artifact_if_requested

        seen = {}

        class FakeState:
            time_col = None
            event_col = None

        class FakeLearner(_FakeLearnerBase):
            _state = FakeState()

            def save_artifact(self, model, path, *, step, time_col, event_col):
                seen['time_col'] = time_col
                seen['event_col'] = event_col
                Path(path).mkdir(parents=True, exist_ok=True)

        args = SimpleNamespace(
            save_artifact=str(tmp_path / 'art'),
            time_col='survival_time',
            event_col='status',
            quiet=False,
        )

        _save_artifact_if_requested(FakeLearner(), None, args, step='train')

        assert seen['time_col'] == 'survival_time'
        assert seen['event_col'] == 'status'


class TestCLIArtifactInspect:
    """Tests for artifact-inspect CLI command."""

    def _make_artifact(self, tmp_path, metadata):
        """Helper to create a fake artifact directory with metadata."""
        artifact_dir = tmp_path / 'artifact'
        artifact_dir.mkdir()
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))
        return artifact_dir

    def test_inspect_full_artifact(self, tmp_path, capsys):
        """artifact-inspect should display a human-readable summary."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {
            'artifact_version': 1,
            'step': 'train',
            'model_name': 'coxph',
            'timestamp': '2026-02-24T00:00:00+00:00',
            'environment': {
                'mlsurv_version': '0.1.0',
                'python_version': '3.11.0',
            },
            'data_contract': {'feature_names': ['f1', 'f2', 'f3']},
            'results': {
                'cv_metric_mean': 0.733,
                'cv_metric_std': 0.051,
                'scoring_metric': 'harrell',
            },
        })

        rc = main(['artifact-inspect', str(artifact_dir)])

        assert rc == 0
        out = capsys.readouterr().out
        assert 'version:' in out
        assert 'coxph' in out
        assert 'train' in out
        assert '0.7330' in out
        assert 'features:   3' in out

    def test_inspect_json_output(self, tmp_path, capsys):
        """artifact-inspect --json should dump raw JSON."""
        from mlsurv.cli import main

        metadata = {'artifact_version': 1, 'step': 'tune', 'model_name': 'rsf'}
        artifact_dir = self._make_artifact(tmp_path, metadata)

        rc = main(['artifact-inspect', str(artifact_dir), '--json'])

        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed['step'] == 'tune'
        assert parsed['model_name'] == 'rsf'

    def test_inspect_missing_directory(self, tmp_path, capsys):
        """artifact-inspect on missing dir should error."""
        from mlsurv.cli import main

        rc = main(['artifact-inspect', str(tmp_path / 'nonexistent')])

        assert rc == 3  # EXIT_DATA
        assert 'not found' in capsys.readouterr().err.lower()

    def test_inspect_missing_metadata(self, tmp_path, capsys):
        """artifact-inspect on dir without artifact.json should error."""
        from mlsurv.cli import main

        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()

        rc = main(['artifact-inspect', str(empty_dir)])

        assert rc == 3  # EXIT_DATA
        assert 'artifact.json' in capsys.readouterr().err

    def test_inspect_corrupt_json(self, tmp_path, capsys):
        """artifact-inspect on corrupt JSON should error."""
        from mlsurv.cli import main

        artifact_dir = tmp_path / 'corrupt'
        artifact_dir.mkdir()
        (artifact_dir / 'artifact.json').write_text('{bad json}')

        rc = main(['artifact-inspect', str(artifact_dir)])

        assert rc == 3  # EXIT_DATA
        assert 'corrupt' in capsys.readouterr().err.lower()

    def test_inspect_pre_versioned_warns(self, tmp_path, capsys):
        """Pre-versioned artifact should show warning about missing version."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {
            'step': 'train', 'model_name': 'coxph',
        })

        rc = main(['artifact-inspect', str(artifact_dir)])

        assert rc == 0
        stderr = capsys.readouterr().err
        assert 'artifact_version' in stderr.lower() or 'pre-versioned' in stderr.lower()

    def test_inspect_no_data_contract_warns(self, tmp_path, capsys):
        """Artifact without data_contract should show warning."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {
            'artifact_version': 1,
            'step': 'train',
            'model_name': 'coxph',
        })

        rc = main(['artifact-inspect', str(artifact_dir)])

        assert rc == 0
        stderr = capsys.readouterr().err
        assert 'data_contract' in stderr.lower()


class TestCLIArtifactUpgrade:
    """Tests for artifact-upgrade CLI command."""

    def _make_artifact(self, tmp_path, metadata, name='artifact'):
        """Helper to create a fake artifact directory with metadata."""
        artifact_dir = tmp_path / name
        artifact_dir.mkdir()
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))
        return artifact_dir

    def test_upgrade_pre_versioned(self, tmp_path, capsys):
        """Pre-versioned artifact should be upgraded with all stubs."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {
            'step': 'train', 'model_name': 'coxph',
        })

        rc = main(['artifact-upgrade', str(artifact_dir)])

        assert rc == 0
        with open(artifact_dir / 'artifact.json') as f:
            upgraded = json.load(f)
        assert upgraded['artifact_version'] == 1
        assert 'data_contract' in upgraded
        assert 'environment' in upgraded
        assert 'results' in upgraded
        assert 'timestamp' in upgraded

    def test_upgrade_already_current(self, tmp_path, capsys):
        """Already-current artifact should report no changes."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {
            'artifact_version': 1,
            'step': 'train',
            'model_name': 'coxph',
            'timestamp': '2026-02-24T00:00:00+00:00',
            'environment': {},
            'data_contract': {},
            'results': {},
        })

        rc = main(['artifact-upgrade', str(artifact_dir)])

        assert rc == 0
        out = capsys.readouterr().out
        assert 'already up to date' in out.lower()

    def test_upgrade_dry_run(self, tmp_path, capsys):
        """--dry-run should show changes without writing."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {
            'step': 'train', 'model_name': 'coxph',
        })
        original_text = (artifact_dir / 'artifact.json').read_text()

        rc = main(['artifact-upgrade', str(artifact_dir), '--dry-run'])

        assert rc == 0
        out = capsys.readouterr().out
        assert 'dry run' in out.lower()
        # File should not have been modified
        assert (artifact_dir / 'artifact.json').read_text() == original_text

    def test_upgrade_output_to_new_dir(self, tmp_path, capsys):
        """--output should copy artifact to new dir and upgrade there."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {
            'step': 'train', 'model_name': 'coxph',
        })
        # Add a model file to verify copytree
        (artifact_dir / 'model.joblib').write_bytes(b'fake-model')
        output_dir = tmp_path / 'upgraded'

        rc = main(['artifact-upgrade', str(artifact_dir), '--output', str(output_dir)])

        assert rc == 0
        # Source should be untouched
        with open(artifact_dir / 'artifact.json') as f:
            source = json.load(f)
        assert 'artifact_version' not in source

        # Output should be upgraded
        with open(output_dir / 'artifact.json') as f:
            upgraded = json.load(f)
        assert upgraded['artifact_version'] == 1
        # Model file should be copied
        assert (output_dir / 'model.joblib').exists()

    def test_upgrade_output_exists_error(self, tmp_path, capsys):
        """--output to existing dir should error."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {'step': 'train'})
        output_dir = tmp_path / 'existing'
        output_dir.mkdir()

        rc = main(['artifact-upgrade', str(artifact_dir), '--output', str(output_dir)])

        assert rc == 1  # EXIT_ERROR
        assert 'already exists' in capsys.readouterr().err.lower()

    def test_upgrade_missing_dir(self, tmp_path, capsys):
        """Upgrade on missing dir should error."""
        from mlsurv.cli import main

        rc = main(['artifact-upgrade', str(tmp_path / 'nonexistent')])

        assert rc == 3  # EXIT_DATA
        assert 'not found' in capsys.readouterr().err.lower()

    def test_upgrade_preserves_existing_values(self, tmp_path):
        """Upgrade should not overwrite existing fields."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {
            'step': 'train',
            'model_name': 'coxph',
            'data_contract': {'feature_names': ['f1', 'f2']},
        })

        rc = main(['artifact-upgrade', str(artifact_dir)])

        assert rc == 0
        with open(artifact_dir / 'artifact.json') as f:
            upgraded = json.load(f)
        # Original data_contract preserved
        assert upgraded['data_contract'] == {'feature_names': ['f1', 'f2']}
        # New stubs added
        assert upgraded['artifact_version'] == 1
        assert 'environment' in upgraded

    def test_upgrade_corrupt_json(self, tmp_path, capsys):
        """artifact-upgrade on corrupt JSON should error."""
        from mlsurv.cli import main

        artifact_dir = tmp_path / 'corrupt'
        artifact_dir.mkdir()
        (artifact_dir / 'artifact.json').write_text('{bad json}')

        rc = main(['artifact-upgrade', str(artifact_dir)])

        assert rc == 3  # EXIT_DATA
        assert 'corrupt' in capsys.readouterr().err.lower()

    def test_upgrade_missing_metadata(self, tmp_path, capsys):
        """artifact-upgrade on dir without artifact.json should error."""
        from mlsurv.cli import main

        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()

        rc = main(['artifact-upgrade', str(empty_dir)])

        assert rc == 3  # EXIT_DATA
        assert 'artifact.json' in capsys.readouterr().err

    def test_upgrade_fixes_invalid_version(self, tmp_path):
        """Upgrade should fix artifact_version=null or other invalid values."""
        from mlsurv.cli import main

        artifact_dir = self._make_artifact(tmp_path, {
            'artifact_version': None,
            'step': 'train',
        })

        rc = main(['artifact-upgrade', str(artifact_dir)])

        assert rc == 0
        with open(artifact_dir / 'artifact.json') as f:
            upgraded = json.load(f)
        assert upgraded['artifact_version'] == 1


class TestUpgradeArtifactMetadata:
    """Unit tests for _upgrade_artifact_metadata()."""

    def test_empty_metadata_gets_all_stubs(self):
        """Empty dict should get all required fields."""
        from mlsurv.cli import _upgrade_artifact_metadata

        metadata = {}
        changes = _upgrade_artifact_metadata(metadata)

        assert len(changes) == 5
        assert metadata['artifact_version'] == 1
        assert metadata['timestamp'] is None
        assert metadata['environment'] == {}
        assert metadata['data_contract'] == {}
        assert metadata['results'] == {}

    def test_full_metadata_no_changes(self):
        """Complete metadata should produce no changes."""
        from mlsurv.cli import _upgrade_artifact_metadata

        metadata = {
            'artifact_version': 1,
            'timestamp': '2026-02-24T00:00:00+00:00',
            'environment': {'mlsurv_version': '0.1.0'},
            'data_contract': {'feature_names': ['f1']},
            'results': {'cv_metric_mean': 0.7},
        }
        changes = _upgrade_artifact_metadata(metadata)

        assert changes == []

    def test_preserves_existing_values(self):
        """Upgrade should not overwrite existing non-missing fields."""
        from mlsurv.cli import _upgrade_artifact_metadata

        metadata = {
            'data_contract': {'feature_names': ['f1', 'f2']},
            'results': {'cv_metric_mean': 0.733},
        }
        _upgrade_artifact_metadata(metadata)

        assert metadata['data_contract'] == {'feature_names': ['f1', 'f2']}
        assert metadata['results'] == {'cv_metric_mean': 0.733}
        assert metadata['artifact_version'] == 1

    @pytest.mark.parametrize("bad_version", [None, 0, 'v1', -1])
    def test_fixes_invalid_version(self, bad_version):
        """Invalid artifact_version values should be fixed to current."""
        from mlsurv.cli import _upgrade_artifact_metadata

        metadata = {'artifact_version': bad_version}
        changes = _upgrade_artifact_metadata(metadata)
        assert metadata['artifact_version'] == 1
        assert any('invalid' in c.lower() or 'fixed' in c.lower() for c in changes)

    def test_future_version_warns_without_downgrading(self):
        """Versions newer than supported should warn, not downgrade."""
        from mlsurv.cli import _upgrade_artifact_metadata
        from mlsurv.constants import ARTIFACT_VERSION

        future_version = ARTIFACT_VERSION + 1
        metadata = {'artifact_version': future_version}
        changes = _upgrade_artifact_metadata(metadata)
        # Version should NOT be changed
        assert metadata['artifact_version'] == future_version
        assert any('newer' in c.lower() or 'upgrade mlsurv' in c.lower() for c in changes)


class TestCLIArtifactContractsNoDataContract:
    """Test that loading an artifact with no data_contract warns."""

    def test_no_data_contract_warns_on_load(self, monkeypatch, tmp_path, capsys):
        """evaluate --load-artifact with no data_contract should warn."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        artifact_dir = tmp_path / 'no_contract'
        artifact_dir.mkdir()
        metadata = {
            'artifact_version': 1,
            'model_name': 'coxph',
            'results': {
                'cv_metric_mean': 0.733,
                'cv_metric_std': 0.051,
            },
        }
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))

        class FakeEvalResult:
            concordance = pd.DataFrame({'c_index_test': [0.719]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def evaluate(self, model_result, **kwargs):
                return FakeEvalResult()

        loaded_sentinel = object()

        def fake_load_artifact(path):
            return metadata, loaded_sentinel, None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.load_artifact',
            staticmethod(fake_load_artifact),
        )

        rc = main([
            'evaluate', str(data_path),
            '--model', 'coxph',
            '--load-artifact', str(artifact_dir),
        ])

        assert rc == 0
        stderr = capsys.readouterr().err
        assert 'data_contract' in stderr.lower()


@pytest.mark.integration
class TestCLIArtifactPipeline:
    """End-to-end CLI artifact pipeline tests."""

    @pytest.fixture
    def sample_data_file(self, tmp_path, synthetic_data_clean):
        X, y = synthetic_data_clean
        df = X.copy()
        df['time'] = y['time']
        df['event'] = y['event'].astype(int)
        filepath = tmp_path / 'data.csv'
        df.to_csv(filepath, index=False)
        return filepath

    def test_train_save_then_evaluate_load(self, sample_data_file, tmp_path):
        """train --save-artifact -> evaluate --load-artifact end-to-end."""
        import json
        from mlsurv.cli import main

        artifact_dir = tmp_path / 'artifact'

        # Train and save
        rc = main([
            'train', str(sample_data_file),
            '--model', 'coxph', '--cv-folds', '3',
            '--save-artifact', str(artifact_dir),
        ])
        assert rc == 0
        assert (artifact_dir / 'artifact.json').exists()
        assert (artifact_dir / 'model.joblib').exists()

        # Verify artifact metadata content
        with open(artifact_dir / 'artifact.json') as f:
            metadata = json.load(f)
        assert metadata['artifact_version'] == 1
        assert metadata['step'] == 'train'
        assert metadata['model_name'] == 'coxph'
        assert 'feature_names' in metadata['data_contract']
        assert 'mlsurv_version' in metadata['environment']

        # Evaluate with loaded artifact
        rc = main([
            'evaluate', str(sample_data_file),
            '--model', 'coxph', '--cv-folds', '3',
            '--load-artifact', str(artifact_dir),
        ])
        assert rc == 0

    def test_train_save_then_bootstrap_load(self, sample_data_file, tmp_path):
        """train --save-artifact -> bootstrap --load-artifact end-to-end."""
        from mlsurv.cli import main

        artifact_dir = tmp_path / 'artifact'

        # Train and save
        rc = main([
            'train', str(sample_data_file),
            '--model', 'coxph', '--cv-folds', '3',
            '--save-artifact', str(artifact_dir),
        ])
        assert rc == 0

        # Bootstrap with loaded artifact
        rc = main([
            'bootstrap', str(sample_data_file),
            '--model', 'coxph', '--cv-folds', '3',
            '--n-bootstrap', '10',
            '--load-artifact', str(artifact_dir),
        ])
        assert rc == 0


class TestCLICompareFlag:
    """Tests for ``mlsurv run --compare`` and ``--compare-output``."""

    def _make_fakes(self, *, has_bootstrap: bool = True, compare_raises: bool = False):
        """Build fake learner objects for monkeypatching (dict-of-ModelResult API)."""
        comp_df = pd.DataFrame({
            'model_a': ['coxph'],
            'model_b': ['rsf'],
            'metric': ['harrell'],
            'method': ['paired_bootstrap'],
            'delta': [0.05],
            'ci_lower': [0.01],
            'ci_upper': [0.09],
            'p_value': [0.02],
            'p_adjusted': [0.02],
        })

        class FakeBootstrapResult:
            def compare(self, metric='harrell', **kwargs):
                if compare_raises:
                    raise ValueError("Need at least 2 models to compare")
                return comp_df

        class FakeEvalResult:
            concordance = pd.DataFrame({'c_index_test': [0.712]})

        class FakeModelResult:
            def __init__(self, name, cv_mean, cv_std, evaluation=None):
                self.name = name
                self.cv_metric_mean = cv_mean
                self.cv_metric_std = cv_std
                self.evaluation = evaluation

        _boot = FakeBootstrapResult() if has_bootstrap else None

        class FakeLearner(_FakeLearnerBase):
            scoring = 'harrell'

            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def run(self, **kwargs):
                # run() now returns dict of {str: ModelResult}
                return {
                    'coxph': FakeModelResult('coxph', 0.69, 0.04, FakeEvalResult()),
                    'rsf': FakeModelResult('rsf', 0.74, 0.03, FakeEvalResult()),
                }

            @property
            def best_model(self):
                return FakeModelResult('rsf', 0.74, 0.03, FakeEvalResult())

            @property
            def bootstrap_result(self):
                return _boot

            @property
            def evaluation_result(self):
                return FakeEvalResult()

            def generate_report(self, output_path, **kwargs):
                Path(output_path).write_text('<html>ok</html>')

        return FakeLearner, comp_df

    def test_run_compare_verbose_prints_comparison(self, monkeypatch, tmp_path, capsys):
        """``--compare --verbose`` prints paired comparison table to stdout."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        FakeLearner, _ = self._make_fakes(has_bootstrap=True)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main(['run', str(data_path), '--compare', '--verbose'])
        assert rc == 0

        captured = capsys.readouterr().out
        assert 'coxph' in captured
        assert 'rsf' in captured

    def test_run_compare_without_verbose_suppresses(self, monkeypatch, tmp_path, capsys):
        """``--compare`` without ``--verbose`` does not print comparison to stdout."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        FakeLearner, _ = self._make_fakes(has_bootstrap=True)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main(['run', str(data_path), '--compare'])
        assert rc == 0

        captured = capsys.readouterr().out
        assert 'Paired Model Comparison' not in captured

    def test_run_compare_output_saves_csv(self, monkeypatch, tmp_path):
        """``--compare-output FILE`` writes comparison DataFrame to CSV."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        csv_path = tmp_path / 'compare.csv'
        FakeLearner, expected_df = self._make_fakes(has_bootstrap=True)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main(['run', str(data_path), '--compare-output', str(csv_path)])
        assert rc == 0
        assert csv_path.exists()

        saved = pd.read_csv(csv_path)
        assert list(saved.columns) == list(expected_df.columns)
        assert len(saved) == 1
        assert saved['model_a'].iloc[0] == 'coxph'

    def test_run_compare_no_bootstrap_warns(self, monkeypatch, tmp_path, capsys):
        """``--compare`` without bootstrap warns to stderr."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        FakeLearner, _ = self._make_fakes(has_bootstrap=False)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main(['run', str(data_path), '--compare'])
        assert rc == 0

        captured = capsys.readouterr()
        assert 'Paired Model Comparison' not in captured.out
        assert '--compare skipped' in captured.err

    def test_run_compare_output_without_verbose(self, monkeypatch, tmp_path):
        """``--compare-output`` works without ``--verbose`` (file write is independent)."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        csv_path = tmp_path / 'compare.csv'
        FakeLearner, _ = self._make_fakes(has_bootstrap=True)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main(['run', str(data_path), '--compare-output', str(csv_path)])
        assert rc == 0
        assert csv_path.exists()

    def test_run_compare_exception_emits_warning(self, monkeypatch, tmp_path, capsys):
        """Exception in compare() emits stderr warning but still returns 0."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        FakeLearner, _ = self._make_fakes(has_bootstrap=True, compare_raises=True)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main(['run', str(data_path), '--compare'])
        assert rc == 0

        captured = capsys.readouterr()
        assert 'Paired comparison skipped' in captured.err

    def test_run_compare_output_no_bootstrap_warns(self, monkeypatch, tmp_path, capsys):
        """``--compare-output`` with no bootstrap warns and creates no file."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        csv_path = tmp_path / 'compare.csv'
        FakeLearner, _ = self._make_fakes(has_bootstrap=False)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main(['run', str(data_path), '--compare-output', str(csv_path)])
        assert rc == 0
        assert not csv_path.exists()

        captured = capsys.readouterr()
        assert '--compare-output skipped' in captured.err


# Shared fake-learner configs for the run/tune experiment-name + resume tests.
# Safe to share across parametrized cases only because no test mutates them; if a
# future test mutates the model result in place, give it its own _Obj instead.
_RUN_MODEL_RESULT = _Obj(
    name='coxph', cv_metric_mean=0.70, cv_metric_std=0.04, evaluation=None,
)
_RUN_BUILD = dict(
    run_return={'coxph': _RUN_MODEL_RESULT},
    best_model=_RUN_MODEL_RESULT,
    bootstrap_result=None,
    evaluation_result=None,
    scoring='harrell',
)
_TUNE_BUILD = dict(
    tune_return=_Obj(name='rsf', cv_metric_mean=0.75, cv_metric_std=0.03,
                     best_params_={}),
)


class TestCLIExperimentName:
    """Tests for ``--experiment-name`` on ``run`` and ``tune``."""

    def test_parser_run_experiment_name_default(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['run', 'test.csv'])
        assert args.experiment_name is None

    def test_parser_run_experiment_name_set(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['run', 'test.csv', '--experiment-name', 'foo'])
        assert args.experiment_name == 'foo'

    def test_parser_tune_experiment_name_set(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'tune', 'test.csv', '--model', 'rsf', '--experiment-name', 'bar',
        ])
        assert args.experiment_name == 'bar'

    @pytest.mark.parametrize(
        'command,argv_tail,build_kwargs,seen_key,expected_name',
        [
            pytest.param(
                'run', ['--models', 'coxph', '--experiment-name', 'my_exp'],
                _RUN_BUILD, 'run_kwargs', 'my_exp', id='run',
            ),
            pytest.param(
                'tune', ['--model', 'rsf', '--experiment-name', 'tune_exp'],
                _TUNE_BUILD, 'tune_kwargs', 'tune_exp', id='tune',
            ),
        ],
    )
    def test_cmd_passes_experiment_name(
        self, monkeypatch, tmp_path, make_fake_learner,
        command, argv_tail, build_kwargs, seen_key, expected_name,
    ):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}
        monkeypatch.setattr(
            mlsurv, 'SurvivalLearner', make_fake_learner(seen, **build_kwargs),
        )

        rc = main([command, str(data_path), *argv_tail])
        assert rc == 0
        assert seen[seen_key]['experiment_name'] == expected_name

    def test_cmd_run_forwards_no_tune_and_n_trials(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        """``--no-tune`` / ``--n-trials`` reach ``learner.run(tune=, n_trials=)``."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}
        monkeypatch.setattr(
            mlsurv, 'SurvivalLearner', make_fake_learner(seen, **_RUN_BUILD),
        )

        rc = main([
            'run', str(data_path), '--models', 'coxph',
            '--no-tune', '--n-trials', '7',
        ])
        assert rc == 0
        assert seen['run_kwargs']['tune'] is False
        assert seen['run_kwargs']['n_trials'] == 7

    def test_cmd_run_tune_defaults_forwarded(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        """Without flags, ``run`` forwards ``tune=True`` and ``n_trials='auto'``."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}
        monkeypatch.setattr(
            mlsurv, 'SurvivalLearner', make_fake_learner(seen, **_RUN_BUILD),
        )

        rc = main(['run', str(data_path), '--models', 'coxph'])
        assert rc == 0
        assert seen['run_kwargs']['tune'] is True
        assert seen['run_kwargs']['n_trials'] == 'auto'

    @pytest.mark.parametrize(
        'command,argv_tail,build_kwargs',
        [
            pytest.param('run', ['--models', 'coxph', '--resume'], _RUN_BUILD, id='run'),
            pytest.param('tune', ['--model', 'rsf', '--resume'], _TUNE_BUILD, id='tune'),
        ],
    )
    def test_cmd_resume_without_experiment_name_warns(
        self, monkeypatch, tmp_path, capsys, make_fake_learner,
        command, argv_tail, build_kwargs,
    ):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}
        monkeypatch.setattr(
            mlsurv, 'SurvivalLearner', make_fake_learner(seen, **build_kwargs),
        )

        rc = main([command, str(data_path), *argv_tail])
        assert rc == 0

        captured = capsys.readouterr()
        assert '--resume without --experiment-name' in captured.err


# ---------------------------------------------------------------------------
# Learner serialization (--save-learner / --load-learner)
# ---------------------------------------------------------------------------


class TestCLILearnerArgs:
    """Parser-level tests for --save-learner / --load-learner flags."""

    @pytest.mark.parametrize('subcommand,argv_extra', [
        ('run', ['test.csv', '--save-learner', '/tmp/s.learner']),
        ('train', ['test.csv', '--model', 'coxph', '--save-learner', '/tmp/s.learner']),
        ('tune', ['test.csv', '--model', 'rsf', '--save-learner', '/tmp/s.learner']),
        ('evaluate', ['test.csv', '--model', 'coxph', '--save-learner', '/tmp/s.learner']),
        ('bootstrap', ['test.csv', '--model', 'coxph', '--save-learner', '/tmp/s.learner']),
        ('validate', [
            'test.csv', '--validation-data', 'ext.csv',
            '--model', 'coxph', '--save-learner', '/tmp/s.learner',
        ]),
        ('benchmark', [
            'test.csv', '--reference-file', 'ref.csv',
            '--save-learner', '/tmp/s.learner',
        ]),
        ('setup', ['test.csv', '--save-learner', '/tmp/s.learner']),
    ])
    def test_subcommand_accepts_save_learner(self, subcommand, argv_extra):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([subcommand] + argv_extra)
        assert args.command == subcommand
        assert args.save_learner == '/tmp/s.learner'

    @pytest.mark.parametrize('subcommand,argv_extra', [
        ('evaluate', ['--load-learner', 'foo.learner', '--model', 'coxph']),
        ('bootstrap', ['--load-learner', 'foo.learner', '--model', 'coxph']),
        ('validate', [
            '--load-learner', 'foo.learner',
            '--validation-data', 'ext.csv', '--model', 'coxph',
        ]),
    ])
    def test_subcommand_accepts_load_learner(self, subcommand, argv_extra):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([subcommand] + argv_extra)
        assert args.command == subcommand
        assert args.load_learner == 'foo.learner'
        assert args.data is None

    def test_train_does_not_have_load_learner(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'train', 'test.csv', '--model', 'coxph',
                '--load-learner', 'foo.learner',
            ])

    def test_run_does_not_have_load_learner(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'run', 'test.csv', '--load-learner', 'foo.learner',
            ])

    def test_data_still_required_for_train(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['train', '--model', 'coxph'])

    def test_data_optional_for_evaluate_with_load_learner(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'evaluate', '--load-learner', 'foo.learner', '--model', 'coxph',
        ])
        assert args.data is None
        assert args.load_learner == 'foo.learner'


class TestCLILearnerValidation:
    """Validation tests for --load-learner mutual exclusivity."""

    def test_load_learner_and_load_artifact_errors(self, tmp_path):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'data.csv')
        with pytest.raises(SystemExit):
            main([
                'evaluate', str(data_path), '--model', 'coxph',
                '--load-learner', 'foo.learner',
                '--load-artifact', '/tmp/art',
            ])

    def test_load_learner_and_data_file_errors(self, tmp_path):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'data.csv')
        with pytest.raises(SystemExit):
            main([
                'evaluate', str(data_path), '--model', 'coxph',
                '--load-learner', 'foo.learner',
            ])

    def test_no_data_no_load_learner_errors(self):
        from mlsurv.cli import main

        with pytest.raises(SystemExit):
            main(['evaluate', '--model', 'coxph'])


class TestCLILearnerContracts:
    """Integration tests for save/load learner round-trips."""

    @pytest.fixture
    def sample_data_file(self, tmp_path, synthetic_data_clean):
        X, y = synthetic_data_clean
        df = X.copy()
        df['time'] = y['time']
        df['event'] = y['event'].astype(int)
        filepath = tmp_path / 'data.csv'
        df.to_csv(filepath, index=False)
        return filepath

    def test_train_save_learner_creates_file(self, sample_data_file, tmp_path):
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        rc = main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        assert rc == 0
        assert learner_path.exists()
        assert learner_path.stat().st_size > 0

    def test_train_then_evaluate_roundtrip(self, sample_data_file, tmp_path):
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        rc = main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        assert rc == 0

        rc = main([
            'evaluate', '--load-learner', str(learner_path),
            '--model', 'coxph', '--n-importance-repeats', '0',
        ])
        assert rc == 0

    def test_train_then_bootstrap_roundtrip(self, sample_data_file, tmp_path):
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        rc = main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        assert rc == 0

        rc = main([
            'bootstrap', '--load-learner', str(learner_path),
            '--model', 'coxph', '--n-bootstrap', '3',
        ])
        assert rc == 0

    def test_save_learner_overwrite_warns(self, sample_data_file, tmp_path, capsys):
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        # First save
        main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        capsys.readouterr()  # clear

        # Second save (should warn about overwrite)
        main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        captured = capsys.readouterr()
        assert 'Overwriting' in captured.err

    def test_save_learner_and_save_artifact_both_work(
        self, sample_data_file, tmp_path,
    ):
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        artifact_dir = tmp_path / 'artifact'
        rc = main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2',
            '--save-learner', str(learner_path),
            '--save-artifact', str(artifact_dir),
        ])
        assert rc == 0
        assert learner_path.exists()
        assert artifact_dir.is_dir()

    def test_load_learner_model_not_found_errors(
        self, sample_data_file, tmp_path, capsys,
    ):
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        # Train coxph only
        main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])

        # Try to evaluate rsf which was not trained
        rc = main([
            'evaluate', '--load-learner', str(learner_path),
            '--model', 'rsf', '--n-importance-repeats', '0',
        ])
        assert rc != 0
        captured = capsys.readouterr()
        assert 'rsf' in captured.err
        assert 'not found' in captured.err.lower()

    def test_load_learner_file_not_found_errors(self, capsys):
        from mlsurv.cli import main

        rc = main([
            'evaluate', '--load-learner', '/nonexistent/path.learner',
            '--model', 'coxph',
        ])
        assert rc != 0
        captured = capsys.readouterr()
        assert 'nonexistent' in captured.err.lower() or 'not found' in captured.err.lower()

    def test_load_learner_prints_summary(self, sample_data_file, tmp_path, capsys):
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        capsys.readouterr()  # clear

        main([
            'evaluate', '--load-learner', str(learner_path),
            '--model', 'coxph', '--n-importance-repeats', '0',
        ])
        captured = capsys.readouterr()
        assert 'Loaded learner from:' in captured.err
        assert '1 model' in captured.err

    def test_run_save_learner_multiple_models(self, sample_data_file, tmp_path):
        # ``--no-tune`` keeps this in the fast tier (no RSF Optuna tuning).
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        rc = main([
            'run', str(sample_data_file),
            '--models', 'coxph', 'rsf', '--no-tune',
            '--cv-folds', '2', '--n-bootstrap', '3',
            '--save-learner', str(learner_path),
        ])
        assert rc == 0
        assert learner_path.exists()

        # Load and evaluate one of the models
        rc = main([
            'evaluate', '--load-learner', str(learner_path),
            '--model', 'coxph', '--n-importance-repeats', '0',
        ])
        assert rc == 0

    def test_run_no_tune_skips_tuning(self, sample_data_file, tmp_path):
        """``run --no-tune`` trains models without tuning: is_tuned is False."""
        from mlsurv import SurvivalLearner
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        rc = main([
            'run', str(sample_data_file),
            '--models', 'coxph', 'rsf', '--no-tune',
            '--cv-folds', '2', '--n-bootstrap', '2',
            '--save-learner', str(learner_path),
        ])
        assert rc == 0

        loaded = SurvivalLearner.load_learner(learner_path)
        assert set(loaded._state.models) == {'coxph', 'rsf'}
        for name, result in loaded._state.models.items():
            assert result.is_tuned is False, f"{name} should not be tuned"

    def test_train_then_validate_roundtrip(self, sample_data_file, tmp_path):
        """Validate command with --load-learner still loads external cohort."""
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        rc = main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        assert rc == 0

        # Use the same CSV as validation data (just testing the plumbing)
        rc = main([
            'validate', '--load-learner', str(learner_path),
            '--model', 'coxph', '--validation-data', str(sample_data_file),
            '--no-bootstrap',
        ])
        assert rc == 0

    def test_load_then_save_roundtrip(self, sample_data_file, tmp_path):
        """Load a learner, evaluate, then re-save with new results."""
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        updated_path = tmp_path / 'updated.learner'

        # Train and save
        rc = main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        assert rc == 0

        # Load, evaluate, and re-save
        rc = main([
            'evaluate', '--load-learner', str(learner_path),
            '--model', 'coxph', '--n-importance-repeats', '0',
            '--save-learner', str(updated_path),
        ])
        assert rc == 0
        assert updated_path.exists()
        assert updated_path.stat().st_size > 0

    def test_save_learner_oserror_degrades_gracefully(
        self, sample_data_file, tmp_path, monkeypatch, capsys,
    ):
        """OSError during save_learner should warn, not crash (rc=0)."""
        from mlsurv.cli import main

        def _raise_oserror(*a, **kw):
            raise OSError("disk full")

        rc = main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(tmp_path / 's.learner'),
        ])
        # sanity: baseline works
        assert rc == 0
        capsys.readouterr()

        monkeypatch.setattr(
            'mlsurv.learner._persistence.PersistenceMixin.save_learner', _raise_oserror,
        )
        rc = main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(tmp_path / 'fail.learner'),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Failed to save learner' in captured.err
        assert 'not persisted' in captured.err

    def test_load_learner_warns_on_ignored_setup_args(
        self, sample_data_file, tmp_path, capsys,
    ):
        """Setup args supplied with --load-learner should emit a warning."""
        from mlsurv.cli import main

        learner_path = tmp_path / 'session.learner'
        main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        capsys.readouterr()

        # --cv-folds 10 differs from default (5) and should trigger warning
        main([
            'evaluate', '--load-learner', str(learner_path),
            '--model', 'coxph', '--n-importance-repeats', '0',
            '--cv-folds', '10',
        ])
        captured = capsys.readouterr()
        assert '--cv-folds' in captured.err
        assert 'ignored' in captured.err.lower()

    @pytest.mark.parametrize('subcommand', ['tune', 'setup', 'benchmark'])
    def test_save_only_commands_reject_load_learner(self, subcommand):
        """Commands with save-only should not accept --load-learner."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([subcommand, 'test.csv', '--load-learner', 'f.learner'])


# ---------------------------------------------------------------------------
# CLI Tier 2 Enhancements
# ---------------------------------------------------------------------------


class TestCLITier2Parser:
    """Parser tests for sweep, report customization, interactions, subgroups."""

    # -- sweep subcommand --

    def test_parser_sweep_subcommand_registered(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'sweep', 'test.csv', '--models', 'rsf',
            '--param', 'n_estimators', '--values', '50', '100',
        ])
        assert args.command == 'sweep'

    def test_parser_sweep_requires_models(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['sweep', 'test.csv', '--param', 'k', '--values', '5'])

    def test_parser_sweep_requires_param(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['sweep', 'test.csv', '--models', 'rsf', '--values', '5'])

    def test_parser_sweep_requires_values(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['sweep', 'test.csv', '--models', 'rsf', '--param', 'k'])

    def test_parser_sweep_defaults(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'sweep', 'test.csv', '--models', 'rsf',
            '--param', 'n_estimators', '--values', '50',
        ])
        assert args.target == 'model'
        assert args.include_test is False
        assert args.n_bootstrap == 0
        assert args.cv_folds == 5

    def test_parser_sweep_values_parsed_as_ints(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'sweep', 'test.csv', '--models', 'rsf',
            '--param', 'n_estimators', '--values', '50', '100', '200',
        ])
        assert args.values == [50, 100, 200]
        assert all(isinstance(v, int) for v in args.values)

    def test_parser_sweep_values_parsed_as_floats(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'sweep', 'test.csv', '--models', 'rsf',
            '--param', 'alpha', '--values', '0.01', '0.1', '1.0',
        ])
        assert args.values == [0.01, 0.1, 1.0]
        assert all(isinstance(v, float) for v in args.values)

    def test_parser_sweep_values_parsed_as_strings(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'sweep', 'test.csv', '--models', 'rsf',
            '--param', 'strategy', '--values', 'median', 'mean',
        ])
        assert args.values == ['median', 'mean']

    def test_parser_sweep_target_choices(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        for target in ('model', 'imputer', 'scaler', 'feature_selector'):
            args = parser.parse_args([
                'sweep', 'test.csv', '--models', 'rsf',
                '--param', 'k', '--values', '5',
                '--target', target,
            ])
            assert args.target == target

    # -- report customization --

    def test_parser_report_args_on_evaluate(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'test.csv', '--model', 'coxph',
            '--sections', 'summary', 'evaluation',
            '--outcome-name', 'death',
            '--manuscript',
            '--feature-groups-file', 'groups.json',
        ])
        assert args.sections == ['summary', 'evaluation']
        assert args.outcome_name == 'death'
        assert args.manuscript is True
        assert args.feature_groups_file == 'groups.json'

    def test_parser_report_args_defaults(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'train', 'test.csv', '--model', 'coxph',
        ])
        assert args.sections is None
        assert args.outcome_name is None
        assert args.manuscript is False
        assert args.feature_groups_file is None

    def test_parser_report_args_on_run(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'run', 'test.csv',
            '--sections', 'summary',
            '--outcome-name', 'progression',
        ])
        assert args.sections == ['summary']
        assert args.outcome_name == 'progression'

    # -- interactions --

    def test_parser_evaluate_interactions_defaults(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'test.csv', '--model', 'coxph',
        ])
        assert args.interactions is False
        assert args.interaction_top_k == 5

    def test_parser_evaluate_interactions_flags(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'test.csv', '--model', 'coxph',
            '--interactions', '--interaction-top-k', '8',
        ])
        assert args.interactions is True
        assert args.interaction_top_k == 8

    # -- subgroups --

    def test_parser_evaluate_subgroups(self):
        """--subgroups accepts one or more column names for evaluate."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'test.csv', '--model', 'coxph',
            '--subgroups', 'stage', 'grade',
        ])
        assert args.subgroups == ['stage', 'grade']

    def test_parser_evaluate_subgroups_default(self):
        """--subgroups defaults to None when not provided for evaluate."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'test.csv', '--model', 'coxph',
        ])
        assert args.subgroups is None

    def test_parser_benchmark_subgroups(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'benchmark', 'test.csv', '--reference-file', 'ref.csv',
            '--subgroups', 'stage', 'grade',
        ])
        assert args.subgroups == ['stage', 'grade']

    def test_parser_benchmark_subgroups_default(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'benchmark', 'test.csv', '--reference-file', 'ref.csv',
        ])
        assert args.subgroups is None


class TestCLITier2Validation:
    """Argument validation tests for tier 2 enhancements."""

    def test_interaction_top_k_too_small(self):
        from mlsurv.cli import _validate_args, create_parser

        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'test.csv', '--model', 'coxph',
            '--interaction-top-k', '1',
        ])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)


class TestCLITier2Commands:
    """Handler tests for sweep, interactions, subgroups, and report kwargs."""

    def test_cmd_sweep_calls_parameter_sweep(self, monkeypatch, tmp_path):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        output_path = tmp_path / 'sweep.csv'
        seen = {}

        class FakeSweepResult:
            data = pd.DataFrame({
                'param_value': [50, 100],
                'model': ['rsf', 'rsf'],
                'mean': [0.7, 0.75],
            })

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                seen.setdefault('trained', []).append(model_name)

            def parameter_sweep(self, **kwargs):
                seen['sweep_kwargs'] = kwargs
                return FakeSweepResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'sweep', str(data_path),
            '--models', 'rsf',
            '--param', 'n_estimators',
            '--values', '50', '100',
            '--output', str(output_path),
        ])

        assert rc == 0
        assert 'rsf' in seen['trained']
        assert seen['sweep_kwargs']['param'] == 'n_estimators'
        assert seen['sweep_kwargs']['values'] == [50, 100]
        assert seen['sweep_kwargs']['target'] == 'model'
        out = pd.read_csv(output_path)
        assert len(out) == 2

    def test_cmd_sweep_empty_result(self, monkeypatch, tmp_path, capsys):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        class FakeSweepResult:
            data = pd.DataFrame()

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                pass

            def parameter_sweep(self, **kwargs):
                return FakeSweepResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'sweep', str(data_path),
            '--models', 'rsf',
            '--param', 'k', '--values', '5',
        ])

        assert rc == 4  # EXIT_COMPUTATION
        captured = capsys.readouterr()
        assert 'No sweep results' in captured.err

    def test_cmd_sweep_skips_failed_model(self, monkeypatch, tmp_path):
        """Partial model failure: surviving models are swept."""
        import warnings

        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}

        class FakeSweepResult:
            data = pd.DataFrame({
                'param_value': [50, 100],
                'model': ['rsf', 'rsf'],
                'mean': [0.7, 0.75],
            })

        class FakeLearner:
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                if model_name == 'deepsurv':
                    raise ImportError("No module named 'pycox'")
                seen.setdefault('trained', []).append(model_name)
                return model_name  # non-None sentinel

            def parameter_sweep(self, **kwargs):
                seen['sweep_kwargs'] = kwargs
                return FakeSweepResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rc = main([
                'sweep', str(data_path),
                '--models', 'rsf', 'deepsurv',
                '--param', 'n_estimators',
                '--values', '50', '100',
            ])

        assert rc == 0
        assert seen['trained'] == ['rsf']
        assert seen['sweep_kwargs']['models'] == ['rsf']
        # The dropped model surfaces via one consolidated MethodologicalWarning,
        # not the old per-model "Missing dependency" SuggestionWarning.
        from mlsurv._warnings import MethodologicalWarning
        drops = [w for w in caught
                 if issubclass(w.category, MethodologicalWarning)
                 and 'dropped' in str(w.message)]
        assert len(drops) == 1
        msg = str(drops[0].message)
        assert 'deepsurv' in msg
        assert '1 of 2 models were dropped' in msg
        assert not any('Missing dependency' in str(w.message) for w in caught)

    def test_cmd_sweep_all_models_fail(self, monkeypatch, tmp_path, capsys):
        """All models fail: RuntimeError propagates as exit code 1."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        class FakeLearner:
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                raise ValueError(f"{model_name} cannot train")

            def parameter_sweep(self, **kwargs):
                raise AssertionError("should not be called")

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'sweep', str(data_path),
            '--models', 'rsf', 'gbsa',
            '--param', 'k', '--values', '5',
        ])

        assert rc == 4  # EXIT_COMPUTATION
        captured = capsys.readouterr()
        assert 'All models failed' in captured.err

    def test_cmd_sweep_memory_error_aborts_immediately(
        self, monkeypatch, tmp_path, capsys,
    ):
        """MemoryError aborts the loop — remaining models are not attempted."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}

        class FakeLearner:
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                seen.setdefault('trained', []).append(model_name)
                if model_name == 'rsf':
                    raise MemoryError("out of memory")

            def parameter_sweep(self, **kwargs):
                raise AssertionError("should not be called")

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        # MemoryError propagates to main() → EXIT_COMPUTATION
        rc = main([
            'sweep', str(data_path),
            '--models', 'rsf', 'coxph',
            '--param', 'k', '--values', '5',
        ])

        assert rc == 4  # EXIT_COMPUTATION
        # rsf was attempted but coxph was never reached
        assert seen['trained'] == ['rsf']
        captured = capsys.readouterr()
        assert 'out of memory' in captured.err

    def test_cmd_evaluate_forwards_interactions(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', make_fake_learner(
            seen, train_return=object(),
            evaluate_return=_Obj(concordance=pd.DataFrame({'c_index_test': [0.7]})),
        ))

        rc = main([
            'evaluate', str(data_path), '--model', 'coxph',
            '--interactions', '--interaction-top-k', '3',
        ])

        assert rc == 0
        assert seen['eval_kwargs']['include_interactions'] is True
        assert seen['eval_kwargs']['interaction_top_k'] == 3

    def test_cmd_evaluate_no_interactions_by_default(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', make_fake_learner(
            seen, train_return=object(),
            evaluate_return=_Obj(concordance=pd.DataFrame({'c_index_test': [0.7]})),
        ))

        rc = main([
            'evaluate', str(data_path), '--model', 'coxph',
        ])

        assert rc == 0
        assert 'include_interactions' not in seen['eval_kwargs']

    def test_cmd_evaluate_forwards_subgroups(self, monkeypatch, tmp_path):
        """--subgroups reaches setup() as subpopulations dict with correct values."""
        import numpy as np

        import mlsurv
        from mlsurv.cli import main

        # Write data CSV with two subgroup columns
        df = pd.DataFrame({
            'f1': [0.1, 0.2, 0.3, 0.4],
            'stage': ['I', 'II', 'I', 'II'],
            'grade': ['A', 'B', 'A', 'B'],
            'time': [10, 12, 8, 15],
            'event': [1, 0, 1, 0],
        })
        data_path = tmp_path / 'data.csv'
        df.to_csv(data_path, index=False)

        seen = {}
        sentinel = object()

        class FakeEvalResult:
            concordance = pd.DataFrame({'c_index_test': [0.7]})

        class FakeLearner:
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                seen['setup_kwargs'] = kwargs

            def train(self, model_name):
                return sentinel

            def evaluate(self, model_result, **kwargs):
                return FakeEvalResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'evaluate', str(data_path), '--model', 'coxph',
            '--subgroups', 'stage', 'grade',
        ])

        assert rc == 0
        subpops = seen['setup_kwargs']['subpopulations']
        assert isinstance(subpops['stage'], np.ndarray)
        assert list(subpops['stage']) == ['I', 'II', 'I', 'II']
        assert isinstance(subpops['grade'], np.ndarray)
        assert list(subpops['grade']) == ['A', 'B', 'A', 'B']

    def test_cmd_evaluate_subgroups_missing_column(self, monkeypatch, tmp_path, capsys):
        """Missing subgroup columns cause error even when some columns are valid."""
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        rc = main([
            'evaluate', str(data_path), '--model', 'coxph',
            '--subgroups', 'f1', 'nonexistent_col',
        ])

        assert rc == 3  # EXIT_DATA
        captured = capsys.readouterr()
        assert 'subgroup columns not found' in captured.err
        assert 'nonexistent_col' in captured.err

    def test_cmd_evaluate_subgroups_with_load_learner_errors(self, tmp_path):
        """--subgroups + --load-learner raises parser error."""
        from mlsurv.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main([
                'evaluate', '--model', 'coxph',
                '--load-learner', str(tmp_path / 'fake.learner'),
                '--subgroups', 'stage',
            ])
        assert exc_info.value.code == 2

    def test_cmd_benchmark_forwards_subgroups(self, monkeypatch, tmp_path):
        import mlsurv
        from mlsurv.cli import main

        # Write data CSV with a subgroup column
        df = pd.DataFrame({
            'f1': [0.1, 0.2, 0.3, 0.4],
            'stage': ['I', 'II', 'I', 'II'],
            'time': [10, 12, 8, 15],
            'event': [1, 0, 1, 0],
        })
        data_path = tmp_path / 'data.csv'
        df.to_csv(data_path, index=False)

        # Write reference CSV
        ref_df = pd.DataFrame({'ref_score': [1.0, 2.0, 3.0, 4.0]})
        ref_path = tmp_path / 'ref.csv'
        ref_df.to_csv(ref_path, index=False)

        seen = {}

        class FakeBenchmarkResult:
            class evaluation:
                concordance = pd.DataFrame({'c_index_test': [0.7]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                seen['setup_kwargs'] = kwargs

            def train(self, model_name, verbose=True):
                pass

            def benchmark(self, **kwargs):
                return FakeBenchmarkResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'benchmark', str(data_path),
            '--reference-file', str(ref_path),
            '--subgroups', 'stage',
        ])

        assert rc == 0
        assert 'subpopulations' in seen['setup_kwargs']
        assert 'stage' in seen['setup_kwargs']['subpopulations']

    def test_cmd_benchmark_subgroups_missing_column(self, monkeypatch, tmp_path, capsys):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')

        # Write reference CSV
        ref_df = pd.DataFrame({'ref_score': [1.0, 2.0, 3.0, 4.0]})
        ref_path = tmp_path / 'ref.csv'
        ref_df.to_csv(ref_path, index=False)

        rc = main([
            'benchmark', str(data_path),
            '--reference-file', str(ref_path),
            '--subgroups', 'nonexistent_col',
        ])

        assert rc == 3  # EXIT_DATA
        captured = capsys.readouterr()
        assert 'subgroup columns not found' in captured.err

    def test_cmd_benchmark_skips_failed_model(self, monkeypatch, tmp_path):
        """Partial model failure: surviving models are benchmarked."""
        import warnings

        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({'ref_score': [0.5, 0.6, 0.7, 0.8]}).to_csv(
            ref_path, index=False,
        )
        seen = {}

        class FakeBenchmarkResult:
            evaluation = type('E', (), {
                'concordance': pd.DataFrame({'c_index_test': [0.7]}),
            })()

        class FakeLearner:
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                if model_name == 'deepsurv':
                    raise ImportError("No module named 'pycox'")
                seen.setdefault('trained', []).append(model_name)
                return model_name

            def benchmark(self, **kwargs):
                seen['benchmark_called'] = True
                return FakeBenchmarkResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rc = main([
                'benchmark', str(data_path),
                '--reference-file', str(ref_path),
                '--models', 'rsf', 'deepsurv',
                '--no-bootstrap',
            ])

        assert rc == 0
        assert seen['trained'] == ['rsf']
        assert seen['benchmark_called']
        # The dropped model surfaces via one consolidated MethodologicalWarning,
        # not the old per-model "Missing dependency" SuggestionWarning.
        from mlsurv._warnings import MethodologicalWarning
        drops = [w for w in caught
                 if issubclass(w.category, MethodologicalWarning)
                 and 'dropped' in str(w.message)]
        assert len(drops) == 1
        msg = str(drops[0].message)
        assert 'deepsurv' in msg
        assert '1 of 2 models were dropped' in msg
        assert not any('Missing dependency' in str(w.message) for w in caught)

    def test_cmd_benchmark_all_models_fail(self, monkeypatch, tmp_path, capsys):
        """All models fail: RuntimeError propagates as exit code 1."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        ref_path = tmp_path / 'refs.csv'
        pd.DataFrame({'ref_score': [0.5, 0.6, 0.7, 0.8]}).to_csv(
            ref_path, index=False,
        )

        class FakeLearner:
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                raise ValueError(f"{model_name} cannot train")

            def benchmark(self, **kwargs):
                raise AssertionError("should not be called")

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'benchmark', str(data_path),
            '--reference-file', str(ref_path),
            '--models', 'rsf', 'gbsa',
            '--no-bootstrap',
        ])

        assert rc == 4  # EXIT_COMPUTATION (RuntimeError from _raise_if_all_failed)
        captured = capsys.readouterr()
        assert 'All models failed' in captured.err

    def test_collect_report_kwargs_extracts_args(self):
        import argparse

        from mlsurv.cli._helpers import _collect_report_kwargs

        args = argparse.Namespace(
            sections=['summary', 'evaluation'],
            outcome_name='death',
            manuscript=True,
            feature_groups_file=None,
        )
        kw = _collect_report_kwargs(args)
        assert kw['sections'] == ['summary', 'evaluation']
        assert kw['outcome_name'] == 'death'
        assert kw['manuscript'] is True
        assert 'feature_groups' not in kw

    def test_collect_report_kwargs_empty_when_defaults(self):
        import argparse

        from mlsurv.cli._helpers import _collect_report_kwargs

        args = argparse.Namespace(
            sections=None,
            outcome_name=None,
            manuscript=False,
            feature_groups_file=None,
        )
        kw = _collect_report_kwargs(args)
        assert kw == {}

    def test_collect_report_kwargs_loads_feature_groups_file(self, tmp_path):
        import argparse
        import json

        from mlsurv.cli._helpers import _collect_report_kwargs

        groups = {'age': 'clinical', 'tumor_size': 'clinical', 'gene_a': 'genomic'}
        groups_file = tmp_path / 'groups.json'
        groups_file.write_text(json.dumps(groups))

        args = argparse.Namespace(
            sections=None,
            outcome_name=None,
            manuscript=False,
            feature_groups_file=str(groups_file),
        )
        kw = _collect_report_kwargs(args)
        assert kw['feature_groups'] == groups

    def test_cmd_train_forwards_report_kwargs(
        self, monkeypatch, tmp_path, make_fake_learner,
    ):
        """Report customization kwargs reach generate_report via cmd_train."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        report_path = tmp_path / 'report.html'
        seen = {}
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', make_fake_learner(
            seen, train_return=_Obj(cv_metric_mean=0.7, cv_metric_std=0.05),
        ))

        rc = main([
            'train', str(data_path), '--model', 'coxph',
            '--report', str(report_path),
            '--sections', 'summary', 'evaluation',
            '--outcome-name', 'death',
            '--manuscript',
        ])

        assert rc == 0
        assert seen['report_kwargs']['sections'] == ['summary', 'evaluation']
        assert seen['report_kwargs']['outcome_name'] == 'death'
        assert seen['report_kwargs']['manuscript'] is True

    def test_cmd_sweep_forwards_include_test(self, monkeypatch, tmp_path):
        """--include-test and --n-bootstrap reach parameter_sweep."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}

        class FakeSweepResult:
            data = pd.DataFrame({'param_value': [50], 'mean': [0.7]})

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, verbose=True):
                pass

            def parameter_sweep(self, **kwargs):
                seen['sweep_kwargs'] = kwargs
                return FakeSweepResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'sweep', str(data_path),
            '--models', 'rsf', 'coxph',
            '--param', 'k', '--values', '5',
            '--include-test', '--n-bootstrap', '10',
        ])

        assert rc == 0
        assert seen['sweep_kwargs']['include_test'] is True
        assert seen['sweep_kwargs']['n_bootstrap'] == 10
        assert seen['sweep_kwargs']['models'] == ['rsf', 'coxph']

    def test_parser_sweep_values_mixed_types(self):
        """Sweep values can be a mix of int, float, and string."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'sweep', 'test.csv', '--models', 'rsf',
            '--param', 'strategy', '--values', '50', '0.1', 'auto',
        ])
        assert args.values == [50, 0.1, 'auto']
        assert isinstance(args.values[0], int)
        assert isinstance(args.values[1], float)
        assert isinstance(args.values[2], str)

    def test_interaction_top_k_minimum_valid(self):
        """--interaction-top-k 2 is the minimum accepted value."""
        from mlsurv.cli import _validate_args, create_parser

        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'test.csv', '--model', 'coxph',
            '--interaction-top-k', '2',
        ])
        # Should not raise
        _validate_args(parser, args)
        assert args.interaction_top_k == 2

    def test_feature_groups_file_not_found_early(self):
        """--feature-groups-file validated before expensive training."""
        from mlsurv.cli import _validate_args, create_parser

        parser = create_parser()
        args = parser.parse_args([
            'train', 'test.csv', '--model', 'coxph',
            '--report', 'r.html',
            '--feature-groups-file', '/nonexistent/groups.json',
        ])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)


# ---------------------------------------------------------------------------
# Quiet flag and progress wiring
# ---------------------------------------------------------------------------


class TestCLIQuietFlag:
    """Tests for -q/--quiet flag and progress wiring."""

    def test_quiet_flag_parsed(self):
        from mlsurv.cli import create_parser
        parser = create_parser()
        assert parser.parse_args(['run', 'test.csv']).quiet is False
        assert parser.parse_args(['-q', 'run', 'test.csv']).quiet is True
        assert parser.parse_args(['--quiet', 'models']).quiet is True

    def test_quiet_verbose_mutual_exclusion(self):
        from mlsurv.cli._data import _validate_args
        from mlsurv.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(['-q', 'run', 'test.csv', '--verbose'])
        with pytest.raises(SystemExit):
            _validate_args(parser, args)

    def test_run_quiet_suppresses_output(self, tmp_path, monkeypatch, capsys):
        import mlsurv
        from mlsurv.cli import main
        data_path = _write_sample_csv(tmp_path / 'data.csv')

        class FakeModelResult:
            name = 'coxph'
            cv_metric_mean = 0.7
            cv_metric_std = 0.02
            evaluation = None

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y): pass
            def setup(self, **kw): pass
            def run(self, **kw): return {'coxph': FakeModelResult()}
            @property
            def best_model(self): return FakeModelResult()
            @property
            def evaluation_result(self): return None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        assert main(['-q', 'run', str(data_path)]) == 0
        assert capsys.readouterr().out == ''

    def test_setup_quiet_suppresses_output(self, tmp_path, monkeypatch, capsys):
        import mlsurv
        from mlsurv.cli import main
        data_path = _write_sample_csv(tmp_path / 'data.csv')

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y): self._X = X
            def setup(self, **kw): pass
            X_train = property(lambda self: self._X.iloc[:2])
            X_test = property(lambda self: self._X.iloc[2:])

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        assert main(['-q', 'setup', str(data_path)]) == 0
        assert capsys.readouterr().out == ''

    def test_models_quiet_still_prints(self, capsys):
        from mlsurv.cli import main
        assert main(['-q', 'models']) == 0
        assert 'coxph' in capsys.readouterr().out

    def test_train_quiet_suppresses_summary(self, tmp_path, monkeypatch, capsys):
        import mlsurv
        from mlsurv.cli import main
        data_path = _write_sample_csv(tmp_path / 'data.csv')
        output_path = tmp_path / 'out.csv'

        class FakeResult:
            name = 'coxph'
            cv_metric_mean = 0.7
            cv_metric_std = 0.02

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y): pass
            def setup(self, **kw): pass
            def train(self, model, **kw): return FakeResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        assert main(['-q', 'train', str(data_path), '--model', 'coxph',
                      '--output', str(output_path)]) == 0
        assert capsys.readouterr().out == ''
        assert output_path.exists()

    def test_quiet_errors_still_on_stderr(self, tmp_path, capsys):
        from mlsurv.cli import main

        rc = main(['-q', 'run', str(tmp_path / 'nonexistent.csv')])
        assert rc == 3  # EXIT_DATA
        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'Error' in captured.err or 'not found' in captured.err.lower()

    def test_run_passes_show_eval_progress(self, tmp_path, monkeypatch):
        import mlsurv
        from mlsurv.cli import main
        data_path = _write_sample_csv(tmp_path / 'data.csv')
        seen = {}

        class FakeModelResult:
            name = 'coxph'
            cv_metric_mean = 0.7
            cv_metric_std = 0.02
            evaluation = None

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y): pass
            def setup(self, **kw): pass
            def run(self, **kw):
                seen.update(kw)
                return {'coxph': FakeModelResult()}
            @property
            def best_model(self): return FakeModelResult()
            @property
            def evaluation_result(self): return None

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        main(['-q', 'run', str(data_path)])
        assert seen.get('show_eval_progress') is False
        assert seen.get('show_progress_bar') is False

    def test_bootstrap_passes_show_progress(self, tmp_path, monkeypatch):
        import mlsurv
        from mlsurv.cli import main
        data_path = _write_sample_csv(tmp_path / 'data.csv')
        seen = {}

        class FakeBootResult:
            summary = pd.DataFrame({'metric': ['c_index'], 'value': [0.7]})
        class FakeResult:
            name = 'coxph'
            cv_metric_mean = 0.7
            cv_metric_std = 0.02
        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y): pass
            def setup(self, **kw): pass
            def train(self, model, **kw): return FakeResult()
            def bootstrap(self, model_result, **kw):
                seen.update(kw)
                return FakeBootResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        main(['-q', 'bootstrap', str(data_path), '--model', 'coxph'])
        assert seen.get('show_progress') is False

    def test_tune_passes_show_progress_bar(self, tmp_path, monkeypatch):
        import mlsurv
        from mlsurv.cli import main
        data_path = _write_sample_csv(tmp_path / 'data.csv')
        seen = {}

        class FakeResult:
            name = 'rsf'
            cv_metric_mean = 0.75
            cv_metric_std = 0.03
            best_params_ = {}
        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y): pass
            def setup(self, **kw): pass
            def tune(self, model, **kw):
                seen.update(kw)
                return FakeResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        main(['-q', 'tune', str(data_path), '--model', 'rsf'])
        assert seen.get('show_progress_bar') is False
        assert seen.get('show_eval_progress') is False

    def test_evaluate_passes_show_progress(self, tmp_path, monkeypatch):
        import mlsurv
        from mlsurv.cli import main
        data_path = _write_sample_csv(tmp_path / 'data.csv')
        seen = {}

        class FakeEvalResult:
            concordance = pd.DataFrame({'c_index_test': [0.7]})
        class FakeResult:
            name = 'coxph'
            cv_metric_mean = 0.7
            cv_metric_std = 0.02
        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y): pass
            def setup(self, **kw): pass
            def train(self, model, **kw): return FakeResult()
            def evaluate(self, model_result, **kw):
                seen.update(kw)
                return FakeEvalResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        main(['-q', 'evaluate', str(data_path), '--model', 'coxph'])
        assert seen.get('show_progress') is False

    def test_export_study_quiet_suppresses_output(self, tmp_path, monkeypatch, capsys):
        from mlsurv.cli import main
        output_path = tmp_path / 'study.json'

        def fake_export_study(**kw):
            output_path.write_text('{}')
            return {'metadata': {'n_trials': 10, 'best_value': 0.75}}

        import mlsurv.tuning
        monkeypatch.setattr(mlsurv.tuning, 'export_study', fake_export_study)
        assert main(['-q', 'export-study', '--experiment', 'exp1',
                      '--model', 'rsf', '--output', str(output_path)]) == 0
        assert capsys.readouterr().out == ''


class TestCLIExplainParser:
    """Parser-level tests for the explain subcommand."""

    def test_parser_explain_requires_load_learner(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['explain', '--patient', '0'])

    def test_parser_explain_requires_patient_selection(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['explain', '--load-learner', 'foo.learner'])

    def test_parser_explain_mutual_exclusion(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'explain', '--load-learner', 'foo.learner',
                '--patient', '0', '--patient-data', 'patient.csv',
            ])

    def test_parser_explain_defaults(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'explain', '--load-learner', 'foo.learner', '--patient', '0',
        ])
        assert args.n_features == 15
        assert args.max_background == 50
        assert args.model is None
        assert args.plot is None
        assert args.output is None
        assert args.patient == 0
        assert args.patient_data is None

    def test_parser_explain_all_flags(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'explain', '--load-learner', 'foo.learner',
            '--patient', '5',
            '--model', 'rsf',
            '--n-features', '10',
            '--max-background', '30',
            '--plot', 'waterfall.html',
            '--output', 'shap.csv',
        ])
        assert args.load_learner == 'foo.learner'
        assert args.patient == 5
        assert args.model == 'rsf'
        assert args.n_features == 10
        assert args.max_background == 30
        assert args.plot == 'waterfall.html'
        assert args.output == 'shap.csv'

    def test_parser_explain_patient_data_flag(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'explain', '--load-learner', 'foo.learner',
            '--patient-data', 'patient.csv',
        ])
        assert args.patient_data == 'patient.csv'
        assert args.patient is None

    def test_parser_explain_rejects_legacy_patient_index_flag(self):
        """Regression: the legacy --patient-index flag must be gone."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'explain', '--load-learner', 'foo.learner',
                '--patient-index', '0',
            ])

    def test_parser_explain_n_features_validation(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'explain', '--load-learner', 'foo.learner',
            '--patient', '0', '--n-features', '0',
        ])
        from mlsurv.cli._data import _validate_args
        with pytest.raises(SystemExit):
            _validate_args(parser, args)

    def test_parser_explain_max_background_validation(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'explain', '--load-learner', 'foo.learner',
            '--patient', '0', '--max-background', '0',
        ])
        from mlsurv.cli._data import _validate_args
        with pytest.raises(SystemExit):
            _validate_args(parser, args)


class TestCLIExplainCommand:
    """Command handler tests for cmd_explain (monkeypatched, fast tier)."""

    def _make_fake_explanation(self):
        """Build a minimal PatientExplanation for testing."""
        from mlsurv.results._explanation import PatientExplanation

        shap_df = pd.DataFrame({
            'model_name': ['coxph', 'coxph'],
            'feature': ['age', 'biomarker'],
            'feature_value': [65.0, 1.2],
            'shap_value': [0.45, -0.25],
            'abs_shap_value': [0.45, 0.25],
            'risk_score': [1.53, 1.53],
        })
        return PatientExplanation(shap_values=shap_df, figure=None)

    def _make_fake_explanation_with_figure(self):
        """Build a PatientExplanation with a mock figure."""
        result = self._make_fake_explanation()

        class FakeFigure:
            def __init__(self):
                self.written_to = None

            def write_html(self, path):
                self.written_to = path

        result.figure = FakeFigure()
        return result

    def _make_fake_learner_cls(self, fake_result, explain_calls=None,
                               model_names=None):
        """Build a FakeLearner class with load_learner classmethod."""
        if model_names is None:
            model_names = ['coxph']

        class _FakeState:
            is_setup = True
            models = {m: object() for m in model_names}

        class FakeLearner(_FakeLearnerBase):
            _state = _FakeState()
            X_train = pd.DataFrame({'f': [1, 2, 3]})
            X_test = pd.DataFrame({'f': [4, 5]})

            @classmethod
            def load_learner(cls, path):
                return cls()

            def explain(self, **kwargs):
                if explain_calls is not None:
                    explain_calls.append(kwargs)
                return fake_result

        return FakeLearner

    def test_cmd_explain_patient(self, monkeypatch, capsys):
        import mlsurv
        from mlsurv.cli import main

        fake_result = self._make_fake_explanation()
        explain_calls = []
        FakeLearner = self._make_fake_learner_cls(fake_result, explain_calls)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient', '3',
        ])
        assert rc == 0
        assert len(explain_calls) == 1
        assert explain_calls[0]['patient'] == 3
        assert explain_calls[0]['X'] is None
        assert explain_calls[0]['model'] is None
        assert explain_calls[0]['n_features'] == 15
        assert explain_calls[0]['max_background'] == 50
        assert explain_calls[0]['plot'] is False

        captured = capsys.readouterr()
        assert 'SHAP Explanation' in captured.out
        assert 'age' in captured.out
        assert 'biomarker' in captured.out

    def test_cmd_explain_patient_data_csv(self, monkeypatch, tmp_path, capsys):
        import mlsurv
        from mlsurv.cli import main

        patient_csv = tmp_path / 'patient.csv'
        pd.DataFrame({'f1': [0.5], 'f2': [1.0]}).to_csv(patient_csv, index=False)

        fake_result = self._make_fake_explanation()
        explain_calls = []
        FakeLearner = self._make_fake_learner_cls(fake_result, explain_calls)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient-data', str(patient_csv),
        ])
        assert rc == 0
        assert len(explain_calls) == 1
        assert explain_calls[0]['patient'] is None
        X_passed = explain_calls[0]['X']
        assert X_passed is not None
        assert len(X_passed) == 1
        assert list(X_passed.columns) == ['f1', 'f2']

    def test_cmd_explain_writes_csv_output(self, monkeypatch, tmp_path, capsys):
        import mlsurv
        from mlsurv.cli import main

        fake_result = self._make_fake_explanation()
        FakeLearner = self._make_fake_learner_cls(fake_result)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        output_csv = tmp_path / 'shap.csv'
        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient', '0',
            '--output', str(output_csv),
        ])
        assert rc == 0
        assert output_csv.exists()
        df = pd.read_csv(output_csv)
        assert 'feature' in df.columns
        assert 'shap_value' in df.columns
        assert len(df) == 2

    def test_cmd_explain_writes_plot_html(self, monkeypatch, tmp_path, capsys):
        import mlsurv
        from mlsurv.cli import main

        fake_result = self._make_fake_explanation_with_figure()
        FakeLearner = self._make_fake_learner_cls(fake_result)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        plot_path = tmp_path / 'waterfall.html'
        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient', '0',
            '--plot', str(plot_path),
        ])
        assert rc == 0
        assert fake_result.figure.written_to == str(plot_path)
        captured = capsys.readouterr()
        assert 'Waterfall plot saved to' in captured.out

    def test_cmd_explain_no_plot_skips_figure(self, monkeypatch, capsys):
        import mlsurv
        from mlsurv.cli import main

        explain_calls = []
        fake_result = self._make_fake_explanation()
        FakeLearner = self._make_fake_learner_cls(fake_result, explain_calls)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient', '0',
        ])
        assert rc == 0
        assert explain_calls[0]['plot'] is False

    def test_cmd_explain_forwards_model_flag(self, monkeypatch, capsys):
        import mlsurv
        from mlsurv.cli import main

        explain_calls = []
        fake_result = self._make_fake_explanation()
        FakeLearner = self._make_fake_learner_cls(
            fake_result, explain_calls, model_names=['rsf'],
        )
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient', '0', '--model', 'rsf',
        ])
        assert rc == 0
        assert explain_calls[0]['model'] == 'rsf'

    def test_cmd_explain_patient_data_multi_row_errors(self, monkeypatch, tmp_path, capsys):
        import mlsurv
        from mlsurv.cli import main

        patient_csv = tmp_path / 'patient.csv'
        pd.DataFrame({'f1': [0.5, 0.6], 'f2': [1.0, 0.0]}).to_csv(
            patient_csv, index=False,
        )

        class FakeLearner(_FakeLearnerBase):
            class _state:
                is_setup = True
                models = {'coxph': object()}
            X_train = pd.DataFrame({'f': [1, 2, 3]})
            X_test = pd.DataFrame({'f': [4, 5]})

            @classmethod
            def load_learner(cls, path):
                return cls()

            def explain(self, **kwargs):
                raise AssertionError("explain should not be called")

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient-data', str(patient_csv),
        ])
        assert rc == 3  # EXIT_DATA
        captured = capsys.readouterr()
        assert 'exactly 1 row' in captured.err

    def test_cmd_explain_error_propagation(self, monkeypatch, capsys):
        import mlsurv
        from mlsurv.cli import main

        class FakeLearner(_FakeLearnerBase):
            class _state:
                is_setup = True
                models = {'coxph': object()}
            X_train = pd.DataFrame({'f': [1, 2, 3]})
            X_test = pd.DataFrame({'f': [4, 5]})

            @classmethod
            def load_learner(cls, path):
                return cls()

            def explain(self, **kwargs):
                raise ValueError("patient=99 is out of bounds")

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient', '99',
        ])
        assert rc == 4  # EXIT_COMPUTATION
        captured = capsys.readouterr()
        assert 'out of bounds' in captured.err

    def test_cmd_explain_missing_patient_data_file(self, monkeypatch, tmp_path, capsys):
        import mlsurv
        from mlsurv.cli import main

        class FakeLearner(_FakeLearnerBase):
            class _state:
                is_setup = True
                models = {'coxph': object()}
            X_train = pd.DataFrame({'f': [1, 2, 3]})
            X_test = pd.DataFrame({'f': [4, 5]})

            @classmethod
            def load_learner(cls, path):
                return cls()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient-data', str(tmp_path / 'nonexistent.csv'),
        ])
        assert rc == 3  # EXIT_DATA
        captured = capsys.readouterr()
        assert 'not found' in captured.err

    def test_cmd_explain_empty_csv_errors(self, monkeypatch, tmp_path, capsys):
        import mlsurv
        from mlsurv.cli import main

        patient_csv = tmp_path / 'empty.csv'
        patient_csv.write_text('f1,f2\n')

        class FakeLearner(_FakeLearnerBase):
            class _state:
                is_setup = True
                models = {'coxph': object()}
            X_train = pd.DataFrame({'f': [1, 2, 3]})
            X_test = pd.DataFrame({'f': [4, 5]})

            @classmethod
            def load_learner(cls, path):
                return cls()

            def explain(self, **kwargs):
                raise AssertionError("explain should not be called")

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient-data', str(patient_csv),
        ])
        assert rc == 3  # EXIT_DATA
        captured = capsys.readouterr()
        assert 'exactly 1 row' in captured.err

    def test_cmd_explain_import_error(self, monkeypatch, capsys):
        import mlsurv
        from mlsurv.cli import main

        class FakeLearner(_FakeLearnerBase):
            class _state:
                is_setup = True
                models = {'coxph': object()}
            X_train = pd.DataFrame({'f': [1, 2, 3]})
            X_test = pd.DataFrame({'f': [4, 5]})

            @classmethod
            def load_learner(cls, path):
                return cls()

            def explain(self, **kwargs):
                raise ImportError("No module named 'shap'")

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient', '0',
        ])
        assert rc == 1  # EXIT_ERROR
        captured = capsys.readouterr()
        assert 'Missing dependency' in captured.err
        assert 'shap' in captured.err

    def test_cmd_explain_plot_write_failure(self, monkeypatch, capsys):
        import mlsurv
        from mlsurv.cli import main

        fake_result = self._make_fake_explanation()

        class FailingFigure:
            def write_html(self, path):
                raise OSError("Permission denied")

        fake_result.figure = FailingFigure()

        FakeLearner = self._make_fake_learner_cls(fake_result)
        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)

        rc = main([
            'explain', '--load-learner', 'test.learner',
            '--patient', '0',
            '--plot', '/readonly/path.html',
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'SHAP Explanation' in captured.out
        assert 'Warning: Failed to write plot' in captured.err


# ---------------------------------------------------------------------------
# Statistical test CLI commands
# ---------------------------------------------------------------------------


class TestCLIStatTestParser:
    """Parser-level tests for statistical test flags and commands."""

    def test_evaluate_accepts_stat_test_flags(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'test.csv', '--model', 'coxph',
            '--ph-test', '--linearity-test', '--logrank-test',
            '--rmst-test', '--importance-test',
        ])
        assert args.ph_test is True
        assert args.linearity_test is True
        assert args.logrank_test is True
        assert args.rmst_test is True
        assert args.importance_test is True

    def test_evaluate_stat_test_flags_default_false(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['evaluate', 'test.csv', '--model', 'coxph'])
        assert args.ph_test is False
        assert args.linearity_test is False
        assert args.logrank_test is False
        assert args.rmst_test is False
        assert args.importance_test is False

    @pytest.mark.parametrize('cmd,extra_args', [
        ('ph-test', ['--load-learner', 'foo.learner']),
        ('logrank-test', ['--load-learner', 'foo.learner']),
        ('rmst-test', ['--load-learner', 'foo.learner']),
        ('importance-test', ['--load-learner', 'foo.learner']),
    ])
    def test_standalone_commands_parse(self, cmd, extra_args):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([cmd] + extra_args)
        assert args.command == cmd
        assert args.load_learner == 'foo.learner'

    @pytest.mark.parametrize('cmd', [
        'ph-test', 'logrank-test', 'rmst-test', 'importance-test',
    ])
    def test_standalone_commands_require_load_learner(self, cmd):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([cmd])

    def test_old_rmst_comparison_subcommand_gone(self):
        """The renamed subcommand no longer resolves."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'rmst-comparison', '--load-learner', 'foo.learner',
            ])

    def test_old_importance_pvalues_subcommand_gone(self):
        """The renamed subcommand no longer resolves."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'importance-pvalues', '--load-learner', 'foo.learner',
            ])

    def test_ph_test_accepts_time_transform(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'ph-test', '--load-learner', 'foo.learner',
            '--time-transform', 'km',
        ])
        assert args.time_transform == 'km'

    def test_ph_test_time_transform_default(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['ph-test', '--load-learner', 'foo.learner'])
        assert args.time_transform == 'rank'

    def test_logrank_test_accepts_n_groups(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'logrank-test', '--load-learner', 'foo.learner',
            '--n-groups', '3',
        ])
        assert args.n_groups == 3

    def test_rmst_test_accepts_tau(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'rmst-test', '--load-learner', 'foo.learner',
            '--tau', '24.0',
        ])
        assert args.tau == 24.0

    def test_rmst_test_tau_default_none(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'rmst-test', '--load-learner', 'foo.learner',
        ])
        assert args.tau is None

    def test_importance_test_accepts_method(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'importance-test', '--load-learner', 'foo.learner',
            '--method', 'bootstrap',
        ])
        assert args.method == 'bootstrap'

    def test_importance_test_defaults(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'importance-test', '--load-learner', 'foo.learner',
        ])
        assert args.method == 'pimp'
        assert args.n_permutations == 100
        assert args.n_importance_repeats == 10
        assert args.p_adjust == 'bh'
        assert args.random_state is None
        assert args.n_jobs is None

    def test_logrank_p_adjust_choices(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'logrank-test', '--load-learner', 'foo.learner',
            '--p-adjust', 'bonferroni',
        ])
        assert args.p_adjust == 'bonferroni'

    def test_standalone_commands_accept_output(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        for cmd in ['ph-test', 'logrank-test', 'rmst-test',
                     'importance-test']:
            args = parser.parse_args([
                cmd, '--load-learner', 'foo.learner',
                '--output', '/tmp/out.csv',
            ])
            assert args.output == '/tmp/out.csv'


class TestCLIParityCommandParser:
    """Parser-level tests for the CLI/API-parity bundle additions."""

    def test_linearity_test_parses(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'linearity-test', '--load-learner', 'foo.learner', '--n-knots', '6',
        ])
        assert args.command == 'linearity-test'
        assert args.load_learner == 'foo.learner'
        assert args.n_knots == 6

    def test_linearity_test_n_knots_default(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['linearity-test', '--load-learner', 'f'])
        assert args.n_knots == 4

    def test_linearity_test_requires_load_learner(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['linearity-test'])

    def test_interactions_parses(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'interactions', '--load-learner', 'foo.learner',
            '--model', 'rsf', '--method', 'shap', '--top-k', '3',
            '--max-samples', '40', '--grid-resolution', '8',
            '--random-state', '7',
        ])
        assert args.command == 'interactions'
        assert args.model == 'rsf'
        assert args.method == 'shap'
        assert args.top_k == 3
        assert args.max_samples == 40
        assert args.grid_resolution == 8
        assert args.random_state == 7

    def test_interactions_defaults(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['interactions', '--load-learner', 'f'])
        assert args.method == 'hstat'
        assert args.top_k == 5
        assert args.max_samples == 100
        assert args.grid_resolution == 25
        assert args.model is None

    def test_interactions_rejects_bad_method(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'interactions', '--load-learner', 'f', '--method', 'pdp',
            ])

    def test_export_results_parses(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'export-results', 'out.xlsx', '--load-learner', 'foo.learner',
            '--format', 'csv', '--audit',
        ])
        assert args.command == 'export-results'
        assert args.output == 'out.xlsx'
        assert args.load_learner == 'foo.learner'
        assert args.format == 'csv'
        assert args.audit is True

    def test_export_results_format_default_none(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'export-results', 'out.xlsx', '--load-learner', 'f',
        ])
        assert args.format is None
        assert args.audit is False

    def test_export_results_requires_output_and_learner(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['export-results', '--load-learner', 'f'])
        with pytest.raises(SystemExit):
            parser.parse_args(['export-results', 'out.xlsx'])

    def test_train_accepts_model_params(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'train', 'd.csv', '--model', 'coxph', '--model-params', 'mp.json',
        ])
        assert args.model_params == 'mp.json'

    def test_train_model_params_default_none(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['train', 'd.csv', '--model', 'coxph'])
        assert args.model_params is None

    def test_benchmark_accepts_dataset(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'benchmark', 'd.csv', '--reference-file', 'r.csv',
            '--dataset', 'train',
        ])
        assert args.dataset == 'train'

    def test_benchmark_dataset_default_test(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'benchmark', 'd.csv', '--reference-file', 'r.csv',
        ])
        assert args.dataset == 'test'

    def test_benchmark_rejects_bad_dataset(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'benchmark', 'd.csv', '--reference-file', 'r.csv',
                '--dataset', 'holdout',
            ])

    def test_predict_accepts_ci_band(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'predict', 'd.csv', '--load-learner', 'f', '--ci', '--ci-band',
        ])
        assert args.ci is True
        assert args.ci_band is True

    def test_predict_ci_band_default_false(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['predict', 'd.csv', '--load-learner', 'f'])
        assert args.ci_band is False


class TestCLIParityFlagWiring:
    """The new flags' CLI glue actually reaches the API (not just parses)."""

    def test_train_model_params_forwarded(self, monkeypatch, tmp_path):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        mp_path = tmp_path / 'mp.json'
        mp_path.write_text(json.dumps({'coxph': {'alpha': 0.3}}))
        seen = {}

        class FakeResult:
            name = 'coxph'
            cv_metric_mean = 0.7
            cv_metric_std = 0.05
            model = 'm'
            is_tuned = False
            best_params_ = None

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, **kwargs):
                seen['model_params'] = kwargs.get('model_params')
                return FakeResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        rc = main([
            'train', str(data_path), '--model', 'coxph',
            '--model-params', str(mp_path),
        ])
        assert rc == 0
        assert seen['model_params'] == {'coxph': {'alpha': 0.3}}

    def test_train_without_model_params_omits_kwarg(self, monkeypatch, tmp_path):
        """No --model-params => train() called without model_params kwarg."""
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        seen = {}

        class FakeResult:
            name = 'coxph'
            cv_metric_mean = 0.7
            cv_metric_std = 0.05
            model = 'm'
            is_tuned = False
            best_params_ = None

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, **kwargs):
                seen['kwargs'] = kwargs
                return FakeResult()

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        rc = main(['train', str(data_path), '--model', 'coxph'])
        assert rc == 0
        assert 'model_params' not in seen['kwargs']

    def test_train_model_params_malformed_json_errors(self, monkeypatch, tmp_path):
        import mlsurv
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'sample.csv')
        bad = tmp_path / 'bad.json'
        bad.write_text('{not valid json')

        class FakeLearner(_FakeLearnerBase):
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def train(self, model_name, **kwargs):
                raise AssertionError('train must not run on malformed JSON')

        monkeypatch.setattr(mlsurv, 'SurvivalLearner', FakeLearner)
        rc = main([
            'train', str(data_path), '--model', 'coxph',
            '--model-params', str(bad),
        ])
        assert rc != 0

    def test_interactions_shap_typeerror_surfaced(
        self, monkeypatch, capsys,
    ):
        """method='shap' on a non-tree model raises TypeError; surface it as a
        clean computation error, not a generic 'Unexpected error'."""
        from mlsurv.cli import main, EXIT_COMPUTATION
        import mlsurv.cli._commands as cmds

        class FakeLearner(_FakeLearnerBase):
            def interactions(self, **kwargs):
                raise TypeError(
                    "SHAP interaction values require a tree-based model"
                )

        monkeypatch.setattr(
            cmds, '_load_learner_if_requested', lambda args: FakeLearner(),
        )
        rc = main([
            'interactions', '--load-learner', 'x.learner',
            '--method', 'shap', '--model', 'coxph',
        ])
        assert rc == EXIT_COMPUTATION
        err = capsys.readouterr().err
        assert 'Unexpected error' not in err

    def test_export_results_infers_csv_without_format(
        self, monkeypatch, tmp_path,
    ):
        from mlsurv.cli import main
        import mlsurv.cli._commands as cmds

        seen = {}

        class FakeLearner(_FakeLearnerBase):
            def export_results(self, path, *, format='xlsx', audit=False):
                seen['format'] = format
                return str(path)

        monkeypatch.setattr(
            cmds, '_load_learner_if_requested', lambda args: FakeLearner(),
        )
        rc = main([
            'export-results', str(tmp_path / 'outdir'),
            '--load-learner', 'x.learner',
        ])
        assert rc == 0
        assert seen['format'] == 'csv'

    def test_export_results_unrecognized_extension_warns(
        self, monkeypatch, tmp_path, capsys,
    ):
        from mlsurv.cli import main
        import mlsurv.cli._commands as cmds

        seen = {}

        class FakeLearner(_FakeLearnerBase):
            def export_results(self, path, *, format='xlsx', audit=False):
                seen['format'] = format
                return str(path)

        monkeypatch.setattr(
            cmds, '_load_learner_if_requested', lambda args: FakeLearner(),
        )
        rc = main([
            'export-results', str(tmp_path / 'results.xls'),
            '--load-learner', 'x.learner',
        ])
        assert rc == 0
        assert seen['format'] == 'csv'
        assert 'unrecognized' in capsys.readouterr().err.lower()

    def test_export_results_explicit_format_overrides_extension(
        self, monkeypatch, tmp_path,
    ):
        from mlsurv.cli import main
        import mlsurv.cli._commands as cmds

        seen = {}

        class FakeLearner(_FakeLearnerBase):
            def export_results(self, path, *, format='xlsx', audit=False):
                seen['format'] = format
                return str(path)

        monkeypatch.setattr(
            cmds, '_load_learner_if_requested', lambda args: FakeLearner(),
        )
        rc = main([
            'export-results', str(tmp_path / 'out.xlsx'),
            '--load-learner', 'x.learner', '--format', 'csv',
        ])
        assert rc == 0
        assert seen['format'] == 'csv'

    @pytest.mark.integration
    def test_benchmark_dataset_passthrough(self, tmp_path, synthetic_data_clean):
        """--dataset train routes references through the train-split path."""
        from mlsurv.cli import main

        X, y = synthetic_data_clean
        df = X.copy()
        df['time'] = y['time']
        df['event'] = y['event'].astype(int)
        data_path = tmp_path / 'data.csv'
        df.to_csv(data_path, index=False)

        ref = pd.DataFrame({'ref_score': X.iloc[:, 0].to_numpy()})
        ref_path = tmp_path / 'ref.csv'
        ref.to_csv(ref_path, index=False)

        rc = main([
            'benchmark', str(data_path), '--reference-file', str(ref_path),
            '--models', 'coxph', '--dataset', 'train', '--no-bootstrap',
            '--cv-folds', '2',
        ])
        assert rc == 0


class TestCLIStatTestValidation:
    """Argument validation tests for statistical test parameters."""

    def test_n_groups_rejects_1(self):
        from mlsurv.cli import main

        with pytest.raises(SystemExit):
            main([
                'logrank-test', '--load-learner', 'foo.learner',
                '--n-groups', '1',
            ])

    def test_n_groups_rejects_0(self):
        from mlsurv.cli import main

        with pytest.raises(SystemExit):
            main([
                'rmst-test', '--load-learner', 'foo.learner',
                '--n-groups', '0',
            ])

    def test_n_permutations_rejects_0(self):
        from mlsurv.cli import main

        with pytest.raises(SystemExit):
            main([
                'importance-test', '--load-learner', 'foo.learner',
                '--n-permutations', '0',
            ])


class TestCLIStatTestCommands:
    """Integration tests for statistical test CLI commands."""

    @pytest.fixture
    def sample_data_file(self, tmp_path, synthetic_data_clean):
        X, y = synthetic_data_clean
        df = X.copy()
        df['time'] = y['time']
        df['event'] = y['event'].astype(int)
        filepath = tmp_path / 'data.csv'
        df.to_csv(filepath, index=False)
        return filepath

    @pytest.fixture
    def trained_learner_path(self, sample_data_file, tmp_path):
        """Save a learner after train (no evaluate)."""
        from mlsurv.cli import main

        learner_path = tmp_path / 'trained.learner'
        rc = main([
            'train', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--save-learner', str(learner_path),
        ])
        assert rc == 0
        return learner_path

    @pytest.fixture
    def evaluated_learner_path(self, sample_data_file, tmp_path):
        """Save a learner after evaluate (has risk scores)."""
        from mlsurv.cli import main

        learner_path = tmp_path / 'evaluated.learner'
        rc = main([
            'evaluate', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--n-importance-repeats', '0',
            '--save-learner', str(learner_path),
        ])
        assert rc == 0
        return learner_path

    # --- Mode A: In-session flags on evaluate ---

    @pytest.mark.integration
    def test_evaluate_with_ph_test_flag(self, sample_data_file, capsys):
        from mlsurv.cli import main

        rc = main([
            'evaluate', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--n-importance-repeats', '0',
            '--ph-test',
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Proportional Hazards Test' in captured.out

    @pytest.mark.integration
    def test_evaluate_with_linearity_test_flag(self, sample_data_file, capsys):
        from mlsurv.cli import main

        rc = main([
            'evaluate', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--n-importance-repeats', '0',
            '--linearity-test',
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Log-Hazard Linearity Test' in captured.out

    @pytest.mark.integration
    def test_evaluate_with_logrank_test_flag(self, sample_data_file, capsys):
        from mlsurv.cli import main

        rc = main([
            'evaluate', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--n-importance-repeats', '0',
            '--logrank-test',
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Log-Rank Test' in captured.out

    @pytest.mark.integration
    def test_evaluate_with_rmst_test_flag(self, sample_data_file, capsys):
        from mlsurv.cli import main

        rc = main([
            'evaluate', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--n-importance-repeats', '0',
            '--rmst-test',
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'RMST Comparison' in captured.out

    @pytest.mark.integration
    def test_evaluate_with_all_stat_flags(self, sample_data_file, capsys):
        from mlsurv.cli import main

        rc = main([
            'evaluate', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--n-importance-repeats', '0',
            '--ph-test', '--linearity-test', '--logrank-test',
            '--rmst-test', '--importance-test',
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Proportional Hazards Test' in captured.out
        # linearity_test may return empty on small synthetic data (sample
        # size below 200 hard floor), so we check the header is printed
        # but don't require non-empty results
        assert 'Log-Hazard Linearity Test' in captured.out
        assert 'Log-Rank Test' in captured.out
        assert 'RMST Comparison' in captured.out
        assert 'Importance P-Values' in captured.out

    # --- Mode B: Standalone commands ---

    @pytest.mark.integration
    def test_ph_test_standalone(self, trained_learner_path, tmp_path, capsys):
        from mlsurv.cli import main

        output_csv = tmp_path / 'ph_test.csv'
        rc = main([
            'ph-test', '--load-learner', str(trained_learner_path),
            '--output', str(output_csv),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Proportional Hazards Test' in captured.out
        assert output_csv.exists()

        import pandas as pd
        df = pd.read_csv(output_csv)
        assert 'feature' in df.columns
        assert 'p_value' in df.columns

    @pytest.mark.integration
    def test_ph_test_standalone_time_transform(self, trained_learner_path, capsys):
        from mlsurv.cli import main

        rc = main([
            'ph-test', '--load-learner', str(trained_learner_path),
            '--time-transform', 'km',
        ])
        assert rc == 0

    @pytest.mark.integration
    def test_logrank_test_standalone(self, evaluated_learner_path, tmp_path, capsys):
        from mlsurv.cli import main

        output_csv = tmp_path / 'logrank.csv'
        rc = main([
            'logrank-test', '--load-learner', str(evaluated_learner_path),
            '--output', str(output_csv),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Log-Rank Test' in captured.out
        assert output_csv.exists()

    @pytest.mark.integration
    def test_rmst_test_standalone(self, evaluated_learner_path, tmp_path, capsys):
        from mlsurv.cli import main

        output_csv = tmp_path / 'rmst.csv'
        rc = main([
            'rmst-test', '--load-learner', str(evaluated_learner_path),
            '--output', str(output_csv),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'RMST Comparison' in captured.out
        assert output_csv.exists()

    @pytest.mark.integration
    def test_importance_test_standalone(self, trained_learner_path, tmp_path, capsys):
        from mlsurv.cli import main

        output_csv = tmp_path / 'pvalues.csv'
        rc = main([
            'importance-test', '--load-learner', str(trained_learner_path),
            '--n-permutations', '5', '--n-importance-repeats', '2',
            '--output', str(output_csv),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Importance P-Values' in captured.out
        assert output_csv.exists()

    @pytest.mark.integration
    def test_logrank_test_requires_evaluate_state(self, trained_learner_path):
        """logrank-test on a train-only learner should fail with clear error."""
        from mlsurv.cli import main

        rc = main([
            'logrank-test', '--load-learner', str(trained_learner_path),
        ])
        assert rc != 0

    @pytest.mark.integration
    def test_rmst_test_requires_evaluate_state(self, trained_learner_path):
        """rmst-test on a train-only learner should fail with clear error."""
        from mlsurv.cli import main

        rc = main([
            'rmst-test', '--load-learner', str(trained_learner_path),
        ])
        assert rc != 0

    @pytest.mark.integration
    def test_logrank_test_error_message_is_cli_friendly(
        self, trained_learner_path, capsys,
    ):
        """logrank-test on train-only learner should give CLI-friendly error."""
        from mlsurv.cli import main

        main([
            'logrank-test', '--load-learner', str(trained_learner_path),
        ])
        captured = capsys.readouterr()
        # Should mention CLI command, not Python API method
        assert 'mlsurv evaluate' in captured.err

    @pytest.mark.integration
    def test_evaluate_stat_test_failure_does_not_block_save(
        self, sample_data_file, tmp_path,
    ):
        """A stat test failure should not prevent --save-learner."""
        from mlsurv.cli import main

        learner_path = tmp_path / 'saved.learner'
        # --importance-test with method=bootstrap will fail (no bootstrap
        # data), but evaluate + save-learner should still succeed
        rc = main([
            'evaluate', str(sample_data_file), '--model', 'coxph',
            '--cv-folds', '2', '--n-importance-repeats', '0',
            '--importance-test',
            '--save-learner', str(learner_path),
        ])
        assert rc == 0
        assert learner_path.exists()

    @pytest.fixture
    def evaluated_rsf_learner_path(self, sample_data_file, tmp_path):
        """Save an rsf learner after evaluate (nonlinear, for interactions)."""
        from mlsurv.cli import main

        learner_path = tmp_path / 'rsf_eval.learner'
        rc = main([
            'evaluate', str(sample_data_file), '--model', 'rsf',
            '--cv-folds', '2', '--n-importance-repeats', '2',
            '--save-learner', str(learner_path),
        ])
        assert rc == 0
        return learner_path

    # --- Parity bundle: linearity-test / interactions / export-results ---

    @pytest.mark.integration
    def test_linearity_test_standalone(
        self, trained_learner_path, tmp_path, capsys,
    ):
        from mlsurv.cli import main

        output_csv = tmp_path / 'linearity.csv'
        rc = main([
            'linearity-test', '--load-learner', str(trained_learner_path),
            '--output', str(output_csv),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Log-Hazard Linearity Test' in captured.out

    @pytest.mark.integration
    def test_interactions_standalone(
        self, evaluated_rsf_learner_path, tmp_path, capsys,
    ):
        from mlsurv.cli import main

        output_csv = tmp_path / 'interactions.csv'
        rc = main([
            'interactions', '--load-learner', str(evaluated_rsf_learner_path),
            '--model', 'rsf', '--top-k', '3', '--max-samples', '40',
            '--grid-resolution', '8', '--output', str(output_csv),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Feature Interactions' in captured.out
        assert output_csv.exists()

    @pytest.mark.integration
    def test_interactions_requires_evaluate(
        self, trained_learner_path, capsys,
    ):
        """interactions on a train-only learner fails with a CLI-friendly error."""
        from mlsurv.cli import main

        rc = main([
            'interactions', '--load-learner', str(trained_learner_path),
        ])
        assert rc != 0
        captured = capsys.readouterr()
        assert 'mlsurv evaluate' in captured.err

    @pytest.mark.integration
    def test_export_results_xlsx_standalone(
        self, evaluated_learner_path, tmp_path, capsys,
    ):
        from mlsurv.cli import main

        out = tmp_path / 'results.xlsx'
        rc = main([
            'export-results', str(out),
            '--load-learner', str(evaluated_learner_path),
        ])
        assert rc == 0
        assert out.exists()
        captured = capsys.readouterr()
        assert 'exported' in captured.out.lower()

    @pytest.mark.integration
    def test_export_results_csv_standalone(
        self, evaluated_learner_path, tmp_path,
    ):
        from mlsurv.cli import main

        out_dir = tmp_path / 'csvout'
        rc = main([
            'export-results', str(out_dir),
            '--load-learner', str(evaluated_learner_path), '--format', 'csv',
        ])
        assert rc == 0
        assert out_dir.exists()

    def test_n_permutations_rejects_negative(self):
        from mlsurv.cli import main

        with pytest.raises(SystemExit):
            main([
                'importance-test', '--load-learner', 'foo.learner',
                '--n-permutations', '-5',
            ])

    def test_n_groups_rejects_negative(self):
        from mlsurv.cli import main

        with pytest.raises(SystemExit):
            main([
                'logrank-test', '--load-learner', 'foo.learner',
                '--n-groups', '-1',
            ])


class TestCLIEvaluationResultAfterLoad:
    """evaluation_result is computed on read; verify it works after a
    save/load round-trip."""

    @pytest.mark.integration
    def test_evaluation_result_accessible_after_load_learner(
        self, tmp_path, synthetic_data_clean,
    ):
        """evaluation_result property must work after save/load round-trip."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train('coxph', verbose=False)
        learner.evaluate(verbose=False, n_importance_repeats=0)

        # Verify evaluation_result works before save
        assert learner.evaluation_result is not None
        assert hasattr(learner.evaluation_result, 'test_set')

        # Save and reload
        path = tmp_path / 'test.learner'
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        # evaluation_result is computed on read, so it must work after load
        eval_result = loaded.evaluation_result
        assert eval_result is not None
        assert hasattr(eval_result, 'test_set')
        assert not eval_result.test_set.empty


# ======================================================================
# Predict CI tests
# ======================================================================

class TestCLIPredictCIParser:
    """Parser-level tests for predict CI flags."""

    def test_predict_accepts_load_learner(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'predict', 'test.csv', '--load-learner', 'foo.learner',
        ])
        assert args.load_learner == 'foo.learner'

    def test_predict_ci_flags_defaults(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'predict', 'test.csv', '--load-learner', 'foo.learner',
        ])
        assert args.ci is False
        assert args.n_bootstrap == 200
        assert args.ci_level == 0.95
        assert args.random_state == 42
        assert args.n_jobs == 1

    def test_predict_ci_flags_custom(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'predict', 'test.csv', '--load-learner', 'foo.learner',
            '--ci', '--n-bootstrap', '50', '--ci-level', '0.9',
            '--random-state', '7', '--n-jobs', '4',
        ])
        assert args.ci is True
        assert args.n_bootstrap == 50
        assert args.ci_level == 0.9
        assert args.random_state == 7
        assert args.n_jobs == 4

    def test_predict_load_artifact_no_longer_required(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'predict', 'test.csv', '--load-learner', 'foo.learner',
        ])
        assert args.load_artifact is None

    def test_predict_model_flag(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'predict', 'test.csv', '--load-learner', 'foo.learner',
            '--model', 'coxph',
        ])
        assert args.model == 'coxph'


class TestCLIPredictCIValidation:
    """Validation tests for predict CI flags."""

    def test_predict_ci_with_load_artifact_errors(self, tmp_path):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'data.csv')
        with pytest.raises(SystemExit):
            main([
                'predict', str(data_path),
                '--load-artifact', str(tmp_path), '--ci',
            ])

    def test_predict_no_source_errors(self, tmp_path):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'data.csv')
        with pytest.raises(SystemExit):
            main([
                'predict', str(data_path),
            ])

    def test_predict_ci_level_out_of_range(self, tmp_path):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'data.csv')
        with pytest.raises(SystemExit):
            main([
                'predict', str(data_path),
                '--load-learner', 'foo.learner',
                '--ci', '--ci-level', '1.5',
            ])

    def test_predict_load_learner_and_artifact_mutual_exclusion(self, tmp_path):
        from mlsurv.cli import main

        data_path = _write_sample_csv(tmp_path / 'data.csv')
        with pytest.raises(SystemExit):
            main([
                'predict', str(data_path),
                '--load-artifact', str(tmp_path),
                '--load-learner', 'foo.learner',
            ])


class TestCLIPredictFromLearner:
    """Integration tests for predict --load-learner path."""

    @pytest.fixture(scope='class')
    def trained_learner_path(self, tmp_path_factory):
        """Train coxph on small data and save learner once for the class."""
        import numpy as np
        from mlsurv import SurvivalLearner

        rng = np.random.RandomState(42)
        n, p = 50, 5
        X = pd.DataFrame(rng.randn(n, p), columns=[f'x{i}' for i in range(p)])
        y = np.array(
            [(bool(rng.rand() > 0.3), rng.exponential(100)) for _ in range(n)],
            dtype=[('event', bool), ('time', float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=2)
        learner.train('coxph')

        tmp = tmp_path_factory.mktemp('predict_ci')
        learner_path = tmp / 'session.learner'
        learner.save_learner(learner_path)

        pred_csv = tmp / 'newpatients.csv'
        X_new = pd.DataFrame(rng.randn(4, p), columns=[f'x{i}' for i in range(p)])
        X_new.to_csv(pred_csv, index=False)

        return learner_path, pred_csv

    def test_predict_from_learner_basic(self, trained_learner_path, capsys):
        from mlsurv.cli import main

        learner_path, pred_csv = trained_learner_path
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert 'risk_score' in out
        assert 'median_survival' in out

    def test_predict_from_learner_without_ci_no_ci_columns(
        self, trained_learner_path, tmp_path, capsys,
    ):
        from mlsurv.cli import main

        learner_path, pred_csv = trained_learner_path
        output_path = tmp_path / 'pred_no_ci.csv'
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--output', str(output_path),
        ])
        assert rc == 0
        out = pd.read_csv(output_path)
        assert 'risk_score' in out.columns
        ci_cols = [c for c in out.columns if c.endswith('_lower') or c.endswith('_upper')]
        assert ci_cols == []

    def test_predict_from_learner_defaults_no_ci_warning(
        self, trained_learner_path, capsys,
    ):
        """Plain predict (no --ci) must not warn — the CLI random_state default
        (42) differs from predict()'s API default, so the gating must keep the
        no-op CI knobs from tripping the ignored-param warning."""
        from mlsurv.cli import main

        learner_path, pred_csv = trained_learner_path
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
        ])
        assert rc == 0
        assert 'require --ci' not in capsys.readouterr().err

    @pytest.fixture(scope='class')
    def two_model_learner_path(self, tmp_path_factory):
        """Class-scoped learner with two trained models (coxph, coxnet)
        plus a 4-patient prediction CSV, for multi-model predict tests."""
        import numpy as np
        from mlsurv import SurvivalLearner

        rng = np.random.RandomState(42)
        n, p = 50, 5
        X = pd.DataFrame(rng.randn(n, p), columns=[f'x{i}' for i in range(p)])
        y = np.array(
            [(bool(rng.rand() > 0.3), rng.exponential(100)) for _ in range(n)],
            dtype=[('event', bool), ('time', float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=2)
        learner.train(['coxph', 'coxnet'])

        tmp = tmp_path_factory.mktemp('predict_two_models')
        learner_path = tmp / 'session.learner'
        learner.save_learner(learner_path)

        pred_csv = tmp / 'newpatients.csv'
        X_new = pd.DataFrame(rng.randn(4, p), columns=[f'x{i}' for i in range(p)])
        X_new.to_csv(pred_csv, index=False)

        return learner_path, pred_csv

    def test_predict_multi_model_without_model_flag_combined_table(
        self, two_model_learner_path, capsys,
    ):
        """Omitting --model with several models emits one combined table."""
        from mlsurv.cli import main

        learner_path, pred_csv = two_model_learner_path
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'Multiple models trained' not in captured.err
        assert 'risk_score' in captured.out
        assert 'coxph' in captured.out
        assert 'coxnet' in captured.out

    def test_predict_multi_model_with_model_flag_succeeds(
        self, two_model_learner_path, capsys,
    ):
        from mlsurv.cli import main

        learner_path, pred_csv = two_model_learner_path
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--model', 'coxph',
        ])
        assert rc == 0
        assert 'risk_score' in capsys.readouterr().out

    def test_predict_multi_model_output_csv_stacked(
        self, two_model_learner_path, tmp_path, capsys,
    ):
        """Combined CSV stacks one block per model, model_name first."""
        from mlsurv.cli import main

        learner_path, pred_csv = two_model_learner_path
        output_path = tmp_path / 'pred_multi.csv'
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--times', '50',
            '--output', str(output_path),
        ])
        assert rc == 0
        out = pd.read_csv(output_path)
        assert out.columns[0] == 'model_name'
        assert set(out['model_name']) == {'coxph', 'coxnet'}
        assert len(out) == 8  # 4 patients x 2 models
        # One contiguous block per model (2 block starts, not interleaved)
        assert out['model_name'].ne(out['model_name'].shift()).sum() == 2
        assert out['risk_score'].notna().all()
        assert 'survival_50' in out.columns

    def test_predict_multi_model_json_format(
        self, two_model_learner_path, capsys,
    ):
        """--format json includes the model_name field for every model."""
        import json

        from mlsurv.cli import main

        learner_path, pred_csv = two_model_learner_path
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--format', 'json',
        ])
        assert rc == 0
        records = json.loads(capsys.readouterr().out)
        assert len(records) == 8  # 4 patients x 2 models
        assert {rec['model_name'] for rec in records} == {'coxph', 'coxnet'}

    @pytest.fixture
    def _degrade_coxnet_survival(self, monkeypatch):
        """Make coxnet's prediction lack survival probabilities."""
        import numpy as np

        from mlsurv.learner._prediction import PredictionMixin
        from mlsurv.results._prediction import PredictionResult

        orig = PredictionMixin._predict_single

        def fake_single(self, model_result, X, **kwargs):
            if model_result.name == 'coxnet':
                return PredictionResult(
                    risk_scores=np.zeros(len(X)),
                    model_name='coxnet',
                    n_patients=len(X),
                    _surv_probs=None,
                )
            return orig(self, model_result, X, **kwargs)

        monkeypatch.setattr(PredictionMixin, '_predict_single', fake_single)

    @pytest.fixture
    def _skip_coxnet_entirely(self, monkeypatch):
        """Make coxnet's prediction raise so predict() skips the model."""
        from mlsurv.learner._prediction import PredictionMixin

        orig = PredictionMixin._predict_single

        def fake_single(self, model_result, X, **kwargs):
            if model_result.name == 'coxnet':
                raise RuntimeError('boom')
            return orig(self, model_result, X, **kwargs)

        monkeypatch.setattr(PredictionMixin, '_predict_single', fake_single)

    def test_predict_multi_model_strict_fails_when_any_model_degrades(
        self, two_model_learner_path, _degrade_coxnet_survival, capsys,
    ):
        """--strict + --times fails when ANY model lacks survival columns."""
        from mlsurv.cli import main
        from mlsurv.cli._exit_codes import EXIT_COMPUTATION

        learner_path, pred_csv = two_model_learner_path
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--times', '50', '--strict',
        ])
        assert rc == EXIT_COMPUTATION
        err = capsys.readouterr().err
        assert 'Survival function prediction failed' in err
        assert 'coxnet' in err

    def test_predict_multi_model_strict_fails_when_model_skipped(
        self, two_model_learner_path, _skip_coxnet_entirely, capsys,
    ):
        """--strict + --times fails when a model is skipped entirely."""
        from mlsurv.cli import main
        from mlsurv.cli._exit_codes import EXIT_COMPUTATION

        learner_path, pred_csv = two_model_learner_path
        with pytest.warns(UserWarning, match="skipped"):
            rc = main([
                'predict', str(pred_csv),
                '--load-learner', str(learner_path),
                '--times', '50', '--strict',
            ])
        assert rc == EXIT_COMPUTATION
        err = capsys.readouterr().err
        assert 'coxnet' in err

    def test_predict_multi_model_strict_fails_when_model_skipped_no_times(
        self, two_model_learner_path, _skip_coxnet_entirely, capsys,
    ):
        """--strict fails on a fully-skipped model even without --times."""
        from mlsurv.cli import main
        from mlsurv.cli._exit_codes import EXIT_COMPUTATION

        learner_path, pred_csv = two_model_learner_path
        with pytest.warns(UserWarning, match="skipped"):
            rc = main([
                'predict', str(pred_csv),
                '--load-learner', str(learner_path),
                '--strict',
            ])
        assert rc == EXIT_COMPUTATION
        err = capsys.readouterr().err
        assert 'Prediction failed' in err
        assert 'coxnet' in err

    def test_predict_multi_model_nonstrict_skipped_model_omitted(
        self, two_model_learner_path, _skip_coxnet_entirely, tmp_path,
        capsys,
    ):
        """Without --strict, a fully-skipped model is omitted from the table
        and the CLI says so on stderr (even without --times)."""
        from mlsurv.cli import main

        learner_path, pred_csv = two_model_learner_path
        output_path = tmp_path / 'pred_skipped.csv'
        with pytest.warns(UserWarning, match="skipped"):
            rc = main([
                'predict', str(pred_csv),
                '--load-learner', str(learner_path),
                '--output', str(output_path),
            ])
        assert rc == 0
        err = capsys.readouterr().err
        assert 'coxnet' in err and 'omitted' in err
        out = pd.read_csv(output_path)
        assert len(out) == 4
        assert set(out['model_name']) == {'coxph'}

    def test_predict_multi_model_nonstrict_degrades_per_model(
        self, two_model_learner_path, _degrade_coxnet_survival, tmp_path,
        capsys,
    ):
        """Without --strict, degraded models keep risk_score rows with the
        survival columns NaN-filled while healthy models keep full output."""
        from mlsurv.cli import main

        learner_path, pred_csv = two_model_learner_path
        output_path = tmp_path / 'pred_degraded.csv'
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--times', '50',
            '--output', str(output_path),
        ])
        assert rc == 0
        err = capsys.readouterr().err
        assert 'coxnet' in err and 'risk_score' in err
        out = pd.read_csv(output_path)
        assert len(out) == 8
        coxph_rows = out[out['model_name'] == 'coxph']
        coxnet_rows = out[out['model_name'] == 'coxnet']
        assert coxph_rows['survival_50'].notna().all()
        assert coxnet_rows['survival_50'].isna().all()
        assert coxnet_rows['risk_score'].notna().all()

    @pytest.mark.integration
    def test_predict_multi_model_with_ci(
        self, two_model_learner_path, tmp_path, capsys,
    ):
        """--ci works through the multi-model path: CI columns present and
        populated for every model block."""
        from mlsurv.cli import main

        learner_path, pred_csv = two_model_learner_path
        output_path = tmp_path / 'pred_multi_ci.csv'
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--ci', '--n-bootstrap', '3',
            '--times', '50',
            '--output', str(output_path),
        ])
        assert rc == 0
        out = pd.read_csv(output_path)
        assert len(out) == 8
        assert 'survival_50_lower' in out.columns
        assert 'survival_50_upper' in out.columns
        for name in ('coxph', 'coxnet'):
            block = out[out['model_name'] == name]
            assert block['survival_50_lower'].notna().all()
            assert block['survival_50_upper'].notna().all()

    def test_predict_from_learner_ci_knob_without_ci_warns(
        self, trained_learner_path, capsys,
    ):
        """Explicitly setting a CI knob without --ci must not be dropped
        silently — emit a stderr note naming the ignored flag."""
        from mlsurv.cli import main

        learner_path, pred_csv = trained_learner_path
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--n-bootstrap', '500',
        ])
        assert rc == 0
        err = capsys.readouterr().err
        assert '--n-bootstrap' in err and 'require --ci' in err

    def test_predict_ci_band_without_ci_warns(
        self, trained_learner_path, capsys,
    ):
        """--ci-band without --ci must warn, not silently drop (no silent
        data loss)."""
        from mlsurv.cli import main

        learner_path, pred_csv = trained_learner_path
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--ci-band',
        ])
        assert rc == 0
        err = capsys.readouterr().err
        assert '--ci-band' in err and 'require --ci' in err

    @pytest.mark.integration
    def test_predict_ci_band_passthrough(
        self, trained_learner_path, tmp_path,
    ):
        """--ci --ci-band threads end-to-end and still yields CI columns."""
        from mlsurv.cli import main

        learner_path, pred_csv = trained_learner_path
        output_path = tmp_path / 'pred_ci_band.csv'
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--ci', '--ci-band', '--n-bootstrap', '3',
            '--times', '10',
            '--output', str(output_path),
        ])
        assert rc == 0
        out = pd.read_csv(output_path)
        assert 'survival_10_lower' in out.columns
        assert 'survival_10_upper' in out.columns

    @pytest.mark.integration
    def test_predict_from_learner_with_ci(
        self, trained_learner_path, tmp_path, capsys,
    ):
        from mlsurv.cli import main

        learner_path, pred_csv = trained_learner_path
        output_path = tmp_path / 'pred_ci.csv'
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--ci', '--n-bootstrap', '3',
            '--output', str(output_path),
        ])
        assert rc == 0
        out = pd.read_csv(output_path)
        assert 'median_survival' in out.columns
        assert 'median_survival_lower' in out.columns
        assert 'median_survival_upper' in out.columns
        assert (out['median_survival_lower'] <= out['median_survival_upper']).all()

    @pytest.mark.integration
    def test_predict_from_learner_with_ci_and_times(
        self, trained_learner_path, tmp_path, capsys,
    ):
        from mlsurv.cli import main

        learner_path, pred_csv = trained_learner_path
        output_path = tmp_path / 'pred_ci_times.csv'
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--ci', '--n-bootstrap', '3',
            '--times', '50', '100',
            '--output', str(output_path),
        ])
        assert rc == 0
        out = pd.read_csv(output_path)

        assert 'survival_50' in out.columns
        assert 'survival_50_lower' in out.columns
        assert 'survival_50_upper' in out.columns
        assert 'survival_100' in out.columns
        assert 'survival_100_lower' in out.columns
        assert 'survival_100_upper' in out.columns

        assert 'risk_50' in out.columns
        assert 'risk_50_lower' in out.columns
        assert 'risk_50_upper' in out.columns

        import numpy as np
        np.testing.assert_allclose(
            out['risk_50_lower'].values,
            1.0 - out['survival_50_upper'].values,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            out['risk_50_upper'].values,
            1.0 - out['survival_50_lower'].values,
            atol=1e-8,
        )

        assert 'rmst_50' in out.columns
        assert 'rmst_50_lower' in out.columns
        assert 'rmst_50_upper' in out.columns

        # Default-tau RMST CI must also be present alongside user-time CIs
        rmst_cols = [c for c in out.columns if c.startswith('rmst_')]
        default_rmst_cols = [c for c in rmst_cols
                             if c not in ('rmst_50', 'rmst_50_lower', 'rmst_50_upper',
                                          'rmst_100', 'rmst_100_lower', 'rmst_100_upper')]
        # Should have rmst_{default_tau}, rmst_{default_tau}_lower, rmst_{default_tau}_upper
        assert len(default_rmst_cols) == 3
        lower_cols = [c for c in default_rmst_cols if c.endswith('_lower')]
        upper_cols = [c for c in default_rmst_cols if c.endswith('_upper')]
        assert len(lower_cols) == 1
        assert len(upper_cols) == 1

    @pytest.mark.integration
    def test_predict_from_learner_csv_output(
        self, trained_learner_path, tmp_path,
    ):
        from mlsurv.cli import main

        learner_path, pred_csv = trained_learner_path
        output_path = tmp_path / 'pred_out.csv'
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--ci', '--n-bootstrap', '3',
            '--output', str(output_path),
        ])
        assert rc == 0
        assert output_path.exists()
        out = pd.read_csv(output_path)
        assert len(out) == 4
        assert 'sample_idx' in out.columns
        assert 'model_name' in out.columns

    def test_predict_from_learner_file_not_found(self, tmp_path, capsys):
        from mlsurv.cli import main

        pred_csv = _write_sample_csv(tmp_path / 'data.csv')
        rc = main([
            'predict', str(pred_csv),
            '--load-learner', '/nonexistent/path.learner',
        ])
        assert rc == 3  # EXIT_DATA
        err = capsys.readouterr().err
        assert 'Error' in err

    def test_predict_from_learner_strict_with_no_survival(
        self, trained_learner_path, monkeypatch, capsys,
    ):
        """--strict returns 1 when survival fails in the learner path."""
        from mlsurv.cli import main
        from mlsurv.results._prediction import PredictionResult
        import numpy as np

        learner_path, pred_csv = trained_learner_path

        def fake_predict(self, X, **kwargs):
            return PredictionResult(
                risk_scores=np.zeros(len(X)),
                model_name='coxph',
                n_patients=len(X),
                _surv_probs=None,
            )

        monkeypatch.setattr(
            'mlsurv.learner._prediction.PredictionMixin.predict',
            fake_predict,
        )

        rc = main([
            'predict', str(pred_csv),
            '--load-learner', str(learner_path),
            '--times', '50', '--strict',
        ])
        assert rc == 4  # EXIT_COMPUTATION
        err = capsys.readouterr().err
        assert 'Survival function prediction failed' in err


class TestCLIWriteHelperOSError:
    """OSError handling in CLI write helpers (_write_csv, _write_json, _write_report)."""

    def test_write_csv_oserror_warns_on_stderr(self, monkeypatch, capsys):
        """_write_csv catches OSError and prints warning to stderr."""
        from mlsurv.cli._helpers import _write_csv

        df = pd.DataFrame({'a': [1, 2]})

        def _failing_to_csv(*args, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr(pd.DataFrame, 'to_csv', _failing_to_csv)

        _write_csv(df, '/fake/path.csv', 'Test output')

        captured = capsys.readouterr()
        assert 'Warning: Failed to write Test output' in captured.err
        assert '/fake/path.csv' in captured.err
        assert captured.out == ''

    def test_write_json_oserror_warns_on_stderr(self, tmp_path, capsys):
        """_write_json catches OSError and prints warning to stderr."""
        from mlsurv.cli._helpers import _write_json

        bad_path = str(tmp_path / 'nonexistent_dir' / 'out.json')

        _write_json({'key': 'val'}, bad_path, 'Tune results')

        captured = capsys.readouterr()
        assert 'Warning: Failed to write Tune results' in captured.err
        assert bad_path in captured.err
        assert captured.out == ''

    def test_write_json_typeerror_warns_on_stderr(self, tmp_path, capsys):
        """_write_json catches TypeError from non-serializable payloads."""
        from mlsurv.cli._helpers import _write_json
        import numpy as np

        out_path = str(tmp_path / 'out.json')
        _write_json({'val': np.int64(42)}, out_path, 'Results')

        captured = capsys.readouterr()
        assert 'Warning: Failed to write Results' in captured.err
        assert out_path in captured.err
        assert captured.out == ''

    def test_write_json_success_prints_confirmation(self, tmp_path, capsys):
        """_write_json prints confirmation on success."""
        from mlsurv.cli._helpers import _write_json

        out_path = str(tmp_path / 'out.json')
        _write_json({'key': 'val'}, out_path, 'Results')

        captured = capsys.readouterr()
        assert 'Results saved to:' in captured.out
        assert captured.err == ''

    def test_write_json_quiet_suppresses_confirmation(self, tmp_path, capsys):
        """_write_json with quiet=True suppresses the success message."""
        from mlsurv.cli._helpers import _write_json

        out_path = str(tmp_path / 'out.json')
        _write_json({'key': 'val'}, out_path, 'Results', quiet=True)

        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''

    def test_write_report_oserror_warns_on_stderr(self, capsys):
        """_write_report catches OSError and prints warning to stderr."""
        from mlsurv.cli._helpers import _write_report

        class FakeLearner:
            def generate_report(self, *args, **kwargs):
                raise OSError("disk full")

        _write_report(FakeLearner(), '/fake/report.html')

        captured = capsys.readouterr()
        assert 'Warning: Failed to write report' in captured.err
        assert '/fake/report.html' in captured.err
        assert captured.out == ''

    def test_write_report_skips_when_no_path(self, capsys):
        """_write_report returns immediately when report_path is falsy."""
        from mlsurv.cli._helpers import _write_report

        class FakeLearner:
            def generate_report(self, *args, **kwargs):
                raise AssertionError("should not be called")

        _write_report(FakeLearner(), None)
        _write_report(FakeLearner(), '')

        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''

    def test_write_report_success_prints_confirmation(self, capsys):
        """_write_report prints confirmation on success."""
        from mlsurv.cli._helpers import _write_report

        class FakeLearner:
            def generate_report(self, *args, **kwargs):
                pass

        _write_report(FakeLearner(), '/some/report.html')

        captured = capsys.readouterr()
        assert 'Report saved to:' in captured.out
        assert captured.err == ''


# ===================================================================
# CLI Nice-to-Haves
# ===================================================================


class TestCLIExitCodes:
    """Exit code constants and semantic exit code behaviour."""

    def test_exit_code_constants_importable(self):
        from mlsurv.cli._exit_codes import (
            EXIT_COMPUTATION,
            EXIT_DATA,
            EXIT_ERROR,
            EXIT_INTERRUPT,
            EXIT_SUCCESS,
            EXIT_USAGE,
        )
        assert EXIT_SUCCESS == 0
        assert EXIT_ERROR == 1
        assert EXIT_USAGE == 2
        assert EXIT_DATA == 3
        assert EXIT_COMPUTATION == 4
        assert EXIT_INTERRUPT == 130

    def test_file_not_found_returns_exit_data(self, tmp_path):
        from mlsurv.cli import main

        rc = main(['setup', str(tmp_path / 'nonexistent.csv')])
        assert rc == 3  # EXIT_DATA

    def test_usage_error_returns_exit_usage(self):
        """argparse exits with code 2 for missing required args."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['train'])  # missing required --model and data
        assert exc_info.value.code == 2

    def test_unknown_command_returns_exit_usage(self):
        """Unknown subcommand triggers argparse error (code 2)."""
        from mlsurv.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(['nonexistent-command'])
        assert exc_info.value.code == 2  # EXIT_USAGE (argparse convention)


class TestCLIFormat:
    """``--format`` output mode tests."""

    def test_parser_accepts_format_json(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['models', '--format', 'json'])
        assert args.output_format == 'json'

    def test_parser_format_default_is_table(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['models'])
        assert args.output_format == 'table'

    def test_parser_format_csv(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['setup', 'data.csv', '--format', 'csv'])
        assert args.output_format == 'csv'

    def test_format_dataframe_json(self):
        from mlsurv.cli._output import format_dataframe

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = format_dataframe(df, 'json')
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0] == {'a': 1, 'b': 3}

    def test_format_dataframe_csv(self):
        from mlsurv.cli._output import format_dataframe

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = format_dataframe(df, 'csv')
        assert 'a,b' in result
        assert '1,3' in result

    def test_format_dataframe_table(self):
        from mlsurv.cli._output import format_dataframe

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = format_dataframe(df, 'table')
        assert 'a' in result
        assert 'b' in result

    def test_format_dict_json(self):
        from mlsurv.cli._output import format_dict

        data = {'key1': 'val1', 'key2': 42}
        result = format_dict(data, 'json')
        parsed = json.loads(result)
        assert parsed == data

    def test_format_dict_csv(self):
        from mlsurv.cli._output import format_dict

        data = {'key1': 'val1', 'key2': 42}
        result = format_dict(data, 'csv')
        assert 'key,value' in result
        assert 'key1,val1' in result

    def test_format_dict_table_returns_none(self):
        from mlsurv.cli._output import format_dict

        result = format_dict({'a': 1}, 'table')
        assert result is None

    def test_format_list_json(self):
        from mlsurv.cli._output import format_list

        result = format_list(['a', 'b', 'c'], 'json')
        parsed = json.loads(result)
        assert parsed == ['a', 'b', 'c']

    def test_format_list_csv(self):
        from mlsurv.cli._output import format_list

        result = format_list(['a', 'b', 'c'], 'csv')
        assert result == 'a\nb\nc'

    def test_cmd_models_format_json(self, capsys):
        from mlsurv.cli import main

        rc = main(['models', '--format', 'json'])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
        assert 'coxph' in parsed

    def test_cmd_models_format_json_verbose(self, capsys):
        from mlsurv.cli import main

        rc = main(['models', '--format', 'json', '--verbose'])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert all('name' in item and 'tunable' in item for item in parsed)

    def test_cmd_models_format_csv(self, capsys):
        from mlsurv.cli import main

        rc = main(['models', '--format', 'csv'])
        assert rc == 0
        out = capsys.readouterr().out
        lines = out.strip().split('\n')
        assert 'coxph' in lines

    def test_format_json_suppresses_headers(self, capsys):
        """JSON output should not contain headers or separators."""
        from mlsurv.cli import main

        rc = main(['models', '--format', 'json'])
        assert rc == 0
        out = capsys.readouterr().out
        assert '---' not in out
        assert '===' not in out

    def test_format_does_not_collide_with_export_study(self):
        """export-study has its own --format; our --format uses dest=output_format."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'export-study', '--experiment', 'e', '--model', 'm',
            '--output', 'out.json', '--format', 'json',
        ])
        # export-study's --format uses default dest='format'
        assert args.format == 'json'
        # our --format is not present on export-study
        assert not hasattr(args, 'output_format')


class TestCLIStdin:
    """Stdin/pipe support tests."""

    def test_validate_csv_path_dash_returns_none(self):
        from mlsurv.cli._data import _validate_csv_path

        result = _validate_csv_path('-')
        assert result is None

    def test_cli_load_stdin_reads_data(self, tmp_path, monkeypatch):
        import io

        csv_content = "f1,f2,time,event\n0.1,1.0,10,1\n0.2,0.0,12,0\n0.3,1.0,8,1\n"
        monkeypatch.setattr('sys.stdin', io.StringIO(csv_content))
        monkeypatch.setattr('sys.stdin.isatty', lambda: False)

        from argparse import Namespace

        from mlsurv.cli._data import _cli_load

        args = Namespace(
            data='-', time_col='time', event_col='event',
            start_col=None, end_col=None, time_unit='days',
            event_values=None,
        )
        X, y = _cli_load(args)
        assert len(X) == 3
        assert 'f1' in X.columns

    def test_cli_load_stdin_empty_exits(self, monkeypatch):
        import io

        from mlsurv.cli._exit_codes import CLIDataError

        monkeypatch.setattr('sys.stdin', io.StringIO(''))
        monkeypatch.setattr('sys.stdin.isatty', lambda: False)

        from argparse import Namespace

        from mlsurv.cli._data import _cli_load

        args = Namespace(
            data='-', time_col='time', event_col='event',
            start_col=None, end_col=None, time_unit='days',
            event_values=None,
        )
        with pytest.raises(CLIDataError):
            _cli_load(args)

    def test_cli_load_stdin_on_tty_exits(self, monkeypatch):
        import io

        from mlsurv.cli._exit_codes import CLIDataError

        mock_stdin = io.StringIO('')
        monkeypatch.setattr('sys.stdin', mock_stdin)
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)

        from argparse import Namespace

        from mlsurv.cli._data import _cli_load

        args = Namespace(
            data='-', time_col='time', event_col='event',
            start_col=None, end_col=None, time_unit='days',
            event_values=None,
        )
        with pytest.raises(CLIDataError):
            _cli_load(args)


class TestCLIArgcomplete:
    """argcomplete integration tests."""

    def test_argcomplete_import_guard(self, monkeypatch):
        """main() handles missing argcomplete gracefully."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'argcomplete':
                raise ImportError("No module named 'argcomplete'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)
        from mlsurv.cli import main

        # Should work fine without argcomplete (just prints help)
        rc = main([])
        assert rc == 0
