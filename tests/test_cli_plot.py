"""Tests for the mlsurv plot CLI subcommand."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mlsurv.cli import create_parser, main
from mlsurv.cli._plot import (
    _ALL_TYPE_FLAGS,
    _FLAG_DEFAULTS,
    _PLOT_REGISTRY,
    _check_prerequisite,
    _generate_inline_plots,
    _validate_type_flags,
    cmd_plot,
)


# ---------------------------------------------------------------------------
# Parser tests (fast tier)
# ---------------------------------------------------------------------------

class TestPlotParser:
    """Verify the plot subparser accepts all expected flags."""

    def test_parser_recognises_plot_subcommand(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', '--list', '--load-learner', 'dummy.learner',
        ])
        assert args.command == 'plot'

    def test_parser_list_without_load_learner(self):
        parser = create_parser()
        args = parser.parse_args(['plot', '--list'])
        assert args.command == 'plot'
        assert args.list is True
        assert args.load_learner is None

    def test_parser_type_positional(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'survival', '--load-learner', 'x.learner',
        ])
        assert args.type == 'survival'

    def test_parser_type_optional(self):
        parser = create_parser()
        args = parser.parse_args(['plot', '--list'])
        assert args.type is None

    def test_parser_output_and_format(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'metric', '--load-learner', 'x.learner',
            '-o', 'out.png', '-f', 'png',
        ])
        assert args.output == 'out.png'
        assert args.format == 'png'

    def test_parser_model_nargs_plus(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'metric', '--load-learner', 'x.learner',
            '--model', 'coxph', 'rsf',
        ])
        assert args.models == ['coxph', 'rsf']

    def test_parser_no_open(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'survival', '--load-learner', 'x.learner', '--no-open',
        ])
        assert args.no_open is True

    def test_parser_survival_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'survival', '--load-learner', 'x.learner',
            '--n-groups', '3', '--show-risk-table', '--time-limit', '60.0',
        ])
        assert args.n_groups == 3
        assert args.show_risk_table is True
        assert args.time_limit == 60.0

    def test_parser_metric_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'metric', '--load-learner', 'x.learner',
            '--style', 'forest', '--metric', 'harrell',
            '--source', 'cv', '--orientation', 'vertical',
        ])
        assert args.style == 'forest'
        assert args.metric == 'harrell'
        assert args.source == 'cv'
        assert args.orientation == 'vertical'

    def test_parser_importance_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'importance', '--load-learner', 'x.learner',
            '--n-features', '10', '--style', 'heatmap',
        ])
        assert args.n_features == 10
        assert args.style == 'heatmap'

    def test_parser_roc_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'roc', '--load-learner', 'x.learner',
            '--eval-time', '12.5', '--no-auc', '--no-ci',
        ])
        assert args.eval_time == 12.5
        assert args.no_auc is True
        assert args.no_ci is True

    def test_parser_calibration_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'calibration', '--load-learner', 'x.learner',
            '--n-bins', '5', '--eval-time', '24.0',
        ])
        assert args.n_bins == 5
        assert args.eval_time == 24.0

    def test_parser_references_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'survival', '--load-learner', 'x.learner', '--references',
        ])
        assert args.references is True

    def test_parser_defaults(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'missing', '--load-learner', 'x.learner',
        ])
        assert args.format == 'html'
        assert args.output is None
        assert args.no_open is False
        assert args.style == 'auto'
        assert args.n_features == 20
        assert args.n_bins == 10
        assert args.models is None
        # Phase 2 defaults
        assert args.kind is None
        assert args.features is None
        assert args.metrics is None
        assert args.threshold == 0.8
        # Phase 3 defaults
        assert args.max_samples == 100
        assert args.grid_resolution == 25
        assert args.eval_times is None
        assert args.return_data is False

    # --- Phase 2 parser tests ---

    def test_parser_kind_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'diagnostics', '--load-learner', 'x.learner',
            '--kind', 'schoenfeld',
        ])
        assert args.kind == 'schoenfeld'

    def test_parser_features_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'diagnostics', '--load-learner', 'x.learner',
            '--features', 'age', 'grade',
        ])
        assert args.features == ['age', 'grade']

    def test_parser_metrics_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'metric-time', '--load-learner', 'x.learner',
            '--metrics', 'harrell', 'uno',
        ])
        assert args.metrics == ['harrell', 'uno']

    def test_parser_threshold_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'feature-correlation', '--load-learner', 'x.learner',
            '--threshold', '0.5',
        ])
        assert args.threshold == 0.5

    # --- Phase 3 parser tests ---

    def test_parser_max_samples_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'interaction-pdp', '--load-learner', 'x.learner',
            '--max-samples', '50',
        ])
        assert args.max_samples == 50

    def test_parser_grid_resolution_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'interaction-pdp', '--load-learner', 'x.learner',
            '--grid-resolution', '30',
        ])
        assert args.grid_resolution == 30

    def test_parser_eval_times_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'calibration-overlay', '--load-learner', 'x.learner',
            '--eval-times', '12.0', '24.0', '36.0',
        ])
        assert args.eval_times == [12.0, 24.0, 36.0]

    def test_parser_return_data_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'plot', 'shap', '--load-learner', 'x.learner', '--return-data',
        ])
        assert args.return_data is True


# ---------------------------------------------------------------------------
# Registry tests (fast tier)
# ---------------------------------------------------------------------------

class TestPlotRegistry:
    """Verify registry entries are consistent."""

    def test_all_registry_methods_exist_on_learner(self):
        from mlsurv import SurvivalLearner
        for name, spec in _PLOT_REGISTRY.items():
            assert hasattr(SurvivalLearner, spec.method), (
                f"Registry type '{name}' references non-existent method '{spec.method}'"
            )

    def test_all_methods_accept_return_data(self):
        import inspect
        from mlsurv import SurvivalLearner
        for name, spec in _PLOT_REGISTRY.items():
            method = getattr(SurvivalLearner, spec.method)
            sig = inspect.signature(method)
            assert 'return_data' in sig.parameters, (
                f"Method '{spec.method}' (type '{name}') missing return_data parameter"
            )

    def test_valid_flags_are_subset_of_all(self):
        for name, spec in _PLOT_REGISTRY.items():
            assert spec.valid_flags <= _ALL_TYPE_FLAGS, (
                f"Type '{name}' has flags not in _ALL_TYPE_FLAGS: "
                f"{spec.valid_flags - _ALL_TYPE_FLAGS}"
            )

    def test_build_kwargs_returns_dict(self):
        args = Namespace(**_FLAG_DEFAULTS)
        for name, spec in _PLOT_REGISTRY.items():
            if name == 'interaction-pdp':
                # interaction-pdp requires exactly 2 features
                pdp_args = Namespace(**{**_FLAG_DEFAULTS, 'features': ['f0', 'f1']})
                result = spec.build_kwargs(pdp_args)
            else:
                result = spec.build_kwargs(args)
            assert isinstance(result, dict), (
                f"build_kwargs for '{name}' returned {type(result)}, expected dict"
            )

    def test_prerequisite_levels_are_valid(self):
        valid_levels = {'setup', 'train', 'evaluate', 'tune', 'benchmark',
                        'validate'}
        for name, spec in _PLOT_REGISTRY.items():
            assert spec.prerequisite in valid_levels, (
                f"Type '{name}' has unknown prerequisite '{spec.prerequisite}'"
            )

    def test_parity_types_in_registry(self):
        """benchmark-scatter and validation-heatmap close the plot parity gap."""
        for t in ('benchmark-scatter', 'validation-heatmap'):
            assert t in _PLOT_REGISTRY, f"Plot type '{t}' missing from registry"

    def test_phase2_types_in_registry(self):
        phase2 = {
            'diagnostics', 'coefficients', 'hazard-ratios',
            'feature-correlation', 'feature-distributions', 'metric-time',
        }
        for t in phase2:
            assert t in _PLOT_REGISTRY, f"Phase 2 type '{t}' missing from registry"

    def test_phase3_types_in_registry(self):
        phase3 = {
            'shap', 'interactions', 'interaction-pdp',
            'calibration-overlay', 'optuna', 'metric-heatmap',
            'hazard-ratios-comparison', 'coefficient-comparison',
            'benchmark-incremental',
        }
        for t in phase3:
            assert t in _PLOT_REGISTRY, f"Phase 3 type '{t}' missing from registry"

    def test_model_param_field(self):
        for name, spec in _PLOT_REGISTRY.items():
            assert spec.model_param in ('models', 'model', None), (
                f"Type '{name}' has invalid model_param '{spec.model_param}'"
            )


# ---------------------------------------------------------------------------
# Prerequisite & flag validation (fast tier)
# ---------------------------------------------------------------------------

class TestPrerequisiteChecking:

    def _make_learner_mock(self, *, n_models=0, has_eval=False):
        learner = MagicMock()
        models = {}
        for i in range(n_models):
            mr = MagicMock()
            mr.evaluation = MagicMock() if has_eval else None
            models[f'model_{i}'] = mr
        learner.models = models
        return learner

    def test_setup_always_passes(self):
        learner = self._make_learner_mock()
        spec = _PLOT_REGISTRY['missing']
        assert _check_prerequisite(learner, spec, 'missing') is None

    def test_train_passes_with_models(self):
        learner = self._make_learner_mock(n_models=1)
        spec = _PLOT_REGISTRY['survival']
        assert _check_prerequisite(learner, spec, 'survival') is None

    def test_train_fails_without_models(self):
        learner = self._make_learner_mock(n_models=0)
        spec = _PLOT_REGISTRY['survival']
        err = _check_prerequisite(learner, spec, 'survival')
        assert err is not None
        assert 'train' in err.lower() or 'run' in err.lower()

    def test_evaluate_passes_with_evaluation(self):
        learner = self._make_learner_mock(n_models=1, has_eval=True)
        spec = _PLOT_REGISTRY['importance']
        assert _check_prerequisite(learner, spec, 'importance') is None

    def test_evaluate_fails_without_evaluation(self):
        learner = self._make_learner_mock(n_models=1, has_eval=False)
        spec = _PLOT_REGISTRY['importance']
        err = _check_prerequisite(learner, spec, 'importance')
        assert err is not None
        assert 'evaluate' in err.lower()

    def test_tune_passes_with_optuna_study(self):
        learner = MagicMock()
        mr = MagicMock()
        mr.optuna_study = MagicMock()
        learner.models = {'rsf': mr}
        spec = _PLOT_REGISTRY['optuna']
        assert _check_prerequisite(learner, spec, 'optuna') is None

    def test_tune_fails_without_optuna_study(self):
        learner = MagicMock()
        mr = MagicMock()
        mr.optuna_study = None
        mr._optuna_tuning_history = None
        learner.models = {'rsf': mr}
        spec = _PLOT_REGISTRY['optuna']
        err = _check_prerequisite(learner, spec, 'optuna')
        assert err is not None
        assert 'tune' in err.lower()

    def test_benchmark_passes_with_result(self):
        learner = MagicMock()
        learner.benchmark_result = MagicMock()
        learner.models = {}
        spec = _PLOT_REGISTRY['benchmark-incremental']
        assert _check_prerequisite(learner, spec, 'benchmark-incremental') is None

    def test_benchmark_fails_without_result(self):
        learner = MagicMock()
        learner.benchmark_result = None
        learner.models = {}
        spec = _PLOT_REGISTRY['benchmark-incremental']
        err = _check_prerequisite(learner, spec, 'benchmark-incremental')
        assert err is not None
        assert 'benchmark' in err.lower()

    def test_benchmark_scatter_fails_without_result(self):
        learner = MagicMock()
        learner.benchmark_result = None
        learner.models = {}
        spec = _PLOT_REGISTRY['benchmark-scatter']
        err = _check_prerequisite(learner, spec, 'benchmark-scatter')
        assert err is not None
        assert 'benchmark' in err.lower()

    def test_validate_passes_with_two_cohorts(self):
        learner = MagicMock()
        learner.validation_results = {'A': MagicMock(), 'B': MagicMock()}
        spec = _PLOT_REGISTRY['validation-heatmap']
        assert _check_prerequisite(learner, spec, 'validation-heatmap') is None

    def test_validate_fails_with_one_cohort(self):
        learner = MagicMock()
        learner.validation_results = {'A': MagicMock()}
        spec = _PLOT_REGISTRY['validation-heatmap']
        err = _check_prerequisite(learner, spec, 'validation-heatmap')
        assert err is not None
        assert 'validation' in err.lower() or 'cohort' in err.lower()


class TestFlagValidation:

    def _make_args(self, **overrides):
        defaults = dict(_FLAG_DEFAULTS)
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_no_error_when_flags_match_type(self):
        args = self._make_args(n_groups=3)
        spec = _PLOT_REGISTRY['survival']
        assert _validate_type_flags(args, spec, 'survival') is None

    def test_error_when_flag_mismatches_type(self):
        args = self._make_args(n_groups=3)
        spec = _PLOT_REGISTRY['missing']
        err = _validate_type_flags(args, spec, 'missing')
        assert err is not None
        assert '--n-groups' in err

    def test_no_error_for_defaults(self):
        args = self._make_args()
        for name, spec in _PLOT_REGISTRY.items():
            assert _validate_type_flags(args, spec, name) is None

    # --- Phase 2 flag validation ---

    def test_kind_invalid_for_calibration(self):
        args = self._make_args(kind='schoenfeld')
        spec = _PLOT_REGISTRY['calibration']
        err = _validate_type_flags(args, spec, 'calibration')
        assert err is not None
        assert '--kind' in err

    def test_features_invalid_for_metric(self):
        args = self._make_args(features=['age'])
        spec = _PLOT_REGISTRY['metric']
        err = _validate_type_flags(args, spec, 'metric')
        assert err is not None
        assert '--features' in err

    def test_threshold_invalid_for_survival(self):
        args = self._make_args(threshold=0.5)
        spec = _PLOT_REGISTRY['survival']
        err = _validate_type_flags(args, spec, 'survival')
        assert err is not None
        assert '--threshold' in err

    def test_metrics_invalid_for_roc(self):
        args = self._make_args(metrics=['harrell'])
        spec = _PLOT_REGISTRY['roc']
        err = _validate_type_flags(args, spec, 'roc')
        assert err is not None
        assert '--metrics' in err

    # --- Phase 3 flag validation ---

    def test_max_samples_invalid_for_shap(self):
        args = self._make_args(max_samples=50)
        spec = _PLOT_REGISTRY['shap']
        err = _validate_type_flags(args, spec, 'shap')
        assert err is not None
        assert '--max-samples' in err

    def test_eval_times_invalid_for_metric(self):
        args = self._make_args(eval_times=[12.0])
        spec = _PLOT_REGISTRY['metric']
        err = _validate_type_flags(args, spec, 'metric')
        assert err is not None
        assert '--eval-times' in err

    def test_grid_resolution_invalid_for_optuna(self):
        args = self._make_args(grid_resolution=30)
        spec = _PLOT_REGISTRY['optuna']
        err = _validate_type_flags(args, spec, 'optuna')
        assert err is not None
        assert '--grid-resolution' in err


# ---------------------------------------------------------------------------
# Handler tests (fast tier, monkeypatched)
# ---------------------------------------------------------------------------

class TestCmdPlot:
    """Test cmd_plot handler without real training."""

    def test_list_returns_zero(self, capsys):
        rc = main(['plot', '--list'])
        assert rc == 0
        captured = capsys.readouterr().out
        assert 'survival' in captured
        assert 'metric' in captured

    def test_unknown_type_returns_one(self, capsys):
        with patch('mlsurv.cli._plot._load_learner_if_requested') as mock_load:
            mock_load.return_value = MagicMock()
            args = Namespace(
                type='nonexistent', list=False, load_learner='x.learner',
                models=None, output=None, format='html', no_open=True,
                quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 1
        captured = capsys.readouterr().err
        assert 'nonexistent' in captured

    def test_missing_load_learner_returns_one(self, capsys):
        args = Namespace(
            type='survival', list=False, load_learner=None,
            models=None, output=None, format='html', no_open=True,
            quiet=False, verbose=False,
            **_FLAG_DEFAULTS,
        )
        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=None):
            rc = cmd_plot(args)
        assert rc == 1

    def test_prerequisite_failure_returns_one(self, capsys):
        learner = MagicMock()
        learner.models = {}  # no trained models
        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='survival', list=False, load_learner='x.learner',
                models=None, output=None, format='html', no_open=True,
                quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 1
        captured = capsys.readouterr().err
        assert 'requires' in captured.lower()

    def test_invalid_model_name_returns_one(self, capsys):
        learner = MagicMock()
        mr = MagicMock()
        mr.evaluation = None
        learner.models = {'coxph': mr}
        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='survival', list=False, load_learner='x.learner',
                models=['nonexistent'], output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 1
        captured = capsys.readouterr().err
        assert 'nonexistent' in captured

    def test_wrong_flag_for_type_returns_one(self, capsys):
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='missing', list=False, load_learner='x.learner',
                models=None, output=None, format='html', no_open=True,
                quiet=False, verbose=False,
                # set a flag that doesn't belong to 'missing'
                **{**_FLAG_DEFAULTS, 'n_groups': 3},
            )
            rc = cmd_plot(args)
        assert rc == 1
        captured = capsys.readouterr().err
        assert '--n-groups' in captured

    def test_successful_plot_to_html(self, tmp_path, capsys):
        fig = MagicMock()
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_survival.return_value = fig

        output_path = str(tmp_path / 'plot.html')
        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='survival', list=False, load_learner='x.learner',
                models=None, output=output_path, format='html',
                no_open=True, quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        fig.write_html.assert_called_once()

    def test_successful_plot_to_png(self, tmp_path):
        fig = MagicMock()
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_survival.return_value = fig

        output_path = str(tmp_path / 'plot.png')
        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='survival', list=False, load_learner='x.learner',
                models=None, output=output_path, format='png',
                no_open=True, quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        fig.write_image.assert_called_once()

    def test_successful_plot_to_json(self, tmp_path):
        fig = MagicMock()
        fig.to_json.return_value = '{"data": []}'
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_survival.return_value = fig

        output_path = str(tmp_path / 'plot.json')
        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='survival', list=False, load_learner='x.learner',
                models=None, output=output_path, format='json',
                no_open=True, quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        fig.to_json.assert_called_once()
        assert Path(output_path).exists()

    def test_no_open_suppresses_browser(self, tmp_path):
        fig = MagicMock()
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_survival.return_value = fig

        output_path = str(tmp_path / 'plot.html')
        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner), \
             patch('mlsurv.cli._plot.webbrowser') as mock_wb:
            args = Namespace(
                type='survival', list=False, load_learner='x.learner',
                models=None, output=output_path, format='html',
                no_open=True, quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        mock_wb.open.assert_not_called()

    def test_models_forwarded_to_plot_method(self, tmp_path):
        fig = MagicMock()
        learner = MagicMock()
        mr = MagicMock()
        mr.evaluation = None
        learner.models = {'coxph': mr, 'rsf': mr}
        learner.plot_survival.return_value = fig

        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='survival', list=False, load_learner='x.learner',
                models=['coxph', 'rsf'], output=str(tmp_path / 'out.html'),
                format='html', no_open=True, quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        call_kwargs = learner.plot_survival.call_args
        assert call_kwargs[1]['models'] == ['coxph', 'rsf']

    def test_kwargs_forwarded_correctly(self, tmp_path):
        fig = MagicMock()
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_survival.return_value = fig

        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='survival', list=False, load_learner='x.learner',
                models=None, output=str(tmp_path / 'out.html'),
                format='html', no_open=True, quiet=False, verbose=False,
                **{**_FLAG_DEFAULTS, 'n_groups': 4, 'show_risk_table': True,
                   'time_limit': 100.0},
            )
            rc = cmd_plot(args)
        assert rc == 0
        call_kwargs = learner.plot_survival.call_args[1]
        assert call_kwargs['n_groups'] == 4
        assert call_kwargs['show_risk_table'] is True
        assert call_kwargs['time_limit'] == 100.0

    def test_no_output_calls_fig_show(self):
        fig = MagicMock()
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_missing.return_value = fig

        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='missing', list=False, load_learner='x.learner',
                models=None, output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        fig.show.assert_called_once()

    def test_plot_method_value_error_returns_one(self, capsys):
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_survival.side_effect = ValueError("bad value")

        with patch('mlsurv.cli._plot._load_learner_if_requested', return_value=learner):
            args = Namespace(
                type='survival', list=False, load_learner='x.learner',
                models=None, output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 1
        assert 'bad value' in capsys.readouterr().err

    def test_each_type_dispatches_correctly(self, tmp_path):
        for plot_type, spec in _PLOT_REGISTRY.items():
            fig = MagicMock()
            fig.to_json.return_value = '{}'
            learner = MagicMock()
            mr = MagicMock()
            mr.evaluation = MagicMock()
            mr.optuna_study = MagicMock()  # satisfy tune prereq
            learner.models = {'coxph': mr}
            learner.benchmark_result = MagicMock()  # satisfy benchmark prereq
            # satisfy validate prereq (needs 2+ cohorts)
            learner.validation_results = {'A': MagicMock(), 'B': MagicMock()}
            getattr(learner, spec.method).return_value = fig

            extra = {}
            if plot_type == 'interaction-pdp':
                extra['features'] = ['age', 'grade']

            with patch('mlsurv.cli._plot._load_learner_if_requested',
                       return_value=learner):
                args = Namespace(
                    type=plot_type, list=False, load_learner='x.learner',
                    models=None, output=str(tmp_path / f'{plot_type}.html'),
                    format='html', no_open=True, quiet=False, verbose=False,
                    return_data=False,
                    **{**_FLAG_DEFAULTS, **extra},
                )
                rc = cmd_plot(args)
            assert rc == 0, f"Plot type '{plot_type}' returned rc={rc}"
            getattr(learner, spec.method).assert_called_once()

    # --- Phase 2 kwarg forwarding tests ---

    def _run_plot(self, plot_type, tmp_path, extra_args=None, n_models=1,
                  has_eval=True, has_tune=False, has_benchmark=False):
        """Helper: run cmd_plot with mocked learner, return (rc, call_kwargs)."""
        fig = MagicMock()
        fig.to_json.return_value = '{}'
        learner = MagicMock()
        mr = MagicMock()
        mr.evaluation = MagicMock() if has_eval else None
        mr.optuna_study = MagicMock() if has_tune else None
        learner.models = {f'm{i}': mr for i in range(n_models)}
        learner.benchmark_result = MagicMock() if has_benchmark else None
        spec = _PLOT_REGISTRY[plot_type]
        getattr(learner, spec.method).return_value = fig

        args_dict = dict(
            type=plot_type, list=False, load_learner='x.learner',
            models=None, output=str(tmp_path / f'{plot_type}.html'),
            format='html', no_open=True, quiet=False, verbose=False,
            return_data=False,
            **_FLAG_DEFAULTS,
        )
        if extra_args:
            args_dict.update(extra_args)
        args = Namespace(**args_dict)

        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            rc = cmd_plot(args)

        call_kwargs = getattr(learner, spec.method).call_args
        return rc, call_kwargs[1] if call_kwargs else {}

    # Per-flag CLI→plot-method kwarg forwarding. Each row passes one (or the
    # minimal supporting set of) extra arg(s) and asserts the resulting plot
    # call kwarg — covering passthrough, rename (n_features→n), and transform
    # (references True→'all') cases. Multi-key / extra-default / negative cases
    # are kept as their own tests below.
    @pytest.mark.parametrize("plot_type, extra_args, check_key, expected", [
        ("diagnostics", {"kind": "schoenfeld"}, "kind", "schoenfeld"),
        ("diagnostics", {"features": ["age", "grade"]}, "features", ["age", "grade"]),
        ("hazard-ratios", {"kind": "multivariate"}, "kind", "multivariate"),
        ("coefficients", {"n_features": 10}, "n_features", 10),
        ("feature-correlation", {"threshold": 0.5}, "threshold", 0.5),
        ("feature-distributions", {"features": ["f0", "f1"]}, "features", ["f0", "f1"]),
        ("metric-time", {"metrics": ["harrell", "uno"]}, "metric", ["harrell", "uno"]),
        ("metric-time", {"references": True}, "references", "all"),  # True -> 'all'
        ("interactions", {"n_features": 15}, "n", 15),               # n_features -> n
        ("interaction-pdp", {"features": ["a", "b"], "max_samples": 50},
         "max_samples", 50),
        ("calibration-overlay", {"eval_times": [12.0, 24.0]},
         "eval_times", [12.0, 24.0]),
        ("metric-heatmap", {"metrics": ["harrell", "uno"]}, "metrics", ["harrell", "uno"]),
        ("hazard-ratios-comparison", {"n_features": 10}, "n_features", 10),
        ("coefficient-comparison", {"n_features": 15}, "n_features", 15),
    ])
    def test_kwarg_forwarded(self, tmp_path, plot_type, extra_args, check_key, expected):
        rc, kw = self._run_plot(plot_type, tmp_path, extra_args=extra_args)
        assert rc == 0
        assert kw[check_key] == expected

    def test_shap_kind_forwarded(self, tmp_path):
        rc, kw = self._run_plot('shap', tmp_path,
                                extra_args={'kind': 'beeswarm'})
        assert rc == 0
        assert kw['kind'] == 'beeswarm'
        assert kw['n'] == 20  # default n_features

    def test_interaction_pdp_features_forwarded(self, tmp_path):
        rc, kw = self._run_plot('interaction-pdp', tmp_path,
                                extra_args={'features': ['age', 'grade']})
        assert rc == 0
        assert kw['feature_1'] == 'age'
        assert kw['feature_2'] == 'grade'

    def test_diagnostics_defaults_no_kwargs(self, tmp_path):
        """Defaults should not pass kind or features."""
        rc, kw = self._run_plot('diagnostics', tmp_path)
        assert rc == 0
        assert 'kind' not in kw
        assert 'features' not in kw

    # --- model_param dispatch tests ---

    def test_model_param_none_rejects_model_flag(self, capsys):
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='hazard-ratios-comparison', list=False,
                load_learner='x.learner',
                models=['coxph'], output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                return_data=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 1
        assert 'not valid' in capsys.readouterr().err.lower()

    def test_model_param_singular_rejects_multiple(self, capsys):
        learner = MagicMock()
        mr = MagicMock()
        mr.evaluation = MagicMock()
        learner.models = {'coxph': mr, 'rsf': mr}
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='interactions', list=False, load_learner='x.learner',
                models=['coxph', 'rsf'], output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                return_data=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 1
        assert 'at most one' in capsys.readouterr().err.lower()

    def test_model_param_singular_passes_one(self, tmp_path):
        fig = MagicMock()
        learner = MagicMock()
        mr = MagicMock()
        mr.evaluation = MagicMock()
        learner.models = {'coxph': mr}
        learner.plot_interactions.return_value = fig
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='interactions', list=False, load_learner='x.learner',
                models=['coxph'], output=str(tmp_path / 'out.html'),
                format='html', no_open=True, quiet=False, verbose=False,
                return_data=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        kw = learner.plot_interactions.call_args[1]
        assert kw['model'] == 'coxph'
        assert 'models' not in kw

    # --- builder validation tests ---

    def test_interaction_pdp_requires_two_features(self, capsys):
        learner = MagicMock()
        mr = MagicMock()
        mr.evaluation = MagicMock()
        learner.models = {'coxph': mr}
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='interaction-pdp', list=False, load_learner='x.learner',
                models=None, output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                return_data=False,
                **{**_FLAG_DEFAULTS, 'features': ['age']},
            )
            rc = cmd_plot(args)
        assert rc == 1
        assert 'exactly 2 features' in capsys.readouterr().err.lower()

    def test_interaction_pdp_no_features_errors(self, capsys):
        learner = MagicMock()
        mr = MagicMock()
        mr.evaluation = MagicMock()
        learner.models = {'coxph': mr}
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='interaction-pdp', list=False, load_learner='x.learner',
                models=None, output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                return_data=False,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 1
        assert 'exactly 2 features' in capsys.readouterr().err.lower()

    # --- return-data tests ---

    def test_return_data_writes_csv(self, tmp_path):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_missing.return_value = df
        output_path = str(tmp_path / 'data.csv')
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='missing', list=False, load_learner='x.learner',
                models=None, output=output_path, format='html',
                no_open=True, quiet=False, verbose=False,
                return_data=True,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        assert Path(output_path).exists()

    def test_return_data_to_stdout(self, capsys):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_missing.return_value = df
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='missing', list=False, load_learner='x.learner',
                models=None, output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                return_data=True,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        captured = capsys.readouterr().out
        assert 'a,b' in captured

    def test_return_data_dict_writes_json(self, tmp_path):
        result_dict = {'key': pd.DataFrame({'a': [1]})}
        learner = MagicMock()
        mr = MagicMock()
        mr.optuna_study = MagicMock()
        learner.models = {'rsf': mr}
        learner.plot_optuna.return_value = result_dict
        output_path = str(tmp_path / 'data.json')
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='optuna', list=False, load_learner='x.learner',
                models=None, output=output_path, format='html',
                no_open=True, quiet=False, verbose=False,
                return_data=True,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        assert Path(output_path).exists()
        import json
        data = json.loads(Path(output_path).read_text(encoding='utf-8'))
        assert 'key' in data

    def test_return_data_dict_to_stdout(self, capsys):
        result_dict = {'metrics': pd.DataFrame({'m': [0.8]})}
        learner = MagicMock()
        mr = MagicMock()
        mr.optuna_study = MagicMock()
        learner.models = {'rsf': mr}
        learner.plot_optuna.return_value = result_dict
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='optuna', list=False, load_learner='x.learner',
                models=None, output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                return_data=True,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 0
        captured = capsys.readouterr().out
        assert 'metrics' in captured

    def test_return_data_unexpected_type_returns_one(self, capsys):
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_missing.return_value = "unexpected string"
        with patch('mlsurv.cli._plot._load_learner_if_requested',
                   return_value=learner):
            args = Namespace(
                type='missing', list=False, load_learner='x.learner',
                models=None, output=None, format='html',
                no_open=True, quiet=False, verbose=False,
                return_data=True,
                **_FLAG_DEFAULTS,
            )
            rc = cmd_plot(args)
        assert rc == 1
        assert 'unexpected' in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Integration tests (real training)
# ---------------------------------------------------------------------------

def _make_small_data(n=50, n_features=5, seed=42):
    """Generate small synthetic survival data inline (no fixture dependency)."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f'f{i}' for i in range(n_features)],
    )
    time = rng.exponential(10, n)
    event = rng.binomial(1, 0.6, n).astype(bool)
    y = np.array(
        [(e, t) for e, t in zip(event, time)],
        dtype=[('event', bool), ('time', float)],
    )
    return X, y


@pytest.fixture(scope="module")
def saved_learner_path(tmp_path_factory):
    """Train, evaluate, and save a learner for integration tests."""
    from mlsurv import SurvivalLearner

    X, y = _make_small_data()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=2)
    learner.run(['coxph', 'coxnet'])
    learner.evaluate(n_importance_repeats=1)

    path = tmp_path_factory.mktemp("learner") / "test.learner"
    learner.save_learner(str(path))
    return str(path)


@pytest.mark.integration
class TestCmdPlotIntegration:
    """Integration tests with a real trained+evaluated learner."""

    def test_plot_survival_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'survival.html')
        rc = main([
            'plot', 'survival',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_importance_png(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'importance.png')
        rc = main([
            'plot', 'importance',
            '--load-learner', saved_learner_path,
            '--output', output, '-f', 'png', '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_missing_json(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'missing.json')
        rc = main([
            'plot', 'missing',
            '--load-learner', saved_learner_path,
            '--output', output, '-f', 'json', '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_metric_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'metric.html')
        rc = main([
            'plot', 'metric',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()

    def test_plot_calibration_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'calibration.html')
        rc = main([
            'plot', 'calibration',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()

    def test_plot_roc_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'roc.html')
        rc = main([
            'plot', 'roc',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()

    # --- Phase 2 integration tests ---

    def test_plot_diagnostics_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'diagnostics.html')
        rc = main([
            'plot', 'diagnostics',
            '--load-learner', saved_learner_path,
            '--kind', 'schoenfeld',
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_coefficients_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'coefficients.html')
        rc = main([
            'plot', 'coefficients',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_feature_correlation_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'feature_corr.html')
        rc = main([
            'plot', 'feature-correlation',
            '--load-learner', saved_learner_path,
            '--threshold', '0.5',
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_metric_time_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'metric_time.html')
        rc = main([
            'plot', 'metric-time',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    # --- Phase 3 integration tests ---

    def test_plot_calibration_overlay_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'calibration_overlay.html')
        rc = main([
            'plot', 'calibration-overlay',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_metric_heatmap_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'metric_heatmap.html')
        rc = main([
            'plot', 'metric-heatmap',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_coefficient_comparison_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'coeff_comparison.html')
        rc = main([
            'plot', 'coefficient-comparison',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_hazard_ratios_comparison_html(self, saved_learner_path, tmp_path):
        output = str(tmp_path / 'hr_comparison.html')
        rc = main([
            'plot', 'hazard-ratios-comparison',
            '--load-learner', saved_learner_path,
            '--output', output, '--no-open',
        ])
        assert rc == 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0


# ---------------------------------------------------------------------------
# Item 3: Nonlinear model description hints
# ---------------------------------------------------------------------------

class TestNonlinearDescriptions:

    def test_nonlinear_description_hint(self):
        for name in ('interactions', 'interaction-pdp'):
            desc = _PLOT_REGISTRY[name].description
            assert 'nonlinear' in desc.lower(), (
                f"Type '{name}' description should mention nonlinear requirement"
            )

    def test_list_shows_nonlinear_hint(self, capsys):
        rc = main(['plot', '--list'])
        assert rc == 0
        output = capsys.readouterr().out
        assert 'nonlinear' in output.lower()


# ---------------------------------------------------------------------------
# Item 2: Optuna serialization fallback
# ---------------------------------------------------------------------------

class TestOptunaSerializationFallback:

    def test_check_prerequisite_accepts_extracted_history(self):
        learner = MagicMock()
        mr = MagicMock()
        mr.optuna_study = None
        mr._optuna_tuning_history = [{'trial': 0, 'value': 0.75}]
        learner.models = {'rsf': mr}
        spec = _PLOT_REGISTRY['optuna']
        assert _check_prerequisite(learner, spec, 'optuna') is None

    def test_check_prerequisite_fails_without_study_or_history(self):
        learner = MagicMock()
        mr = MagicMock()
        mr.optuna_study = None
        mr._optuna_tuning_history = None
        learner.models = {'rsf': mr}
        spec = _PLOT_REGISTRY['optuna']
        err = _check_prerequisite(learner, spec, 'optuna')
        assert err is not None
        assert 'tune' in err.lower()


# ---------------------------------------------------------------------------
# Item 1: Inline plot generation
# ---------------------------------------------------------------------------

class TestInlinePlotParser:

    def test_evaluate_parser_accepts_plot_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'data.csv', '--model', 'coxph',
            '--plot', 'metric', 'importance',
        ])
        assert args.plot == ['metric', 'importance']

    def test_evaluate_parser_no_plot_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'evaluate', 'data.csv', '--model', 'coxph',
        ])
        assert args.plot is None

    def test_bootstrap_parser_accepts_plot_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            'bootstrap', 'data.csv', '--model', 'coxph',
            '--plot', 'metric',
        ])
        assert args.plot == ['metric']


class TestGenerateInlinePlots:

    def test_empty_list_does_nothing(self, tmp_path):
        learner = MagicMock()
        n_failed = _generate_inline_plots(learner, [], tmp_path, quiet=True)
        assert n_failed == 0

    def test_unknown_type_warns_and_skips(self, tmp_path, capsys):
        learner = MagicMock()
        n_failed = _generate_inline_plots(
            learner, ['nonexistent'], tmp_path, quiet=False,
        )
        assert n_failed == 1
        assert 'nonexistent' in capsys.readouterr().err

    def test_prerequisite_failure_warns_and_skips(self, tmp_path, capsys):
        learner = MagicMock()
        learner.models = {}
        n_failed = _generate_inline_plots(
            learner, ['survival'], tmp_path, quiet=False,
        )
        assert n_failed == 1
        assert 'skipping' in capsys.readouterr().err.lower()

    def test_successful_plot_saves_html(self, tmp_path):
        fig = MagicMock()
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_survival.return_value = fig
        n_failed = _generate_inline_plots(
            learner, ['survival'], tmp_path, quiet=False,
        )
        assert n_failed == 0
        fig.write_html.assert_called_once()

    def test_multiple_types_independent_failures(self, tmp_path, capsys):
        fig = MagicMock()
        learner = MagicMock()
        mr = MagicMock()
        mr.evaluation = MagicMock()
        mr.optuna_study = None
        mr._optuna_tuning_history = None
        learner.models = {'coxph': mr}
        learner.benchmark_result = None
        learner.plot_importance.return_value = fig
        # importance should succeed, optuna should fail (no tune data)
        n_failed = _generate_inline_plots(
            learner, ['importance', 'optuna'], tmp_path, quiet=False,
        )
        assert n_failed == 1
        learner.plot_importance.assert_called_once()

    def test_output_dir_none_uses_cwd(self):
        fig = MagicMock()
        learner = MagicMock()
        learner.models = {'coxph': MagicMock()}
        learner.plot_survival.return_value = fig
        with patch('mlsurv.cli._plot._write_figure') as mock_write:
            _generate_inline_plots(learner, ['survival'], None, quiet=True)
        call_path = mock_write.call_args[0][1]
        assert 'survival.html' in call_path
