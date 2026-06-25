"""Tests for mlsurv.results.printing console output functions.

Covers formatting helpers, single-model and multi-model table printers,
bootstrap tables, evaluation tables, and best-params display functions.
"""

import warnings

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from mlsurv._warnings import InternalWarning
from mlsurv.results.printing import (
    _BEST_MARKER,
    _is_valid_numeric,
    _filter_by_model_type,
    _format_mean_std,
    _format_ci,
    _format_value,
    _format_tau,
    _find_best_cv_models,
    _format_cv_fold_label,
    _mark_best_in_cells,
    _format_param_value,
    _group_best_params,
    _format_params_inline,
    print_cv_metrics_table,
    print_multi_model_cv_table,
    print_bootstrap_table,
    print_multi_model_bootstrap_table,
    print_evaluation_table,
    print_multi_model_evaluation_table,
    print_best_params,
    print_multi_model_best_params,
    print_ph_test,
    print_rmst_test,
    print_rmst_pairwise,
    print_calibration_test,
    _format_delta,
    _filter_baseline,
    _DELTA_NOTE,
)


class TestValidNumericHelper:
    """Tests for _is_valid_numeric() null/NaN detection.

    This helper guards against NaN/None values propagating into
    formatted output strings.
    """

    @pytest.mark.parametrize("value, expected", [
        (0.723, True),                 # valid float
        (42, True),                    # valid int
        (None, False),                 # None
        (float('nan'), False),         # NaN
        (pd.NA, False),                # pandas NA
        (float('inf'), False),         # +inf
        (float('-inf'), False),        # -inf
        (np.inf, False),               # numpy +inf
        (-np.inf, False),              # numpy -inf
        ('oops', False),               # non-numeric string (object-dtype cell)
        ('3.14', False),               # numeric-looking string is still not a number
    ])
    def test_is_valid_numeric(self, value, expected):
        assert _is_valid_numeric(value) is expected


class TestFormatTau:
    """Tests for _format_tau() RMST horizon formatting."""

    @pytest.mark.parametrize("tau, expected", [
        (5.0, "5"),       # whole float -> integer display
        (10, "10"),       # int -> integer display
        (5.5, "5.50"),    # fractional -> precision display
    ])
    def test_valid_tau(self, tau, expected):
        assert _format_tau(tau, precision=2) == expected

    @pytest.mark.parametrize("tau", [
        float('nan'),     # int(nan) raises ValueError
        float('inf'),     # int(inf) raises OverflowError
        -np.inf,
        None,             # int(None) raises TypeError
    ])
    def test_degenerate_tau_returns_na(self, tau):
        """Non-finite / missing tau should degrade to 'N/A', not raise."""
        assert _format_tau(tau, precision=2) == "N/A"


class TestFormatHelpers:
    """Tests for string formatting helpers."""

    def test_format_mean_std(self):
        assert _format_mean_std(0.723, 0.032) == "0.723 (0.032)"

    def test_format_mean_std_na(self):
        assert _format_mean_std(None, 0.032) == "N/A"
        assert _format_mean_std(0.723, None) == "N/A"

    def test_format_ci(self):
        assert _format_ci(0.665, 0.781) == "[0.665, 0.781]"

    def test_format_ci_na(self):
        assert _format_ci(None, 0.781) == "[N/A]"

    def test_format_value(self):
        assert _format_value(0.723) == "0.723"

    def test_format_value_na(self):
        assert _format_value(None) == "N/A"

    def test_format_value_precision(self):
        assert _format_value(0.72345, precision=2) == "0.72"

    @pytest.mark.parametrize("n_scores,cv_repeats,expected", [
        (5, 1, "5-fold"),
        (5, 0, "5-fold"),            # cv_repeats < 1 falls back to plain label
        (15, 3, "5-fold x3 repeats"),
        (10, 2, "5-fold x2 repeats"),
        (7, 3, "7-fold"),            # non-divisible -> plain total, no wrong base
    ])
    def test_format_cv_fold_label(self, n_scores, cv_repeats, expected):
        assert _format_cv_fold_label(n_scores, cv_repeats) == expected


class TestPrintCvMetricsTable:
    """Tests for print_cv_metrics_table() single-model output."""

    def test_basic_output(self, capsys):
        """Should print formatted table with metric names."""
        cv_metrics = {
            'harrell': {'mean': 0.723, 'std': 0.032, 'scores': np.array([0.7, 0.75])},
        }
        print_cv_metrics_table(cv_metrics, n_folds=5)
        output = capsys.readouterr().out
        assert "Cross-Validation Results (5-fold)" in output
        assert "Harrell" in output
        assert "0.723" in output

    def test_with_model_name(self, capsys):
        """Model name should appear in header."""
        cv_metrics = {'harrell': {'mean': 0.7, 'std': 0.03, 'scores': []}}
        print_cv_metrics_table(cv_metrics, n_folds=5, model_name='RSF')
        output = capsys.readouterr().out
        assert "RSF:" in output

    def test_lower_is_better_asterisk(self, capsys):
        """IBS should get asterisk marking."""
        cv_metrics = {
            'integrated_brier': {'mean': 0.185, 'std': 0.015, 'scores': []},
        }
        print_cv_metrics_table(cv_metrics, n_folds=5)
        output = capsys.readouterr().out
        assert "*" in output
        assert "Lower is better" in output

    def test_repeated_cv_label(self, capsys):
        """Repeated CV should read base folds and repeat count, not the total."""
        cv_metrics = {'harrell': {'mean': 0.7, 'std': 0.03, 'scores': []}}
        # 15 total scores = 5 folds x 3 repeats.
        print_cv_metrics_table(cv_metrics, n_folds=15, cv_repeats=3)
        output = capsys.readouterr().out
        assert "5-fold x3 repeats" in output
        assert "15-fold" not in output


class TestPrintMultiModelCvTable:
    """Tests for print_multi_model_cv_table() comparison output."""

    @pytest.fixture
    def model_results(self):
        """Create mock model results for table printing."""
        results = {}
        for name, mean, std in [('rsf', 0.723, 0.032), ('coxph', 0.698, 0.045)]:
            r = MagicMock(spec=['cv_metrics', 'cv_metric_mean', 'cv_metric_std'])
            r.cv_metrics = {
                'harrell': {'mean': mean, 'std': std, 'scores': []},
            }
            r.cv_metric_mean = mean
            r.cv_metric_std = std
            results[name] = r
        return results

    def test_basic_multi_model(self, model_results, capsys):
        """Should print table with all models."""
        print_multi_model_cv_table(model_results, n_folds=5)
        output = capsys.readouterr().out
        assert 'rsf' in output
        assert 'coxph' in output
        assert '0.723' in output

    def test_sorted_by_performance(self, model_results, capsys):
        """Models should appear sorted by cv_metric_mean descending."""
        print_multi_model_cv_table(model_results, n_folds=5)
        output = capsys.readouterr().out
        rsf_pos = output.find('rsf')
        coxph_pos = output.find('coxph')
        assert rsf_pos < coxph_pos  # RSF (0.723) before CoxPH (0.698)

    def test_tuning_status_in_header(self, model_results, capsys):
        """An explicit tuning_status override should appear in the header."""
        print_multi_model_cv_table(model_results, n_folds=5, tuning_status="with tuning")
        output = capsys.readouterr().out
        assert "with tuning" in output

    @staticmethod
    def _tuned_results(*tuned_flags):
        """Build {name: result} with the given per-model is_tuned flags."""
        from types import SimpleNamespace
        out = {}
        for i, tuned in enumerate(tuned_flags):
            mean = 0.72 - 0.01 * i
            out[f"m{i}"] = SimpleNamespace(
                cv_metrics={'harrell': {'mean': mean, 'std': 0.03, 'scores': []}},
                cv_metric_mean=mean, cv_metric_std=0.03, is_tuned=tuned,
            )
        return out

    def test_any_tuned_derives_with_tuning(self, capsys):
        """When any model was tuned, the header reads 'with tuning'."""
        print_multi_model_cv_table(self._tuned_results(True, True), n_folds=5)
        output = capsys.readouterr().out
        assert "with tuning" in output

    def test_none_tuned_no_annotation(self, capsys):
        """When no model was tuned, the header carries no tuning annotation."""
        print_multi_model_cv_table(self._tuned_results(False, False), n_folds=5)
        output = capsys.readouterr().out
        assert "with tuning" not in output
        assert "baseline" not in output  # the old wording is gone

    def test_mixed_tuning_still_with_tuning(self, capsys):
        """A model that wasn't tuned (e.g. coxph) just sits in a 'with tuning'
        table -- no per-row tag (its skip is explained by a warning elsewhere)."""
        # m0 untuned, m1 tuned.
        print_multi_model_cv_table(self._tuned_results(False, True), n_folds=5)
        output = capsys.readouterr().out
        assert "with tuning" in output
        assert "(tuned)" not in output
        assert "(untuned)" not in output

    def test_cells_legend_in_header(self, model_results, capsys):
        """Header should label the parenthetical as the per-split Std."""
        print_multi_model_cv_table(model_results, n_folds=5)
        output = capsys.readouterr().out
        assert "mean (Std)" in output
        assert "repeats" not in output  # default cv_repeats=1 -> plain fold label

    def test_repeated_cv_label(self, model_results, capsys):
        """Repeated CV should read base folds and repeat count, not the total."""
        # 15 total scores = 5 folds x 3 repeats.
        print_multi_model_cv_table(model_results, n_folds=15, cv_repeats=3)
        output = capsys.readouterr().out
        assert "5-fold x3 repeats" in output
        assert "15-fold" not in output
        assert "mean (Std)" in output


class TestRepeatedCvLabelWiring:
    """Check that the learner threads cv_repeats into the CV table header."""

    def test_print_cv_summary_threads_cv_repeats(self, capsys):
        """Single-model path (_print_cv_summary, used by train()/tune()) should
        read self._state.cv_repeats and label the header accordingly. Calling the
        printer directly would not catch a dropped argument at this call site."""
        from types import SimpleNamespace
        from mlsurv.learner._core import SurvivalLearner
        result = SimpleNamespace(
            name='coxph',
            best_params_=None,
            cv_metric_scores=list(range(15)),  # 15 = 5 folds x 3 repeats
            cv_metrics={'harrell': {'mean': 0.7, 'std': 0.03, 'scores': []}},
        )
        learner = SimpleNamespace(_state=SimpleNamespace(cv_repeats=3))
        SurvivalLearner._print_cv_summary(learner, result)
        output = capsys.readouterr().out
        assert "5-fold x3 repeats" in output

    @pytest.mark.integration
    def test_run_header_reports_repeats(self, small_data, capsys):
        """run() with cv_repeats>1 should label the table '2-fold x2 repeats'."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=2, cv_repeats=2, random_state=42)
        capsys.readouterr()  # discard setup output
        learner.run(['coxph', 'coxnet'], tune=False)
        output = capsys.readouterr().out
        assert "2-fold x2 repeats" in output
        assert "mean (Std)" in output


class TestPrintBootstrapTable:
    """Tests for print_bootstrap_table() bootstrap CI output."""

    @pytest.fixture
    def bootstrap_summary(self):
        return pd.DataFrame({
            'model_type': ['rsf', 'rsf'],
            'metric': ['harrell', 'integrated_brier'],
            'point': [0.725, 0.183],
            'mean': [0.723, 0.185],
            'std': [0.032, 0.015],
            'ci_lower': [0.665, 0.158],
            'ci_upper': [0.781, 0.212],
        })

    def test_basic_bootstrap_output(self, bootstrap_summary, capsys):
        """Should print formatted bootstrap results using point estimates."""
        print_bootstrap_table(bootstrap_summary, n_bootstrap=200)
        output = capsys.readouterr().out
        assert "Bootstrap Results" in output
        assert "200 iterations" in output
        assert "0.725" in output  # point estimate, not bootstrap mean

    def test_model_name_filtering(self, bootstrap_summary, capsys):
        """When model_name provided, should filter to that model."""
        print_bootstrap_table(bootstrap_summary, n_bootstrap=200, model_name='rsf')
        output = capsys.readouterr().out
        assert "rsf:" in output

    def test_point_estimates_override(self, bootstrap_summary, capsys):
        """Point estimates should override bootstrap means."""
        point_estimates = {'harrell': 0.750}
        print_bootstrap_table(
            bootstrap_summary, n_bootstrap=200,
            model_name='rsf', point_estimates=point_estimates
        )
        output = capsys.readouterr().out
        assert "0.750" in output


class TestPrintEvaluationTable:
    """Tests for print_evaluation_table() holdout results."""

    def test_basic_evaluation(self, capsys):
        """Should print test set evaluation results."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': [0.723],
        })
        print_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert "Test Set Evaluation" in output
        assert "0.723" in output

    def test_with_model_name(self, capsys):
        """Model name should appear in header and filter rows."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': [0.723],
        })
        print_evaluation_table(concordance, model_name='rsf', display_name='RSF')
        output = capsys.readouterr().out
        assert "RSF:" in output

    def test_multiple_metrics(self, capsys):
        """Should display all available metrics."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': [0.723],
            'integrated_brier_test': [0.185],
        })
        print_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert "Harrell" in output
        assert "Integrated Brier Score" in output

    def test_model_name_filters_multimodel(self, capsys):
        """When model_name is given, only that model's row should be used."""
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.800, 0.600],
        })
        print_evaluation_table(concordance, model_name='rsf')
        output = capsys.readouterr().out
        assert "0.800" in output
        assert "0.600" not in output

    def test_model_name_no_match_falls_back(self, capsys):
        """When model_name matches no rows, should fall back with warning."""
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.800, 0.600],
        })
        with pytest.warns(InternalWarning, match="model_name='gbsa' not found"):
            print_evaluation_table(concordance, model_name='gbsa')
        output = capsys.readouterr().out
        assert "Test Set Evaluation" in output
        assert "0.800" in output

    def test_model_name_no_match_warning_lists_available(self):
        """Warning message should list available model names."""
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.800, 0.600],
        })
        with pytest.warns(InternalWarning, match=r"Available models: \['coxph', 'rsf'\]"):
            print_evaluation_table(concordance, model_name='gbsa')

    def test_model_name_match_no_warning(self):
        """When model_name matches, no warning should be emitted."""
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.800, 0.600],
        })
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            print_evaluation_table(concordance, model_name='rsf')

    def test_nan_cell_skipped(self, capsys):
        """A NaN metric cell should be skipped, not crash or print 'nan'."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': [0.723],
            'integrated_brier_test': [np.nan],
        })
        print_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert "0.723" in output
        assert "nan" not in output.lower()

    def test_object_dtype_cell_skipped(self, capsys):
        """An object-dtype (non-numeric) cell should be skipped, not raise TypeError."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': ['oops'],          # stray string -> object dtype
            'integrated_brier_test': [0.185],
        })
        # np.isnan('oops') would raise TypeError; the _is_valid_numeric guard skips it.
        print_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert "Test Set Evaluation" in output
        assert "0.185" in output
        assert "oops" not in output

    def test_nullable_pd_na_cell_skipped(self, capsys):
        """A pd.NA cell in a nullable Float64 column should be skipped, not crash.

        This is the realistic way pd.NA reaches the printer (nullable dtypes).
        The pd.isna() guard in _is_valid_numeric catches it before any numeric
        coercion (np.isnan(pd.NA) would otherwise raise TypeError).
        """
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': pd.array([pd.NA], dtype='Float64'),
            'integrated_brier_test': [0.185],
        })
        print_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert "Test Set Evaluation" in output
        assert "0.185" in output
        assert "<NA>" not in output       # the pd.NA cell is skipped, not rendered


class TestPrintMultiModelBootstrapTable:
    """Tests for print_multi_model_bootstrap_table() multi-model bootstrap output.

    This 128-line function handles direction-aware best-model detection,
    concordance fallback to bootstrap means, and the <best annotation.
    """

    @pytest.fixture
    def bootstrap_data(self):
        """Two models with harrell + IBS metrics in bootstrap summary and concordance."""
        summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf', 'coxph', 'coxph'],
            'metric': ['harrell', 'integrated_brier', 'harrell', 'integrated_brier'],
            'point': [0.723, 0.188, 0.698, 0.205],
            'mean': [0.720, 0.190, 0.700, 0.200],
            'std': [0.030, 0.015, 0.040, 0.020],
            'ci_lower': [0.665, 0.162, 0.625, 0.165],
            'ci_upper': [0.775, 0.218, 0.775, 0.235],
        })
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.723, 0.698],
            'integrated_brier_test': [0.185, 0.210],
        })
        return summary, concordance

    def test_basic_multi_model_output(self, bootstrap_data, capsys):
        """Should print table with header, both models, and metric values."""
        summary, concordance = bootstrap_data
        print_multi_model_bootstrap_table(summary, concordance, n_bootstrap=200)
        output = capsys.readouterr().out
        assert "Test Set Results" in output
        assert "200 bootstrap iterations" in output
        assert "rsf" in output
        assert "coxph" in output
        assert "0.723" in output

    def test_best_model_annotation(self, bootstrap_data, capsys):
        """RSF is best on both metrics (higher harrell, lower IBS) -> gets <best and arrow markers."""
        summary, concordance = bootstrap_data
        print_multi_model_bootstrap_table(summary, concordance, n_bootstrap=200)
        output = capsys.readouterr().out
        for line in output.splitlines():
            if line.strip().startswith('rsf'):
                assert '<best' in line
                assert _BEST_MARKER in line
            elif line.strip().startswith('coxph'):
                assert '<best' not in line
                assert _BEST_MARKER not in line

    def test_no_best_when_mixed_winners(self, capsys):
        """When different models win different metrics, no model gets <best but each gets arrow on its best."""
        summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf', 'coxph', 'coxph'],
            'metric': ['harrell', 'integrated_brier', 'harrell', 'integrated_brier'],
            'point': [0.723, 0.200, 0.698, 0.180],
            'mean': [0.720, 0.200, 0.700, 0.180],
            'std': [0.030, 0.020, 0.040, 0.015],
            'ci_lower': [0.665, 0.165, 0.625, 0.150],
            'ci_upper': [0.775, 0.235, 0.775, 0.210],
        })
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.723, 0.698],  # RSF wins harrell (higher)
            'integrated_brier_test': [0.200, 0.180],  # CoxPH wins IBS (lower)
        })
        print_multi_model_bootstrap_table(summary, concordance, n_bootstrap=200)
        output = capsys.readouterr().out
        assert '<best' not in output
        # Both models should have arrow marker on their winning metric
        for line in output.splitlines():
            if line.strip().startswith('rsf'):
                assert _BEST_MARKER in line  # RSF wins harrell
            elif line.strip().startswith('coxph'):
                assert _BEST_MARKER in line  # CoxPH wins IBS

    def test_model_order_respected(self, bootstrap_data, capsys):
        """model_order should control display order (coxph first)."""
        summary, concordance = bootstrap_data
        print_multi_model_bootstrap_table(
            summary, concordance, n_bootstrap=200,
            model_order=['coxph', 'rsf'],
        )
        output = capsys.readouterr().out
        # Skip header occurrences - find data rows
        lines = [l for l in output.splitlines() if l.strip().startswith(('coxph', 'rsf'))]
        assert lines[0].strip().startswith('coxph')
        assert lines[1].strip().startswith('rsf')

    def test_fallback_to_point_estimates(self, capsys):
        """When concordance lacks metric columns, should use point estimates for best-model."""
        summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf', 'coxph', 'coxph'],
            'metric': ['harrell', 'integrated_brier', 'harrell', 'integrated_brier'],
            'point': [0.723, 0.188, 0.698, 0.205],
            'mean': [0.720, 0.190, 0.700, 0.200],
            'std': [0.030, 0.015, 0.040, 0.020],
            'ci_lower': [0.665, 0.162, 0.625, 0.165],
            'ci_upper': [0.775, 0.218, 0.775, 0.235],
        })
        # Concordance has model_type but no metric columns
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
        })
        print_multi_model_bootstrap_table(summary, concordance, n_bootstrap=200)
        output = capsys.readouterr().out
        # Should still produce output without crashing
        assert "rsf" in output
        assert "coxph" in output

    def test_lower_is_better_asterisk(self, bootstrap_data, capsys):
        """IBS column should get asterisk and footer note."""
        summary, concordance = bootstrap_data
        print_multi_model_bootstrap_table(summary, concordance, n_bootstrap=200)
        output = capsys.readouterr().out
        assert "* Lower is better" in output


class TestPrintMultiModelEvaluationTable:
    """Tests for print_multi_model_evaluation_table() — single combined table
    for multi-model evaluate() output without bootstrap."""

    @pytest.fixture
    def concordance(self):
        return pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.723, 0.698],
            'integrated_brier_test': [0.185, 0.210],
        })

    def test_basic_multi_model_output(self, concordance, capsys):
        """One table containing both models and their metric values."""
        print_multi_model_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert "Test Set Evaluation" in output
        assert "rsf" in output
        assert "coxph" in output
        assert "0.723" in output
        assert "0.698" in output
        # Single header — not one per model
        assert output.count("Test Set Evaluation") == 1

    def test_no_bootstrap_ci_in_output(self, concordance, capsys):
        """Output is point estimates only — no `[lo, hi]` CI strings."""
        print_multi_model_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert "[" not in output
        assert "]" not in output

    def test_best_model_annotation(self, concordance, capsys):
        """RSF wins both metrics (higher harrell, lower IBS) -> <best + arrow."""
        print_multi_model_evaluation_table(concordance)
        output = capsys.readouterr().out
        for line in output.splitlines():
            if line.strip().startswith('rsf'):
                assert '<best' in line
                assert _BEST_MARKER in line
            elif line.strip().startswith('coxph'):
                assert '<best' not in line
                assert _BEST_MARKER not in line

    def test_no_best_when_mixed_winners(self, capsys):
        """Different winners per metric -> no <best, but each gets arrow on its win."""
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.723, 0.698],            # RSF wins
            'integrated_brier_test': [0.200, 0.180],   # CoxPH wins
        })
        print_multi_model_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert '<best' not in output
        for line in output.splitlines():
            if line.strip().startswith(('rsf', 'coxph')):
                assert _BEST_MARKER in line

    def test_model_order_respected(self, concordance, capsys):
        """model_order controls display order."""
        print_multi_model_evaluation_table(concordance, model_order=['coxph', 'rsf'])
        output = capsys.readouterr().out
        lines = [l for l in output.splitlines()
                 if l.strip().startswith(('coxph', 'rsf'))]
        assert lines[0].strip().startswith('coxph')
        assert lines[1].strip().startswith('rsf')

    def test_display_names_used(self, concordance, capsys):
        """display_names mapping replaces raw keys in row labels."""
        print_multi_model_evaluation_table(
            concordance,
            display_names={'rsf': 'RSF', 'coxph': 'CoxPH'},
        )
        output = capsys.readouterr().out
        assert 'RSF' in output
        assert 'CoxPH' in output

    def test_lower_is_better_asterisk(self, concordance, capsys):
        """IBS column should trigger the lower-is-better footer."""
        print_multi_model_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert "* Lower is better" in output

    def test_warns_on_missing_model_order_keys(self, concordance, capsys):
        """model_order entries absent from concordance trigger an InternalWarning
        and render an N/A row (no silent data loss)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            print_multi_model_evaluation_table(
                concordance, model_order=['rsf', 'coxph', 'nonexistent'],
            )
        assert len(w) == 1
        assert issubclass(w[0].category, InternalWarning)
        assert "nonexistent" in str(w[0].message)
        output = capsys.readouterr().out
        assert "nonexistent" in output
        assert "N/A" in output

    def test_direct_call_stacklevel_points_to_caller(self, concordance):
        """Default _warn_stacklevel=2 points the warning at the caller (this test),
        not at printing.py."""
        import os
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            print_multi_model_evaluation_table(
                concordance, model_order=['rsf', 'coxph', 'missing'],
            )
        assert len(w) == 1
        assert os.path.basename(w[0].filename) == "test_printing.py"

    def test_single_model_no_best_marker(self, capsys):
        """Single-model concordance must not show <best or arrow markers."""
        single = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': [0.723],
            'integrated_brier_test': [0.185],
        })
        print_multi_model_evaluation_table(single)
        output = capsys.readouterr().out
        assert '<best' not in output
        assert _BEST_MARKER not in output

    def test_empty_concordance_no_crash(self, capsys):
        """Empty concordance (no model_type column) prints header-only without crashing."""
        empty = pd.DataFrame()
        print_multi_model_evaluation_table(empty)
        output = capsys.readouterr().out
        assert "Test Set Evaluation" in output


class TestDeltaAnnotation:
    """Tests for the optional held-out-test delta annotation (``validate()``).

    Covers the shared ``_format_delta`` / ``_filter_baseline`` helpers and the
    ``delta_baseline=`` path of all four table printers: each metric cell gains
    a ``(Δ ...)`` change-from-baseline annotation, a footnote defines Δ once,
    and rows/metrics absent from the baseline (references, never-evaluated
    models) print with no Δ.
    """

    @pytest.fixture
    def baseline(self):
        """Held-out-test concordance for two models (the dev baseline)."""
        return pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.760, 0.740],
            'integrated_brier_test': [0.160, 0.170],
        })

    @pytest.fixture
    def concordance(self):
        """Validation concordance: rsf/coxph have a baseline, ``someref`` does not."""
        return pd.DataFrame({
            'model_type': ['rsf', 'coxph', 'someref'],
            'c_index_test': [0.723, 0.698, 0.600],
            'integrated_brier_test': [0.185, 0.210, 0.250],
        })

    @pytest.fixture
    def bootstrap_summary(self, concordance):
        """Bootstrap summary aligned with ``concordance`` (point == concordance)."""
        rows = []
        cols = {'harrell': 'c_index_test', 'integrated_brier': 'integrated_brier_test'}
        for _, r in concordance.iterrows():
            for metric, col in cols.items():
                pt = r[col]
                rows.append({
                    'model_type': r['model_type'], 'metric': metric,
                    'point': pt, 'mean': pt, 'std': 0.02,
                    'ci_lower': pt - 0.03, 'ci_upper': pt + 0.03,
                })
        return pd.DataFrame(rows)

    # ---- helper: _format_delta -------------------------------------------

    def test_format_delta_signed_difference(self, baseline):
        """Higher-is-better drop is negative; IBS rise is positive; sign explicit."""
        rsf_base = _filter_baseline(baseline, 'rsf')
        assert _format_delta(0.723, rsf_base, 'c_index_test', 3) == " (Δ -0.037)"
        assert _format_delta(0.185, rsf_base, 'integrated_brier_test', 3) == " (Δ +0.025)"

    def test_format_delta_no_baseline_returns_empty(self, baseline):
        """None baseline, missing metric column, and non-finite values yield ''."""
        rsf_base = _filter_baseline(baseline, 'rsf')
        assert _format_delta(0.7, None, 'c_index_test', 3) == ""
        assert _format_delta(0.7, rsf_base, 'c_index_td_test', 3) == ""  # col absent
        assert _format_delta(np.nan, rsf_base, 'c_index_test', 3) == ""  # non-finite current
        # Non-finite *baseline* cell (e.g. a metric that failed to compute on
        # the held-out test split) also yields no delta.
        nan_base = _filter_baseline(
            pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [np.nan]}), 'rsf',
        )
        assert _format_delta(0.7, nan_base, 'c_index_test', 3) == ""

    def test_filter_baseline(self, baseline):
        """Slices to one model's rows; None/absent model_type column -> None."""
        assert _filter_baseline(None, 'rsf') is None
        assert _filter_baseline(pd.DataFrame({'x': [1]}), 'rsf') is None
        sliced = _filter_baseline(baseline, 'coxph')
        assert list(sliced['model_type']) == ['coxph']
        # A model absent from the baseline yields an empty (not None) slice.
        assert len(_filter_baseline(baseline, 'ghost')) == 0

    # ---- multi-model evaluation table ------------------------------------

    def test_multi_eval_shows_delta_and_note(self, concordance, baseline, capsys):
        print_multi_model_evaluation_table(concordance, delta_baseline=baseline)
        output = capsys.readouterr().out
        assert "(Δ -0.037)" in output   # rsf harrell 0.723 - 0.760
        assert "(Δ +0.025)" in output   # rsf IBS 0.185 - 0.160
        assert _DELTA_NOTE in output

    def test_multi_eval_reference_row_has_no_delta(self, concordance, baseline, capsys):
        """``someref`` is absent from the baseline -> its row carries no Δ."""
        print_multi_model_evaluation_table(concordance, delta_baseline=baseline)
        output = capsys.readouterr().out
        ref_line = next(l for l in output.splitlines() if l.strip().startswith('someref'))
        assert "Δ" not in ref_line

    def test_multi_eval_without_baseline_unchanged(self, concordance, capsys):
        """delta_baseline=None reproduces the no-delta output (regression guard)."""
        print_multi_model_evaluation_table(concordance)
        output = capsys.readouterr().out
        assert "Δ" not in output
        assert _DELTA_NOTE not in output

    # ---- multi-model bootstrap table -------------------------------------

    def test_multi_bootstrap_shows_delta_with_ci(self, bootstrap_summary, concordance,
                                                 baseline, capsys):
        print_multi_model_bootstrap_table(
            bootstrap_summary, concordance, n_bootstrap=200, delta_baseline=baseline,
        )
        output = capsys.readouterr().out
        assert "(Δ -0.037)" in output       # rsf harrell delta
        assert "[" in output and "]" in output  # CI still present
        assert _DELTA_NOTE in output

    def test_multi_bootstrap_without_baseline_unchanged(self, bootstrap_summary,
                                                        concordance, capsys):
        print_multi_model_bootstrap_table(bootstrap_summary, concordance, n_bootstrap=200)
        output = capsys.readouterr().out
        assert "Δ" not in output

    # ---- single-model tables ---------------------------------------------

    def test_single_eval_shows_delta_and_note(self, concordance, baseline, capsys):
        print_evaluation_table(
            concordance[concordance.model_type == 'coxph'],
            model_name='coxph', delta_baseline=baseline,
        )
        output = capsys.readouterr().out
        assert "(Δ -0.042)" in output   # coxph harrell 0.698 - 0.740
        assert _DELTA_NOTE in output

    def test_single_bootstrap_shows_delta_and_note(self, bootstrap_summary, baseline,
                                                   capsys):
        print_bootstrap_table(
            bootstrap_summary, n_bootstrap=200, model_name='coxph',
            delta_baseline=baseline,
        )
        output = capsys.readouterr().out
        assert "(Δ -0.042)" in output   # coxph harrell delta
        assert _DELTA_NOTE in output

    def test_single_tables_without_baseline_unchanged(self, concordance, bootstrap_summary,
                                                      capsys):
        """Single-model printers carry no Δ when delta_baseline is None."""
        print_evaluation_table(concordance[concordance.model_type == 'coxph'],
                               model_name='coxph')
        print_bootstrap_table(bootstrap_summary, n_bootstrap=200, model_name='coxph')
        output = capsys.readouterr().out
        assert "Δ" not in output

    # ---- dispatch layer (print_eval_metrics threading) -------------------

    def _eval_result(self, concordance, bootstrap_summary=None):
        from mlsurv.results.containers import EvaluationResult, BootstrapResult
        bootstrap = None
        if bootstrap_summary is not None:
            models = list(concordance['model_type'])
            iterations = pd.DataFrame({
                'iteration': list(range(5)) * len(models),
                'model_type': [m for m in models for _ in range(5)],
            })
            bootstrap = BootstrapResult(
                summary=bootstrap_summary, iterations=iterations, ci_level=0.95,
            )
        return EvaluationResult(
            concordance=concordance, test_set=pd.DataFrame(), auc_t=pd.DataFrame(),
            importance=pd.DataFrame(), importance_iterations=pd.DataFrame(),
            model_order=list(concordance['model_type']), bootstrap=bootstrap,
        )

    def test_dispatch_multi_threads_delta(self, concordance, baseline, capsys):
        """print_eval_metrics forwards delta_baseline through the multi-model arm."""
        from mlsurv.results._containers_printing import print_eval_metrics
        print_eval_metrics(self._eval_result(concordance), delta_baseline=baseline)
        output = capsys.readouterr().out
        assert "(Δ" in output
        assert _DELTA_NOTE in output

    def test_dispatch_single_threads_delta(self, concordance, baseline, capsys):
        """print_eval_metrics forwards delta_baseline through the single-model arm."""
        from mlsurv.results._containers_printing import print_eval_metrics
        one = concordance[concordance.model_type == 'coxph']
        print_eval_metrics(self._eval_result(one), delta_baseline=baseline)
        output = capsys.readouterr().out
        assert "(Δ -0.042)" in output   # coxph harrell 0.698 - 0.740
        assert _DELTA_NOTE in output


class TestPrintEvalMetricsDispatch:
    """Tests for print_eval_metrics() routing of multi-model EvaluationResult
    through the new combined-table printer (vs. the old per-model loop)."""

    @pytest.fixture
    def multi_model_eval_result(self):
        from mlsurv.results.containers import EvaluationResult
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.72, 0.68],
            'integrated_brier_test': [0.18, 0.21],
        })
        return EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
            model_order=['rsf', 'coxph'],
        )

    def test_single_combined_table_no_bootstrap(self, multi_model_eval_result, capsys):
        """print_metrics() on multi-model no-bootstrap EvaluationResult prints
        ONE combined 'Test Set Evaluation' header, not one per model."""
        multi_model_eval_result.print_metrics()
        output = capsys.readouterr().out
        assert output.count("Test Set Evaluation") == 1
        # Dispatch path applies model display-name mapping (rsf -> RSF, etc.)
        assert "RSF" in output
        assert "CoxPH" in output

    def test_dispatch_warning_stacklevel_points_to_caller(self, capsys):
        """When dispatched via print_metrics() with a model_order entry missing
        from concordance, the warning points to user code, not internal modules."""
        import os
        from mlsurv.results.containers import EvaluationResult
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.72, 0.68],
        })
        result = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
            model_order=['rsf', 'coxph', 'missing'],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result.print_metrics()
        assert len(w) == 1
        assert "missing" in str(w[0].message)
        basename = os.path.basename(w[0].filename)
        assert basename != "printing.py"
        assert basename != "_containers_printing.py"
        assert basename != "containers.py"


class TestPrintBootstrapSummaryDispatch:
    """Tests for print_bootstrap_summary() routing of multi-model BootstrapResult
    through the consolidated table printer."""

    @pytest.fixture
    def multi_model_boot_result(self):
        from mlsurv.results.containers import BootstrapResult
        summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf', 'coxph', 'coxph'],
            'metric': ['harrell', 'integrated_brier', 'harrell', 'integrated_brier'],
            'point': [0.725, 0.183, 0.690, 0.205],
            'mean': [0.723, 0.185, 0.688, 0.207],
            'std': [0.032, 0.015, 0.030, 0.018],
            'ci_lower': [0.665, 0.158, 0.630, 0.175],
            'ci_upper': [0.781, 0.212, 0.745, 0.240],
        })
        iterations = pd.DataFrame({
            'iteration': list(range(50)) * 2,
            'model_type': ['rsf'] * 50 + ['coxph'] * 50,
        })
        return BootstrapResult(summary=summary, iterations=iterations)

    def test_single_combined_table(self, multi_model_boot_result, capsys):
        """Multi-model bootstrap prints ONE consolidated table, not one per model."""
        multi_model_boot_result.print_summary()
        output = capsys.readouterr().out
        # Multi-model header from print_multi_model_bootstrap_table
        assert "Test Set Results" in output
        assert output.count("Test Set Results") == 1
        # Old per-model header should NOT appear
        assert "Bootstrap Results" not in output
        # Both model display names appear as rows, in input order (model_order honored)
        assert "RSF" in output
        assert "CoxPH" in output
        assert output.index("RSF") < output.index("CoxPH")
        # Point estimates from summary['point'] surface (empty-concordance fallback contract)
        assert "0.725" in output  # rsf harrell
        assert "0.690" in output  # coxph harrell
        # Best-model marker appears (rsf wins harrell, IBS; coxph wins none — no <best row marker)
        assert "→" in output  # per-metric best arrow

    def test_iteration_count_in_header(self, multi_model_boot_result, capsys):
        """Header reflects actual bootstrap iteration count from iterations df."""
        multi_model_boot_result.print_summary()
        output = capsys.readouterr().out
        assert "50 bootstrap iterations" in output

    def test_single_model_uses_per_model_table(self, capsys):
        """Single-model bootstrap should still use the per-model 'Bootstrap Results' header."""
        from mlsurv.results.containers import BootstrapResult
        summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf'],
            'metric': ['harrell', 'integrated_brier'],
            'point': [0.725, 0.183],
            'mean': [0.723, 0.185],
            'std': [0.032, 0.015],
            'ci_lower': [0.665, 0.158],
            'ci_upper': [0.781, 0.212],
        })
        iterations = pd.DataFrame({'iteration': list(range(50)), 'model_type': ['rsf'] * 50})
        BootstrapResult(summary=summary, iterations=iterations).print_summary()
        output = capsys.readouterr().out
        assert "Bootstrap Results" in output
        assert "Test Set Results" not in output


class TestFindBestCvModels:
    """Tests for _find_best_cv_models() helper."""

    def test_higher_is_better(self):
        """Harrell C-index: higher is better."""
        results = {}
        for name, mean in [('rsf', 0.723), ('coxph', 0.698)]:
            r = MagicMock(spec=['cv_metrics', 'cv_metric_mean'])
            r.cv_metrics = {'harrell': {'mean': mean}}
            r.cv_metric_mean = mean
            results[name] = r
        best = _find_best_cv_models(results, ['harrell'])
        assert best == {'harrell': 'rsf'}

    def test_lower_is_better(self):
        """IBS: lower is better."""
        results = {}
        for name, mean in [('rsf', 0.190), ('coxph', 0.180)]:
            r = MagicMock(spec=['cv_metrics', 'cv_metric_mean'])
            r.cv_metrics = {'integrated_brier': {'mean': mean}}
            r.cv_metric_mean = None
            results[name] = r
        best = _find_best_cv_models(results, ['integrated_brier'])
        assert best == {'integrated_brier': 'coxph'}

    def test_mixed_winners(self):
        """Different models win different metrics."""
        results = {}
        # RSF: better harrell, worse IBS
        r = MagicMock(spec=['cv_metrics', 'cv_metric_mean'])
        r.cv_metrics = {'harrell': {'mean': 0.723}, 'integrated_brier': {'mean': 0.200}}
        r.cv_metric_mean = 0.723
        results['rsf'] = r
        # CoxPH: worse harrell, better IBS
        r = MagicMock(spec=['cv_metrics', 'cv_metric_mean'])
        r.cv_metrics = {'harrell': {'mean': 0.698}, 'integrated_brier': {'mean': 0.180}}
        r.cv_metric_mean = 0.698
        results['coxph'] = r
        best = _find_best_cv_models(results, ['harrell', 'integrated_brier'])
        assert best == {'harrell': 'rsf', 'integrated_brier': 'coxph'}

    def test_fallback_to_cv_metric_mean(self):
        """When cv_metrics lacks harrell, falls back to cv_metric_mean."""
        r = MagicMock(spec=['cv_metrics', 'cv_metric_mean'])
        r.cv_metrics = {}
        r.cv_metric_mean = 0.710
        results = {'rsf': r}
        best = _find_best_cv_models(results, ['harrell'])
        assert best == {'harrell': 'rsf'}

    def test_all_nan_values_skipped(self):
        """When all models have NaN mean for a metric, metric is absent from result."""
        results = {}
        for name in ['rsf', 'coxph']:
            r = MagicMock(spec=['cv_metrics', 'cv_metric_mean'])
            r.cv_metrics = {'harrell': {'mean': float('nan')}}
            r.cv_metric_mean = float('nan')
            results[name] = r
        best = _find_best_cv_models(results, ['harrell'])
        assert 'harrell' not in best


class TestMarkBestInCells:
    """Tests for _mark_best_in_cells() helper."""

    def test_multi_model_marks_best(self):
        """Best cell gets arrow prefix, others get 2-space indent."""
        cells = ["0.723 (0.032)   ", "0.185 (0.015)   "]
        marked = _mark_best_in_cells(
            cells, 'rsf', ['harrell', 'integrated_brier'],
            {'harrell': 'rsf', 'integrated_brier': 'coxph'}, 16, True
        )
        assert marked[0].startswith(_BEST_MARKER)
        assert marked[1].startswith('  ')

    def test_single_model_no_markers(self):
        """Single model (multi=False) returns cells unchanged."""
        cells = ["0.723 (0.032)   "]
        result = _mark_best_in_cells(
            cells, 'rsf', ['harrell'], {'harrell': 'rsf'}, 16, False
        )
        assert result is cells  # same object, no modification

    def test_column_width_preserved(self):
        """Marked cells should maintain the original column width."""
        cells = ["0.723 (0.032)   "]
        marked = _mark_best_in_cells(
            cells, 'rsf', ['harrell'], {'harrell': 'rsf'}, 16, True
        )
        assert len(marked[0]) == 16

    def test_column_width_preserved_non_best(self):
        """Non-best cell should also maintain the original column width."""
        cells = ["0.723 (0.032)   "]
        marked = _mark_best_in_cells(
            cells, 'coxph', ['harrell'], {'harrell': 'rsf'}, 16, True
        )
        assert len(marked[0]) == 16
        assert marked[0].startswith('  ')


class TestCvTableBestMarkers:
    """Integration tests for best-metric markers in CV table output."""

    def test_best_markers_in_cv_table(self, capsys):
        """Best model per metric should have arrow marker in multi-model CV table."""
        results = {}
        for name, mean, std in [('rsf', 0.723, 0.032), ('coxph', 0.698, 0.045)]:
            r = MagicMock(spec=['cv_metrics', 'cv_metric_mean', 'cv_metric_std'])
            r.cv_metrics = {'harrell': {'mean': mean, 'std': std, 'scores': []}}
            r.cv_metric_mean = mean
            r.cv_metric_std = std
            results[name] = r
        print_multi_model_cv_table(results, n_folds=5)
        output = capsys.readouterr().out
        for line in output.splitlines():
            if line.strip().startswith('rsf'):
                assert _BEST_MARKER in line
            elif line.strip().startswith('coxph'):
                assert _BEST_MARKER not in line

    def test_no_markers_single_model(self, capsys):
        """Single model should not have arrow markers."""
        r = MagicMock(spec=['cv_metrics', 'cv_metric_mean', 'cv_metric_std'])
        r.cv_metrics = {'harrell': {'mean': 0.723, 'std': 0.032, 'scores': []}}
        r.cv_metric_mean = 0.723
        r.cv_metric_std = 0.032
        print_multi_model_cv_table({'rsf': r}, n_folds=5)
        output = capsys.readouterr().out
        assert _BEST_MARKER not in output

    def test_cv_mixed_winners(self, capsys):
        """Different models winning different CV metrics each get arrow marker on their best."""
        results = {}
        # RSF: better harrell, worse IBS
        r = MagicMock(spec=['cv_metrics', 'cv_metric_mean', 'cv_metric_std'])
        r.cv_metrics = {
            'harrell': {'mean': 0.723, 'std': 0.032, 'scores': []},
            'integrated_brier': {'mean': 0.200, 'std': 0.020, 'scores': []},
        }
        r.cv_metric_mean = 0.723
        r.cv_metric_std = 0.032
        results['rsf'] = r
        # CoxPH: worse harrell, better IBS
        r = MagicMock(spec=['cv_metrics', 'cv_metric_mean', 'cv_metric_std'])
        r.cv_metrics = {
            'harrell': {'mean': 0.698, 'std': 0.045, 'scores': []},
            'integrated_brier': {'mean': 0.180, 'std': 0.015, 'scores': []},
        }
        r.cv_metric_mean = 0.698
        r.cv_metric_std = 0.045
        results['coxph'] = r
        print_multi_model_cv_table(results, n_folds=5)
        output = capsys.readouterr().out
        for line in output.splitlines():
            if line.strip().startswith('rsf') or line.strip().startswith('coxph'):
                assert _BEST_MARKER in line  # Each wins one metric


class TestBootstrapTableBestMarkers:
    """Integration tests for best-metric markers in bootstrap table output."""

    def test_best_markers_in_bootstrap_table(self, capsys):
        """Best model per metric should have arrow marker in multi-model bootstrap table."""
        summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf', 'coxph', 'coxph'],
            'metric': ['harrell', 'integrated_brier', 'harrell', 'integrated_brier'],
            'point': [0.723, 0.188, 0.698, 0.205],
            'mean': [0.720, 0.190, 0.700, 0.200],
            'std': [0.030, 0.015, 0.040, 0.020],
            'ci_lower': [0.665, 0.162, 0.625, 0.165],
            'ci_upper': [0.775, 0.218, 0.775, 0.235],
        })
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.723, 0.698],
            'integrated_brier_test': [0.185, 0.210],
        })
        print_multi_model_bootstrap_table(summary, concordance, n_bootstrap=200)
        output = capsys.readouterr().out
        for line in output.splitlines():
            if line.strip().startswith('rsf'):
                assert _BEST_MARKER in line  # RSF best on both
            elif line.strip().startswith('coxph'):
                assert _BEST_MARKER not in line

    def test_single_model_no_markers(self, capsys):
        """Single model bootstrap table should not have arrow markers."""
        summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf'],
            'metric': ['harrell', 'integrated_brier'],
            'point': [0.723, 0.188],
            'mean': [0.720, 0.190],
            'std': [0.030, 0.015],
            'ci_lower': [0.665, 0.162],
            'ci_upper': [0.775, 0.218],
        })
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': [0.723],
            'integrated_brier_test': [0.185],
        })
        print_multi_model_bootstrap_table(
            summary, concordance, n_bootstrap=200, model_order=['rsf']
        )
        output = capsys.readouterr().out
        assert _BEST_MARKER not in output


class TestPrintBestParams:
    """Tests for print_best_params() single-model best params output."""

    def test_simple_flat_params(self, capsys):
        """Flat params should print on one line."""
        print_best_params({'n_estimators': 200, 'max_depth': 15}, model_name='rsf')
        output = capsys.readouterr().out
        assert "Best Parameters (rsf): n_estimators=200, max_depth=15" in output

    def test_prefixed_params_cleaned(self, capsys):
        """model__ prefix should be stripped in output."""
        print_best_params({'model__n_estimators': 200}, model_name='rsf')
        output = capsys.readouterr().out
        assert "n_estimators=200" in output
        assert "model__" not in output

    def test_multi_component_grouped(self, capsys):
        """Multiple components should produce grouped output."""
        params = {
            'model__n_estimators': 200,
            'feature_selector__k': 10,
        }
        print_best_params(params, model_name='rsf')
        output = capsys.readouterr().out
        assert "Model: n_estimators=200" in output
        assert "Feature Selector: k=10" in output

    def test_empty_dict_noop(self, capsys):
        """Empty params should produce no output."""
        print_best_params({})
        output = capsys.readouterr().out
        assert output == ""

    def test_float_formatting(self, capsys):
        """Floats should use g format (no trailing zeros)."""
        print_best_params({'learning_rate': 0.001}, model_name='rsf')
        output = capsys.readouterr().out
        assert "learning_rate=0.001" in output

    def test_selector_k_routing(self, capsys):
        """Prefixed feature_selector__k routes to the Feature Selector group."""
        params = {'n_estimators': 200, 'feature_selector__k': 10}
        print_best_params(params, model_name='rsf')
        output = capsys.readouterr().out
        assert "Model: n_estimators=200" in output
        assert "Feature Selector: k=10" in output

    def test_no_model_name(self, capsys):
        """Without model_name, prefix should be just 'Best Parameters'."""
        print_best_params({'n_estimators': 200})
        output = capsys.readouterr().out
        assert "Best Parameters: n_estimators=200" in output


class TestPrintMultiModelBestParams:
    """Tests for print_multi_model_best_params() multi-model output."""

    def test_basic_multi_model(self, capsys):
        """Should print one line per tuned model."""
        r1 = MagicMock()
        r1.best_params_ = {'n_estimators': 200, 'max_depth': 15}
        r2 = MagicMock()
        r2.best_params_ = {'alpha': 0.1}
        print_multi_model_best_params({'rsf': r1, 'coxnet': r2})
        output = capsys.readouterr().out
        assert "Best Parameters:" in output
        assert "rsf: n_estimators=200, max_depth=15" in output
        assert "coxnet: alpha=0.1" in output

    def test_skips_non_tuned(self, capsys):
        """Models without best_params_ should be skipped."""
        r1 = MagicMock()
        r1.best_params_ = {'n_estimators': 200}
        r2 = MagicMock(spec=['cv_metrics'])  # No best_params_
        print_multi_model_best_params({'rsf': r1, 'coxph': r2})
        output = capsys.readouterr().out
        assert "rsf:" in output
        assert "coxph" not in output

    def test_empty_results_noop(self, capsys):
        """Empty dict should produce no output."""
        print_multi_model_best_params({})
        output = capsys.readouterr().out
        assert output == ""

    def test_prefixed_params_cleaned(self, capsys):
        """Prefixed keys should be cleaned in multi-model output."""
        r1 = MagicMock()
        r1.best_params_ = {'model__n_estimators': 200, 'feature_selector__k': 10}
        print_multi_model_best_params({'rsf': r1})
        output = capsys.readouterr().out
        assert "n_estimators=200" in output
        assert "k=10" in output
        assert "model__" not in output


class TestFormatParamValue:
    """Tests for _format_param_value() formatting helper."""

    @pytest.mark.parametrize("value, expected", [
        (200, "200"),                  # int
        (0.001, "0.001"),              # float
        (1.0, "1"),                    # float with trailing zeros
        ([64, 64], "[64, 64]"),        # list
        ('relu', "'relu'"),            # string
        (True, "True"),                # bool True
        (False, "False"),              # bool False
    ])
    def test_format_param_value(self, value, expected):
        assert _format_param_value(value) == expected


class TestGroupBestParams:
    """Tests for _group_best_params() helper."""

    def test_flat_params(self):
        result = _group_best_params({'n_estimators': 200, 'max_depth': 15})
        assert 'Model' in result
        assert result['Model'] == {'n_estimators': '200', 'max_depth': '15'}

    def test_prefixed_params(self):
        result = _group_best_params({'model__lr': 0.01, 'imputer__n_neighbors': 5})
        assert result['Model'] == {'lr': '0.01'}
        assert result['Imputer'] == {'n_neighbors': '5'}

    def test_bare_k_is_model_param(self):
        # An unprefixed 'k' is the model's own hyperparameter,
        # not the selector's. The selector's k arrives as feature_selector__k.
        result = _group_best_params({'k': 10, 'n_estimators': 200})
        assert result['Model'] == {'k': '10', 'n_estimators': '200'}
        assert 'Feature Selector' not in result

    def test_prefixed_selector_k(self):
        result = _group_best_params({'feature_selector__k': 10, 'n_estimators': 200})
        assert result['Feature Selector'] == {'k': '10'}
        assert result['Model'] == {'n_estimators': '200'}


class TestFormatParamsInline:
    """Tests for _format_params_inline() helper."""

    def test_basic(self):
        assert _format_params_inline({'a': '1', 'b': '2'}) == "a=1, b=2"

    def test_single(self):
        assert _format_params_inline({'x': '42'}) == "x=42"


# =====================================================================
# Statistical test print functions
# =====================================================================


class TestPrintPhTest:
    """Tests for print_ph_test() formatted PH test output."""

    @pytest.fixture
    def ph_df(self):
        return pd.DataFrame({
            'feature': ['age', 'tumor_size', 'GLOBAL'],
            'test_statistic': [2.341, 8.712, 18.503],
            'p_value': [0.126, 3.15e-3, 3.45e-4],
            'p_adjusted': [0.126, 6.30e-3, float('nan')],
            'significant': [False, True, True],
        })

    def test_header(self, ph_df, capsys):
        print_ph_test(ph_df)
        output = capsys.readouterr().out
        assert "Proportional Hazards Test (Grambsch-Therneau)" in output
        assert "Benjamini-Hochberg FDR-corrected" in output

    def test_feature_names(self, ph_df, capsys):
        print_ph_test(ph_df)
        output = capsys.readouterr().out
        assert "age" in output
        assert "tumor_size" in output

    def test_significance_stars(self, ph_df, capsys):
        print_ph_test(ph_df)
        output = capsys.readouterr().out
        # tumor_size p=3.15e-3 → **
        # GLOBAL p=3.45e-4 → ***
        assert "**" in output
        assert "***" in output

    def test_global_separator(self, ph_df, capsys):
        """GLOBAL row should be separated from feature rows by a separator line."""
        print_ph_test(ph_df)
        output = capsys.readouterr().out
        lines = output.strip().splitlines()
        # Find GLOBAL line and check that it's preceded by a separator
        for i, line in enumerate(lines):
            if line.strip().startswith("GLOBAL"):
                assert "---" in lines[i - 1]
                break

    def test_empty_df(self, capsys):
        print_ph_test(pd.DataFrame(columns=['feature', 'test_statistic', 'p_value', 'p_adjusted', 'significant']))
        output = capsys.readouterr().out
        assert output == ""

    def test_none_input(self, capsys):
        print_ph_test(None)
        output = capsys.readouterr().out
        assert output == ""

    def test_nan_pvalue(self, capsys):
        df = pd.DataFrame({
            'feature': ['age'],
            'test_statistic': [float('nan')],
            'p_value': [float('nan')],
            'p_adjusted': [float('nan')],
            'significant': [False],
        })
        print_ph_test(df)
        output = capsys.readouterr().out
        assert "NaN" in output

    def test_precision(self, ph_df, capsys):
        print_ph_test(ph_df, precision=1)
        output = capsys.readouterr().out
        assert "2.3" in output


class TestPrintRmstTest:
    """Tests for print_rmst_test() formatted RMST output."""

    @pytest.fixture
    def rmst_result(self):
        from collections import namedtuple
        RMSTResult = namedtuple(
            'RMSTResult',
            ['rmst', 'se', 'diff', 'diff_se', 'ci_lower', 'ci_upper', 'z_stat', 'p_value', 'tau']
        )
        return RMSTResult(
            rmst={'Low Risk': 28.432, 'High Risk': 22.156},
            se={'Low Risk': 1.234, 'High Risk': 1.567},
            diff=-6.276,
            diff_se=1.992,
            ci_lower=-8.987,
            ci_upper=-3.565,
            z_stat=-3.145,
            p_value=0.002,
            tau=36.0,
        )

    def test_header_with_tau(self, rmst_result, capsys):
        print_rmst_test(rmst_result)
        output = capsys.readouterr().out
        assert "RMST Comparison (tau = 36)" in output

    def test_group_names(self, rmst_result, capsys):
        print_rmst_test(rmst_result)
        output = capsys.readouterr().out
        assert "Low Risk" in output
        assert "High Risk" in output

    def test_difference_and_ci(self, rmst_result, capsys):
        print_rmst_test(rmst_result)
        output = capsys.readouterr().out
        assert "-6.276" in output
        assert "-8.987" in output
        assert "-3.565" in output

    def test_pvalue_stars(self, rmst_result, capsys):
        print_rmst_test(rmst_result)
        output = capsys.readouterr().out
        assert "**" in output  # p=0.002 → **

    def test_positive_diff(self, capsys):
        from collections import namedtuple
        RMSTResult = namedtuple(
            'RMSTResult',
            ['rmst', 'se', 'diff', 'diff_se', 'ci_lower', 'ci_upper', 'z_stat', 'p_value', 'tau']
        )
        result = RMSTResult(
            rmst={'A': 30.0, 'B': 25.0},
            se={'A': 1.0, 'B': 1.0},
            diff=5.0, diff_se=1.414,
            ci_lower=2.228, ci_upper=7.772,
            z_stat=3.536, p_value=0.0004,
            tau=36.0,
        )
        print_rmst_test(result)
        output = capsys.readouterr().out
        assert "+5.000" in output

    def test_fractional_tau(self, capsys):
        from collections import namedtuple
        RMSTResult = namedtuple(
            'RMSTResult',
            ['rmst', 'se', 'diff', 'diff_se', 'ci_lower', 'ci_upper', 'z_stat', 'p_value', 'tau']
        )
        result = RMSTResult(
            rmst={'A': 10.0, 'B': 8.0},
            se={'A': 0.5, 'B': 0.5},
            diff=2.0, diff_se=0.707,
            ci_lower=0.614, ci_upper=3.386,
            z_stat=2.828, p_value=0.005,
            tau=12.5,
        )
        print_rmst_test(result)
        output = capsys.readouterr().out
        assert "tau = 12.500" in output


class TestPrintRmstPairwise:
    """Tests for print_rmst_pairwise() formatted pairwise RMST output."""

    @pytest.fixture
    def pairwise_result(self):
        from collections import namedtuple
        PairwiseRMSTResult = namedtuple(
            'PairwiseRMSTResult',
            ['rmst', 'se', 'pairwise', 'tau', 'p_adjust_method']
        )
        return PairwiseRMSTResult(
            rmst={'Low': 30.0, 'Medium': 25.0, 'High': 20.0},
            se={'Low': 1.0, 'Medium': 1.2, 'High': 1.5},
            pairwise=[
                {'groups': ('Low', 'Medium'), 'diff': -5.0, 'diff_se': 1.56,
                 'ci_lower': -8.06, 'ci_upper': -1.94, 'raw_p': 0.001, 'adjusted_p': 0.003},
                {'groups': ('Low', 'High'), 'diff': -10.0, 'diff_se': 1.80,
                 'ci_lower': -13.53, 'ci_upper': -6.47, 'raw_p': 0.0001, 'adjusted_p': 0.0003},
                {'groups': ('Medium', 'High'), 'diff': -5.0, 'diff_se': 1.92,
                 'ci_lower': -8.76, 'ci_upper': -1.24, 'raw_p': 0.009, 'adjusted_p': 0.018},
            ],
            tau=36.0,
            p_adjust_method='holm',
        )

    def test_per_group_table(self, pairwise_result, capsys):
        print_rmst_pairwise(pairwise_result)
        output = capsys.readouterr().out
        assert "Low" in output
        assert "Medium" in output
        assert "High" in output
        assert "30.000" in output

    def test_pairwise_table(self, pairwise_result, capsys):
        print_rmst_pairwise(pairwise_result)
        output = capsys.readouterr().out
        assert "Low vs Medium" in output
        assert "Low vs High" in output
        assert "Medium vs High" in output

    def test_correction_label(self, pairwise_result, capsys):
        print_rmst_pairwise(pairwise_result)
        output = capsys.readouterr().out
        assert "holm correction" in output

    def test_adjusted_p_used(self, pairwise_result, capsys):
        print_rmst_pairwise(pairwise_result)
        output = capsys.readouterr().out
        assert "p-adj" in output

    def test_significance_stars(self, pairwise_result, capsys):
        print_rmst_pairwise(pairwise_result)
        output = capsys.readouterr().out
        assert "**" in output  # adjusted_p=0.003 → **
        assert "***" in output  # adjusted_p=0.0003 → ***


class TestPrintCalibrationTest:
    """Tests for print_calibration_test() one-line output."""

    def test_gnd_method(self, capsys):
        print_calibration_test(8.234, 0.041)
        output = capsys.readouterr().out
        assert "Calibration GOF Test (GND)" in output
        assert "8.234" in output
        assert "0.041" in output
        assert "predicted and observed survival probabilities diverge" in output

    def test_hl_method(self, capsys):
        print_calibration_test(5.0, 0.3, method='hl')
        output = capsys.readouterr().out
        assert "Hosmer-Lemeshow" in output

    def test_significance_star(self, capsys):
        print_calibration_test(10.0, 0.005)
        output = capsys.readouterr().out
        assert "**" in output

    def test_nan_handling(self, capsys):
        print_calibration_test(float('nan'), float('nan'))
        output = capsys.readouterr().out
        assert "NaN" in output

    def test_precision(self, capsys):
        print_calibration_test(8.23456, 0.04123, precision=2)
        output = capsys.readouterr().out
        assert "8.23" in output


class TestPrintPhTestEdgeCases:
    """Additional edge cases for print_ph_test."""

    def test_no_global_row(self, capsys):
        """PH test result without GLOBAL row should still print correctly."""
        df = pd.DataFrame({
            'feature': ['age', 'tumor_size'],
            'test_statistic': [2.341, 0.512],
            'p_value': [0.126, 0.474],
            'p_adjusted': [0.252, 0.474],
            'significant': [False, False],
        })
        print_ph_test(df)
        output = capsys.readouterr().out
        assert "age" in output
        assert "tumor_size" in output
        # No GLOBAL data row (footer may mention "GLOBAL" in explanation text)
        lines = output.strip().splitlines()
        data_lines = [l for l in lines if l.strip().startswith("GLOBAL") and "|" in l]
        assert len(data_lines) == 0
        # Should not have a double separator before footer
        # Last content line before footer should not be a separator
        data_lines = [l for l in lines if l.startswith("age") or l.startswith("tumor")]
        assert len(data_lines) == 2


class TestPrintRmstEdgeCases:
    """Additional edge cases for RMST print functions."""

    def _make_rmst_result(self, **overrides):
        from collections import namedtuple
        RMSTResult = namedtuple(
            'RMSTResult',
            ['rmst', 'se', 'diff', 'diff_se', 'ci_lower', 'ci_upper', 'z_stat', 'p_value', 'tau']
        )
        defaults = dict(
            rmst={'A': 10.0, 'B': 8.0}, se={'A': 0.5, 'B': 0.5},
            diff=2.0, diff_se=0.707, ci_lower=0.614, ci_upper=3.386,
            z_stat=2.828, p_value=0.005, tau=12.0,
        )
        defaults.update(overrides)
        return RMSTResult(**defaults)

    def test_none_input(self, capsys):
        print_rmst_test(None)
        output = capsys.readouterr().out
        assert output == ""

    def test_nan_diff(self, capsys):
        result = self._make_rmst_result(diff=float('nan'), z_stat=float('nan'), p_value=float('nan'))
        print_rmst_test(result)
        output = capsys.readouterr().out
        assert "NaN" in output

    def test_pairwise_none_input(self, capsys):
        print_rmst_pairwise(None)
        output = capsys.readouterr().out
        assert output == ""

    def test_pairwise_empty_list(self, capsys):
        from collections import namedtuple
        PairwiseRMSTResult = namedtuple(
            'PairwiseRMSTResult',
            ['rmst', 'se', 'pairwise', 'tau', 'p_adjust_method']
        )
        result = PairwiseRMSTResult(
            rmst={'A': 10.0, 'B': 8.0}, se={'A': 0.5, 'B': 0.5},
            pairwise=[], tau=12.0, p_adjust_method='holm',
        )
        print_rmst_pairwise(result)
        output = capsys.readouterr().out
        # Per-group table should still print
        assert "A" in output
        assert "B" in output
        # Pairwise section should not appear
        assert "Pair" not in output


class TestPrintCalibrationEdgeCases:
    """Additional edge cases for print_calibration_test."""

    def test_unknown_method(self, capsys):
        print_calibration_test(5.0, 0.3, method='custom')
        output = capsys.readouterr().out
        assert "CUSTOM" in output


# =====================================================================
# _filter_by_model_type and _warn_stacklevel
# =====================================================================


class TestFilterByModelTypeStacklevel:
    """Tests for _filter_by_model_type _warn_stacklevel propagation."""

    @pytest.fixture
    def multi_model_df(self):
        return pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.72, 0.68],
        })

    def test_bad_model_name_warns(self, multi_model_df):
        """Unknown model_name emits UserWarning with 'not found'."""
        with pytest.warns(InternalWarning, match="not found"):
            result = _filter_by_model_type(multi_model_df, "nonexistent")
        # Falls back to unfiltered
        assert len(result) == 2

    def test_stacklevel_points_to_caller(self, multi_model_df):
        """_warn_stacklevel=2 makes warning point to this test file."""
        import os
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _filter_by_model_type(multi_model_df, "bad", _warn_stacklevel=2)
        assert len(w) == 1
        assert os.path.basename(w[0].filename) == "test_printing.py"

    def test_default_stacklevel(self, multi_model_df):
        """Default stacklevel=3 does not point to the internal printing module."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _filter_by_model_type(multi_model_df, "bad")
        assert len(w) == 1
        import os
        assert os.path.basename(w[0].filename) != "printing.py"

    def test_container_dispatch_stacklevel(self):
        """_warn_stacklevel=5 in container path does not blame printing.py or containers.py."""
        import os
        from mlsurv.results.containers import EvaluationResult

        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.72, 0.68],
        })
        # Construct minimal EvaluationResult (no bootstrap -> evaluation table path)
        result = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
            model_order=['rsf', 'coxph', 'nonexistent'],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result.print_metrics()
        # The 'nonexistent' model_order entry should trigger a warning
        assert len(w) == 1
        assert "not found" in str(w[0].message)
        basename = os.path.basename(w[0].filename)
        assert basename != "printing.py"
        assert basename != "containers.py"
