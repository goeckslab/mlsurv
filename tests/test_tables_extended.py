"""Tests for mlsurv.results.tables formatting functions.

Covers format_metrics_table(), format_importance_table(),
format_bootstrap_summary(), format_hazard_ratios(), to_latex(),
to_markdown(), and to_html().
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from mlsurv.results.tables import (
    format_metrics_table,
    format_importance_table,
    format_bootstrap_summary,
    format_hazard_ratios,
    to_latex,
    to_markdown,
    to_html,
)


class TestFormatMetricsTable:
    """Tests for format_metrics_table() display formatting."""

    @pytest.fixture
    def metrics_df(self):
        return pd.DataFrame({
            'model_type': ['rsf', 'coxph', 'gbsa'],
            'cv_metric_mean': [0.723, 0.698, 0.715],
            'cv_metric_std': [0.032, 0.045, 0.038],
        })

    def test_auto_detect_numeric_columns(self, metrics_df):
        """Should auto-detect numeric columns when metrics=None."""
        result = format_metrics_table(metrics_df)
        assert 'model_type' in result.columns
        assert 'cv_metric_mean' in result.columns

    def test_sort_by_column(self, metrics_df):
        """sort_by should order by specified column."""
        result = format_metrics_table(metrics_df, sort_by='cv_metric_mean', ascending=False)
        assert result.iloc[0]['model_type'] == 'rsf'

    def test_precision_rounding(self, metrics_df):
        """Numeric values should be rounded to specified precision."""
        result = format_metrics_table(metrics_df, precision=2)
        assert result['cv_metric_mean'].iloc[0] == 0.72

    def test_specific_metrics(self, metrics_df):
        """Only specified metrics should appear."""
        result = format_metrics_table(metrics_df, metrics=['cv_metric_mean'])
        assert 'cv_metric_std' not in result.columns

    def test_reset_index(self, metrics_df):
        """Result should have reset index."""
        result = format_metrics_table(metrics_df)
        assert list(result.index) == list(range(len(result)))


class TestFormatImportanceTable:
    """Tests for format_importance_table() feature ranking display."""

    @pytest.fixture
    def importance_df(self):
        return pd.DataFrame({
            'feature': ['age', 'stage', 'grade', 'size', 'marker'],
            'perm_mean': [0.05, 0.03, 0.08, 0.01, 0.04],
            'perm_std': [0.01, 0.005, 0.02, 0.003, 0.008],
        })

    def test_top_n_features(self, importance_df):
        """Should return top N features by importance."""
        result = format_importance_table(importance_df, n=3)
        assert len(result) == 3

    def test_rank_column_added(self, importance_df):
        """Should add a Rank column starting at 1."""
        result = format_importance_table(importance_df, n=3)
        assert 'Rank' in result.columns
        assert list(result['Rank']) == [1, 2, 3]

    def test_sorted_by_importance(self, importance_df):
        """Top feature should have highest importance."""
        result = format_importance_table(importance_df, n=1)
        assert result['Feature'].iloc[0] == 'grade'

    def test_std_column_included(self, importance_df):
        """Std Dev column should be present when std data exists."""
        result = format_importance_table(importance_df)
        assert 'Std Dev' in result.columns

    def test_missing_importance_col_raises_valueerror(self, importance_df):
        """A bad importance_col should raise a clear ValueError, not a bare KeyError."""
        with pytest.raises(ValueError, match=r"importance_col='nope'"):
            format_importance_table(importance_df, importance_col='nope')

    def test_missing_feature_col_raises_valueerror(self, importance_df):
        """A bad feature_col should raise a clear ValueError, not a bare KeyError."""
        with pytest.raises(ValueError, match=r"feature_col='nope'"):
            format_importance_table(importance_df, feature_col='nope')

    def test_missing_col_error_lists_available_columns(self, importance_df):
        """The error message should list the available columns to aid debugging."""
        with pytest.raises(ValueError, match=r"Available columns:.*perm_mean"):
            format_importance_table(importance_df, importance_col='nope')

    def test_missing_std_col_degrades_gracefully(self, importance_df):
        """std_col is optional: a missing one should drop Std Dev, not raise."""
        result = format_importance_table(importance_df, std_col='nonexistent')
        assert 'Std Dev' not in result.columns
        assert 'Importance' in result.columns


class TestFormatBootstrapSummary:
    """Tests for format_bootstrap_summary() CI formatting."""

    @pytest.fixture
    def bootstrap_df(self):
        df = pd.DataFrame({
            'metric': ['harrell', 'integrated_brier'],
            'mean': [0.723, 0.185],
            'std': [0.032, 0.015],
            'ci_lower': [0.665, 0.158],
            'ci_upper': [0.781, 0.212],
        })
        df.attrs['ci_level'] = 0.95  # as stamped by BootstrapResult
        return df

    def test_ci_column_created(self, bootstrap_df):
        """Should create combined 95% CI column."""
        result = format_bootstrap_summary(bootstrap_df)
        assert '95% CI' in result.columns
        assert 'ci_lower' not in result.columns
        assert 'ci_upper' not in result.columns

    def test_column_renaming(self, bootstrap_df):
        """Raw column names should be renamed to display-friendly names."""
        result = format_bootstrap_summary(bootstrap_df)
        assert 'metric' not in result.columns
        assert 'Mean' in result.columns

    def test_precision_applied(self, bootstrap_df):
        """Values should be rounded to specified precision."""
        result = format_bootstrap_summary(bootstrap_df, precision=2)
        assert result['Mean'].iloc[0] == 0.72

    def test_ci_level_param(self, bootstrap_df):
        """Explicit ci_level should set the CI column label."""
        result = format_bootstrap_summary(bootstrap_df, ci_level=0.80)
        assert '80% CI' in result.columns
        assert '95% CI' not in result.columns

    def test_ci_level_from_attrs(self, bootstrap_df):
        """ci_level stamped on df.attrs (by BootstrapResult) should be used."""
        bootstrap_df.attrs['ci_level'] = 0.80
        result = format_bootstrap_summary(bootstrap_df)
        assert '80% CI' in result.columns
        assert '95% CI' not in result.columns

    def test_explicit_ci_level_overrides_attrs(self, bootstrap_df):
        """Explicit ci_level wins over the attrs value."""
        bootstrap_df.attrs['ci_level'] = 0.80
        result = format_bootstrap_summary(bootstrap_df, ci_level=0.90)
        assert '90% CI' in result.columns

    def test_non_integer_ci_level(self, bootstrap_df):
        """Non-integer percent levels should keep their decimals."""
        result = format_bootstrap_summary(bootstrap_df, ci_level=0.975)
        assert '97.5% CI' in result.columns

    def test_ci_level_float_noise(self, bootstrap_df):
        """0.9 * 100 float noise must not leak into the label."""
        result = format_bootstrap_summary(bootstrap_df, ci_level=0.9)
        assert '90% CI' in result.columns

    @pytest.mark.parametrize('bad', [0, 1, -0.5, 95])
    def test_ci_level_out_of_range_raises(self, bootstrap_df, bad):
        """ci_level outside (0, 1) raises a clear ValueError, not a bad label."""
        with pytest.raises(ValueError, match=r"ci_level must be between 0 and 1"):
            format_bootstrap_summary(bootstrap_df, ci_level=bad)

    def test_unknown_ci_level_warns(self):
        """No ci_level and no attrs -> SuggestionWarning before labeling 95%."""
        from mlsurv._warnings import SuggestionWarning
        bare_df = pd.DataFrame({
            'metric': ['harrell'],
            'mean': [0.723],
            'ci_lower': [0.665],
            'ci_upper': [0.781],
        })
        with pytest.warns(SuggestionWarning, match="CI level unknown"):
            result = format_bootstrap_summary(bare_df)
        assert '95% CI' in result.columns

    def test_no_warning_when_attrs_present(self, bootstrap_df):
        """attrs-stamped DataFrames format without warning."""
        import warnings as _warnings
        bootstrap_df.attrs['ci_level'] = 0.95
        with _warnings.catch_warnings():
            _warnings.simplefilter('error')
            result = format_bootstrap_summary(bootstrap_df)
        assert '95% CI' in result.columns


class TestToLatex:
    """Tests for to_latex() LaTeX table generation."""

    def test_basic_latex_output(self):
        """Should produce valid LaTeX table structure."""
        df = pd.DataFrame({'Model': ['RSF', 'CoxPH'], 'C-Index': [0.723, 0.698]})
        latex = to_latex(df, caption='Test Table', label='tab:test')
        assert '\\begin{table}' in latex
        assert '\\caption{Test Table}' in latex
        assert '\\label{tab:test}' in latex
        assert 'RSF' in latex

    def test_precision(self):
        """Float values should respect precision parameter."""
        df = pd.DataFrame({'value': [0.12345]})
        latex = to_latex(df, precision=2)
        assert '0.12' in latex

    def test_escapes_special_chars(self):
        """LaTeX special characters should be escaped."""
        df = pd.DataFrame({'name': ['model_1']})
        latex = to_latex(df)
        assert r'model\_1' in latex


class TestToMarkdown:
    """Tests for to_markdown() Markdown table generation."""

    def test_basic_markdown(self):
        """Should produce a valid Markdown table."""
        df = pd.DataFrame({'Model': ['RSF'], 'Score': [0.723]})
        md = to_markdown(df)
        assert '|' in md
        assert 'RSF' in md

    def test_precision(self):
        """Float precision should be applied."""
        df = pd.DataFrame({'val': [0.12345]})
        md = to_markdown(df, precision=2)
        assert '0.12' in md


class TestToHtml:
    """Tests for to_html() HTML table generation."""

    def test_basic_html(self):
        """Should produce a valid HTML table."""
        df = pd.DataFrame({'Model': ['RSF'], 'Score': [0.723]})
        html = to_html(df)
        assert '<table' in html
        assert 'RSF' in html

    def test_css_classes(self):
        """CSS classes should be applied."""
        df = pd.DataFrame({'a': [1]})
        html = to_html(df, classes='custom-table')
        assert 'custom-table' in html

    def test_no_index(self):
        """Should not include DataFrame index."""
        df = pd.DataFrame({'a': ['x', 'y']}, index=[100, 101])
        html = to_html(df)
        assert '<th>100</th>' not in html
        assert '<th>101</th>' not in html
        assert '<th>a</th>' in html


class TestFormatHazardRatios:
    """Tests for format_hazard_ratios() Cox model hazard ratio table.

    Publication-facing function - incorrect math produces wrong HR values
    in manuscripts. Tests all three branches: with covariance, without
    covariance, and no coefficients.
    """

    def _make_cox_model(self, coef_values, feature_names=None, covariance=None):
        """Helper to create a mock Cox model."""
        coef = pd.Series(
            coef_values,
            index=feature_names or [f'Feature {i}' for i in range(len(coef_values))],
        )
        model = MagicMock()
        model.get_coefficients.return_value = coef

        if covariance is not None:
            model.model_ = MagicMock()
            model.model_.covariance_matrix_ = covariance
        else:
            # No model_ attribute -> no covariance path
            del model.model_

        return model

    def test_with_covariance_matrix(self):
        """Should compute HRs and 95% CIs from covariance matrix."""
        coef = [0.5, -0.3]
        cov = np.diag([0.01, 0.02])  # SE = sqrt(diag) = [0.1, 0.1414]
        model = self._make_cox_model(coef, ['age', 'grade'], cov)

        result = format_hazard_ratios(model)

        assert len(result) == 2
        assert list(result.columns) == ['Feature', 'HR', '95% CI']
        # HR = exp(coef): exp(0.5) ~= 1.649, exp(-0.3) ~= 0.741
        assert result['HR'].iloc[0] == pytest.approx(np.exp(0.5), abs=0.001)
        # CI should be formatted strings, not 'N/A'
        assert result['95% CI'].iloc[0] != 'N/A'
        assert '[' in result['95% CI'].iloc[0]

    def test_without_covariance_matrix(self):
        """Should produce 'N/A' CIs when covariance is unavailable."""
        model = self._make_cox_model([0.5, -0.3], ['age', 'grade'])

        result = format_hazard_ratios(model)

        assert len(result) == 2
        assert all(ci == 'N/A' for ci in result['95% CI'])
        # HR should still be correct
        assert result['HR'].iloc[0] == pytest.approx(np.exp(0.5), abs=0.001)

    def test_malformed_covariance_falls_back_to_na(self):
        """Malformed covariance should trigger fallback and return N/A CIs."""
        model = self._make_cox_model([0.5, -0.3], ['age', 'grade'], covariance=object())

        result = format_hazard_ratios(model)

        assert all(ci == 'N/A' for ci in result['95% CI'])

    def test_no_coefficients_raises(self):
        """Should raise ValueError when model has no coefficients."""
        model = MagicMock()
        model.get_coefficients.return_value = None

        with pytest.raises(ValueError, match="does not have coefficients"):
            format_hazard_ratios(model)

    def test_sorted_by_absolute_coefficient(self):
        """Features should be sorted by absolute coefficient (largest first)."""
        model = self._make_cox_model(
            [0.1, -0.8, 0.5],
            ['small', 'big_neg', 'medium'],
        )

        result = format_hazard_ratios(model)

        # big_neg (|-0.8|) should be first, then medium (|0.5|), then small (|0.1|)
        assert result['Feature'].iloc[0] == 'big_neg'
        assert result['Feature'].iloc[1] == 'medium'
        assert result['Feature'].iloc[2] == 'small'

    def test_top_n_features(self):
        """n parameter should limit number of features returned."""
        model = self._make_cox_model([0.1, -0.8, 0.5], ['a', 'b', 'c'])

        result = format_hazard_ratios(model, n=2)
        assert len(result) == 2
