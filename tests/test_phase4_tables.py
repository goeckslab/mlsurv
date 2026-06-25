"""Test results tables module for Phase 4 (Session 3)."""

import numpy as np
import pandas as pd
import pytest


class TestFormatMetricsTable:
    """Test format_metrics_table function."""

    def test_format_metrics_basic(self):
        """Verify basic formatting works."""
        from mlsurv.results.tables import format_metrics_table

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf', 'gbsa'],
            'cv_metric_mean': [0.7512345, 0.7823456, 0.7612345],
            'cv_metric_std': [0.0312345, 0.0212345, 0.0412345]
        })

        result = format_metrics_table(df, precision=3)

        assert 'model_type' in result.columns
        assert 'cv_metric_mean' in result.columns
        # Check rounding
        assert result['cv_metric_mean'].iloc[0] == 0.751

    def test_format_metrics_sort(self):
        """Verify sorting works."""
        from mlsurv.results.tables import format_metrics_table

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf', 'gbsa'],
            'cv_metric_mean': [0.75, 0.80, 0.77]
        })

        result = format_metrics_table(df, sort_by='cv_metric_mean', ascending=False)

        assert result['model_type'].iloc[0] == 'rsf'  # Highest c-index


class TestFormatImportanceTable:
    """Test format_importance_table function."""

    def test_format_importance_basic(self):
        """Verify basic formatting works."""
        from mlsurv.results.tables import format_importance_table

        np.random.seed(42)
        df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(25)],
            'perm_mean': np.random.rand(25),
            'perm_std': np.random.rand(25) * 0.1
        })

        result = format_importance_table(df, n=10)

        assert len(result) == 10
        assert 'Rank' in result.columns
        assert 'Feature' in result.columns
        assert 'Importance' in result.columns

    def test_format_importance_adds_rank(self):
        """Verify rank column is added correctly."""
        from mlsurv.results.tables import format_importance_table

        df = pd.DataFrame({
            'feature': ['A', 'B', 'C'],
            'perm_mean': [0.3, 0.1, 0.2]
        })

        result = format_importance_table(df, n=3)

        assert result['Rank'].tolist() == [1, 2, 3]
        assert result['Feature'].iloc[0] == 'A'  # Highest importance


class TestFormatBootstrapSummary:
    """Test format_bootstrap_summary function."""

    def test_format_bootstrap_basic(self):
        """Verify basic formatting works."""
        from mlsurv.results.tables import format_bootstrap_summary

        df = pd.DataFrame({
            'metric': ['harrell', 'ipcw'],
            'mean': [0.75123, 0.74567],
            'std': [0.02345, 0.02567],
            'ci_lower': [0.70567, 0.69456],
            'ci_upper': [0.79678, 0.79678]
        })
        df.attrs['ci_level'] = 0.95  # as stamped by BootstrapResult

        result = format_bootstrap_summary(df)

        assert '95% CI' in result.columns
        assert 'ci_lower' not in result.columns
        assert 'ci_upper' not in result.columns


class TestFormatHazardRatios:
    """Test format_hazard_ratios function."""

    def test_format_hr_basic(self, synthetic_data_clean):
        """Verify hazard ratio formatting works."""
        from mlsurv.results.tables import format_hazard_ratios
        from mlsurv.models import CoxPHModel

        X, y = synthetic_data_clean
        model = CoxPHModel()
        model.fit(X, y)

        result = format_hazard_ratios(model, n=5)

        assert 'Feature' in result.columns
        assert 'HR' in result.columns
        assert '95% CI' in result.columns
        assert len(result) == 5

    def test_format_hr_gates_non_hr_model(self):
        """format_hazard_ratios raises for models whose exp(coef) is not an HR."""
        from mlsurv.results.tables import format_hazard_ratios

        class PenalizedModel:
            # Stub (not MagicMock) so the flag is genuinely False, like CoxNet.
            hazard_ratio_valid = False

            def get_coefficients(self):
                return pd.Series([0.5, -0.3], index=["a", "b"])

        with pytest.raises(ValueError, match="not hazard ratios"):
            format_hazard_ratios(PenalizedModel())


class TestTableExports:
    """Test table export functions."""

    def test_to_markdown(self):
        """Verify markdown export works."""
        from mlsurv.results.tables import to_markdown

        df = pd.DataFrame({
            'Model': ['coxph', 'rsf'],
            'C-Index': [0.75, 0.78]
        })

        result = to_markdown(df)

        assert '|' in result
        assert 'coxph' in result
        assert 'rsf' in result

    def test_to_html(self):
        """Verify HTML export works."""
        from mlsurv.results.tables import to_html

        df = pd.DataFrame({
            'Model': ['coxph', 'rsf'],
            'C-Index': [0.75, 0.78]
        })

        result = to_html(df)

        assert '<table' in result
        assert 'coxph' in result

    def test_to_latex(self):
        """Verify LaTeX export works."""
        from mlsurv.results.tables import to_latex

        df = pd.DataFrame({
            'Model': ['coxph', 'rsf'],
            'C-Index': [0.75, 0.78]
        })

        result = to_latex(df, caption='Test Table')

        assert '\\begin{tabular}' in result
        assert 'coxph' in result
