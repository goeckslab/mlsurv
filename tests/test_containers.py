"""Test result container classes and their convenience methods."""

import json

import pytest
import numpy as np
import pandas as pd
import warnings
from types import SimpleNamespace

from mlsurv._warnings import InternalWarning
from mlsurv.results.containers import (
    EvaluationResult,
    BootstrapResult,
    ModelResult,
    ValidationResult,
    extract_model_evaluation,
)
from mlsurv.results._containers_helpers import DictExportMixin, _df_to_records


class TestEvaluationResultProperties:
    """Test EvaluationResult scalar properties."""

    @pytest.fixture
    def eval_result(self):
        """Create a basic EvaluationResult for testing."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_train': [0.85],
            'c_index_test': [0.72]
        })
        test_set = pd.DataFrame({
            'model_type': ['rsf'] * 5,
            'index': range(5),
            'risk_score': [0.1, 0.2, 0.3, 0.4, 0.5],
            'time': [1.0, 2.0, 3.0, 4.0, 5.0],
            'event': [True, False, True, True, False]
        })
        auc_t = pd.DataFrame({
            'model_type': ['rsf'] * 3,
            'eval_time': [1.0, 2.0, 3.0],
            'auc': [0.70, 0.72, 0.68]
        })
        importance = pd.DataFrame({
            'model_type': ['rsf'] * 2,
            'feature': ['age', 'stage'],
            'coefficient': [0.5, 0.3],
            'perm_mean': [0.02, 0.01],
            'perm_std': [0.005, 0.003]
        })
        importance_iterations = pd.DataFrame({
            'model_type': ['rsf'] * 4,
            'feature': ['age', 'age', 'stage', 'stage'],
            'iteration': [0, 1, 0, 1],
            'importance': [0.02, 0.025, 0.01, 0.012]
        })

        return EvaluationResult(
            concordance=concordance,
            test_set=test_set,
            auc_t=auc_t,
            importance=importance,
            importance_iterations=importance_iterations
        )

    def test_c_index_test_returns_float(self, eval_result):
        """c_index_test should return float value."""
        assert eval_result.c_index_test == pytest.approx(0.72)
        assert isinstance(eval_result.c_index_test, float)

    def test_c_index_train_returns_float(self, eval_result):
        """c_index_train should return float value."""
        assert eval_result.c_index_train == pytest.approx(0.85)
        assert isinstance(eval_result.c_index_train, float)

    def test_c_index_test_none_when_missing_column(self):
        """c_index_test should return None when column missing."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_train': [0.85]
        })
        result = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame()
        )
        assert result.c_index_test is None

    def test_ibs_test_returns_none_without_bootstrap(self, eval_result):
        """ibs_test should return None when no bootstrap."""
        assert eval_result.ibs_test is None

    def test_ibs_test_with_bootstrap(self, eval_result):
        """ibs_test should return value from bootstrap summary."""
        bootstrap_summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf'],
            'metric': ['harrell', 'integrated_brier'],
            'mean': [0.72, 0.18],
            'std': [0.02, 0.01],
            'ci_lower': [0.68, 0.16],
            'ci_upper': [0.76, 0.20]
        })
        eval_result.bootstrap = BootstrapResult(
            summary=bootstrap_summary,
            iterations=pd.DataFrame()
        )
        assert eval_result.ibs_test == pytest.approx(0.18)


class TestBootstrapResultProperties:
    """Test BootstrapResult scalar properties."""

    @pytest.fixture
    def bootstrap_result(self):
        """Create a basic BootstrapResult for testing."""
        summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf', 'rsf'],
            'metric': ['harrell', 'ipcw', 'integrated_brier'],
            'point': [0.715, 0.695, 0.175],
            'mean': [0.72, 0.70, 0.18],
            'std': [0.025, 0.030, 0.015],
            'ci_lower': [0.67, 0.64, 0.15],
            'ci_upper': [0.77, 0.76, 0.21]
        })
        iterations = pd.DataFrame({
            'model_type': ['rsf'] * 6,
            'iteration': [0, 0, 0, 1, 1, 1],
            'metric_name': ['harrell', 'ipcw', 'integrated_brier'] * 2,
            'value': [0.71, 0.69, 0.17, 0.73, 0.71, 0.19]
        })
        return BootstrapResult(summary=summary, iterations=iterations)

    def test_summary_harrell_mean(self, bootstrap_result):
        """Summary should contain harrell mean value."""
        mask = bootstrap_result.summary['metric'] == 'harrell'
        mean_val = float(bootstrap_result.summary.loc[mask, 'mean'].iloc[0])
        assert mean_val == pytest.approx(0.72)
        assert isinstance(mean_val, float)

    def test_compute_ci_harrell(self, bootstrap_result):
        """compute_ci('harrell') should return (lower, upper) tuple with correct values."""
        ci = bootstrap_result.compute_ci('harrell')
        assert ci is not None
        assert len(ci) == 2
        assert ci[0] == pytest.approx(0.7105, abs=1e-4)
        assert ci[1] == pytest.approx(0.7295, abs=1e-4)


class TestModelResultGetMetric:
    """Test ModelResult.get_metric() method."""

    @pytest.fixture
    def model_result_full(self):
        """Create a ModelResult with all phases populated."""
        # CV metrics
        cv_metrics = {
            'harrell': {'scores': np.array([0.71, 0.73, 0.72, 0.70, 0.74]), 'mean': 0.72, 'std': 0.015},
            'ipcw': {'scores': np.array([0.68, 0.70, 0.69, 0.67, 0.71]), 'mean': 0.69, 'std': 0.016},
            'integrated_brier': {'scores': np.array([0.15, 0.14, 0.16, 0.15, 0.14]), 'mean': 0.148, 'std': 0.008}
        }

        # Evaluation result
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_train': [0.85],
            'c_index_test': [0.71]
        })
        evaluation = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame()
        )

        # Bootstrap result
        bootstrap_summary = pd.DataFrame({
            'model_type': ['rsf', 'rsf', 'rsf'],
            'metric': ['harrell', 'ipcw', 'integrated_brier'],
            'point': [0.715, 0.690, 0.142],
            'mean': [0.712, 0.685, 0.145],
            'std': [0.025, 0.030, 0.010],
            'ci_lower': [0.665, 0.630, 0.126],
            'ci_upper': [0.758, 0.740, 0.164]
        })
        bootstrap_iterations = pd.DataFrame({
            'model_type': ['rsf'] * 6,
            'iteration': [0, 0, 0, 1, 1, 1],
            'metric_name': ['harrell', 'ipcw', 'integrated_brier'] * 2,
            'value': [0.70, 0.68, 0.14, 0.72, 0.69, 0.15]
        })
        bootstrap = BootstrapResult(
            summary=bootstrap_summary,
            iterations=bootstrap_iterations
        )

        return ModelResult(
            name='rsf',
            model=None,  # Mock
            cv_metric_scores=np.array([0.71, 0.73, 0.72, 0.70, 0.74]),
            cv_metric_mean=0.72,
            cv_metric_std=0.015,
            cv_metrics=cv_metrics,
            evaluation=evaluation,
            bootstrap=bootstrap
        )

    def test_get_metric_cv_mean(self, model_result_full):
        """get_metric should return CV mean."""
        result = model_result_full.get_metric('harrell', 'cv', 'mean')
        assert result == pytest.approx(0.72)

    def test_get_metric_cv_std(self, model_result_full):
        """get_metric should return CV std."""
        result = model_result_full.get_metric('harrell', 'cv', 'std')
        assert result == pytest.approx(0.015)

    def test_get_metric_cv_scores(self, model_result_full):
        """get_metric should return CV scores array."""
        result = model_result_full.get_metric('harrell', 'cv', 'scores')
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_get_metric_test_mean(self, model_result_full):
        """get_metric should return test set value."""
        result = model_result_full.get_metric('harrell', 'test', 'mean')
        assert result == pytest.approx(0.71)

    def test_get_metric_bootstrap_mean(self, model_result_full):
        """get_metric should return bootstrap mean."""
        result = model_result_full.get_metric('harrell', 'bootstrap', 'mean')
        assert result == pytest.approx(0.712)

    def test_get_metric_bootstrap_ci_lower(self, model_result_full):
        """get_metric should return bootstrap ci_lower."""
        result = model_result_full.get_metric('harrell', 'bootstrap', 'ci_lower')
        assert result == pytest.approx(0.665)

    def test_get_metric_bootstrap_ci_upper(self, model_result_full):
        """get_metric should return bootstrap ci_upper."""
        result = model_result_full.get_metric('harrell', 'bootstrap', 'ci_upper')
        assert result == pytest.approx(0.758)

    def test_get_metric_bootstrap_point(self, model_result_full):
        """get_metric should return bootstrap point estimate."""
        result = model_result_full.get_metric('harrell', 'bootstrap', 'point')
        assert result == pytest.approx(0.715)

    def test_get_metric_different_metrics(self, model_result_full):
        """get_metric should work for all metric types."""
        assert model_result_full.get_metric('ipcw', 'cv', 'mean') == pytest.approx(0.69)
        assert model_result_full.get_metric('integrated_brier', 'cv', 'mean') == pytest.approx(0.148)
        assert model_result_full.get_metric('integrated_brier', 'bootstrap', 'mean') == pytest.approx(0.145)

    def test_get_metric_unknown_phase_warns(self, model_result_full):
        """get_metric should warn for unknown phase."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model_result_full.get_metric('harrell', 'unknown', 'mean')
            assert result is None
            assert len(w) == 1
            assert "Unknown phase" in str(w[0].message)

    def test_get_metric_test_unrecognized_metric_warns(self, model_result_full):
        """get_metric should warn with specific message for unrecognized metric names."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model_result_full.get_metric('harell', 'test', 'mean')
            assert result is None
            assert len(w) == 1
            assert "not a recognized metric name" in str(w[0].message)

    def test_get_metric_missing_evaluation_warns(self):
        """get_metric should warn when evaluation not available."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.0,
            evaluation=None
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model_result.get_metric('harrell', 'test', 'mean')
            assert result is None
            assert len(w) == 1
            assert "evaluation" in str(w[0].message).lower()

    def test_get_metric_missing_bootstrap_warns(self):
        """get_metric should warn when bootstrap not available."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.0,
            bootstrap=None
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model_result.get_metric('harrell', 'bootstrap', 'mean')
            assert result is None
            assert len(w) == 1
            assert "bootstrap" in str(w[0].message).lower()

    def test_get_metric_fallback_to_primary(self):
        """get_metric should fallback to cv_metric_* when cv_metrics missing."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.71, 0.73, 0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            cv_metrics=None  # No multi-metric CV
        )
        assert model_result.get_metric('harrell', 'cv', 'mean') == pytest.approx(0.72)
        assert model_result.get_metric('harrell', 'cv', 'std') == pytest.approx(0.01)


class TestModelResultCvResultsDf:
    """Test ModelResult.cv_results_df() method."""

    def test_cv_results_df_with_cv_metrics(self):
        """cv_results_df should return DataFrame with all metrics."""
        cv_metrics = {
            'harrell': {'scores': np.array([0.71, 0.73, 0.72]), 'mean': 0.72, 'std': 0.01},
            'ipcw': {'scores': np.array([0.68, 0.70, 0.69]), 'mean': 0.69, 'std': 0.01}
        }
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.71, 0.73, 0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            cv_metrics=cv_metrics
        )

        df = model_result.cv_results_df()
        assert 'fold' in df.columns
        assert 'harrell' in df.columns
        assert 'ipcw' in df.columns
        assert len(df) == 3

    def test_cv_results_df_fallback(self):
        """cv_results_df should fallback to primary metric when cv_metrics missing."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.71, 0.73, 0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            cv_metrics=None
        )

        df = model_result.cv_results_df()
        assert 'fold' in df.columns
        assert 'harrell' in df.columns
        assert len(df) == 3

    def test_cv_results_df_mismatched_lengths_pads_with_nan(self):
        """cv_results_df should pad shorter metric arrays with NaN."""
        cv_metrics = {
            'harrell': {
                'scores': np.array([0.71, 0.73, 0.72, 0.70, 0.74]),
                'mean': 0.72, 'std': 0.015,
            },
            'ipcw': {
                'scores': np.array([0.68, 0.70, 0.69, 0.67, 0.71]),
                'mean': 0.69, 'std': 0.015,
            },
            'integrated_brier': {
                'scores': np.array([0.20, 0.22, 0.21]),  # Only 3 of 5
                'mean': 0.21, 'std': 0.01,
            },
        }
        model_result = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.71, 0.73, 0.72, 0.70, 0.74]),
            cv_metric_mean=0.72, cv_metric_std=0.015,
            cv_metrics=cv_metrics,
        )

        with pytest.warns(InternalWarning, match="padding with NaN"):
            df = model_result.cv_results_df()

        assert len(df) == 5
        assert 'integrated_brier' in df.columns
        assert df['integrated_brier'].isna().sum() == 2
        assert not df['harrell'].isna().any()


class TestModelResultSummary:
    """Test ModelResult.summary() method."""

    @pytest.fixture
    def model_result_with_all(self):
        """Create a ModelResult with all phases for summary testing."""
        cv_metrics = {
            'harrell': {'scores': np.array([0.72]), 'mean': 0.72, 'std': 0.01},
        }
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_train': [0.85],
            'c_index_test': [0.71]
        })
        evaluation = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame()
        )
        bootstrap_summary = pd.DataFrame({
            'model_type': ['rsf'],
            'metric': ['harrell'],
            'mean': [0.712],
            'std': [0.025],
            'ci_lower': [0.665],
            'ci_upper': [0.758]
        })
        bootstrap = BootstrapResult(
            summary=bootstrap_summary,
            iterations=pd.DataFrame()
        )

        return ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            cv_metrics=cv_metrics,
            evaluation=evaluation,
            bootstrap=bootstrap
        )

    def test_summary_returns_dataframe(self, model_result_with_all):
        """summary should return DataFrame."""
        df = model_result_with_all.summary()
        assert isinstance(df, pd.DataFrame)
        assert 'metric' in df.columns
        assert 'phase' in df.columns
        assert 'mean' in df.columns
        assert 'std' in df.columns
        assert 'ci_lower' in df.columns
        assert 'ci_upper' in df.columns

    def test_summary_includes_all_phases(self, model_result_with_all):
        """summary should include cv and test phases (bootstrap merged into test)."""
        df = model_result_with_all.summary(metrics=['harrell'])
        phases = df['phase'].unique()
        assert 'cv' in phases
        assert 'test' in phases
        assert 'bootstrap' not in phases
        # Test row: point from test set, CI from bootstrap
        test_row = df[df['phase'] == 'test'].iloc[0]
        assert test_row['point'] == pytest.approx(0.71)
        assert test_row['mean'] == pytest.approx(0.712)
        assert test_row['std'] == pytest.approx(0.025)
        assert test_row['ci_lower'] == pytest.approx(0.665)
        assert test_row['ci_upper'] == pytest.approx(0.758)

    def test_summary_custom_metrics(self, model_result_with_all):
        """summary should accept custom metrics list."""
        df = model_result_with_all.summary(metrics=['harrell'])
        assert (df['metric'] == 'harrell').all()

    def test_summary_only_cv_when_minimal_data(self):
        """summary should only include CV when no evaluation/bootstrap."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            cv_metrics={'harrell': {'scores': np.array([0.72]), 'mean': 0.72, 'std': 0.01}}
        )
        df = model_result.summary(metrics=['harrell'])
        # Should only have CV phase since no evaluation or bootstrap
        assert len(df) == 1
        assert df.iloc[0]['phase'] == 'cv'


class TestResolveBootstrap:
    """Test ModelResult.resolved_bootstrap property resolution order."""

    def test_prefers_direct_bootstrap(self):
        """resolved_bootstrap should prefer ModelResult.bootstrap over evaluation.bootstrap."""
        direct_boot = BootstrapResult(
            summary=pd.DataFrame({'metric': ['harrell'], 'mean': [0.71],
                                  'std': [0.02], 'ci_lower': [0.67], 'ci_upper': [0.75]}),
            iterations=pd.DataFrame()
        )
        eval_boot = BootstrapResult(
            summary=pd.DataFrame({'metric': ['harrell'], 'mean': [0.60],
                                  'std': [0.03], 'ci_lower': [0.54], 'ci_upper': [0.66]}),
            iterations=pd.DataFrame()
        )
        concordance = pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.71]})
        evaluation = EvaluationResult(
            concordance=concordance, test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(), importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(), bootstrap=eval_boot
        )
        mr = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.72]), cv_metric_mean=0.72, cv_metric_std=0.01,
            evaluation=evaluation, bootstrap=direct_boot
        )
        assert mr.resolved_bootstrap is direct_boot

    def test_falls_back_to_evaluation_bootstrap(self):
        """_resolve_bootstrap should fall back to evaluation.bootstrap when direct is None."""
        eval_boot = BootstrapResult(
            summary=pd.DataFrame({'metric': ['harrell'], 'mean': [0.71],
                                  'std': [0.02], 'ci_lower': [0.67], 'ci_upper': [0.75]}),
            iterations=pd.DataFrame()
        )
        concordance = pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.71]})
        evaluation = EvaluationResult(
            concordance=concordance, test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(), importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(), bootstrap=eval_boot
        )
        mr = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.72]), cv_metric_mean=0.72, cv_metric_std=0.01,
            evaluation=evaluation, bootstrap=None
        )
        assert mr.resolved_bootstrap is eval_boot

    def test_returns_none_when_both_absent(self):
        """_resolve_bootstrap should return None when no bootstrap is available."""
        mr = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.72]), cv_metric_mean=0.72, cv_metric_std=0.01,
            evaluation=None, bootstrap=None
        )
        assert mr.resolved_bootstrap is None

    def test_summary_test_row_without_bootstrap(self):
        """Test row should have point but NaN CI when no bootstrap exists."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'], 'c_index_test': [0.71]
        })
        evaluation = EvaluationResult(
            concordance=concordance, test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(), importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame()
        )
        mr = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.72]), cv_metric_mean=0.72, cv_metric_std=0.01,
            cv_metrics={'harrell': {'scores': np.array([0.72]), 'mean': 0.72, 'std': 0.01}},
            evaluation=evaluation, bootstrap=None
        )
        df = mr.summary(metrics=['harrell'])
        test_row = df[df['phase'] == 'test'].iloc[0]
        assert test_row['point'] == pytest.approx(0.71)
        assert pd.isna(test_row['mean'])
        assert pd.isna(test_row['std'])
        assert pd.isna(test_row['ci_lower'])
        assert pd.isna(test_row['ci_upper'])


class TestExistingFunctionality:
    """Test that existing functionality still works."""

    def test_bootstrap_result_get_distribution(self):
        """Existing get_distribution should still work."""
        iterations = pd.DataFrame({
            'model_type': ['rsf'] * 4,
            'iteration': [0, 0, 1, 1],
            'metric_name': ['harrell', 'ipcw', 'harrell', 'ipcw'],
            'value': [0.71, 0.68, 0.73, 0.70]
        })
        result = BootstrapResult(
            summary=pd.DataFrame(),
            iterations=iterations
        )
        dist = result.get_distribution('harrell')
        assert len(dist) == 2
        assert 0.71 in dist
        assert 0.73 in dist

    def test_bootstrap_result_compute_ci(self):
        """Existing compute_ci should still work."""
        iterations = pd.DataFrame({
            'model_type': ['rsf'] * 100,
            'iteration': list(range(100)),
            'metric_name': ['harrell'] * 100,
            'value': np.random.normal(0.72, 0.02, 100)
        })
        result = BootstrapResult(
            summary=pd.DataFrame(),
            iterations=iterations
        )
        lower, upper = result.compute_ci('harrell', ci=0.95, method='percentile')
        assert lower < upper
        assert 0.65 < lower < 0.75
        assert 0.70 < upper < 0.80

    def test_model_result_existing_properties(self):
        """Existing ModelResult properties should still work."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_train': [0.85],
            'c_index_test': [0.71]
        })
        evaluation = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame({'col': [1, 2, 3]}),
            auc_t=pd.DataFrame({'col': [1]}),
            importance=pd.DataFrame({'col': [1]}),
            importance_iterations=pd.DataFrame()
        )

        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            evaluation=evaluation
        )

        # Access evaluation properties through .evaluation
        assert model_result.evaluation.c_index_train == 0.85
        assert model_result.evaluation.c_index_test == 0.71
        assert model_result.evaluation.test_set is not None
        assert model_result.evaluation.auc_t is not None
        assert model_result.evaluation.importance is not None

    def test_model_result_nan_c_index_returns_none(self):
        """NaN c-index values should be returned as None, not nan."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_train': [float('nan')],
            'c_index_test': [float('nan')]
        })
        evaluation = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame({'col': [1, 2, 3]}),
            auc_t=pd.DataFrame({'col': [1]}),
            importance=pd.DataFrame({'col': [1]}),
            importance_iterations=pd.DataFrame()
        )

        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            evaluation=evaluation
        )

        # Properties must return None, not nan
        assert model_result.evaluation.c_index_train is None
        assert model_result.evaluation.c_index_test is None

        # to_dict() must serialize as None, not nan
        d = model_result.to_dict()
        assert d['c_index_train'] is None
        assert d['c_index_test'] is None

    def test_to_dict_schema(self):
        """to_dict() should include expected keys and exclude bootstrap_ci."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_train': [0.85],
            'c_index_test': [0.71]
        })
        evaluation = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame()
        )
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            evaluation=evaluation
        )
        d = model_result.to_dict()
        expected_keys = {'name', 'cv_metric_mean', 'cv_metric_std',
                         'cv_ci_lower', 'cv_ci_upper', 'cv_ci_level',
                         'c_index_train', 'c_index_test'}
        assert expected_keys <= set(d.keys())
        assert 'bootstrap_ci' not in d
        assert d['c_index_train'] == pytest.approx(0.85)
        assert d['c_index_test'] == pytest.approx(0.71)

    def test_to_dict_json_safe_nan(self):
        """to_dict() should convert NaN values to None for JSON compatibility."""
        import json
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=np.nan,
            cv_metric_std=np.nan,
            cv_ci_lower=np.nan,
            cv_ci_upper=np.nan,
        )
        d = model_result.to_dict()
        assert d['cv_metric_mean'] is None
        assert d['cv_ci_lower'] is None
        assert d['cv_ci_upper'] is None
        # Must not raise ValueError for NaN
        json_str = json.dumps(d)
        assert 'NaN' not in json_str

    def test_to_dict_cv_metrics_includes_ci(self):
        """to_dict() should include CI keys in per-metric cv_metrics."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            cv_metrics={
                'harrell': {
                    'mean': 0.72, 'std': 0.01,
                    'scores': np.array([0.72]),
                    'ci_lower': 0.70, 'ci_upper': 0.74,
                }
            },
        )
        d = model_result.to_dict()
        assert 'ci_lower' in d['cv_metrics']['harrell']
        assert d['cv_metrics']['harrell']['ci_lower'] == pytest.approx(0.70)
        assert d['cv_metrics']['harrell']['ci_upper'] == pytest.approx(0.74)

    def test_to_dict_prefers_best_params_over_optuna(self):
        """to_dict() should use best_params_ when both best_params_ and optuna_study exist."""
        mock_study = SimpleNamespace(best_params={'n_estimators': 100})
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            optuna_study=mock_study,
            best_params_={'model__n_estimators': 200},
        )
        d = model_result.to_dict()
        assert d['best_params'] == {'model__n_estimators': 200}

    def test_to_dict_falls_back_to_optuna_study(self):
        """to_dict() should fall back to optuna_study.best_params when best_params_ is None."""
        mock_study = SimpleNamespace(best_params={'n_estimators': 100})
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            optuna_study=mock_study,
        )
        d = model_result.to_dict()
        assert d['best_params'] == {'n_estimators': 100}

    def test_to_dict_no_best_params_when_both_none(self):
        """to_dict() should omit best_params when both best_params_ and optuna_study are None."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
        )
        d = model_result.to_dict()
        assert 'best_params' not in d

    def test_to_dict_empty_best_params_included(self):
        """to_dict() should include best_params even when it is an empty dict."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            best_params_={},
        )
        d = model_result.to_dict()
        assert d['best_params'] == {}

    def test_to_dict_optuna_no_completed_trials(self):
        """to_dict() should not crash when optuna_study has no completed trials."""
        class _NoTrialsStudy:
            @property
            def best_params(self):
                raise ValueError("Study has no completed trials.")

        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            optuna_study=_NoTrialsStudy(),
        )
        d = model_result.to_dict()
        assert 'best_params' not in d


class TestToDict:
    """Uniform to_dict() / to_dict(full=True) JSON export across containers.

    All fixtures are built from tiny in-memory DataFrames (no training), so
    these run in the fast tier with no marker.
    """

    @staticmethod
    def _assert_json_safe(obj):
        """Round-trip through json and confirm no NaN/Infinity leaked."""
        text = json.dumps(obj)
        assert 'NaN' not in text
        assert 'Infinity' not in text
        json.loads(text)  # parses back without error
        return text

    # -- fixtures -----------------------------------------------------------

    @pytest.fixture
    def eval_result(self):
        """Single-model EvaluationResult with NaN and inf cells."""
        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_train': [0.85],
            'c_index_test': [0.72],
            'integrated_brier_test': [np.nan],  # NaN must serialize to null
        })
        test_set = pd.DataFrame({
            'model_type': ['rsf'] * 2, 'index': [0, 1],
            'risk_score': [0.1, np.inf],  # inf must serialize to null
            'time': [1.0, 2.0], 'event': [True, False],
        })
        auc_t = pd.DataFrame({
            'model_type': ['rsf'] * 2, 'eval_time': [1.0, 2.0], 'auc': [0.7, 0.68],
        })
        return EvaluationResult(
            concordance=concordance, test_set=test_set, auc_t=auc_t,
            importance=pd.DataFrame(), importance_iterations=pd.DataFrame(),
        )

    @pytest.fixture
    def bootstrap_result(self):
        summary = pd.DataFrame({
            'model_type': ['rsf'], 'metric': ['harrell'], 'point': [0.71],
            'mean': [0.72], 'std': [0.02], 'ci_lower': [0.68], 'ci_upper': [0.76],
        })
        iterations = pd.DataFrame({
            'model_type': ['rsf', 'rsf'], 'iteration': [0, 1],
            'metric_name': ['harrell', 'harrell'], 'value': [0.71, 0.73],
        })
        return BootstrapResult(summary=summary, iterations=iterations)

    # -- EvaluationResult ---------------------------------------------------

    def test_eval_summary_keys(self, eval_result):
        d = eval_result.to_dict()
        assert d['n_models'] == 1
        assert d['models'] == ['rsf']
        assert d['c_index_test'] == pytest.approx(0.72)
        assert 'tables' not in d  # summary mode omits tables
        self._assert_json_safe(d)

    def test_eval_full_tables(self, eval_result):
        d = eval_result.to_dict(full=True)
        assert 'tables' in d
        # concordance is exported under the 'metrics' key
        assert 'metrics' in d['tables']
        assert isinstance(d['tables']['metrics'], list)
        assert isinstance(d['tables']['metrics'][0], dict)
        # NaN cell became null
        assert d['tables']['metrics'][0]['integrated_brier_test'] is None
        self._assert_json_safe(d)

    def test_eval_multi_model_omits_scalar_metrics_without_warning(self):
        """Multi-model summary lists models but skips warning-prone scalars."""
        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_train': [0.85, 0.80], 'c_index_test': [0.72, 0.70],
        })
        ev = EvaluationResult(
            concordance=concordance, test_set=pd.DataFrame(), auc_t=pd.DataFrame(),
            importance=pd.DataFrame(), importance_iterations=pd.DataFrame(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any warning fails the test
            d = ev.to_dict()
        assert d['n_models'] == 2
        assert set(d['models']) == {'rsf', 'coxph'}
        assert 'c_index_test' not in d
        # Full dump still carries both models' rows under tables['metrics'].
        full = ev.to_dict(full=True)
        assert len(full['tables']['metrics']) == 2

    # -- BootstrapResult ----------------------------------------------------

    def test_bootstrap_summary_and_full(self, bootstrap_result):
        d = bootstrap_result.to_dict()
        assert d['n_models'] == 1
        assert d['n_bootstrap'] == 2
        assert d['ci_level'] == 0.95
        assert d['metrics'] == ['harrell']
        assert 'tables' not in d
        full = bootstrap_result.to_dict(full=True)
        assert 'summary' in full['tables'] and 'iterations' in full['tables']
        self._assert_json_safe(full)

    # -- ModelResult (backward compatibility + full) ------------------------

    def test_model_result_summary_matches_legacy(self, eval_result):
        """to_dict(full=False) returns the historical scalar payload."""
        mr = ModelResult(
            name='rsf', model=None, cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72, cv_metric_std=0.01, evaluation=eval_result,
        )
        d = mr.to_dict()
        assert {'name', 'cv_metric_mean', 'cv_metric_std', 'cv_ci_lower',
                'cv_ci_upper', 'cv_ci_level', 'c_index_train', 'c_index_test'} <= set(d)
        assert 'tables' not in d
        self._assert_json_safe(d)

    def test_model_result_full_adds_tables(self, eval_result):
        mr = ModelResult(
            name='rsf', model=None, cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72, cv_metric_std=0.01, evaluation=eval_result,
        )
        full = mr.to_dict(full=True)
        assert 'tables' in full
        # ModelResult.to_dataframes() prefixes evaluation tables with 'eval_'.
        assert any(k.startswith('eval_') for k in full['tables'])
        self._assert_json_safe(full)

    # -- ValidationResult ---------------------------------------------------

    def test_validation_result(self, eval_result):
        vr = ValidationResult(name='TCGA', evaluation=eval_result,
                              n_samples=10, n_events=4)
        d = vr.to_dict()
        assert d['name'] == 'TCGA'
        assert d['n_samples'] == 10
        assert d['event_rate'] == pytest.approx(0.4)
        assert d['c_index_test'] == pytest.approx(0.72)
        full = vr.to_dict(full=True)
        assert 'cohort_info' in full['tables']
        self._assert_json_safe(full)

    # -- BenchmarkResult ----------------------------------------------------

    def test_benchmark_result(self, eval_result):
        from mlsurv.results.containers import BenchmarkResult
        bench = BenchmarkResult(
            evaluation=eval_result,
            univariate_cox=pd.DataFrame({'reference': ['tmb'], 'hazard_ratio': [1.2],
                                         'p_value': [0.03]}),
            incremental=pd.DataFrame(),
            correlation=pd.DataFrame(),
            reference_names=['tmb'], model_names=['rsf'],
        )
        d = bench.to_dict()
        assert d['model_names'] == ['rsf']
        assert d['reference_names'] == ['tmb']
        assert d['dataset'] == 'test'
        full = bench.to_dict(full=True)
        assert 'univariate_cox' in full['tables']
        self._assert_json_safe(full)

    # -- SweepResult --------------------------------------------------------

    def test_sweep_result(self):
        from mlsurv.results.containers import SweepResult
        data = pd.DataFrame({
            'param_value': [1, 1, 2, 2], 'model': ['rsf', 'coxph', 'rsf', 'coxph'],
            'metric': ['harrell'] * 4, 'mean': [0.7, 0.68, 0.72, 0.69],
            'std': [0.02] * 4, 'ci_lower': [0.66] * 4, 'ci_upper': [0.74] * 4,
        })
        sweep = SweepResult(data=data, param_name='k', target='feature_selector')
        d = sweep.to_dict()
        assert d['param_name'] == 'k'
        assert d['n_values'] == 2
        assert d['models'] == ['coxph', 'rsf']
        full = sweep.to_dict(full=True)
        assert 'sweep' in full['tables']
        self._assert_json_safe(full)

    def test_sweep_result_empty_data(self):
        from mlsurv.results.containers import SweepResult
        sweep = SweepResult(data=pd.DataFrame(), param_name='k', target='model')
        d = sweep.to_dict()
        assert d['n_values'] == 0
        assert d['models'] == []
        full = sweep.to_dict(full=True)
        assert full['tables'] == {}  # nothing to export

    # -- PredictionResult ---------------------------------------------------

    def test_prediction_result_preserves_time_index(self):
        from mlsurv.results import PredictionResult
        pred = PredictionResult(
            risk_scores=np.array([0.3, 0.7]), model_name='rsf', n_patients=2,
            _time_grid=np.array([1.0, 2.0, 3.0]),
            _surv_probs=np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]),
        )
        d = pred.to_dict()
        assert d['model_name'] == 'rsf'
        assert d['n_patients'] == 2
        assert d['has_survival_function'] is True
        full = pred.to_dict(full=True)
        assert 'survival_function' in full['tables']
        # The time index must survive as a 'time' column (records drop indexes).
        first_row = full['tables']['survival_function'][0]
        assert 'time' in first_row
        self._assert_json_safe(full)

    def test_prediction_result_no_survival_function(self):
        """Risk-score-only prediction: to_dict works, survival table absent."""
        from mlsurv.results import PredictionResult
        pred = PredictionResult(
            risk_scores=np.array([0.3, 0.7]), model_name='rsf', n_patients=2,
        )
        d = pred.to_dict()
        assert d['has_survival_function'] is False
        full = pred.to_dict(full=True)
        assert 'survival_function' not in full['tables']
        # risk_scores carries a named 'patient' RangeIndex -> promoted to a column.
        assert 'patient' in full['tables']['risk_scores'][0]
        self._assert_json_safe(full)

    # -- _df_to_records helper (direct unit coverage) -----------------------

    def test_df_to_records_neutralizes_nan_and_inf(self):
        """Both NaN and inf table cells must become null in the records."""
        df = pd.DataFrame({'a': [1.0, np.nan], 'b': [np.inf, -np.inf]})
        records = _df_to_records(df)
        assert records == [{'a': 1.0, 'b': None}, {'a': None, 'b': None}]
        json.dumps(records)  # the exact to_dict()-vs-to_json regression guard

    def test_df_to_records_empty_and_none(self):
        assert _df_to_records(None) == []
        assert _df_to_records(pd.DataFrame()) == []

    def test_df_to_records_drops_unnamed_filtered_index(self):
        """A row-filtered (non-default) unnamed index must not leak an 'index' key."""
        df = pd.DataFrame({'metric': ['a', 'drop', 'b'], 'v': [1, 2, 3]})
        filtered = df[df['metric'] != 'drop']  # index is no longer contiguous 0..n
        records = _df_to_records(filtered)
        assert all('index' not in r for r in records)
        assert records == [{'metric': 'a', 'v': 1}, {'metric': 'b', 'v': 3}]

    def test_df_to_records_named_index_collision_keeps_column(self):
        """When the index name duplicates a column, keep the column, drop the index."""
        df = pd.DataFrame({'time': [10, 20]})
        df.index = pd.Index([0, 1], name='time')  # name collides with column
        records = _df_to_records(df)  # must not raise
        assert records == [{'time': 10}, {'time': 20}]

    # -- mixin contract -----------------------------------------------------

    def test_all_mixin_subclasses_define_to_dataframes(self):
        """Every DictExportMixin subclass must supply its own to_dataframes()."""
        # Import the modules so all containers are registered as subclasses.
        import mlsurv.results.containers  # noqa: F401
        import mlsurv.results._prediction  # noqa: F401
        subclasses = DictExportMixin.__subclasses__()
        assert len(subclasses) >= 7
        for cls in subclasses:
            assert 'to_dataframes' in cls.__dict__, (
                f"{cls.__name__} mixes in DictExportMixin but does not define "
                "to_dataframes()"
            )


def test_evaluation_result_repr_single_model():
    """Test __repr__ for single model with metrics."""
    concordance = pd.DataFrame({
        'model_type': ['rsf'],
        'c_index_train': [0.85],
        'c_index_test': [0.78]
    })
    result = EvaluationResult(
        concordance=concordance,
        test_set=pd.DataFrame(),
        auc_t=pd.DataFrame(),
        importance=pd.DataFrame(),
        importance_iterations=pd.DataFrame()
    )
    repr_str = repr(result)
    
    assert "EvaluationResult" in repr_str
    assert "rsf" in repr_str
    assert "c_index_test" in repr_str
    assert "c_index_train" in repr_str


def test_evaluation_result_repr_no_models():
    """Test __repr__ handles empty result."""
    result = EvaluationResult(
        concordance=pd.DataFrame(),
        test_set=pd.DataFrame(),
        auc_t=pd.DataFrame(),
        importance=pd.DataFrame(),
        importance_iterations=pd.DataFrame()
    )
    repr_str = repr(result)
    
    assert "EvaluationResult" in repr_str
    assert "no models" in repr_str


def test_evaluation_result_repr_none_concordance():
    """Test __repr__ handles None concordance without crashing."""
    result = EvaluationResult(
        concordance=None,
        test_set=pd.DataFrame(),
        auc_t=pd.DataFrame(),
        importance=pd.DataFrame(),
        importance_iterations=pd.DataFrame()
    )
    repr_str = repr(result)

    assert "EvaluationResult" in repr_str
    assert "no models" in repr_str


def test_evaluation_result_repr_missing_metrics():
    """Test __repr__ handles None/NaN metrics without crashing."""
    concordance = pd.DataFrame({
        'model_type': ['test_model'],
        'c_index_train': [None],
        'c_index_test': [np.nan]
    })
    result = EvaluationResult(
        concordance=concordance,
        test_set=pd.DataFrame(),
        auc_t=pd.DataFrame(),
        importance=pd.DataFrame(),
        importance_iterations=pd.DataFrame()
    )
    
    # Should not raise
    repr_str = repr(result)
    assert "EvaluationResult" in repr_str
    assert "N/A" in repr_str


def test_evaluation_result_repr_none_c_index_test():
    """Test __repr__ handles None c_index_test without crashing."""
    concordance = pd.DataFrame({
        'model_type': ['test_model'],
        'c_index_train': [0.81],
        'c_index_test': [None]
    })
    result = EvaluationResult(
        concordance=concordance,
        test_set=pd.DataFrame(),
        auc_t=pd.DataFrame(),
        importance=pd.DataFrame(),
        importance_iterations=pd.DataFrame()
    )

    repr_str = repr(result)
    assert "EvaluationResult" in repr_str
    assert "c_index_test=N/A" in repr_str


def test_evaluation_result_repr_multiple_models():
    """Test __repr__ for multi-model results."""
    concordance = pd.DataFrame({
        'model_type': ['rsf', 'coxph', 'gbsa'],
        'c_index_train': [0.85, 0.82, 0.87],
        'c_index_test': [0.78, 0.75, 0.80]
    })
    result = EvaluationResult(
        concordance=concordance,
        test_set=pd.DataFrame(),
        auc_t=pd.DataFrame(),
        importance=pd.DataFrame(),
        importance_iterations=pd.DataFrame()
    )
    repr_str = repr(result)
    
    assert "EvaluationResult" in repr_str
    assert "3 models" in repr_str
    assert "rsf" in repr_str


def test_bootstrap_result_repr():
    """Test __repr__ for BootstrapResult."""
    summary = pd.DataFrame({
        'model_type': ['rsf', 'rsf'],
        'metric': ['harrell', 'ipcw'],
        'mean': [0.75, 0.73],
        'std': [0.02, 0.03],
        'ci_lower': [0.71, 0.67],
        'ci_upper': [0.79, 0.79]
    })
    np.random.seed(42)
    iterations = pd.DataFrame({
        'model_type': ['rsf'] * 10,
        'iteration': list(range(10)),
        'metric_name': ['harrell'] * 10,
        'value': np.random.rand(10)
    })
    result = BootstrapResult(summary=summary, iterations=iterations)
    
    repr_str = repr(result)
    assert "BootstrapResult" in repr_str
    assert "rsf" in repr_str
    assert "n_bootstrap" in repr_str


def test_bootstrap_result_repr_empty_iterations():
    """Test __repr__ with non-empty summary but empty iterations DataFrame."""
    summary = pd.DataFrame({
        'model_type': ['rsf'],
        'metric': ['harrell'],
        'mean': [0.75],
        'std': [0.02],
        'ci_lower': [0.71],
        'ci_upper': [0.79]
    })
    iterations = pd.DataFrame(columns=['model_type', 'iteration', 'metric_name', 'value'])
    result = BootstrapResult(summary=summary, iterations=iterations)
    
    repr_str = repr(result)
    assert "BootstrapResult" in repr_str
    assert "rsf" in repr_str
    assert "n_bootstrap=0" in repr_str


def test_bootstrap_result_repr_no_models():
    """Test __repr__ with empty summary (no models)."""
    summary = pd.DataFrame()
    iterations = pd.DataFrame(columns=['model_type', 'iteration', 'metric_name', 'value'])
    result = BootstrapResult(summary=summary, iterations=iterations)
    
    repr_str = repr(result)
    assert "BootstrapResult" in repr_str
    assert "no models" in repr_str


def test_model_result_repr():
    """Test __repr__ for ModelResult."""
    result = ModelResult(
        name='rsf',
        model=None,
        cv_metric_scores=np.array([0.75, 0.78, 0.80]),
        cv_metric_mean=0.777,
        cv_metric_std=0.021,
        is_tuned=True
    )
    
    repr_str = repr(result)
    assert "ModelResult" in repr_str
    assert "rsf" in repr_str
    assert "tuned" in repr_str
    assert "cv_mean" in repr_str


def test_model_result_repr_with_nan():
    """Test __repr__ handles NaN metrics without crashing."""
    result = ModelResult(
        name='test_model',
        model=None,
        cv_metric_scores=np.array([np.nan, np.nan]),
        cv_metric_mean=np.nan,
        cv_metric_std=np.nan,
        is_tuned=False
    )
    
    # Should not raise
    repr_str = repr(result)
    assert "ModelResult" in repr_str
    assert "N/A" in repr_str


def test_model_result_repr_with_evaluation():
    """Test __repr__ for ModelResult with evaluation (c_index_test branch)."""
    concordance = pd.DataFrame({
        'model_type': ['rsf'],
        'c_index_train': [0.85],
        'c_index_test': [0.78]
    })
    evaluation = EvaluationResult(
        concordance=concordance,
        test_set=pd.DataFrame(),
        auc_t=pd.DataFrame(),
        importance=pd.DataFrame(),
        importance_iterations=pd.DataFrame()
    )
    result = ModelResult(
        name='rsf',
        model=None,
        cv_metric_scores=np.array([0.75, 0.78]),
        cv_metric_mean=0.765,
        cv_metric_std=0.015,
        evaluation=evaluation,
        is_tuned=False
    )
    
    repr_str = repr(result)
    assert "ModelResult" in repr_str
    assert "rsf" in repr_str
    assert "c_index_test" in repr_str
    assert "0.780" in repr_str


def test_validation_result_repr():
    """Test __repr__ for ValidationResult."""
    concordance = pd.DataFrame({
        'model_type': ['rsf'],
        'c_index_test': [0.82]
    })
    eval_result = EvaluationResult(
        concordance=concordance,
        test_set=pd.DataFrame(),
        auc_t=pd.DataFrame(),
        importance=pd.DataFrame(),
        importance_iterations=pd.DataFrame()
    )
    result = ValidationResult(
        name='TCGA',
        evaluation=eval_result,
        n_samples=100,
        n_events=45
    )
    
    repr_str = repr(result)
    assert "ValidationResult" in repr_str
    assert "TCGA" in repr_str
    assert "n_samples=100" in repr_str
    assert "c_index" in repr_str


def test_validation_result_repr_none_c_index():
    """Test __repr__ handles None/NaN c_index without crashing."""
    concordance = pd.DataFrame({
        'model_type': ['rsf'],
        'c_index_train': [None],
        'c_index_test': [np.nan]
    })
    eval_result = EvaluationResult(
        concordance=concordance,
        test_set=pd.DataFrame(),
        auc_t=pd.DataFrame(),
        importance=pd.DataFrame(),
        importance_iterations=pd.DataFrame()
    )
    result = ValidationResult(
        name='External',
        evaluation=eval_result,
        n_samples=50,
        n_events=20
    )
    
    repr_str = repr(result)
    assert "ValidationResult" in repr_str
    assert "External" in repr_str
    assert "c_index=N/A" in repr_str


class TestModelResultFromArtifact:
    """Test ModelResult.from_artifact() classmethod."""

    def test_from_artifact_full_metadata(self):
        """from_artifact should construct correct ModelResult from full metadata."""
        metadata = {
            'model_name': 'rsf',
            'results': {
                'cv_metric_mean': 0.750,
                'cv_metric_std': 0.025,
                'is_tuned': True,
                'best_params': {'model__n_estimators': 200},
                'cv_metric_scores': [0.73, 0.75, 0.77],
            },
            'config': {'scoring': 'harrell'},
        }
        sentinel_model = object()
        result = ModelResult.from_artifact(metadata, sentinel_model)

        assert result.name == 'rsf'
        assert result.model is sentinel_model
        assert result.cv_metric_mean == pytest.approx(0.750)
        assert result.cv_metric_std == pytest.approx(0.025)
        assert result.is_tuned is True
        assert result.best_params_ == {'model__n_estimators': 200}
        np.testing.assert_array_almost_equal(
            result.cv_metric_scores, [0.73, 0.75, 0.77],
        )
        assert result.scoring_metric == 'harrell'

    def test_from_artifact_missing_scores(self):
        """from_artifact should handle old artifacts without cv_metric_scores."""
        metadata = {
            'model_name': 'coxph',
            'results': {
                'cv_metric_mean': 0.680,
                'cv_metric_std': 0.040,
            },
        }
        result = ModelResult.from_artifact(metadata, 'dummy_model')

        assert result.name == 'coxph'
        assert result.cv_metric_mean == pytest.approx(0.680)
        assert result.cv_metric_std == pytest.approx(0.040)
        assert len(result.cv_metric_scores) == 0
        assert result.is_tuned is False
        assert result.best_params_ is None
        assert result.scoring_metric is None

    def test_from_artifact_no_results(self):
        """from_artifact should handle metadata with no results at all."""
        metadata = {'model_name': 'gbsa'}
        result = ModelResult.from_artifact(metadata, 'dummy_model')

        assert result.name == 'gbsa'
        assert np.isnan(result.cv_metric_mean)
        assert np.isnan(result.cv_metric_std)
        assert len(result.cv_metric_scores) == 0

    def test_from_artifact_scoring_metric_prefers_results(self):
        """scoring_metric should prefer results.scoring_metric over config.scoring."""
        metadata = {
            'model_name': 'rsf',
            'results': {
                'cv_metric_mean': 0.75,
                'cv_metric_std': 0.02,
                'scoring_metric': 'ipcw',
            },
            'config': {'scoring': 'harrell'},
        }
        result = ModelResult.from_artifact(metadata, 'dummy')
        assert result.scoring_metric == 'ipcw'

    def test_from_artifact_scoring_metric_falls_back_to_config(self):
        """scoring_metric should fall back to config.scoring for old artifacts."""
        metadata = {
            'model_name': 'rsf',
            'results': {
                'cv_metric_mean': 0.75,
                'cv_metric_std': 0.02,
            },
            'config': {'scoring': 'harrell'},
        }
        result = ModelResult.from_artifact(metadata, 'dummy')
        assert result.scoring_metric == 'harrell'


class TestModelResultCvCI:
    """Test CV confidence intervals in ModelResult."""

    def test_summary_cv_ci_populated(self):
        """summary() should return CI values for CV phase when populated."""
        cv_metrics = {
            'harrell': {
                'scores': np.array([0.71, 0.73, 0.72, 0.70, 0.74]),
                'mean': 0.72,
                'std': 0.015,
                'ci_lower': 0.701,
                'ci_upper': 0.739,
            },
        }
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.71, 0.73, 0.72, 0.70, 0.74]),
            cv_metric_mean=0.72,
            cv_metric_std=0.015,
            cv_metrics=cv_metrics,
            cv_ci_lower=0.701,
            cv_ci_upper=0.739,
            cv_ci_level=0.95,
        )
        df = model_result.summary(metrics=['harrell'])
        cv_row = df[df['phase'] == 'cv'].iloc[0]
        assert pd.notna(cv_row['ci_lower'])
        assert pd.notna(cv_row['ci_upper'])
        assert cv_row['ci_lower'] == pytest.approx(0.701, abs=0.001)
        assert cv_row['ci_upper'] == pytest.approx(0.739, abs=0.001)

    def test_summary_cv_ci_nan_without_data(self):
        """summary() should return NaN CI when cv_ci fields are NaN (default)."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.0,
            cv_metrics={'harrell': {'scores': np.array([0.72]), 'mean': 0.72, 'std': 0.0}},
        )
        df = model_result.summary(metrics=['harrell'])
        cv_row = df[df['phase'] == 'cv'].iloc[0]
        # No ci_lower/ci_upper keys in cv_metrics → should be NaN
        assert pd.isna(cv_row['ci_lower'])
        assert pd.isna(cv_row['ci_upper'])

    def test_get_metric_cv_ci_lower(self):
        """get_metric should return cv ci_lower."""
        cv_metrics = {
            'harrell': {
                'scores': np.array([0.71, 0.73]),
                'mean': 0.72,
                'std': 0.01,
                'ci_lower': 0.65,
                'ci_upper': 0.79,
            },
        }
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.71, 0.73]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            cv_metrics=cv_metrics,
            cv_ci_lower=0.65,
            cv_ci_upper=0.79,
        )
        assert model_result.get_metric('harrell', 'cv', 'ci_lower') == pytest.approx(0.65)
        assert model_result.get_metric('harrell', 'cv', 'ci_upper') == pytest.approx(0.79)

    def test_get_metric_cv_ci_fallback_primary(self):
        """get_metric ci_lower/ci_upper should work via primary metric fallback."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.71, 0.73]),
            cv_metric_mean=0.72,
            cv_metric_std=0.01,
            cv_metrics=None,
            cv_ci_lower=0.65,
            cv_ci_upper=0.79,
        )
        assert model_result.get_metric('harrell', 'cv', 'ci_lower') == pytest.approx(0.65)
        assert model_result.get_metric('harrell', 'cv', 'ci_upper') == pytest.approx(0.79)

    def test_get_metric_cv_ci_nan_valued_keys(self):
        """get_metric should return NaN when ci_lower/ci_upper keys exist but are NaN."""
        cv_metrics = {
            'harrell': {
                'scores': np.array([0.72]),
                'mean': 0.72,
                'std': 0.0,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
            },
        }
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.0,
            cv_metrics=cv_metrics,
        )
        result = model_result.get_metric('harrell', 'cv', 'ci_lower')
        assert isinstance(result, float)
        assert np.isnan(result)

    def test_cv_ci_level_stored(self):
        """cv_ci_level should be stored on ModelResult."""
        model_result = ModelResult(
            name='rsf',
            model=None,
            cv_metric_scores=np.array([0.72]),
            cv_metric_mean=0.72,
            cv_metric_std=0.0,
            cv_ci_level=0.90,
        )
        assert model_result.cv_ci_level == 0.90


class TestComputeCvCI:
    """Test the _compute_cv_ci helper function."""

    def test_basic_ci(self):
        """CI should bracket the mean."""
        from mlsurv.learner._training import _compute_cv_ci

        scores = [0.70, 0.72, 0.74, 0.71, 0.73]
        lo, hi = _compute_cv_ci(scores, 0.95)
        mean = np.mean(scores)
        assert lo < mean < hi

    def test_fewer_than_2_returns_nan(self):
        """< 2 finite scores should return NaN."""
        from mlsurv.learner._training import _compute_cv_ci

        lo, hi = _compute_cv_ci([0.72], 0.95)
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_all_nan_returns_nan(self):
        """All-NaN input should return NaN."""
        from mlsurv.learner._training import _compute_cv_ci

        lo, hi = _compute_cv_ci([np.nan, np.nan], 0.95)
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_filters_non_finite(self):
        """Should ignore NaN and inf values."""
        from mlsurv.learner._training import _compute_cv_ci

        scores = [0.70, np.nan, 0.74, np.inf, 0.72]
        lo, hi = _compute_cv_ci(scores, 0.95)
        assert not np.isnan(lo)
        assert not np.isnan(hi)
        assert lo < hi

    def test_narrower_ci_with_lower_level(self):
        """Lower confidence level should produce narrower CI."""
        from mlsurv.learner._training import _compute_cv_ci

        scores = [0.70, 0.72, 0.74, 0.71, 0.73]
        lo_95, hi_95 = _compute_cv_ci(scores, 0.95)
        lo_90, hi_90 = _compute_cv_ci(scores, 0.90)
        assert (hi_95 - lo_95) > (hi_90 - lo_90)

    def test_constant_scores_produces_point_ci(self):
        """Constant scores should produce CI equal to the mean (zero width)."""
        from mlsurv.learner._training import _compute_cv_ci

        scores = [0.72, 0.72, 0.72, 0.72, 0.72]
        lo, hi = _compute_cv_ci(scores, 0.95)
        assert lo == pytest.approx(0.72)
        assert hi == pytest.approx(0.72)

    def test_ci_matches_known_values(self):
        """CI should match hand-computed t-distribution interval."""
        from mlsurv.learner._training import _compute_cv_ci
        from scipy.stats import t as t_dist

        scores = [0.70, 0.72, 0.74, 0.71, 0.73]
        lo, hi = _compute_cv_ci(scores, 0.95)
        # mean=0.72, std(ddof=1)=0.01581, k=5
        # t_crit = t.ppf(0.975, df=4) = 2.7764
        # margin = 2.7764 * 0.01581 / sqrt(5) = 0.01963
        t_crit = t_dist.ppf(0.975, df=4)
        margin = t_crit * (0.015811388300842 / np.sqrt(5))
        assert lo == pytest.approx(0.72 - margin, abs=1e-6)
        assert hi == pytest.approx(0.72 + margin, abs=1e-6)

    def test_two_scores_produces_wide_ci(self):
        """Two scores (df=1) should produce a valid but very wide CI."""
        from mlsurv.learner._training import _compute_cv_ci

        scores = [0.70, 0.74]
        lo, hi = _compute_cv_ci(scores, 0.95)
        assert lo < 0.72 < hi
        assert (hi - lo) > 0.2  # Very wide due to df=1

    def test_empty_scores_returns_nan(self):
        """Empty score list should return NaN CI."""
        from mlsurv.learner._training import _compute_cv_ci

        lo, hi = _compute_cv_ci([], 0.95)
        assert np.isnan(lo)
        assert np.isnan(hi)


# ---------------------------------------------------------------------------
# importance_summary() tests
# ---------------------------------------------------------------------------

class TestImportanceSummary:
    """Tests for BootstrapResult.importance_summary()."""

    def _make_result(self, importance_df):
        """Helper: build a BootstrapResult with given importance DataFrame."""
        summary = pd.DataFrame({
            "model_type": ["rsf"],
            "metric": ["harrell"],
            "mean": [0.72],
            "std": [0.02],
            "ci_lower": [0.68],
            "ci_upper": [0.76],
        })
        iterations = pd.DataFrame({
            "model_type": ["rsf"] * 2,
            "iteration": [0, 1],
            "metric_name": ["harrell", "harrell"],
            "value": [0.71, 0.73],
        })
        return BootstrapResult(
            summary=summary,
            iterations=iterations,
            importance=importance_df,
        )

    def test_importance_summary_basic(self):
        """Output has correct columns, shape, sorted by p_value."""
        n_iter = 20
        rng = np.random.RandomState(42)
        importance = pd.DataFrame({
            "model_type": ["rsf"] * (n_iter * 3),
            "iteration": list(range(n_iter)) * 3,
            "feature": ["age"] * n_iter + ["stage"] * n_iter + ["grade"] * n_iter,
            "importance": np.concatenate([
                rng.normal(0.05, 0.01, n_iter),
                rng.normal(0.02, 0.01, n_iter),
                rng.normal(0.01, 0.005, n_iter),
            ]),
        })
        result = self._make_result(importance)
        df = result.importance_summary()

        assert list(df.columns) == [
            "feature", "mean", "std", "ci_lower", "ci_upper",
            "p_value", "p_adjusted",
        ]
        assert len(df) == 3
        # Sorted by p_value ascending
        assert (df["p_value"].diff().dropna() >= 0).all()

    def test_importance_summary_pvalues(self):
        """Controlled data: verify p-values are correct."""
        n = 100
        importance = pd.DataFrame({
            "model_type": ["rsf"] * (n * 3),
            "iteration": list(range(n)) * 3,
            "feature": (["always_pos"] * n + ["always_neg"] * n
                        + ["half_half"] * n),
            "importance": np.concatenate([
                np.abs(np.arange(1, n + 1, dtype=float)),   # all > 0
                -np.abs(np.arange(1, n + 1, dtype=float)),  # all ≤ 0
                np.arange(1, n + 1, dtype=float) - 50.5,    # 50 ≤ 0, 50 > 0
            ]),
        })
        result = self._make_result(importance)
        df = result.importance_summary(p_adjust="none")

        always_pos = df.loc[df["feature"] == "always_pos", "p_value"].iloc[0]
        always_neg = df.loc[df["feature"] == "always_neg", "p_value"].iloc[0]
        half_half = df.loc[df["feature"] == "half_half", "p_value"].iloc[0]

        assert always_pos == pytest.approx(0.0)
        assert always_neg == pytest.approx(1.0)
        assert half_half == pytest.approx(0.5)

    def test_importance_summary_adjustment(self):
        """p_adjusted >= p_value with holm; none gives raw values."""
        n = 50
        rng = np.random.RandomState(7)
        importance = pd.DataFrame({
            "model_type": ["rsf"] * (n * 2),
            "iteration": list(range(n)) * 2,
            "feature": ["a"] * n + ["b"] * n,
            "importance": np.concatenate([
                rng.normal(0.05, 0.01, n),
                rng.normal(0.01, 0.02, n),
            ]),
        })
        result = self._make_result(importance)

        df_holm = result.importance_summary(p_adjust="holm")
        assert (df_holm["p_adjusted"] >= df_holm["p_value"] - 1e-15).all()

        df_none = result.importance_summary(p_adjust="none")
        assert (df_none["p_value"] == df_none["p_adjusted"]).all()

    def test_importance_summary_no_importance(self):
        """Raises ValueError when importance is None."""
        result = self._make_result(importance_df=None)
        with pytest.raises(ValueError, match="No importance data"):
            result.importance_summary()

    def test_importance_summary_multi_model(self):
        """Errors without model_type; works with it specified."""
        n = 20
        rng = np.random.RandomState(99)
        importance = pd.DataFrame({
            "model_type": ["rsf"] * n + ["coxph"] * n,
            "iteration": list(range(n)) * 2,
            "feature": ["age"] * (n * 2),
            "importance": rng.normal(0.03, 0.01, n * 2),
        })
        result = self._make_result(importance)

        with pytest.raises(ValueError, match="Multiple model_types"):
            result.importance_summary()

        df = result.importance_summary(model_type="rsf")
        assert len(df) == 1
        assert df["feature"].iloc[0] == "age"

    def test_importance_summary_empty_dataframe(self):
        """Raises ValueError when importance is an empty DataFrame."""
        result = self._make_result(pd.DataFrame(
            columns=["model_type", "iteration", "feature", "importance"]
        ))
        with pytest.raises(ValueError, match="No importance data"):
            result.importance_summary()

    def test_importance_summary_single_iteration(self):
        """Single bootstrap iteration: std=0, p-value well-defined."""
        importance = pd.DataFrame({
            "model_type": ["rsf"],
            "iteration": [0],
            "feature": ["age"],
            "importance": [0.05],
        })
        result = self._make_result(importance)
        df = result.importance_summary(p_adjust="none")
        assert df["std"].iloc[0] == 0.0
        assert df["p_value"].iloc[0] == 0.0  # 0.05 > 0

    def test_importance_summary_ci_narrows(self):
        """80% CI is strictly narrower than 95% CI."""
        n = 100
        rng = np.random.RandomState(0)
        importance = pd.DataFrame({
            "model_type": ["rsf"] * n,
            "iteration": list(range(n)),
            "feature": ["age"] * n,
            "importance": rng.normal(0.05, 0.02, n),
        })
        result = self._make_result(importance)
        df_95 = result.importance_summary(ci=0.95)
        df_80 = result.importance_summary(ci=0.80)
        assert df_80["ci_lower"].iloc[0] > df_95["ci_lower"].iloc[0]
        assert df_80["ci_upper"].iloc[0] < df_95["ci_upper"].iloc[0]

    def test_importance_summary_nan_values_excluded(self):
        """NaN importance values are excluded from statistics."""
        importance = pd.DataFrame({
            "model_type": ["rsf"] * 4,
            "iteration": [0, 1, 2, 3],
            "feature": ["age"] * 4,
            "importance": [0.05, np.nan, 0.03, np.nan],
        })
        result = self._make_result(importance)
        df = result.importance_summary(p_adjust="none")
        # Only 2 finite values (0.05, 0.03), both > 0
        assert df["p_value"].iloc[0] == 0.0
        assert df["mean"].iloc[0] == pytest.approx(0.04)

    def test_importance_summary_unknown_model_type(self):
        """Raises ValueError for unknown model_type with helpful message."""
        n = 10
        importance = pd.DataFrame({
            "model_type": ["rsf"] * n,
            "iteration": list(range(n)),
            "feature": ["age"] * n,
            "importance": np.ones(n) * 0.05,
        })
        result = self._make_result(importance)
        with pytest.raises(ValueError, match="not found in importance"):
            result.importance_summary(model_type="typo")

    @pytest.mark.parametrize("bad_ci", [0.0, 1.0, -0.5, 1.5])
    def test_importance_summary_invalid_ci(self, bad_ci):
        """Raises ValueError for ci outside (0, 1)."""
        importance = pd.DataFrame({
            "model_type": ["rsf"] * 10,
            "iteration": list(range(10)),
            "feature": ["age"] * 10,
            "importance": np.ones(10) * 0.05,
        })
        result = self._make_result(importance)
        with pytest.raises(ValueError, match="ci must be between"):
            result.importance_summary(ci=bad_ci)

    def test_importance_summary_all_nan_feature(self):
        """Feature with all-NaN/Inf importance gets an all-NaN row."""
        importance = pd.DataFrame({
            "model_type": ["rsf"] * 6,
            "iteration": [0, 1, 2, 0, 1, 2],
            "feature": ["good"] * 3 + ["broken"] * 3,
            "importance": [0.05, 0.03, 0.04, np.nan, np.nan, np.inf],
        })
        result = self._make_result(importance)
        df = result.importance_summary(p_adjust="none")
        broken = df.loc[df["feature"] == "broken"]
        assert len(broken) == 1
        assert np.isnan(broken["mean"].iloc[0])
        assert np.isnan(broken["p_value"].iloc[0])
        assert np.isnan(broken["p_adjusted"].iloc[0])


# ---------------------------------------------------------------------------
# Survival probabilities API
# ---------------------------------------------------------------------------

def _make_minimal_eval(**kwargs):
    """Create a minimal EvaluationResult with empty DataFrames."""
    defaults = dict(
        concordance=pd.DataFrame({'model_type': ['m'], 'c_index_test': [0.7]}),
        test_set=pd.DataFrame({'model_type': ['m']}),
        auc_t=pd.DataFrame({'model_type': ['m']}),
        importance=pd.DataFrame({'model_type': ['m']}),
        importance_iterations=pd.DataFrame({'model_type': ['m']}),
    )
    defaults.update(kwargs)
    return EvaluationResult(**defaults)


class TestSurvivalProbabilitiesProperty:
    """Tests for EvaluationResult.survival_probabilities property."""

    def test_none_when_cache_is_none(self):
        result = _make_minimal_eval(_surv_probs_cache=None)
        assert result.survival_probabilities is None

    def test_single_model(self):
        probs = np.array([[0.9, 0.8, 0.7], [0.85, 0.75, 0.6]])
        times = np.array([1.0, 2.0, 3.0])
        result = _make_minimal_eval(_surv_probs_cache={'rsf': (probs, times)})

        sp = result.survival_probabilities
        assert sp is not None
        assert 'rsf' in sp
        df = sp['rsf']
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        np.testing.assert_array_almost_equal(df.values, probs)
        np.testing.assert_array_equal(df.columns.values, times)

    def test_multi_model(self):
        probs1 = np.array([[0.9, 0.8]])
        probs2 = np.array([[0.7, 0.6]])
        times = np.array([1.0, 5.0])
        cache = {'rsf': (probs1, times), 'coxph': (probs2, times)}
        result = _make_minimal_eval(_surv_probs_cache=cache)

        sp = result.survival_probabilities
        assert sp is not None
        assert set(sp.keys()) == {'rsf', 'coxph'}

    def test_skips_none_probs_entry(self):
        """A model's None probs are excluded; an all-None cache yields {}.

        The cache exists, so the result is an empty dict — distinct from None,
        which means "no cache at all". This keeps "never computed" (None)
        distinguishable from "computed, none usable" ({}).
        """
        cache = {'rsf': (None, np.array([1.0]))}
        result = _make_minimal_eval(_surv_probs_cache=cache)
        assert result.survival_probabilities == {}

    def test_empty_vs_none_are_distinct(self):
        """None (no cache) and {} (cache present, none usable) must differ."""
        no_cache = _make_minimal_eval(_surv_probs_cache=None)
        empty_cache = _make_minimal_eval(_surv_probs_cache={'rsf': (None, np.array([1.0]))})
        assert no_cache.survival_probabilities is None
        assert empty_cache.survival_probabilities == {}


class TestToDataframesSurvivalProbabilities:
    """Tests for survival_probabilities in to_dataframes()."""

    def test_includes_when_cache_present(self):
        probs = np.array([[0.9, 0.8], [0.7, 0.6]])
        times = np.array([1.0, 2.0])
        result = _make_minimal_eval(_surv_probs_cache={'rsf': (probs, times)})

        dfs = result.to_dataframes()
        assert 'survival_probabilities' in dfs
        sp_df = dfs['survival_probabilities']
        assert 'model_type' in sp_df.columns
        assert 'sample_idx' in sp_df.columns
        assert sp_df.shape[0] == 2  # 2 samples
        assert sp_df['model_type'].iloc[0] == 'rsf'

    def test_excludes_when_cache_is_none(self):
        result = _make_minimal_eval(_surv_probs_cache=None)
        dfs = result.to_dataframes()
        assert 'survival_probabilities' not in dfs


class TestExtractModelEvaluationCachePropagation:
    """extract_model_evaluation() propagates _surv_probs_cache."""

    def test_propagates_cache(self):
        probs_rsf = np.array([[0.9, 0.8]])
        probs_cox = np.array([[0.7, 0.6]])
        times = np.array([1.0, 2.0])
        cache = {'rsf': (probs_rsf, times), 'coxph': (probs_cox, times)}

        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.75, 0.70],
        })
        multi = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame({'model_type': ['rsf', 'coxph']}),
            auc_t=pd.DataFrame({'model_type': ['rsf', 'coxph']}),
            importance=pd.DataFrame({'model_type': ['rsf', 'coxph']}),
            importance_iterations=pd.DataFrame({'model_type': ['rsf', 'coxph']}),
            _surv_probs_cache=cache,
        )

        single = extract_model_evaluation(multi, 'rsf')
        assert single._surv_probs_cache is not None
        assert 'rsf' in single._surv_probs_cache
        assert 'coxph' not in single._surv_probs_cache

    def test_none_when_model_not_in_cache(self):
        cache = {'rsf': (np.array([[0.9]]), np.array([1.0]))}
        multi = _make_minimal_eval(_surv_probs_cache=cache)
        # model name 'm' matches the minimal concordance, but not the cache
        single = extract_model_evaluation(multi, 'missing')
        assert single._surv_probs_cache is None


class TestFilterEvalResultCachePropagation:
    """_filter_eval_result() propagates _surv_probs_cache."""

    def test_filters_to_matching_models(self):
        from mlsurv.learner._visualization import VisualizationMixin
        probs_rsf = np.array([[0.9, 0.8]])
        probs_cox = np.array([[0.7, 0.6]])
        times = np.array([1.0, 2.0])
        cache = {'rsf': (probs_rsf, times), 'coxph': (probs_cox, times)}
        boot = BootstrapResult(
            summary=pd.DataFrame({'model_type': ['rsf'], 'metric': ['harrell'],
                                  'mean': [0.7]}),
            iterations=pd.DataFrame({'model_type': ['rsf'], 'iteration': [0]}),
            ci_level=0.80,
        )
        result = _make_minimal_eval(_surv_probs_cache=cache, bootstrap=boot)

        filtered = VisualizationMixin._filter_eval_result(result, ['rsf'])
        assert filtered._surv_probs_cache is not None
        assert 'rsf' in filtered._surv_probs_cache
        assert 'coxph' not in filtered._surv_probs_cache
        # ci_level survives the viz filter path
        assert filtered.bootstrap.ci_level == 0.80

    def test_none_when_no_models_match(self):
        from mlsurv.learner._visualization import VisualizationMixin
        cache = {'rsf': (np.array([[0.9]]), np.array([1.0]))}
        result = _make_minimal_eval(_surv_probs_cache=cache)

        filtered = VisualizationMixin._filter_eval_result(result, ['missing'])
        assert filtered._surv_probs_cache is None

    def test_none_when_cache_is_none(self):
        from mlsurv.learner._visualization import VisualizationMixin
        result = _make_minimal_eval(_surv_probs_cache=None)

        filtered = VisualizationMixin._filter_eval_result(result, ['rsf'])
        assert filtered._surv_probs_cache is None


class TestToDataframesSurvProbsMultiModel:
    """Multi-model and edge-case exports for survival_probabilities."""

    def test_multi_model_export(self):
        probs1 = np.array([[0.9, 0.8], [0.7, 0.6]])
        probs2 = np.array([[0.5, 0.4]])
        times = np.array([1.0, 2.0])
        cache = {'rsf': (probs1, times), 'coxph': (probs2, times)}
        result = _make_minimal_eval(_surv_probs_cache=cache)

        dfs = result.to_dataframes()
        sp_df = dfs['survival_probabilities']
        assert sp_df.shape[0] == 3  # 2 rsf + 1 coxph
        assert set(sp_df['model_type']) == {'rsf', 'coxph'}
        assert 't_1.0' in sp_df.columns
        assert 't_2.0' in sp_df.columns

    def test_all_none_probs_excluded(self):
        cache = {'rsf': (None, np.array([1.0])), 'coxph': (None, np.array([1.0]))}
        result = _make_minimal_eval(_surv_probs_cache=cache)
        dfs = result.to_dataframes()
        assert 'survival_probabilities' not in dfs


class TestInteractionRawCacheExtractPropagation:
    """extract_model_evaluation() propagates _interaction_raw_cache."""

    def test_propagates_cache(self):
        np.random.seed(42)
        raw_rsf = np.random.randn(10, 3, 3)
        raw_cox = np.random.randn(10, 3, 3)
        feat_names = ['f1', 'f2', 'f3']
        cache = {
            'rsf': (raw_rsf, feat_names),
            'coxph': (raw_cox, feat_names),
        }

        concordance = pd.DataFrame({
            'model_type': ['rsf', 'coxph'],
            'c_index_test': [0.75, 0.70],
        })
        multi = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame({'model_type': ['rsf', 'coxph']}),
            auc_t=pd.DataFrame({'model_type': ['rsf', 'coxph']}),
            importance=pd.DataFrame({'model_type': ['rsf', 'coxph']}),
            importance_iterations=pd.DataFrame({'model_type': ['rsf', 'coxph']}),
            _interaction_raw_cache=cache,
            interaction_method='shap',
        )

        single = extract_model_evaluation(multi, 'rsf')
        assert single.interaction_values_raw is not None
        np.testing.assert_array_equal(single.interaction_values_raw, raw_rsf)
        assert single.interaction_feature_names == feat_names
        assert single._interaction_raw_cache is not None
        assert 'rsf' in single._interaction_raw_cache
        assert 'coxph' not in single._interaction_raw_cache

    def test_none_when_model_not_in_cache(self):
        np.random.seed(42)
        cache = {'rsf': (np.random.randn(5, 2, 2), ['a', 'b'])}
        multi = _make_minimal_eval(_interaction_raw_cache=cache)
        single = extract_model_evaluation(multi, 'missing')
        assert single.interaction_values_raw is None
        assert single.interaction_feature_names is None
        assert single._interaction_raw_cache is None

    def test_none_when_cache_is_none(self):
        multi = _make_minimal_eval(_interaction_raw_cache=None)
        single = extract_model_evaluation(multi, 'm')
        assert single.interaction_values_raw is None
        assert single._interaction_raw_cache is None


class TestInteractionRawCacheFilterPropagation:
    """filter_eval_result() propagates _interaction_raw_cache."""

    def test_filters_to_matching_models(self):
        from mlsurv.learner._visualization import VisualizationMixin
        np.random.seed(42)
        cache = {
            'rsf': (np.random.randn(5, 2, 2), ['a', 'b']),
            'coxph': (np.random.randn(5, 2, 2), ['a', 'b']),
        }
        result = _make_minimal_eval(_interaction_raw_cache=cache)
        filtered = VisualizationMixin._filter_eval_result(result, ['rsf'])
        assert filtered._interaction_raw_cache is not None
        assert 'rsf' in filtered._interaction_raw_cache
        assert 'coxph' not in filtered._interaction_raw_cache

    def test_none_when_no_models_match(self):
        from mlsurv.learner._visualization import VisualizationMixin
        np.random.seed(42)
        cache = {'rsf': (np.random.randn(5, 2, 2), ['a', 'b'])}
        result = _make_minimal_eval(_interaction_raw_cache=cache)
        filtered = VisualizationMixin._filter_eval_result(result, ['missing'])
        assert filtered._interaction_raw_cache is None

    def test_none_when_cache_is_none(self):
        from mlsurv.learner._visualization import VisualizationMixin
        result = _make_minimal_eval(_interaction_raw_cache=None)
        filtered = VisualizationMixin._filter_eval_result(result, ['rsf'])
        assert filtered._interaction_raw_cache is None


class TestInteractionRawCacheCombine:
    """EvaluationResult.combine() merges _interaction_raw_cache."""

    def test_combine_merges_caches(self):
        np.random.seed(42)
        raw1 = np.random.randn(5, 2, 2)
        raw2 = np.random.randn(5, 2, 2)
        feat = ['a', 'b']

        mr1 = ModelResult(name='rsf', model=None, cv_metric_scores=np.array([0.7]), cv_metric_mean=0.7, cv_metric_std=0.01)
        mr1.evaluation = _make_minimal_eval(
            concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.75]}),
            test_set=pd.DataFrame({'model_type': ['rsf']}),
            auc_t=pd.DataFrame({'model_type': ['rsf']}),
            importance=pd.DataFrame({'model_type': ['rsf']}),
            importance_iterations=pd.DataFrame({'model_type': ['rsf']}),
            _interaction_raw_cache={'rsf': (raw1, feat)},
            interaction_method='shap',
        )
        mr2 = ModelResult(name='coxph', model=None, cv_metric_scores=np.array([0.65]), cv_metric_mean=0.65, cv_metric_std=0.02)
        mr2.evaluation = _make_minimal_eval(
            concordance=pd.DataFrame({'model_type': ['coxph'], 'c_index_test': [0.70]}),
            test_set=pd.DataFrame({'model_type': ['coxph']}),
            auc_t=pd.DataFrame({'model_type': ['coxph']}),
            importance=pd.DataFrame({'model_type': ['coxph']}),
            importance_iterations=pd.DataFrame({'model_type': ['coxph']}),
            _interaction_raw_cache={'coxph': (raw2, feat)},
            interaction_method='shap',
        )

        combined = EvaluationResult.combine({'rsf': mr1, 'coxph': mr2})
        assert combined._interaction_raw_cache is not None
        assert 'rsf' in combined._interaction_raw_cache
        assert 'coxph' in combined._interaction_raw_cache
        np.testing.assert_array_equal(combined._interaction_raw_cache['rsf'][0], raw1)
        np.testing.assert_array_equal(combined._interaction_raw_cache['coxph'][0], raw2)

    def test_combine_none_when_no_caches(self):
        mr1 = ModelResult(name='rsf', model=None, cv_metric_scores=np.array([0.7]), cv_metric_mean=0.7, cv_metric_std=0.01)
        mr1.evaluation = _make_minimal_eval(
            concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.75]}),
            test_set=pd.DataFrame({'model_type': ['rsf']}),
            auc_t=pd.DataFrame({'model_type': ['rsf']}),
            importance=pd.DataFrame({'model_type': ['rsf']}),
            importance_iterations=pd.DataFrame({'model_type': ['rsf']}),
        )
        combined = EvaluationResult.combine({'rsf': mr1})
        assert combined._interaction_raw_cache is None


class TestCombineEmptyFrameAliasing:
    """Empty fields from combine() must be distinct objects.

    A mutable ``default=pd.DataFrame()`` in the nested ``_concat`` helper was
    created once per combine() call, so every empty required field aliased the
    same DataFrame — mutating one silently corrupted the others.
    """

    @staticmethod
    def _combine_with_empty_frames():
        """Combine two models whose required frames (except concordance) are empty."""
        empty = dict(
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
        )
        mr1 = ModelResult(name='rsf', model=None, cv_metric_scores=np.array([0.7]), cv_metric_mean=0.7, cv_metric_std=0.01)
        mr1.evaluation = _make_minimal_eval(
            concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.75]}),
            **empty,
        )
        mr2 = ModelResult(name='coxph', model=None, cv_metric_scores=np.array([0.65]), cv_metric_mean=0.65, cv_metric_std=0.02)
        mr2.evaluation = _make_minimal_eval(
            concordance=pd.DataFrame({'model_type': ['coxph'], 'c_index_test': [0.70]}),
            **empty,
        )
        return EvaluationResult.combine({'rsf': mr1, 'coxph': mr2})

    def test_empty_fields_are_distinct_objects(self):
        agg = self._combine_with_empty_frames()
        frames = [agg.test_set, agg.auc_t, agg.importance, agg.importance_iterations]
        for i, a in enumerate(frames):
            assert a.empty
            for b in frames[i + 1:]:
                assert a is not b

    def test_mutating_one_empty_field_does_not_corrupt_others(self):
        agg = self._combine_with_empty_frames()
        agg.importance['x'] = 1
        assert 'x' not in agg.test_set.columns
        assert 'x' not in agg.auc_t.columns
        assert 'x' not in agg.importance_iterations.columns

    def test_optional_fields_stay_none_when_absent(self):
        """Explicit default=None call sites still return None, not empty frames."""
        agg = self._combine_with_empty_frames()
        assert agg.importance_per_column is None
        assert agg.shap_values is None
        assert agg.shap_values_per_column is None
        assert agg.interactions is None


class TestEvaluationResultSetstate:
    """EvaluationResult.__setstate__ fills missing fields from old pickles."""

    def test_missing_interaction_raw_cache(self):
        """Unpickling old EvaluationResult without _interaction_raw_cache gets None."""
        er = _make_minimal_eval()
        # Simulate old pickle without _interaction_raw_cache
        state = er.__dict__.copy()
        del state['_interaction_raw_cache']
        new_er = EvaluationResult.__new__(EvaluationResult)
        new_er.__setstate__(state)
        assert new_er._interaction_raw_cache is None

    def test_roundtrip_preserves_cache(self):
        """Pickle roundtrip preserves _interaction_raw_cache."""
        import pickle
        np.random.seed(42)
        raw = np.random.randn(5, 2, 2)
        er = _make_minimal_eval(_interaction_raw_cache={'rsf': (raw, ['a', 'b'])})
        restored = pickle.loads(pickle.dumps(er))
        assert restored._interaction_raw_cache is not None
        assert 'rsf' in restored._interaction_raw_cache
        np.testing.assert_array_equal(restored._interaction_raw_cache['rsf'][0], raw)


class TestBootstrapResultSetstate:
    """BootstrapResult.__setstate__ fills missing fields from old pickles."""

    def test_missing_field_gets_default(self):
        """Unpickling old BootstrapResult without optional fields gets None."""
        from mlsurv.results.containers import BootstrapResult
        br = BootstrapResult(
            summary=pd.DataFrame({'model_type': ['rsf']}),
            iterations=pd.DataFrame({'model_type': ['rsf']}),
        )
        state = br.__dict__.copy()
        del state['subpopulation_auc_t']  # Simulate old pickle
        new_br = BootstrapResult.__new__(BootstrapResult)
        new_br.__setstate__(state)
        assert new_br.subpopulation_auc_t is None

    def test_roundtrip_preserves_fields(self):
        """Pickle roundtrip preserves all fields."""
        import pickle
        from mlsurv.results.containers import BootstrapResult
        br = BootstrapResult(
            summary=pd.DataFrame({'model_type': ['rsf'], 'metric': ['harrell']}),
            iterations=pd.DataFrame({'model_type': ['rsf']}),
            importance=pd.DataFrame({'model_type': ['rsf'], 'feature': ['age']}),
        )
        restored = pickle.loads(pickle.dumps(br))
        assert restored.importance is not None
        assert 'feature' in restored.importance.columns


class TestBootstrapResultCiLevel:
    """BootstrapResult stores the CI level it was computed at."""

    @staticmethod
    def _make(ci_level=None, model='rsf'):
        summary = pd.DataFrame({
            'model_type': [model],
            'metric': ['harrell'],
            'point': [0.71],
            'mean': [0.72],
            'std': [0.02],
            'ci_lower': [0.67],
            'ci_upper': [0.77],
        })
        iterations = pd.DataFrame({
            'model_type': [model, model],
            'iteration': [0, 1],
            'metric_name': ['harrell', 'harrell'],
            'value': [0.71, 0.73],
        })
        kwargs = {} if ci_level is None else {'ci_level': ci_level}
        return BootstrapResult(summary=summary, iterations=iterations, **kwargs)

    def test_default_field_and_attrs(self):
        """Default ci_level is 0.95 and is stamped on summary.attrs."""
        br = self._make()
        assert br.ci_level == 0.95
        assert br.summary.attrs['ci_level'] == 0.95

    def test_custom_field_and_attrs(self):
        """A custom ci_level lands on both the field and summary.attrs."""
        br = self._make(ci_level=0.80)
        assert br.ci_level == 0.80
        assert br.summary.attrs['ci_level'] == 0.80

    def test_combine_preserves_ci_level(self):
        """combine() carries ci_level through (concat drops attrs, field re-stamps)."""
        mr1 = ModelResult(name='rsf', model=None, cv_metric_scores=np.array([0.7]), cv_metric_mean=0.7, cv_metric_std=0.01)
        mr1.bootstrap = self._make(ci_level=0.80, model='rsf')
        mr2 = ModelResult(name='coxph', model=None, cv_metric_scores=np.array([0.65]), cv_metric_mean=0.65, cv_metric_std=0.02)
        mr2.bootstrap = self._make(ci_level=0.80, model='coxph')
        combined = BootstrapResult.combine({'rsf': mr1, 'coxph': mr2})
        assert combined.ci_level == 0.80
        assert combined.summary.attrs['ci_level'] == 0.80

    def test_combine_warns_on_mixed_ci_levels(self):
        """combine() warns when per-model levels differ; first model's wins."""
        from mlsurv._warnings import MethodologicalWarning
        mr1 = ModelResult(name='rsf', model=None, cv_metric_scores=np.array([0.7]), cv_metric_mean=0.7, cv_metric_std=0.01)
        mr1.bootstrap = self._make(ci_level=0.80, model='rsf')
        mr2 = ModelResult(name='coxph', model=None, cv_metric_scores=np.array([0.65]), cv_metric_mean=0.65, cv_metric_std=0.02)
        mr2.bootstrap = self._make(ci_level=0.95, model='coxph')
        with pytest.warns(MethodologicalWarning, match="different CI levels"):
            combined = BootstrapResult.combine({'rsf': mr1, 'coxph': mr2})
        assert combined.ci_level == 0.80  # first model's level

    def test_extract_model_bootstrap_preserves_ci_level(self):
        """Per-model split keeps the multi-model result's ci_level."""
        from mlsurv.results._containers_extraction import _extract_model_bootstrap
        mr1 = ModelResult(name='rsf', model=None, cv_metric_scores=np.array([0.7]), cv_metric_mean=0.7, cv_metric_std=0.01)
        mr1.bootstrap = self._make(ci_level=0.80, model='rsf')
        mr2 = ModelResult(name='coxph', model=None, cv_metric_scores=np.array([0.65]), cv_metric_mean=0.65, cv_metric_std=0.02)
        mr2.bootstrap = self._make(ci_level=0.80, model='coxph')
        multi = BootstrapResult.combine({'rsf': mr1, 'coxph': mr2})
        single = _extract_model_bootstrap(multi, 'rsf')
        assert single.ci_level == 0.80
        assert single.summary.attrs['ci_level'] == 0.80

    def test_print_metrics_uses_bootstrap_ci_level(self, capsys):
        """EvaluationResult.print_metrics() labels at the bootstrap's level."""
        ev = _make_minimal_eval(
            concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.71]}),
            bootstrap=self._make(ci_level=0.80, model='rsf'),
        )
        ev.print_metrics()
        out = capsys.readouterr().out
        assert '80% CI' in out
        assert '95% CI' not in out

    def test_setstate_backfills_ci_level(self):
        """Old pickles without ci_level get the 0.95 default and attrs stamp."""
        br = self._make(ci_level=0.80)
        state = br.__dict__.copy()
        del state['ci_level']  # Simulate old pickle
        state['summary'] = state['summary'].copy()
        state['summary'].attrs.clear()
        new_br = BootstrapResult.__new__(BootstrapResult)
        new_br.__setstate__(state)
        assert new_br.ci_level == 0.95
        assert new_br.summary.attrs['ci_level'] == 0.95

    def test_print_summary_uses_stored_ci_level(self, capsys):
        """print_summary() labels intervals at the stored level by default."""
        self._make(ci_level=0.80).print_summary()
        out = capsys.readouterr().out
        assert '80% CI' in out
        assert '95% CI' not in out

    def test_print_summary_explicit_override(self, capsys):
        """An explicit ci_level still wins over the stored one."""
        self._make(ci_level=0.80).print_summary(ci_level=0.95)
        out = capsys.readouterr().out
        assert '95% CI' in out
        assert '80% CI' not in out

    def test_print_summary_fractional_ci_level(self, capsys):
        """0.975 prints as '97.5% CI' — no int() truncation to 97."""
        self._make(ci_level=0.975).print_summary()
        out = capsys.readouterr().out
        assert '97.5% CI' in out
        assert '97% CI' not in out.replace('97.5% CI', '')


class TestModelResultSetstate:
    """ModelResult.__setstate__ keeps base_name a property and fills defaults.

    ``base_name`` used to be a stored field; it is now a read-only @property
    (equal to ``name``). These tests pin that unpickling works and that a stale
    stored ``base_name`` from an old pickle is harmless — the property (a data
    descriptor) wins over an instance ``__dict__`` entry as long as ``base_name``
    stays a setter-less property, which is why the former
    ``state.pop('base_name', None)`` was dead and removed.
    """

    @staticmethod
    def _make(name='coxnet'):
        return ModelResult(
            name=name, model=None,
            cv_metric_scores=np.array([0.70]),
            cv_metric_mean=0.70, cv_metric_std=0.01,
        )

    def test_roundtrip_resolves_base_name(self):
        """Pickle roundtrip preserves name and base_name equals name."""
        import pickle
        restored = pickle.loads(pickle.dumps(self._make('coxnet')))
        assert restored.name == 'coxnet'
        assert restored.base_name == 'coxnet'

    def test_stale_stored_base_name_is_ignored(self):
        """A stored base_name from an old pickle never shadows the property."""
        mr = self._make('coxnet')
        state = mr.__dict__.copy()
        state['base_name'] = 'WRONG'  # Simulate the pre-property stored field
        restored = ModelResult.__new__(ModelResult)
        restored.__setstate__(state)
        assert restored.base_name == 'coxnet'

    def test_missing_field_gets_default(self):
        """Unpickling an old state without a later-added field fills the default."""
        mr = self._make('rsf')
        state = mr.__dict__.copy()
        del state['is_synthetic']  # Simulate a pickle from before this field existed
        restored = ModelResult.__new__(ModelResult)
        restored.__setstate__(state)
        # Assert on the instance dict, not attribute access: simple dataclass
        # defaults live as class attributes, so ``restored.is_synthetic`` would
        # return False via class-attr fallback even if the fill loop never ran.
        # The dict membership is what the loop actually populates (and what the
        # next __getstate__ re-serializes).
        assert 'is_synthetic' in restored.__dict__
        assert restored.is_synthetic is False


# ---------------------------------------------------------------------------
# ModelResult.coefficients
# ---------------------------------------------------------------------------

_COEFFICIENT_COLUMNS = [
    'feature', 'coefficient', 'hazard_ratio',
    'hr_lower_95', 'hr_upper_95', 'p_value',
]


def _make_coef_data(n=80, p=5, seed=0):
    """Small tie-free survival dataset for coefficient tests."""
    from mlsurv.utils import SurvivalArray

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)),
                     columns=[f"f{i}" for i in range(p)])
    risk = 0.8 * X["f0"] - 0.5 * X["f1"]
    time = rng.exponential(scale=np.exp(-risk))  # continuous, tie-free
    event = rng.random(n) < 0.7
    y = SurvivalArray(np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    ))
    return X, y


@pytest.fixture(scope="module")
def _coef_template():
    """Train coxph + coxnet + rsf once; deep-copied per test."""
    from mlsurv import SurvivalLearner

    X, y = _make_coef_data()
    learner = SurvivalLearner(X, y)
    learner.setup(feature_selector=None, random_state=42, n_jobs=1)
    learner.run(["coxph", "coxnet", "rsf"], tune=False, verbose=False)
    return learner


@pytest.fixture
def coef_learner(_coef_template):
    import copy
    return copy.deepcopy(_coef_template)


class TestModelResultCoefficients:
    """ModelResult.coefficients accessor with gated inference."""

    def test_coxph_full_inference(self, coef_learner):
        """CoxPH populates coefficient, HR, CI bounds, and p-value."""
        df = coef_learner.models["coxph"].coefficients
        assert list(df.columns) == _COEFFICIENT_COLUMNS
        assert df["coefficient"].notna().all()
        # HR == exp(coefficient)
        np.testing.assert_allclose(
            df["hazard_ratio"].values, np.exp(df["coefficient"].values)
        )
        # CI brackets the HR; p-value in (0, 1].
        assert (df["hr_lower_95"] < df["hazard_ratio"]).all()
        assert (df["hazard_ratio"] < df["hr_upper_95"]).all()
        assert (df["p_value"] > 0).all()
        assert (df["p_value"] <= 1).all()

    def test_coxnet_gated(self, coef_learner):
        """CoxNet: coefficients present, HR/CI/p-value blank (NaN)."""
        df = coef_learner.models["coxnet"].coefficients
        assert df["coefficient"].notna().any()
        for col in ("hazard_ratio", "hr_lower_95", "hr_upper_95", "p_value"):
            assert df[col].isna().all(), col

    def test_coxnet_preserves_zero_coefficients(self, coef_learner):
        """L1-zeroed coefficients survive as 0.0 rows, not NaN/dropped.

        0.0 means 'in the model, no effect'; NaN means 'selector dropped
        it'. The distinction is load-bearing and must not be lost to a
        truthiness check.
        """
        df = coef_learner.models["coxnet"].coefficients
        # No selector here, so every feature is a row with a real (possibly
        # exactly-zero) coefficient — none should be NaN.
        assert df["coefficient"].notna().all()
        # CoxNet's L1 penalty zeroes at least one coefficient on this data.
        assert (df["coefficient"] == 0.0).any()

    def test_extraction_failure_always_warns_and_returns_empty(self):
        """A get_coefficients() failure surfaces even under warn=False.

        CoxPH fit on an unnamed array raises inside get_coefficients()
        (it indexes by feature_names_in_). That is a genuine failure, not
        a benign "no coefficients" skip, so it must warn regardless of the
        warn flag and return the empty sentinel rather than wrong data.

        The category is MetricFailureWarning rather than InternalWarning
        because the default filters drop InternalWarning, which left this
        failure reaching the user through no channel at all.
        """
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients
        from mlsurv.models import CoxPHModel

        X, y = _make_coef_data(n=120, p=4, seed=5)
        model = CoxPHModel(alpha=0)
        model.fit(X.to_numpy(), y)  # no column names

        # Even with warn=False (the multi-model sweep path), the genuine
        # extraction error is not swallowed.
        with pytest.warns(MetricFailureWarning, match="Could not extract"):
            df = compute_model_coefficients(model, warn=False)
        assert list(df.columns) == _COEFFICIENT_COLUMNS
        assert len(df) == 0

    def test_numpy_fit_falls_back_to_generic_names(self):
        """Unnamed-array fit with a non-string coef index -> feature_{i}.

        Weibull-AFT on a bare numpy array makes lifelines assign integer
        column names, so get_coefficients() succeeds but carries a
        non-string index. With no pipeline names to recover, the accessor
        falls back to generic feature_{i} labels and warns.

        SuggestionWarning, not InternalWarning: the default filters drop
        InternalWarning, and losing the names is actionable (fit on a
        DataFrame).
        """
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients
        from mlsurv.models import WeibullAFTModel

        X, y = _make_coef_data(n=120, p=4, seed=5)
        model = WeibullAFTModel(alpha=0.01)
        model.fit(X.to_numpy(), y)  # no column names

        with pytest.warns(SuggestionWarning, match="generic feature"):
            df = compute_model_coefficients(model)
        assert list(df["feature"]) == [f"feature_{i}" for i in range(4)]

    def test_empty_coefficient_vector_is_a_failure_not_a_skip(self):
        """Returning zero coefficients is not the same as having none.

        None is the documented "this model has no coefficients" answer. An
        empty vector means the model advertised get_coefficients(), was
        asked, and handed back nothing, so it warns even under warn=False
        rather than looking identical to an RSF in the aggregate.
        """
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        class _EmptyCoefModel:
            def get_coefficients(self):
                return pd.Series([], dtype=float)

        with pytest.warns(MetricFailureWarning, match="0 coefficients"):
            df = compute_model_coefficients(_EmptyCoefModel(), warn=False)

        assert list(df.columns) == _COEFFICIENT_COLUMNS
        assert len(df) == 0

    def test_no_coefficient_model_sentinel(self, coef_learner):
        """RSF: 0-row DataFrame with the full schema + SuggestionWarning."""
        from mlsurv._warnings import SuggestionWarning

        with pytest.warns(SuggestionWarning):
            df = coef_learner.models["rsf"].coefficients
        assert list(df.columns) == _COEFFICIENT_COLUMNS
        assert len(df) == 0

    def test_singular_hessian_no_crash(self, coef_learner):
        """Missing covariance: coefficient + HR present, CI/p NaN, warns.

        The tier is ``SuggestionWarning``, not ``InternalWarning``: the default
        filters ignore the latter, so losing every CI and p-value on the
        coefficient table used to reach the user as bare NaN columns. See
        ``tests/test_missing_ci_is_explained.py`` for the default-filter guard.
        """
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.utils import unwrap_pipeline

        inner = unwrap_pipeline(coef_learner.models["coxph"].model)
        # Deterministically simulate a singular Hessian (rank-deficient
        # designs may still invert via the ridge fallback).
        del inner.model_.covariance_matrix_

        with pytest.warns(SuggestionWarning, match="[Cc]ovariance"):
            df = coef_learner.models["coxph"].coefficients
        assert df["coefficient"].notna().all()
        assert df["hazard_ratio"].notna().all()
        assert df["hr_lower_95"].isna().all()
        assert df["hr_upper_95"].isna().all()
        assert df["p_value"].isna().all()

    def test_selector_alignment(self):
        """TopK selector: all pre-selection features present, dropped NaN."""
        from mlsurv import SurvivalLearner

        # 20 features so topk drops some.
        rng = np.random.default_rng(1)
        n, p = 120, 20
        X = pd.DataFrame(rng.normal(size=(n, p)),
                         columns=[f"f{i}" for i in range(p)])
        risk = 0.7 * X["f0"] - 0.6 * X["f1"]
        from mlsurv.utils import SurvivalArray
        time = rng.exponential(scale=np.exp(-risk))
        event = rng.random(n) < 0.7
        y = SurvivalArray(np.array(
            [(bool(e), float(t)) for e, t in zip(event, time)],
            dtype=[("event", bool), ("time", float)],
        ))

        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42, n_jobs=1)
        learner.train("coxph", tune=False)

        df = learner.models["coxph"].coefficients
        # Every pre-selection feature appears as a row.
        assert list(df["feature"]) == list(X.columns)
        # Some features were dropped by the selector (NaN coefficient rows).
        assert df["coefficient"].isna().any()
        # Selected features carry full inference.
        selected = df[df["coefficient"].notna()]
        assert selected["hazard_ratio"].notna().all()

    def test_aft_selector_alignment_labels_match_values(self):
        """AFT under a selector: each coefficient stays on its own feature.

        lifelines returns AFT params in alphabetical order, which differs
        from the selector's emission order, so positional alignment would
        silently mislabel. Align must be by the model's own labels.
        """
        from mlsurv import SurvivalLearner
        from mlsurv.utils import SurvivalArray, unwrap_pipeline

        rng = np.random.default_rng(3)
        n, p = 120, 12
        X = pd.DataFrame(rng.normal(size=(n, p)),
                         columns=[f"f{i}" for i in range(p)])
        risk = 0.7 * X["f0"] - 0.6 * X["f1"]
        time = rng.exponential(scale=np.exp(-risk))
        event = rng.random(n) < 0.7
        y = SurvivalArray(np.array(
            [(bool(e), float(t)) for e, t in zip(event, time)],
            dtype=[("event", bool), ("time", float)],
        ))

        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42, n_jobs=1)
        learner.train("weibull_aft", tune=False)

        result = learner.models["weibull_aft"]
        df = result.coefficients.set_index("feature")
        # Ground truth straight from the model, keyed by feature label.
        truth = unwrap_pipeline(result.model).get_coefficients()

        for feat, coef in truth.items():
            np.testing.assert_allclose(df.loc[feat, "coefficient"], coef)

    # -- label resolution: cases the two surfaces used to answer differently

    @staticmethod
    def _stub_pipeline(all_features, selected, coef_index, coefs):
        """Pipeline with each label source under the test's control.

        The selector step carries the pre-selection names, which is where
        ``_resolve_all_features`` reads them, and reports its own selection.
        The model returns its coefficients under whatever index the case
        needs.
        """
        from sklearn.pipeline import Pipeline

        class _Selector:
            feature_names_in_ = np.array(all_features)
            selected_features_ = list(selected)

        class _Model:
            hazard_ratio_valid = False

            def get_coefficients(self):
                return pd.Series(coefs, index=coef_index)

        return Pipeline([('selector', _Selector()), ('model', _Model())])

    def test_model_order_wins_over_selector_order(self):
        """A reordered coefficient index keys by name, never by position.

        This is the rule the whole shared resolver exists to enforce:
        lifelines sorts Weibull-AFT covariates, so a model can hand back its
        coefficients in an order the selector never emitted. Pinned with a
        stub rather than a real AFT fit because whether lifelines' sort
        actually disagrees with the selector's pick depends on the feature
        names and the seed, which makes a live fit an unreliable guard.
        Mirrors the importance-side test of the same rule.
        """
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        pipeline = self._stub_pipeline(
            ['a', 'b', 'c', 'd'], ['a', 'b', 'c'],
            ['c', 'a', 'b'], [3.0, 1.0, 2.0],
        )
        df = compute_model_coefficients(pipeline).set_index('feature')

        # Positional keying would read a=3.0, b=1.0, c=2.0.
        assert df.loc['a', 'coefficient'] == pytest.approx(1.0)
        assert df.loc['b', 'coefficient'] == pytest.approx(2.0)
        assert df.loc['c', 'coefficient'] == pytest.approx(3.0)
        assert np.isnan(df.loc['d', 'coefficient'])

    def test_all_zero_coefficients_keep_their_rows(self):
        """0.0 is a real coefficient, so the finiteness gate must pass it.

        An all-zero vector is what a fully-penalized L1 fit produces. A
        truthiness or `.any()`-on-values check would drop every row and
        report the model as having no coefficients at all.
        """
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        pipeline = self._stub_pipeline(
            ['a', 'b'], ['a', 'b'], ['a', 'b'], [0.0, 0.0],
        )
        df = compute_model_coefficients(pipeline).set_index('feature')

        assert len(df) == 2
        assert (df['coefficient'] == 0.0).all()

    def test_infinite_coefficients_hit_the_same_sentinel_as_nan(self):
        """isfinite, not notna: an overflowed value is not usable either."""
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        pipeline = self._stub_pipeline(
            ['a', 'b'], ['a', 'b'], ['a', 'b'], [np.inf, -np.inf],
        )

        with pytest.warns(MetricFailureWarning, match="no finite"):
            df = compute_model_coefficients(pipeline)

        assert len(df) == 0

    def test_non_numeric_coefficients_degrade_rather_than_raise(self):
        """The float cast runs inside the guard, like every other failure.

        SurvivalLearner.coefficients loops with no try of its own, so a
        wrapper handing back strings must not cost the other models their
        rows.
        """
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        pipeline = self._stub_pipeline(
            ['a', 'b'], ['a', 'b'], ['a', 'b'], ['not', 'numeric'],
        )

        with pytest.warns(MetricFailureWarning, match="extract"):
            df = compute_model_coefficients(pipeline, warn=False)

        assert list(df.columns) == _COEFFICIENT_COLUMNS
        assert len(df) == 0

    @pytest.mark.parametrize(
        "coef_index, case",
        [(['x0', 'x1'], "names matching no feature"),
         (['a', 1], "a mixed string/integer index")],
    )
    def test_unusable_index_falls_back_to_the_selector(self, coef_index, case):
        """An index that cannot key the features must not corrupt the table.

        Both indices passed the old first-element-only string check and were
        then reindexed onto labels the features did not have, silently and
        with no warning. They failed differently: ``['x0', 'x1']`` matched
        nothing, so the whole table came back NaN, while ``['a', 1]`` kept
        ``a`` and quietly lost ``c``, which is the worse of the two.

        They are also rejected by different guards, so both are worth
        keeping: ``['a', 1]`` fails ``_string_coef_index`` (not every label
        is a string), whereas ``['x0', 'x1']`` passes that and fails the
        intersection check inside ``_resolve_coef_labels``.

        Alignment then falls back to the selector's order. That assumes the
        model did not reorder its coefficients, which holds for an index
        that is positional by construction (sklearn's ``x0, x1``) and is the
        resolver's documented assumption for the rest.
        """
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        pipeline = self._stub_pipeline(
            ['a', 'b', 'c', 'd'], ['a', 'c'], coef_index, [0.5, -0.3],
        )
        df = compute_model_coefficients(pipeline).set_index('feature')

        assert list(df.index) == ['a', 'b', 'c', 'd'], case
        assert df.loc['a', 'coefficient'] == pytest.approx(0.5), case
        assert df.loc['c', 'coefficient'] == pytest.approx(-0.3), case
        # Only the selector-dropped features are blank.
        assert np.isnan(df.loc['b', 'coefficient']), case
        assert np.isnan(df.loc['d', 'coefficient']), case

    def test_alignment_failure_warns_and_returns_empty(self):
        """Duplicate labels degrade to the sentinel instead of raising.

        This used to escape as an uncaught ValueError. It must also warn
        under warn=False: that flag suppresses the benign "no coefficients"
        skip, not a genuine failure.
        """
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        pipeline = self._stub_pipeline(['a', 'b'], ['a', 'a'], [0, 1],
                                       [0.5, -0.3])

        with pytest.warns(MetricFailureWarning, match="align"):
            df = compute_model_coefficients(pipeline, warn=False)

        assert list(df.columns) == _COEFFICIENT_COLUMNS
        assert len(df) == 0

    def test_all_nan_alignment_returns_the_sentinel(self):
        """Nothing finite landed, so report nothing rather than NaN rows.

        A NaN-filled frame passes the ``if not df.empty`` check in
        ``SurvivalLearner.coefficients`` and would reach feature_summary()
        and the exports carrying none of the warning that explains it.
        """
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        # Labels resolve (the selector names them) and align cleanly, but
        # every value is NaN, so no feature carries a usable coefficient.
        pipeline = self._stub_pipeline(
            ['a', 'b'], ['a', 'b'], [0, 1], [np.nan, np.nan],
        )

        with pytest.warns(MetricFailureWarning, match="no finite"):
            df = compute_model_coefficients(pipeline)

        assert list(df.columns) == _COEFFICIENT_COLUMNS
        assert len(df) == 0

    def test_unmatched_labels_are_reported_and_flagged(self):
        """Rows keyed by the model's own vocabulary have to say so.

        The resolver gives up when a string index matches no feature, no
        selector reports a selection, and the lengths differ. The values are
        still worth showing, but the feature column is then the model's
        labels rather than the pipeline's, which the user cannot tell by
        looking.
        """
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients
        from sklearn.pipeline import Pipeline

        class _Selector:  # names the features, reports no selection
            feature_names_in_ = np.array(['a', 'b', 'c'])

        class _Model:
            hazard_ratio_valid = False

            def get_coefficients(self):
                return pd.Series([0.5, -0.3], index=['x0', 'x1'])

        pipeline = Pipeline([('selector', _Selector()), ('model', _Model())])

        with pytest.warns(MetricFailureWarning, match="could not be matched"):
            df = compute_model_coefficients(pipeline, warn=False)

        assert list(df['feature']) == ['x0', 'x1']
        assert df['coefficient'].notna().all()

    def test_named_pipeline_with_unnamed_coefficients_is_not_a_lost_names_skip(
            self):
        """Positional rows under a named pipeline are a placement failure.

        The pipeline knows its feature names here, so telling the user to
        fit on a DataFrame would be false advice. It is also ungated,
        because the rows below are numbered rather than named and would
        otherwise reach feature_summary() looking like real features with no
        way to tell.
        """
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients
        from sklearn.pipeline import Pipeline

        class _Selector:  # names the features, reports no usable selection
            feature_names_in_ = np.array(['a', 'b', 'c'])

        class _Model:
            hazard_ratio_valid = False

            def get_coefficients(self):
                return pd.Series([0.5, -0.3], index=[0, 1])

        pipeline = Pipeline([('selector', _Selector()), ('model', _Model())])

        with pytest.warns(MetricFailureWarning, match="no usable names"):
            df = compute_model_coefficients(pipeline, warn=False)

        # position_, not feature_: this branch only runs when the pipeline has
        # real names, and `feature_0` is a name a real feature can have. The
        # feature_summary() join would silently pair a positional row with a
        # real feature of the same name.
        assert list(df['feature']) == ['position_0', 'position_1']

    def test_the_finiteness_gate_covers_the_unlabeled_paths_too(self):
        """An all-NaN table is nothing computed however its rows got named.

        The gate used to sit inside the aligned branch only, so a model that
        reached the positional path with NaN values still returned a
        NaN-filled frame, which passes the non-empty checks downstream.
        """
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        class _Model:  # no pipeline, so the positional path
            hazard_ratio_valid = False

            def get_coefficients(self):
                return pd.Series([np.nan, np.nan], index=[0, 1])

        with pytest.warns(MetricFailureWarning, match="no finite value"):
            df = compute_model_coefficients(_Model(), warn=False)

        assert len(df) == 0

    def test_generic_names_notice_is_gated_by_warn(self):
        """warn=False silences the benign skips, and only those.

        The aggregate accessor passes warn=False, so the gated half of that
        contract is what it actually exercises. Losing the names is a skip,
        not a failure: the rows are still real, they are just labeled
        positionally.
        """
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        class _Model:  # no pipeline, non-string index, so no names anywhere
            hazard_ratio_valid = False

            def get_coefficients(self):
                return pd.Series([0.5, -0.3], index=[0, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = compute_model_coefficients(_Model(), warn=False)

        assert list(df['feature']) == ['feature_0', 'feature_1']

    def test_wald_matches_lifelines(self):
        """Wald p-values + CIs match a lifelines CoxPHFitter refit."""
        from lifelines import CoxPHFitter
        from mlsurv.models import CoxPHModel

        X, y = _make_coef_data(n=150, p=4, seed=7)

        # Unpenalized mlsurv CoxPH, no scaler — coefficients on raw scale.
        model = CoxPHModel(alpha=0)
        model.fit(X, y)
        assert getattr(model, "alpha_", 0) == 0, "ridge fallback engaged"
        from mlsurv.evaluation._coefficients import compute_model_coefficients
        df = compute_model_coefficients(model).set_index("feature")

        # lifelines reference on the same design (penalizer=0).
        ref = X.copy()
        ref["duration"] = y["time"]
        ref["event"] = y["event"].astype(int)
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(ref, duration_col="duration", event_col="event")
        summ = cph.summary

        for feat in X.columns:
            np.testing.assert_allclose(
                df.loc[feat, "p_value"], summ.loc[feat, "p"], rtol=1e-3,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                df.loc[feat, "hr_lower_95"],
                summ.loc[feat, "exp(coef) lower 95%"], rtol=1e-3,
            )
            np.testing.assert_allclose(
                df.loc[feat, "hr_upper_95"],
                summ.loc[feat, "exp(coef) upper 95%"], rtol=1e-3,
            )


class TestStringCoefIndex:
    """The shared predicate both coefficient surfaces resolve labels with.

    Tested directly because two of its answers are hard to reach through
    either caller, and because getting one of them wrong is what put every
    coefficient on the wrong feature before.
    """

    @pytest.mark.parametrize("index, expected", [
        (['a', 'b'], ['a', 'b']),
        ([], None),
        ([0, 1], None),
        (['a', 1], None),
    ])
    def test_only_an_all_string_index_is_usable(self, index, expected):
        from mlsurv.evaluation.importance import _string_coef_index

        series = pd.Series([1.0] * len(index), index=index, dtype=float)
        assert _string_coef_index(series) == expected

    def test_multiindex_is_rejected(self):
        """lifelines' params_ is a MultiIndex, so this is a live shape.

        Tuples are not strings, so the index cannot key features by name and
        the resolver has to fall back rather than align tuples positionally.
        The Weibull-AFT wrapper flattens to the lambda_ block before we ever
        see it, which is what keeps that model on the by-name path.
        """
        from mlsurv.evaluation.importance import _string_coef_index

        series = pd.Series([1.0, 2.0], index=pd.MultiIndex.from_tuples(
            [('lambda_', 'b'), ('lambda_', 'a')],
        ))
        assert _string_coef_index(series) is None

    def test_a_bare_array_has_no_index(self):
        from mlsurv.evaluation.importance import _string_coef_index

        assert _string_coef_index(np.array([1.0, 2.0])) is None

    @pytest.mark.parametrize("returned", [[0.5, 0.3], (0.5, 0.3)])
    def test_a_list_or_tuple_index_attribute_is_not_an_index(self, returned):
        """`list.index` is a METHOD, so a name check alone would crash.

        A wrapper returning a bare list used to reach `len(index)` on the
        bound method and raise TypeError, which escaped every guard and took
        down the whole aggregate table.
        """
        from mlsurv.evaluation.importance import _string_coef_index

        assert _string_coef_index(returned) is None

    def test_a_bare_list_return_degrades_rather_than_crashing(self):
        """The same shape, end to end through the accessor."""
        from mlsurv.evaluation._coefficients import compute_model_coefficients

        class _Model:
            hazard_ratio_valid = False

            def get_coefficients(self):
                return [0.5, -0.3]

        df = compute_model_coefficients(_Model(), warn=False)
        assert list(df['feature']) == ['feature_0', 'feature_1']
        assert df['coefficient'].tolist() == [0.5, -0.3]


class TestLearnerCoefficients:
    """SurvivalLearner.coefficients long-format accessor."""

    def test_long_format(self, coef_learner):
        """Long format with model_type first; gating per model; RSF omitted."""
        df = coef_learner.coefficients
        assert df.columns[0] == "model_type"
        assert list(df.columns) == ["model_type", *_COEFFICIENT_COLUMNS]
        assert set(df["model_type"]) == {"coxph", "coxnet"}  # rsf omitted
        coxph_rows = df[df["model_type"] == "coxph"]
        coxnet_rows = df[df["model_type"] == "coxnet"]
        assert coxph_rows["p_value"].notna().all()
        assert coxnet_rows["p_value"].isna().all()

    def test_no_warning_spam_for_noncoef_models(self, coef_learner):
        """Accessing learner.coefficients with an RSF model present is silent.

        The aggregate calls compute_model_coefficients(warn=False) so that
        coefficient-less models (RSF) do not spray SuggestionWarnings on a
        routine property access.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning -> failure
            df = coef_learner.coefficients
        assert set(df["model_type"]) == {"coxph", "coxnet"}

    def test_one_failing_model_keeps_the_others(self, coef_learner):
        """The aggregate loops every model with no guard of its own.

        A coefficient failure on one model used to raise out of the loop and
        cost every other model its rows, and the report the whole feature
        table with them.
        """
        from mlsurv._warnings import MetricFailureWarning

        coef_learner.models["coxnet"].model = (
            TestModelResultCoefficients._stub_pipeline(
                ["a", "b"], ["a", "a"], [0, 1], [0.5, -0.3],
            )
        )

        # Named by its registry key, not its wrapper class: that is the name
        # the user asked for and the one the model_type column shows.
        with pytest.warns(MetricFailureWarning, match="coxnet"):
            df = coef_learner.coefficients

        assert set(df["model_type"]) == {"coxph"}
        assert df["coefficient"].notna().all()

    def test_a_dropped_model_does_not_break_feature_summary(self, coef_learner):
        """feature_summary() is where the dropped rows actually show up.

        It joins the coefficient table onto the per-feature summary, so a
        model returning no rows has to leave the other models' columns
        intact rather than misaligning the join.
        """
        from mlsurv._warnings import MetricFailureWarning

        coef_learner.models["coxnet"].model = (
            TestModelResultCoefficients._stub_pipeline(
                ["a", "b"], ["a", "a"], [0, 1], [0.5, -0.3],
            )
        )

        with pytest.warns(MetricFailureWarning):
            summary = coef_learner.feature_summary()

        assert not summary.empty
        # coxph still carries its multivariable coefficients.
        coxph_rows = summary[summary["model_type"] == "coxph"]
        assert coxph_rows["mv_coefficient"].notna().any()

    def test_empty_before_training(self):
        """No trained models: 0-row DataFrame with full schema."""
        from mlsurv import SurvivalLearner

        X, y = _make_coef_data()
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=42, n_jobs=1)

        df = learner.coefficients
        assert list(df.columns) == ["model_type", *_COEFFICIENT_COLUMNS]
        assert len(df) == 0

    def test_reflects_retrain(self, coef_learner):
        """learner.coefficients tracks the current trained-model set."""
        before = set(coef_learner.coefficients["model_type"])
        assert "coxph" in before
        # Drop coxph and confirm it disappears from the live property.
        del coef_learner._state.models["coxph"]
        after = set(coef_learner.coefficients["model_type"])
        assert "coxph" not in after
