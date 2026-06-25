"""Tests for the results architecture refactor.

Tests for:
- EvaluationResult.combine() classmethod
- BootstrapResult.combine() classmethod
- SurvivalLearner computed properties (evaluation_result, bootstrap_result, best_model)
- SurvivalLearner.summary() method
"""

import numpy as np
import pandas as pd
import pytest

from mlsurv.results.containers import (
    BootstrapResult,
    EvaluationResult,
    ModelResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval(model_name, c_test=0.7):
    """Create a minimal single-model EvaluationResult."""
    return EvaluationResult(
        concordance=pd.DataFrame({
            'model_type': [model_name],
            'c_index_train': [0.8],
            'c_index_test': [c_test],
        }),
        test_set=pd.DataFrame({
            'model_type': [model_name],
            'index': [0], 'risk_score': [0.5],
            'time': [1.0], 'event': [1],
        }),
        auc_t=pd.DataFrame({
            'model_type': [model_name],
            'eval_time': [1.0], 'auc': [0.7],
        }),
        importance=pd.DataFrame({
            'model_type': [model_name],
            'feature': ['age'], 'coefficient': [0.1],
            'perm_mean': [0.05], 'perm_std': [0.01],
        }),
        importance_iterations=pd.DataFrame({
            'model_type': [model_name],
            'feature': ['age'], 'iteration': [0], 'importance': [0.05],
        }),
    )


def _make_boot(model_name, metric_mean=0.72):
    """Create a minimal single-model BootstrapResult."""
    return BootstrapResult(
        summary=pd.DataFrame({
            'model_type': [model_name],
            'metric': ['harrell'],
            'point': [metric_mean], 'mean': [metric_mean],
            'std': [0.02],
            'ci_lower': [metric_mean - 0.05],
            'ci_upper': [metric_mean + 0.05],
        }),
        iterations=pd.DataFrame({
            'model_type': [model_name] * 3,
            'iteration': [0, 1, 2],
            'metric_name': ['harrell'] * 3,
            'value': [metric_mean - 0.01, metric_mean, metric_mean + 0.01],
        }),
    )


def _make_model(name, cv_mean=0.72, evaluation=None, bootstrap=None):
    """Create a minimal ModelResult."""
    return ModelResult(
        name=name,
        model=None,
        cv_metric_scores=np.array([cv_mean - 0.01, cv_mean, cv_mean + 0.01]),
        cv_metric_mean=cv_mean,
        cv_metric_std=0.01,
        evaluation=evaluation,
        bootstrap=bootstrap,
    )


# ---------------------------------------------------------------------------
# EvaluationResult.combine()
# ---------------------------------------------------------------------------


class TestEvaluationResultCombine:

    def test_two_models(self):
        """combine() concatenates DataFrames from two models."""
        models = {
            'rsf': _make_model('rsf', evaluation=_make_eval('rsf', 0.75)),
            'coxph': _make_model('coxph', evaluation=_make_eval('coxph', 0.70)),
        }
        result = EvaluationResult.combine(models)

        assert len(result.concordance) == 2
        assert set(result.concordance['model_type']) == {'rsf', 'coxph'}
        assert len(result.test_set) == 2
        assert len(result.auc_t) == 2
        assert len(result.importance) == 2
        assert len(result.importance_iterations) == 2

    def test_model_order_preserved(self):
        """combine() sets model_order from dict key order."""
        models = {
            'rsf': _make_model('rsf', evaluation=_make_eval('rsf')),
            'coxph': _make_model('coxph', evaluation=_make_eval('coxph')),
        }
        result = EvaluationResult.combine(models)
        assert result.model_order == ['rsf', 'coxph']

    def test_empty_dict_raises(self):
        """combine() raises ValueError on empty dict."""
        with pytest.raises(ValueError, match="No models have evaluation"):
            EvaluationResult.combine({})

    def test_no_evaluations_raises(self):
        """combine() raises ValueError when all evaluations are None."""
        models = {
            'rsf': _make_model('rsf'),  # no evaluation
            'coxph': _make_model('coxph'),  # no evaluation
        }
        with pytest.raises(ValueError, match="No models have evaluation"):
            EvaluationResult.combine(models)

    def test_partial_evaluations(self):
        """combine() includes only models with evaluations."""
        models = {
            'rsf': _make_model('rsf', evaluation=_make_eval('rsf')),
            'coxph': _make_model('coxph'),  # no evaluation
        }
        result = EvaluationResult.combine(models)
        assert len(result.concordance) == 1
        assert result.concordance['model_type'].iloc[0] == 'rsf'

    def test_shap_merge(self):
        """combine() merges SHAP from models that have it."""
        eval_with_shap = _make_eval('rsf')
        eval_with_shap.shap_values = pd.DataFrame({
            'model_type': ['rsf'], 'feature': ['age'],
            'sample_idx': [0], 'shap_value': [0.1],
        })
        models = {
            'rsf': _make_model('rsf', evaluation=eval_with_shap),
            'coxph': _make_model('coxph', evaluation=_make_eval('coxph')),
        }
        result = EvaluationResult.combine(models)
        assert result.shap_values is not None
        assert len(result.shap_values) == 1

    def test_surv_probs_cache_merge(self):
        """combine() merges _surv_probs_cache from all models."""
        eval1 = _make_eval('rsf')
        eval1._surv_probs_cache = {'rsf': (np.array([0.9]), np.array([1.0]))}
        eval2 = _make_eval('coxph')
        eval2._surv_probs_cache = {'coxph': (np.array([0.8]), np.array([1.0]))}
        models = {
            'rsf': _make_model('rsf', evaluation=eval1),
            'coxph': _make_model('coxph', evaluation=eval2),
        }
        result = EvaluationResult.combine(models)
        assert result._surv_probs_cache is not None
        assert 'rsf' in result._surv_probs_cache
        assert 'coxph' in result._surv_probs_cache

    def test_empty_dataframes_handled(self):
        """combine() handles models with empty DataFrames."""
        eval_empty = EvaluationResult(
            concordance=pd.DataFrame(),
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
        )
        models = {
            'rsf': _make_model('rsf', evaluation=eval_empty),
            'coxph': _make_model('coxph', evaluation=_make_eval('coxph')),
        }
        result = EvaluationResult.combine(models)
        assert len(result.concordance) == 1  # only coxph


# ---------------------------------------------------------------------------
# BootstrapResult.combine()
# ---------------------------------------------------------------------------


class TestBootstrapResultCombine:

    def test_two_models_standalone(self):
        """combine() concatenates standalone bootstraps from two models."""
        models = {
            'rsf': _make_model('rsf', bootstrap=_make_boot('rsf', 0.75)),
            'coxph': _make_model('coxph', bootstrap=_make_boot('coxph', 0.70)),
        }
        result = BootstrapResult.combine(models)

        assert result is not None
        assert len(result.summary) == 2
        assert set(result.summary['model_type']) == {'rsf', 'coxph'}
        assert len(result.iterations) == 6  # 3 per model

    def test_no_bootstrap_returns_none(self):
        """combine() returns None when no models have bootstrap."""
        models = {
            'rsf': _make_model('rsf'),
            'coxph': _make_model('coxph'),
        }
        assert BootstrapResult.combine(models) is None

    def test_embedded_bootstrap(self):
        """combine() finds bootstrap embedded in evaluation."""
        eval_with_boot = _make_eval('rsf')
        eval_with_boot.bootstrap = _make_boot('rsf')
        models = {
            'rsf': _make_model('rsf', evaluation=eval_with_boot),
        }
        result = BootstrapResult.combine(models)
        assert result is not None
        assert len(result.summary) == 1

    def test_standalone_preferred_over_embedded(self):
        """combine() prefers standalone .bootstrap over .evaluation.bootstrap."""
        eval_with_boot = _make_eval('rsf')
        eval_with_boot.bootstrap = _make_boot('rsf', 0.60)
        standalone = _make_boot('rsf', 0.75)
        models = {
            'rsf': _make_model('rsf', evaluation=eval_with_boot, bootstrap=standalone),
        }
        result = BootstrapResult.combine(models)
        assert result is not None
        # Should use standalone (0.75), not embedded (0.60)
        assert result.summary['mean'].iloc[0] == pytest.approx(0.75)

    def test_lower_is_better_union(self):
        """combine() merges lower_is_better as union."""
        boot1 = _make_boot('rsf')
        boot1.lower_is_better = {'integrated_brier'}
        boot2 = _make_boot('coxph')
        boot2.lower_is_better = {'integrated_brier', 'brier'}
        models = {
            'rsf': _make_model('rsf', bootstrap=boot1),
            'coxph': _make_model('coxph', bootstrap=boot2),
        }
        result = BootstrapResult.combine(models)
        assert result.lower_is_better == {'integrated_brier', 'brier'}

    def test_lower_is_better_empty_set_preserved(self):
        """An explicit empty lower_is_better set() round-trips, not collapsed to None."""
        boot1 = _make_boot('rsf')
        boot1.lower_is_better = set()
        boot2 = _make_boot('coxph')
        boot2.lower_is_better = set()
        models = {
            'rsf': _make_model('rsf', bootstrap=boot1),
            'coxph': _make_model('coxph', bootstrap=boot2),
        }
        result = BootstrapResult.combine(models)
        assert result.lower_is_better == set()
        assert result.lower_is_better is not None

    def test_lower_is_better_none_when_unset(self):
        """Stays None when no bootstrap sets lower_is_better."""
        models = {
            'rsf': _make_model('rsf', bootstrap=_make_boot('rsf')),
            'coxph': _make_model('coxph', bootstrap=_make_boot('coxph')),
        }
        result = BootstrapResult.combine(models)
        assert result.lower_is_better is None

    @pytest.mark.parametrize('lib1,lib2', [
        (None, {'brier'}),       # first bootstrap unset, second sets it
        ({'brier'}, None),       # order-independence
        (set(), {'brier'}),      # empty-set carried into the union must not drop 'brier'
    ])
    def test_lower_is_better_mixed_none_and_set(self, lib1, lib2):
        """Union is correct across mixed None / set inputs in any order.

        Guards the `(merged_lib or set()) | b.lower_is_better` line against a
        future simplification to `merged_lib | b.lower_is_better`, which would
        crash on the None|set path or drop elements.
        """
        boot1 = _make_boot('rsf')
        boot1.lower_is_better = lib1
        boot2 = _make_boot('coxph')
        boot2.lower_is_better = lib2
        models = {
            'rsf': _make_model('rsf', bootstrap=boot1),
            'coxph': _make_model('coxph', bootstrap=boot2),
        }
        result = BootstrapResult.combine(models)
        assert result.lower_is_better == {'brier'}

    def test_lower_is_better_single_model_empty_set(self):
        """Single-model combine preserves an explicit empty set."""
        boot = _make_boot('rsf')
        boot.lower_is_better = set()
        result = BootstrapResult.combine({'rsf': _make_model('rsf', bootstrap=boot)})
        assert result.lower_is_better == set()
        assert result.lower_is_better is not None

    def test_partial_optional_fields(self):
        """combine() handles one model with predictions, one without."""
        boot1 = _make_boot('rsf')
        boot1.predictions = pd.DataFrame({
            'model_type': ['rsf'], 'iteration': [0],
            'index': [0], 'risk_score': [0.5],
        })
        boot2 = _make_boot('coxph')
        models = {
            'rsf': _make_model('rsf', bootstrap=boot1),
            'coxph': _make_model('coxph', bootstrap=boot2),
        }
        result = BootstrapResult.combine(models)
        assert result.predictions is not None
        assert len(result.predictions) == 1  # only from rsf

    def test_none_summary_handled(self):
        """combine() returns None when all bootstrap data is empty."""
        boot = BootstrapResult(summary=None, iterations=None)
        models = {'rsf': _make_model('rsf', bootstrap=boot)}
        result = BootstrapResult.combine(models)
        # Bootstrap with no summary or iterations is effectively no bootstrap
        assert result is None


# ---------------------------------------------------------------------------
# Computed properties
# ---------------------------------------------------------------------------


class TestLearnerComputedProperties:
    """Test computed properties on SurvivalLearner."""

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        n = 50
        X = pd.DataFrame({
            'age': np.random.normal(60, 10, n),
            'stage': np.random.randint(1, 4, n),
            'biomarker': np.random.random(n),
        })
        y = np.array(
            [(bool(e), t) for e, t in
             zip(np.random.binomial(1, 0.6, n),
                 np.random.exponential(2, n))],
            dtype=[('event', '?'), ('time', '<f8')],
        )
        return X, y

    def test_evaluation_result_after_evaluate(self, small_data):
        """evaluation_result property returns combined result after evaluate()."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train('coxph', verbose=False)
        learner.evaluate(verbose=False)

        er = learner.evaluation_result
        assert er is not None
        assert hasattr(er, 'concordance')
        assert len(er.concordance) == 1

    def test_evaluation_result_none_before_evaluate(self, small_data):
        """evaluation_result is None before evaluate()."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train('coxph', verbose=False)

        assert learner.evaluation_result is None

    def test_evaluation_result_multi_model(self, small_data):
        """evaluation_result combines evaluations from multiple models."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train('coxph', verbose=False)
        learner.train('rsf', verbose=False)
        learner.evaluate(verbose=False)

        er = learner.evaluation_result
        assert er is not None
        assert len(er.concordance) == 2
        models = set(er.concordance['model_type'])
        assert 'coxph' in models
        assert 'rsf' in models

    def test_best_model_property(self, small_data):
        """best_model returns model with highest CV metric mean."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train('coxph', verbose=False)
        learner.train('rsf', verbose=False)

        best = learner.best_model
        assert best is not None
        assert best.name in ('coxph', 'rsf')
        # Should be the one with highest cv_metric_mean
        other_name = 'coxph' if best.name == 'rsf' else 'rsf'
        other = learner.models[other_name]
        assert best.cv_metric_mean >= other.cv_metric_mean

    def test_best_model_none_when_empty(self, small_data):
        """best_model is None when no models trained."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        assert learner.best_model is None

    def test_summary_method(self, small_data):
        """summary() returns DataFrame with all models."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train('coxph', verbose=False)
        learner.train('rsf', verbose=False)

        df = learner.summary()
        assert isinstance(df, pd.DataFrame)
        assert 'model' in df.columns
        assert set(df['model'].unique()) == {'coxph', 'rsf'}

    def test_summary_empty_when_no_models(self, small_data):
        """summary() returns empty DataFrame when no models trained."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)

        df = learner.summary()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_bootstrap_result_after_bootstrap(self, small_data):
        """bootstrap_result returns combined result after bootstrap()."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train('coxph', verbose=False)
        learner.bootstrap(n_bootstrap=2, verbose=False, show_progress=False)

        br = learner.bootstrap_result
        assert br is not None
        assert hasattr(br, 'summary')

    def test_bootstrap_result_none_before_bootstrap(self, small_data):
        """bootstrap_result is None before bootstrap()."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train('coxph', verbose=False)

        assert learner.bootstrap_result is None


# ---------------------------------------------------------------------------
# Step 1: combine() edge cases
# ---------------------------------------------------------------------------


class TestCombineEdgeCases:
    """Edge case tests for EvaluationResult.combine() and BootstrapResult.combine()."""

    def test_single_model_combine(self):
        """combine() works with a single-model dict."""
        models = {'rsf': _make_model('rsf', evaluation=_make_eval('rsf', 0.75))}
        result = EvaluationResult.combine(models)
        assert len(result.concordance) == 1
        assert result.model_order == ['rsf']

    def test_mismatched_columns(self):
        """combine() handles models with different DataFrame columns via concat NaN fill."""
        eval1 = _make_eval('rsf', 0.75)
        eval1.concordance['extra_col'] = [1.0]
        eval2 = _make_eval('coxph', 0.70)
        models = {
            'rsf': _make_model('rsf', evaluation=eval1),
            'coxph': _make_model('coxph', evaluation=eval2),
        }
        result = EvaluationResult.combine(models)
        assert len(result.concordance) == 2
        assert 'extra_col' in result.concordance.columns
        # coxph row should have NaN for extra_col
        coxph_row = result.concordance[result.concordance['model_type'] == 'coxph']
        assert pd.isna(coxph_row['extra_col'].iloc[0])

    def test_different_shap_features(self):
        """combine() handles different SHAP feature sets across models."""
        eval1 = _make_eval('rsf')
        eval1.shap_values = pd.DataFrame({
            'model_type': ['rsf', 'rsf'],
            'feature': ['age', 'stage'],
            'sample_idx': [0, 0],
            'shap_value': [0.1, 0.2],
        })
        eval2 = _make_eval('coxph')
        eval2.shap_values = pd.DataFrame({
            'model_type': ['coxph', 'coxph'],
            'feature': ['age', 'biomarker'],
            'sample_idx': [0, 0],
            'shap_value': [0.3, 0.4],
        })
        models = {
            'rsf': _make_model('rsf', evaluation=eval1),
            'coxph': _make_model('coxph', evaluation=eval2),
        }
        result = EvaluationResult.combine(models)
        assert result.shap_values is not None
        assert len(result.shap_values) == 4
        features = set(result.shap_values['feature'])
        assert features == {'age', 'stage', 'biomarker'}

    def test_single_model_bootstrap_combine(self):
        """BootstrapResult.combine() works with a single-model dict."""
        models = {'rsf': _make_model('rsf', bootstrap=_make_boot('rsf', 0.75))}
        result = BootstrapResult.combine(models)
        assert result is not None
        assert len(result.summary) == 1

    def test_mismatched_iteration_columns(self):
        """combine() handles bootstrap with different optional columns."""
        boot1 = _make_boot('rsf')
        boot1.predictions = pd.DataFrame({
            'model_type': ['rsf'], 'iteration': [0],
            'index': [0], 'risk_score': [0.5],
        })
        boot2 = _make_boot('coxph')
        boot2.importance = pd.DataFrame({
            'model_type': ['coxph'], 'iteration': [0],
            'feature': ['age'], 'importance': [0.1],
        })
        models = {
            'rsf': _make_model('rsf', bootstrap=boot1),
            'coxph': _make_model('coxph', bootstrap=boot2),
        }
        result = BootstrapResult.combine(models)
        assert result.predictions is not None
        assert result.importance is not None


# ---------------------------------------------------------------------------
# Step 2: resolved_bootstrap
# ---------------------------------------------------------------------------


class TestResolvedBootstrap:

    def test_resolved_bootstrap_standalone(self):
        """resolved_bootstrap returns standalone .bootstrap."""
        boot = _make_boot('rsf')
        mr = _make_model('rsf', bootstrap=boot)
        assert mr.resolved_bootstrap is boot

    def test_resolved_bootstrap_embedded(self):
        """resolved_bootstrap finds bootstrap embedded in evaluation."""
        boot = _make_boot('rsf')
        eval_r = _make_eval('rsf')
        eval_r.bootstrap = boot
        mr = _make_model('rsf', evaluation=eval_r)
        assert mr.resolved_bootstrap is boot

    def test_resolved_bootstrap_prefers_standalone(self):
        """resolved_bootstrap prefers standalone over embedded."""
        standalone = _make_boot('rsf', 0.80)
        embedded = _make_boot('rsf', 0.60)
        eval_r = _make_eval('rsf')
        eval_r.bootstrap = embedded
        mr = _make_model('rsf', evaluation=eval_r, bootstrap=standalone)
        assert mr.resolved_bootstrap is standalone
        assert mr.resolved_bootstrap.summary['mean'].iloc[0] == pytest.approx(0.80)

    def test_resolved_bootstrap_none(self):
        """resolved_bootstrap returns None when no bootstrap exists."""
        mr = _make_model('rsf')
        assert mr.resolved_bootstrap is None

    def test_evaluate_hoists_bootstrap(self, small_data):
        """evaluate() with bootstrap hoists .evaluation.bootstrap to .bootstrap."""
        from mlsurv import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train('coxph', verbose=False)
        learner.evaluate(bootstrap=True, n_bootstrap=2, verbose=False, show_progress=False)

        mr = learner.models['coxph']
        # Bootstrap should be hoisted to top-level
        assert mr.bootstrap is not None
        assert mr.evaluation.bootstrap is not None

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        n = 50
        X = pd.DataFrame({
            'age': np.random.normal(60, 10, n),
            'stage': np.random.randint(1, 4, n),
            'biomarker': np.random.random(n),
        })
        y = np.array(
            [(bool(e), t) for e, t in
             zip(np.random.binomial(1, 0.6, n),
                 np.random.exponential(2, n))],
            dtype=[('event', '?'), ('time', '<f8')],
        )
        return X, y


# ---------------------------------------------------------------------------
# Step 4: Compute-on-read aggregates
# ---------------------------------------------------------------------------


class TestComputeOnRead:
    """evaluation_result/bootstrap_result are computed on each access (no
    cache), so they always reflect the current per-model state. There is no
    `_invalidate_result_caches()` to remember."""

    def test_recomputes_on_each_access(self):
        """Two accesses without mutation return equal content."""
        from mlsurv.learner._state import LearnerState
        from mlsurv.learner._core import SurvivalLearner

        mr = _make_model('rsf', evaluation=_make_eval('rsf'))
        state = LearnerState()
        state.models['rsf'] = mr

        # Create a minimal learner with injected state
        learner = object.__new__(SurvivalLearner)
        learner._state = state

        er1 = learner.evaluation_result
        er2 = learner.evaluation_result
        assert er1 is not None and er2 is not None
        assert len(er1.concordance) == len(er2.concordance) == 1

    def test_reflects_new_model(self):
        """Adding a new model is reflected on the next read."""
        from mlsurv.learner._state import LearnerState
        from mlsurv.learner._core import SurvivalLearner

        mr1 = _make_model('rsf', evaluation=_make_eval('rsf', 0.75))
        state = LearnerState()
        state.models['rsf'] = mr1

        learner = object.__new__(SurvivalLearner)
        learner._state = state

        er1 = learner.evaluation_result
        assert er1 is not None
        assert len(er1.concordance) == 1

        # Add a second model — no manual invalidation needed
        mr2 = _make_model('coxph', evaluation=_make_eval('coxph', 0.70))
        state.models['coxph'] = mr2

        er2 = learner.evaluation_result
        assert len(er2.concordance) == 2

    def test_reflects_in_place_evaluation(self):
        """Storing evaluation in place on a ModelResult is reflected on read.

        This is the regression the cache removal protects: a code path that
        sets ``mr.evaluation = ...`` no longer has to remember to invalidate.
        """
        from mlsurv.learner._state import LearnerState
        from mlsurv.learner._core import SurvivalLearner

        mr = _make_model('rsf')  # no evaluation
        state = LearnerState()
        state.models['rsf'] = mr

        learner = object.__new__(SurvivalLearner)
        learner._state = state

        assert learner.evaluation_result is None

        # Store evaluation in place — next read reflects it with no invalidation
        mr.evaluation = _make_eval('rsf')

        er = learner.evaluation_result
        assert er is not None
        assert len(er.concordance) == 1
        assert er.concordance['model_type'].iloc[0] == 'rsf'

    def test_reflects_in_place_bootstrap(self):
        """Storing bootstrap in place on a ModelResult is reflected on read.

        bootstrap_result is a separate compute-on-read path from
        evaluation_result (it goes through ``_has_any_bootstrap`` +
        ``BootstrapResult.combine``), so it needs its own coverage.
        """
        from mlsurv.learner._state import LearnerState
        from mlsurv.learner._core import SurvivalLearner

        mr = _make_model('rsf')  # no bootstrap
        state = LearnerState()
        state.models['rsf'] = mr

        learner = object.__new__(SurvivalLearner)
        learner._state = state

        assert learner.bootstrap_result is None

        # Store bootstrap in place — next read reflects it with no invalidation
        mr.bootstrap = _make_boot('rsf')

        br = learner.bootstrap_result
        assert br is not None
        assert len(br.summary) == 1
        assert br.summary['model_type'].iloc[0] == 'rsf'


# ---------------------------------------------------------------------------
# Step 5: Synthetic models
# ---------------------------------------------------------------------------


class TestSyntheticModels:

    def _make_learner_with_models(self):
        """Create a SurvivalLearner with injected models (no real data needed)."""
        from mlsurv.learner._state import LearnerState
        from mlsurv.learner._core import SurvivalLearner

        real = _make_model('rsf', cv_mean=0.75, evaluation=_make_eval('rsf', 0.75))
        synthetic = ModelResult(
            name='external_ref',
            model=None,
            cv_metric_scores=np.array([]),
            cv_metric_mean=np.nan,
            cv_metric_std=np.nan,
            evaluation=_make_eval('external_ref', 0.90),
            is_synthetic=True,
        )
        state = LearnerState()
        state.models['rsf'] = real
        state.models['external_ref'] = synthetic

        learner = object.__new__(SurvivalLearner)
        learner._state = state
        return learner

    def test_best_model_excludes_synthetic(self):
        """best_model should not return synthetic models."""
        learner = self._make_learner_with_models()
        best = learner.best_model
        assert best is not None
        assert best.name == 'rsf'
        assert not best.is_synthetic

    def test_summary_excludes_synthetic(self):
        """summary() should not include synthetic models."""
        learner = self._make_learner_with_models()
        df = learner.summary()
        assert 'external_ref' not in df['model'].values
        assert 'rsf' in df['model'].values

    def test_evaluation_result_includes_synthetic(self):
        """evaluation_result should include synthetic models (needed for benchmark plots)."""
        learner = self._make_learner_with_models()
        er = learner.evaluation_result
        assert er is not None
        models_in_er = set(er.concordance['model_type'])
        assert 'rsf' in models_in_er
        assert 'external_ref' in models_in_er
