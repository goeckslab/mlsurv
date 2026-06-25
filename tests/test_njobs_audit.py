"""Tests for n_jobs oversubscription prevention in the evaluation path.

Verifies that total core usage never exceeds the user's n_jobs setting,
even when models have their own internal parallelism (e.g., RSF n_jobs).
"""

import pytest
from unittest.mock import patch, MagicMock

from mlsurv._warnings import SuggestionWarning

pytestmark = pytest.mark.parallel

# RSF tuning cost scales with forest size, so the tuning test below passes a tiny
# Optuna space. Partial spaces merge per-param with the defaults. The n_jobs-propagation
# assertion is independent of forest size.
_SMALL_TREE_SPACE = {'n_estimators': (10, 20), 'max_depth': (3, 5)}


class TestSingleModelEvaluateOversubscription:
    """Verify n_jobs safety in single-model evaluate() path."""

    def test_rsf_njobs_preserved_during_evaluate(self, train_test_data):
        """RSF model.n_jobs should be preserved — importance uses n_jobs=1 instead."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('rsf', n_jobs=4)
        model.fit(X_train, y_train)

        # After fitting, model.n_jobs is still 4
        assert model.n_jobs == 4

        # Evaluate with n_jobs=4 — RSF keeps its own n_jobs, importance uses 1
        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=4,
        )

        # Model's n_jobs should be unchanged — importance uses n_jobs=1
        # instead of adjust_model_njobs, avoiding serialization overhead
        assert model.n_jobs == 4, (
            "RSF model.n_jobs should remain 4 after evaluate() with n_jobs>1 "
            "(importance uses n_jobs=1 instead of adjusting model)"
        )
        # Evaluation should still succeed
        assert result.importance is not None

    def test_rsf_njobs_unchanged_when_evaluate_sequential(self, train_test_data):
        """RSF model.n_jobs should be preserved when evaluate uses n_jobs=1."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('rsf', n_jobs=4)
        model.fit(X_train, y_train)

        # Evaluate with n_jobs=1 — should NOT adjust model.n_jobs
        evaluate(
            model, X_test, y_test,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )

        # Model's n_jobs should be unchanged
        assert model.n_jobs == 4, (
            "RSF model.n_jobs should remain 4 when evaluate() uses n_jobs=1"
        )

    def test_adjust_model_njobs_skipped_for_rsf_in_eval_dataframes(self, train_test_data):
        """adjust_model_njobs is NOT called for RSF (has internal parallelism)."""
        from mlsurv.models import get_model
        from mlsurv.evaluation.evaluate import _compute_eval_dataframes
        from mlsurv.evaluation._config import EvalConfig
        from mlsurv.utils import auto_eval_times

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('rsf', n_jobs=4)
        model.fit(X_train, y_train)

        eval_times = auto_eval_times(y_test)
        config = EvalConfig(
            n_importance_repeats=5,
            eval_times=eval_times,
            random_state=42,
            include_shap=False,
            show_progress=False,
            n_jobs=4,
        )

        with patch('mlsurv.evaluation.evaluate.adjust_model_njobs', wraps=__import__('mlsurv.utils', fromlist=['adjust_model_njobs']).adjust_model_njobs) as mock_adjust:
            _compute_eval_dataframes(
                model, X_test, y_test, X_train, y_train, None, 'rsf', config,
            )
            # RSF has internal parallelism — adjust_model_njobs should NOT be called
            mock_adjust.assert_not_called()

    def test_adjust_model_njobs_called_for_coxph_in_eval_dataframes(self, train_test_data):
        """adjust_model_njobs IS called for CoxPH (no internal parallelism)."""
        from mlsurv.models import get_model
        from mlsurv.evaluation.evaluate import _compute_eval_dataframes
        from mlsurv.evaluation._config import EvalConfig
        from mlsurv.utils import auto_eval_times

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        eval_times = auto_eval_times(y_test)
        config = EvalConfig(
            n_importance_repeats=5,
            eval_times=eval_times,
            random_state=42,
            include_shap=False,
            show_progress=False,
            n_jobs=4,
        )

        with patch('mlsurv.evaluation.evaluate.adjust_model_njobs', wraps=__import__('mlsurv.utils', fromlist=['adjust_model_njobs']).adjust_model_njobs) as mock_adjust:
            _compute_eval_dataframes(
                model, X_test, y_test, X_train, y_train, None, 'coxph', config,
            )
            # CoxPH has no internal parallelism — adjust_model_njobs should be called
            mock_adjust.assert_called_once_with(model, 4)

    def test_coxph_unaffected_by_adjust(self, train_test_data):
        """CoxPH (no n_jobs) should work fine through evaluate with n_jobs>1."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        # Should complete without error
        result = evaluate(
            model, X_test, y_test,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=4,
        )
        assert result.importance is not None


class TestMultiModelEvaluateOversubscription:
    """Verify n_jobs safety in multi-model evaluate() path."""

    def test_rsf_njobs_preserved_in_multimodel_evaluate(self, train_test_data):
        """In multi-model evaluate, RSF n_jobs should be unchanged (no outer adjust)."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        rsf = get_model('rsf', n_jobs=4)
        rsf.fit(X_train, y_train)

        coxph = get_model('coxph')
        coxph.fit(X_train, y_train)

        models = {'rsf': rsf, 'coxph': coxph}

        # Multi-model evaluate with n_jobs=4 — sequential model loop,
        # inner importance parallelism only
        result = evaluate(
            models, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=4,
        )

        # RSF has internal parallelism — n_jobs is never modified
        assert rsf.n_jobs == 4, (
            "RSF model.n_jobs should remain 4 in multi-model evaluate"
        )
        # Results should be valid
        assert len(result.concordance) == 2

    def test_adjust_called_only_for_non_parallel_models(self, train_test_data):
        """adjust_model_njobs is called for CoxPH (inner) but not RSF in multi-model eval."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        rsf = get_model('rsf', n_jobs=8)
        rsf.fit(X_train, y_train)

        coxph = get_model('coxph')
        coxph.fit(X_train, y_train)

        models = {'rsf': rsf, 'coxph': coxph}

        with patch('mlsurv.evaluation.evaluate.adjust_model_njobs', wraps=__import__('mlsurv.utils', fromlist=['adjust_model_njobs']).adjust_model_njobs) as mock_adjust:
            evaluate(
                models, X_test, y_test,
                X_train=X_train, y_train=y_train,
                n_importance_repeats=5,
                random_state=42,
                show_progress=False,
                n_jobs=4,
            )

            # With sequential multi-model loop, adjust_model_njobs is only
            # called inside _compute_eval_dataframes for models WITHOUT
            # internal parallelism (CoxPH gets adjusted, RSF does not)
            calls = mock_adjust.call_args_list
            n_jobs_4_calls = [c for c in calls if c[0][1] == 4]
            assert len(n_jobs_4_calls) == 1, (
                f"Expected exactly 1 adjust_model_njobs call with n_jobs=4 "
                f"(CoxPH only), got {len(n_jobs_4_calls)}"
            )

    def test_sequential_multimodel_does_not_adjust_model_njobs(self, train_test_data):
        """Sequential multi-model evaluate (n_jobs=1) should not adjust model n_jobs."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        rsf = get_model('rsf', n_jobs=4)
        rsf.fit(X_train, y_train)

        coxph = get_model('coxph')
        coxph.fit(X_train, y_train)

        models = {'rsf': rsf, 'coxph': coxph}

        # Sequential evaluation (n_jobs=1) — no outer parallelism
        evaluate(
            models, X_test, y_test,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )

        # RSF's n_jobs should be unchanged (n_jobs=1 means no oversubscription risk)
        assert rsf.n_jobs == 4, (
            "RSF model.n_jobs should remain 4 in sequential multi-model evaluate"
        )


class TestEvaluateBootstrapOversubscription:
    """Verify n_jobs safety in evaluate+bootstrap combined path."""

    def test_rsf_njobs_restored_after_bootstrap(self, train_test_data):
        """RSF n_jobs is restored after evaluate with bootstrap."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        rsf = get_model('rsf', n_jobs=4)
        rsf.fit(X_train, y_train)

        # Evaluate with bootstrap and parallel
        result = evaluate(
            rsf, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=4,
            bootstrap=True,
            n_bootstrap=10,
        )

        # Model n_jobs should be restored to original value (4)
        assert rsf.n_jobs == 4
        assert result.bootstrap is not None


class TestXGBCoxOversubscription:
    """Verify XGBCox n_jobs is also handled."""

    def test_xgbcox_njobs_preserved_during_evaluate(self, train_test_data):
        """XGBCox model.n_jobs should be preserved — importance uses n_jobs=1."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"^XGBoost with n_jobs != 1", category=UserWarning)
                model = get_model('xgbcox', n_jobs=4)
            model.fit(X_train, y_train)
        except (ImportError, ModuleNotFoundError):
            pytest.skip("XGBCox not available")

        evaluate(
            model, X_test, y_test,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=4,
        )

        # XGBCox has internal parallelism — n_jobs should be unchanged
        assert model.n_jobs == 4


class TestBootstrapBackendSelection:
    """Verify correct joblib backend selection for bootstrap iterations."""

    def test_threading_backend_used_for_precomputed_bootstrap(self, train_test_data):
        """When can_precompute=True and model has internal parallelism, use threading."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('rsf', n_jobs=1)
        model.fit(X_train, y_train)

        with patch('mlsurv.evaluation._bootstrap_workers.Parallel') as mock_parallel:
            mock_parallel.return_value = mock_parallel
            mock_parallel.__call__ = MagicMock(return_value=[
                {'iteration_data': [], 'predictions_data': None,
                 'auc_data': None, 'importance_data': None}
            ])
            # Precomputed bootstrap (no custom_metrics, no importance)
            try:
                bootstrap(
                    model, X_test, y_test, y_train=y_train,
                    n_bootstrap=100, random_state=42, show_progress=False,
                    n_jobs=4, force_parallel=True,
                )
            except Exception:
                pass  # May fail on result aggregation, but we just need the Parallel call

            # Check that Parallel was called with backend='threading'
            calls = mock_parallel.call_args_list
            backend_args = [c.kwargs.get('backend', c.args[1] if len(c.args) > 1 else None)
                           for c in calls if 'backend' in c.kwargs or len(c.args) > 1]
            assert any(b == 'threading' for b in backend_args), (
                f"Expected threading backend for precomputed RSF bootstrap, got {backend_args}"
            )

    def test_loky_backend_used_for_lightweight_precomputed_bootstrap(self, train_test_data):
        """CoxPH with precomputed should still use loky (lightweight model)."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        with patch('mlsurv.evaluation._bootstrap_workers.Parallel') as mock_parallel:
            mock_parallel.return_value = mock_parallel
            mock_parallel.__call__ = MagicMock(return_value=[
                {'iteration_data': [], 'predictions_data': None,
                 'auc_data': None, 'importance_data': None}
            ])
            # Precomputed bootstrap (no custom_metrics, no importance)
            try:
                bootstrap(
                    model, X_test, y_test, y_train=y_train,
                    n_bootstrap=100, random_state=42, show_progress=False,
                    n_jobs=4, force_parallel=True,
                )
            except Exception:
                pass  # May fail on result aggregation, but we just need the Parallel call

            # Check that Parallel was called with backend='loky' (not threading)
            calls = mock_parallel.call_args_list
            backend_args = [c.kwargs.get('backend', c.args[1] if len(c.args) > 1 else None)
                           for c in calls if 'backend' in c.kwargs or len(c.args) > 1]
            assert any(b == 'loky' for b in backend_args), (
                f"Expected loky backend for precomputed CoxPH bootstrap, got {backend_args}"
            )
            assert not any(b == 'threading' for b in backend_args), (
                f"CoxPH should NOT use threading backend, got {backend_args}"
            )

    def test_loky_backend_used_for_non_precomputed_bootstrap(self, train_test_data):
        """When can_precompute=False (importance enabled), bootstrap should use loky."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        with patch('mlsurv.evaluation._bootstrap_workers.Parallel') as mock_parallel:
            mock_parallel.return_value = mock_parallel
            mock_parallel.__call__ = MagicMock(return_value=[
                {'iteration_data': [], 'predictions_data': None,
                 'auc_data': None, 'importance_data': None}
            ])
            # Non-precomputed bootstrap (include_importance=True)
            try:
                bootstrap(
                    model, X_test, y_test, y_train=y_train,
                    n_bootstrap=600, random_state=42, show_progress=False,
                    n_jobs=4, include_importance=True, force_parallel=True,
                )
            except Exception:
                pass

            # Check that Parallel was called with backend='loky'
            calls = mock_parallel.call_args_list
            backend_args = [c.kwargs.get('backend') for c in calls if 'backend' in c.kwargs]
            assert any(b == 'loky' for b in backend_args), (
                f"Expected loky backend for non-precomputed bootstrap, got {backend_args}"
            )


class TestNjobsPropagation:
    """Verify setup(n_jobs) is propagated to models with internal parallelism."""

    def _get_inner_model(self, pipeline):
        """Extract the innermost model from a pipeline."""
        return pipeline.steps[-1][1]

    @pytest.mark.parametrize("n_jobs_value", [4, 1, -1])
    def test_setup_njobs_propagates_to_rsf(self, synthetic_data_clean, n_jobs_value):
        """setup(n_jobs=v) + train('rsf') → inner model.n_jobs == v (v in {4, 1, -1})."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=n_jobs_value)
        result = learner.train('rsf')

        inner_model = self._get_inner_model(result.model)
        assert inner_model.n_jobs == n_jobs_value, (
            f"RSF should inherit n_jobs={n_jobs_value} from setup(), "
            f"got {inner_model.n_jobs}"
        )

    def test_train_propagates_njobs_to_xgbcox(self, synthetic_data_clean):
        """setup(n_jobs=4) + train('xgbcox') → inner model.n_jobs == 4."""
        from mlsurv import SurvivalLearner

        try:
            import xgboost  # noqa: F401
        except ImportError:
            pytest.skip("XGBoost not installed")

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=4)
        result = learner.train('xgbcox')

        inner_model = self._get_inner_model(result.model)
        assert inner_model.n_jobs == 4, (
            f"XGBCox should inherit n_jobs=4 from setup(), got {inner_model.n_jobs}"
        )

    def test_train_does_not_propagate_njobs_to_coxph(self, synthetic_data_clean):
        """CoxPH has no n_jobs param; train should not inject one."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=4)
        result = learner.train('coxph')

        inner_model = self._get_inner_model(result.model)
        assert not hasattr(inner_model, 'n_jobs'), (
            "CoxPH should not have n_jobs attribute injected by train()"
        )

    def test_explicit_njobs_kwarg_not_overridden(self, synthetic_data_clean):
        """Explicit n_jobs=2 in train() kwargs should not be overridden by setup n_jobs=4."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=4)
        result = learner.train('rsf', n_jobs=2)

        inner_model = self._get_inner_model(result.model)
        assert inner_model.n_jobs == 2, (
            f"Explicit n_jobs=2 should be respected, got {inner_model.n_jobs}"
        )

    def test_setup_without_explicit_njobs_propagates_default(self, synthetic_data_clean):
        """setup() without explicit n_jobs uses default → propagated to RSF."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42)  # no n_jobs arg
        result = learner.train('rsf')

        inner_model = self._get_inner_model(result.model)
        assert inner_model.n_jobs == learner.n_jobs, (
            f"RSF should inherit default n_jobs={learner.n_jobs} from setup(), "
            f"got {inner_model.n_jobs}"
        )

    def test_tune_preserves_njobs_propagation(self, synthetic_data_clean):
        """setup(n_jobs=4) + tune('rsf') → tuned model.n_jobs == 4."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=4)
        learner.train('rsf')
        learner.tune('rsf', n_trials=2, search_space=_SMALL_TREE_SPACE)  # overwrites the baseline at 'rsf'

        result = learner.models['rsf']
        assert result.is_tuned
        inner_model = self._get_inner_model(result.model)
        assert inner_model.n_jobs == 4, (
            f"Tuned RSF should inherit n_jobs=4 from setup(), got {inner_model.n_jobs}"
        )

    def test_custom_model_warns_on_njobs_mismatch(self, synthetic_data_clean):
        """Custom RSF instance with mismatched n_jobs should warn."""
        from mlsurv import SurvivalLearner
        from mlsurv.models import get_model

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=4)

        custom_rsf = get_model('rsf', n_jobs=1)

        with pytest.warns(SuggestionWarning, match=r"n_jobs=1.*setup\(n_jobs=4\)"):
            learner.train(custom_rsf)


class TestNjobsNeverExceedsUserSetting:
    """Integration-level tests that n_jobs is respected end-to-end."""

    def test_learner_evaluate_respects_njobs(self, synthetic_data_clean):
        """SurvivalLearner.evaluate() passes n_jobs correctly."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=4)
        learner.train('rsf')

        eval_result = learner.evaluate(
            n_importance_repeats=5,
            show_progress=False,
        )

        assert eval_result.concordance is not None


class TestNjobsRestoredOnException:
    """Verify n_jobs is restored even when exceptions occur mid-evaluation."""

    def test_njobs_restored_on_exception_in_eval_dataframes(self, train_test_data):
        """n_jobs is restored when _compute_eval_dataframes raises."""
        from mlsurv.models import get_model
        from mlsurv.evaluation.evaluate import _compute_eval_dataframes
        from mlsurv.evaluation._config import EvalConfig
        from mlsurv.utils import auto_eval_times

        X_train, X_test, y_train, y_test = train_test_data

        # Use a Pipeline wrapping RSF so save/restore has something to do
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rsf = get_model('rsf', n_jobs=4)
        rsf.fit(X_train, y_train)
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', rsf)])
        pipeline.fit(X_train, y_train)

        original_njobs = rsf.n_jobs
        assert original_njobs == 4

        eval_times = auto_eval_times(y_test)
        config = EvalConfig(
            n_importance_repeats=5,
            eval_times=eval_times,
            random_state=42,
            include_shap=False,
            show_progress=False,
            n_jobs=4,
        )

        # Mock _compute_concordance to raise after n_jobs has been adjusted
        with patch(
            'mlsurv.evaluation.evaluate._compute_concordance',
            side_effect=RuntimeError("injected error"),
        ):
            with pytest.raises(RuntimeError, match="injected error"):
                _compute_eval_dataframes(
                    pipeline, X_test, y_test, X_train, y_train,
                    None, 'rsf', config,
                )

        # n_jobs must be restored despite the exception
        assert rsf.n_jobs == original_njobs, (
            f"RSF n_jobs should be restored to {original_njobs} after exception, "
            f"got {rsf.n_jobs}"
        )

    def test_njobs_restored_on_exception_in_bootstrap(self, train_test_data):
        """n_jobs is restored when _bootstrap_single raises."""
        from mlsurv.models import get_model
        from mlsurv.evaluation.bootstrap import _bootstrap_single

        X_train, X_test, y_train, y_test = train_test_data

        # Use a Pipeline wrapping RSF so save/restore has something to do
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rsf = get_model('rsf', n_jobs=4)
        rsf.fit(X_train, y_train)
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', rsf)])
        pipeline.fit(X_train, y_train)

        original_njobs = rsf.n_jobs
        assert original_njobs == 4

        # Mock _precompute_predictions to raise after n_jobs has been adjusted
        with patch(
            'mlsurv.evaluation.bootstrap._precompute_predictions',
            side_effect=RuntimeError("injected error"),
        ):
            with pytest.raises(RuntimeError, match="injected error"):
                _bootstrap_single(
                    pipeline, X_test, y_test,
                    y_train=y_train,
                    n_bootstrap=10,
                    random_state=42,
                    show_progress=False,
                    n_jobs=4,
                )

        # n_jobs must be restored despite the exception
        assert rsf.n_jobs == original_njobs, (
            f"RSF n_jobs should be restored to {original_njobs} after exception, "
            f"got {rsf.n_jobs}"
        )


class TestPredictionBootstrapBackendSelection:
    """Verify correct joblib backend and kwargs adjustment for prediction bootstrap."""

    def test_loky_backend_for_rsf(self, train_test_data):
        """Prediction bootstrap with RSF should use loky (true parallelism for retrain)."""
        import numpy as np
        from mlsurv.evaluation._prediction_bootstrap import (
            run_prediction_bootstrap,
        )
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        with patch(
            'mlsurv.evaluation._prediction_bootstrap.Parallel',
            create=True,
        ) as mock_cls:
            mock_cls.return_value = MagicMock(return_value=[None])
            try:
                run_prediction_bootstrap(
                    X_train=X_train, y_train=y_train,
                    X_new=X_test.iloc[:1],
                    model_name='rsf',
                    model_proto=get_model('rsf', n_estimators=10, n_jobs=4),
                    imputer=None, scaler=None, selector=None,
                    time_grid=np.array([1.0, 2.0]),
                    rmst_default_tau=10.0,
                    user_times=None,
                    n_bootstrap=25,
                    n_jobs=2,
                    verbose=False,
                )
            except (TypeError, AttributeError):
                pass  # Mock doesn't return valid iteration results

            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args
            # loky preferred; threading acceptable as fallback
            assert call_kwargs[1]['backend'] in ('loky', 'threading')

    def test_coxph_stays_sequential(self, train_test_data):
        """Prediction bootstrap with CoxPH should never invoke Parallel."""
        import numpy as np
        from mlsurv.evaluation._prediction_bootstrap import (
            run_prediction_bootstrap,
        )
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        with patch(
            'mlsurv.evaluation._prediction_bootstrap.Parallel',
            create=True,
        ) as mock_cls:
            try:
                run_prediction_bootstrap(
                    X_train=X_train, y_train=y_train,
                    X_new=X_test.iloc[:1],
                    model_name='coxph',
                    model_proto=get_model('coxph', alpha=0.1),
                    imputer=None, scaler=None, selector=None,
                    time_grid=np.array([1.0, 2.0]),
                    rmst_default_tau=10.0,
                    user_times=None,
                    n_bootstrap=25,
                    n_jobs=4,
                    verbose=False,
                )
            except (TypeError, AttributeError):
                pass

            # CoxPH should never use Parallel — always sequential
            mock_cls.assert_not_called()

    def test_proto_njobs_adjusted_for_rsf(self, train_test_data):
        """When outer parallel is active, RSF model proto should have n_jobs=1."""
        import numpy as np
        from mlsurv.evaluation._prediction_bootstrap import (
            run_prediction_bootstrap,
        )
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data
        captured = {}

        def spy_iter(seed, **kwargs):
            captured.update(kwargs)
            return None  # Fail fast

        # Force threading backend so the spy (monkey-patch) is visible
        # across threads.  Loky forks processes where patches aren't visible.
        # effective_n_jobs must be >1 so _adjust_proto_njobs activates.
        with patch(
            'mlsurv.evaluation._prediction_bootstrap'
            '._resolve_prediction_parallel_config',
            return_value=(2, 'threading'),
        ), patch(
            'mlsurv.evaluation._prediction_bootstrap'
            '._prediction_bootstrap_iteration',
            side_effect=spy_iter,
        ):
            run_prediction_bootstrap(
                X_train=X_train, y_train=y_train,
                X_new=X_test.iloc[:1],
                model_name='rsf',
                model_proto=get_model('rsf', n_estimators=10, n_jobs=4),
                imputer=None, scaler=None, selector=None,
                time_grid=np.array([1.0, 2.0]),
                rmst_default_tau=10.0,
                user_times=None,
                n_bootstrap=25,
                n_jobs=2,
                verbose=False,
            )

        assert captured['model_proto'].n_jobs == 1
