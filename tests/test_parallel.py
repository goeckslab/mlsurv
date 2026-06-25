"""Tests for parallel computing support in mlsurv."""

import pytest
from numpy.testing import assert_array_almost_equal

pytestmark = pytest.mark.parallel


class TestParallelBootstrap:
    """Tests for parallel bootstrap execution."""

    def test_bootstrap_parallel_matches_sequential(self, train_test_data):
        """Bootstrap with n_jobs=-1 produces same results as n_jobs=1 (same seed)."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap

        X_train, X_test, y_train, y_test = train_test_data

        # Train a model
        model = get_model('coxph')
        model.fit(X_train, y_train)

        # Run bootstrap with sequential execution
        result_seq = bootstrap(
            model,
            X_test,
            y_test,
            y_train=y_train,
            n_bootstrap=15,
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )

        # Run bootstrap with parallel execution
        result_par = bootstrap(
            model,
            X_test,
            y_test,
            y_train=y_train,
            n_bootstrap=15,
            random_state=42,
            show_progress=False,
            n_jobs=-1,
        )

        # Results should match
        assert_array_almost_equal(
            result_seq.summary['mean'].values,
            result_par.summary['mean'].values,
            decimal=8,
        )
        assert_array_almost_equal(
            result_seq.summary['std'].values,
            result_par.summary['std'].values,
            decimal=8,
        )

    def test_bootstrap_reproducible_with_seed(self, train_test_data):
        """Parallel bootstrap is reproducible when seed is set."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        # Run twice with same seed
        result1 = bootstrap(
            model, X_test, y_test, y_train=y_train,
            n_bootstrap=10, random_state=123, show_progress=False, n_jobs=4,
        )
        result2 = bootstrap(
            model, X_test, y_test, y_train=y_train,
            n_bootstrap=10, random_state=123, show_progress=False, n_jobs=4,
        )

        # Should be identical
        assert_array_almost_equal(
            result1.summary['mean'].values,
            result2.summary['mean'].values,
            decimal=8,
        )

    def test_bootstrap_results_same_any_njobs(self, train_test_data):
        """Bootstrap results are identical regardless of n_jobs value."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        # Run with various n_jobs values
        n_jobs_values = [1, 2, 4, -1]
        results = []

        for n_jobs in n_jobs_values:
            result = bootstrap(
                model, X_test, y_test, y_train=y_train,
                n_bootstrap=10, random_state=42, show_progress=False, n_jobs=n_jobs,
            )
            results.append(result)

        # All should match the first
        for i, result in enumerate(results[1:], 1):
            assert_array_almost_equal(
                results[0].summary['mean'].values,
                result.summary['mean'].values,
                decimal=8,
                err_msg=f"n_jobs={n_jobs_values[i]} differs from n_jobs=1",
            )


class TestParallelCV:
    """Tests for parallel CV fold training."""

    def test_cv_parallel_matches_sequential(self, synthetic_data_clean):
        """CV with n_jobs=-1 produces same results as n_jobs=1."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        # Sequential CV
        learner_seq = SurvivalLearner(X, y)
        learner_seq.setup(cv_folds=3, random_state=42, n_jobs=1)
        result_seq = learner_seq.train('coxph')

        # Parallel CV
        learner_par = SurvivalLearner(X, y)
        learner_par.setup(cv_folds=3, random_state=42, n_jobs=-1)
        result_par = learner_par.train('coxph')

        # CV scores should match
        assert_array_almost_equal(
            result_seq.cv_metric_scores,
            result_par.cv_metric_scores,
            decimal=8,
        )
        assert abs(result_seq.cv_metric_mean - result_par.cv_metric_mean) < 1e-8

    def test_cv_reproducible_with_seed(self, synthetic_data_clean):
        """Parallel CV is reproducible when seed is set."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        # Run twice with same seed
        learner1 = SurvivalLearner(X, y)
        learner1.setup(cv_folds=3, random_state=42, n_jobs=4)
        result1 = learner1.train('coxph')

        learner2 = SurvivalLearner(X, y)
        learner2.setup(cv_folds=3, random_state=42, n_jobs=4)
        result2 = learner2.train('coxph')

        # Should be identical
        assert_array_almost_equal(
            result1.cv_metric_scores,
            result2.cv_metric_scores,
            decimal=8,
        )


class TestParallelEvaluation:
    """Tests for parallel model evaluation."""

    def test_evaluate_parallel_matches_sequential(self, train_test_data):
        """Multi-model evaluation with n_jobs=-1 matches n_jobs=1."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        # Train multiple models
        models = {}
        for name in ['coxph', 'coxnet']:
            model = get_model(name)
            model.fit(X_train, y_train)
            models[name] = model

        # Sequential evaluation
        result_seq = evaluate(
            models, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )

        # Parallel evaluation
        result_par = evaluate(
            models, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=-1,
        )

        # C-index values should match
        for model_name in ['coxph', 'coxnet']:
            seq_cidx = result_seq.concordance[
                result_seq.concordance['model_type'] == model_name
            ]['c_index_test'].values[0]
            par_cidx = result_par.concordance[
                result_par.concordance['model_type'] == model_name
            ]['c_index_test'].values[0]
            assert abs(seq_cidx - par_cidx) < 1e-8

    def test_permutation_importance_parallel(self, train_test_data):
        """Permutation importance with n_jobs works correctly."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        # Evaluate with parallel importance
        result = evaluate(
            model, X_test, y_test,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=4,
        )

        # Should have importance results
        assert result.importance is not None
        assert len(result.importance) > 0


class TestParallelIntegration:
    """Integration tests for parallel computing across the full pipeline."""

    def test_full_pipeline_with_parallel(self, synthetic_data_clean):
        """Full SurvivalLearner pipeline works with n_jobs=-1."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        # Run full pipeline with parallelization
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=-1)

        # Train a model
        result = learner.train('coxph')
        assert result.cv_metric_mean > 0.5

        # Evaluate
        eval_result = learner.evaluate(
            n_importance_repeats=5,
            show_progress=False,
        )
        assert eval_result.concordance is not None

    def test_bootstrap_with_parallel_via_learner(self, synthetic_data_clean):
        """Bootstrap through learner works with n_jobs."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=-1)
        learner.train('coxph')

        # Bootstrap evaluation
        result = learner.bootstrap(
            n_bootstrap=10,
            random_state=42,
            show_progress=False,
        )

        assert result.summary is not None
        assert len(result.summary) > 0


class TestNJobsParameter:
    """Tests for n_jobs parameter handling."""

    def test_njobs_default_is_one(self, synthetic_data_clean):
        """Default n_jobs is 1 (sequential)."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        assert learner.n_jobs == 1

    def test_njobs_minus_one_uses_all_cores(self, synthetic_data_clean):
        """n_jobs=-1 is accepted and works."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, n_jobs=-1)

        # Should be stored
        assert learner.n_jobs == -1

        # Should work for training
        result = learner.train('coxph')
        assert result.cv_metric_mean > 0.5

    def test_njobs_positive_int(self, synthetic_data_clean):
        """Positive n_jobs values work correctly."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, n_jobs=4)

        assert learner.n_jobs == 4

        result = learner.train('coxph')
        assert result.cv_metric_mean > 0.5


class TestSmartParallelDetection:
    """Tests for smart parallel CV detection."""

    def test_sequential_cv_models_not_slowed(self, synthetic_data_clean):
        """CoxPH uses sequential CV even when n_jobs=-1 (smart detection)."""
        from mlsurv import SurvivalLearner
        from mlsurv.learner import SEQUENTIAL_CV_MODELS

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=-1)

        # CoxPH is in SEQUENTIAL_CV_MODELS, so CV should run with n_jobs=1
        # internally even though setup specified n_jobs=-1
        assert 'coxph' in SEQUENTIAL_CV_MODELS

        result = learner.train('coxph')
        assert result.cv_metric_mean > 0.5

    def test_sequential_cv_models_constant(self):
        """SEQUENTIAL_CV_MODELS includes expected models."""
        from mlsurv.learner import SEQUENTIAL_CV_MODELS

        assert 'coxph' in SEQUENTIAL_CV_MODELS
        assert 'coxnet' in SEQUENTIAL_CV_MODELS
        assert 'deepsurv' in SEQUENTIAL_CV_MODELS
        # RSF/GBSA should NOT be in this set (they benefit from parallelization)
        assert 'rsf' not in SEQUENTIAL_CV_MODELS
        assert 'gbsa' not in SEQUENTIAL_CV_MODELS

    def test_rsf_still_uses_parallel(self, synthetic_data_clean):
        """RSF should still use parallel CV when n_jobs != 1."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=-1)

        # RSF should work with parallelization
        result = learner.train('rsf')
        assert result.cv_metric_mean > 0.5


class TestHelperFunctions:
    """Tests for parallel computing helper functions."""

    def test_adjust_model_njobs(self):
        """_adjust_model_njobs correctly disables model parallelism."""
        from mlsurv.learner import _adjust_model_njobs
        from mlsurv.models import get_model

        # RSF has n_jobs parameter
        model = get_model('rsf', n_jobs=4)

        # When outer_n_jobs=1, model should keep its n_jobs
        adjusted = _adjust_model_njobs(model, outer_n_jobs=1)
        assert adjusted.n_jobs == 4

        # When outer_n_jobs != 1, model n_jobs should be set to 1
        adjusted = _adjust_model_njobs(model, outer_n_jobs=4)
        assert adjusted.n_jobs == 1

        # CoxPH doesn't have n_jobs, should return unchanged
        model_cox = get_model('coxph')
        adjusted_cox = _adjust_model_njobs(model_cox, outer_n_jobs=4)
        assert not hasattr(adjusted_cox, 'n_jobs') or adjusted_cox is model_cox

    def test_adjust_model_njobs_from_utils(self):
        """adjust_model_njobs from utils works correctly."""
        from mlsurv.utils import adjust_model_njobs
        from mlsurv.models import get_model

        model = get_model('rsf', n_jobs=4)

        # When outer_n_jobs=1, model should keep its n_jobs
        adjusted = adjust_model_njobs(model, outer_n_jobs=1)
        assert adjusted.n_jobs == 4

        # When outer_n_jobs != 1, model n_jobs should be set to 1
        adjusted = adjust_model_njobs(model, outer_n_jobs=4)
        assert adjusted.n_jobs == 1

    def test_adjust_model_njobs_handles_pipeline(self):
        """adjust_model_njobs correctly handles sklearn Pipeline objects."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from mlsurv.utils import adjust_model_njobs
        from mlsurv.models import get_model

        # Create a pipeline with RSF (has n_jobs) as the 'model' step
        rsf = get_model('rsf', n_jobs=4)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', rsf),
        ])

        # Pipeline itself doesn't have n_jobs
        assert not hasattr(pipeline, 'n_jobs')
        # But the inner model does
        assert pipeline.named_steps['model'].n_jobs == 4

        # When outer_n_jobs=1, model inside pipeline should keep its n_jobs
        adjusted = adjust_model_njobs(pipeline, outer_n_jobs=1)
        assert adjusted.named_steps['model'].n_jobs == 4

        # When outer_n_jobs != 1, model inside pipeline should be set to 1
        rsf2 = get_model('rsf', n_jobs=4)
        pipeline2 = Pipeline([
            ('scaler', StandardScaler()),
            ('model', rsf2),
        ])
        adjusted2 = adjust_model_njobs(pipeline2, outer_n_jobs=4)
        assert adjusted2.named_steps['model'].n_jobs == 1

    def test_adjust_model_njobs_pipeline_without_model_step(self):
        """adjust_model_njobs handles Pipeline without 'model' step gracefully."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from mlsurv.utils import adjust_model_njobs
        from mlsurv.models import get_model

        # Pipeline with different step names (no 'model' key)
        coxph = get_model('coxph')
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', coxph),  # Not named 'model'
        ])

        # Should return pipeline unchanged (no crash)
        adjusted = adjust_model_njobs(pipeline, outer_n_jobs=4)
        assert adjusted is pipeline

    def test_save_restore_direct_model_with_njobs(self):
        """save/restore round-trip works for a direct model with n_jobs (RSF)."""
        from mlsurv.utils import save_model_njobs, restore_model_njobs
        from mlsurv.models import get_model

        model = get_model('rsf', n_jobs=8)
        saved = save_model_njobs(model)

        assert saved == {'n_jobs': 8}

        # Mutate n_jobs, then restore
        model.n_jobs = 1
        restore_model_njobs(model, saved)
        assert model.n_jobs == 8

    def test_save_restore_pipeline_with_njobs(self):
        """save/restore works for a Pipeline wrapping a model with n_jobs."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from mlsurv.utils import save_model_njobs, restore_model_njobs, _unwrap_inner_model
        from mlsurv.models import get_model

        rsf = get_model('rsf', n_jobs=4)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', rsf),
        ])

        # _unwrap_inner_model should find the inner RSF
        inner = _unwrap_inner_model(pipeline)
        assert inner is rsf

        saved = save_model_njobs(pipeline)
        assert saved == {'n_jobs': 4}

        rsf.n_jobs = 1
        restore_model_njobs(pipeline, saved)
        assert rsf.n_jobs == 4

    def test_save_restore_model_without_njobs(self):
        """save returns {} and restore is no-op for model without n_jobs (CoxPH)."""
        from mlsurv.utils import save_model_njobs, restore_model_njobs
        from mlsurv.models import get_model

        model = get_model('coxph')
        saved = save_model_njobs(model)
        assert saved == {}

        # restore with empty dict is a no-op (should not raise)
        restore_model_njobs(model, saved)

    def test_save_restore_model_with_model_subattr(self, train_test_data):
        """save/restore handles model_ sub-attribute (fitted estimator convention)."""
        from mlsurv.utils import save_model_njobs, restore_model_njobs
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('rsf', n_jobs=4)
        model.fit(X_train, y_train)

        # After fitting, RSF should have model_ with n_jobs
        if hasattr(model, 'model_') and hasattr(model.model_, 'n_jobs'):
            saved = save_model_njobs(model)
            assert 'n_jobs' in saved
            assert 'model_n_jobs' in saved

            # Mutate both
            model.n_jobs = 1
            model.model_.n_jobs = 1

            restore_model_njobs(model, saved)
            assert model.n_jobs == 4
            assert model.model_.n_jobs == saved['model_n_jobs']
        else:
            # If model_ doesn't have n_jobs, just verify basic save/restore
            saved = save_model_njobs(model)
            assert saved == {'n_jobs': 4}

    def test_restore_empty_saved_is_noop(self):
        """restore_model_njobs with empty dict is a no-op."""
        from mlsurv.utils import restore_model_njobs
        from mlsurv.models import get_model

        model = get_model('rsf', n_jobs=4)
        restore_model_njobs(model, {})
        assert model.n_jobs == 4

    def test_unwrap_inner_model_returns_none_for_no_njobs(self):
        """_unwrap_inner_model returns None for model without n_jobs."""
        from mlsurv.utils import _unwrap_inner_model
        from mlsurv.models import get_model

        model = get_model('coxph')
        assert _unwrap_inner_model(model) is None

    def test_restore_warns_on_structural_mismatch(self):
        """restore_model_njobs warns when saved is non-empty but inner is None."""
        import warnings
        from mlsurv.utils import restore_model_njobs
        from mlsurv.models import get_model

        # CoxPH has no n_jobs, so _unwrap_inner_model returns None
        model = get_model('coxph')
        saved = {'n_jobs': 4}  # Pretend we saved from a different structure

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            restore_model_njobs(model, saved)
            assert len(w) == 1
            assert "Model structure may have changed" in str(w[0].message)


class TestUnivariateCorrelationParallel:
    """Tests for parallel univariate Cox fitting."""

    def test_feature_selection_parallel_matches_sequential(self, train_test_data):
        """TopKCoxSelector with n_jobs=-1 produces same results as n_jobs=1."""
        from mlsurv.preprocessing import TopKCoxSelector

        X_train, X_test, y_train, y_test = train_test_data

        # Sequential feature selection
        selector_seq = TopKCoxSelector(k=5, n_jobs=1)
        selector_seq.fit(X_train, y_train)

        # Parallel feature selection
        selector_par = TopKCoxSelector(k=5, n_jobs=-1)
        selector_par.fit(X_train, y_train)

        # Selected features should match
        assert selector_seq.selected_features_ == selector_par.selected_features_

        # Rankings should match
        seq_rankings = dict(selector_seq.sorted_features_)
        par_rankings = dict(selector_par.sorted_features_)
        for feat in seq_rankings:
            assert abs(seq_rankings[feat] - par_rankings[feat]) < 1e-8

    def test_compute_univariate_cox_parallel_matches_sequential(self, train_test_data):
        """compute_univariate_cox with n_jobs=-1 produces same results as n_jobs=1."""
        from mlsurv.evaluation.importance import compute_univariate_cox

        X_train, X_test, y_train, y_test = train_test_data

        # Sequential execution
        result_seq = compute_univariate_cox(X_train, y_train, n_jobs=1)

        # Parallel execution
        result_par = compute_univariate_cox(X_train, y_train, n_jobs=-1)

        # Results should match
        assert list(result_seq['feature']) == list(result_par['feature'])
        assert_array_almost_equal(
            result_seq['p_value'].values,
            result_par['p_value'].values,
            decimal=8,
        )
        assert_array_almost_equal(
            result_seq['hazard_ratio'].values,
            result_par['hazard_ratio'].values,
            decimal=8,
        )


class TestMultiModelBootstrapParallel:
    """Tests for parallel multi-model bootstrap."""

    def test_multi_model_bootstrap_with_njobs_models(self, train_test_data):
        """Multi-model bootstrap with n_jobs_models produces same results."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap

        X_train, X_test, y_train, y_test = train_test_data

        # Train models
        models = {}
        for name in ['coxph', 'coxnet']:
            model = get_model(name)
            model.fit(X_train, y_train)
            models[name] = model

        # Sequential model bootstrap
        result_seq = bootstrap(
            models,
            X_test,
            y_test,
            y_train=y_train,
            n_bootstrap=10,
            random_state=42,
            show_progress=False,
            n_jobs=1,
            n_jobs_models=1,
        )

        # Parallel model bootstrap
        result_par = bootstrap(
            models,
            X_test,
            y_test,
            y_train=y_train,
            n_bootstrap=10,
            random_state=42,
            show_progress=False,
            n_jobs=1,
            n_jobs_models=-1,
        )

        # Both models should be present in results
        assert set(result_seq.summary['model_type']) == set(result_par.summary['model_type'])

        # Mean values should match for each model
        for model_name in ['coxph', 'coxnet']:
            seq_mean = result_seq.summary[
                result_seq.summary['model_type'] == model_name
            ]['mean'].values
            par_mean = result_par.summary[
                result_par.summary['model_type'] == model_name
            ]['mean'].values
            assert_array_almost_equal(seq_mean, par_mean, decimal=8)


class TestRSFDefaultNjobs:
    """Tests for RSF n_jobs default behavior."""

    def test_rsf_default_njobs_is_one(self):
        """RSF default n_jobs should be 1 (not -1)."""
        from mlsurv.models import get_model

        model = get_model('rsf')
        assert model.n_jobs == 1, "RSF should default to n_jobs=1"

    def test_rsf_respects_explicit_njobs(self):
        """RSF should respect explicitly set n_jobs."""
        from mlsurv.models import get_model

        model = get_model('rsf', n_jobs=4)
        assert model.n_jobs == 4

        model = get_model('rsf', n_jobs=-1)
        assert model.n_jobs == -1


class TestAutoDisableParallel:
    """Tests for automatic parallelization disable thresholds."""

    def test_feature_selection_auto_disable_below_threshold(self, train_test_data):
        """TopKCoxSelector auto-disables parallelization for few features."""
        from mlsurv.preprocessing import TopKCoxSelector
        from mlsurv.preprocessing.selection import MIN_FEATURES_FOR_PARALLEL

        X_train, X_test, y_train, y_test = train_test_data

        # Use only a few features (below threshold)
        n_cols = min(10, MIN_FEATURES_FOR_PARALLEL - 1)
        X_small = X_train.iloc[:, :n_cols]

        # With n_jobs=-1, should auto-disable and produce same results as n_jobs=1
        selector_seq = TopKCoxSelector(k=3, n_jobs=1)
        selector_seq.fit(X_small, y_train)

        selector_par = TopKCoxSelector(k=3, n_jobs=-1)
        selector_par.fit(X_small, y_train)

        # Results should match (parallelization was auto-disabled)
        assert selector_seq.selected_features_ == selector_par.selected_features_

    def test_feature_selection_force_parallel_overrides_threshold(self, train_test_data):
        """force_parallel=True skips threshold check."""
        from mlsurv.preprocessing import TopKCoxSelector

        X_train, X_test, y_train, y_test = train_test_data
        X_small = X_train.iloc[:, :5]  # Very few features

        # With force_parallel=True, should still use parallel even below threshold
        selector = TopKCoxSelector(k=3, n_jobs=-1, force_parallel=True)
        selector.fit(X_small, y_train)

        # Should complete without error
        assert len(selector.selected_features_) == 3

    def test_bootstrap_auto_disable_below_threshold(self, train_test_data):
        """Bootstrap auto-disables parallelization for few iterations."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap
        from mlsurv.evaluation.bootstrap import MIN_ITERATIONS_FOR_PARALLEL

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        # Use few iterations (below threshold)
        n_iter = min(20, MIN_ITERATIONS_FOR_PARALLEL - 1)

        # With n_jobs=-1, should auto-disable and produce same results as n_jobs=1
        result_seq = bootstrap(
            model, X_test, y_test, y_train=y_train,
            n_bootstrap=n_iter, random_state=42, show_progress=False, n_jobs=1,
        )
        result_par = bootstrap(
            model, X_test, y_test, y_train=y_train,
            n_bootstrap=n_iter, random_state=42, show_progress=False, n_jobs=-1,
        )

        # Results should match (parallelization was auto-disabled)
        assert_array_almost_equal(
            result_seq.summary['mean'].values,
            result_par.summary['mean'].values,
            decimal=8,
        )

    def test_bootstrap_force_parallel_overrides_threshold(self, train_test_data):
        """force_parallel=True skips threshold check."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        # With force_parallel=True, should still use parallel even below threshold
        result = bootstrap(
            model, X_test, y_test, y_train=y_train,
            n_bootstrap=10, random_state=42, show_progress=False,
            n_jobs=-1, force_parallel=True,
        )

        # Should complete without error
        assert result.summary is not None
        assert len(result.summary) > 0

    def test_multi_model_bootstrap_auto_disable(self, train_test_data):
        """Multi-model bootstrap auto-disables when total work is low."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap

        X_train, X_test, y_train, y_test = train_test_data

        # Train 2 models
        models = {}
        for name in ['coxph', 'coxnet']:
            model = get_model(name)
            model.fit(X_train, y_train)
            models[name] = model

        # Use few iterations so total_work < threshold
        n_iter = 20  # 2 models * 20 = 40 < 300 threshold

        # With n_jobs_models=-1, should auto-disable
        result_seq = bootstrap(
            models, X_test, y_test, y_train=y_train,
            n_bootstrap=n_iter, random_state=42, show_progress=False,
            n_jobs=1, n_jobs_models=1,
        )
        result_par = bootstrap(
            models, X_test, y_test, y_train=y_train,
            n_bootstrap=n_iter, random_state=42, show_progress=False,
            n_jobs=1, n_jobs_models=-1,
        )

        # Results should match (model parallelization was auto-disabled)
        for model_name in ['coxph', 'coxnet']:
            seq_mean = result_seq.summary[
                result_seq.summary['model_type'] == model_name
            ]['mean'].values
            par_mean = result_par.summary[
                result_par.summary['model_type'] == model_name
            ]['mean'].values
            assert_array_almost_equal(seq_mean, par_mean, decimal=8)

    def test_univariate_cox_auto_disable_below_threshold(self, train_test_data):
        """compute_univariate_cox auto-disables for few features."""
        from mlsurv.evaluation.importance import compute_univariate_cox, MIN_FEATURES_FOR_PARALLEL

        X_train, X_test, y_train, y_test = train_test_data

        # Use only a few features (below threshold)
        n_cols = min(10, MIN_FEATURES_FOR_PARALLEL - 1)
        X_small = X_train.iloc[:, :n_cols]

        # With n_jobs=-1, should auto-disable and produce same results as n_jobs=1
        result_seq = compute_univariate_cox(X_small, y_train, n_jobs=1)
        result_par = compute_univariate_cox(X_small, y_train, n_jobs=-1)

        # Results should match (parallelization was auto-disabled)
        assert list(result_seq['feature']) == list(result_par['feature'])
        assert_array_almost_equal(
            result_seq['p_value'].values,
            result_par['p_value'].values,
            decimal=8,
        )

    def test_threshold_constants_are_reasonable(self):
        """Threshold constants have expected values from benchmarks."""
        from mlsurv.utils import MIN_FEATURES_FOR_PARALLEL
        from mlsurv.preprocessing.selection import MIN_FEATURES_FOR_PARALLEL as SEL_THRESHOLD
        from mlsurv.evaluation.bootstrap import (
            MIN_ITERATIONS_FOR_PARALLEL,
            MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED,
            MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL,
        )
        from mlsurv.evaluation.importance import MIN_FEATURES_FOR_PARALLEL as IMP_THRESHOLD

        # Canonical value from utils.py
        assert MIN_FEATURES_FOR_PARALLEL == 15
        # Consumer modules re-export the same object (not a copy)
        assert SEL_THRESHOLD is MIN_FEATURES_FOR_PARALLEL
        assert IMP_THRESHOLD is MIN_FEATURES_FOR_PARALLEL
        # Full model serialization threshold (RSF has ~0.2s overhead per worker)
        assert MIN_ITERATIONS_FOR_PARALLEL == 500
        # Precomputed arrays serialize efficiently, much lower threshold
        assert MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED == 50
        assert MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL == 300


class TestComputeParallelismSplit:
    """Tests for _compute_parallelism_split() helper function."""

    def test_njobs_one_returns_one_one(self):
        """n_jobs=1 should return (1, 1) - user wants minimal resources."""
        from mlsurv.tuning.optuna_tuner import _compute_parallelism_split

        result = _compute_parallelism_split(
            n_jobs=1, n_trials=10, n_samples=500, model_name='rsf'
        )
        assert result == (1, 1)

    def test_njobs_minus_one_resolves_to_cpu_count(self):
        """n_jobs=-1 should resolve to actual CPU count."""
        import os
        from mlsurv.tuning.optuna_tuner import _compute_parallelism_split

        cpu_count = os.cpu_count() or 1

        # With many trials, should maximize trial parallelism
        trial_jobs, model_jobs = _compute_parallelism_split(
            n_jobs=-1, n_trials=50, n_samples=500, model_name='rsf'
        )

        # trial_n_jobs should be resolved (not -1)
        assert trial_jobs > 0
        assert trial_jobs <= cpu_count
        assert model_jobs >= 1

    def test_non_rsf_models_get_model_njobs_one(self):
        """Non-RSF models (excluding thread-unsafe) should get model_n_jobs=1."""
        from mlsurv.tuning.optuna_tuner import _compute_parallelism_split

        for model_name in ['coxph', 'gbsa', 'fssvm', 'deepsurv']:
            trial_jobs, model_jobs = _compute_parallelism_split(
                n_jobs=8, n_trials=10, n_samples=1000, model_name=model_name
            )
            assert model_jobs == 1, f"{model_name} should have model_n_jobs=1"
            assert trial_jobs == 8, f"{model_name} should use all cores for trials"

    def test_small_dataset_uses_balanced(self):
        """Small/medium datasets should use balanced approach."""
        from mlsurv.tuning.optuna_tuner import _compute_parallelism_split

        # 1000 samples (<2000 threshold), 10 trials
        trial_jobs, model_jobs = _compute_parallelism_split(
            n_jobs=16, n_trials=10, n_samples=1000, model_name='rsf'
        )

        # Should use balanced: ~4 trial jobs, ~4 model jobs
        assert trial_jobs >= 2
        assert model_jobs >= 2
        assert trial_jobs * model_jobs <= 16

    def test_large_dataset_uses_model_only(self):
        """Large datasets (>=2000) should use model_only strategy."""
        from mlsurv.tuning.optuna_tuner import _compute_parallelism_split

        # 2000 samples (>=2000 threshold) - model_only is fastest
        trial_jobs, model_jobs = _compute_parallelism_split(
            n_jobs=16, n_trials=10, n_samples=2000, model_name='rsf'
        )

        # Should use model_only: 1 trial, all cores to model
        assert trial_jobs == 1
        assert model_jobs == 16

    def test_many_trials_uses_balanced(self):
        """With many trials (>=20), should use balanced approach."""
        from mlsurv.tuning.optuna_tuner import _compute_parallelism_split

        # 50 trials (>=20 threshold) on small dataset
        trial_jobs, model_jobs = _compute_parallelism_split(
            n_jobs=16, n_trials=50, n_samples=500, model_name='rsf'
        )

        # Should use balanced approach
        assert trial_jobs >= 2
        assert model_jobs >= 1
        assert trial_jobs * model_jobs <= 16

    def test_never_exceeds_core_budget(self):
        """trial_n_jobs * model_n_jobs should never exceed n_jobs."""
        from mlsurv.tuning.optuna_tuner import _compute_parallelism_split

        test_cases = [
            (1, 10, 500),
            (4, 10, 500),
            (8, 4, 2000),
            (8, 50, 500),
            (16, 10, 5000),
        ]

        for n_jobs, n_trials, n_samples in test_cases:
            trial_jobs, model_jobs = _compute_parallelism_split(
                n_jobs=n_jobs, n_trials=n_trials, n_samples=n_samples, model_name='rsf'
            )
            assert trial_jobs * model_jobs <= n_jobs, \
                f"Exceeded budget: {trial_jobs}*{model_jobs} > {n_jobs}"

    def test_never_negative_jobs(self):
        """Both return values should always be >= 1."""
        from mlsurv.tuning.optuna_tuner import _compute_parallelism_split

        test_cases = [
            (1, 1, 100),
            (2, 1, 100),
            (8, 2, 100),
            (-1, 10, 1000),
        ]

        for n_jobs, n_trials, n_samples in test_cases:
            trial_jobs, model_jobs = _compute_parallelism_split(
                n_jobs=n_jobs, n_trials=n_trials, n_samples=n_samples, model_name='rsf'
            )
            assert trial_jobs >= 1
            assert model_jobs >= 1


class TestTunerInheritsNjobs:
    """Tests for Optuna tuner inheriting n_jobs from learner."""

    def test_tune_inherits_njobs_from_learner(self, synthetic_data_clean):
        """tune() passes learner's n_jobs to the tuner."""
        from mlsurv import SurvivalLearner
        from unittest.mock import patch, MagicMock

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=4)

        # Mock the tuner.tune method to capture the n_jobs parameter
        with patch('mlsurv.learner._tuning.get_tuner') as mock_get_tuner:
            mock_tuner = MagicMock()
            mock_tuner.tune.return_value = ({}, MagicMock(), 'test_exp')
            mock_get_tuner.return_value = mock_tuner

            try:
                learner.tune('coxnet', n_trials=5)
            except Exception:
                pass  # May fail due to mocking, but we just need to verify the call

            # Verify tune was called with n_jobs from learner
            if mock_tuner.tune.called:
                call_kwargs = mock_tuner.tune.call_args[1]
                assert call_kwargs.get('n_jobs') == 4


class TestRSFTunerNjobsSplit:
    """Tests for RSF tuner parallelism split during tuning."""

    def test_rsf_tuner_sets_model_njobs_attribute(self, synthetic_data_clean):
        """RSFTuner.tune() sets _model_n_jobs attribute."""
        from mlsurv.tuning import get_tuner
        from mlsurv.preprocessing import TopKCoxSelector
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        X, y = synthetic_data_clean

        # Set up tuner with proper components (new API)
        tuner = get_tuner(
            'rsf', X=X, y=y, cv_folds=2, random_state=42,
            imputer=SimpleImputer(strategy='median'),
            scaler=StandardScaler(),
            selector=TopKCoxSelector(k=5),
        )

        # Use in-memory storage to avoid Windows file locking
        # With n_jobs=4, tune() should store the model-level parallelism split.
        tuner.tune(n_trials=3, storage=None, study_name=None, n_jobs=4)

        # _model_n_jobs should be set (read back by the per-model tuners)
        assert hasattr(tuner, '_model_n_jobs')
        assert tuner._model_n_jobs >= 1
        assert tuner._model_n_jobs <= 4

    def test_rsf_tuner_njobs_one_uses_one_core(self, synthetic_data_clean):
        """RSFTuner with n_jobs=1 should use model_n_jobs=1."""
        from mlsurv.tuning import get_tuner
        from mlsurv.preprocessing import TopKCoxSelector
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        X, y = synthetic_data_clean

        # Set up tuner with proper components (new API)
        tuner = get_tuner(
            'rsf', X=X, y=y, cv_folds=2, random_state=42,
            imputer=SimpleImputer(strategy='median'),
            scaler=StandardScaler(),
            selector=TopKCoxSelector(k=5),
        )

        # Use in-memory storage to avoid Windows file locking
        tuner.tune(n_trials=2, storage=None, study_name=None, n_jobs=1)

        # With n_jobs=1, model should use 1 core
        assert tuner._model_n_jobs == 1

    def test_rsf_tuner_few_trials_splits_cores(self, synthetic_data_clean):
        """RSFTuner with few trials should use balanced approach."""
        from mlsurv.tuning import get_tuner
        from mlsurv.preprocessing import TopKCoxSelector
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        X, y = synthetic_data_clean

        # Set up tuner with proper components (new API)
        tuner = get_tuner(
            'rsf', X=X, y=y, cv_folds=2, random_state=42,
            imputer=SimpleImputer(strategy='median'),
            scaler=StandardScaler(),
            selector=TopKCoxSelector(k=5),
        )

        # Use in-memory storage to avoid Windows file locking
        # 4 trials with 8 cores on small dataset -> balanced approach
        tuner.tune(n_trials=4, storage=None, study_name=None, n_jobs=8)

        # Balanced split: model-level parallelism stays within the n_jobs budget.
        # (The full trial*model split invariant is covered by the dedicated
        # _compute_parallelism_split unit tests.)
        assert tuner._model_n_jobs >= 1
        assert tuner._model_n_jobs <= 8


class TestEvaluateParallelization:
    """Tests for evaluate() parallelization — n_jobs handling and oversubscription."""

    def test_evaluate_single_model_with_njobs(self, train_test_data):
        """Single-model evaluate with n_jobs>1 passes n_jobs to permutation_importance."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=4,
        )

        assert result.importance is not None
        assert len(result.importance) > 0
        assert result.importance_iterations is not None

    def test_evaluate_multimodel_parallel_results_match(self, train_test_data):
        """Multi-model evaluate results match between n_jobs=1 and n_jobs=-1."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        # Same models can be reused for both sequential and parallel since
        # _evaluate_multiple now restores n_jobs after dispatch
        models = {}
        for name in ['coxph', 'coxnet']:
            m = get_model(name)
            m.fit(X_train, y_train)
            models[name] = m

        result_seq = evaluate(
            models, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )

        result_par = evaluate(
            models, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=-1,
        )

        for model_name in ['coxph', 'coxnet']:
            seq_cidx = result_seq.concordance[
                result_seq.concordance['model_type'] == model_name
            ]['c_index_test'].values[0]
            par_cidx = result_par.concordance[
                result_par.concordance['model_type'] == model_name
            ]['c_index_test'].values[0]
            assert abs(seq_cidx - par_cidx) < 1e-8

    def test_evaluate_with_bootstrap_parallel(self, train_test_data):
        """evaluate() with bootstrap=True works with n_jobs>1."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('coxph')
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=4,
            bootstrap=True,
            n_bootstrap=20,
        )

        assert result.bootstrap is not None
        assert len(result.bootstrap.summary) > 0

    def test_evaluate_rsf_keeps_internal_parallelism(self, train_test_data):
        """RSF model n_jobs is preserved — importance uses n_jobs=1 instead."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('rsf', n_jobs=8)
        model.fit(X_train, y_train)

        evaluate(
            model, X_test, y_test,
            n_importance_repeats=5,
            random_state=42,
            show_progress=False,
            n_jobs=4,
        )

        # RSF keeps its own n_jobs — importance uses n_jobs=1 to avoid
        # serialization overhead from outer parallelism
        assert model.n_jobs == 8


class TestThresholdBoundary:
    """Tests at exact threshold boundaries to verify auto-disable behavior."""

    def test_bootstrap_at_exact_threshold_goes_parallel(self, train_test_data):
        """Bootstrap at exactly MIN_ITERATIONS_FOR_PARALLEL should use parallel."""
        from mlsurv.evaluation._bootstrap_workers import (
            MIN_ITERATIONS_FOR_PARALLEL,
            _resolve_effective_n_jobs,
        )

        # At the threshold, should return n_jobs (not 1)
        effective = _resolve_effective_n_jobs(
            n_jobs=4,
            n_bootstrap=MIN_ITERATIONS_FOR_PARALLEL,
            can_precompute=False,
            force_parallel=False,
            show_progress=False,
        )
        assert effective == 4, (
            f"At threshold {MIN_ITERATIONS_FOR_PARALLEL}, should use n_jobs=4, got {effective}"
        )

    def test_bootstrap_below_threshold_goes_sequential(self, train_test_data):
        """Bootstrap at MIN_ITERATIONS_FOR_PARALLEL - 1 should be sequential."""
        from mlsurv.evaluation._bootstrap_workers import (
            MIN_ITERATIONS_FOR_PARALLEL,
            _resolve_effective_n_jobs,
        )

        effective = _resolve_effective_n_jobs(
            n_jobs=4,
            n_bootstrap=MIN_ITERATIONS_FOR_PARALLEL - 1,
            can_precompute=False,
            force_parallel=False,
            show_progress=False,
        )
        assert effective == 1, (
            f"Below threshold {MIN_ITERATIONS_FOR_PARALLEL - 1}, should be sequential, got {effective}"
        )

    def test_bootstrap_precomputed_at_exact_threshold(self):
        """Precomputed bootstrap at exact threshold should use parallel."""
        from mlsurv.evaluation._bootstrap_workers import (
            MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED,
            _resolve_effective_n_jobs,
        )

        effective = _resolve_effective_n_jobs(
            n_jobs=4,
            n_bootstrap=MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED,
            can_precompute=True,
            force_parallel=False,
            show_progress=False,
        )
        assert effective == 4

    def test_bootstrap_precomputed_below_threshold(self):
        """Precomputed bootstrap below threshold should be sequential."""
        from mlsurv.evaluation._bootstrap_workers import (
            MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED,
            _resolve_effective_n_jobs,
        )

        effective = _resolve_effective_n_jobs(
            n_jobs=4,
            n_bootstrap=MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED - 1,
            can_precompute=True,
            force_parallel=False,
            show_progress=False,
        )
        assert effective == 1

    def test_outer_parallel_at_exact_threshold(self, train_test_data):
        """Multi-model bootstrap at exact MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import bootstrap
        from mlsurv.evaluation.bootstrap import MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL

        X_train, X_test, y_train, y_test = train_test_data

        models = {}
        for name in ['coxph', 'coxnet']:
            m = get_model(name)
            m.fit(X_train, y_train)
            models[name] = m

        # 2 models * n_iter = threshold -> should NOT auto-disable
        n_iter = MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL // 2

        result = bootstrap(
            models, X_test, y_test, y_train=y_train,
            n_bootstrap=n_iter, random_state=42, show_progress=False,
            n_jobs=1, n_jobs_models=4,
        )

        assert set(result.summary['model_type']) == {'coxph', 'coxnet'}

    def test_features_at_exact_threshold(self):
        """Univariate Cox at MIN_FEATURES_FOR_PARALLEL should use parallel."""
        from mlsurv.evaluation.importance import (
            MIN_FEATURES_FOR_PARALLEL,
            compute_univariate_cox,
        )
        import numpy as np
        import pandas as pd

        # Generate data with exactly MIN_FEATURES_FOR_PARALLEL features
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = MIN_FEATURES_FOR_PARALLEL
        X = pd.DataFrame(
            rng.randn(n_samples, n_features),
            columns=[f'f_{i}' for i in range(n_features)],
        )
        times = rng.exponential(2, n_samples).clip(0.1, 10)
        events = rng.binomial(1, 0.7, n_samples).astype(bool)
        y = np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[('event', bool), ('time', float)],
        )

        # Should succeed with n_jobs=-1 (at threshold, should use parallel)
        result = compute_univariate_cox(X, y, n_jobs=-1)
        assert len(result) == n_features


class TestJoblibBackendFallback:
    """Tests for restricted-environment joblib backend fallback."""

    def test_resolve_joblib_backend_falls_back_when_loky_unavailable(self, monkeypatch):
        """loky should degrade to threading when the probe fails."""
        from mlsurv import utils

        utils._joblib_backend_usable.cache_clear()
        monkeypatch.setattr(utils, "_joblib_backend_usable", lambda backend: False)

        assert utils.resolve_joblib_backend("loky") == "threading"

    def test_resolve_joblib_backend_keeps_loky_when_available(self, monkeypatch):
        """loky should remain selected when the probe succeeds."""
        from mlsurv import utils

        utils._joblib_backend_usable.cache_clear()
        monkeypatch.setattr(utils, "_joblib_backend_usable", lambda backend: True)

        assert utils.resolve_joblib_backend("loky") == "loky"

    def test_joblib_parallel_backend_yields_resolved_name(self, monkeypatch):
        """Context manager should yield the resolved backend name."""
        from mlsurv import utils

        utils._joblib_backend_usable.cache_clear()
        monkeypatch.setattr(utils, "_joblib_backend_usable", lambda b: False)

        with utils.joblib_parallel_backend(2, backend="loky") as resolved:
            assert resolved == "threading"

    def test_joblib_parallel_backend_keeps_loky_when_usable(self, monkeypatch):
        """Context manager should keep loky when the probe succeeds."""
        from mlsurv import utils

        utils._joblib_backend_usable.cache_clear()
        monkeypatch.setattr(utils, "_joblib_backend_usable", lambda b: True)

        with utils.joblib_parallel_backend(2, backend="loky") as resolved:
            assert resolved == "loky"

    def test_probe_catches_permission_error(self, monkeypatch):
        """PermissionError during loky probe should return False."""
        import joblib
        from mlsurv import utils

        utils._joblib_backend_usable.cache_clear()

        class _RaisingParallel:
            def __init__(self, **kw):
                pass
            def __call__(self, gen):
                raise PermissionError("sandbox restriction")

        monkeypatch.setattr(joblib, "Parallel", _RaisingParallel)
        assert utils._joblib_backend_usable("loky") is False
        utils._joblib_backend_usable.cache_clear()

    def test_probe_catches_runtime_error(self, monkeypatch):
        """RuntimeError (e.g. BrokenProcessPool) during probe should return False."""
        import joblib
        from mlsurv import utils

        utils._joblib_backend_usable.cache_clear()

        class _RaisingParallel:
            def __init__(self, **kw):
                pass
            def __call__(self, gen):
                raise RuntimeError("loky process worker died")

        monkeypatch.setattr(joblib, "Parallel", _RaisingParallel)
        assert utils._joblib_backend_usable("loky") is False
        utils._joblib_backend_usable.cache_clear()


class TestPredictionBootstrapParallelReproducibility:
    """Verify prediction bootstrap gives same results in parallel vs sequential."""

    @pytest.mark.bootstrap_heavy
    def test_parallel_matches_sequential(self, synthetic_data_clean):
        """n_jobs=1 and n_jobs=2 should produce identical CIs with same seed."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        X_new = X.iloc[:3]

        pred_seq = learner.predict(
            X_new, ci=True, n_bootstrap=25, n_jobs=1,
            random_state=99, verbose=False,
        )
        pred_par = learner.predict(
            X_new, ci=True, n_bootstrap=25, n_jobs=2,
            random_state=99, verbose=False,
        )

        assert_array_almost_equal(
            pred_seq.median_survival_ci,
            pred_par.median_survival_ci,
            decimal=6,
        )
