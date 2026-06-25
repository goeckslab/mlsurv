"""End-to-end integration tests."""

import pytest
import numpy as np
import pandas as pd

pytestmark = pytest.mark.integration

# RSF fit/predict cost scales with forest size, so workflow-plumbing tests (which
# only assert structure, keys, or no-crash) shrink the forest to keep them fast.
# Tuning tests pass a tiny Optuna space; partial spaces merge per-param with the
# defaults, so the unlisted RSF params keep their default ranges. Untuned tests
# pass a small n_estimators directly, since search_spaces is a no-op without tuning.
_SMALL_RSF_SPACE = {'n_estimators': (10, 20), 'max_depth': (3, 5)}
_SMALL_RSF_PARAMS = {'n_estimators': 20}


class TestFullWorkflow:
    """Test complete analysis workflows."""

    @pytest.mark.integration_heavy
    def test_setup_compare_evaluate_bootstrap(self, synthetic_data_clean):
        """Test full workflow: setup -> compare -> evaluate -> bootstrap."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        # Step 1: Setup
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        assert learner._state.is_setup
        assert len(learner.X_train) > 0
        assert len(learner.X_test) > 0

        # Step 2: Compare models (run() tunes by default)
        comparison = learner.run(
            models=['coxph', 'rsf'],
            verbose=False,
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
        )

        assert comparison is not None
        assert len(comparison) == 2
        assert learner.best_model is not None

        # Step 3: Evaluate best model
        best_result = learner.best_model
        eval_result = learner.evaluate(best_result, n_importance_repeats=5)

        assert eval_result is not None
        assert eval_result.concordance is not None
        assert eval_result.importance is not None

        # Step 4: Bootstrap confidence intervals
        boot_result = learner.bootstrap(best_result, n_bootstrap=10)

        assert boot_result is not None
        assert boot_result.summary is not None
        assert 'ci_lower' in boot_result.summary.columns
        assert 'ci_upper' in boot_result.summary.columns

    @pytest.mark.integration_heavy
    def test_setup_tune_evaluate(self, small_data):
        """Test workflow with tuning."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)

        # Setup
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Tune a model
        tuned = learner.tune(
            'rsf',
            n_trials=2,
            search_space=_SMALL_RSF_SPACE
        )

        assert tuned is not None
        assert tuned.best_params_ is not None
        # Check for params with or without model__ prefix
        has_params = (
            'n_estimators' in tuned.best_params_ or
            'max_depth' in tuned.best_params_ or
            'model__n_estimators' in tuned.best_params_ or
            'model__max_depth' in tuned.best_params_
        )
        assert has_params, f"Expected tuned params not found in: {tuned.best_params_}"

        # Evaluate tuned model
        eval_result = learner.evaluate(tuned, n_importance_repeats=5)

        assert eval_result is not None
        assert 0.0 <= eval_result.concordance['c_index_test'].iloc[0] <= 1.0

    @pytest.mark.integration_heavy
    def test_multiple_model_comparison(self, synthetic_data_clean):
        """Test comparing multiple models end-to-end."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3)

        # Compare multiple models (run() tunes by default)
        result = learner.run(
            models=['coxph', 'coxnet', 'rsf'],
            verbose=False,
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
        )

        # Check all models were trained
        assert len(result) == 3
        assert 'coxph' in result
        assert 'coxnet' in result
        assert 'rsf' in result

        # Metrics DataFrame should have all models
        # run() returns a bare dict; the CV metrics summary lives on learner.summary()
        metrics = learner.summary()
        assert metrics['model'].nunique() == 3
        assert 'model' in metrics.columns
        assert 'mean' in metrics.columns

        # Best model should be identified
        assert learner.best_model.name in ['coxph', 'coxnet', 'rsf']

    def test_run_quick_api(self, small_data):
        """Test run() quick API end-to-end."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)

        # Setup first, then compare models
        learner.setup(test_fraction=0.3, cv_folds=3)
        result = learner.run(
            models=['coxph'],
            verbose=False,
            evaluate=False,  # Skip evaluation for faster test
        )

        assert result is not None
        assert len(result) == 1
        assert learner.best_model is not None

    def test_external_validation(self, synthetic_data_clean):
        """Test external validation workflow."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        # Split data to simulate external cohort
        split_idx = int(len(X) * 0.7)
        X_internal = X.iloc[:split_idx]
        y_internal = y[:split_idx]
        X_external = X.iloc[split_idx:]
        y_external = y[split_idx:]

        # Train on internal data
        learner = SurvivalLearner(X_internal, y_internal)
        learner.setup(test_fraction=0.2)

        model_result = learner.train('coxph')

        # Validate on external data
        val_result = learner.validate(
            X_external, y_external,
            models=model_result,
            name='External_Cohort'
        )

        assert val_result is not None
        assert val_result.name == 'External_Cohort'
        assert 0.0 <= val_result.evaluation.c_index_test <= 1.0
        assert len(val_result.test_set) == len(X_external)


class TestRealDatasets:
    """Test with real survival datasets."""

    def _convert_gbsg2_y(self, y):
        """Convert GBSG2 y from ('cens', 'time') to ('event', 'time')."""
        # GBSG2 uses 'cens' for censoring indicator (0=censored, 1=event)
        return np.array(
            [(bool(row['cens']), row['time']) for row in y],
            dtype=[('event', bool), ('time', float)]
        )

    def test_gbsg2_full_pipeline(self, gbsg2_data):
        """Test full pipeline on GBSG2 breast cancer data."""
        from mlsurv import SurvivalLearner

        X, y_orig = gbsg2_data
        y = self._convert_gbsg2_y(y_orig)

        # One-hot encode categorical columns
        X_encoded = pd.get_dummies(X, drop_first=True)

        learner = SurvivalLearner(X_encoded, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Train CoxPH (fast)
        result = learner.train('coxph')

        assert result is not None
        assert 0.0 <= result.cv_metric_mean <= 1.0

        # Evaluate
        eval_result = learner.evaluate(result, n_importance_repeats=5)

        assert eval_result is not None
        # Real data should give reasonable C-index
        c_index = eval_result.concordance['c_index_test'].iloc[0]
        assert 0.5 <= c_index <= 0.9

    @pytest.mark.integration_heavy
    def test_gbsg2_run(self, gbsg2_data):
        """Test model comparison on GBSG2 data."""
        from mlsurv import SurvivalLearner

        X, y_orig = gbsg2_data
        y = self._convert_gbsg2_y(y_orig)
        X_encoded = pd.get_dummies(X, drop_first=True)

        learner = SurvivalLearner(X_encoded, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # A 30-tree RSF still clears the 0.5 floor on GBSG2 while running far
        # faster than the 100-tree auto-default forest (9 features, floored at 100).
        result = learner.run(
            models=['coxph', 'rsf'],
            tune=False,
            verbose=False,
            model_params={'rsf': {'n_estimators': 30}},
        )

        assert len(result) == 2

        # Both models should have reasonable performance
        for name, model_result in result.items():
            assert 0.5 <= model_result.cv_metric_mean <= 0.9


class TestPipelineWithFeatureSelection:
    """Test pipelines with feature selection."""

    def test_full_pipeline_with_topk_selection(self, synthetic_data_clean):
        """Test full pipeline with TopK feature selection."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        # Setup with feature selector
        selector = TopKCoxSelector(k=10)
        learner.setup(
            test_fraction=0.3,
            cv_folds=3,
            feature_selector=selector
        )

        result = learner.train('coxph')

        assert result is not None
        # Pipeline should contain selector
        assert 'selector' in [name for name, _ in result.model.steps]

    @pytest.mark.integration_heavy
    def test_compare_with_feature_selection(self, synthetic_data_clean):
        """Test model comparison with feature selection in pipeline."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        selector = TopKCoxSelector(k=10)
        learner.setup(
            test_fraction=0.3,
            cv_folds=3,
            feature_selector=selector
        )

        # run() tunes by default
        result = learner.run(
            models=['coxph', 'rsf'],
            verbose=False,
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
        )

        assert len(result) == 2


class TestErrorHandling:
    """Test error handling in workflows."""

    def test_invalid_model_name(self, small_data):
        """Verify error on invalid model name."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        with pytest.raises(ValueError, match="Unknown model"):
            learner.train('nonexistent_model')

    def test_groupkfold_without_groups(self, small_data):
        """Verify error when using GroupKFold without groups."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)

        with pytest.raises(ValueError, match="groups required"):
            learner.setup(cv_strategy='groupkfold')


class TestFullPipelineConfigurations:
    """Test complete pipelines in varying configurations."""

    def test_pipeline_coxph_no_preprocessing(self, synthetic_data_clean):
        """Test CoxPH pipeline without preprocessing."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=3,
            imputer=None,
            scaler=None,
            random_state=42
        )

        result = learner.train('coxph')

        assert result is not None
        assert 0.0 <= result.cv_metric_mean <= 1.0
        # Pipeline should only have model step
        assert 'imputer' not in result.model.named_steps
        assert 'scaler' not in result.model.named_steps

    def test_pipeline_rsf_with_feature_selection(self, synthetic_data_clean):
        """Test RSF pipeline with feature selection."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        selector = TopKCoxSelector(k=10)
        learner.setup(
            test_fraction=0.3,
            cv_folds=3,
            feature_selector=selector,
            random_state=42
        )

        result = learner.train('rsf', n_estimators=10, random_state=42)

        assert result is not None
        assert 'selector' in result.model.named_steps

    @pytest.mark.integration_heavy
    def test_pipeline_gbsa_with_tuning(self, small_data):
        """Test GBSA pipeline with tuning."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.tune(
            'gbsa',
            n_trials=3,
            search_space={
                'n_estimators': (10, 30),
                'max_depth': (2, 4)
            }
        )

        assert result is not None
        assert result.best_params_ is not None

    @pytest.mark.integration_heavy
    def test_full_pipeline_tune_select_evaluate_bootstrap(self, synthetic_data_clean):
        """Test complete pipeline: tune -> select -> evaluate -> bootstrap."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        selector = TopKCoxSelector(k=10)
        learner.setup(
            test_fraction=0.3,
            cv_folds=3,
            feature_selector=selector,
            random_state=42
        )

        # Tune
        tuned = learner.tune(
            'rsf',
            n_trials=2,
            search_space=_SMALL_RSF_SPACE
        )
        assert tuned.best_params_ is not None

        # Evaluate
        eval_result = learner.evaluate(tuned, n_importance_repeats=5, bootstrap=False)
        assert eval_result is not None
        assert eval_result.concordance is not None

        # Bootstrap
        boot_result = learner.bootstrap(tuned, n_bootstrap=10)
        assert boot_result is not None
        assert 'ci_lower' in boot_result.summary.columns

    @pytest.mark.integration_heavy
    def test_full_pipeline_compare_tune_best_model_report(self, small_data, tmp_path):
        """Test compare -> tune -> report pipeline."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Compare with tuning
        comparison = learner.run(
            models=['coxph', 'rsf'],
            tune=True,
            n_trials=2,
            verbose=False,
            evaluate=True,
            bootstrap=False,  # Skip bootstrap for speed
            search_spaces={'rsf': _SMALL_RSF_SPACE}
        )

        assert comparison is not None
        assert learner.best_model is not None

        # Generate report
        report_path = tmp_path / "test_report.html"
        learner.generate_report(str(report_path))

        assert report_path.exists()

    def test_pipeline_with_stratified_cv(self, synthetic_data_clean):
        """Test pipeline with stratified cross-validation."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=5,  # 5 folds retained: assertion below verifies fold count
            cv_strategy='stratifiedkfold',
            random_state=42
        )

        result = learner.train('coxph')
        assert result is not None
        # CV scores should have at least 5 entries (some may be NaN for failed folds)
        # The framework may compute multiple metrics per fold
        assert len(result.cv_metric_scores) >= 5
        # Should have a valid mean
        assert result.cv_metric_mean > 0

    @pytest.mark.integration_heavy
    def test_pipeline_with_repeated_cv(self, synthetic_data_clean):
        """Test pipeline with repeated cross-validation."""
        from mlsurv import SurvivalLearner
        import warnings

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.setup(
                test_fraction=0.3,
                cv_folds=3,
                cv_repeats=2,
                random_state=42
            )

        result = learner.train('coxph')
        assert result is not None
        # Should have at least 3 folds * 2 repeats = 6 scores
        assert len(result.cv_metric_scores) >= 6
        # Should have a valid mean
        assert result.cv_metric_mean > 0

    @pytest.mark.integration_heavy
    def test_pipeline_with_custom_cv_object(self, synthetic_data_clean):
        """Test pipeline with custom CV object."""
        from mlsurv import SurvivalLearner
        from sklearn.model_selection import RepeatedStratifiedKFold

        X, y = synthetic_data_clean

        custom_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)

        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_strategy=custom_cv,
            random_state=42
        )

        result = learner.train('coxph')
        assert result is not None
        # Custom CV: should have at least 3 splits * 2 repeats = 6 scores
        assert len(result.cv_metric_scores) >= 6
        # Should have a valid mean
        assert result.cv_metric_mean > 0

    def test_compare_all_models_pipeline(self, synthetic_data_clean):
        """Test comparing all available models."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Compare multiple models (excluding slow ones); run() tunes by default
        result = learner.run(
            models=['coxph', 'coxnet', 'rsf'],
            verbose=False,
            evaluate=False,
            bootstrap=False,
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
        )

        assert len(result) == 3
        assert learner.best_model is not None

    @pytest.mark.integration_heavy
    def test_compare_selected_models_with_tuning(self, small_data):
        """Test comparing selected models with tuning."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.run(
            models=['coxnet', 'rsf'],
            tune=True,
            n_trials=2,
            verbose=False,
            evaluate=False,
            bootstrap=False,
            search_spaces={
                'coxnet': {'l1_ratio': (0.1, 0.9)},
                'rsf': _SMALL_RSF_SPACE
            }
        )

        assert len(result) == 2
        # At least one should have been tuned
        assert any(
            hasattr(result[m], 'best_params_') and result[m].best_params_
            for m in result
        )


@pytest.mark.integration_heavy
class TestPipelineWithDeepSurv:
    """Integration tests specifically for DeepSurv."""

    @pytest.fixture
    def check_torch(self):
        """Skip tests if torch not available."""
        pytest.importorskip("torch")

    def test_deepsurv_in_comparison_pipeline(self, small_data, check_torch):
        """Test DeepSurv in model comparison."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.run(
            models=['coxph', 'deepsurv'],
            verbose=False,
            evaluate=False,
            bootstrap=False
        )

        assert 'deepsurv' in result
        assert result['deepsurv'].cv_metric_mean > 0

    def test_deepsurv_tuning_pipeline(self, small_data, check_torch):
        """Test DeepSurv tuning."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.tune(
            'deepsurv',
            n_trials=2,
            search_space={
                'n_layers': (1, 2),
                'n_nodes': (8, 16)
            }
        )

        assert result is not None
        assert result.best_params_ is not None

    def test_deepsurv_with_feature_selection(self, small_data, check_torch):
        """Test DeepSurv with feature selection."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = small_data
        learner = SurvivalLearner(X, y)

        selector = TopKCoxSelector(k=3)
        learner.setup(
            test_fraction=0.3,
            cv_folds=3,
            feature_selector=selector,
            random_state=42
        )

        result = learner.train('deepsurv', layers=[8], epochs=10)

        assert result is not None
        assert 'selector' in result.model.named_steps

    def test_deepsurv_dynamic_defaults_applied(self, small_data, check_torch):
        """Test that dynamic defaults are applied when training without explicit params."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Train without specifying parameters - should use dynamic defaults
        result = learner.train('deepsurv')

        # Verify model was created (dynamic defaults worked)
        assert result is not None
        assert result.cv_metric_mean > 0

        # Get the actual model and verify it has reasonable defaults
        model = result.model.named_steps['model']

        # For small data (50 samples), expect high dropout, few nodes
        assert model.dropout >= 0.15  # Dynamic defaults use EPV-based dropout
        assert len(model.layers) == 1  # Paper found 1 layer optimal
        assert model.activation == 'selu'  # Optimal for 3/4 datasets

    def test_deepsurv_user_params_override_defaults(self, small_data, check_torch):
        """Test that user-provided params override dynamic defaults."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Train with explicit parameters
        result = learner.train(
            'deepsurv',
            dropout=0.05,  # Lower than what defaults would compute
            layers=[64, 64],  # More than defaults would suggest
            activation='relu'  # Different from SELU default
        )

        # Get the actual model
        model = result.model.named_steps['model']

        # User params should override defaults
        assert model.dropout == 0.05
        assert model.layers == [64, 64]
        assert model.activation == 'relu'


class TestNestedCVExecution:
    """Test actual nested CV execution."""

    def test_nested_cv_basic_structure(self, synthetic_data_clean):
        """Test nested CV structure is correct."""
        from sklearn.model_selection import StratifiedKFold

        X, y = synthetic_data_clean

        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        outer_scores = []

        for outer_train_idx, outer_test_idx in outer_cv.split(X, y['event']):
            X_train_outer = X.iloc[outer_train_idx]
            y_train_outer = y[outer_train_idx]
            # Inner CV for tuning would go here
            # For now, just verify structure
            inner_splits = list(inner_cv.split(X_train_outer, y_train_outer['event']))
            assert len(inner_splits) == 3

            # Each inner split should be within outer training bounds
            for inner_train_idx, inner_val_idx in inner_splits:
                assert all(i < len(X_train_outer) for i in inner_train_idx)
                assert all(i < len(X_train_outer) for i in inner_val_idx)

            outer_scores.append(len(outer_test_idx))

        assert len(outer_scores) == 3

    def test_nested_cv_outer_inner_scores_differ(self, small_data):
        """Test that inner and outer CV scores can differ."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Get CV score
        result = learner.train('coxph')
        cv_score = result.cv_metric_mean

        # Get test score
        eval_result = learner.evaluate(result, n_importance_repeats=5, bootstrap=False)

        # These can differ (this is expected behavior)
        # We just verify both produce valid results
        assert 0 <= cv_score <= 1
        assert eval_result.concordance is not None


@pytest.mark.integration_heavy
class TestRealDatasetsExtended:
    """Extended tests with real survival datasets."""

    def _convert_gbsg2_y(self, y):
        """Convert GBSG2 y from ('cens', 'time') to ('event', 'time')."""
        return np.array(
            [(bool(row['cens']), row['time']) for row in y],
            dtype=[('event', bool), ('time', float)]
        )

    def _convert_veterans_y(self, y):
        """Convert Veterans y to standard format."""
        return np.array(
            [(bool(row[0]), float(row[1])) for row in y],
            dtype=[('event', bool), ('time', float)]
        )

    def test_veterans_full_pipeline(self, veterans_data):
        """Test full pipeline on Veterans lung cancer data."""
        from mlsurv import SurvivalLearner

        X, y_orig = veterans_data
        y = self._convert_veterans_y(y_orig)

        # One-hot encode categorical columns
        X_encoded = pd.get_dummies(X, drop_first=True)

        learner = SurvivalLearner(X_encoded, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.train('coxph')

        assert result is not None
        assert 0.0 <= result.cv_metric_mean <= 1.0

    def test_gbsg2_full_pipeline_with_report(self, gbsg2_data, tmp_path):
        """Test full pipeline with report on GBSG2."""
        from mlsurv import SurvivalLearner

        X, y_orig = gbsg2_data
        y = self._convert_gbsg2_y(y_orig)
        X_encoded = pd.get_dummies(X, drop_first=True)

        learner = SurvivalLearner(X_encoded, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Train and evaluate
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=5, bootstrap=False)

        assert eval_result is not None

        # Generate report
        report_path = tmp_path / "gbsg2_report.html"
        learner.generate_report(str(report_path))

        assert report_path.exists()


@pytest.mark.integration_heavy
class TestLargeDatasetHandling:
    """Test handling of larger datasets."""

    def test_large_dataset_coxph(self, large_synthetic_data):
        """Test CoxPH with large dataset."""
        from mlsurv import SurvivalLearner

        X, y = large_synthetic_data

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.2, cv_folds=3, random_state=42)

        result = learner.train('coxph')

        assert result is not None
        assert len(learner.X_train) >= 800  # 80% of 1000

    def test_large_dataset_rsf(self, large_synthetic_data):
        """Test RSF with large dataset."""
        from mlsurv import SurvivalLearner

        X, y = large_synthetic_data

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.2, cv_folds=3, random_state=42)

        # Use fewer trees for speed
        result = learner.train('rsf', n_estimators=20, random_state=42)

        assert result is not None
        assert 0.4 <= result.cv_metric_mean <= 1.0  # Should be better than random


class TestNewModels:
    """Integration tests for newly added models.

    The per-model ``learner.train()`` smokes (xgbcox / deephit / weibull_aft)
    were removed: each model's fit/predict/score contract is unit-tested
    in test_new_models.py, and the learner-pipeline training of all three is
    still exercised by ``test_compare_new_models`` below (which runs them via
    ``learner.run``). ``test_compare_new_models`` is retained because the new
    models are not part of ``test_compare_all_models_pipeline`` (coxph/coxnet/rsf).
    """

    @pytest.mark.integration_heavy
    def test_compare_new_models(self, synthetic_data_clean):
        """Test comparing new models together."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Compare new models (skip deephit if torch not available)
        models = ['xgbcox', 'weibull_aft']
        try:
            import torch  # noqa: F401
            import pycox  # noqa: F401
            models.append('deephit')
        except ImportError:
            pass

        result = learner.run(models=models, verbose=False, tune=False, evaluate=False)

        assert result is not None
        assert len(result) >= 2
        assert learner.best_model is not None


@pytest.mark.integration
class TestRunStringModels:
    """Test run() with string model argument."""

    def test_run_string_single_model(self, small_data):
        """run('rsf') should work like run(models=['rsf'])."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.run('coxph', tune=False, verbose=False, evaluate=False)

        assert result is not None
        assert len(result) == 1
        assert 'coxph' in result

    def test_run_string_with_kwargs(self, small_data):
        """run('rsf', tune=False, n_estimators=50) forwards kwargs."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.run(
            'rsf', tune=False, verbose=False, evaluate=False,
            n_estimators=50,
        )

        assert result is not None
        assert 'rsf' in result
        # Verify the model was trained with the specified params
        model = result['rsf'].model.named_steps['model']
        assert model.n_estimators == 50


@pytest.mark.integration
class TestModelParams:
    """Test model_params for run() and train()."""

    def test_run_model_params_single(self, small_data):
        """run() with model_params for a single model."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.run(
            'rsf', tune=False, verbose=False, evaluate=False,
            model_params={'rsf': {'n_estimators': 50}},
        )

        assert 'rsf' in result
        model = result['rsf'].model.named_steps['model']
        assert model.n_estimators == 50

    def test_run_model_params_multiple(self, small_data):
        """run() with model_params for multiple models."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.run(
            ['coxph', 'rsf'], tune=False, verbose=False, evaluate=False,
            model_params={'rsf': {'n_estimators': 50}},
        )

        assert len(result) == 2
        rsf_model = result['rsf'].model.named_steps['model']
        assert rsf_model.n_estimators == 50

    def test_run_kwargs_multi_model_warns(self, small_data):
        """run() with **kwargs and multiple models should warn."""
        from mlsurv import SurvivalLearner
        import warnings as w

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            result = learner.run(
                ['coxph', 'rsf'], tune=False, verbose=False, evaluate=False,
                n_estimators=50,
            )

        assert result is not None
        warned = [str(c.message) for c in caught if "model_params" in str(c.message)]
        assert len(warned) > 0, "Expected warning about kwargs with multiple models"

    def test_run_tune_with_model_params_warns(self, small_data):
        """run() with tune=True + model_params should warn."""
        from mlsurv import SurvivalLearner
        import warnings as w

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            learner.run(
                'coxph', tune=True, n_trials=2, verbose=False,
                evaluate=False, bootstrap=False,
                model_params={'coxph': {'alpha': 0.1}},
            )

        warned = [str(c.message) for c in caught if "ignored when tune=True" in str(c.message)]
        assert len(warned) > 0, "Expected warning about model_params with tune=True"

    def test_run_both_model_params_and_kwargs_raises(self, small_data):
        """run() with both model_params and **kwargs should raise ValueError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with pytest.raises(ValueError, match="Cannot specify both"):
            learner.run(
                'rsf', tune=False, verbose=False, evaluate=False,
                model_params={'rsf': {'n_estimators': 50}},
                n_estimators=100,
            )

    def test_train_model_params_single(self, small_data):
        """train() with model_params for a single model."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.train(
            'rsf', verbose=False,
            model_params={'rsf': {'n_estimators': 50}},
        )

        assert result is not None
        model = result.model.named_steps['model']
        assert model.n_estimators == 50

    def test_train_model_params_multiple(self, small_data):
        """train() with model_params for multiple models."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        results = learner.train(
            ['coxph', 'rsf'], verbose=False,
            model_params={'rsf': {'n_estimators': 50}},
        )

        assert len(results) == 2
        rsf_model = results['rsf'].model.named_steps['model']
        assert rsf_model.n_estimators == 50

    def test_train_both_model_params_and_kwargs_raises(self, small_data):
        """train() with both model_params and **kwargs should raise ValueError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with pytest.raises(ValueError, match="Cannot specify both"):
            learner.train(
                'rsf', verbose=False,
                model_params={'rsf': {'n_estimators': 50}},
                n_estimators=100,
            )


@pytest.mark.integration
class TestGetMetricInference:
    """Test ModelResult.get_metric() via dict returned by run()."""

    def test_single_model_infers(self, small_data):
        """get_metric() on the sole ModelResult returns a valid value."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.run('coxph', tune=False, verbose=False, evaluate=False)

        # run() returns dict; access the single ModelResult directly
        metric_val = result['coxph'].get_metric(metric='harrell', phase='cv')
        assert metric_val is not None
        assert 0.0 <= metric_val <= 1.0

    def test_multiple_models_uses_best(self, small_data):
        """get_metric() on best_model matches direct access."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.run(
            ['coxph', 'rsf'], tune=False, verbose=False, evaluate=False,
        )

        # run() returns dict; use learner.best_model to find the best
        best = learner.best_model
        metric_val = best.get_metric(metric='harrell', phase='cv')
        best_direct = result[best.name].get_metric('harrell', 'cv')
        assert metric_val == best_direct


@pytest.mark.integration
class TestNTrialsDict:
    """Test n_trials as dict for per-model trial counts."""

    @pytest.mark.integration_heavy
    def test_train_n_trials_dict(self, small_data):
        """train() with n_trials dict resolves per-model."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        # Different trial counts per model
        results = learner.train(
            ['coxph', 'rsf'], tune=True, verbose=False,
            n_trials={'coxph': 2, 'rsf': 2},
            search_spaces={'rsf': _SMALL_RSF_SPACE},
        )

        assert len(results) == 2
        assert 'coxph' in results
        assert 'rsf' in results

    @pytest.mark.integration_heavy
    def test_run_n_trials_dict(self, small_data):
        """run() with n_trials dict resolves per-model."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.run(
            ['coxph', 'rsf'], tune=True, n_trials={'coxph': 2, 'rsf': 2},
            verbose=False, evaluate=False, bootstrap=False,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
        )

        assert len(result) == 2


@pytest.mark.integration
class TestNImportanceRepeatsDict:
    """Test n_importance_repeats as dict for per-model repeat counts."""

    def test_evaluate_n_importance_repeats_dict(self, small_data):
        """evaluate() with n_importance_repeats dict resolves per-model."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        learner.train(
            ['coxph', 'rsf'], verbose=False,
            model_params={'rsf': _SMALL_RSF_PARAMS},
        )

        result = learner.evaluate(
            n_importance_repeats={'coxph': 3, 'rsf': 2},
            verbose=False, bootstrap=False,
        )

        assert result is not None
        assert result.importance is not None
        # Should have importance for both models
        model_types = result.importance['model_type'].unique()
        assert 'coxph' in model_types
        assert 'rsf' in model_types

    def test_evaluate_n_importance_repeats_dict_fallback(self, small_data):
        """Models not in the dict fall back to 'auto'."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        learner.train(['coxph', 'rsf'], verbose=False)

        # Only coxph specified; rsf falls back to auto
        result = learner.evaluate(
            n_importance_repeats={'coxph': 3},
            verbose=False, bootstrap=False,
        )

        assert result is not None
        assert result.importance is not None


# TestResolvePerModel (pure _resolve_per_model dict-lookup unit tests) was moved
# to test_kwarg_forwarding.py — it does no model fitting and belongs with
# the per-model kwarg-resolution unit tests, not in the integration suite.


@pytest.mark.integration
class TestValidation:
    """Test input validation for model_params, n_trials dict, etc."""

    def test_model_params_bad_structure_raises(self, small_data):
        """model_params values must be dicts, not scalars."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with pytest.raises(TypeError, match="model_params.*must be a dict"):
            learner.train('rsf', model_params={'rsf': 42})

    def test_run_model_params_bad_structure_raises(self, small_data):
        """run() model_params values must be dicts."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with pytest.raises(TypeError, match="model_params.*must be a dict"):
            learner.run(['rsf'], tune=False, model_params={'rsf': [1, 2]})

    def test_unused_model_params_keys_warns(self, small_data):
        """model_params with keys not in models list emits warning."""
        import warnings as w
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            learner.train(
                ['coxph'], verbose=False,
                model_params={'coxph': {}, 'typo_model': {'n_estimators': 50}},
            )

        warned = [str(c.message) for c in caught if "typo_model" in str(c.message)]
        assert len(warned) > 0

    def test_n_trials_dict_invalid_value_raises(self, small_data):
        """n_trials dict with invalid per-model value raises ValueError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with pytest.raises(ValueError, match="n_trials.*must be a non-negative integer"):
            learner.tune('coxph', n_trials={'coxph': -5})

    def test_n_importance_repeats_dict_invalid_value_raises(self, small_data):
        """n_importance_repeats dict with invalid value raises."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)
        learner.train('coxph', verbose=False)

        with pytest.raises(ValueError, match="n_importance_repeats.*must be"):
            learner.evaluate(
                n_importance_repeats={'coxph': 'garbage'},
                verbose=False, bootstrap=False,
            )


@pytest.mark.integration_heavy
class TestNTrialsDictSingleModel:
    """Test n_trials dict with single-model entry points."""

    def test_train_single_model_n_trials_dict(self, small_data):
        """train() single model with n_trials dict resolves correctly."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.train('coxph', tune=True, verbose=False, n_trials={'coxph': 2})
        assert result is not None

    def test_tune_single_model_n_trials_dict(self, small_data):
        """tune() single model with n_trials dict resolves correctly."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        result = learner.tune('coxph', n_trials={'coxph': 2})
        assert result is not None


@pytest.mark.integration
class TestValidationExtended:
    """Extended validation tests for review findings."""

    def test_n_trials_bool_scalar_raises(self, small_data):
        """n_trials=True should raise TypeError (bool is subclass of int)."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with pytest.raises(TypeError, match="n_trials.*bool"):
            learner.tune('coxph', n_trials=True)

    def test_n_trials_dict_bool_value_raises(self, small_data):
        """n_trials={'coxph': False} should raise TypeError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with pytest.raises(TypeError, match="n_trials.*bool"):
            learner.tune('coxph', n_trials={'coxph': False})

    def test_n_importance_repeats_dict_bool_value_raises(self, small_data):
        """n_importance_repeats={'coxph': True} should raise TypeError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)
        learner.train('coxph', verbose=False)

        with pytest.raises(TypeError, match="n_importance_repeats.*bool"):
            learner.evaluate(
                n_importance_repeats={'coxph': True},
                verbose=False, bootstrap=False,
            )

    def test_run_kwargs_no_models_warns(self, small_data):
        """run(tune=False, n_estimators=50) without models should warn."""
        import warnings as w
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            learner.run(
                tune=False, verbose=False, evaluate=False,
                n_estimators=50,
            )

        warned = [str(c.message) for c in caught if "models is not specified" in str(c.message)]
        assert len(warned) > 0, "Expected warning about kwargs with no models"

    def test_train_single_model_unused_model_params_warns(self, small_data):
        """train('rsf', model_params={'coxph': ...}) should warn about unused key."""
        import warnings as w
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            learner.train(
                'rsf', verbose=False,
                model_params={'rsf': {'n_estimators': 50}, 'coxph': {'alpha': 0.1}},
            )

        warned = [str(c.message) for c in caught if "coxph" in str(c.message)]
        assert len(warned) > 0, "Expected warning about unused 'coxph' key"

    def test_model_params_top_level_not_dict_raises(self, small_data):
        """model_params='bad' should raise TypeError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with pytest.raises(TypeError, match="model_params must be a dict"):
            learner.train('rsf', model_params="bad")

    def test_model_params_non_string_key_raises(self, small_data):
        """model_params with non-string keys should raise TypeError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with pytest.raises(TypeError, match="model_params keys must be strings"):
            learner.train('rsf', model_params={42: {'n_estimators': 50}})

    def test_model_params_unused_key_warns(self, small_data):
        """model_params with typo key should warn."""
        import warnings as w
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            learner.train(
                ['coxph'], verbose=False,
                model_params={'coxph': {}, 'rfs_typo': {'n_estimators': 50}},
            )

        warned = [str(c.message) for c in caught if "rfs_typo" in str(c.message)]
        assert len(warned) > 0, "Expected warning about unused 'rfs_typo' key"

    def test_custom_model_kwargs_warns(self, small_data):
        """train(custom_model, n_estimators=50) warns about ignored kwargs."""
        import warnings as w
        from mlsurv import SurvivalLearner
        from mlsurv.models.rsf import RSFModel

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)
        custom = RSFModel(n_estimators=10, random_state=42)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            learner.train(custom, verbose=False, n_estimators=50)

        warned = [str(c.message) for c in caught if "ignored when training" in str(c.message)]
        assert len(warned) > 0, "Expected warning about kwargs with custom model"

    def test_n_importance_repeats_unused_key_warns(self, small_data):
        """n_importance_repeats dict with typo key should warn."""
        import warnings as w
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)
        learner.train('coxph', verbose=False)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            learner.evaluate(
                n_importance_repeats={'coxph': 3, 'typo_model': 5},
                verbose=False, bootstrap=False,
            )

        warned = [str(c.message) for c in caught if "typo_model" in str(c.message)]
        assert len(warned) > 0, "Expected warning about unused 'typo_model' key"

    def test_tune_single_model_n_trials_dict_unused_key_warns(self, small_data):
        """tune('rsf', n_trials={'rsf': 2, 'typo': 10}) warns about 'typo'."""
        import warnings as w
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            learner.tune('coxph', n_trials={'coxph': 2, 'typo': 10}, verbose=False)

        warned = [str(c.message) for c in caught if "typo" in str(c.message)]
        assert len(warned) > 0, "Expected warning about unused 'typo' key in n_trials"

    def test_train_single_model_n_trials_dict_unused_key_warns(self, small_data):
        """train('rsf', tune=True, n_trials={'rsf': 2, 'typo': 10}) warns."""
        import warnings as w
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            learner.train('coxph', tune=True, verbose=False,
                          n_trials={'coxph': 2, 'typo': 10})

        warned = [str(c.message) for c in caught if "typo" in str(c.message)]
        assert len(warned) > 0, "Expected warning about unused 'typo' key in n_trials"
