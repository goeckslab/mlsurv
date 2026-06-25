"""Test hyperparameter tuning for Phase 3.

NOTE: search-space / auto_* coverage (DEFAULT_SPACES membership, auto_n_trials,
auto_cv_folds, auto_deepsurv_defaults) lives in ``tests/test_search_spaces.py``,
which is its canonical home. The duplicated ``TestSearchSpaces`` class that used
to open this file was removed; the one genuinely-unique case it
held (``get_model`` dynamic-default override) was relocated to
``TestGetModelDynamicDefaults`` below.
"""

import warnings

import pytest
import pandas as pd

from mlsurv._warnings import SuggestionWarning

# Tiny forest for tuning tests that only assert tuning *plumbing* (keys, names,
# is_tuned, notifications) rather than tuned model quality. Without an override
# these tune RSF/GBSA over the full default space (n_estimators 100-300). Partial
# spaces merge per-param with the defaults, so unlisted params keep their default
# ranges. The keys are generic forest params, so the same dict works for GBSA.
_SMALL_RSF_SPACE = {'n_estimators': (10, 20), 'max_depth': (3, 5)}
_SMALL_RSF_PARAMS = {'n_estimators': 20}


class TestGetModelDynamicDefaults:
    """get_model() applies data-adaptive defaults but lets user kwargs win."""

    def test_auto_deepsurv_defaults_user_override(self):
        """Verify user-provided kwargs override computed defaults."""
        from mlsurv.models import get_model

        # Get model with dynamic defaults
        model = get_model(
            'deepsurv',
            n_samples=686, n_features=7, n_events=299,
            # User override
            dropout=0.1, layers=[32, 32]
        )

        # User values should override defaults
        assert model.dropout == 0.1
        assert model.layers == [32, 32]
        # Other params should use dynamic defaults
        assert model.activation == 'selu'


class TestBaseTuner:
    """Test BaseTuner class."""

    def test_suggest_param_int_range(self, synthetic_data_clean):
        """Verify integer range suggestion."""
        from mlsurv.tuning import BaseTuner

        X, y = synthetic_data_clean

        class MockTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}

            def objective(self, trial):
                return 0.5

        tuner = MockTuner(X, y)

        # Check that search_space is set
        assert 'n_estimators' in tuner.search_space
        assert tuner.search_space['n_estimators'] == (100, 500)

    def test_suggest_param_log_range(self, synthetic_data_clean):
        """Verify log-scale suggestion."""
        from mlsurv.tuning import BaseTuner

        X, y = synthetic_data_clean

        class MockTuner(BaseTuner):
            DEFAULT_SPACE = {'alpha': (1e-5, 1.0, 'log')}

            def objective(self, trial):
                return 0.5

        tuner = MockTuner(X, y)

        assert tuner.search_space['alpha'] == (1e-5, 1.0, 'log')

    def test_suggest_param_categorical(self, synthetic_data_clean):
        """Verify categorical suggestion."""
        from mlsurv.tuning import BaseTuner

        X, y = synthetic_data_clean

        class MockTuner(BaseTuner):
            DEFAULT_SPACE = {'max_features': ['sqrt', 'log2', 0.5]}

            def objective(self, trial):
                return 0.5

        tuner = MockTuner(X, y)

        assert tuner.search_space['max_features'] == ['sqrt', 'log2', 0.5]

    def test_suggest_value_non_primitive_categorical(self, synthetic_data_clean):
        """Verify non-primitive categorical values are encoded and decoded safely."""
        from optuna.trial import FixedTrial
        from mlsurv.tuning import BaseTuner

        X, y = synthetic_data_clean

        class MockTuner(BaseTuner):
            DEFAULT_SPACE = {}

            def objective(self, trial):
                return 0.5

        tuner = MockTuner(X, y)
        trial = FixedTrial({'layers': 1})

        value = tuner._suggest_value(trial, 'layers', [[16], [32, 32]])
        assert value == [32, 32]

        decoded = tuner._decode_best_params({'layers': 1})
        assert decoded['layers'] == [32, 32]


class TestTunerRegistry:
    """Test tuner registry."""

    def test_get_tuner_rsf(self, synthetic_data_clean):
        """Verify RSFTuner can be retrieved."""
        from mlsurv.tuning import get_tuner, RSFTuner

        X, y = synthetic_data_clean
        tuner = get_tuner('rsf', X=X, y=y)

        assert isinstance(tuner, RSFTuner)

    def test_get_tuner_coxnet(self, synthetic_data_clean):
        """Verify CoxNetTuner can be retrieved."""
        from mlsurv.tuning import get_tuner, CoxNetTuner

        X, y = synthetic_data_clean
        tuner = get_tuner('coxnet', X=X, y=y)

        assert isinstance(tuner, CoxNetTuner)

    def test_get_tuner_coxph(self, synthetic_data_clean):
        """Verify CoxPHTuner can be retrieved for preprocessing tuning."""
        from mlsurv.tuning import get_tuner, CoxPHTuner

        X, y = synthetic_data_clean
        tuner = get_tuner('coxph', X=X, y=y)

        assert isinstance(tuner, CoxPHTuner)

    def test_get_tuner_unknown_raises(self, synthetic_data_clean):
        """Verify unknown model raises."""
        from mlsurv.tuning import get_tuner

        X, y = synthetic_data_clean

        with pytest.raises(ValueError, match="No tuner"):
            get_tuner('unknown_model', X=X, y=y)


class TestRSFTuner:
    """Test RSFTuner class."""

    def test_rsf_tuner_objective(self, small_data):
        """Verify RSFTuner objective runs without error."""
        from mlsurv.tuning import RSFTuner
        import optuna

        X, y = small_data

        # Override search space with fast settings for testing
        fast_space = {
            'n_estimators': (10, 20),
            'max_depth': (3, 5),
            'min_samples_split': (2, 5),
            'min_samples_leaf': (1, 3),
            'max_features': ['sqrt'],
            'bootstrap': [True],
        }

        tuner = RSFTuner(X, y, cv_folds=3, random_state=42, search_space=fast_space)

        # Create a study and run the objective directly
        study = optuna.create_study(direction='maximize')
        study.optimize(tuner.objective, n_trials=1, show_progress_bar=False)
        assert len(study.trials) == 1


class TestCoxNetTuner:
    """Test CoxNetTuner class."""

    def test_coxnet_tuner_default_space(self, synthetic_data_clean):
        """Verify CoxNetTuner has correct default space."""
        from mlsurv.tuning import CoxNetTuner

        X, y = synthetic_data_clean
        tuner = CoxNetTuner(X, y)

        # CoxNet tunes l1_ratio and alpha_index (path position)
        assert 'l1_ratio' in tuner.search_space
        assert 'alpha_index' in tuner.search_space
        # These should NOT be in the default search space:
        assert 'alpha_min_ratio' not in tuner.search_space
        assert 'n_alphas' not in tuner.search_space
        assert 'max_iter' not in tuner.search_space
        assert 'tol' not in tuner.search_space


class TestCustomSearchSpace:
    """Test custom search space handling."""

    def test_custom_space_merge(self, synthetic_data_clean):
        """Verify custom search space merges with defaults."""
        from mlsurv.tuning import RSFTuner

        X, y = synthetic_data_clean

        custom = {'n_estimators': (50, 100)}
        tuner = RSFTuner(X, y, search_space=custom)

        # Custom value should override
        assert tuner.search_space['n_estimators'] == (50, 100)
        # Other defaults should remain
        assert 'max_depth' in tuner.search_space


class TestTunedParametersApplied:
    """Test that all tuned parameters are correctly applied to final models.

    This tests the fix for the bug where tuned parameters (especially 'k' for
    feature selection) were being discarded instead of applied to the final model.
    """

    def test_all_search_space_params_returned(self, small_data):
        """Verify all search space parameters appear in best_params (prefixed or unprefixed)."""
        from mlsurv.tuning import RSFTuner

        X, y = small_data

        # Use fast settings
        fast_space = {
            'n_estimators': (10, 20),
            'max_depth': (3, 5),
            'min_samples_split': (2, 5),
            'min_samples_leaf': (1, 3),
            'max_features': ['sqrt'],
            'bootstrap': [True],
        }

        tuner = RSFTuner(X, y, cv_folds=2, random_state=42, search_space=fast_space)
        best_params, study, _ = tuner.tune(n_trials=2)

        # All parameters in search space should be in best_params (prefixed or unprefixed)
        for param_name in fast_space.keys():
            # Check for param with or without model__ prefix
            has_param = (
                param_name in best_params or
                f'model__{param_name}' in best_params
            )
            assert has_param, f"Parameter '{param_name}' missing from best_params"

    def test_k_applied_to_feature_selector_run(self, small_data):
        """Verify tuned k is applied to feature selector via run()."""
        from mlsurv import SurvivalLearner

        X, y = small_data

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42,
                      feature_selector='topk')

        # Per-model search space: selector k via the feature_selector component
        fast_space = {
            'rsf': {
                'model': {
                    'n_estimators': (10, 20),
                    'max_depth': (3, 5),
                    'min_samples_split': (2, 5),
                    'min_samples_leaf': (1, 3),
                    'max_features': ['sqrt'],
                    'bootstrap': [True],
                },
                'feature_selector': {'k': (3, 4)},  # Small k range for testing
            }
        }

        comparison = learner.run(
            models=['rsf'],
            tune=True,
            n_trials=2,
            search_spaces=fast_space,
            verbose=False
        )

        # run() returns dict of {str: ModelResult}
        assert isinstance(comparison, dict)
        assert 'rsf' in comparison

        rsf_result = comparison['rsf']
        pipeline = rsf_result.model

        # Verify pipeline has selector step
        assert 'selector' in pipeline.named_steps, "Feature selector not in pipeline"

        selector = pipeline.named_steps['selector']
        # k should be within the tuned range
        assert 3 <= selector.k <= 4, f"Selector k={selector.k} not in range [3, 4]"

    def test_k_applied_to_feature_selector_tune(self, small_data):
        """Verify tuned k is applied to feature selector via tune()."""
        from mlsurv import SurvivalLearner

        X, y = small_data

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42,
                      feature_selector='topk')

        fast_space = {
            'model': {
                'n_estimators': (10, 20),
                'max_depth': (3, 5),
                'min_samples_split': (2, 5),
                'min_samples_leaf': (1, 3),
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'feature_selector': {'k': (3, 4)},
        }

        result = learner.tune('rsf', n_trials=2, search_spaces=fast_space)

        pipeline = result.model

        # Verify pipeline has selector step
        assert 'selector' in pipeline.named_steps, "Feature selector not in pipeline"

        selector = pipeline.named_steps['selector']
        assert 3 <= selector.k <= 4, f"Selector k={selector.k} not in range [3, 4]"

        # Verify best_params includes the selector k under its component prefix
        assert result.best_params_ is not None
        assert 'feature_selector__k' in result.best_params_, (
            f"'feature_selector__k' not found in best_params: {result.best_params_}")

    def test_model_params_applied_correctly(self, small_data):
        """Verify model-specific params are passed to the model."""
        from mlsurv import SurvivalLearner

        X, y = small_data

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        # Use specific values we can verify
        fast_space = {
            'n_estimators': [15],  # Fixed value
            'max_depth': [4],      # Fixed value
            'min_samples_split': (2, 3),
            'min_samples_leaf': (1, 2),
            'max_features': ['sqrt'],
            'bootstrap': [True],
        }

        result = learner.tune('rsf', n_trials=1, search_space=fast_space)

        pipeline = result.model
        model = pipeline.named_steps['model']

        # Verify fixed params were applied to the model
        assert model.model_.n_estimators == 15, f"n_estimators={model.model_.n_estimators}, expected 15"
        assert model.model_.max_depth == 4, f"max_depth={model.model_.max_depth}, expected 4"

    @pytest.mark.parametrize("model_name", ['rsf', 'gbsa'])
    def test_all_tunable_models_apply_k(self, small_data, model_name):
        """Verify selector k is applied via feature_selector__k."""
        from mlsurv import SurvivalLearner

        X, y = small_data

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42,
                      feature_selector='topk')

        # Selector k is tuned only via the explicit feature_selector component.
        model_space = {}

        # Add minimal model-specific params for fast execution
        if model_name == 'rsf':
            model_space = {
                'n_estimators': [10],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            }
        elif model_name == 'gbsa':
            model_space = {
                'loss': ['coxph'],
                'learning_rate': [0.1],
                'n_estimators': [10],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'subsample': [1.0],
                'dropout_rate': [0.0],
            }

        fast_space = {'model': model_space, 'feature_selector': {'k': [4]}}

        result = learner.tune(model_name, n_trials=1, search_spaces=fast_space)

        pipeline = result.model
        assert 'selector' in pipeline.named_steps, f"No selector for {model_name}"
        assert pipeline.named_steps['selector'].k == 4, f"k not applied for {model_name}"

    @pytest.mark.integration
    def test_fssvm_applies_k(self, synthetic_data_clean):
        """Verify selector k is applied for FSSVM (needs larger dataset)."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42,
                      feature_selector='topk')

        # FSSVM needs more stable settings on small data
        fast_space = {
            'model': {
                'alpha': [1.0],
                'max_iter': [500],
                'tol': [1e-3],
                'fit_intercept': [False],
                'rank_ratio': [1.0],
                'optimizer': ['avltree'],  # Cross-platform (PRSVM/simple segfault on Windows)
            },
            'feature_selector': {'k': [5]},
        }

        result = learner.tune('fssvm', n_trials=1, search_spaces=fast_space)

        pipeline = result.model
        assert 'selector' in pipeline.named_steps, "No selector for fssvm"
        assert pipeline.named_steps['selector'].k == 5, "k not applied for fssvm"


class TestFSSVMTunerSimplified:
    """Test simplified FSSVM tuner (optimizer/fit_intercept not tuned)."""

    def test_search_space_excludes_optimizer(self):
        """Verify optimizer is not in the FSSVM search space."""
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES

        assert 'optimizer' not in DEFAULT_SPACES['fssvm']

    def test_search_space_excludes_fit_intercept(self):
        """Verify fit_intercept is not in the FSSVM search space."""
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES

        assert 'fit_intercept' not in DEFAULT_SPACES['fssvm']


class TestComponentTuning:
    """Test customizable pipeline components with tunable parameters."""

    def test_custom_imputer_used(self, small_data):
        """Custom imputer from setup() is used in pipeline."""
        from mlsurv import SurvivalLearner
        from sklearn.impute import KNNImputer

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            imputer=KNNImputer(n_neighbors=3),
        )

        result = learner.train('coxph')
        pipeline = result.model

        # Verify KNNImputer is in the pipeline
        assert 'imputer' in pipeline.named_steps
        assert isinstance(pipeline.named_steps['imputer'], KNNImputer)

    def test_custom_scaler_used(self, small_data):
        """Custom scaler from setup() is used in pipeline."""
        from mlsurv import SurvivalLearner
        from sklearn.preprocessing import RobustScaler

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            scaler=RobustScaler(),
        )

        result = learner.train('coxph')
        pipeline = result.model

        # Verify RobustScaler is in the pipeline
        assert 'scaler' in pipeline.named_steps
        assert isinstance(pipeline.named_steps['scaler'], RobustScaler)

    def test_custom_selector_used(self, small_data):
        """Custom selector from setup() is used in pipeline."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            feature_selector=TopKCoxSelector(k=3),
        )

        result = learner.train('coxph')
        pipeline = result.model

        # Verify TopKCoxSelector is in the pipeline
        assert 'selector' in pipeline.named_steps
        assert isinstance(pipeline.named_steps['selector'], TopKCoxSelector)
        assert pipeline.named_steps['selector'].k == 3

    def test_skip_scaler(self, small_data):
        """Pipeline has no scaler when scaler=None in setup()."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            scaler=None,
        )

        result = learner.train('coxph')
        pipeline = result.model

        # Verify scaler is NOT in the pipeline
        assert 'scaler' not in pipeline.named_steps
        # Imputer should still be there
        assert 'imputer' in pipeline.named_steps

    def test_skip_imputer(self, small_data):
        """Pipeline has no imputer when imputer=None in setup()."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            imputer=None,
        )

        result = learner.train('coxph')
        pipeline = result.model

        # Verify imputer is NOT in the pipeline
        assert 'imputer' not in pipeline.named_steps
        # Scaler should still be there
        assert 'scaler' in pipeline.named_steps

    def test_skip_all_preprocessing(self, synthetic_data_clean):
        """Pipeline has only model when imputer=None, scaler=None, selector=None."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            imputer=None,
            scaler=None,
            feature_selector=None,
        )

        result = learner.train('coxph')
        pipeline = result.model

        # Pipeline should only have model
        assert list(pipeline.named_steps.keys()) == ['model']

    def test_defaults_used_when_not_specified(self, small_data):
        """SimpleImputer and StandardScaler used by default."""
        from mlsurv import SurvivalLearner
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        result = learner.train('coxph')
        pipeline = result.model

        # Verify defaults
        assert 'imputer' in pipeline.named_steps
        assert isinstance(pipeline.named_steps['imputer'], SimpleImputer)
        assert 'scaler' in pipeline.named_steps
        assert isinstance(pipeline.named_steps['scaler'], StandardScaler)

    def test_selector_params_tuned_with_prefix(self, small_data):
        """feature_selector__k in best_params when search_spaces used."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            feature_selector=TopKCoxSelector(k=5),
        )

        result = learner.tune('rsf', n_trials=2, search_spaces={
            'model': {'n_estimators': [15], 'max_depth': [3], 'min_samples_split': [2],
                      'min_samples_leaf': [1], 'max_features': ['sqrt'], 'bootstrap': [True]},
            'feature_selector': {'k': (3, 5)},
        })

        # Verify prefixed param in best_params
        assert result.best_params_ is not None
        assert any('feature_selector__k' in k for k in result.best_params_.keys())

    def test_model_params_tuned_with_prefix(self, small_data):
        """model__n_estimators in best_params when search_spaces used."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            feature_selector=TopKCoxSelector(k=5),
        )

        result = learner.tune('rsf', n_trials=2, search_spaces={
            'model': {'n_estimators': (10, 20), 'max_depth': [3], 'min_samples_split': [2],
                      'min_samples_leaf': [1], 'max_features': ['sqrt'], 'bootstrap': [True]},
        })

        # Verify prefixed param in best_params
        assert result.best_params_ is not None
        assert any('model__n_estimators' in k for k in result.best_params_.keys())

    def test_flat_search_space_is_model_space(self, small_data):
        """A flat (component-less) search_spaces dict is treated as the model space."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        # Flat dict with no component keys -> all keys are model params.
        result = learner.tune('rsf', n_trials=2, search_spaces={
            'n_estimators': [15],
            'max_depth': [3],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt'],
            'bootstrap': [True],
        })

        # Should work without errors
        assert result is not None
        assert result.best_params_ is not None

    def test_no_selector_in_pipeline_when_none(self, small_data):
        """Pipeline has no selector when feature_selector=None in setup()."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            feature_selector=None,
        )

        result = learner.train('coxph')
        pipeline = result.model

        # Verify selector is NOT in the pipeline
        assert 'selector' not in pipeline.named_steps


class TestNewModelsTuning:
    """Tests for tuning newly added models."""

    def test_tune_xgbcox(self, small_data):
        """Test XGBoost-Cox tuning."""
        pytest.importorskip("xgboost")
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        result = learner.tune('xgbcox', n_trials=3, search_space={
            'n_estimators': (10, 50),
            'max_depth': (3, 6),
        })

        assert result is not None
        assert result.best_params_ is not None
        assert result.name == 'xgbcox'

    def test_tune_deephit(self, small_data):
        """Test DeepHit tuning."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        result = learner.tune('deephit', n_trials=2, search_space={
            'layers': [[16]],  # Use layers directly instead of n_layers/n_nodes
            'epochs': [5],
        })

        assert result is not None
        assert result.best_params_ is not None
        assert result.name == 'deephit'

    def test_tune_weibull_aft(self, small_data):
        """Test Weibull AFT tuning."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        result = learner.tune('weibull_aft', n_trials=3, search_space={
            'alpha': (0.001, 1.0, 'log'),
        })

        assert result is not None
        assert result.best_params_ is not None
        assert result.name == 'weibull_aft'


class TestSkipEmptyTuning:
    """Test that tuning is skipped when model has no searchable params."""

    def test_tune_coxph_no_preprocessing_skips(self, small_data, capsys):
        """tune('coxph') with no preprocessing search space skips tuning."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        result = learner.tune('coxph', n_trials=10, verbose=False)

        captured = capsys.readouterr()
        assert "no hyperparameters to search" in captured.out

        # Returns a valid result trained with defaults, NOT marked as tuned
        assert result is not None
        assert result.name == 'coxph'
        assert not result.is_tuned

    def test_tune_coxph_unsupported_model_space_warns_and_skips(self, small_data, capsys):
        """A model key CoxPH cannot apply (not alpha) warns and is not tuned.

        ``alpha`` is now a tunable CoxPH model param, so the guard fires only
        for genuinely unsupported keys (here a ``penalizer`` typo).
        """
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = learner.tune(
                'coxph', n_trials=5, verbose=False,
                search_spaces={'model': {'penalizer': (0.0, 1.0)}},
            )

        suggestion = [w for w in caught
                      if issubclass(w.category, SuggestionWarning)
                      and 'penalizer' in str(w.message)]
        assert len(suggestion) == 1

        captured = capsys.readouterr()
        assert "no hyperparameters to search" in captured.out
        # Trained with defaults, NOT falsely reported as tuned
        assert result is not None
        assert result.name == 'coxph'
        assert not result.is_tuned

    def test_tune_coxph_with_selector_space_still_tunes(self, small_data):
        """coxph with selector search space should still run tuning."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3, cv_folds=2, random_state=42,
            feature_selector=TopKCoxSelector(k=5),
        )

        result = learner.tune('coxph', n_trials=3, verbose=False, search_spaces={
            'feature_selector': {'k': (3, 5)},
        })

        assert result is not None
        assert result.name == 'coxph'
        assert result.is_tuned
        assert result.best_params_ is not None

    def test_tune_coxph_after_train_preserves_original(self, small_data, capsys):
        """tune('coxph') after train() returns baseline, no '_tuned' entry."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        learner.train('coxph')
        original = learner.models['coxph']

        result = learner.tune('coxph', n_trials=10, verbose=False)

        captured = capsys.readouterr()
        assert "no hyperparameters to search" in captured.out

        # Original trained model preserved, no _tuned variant created
        assert 'coxph' in learner.models
        assert 'coxph_tuned' not in learner.models
        # Returned result IS the existing baseline
        assert result is original
        assert not result.is_tuned

    def test_run_coxph_with_tune_skips_with_note(self, small_data, capsys):
        """run() with tune=True prints skip note for coxph tuning."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        comparison = learner.run(
            models=['coxph'], tune=True, n_trials=10, verbose=False,
        )

        captured = capsys.readouterr()
        assert "no hyperparameters to search" in captured.out

        # run() returns dict of {str: ModelResult}
        assert isinstance(comparison, dict)
        assert 'coxph' in comparison
        result = comparison['coxph']
        assert not result.is_tuned

    def test_tune_all_excludes_non_tunable(self, small_data, capsys):
        """tune() return dict excludes models with no tunable params."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)
        learner.run(
            models=['coxph', 'rsf'], tune=False,
            evaluate=False, bootstrap=False, verbose=False,
            model_params={'rsf': _SMALL_RSF_PARAMS},
        )

        # Note: a standalone multi-model tune() applies one search_spaces to
        # every model, so we can't narrow only RSF here without also handing
        # CoxPH a forest space; just trim n_trials. The untuned RSF forest is
        # already shrunk via model_params above.
        tuned = learner.tune(n_trials=2, verbose=False)

        captured = capsys.readouterr()
        # CoxPH has no tunable params — excluded from tuned dict
        assert 'coxph' not in tuned
        # RSF was tuned — dict is keyed by the plain model name
        assert 'rsf' in tuned
        assert tuned['rsf'].is_tuned
        # No _tuned storage suffix exists anymore
        assert 'rsf_tuned' not in learner.models
        # Print message was emitted
        assert "no hyperparameters to search" in captured.out

    def test_tune_all_non_tunable_returns_empty(self, small_data, capsys):
        """tune() returns empty dict when all models are non-tunable."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)
        learner.run(
            models=['coxph'], tune=False,
            evaluate=False, bootstrap=False, verbose=False,
        )

        tuned = learner.tune(verbose=False)

        captured = capsys.readouterr()
        # Empty dict (no crash), print message was emitted
        assert tuned == {}
        assert 'coxph_tuned' not in learner.models
        assert "no hyperparameters to search" in captured.out


class TestTuneOverwritesBaseline:
    """tune() overwrites the baseline at the plain model name."""

    def test_tune_overwrites_baseline_in_place(self, small_data, capsys):
        """tune('rsf') after train('rsf') overwrites 'rsf' — no _tuned key, no note."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        learner.train('rsf', n_estimators=20)
        assert not learner.models['rsf'].is_tuned
        learner.tune('rsf', n_trials=2, verbose=True,
                     search_spaces=_SMALL_RSF_SPACE)

        captured = capsys.readouterr().out
        assert "Note: Tuned model stored as" not in captured
        assert learner.models['rsf'].is_tuned
        assert 'rsf_tuned' not in learner.models

    def test_no_notification_when_tune_without_prior_train(self, small_data, capsys):
        """tune() without prior train() stores under the plain key — no notification."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        learner.tune('rsf', n_trials=2, verbose=True,
                     search_spaces=_SMALL_RSF_SPACE)

        captured = capsys.readouterr().out
        assert "Note: Tuned model stored as" not in captured
        assert learner.models['rsf'].is_tuned

    def test_multi_model_returns_plain_keys(self, small_data, capsys):
        """tune() with multiple pre-trained models returns a dict keyed by the
        plain model names, matching learner.models."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        learner.train(
            ['rsf', 'gbsa'],
            model_params={'rsf': _SMALL_RSF_PARAMS, 'gbsa': _SMALL_RSF_PARAMS},
        )
        # Both models are forests, so a single component-keyed model space
        # ({'model': ...}) narrows n_estimators/max_depth for each of them.
        tuned = learner.tune(
            verbose=True, n_trials=2,
            search_spaces={'model': _SMALL_RSF_SPACE},
        )

        assert set(tuned) == {'rsf', 'gbsa'}
        for key in ('rsf', 'gbsa'):
            assert tuned[key].is_tuned
            assert tuned[key].name == key
            assert key in learner.models
            # Returned reference is the same object stored in _state.models
            assert tuned[key] is learner._state.models[key]
            assert f'{key}_tuned' not in learner.models


class TestSamplerSeeding:
    """Test that Optuna TPESampler is seeded correctly for reproducibility."""

    # Minimal search space for fast trials
    FAST_SPACE = {
        'n_estimators': (10, 20),
        'max_depth': (3, 5),
        'min_samples_split': (2, 5),
        'min_samples_leaf': (1, 3),
        'max_features': ['sqrt'],
        'bootstrap': [True],
    }

    @staticmethod
    def _trial_params(study):
        """Extract list of trial param dicts from a study."""
        return [t.params for t in study.trials]

    def test_reproducible_with_seed(self, small_data, tmp_path):
        """Two runs with same random_state produce identical trial params."""
        from mlsurv.tuning import RSFTuner

        X, y = small_data

        results = []
        for i in range(2):
            tuner = RSFTuner(
                X, y, cv_folds=2, random_state=42, search_space=self.FAST_SPACE,
            )
            _, study, _ = tuner.tune(
                n_trials=5,
                experiment_name=tmp_path / f"repro_run_{i}",
                optuna_verbose=False,
            )
            results.append(self._trial_params(study))

        assert results[0] == results[1], (
            "Seeded local runs should produce identical trial params"
        )

    def test_unseeded_with_user_storage(self, small_data, tmp_path):
        """Explicit storage disables sampler seeding (params likely differ)."""
        from mlsurv.tuning import RSFTuner

        X, y = small_data

        results = []
        for i in range(2):
            db_path = tmp_path / f"user_storage_{i}.db"
            tuner = RSFTuner(
                X, y, cv_folds=2, random_state=42, search_space=self.FAST_SPACE,
            )
            _, study, _ = tuner.tune(
                n_trials=5,
                storage=f"sqlite:///{db_path}",
                study_name=f"test_study_{i}",
                optuna_verbose=False,
            )
            results.append(self._trial_params(study))

        assert results[0] != results[1], (
            "User-provided storage should leave sampler unseeded (params differ)"
        )

    def test_unseeded_when_no_random_state(self, small_data, tmp_path):
        """Without random_state, two runs produce different trial params."""
        from mlsurv.tuning import RSFTuner

        X, y = small_data

        results = []
        for i in range(2):
            tuner = RSFTuner(
                X, y, cv_folds=2, random_state=None, search_space=self.FAST_SPACE,
            )
            _, study, _ = tuner.tune(
                n_trials=5,
                experiment_name=tmp_path / f"noseed_run_{i}",
                optuna_verbose=False,
            )
            results.append(self._trial_params(study))

        assert results[0] != results[1], (
            "Unseeded runs should produce different trial params"
        )


class TestWarnEpvIfLow:
    """Test _warn_epv_if_low extracted helper."""

    @staticmethod
    def _make_learner(n_events, n_features):
        """Create a mock learner with controlled EPV."""
        from unittest.mock import MagicMock
        learner = MagicMock()
        learner.y_train = {'event': pd.Series([1] * n_events + [0] * 10)}
        learner.X_train = MagicMock()
        learner.X_train.columns = list(range(n_features))
        learner._state.epv_warned_at_setup = False
        return learner

    def test_very_low_epv_warns(self):
        """EPV < 2 should issue a UserWarning."""
        import warnings
        from mlsurv.learner._tuning import TuningMixin

        learner = self._make_learner(n_events=3, n_features=5)  # EPV=0.6
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TuningMixin._warn_epv_if_low(learner, stacklevel=2)
        assert len(w) == 1
        assert "EPV=0.6" in str(w[0].message)

    def test_very_low_epv_warns_with_default_stacklevel(self):
        """Default stacklevel should attribute the warning to the call site."""
        import warnings
        from mlsurv.learner._tuning import TuningMixin

        learner = self._make_learner(n_events=3, n_features=5)  # EPV=0.6
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TuningMixin._warn_epv_if_low(learner)
        assert len(w) == 1
        assert "EPV=0.6" in str(w[0].message)
        assert w[0].filename == __file__

    def test_low_epv_logs_info(self):
        """2 <= EPV < 5 should log info, not warn."""
        import warnings
        import logging
        from mlsurv.learner._tuning import TuningMixin

        learner = self._make_learner(n_events=12, n_features=4)  # EPV=3.0
        logger = logging.getLogger("mlsurv.learner")
        records = []

        class Collector(logging.Handler):
            def emit(self, record):
                records.append(record)

        collector = Collector()
        logger.addHandler(collector)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                TuningMixin._warn_epv_if_low(learner, stacklevel=2)
            assert len(w) == 0
            assert any("EPV=3.0" in r.getMessage() for r in records)
        finally:
            logger.removeHandler(collector)
            logger.setLevel(old_level)

    def test_normal_epv_silent(self):
        """EPV >= 5 should produce no warning."""
        import warnings
        from mlsurv.learner._tuning import TuningMixin

        learner = self._make_learner(n_events=50, n_features=5)  # EPV=10
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TuningMixin._warn_epv_if_low(learner, stacklevel=2)
        assert len(w) == 0

    def test_zero_features_no_crash(self):
        """n_features=0 should not crash (EPV=inf, no warning)."""
        import warnings
        from mlsurv.learner._tuning import TuningMixin

        learner = self._make_learner(n_events=10, n_features=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TuningMixin._warn_epv_if_low(learner, stacklevel=2)
        assert len(w) == 0
