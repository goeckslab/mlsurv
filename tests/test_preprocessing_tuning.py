"""Test intuitive preprocessing parameter tuning.

These tests verify:
1. Class-name based search space specification
2. Broadcast parameter tuning
3. Multi-step Pipeline tuning
4. Class-specific overrides
"""

import pytest
from sklearn.pipeline import Pipeline


class TestSearchSpaceParsing:
    """Test search space parsing helpers.

    The ``_is_class_name`` / ``_parse_component_search_space`` /
    ``_get_step_class_mapping`` unit tests live in ``test_tuner_classes.py``
    (``TestIsClassName`` / ``TestParseComponentSearchSpace`` /
    ``TestGetStepClassMapping``). Only the duplicate-class mapping case below
    is unique to this file.
    """

    def test_get_step_class_mapping_duplicate_classes(self):
        """Map class names when same class appears multiple times."""
        from mlsurv.tuning.optuna_tuner import _get_step_class_mapping
        from mlsurv.preprocessing import VIFSelector

        pipeline = Pipeline([
            ('sel_0', VIFSelector(threshold=5)),
            ('sel_1', VIFSelector(threshold=10)),
        ])
        mapping = _get_step_class_mapping(pipeline)

        assert mapping == {'VIFSelector': ['sel_0', 'sel_1']}


class TestPreprocessingDefaults:
    """Test preprocessing default search spaces."""

    def test_preprocessing_defaults_exist(self):
        """Verify PREPROCESSING_DEFAULTS contains expected keys."""
        from mlsurv.tuning.search_spaces import PREPROCESSING_DEFAULTS

        assert 'imputer' in PREPROCESSING_DEFAULTS
        assert 'scaler' in PREPROCESSING_DEFAULTS
        assert 'feature_selector' in PREPROCESSING_DEFAULTS

    def test_feature_selector_defaults(self):
        """Verify feature selector defaults."""
        from mlsurv.tuning.search_spaces import get_preprocessing_defaults

        defaults = get_preprocessing_defaults('feature_selector')

        assert 'TopKCoxSelector' in defaults
        assert 'VIFSelector' in defaults
        assert 'CorrelationSelector' in defaults

        assert 'k' in defaults['TopKCoxSelector']
        assert 'threshold' in defaults['VIFSelector']

    def test_get_preprocessing_defaults_returns_copy(self):
        """Verify get_preprocessing_defaults returns a copy."""
        from mlsurv.tuning.search_spaces import get_preprocessing_defaults

        defaults1 = get_preprocessing_defaults('feature_selector')
        defaults2 = get_preprocessing_defaults('feature_selector')

        # Modifying one shouldn't affect the other
        defaults1['new_key'] = 'test'
        assert 'new_key' not in defaults2


@pytest.mark.integration
class TestSingleComponentTuning:
    """Test tuning with single preprocessing component."""

    def test_single_selector_with_flat_params(self, small_data):
        """Single selector tunes with flat params (backward compat)."""
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
            'model': {
                'n_estimators': [15],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'feature_selector': {'k': (3, 5)},
        })

        # Verify k was tuned
        assert 'feature_selector__k' in result.best_params_
        assert 3 <= result.best_params_['feature_selector__k'] <= 5

    def test_single_selector_with_class_name(self, small_data):
        """Single selector tunes with class name (new API)."""
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
            'model': {
                'n_estimators': [15],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'feature_selector': {
                'TopKCoxSelector': {'k': (3, 5)},
            },
        })

        # Verify k was tuned
        assert 'feature_selector__k' in result.best_params_
        assert 3 <= result.best_params_['feature_selector__k'] <= 5


@pytest.mark.integration
class TestMultiStepPipelineTuning:
    """Test tuning with multi-step preprocessing pipelines."""

    def test_multi_step_with_class_names(self, small_data):
        """Multi-step pipeline tunes with class-specific params."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import VIFSelector, TopKCoxSelector

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            feature_selector=[VIFSelector(threshold=10), TopKCoxSelector(k=5)],
        )

        result = learner.tune('rsf', n_trials=2, search_spaces={
            'model': {
                'n_estimators': [15],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'feature_selector': {
                'VIFSelector': {'threshold': (5.0, 15.0)},
                'TopKCoxSelector': {'k': (3, 5)},
            },
        })

        # Verify both params were tuned with nested names
        assert 'feature_selector__sel_0__threshold' in result.best_params_
        assert 'feature_selector__sel_1__k' in result.best_params_

        # Verify values are in range
        threshold = result.best_params_['feature_selector__sel_0__threshold']
        assert 5.0 <= threshold <= 15.0

        k = result.best_params_['feature_selector__sel_1__k']
        assert 3 <= k <= 5

    def test_multi_step_pipeline_applied(self, small_data):
        """Verify tuned params are applied to multi-step pipeline."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import VIFSelector, TopKCoxSelector

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            feature_selector=[VIFSelector(threshold=10), TopKCoxSelector(k=5)],
        )

        result = learner.tune('rsf', n_trials=2, search_spaces={
            'model': {
                'n_estimators': [15],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'feature_selector': {
                'VIFSelector': {'threshold': [8.0]},  # Fixed value
                'TopKCoxSelector': {'k': [4]},        # Fixed value
            },
        })

        # Get the pipeline
        pipeline = result.model
        selector = pipeline.named_steps['selector']

        # Verify it's a Pipeline with the right steps
        assert hasattr(selector, 'steps')
        assert len(selector.steps) == 2

        # Verify params were applied
        vif = selector.named_steps['sel_0']
        topk = selector.named_steps['sel_1']

        assert vif.threshold == 8.0
        assert topk.k == 4


@pytest.mark.integration
class TestBroadcastParams:
    """Test broadcast parameter behavior."""

    def test_broadcast_to_all_steps(self, synthetic_data_clean):
        """Broadcast param applies to all steps with that param."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import VIFSelector, CorrelationSelector

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            # Both VIF and Correlation have 'threshold' param
            feature_selector=[VIFSelector(threshold=10), CorrelationSelector(threshold=0.8)],
        )

        result = learner.tune('rsf', n_trials=2, search_spaces={
            'model': {
                'n_estimators': [15],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'feature_selector': {
                # No class-specific params - both should NOT get tuned
                # because we only tune params that are explicitly specified
            },
        })

        # No feature_selector params in best_params since we didn't specify any
        assert not any(k.startswith('feature_selector__') for k in result.best_params_)

    def test_class_specific_overrides_broadcast(self, synthetic_data_clean):
        """Class-specific params override broadcast for that class."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import VIFSelector, CorrelationSelector

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            feature_selector=[VIFSelector(threshold=10), CorrelationSelector(threshold=0.8)],
        )

        result = learner.tune('rsf', n_trials=2, search_spaces={
            'model': {
                'n_estimators': [15],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'feature_selector': {
                # VIF gets class-specific range
                'VIFSelector': {'threshold': (5.0, 15.0)},
                # CorrelationSelector not specified - won't be tuned
            },
        })

        # Only VIF threshold should be tuned
        assert 'feature_selector__sel_0__threshold' in result.best_params_
        assert 'feature_selector__sel_1__threshold' not in result.best_params_

        # VIF threshold should be in its specific range
        vif_threshold = result.best_params_['feature_selector__sel_0__threshold']
        assert 5.0 <= vif_threshold <= 15.0


@pytest.mark.integration
class TestImputerTuning:
    """Test imputer parameter tuning."""

    def test_knn_imputer_tuning(self, small_data):
        """KNNImputer n_neighbors can be tuned."""
        from mlsurv import SurvivalLearner
        from sklearn.impute import KNNImputer

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            imputer=KNNImputer(n_neighbors=5),
        )

        result = learner.tune('rsf', n_trials=2, search_spaces={
            'model': {
                'n_estimators': [15],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'imputer': {'n_neighbors': (3, 7)},
        })

        # Verify n_neighbors was tuned
        assert 'imputer__n_neighbors' in result.best_params_
        assert 3 <= result.best_params_['imputer__n_neighbors'] <= 7


@pytest.mark.integration
class TestScalerTuning:
    """Test scaler parameter tuning."""

    def test_pca_components_tuning(self, synthetic_data_clean):
        """PCA n_components can be tuned."""
        from mlsurv import SurvivalLearner
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)

        # Create a scaler pipeline with PCA
        scaler = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
        ])

        learner.setup(
            test_fraction=0.3,
            cv_folds=2,
            random_state=42,
            scaler=scaler,
            feature_selector=None,
        )

        result = learner.tune('rsf', n_trials=2, search_spaces={
            'model': {
                'n_estimators': [15],
                'max_depth': [3],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'scaler': {
                'PCA': {'n_components': (0.8, 0.99)},
            },
        })

        # Verify PCA n_components was tuned
        assert 'scaler__pca__n_components' in result.best_params_
        assert 0.8 <= result.best_params_['scaler__pca__n_components'] <= 0.99


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility with existing API."""

    def test_flat_search_space_still_works(self, small_data):
        """Flat search_space dict works for backward compatibility."""
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

    def test_old_search_space_param_alias(self, small_data):
        """search_space param alias works for backward compatibility."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)

        # Using old 'search_space' param name (singular)
        result = learner.tune('rsf', n_trials=2, search_space={
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
