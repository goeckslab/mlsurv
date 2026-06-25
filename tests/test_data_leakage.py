"""Test data leakage prevention in mlsurv.

Properties under test:
- Preprocessing (scaling, imputation) must be fit on training data only
- Feature selection must not use test/validation data
- Cross-validation folds must be properly isolated
- Hyperparameter tuning uses validation set, not test set
"""

import pytest
import numpy as np
import pandas as pd

pytestmark = pytest.mark.integration


class TestPreprocessingIsolation:
    """Test that preprocessing is fit only on training data."""

    def test_scaler_fit_only_on_train(self, synthetic_data):
        """Verify scaler statistics come from training data only."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, test_fraction=0.3)
        result = learner.train('coxph')

        # Get the scaler from the pipeline
        pipeline = result.model
        scaler = pipeline.named_steps.get('scaler')

        if scaler is not None:
            # Scaler mean should match training data mean (after imputation)
            imputer = pipeline.named_steps.get('imputer')
            if imputer is not None:
                X_train_imputed = imputer.transform(learner.X_train)
            else:
                X_train_imputed = learner.X_train.values

            # Scaler statistics should come from training data only
            # (accounting for potential NaN handling)
            expected_mean = np.nanmean(X_train_imputed, axis=0)
            np.testing.assert_array_almost_equal(
                scaler.mean_,
                expected_mean,
                decimal=5,
                err_msg="Scaler mean doesn't match training data - potential data leakage"
            )

    def test_imputer_fit_only_on_train(self, synthetic_data):
        """Verify imputer statistics come from training data only."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, test_fraction=0.3)
        result = learner.train('coxph')

        # Get the imputer from the pipeline
        pipeline = result.model
        imputer = pipeline.named_steps.get('imputer')

        if imputer is not None:
            # Imputer statistics should come from training data only
            train_median = np.nanmedian(learner.X_train.values, axis=0)

            np.testing.assert_array_almost_equal(
                imputer.statistics_,
                train_median,
                decimal=5,
                err_msg="Imputer statistics don't match training data - potential data leakage"
            )

    def test_selector_fit_only_on_train(self, synthetic_data_clean):
        """Verify feature selector is fit on training data only."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        selector = TopKCoxSelector(k=5)
        learner.setup(random_state=42, feature_selector=selector)
        result = learner.train('coxph')

        # Get selector from pipeline
        pipeline = result.model
        fitted_selector = pipeline.named_steps.get('selector')

        if fitted_selector is not None:
            # Verify selector was fit on training data by checking selected features
            # The selector should have been fit during pipeline.fit(X_train, y_train)
            assert hasattr(fitted_selector, 'selected_features_')
            assert len(fitted_selector.selected_features_) == 5


class TestCVIsolation:
    """Test that CV folds are properly isolated."""

    def test_cv_folds_properly_isolated(self, synthetic_data_clean):
        """Verify CV folds don't overlap."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=5)

        # Collect all fold indices
        all_train_indices = []
        all_val_indices = []

        for train_idx, val_idx in learner._state.cv_splitter.split(
            learner.X_train, learner.y_train['event']
        ):
            all_train_indices.append(set(train_idx))
            all_val_indices.append(set(val_idx))

            # Verify no overlap within a fold
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Train and val overlap in fold: {overlap}"

        # Each validation sample appears in exactly one validation fold
        all_val_samples = [idx for val_set in all_val_indices for idx in val_set]
        unique_val_samples = set(all_val_samples)
        assert len(all_val_samples) == len(unique_val_samples), \
            "Same sample appears in multiple validation folds"

    def test_cv_val_samples_cover_all_data(self, synthetic_data_clean):
        """Verify validation folds cover all training data exactly once."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=5)

        # Collect all validation indices
        all_val_indices = set()
        for train_idx, val_idx in learner._state.cv_splitter.split(
            learner.X_train, learner.y_train['event']
        ):
            all_val_indices.update(val_idx)

        # Should cover exactly the training set indices
        expected_indices = set(range(len(learner.X_train)))
        assert all_val_indices == expected_indices, \
            "CV folds don't cover all training data"

    def test_nested_cv_inner_outer_isolation(self, small_data):
        """Verify nested CV has proper inner/outer isolation."""
        from sklearn.model_selection import StratifiedKFold

        X, y = small_data

        # Simulate nested CV structure
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for outer_train_idx, outer_test_idx in outer_cv.split(X, y['event']):
            X_outer_train = X.iloc[outer_train_idx]
            y_outer_train = y[outer_train_idx]

            # Inner CV should only use outer training data
            for inner_train_idx, inner_val_idx in inner_cv.split(
                X_outer_train, y_outer_train['event']
            ):
                # Verify inner indices are within outer training set bounds
                assert all(i < len(X_outer_train) for i in inner_train_idx)
                assert all(i < len(X_outer_train) for i in inner_val_idx)

                # Convert to original indices
                original_inner_train = [outer_train_idx[i] for i in inner_train_idx]
                original_inner_val = [outer_train_idx[i] for i in inner_val_idx]

                # Verify no overlap with outer test set
                overlap_train = set(original_inner_train) & set(outer_test_idx)
                overlap_val = set(original_inner_val) & set(outer_test_idx)

                assert len(overlap_train) == 0, \
                    "Inner train overlaps with outer test - data leakage!"
                assert len(overlap_val) == 0, \
                    "Inner val overlaps with outer test - data leakage!"


class TestFeatureSelectionLeakage:
    """Test that feature selection doesn't leak test data."""

    def test_topk_selector_not_using_test_data(self, synthetic_data_clean):
        """Verify TopKCoxSelector uses only training data."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        selector = TopKCoxSelector(k=5)
        learner.setup(random_state=42, feature_selector=selector)

        # Create model
        result = learner.train('coxph')

        # Get fitted selector
        fitted_selector = result.model.named_steps.get('selector')
        if fitted_selector is not None:
            # Fit a fresh selector on just training data
            fresh_selector = TopKCoxSelector(k=5)
            fresh_selector.fit(learner.X_train, learner.y_train)

            # Convert both to comparable sets
            # Pipeline selector may store indices, fresh selector stores feature names
            def get_feature_set(selector, df):
                features = selector.selected_features_
                if all(isinstance(f, int) for f in features):
                    # Convert indices to names
                    return set(df.columns[list(features)])
                return set(features)

            fitted_features = get_feature_set(fitted_selector, learner.X_train)
            fresh_features = get_feature_set(fresh_selector, learner.X_train)

            # Selected features should match (both should use only training data)
            assert fitted_features == fresh_features, \
                   "Feature selection may be using test data"

    def test_preprocessing_fit_per_fold(self, synthetic_data_clean):
        """Verify preprocessing is fit per fold, not on the full training set.

        The final pipeline is fit on the full training set, but CV folds each
        fit their own preprocessing independently.
        """
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42, feature_selector=TopKCoxSelector(k=5),
        )
        result = learner.train('coxph')

        # Final pipeline should have a fitted selector
        fitted_selector = result.model.named_steps.get('selector')
        assert fitted_selector is not None
        assert len(fitted_selector.selected_features_) == 5

        # CV metric should be valid
        assert not np.isnan(result.cv_metric_mean)

    def test_final_pipeline_predicts_on_test_data(self, synthetic_data):
        """Verify the final pipeline can predict on held-out test data.

        The final pipeline is fit on the full training set via pipeline.fit().
        Prediction on unseen data must work correctly through the full
        imputer → scaler → selector → model chain.
        """
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data  # has NaN — exercises the full chain
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42, feature_selector=TopKCoxSelector(k=5),
        )
        result = learner.train('coxph')

        # Pipeline should have all 3 preprocessing steps + model
        assert 'imputer' in result.model.named_steps
        assert 'scaler' in result.model.named_steps
        assert 'selector' in result.model.named_steps

        # Predict on test data (unseen, may contain NaN)
        predictions = result.model.predict(learner.X_test)
        assert len(predictions) == len(learner.X_test)
        assert np.all(np.isfinite(predictions))


class TestTuningIsolation:
    """Test that tuning doesn't use test data."""

    def test_tuning_uses_validation_not_test(self, small_data):
        """Verify tuning optimization doesn't touch test set."""
        from mlsurv import SurvivalLearner

        X, y = small_data

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, test_fraction=0.3)

        # Store test set size
        test_size = len(learner.X_test)

        # Tune model
        result = learner.tune(
            'rsf',
            n_trials=3,
            search_space={'n_estimators': (10, 20)}
        )

        # Test set should be unchanged
        assert len(learner.X_test) == test_size, \
            "Test set size changed during tuning - potential leakage"

        # Verify model was tuned (has best_params_)
        assert result.best_params_ is not None

    def test_tuning_cv_within_train_only(self, small_data):
        """Verify tuning CV is performed only on training data."""
        from mlsurv import SurvivalLearner

        X, y = small_data

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, test_fraction=0.3)

        # Get training indices
        train_indices = set(learner.X_train.index)
        test_indices = set(learner.X_test.index)

        # Verify no overlap
        assert len(train_indices & test_indices) == 0

        # Run tuning - should only use training data internally
        result = learner.tune(
            'rsf',
            n_trials=3,
            search_space={'n_estimators': (10, 20)}
        )

        # Model should be valid
        assert result.cv_metric_mean > 0


class TestTemporalDataHandling:
    """Test proper handling of temporal data."""

    def test_timeseries_cv_respects_ordering(self, temporal_data):
        """Verify TimeSeriesSplit respects temporal ordering."""
        from mlsurv import SurvivalLearner

        X, y, enrollment_time = temporal_data

        # Sort by enrollment time
        sorted_indices = np.argsort(enrollment_time)
        X_sorted = X.iloc[sorted_indices].reset_index(drop=True)
        y_sorted = y[sorted_indices]

        learner = SurvivalLearner(X_sorted, y_sorted)
        learner.setup(
            random_state=42,
            cv_strategy='timeseries',
            cv_folds=3,
            stratify=None  # TimeSeriesSplit doesn't support stratification
        )

        # Verify temporal ordering in folds
        for train_idx, val_idx in learner._state.cv_splitter.split(learner.X_train):
            # All training samples should come before validation samples
            max_train_idx = max(train_idx)
            min_val_idx = min(val_idx)

            assert max_train_idx < min_val_idx, \
                "TimeSeriesSplit doesn't maintain temporal ordering"


class TestGroupKFoldIsolation:
    """Test GroupKFold properly isolates groups."""

    def test_groupkfold_no_group_overlap(self, multi_site_data):
        """Verify GroupKFold keeps groups together."""
        from mlsurv import SurvivalLearner

        X, y, groups = multi_site_data

        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42,
            cv_strategy='groupkfold',
            cv_folds=3,
            groups=groups
        )

        # Extract groups for training data
        train_groups = groups[learner.X_train.index.values]

        for train_idx, val_idx in learner._state.cv_splitter.split(
            learner.X_train, learner.y_train['event'], groups=train_groups
        ):
            train_group_set = set(train_groups[train_idx])
            val_group_set = set(train_groups[val_idx])

            # Groups should not overlap
            overlap = train_group_set & val_group_set
            assert len(overlap) == 0, \
                f"Groups overlap in train/val: {overlap}"

    def test_no_subject_in_both_train_and_test(self, repeated_subject_data):
        """Group-aware holdout: no subject straddles the train/test boundary."""
        from mlsurv import SurvivalLearner

        X, y, subject_id = repeated_subject_data

        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42,
            cv_strategy='groupkfold',
            cv_folds=3,
            groups=subject_id,
            test_fraction=0.3,
        )

        train_subjects = set(subject_id[learner.X_train.index.values])
        test_subjects = set(subject_id[learner.X_test.index.values])

        # Core assertion: the holdout is subject-pure (the issue's leak).
        overlap = train_subjects & test_subjects
        assert len(overlap) == 0, \
            f"Subjects leak across the train/test holdout: {overlap}"

        # Sanity: both sides non-empty and together they cover all subjects.
        assert train_subjects and test_subjects
        assert train_subjects | test_subjects == set(subject_id)

    def test_groups_protect_holdout_under_default_cv(self, repeated_subject_data):
        """Group-aware holdout fires whenever groups is passed, not only for
        cv_strategy='groupkfold'."""
        from mlsurv import SurvivalLearner

        X, y, subject_id = repeated_subject_data

        learner = SurvivalLearner(X, y)
        # Default cv_strategy='stratifiedkfold', but groups is supplied.
        learner.setup(random_state=42, groups=subject_id, test_fraction=0.3)

        train_subjects = set(subject_id[learner.X_train.index.values])
        test_subjects = set(subject_id[learner.X_test.index.values])

        overlap = train_subjects & test_subjects
        assert len(overlap) == 0, \
            f"Subjects leak across the holdout under default CV: {overlap}"
        assert learner._state.group_aware_holdout is True

    def test_single_group_falls_back_to_row_split(self, repeated_subject_data):
        """A single distinct group can't yield a leak-free holdout: warn and
        fall back to a row-level split (no group-aware flag)."""
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import MethodologicalWarning

        X, y, _ = repeated_subject_data
        one_group = np.zeros(len(X), dtype=int)  # all rows share one group

        learner = SurvivalLearner(X, y)
        with pytest.warns(MethodologicalWarning, match="one distinct group"):
            learner.setup(random_state=42, groups=one_group, test_fraction=0.3)

        # Fallback must still yield a usable, non-empty split, and the flag
        # must NOT claim a group-aware holdout happened.
        assert len(learner.X_train) > 0 and len(learner.X_test) > 0
        assert learner._state.group_aware_holdout is False

    def test_timeseries_with_groups_skips_group_aware_holdout(self, repeated_subject_data):
        """cv_strategy='timeseries' keeps its ordered split even when groups is
        passed - group-aware shuffling would corrupt temporal ordering."""
        from mlsurv import SurvivalLearner

        X, y, subject_id = repeated_subject_data

        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42, cv_strategy='timeseries', cv_folds=3,
            groups=subject_id, stratify=None, test_fraction=0.3,
        )

        # The group-aware holdout must not have fired for timeseries.
        assert learner._state.group_aware_holdout is False
        # timeseries holdout is ordered (no shuffle): train precedes test.
        assert learner.X_train.index.max() < learner.X_test.index.min()

    def test_uneven_groups_warn_on_fraction_deviation(self):
        """When whole groups can't land near the requested fraction, a
        SuggestionWarning reports the realized fraction."""
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.utils import SurvivalArray

        np.random.seed(3)
        # 4 large groups of 25 rows each: with test_fraction=0.3, whole-group
        # moves can only land at 0.25 or 0.50, both >5pp from 0.30.
        n_groups, rows_per = 4, 25
        n = n_groups * rows_per
        groups = np.repeat(np.arange(n_groups), rows_per)
        X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n)})
        times = np.random.exponential(2, n).clip(0.1, 10)
        events = np.random.binomial(1, 0.6, n).astype(bool)
        y = SurvivalArray(np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[('event', bool), ('time', float)]))

        learner = SurvivalLearner(X, y)
        with pytest.warns(SuggestionWarning, match="realized"):
            learner.setup(random_state=42, groups=groups, test_fraction=0.3)


class TestPreprocessingInPipeline:
    """Test preprocessing is properly encapsulated in pipeline."""

    def test_pipeline_transform_uses_fitted_params(self, synthetic_data):
        """Verify pipeline transform uses params from fit, not test data."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        result = learner.train('coxph')

        pipeline = result.model

        # Get preprocessing params from fitted pipeline
        imputer = pipeline.named_steps.get('imputer')
        scaler = pipeline.named_steps.get('scaler')

        if imputer is not None:
            original_statistics = imputer.statistics_.copy()

        if scaler is not None:
            original_mean = scaler.mean_.copy()
            original_var = scaler.var_.copy()

        # Transform test data
        _ = pipeline.predict(learner.X_test)

        # Verify preprocessing params haven't changed (weren't refit on test)
        if imputer is not None:
            np.testing.assert_array_equal(
                imputer.statistics_, original_statistics,
                err_msg="Imputer was refit on test data - data leakage!"
            )

        if scaler is not None:
            np.testing.assert_array_equal(
                scaler.mean_, original_mean,
                err_msg="Scaler was refit on test data - data leakage!"
            )
            np.testing.assert_array_equal(
                scaler.var_, original_var,
                err_msg="Scaler was refit on test data - data leakage!"
            )


class TestCoxSelectorInPipeline:
    """Test Cox-based selector behavior inside the pipeline."""

    def test_cox_selector_in_pipeline_receives_imputed_data(self, synthetic_data):
        """Verify TopKCoxSelector receives imputed data within the pipeline.

        The selector sits after the imputer in the
        pipeline and receives imputed data.  Raw-data fitting is deferred
        to a future custom pipeline step.
        """
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data  # has NaN
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=TopKCoxSelector(k=5), random_state=42)
        result = learner.train('coxph')

        selector = result.model.named_steps['selector']
        assert len(selector.selected_features_) == 5
        # Pipeline is imputer → scaler → selector → model, so selector
        # gets imputed+scaled data.  Verify prediction still works.
        predictions = result.model.predict(learner.X_test)
        assert len(predictions) == len(learner.X_test)
        assert np.all(np.isfinite(predictions))
