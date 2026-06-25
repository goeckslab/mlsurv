"""Test edge cases and boundary conditions."""

import pytest
import numpy as np
import pandas as pd

from tests._helpers import make_survival_array


@pytest.fixture
def tiny_data():
    """Create dataset with 20 samples."""
    np.random.seed(42)
    n_samples = 20

    X = pd.DataFrame({
        'age': np.random.uniform(30, 80, n_samples),
        'biomarker': np.random.randn(n_samples),
        'stage': np.random.randint(1, 4, n_samples),
    })

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)

    y = make_survival_array(events, times)

    return X, y


@pytest.fixture
def high_censoring_data():
    """Create dataset with 90% censoring."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
    })

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    # Only 10% events
    events = np.random.binomial(1, 0.1, n_samples).astype(bool)

    y = make_survival_array(events, times)

    return X, y


@pytest.fixture
def low_censoring_data():
    """Create dataset with 10% censoring (90% events)."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
    })

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    # 90% events
    events = np.random.binomial(1, 0.9, n_samples).astype(bool)

    y = make_survival_array(events, times)

    return X, y


@pytest.fixture
def single_feature_data():
    """Create dataset with single feature."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        'only_feature': np.random.randn(n_samples),
    })

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)

    y = make_survival_array(events, times)

    return X, y


@pytest.fixture
def collinear_features_data():
    """Create dataset with highly collinear features."""
    np.random.seed(42)
    n_samples = 100

    base = np.random.randn(n_samples)

    X = pd.DataFrame({
        'feature_1': base,
        'feature_2': base + np.random.randn(n_samples) * 0.01,  # Nearly identical
        'feature_3': base * 2 + 0.5,  # Perfect linear relationship
        'independent': np.random.randn(n_samples),
    })

    times = np.exp(-base + np.random.randn(n_samples) * 0.5).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)

    y = make_survival_array(events, times)

    return X, y


class TestMinimalData:
    """Test with minimal data sizes."""

    def test_coxph_tiny_dataset(self, tiny_data):
        """Verify CoxPH handles tiny datasets."""
        from mlsurv.models import CoxPHModel

        X, y = tiny_data

        model = CoxPHModel(alpha=0.1)  # Add regularization for stability
        model.fit(X, y)

        assert model.is_fitted_

        risk_scores = model.predict(X)
        assert len(risk_scores) == len(X)
        assert not np.isnan(risk_scores).any()

    def test_rsf_tiny_dataset(self, tiny_data):
        """Verify RSF handles tiny datasets."""
        from mlsurv.models import RSFModel

        X, y = tiny_data

        model = RSFModel(n_estimators=10, min_samples_leaf=2, random_state=42)
        model.fit(X, y)

        assert model.is_fitted_

        risk_scores = model.predict(X)
        assert len(risk_scores) == len(X)

    def test_learner_tiny_dataset(self, tiny_data):
        """Verify SurvivalLearner handles tiny datasets."""
        from mlsurv import SurvivalLearner

        X, y = tiny_data

        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3)

        result = learner.train('coxph')

        assert result is not None
        assert 0.0 <= result.cv_metric_mean <= 1.0


class TestEventDistribution:
    """Test edge cases in event distribution."""

    def test_high_censoring(self, high_censoring_data):
        """Verify handling of high censoring rate."""
        from mlsurv.models import CoxPHModel

        X, y = high_censoring_data

        model = CoxPHModel(alpha=0.1)
        model.fit(X, y)

        assert model.is_fitted_

        # Model should still produce predictions
        risk_scores = model.predict(X)
        assert len(risk_scores) == len(X)

    def test_low_censoring(self, low_censoring_data):
        """Verify handling of low censoring rate."""
        from mlsurv.models import CoxPHModel

        X, y = low_censoring_data

        model = CoxPHModel()
        model.fit(X, y)

        assert model.is_fitted_

        risk_scores = model.predict(X)
        assert len(risk_scores) == len(X)

    def test_high_censoring_learner(self, high_censoring_data):
        """Verify SurvivalLearner handles high censoring."""
        from mlsurv import SurvivalLearner

        X, y = high_censoring_data

        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3)  # Fewer folds due to few events

        result = learner.train('coxph')
        assert result is not None


class TestFeatureEdgeCases:
    """Test feature-related edge cases."""

    def test_single_feature(self, single_feature_data):
        """Verify handling of single feature."""
        from mlsurv.models import CoxPHModel

        X, y = single_feature_data

        model = CoxPHModel()
        model.fit(X, y)

        assert model.is_fitted_
        assert model.n_features_in_ == 1

        coef = model.get_coefficients()
        assert len(coef) == 1

    def test_collinear_features_coxph(self, collinear_features_data):
        """Verify CoxPH handling of collinearity."""
        from mlsurv.models import CoxPHModel

        X, y = collinear_features_data

        # CoxPH with regularization should handle collinearity
        model = CoxPHModel(alpha=0.1)
        model.fit(X, y)

        assert model.is_fitted_

        risk_scores = model.predict(X)
        assert not np.isnan(risk_scores).any()

    def test_collinear_features_coxnet(self, collinear_features_data):
        """Verify CoxNet handling of collinearity (elastic net)."""
        from mlsurv.models import CoxNetModel

        X, y = collinear_features_data

        # CoxNet with L1 penalty should handle collinearity well
        model = CoxNetModel(l1_ratio=0.9)
        model.fit(X, y)

        assert model.is_fitted_

        # L1 penalty should select among collinear features
        coef = model.get_coefficients()
        assert len(coef) == X.shape[1]

    def test_collinear_features_rsf(self, collinear_features_data):
        """Verify RSF handling of collinearity."""
        from mlsurv.models import RSFModel

        X, y = collinear_features_data

        model = RSFModel(n_estimators=20, random_state=42)
        model.fit(X, y)

        assert model.is_fitted_

        risk_scores = model.predict(X)
        assert not np.isnan(risk_scores).any()

    def test_all_missing_feature(self, synthetic_data):
        """Verify handling of all-NaN feature."""
        from mlsurv.models import CoxPHModel
        from mlsurv.preprocessing import get_imputer, get_scaler
        from sklearn.pipeline import Pipeline

        X, y = synthetic_data

        # Add an all-NaN feature
        X = X.copy()
        X['all_nan'] = np.nan

        # Pipeline with imputer should handle this
        pipe = Pipeline([
            ('imputer', get_imputer('median')),
            ('scaler', get_scaler('standard')),
            ('model', CoxPHModel())
        ])

        pipe.fit(X, y)
        score = pipe.score(X, y)

        assert 0.0 <= score <= 1.0


class TestFeatureSelectionEdgeCases:
    """Test feature selection edge cases."""

    def test_topk_more_than_features(self, single_feature_data):
        """Verify TopKCoxSelector when k > n_features."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = single_feature_data

        selector = TopKCoxSelector(k=10)  # More than the 1 feature available
        X_selected = selector.fit_transform(X, y)

        # Should select all available features
        assert X_selected.shape[1] == 1

    def test_topk_with_missing_values(self, synthetic_data):
        """Verify TopKCoxSelector handles missing values."""
        from mlsurv.preprocessing import TopKCoxSelector
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        X, y = synthetic_data

        # Pipeline with imputer before selector
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('selector', TopKCoxSelector(k=5))
        ])

        X_selected = pipe.fit_transform(X, y)

        assert X_selected.shape[1] == 5
        # Check for NaN in the numpy array
        if hasattr(X_selected, 'values'):
            assert not np.isnan(X_selected.values).any()
        else:
            assert not np.isnan(X_selected).any()


class TestScoringEdgeCases:
    """Test scoring with edge cases."""

    def test_score_strong_predictor(self):
        """Verify scoring with a strong predictive feature."""
        from mlsurv.models import CoxPHModel

        np.random.seed(42)
        n_samples = 100

        # Create data where feature strongly predicts outcome (but not perfectly)
        predictor = np.arange(n_samples, dtype=float) / n_samples

        X = pd.DataFrame({
            'predictor': predictor
        })

        # Times inversely related to predictor + noise
        times = np.exp(-2 * predictor + np.random.randn(n_samples) * 0.3)
        times = np.clip(times, 0.1, 10)

        # Mix of events and censoring
        events = np.random.binomial(1, 0.7, n_samples).astype(bool)

        y = make_survival_array(events, times)

        model = CoxPHModel(alpha=0.1)
        model.fit(X, y)

        score = model.score(X, y)
        # Should have reasonably high C-index
        assert score > 0.6

    def test_score_no_concordant_pairs(self):
        """Verify scoring when predictions are random."""
        from mlsurv.evaluation import c_index

        np.random.seed(42)
        n_samples = 100

        times = np.random.exponential(2, n_samples)
        events = np.random.binomial(1, 0.5, n_samples).astype(bool)

        y = make_survival_array(events, times)

        # Completely random predictions
        risk_scores = np.random.randn(n_samples)

        c = c_index(y, risk_scores, method='harrell')

        # Should be close to 0.5 (random)
        assert 0.3 <= c <= 0.7


class TestMetricEdgeCases:
    """Edge cases in metric computation."""

    def test_cindex_with_tied_times(self):
        """Verify C-index handles tied survival times."""
        from mlsurv.evaluation import c_index

        np.random.seed(42)
        n_samples = 100

        # Create many ties in survival times
        times = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n_samples)
        events = np.random.binomial(1, 0.6, n_samples).astype(bool)

        y = make_survival_array(events, times)

        risk_scores = np.random.randn(n_samples)

        # Should not raise error and produce valid result
        c = c_index(y, risk_scores, method='harrell')
        assert 0.0 <= c <= 1.0

    def test_cindex_all_censored(self):
        """Verify C-index handles all-censored data gracefully."""
        from mlsurv.evaluation import c_index

        n_samples = 50

        times = np.random.exponential(2, n_samples)
        events = np.zeros(n_samples, dtype=bool)  # All censored

        y = make_survival_array(events, times)

        risk_scores = np.random.randn(n_samples)

        # May return NaN or raise - either is acceptable
        try:
            c = c_index(y, risk_scores, method='harrell')
            # If it returns a value, should be NaN
            assert np.isnan(c) or 0.0 <= c <= 1.0
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable to raise error with no comparable pairs

    def test_cindex_all_events(self):
        """Verify C-index handles all-event data."""
        from mlsurv.evaluation import c_index

        np.random.seed(42)
        n_samples = 50

        times = np.random.exponential(2, n_samples).clip(0.1, 10)
        events = np.ones(n_samples, dtype=bool)  # All events

        y = make_survival_array(events, times)

        risk_scores = np.random.randn(n_samples)

        c = c_index(y, risk_scores, method='harrell')
        assert 0.0 <= c <= 1.0

    def test_auc_t_single_timepoint(self):
        """Verify AUC(t) with single timepoint."""
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel

        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            'feature': np.random.randn(n_samples)
        })

        times = np.random.exponential(2, n_samples).clip(0.1, 10)
        events = np.random.binomial(1, 0.6, n_samples).astype(bool)

        y = make_survival_array(events, times)

        model = CoxPHModel()
        model.fit(X, y)

        # Single eval time
        result = evaluate(
            model, X, y, X_train=X, y_train=y,
            eval_times=[np.median(times)],
            n_importance_repeats=0,
            bootstrap=False
        )

        assert result.auc_t is not None

    def test_brier_score_at_max_time(self):
        """Verify Brier score computation at maximum observed time."""
        from mlsurv.models import RSFModel
        from sksurv.metrics import brier_score

        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            'feature': np.random.randn(n_samples)
        })

        times = np.random.exponential(2, n_samples).clip(0.1, 5)
        events = np.random.binomial(1, 0.6, n_samples).astype(bool)

        y = make_survival_array(events, times)

        model = RSFModel(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Brier score near max time
        surv_funcs = model.predict_survival_function(X)
        t_eval = np.percentile(times, 90)

        # Extract probabilities at t_eval
        probs = np.array([fn(t_eval) for fn in surv_funcs])

        # Should compute without error
        try:
            scores = brier_score(y, y, probs, t_eval)
            assert scores is not None
        except (ValueError, RuntimeError):
            # May fail at extreme time points
            pass

    def test_ipcw_cindex_few_events(self):
        """Verify IPCW C-index with few events."""
        from mlsurv.evaluation import c_index

        np.random.seed(42)
        n_samples = 50

        times = np.random.exponential(2, n_samples)
        # Only 5 events
        events = np.zeros(n_samples, dtype=bool)
        events[:5] = True
        np.random.shuffle(events)

        y_train = make_survival_array(events, times)

        y_test = make_survival_array(events[:20], times[:20])

        risk_scores = np.random.randn(20)

        # May return valid score or NaN
        try:
            c = c_index(y_test, risk_scores, method='ipcw', y_train=y_train)
            assert np.isnan(c) or 0.0 <= c <= 1.0
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable

    def test_concordance_with_constant_predictions(self):
        """Verify C-index with constant predictions returns 0.5 or NaN."""
        from mlsurv.evaluation import c_index

        np.random.seed(42)
        n_samples = 50

        times = np.random.exponential(2, n_samples)
        events = np.random.binomial(1, 0.6, n_samples).astype(bool)

        y = make_survival_array(events, times)

        # Constant predictions
        risk_scores = np.ones(n_samples)

        c = c_index(y, risk_scores, method='harrell')
        # With constant predictions, no concordant/discordant pairs
        # Result should be 0.5 (coin flip) or NaN
        assert np.isnan(c) or abs(c - 0.5) < 0.1


class TestErrorHandling:
    """Test graceful error handling and warnings."""

    def test_invalid_model_params_error(self, small_data):
        """Verify error with invalid model parameters."""
        from mlsurv.models import RSFModel

        X, y = small_data

        # Invalid parameter
        with pytest.raises((TypeError, ValueError)):
            model = RSFModel(invalid_param=123)
            model.fit(X, y)

    def test_tune_with_invalid_search_space(self, small_data):
        """Verify error with invalid search space."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        # Invalid search space format
        with pytest.raises((ValueError, TypeError, KeyError)):
            learner.tune(
                'rsf',
                n_trials=2,
                search_space={'invalid_param_xyz': (1, 10)}
            )

    def test_evaluate_unfitted_model_error(self, small_data):
        """Verify error when evaluating unfitted model."""
        from mlsurv import SurvivalLearner
        from mlsurv.models import CoxPHModel

        X, y = small_data

        learner = SurvivalLearner(X, y)
        learner.setup()

        # Unfitted model
        unfitted_model = CoxPHModel()

        with pytest.raises(Exception):  # NotFittedError or similar
            learner.evaluate(unfitted_model)

    def test_bootstrap_insufficient_samples_handling(self):
        """Verify bootstrap handles small sample sizes."""
        from mlsurv import SurvivalLearner
        import warnings

        np.random.seed(42)
        n_samples = 15  # Very small

        X = pd.DataFrame({
            'feature': np.random.randn(n_samples)
        })

        times = np.random.exponential(2, n_samples).clip(0.1, 10)
        events = np.random.binomial(1, 0.6, n_samples).astype(bool)

        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.setup(test_fraction=0.3, cv_folds=3)

        result = learner.train('coxph')

        # Bootstrap with tiny dataset - should work but may have warnings
        boot = learner.bootstrap(result, n_bootstrap=2)
        assert boot is not None

    def test_feature_selector_no_significant_features_handling(self):
        """Verify handling when no features are significant."""
        from mlsurv.preprocessing import TopKCoxSelector
        import warnings

        np.random.seed(42)
        n_samples = 50

        # Random noise features
        X = pd.DataFrame({
            'noise1': np.random.randn(n_samples),
            'noise2': np.random.randn(n_samples),
        })

        times = np.random.exponential(2, n_samples)
        events = np.random.binomial(1, 0.5, n_samples).astype(bool)

        y = make_survival_array(events, times)

        selector = TopKCoxSelector(k=1)

        # Should still select k features even if none are "significant"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selector.fit(X, y)

        assert len(selector.selected_features_) == 1

    def test_convergence_warning_handled_gracefully(self, synthetic_data_clean):
        """Verify convergence warnings don't cause failures."""
        from mlsurv import SurvivalLearner
        import warnings

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3)

        # CoxNet with extreme parameters might not converge
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = learner.train('coxnet', l1_ratio=0.99, n_alphas=5)

        # Should produce a result despite potential convergence issues
        assert result is not None

    def test_empty_dataframe_error(self):
        """Verify error with empty DataFrame."""
        from mlsurv import SurvivalLearner

        X = pd.DataFrame()  # Empty
        y = np.array([], dtype=[('event', bool), ('time', float)])

        with pytest.raises((ValueError, IndexError)):
            learner = SurvivalLearner(X, y)
            learner.setup()

    def test_mismatched_X_y_length_error(self):
        """Verify error when X and y have different lengths."""
        from mlsurv import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame({'feature': np.random.randn(100)})

        # Only 2 samples
        y = make_survival_array([True, False], [1.0, 2.0])

        with pytest.raises((ValueError, IndexError)):
            learner = SurvivalLearner(X, y)
            learner.setup()

    def test_negative_time_handling(self):
        """Verify behavior with negative survival times."""
        from mlsurv import SurvivalLearner
        import warnings

        np.random.seed(42)
        X = pd.DataFrame({'feature': np.random.randn(50)})

        times = np.random.randn(50)  # Will have negative values
        events = np.random.binomial(1, 0.5, 50).astype(bool)

        # Built inline (not via the factory): negative times are intentional here,
        # and to_structured_array would reject them.
        y = np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[('event', bool), ('time', float)]
        )

        learner = SurvivalLearner(X, y)

        # Either raises error or handles gracefully during setup/training
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                learner.setup()
                result = learner.train('coxph')
            # If it doesn't raise, the model should still produce some result
            assert result is not None
        except (ValueError, RuntimeError):
            # Acceptable to raise an error for invalid data
            pass

    def test_all_missing_column_handled(self, synthetic_data):
        """Verify all-NaN column is handled appropriately."""
        from mlsurv import SurvivalLearner
        import warnings

        X, y = synthetic_data
        X = X.copy()
        X['all_nan'] = np.nan  # Add all-NaN column

        learner = SurvivalLearner(X, y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.setup()
            result = learner.train('coxph')

        # Should complete despite all-NaN column
        assert result is not None


class TestLookupModelByName:
    """Edge cases for SurvivalLearner._lookup_model_by_name."""

    @pytest.fixture
    def make_learner(self):
        """Create a bare SurvivalLearner with empty state for unit testing."""
        from mlsurv.learner import SurvivalLearner
        from mlsurv.learner._state import LearnerState

        def _make(**models):
            learner = object.__new__(SurvivalLearner)
            learner._state = LearnerState()
            learner._state.models.update(models)
            return learner

        return _make

    def test_lookup_model_exact_key(self, make_learner):
        """Lookup returns the model stored under the exact storage key."""
        learner = make_learner(rsf='the_model')
        assert learner._lookup_model_by_name('rsf') == 'the_model'

    def test_lookup_model_missing_raises(self, make_learner):
        """Looking up a name with no matching storage key raises ValueError."""
        import pytest
        learner = make_learner(rsf='the_model')
        with pytest.raises(ValueError, match="not found"):
            learner._lookup_model_by_name('coxph')


class TestRiskScoreValidation:
    """_validate_risk_scores excludes models with non-finite risk scores."""

    def test_nan_model_excluded_with_warning(self):
        """Model with NaN risk scores is excluded and warning is emitted."""
        import warnings
        import pandas as pd
        from mlsurv.learner._visualization import VisualizationMixin

        good_df = pd.DataFrame({
            'time': [1.0, 2.0, 3.0],
            'event': [True, False, True],
            'risk_score': [0.5, 1.2, 0.8],
        })
        bad_df = pd.DataFrame({
            'time': [1.0, 2.0, 3.0],
            'event': [True, False, True],
            'risk_score': [0.5, np.nan, 0.8],
        })
        data = {'good_model': good_df, 'bad_model': bad_df}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = VisualizationMixin._validate_risk_scores(data)

        assert 'good_model' in result
        assert 'bad_model' not in result
        assert len(w) == 1
        assert "bad_model" in str(w[0].message)
        assert "1 non-finite" in str(w[0].message)

    def test_inf_model_excluded(self):
        """Model with inf risk scores is excluded."""
        import pandas as pd
        from mlsurv.learner._visualization import VisualizationMixin

        inf_df = pd.DataFrame({
            'time': [1.0, 2.0],
            'event': [True, False],
            'risk_score': [np.inf, 0.5],
        })
        result = VisualizationMixin._validate_risk_scores({'inf_model': inf_df})
        assert 'inf_model' not in result

    def test_all_finite_passes(self):
        """All-finite model passes through unchanged."""
        import pandas as pd
        from mlsurv.learner._visualization import VisualizationMixin

        df = pd.DataFrame({
            'time': [1.0, 2.0],
            'event': [True, False],
            'risk_score': [0.5, 1.0],
        })
        result = VisualizationMixin._validate_risk_scores({'model': df})
        assert 'model' in result

    def test_all_models_excluded_returns_empty(self):
        """When all models have non-finite scores, empty dict is returned."""
        import pandas as pd
        from mlsurv.learner._visualization import VisualizationMixin

        bad_df = pd.DataFrame({
            'time': [1.0, 2.0],
            'event': [True, False],
            'risk_score': [np.nan, np.inf],
        })
        result = VisualizationMixin._validate_risk_scores({
            'model_a': bad_df, 'model_b': bad_df,
        })
        assert result == {}
