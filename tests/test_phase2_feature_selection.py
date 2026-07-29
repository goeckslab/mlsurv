"""Test feature selection for Phase 2."""

import warnings

import pytest
import numpy as np
import pandas as pd

from mlsurv._warnings import MethodologicalWarning

from tests._helpers import make_survival_array, make_y


class TestTopKCoxSelector:
    """Test TopKCoxSelector feature selection."""

    def test_basic_selection(self, synthetic_data_clean):
        """Verify TopKCoxSelector selects k features."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean

        selector = TopKCoxSelector(k=10)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[1] == 10
        assert hasattr(selector, 'selected_features_')
        assert len(selector.selected_features_) == 10

    def test_selector_attributes(self, synthetic_data_clean):
        """Verify TopKCoxSelector sets correct attributes after fit."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean

        selector = TopKCoxSelector(k=5)
        selector.fit(X, y)

        # Check sorted_features_ contains tuples
        assert hasattr(selector, 'sorted_features_')
        assert len(selector.sorted_features_) == X.shape[1]
        assert isinstance(selector.sorted_features_[0], tuple)
        assert len(selector.sorted_features_[0]) == 2  # (feature_name, p_value)

    def test_transform_preserves_column_names(self, synthetic_data_clean):
        """Verify transform returns DataFrame with correct columns."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean

        selector = TopKCoxSelector(k=5)
        X_selected = selector.fit_transform(X, y)

        assert isinstance(X_selected, pd.DataFrame)
        assert list(X_selected.columns) == selector.selected_features_

    def test_get_support(self, synthetic_data_clean):
        """Verify get_support returns correct format."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean

        selector = TopKCoxSelector(k=5)
        selector.fit(X, y)

        # Boolean mask
        support = selector.get_support()
        assert len(support) == X.shape[1]
        assert sum(support) == 5

        # Indices
        indices = selector.get_support(indices=True)
        assert len(indices) == 5

    def test_get_feature_names_out(self, synthetic_data_clean):
        """Verify sklearn 1.0+ compatibility."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean

        selector = TopKCoxSelector(k=5)
        selector.fit(X, y)

        feature_names = selector.get_feature_names_out()
        assert feature_names == selector.selected_features_

    def test_sklearn_params(self):
        """Verify sklearn get/set_params compatibility."""
        from mlsurv.preprocessing import TopKCoxSelector

        selector = TopKCoxSelector(k=10, penalizer=0.05)

        params = selector.get_params()
        assert params['k'] == 10
        assert params['penalizer'] == 0.05

        selector.set_params(k=20)
        assert selector.k == 20

    def test_k_larger_than_features(self, synthetic_data_clean):
        """Verify behavior when k > n_features."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean

        selector = TopKCoxSelector(k=100)  # More than available features
        X_selected = selector.fit_transform(X, y)

        # Should select all features
        assert X_selected.shape[1] == X.shape[1]


class TestPipelineIntegration:
    """Test feature selection in sklearn Pipeline."""

    def test_selector_in_pipeline(self, synthetic_data_clean):
        """Verify TopKCoxSelector works in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from mlsurv.preprocessing import TopKCoxSelector
        from mlsurv.models import CoxPHModel

        X, y = synthetic_data_clean

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', TopKCoxSelector(k=10)),
            ('model', CoxPHModel())
        ])

        pipe.fit(X, y)
        score = pipe.score(X, y)

        assert 0.0 <= score <= 1.0

    def test_no_data_leakage_in_cv(self, synthetic_data_clean):
        """Verify feature selection doesn't leak data in CV."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from mlsurv.preprocessing import TopKCoxSelector
        from mlsurv.models import CoxPHModel

        X, y = synthetic_data_clean

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', TopKCoxSelector(k=10)),
            ('model', CoxPHModel())
        ])

        # This should work without data leakage
        scores = cross_val_score(pipe, X, y, cv=3)

        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestTopKCoxSelectorAutoK:
    """Test k='auto' (Harrell m/15 rule) and related features."""

    def test_default_is_auto(self):
        """Verify default k is 'auto'."""
        from mlsurv.preprocessing import TopKCoxSelector

        selector = TopKCoxSelector()
        assert selector.k == 'auto'

    def test_auto_resolves_harrell_rule(self, synthetic_data_clean):
        """Verify k='auto' resolves via Harrell m/15 rule."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        # synthetic_data_clean: 200 samples, 20 features, ~70% events
        n_events = int(y['event'].sum())
        n_features = X.shape[1]
        expected_k = min(n_features, max(1, n_events // 15))

        selector = TopKCoxSelector()
        selector.fit(X, y)

        assert selector.effective_k_ == expected_k
        assert len(selector.selected_features_) == expected_k

    def test_auto_few_events(self):
        """Verify k='auto' with very few events gives effective_k=1."""
        from mlsurv.preprocessing import TopKCoxSelector

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f'f{i}' for i in range(5)])
        # Only 5 events -> 5 // 15 = 0, so max(1, 0) = 1
        events = np.zeros(50, dtype=bool)
        events[:5] = True
        times = np.random.exponential(2, 50).clip(0.1, 10)
        y = make_survival_array(events, times)

        selector = TopKCoxSelector()
        selector.fit(X, y)
        assert selector.effective_k_ == 1

    def test_auto_more_events_than_features(self):
        """Verify k='auto' clamps to n_features when m/15 > n_features."""
        from mlsurv.preprocessing import TopKCoxSelector

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])
        # ~140 events -> 140 // 15 = 9, but only 3 features -> effective_k = 3
        events = np.random.binomial(1, 0.7, 200).astype(bool)
        times = np.random.exponential(2, 200).clip(0.1, 10)
        y = make_survival_array(events, times)

        selector = TopKCoxSelector()
        selector.fit(X, y)
        assert selector.effective_k_ == 3
        assert len(selector.selected_features_) == 3

    def test_explicit_k_clamped(self, synthetic_data_clean):
        """Verify explicit k > n_features is clamped."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        selector = TopKCoxSelector(k=100)
        selector.fit(X, y)
        assert selector.effective_k_ == X.shape[1]
        assert len(selector.selected_features_) == X.shape[1]

    def test_invalid_k_raises(self):
        """Verify invalid k values raise ValueError."""
        from mlsurv.preprocessing import TopKCoxSelector

        with pytest.raises(ValueError, match="k must be 'auto', 'tune', or an integer >= 1"):
            TopKCoxSelector(k=0)
        with pytest.raises(ValueError, match="k must be 'auto', 'tune', or an integer >= 1"):
            TopKCoxSelector(k=-1)
        with pytest.raises(ValueError, match="k must be 'auto', 'tune', or an integer >= 1"):
            TopKCoxSelector(k=3.5)

    def test_bool_k_rejected(self):
        """Verify k=True/False are rejected (bool is subclass of int)."""
        from mlsurv.preprocessing import TopKCoxSelector

        with pytest.raises(ValueError, match="k must be 'auto', 'tune', or an integer >= 1"):
            TopKCoxSelector(k=True)
        with pytest.raises(ValueError, match="k must be 'auto', 'tune', or an integer >= 1"):
            TopKCoxSelector(k=False)

    def test_explicit_k_equals_n_features(self, synthetic_data_clean):
        """Verify k == n_features is a no-op (selects all features)."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        n_features = X.shape[1]
        selector = TopKCoxSelector(k=n_features)
        selector.fit(X, y)
        assert selector.effective_k_ == n_features
        assert len(selector.selected_features_) == n_features

    def test_auto_few_events_warns(self):
        """Verify k='auto' emits UserWarning when < 15 events."""
        from mlsurv.preprocessing import TopKCoxSelector

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f'f{i}' for i in range(5)])
        events = np.zeros(50, dtype=bool)
        events[:5] = True
        times = np.random.exponential(2, 50).clip(0.1, 10)
        y = make_survival_array(events, times)

        selector = TopKCoxSelector()
        with pytest.warns(MethodologicalWarning, match="feature selection is highly constrained"):
            selector.fit(X, y)

    def test_verbose_auto_resolution(self, synthetic_data_clean, capsys):
        """Verify verbose=True prints auto-resolution message."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        selector = TopKCoxSelector(verbose=True)
        selector.fit(X, y)

        captured = capsys.readouterr()
        assert "k='auto' resolved to" in captured.out
        assert "Harrell m/15" in captured.out

    def test_verbose_clamped_message(self, capsys):
        """Verify verbose=True prints clamped message when k > n_features."""
        from mlsurv.preprocessing import TopKCoxSelector

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])
        events = np.random.binomial(1, 0.7, 200).astype(bool)
        times = np.random.exponential(2, 200).clip(0.1, 10)
        y = make_survival_array(events, times)

        selector = TopKCoxSelector(k=100, verbose=True)
        selector.fit(X, y)

        captured = capsys.readouterr()
        assert "clamped to 3" in captured.out

    def test_verbose_noop_message(self, capsys):
        """Verify verbose=True prints no-op message when k == n_features."""
        from mlsurv.preprocessing import TopKCoxSelector

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])
        events = np.random.binomial(1, 0.7, 200).astype(bool)
        times = np.random.exponential(2, 200).clip(0.1, 10)
        y = make_survival_array(events, times)

        selector = TopKCoxSelector(k=3, verbose=True)
        selector.fit(X, y)

        captured = capsys.readouterr()
        assert "no-op" in captured.out

    def test_repr_before_fit(self):
        """Verify repr shows k before fit."""
        from mlsurv.preprocessing import TopKCoxSelector

        selector = TopKCoxSelector()
        r = repr(selector)
        assert r == "TopKCoxSelector(k='auto')"

        selector2 = TopKCoxSelector(k=5)
        assert repr(selector2) == "TopKCoxSelector(k=5)"

    def test_repr_after_fit(self, synthetic_data_clean):
        """Verify repr shows k and effective_k after fit."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        selector = TopKCoxSelector()
        selector.fit(X, y)
        r = repr(selector)
        assert "k='auto'" in r
        assert f"effective_k={selector.effective_k_}" in r

    def test_repr_non_default_params(self):
        """Verify repr shows non-default params."""
        from mlsurv.preprocessing import TopKCoxSelector

        selector = TopKCoxSelector(k=5, penalizer=0.1, verbose=True)
        r = repr(selector)
        assert "k=5" in r
        assert "penalizer=0.1" in r
        assert "verbose=True" in r

    def test_sklearn_get_params(self):
        """Verify get_params() returns k='auto'."""
        from mlsurv.preprocessing import TopKCoxSelector

        selector = TopKCoxSelector()
        params = selector.get_params()
        assert params['k'] == 'auto'

    def test_pickle_roundtrip(self, synthetic_data_clean):
        """Verify k='auto' survives pickle serialization."""
        import pickle
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data_clean
        selector = TopKCoxSelector()
        selector.fit(X, y)

        data = pickle.dumps(selector)
        loaded = pickle.loads(data)

        assert loaded.k == 'auto'
        assert loaded.effective_k_ == selector.effective_k_
        assert loaded.selected_features_ == selector.selected_features_


class TestSetupSummaryEcho:
    """Test that setup() prints a boxed configuration summary."""

    def test_setup_prints_summary(self, capsys):
        """Verify setup() prints Pipeline and Data lines."""
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector='topk')

        captured = capsys.readouterr()
        assert "Pipeline:" in captured.out
        assert "Data:" in captured.out
        assert "TopKCoxSelector" in captured.out
        assert "features" in captured.out
        assert "EPV=" in captured.out

    def test_setup_summary_default_no_selector(self, capsys):
        """Verify setup() summary with default pipeline (no selector)."""
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner.setup()

        captured = capsys.readouterr()
        assert "Pipeline:" in captured.out
        assert "SimpleImputer" in captured.out
        assert "StandardScaler" in captured.out
        assert "TopKCoxSelector" not in captured.out
        assert "Data:" in captured.out

    def test_setup_verbose_false_suppresses_summary(self, capsys):
        """Verify setup(verbose=False) prints nothing."""
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_setup_summary_box_format(self, capsys):
        """Verify setup() prints boxed format with all 4 sections."""
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner.setup()

        captured = capsys.readouterr()
        assert "\u250c\u2500 Setup" in captured.out
        assert "\u2514\u2500" in captured.out
        assert "Pipeline:" in captured.out
        assert "Data:" in captured.out
        assert "CV:" in captured.out
        assert "State:" in captured.out
        assert "seed=42" in captured.out

    def test_setup_summary_data_line(self, capsys):
        """Verify Data line contains test%, features, and EPV."""
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner.setup()

        captured = capsys.readouterr()
        assert "30%" in captured.out
        assert "features" in captured.out
        assert "EPV=" in captured.out

    def test_setup_summary_cv_line(self, capsys):
        """Verify CV line shows strategy, folds, and scoring."""
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner.setup()

        captured = capsys.readouterr()
        assert "stratifiedkfold" in captured.out
        assert "k=5" in captured.out
        assert "scoring=harrell" in captured.out
        assert " repeats" not in captured.out

    def test_setup_summary_with_repeats_and_experiment(self, capsys):
        """Verify CV repeats and experiment_name appear when set."""
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner.setup(cv_repeats=3, experiment_name="my_study")

        captured = capsys.readouterr()
        assert "3 repeats" in captured.out
        assert "experiment=my_study" in captured.out

    def test_ensure_setup_is_silent(self, capsys):
        """Verify auto-setup via _ensure_setup does not print the summary."""
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner._ensure_setup()

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_setup_summary_custom_cv_object(self, capsys):
        """Verify custom CV object displays class name."""
        from sklearn.model_selection import StratifiedKFold
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner.setup(cv_strategy=StratifiedKFold(n_splits=3))

        captured = capsys.readouterr()
        assert "StratifiedKFold" in captured.out

    def test_setup_summary_callable_scoring(self, capsys):
        """Verify callable scoring displays function name."""
        from mlsurv.learner import SurvivalLearner

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        events = np.random.binomial(1, 0.7, 100).astype(bool)
        times = np.random.exponential(2, 100).clip(0.1, 10)
        y = make_survival_array(events, times)

        def my_custom_scorer(y_true, y_pred):
            pass

        learner = SurvivalLearner(X, y)
        learner.setup(scoring=my_custom_scorer)

        captured = capsys.readouterr()
        assert "scoring=my_custom_scorer" in captured.out

    def test_setup_summary_ipcw_note(self, capsys):
        """Verify Setup box shows IPCW note when test times exceed training max."""
        from mlsurv.learner import SurvivalLearner

        rng = np.random.RandomState(42)
        n = 15  # Enough for min_samples + min_events checks
        X_train = pd.DataFrame(rng.randn(n, 3), columns=['f0', 'f1', 'f2'])
        X_test = pd.DataFrame(rng.randn(n, 3), columns=['f0', 'f1', 'f2'])
        # Train times in [1, 10); test has 2 observations at 15.0 and 20.0
        events_train = np.array([True, False] * 7 + [True])
        times_train = rng.uniform(1, 10, n)
        y_train = make_survival_array(events_train, times_train)
        events_test = np.array([True, False] * 7 + [True])
        times_test = np.concatenate([rng.uniform(1, 9, n - 2), [15.0, 20.0]])
        y_test = make_survival_array(events_test, times_test)

        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(random_state=42)

        captured = capsys.readouterr()
        assert "IPCW note" in captured.out
        assert "2 y_test obs" in captured.out


class TestCoxSentinelConsistency:
    """Test that Cox failure sentinels are consistent across modules."""

    def test_default_cox_result_uses_failed_pval(self):
        """_default_full_result uses FAILED_PVAL (inf), not 1.0."""
        from mlsurv.preprocessing._cox_fit import (
            _default_full_result as _default_cox_result,
            FAILED_PVAL,
        )

        result = _default_cox_result("dummy")
        assert result['p_value'] == FAILED_PVAL
        assert result['p_value'] == float('inf')
        assert np.isnan(result['hazard_ratio'])
        assert np.isnan(result['coefficient'])

    def test_compute_univariate_cox_constant_feature_returns_inf(self):
        """compute_univariate_cox assigns inf p-value to constant features."""
        from mlsurv.evaluation.importance import compute_univariate_cox

        X = pd.DataFrame({
            'good': np.random.randn(50),
            'constant': np.ones(50),
        })
        y = np.array(
            [(True, float(i)) for i in range(1, 26)]
            + [(False, float(i)) for i in range(26, 51)],
            dtype=[('event', '?'), ('time', '<f8')],
        )
        result = compute_univariate_cox(X, y)
        const_row = result[result['feature'] == 'constant'].iloc[0]
        assert const_row['p_value'] == float('inf')
        assert np.isnan(const_row['hazard_ratio'])


class TestSkippedVsFailedWarnings:
    """Test that _sort_features distinguishes skipped from failed features."""

    @staticmethod
    def _make_data(n=50):
        """Create minimal survival data with one good and one constant feature."""
        np.random.seed(42)
        X = pd.DataFrame({
            'good': np.random.randn(n),
            'constant': np.ones(n),
        })
        y = np.array(
            [(True, float(i)) for i in range(1, n // 2 + 1)]
            + [(False, float(i)) for i in range(n // 2 + 1, n + 1)],
            dtype=[('event', '?'), ('time', '<f8')],
        )
        return X, y

    def test_sort_features_warning_skipped_only(self):
        """Constant feature triggers warning with 'skipped', not 'failed'."""
        from mlsurv.preprocessing.selection import _sort_features

        X, y = self._make_data()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _sort_features(X, y)

        skip_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "skipped" in str(w.message).lower()
        ]
        assert len(skip_warnings) == 1
        assert "constant/near-zero variance" in str(skip_warnings[0].message)
        # Should NOT say "failed" because no fit was attempted
        assert "failed" not in str(skip_warnings[0].message).lower()

        # Result still returns (col, pval) pairs
        assert len(result) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    def test_sort_features_warning_failed_only(self):
        """When one feature fails Cox fitting, warning says 'failed', not 'skipped'."""
        from unittest.mock import patch
        from mlsurv.preprocessing.selection import _sort_features

        np.random.seed(42)
        # Use non-constant features so the pre-check doesn't skip them
        X = pd.DataFrame({
            'good': np.random.randn(50),
            'bad': np.random.randn(50),
        })
        y = np.array(
            [(True, float(i)) for i in range(1, 26)]
            + [(False, float(i)) for i in range(26, 51)],
            dtype=[('event', '?'), ('time', '<f8')],
        )

        from mlsurv.preprocessing import _cox_fit as _cox_mod
        real_try = _cox_mod.try_cox_fit

        def _fail_bad_only(df, col_name, **kwargs):
            if col_name == 'bad':
                return None
            return real_try(df, col_name, **kwargs)

        with patch('mlsurv.preprocessing._cox_fit.try_cox_fit',
                   side_effect=_fail_bad_only):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _sort_features(X, y)

        user_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
        ]
        assert len(user_warnings) == 1
        msg = str(user_warnings[0].message)
        assert "failed cox fitting" in msg.lower()
        assert "skipped" not in msg.lower()

    def test_sort_features_warning_mixed(self):
        """Mix of skipped and failed features mentions both in warning."""
        from unittest.mock import patch
        from mlsurv.preprocessing.selection import _sort_features

        np.random.seed(42)
        X = pd.DataFrame({
            'good': np.random.randn(50),
            'constant': np.ones(50),        # will be skipped
            'bad': np.random.randn(50),     # will be failed via mock
        })
        y = np.array(
            [(True, float(i)) for i in range(1, 26)]
            + [(False, float(i)) for i in range(26, 51)],
            dtype=[('event', '?'), ('time', '<f8')],
        )

        # Capture the real function BEFORE patching
        from mlsurv.preprocessing import _cox_fit as _cox_mod
        real_try_cox_fit = _cox_mod.try_cox_fit

        def _selective_mock(df, col_name, **kwargs):
            """Return None only for 'bad', let others through."""
            if col_name == 'bad':
                return None
            return real_try_cox_fit(df, col_name, **kwargs)

        with patch('mlsurv.preprocessing._cox_fit.try_cox_fit', side_effect=_selective_mock):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _sort_features(X, y)

        user_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
        ]
        assert len(user_warnings) >= 1
        msg = str(user_warnings[0].message)
        assert "skipped" in msg.lower()
        assert "failed" in msg.lower()

    def test_fit_univariate_cox_returns_dict(self):
        """fit_univariate_cox returns dict with feature/p_value/status/error_detail."""
        from mlsurv.preprocessing._cox_fit import fit_univariate_cox, FAILED_PVAL

        np.random.seed(42)
        n = 50
        y = np.array(
            [(True, float(i)) for i in range(1, n // 2 + 1)]
            + [(False, float(i)) for i in range(n // 2 + 1, n + 1)],
            dtype=[('event', '?'), ('time', '<f8')],
        )

        # Good feature
        good = pd.Series(np.random.randn(n))
        result = fit_univariate_cox(good, y, 'good')
        assert result['status'] == 'ok'
        assert np.isfinite(result['p_value'])
        assert result['error_detail'] is None

        # Constant feature (skipped)
        const = pd.Series(np.ones(n))
        result = fit_univariate_cox(const, y, 'constant')
        assert result['status'] == 'skipped'
        assert result['p_value'] == FAILED_PVAL
        assert result['error_detail'] is None

    def test_fit_univariate_cox_error_status(self):
        """fit_univariate_cox returns 'error' status with error_detail string on unexpected failure."""
        from unittest.mock import patch
        from mlsurv.preprocessing._cox_fit import fit_univariate_cox

        np.random.seed(42)
        n = 50
        y = np.array(
            [(True, float(i)) for i in range(1, n // 2 + 1)]
            + [(False, float(i)) for i in range(n // 2 + 1, n + 1)],
            dtype=[('event', '?'), ('time', '<f8')],
        )
        feature = pd.Series(np.random.randn(n))

        # Patch try_cox_fit to return an error string (unexpected error path)
        with patch('mlsurv.preprocessing._cox_fit.try_cox_fit') as mock_try:
            mock_try.return_value = "Unexpected error for 'x': RuntimeError: boom"
            result = fit_univariate_cox(feature, y, 'x')

        assert result['status'] == 'error'
        assert isinstance(result['error_detail'], str)
        assert 'boom' in result['error_detail']

    def test_warn_cox_errors_pipeline(self):
        """Error strings from try_cox_fit are surfaced as warnings via _warn_cox_errors."""
        from unittest.mock import patch
        from mlsurv.preprocessing.selection import _sort_features

        np.random.seed(42)
        n = 80
        y = np.array(
            [(True, float(i)) for i in range(1, n // 2 + 1)]
            + [(False, float(i)) for i in range(n // 2 + 1, n + 1)],
            dtype=[('event', '?'), ('time', '<f8')],
        )
        X = pd.DataFrame({
            'good': np.random.randn(n),
            'bad': np.random.randn(n),
        })

        def mock_try(*args, **kwargs):
            # args: (df, col_name, ...) or keyword col_name
            col = args[1] if len(args) > 1 else kwargs.get('col_name', '')
            if col == 'bad':
                return "Unexpected error for 'bad': RuntimeError: kaboom"
            return {'p_value': 0.05}  # Successful pval-mode fit for 'good'

        with patch('mlsurv.preprocessing._cox_fit.try_cox_fit', side_effect=mock_try):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _sort_features(X, y)

        error_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning) and 'unexpected' in str(w.message).lower()
        ]
        assert len(error_warnings) >= 1
        assert 'kaboom' in str(error_warnings[0].message)


class TestUnivCoxNaN:
    """Tests for NaN handling in univariate Cox."""

    @staticmethod
    def _make_survival_y(n):
        return np.array(
            [(True, float(i)) for i in range(1, n // 2 + 1)]
            + [(False, float(i)) for i in range(n // 2 + 1, n + 1)],
            dtype=[('event', '?'), ('time', '<f8')],
        )

    def test_nan_column_no_unexpected_error_warning(self):
        """Columns with NaN don't produce 'Unexpected error' warnings."""
        from mlsurv.evaluation.importance import compute_univariate_cox

        np.random.seed(42)
        n = 60
        X = pd.DataFrame({
            'good': np.random.randn(n),
            'has_nan': np.random.randn(n),
        })
        X.loc[0:9, 'has_nan'] = np.nan
        y = self._make_survival_y(n)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_univariate_cox(X, y)

        unexpected = [
            w for w in caught
            if 'unexpected' in str(w.message).lower()
        ]
        assert len(unexpected) == 0
        # Both features should have results
        assert len(result) == 2

    def test_all_nan_column_returns_default_with_n_missing(self):
        """An all-NaN column returns default result with n_missing metadata."""
        from mlsurv.evaluation.importance import compute_univariate_cox

        np.random.seed(42)
        n = 50
        X = pd.DataFrame({
            'good': np.random.randn(n),
            'all_nan': np.full(n, np.nan),
        })
        y = self._make_survival_y(n)

        result = compute_univariate_cox(X, y)
        nan_row = result[result['feature'] == 'all_nan'].iloc[0]
        assert nan_row['p_value'] == float('inf')
        assert np.isnan(nan_row['hazard_ratio'])
        assert nan_row['n_missing'] == n

    def test_typeerror_caught_by_try_cox_fit(self):
        """TypeError from lifelines is caught as expected exception, not 'Unexpected error'."""
        from unittest.mock import patch
        from mlsurv.preprocessing._cox_fit import try_cox_fit

        df = pd.DataFrame({
            'T': [1.0, 2.0, 3.0],
            'E': [True, False, True],
            'x': [0.1, 0.2, 0.3],
        })

        with patch('mlsurv.preprocessing._cox_fit.CoxPHFitter') as MockCPH:
            MockCPH.return_value.fit.side_effect = TypeError("NaN in data")
            result = try_cox_fit(df, 'x', extract='full')

        # TypeError should be caught → returns None (expected failure),
        # NOT a string (unexpected error)
        assert result is None

    def test_convergence_warning_elevated_in_pval_mode(self):
        """ConvergenceWarning aborts the fit in extract='pval' (selection retry path)
        but is accepted in extract='full' (importance descriptive path)."""
        from unittest.mock import patch, MagicMock
        from lifelines.exceptions import ConvergenceWarning
        from mlsurv.preprocessing._cox_fit import try_cox_fit

        df = pd.DataFrame({
            'T': [1.0, 2.0, 3.0, 4.0],
            'E': [True, False, True, False],
            'x': [0.1, 0.2, 0.3, 0.4],
        })

        def fit_with_warning(self, *args, **kwargs):
            warnings.warn("did not converge", ConvergenceWarning)
            self.summary = pd.DataFrame({
                'p': [0.05], 'coef': [0.5],
                'exp(coef)': [1.65],
                'exp(coef) lower 95%': [0.9],
                'exp(coef) upper 95%': [3.0],
            }, index=['x'])

        with patch('mlsurv.preprocessing._cox_fit.CoxPHFitter') as MockCPH:
            instance = MagicMock()
            instance.fit.side_effect = lambda *a, **kw: fit_with_warning(instance, *a, **kw)
            MockCPH.return_value = instance
            pval_result = try_cox_fit(df, 'x', extract='pval')
            full_result = try_cox_fit(df, 'x', extract='full')

        # pval mode elevates ConvergenceWarning → expected failure → None
        assert pval_result is None
        # full mode accepts it → success dict
        assert isinstance(full_result, dict)
        assert full_result['p_value'] == 0.05


class TestCasewiseDeletion:
    """Test per-feature casewise deletion in univariate Cox fitting."""

    def test_topk_handles_nan(self, synthetic_data):
        """TopKCoxSelector produces valid rankings when X has NaN."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data  # has NaN in cols 0 and 5
        selector = TopKCoxSelector(k=5)
        selector.fit(X, y)

        assert len(selector.selected_features_) == 5
        assert len(selector.sorted_features_) == X.shape[1]
        # All p-values should be finite or inf (not NaN)
        for _name, pval in selector.sorted_features_:
            assert not np.isnan(pval)

    def test_aggregate_casewise_deletion_warning(self, synthetic_data):
        """Aggregate MethodologicalWarning emitted when features have NaN."""
        from mlsurv.preprocessing import TopKCoxSelector

        X, y = synthetic_data  # has NaN in cols 0 and 5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selector = TopKCoxSelector(k=5)
            selector.fit(X, y)

        meth_warnings = [x for x in w if issubclass(x.category, MethodologicalWarning)]
        casewise_warns = [x for x in meth_warnings
                          if 'Casewise deletion' in str(x.message)]
        assert len(casewise_warns) == 1
        msg = str(casewise_warns[0].message)
        assert 'features had NaN rows dropped' in msg

    def test_univariate_cox_drops_nan_rows(self):
        """fit_univariate_cox applies casewise deletion for NaN."""
        from mlsurv.preprocessing._cox_fit import fit_univariate_cox

        np.random.seed(42)
        n = 100
        feature = pd.Series(np.random.randn(n))
        # Inject NaN in 20% of rows
        feature.iloc[:20] = np.nan

        y = make_y(np.random.exponential(5, n))

        result = fit_univariate_cox(feature, y, 'test_feat')
        assert result['status'] == 'ok'
        assert 0 < result['p_value'] < float('inf')

    def test_all_nan_feature_skipped(self):
        """Feature with all NaN values is skipped."""
        from mlsurv.preprocessing._cox_fit import fit_univariate_cox, FAILED_PVAL

        feature = pd.Series([np.nan] * 50)
        y = make_y(np.random.exponential(5, 50))

        result = fit_univariate_cox(feature, y, 'all_nan')
        assert result['p_value'] == FAILED_PVAL
        assert result['status'] == 'skipped'

    def test_low_nonnan_rows_warning(self):
        """Features with <10 non-NaN rows emit aggregate MethodologicalWarning."""
        from mlsurv.preprocessing import TopKCoxSelector

        np.random.seed(42)
        n = 50
        X = pd.DataFrame({
            'good': np.random.randn(n),
            'few_rows': [np.nan] * (n - 5) + list(np.random.randn(5)),
        })
        y = make_y(np.random.exponential(5, n))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selector = TopKCoxSelector(k=2)
            selector.fit(X, y)

        meth_warnings = [x for x in w if issubclass(x.category, MethodologicalWarning)]
        # Aggregate warning about casewise deletion with low-row features
        casewise_warns = [x for x in meth_warnings
                          if 'fewer than' in str(x.message) and 'few_rows' in str(x.message)]
        assert len(casewise_warns) >= 1

    def test_casewise_vs_imputed_rankings_differ(self, synthetic_data):
        """Rankings on raw data (casewise) differ from imputed data rankings."""
        from mlsurv.preprocessing import TopKCoxSelector
        from sklearn.impute import SimpleImputer

        X, y = synthetic_data
        # Make NaN more impactful: 30% NaN in feature_0
        X_heavy_nan = X.copy()
        np.random.seed(99)
        nan_mask = np.random.rand(len(X)) < 0.3
        X_heavy_nan.iloc[nan_mask, 0] = np.nan

        # Fit on raw data (casewise deletion)
        sel_raw = TopKCoxSelector(k=X.shape[1])
        sel_raw.fit(X_heavy_nan, y)
        raw_rankings = {name: pval for name, pval in sel_raw.sorted_features_}

        # Fit on imputed data
        imp = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imp.fit_transform(X_heavy_nan), columns=X_heavy_nan.columns
        )
        sel_imp = TopKCoxSelector(k=X.shape[1])
        sel_imp.fit(X_imputed, y)
        imp_rankings = {name: pval for name, pval in sel_imp.sorted_features_}

        # p-values for feature_0 should differ (different sample sizes)
        assert raw_rankings['feature_0'] != imp_rankings['feature_0']
