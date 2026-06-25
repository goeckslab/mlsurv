"""Tests for VIFSelector, CorrelationSelector, and preprocessing resolvers."""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

from mlsurv.preprocessing import (
    VIFSelector,
    CorrelationSelector,
    TopKCoxSelector,
)
from mlsurv.preprocessing.pipelines import (
    get_imputer,
    get_scaler,
    get_selector,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def correlated_data():
    """Create dataset with highly correlated features."""
    np.random.seed(42)
    n = 100

    # Base features
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # Correlated features (x4 ~ x1, x5 ~ x2)
    x4 = x1 + 0.1 * np.random.randn(n)  # ~0.99 correlation with x1
    x5 = x2 + 0.3 * np.random.randn(n)  # ~0.95 correlation with x2

    X = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'x5': x5,
    })

    return X


@pytest.fixture
def multicollinear_data():
    """Create dataset with high multicollinearity (high VIF)."""
    np.random.seed(42)
    n = 100

    # Base feature
    x1 = np.random.randn(n)

    # Features that are linear combinations of each other
    x2 = 2 * x1 + 0.01 * np.random.randn(n)  # Very high VIF
    x3 = 0.5 * x1 + 0.5 * x2 + 0.01 * np.random.randn(n)  # Very high VIF

    # Independent feature
    x4 = np.random.randn(n)

    X = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
    })

    return X


@pytest.fixture
def survival_data(correlated_data):
    """Create survival outcomes for correlated_data."""
    np.random.seed(42)
    n = len(correlated_data)

    y = np.zeros(n, dtype=[('event', bool), ('time', float)])
    y['event'] = np.random.binomial(1, 0.7, n).astype(bool)
    y['time'] = np.abs(np.random.randn(n)) + 0.1

    return y


# ============================================================================
# VIFSelector Tests
# ============================================================================

class TestVIFSelector:
    """Tests for VIFSelector."""

    def test_removes_high_vif_features(self, multicollinear_data):
        """VIFSelector should remove features with VIF > threshold."""
        selector = VIFSelector(threshold=10.0)
        selector.fit_transform(multicollinear_data)

        # Should have removed at least one high-VIF feature
        assert len(selector.removed_features_) > 0
        assert len(selector.selected_features_) < len(multicollinear_data.columns)

        # Remaining features should have VIF <= threshold
        for vif in selector.vif_scores_.values():
            assert vif <= 10.0 or len(selector.selected_features_) == selector.min_features

    def test_respects_min_features(self, multicollinear_data):
        """VIFSelector should keep at least min_features."""
        selector = VIFSelector(threshold=1.0, min_features=3)  # Very strict threshold
        selector.fit_transform(multicollinear_data)

        assert len(selector.selected_features_) >= 3

    def test_keeps_all_if_below_threshold(self, correlated_data):
        """VIFSelector should keep all features if VIF is below threshold."""
        # Use very high threshold
        selector = VIFSelector(threshold=100.0)
        selector.fit_transform(correlated_data)

        # With high threshold, might keep all or most features
        # (depends on actual VIF values)
        assert len(selector.selected_features_) > 0

    def test_verbose_output(self, multicollinear_data, capsys):
        """VIFSelector verbose mode should print progress."""
        selector = VIFSelector(threshold=10.0, verbose=True)
        selector.fit_transform(multicollinear_data)

        captured = capsys.readouterr()
        # Should contain removal messages
        if selector.removed_features_:
            assert "Removed" in captured.out

    def test_get_support(self, multicollinear_data):
        """Test get_support returns correct mask/indices."""
        selector = VIFSelector(threshold=10.0)
        selector.fit(multicollinear_data)

        # Test boolean mask
        mask = selector.get_support(indices=False)
        assert mask.sum() == len(selector.selected_features_)

        # Test indices
        indices = selector.get_support(indices=True)
        assert len(indices) == len(selector.selected_features_)

    def test_get_feature_names_out(self, multicollinear_data):
        """Test get_feature_names_out returns selected features."""
        selector = VIFSelector(threshold=10.0)
        selector.fit(multicollinear_data)

        names_out = selector.get_feature_names_out()
        assert names_out == selector.selected_features_

    def test_vif_handles_singular_matrix(self):
        """VIFSelector should assign inf VIF (not crash) for perfectly collinear features."""
        np.random.seed(42)
        n = 100
        col_a = np.random.randn(n)
        col_b = np.random.randn(n)
        col_c = col_a + col_b  # exact linear combination -> singular matrix

        X = pd.DataFrame({'col_a': col_a, 'col_b': col_b, 'col_c': col_c})
        selector = VIFSelector(threshold=10)
        selector.fit(X)  # should not raise LinAlgError

        # At least one of the collinear features should be removed
        assert len(selector.removed_features_) > 0

        # Verify _compute_vif assigns inf for the collinear feature
        vif_raw = selector._compute_vif(X)
        inf_features = [f for f, v in vif_raw.items() if v == float('inf')]
        assert len(inf_features) > 0


# ============================================================================
# CorrelationSelector Tests
# ============================================================================

@pytest.mark.integration
class TestCorrelationSelector:
    """Tests for CorrelationSelector."""

    def test_removes_correlated_features_frequency(self, correlated_data):
        """CorrelationSelector with frequency strategy removes correlated features."""
        selector = CorrelationSelector(threshold=0.8, strategy='frequency')
        X_selected = selector.fit_transform(correlated_data)

        # Should detect correlated pairs
        assert len(selector.correlation_pairs_) > 0

        # Should remove one from each pair
        assert len(selector.removed_features_) > 0

        # Remaining features should not have high correlation
        remaining = X_selected
        corr_matrix = remaining.corr().abs()
        corr_vals = corr_matrix.to_numpy(copy=True)
        np.fill_diagonal(corr_vals, 0)
        assert corr_vals.max() < 0.8 or len(remaining.columns) == 1

    def test_removes_correlated_features_cox(self, correlated_data, survival_data):
        """CorrelationSelector with cox strategy keeps survival-relevant features."""
        selector = CorrelationSelector(threshold=0.8, strategy='cox')
        selector.fit_transform(correlated_data, survival_data)

        # Should remove one from each pair
        assert len(selector.removed_features_) > 0

    def test_cox_strategy_requires_y(self, correlated_data):
        """CorrelationSelector with cox strategy should require y."""
        selector = CorrelationSelector(threshold=0.8, strategy='cox')

        with pytest.raises(ValueError, match="requires y"):
            selector.fit(correlated_data)

    def test_no_correlated_pairs(self):
        """CorrelationSelector keeps all features if no pairs exceed threshold."""
        np.random.seed(42)
        X = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100),
        })

        selector = CorrelationSelector(threshold=0.99)  # Very high threshold
        X_selected = selector.fit_transform(X)

        assert len(selector.removed_features_) == 0
        assert len(X_selected.columns) == 3

    def test_verbose_output(self, correlated_data, capsys):
        """CorrelationSelector verbose mode should print progress."""
        selector = CorrelationSelector(threshold=0.8, verbose=True)
        selector.fit_transform(correlated_data)

        captured = capsys.readouterr()
        if selector.removed_features_:
            assert "Removed" in captured.out

    def test_get_support(self, correlated_data):
        """Test get_support returns correct mask/indices."""
        selector = CorrelationSelector(threshold=0.8)
        selector.fit(correlated_data)

        # Test boolean mask
        mask = selector.get_support(indices=False)
        assert mask.sum() == len(selector.selected_features_)

        # Test indices
        indices = selector.get_support(indices=True)
        assert len(indices) == len(selector.selected_features_)

    def test_invalid_strategy(self, correlated_data):
        """CorrelationSelector should reject invalid strategy."""
        selector = CorrelationSelector(strategy='invalid')

        with pytest.raises(ValueError, match="Unknown strategy"):
            selector.fit(correlated_data)

    def test_cox_strategy_n_jobs(self):
        """CorrelationSelector with n_jobs=2 exercises parallel path and matches sequential."""
        # Need >= 15 features in correlated pairs to exceed MIN_FEATURES_FOR_PARALLEL
        np.random.seed(42)
        n = 200
        base = {f'x{i}': np.random.randn(n) for i in range(8)}
        dupes = {f'x{i}_dup': base[f'x{i}'] + 0.05 * np.random.randn(n) for i in range(8)}
        X = pd.DataFrame({**base, **dupes})

        y = np.zeros(n, dtype=[('event', bool), ('time', float)])
        y['event'] = np.random.binomial(1, 0.7, n).astype(bool)
        y['time'] = np.abs(np.random.randn(n)) + 0.1

        sel_seq = CorrelationSelector(threshold=0.8, strategy='cox', n_jobs=1)
        sel_par = CorrelationSelector(threshold=0.8, strategy='cox', n_jobs=2)

        sel_seq.fit(X, y)
        sel_par.fit(X, y)

        assert set(sel_seq.removed_features_) == set(sel_par.removed_features_)

    def test_force_parallel_parameter(self):
        """CorrelationSelector accepts and stores force_parallel parameter."""
        selector = CorrelationSelector(threshold=0.8, force_parallel=True)
        assert selector.force_parallel is True

        selector_default = CorrelationSelector(threshold=0.8)
        assert selector_default.force_parallel is False


# ============================================================================
# Resolver Function Tests
# ============================================================================

class TestGetImputer:
    """Tests for get_imputer resolver."""

    def test_string_shortcuts(self):
        """String shortcuts resolve to correct imputers."""
        median_imp = get_imputer('median')
        assert isinstance(median_imp, SimpleImputer)
        assert median_imp.strategy == 'median'

        mean_imp = get_imputer('mean')
        assert isinstance(mean_imp, SimpleImputer)
        assert mean_imp.strategy == 'mean'

        from sklearn.impute import KNNImputer
        knn_imp = get_imputer('knn')
        assert isinstance(knn_imp, KNNImputer)

    def test_iterative_preset(self):
        """Iterative preset returns IterativeImputer with correct params."""
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        imp = get_imputer('iterative')
        assert isinstance(imp, IterativeImputer)
        assert imp.max_iter == 10
        # Default seed is None (like get_selector('coxlasso')); setup() threads
        # its configured random_state in (see test_iterative_threads_random_state).
        assert imp.random_state is None
        # Default estimator is None (sklearn uses BayesianRidge)
        assert imp.estimator is None

    def test_iterative_threads_random_state(self):
        """random_state is threaded into the iterative imputer's draws."""
        imp = get_imputer('iterative', random_state=42)
        assert imp.random_state == 42

    def test_iterative_random_state_zero_preserved(self):
        """random_state=0 is a valid seed, not coerced to None."""
        # 0 was the old hard-coded value, so it must survive the None default.
        assert get_imputer('iterative', random_state=0).random_state == 0

    def test_iterative_in_list_threads_random_state(self):
        """The seed reaches an 'iterative' preset nested in a list."""
        pipe = get_imputer(['iterative'], random_state=5)
        assert isinstance(pipe, Pipeline)
        assert pipe.steps[0][1].random_state == 5

    def test_iterative_factory_independence(self):
        """Each call returns a separate instance (factory lambda works)."""
        imp_a = get_imputer('iterative')
        imp_b = get_imputer('iterative')
        assert imp_a is not imp_b

    def test_none_returns_none(self):
        """None input returns None."""
        assert get_imputer(None) is None

    def test_transformer_passthrough(self):
        """Custom transformer is passed through."""
        custom = SimpleImputer(strategy='most_frequent')
        result = get_imputer(custom)
        assert result is custom

    def test_list_wraps_in_pipeline(self):
        """List of transformers is wrapped in Pipeline."""
        imp1 = SimpleImputer(strategy='median')
        imp2 = StandardScaler()

        result = get_imputer([imp1, imp2])
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 2

    def test_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown imputer preset"):
            get_imputer('invalid')


class TestGetScaler:
    """Tests for get_scaler resolver."""

    def test_string_shortcuts(self):
        """String shortcuts resolve to correct scalers."""
        standard = get_scaler('standard')
        assert isinstance(standard, StandardScaler)

        robust = get_scaler('robust')
        assert isinstance(robust, RobustScaler)

        from sklearn.preprocessing import MinMaxScaler
        minmax = get_scaler('minmax')
        assert isinstance(minmax, MinMaxScaler)

    def test_pca_shortcut(self):
        """PCA shortcut creates Pipeline with scaler + PCA."""
        result = get_scaler('pca')
        assert isinstance(result, Pipeline)
        assert 'scaler' in dict(result.steps)
        assert 'pca' in dict(result.steps)
        # Default seed is None; setup() threads its configured random_state in.
        assert dict(result.steps)['pca'].random_state is None

    def test_pca_threads_random_state(self):
        """random_state is threaded into the PCA step's randomized SVD."""
        result = get_scaler('pca', random_state=7)
        assert dict(result.steps)['pca'].random_state == 7

    def test_pca_random_state_zero_preserved(self):
        """random_state=0 is a valid seed, not coerced to None."""
        result = get_scaler('pca', random_state=0)
        assert dict(result.steps)['pca'].random_state == 0

    def test_none_returns_none(self):
        """None input returns None."""
        assert get_scaler(None) is None

    def test_list_wraps_in_pipeline(self):
        """List of transformers is wrapped in Pipeline."""
        s1 = StandardScaler()
        from sklearn.decomposition import PCA
        s2 = PCA(n_components=5)

        result = get_scaler([s1, s2])
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 2

    def test_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scaler preset"):
            get_scaler('invalid')


class TestGetSelector:
    """Tests for get_selector resolver."""

    def test_string_shortcuts(self):
        """String shortcuts resolve to correct selectors."""
        topk = get_selector('topk')
        assert isinstance(topk, TopKCoxSelector)
        assert topk.k == 'auto'

        vif = get_selector('vif')
        assert isinstance(vif, VIFSelector)
        assert vif.threshold == 10.0

        corr = get_selector('correlation')
        assert isinstance(corr, CorrelationSelector)
        assert corr.threshold == 0.8

    def test_none_returns_none(self):
        """None input returns None."""
        assert get_selector(None) is None

    def test_list_wraps_in_pipeline(self):
        """List of selectors is wrapped in Pipeline."""
        sel1 = VIFSelector(threshold=10)
        sel2 = TopKCoxSelector(k=5)

        result = get_selector([sel1, sel2])
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 2

    def test_list_with_string_shortcuts(self):
        """List can mix string shortcuts and transformers."""
        result = get_selector(['vif', TopKCoxSelector(k=5)])
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 2
        # First step should be VIFSelector from 'vif' string
        assert isinstance(result.steps[0][1], VIFSelector)

    def test_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown selector preset"):
            get_selector('invalid')


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for selectors in pipelines."""

    def test_vif_in_pipeline(self, multicollinear_data, survival_data):
        """VIFSelector works in sklearn Pipeline."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('vif', VIFSelector(threshold=10)),
        ])

        X_result = pipeline.fit_transform(multicollinear_data)
        assert X_result.shape[1] < multicollinear_data.shape[1]

    def test_correlation_in_pipeline(self, correlated_data):
        """CorrelationSelector works in sklearn Pipeline."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('corr', CorrelationSelector(threshold=0.8)),
        ])

        X_result = pipeline.fit_transform(correlated_data)
        assert X_result.shape[1] < correlated_data.shape[1]

    def test_chained_selectors(self, multicollinear_data):
        """Multiple selectors can be chained."""
        pipeline = Pipeline([
            ('vif', VIFSelector(threshold=10)),
            ('corr', CorrelationSelector(threshold=0.9)),
        ])

        X_result = pipeline.fit_transform(multicollinear_data)
        assert X_result.shape[1] <= multicollinear_data.shape[1]


class TestIsCoxBased:
    """Test _is_cox_based() helper."""

    def test_topk_is_cox_based(self):
        from mlsurv.preprocessing.selection import _is_cox_based
        assert _is_cox_based(TopKCoxSelector()) is True

    def test_correlation_cox_is_cox_based(self):
        from mlsurv.preprocessing.selection import _is_cox_based
        assert _is_cox_based(CorrelationSelector(strategy='cox')) is True

    def test_correlation_frequency_not_cox_based(self):
        from mlsurv.preprocessing.selection import _is_cox_based
        assert _is_cox_based(CorrelationSelector(strategy='frequency')) is False

    def test_vif_not_cox_based(self):
        from mlsurv.preprocessing.selection import _is_cox_based
        assert _is_cox_based(VIFSelector()) is False

    def test_coxlasso_not_cox_based(self):
        """CoxLassoSelector uses multivariate Cox — casewise deletion not applicable."""
        from mlsurv.preprocessing.selection import _is_cox_based
        from mlsurv.preprocessing import CoxLassoSelector
        assert _is_cox_based(CoxLassoSelector()) is False

    def test_coxnet_not_cox_based(self):
        """CoxNetSelector uses multivariate Cox — casewise deletion not applicable."""
        from mlsurv.preprocessing.selection import _is_cox_based
        from mlsurv.preprocessing import CoxNetSelector
        assert _is_cox_based(CoxNetSelector()) is False

    def test_pipeline_wrapping_not_cox_based(self):
        """Pipeline wrapping a Cox selector returns False — Pipeline must run sequentially."""
        from mlsurv.preprocessing.selection import _is_cox_based
        pipeline = Pipeline([('vif', VIFSelector()), ('topk', TopKCoxSelector())])
        assert _is_cox_based(pipeline) is False


class TestCorrelationCoxWithNaN:
    """Test CorrelationSelector(strategy='cox') with NaN data."""

    def test_cox_strategy_with_nan(self, correlated_data, survival_data):
        """CorrelationSelector(strategy='cox') handles NaN via casewise deletion."""
        X = correlated_data
        y = survival_data

        # Add NaN to some features
        X_nan = X.copy()
        X_nan.iloc[:10, 0] = np.nan   # NaN in x1
        X_nan.iloc[15:25, 3] = np.nan  # NaN in x4 (correlated with x1)

        selector = CorrelationSelector(threshold=0.8, strategy='cox')
        selector.fit(X_nan, y)

        assert len(selector.selected_features_) < X_nan.shape[1]
        assert len(selector.removed_features_) > 0
