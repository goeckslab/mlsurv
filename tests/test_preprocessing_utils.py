"""Tests for mlsurv.preprocessing.utils shared utility functions.

Covers get_default_imputer(), get_default_scaler(), extract_component_params(),
and get_or_create_selector().
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from mlsurv.preprocessing.selection import TopKCoxSelector
from mlsurv.preprocessing.utils import (
    get_default_imputer,
    get_default_scaler,
    extract_component_params,
    get_or_create_selector,
    DEFAULT_IMPUTER_STRATEGY,
)


class TestGetDefaultImputer:
    """Tests for get_default_imputer() factory function."""

    def test_returns_simple_imputer(self):
        """Should return a SimpleImputer instance."""
        imputer = get_default_imputer()
        assert isinstance(imputer, SimpleImputer)

    def test_median_strategy(self):
        """Should use median strategy."""
        imputer = get_default_imputer()
        assert imputer.strategy == DEFAULT_IMPUTER_STRATEGY

    def test_fresh_instance(self):
        """Each call should return a new instance."""
        imp1 = get_default_imputer()
        imp2 = get_default_imputer()
        assert imp1 is not imp2


class TestGetDefaultScaler:
    """Tests for get_default_scaler() factory function."""

    def test_returns_standard_scaler(self):
        """Should return a StandardScaler instance."""
        scaler = get_default_scaler()
        assert isinstance(scaler, StandardScaler)

    def test_fresh_instance(self):
        """Each call should return a new instance."""
        s1 = get_default_scaler()
        s2 = get_default_scaler()
        assert s1 is not s2


class TestExtractComponentParams:
    """Tests for extract_component_params() prefix filtering.

    Correct prefix extraction is critical for the tuning pipeline —
    wrong parameters get passed to wrong components silently.
    """

    def test_basic_extraction(self):
        params = {'model__lr': 0.1, 'model__dropout': 0.5}
        result = extract_component_params(params, 'model')
        assert result == {'lr': 0.1, 'dropout': 0.5}

    def test_filters_other_prefixes(self):
        """Should only extract matching prefix."""
        params = {'model__lr': 0.1, 'imputer__strategy': 'mean'}
        result = extract_component_params(params, 'model')
        assert result == {'lr': 0.1}
        assert 'strategy' not in result

    def test_empty_when_no_match(self):
        """Should return empty dict when no params match."""
        params = {'imputer__strategy': 'mean'}
        result = extract_component_params(params, 'model')
        assert result == {}

    def test_empty_input(self):
        result = extract_component_params({}, 'model')
        assert result == {}

    def test_nested_prefix_preserved(self):
        """str.replace bug: model__model__x should become model__x, not x."""
        params = {'model__model__n_estimators': 100, 'model__lr': 0.01}
        result = extract_component_params(params, 'model')
        assert result == {'model__n_estimators': 100, 'lr': 0.01}


class TestGetOrCreateSelector:
    """Tests for get_or_create_selector() selector creation/configuration."""

    def test_none_and_none(self):
        """Both None should return None."""
        assert get_or_create_selector(None, k=None) is None

    def test_none_with_k_creates_selector(self):
        """None selector with k should create TopKCoxSelector."""
        selector = get_or_create_selector(None, k=10)
        assert isinstance(selector, TopKCoxSelector)
        assert selector.k == 10

    def test_existing_selector_with_k_clones(self):
        """Existing selector with k should clone and update k."""
        original = TopKCoxSelector(k=5)
        updated = get_or_create_selector(original, k=15)
        assert updated.k == 15
        assert original.k == 5  # Original unchanged

    def test_existing_selector_without_k(self):
        """Existing selector without k should return as-is."""
        original = TopKCoxSelector(k=5)
        result = get_or_create_selector(original, k=None)
        assert result is original
