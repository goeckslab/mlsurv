"""Unit tests for _filter_numeric_for_corr."""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import InternalWarning
from mlsurv.viz._features_helpers import _filter_numeric_for_corr


class TestFilterNumericForCorr:
    """Tests for _filter_numeric_for_corr."""

    def test_bool_columns_included(self):
        """Bool columns from pd.get_dummies() are kept (pandas >= 2.0)."""
        X = pd.get_dummies(
            pd.DataFrame({"cat": ["A", "B", "C"]}), columns=["cat"]
        )
        assert X.dtypes.apply(lambda d: d == np.dtype("bool")).all()
        result = _filter_numeric_for_corr(X)
        pd.testing.assert_frame_equal(result, X)

    def test_mixed_numeric_and_bool(self):
        """Both float and bool columns are retained."""
        X = pd.DataFrame({
            "age": [30.0, 40.0, 50.0],
            "flag": [True, False, True],
        })
        result = _filter_numeric_for_corr(X)
        assert list(result.columns) == ["age", "flag"]

    def test_non_numeric_dropped_with_warning(self):
        """String columns are dropped and InternalWarning is emitted."""
        X = pd.DataFrame({
            "age": [30.0, 40.0],
            "name": ["Alice", "Bob"],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _filter_numeric_for_corr(X)
        assert list(result.columns) == ["age"]
        assert len(w) == 1
        assert issubclass(w[0].category, InternalWarning)
        assert "1 non-numeric" in str(w[0].message)

    def test_all_non_numeric_raises(self):
        """ValueError raised when no numeric columns exist."""
        X = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
        with pytest.raises(ValueError, match="No numeric columns"):
            _filter_numeric_for_corr(X)

    def test_bool_not_counted_as_dropped(self):
        """Bool columns are not included in the non-numeric dropped count."""
        X = pd.DataFrame({
            "flag": [True, False, True],
            "name": ["A", "B", "C"],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _filter_numeric_for_corr(X)
        assert list(result.columns) == ["flag"]
        assert len(w) == 1
        assert "1 non-numeric" in str(w[0].message)

    def test_pure_numeric_no_warning(self):
        """All-float DataFrame produces no warning."""
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _filter_numeric_for_corr(X)
        assert len(w) == 0
        pd.testing.assert_frame_equal(result, X)
