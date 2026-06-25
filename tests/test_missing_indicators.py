"""Tests for add_missing_indicators."""

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.preprocessing import add_missing_indicators


@pytest.fixture
def df():
    """DataFrame with some missing values."""
    return pd.DataFrame({
        "albumin": [3.5, np.nan, 4.0, np.nan],
        "stage": [1, 2, np.nan, 4],
        "age": [55, 60, 45, 70],
    })


class TestAddMissingIndicators:
    def test_basic(self, df):
        result = add_missing_indicators(df, columns=["albumin"])
        assert "albumin_is_missing" in result.columns
        assert list(result["albumin_is_missing"]) == [0, 1, 0, 1]

    def test_preserves_original(self, df):
        original_cols = list(df.columns)
        add_missing_indicators(df, columns=["albumin"])
        assert list(df.columns) == original_cols

    def test_multiple_columns(self, df):
        result = add_missing_indicators(df, columns=["albumin", "stage"])
        assert "albumin_is_missing" in result.columns
        assert "stage_is_missing" in result.columns
        assert list(result["albumin_is_missing"]) == [0, 1, 0, 1]
        assert list(result["stage_is_missing"]) == [0, 0, 1, 0]

    def test_unknown_column_raises(self, df):
        with pytest.raises(ValueError, match="not_a_column"):
            add_missing_indicators(df, columns=["not_a_column"])

    def test_no_nan_warns(self, df):
        with pytest.warns(SuggestionWarning, match="no missing values"):
            result = add_missing_indicators(df, columns=["age"])
        assert list(result["age_is_missing"]) == [0, 0, 0, 0]

    def test_all_nan(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        result = add_missing_indicators(df, columns=["x"])
        assert list(result["x_is_missing"]) == [1, 1, 1]

    def test_column_order(self, df):
        result = add_missing_indicators(df, columns=["albumin", "stage"])
        expected = ["albumin", "stage", "age", "albumin_is_missing", "stage_is_missing"]
        assert list(result.columns) == expected

    def test_empty_columns_list(self, df):
        result = add_missing_indicators(df, columns=[])
        pd.testing.assert_frame_equal(result, df)
        assert result is not df  # still a copy

    def test_index_preserved(self, df):
        df.index = [10, 20, 30, 40]
        result = add_missing_indicators(df, columns=["albumin"])
        assert list(result.index) == [10, 20, 30, 40]

    def test_invalid_columns_type(self, df):
        with pytest.raises(TypeError, match="list or tuple"):
            add_missing_indicators(df, columns="albumin")
