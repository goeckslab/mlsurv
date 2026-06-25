"""Tests for mlsurv.viz._plotly_helpers."""

import numpy as np
import pytest

from mlsurv.viz._plotly_helpers import (
    _to_list,
    asymmetric_error_bar,
    symmetric_error_bar,
)


class TestToList:
    """Tests for _to_list array normalisation."""

    def test_numpy_array_to_list(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_list(arr)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_plain_list_passthrough(self):
        lst = [1, 2, 3]
        assert _to_list(lst) is lst

    def test_pandas_series_to_list(self):
        import pandas as pd
        s = pd.Series([4.0, 5.0])
        result = _to_list(s)
        assert isinstance(result, list)

    def test_scalar_passthrough(self):
        assert _to_list(42) == 42


class TestSymmetricErrorBar:
    """Tests for symmetric_error_bar builder."""

    def test_basic(self):
        d = symmetric_error_bar([0.1, 0.2])
        assert d["type"] == "data"
        assert d["array"] == [0.1, 0.2]
        assert d["visible"] is True
        assert "color" not in d

    def test_with_color(self):
        d = symmetric_error_bar([0.1], color="red")
        assert d["color"] == "red"

    def test_numpy_serialised(self):
        d = symmetric_error_bar(np.array([0.1, 0.2]))
        assert isinstance(d["array"], list)

    def test_extra_keys(self):
        d = symmetric_error_bar([0.1], thickness=1.5, width=4)
        assert d["thickness"] == 1.5
        assert d["width"] == 4


class TestAsymmetricErrorBar:
    """Tests for asymmetric_error_bar builder."""

    def test_basic(self):
        d = asymmetric_error_bar([0.1, 0.2], [0.05, 0.1])
        assert d["symmetric"] is False
        assert d["array"] == pytest.approx([0.1, 0.2])
        assert d["arrayminus"] == pytest.approx([0.05, 0.1])

    def test_returns_none_all_nan(self):
        result = asymmetric_error_bar(
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
        )
        assert result is None

    def test_with_color_and_extras(self):
        d = asymmetric_error_bar(
            [0.1], [0.05],
            color="blue", thickness=1.5, width=4,
        )
        assert d["color"] == "blue"
        assert d["thickness"] == 1.5

    def test_numpy_serialised(self):
        d = asymmetric_error_bar(np.array([0.1]), np.array([0.05]))
        assert isinstance(d["array"], list)
        assert isinstance(d["arrayminus"], list)
