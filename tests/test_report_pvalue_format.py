"""Tests for the unified report p-value formatter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlsurv.reports._constants import format_pvalue


class TestDecimalPath:
    def test_large_value(self):
        assert format_pvalue(0.673) == "0.673"

    def test_midrange_value(self):
        assert format_pvalue(0.169) == "0.169"

    def test_rounds_to_three_decimals(self):
        assert format_pvalue(0.050) == "0.050"

    def test_boundary_0_01_uses_decimal(self):
        assert format_pvalue(0.01) == "0.010"


class TestScientificPath:
    def test_just_below_boundary(self):
        # 0.009 → decimal path would be ".009"; scientific path kicks in.
        assert format_pvalue(0.009) == "9.00e-03"

    def test_very_small(self):
        assert format_pvalue(4.4e-04) == "4.40e-04"

    def test_contains_e_minus(self):
        assert "e-" in format_pvalue(1e-10)


class TestInvalidInputs:
    def test_nan(self):
        assert format_pvalue(float("nan")) == "—"

    def test_numpy_nan(self):
        assert format_pvalue(np.nan) == "—"

    def test_none(self):
        assert format_pvalue(None) == "—"

    def test_negative(self):
        assert format_pvalue(-0.1) == "—"

    def test_above_one(self):
        assert format_pvalue(1.5) == "—"

    def test_infinite(self):
        assert format_pvalue(math.inf) == "—"

    def test_non_numeric_string(self):
        assert format_pvalue("not-a-number") == "—"

    def test_pandas_na(self):
        import pandas as pd
        # pd.NA: float(pd.NA) raises TypeError — caught by the helper.
        assert format_pvalue(pd.NA) == "—"


class TestStarsOff:
    """Default stars=False should never append a suffix."""

    @pytest.mark.parametrize("p", [0.04, 0.005, 0.0005, 1e-10, 0.5, 0.01])
    def test_no_suffix(self, p):
        out = format_pvalue(p)
        assert not out.endswith("*")


class TestStarsOn:
    def test_star_single(self):
        assert format_pvalue(0.04, stars=True) == "0.040*"

    def test_star_double(self):
        assert format_pvalue(0.005, stars=True) == "5.00e-03**"

    def test_star_triple(self):
        assert format_pvalue(0.0005, stars=True) == "5.00e-04***"

    def test_no_star_when_not_significant(self):
        assert format_pvalue(0.1, stars=True) == "0.100"

    def test_boundary_0_05_no_star(self):
        # P_STARS uses strict <, so exactly 0.05 gets no star.
        assert format_pvalue(0.05, stars=True) == "0.050"

    def test_boundary_0_01_double_star(self):
        # Exactly 0.01 is not < 0.01 but is < 0.05 → single star.
        assert format_pvalue(0.01, stars=True) == "0.010*"

    def test_nan_stars_still_dash(self):
        assert format_pvalue(float("nan"), stars=True) == "—"


class TestSummaryClassifyPValue:
    """Smoke checks for summary._classify_p_value wrapping format_pvalue."""

    def test_excellent_branch_uses_scientific(self):
        from mlsurv.reports.sections.summary import _classify_p_value
        quality, p_str = _classify_p_value(1e-5)
        assert quality == "excellent"
        assert p_str == "p = *1.00e-05*"

    def test_fair_branch_uses_three_decimals(self):
        from mlsurv.reports.sections.summary import _classify_p_value
        quality, p_str = _classify_p_value(0.03)
        assert quality == "fair"
        assert p_str == "p = *0.030*"

    def test_poor_branch_uses_three_decimals(self):
        from mlsurv.reports.sections.summary import _classify_p_value
        quality, p_str = _classify_p_value(0.4)
        assert quality == "poor"
        assert p_str == "p = *0.400*"

    def test_nan_input_unknown(self):
        from mlsurv.reports.sections.summary import _classify_p_value
        quality, p_str = _classify_p_value(float("nan"))
        assert quality == "unknown"
        assert p_str == "p = *—*"

    def test_none_input_unknown(self):
        from mlsurv.reports.sections.summary import _classify_p_value
        quality, p_str = _classify_p_value(None)
        assert quality == "unknown"
        assert p_str == "p = *—*"
