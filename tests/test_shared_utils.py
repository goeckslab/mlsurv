"""Tests for mlsurv._shared utilities."""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import InternalWarning, SuggestionWarning
from mlsurv._shared import (
    display_name,
    iter_subpopulations,
    nan_summary,
    percentile_ci,
    safe_call,
    tidy_model_col,
    try_verbose_print,
)


# ── safe_call ──────────────────────────────────────────────────────────────

class TestSafeCall:
    """Tests for safe_call exception guard."""

    def test_returns_function_result_on_success(self):
        assert safe_call(lambda x: x + 1, 2) == 3

    def test_returns_default_on_failure(self):
        assert np.isnan(safe_call(lambda: 1 / 0))

    def test_custom_default_on_failure(self):
        assert safe_call(lambda: 1 / 0, default=-1) == -1

    def test_reraises_memory_error(self):
        def boom():
            raise MemoryError("oom")
        with pytest.raises(MemoryError):
            safe_call(boom)

    def test_reraises_recursion_error(self):
        def boom():
            raise RecursionError("too deep")
        with pytest.raises(RecursionError):
            safe_call(boom)

    def test_logs_warning_on_failure(self, caplog):
        with caplog.at_level(logging.WARNING, logger="mlsurv.shared"):
            safe_call(lambda: 1 / 0, context="my_metric")
        assert "my_metric" in caplog.text

    def test_warn_user_emits_user_warning(self):
        with pytest.warns(SuggestionWarning, match="bad_fn"):
            safe_call(lambda: 1 / 0, context="bad_fn", warn_user=True)

    def test_kwargs_forwarded(self):
        def add(a, b=0):
            return a + b
        assert safe_call(add, 3, b=4) == 7


# ── iter_subpopulations ─────────────────────────────────────────────────────────

class TestIterSubpopulations:
    """Tests for iter_subpopulations generator."""

    @pytest.fixture()
    def y(self):
        """Minimal SurvivalArray with 6 samples."""
        return np.array(
            [(True, 1.0), (True, 2.0), (False, 3.0),
             (True, 4.0), (True, 5.0), (False, 6.0)],
            dtype=[("event", bool), ("time", float)],
        )

    def test_dict_subpopulations(self, y):
        labels = np.array(["A", "A", "A", "B", "B", "B"])
        results = list(iter_subpopulations({"cancer": labels}, y, min_events=1))
        assert len(results) == 2
        gvars = [r[0] for r in results]
        assert all(g == "cancer" for g in gvars)
        labels_out = [r[1] for r in results]
        assert set(labels_out) == {"A", "B"}

    def test_flat_array_wraps_as_bootstrap(self, y):
        labels = np.array(["X", "X", "X", "Y", "Y", "Y"])
        results = list(iter_subpopulations(labels, y, min_events=1))
        assert all(r[0] == "_bootstrap" for r in results)

    def test_min_events_filter(self, y):
        labels = np.array(["A", "A", "A", "B", "B", "B"])
        # Group B has 2 events (indices 3, 4), group A has 2 events (indices 0, 1)
        results = list(iter_subpopulations({"g": labels}, y, min_events=3))
        assert len(results) == 0

    def test_nan_labels_excluded(self, y):
        labels = np.array(["A", "A", np.nan, "B", "B", np.nan], dtype=object)
        results = list(iter_subpopulations({"g": labels}, y, min_events=1))
        labels_out = {r[1] for r in results}
        assert labels_out == {"A", "B"}

    def test_mask_and_y_sub_correct(self, y):
        labels = np.array(["A", "A", "A", "B", "B", "B"])
        for gvar, label, mask, y_sub in iter_subpopulations({"g": labels}, y, min_events=1):
            assert mask.sum() == 3
            assert len(y_sub) == 3


# ── try_verbose_print ──────────────────────────────────────────────────────

class TestTryVerbosePrint:
    """Tests for try_verbose_print terminal guard."""

    def test_calls_function_normally(self):
        called = []
        try_verbose_print(lambda: called.append(1))
        assert called == [1]

    def test_catches_broken_pipe(self):
        def boom():
            raise BrokenPipeError("pipe gone")
        with pytest.warns(InternalWarning, match="could not be printed"):
            try_verbose_print(boom, context="Test print")

    def test_catches_unicode_encode_error(self):
        def boom():
            raise UnicodeEncodeError("cp1252", "", 0, 1, "bad char")
        with pytest.warns(InternalWarning, match="could not be printed"):
            try_verbose_print(boom, context="Test print")

    def test_catches_os_error(self):
        def boom():
            raise OSError("broken")
        with pytest.warns(InternalWarning, match="could not be printed"):
            try_verbose_print(boom, context="Test print")

    def test_detail_appended_to_warning(self):
        def boom():
            raise BrokenPipeError("pipe gone")
        with pytest.warns(InternalWarning, match="All configuration is active"):
            try_verbose_print(
                boom, context="Setup", detail="All configuration is active.",
            )

    def test_silent_mode(self):
        def boom():
            raise BrokenPipeError("pipe gone")
        # Should not raise or warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try_verbose_print(boom, warn=False)


# ── tidy_model_col ─────────────────────────────────────────────────────────

class TestTidyModelCol:
    """Tests for tidy_model_col rename helper."""

    def test_renames_model_type_to_model(self):
        df = pd.DataFrame({"model_type": ["coxph"], "value": [0.7]})
        result = tidy_model_col(df)
        assert "model" in result.columns
        assert "model_type" not in result.columns

    def test_no_rename_when_model_exists(self):
        df = pd.DataFrame({"model_type": ["coxph"], "model": ["CoxPH"]})
        result = tidy_model_col(df)
        assert "model_type" in result.columns  # unchanged

    def test_no_rename_when_no_model_type(self):
        df = pd.DataFrame({"name": ["coxph"], "value": [0.7]})
        result = tidy_model_col(df)
        assert list(result.columns) == ["name", "value"]


# ── display_name ───────────────────────────────────────────────────────────

class TestDisplayName:
    """Tests for display_name convenience wrapper."""

    def test_known_model(self):
        result = display_name("coxph")
        assert result == "CoxPH"

    def test_unknown_model_passthrough(self):
        # Unknown models are returned as-is by resolve_display_name
        result = display_name("some_unknown_model_xyz")
        assert isinstance(result, str)

    def test_tuned_suffix(self):
        result = display_name("coxph_tuned")
        assert "tuned" in result.lower()


# ── percentile_ci ──────────────────────────────────────────────────────────

class TestPercentileCi:
    """Tests for percentile_ci bootstrap confidence interval."""

    def test_1d_scalar_output(self):
        rng = np.random.RandomState(42)
        values = rng.normal(10, 2, size=1000)
        lower, upper = percentile_ci(values, ci_level=0.95)
        assert np.isscalar(lower) or lower.ndim == 0
        assert lower < 10 < upper

    def test_2d_array_output(self):
        rng = np.random.RandomState(42)
        values = rng.normal(0, 1, size=(500, 3))
        lower, upper = percentile_ci(values, ci_level=0.95)
        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert np.all(lower < upper)

    def test_nan_handling(self):
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        lower, upper = percentile_ci(values, ci_level=0.90)
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert lower < upper

    def test_wider_ci_produces_wider_interval(self):
        rng = np.random.RandomState(0)
        values = rng.normal(0, 1, size=500)
        lo_90, hi_90 = percentile_ci(values, ci_level=0.90)
        lo_99, hi_99 = percentile_ci(values, ci_level=0.99)
        assert lo_99 < lo_90
        assert hi_99 > hi_90

    def test_single_element(self):
        values = np.array([5.0])
        lower, upper = percentile_ci(values, ci_level=0.95)
        assert lower == 5.0
        assert upper == 5.0


# ── nan_summary ───────────────────────────────────────────────────────────

class TestNanSummary:
    """Tests for nan_summary NaN sentinel dict builder."""

    def test_base_four_keys(self):
        result = nan_summary()
        assert set(result.keys()) == {"mean", "std", "ci_lower", "ci_upper"}
        assert all(np.isnan(v) for v in result.values())

    def test_extra_keys_merged(self):
        result = nan_summary(p_value=np.nan)
        assert set(result.keys()) == {"mean", "std", "ci_lower", "ci_upper", "p_value"}
        assert all(np.isnan(v) for v in result.values())

    def test_extra_non_nan_value(self):
        result = nan_summary(count=0)
        assert result["count"] == 0
        assert np.isnan(result["mean"])

    def test_returns_fresh_dict_each_call(self):
        a = nan_summary()
        b = nan_summary()
        assert a is not b
        a["mean"] = 99
        assert np.isnan(b["mean"])

    def test_unpack_into_larger_dict(self):
        row = {"model": "coxph", "metric": "c_index", **nan_summary()}
        assert row["model"] == "coxph"
        assert np.isnan(row["mean"])
        assert len(row) == 6  # 2 context + 4 NaN
