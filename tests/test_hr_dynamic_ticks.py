"""Tests for dynamic HR tick generation and annotation positioning."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv._warnings import ClampWarning, InternalWarning
from mlsurv.viz._features_hazard_ratios import (
    _compute_hr_ticks,
    _error_bar_axis_label,
    _finite_or_nan,
    _format_hr_tick,
    _hr_data_range,
    _hr_data_range_multi,
    _log_ratio_delta,
    _whisker_delta,
    _add_hr_risk_annotations,
)
from mlsurv.viz._palettes import MANUSCRIPT_DEFAULTS


# ── _error_bar_axis_label ────────────────────────────────────────────────

class TestErrorBarAxisLabel:
    """CI percent labels render via :g — no int() truncation."""

    def test_whole_percent(self):
        assert _error_bar_axis_label(True, "ci", 0.95) == "Hazard Ratio (95% CI)"

    def test_fractional_percent(self):
        """0.975 must render as 97.5, not truncate to 97 or round to 98."""
        assert _error_bar_axis_label(True, "ci", 0.975) == "Hazard Ratio (97.5% CI)"

    def test_float_noise_absorbed(self):
        """0.9 * 100 float noise must not leak into the label."""
        assert _error_bar_axis_label(True, "ci", 0.9) == "Hazard Ratio (90% CI)"


# ── _format_hr_tick ──────────────────────────────────────────────────────

class TestFormatHrTick:
    def test_integer_values(self):
        assert _format_hr_tick(1) == "1"
        assert _format_hr_tick(2) == "2"
        assert _format_hr_tick(10) == "10"

    def test_decimal_values(self):
        assert _format_hr_tick(0.5) == "0.5"
        assert _format_hr_tick(0.25) == "0.25"
        assert _format_hr_tick(0.1) == "0.1"

    def test_no_trailing_zeros(self):
        assert _format_hr_tick(2.0) == "2"
        assert _format_hr_tick(0.50) == "0.5"

    def test_small_values(self):
        assert _format_hr_tick(0.01) == "0.01"
        assert _format_hr_tick(0.001) == "0.001"


# ── _compute_hr_ticks ────────────────────────────────────────────────────

class TestComputeHrTicks:
    def test_always_includes_one(self):
        ticks = _compute_hr_ticks(0.3, 3.0)
        assert 1.0 in ticks

    def test_covers_data_range(self):
        ticks = _compute_hr_ticks(0.3, 3.0)
        assert ticks[0] <= 0.3
        assert ticks[-1] >= 3.0

    def test_narrow_range(self):
        """HRs from 0.8 to 1.2 — should still produce useful ticks."""
        ticks = _compute_hr_ticks(0.8, 1.2)
        assert 1.0 in ticks
        assert len(ticks) >= 3

    def test_wide_range(self):
        """HRs from 0.01 to 100 — should span many decades."""
        ticks = _compute_hr_ticks(0.01, 100)
        assert 1.0 in ticks
        assert ticks[0] <= 0.01
        assert ticks[-1] >= 100

    def test_sorted_output(self):
        ticks = _compute_hr_ticks(0.1, 10)
        assert ticks == sorted(ticks)

    def test_minimum_three_ticks(self):
        ticks = _compute_hr_ticks(0.95, 1.05)
        assert len(ticks) >= 3

    def test_typical_clinical_range(self):
        """Typical HR range for clinical data: 0.5 to 4."""
        ticks = _compute_hr_ticks(0.5, 4.0)
        assert 1.0 in ticks
        assert len(ticks) >= 3
        # Should include nice numbers
        assert all(isinstance(t, (int, float)) for t in ticks)

    def test_all_protective(self):
        """All HRs < 1 (protective features only)."""
        ticks = _compute_hr_ticks(0.1, 0.8)
        assert 1.0 in ticks
        assert ticks[0] <= 0.1

    def test_all_risky(self):
        """All HRs > 1 (risk features only)."""
        ticks = _compute_hr_ticks(1.5, 8.0)
        assert 1.0 in ticks
        assert ticks[-1] >= 8.0


# ── _hr_data_range ───────────────────────────────────────────────────────

class TestHrDataRange:
    def test_with_ci(self):
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "hr": [0.5, 1.0, 2.0],
            "ci_lower": [0.3, 0.8, 1.5],
            "ci_upper": [0.7, 1.2, 3.0],
        })
        lo, hi = _hr_data_range(df)
        assert lo == 0.3
        assert hi == 3.0

    def test_without_ci_columns(self):
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [0.5, 2.0],
        })
        lo, hi = _hr_data_range(df, ci_lower_col="missing_lo",
                                ci_upper_col="missing_hi")
        assert lo == 0.5
        assert hi == 2.0

    def test_nan_ci_values(self):
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [0.5, 2.0],
            "ci_lower": [np.nan, 1.5],
            "ci_upper": [np.nan, 3.0],
        })
        lo, hi = _hr_data_range(df)
        assert lo == 0.5
        assert hi == 3.0

    def test_safety_floor(self):
        """Ensure lo is clamped to 1e-6 even with very small values.

        The value has to be a tiny *positive* one to reach the clamp: 0.0 is
        no longer plottable, so it takes the default-range early return
        instead and never touches this code path.
        """
        df = pd.DataFrame({
            "feature": ["a"],
            "hr": [1e-9],  # edge case
            "ci_lower": [1e-9],
            "ci_upper": [1.0],
        })
        lo, hi = _hr_data_range(df)
        assert lo == 1e-6

    def test_nan_hr_values(self):
        """NaN in the hr column should not produce NaN boundaries."""
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "hr": [np.nan, 0.5, 2.0],
            "ci_lower": [np.nan, 0.3, 1.5],
            "ci_upper": [np.nan, 0.7, 3.0],
        })
        lo, hi = _hr_data_range(df)
        assert np.isfinite(lo)
        assert np.isfinite(hi)
        assert lo == 0.3
        assert hi == 3.0

    def test_identical_min_max(self):
        """When all HRs and CIs are the same, hi > lo is still guaranteed."""
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [1.0, 1.0],
            "ci_lower": [1.0, 1.0],
            "ci_upper": [1.0, 1.0],
        })
        lo, hi = _hr_data_range(df)
        assert lo < hi  # hi = max(1.0, 1.0 * 2) = 2.0

    def test_all_nan_hr(self):
        """All-NaN hr column should return safe defaults."""
        df = pd.DataFrame({
            "feature": ["a"],
            "hr": [np.nan],
            "ci_lower": [np.nan],
            "ci_upper": [np.nan],
        })
        lo, hi = _hr_data_range(df)
        assert lo == 0.25
        assert hi == 4.0


# ── _hr_data_range_multi ─────────────────────────────────────────────────

class TestHrDataRangeMulti:
    def test_spans_all_models(self):
        hr_data = {
            "model_a": pd.DataFrame({
                "feature": ["x", "y"], "hr": [0.5, 1.0],
                "ci_lower": [0.3, 0.8], "ci_upper": [0.7, 1.2],
            }),
            "model_b": pd.DataFrame({
                "feature": ["x", "y"], "hr": [1.0, 3.0],
                "ci_lower": [0.8, 2.0], "ci_upper": [1.2, 5.0],
            }),
        }
        lo, hi = _hr_data_range_multi(hr_data, ["x", "y"])
        assert lo == 0.3
        assert hi == 5.0

    def test_empty_fallback(self):
        import warnings
        from mlsurv._warnings import InternalWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lo, hi = _hr_data_range_multi({}, ["x"])
        assert lo == 0.25
        assert hi == 4.0
        assert any(issubclass(warning.category, InternalWarning) for warning in w)


# ── hazard ratios a log axis cannot place ────────────────────────────────
#
# A univariate Cox fit on a separated or badly scaled feature overflows
# exp(coef + 1.96*se) to +inf and underflows exp(coef - 1.96*se) to 0.0.
# Series.dropna() removes neither, so before this was guarded the infinity
# reached math.ceil(math.log10(inf)) and raised OverflowError, which cost the
# reader the whole Feature Analysis section.

def _catch_clamp(fn, *args, **kwargs):
    """Run *fn*, returning ``(result, [unplottable-value warning messages])``.

    Filtered to this change's message: ``_safe_exp`` raises a different
    ``ClampWarning`` from the same module, and a bare category check would
    silently start counting it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught
                    if issubclass(w.category, ClampWarning)
                    and "cannot be placed on a log axis" in str(w.message)]


class TestUnplottableHrDataRange:
    def test_inf_ci_upper_excluded_from_range(self):
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [0.5, 2.0],
            "ci_lower": [0.3, 1.5],
            "ci_upper": [0.7, np.inf],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo == 0.3
        assert hi == 2.0  # the finite hr max, not inf
        assert len(clamps) == 1

    def test_inf_hr_excluded_from_range(self):
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [np.inf, 2.0],
            "ci_lower": [1.5, 1.5],
            "ci_upper": [3.0, 3.0],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert np.isfinite(lo) and np.isfinite(hi)
        assert hi == 3.0
        assert len(clamps) == 1

    def test_negative_inf_ci_lower_excluded(self):
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [0.5, 2.0],
            "ci_lower": [-np.inf, 1.5],
            "ci_upper": [0.7, 3.0],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo == 0.5  # the finite hr min, not -inf
        assert len(clamps) == 1

    def test_nan_and_inf_together(self):
        """NaN was always dropped; only the inf is newly reported."""
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "hr": [np.nan, 0.5, 2.0],
            "ci_lower": [np.nan, 0.3, 1.5],
            "ci_upper": [np.inf, 0.7, 3.0],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert (lo, hi) == (0.3, 3.0)
        assert len(clamps) == 1
        assert clamps[0].startswith("1 hazard ratio")

    def test_all_inf_hr_falls_back_to_default(self):
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [np.inf, np.inf],
            "ci_lower": [np.inf, np.inf],
            "ci_upper": [np.inf, np.inf],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert (lo, hi) == (0.25, 4.0)
        assert len(clamps) == 1
        # All six values are counted: the CI columns must not be skipped just
        # because the early return fires on an all-infinite hr column.
        assert clamps[0].startswith("6 hazard ratio")

    def test_finite_frame_emits_no_warning(self):
        """The no-false-positive guard: clean data must stay silent."""
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [0.5, 2.0],
            "ci_lower": [0.3, 1.5],
            "ci_upper": [0.7, 3.0],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert (lo, hi) == (0.3, 3.0)
        assert clamps == []

    def test_zero_ci_lower_is_reported_not_silently_floored(self):
        """0.0 is the underflow twin of inf: exp(coef - 1.96*se) hitting zero.

        A log axis cannot place it either, so it gets the same treatment as
        the infinite upper bound rather than silently stretching the axis
        floor down to 1e-6.
        """
        df = pd.DataFrame({
            "feature": ["a"],
            "hr": [2.0], "ci_lower": [0.0], "ci_upper": [3.0],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert lo == 2.0  # the finite positive hr, not the 1e-6 floor
        assert hi == 4.0  # the pre-existing hi >= lo * 2 spread
        assert len(clamps) == 1

    def test_negative_hr_is_reported(self):
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [-3.0, 2.0], "ci_lower": [1.5, 1.5], "ci_upper": [3.0, 3.0],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert lo == 1.5 and hi == 3.0
        assert len(clamps) == 1

    def test_non_numeric_entry_is_reported_not_coerced_away(self):
        """A corrupt column must not quietly change the axis range."""
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": ["oops", 2.0], "ci_lower": [1.5, 1.5], "ci_upper": [3.0, 3.0],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert len(clamps) == 1

    def test_missing_ci_columns_with_inf_hr(self):
        """The 'column absent' branches must still handle a bad hr."""
        df = pd.DataFrame({"feature": ["a", "b"], "hr": [np.inf, 2.0]})
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert np.isfinite(lo) and np.isfinite(hi)
        assert len(clamps) == 1

    def test_all_nan_ci_beside_finite_hr_stays_silent(self):
        """The 'no plottable CI values' branch, with nothing worth reporting."""
        df = pd.DataFrame({
            "feature": ["a"],
            "hr": [2.0], "ci_lower": [np.nan], "ci_upper": [np.nan],
        })
        (lo, hi), clamps = _catch_clamp(_hr_data_range, df)
        assert (lo, hi) == (2.0, 4.0)
        assert clamps == []


class TestUnplottableHrDataRangeMulti:
    def test_one_warning_across_models(self):
        """Three models each carrying an inf must warn once, not three times."""
        def _frame(ci_upper):
            return pd.DataFrame({
                "feature": ["x", "y"], "hr": [0.5, 2.0],
                "ci_lower": [0.3, 1.5], "ci_upper": ci_upper,
            })

        hr_data = {
            "model_a": _frame([0.7, np.inf]),
            "model_b": _frame([0.7, np.inf]),
            "model_c": _frame([0.7, np.inf]),
        }
        (lo, hi), clamps = _catch_clamp(
            _hr_data_range_multi, hr_data, ["x", "y"]
        )
        assert np.isfinite(lo) and np.isfinite(hi)
        assert len(clamps) == 1
        assert clamps[0].startswith("3 hazard ratio")  # total across models

    def test_single_model(self):
        """n=1 is the minimum meaningful model count."""
        hr_data = {
            "only": pd.DataFrame({
                "feature": ["x"], "hr": [1.0],
                "ci_lower": [0.8], "ci_upper": [np.inf],
            }),
        }
        (lo, hi), clamps = _catch_clamp(_hr_data_range_multi, hr_data, ["x"])
        assert np.isfinite(lo) and np.isfinite(hi)
        assert len(clamps) == 1


class TestUnplottableComputeHrTicks:
    """Direct callers must not be able to blow the helper up either."""

    @pytest.mark.parametrize("hr_min, hr_max", [
        (0.5, np.inf),          # the reported crash
        (np.inf, 4.0),
        (np.inf, np.inf),
        (0.5, np.nan),
        (np.nan, 4.0),
        (0.0, 4.0),             # math.log10(0) raises ValueError
        (0.5, 0.0),
        (-1.0, 4.0),            # a log axis has no negative bound
        (4.0, 0.5),             # handed over backwards
        (5.0, 5.0),             # a single point, no span
        (10.0, np.inf),         # substituted bound below the real one
        (1e308, 1e308),         # widening `* 2` overflows to inf
        (5e-324, 5e-324),       # widening `/ 2` flushes to zero
        # Finite but enormous: `10 ** 309` is an exact int float() cannot
        # convert, so these raised the same OverflowError as an infinity.
        (0.5, 1e307),
        (0.5, 1.35e308),
        (1e-310, 4.0),
        # hr_min large enough to keep the 1-2-5 multipliers: 10.0**308 is
        # finite but 2 * 10.0**308 is not, so an infinite candidate used to be
        # picked as the bracket and returned as a tick.
        (1e302, 1.35e308),
        (1e305, 1.35e308),
        (1e303, 1.797e308),
    ])
    def test_degenerate_bounds_produce_finite_ticks(self, hr_min, hr_max):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InternalWarning)
            ticks = _compute_hr_ticks(hr_min, hr_max)
        assert len(ticks) >= 3
        assert all(np.isfinite(t) and t > 0 for t in ticks)
        assert ticks == sorted(ticks)

    @pytest.mark.parametrize("hr_min, hr_max", [
        (0.5, np.inf), (np.inf, 4.0), (np.nan, 4.0), (0.0, 4.0), (-1.0, 4.0),
    ])
    def test_default_range_keeps_the_null_line(self, hr_min, hr_max):
        """Bounds that fall back to the default must still label HR = 1."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InternalWarning)
            ticks = _compute_hr_ticks(hr_min, hr_max)
        assert 1.0 in ticks

    def test_wide_range_stays_readable(self):
        """A 300-decade span must not label every decade."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InternalWarning)
            ticks = _compute_hr_ticks(0.5, 1e300)
        assert len(ticks) <= 20

    def test_huge_finite_bound_does_not_raise(self):
        """The crash reached through a finite bound rather than an infinity."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InternalWarning)
            ticks = _compute_hr_ticks(0.5, 1e307)
        assert all(np.isfinite(t) and t > 0 for t in ticks)

    def test_plot_axis_never_gets_an_infinite_tick(self):
        """End to end: no infinity may reach the axis, and no 300-digit label."""
        from mlsurv.viz.features import plot_hazard_ratios

        df = pd.DataFrame({
            "feature": ["a", "b"],
            "hr": [1e303, 1.7e308],
            "ci_lower": [1e303, 1.7e308],
            "ci_upper": [1e303, 1.7e308],
            "ci_available": [True, True],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plot_hazard_ratios(df)

        tickvals = fig.layout.xaxis.tickvals
        assert all(np.isfinite(v) for v in tickvals)
        assert all(len(t) <= 12 for t in fig.layout.xaxis.ticktext)

    def test_inverted_bounds_span_the_same_range(self):
        """Backwards bounds describe the same span, so give the same ticks."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InternalWarning)
            assert _compute_hr_ticks(4.0, 0.5) == _compute_hr_ticks(0.5, 4.0)

    def test_substituted_bound_warns(self):
        """A silent substitution would draw an axis that misstates the data."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _compute_hr_ticks(0.5, np.inf)
        assert any(issubclass(w.category, InternalWarning) for w in caught)

    def test_finite_bounds_unchanged(self):
        """The guard must not perturb the ordinary path."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ticks = _compute_hr_ticks(0.3, 3.0)
        assert ticks[0] <= 0.3
        assert ticks[-1] >= 3.0
        assert 1.0 in ticks
        assert not [w for w in caught
                    if issubclass(w.category, InternalWarning)]


class TestFormatHrTickNonFinite:
    def test_infinities_do_not_raise(self):
        assert _format_hr_tick(np.inf) == "∞"
        assert _format_hr_tick(-np.inf) == "-∞"

    def test_nan_renders_blank(self):
        assert _format_hr_tick(np.nan) == ""


class TestFiniteOrNan:
    """Marker positions: non-finite in, NaN out."""

    def test_finite_values_pass_through(self):
        out = _finite_or_nan(np.array([0.5, 1.5]))
        assert out.tolist() == [0.5, 1.5]

    def test_infinite_value_becomes_nan(self):
        out = _finite_or_nan(np.array([np.inf, -np.inf, np.nan, 1.0]))
        assert np.isnan(out[:3]).all()
        assert out[3] == 1.0


class TestWhiskerDelta:
    """Forest-plot whiskers are gated on the bound and on the sign."""

    def test_ordinary_bounds(self):
        out, n_wrong = _whisker_delta(np.array([3.0]), np.array([2.0]), "upper")
        assert out.tolist() == [1.0]
        assert n_wrong == 0
        out, n_wrong = _whisker_delta(np.array([0.5]), np.array([1.0]), "lower")
        assert out.tolist() == [0.5]
        assert n_wrong == 0

    def test_zero_lower_bound_draws_nothing(self):
        """hr - 0.0 is finite, so a delta-only check would draw a whisker to a
        bound the axis range excludes."""
        out, _ = _whisker_delta(np.array([0.0]), np.array([1.0e9]), "lower")
        assert np.isnan(out[0])

    def test_infinite_upper_bound_draws_nothing(self):
        out, _ = _whisker_delta(np.array([np.inf]), np.array([2.0]), "upper")
        assert np.isnan(out[0])

    def test_unplottable_point_estimate_draws_nothing(self):
        out, _ = _whisker_delta(np.array([2.0]), np.array([np.inf]), "upper")
        assert np.isnan(out[0])

    def test_nan_bound_draws_nothing(self):
        out, _ = _whisker_delta(np.array([np.nan]), np.array([2.0]), "upper")
        assert np.isnan(out[0])

    def test_wrong_side_bound_invents_nothing(self):
        """An upper bound below the point estimate must not be mirrored.

        Taking the magnitude would draw a whisker out to 5.0 for hr=3, bound=1
        -- a value in neither the data nor the interval.
        """
        out, n_wrong = _whisker_delta(np.array([1.0]), np.array([3.0]), "upper")
        assert np.isnan(out[0])
        assert n_wrong == 1

    def test_no_bare_numpy_warning(self):
        """np.where evaluates both branches; inf - inf must stay quiet."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, _ = _whisker_delta(np.array([np.inf, 3.0]),
                                    np.array([np.inf, 2.0]), "upper")
        assert np.isnan(out[0])
        assert out[1] == 1.0
        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)]

    def test_plot_warns_once_for_swapped_columns(self):
        """One warning per figure, not one per category trace."""
        from mlsurv.viz.features import plot_hazard_ratios

        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "hr": [2.0, 1.0, 0.5],
            # Deliberately swapped: lower above hr, upper below it.
            "ci_lower": [3.0, 1.5, 0.8],
            "ci_upper": [1.5, 0.8, 0.3],
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_hazard_ratios(df)
        msgs = [w for w in caught if issubclass(w.category, InternalWarning)
                and "wrong side" in str(w.message)]
        assert len(msgs) == 1


class TestLogRatioDelta:
    """Comparison-plot error-bar deltas: non-finite in, NaN out, no warning."""

    def test_finite_ratio_is_log10(self):
        out = _log_ratio_delta(np.array([100.0]), np.array([1.0]))
        assert out[0] == pytest.approx(2.0)

    def test_non_finite_becomes_nan_without_runtimewarning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = _log_ratio_delta(
                np.array([np.inf, 1.0, 0.0, np.inf, 2.0]),
                np.array([1.0, 0.0, 1.0, np.inf, 1.0]),
            )
        assert np.isnan(out[0])          # inf / 1   -> inf
        assert np.isnan(out[1])          # 1 / 0     -> divide by zero
        assert np.isnan(out[2])          # 0 / 1     -> log10(0) = -inf
        assert np.isnan(out[3])          # inf / inf -> invalid
        assert out[4] == pytest.approx(np.log10(2.0))
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


class TestTruncationDoesNotHideUnplottableFeatures:
    """`nlargest` truncates before the axis range is computed.

    A feature that sorts low is cut before anything can count it, so an
    unplottable hazard ratio has to rank at the TOP or it disappears with no
    warning at all -- the one outcome this whole change exists to prevent.
    """

    @staticmethod
    def _frame(hr_values):
        n = len(hr_values)
        return pd.DataFrame({
            "feature": [f"f{i}" for i in range(n)],
            "hr": hr_values,
            "ci_lower": [v * 0.9 if v > 0 else 0.1 for v in hr_values],
            "ci_upper": [v * 1.1 if v > 0 else 0.2 for v in hr_values],
        })

    @pytest.mark.parametrize("bad", [np.inf, 0.0, -3.0])
    def test_unplottable_survives_truncation_and_warns(self, bad):
        from mlsurv.viz.features import plot_hazard_ratios

        df = self._frame([bad, 1.1, 1.2, 1.3, 1.4, 1.5])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_hazard_ratios(df, n_features=3)

        assert [w for w in caught
                if issubclass(w.category, ClampWarning)
                and "cannot be placed on a log axis" in str(w.message)]

    def test_clean_frame_still_truncates_quietly(self):
        from mlsurv.viz.features import plot_hazard_ratios

        df = self._frame([1.1, 1.2, 1.3, 1.4, 1.5, 4.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_hazard_ratios(df, n_features=3)

        assert not [w for w in caught
                    if issubclass(w.category, ClampWarning)
                    and "cannot be placed on a log axis" in str(w.message)]


# ── _add_hr_risk_annotations ─────────────────────────────────────────────

class TestAnnotationPositioning:
    def test_annotations_use_domain_coordinates(self):
        """Annotations should use domain-relative coordinates (0-1)."""
        fig = go.Figure()
        d = MANUSCRIPT_DEFAULTS
        _add_hr_risk_annotations(fig, ann_y=1.02, d=d)
        annotations = fig.layout.annotations
        assert len(annotations) == 2

        lower_ann = annotations[0]
        upper_ann = annotations[1]

        # Should be at 25% and 75% of the axis domain
        assert lower_ann.x == 0.25
        assert upper_ann.x == 0.75
        assert lower_ann.xref == "x domain"
        assert upper_ann.xref == "x domain"

    def test_annotations_well_separated(self):
        """Annotations should always be 50% apart in domain space."""
        fig = go.Figure()
        d = MANUSCRIPT_DEFAULTS
        _add_hr_risk_annotations(fig, ann_y=1.02, d=d)
        annotations = fig.layout.annotations
        separation = annotations[1].x - annotations[0].x
        assert separation == 0.5

    def test_annotations_respect_xref(self):
        """When xref='x2', annotations should use 'x2 domain'."""
        fig = go.Figure()
        d = MANUSCRIPT_DEFAULTS
        _add_hr_risk_annotations(fig, ann_y=1.02, d=d, xref="x2")
        annotations = fig.layout.annotations
        assert annotations[0].xref == "x2 domain"
        assert annotations[1].xref == "x2 domain"


# ── Integration: plot_hazard_ratios ──────────────────────────────────────

class TestPlotHazardRatiosDynamicTicks:
    def _make_hr_df(self, hr_range=(0.3, 5.0), n=10):
        """Create a synthetic HR DataFrame with a given range."""
        hrs = np.linspace(hr_range[0], hr_range[1], n)
        return pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(n)],
            "hr": hrs,
            "ci_lower": hrs * 0.8,
            "ci_upper": hrs * 1.2,
        })

    def test_ticks_adapt_to_data(self):
        from mlsurv.viz.features import plot_hazard_ratios

        df = self._make_hr_df(hr_range=(0.05, 15.0))
        fig = plot_hazard_ratios(df, n_features=10)

        # Extract x-axis tickvals — may be on xaxis or xaxis2
        tickvals = fig.layout.xaxis.tickvals
        if tickvals is None and hasattr(fig.layout, "xaxis2"):
            tickvals = fig.layout.xaxis2.tickvals
        assert tickvals is not None

        tickvals = list(tickvals)
        assert 1.0 in tickvals
        # Ticks should extend beyond the old hardcoded [0.25, 4]
        assert min(tickvals) <= 0.05
        assert max(tickvals) >= 15.0

    def test_ticks_different_for_different_data(self):
        from mlsurv.viz.features import plot_hazard_ratios

        df_narrow = self._make_hr_df(hr_range=(0.8, 1.3))
        df_wide = self._make_hr_df(hr_range=(0.05, 20.0))

        fig_narrow = plot_hazard_ratios(df_narrow, n_features=10)
        fig_wide = plot_hazard_ratios(df_wide, n_features=10)

        ticks_narrow = list(fig_narrow.layout.xaxis.tickvals or [])
        ticks_wide = list(fig_wide.layout.xaxis.tickvals or [])

        # Wide range should have more tick spread
        assert max(ticks_wide) > max(ticks_narrow)


class TestPlotHazardRatiosComparisonDynamicTicks:
    def test_comparison_ticks_adapt(self):
        from mlsurv.viz.features import plot_hazard_ratios_comparison

        models = {
            "model_a": pd.DataFrame({
                "feature": ["x", "y", "z"],
                "hr": [0.1, 1.0, 8.0],
                "ci_lower": [0.05, 0.8, 6.0],
                "ci_upper": [0.2, 1.2, 12.0],
            }),
        }
        fig = plot_hazard_ratios_comparison(models, n_features=3)

        tickvals = fig.layout.xaxis.tickvals
        assert tickvals is not None
        tickvals = list(tickvals)
        assert 1.0 in tickvals
        assert min(tickvals) <= 0.1
        assert max(tickvals) >= 8.0

    def test_multi_model_union_range(self):
        """Ticks should cover the union of all models' HR ranges."""
        from mlsurv.viz.features import plot_hazard_ratios_comparison

        models = {
            "narrow_model": pd.DataFrame({
                "feature": ["x", "y", "z"],
                "hr": [0.8, 1.0, 1.2],
                "ci_lower": [0.7, 0.9, 1.1],
                "ci_upper": [0.9, 1.1, 1.3],
            }),
            "wide_model": pd.DataFrame({
                "feature": ["x", "y", "z"],
                "hr": [0.05, 1.0, 15.0],
                "ci_lower": [0.02, 0.8, 10.0],
                "ci_upper": [0.1, 1.2, 20.0],
            }),
        }
        fig = plot_hazard_ratios_comparison(models, n_features=3)

        tickvals = fig.layout.xaxis.tickvals
        assert tickvals is not None
        tickvals = list(tickvals)
        # Ticks must cover the wide model's range, not just the narrow one
        assert min(tickvals) <= 0.05
        assert max(tickvals) >= 15.0
