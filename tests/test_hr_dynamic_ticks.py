"""Tests for dynamic HR tick generation and annotation positioning."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv.viz._features_hazard_ratios import (
    _compute_hr_ticks,
    _error_bar_axis_label,
    _format_hr_tick,
    _hr_data_range,
    _hr_data_range_multi,
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
        """Ensure lo is clamped to 1e-6 even with very small values."""
        df = pd.DataFrame({
            "feature": ["a"],
            "hr": [0.0],  # edge case
            "ci_lower": [0.0],
            "ci_upper": [1.0],
        })
        lo, hi = _hr_data_range(df)
        assert lo >= 1e-6

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
