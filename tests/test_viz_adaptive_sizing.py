"""Tests for adaptive_size(), truncate_labels(), margin_for_labels(),
and their integration with plot functions."""

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import InternalWarning
from mlsurv.viz._theme import adaptive_size, margin_for_labels, truncate_labels
from mlsurv.viz._palettes import MANUSCRIPT_DEFAULTS
from mlsurv.viz.features import (
    plot_coefficients,
    plot_hazard_ratios,
    plot_hazard_ratios_comparison,
    plot_importance,
)
from mlsurv.viz.metrics import plot_metric


# ---------------------------------------------------------------------------
# Unit tests — adaptive_size()
# ---------------------------------------------------------------------------

class TestAdaptiveSize:
    def test_basic_calculation(self):
        assert adaptive_size(10, per_item=30, base=100) == 400

    def test_min_clamp(self):
        assert adaptive_size(1, per_item=10, base=50, min_size=300) == 300

    def test_max_clamp(self):
        assert adaptive_size(100, per_item=30, base=100, max_size=1200) == 1200

    def test_zero_items(self):
        assert adaptive_size(0, per_item=30, base=100, min_size=300) == 300

    def test_exact_min_boundary(self):
        # base + n * per_item == min_size exactly
        assert adaptive_size(10, per_item=20, base=100, min_size=300) == 300

    def test_exact_max_boundary(self):
        # base + n * per_item == max_size exactly
        assert adaptive_size(10, per_item=100, base=200, max_size=1200) == 1200

    def test_between_bounds(self):
        result = adaptive_size(5, per_item=50, base=100, min_size=200, max_size=800)
        assert result == 350
        assert 200 <= result <= 800

    def test_returns_int(self):
        result = adaptive_size(3)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Integration tests — figure dimensions scale with content
# ---------------------------------------------------------------------------


def _importance_df(n):
    """Create a single-model importance DataFrame with *n* features."""
    return pd.DataFrame({
        "feature": [f"f{i}" for i in range(n)],
        "perm_mean": np.linspace(0.5, 0.01, n),
        "perm_std": [0.02] * n,
    })


def _metric_df(n_models):
    """Create a summary metric DataFrame with *n_models* models."""
    return pd.DataFrame({
        "model": [f"model_{i}" for i in range(n_models)],
        "point": np.linspace(0.55, 0.85, n_models),
        "ci_lower": np.linspace(0.50, 0.80, n_models),
        "ci_upper": np.linspace(0.60, 0.90, n_models),
    })


class TestImportanceSizing:
    def test_more_features_taller_figure(self):
        fig_small = plot_importance(_importance_df(5), n=5)
        fig_large = plot_importance(_importance_df(20), n=20)
        assert fig_large.layout.height > fig_small.layout.height

    def test_max_cap_enforced(self):
        fig = plot_importance(_importance_df(50), n=50)
        assert fig.layout.height <= 1200

    def test_single_feature_readable(self):
        fig = plot_importance(_importance_df(1), n=1)
        assert fig.layout.height >= MANUSCRIPT_DEFAULTS["height_small"]

    def test_user_override_wins(self):
        fig = plot_importance(_importance_df(10), n=10, height=999, width=777)
        assert fig.layout.height == 999
        assert fig.layout.width == 777


class TestMetricSizing:
    def test_more_models_taller_forest(self):
        fig_few = plot_metric(_metric_df(2))
        fig_many = plot_metric(_metric_df(9))
        assert fig_many.layout.height > fig_few.layout.height

    def test_single_model_readable(self):
        fig = plot_metric(_metric_df(1))
        assert fig.layout.height >= MANUSCRIPT_DEFAULTS["height_small"]


class TestHazardRatiosSizing:
    def _hr_df(self, n):
        return pd.DataFrame({
            "feature": [f"f{i}" for i in range(n)],
            "hr": np.exp(np.linspace(-1, 1, n)),
            "ci_lower": np.exp(np.linspace(-1.5, 0.5, n)),
            "ci_upper": np.exp(np.linspace(-0.5, 1.5, n)),
        })

    def test_more_features_taller(self):
        fig_small = plot_hazard_ratios(self._hr_df(3))
        fig_large = plot_hazard_ratios(self._hr_df(15))
        assert fig_large.layout.height > fig_small.layout.height

    def test_min_height(self):
        fig = plot_hazard_ratios(self._hr_df(1))
        assert fig.layout.height >= MANUSCRIPT_DEFAULTS["height_small"]


class TestHazardRatiosComparisonSizing:
    def test_width_at_least_medium(self):
        """plot_hazard_ratios_comparison width should be at least 'medium' (400)."""
        from mlsurv.viz.features import plot_hazard_ratios_comparison

        class FakeModel:
            hazard_ratio_valid = True  # pass the HR-validity gate

            def get_coefficients(self):
                return pd.Series({"f0": 0.5, "f1": -0.3})

        models = {"m1": FakeModel(), "m2": FakeModel()}
        fig = plot_hazard_ratios_comparison(models, n_features=2)
        assert fig.layout.width >= MANUSCRIPT_DEFAULTS["height_medium"]


class TestCoefficientsSizing:
    def test_scales_with_features(self):
        df_small = pd.DataFrame({
            "feature": [f"f{i}" for i in range(3)],
            "coefficient": [0.5, -0.3, 0.1],
        })
        df_large = pd.DataFrame({
            "feature": [f"f{i}" for i in range(20)],
            "coefficient": np.linspace(-1, 1, 20),
        })
        fig_small = plot_coefficients(df_small, n=3)
        fig_large = plot_coefficients(df_large, n=20)
        assert fig_large.layout.height > fig_small.layout.height


# ---------------------------------------------------------------------------
# Unit tests — truncate_labels()
# ---------------------------------------------------------------------------

class TestTruncateLabels:
    def test_short_names_unchanged(self):
        names = ["age", "sex", "bmi"]
        display, full = truncate_labels(names)
        assert display == names
        assert full == names

    def test_long_name_truncated(self):
        long = "a" * 50
        display, full = truncate_labels([long])
        assert full == [long]
        assert len(display[0]) == 30
        assert display[0].endswith("...")

    def test_custom_max_chars(self):
        name = "a" * 20
        display, _ = truncate_labels([name], max_chars=15)
        assert len(display[0]) == 15
        assert display[0].endswith("...")

    def test_empty_list(self):
        display, full = truncate_labels([])
        assert display == []
        assert full == []

    def test_exact_boundary(self):
        name = "a" * 30
        display, _ = truncate_labels([name], max_chars=30)
        assert display[0] == name  # exactly at boundary, no truncation

    def test_one_over_boundary(self):
        name = "a" * 31
        display, _ = truncate_labels([name], max_chars=30)
        assert display[0] == "a" * 27 + "..."


# ---------------------------------------------------------------------------
# Unit tests — margin_for_labels()
# ---------------------------------------------------------------------------

class TestMarginForLabels:
    def test_empty_list_returns_min(self):
        assert margin_for_labels([]) == 60

    def test_short_labels(self):
        result = margin_for_labels(["abc", "de"])
        # 3 chars * 7 + 10 = 31 → clamped to min_margin=60
        assert result == 60

    def test_long_labels_capped(self):
        result = margin_for_labels(["a" * 100])
        assert result == 250  # max_margin default

    def test_custom_params(self):
        result = margin_for_labels(
            ["a" * 20], per_char=10, padding=5, min_margin=50, max_margin=300,
        )
        assert result == 205  # 20*10 + 5

    def test_scaling(self):
        short = margin_for_labels(["abc"])
        long = margin_for_labels(["a" * 25])
        assert long >= short


# ---------------------------------------------------------------------------
# Integration — label truncation in plots
# ---------------------------------------------------------------------------

def _long_importance_df(n, name_len=50):
    """Create importance DataFrame with long feature names."""
    return pd.DataFrame({
        "feature": [f"feature_{'x' * name_len}_{i}" for i in range(n)],
        "perm_mean": np.linspace(0.5, 0.01, n),
        "perm_std": [0.02] * n,
    })


class TestImportanceLabelTruncation:
    def test_long_names_margin_capped(self):
        fig = plot_importance(_long_importance_df(10), n=10)
        assert fig.layout.margin.l <= 250

    def test_ticktext_truncated(self):
        fig = plot_importance(_long_importance_df(5, name_len=50), n=5)
        yaxis = fig.layout.yaxis
        assert yaxis.ticktext is not None, "ticktext should always be set"
        for label in yaxis.ticktext:
            assert len(label) <= 30


class TestCoefficientLabelTruncation:
    def test_long_names_margin_capped(self):
        df = pd.DataFrame({
            "feature": [f"very_long_feature_name_{'x' * 40}_{i}" for i in range(5)],
            "coefficient": np.linspace(-1, 1, 5),
        })
        fig = plot_coefficients(df, n=5)
        assert fig.layout.margin.l <= 250


class TestHazardRatioLabelTruncation:
    def test_long_names_margin_capped(self):
        long_names = [f"hazard_feature_{'y' * 40}_{i}" for i in range(5)]
        df = pd.DataFrame({
            "feature": long_names,
            "hr": np.exp(np.linspace(-1, 1, 5)),
            "ci_lower": np.exp(np.linspace(-1.5, 0.5, 5)),
            "ci_upper": np.exp(np.linspace(-0.5, 1.5, 5)),
        })
        fig = plot_hazard_ratios(df)
        assert fig.layout.margin.l <= 250

    def test_ticktext_truncated(self):
        long_names = [f"hazard_feature_{'y' * 40}_{i}" for i in range(5)]
        df = pd.DataFrame({
            "feature": long_names,
            "hr": np.exp(np.linspace(-1, 1, 5)),
            "ci_lower": np.exp(np.linspace(-1.5, 0.5, 5)),
            "ci_upper": np.exp(np.linspace(-0.5, 1.5, 5)),
        })
        fig = plot_hazard_ratios(df)
        yaxis = fig.layout.yaxis
        assert yaxis.ticktext is not None
        for label in yaxis.ticktext:
            assert len(label) <= 30


class TestHRComparisonLabelTruncation:
    def test_long_names_margin_capped(self):
        long_names = [f"comparison_feat_{'z' * 40}_{i}" for i in range(3)]

        class FakeModel:
            hazard_ratio_valid = True  # pass the HR-validity gate

            def __init__(self, names):
                self._names = names
            def get_coefficients(self):
                return pd.Series(np.linspace(-0.5, 0.5, len(self._names)),
                                 index=self._names)

        models = {"m1": FakeModel(long_names), "m2": FakeModel(long_names)}
        fig = plot_hazard_ratios_comparison(models, n_features=3)
        assert fig.layout.margin.l <= 250


# ---------------------------------------------------------------------------
# multi-model raincloud/boxplot falls back to bar with warning
# ---------------------------------------------------------------------------

class TestMultiModelRaincloudFallback:
    def test_raincloud_falls_back_to_bar(self):
        df = pd.DataFrame({
            "feature": ["f0", "f1"] * 2,
            "perm_mean": [0.5, 0.3, 0.4, 0.2],
            "perm_std": [0.01] * 4,
            "model_type": ["m1", "m1", "m2", "m2"],
        })
        with pytest.warns(InternalWarning, match="Falling back to 'bar' style"):
            fig = plot_importance(df, style="raincloud")
        assert fig is not None

    def test_boxplot_falls_back_to_bar(self):
        df = pd.DataFrame({
            "feature": ["f0", "f1"] * 2,
            "perm_mean": [0.5, 0.3, 0.4, 0.2],
            "perm_std": [0.01] * 4,
            "model_type": ["m1", "m1", "m2", "m2"],
        })
        with pytest.warns(InternalWarning, match="Falling back to 'bar' style"):
            fig = plot_importance(df, style="boxplot")
        assert fig is not None
