"""Tests for feature group coloring across visualization functions.

63 tests across 12 classes covering resolve_feature_group_colors, all plot
functions with groups, aggregation, auto-detection, partial mappings, edge
cases, and helper functions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv.viz._palettes import (
    QUALITATIVE_PALETTE,
    resolve_feature_group_colors,
)
from mlsurv.viz.features import (
    plot_coefficients,
    plot_feature_correlation,
    plot_hazard_ratios,
    plot_hazard_ratios_comparison,
    plot_importance,
    plot_importance_heatmap,
)
from mlsurv.viz.shap import (
    plot_shap_beeswarm,
    plot_shap_importance,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

FEATURES = ["age", "bmi", "wbc", "albumin", "crp"]
GROUPS_DICT = {
    "age": "demographic",
    "bmi": "demographic",
    "wbc": "clinical",
    "albumin": "clinical",
    "crp": "clinical",
}
GROUPS_LIST = ["demographic", "demographic", "clinical", "clinical", "clinical"]


@pytest.fixture
def importance_df():
    """Single-model importance DataFrame with 5 features."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "feature": FEATURES,
        "perm_mean": rng.uniform(0.01, 0.5, 5),
        "perm_std": rng.uniform(0.001, 0.05, 5),
    })


@pytest.fixture
def multi_model_df():
    """Multi-model importance DataFrame."""
    rng = np.random.RandomState(42)
    rows = []
    for model in ["coxph", "rsf"]:
        for feat in FEATURES:
            rows.append({
                "feature": feat,
                "model": model,
                "perm_mean": rng.uniform(0.01, 0.5),
                "perm_std": rng.uniform(0.001, 0.05),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def iterations_df():
    """Per-iteration importance DataFrame for raincloud/boxplot."""
    rng = np.random.RandomState(42)
    rows = []
    for feat in FEATURES:
        for _ in range(20):
            rows.append({"feature": feat, "importance": rng.uniform(0.01, 0.5)})
    return pd.DataFrame(rows)


@pytest.fixture
def hr_df():
    """Hazard ratio DataFrame."""
    rng = np.random.RandomState(42)
    hrs = np.exp(rng.uniform(-1, 1, 5))
    return pd.DataFrame({
        "feature": FEATURES,
        "hr": hrs,
        "ci_lower": hrs * 0.7,
        "ci_upper": hrs * 1.3,
    })


@pytest.fixture
def shap_df():
    """Long-format SHAP DataFrame."""
    rng = np.random.RandomState(42)
    rows = []
    for feat in FEATURES:
        for i in range(30):
            rows.append({
                "feature": feat,
                "shap_value": rng.normal(0, 0.1),
                "sample_idx": i,
                "feature_value": rng.uniform(0, 1),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def feature_matrix():
    """Numeric feature DataFrame for correlation."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        rng.randn(50, 5),
        columns=FEATURES,
    )


# ══════════════════════════════════════════════════════════════════════════════
# resolve_feature_group_colors()
# ══════════════════════════════════════════════════════════════════════════════


class TestResolveFeatureGroupColors:
    """Tests for resolve_feature_group_colors()."""

    def test_dict_input(self):
        f2g, g2c = resolve_feature_group_colors(GROUPS_DICT, FEATURES)
        assert f2g == GROUPS_DICT
        assert "demographic" in g2c
        assert "clinical" in g2c

    def test_list_input(self):
        f2g, g2c = resolve_feature_group_colors(GROUPS_LIST, FEATURES)
        assert f2g["age"] == "demographic"
        assert f2g["crp"] == "clinical"
        assert len(g2c) == 2

    def test_series_input(self):
        s = pd.Series(GROUPS_LIST)
        f2g, g2c = resolve_feature_group_colors(s, FEATURES)
        assert f2g["bmi"] == "demographic"

    def test_length_mismatch_list(self):
        with pytest.raises(ValueError, match="length"):
            resolve_feature_group_colors(["a", "b"], FEATURES)

    def test_length_mismatch_series(self):
        with pytest.raises(ValueError, match="length"):
            resolve_feature_group_colors(pd.Series(["a"]), FEATURES)

    def test_all_groups_get_qualitative_palette_colors(self):
        f2g, g2c = resolve_feature_group_colors(GROUPS_DICT, FEATURES)
        # All groups assigned from QUALITATIVE_PALETTE (sorted alphabetically)
        sorted_groups = sorted(g2c.keys())
        for i, grp in enumerate(sorted_groups):
            assert g2c[grp] == QUALITATIVE_PALETTE[i]

    def test_unknown_names_get_qualitative_palette(self):
        custom = {"age": "bloodwork", "bmi": "bloodwork", "wbc": "imaging_new"}
        f2g, g2c = resolve_feature_group_colors(custom, FEATURES)
        assert g2c["bloodwork"] == QUALITATIVE_PALETTE[0]
        assert g2c["imaging_new"] == QUALITATIVE_PALETTE[1]

    def test_explicit_group_colors_override(self):
        overrides = {"clinical": "#ff0000"}
        f2g, g2c = resolve_feature_group_colors(
            GROUPS_DICT, FEATURES, group_colors=overrides,
        )
        assert g2c["clinical"] == "#ff0000"
        # demographic not overridden — uses QUALITATIVE_PALETTE
        assert g2c["demographic"] in QUALITATIVE_PALETTE

    def test_features_missing_from_dict_excluded(self):
        partial = {"age": "demo", "wbc": "lab"}
        f2g, g2c = resolve_feature_group_colors(partial, FEATURES)
        assert "bmi" not in f2g
        assert "albumin" not in f2g
        assert len(f2g) == 2

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="dict, list, or Series"):
            resolve_feature_group_colors(42, FEATURES)

    def test_case_preserved_in_group_names(self):
        custom = {"age": "Clinical", "bmi": "GENOMIC"}
        f2g, g2c = resolve_feature_group_colors(custom, FEATURES)
        # Group names are preserved as-is; colours from QUALITATIVE_PALETTE
        assert "Clinical" in g2c
        assert "GENOMIC" in g2c
        assert g2c["Clinical"] in QUALITATIVE_PALETTE
        assert g2c["GENOMIC"] in QUALITATIVE_PALETTE

    def test_color_cycling_many_groups(self):
        n = 15
        feats = [f"f{i}" for i in range(n)]
        groups = {f"f{i}": f"grp{i}" for i in range(n)}
        f2g, g2c = resolve_feature_group_colors(groups, feats)
        assert len(g2c) == n
        # Groups beyond palette length should cycle
        colors = list(g2c.values())
        assert colors[0] == QUALITATIVE_PALETTE[0]
        assert colors[12] == QUALITATIVE_PALETTE[0]  # cycle at len(QUALITATIVE_PALETTE)


# ══════════════════════════════════════════════════════════════════════════════
# plot_importance() with groups
# ══════════════════════════════════════════════════════════════════════════════


class TestPlotImportanceGroups:
    """Tests for feature group support in plot_importance()."""

    def test_bar_traces_per_group(self, importance_df):
        fig = plot_importance(
            importance_df, style="bar", n=5,
            feature_groups=GROUPS_DICT,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        group_names = {t.name for t in bar_traces}
        assert "demographic" in group_names
        assert "clinical" in group_names

    def test_bar_correct_colors(self, importance_df):
        fig = plot_importance(
            importance_df, style="bar", n=5,
            feature_groups=GROUPS_DICT,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        for t in bar_traces:
            if t.name == "clinical":
                assert t.marker.color in QUALITATIVE_PALETTE

    def test_forest_dots_colored_by_group(self, importance_df):
        fig = plot_importance(
            importance_df, style="forest", n=5,
            feature_groups=GROUPS_DICT,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.name in ("clinical", "demographic")]
        assert len(scatter_traces) >= 2

    def test_raincloud_per_feature_coloring(self, importance_df, iterations_df):
        fig = plot_importance(
            importance_df, style="raincloud", n=5,
            feature_groups=GROUPS_DICT,
            iterations_df=iterations_df,
        )
        assert isinstance(fig, go.Figure)
        # Should have legend traces for groups
        legend_names = {t.name for t in fig.data if t.showlegend}
        assert "clinical" in legend_names
        assert "demographic" in legend_names

    def test_boxplot_per_feature_coloring(self, importance_df, iterations_df):
        fig = plot_importance(
            importance_df, style="boxplot", n=5,
            feature_groups=GROUPS_DICT,
            iterations_df=iterations_df,
        )
        assert isinstance(fig, go.Figure)

    def test_groups_override_coef_col(self, importance_df):
        importance_df["coefficient"] = [0.5, -0.3, 0.1, -0.8, 0.2]
        fig = plot_importance(
            importance_df, style="bar", n=5,
            coef_col="coefficient",
            feature_groups=GROUPS_DICT,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        # Groups should win: names should be group names, not coef signs
        group_names = {t.name for t in bar_traces}
        assert "clinical" in group_names
        assert "demographic" in group_names
        assert "Positive (HR>1)" not in group_names

    def test_multi_model_forest_group_strip(self, multi_model_df):
        fig = plot_importance(
            multi_model_df, style="forest", n=5,
            feature_groups=GROUPS_DICT,
        )
        # Should have a group color strip heatmap trace instead of dot ticks
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1, "forest with groups should have exactly 1 color strip heatmap"
        # No axis should have dot-prefixed tick labels
        for attr in dir(fig.layout):
            if attr.startswith(("xaxis", "yaxis")):
                ax = getattr(fig.layout, attr)
                if ax and ax.ticktext is not None:
                    assert not any("\u25cf" in str(t) for t in ax.ticktext), f"dot found in {attr}"

    def test_facet_by_group(self, importance_df):
        fig = plot_importance(
            importance_df, style="bar", n=5,
            feature_groups=GROUPS_DICT,
            facet_by_group=True,
        )
        # Should be a subplots figure
        assert isinstance(fig, go.Figure)
        # Check that we have bars
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 2

    def test_without_groups_unchanged(self, importance_df):
        fig = plot_importance(importance_df, style="bar", n=5)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        # Default coloring — no group names
        for t in bar_traces:
            assert t.name not in ("clinical", "demographic")

    def test_list_input(self, importance_df):
        groups = [GROUPS_DICT[f] for f in importance_df["feature"]]
        fig = plot_importance(
            importance_df, style="bar", n=5,
            feature_groups=groups,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        group_names = {t.name for t in bar_traces}
        assert "clinical" in group_names


# ══════════════════════════════════════════════════════════════════════════════
# plot_importance_heatmap() with groups
# ══════════════════════════════════════════════════════════════════════════════


class TestPlotImportanceHeatmapGroups:
    """Tests for feature group support in plot_importance_heatmap()."""

    def test_group_strip_present(self, multi_model_df):
        fig = plot_importance_heatmap(
            multi_model_df, n=5,
            feature_groups=GROUPS_DICT,
        )
        # Should have multiple heatmap traces (strip + main)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) >= 2

    def test_colors_match_resolution(self, multi_model_df):
        fig = plot_importance_heatmap(
            multi_model_df, n=5,
            feature_groups=GROUPS_DICT,
        )
        # Should have legend traces for groups
        legend_names = {t.name for t in fig.data if getattr(t, 'showlegend', False)}
        assert "clinical" in legend_names
        assert "demographic" in legend_names

    def test_without_groups_no_subplot(self, multi_model_df):
        fig = plot_importance_heatmap(multi_model_df, n=5)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1


# ══════════════════════════════════════════════════════════════════════════════
# plot_hazard_ratios() with groups
# ══════════════════════════════════════════════════════════════════════════════


class TestPlotHazardRatiosGroups:
    """Tests for feature group support in plot_hazard_ratios()."""

    def test_risk_coloring_kept_with_groups(self, hr_df):
        """With groups, markers keep risk-category coloring; strip shows groups."""
        fig = plot_hazard_ratios(
            hr_df, n_features=5,
            feature_groups=GROUPS_DICT,
        )
        scatter_names = {t.name for t in fig.data if isinstance(t, go.Scatter)}
        # Risk categories should be present (not replaced by group names)
        assert any(n in scatter_names for n in ("Higher risk", "Lower risk", "N.S."))

    def test_group_strip_present_with_groups(self, hr_df):
        """With groups, a Heatmap strip trace should be added."""
        fig = plot_hazard_ratios(
            hr_df, n_features=5,
            feature_groups=GROUPS_DICT,
        )
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_group_legend_traces_present(self, hr_df):
        """With groups, group legend traces should appear alongside risk traces."""
        fig = plot_hazard_ratios(
            hr_df, n_features=5,
            feature_groups=GROUPS_DICT,
        )
        legend_names = {t.name for t in fig.data if getattr(t, 'showlegend', False)}
        assert "clinical" in legend_names
        assert "demographic" in legend_names

    def test_without_groups_hr_direction(self, hr_df):
        fig = plot_hazard_ratios(hr_df, n_features=5)
        scatter_names = {t.name for t in fig.data if isinstance(t, go.Scatter)}
        # Should have HR-direction names
        assert any(n in scatter_names for n in ("Higher risk", "Lower risk", "N.S."))
        # No heatmap strip
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 0


# ══════════════════════════════════════════════════════════════════════════════
# plot_shap_importance() with groups
# ══════════════════════════════════════════════════════════════════════════════


class TestPlotShapImportanceGroups:
    """Tests for feature group support in plot_shap_importance()."""

    def test_traces_colored_by_group_violin(self, shap_df):
        fig = plot_shap_importance(
            shap_df, n=5, show_violin=True,
            feature_groups=GROUPS_DICT,
        )
        # Should have legend traces for groups
        legend_names = {t.name for t in fig.data if getattr(t, 'showlegend', False)}
        assert "clinical" in legend_names
        assert "demographic" in legend_names

    def test_traces_colored_by_group_bar(self, shap_df):
        fig = plot_shap_importance(
            shap_df, n=5, show_violin=False,
            feature_groups=GROUPS_DICT,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        bar_names = {t.name for t in bar_traces}
        assert "clinical" in bar_names
        assert "demographic" in bar_names

    def test_without_groups_uniform_blue(self, shap_df):
        fig = plot_shap_importance(shap_df, n=5, show_violin=False)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 1
        assert bar_traces[0].marker.color == '#1f77b4'


# ══════════════════════════════════════════════════════════════════════════════
# plot_shap_beeswarm() with groups
# ══════════════════════════════════════════════════════════════════════════════


class TestPlotShapBeeswarmGroups:
    """Tests for feature group support in plot_shap_beeswarm()."""

    def test_beeswarm_group_strip(self, shap_df):
        fig = plot_shap_beeswarm(
            shap_df, n=5,
            feature_groups=GROUPS_DICT,
        )
        # Should have a group color strip heatmap trace instead of dot ticks
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1, "beeswarm with groups should have exactly 1 color strip heatmap"
        # No axis should have dot-prefixed tick labels
        for attr in dir(fig.layout):
            if attr.startswith(("xaxis", "yaxis")):
                ax = getattr(fig.layout, attr)
                if ax and ax.ticktext is not None:
                    assert not any("\u25cf" in str(t) for t in ax.ticktext), f"dot found in {attr}"

    def test_scatter_colors_encode_feature_value(self, shap_df):
        fig = plot_shap_beeswarm(
            shap_df, n=5,
            feature_groups=GROUPS_DICT,
        )
        # Scatter traces should still use RdBu_r colorscale
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)
                         and t.marker is not None
                         and getattr(t.marker, 'colorscale', None) is not None]
        # At least the main beeswarm traces should have colorscale
        assert len(scatter_traces) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# plot_feature_correlation() with groups
# ══════════════════════════════════════════════════════════════════════════════


class TestPlotFeatureCorrelationGroups:
    """Tests for feature group support in plot_feature_correlation()."""

    def test_subplot_structure(self, feature_matrix):
        fig = plot_feature_correlation(
            feature_matrix, max_features=5,
            feature_groups=GROUPS_DICT,
        )
        # Should have 3 heatmap traces: top strip, left strip, main
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 3

    def test_color_strips_present(self, feature_matrix):
        fig = plot_feature_correlation(
            feature_matrix, max_features=5,
            feature_groups=GROUPS_DICT,
        )
        # Legend should have group names
        legend_names = {t.name for t in fig.data if getattr(t, 'showlegend', False)}
        assert "clinical" in legend_names
        assert "demographic" in legend_names

    def test_without_groups_simple(self, feature_matrix):
        fig = plot_feature_correlation(feature_matrix, max_features=5)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1


# ══════════════════════════════════════════════════════════════════════════════
# plot_feature_correlation() label truncation & adaptive margins
# ══════════════════════════════════════════════════════════════════════════════


class TestPlotFeatureCorrelationLabelOverflow:
    """Labels are truncated and margins adapt to long feature names."""

    @pytest.fixture()
    def long_name_matrix(self):
        """DataFrame with 6 features whose names exceed 30 characters."""
        rng = np.random.default_rng(42)
        cols = [f"very_long_feature_name_number_{i}_extra_padding" for i in range(6)]
        return pd.DataFrame(rng.random((30, 6)), columns=cols)

    def test_labels_truncated_no_groups(self, long_name_matrix):
        fig = plot_feature_correlation(long_name_matrix, max_features=6)
        x_ticktext = fig.layout.xaxis.ticktext
        assert x_ticktext is not None
        for label in x_ticktext:
            assert len(label) <= 30

    def test_labels_truncated_with_groups(self, long_name_matrix):
        cols = long_name_matrix.columns.tolist()
        groups = {c: "G1" if i < 3 else "G2" for i, c in enumerate(cols)}
        fig = plot_feature_correlation(
            long_name_matrix, max_features=6, feature_groups=groups,
        )
        # Bottom strip x-axis (row=2, col=2) → xaxis4 in 2x2 subplot
        bottom_x = fig.layout.xaxis4
        assert bottom_x.ticktext is not None
        for label in bottom_x.ticktext:
            assert len(label) <= 30

    def test_adaptive_margin_scales_with_name_length(self, long_name_matrix):
        fig = plot_feature_correlation(long_name_matrix, max_features=6)
        # With 30-char truncated labels at per_char=5, margin should be > 60
        assert fig.layout.margin.l > 60
        assert fig.layout.margin.b > 60

    def test_short_names_get_smaller_margin(self):
        rng = np.random.default_rng(42)
        X_short = pd.DataFrame(rng.random((30, 4)), columns=["a", "b", "c", "d"])
        fig = plot_feature_correlation(X_short, max_features=4)
        # Short names → margin should be at the minimum (60)
        assert fig.layout.margin.l == 60
        assert fig.layout.margin.b == 60

    def test_full_names_in_tickvals(self, long_name_matrix):
        fig = plot_feature_correlation(long_name_matrix, max_features=6)
        # tickvals should contain the full (untruncated) feature names
        x_tickvals = fig.layout.xaxis.tickvals
        assert x_tickvals is not None
        for val in x_tickvals:
            assert "very_long_feature_name" in val


# ══════════════════════════════════════════════════════════════════════════════
# group_agg in plot_importance()
# ══════════════════════════════════════════════════════════════════════════════


class TestGroupAgg:
    """Tests for group_agg=True in plot_importance()."""

    def test_bar_one_per_group(self, importance_df):
        fig = plot_importance(
            importance_df, style="bar", n=5,
            feature_groups=GROUPS_DICT,
            group_agg=True,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1
        # y-values should be group names
        all_y = []
        for t in bar_traces:
            if t.y is not None:
                all_y.extend(list(t.y))
        assert "clinical" in all_y or "demographic" in all_y

    def test_boxplot_one_per_group(self, importance_df):
        fig = plot_importance(
            importance_df, style="boxplot", n=5,
            feature_groups=GROUPS_DICT,
            group_agg=True,
        )
        box_traces = [t for t in fig.data if isinstance(t, go.Box)]
        assert len(box_traces) >= 1

    def test_group_agg_without_groups_raises(self, importance_df):
        with pytest.raises(ValueError, match="group_agg.*requires"):
            plot_importance(
                importance_df, style="bar", n=5,
                group_agg=True,
            )

    def test_colors_match_resolution(self, importance_df):
        custom_colors = {"clinical": "#ff0000", "demographic": "#00ff00"}
        fig = plot_importance(
            importance_df, style="bar", n=5,
            feature_groups=GROUPS_DICT,
            group_colors=custom_colors,
            group_agg=True,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        colors = []
        for t in bar_traces:
            if t.marker and t.marker.color:
                if isinstance(t.marker.color, (list, tuple)):
                    colors.extend(t.marker.color)
                else:
                    colors.append(t.marker.color)
        assert "#ff0000" in colors
        assert "#00ff00" in colors


# ══════════════════════════════════════════════════════════════════════════════
# Auto-detection of groups
# ══════════════════════════════════════════════════════════════════════════════


class TestAutoDetection:
    """Tests for auto-detection of feature groups from DataFrame columns."""

    def test_group_column_auto_detected(self, importance_df):
        df = importance_df.copy()
        df["group"] = [GROUPS_DICT[f] for f in df["feature"]]
        fig = plot_importance(df, style="bar", n=5)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        group_names = {t.name for t in bar_traces}
        assert "clinical" in group_names or "demographic" in group_names

    def test_feature_group_column_auto_detected(self, importance_df):
        df = importance_df.copy()
        df["feature_group"] = [GROUPS_DICT[f] for f in df["feature"]]
        fig = plot_importance(df, style="bar", n=5)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        group_names = {t.name for t in bar_traces}
        assert "clinical" in group_names or "demographic" in group_names

    def test_heatmap_auto_detected(self, multi_model_df):
        df = multi_model_df.copy()
        df["group"] = [GROUPS_DICT[f] for f in df["feature"]]
        fig = plot_importance_heatmap(df, n=5)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# Partial group mapping (ungrouped features)
# ══════════════════════════════════════════════════════════════════════════════


class TestPartialGroupMapping:
    """Tests that ungrouped features are not silently dropped."""

    def test_bar_partial_groups_includes_ungrouped(self, importance_df):
        partial = {"age": "demo", "wbc": "lab"}
        fig = plot_importance(
            importance_df, style="bar", n=5,
            feature_groups=partial,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        all_y = []
        for t in bar_traces:
            if t.y is not None:
                all_y.extend(list(t.y))
        # All 5 features should be present (3 ungrouped + 2 grouped)
        assert len(set(all_y)) == 5
        # Should have an "Ungrouped" trace
        trace_names = {t.name for t in bar_traces}
        assert "Ungrouped" in trace_names

    def test_forest_partial_groups_includes_ungrouped(self, importance_df):
        partial = {"age": "demo"}
        fig = plot_importance(
            importance_df, style="forest", n=5,
            feature_groups=partial,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)
                         and t.name in ("demo", "Ungrouped")]
        assert len(scatter_traces) >= 2

    def test_hr_partial_groups_has_strip(self, hr_df):
        partial = {"age": "demo"}
        fig = plot_hazard_ratios(
            hr_df, n_features=5,
            feature_groups=partial,
        )
        # Markers keep risk-category coloring even with groups
        scatter_names = {t.name for t in fig.data if isinstance(t, go.Scatter)}
        assert any(n in scatter_names for n in ("Higher risk", "Lower risk", "N.S."))
        # Should have a group strip heatmap
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_shap_bar_partial_groups_includes_ungrouped(self, shap_df):
        partial = {"age": "demo", "wbc": "lab"}
        fig = plot_shap_importance(
            shap_df, n=5, show_violin=False,
            feature_groups=partial,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        trace_names = {t.name for t in bar_traces}
        assert "Ungrouped" in trace_names


# ══════════════════════════════════════════════════════════════════════════════
# Edge cases from review findings
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases found during review."""

    def test_non_string_group_values_filtered(self):
        """Non-string group values (None, NaN) are filtered with warning."""
        groups = {"age": "demo", "bmi": None, "wbc": "lab"}
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f2g, g2c = resolve_feature_group_colors(groups, FEATURES)
        assert "bmi" not in f2g
        assert len(f2g) == 2
        assert any("non-string" in str(x.message) for x in w)

    def test_nan_group_values_filtered(self):
        """float('nan') group values are filtered out."""
        groups = {"age": "demo", "bmi": float("nan"), "wbc": "lab"}
        f2g, g2c = resolve_feature_group_colors(groups, FEATURES)
        assert "bmi" not in f2g
        assert len(f2g) == 2

    def test_group_agg_no_match_raises(self, importance_df):
        """group_agg with no matching features raises ValueError."""
        # Features in df are FEATURES, but groups dict maps different names
        groups = {"x1": "grp1", "x2": "grp2"}
        with pytest.raises(ValueError, match="No features matched"):
            plot_importance(
                importance_df, style="bar", n=5,
                feature_groups=groups,
                group_agg=True,
            )

    def test_no_phantom_legend_entries(self):
        """grp_to_color should not contain groups not in top features."""
        # Use 10 features but only display top 3
        feats = [f"f{i}" for i in range(10)]
        groups = {f"f{i}": f"grp{i}" for i in range(10)}
        df = pd.DataFrame({
            "feature": feats,
            "perm_mean": list(range(10)),
            "perm_std": [0.01] * 10,
        })
        fig = plot_importance(df, style="bar", n=3, feature_groups=groups)
        # Only 3 features displayed, so at most 3 group legend entries
        legend_traces = [t for t in fig.data if getattr(t, "showlegend", False)]
        assert len(legend_traces) <= 3


# ══════════════════════════════════════════════════════════════════════════════
# Helper function unit tests
# ══════════════════════════════════════════════════════════════════════════════


class TestHelperFunctions:
    """Direct unit tests for internal helper functions."""

    def test_resolve_groups_none_passthrough(self):
        from mlsurv.viz._palettes import _resolve_groups
        f2g, g2c = _resolve_groups(None, ["a", "b", "c"])
        assert f2g is None
        assert g2c is None

    def test_build_discrete_colorscale_boundaries(self):
        from mlsurv.viz._palettes import _build_discrete_colorscale
        grp_to_color = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        groups, idx_map, cs = _build_discrete_colorscale(grp_to_color)
        assert groups == ["A", "B", "C"]
        assert idx_map == {"A": 0, "B": 1, "C": 2}
        # 3 groups -> 6 colorscale entries (lo, hi per group)
        assert len(cs) == 6
        assert cs[0][0] == pytest.approx(0.0)
        assert cs[1][0] == pytest.approx(1 / 3)
        assert cs[4][0] == pytest.approx(2 / 3)
        assert cs[5][0] == pytest.approx(1.0)
        assert cs[0][1] == "#ff0000"
        assert cs[2][1] == "#00ff00"
        assert cs[4][1] == "#0000ff"

    def test_narrow_groups(self):
        from mlsurv.viz._palettes import _narrow_groups
        f2g = {"a": "g1", "b": "g2", "c": "g1"}
        g2c = {"g1": "#ff0000", "g2": "#00ff00"}
        nf, nc = _narrow_groups(f2g, g2c, ["a", "c"])
        assert nf == {"a": "g1", "c": "g1"}
        assert nc == {"g1": "#ff0000"}
        # g2 is excluded since no features in subset belong to it

    def test_narrow_groups_none_passthrough(self):
        from mlsurv.viz._palettes import _narrow_groups
        f2g, g2c = _narrow_groups(None, None, ["a", "b"])
        assert f2g is None
        assert g2c is None

    def test_make_group_strip_trace_vertical(self):
        """Vertical strip produces column-vector Z with correct group indices."""
        from mlsurv.viz._palettes import _make_group_strip_trace
        features = ["a", "b", "c"]
        f2g = {"a": "G1", "b": "G2", "c": "G1"}
        g2c = {"G1": "#ff0000", "G2": "#00ff00"}
        trace = _make_group_strip_trace(features, f2g, g2c, orientation="vertical")
        assert isinstance(trace, go.Heatmap)
        assert list(trace.y) == ["a", "b", "c"]
        assert list(trace.x) == ["Group"]
        # Z encodes group index: G1=0, G2=1 (Plotly stores as tuples)
        assert [list(row) for row in trace.z] == [[0], [1], [0]]
        assert [list(row) for row in trace.customdata] == [["G1"], ["G2"], ["G1"]]

    def test_make_group_strip_trace_horizontal(self):
        """Horizontal strip produces single-row Z."""
        from mlsurv.viz._palettes import _make_group_strip_trace
        features = ["a", "b"]
        f2g = {"a": "G1", "b": "G1"}
        g2c = {"G1": "#ff0000"}
        trace = _make_group_strip_trace(features, f2g, g2c, orientation="horizontal")
        assert isinstance(trace, go.Heatmap)
        assert list(trace.x) == ["a", "b"]
        assert [list(row) for row in trace.z] == [[0, 0]]

    def test_make_group_strip_trace_ungrouped(self):
        """Ungrouped features get an 'Ungrouped' bucket."""
        from mlsurv.viz._palettes import _make_group_strip_trace
        features = ["a", "b", "c"]
        f2g = {"a": "G1"}
        g2c = {"G1": "#ff0000"}
        trace = _make_group_strip_trace(features, f2g, g2c, orientation="vertical")
        # b and c are ungrouped
        assert trace.customdata[0] == ["G1"]
        assert trace.customdata[1] == ["Ungrouped"]
        assert trace.customdata[2] == ["Ungrouped"]

    def test_dict_total_miss_warns(self):
        """Dict with no matching feature names emits a warning."""
        import warnings
        groups = {"x1": "grp1", "x2": "grp2"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f2g, g2c = resolve_feature_group_colors(groups, FEATURES)
        assert len(f2g) == 0
        assert any("none matched" in str(x.message).lower() for x in w)

    def test_multi_model_bar_group_strip(self, multi_model_df):
        """Multi-model bar style uses a bottom color strip instead of dot ticks."""
        fig = plot_importance(
            multi_model_df, style="bar", n=5,
            feature_groups=GROUPS_DICT,
        )
        # Should have a group color strip heatmap trace
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1, "bar with groups should have exactly 1 color strip heatmap"
        # No axis should have dot-prefixed tick labels
        for attr in dir(fig.layout):
            if attr.startswith(("xaxis", "yaxis")):
                ax = getattr(fig.layout, attr)
                if ax and ax.ticktext is not None:
                    assert not any("\u25cf" in str(t) for t in ax.ticktext), f"dot found in {attr}"

    def test_iter_group_segments_ordering_and_ungrouped(self):
        """_iter_group_segments returns first-occurrence ordered segments + trailing Ungrouped."""
        from mlsurv.viz._palettes import _iter_group_segments
        features = ["a", "b", "c", "d", "e"]
        f2g = {"a": "G1", "b": "G2", "d": "G1"}
        g2c = {"G1": "#ff0000", "G2": "#00ff00"}
        segments = _iter_group_segments(features, f2g, g2c)
        assert segments[0] == ("G1", "#ff0000", ["a", "d"])
        assert segments[1] == ("G2", "#00ff00", ["b"])
        assert segments[2] == ("Ungrouped", "#7f7f7f", ["c", "e"])

    def test_iter_group_segments_all_grouped(self):
        """_iter_group_segments with no ungrouped features produces no Ungrouped segment."""
        from mlsurv.viz._palettes import _iter_group_segments
        features = ["a", "b"]
        f2g = {"a": "G1", "b": "G1"}
        g2c = {"G1": "#ff0000"}
        segments = _iter_group_segments(features, f2g, g2c)
        assert len(segments) == 1
        assert segments[0][0] == "G1"


# ══════════════════════════════════════════════════════════════════════════════
# legendgrouptitle only on first trace
# ══════════════════════════════════════════════════════════════════════════════


class TestLegendGroupTitle:
    """legendgrouptitle should only appear on the first trace."""

    def test_legendgrouptitle_on_exactly_one_trace(self, importance_df):
        """Only the first group legend trace should have legendgrouptitle."""
        plot_importance(
            importance_df, style="bar", n=5,
            feature_groups=GROUPS_DICT,
        )
        from mlsurv.viz._palettes import _add_group_legend_traces
        # Create a standalone figure to test the helper directly
        test_fig = go.Figure()
        g2c = {"clinical": "#1f77b4", "demographic": "#ff7f0e"}
        _add_group_legend_traces(test_fig, g2c)
        traces_with_title = [
            t for t in test_fig.data
            if getattr(t, "legendgrouptitle", None) is not None
            and getattr(t.legendgrouptitle, "text", None) is not None
        ]
        assert len(traces_with_title) == 1
        assert traces_with_title[0].legendgrouptitle.text == "Feature Group"

    def test_subsequent_traces_no_legendgrouptitle(self):
        """Traces after the first should not have legendgrouptitle at all."""
        from mlsurv.viz._palettes import _add_group_legend_traces
        test_fig = go.Figure()
        g2c = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        _add_group_legend_traces(test_fig, g2c)
        for i, trace in enumerate(test_fig.data):
            if i == 0:
                assert trace.legendgrouptitle.text == "Feature Group"
            else:
                # Should be None or missing entirely (not explicitly set to None)
                title = getattr(trace, "legendgrouptitle", None)
                if title is not None:
                    assert getattr(title, "text", None) is None


# ══════════════════════════════════════════════════════════════════════════════
# plot_coefficients() with groups — strip + risk coloring
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def coef_df():
    """Single-model coefficient DataFrame."""
    return pd.DataFrame({
        "feature": FEATURES,
        "coefficient": [0.5, -0.3, 0.1, -0.8, 0.2],
        "coef_std": [0.05, 0.03, 0.02, 0.08, 0.04],
    })


@pytest.fixture
def multi_coef_df():
    """Multi-model coefficient DataFrame."""
    rows = []
    for model in ["coxph", "coxnet"]:
        for i, feat in enumerate(FEATURES):
            rows.append({
                "feature": feat,
                "model": model,
                "coefficient": [0.5, -0.3, 0.1, -0.8, 0.2][i] * (1.1 if model == "coxnet" else 1.0),
            })
    return pd.DataFrame(rows)


class TestPlotCoefficientsGroups:
    """Coefficient plots keep risk coloring + add group strip."""

    def test_single_model_group_coloring_with_groups(self, coef_df):
        """Single-model + groups should use group colors, not risk coloring."""
        fig = plot_coefficients(
            coef_df, n=5, feature_groups=GROUPS_DICT,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1
        # Should have group-named bar traces
        bar_names = {t.name for t in bar_traces if t.name}
        assert "clinical" in bar_names or "demographic" in bar_names

    def test_single_model_group_strip_with_groups(self, coef_df):
        """Single-model + groups should have a Heatmap color strip."""
        fig = plot_coefficients(
            coef_df, n=5, feature_groups=GROUPS_DICT,
        )
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_single_model_group_legend_traces(self, coef_df):
        """Single-model + groups should have group legend from data traces."""
        fig = plot_coefficients(
            coef_df, n=5, feature_groups=GROUPS_DICT,
        )
        legend_names = {t.name for t in fig.data
                        if getattr(t, 'showlegend', None) is True}
        assert "clinical" in legend_names
        assert "demographic" in legend_names

    def test_without_groups_no_strip(self, coef_df):
        """Without groups, no heatmap strip should be present."""
        fig = plot_coefficients(coef_df, n=5)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 0


# ══════════════════════════════════════════════════════════════════════════════
# plot_hazard_ratios_comparison() with groups — strip
# ══════════════════════════════════════════════════════════════════════════════


class TestPlotHRComparisonGroups:
    """Tests for HR comparison plots with groups."""

    def test_comparison_group_strip(self, hr_df):
        """HR comparison with groups should have a Heatmap strip."""
        models = {"model_a": hr_df, "model_b": hr_df.copy()}
        fig = plot_hazard_ratios_comparison(
            models, n_features=5,
            feature_groups=GROUPS_DICT,
        )
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_comparison_group_legend(self, hr_df):
        """HR comparison with groups should have group legend traces."""
        models = {"model_a": hr_df, "model_b": hr_df.copy()}
        fig = plot_hazard_ratios_comparison(
            models, n_features=5,
            feature_groups=GROUPS_DICT,
        )
        legend_names = {t.name for t in fig.data if getattr(t, 'showlegend', False)}
        assert "clinical" in legend_names
        assert "demographic" in legend_names

    def test_comparison_without_groups_no_strip(self, hr_df):
        """HR comparison without groups: no heatmap strip."""
        models = {"model_a": hr_df, "model_b": hr_df.copy()}
        fig = plot_hazard_ratios_comparison(
            models, n_features=5,
        )
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 0


# ══════════════════════════════════════════════════════════════════════════════
# multi-model coefficients with groups
# ══════════════════════════════════════════════════════════════════════════════


class TestPlotCoefficientsMultiModelGroups:
    """Multi-model coefficient plots with groups use strip + model coloring."""

    def test_multi_model_group_strip(self, multi_coef_df):
        fig = plot_coefficients(
            multi_coef_df, n=5, feature_groups=GROUPS_DICT,
        )
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_multi_model_group_legend(self, multi_coef_df):
        fig = plot_coefficients(
            multi_coef_df, n=5, feature_groups=GROUPS_DICT,
        )
        legend_names = {t.name for t in fig.data if getattr(t, 'showlegend', False)}
        assert "clinical" in legend_names
        assert "demographic" in legend_names


# ══════════════════════════════════════════════════════════════════════════════
# CV summary fallback from self._state.models
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildCvSummaryFromModels:
    """Tests for build_cv_summary_from_models fallback path."""

    @staticmethod
    def _build(models_dict, metric, model_names):
        """Call the module-level function directly."""
        from mlsurv.viz._data_metrics import (
            build_cv_summary_from_models,
        )
        return build_cv_summary_from_models(models_dict, metric, model_names)

    def test_cv_metrics_dict_path(self):
        """Path A: metric found in cv_metrics dict."""
        from types import SimpleNamespace

        result = SimpleNamespace(
            cv_metrics={"c_index": {"mean": 0.75, "std": 0.03}},
            scoring_metric="c_index",
        )
        df = self._build(
            {"coxph": result}, "c_index", None,
        )
        assert len(df) == 1
        assert df.iloc[0]["cv_metric_mean"] == pytest.approx(0.75)
        assert df.iloc[0]["cv_metric_std"] == pytest.approx(0.03)

    def test_scoring_metric_fallback(self):
        """Path B: metric == scoring_metric, uses cv_metric_mean/std attrs."""
        from types import SimpleNamespace

        result = SimpleNamespace(
            cv_metrics={},
            scoring_metric="c_index",
            cv_metric_mean=0.80,
            cv_metric_std=0.05,
        )
        df = self._build(
            {"rsf": result}, "c_index", None,
        )
        assert len(df) == 1
        assert df.iloc[0]["cv_metric_mean"] == pytest.approx(0.80)

    def test_no_data_raises(self):
        """All models skipped -> ValueError."""
        from types import SimpleNamespace

        result = SimpleNamespace(
            cv_metrics={},
            scoring_metric="brier_score",
        )
        with pytest.raises(ValueError, match="No CV summary data"):
            self._build(
                {"coxph": result}, "c_index", None,
            )

    def test_std_none_defaults_nan(self):
        """std_val is None -> NaN in output."""
        from types import SimpleNamespace

        result = SimpleNamespace(
            cv_metrics={"c_index": {"mean": 0.70}},
            scoring_metric="c_index",
        )
        df = self._build(
            {"coxph": result}, "c_index", None,
        )
        assert pd.isna(df.iloc[0]["cv_metric_std"])

    def test_model_names_filter(self):
        """model_names filters to requested models only."""
        from types import SimpleNamespace

        r1 = SimpleNamespace(cv_metrics={"c_index": {"mean": 0.75, "std": 0.03}}, scoring_metric="c_index")
        r2 = SimpleNamespace(cv_metrics={"c_index": {"mean": 0.80, "std": 0.02}}, scoring_metric="c_index")
        df = self._build(
            {"coxph": r1, "rsf": r2}, "c_index", ["coxph"],
        )
        assert len(df) == 1
        assert df.iloc[0]["model"] == "coxph"

    def test_skipped_model_warns(self):
        """Model without metric data emits a warning."""
        import warnings as _w
        from types import SimpleNamespace

        result = SimpleNamespace(
            cv_metrics={},
            scoring_metric="brier_score",
            cv_metric_mean=0.3,
            cv_metric_std=0.01,
        )
        good = SimpleNamespace(
            cv_metrics={"c_index": {"mean": 0.75, "std": 0.03}},
            scoring_metric="c_index",
        )
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            df = self._build(
                {"bad_model": result, "good_model": good}, "c_index", None,
            )
        assert len(df) == 1
        assert any("bad_model" in str(x.message) for x in w)


# ══════════════════════════════════════════════════════════════════════════════
# Consistent group coloring for coefficients
# ══════════════════════════════════════════════════════════════════════════════


class TestGroupColoredCoefficients:
    """Coefficient bars colored by group when groups provided."""

    def test_bar_traces_per_group(self, coef_df):
        """With groups, one Bar trace per group."""
        fig = plot_coefficients(coef_df, n=5, feature_groups=GROUPS_DICT)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        bar_names = {t.name for t in bar_traces}
        assert "clinical" in bar_names
        assert "demographic" in bar_names

    def test_bar_group_colors_from_palette(self, coef_df):
        """Bar traces use QUALITATIVE_PALETTE colors."""
        fig = plot_coefficients(coef_df, n=5, feature_groups=GROUPS_DICT)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        for t in bar_traces:
            assert t.marker.color in QUALITATIVE_PALETTE

    def test_bar_no_risk_coloring_with_groups(self, coef_df):
        """With groups, hardcoded red/blue risk colors should not appear."""
        fig = plot_coefficients(coef_df, n=5, feature_groups=GROUPS_DICT)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        for t in bar_traces:
            # marker.color should be a single string (group color), not a list
            assert isinstance(t.marker.color, str)
            assert t.marker.color != "#d62728"
            assert t.marker.color != "#1f77b4"

    def test_without_groups_keeps_risk_coloring(self, coef_df):
        """Without groups, original red/blue risk coloring is preserved."""
        fig = plot_coefficients(coef_df, n=5)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1
        for t in bar_traces:
            if isinstance(t.marker.color, (list, tuple)):
                colors = set(t.marker.color)
                assert colors <= {"#d62728", "#1f77b4"}

    def test_forest_traces_per_group(self, coef_df):
        """Forest style: one Scatter trace per group."""
        fig = plot_coefficients(
            coef_df, n=5, style="forest", feature_groups=GROUPS_DICT,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)
                          and t.name in ("clinical", "demographic")]
        assert len(scatter_traces) >= 2

    def test_custom_group_colors(self, coef_df):
        """Explicit group_colors override default palette."""
        custom = {"clinical": "#ff0000", "demographic": "#00ff00"}
        fig = plot_coefficients(
            coef_df, n=5, feature_groups=GROUPS_DICT, group_colors=custom,
        )
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        colors = {t.name: t.marker.color for t in bar_traces}
        assert colors.get("clinical") == "#ff0000"
        assert colors.get("demographic") == "#00ff00"


