"""Tests for feature importance enhancements."""

import numpy as np
import pandas as pd

# ── Per-feature sample size in importance table ────────────────────────────


def _make_summary_row(feature="feat_0", uni_p=0.01, uni_hr=1.5):
    """Minimal feature-summary row for the row builders."""
    return pd.Series({
        "feature": feature,
        "perm_mean": 0.05,
        "perm_std": 0.01,
        "uni_p_value": uni_p,
        "uni_hazard_ratio": uni_hr,
        "p_adjusted": np.nan,
        "sig_stars": None,
        "mean_abs_shap": np.nan,
        "correlated": False,
    })


def _make_summary_block(features, model_type="coxph", uni_p=0.01, uni_hr=1.5):
    """Minimal feature-summary block (one model) for the row builders."""
    n = len(features)
    return pd.DataFrame({
        "model_type": [model_type] * n,
        "feature": features,
        "perm_mean": np.linspace(0.05, 0.03, n),
        "perm_std": [0.01] * n,
        "uni_p_value": [uni_p] * n,
        "uni_hazard_ratio": [uni_hr] * n,
        "p_adjusted": [np.nan] * n,
        "sig_stars": [None] * n,
        "mean_abs_shap": [np.nan] * n,
        "correlated": [False] * n,
    })


class TestPerFeatureSampleSize:
    """build_feature_row / build_multi_model_feature_row should add 'n' column
    when univar_cox has n_missing > 0 for any feature."""

    def _make_univar_cox(self, n_missing_vals):
        """Helper: create a univar_cox DataFrame with given n_missing values."""
        features = [f"feat_{i}" for i in range(len(n_missing_vals))]
        return pd.DataFrame({
            "feature": features,
            "p_value": [0.01] * len(features),
            "hazard_ratio": [1.5] * len(features),
            "n_missing": n_missing_vals,
        })

    def test_n_column_present_when_missing_data(self):
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_feature_row,
        )
        univar = self._make_univar_cox([10, 0, 5])
        result = build_feature_row(
            _make_summary_row("feat_0"), "coxph", 0.7,
            univar_cox=univar, total_samples=100,
        )
        assert "n" in result
        assert result["n"] == "90"  # 100 - 10

    def test_n_column_absent_when_no_missing(self):
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_feature_row,
        )
        univar = self._make_univar_cox([0, 0, 0])
        result = build_feature_row(
            _make_summary_row("feat_0"), "coxph", 0.7,
            univar_cox=univar, total_samples=100,
        )
        assert "n" not in result

    def test_n_column_absent_when_no_univar(self):
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_feature_row,
        )
        result = build_feature_row(
            _make_summary_row("feat_0"), "coxph", 0.7,
            univar_cox=None, total_samples=100,
        )
        assert "n" not in result

    def test_multi_model_n_column(self):
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_multi_model_feature_row,
        )
        univar = self._make_univar_cox([20, 0])
        blocks = {"coxph": _make_summary_block(["feat_0", "feat_1"])}
        result = build_multi_model_feature_row(
            "feat_0", ["coxph"], blocks, {"coxph": 0.7},
            univar_cox=univar, total_samples=200,
        )
        assert "n" in result
        assert result["n"] == "180"  # 200 - 20

    def test_multi_model_n_absent_when_no_missing(self):
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_multi_model_feature_row,
        )
        univar = self._make_univar_cox([0, 0])
        blocks = {"coxph": _make_summary_block(["feat_0", "feat_1"])}
        result = build_multi_model_feature_row(
            "feat_0", ["coxph"], blocks, {"coxph": 0.7},
            univar_cox=univar, total_samples=200,
        )
        assert "n" not in result

    def test_n_column_feature_not_in_univar(self):
        """Feature not in univar_cox should show total_samples as n."""
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_feature_row,
        )
        # univar has feat_0 with missing, but we query feat_99 (not in univar)
        univar = self._make_univar_cox([10])  # feat_0 only
        result = build_feature_row(
            _make_summary_row("feat_99"), "coxph", 0.7,
            univar_cox=univar, total_samples=100,
        )
        assert "n" in result
        assert result["n"] == "100"  # total_samples since feature not found

    def test_n_column_absent_when_total_samples_none(self):
        """total_samples=None should not crash or produce 'None' string."""
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_feature_row,
        )
        univar = self._make_univar_cox([10, 0])
        result = build_feature_row(
            _make_summary_row("feat_0"), "coxph", 0.7,
            univar_cox=univar, total_samples=None,
        )
        assert "n" not in result

    def test_n_column_nan_n_missing(self):
        """NaN in n_missing should fall back to total_samples."""
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_feature_row,
        )
        univar = pd.DataFrame({
            "feature": ["feat_0", "feat_1"],
            "p_value": [0.01, 0.05],
            "hazard_ratio": [1.5, 1.2],
            "n_missing": [float("nan"), 5],
        })
        result = build_feature_row(
            _make_summary_row("feat_0"), "coxph", 0.7,
            univar_cox=univar, total_samples=100,
        )
        assert "n" in result
        assert result["n"] == "100"  # NaN n_missing → total_samples


# ── build_feature_row univariate columns (from the summary row) ────────────


class TestBuildFeatureRowCoxLookup:
    """build_feature_row formats the uni_* columns of the summary row."""

    def test_extracts_pvalue_and_hr_from_summary_row(self):
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_feature_row,
        )
        result = build_feature_row(
            _make_summary_row("feat_0", uni_p=0.003, uni_hr=2.1),
            "coxph", 0.7, total_samples=100,
        )
        assert result["p-value"] != "N/A"
        assert result["HR"] == "2.100"

    def test_nan_uni_values_return_na(self):
        """A feature absent from univariate Cox carries NaN uni_* values."""
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_feature_row,
        )
        result = build_feature_row(
            _make_summary_row("feat_0", uni_p=np.nan, uni_hr=np.nan),
            "coxph", 0.7, total_samples=100,
        )
        assert result["p-value"] == "N/A"
        assert result["HR"] == "N/A"

    def test_inf_uni_pvalue_returns_na(self):
        """The p_value=inf failure sentinel renders as N/A."""
        from mlsurv.reports.sections.feature_analysis._importance import (
            build_feature_row,
        )
        result = build_feature_row(
            _make_summary_row("feat_0", uni_p=np.inf, uni_hr=np.nan),
            "coxph", 0.7, total_samples=100,
        )
        assert result["p-value"] == "N/A"
        assert result["HR"] == "N/A"


# ── PIMP for multiple models ───────────────────────────────────────────────


class TestPimpMultiModel:
    """The per-model PIMP dict should accumulate per-model results."""

    def test_pimp_results_dict_exists(self):
        from collections.abc import Mapping
        from mlsurv.learner._state import LearnerState
        state = LearnerState()
        assert hasattr(state, "pimp_importance_test_results")
        # Now a read-only mappingproxy view over _state.models.
        assert isinstance(state.pimp_importance_test_results, Mapping)
        assert len(state.pimp_importance_test_results) == 0

    def test_get_importance_pvalues_reads_dict(self):
        """get_importance_pvalues reads the per-model PIMP dict."""
        from mlsurv.reports.sections.feature_analysis._importance import (
            get_importance_pvalues,
        )

        # Create a mock section with per-model PIMP data
        coxph_result = pd.DataFrame({
            "feature": ["a", "b"],
            "p_value": [0.01, 0.05],
            "p_adjusted": [0.02, 0.10],
        })
        coxph_result.attrs["p_adjust"] = "bh"
        coxph_result.attrs["model_name"] = "coxph"

        rsf_result = pd.DataFrame({
            "feature": ["a", "b"],
            "p_value": [0.03, 0.07],
            "p_adjusted": [0.06, 0.14],
        })
        rsf_result.attrs["p_adjust"] = "bh"
        rsf_result.attrs["model_name"] = "rsf"

        class MockState:
            pimp_importance_test_results = {
                "coxph": coxph_result, "rsf": rsf_result,
            }
            bootstrap_result = None

        class MockLearner:
            _state = MockState()
            # Mirror the real learner's read-only dict accessor: the
            # report reads the per-model dict through this property.
            pimp_importance_test_results = _state.pimp_importance_test_results

        class MockSection:
            learner = MockLearner()
            eval_result = None
            _pvalue_source = None
            _pvalue_method = "bh"

            def _log_recoverable_exception(self, *a, **kw):
                pass

        section = MockSection()

        # Request RSF — should get the RSF entry, not another model's
        result = get_importance_pvalues(section, "rsf")
        assert result is not None
        assert result is rsf_result
        assert section._pvalue_source == "pimp"

        # Request coxph — should also work
        result2 = get_importance_pvalues(section, "coxph")
        assert result2 is coxph_result


# ── Sort features by group in plots ─────────────────────────────────────────


class TestSortFeaturesByGroup:
    """_sort_features_by_group utility tests."""

    def test_basic_sorting(self):
        from mlsurv.viz._palettes import _sort_features_by_group

        features = ["a", "b", "c", "d", "e"]
        feat_to_grp = {"a": "G2", "b": "G1", "c": "G1", "d": "G2"}
        grp_to_color = {"G1": "#aaa", "G2": "#bbb"}  # G1 first

        result = _sort_features_by_group(features, feat_to_grp, grp_to_color)

        # G1 features first (b, c), then G2 (a, d), then ungrouped (e)
        g1_feats = [f for f in result if feat_to_grp.get(f) == "G1"]
        g2_feats = [f for f in result if feat_to_grp.get(f) == "G2"]
        ungrouped = [f for f in result if f not in feat_to_grp]

        assert g1_feats == result[: len(g1_feats)]
        assert g2_feats == result[len(g1_feats) : len(g1_feats) + len(g2_feats)]
        assert ungrouped == result[len(g1_feats) + len(g2_feats) :]
        assert set(result) == set(features)

    def test_sorting_with_metric_values(self):
        from mlsurv.viz._palettes import _sort_features_by_group

        features = ["a", "b", "c"]
        feat_to_grp = {"a": "G1", "b": "G1", "c": "G1"}
        grp_to_color = {"G1": "#aaa"}
        metric_values = {"a": 1.0, "b": 3.0, "c": 2.0}

        result = _sort_features_by_group(
            features, feat_to_grp, grp_to_color, metric_values=metric_values,
        )
        # Within G1, sorted by metric descending: b (3), c (2), a (1)
        assert result == ["b", "c", "a"]

    def test_all_ungrouped(self):
        from mlsurv.viz._palettes import _sort_features_by_group

        features = ["a", "b", "c"]
        feat_to_grp = {}
        grp_to_color = {}
        result = _sort_features_by_group(features, feat_to_grp, grp_to_color)
        assert result == ["a", "b", "c"]  # original order preserved


class TestAddGroupSeparatorShapes:
    """_add_group_separator_shapes should add hlines between groups."""

    def test_separator_shapes_added(self):
        import plotly.graph_objects as go
        from mlsurv.viz._palettes import _add_group_separator_shapes

        fig = go.Figure()
        features = ["a", "b", "c", "d"]
        feat_to_grp = {"a": "G1", "b": "G1", "c": "G2", "d": "G2"}

        _add_group_separator_shapes(fig, features, feat_to_grp, axis="y")

        # There should be one separator between G1 and G2 (at index 2)
        shapes = fig.layout.shapes
        assert len(shapes) == 1
        assert shapes[0].line.dash == "dot"

    def test_no_separators_single_group(self):
        import plotly.graph_objects as go
        from mlsurv.viz._palettes import _add_group_separator_shapes

        fig = go.Figure()
        features = ["a", "b", "c"]
        feat_to_grp = {"a": "G1", "b": "G1", "c": "G1"}

        _add_group_separator_shapes(fig, features, feat_to_grp, axis="y")
        assert len(fig.layout.shapes) == 0


class TestPlotImportanceSortByGroup:
    """plot_importance with sort_by_group=True."""

    def _make_df(self, n=6):
        return pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(n)],
            "perm_mean": np.linspace(0.1, 0.6, n),
            "perm_std": np.full(n, 0.01),
        })

    def test_sort_by_group_no_error(self):
        from mlsurv.viz.features import plot_importance

        df = self._make_df()
        groups = {f"feat_{i}": f"G{i % 2}" for i in range(6)}
        fig = plot_importance(
            df, feature_groups=groups, sort_by_group=True,
        )
        assert fig is not None

    def test_sort_by_group_adds_separators(self):
        from mlsurv.viz.features import plot_importance

        df = self._make_df()
        groups = {f"feat_{i}": f"G{i % 2}" for i in range(6)}
        fig = plot_importance(
            df, feature_groups=groups, sort_by_group=True,
        )
        # Should have at least one separator shape
        shapes = [s for s in fig.layout.shapes if s.line.dash == "dot"]
        assert len(shapes) >= 1

    def test_sort_by_group_false_no_separators(self):
        from mlsurv.viz.features import plot_importance

        df = self._make_df()
        groups = {f"feat_{i}": f"G{i % 2}" for i in range(6)}
        fig = plot_importance(
            df, feature_groups=groups, sort_by_group=False,
        )
        # No separator shapes (other shapes may exist, but not dotted)
        shapes = [s for s in fig.layout.shapes if s.line.dash == "dot"]
        assert len(shapes) == 0

    def test_sort_by_group_no_groups_no_error(self):
        """sort_by_group=True with feature_groups=None should not error."""
        from mlsurv.viz.features import plot_importance

        df = self._make_df()
        fig = plot_importance(df, sort_by_group=True)
        assert fig is not None


class TestPlotCoefficientsSortByGroup:
    """plot_coefficients with sort_by_group=True."""

    def test_sort_by_group(self):
        from mlsurv.viz.features import plot_coefficients

        df = pd.DataFrame({
            "feature": ["a", "b", "c", "d"],
            "coefficient": [0.5, -0.3, 0.2, -0.1],
        })
        groups = {"a": "G1", "b": "G1", "c": "G2", "d": "G2"}
        fig = plot_coefficients(
            df, feature_groups=groups, sort_by_group=True,
        )
        assert fig is not None
        shapes = [s for s in fig.layout.shapes if s.line.dash == "dot"]
        assert len(shapes) >= 1


class TestPlotCoefficientsStyle:
    """plot_coefficients style handling: multi-model forest + fallback warning."""

    def _make_multi_model_df(self, n=4):
        """Two-model coefficient frame with standard errors."""
        rows = []
        for model in ("CoxPH", "Coxnet"):
            for i in range(n):
                rows.append({
                    "feature": f"feat_{i}",
                    "coefficient": 0.5 - 0.1 * i + (0.05 if model == "Coxnet" else 0.0),
                    "coef_std": 0.05,
                    "model": model,
                })
        return pd.DataFrame(rows)

    def test_multi_model_forest_produces_scatter_not_bars(self):
        """style='forest' with 2+ models renders scatter markers, not bars."""
        import plotly.graph_objects as go
        from mlsurv.viz.features import plot_coefficients

        fig = plot_coefficients(self._make_multi_model_df(), style="forest")
        scatters = [t for t in fig.data if isinstance(t, go.Scatter)]
        bars = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bars) == 0, "forest style must not emit bar traces"
        # one marker trace per model
        assert len(scatters) == 2

    def test_multi_model_bar_still_grouped(self):
        """style='bar' regression: grouped bar traces remain."""
        import plotly.graph_objects as go
        from mlsurv.viz.features import plot_coefficients

        fig = plot_coefficients(self._make_multi_model_df(), style="bar")
        bars = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bars) == 2
        assert fig.layout.barmode == "group"

    def test_multi_model_forest_with_groups(self):
        """Multi-model forest + feature_groups + sort_by_group (the demo combo)."""
        import plotly.graph_objects as go
        from mlsurv.viz.features import plot_coefficients

        groups = {"feat_0": "G1", "feat_1": "G1", "feat_2": "G2", "feat_3": "G2"}
        fig = plot_coefficients(
            self._make_multi_model_df(), style="forest",
            feature_groups=groups, sort_by_group=True,
        )
        scatters = [t for t in fig.data if isinstance(t, go.Scatter)]
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        # 2 model marker traces (plus group legend traces); 1 group strip
        assert len(scatters) >= 2
        assert len(heatmaps) >= 1

    def test_unrecognized_style_warns_and_falls_back_to_bar(self):
        """An unknown style emits SuggestionWarning and renders bars."""
        import plotly.graph_objects as go
        import pytest
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.viz.features import plot_coefficients

        with pytest.warns(SuggestionWarning, match="not recognized"):
            fig = plot_coefficients(self._make_multi_model_df(), style="raincloud")
        assert any(isinstance(t, go.Bar) for t in fig.data)

    def test_valid_styles_emit_no_warning(self):
        """'bar' and 'forest' must not trip the fallback warning."""
        import warnings as _w
        from mlsurv.viz.features import plot_coefficients

        df = self._make_multi_model_df()
        for style in ("bar", "forest"):
            with _w.catch_warnings():
                _w.simplefilter("error")
                plot_coefficients(df, style=style)

    def test_multi_model_forest_offsets_models(self):
        """Each model's markers sit at distinct y-positions (no overlap)."""
        import numpy as np
        import plotly.graph_objects as go
        from mlsurv.viz.features import plot_coefficients

        df = self._make_multi_model_df(n=4)
        fig = plot_coefficients(df, style="forest")
        scatters = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatters) == 2
        y0 = np.asarray(scatters[0].y, dtype=float)
        y1 = np.asarray(scatters[1].y, dtype=float)
        # Same feature rows, but offset so the two models never coincide.
        assert not np.allclose(y0, y1)
        # Ticks land on the integer feature indices, not the feature names.
        assert tuple(fig.layout.yaxis.tickvals) == (0, 1, 2, 3)

    def test_multi_model_forest_missing_feature_keeps_trace(self):
        """A model lacking a top-N feature still gets a trace; the gap is a NaN."""
        import numpy as np
        import plotly.graph_objects as go
        from mlsurv.viz.features import plot_coefficients

        df = self._make_multi_model_df(n=4)
        # Drop one feature from Coxnet only -> asymmetric frame.
        df = df[~((df["model"] == "Coxnet") & (df["feature"] == "feat_3"))]
        fig = plot_coefficients(df, style="forest")
        scatters = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatters) == 2  # both models keep a legend entry
        by_name = {t.name: np.asarray(t.x, dtype=float) for t in scatters}
        # CoxPH has all four features; Coxnet is missing exactly one (a NaN gap).
        assert np.count_nonzero(np.isnan(by_name["CoxPH"])) == 0
        assert np.count_nonzero(np.isnan(by_name["Coxnet"])) == 1
        assert len(by_name["CoxPH"]) == len(by_name["Coxnet"])

    def test_multi_model_forest_all_nan_std_no_error_bars(self):
        """All-NaN coef_std yields markers with no error bars (no crash)."""
        import numpy as np
        import plotly.graph_objects as go
        from mlsurv.viz.features import plot_coefficients

        df = self._make_multi_model_df(n=3)
        df["coef_std"] = np.nan
        fig = plot_coefficients(df, style="forest")
        scatters = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatters) == 2
        for t in scatters:
            assert t.error_x.array is None

    def test_single_model_forest_keeps_categorical_axis(self):
        """Single-model forest stays on the categorical (feature-name) y-axis."""
        import plotly.graph_objects as go
        from mlsurv.viz.features import plot_coefficients

        single = self._make_multi_model_df(n=4)
        single = single[single["model"] == "CoxPH"].drop(columns=["model"])
        fig = plot_coefficients(single, style="forest")
        assert any(isinstance(t, go.Scatter) for t in fig.data)
        # Ticks are the feature-name strings, not integer indices.
        tickvals = fig.layout.yaxis.tickvals
        assert all(isinstance(v, str) for v in tickvals)


class TestPlotHazardRatiosSortByGroup:
    """plot_hazard_ratios with sort_by_group=True."""

    def test_sort_by_group(self):
        from mlsurv.viz.features import plot_hazard_ratios

        df = pd.DataFrame({
            "feature": ["a", "b", "c", "d"],
            "hr": [1.5, 0.7, 2.0, 0.5],
            "ci_lower": [1.2, 0.5, 1.5, 0.3],
            "ci_upper": [1.8, 0.9, 2.5, 0.7],
        })
        groups = {"a": "G1", "b": "G1", "c": "G2", "d": "G2"}
        fig = plot_hazard_ratios(
            df, feature_groups=groups, sort_by_group=True,
        )
        assert fig is not None
        shapes = [s for s in fig.layout.shapes
                  if hasattr(s, "line") and s.line.dash == "dot"]
        assert len(shapes) >= 1


class TestPlotHazardRatiosComparisonSortByGroup:
    """plot_hazard_ratios_comparison: sort_by_group gates reorder + separators,
    while the group strip/legend appear whenever feature_groups resolve."""

    # Two identical models -> ranking by |log(HR)| gives order [d, b, a, c];
    # groups are interleaved in that order so reordering is observable.
    GROUPS = {"a": "G1", "b": "G2", "c": "G2", "d": "G1"}

    def _models(self):
        df = pd.DataFrame({
            "feature": ["a", "b", "c", "d"],
            "hr": [1.5, 0.5, 1.2, 0.3],
            "ci_lower": [1.2, 0.4, 1.0, 0.2],
            "ci_upper": [1.8, 0.6, 1.4, 0.4],
        })
        return {"model_a": df, "model_b": df.copy()}

    @staticmethod
    def _separator_shapes(fig):
        return [s for s in fig.layout.shapes
                if hasattr(s, "line") and s.line.dash == "dot"]

    @staticmethod
    def _strip_traces(fig):
        import plotly.graph_objects as go
        return [t for t in fig.data if isinstance(t, go.Heatmap)]

    @staticmethod
    def _is_group_contiguous(ordered_features, group_map):
        """True if each group occupies a single contiguous run."""
        seen_runs = []
        for feat in ordered_features:
            grp = group_map[feat]
            if not seen_runs or seen_runs[-1] != grp:
                seen_runs.append(grp)
        return len(seen_runs) == len(set(seen_runs))

    def test_sort_by_group_true_adds_separators_and_reorders(self):
        from mlsurv.viz.features import plot_hazard_ratios_comparison

        fig = plot_hazard_ratios_comparison(
            self._models(), feature_groups=self.GROUPS, sort_by_group=True,
        )
        # Separators present
        assert len(self._separator_shapes(fig)) >= 1
        # Strip + legend still present
        assert len(self._strip_traces(fig)) == 1
        legend_names = {t.name for t in fig.data
                        if getattr(t, "showlegend", False)}
        assert {"G1", "G2"} <= legend_names
        # Features reordered group-contiguous (strip y-axis ticktext is the
        # display order, bottom-to-top)
        order = list(fig.layout.yaxis.ticktext)
        assert self._is_group_contiguous(order, self.GROUPS)
        # Within-group |log(HR)| ranking is preserved (metric_values passed
        # through): b (hr=0.5, |log|=0.69) outranks c (hr=1.2, |log|=0.18),
        # so c precedes b in the reversed display order.
        assert order.index("c") < order.index("b")

    def test_sort_by_group_false_no_separators_but_strip_and_legend(self):
        from mlsurv.viz.features import plot_hazard_ratios_comparison

        fig = plot_hazard_ratios_comparison(
            self._models(), feature_groups=self.GROUPS, sort_by_group=False,
        )
        # No separators when the flag is off
        assert len(self._separator_shapes(fig)) == 0
        # ...but the colored group strip and legend still appear
        assert len(self._strip_traces(fig)) == 1
        legend_names = {t.name for t in fig.data
                        if getattr(t, "showlegend", False)}
        assert {"G1", "G2"} <= legend_names
        # Features stay in raw |log(HR)| ranking (interleaved groups, not sorted)
        order = list(fig.layout.yaxis.ticktext)
        assert not self._is_group_contiguous(order, self.GROUPS)

    def test_sort_by_group_no_groups_no_error(self):
        from mlsurv.viz.features import plot_hazard_ratios_comparison

        # sort_by_group=True with no feature_groups is a harmless no-op:
        # no separators, no group strip, no crash.
        fig = plot_hazard_ratios_comparison(
            self._models(), sort_by_group=True,
        )
        assert fig is not None
        assert len(self._separator_shapes(fig)) == 0
        assert len(self._strip_traces(fig)) == 0


class TestPlotImportanceHeatmapSortByGroup:
    """plot_importance_heatmap with sort_by_group=True."""

    def test_sort_by_group(self):
        from mlsurv.viz.features import plot_importance_heatmap

        df = pd.DataFrame({
            "feature": ["a", "b", "c"] * 2,
            "perm_mean": [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
            "model": ["m1"] * 3 + ["m2"] * 3,
        })
        groups = {"a": "G1", "b": "G1", "c": "G2"}
        fig = plot_importance_heatmap(
            df, feature_groups=groups, sort_by_group=True,
        )
        assert fig is not None


class TestPlotFeatureCorrelationSortByGroup:
    """plot_feature_correlation with sort_by_group=True."""

    def test_sort_by_group(self):
        from mlsurv.viz.features import plot_feature_correlation

        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            rng.random((50, 4)),
            columns=["a", "b", "c", "d"],
        )
        groups = {"a": "G1", "b": "G1", "c": "G2", "d": "G2"}
        fig = plot_feature_correlation(
            X, feature_groups=groups, sort_by_group=True,
        )
        assert fig is not None
