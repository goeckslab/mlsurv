"""Tests for mlsurv.viz.missing (plot_missing)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest


# ---------------------------------------------------------------------------
# Standalone viz function tests
# ---------------------------------------------------------------------------

class TestPlotMissing:

    def test_basic(self):
        """DataFrame with NaN returns a Figure."""
        from mlsurv.viz.missing import plot_missing

        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [np.nan, 2, np.nan],
            "c": [1, 2, 3],
        })
        fig = plot_missing(df)
        assert isinstance(fig, go.Figure)

    def test_auto_columns(self):
        """columns=None auto-detects NaN columns, excludes clean ones."""
        from mlsurv.viz.missing import plot_missing

        df = pd.DataFrame({
            "has_nan": [1, np.nan, 3],
            "clean": [1, 2, 3],
        })
        fig = plot_missing(df)
        # Only the column with NaN should appear on x-axis
        assert list(fig.data[0].x) == ["has_nan"]

    def test_no_missing(self):
        """DataFrame with no NaN returns a figure with annotation."""
        from mlsurv.viz.missing import plot_missing

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        fig = plot_missing(df)
        assert isinstance(fig, go.Figure)
        # Should have an annotation saying no missing data
        assert any("No missing data" in a.text
                    for a in fig.layout.annotations)

    def test_clustering_fallback(self, caplog):
        """Clustering error falls back to original order and warns."""
        import logging
        import warnings
        from mlsurv.viz.missing import plot_missing

        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(10, 4), columns=["a", "b", "c", "d"],
        )
        data.iloc[0, 0] = np.nan
        data.iloc[1, 1] = np.nan

        with patch(
            "mlsurv.viz.missing.leaves_list",
            side_effect=ValueError("linkage failed"),
        ):
            with caplog.at_level(logging.ERROR, logger="mlsurv.viz.missing"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    fig = plot_missing(data, columns=["a", "b", "c", "d"])

        assert isinstance(fig, go.Figure)
        assert "linkage failed" in caplog.text
        # Y-axis should reflect unclustered state
        assert fig.layout.yaxis.title.text == "Sample (original order)"

    def test_max_samples(self):
        """Subsampling respects max_samples."""
        from mlsurv.viz.missing import plot_missing

        np.random.seed(0)
        df = pd.DataFrame({
            "a": np.where(np.random.rand(500) < 0.3, np.nan, 1.0),
            "b": np.where(np.random.rand(500) < 0.3, np.nan, 2.0),
        })
        fig = plot_missing(df, max_samples=100)
        # Heatmap z-matrix rows should be <= 100
        assert fig.data[0].z.shape[0] <= 100

    def test_custom_title(self):
        """Custom title is applied."""
        from mlsurv.viz.missing import plot_missing

        df = pd.DataFrame({"a": [1, np.nan]})
        fig = plot_missing(df, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    # -- Accessibility: colorblind-/grayscale-safe encoding + legend -------

    def test_legend_labels_present(self):
        """Figure carries a legend keyed 'Present' / 'Missing'."""
        from mlsurv.viz.missing import plot_missing

        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
        fig = plot_missing(df)

        assert fig.layout.showlegend is True
        legend_names = {
            t.name for t in fig.data
            if isinstance(t, go.Scatter) and t.showlegend
        }
        assert legend_names == {"Present", "Missing"}

    def test_legend_swatch_colors_match_labels(self):
        """Each legend swatch shows the color it claims to label.

        Guards against silently swapping the (color, label) pairing, which
        would make the legend lie about which color means missing.
        """
        from mlsurv.viz.missing import plot_missing
        from mlsurv.viz.missing import _MISSING_COLOR, _PRESENT_COLOR

        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
        fig = plot_missing(df)

        swatch = {
            t.name: t.marker.color for t in fig.data
            if isinstance(t, go.Scatter) and t.showlegend
        }
        assert swatch["Missing"] == _MISSING_COLOR
        assert swatch["Present"] == _PRESENT_COLOR

    def test_colorscale_is_accessible(self):
        """Heatmap uses the luminance-distinct scheme, not red/green."""
        from mlsurv.viz.missing import plot_missing
        from mlsurv.viz.missing import _MISSING_COLOR, _PRESENT_COLOR

        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
        fig = plot_missing(df)

        heatmap = fig.data[0]
        colors = {pair[1].lower() for pair in heatmap.colorscale}
        assert _PRESENT_COLOR.lower() in colors
        assert _MISSING_COLOR.lower() in colors
        # The old red/green pair must not creep back in.
        assert "#2ecc71" not in colors
        assert "#e74c3c" not in colors

    def test_encoding_and_pinned_range(self):
        """z encodes present=0 / missing=1 with the range pinned to [0, 1].

        Pinning matters: an all-present or all-missing clustered subset would
        otherwise auto-scale and map its single value to the wrong color.
        A single deterministic column skips clustering, so row order is stable.
        """
        from mlsurv.viz.missing import plot_missing

        df = pd.DataFrame({"a": [1.0, np.nan]})  # row 0 present, row 1 missing
        fig = plot_missing(df)

        heatmap = fig.data[0]
        assert heatmap.zmin == 0
        assert heatmap.zmax == 1
        assert set(np.unique(heatmap.z)) <= {0, 1}
        # Present row encodes 0, missing row encodes 1.
        assert list(heatmap.z[0]) == [0]
        assert list(heatmap.z[1]) == [1]


# ---------------------------------------------------------------------------
# Learner wrapper tests
# ---------------------------------------------------------------------------

class TestLearnerPlotMissing:

    @pytest.fixture()
    def learner(self, synthetic_data):
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data
        sl = SurvivalLearner(X, y)
        sl.setup()
        return sl

    def test_returns_figure(self, learner):
        """plot_missing() returns a Figure after setup."""
        fig = learner.plot_missing()
        assert isinstance(fig, go.Figure)

    def test_return_data(self, learner):
        """return_data=True returns boolean DataFrame."""
        result = learner.plot_missing(return_data=True)
        assert isinstance(result, pd.DataFrame)
        assert result.dtypes.apply(lambda d: d == bool).all()
        assert result.shape == learner._state.X_train.shape

    def test_before_setup_auto_runs(self, synthetic_data):
        """plot_missing() auto-runs setup() if not called explicitly."""
        from mlsurv.learner import SurvivalLearner

        X, y = synthetic_data
        sl = SurvivalLearner(X, y)
        # Should auto-setup and return a figure without error
        fig = sl.plot_missing()
        assert isinstance(fig, go.Figure)
