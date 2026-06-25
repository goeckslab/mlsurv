"""Tests for mlsurv.viz.features — silent fallback warnings."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv._warnings import InternalWarning
from mlsurv.viz.features import plot_importance


class TestImportanceFallbackWarning:
    """plot_importance should warn when falling back from raincloud/boxplot to bar."""

    def _make_importance_df(self, n=5):
        return pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(n)],
            "perm_mean": np.linspace(0.1, 0.5, n),
            "perm_std": np.full(n, 0.01),
            "model": "RSF",
        })

    def test_raincloud_without_iterations_warns(self):
        df = self._make_importance_df()
        with pytest.warns(InternalWarning, match="style='raincloud'.*was not provided"):
            fig = plot_importance(df, style="raincloud")
        # Should still produce a figure (bar fallback)
        assert fig is not None

    def test_boxplot_without_iterations_warns(self):
        df = self._make_importance_df()
        with pytest.warns(InternalWarning, match="style='boxplot'.*was not provided"):
            fig = plot_importance(df, style="boxplot")
        assert fig is not None

    def test_iterations_df_with_non_matching_features_warns(self):
        df = self._make_importance_df()
        iterations = pd.DataFrame({
            "feature": ["unknown_a"] * 10,
            "importance": np.random.default_rng(0).random(10),
        })
        with pytest.warns(InternalWarning, match="none of its features match"):
            fig = plot_importance(df, style="raincloud", iterations_df=iterations)
        assert fig is not None

    def test_raincloud_with_valid_iterations_no_warning(self):
        df = self._make_importance_df()
        rng = np.random.default_rng(0)
        features = [f"feat_{i}" for i in range(5)]
        iterations = pd.DataFrame({
            "feature": [f for f in features for _ in range(10)],
            "importance": rng.random(50),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_importance(df, style="raincloud", iterations_df=iterations)
        # Single-model raincloud uses the dedicated single-model renderer
        # (one violin per feature), not the multi-model grouped path.
        violins = [t for t in fig.data if isinstance(t, go.Violin)]
        assert len(violins) >= 1

    def test_bar_style_no_warning(self):
        df = self._make_importance_df()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_importance(df, style="bar")
        assert fig is not None

    def test_invalid_style_raises(self):
        df = self._make_importance_df()
        with pytest.raises(ValueError, match="style.*not supported"):
            plot_importance(df, style="invalid")


class TestMultiModelDistributionStyles:
    """Multi-model raincloud/boxplot render grouped per-model distributions."""

    MODELS = ["coxph", "rsf"]
    FEATURES = [f"feat_{i}" for i in range(5)]

    def _summary(self):
        rows = []
        for m in self.MODELS:
            for fi, f in enumerate(self.FEATURES):
                rows.append({
                    "feature": f, "perm_mean": 0.05 + fi * 0.01,
                    "perm_std": 0.01, "model": m,
                })
        return pd.DataFrame(rows)

    def _iterations(self):
        rng = np.random.default_rng(1)
        rows = []
        for m in self.MODELS:
            for fi, f in enumerate(self.FEATURES):
                for j in range(8):
                    rows.append({
                        "model": m, "feature": f, "iteration": j,
                        "importance": 0.05 + fi * 0.01 + rng.normal(0, 0.01),
                    })
        return pd.DataFrame(rows)

    def test_raincloud_multi_model_renders_grouped_violins(self):
        fig = plot_importance(self._summary(), style="raincloud",
                              iterations_df=self._iterations())
        violins = [t for t in fig.data if isinstance(t, go.Violin)]
        assert len(violins) >= 1
        assert {t.name for t in violins} == {"CoxPH", "RSF"}
        assert fig.layout.violinmode == "group"

    def test_boxplot_multi_model_renders_grouped_boxes(self):
        fig = plot_importance(self._summary(), style="boxplot",
                              iterations_df=self._iterations())
        boxes = [t for t in fig.data if isinstance(t, go.Box)]
        assert {t.name for t in boxes} == {"CoxPH", "RSF"}
        assert fig.layout.boxmode == "group"

    def test_multi_model_distribution_without_iterations_falls_back_to_bar(self):
        with pytest.warns(InternalWarning, match="Falling back to 'bar'"):
            fig = plot_importance(self._summary(), style="raincloud",
                                  iterations_df=None)
        assert any(isinstance(t, go.Bar) for t in fig.data)
        assert not any(isinstance(t, (go.Violin, go.Box)) for t in fig.data)

    def test_multi_model_empty_iterations_falls_back_to_bar(self):
        empty = pd.DataFrame({"model": [], "feature": [], "importance": []})
        with pytest.warns(InternalWarning, match="Falling back to 'bar'"):
            fig = plot_importance(self._summary(), style="boxplot",
                                  iterations_df=empty)
        assert any(isinstance(t, go.Bar) for t in fig.data)
        assert not any(isinstance(t, (go.Violin, go.Box)) for t in fig.data)

    def test_multi_model_iterations_missing_model_column_falls_back(self):
        # iterations lack the model column -> required columns not present -> bar
        iters = self._iterations().drop(columns=["model"])
        with pytest.warns(InternalWarning, match="Falling back to 'bar'"):
            fig = plot_importance(self._summary(), style="raincloud",
                                  iterations_df=iters)
        assert any(isinstance(t, go.Bar) for t in fig.data)
        assert not any(isinstance(t, (go.Violin, go.Box)) for t in fig.data)

    def test_multi_model_non_matching_features_falls_back(self):
        # iterations carry the model column but no feature matches the top list
        iters = self._iterations().assign(feature="ghost")
        with pytest.warns(InternalWarning, match="Falling back to 'bar'"):
            fig = plot_importance(self._summary(), style="raincloud",
                                  iterations_df=iters)
        assert any(isinstance(t, go.Bar) for t in fig.data)
        assert not any(isinstance(t, (go.Violin, go.Box)) for t in fig.data)

    def test_multi_model_partial_iterations_warns_and_skips_absent_model(self):
        # Only coxph has iteration rows; rsf is in the summary but absent here.
        iters = self._iterations()
        iters = iters[iters["model"] == "coxph"]
        with pytest.warns(InternalWarning, match="No per-iteration importance data"):
            fig = plot_importance(self._summary(), style="raincloud",
                                  iterations_df=iters)
        violins = [t for t in fig.data if isinstance(t, go.Violin)]
        assert {t.name for t in violins} == {"CoxPH"}  # rsf skipped, not crashed

    def test_multi_model_missing_feature_warns_blank_row(self):
        # Drop one top feature from the iterations entirely (both models).
        iters = self._iterations()
        iters = iters[iters["feature"] != self.FEATURES[0]]
        with pytest.warns(InternalWarning, match="appear as blank rows"):
            plot_importance(self._summary(), style="boxplot",
                            iterations_df=iters)

    def test_auto_multi_model_stays_bar(self):
        fig = plot_importance(self._summary(), style="auto",
                              iterations_df=self._iterations())
        assert any(isinstance(t, go.Bar) for t in fig.data)
        assert not any(isinstance(t, (go.Violin, go.Box)) for t in fig.data)

    def test_multi_model_boxplot_with_feature_groups_adds_separators(self):
        groups = {f: ("A" if i < 3 else "B")
                  for i, f in enumerate(self.FEATURES)}
        fig = plot_importance(self._summary(), style="boxplot",
                              iterations_df=self._iterations(),
                              feature_groups=groups, sort_by_group=True)
        assert any(isinstance(t, go.Box) for t in fig.data)
        # sort_by_group draws dotted separators between the two groups.
        assert len(fig.layout.shapes) >= 1

    def test_multi_model_raincloud_with_feature_groups(self):
        groups = {f: ("A" if i < 3 else "B")
                  for i, f in enumerate(self.FEATURES)}
        fig = plot_importance(self._summary(), style="raincloud",
                              iterations_df=self._iterations(),
                              feature_groups=groups, sort_by_group=True)
        assert any(isinstance(t, go.Violin) for t in fig.data)
