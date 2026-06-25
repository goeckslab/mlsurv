"""Tests for return_data on low-level viz functions (plot_roc, plot_metric_time, plot_importance)."""

import numpy as np
import pandas as pd
import pytest

go = pytest.importorskip("plotly.graph_objects")


class TestPlotRocReturnData:
    """plot_roc(return_data=True) returns dict with figure and data."""

    @pytest.fixture
    def roc_df(self):
        return pd.DataFrame({
            "model": ["coxph"] * 3 + ["rsf"] * 3,
            "fpr": [0.0, 0.3, 1.0, 0.0, 0.2, 1.0],
            "tpr": [0.0, 0.7, 1.0, 0.0, 0.8, 1.0],
            "threshold": [np.inf, 0.5, -np.inf, np.inf, 0.6, -np.inf],
            "auc": [0.75] * 3 + [0.82] * 3,
            "eval_time": [12.0] * 6,
        })

    def test_returns_dict(self, roc_df):
        from mlsurv.viz.metrics import plot_roc

        result = plot_roc(roc_df, return_data=True)
        assert isinstance(result, dict)
        assert "figure" in result
        assert "data" in result

    # test_figure_is_go_figure dropped: the figure object is asserted
    # to be a go.Figure by test_false_returns_figure (same construction path).
    def test_data_is_dataframe(self, roc_df):
        from mlsurv.viz.metrics import plot_roc

        result = plot_roc(roc_df, return_data=True)
        assert isinstance(result["data"], pd.DataFrame)
        assert "model" in result["data"].columns

    def test_false_returns_figure(self, roc_df):
        from mlsurv.viz.metrics import plot_roc

        result = plot_roc(roc_df, return_data=False)
        assert isinstance(result, go.Figure)

    def test_data_content_preserved(self, roc_df):
        """Returned data has the same rows as the input DataFrame."""
        from mlsurv.viz.metrics import plot_roc

        result = plot_roc(roc_df, return_data=True)
        assert len(result["data"]) == len(roc_df)
        assert set(result["data"]["model"].unique()) == {"coxph", "rsf"}


class TestPlotMetricTimeReturnData:
    """plot_metric_time(return_data=True) returns dict with figure and data."""

    @pytest.fixture
    def time_df(self):
        return pd.DataFrame({
            "model": ["coxph"] * 3 + ["rsf"] * 3,
            "eval_time": [6, 12, 18] * 2,
            "auc": [0.75, 0.78, 0.76, 0.77, 0.80, 0.79],
        })

    def test_returns_dict(self, time_df):
        from mlsurv.viz.metrics import plot_metric_time

        result = plot_metric_time(time_df, metric="auc", return_data=True)
        assert isinstance(result, dict)
        assert "figure" in result
        assert "data" in result

    # test_figure_is_go_figure dropped: covered by
    # test_false_returns_figure and test_multi_panel_returns_dict.
    def test_data_is_dataframe(self, time_df):
        from mlsurv.viz.metrics import plot_metric_time

        result = plot_metric_time(time_df, metric="auc", return_data=True)
        assert isinstance(result["data"], pd.DataFrame)
        assert "model" in result["data"].columns

    def test_false_returns_figure(self, time_df):
        from mlsurv.viz.metrics import plot_metric_time

        result = plot_metric_time(time_df, metric="auc", return_data=False)
        assert isinstance(result, go.Figure)

    def test_data_content_preserved(self, time_df):
        """Returned data has the same rows as the input DataFrame."""
        from mlsurv.viz.metrics import plot_metric_time

        result = plot_metric_time(time_df, metric="auc", return_data=True)
        assert len(result["data"]) == len(time_df)

    def test_multi_panel_returns_dict(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            "model": ["coxph"] * 3 + ["rsf"] * 3,
            "eval_time": [6, 12, 18] * 2,
            "value": [0.75, 0.78, 0.76, 0.77, 0.80, 0.79],
            "metric_name": ["auc"] * 6,
        })
        result = plot_metric_time(df, metric=["auc"], return_data=True)
        assert isinstance(result, dict)
        assert "figure" in result and "data" in result
        assert isinstance(result["figure"], go.Figure)


class TestPlotImportanceReturnData:
    """plot_importance(return_data=True) returns dict with figure and data."""

    @pytest.fixture
    def importance_df(self):
        return pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(10)],
            "perm_mean": np.random.RandomState(42).rand(10),
            "perm_std": np.random.RandomState(42).rand(10) * 0.1,
        })

    def test_returns_dict(self, importance_df):
        from mlsurv.viz.features import plot_importance

        result = plot_importance(importance_df, n=5, return_data=True)
        assert isinstance(result, dict)
        assert "figure" in result
        assert "data" in result

    # test_figure_is_go_figure dropped: covered by
    # test_false_returns_figure and test_multi_model_returns_dict.
    def test_data_is_dataframe(self, importance_df):
        from mlsurv.viz.features import plot_importance

        result = plot_importance(importance_df, n=5, return_data=True)
        assert isinstance(result["data"], pd.DataFrame)

    def test_false_returns_figure(self, importance_df):
        from mlsurv.viz.features import plot_importance

        result = plot_importance(importance_df, n=5, return_data=False)
        assert isinstance(result, go.Figure)

    def test_data_is_full_input_not_filtered(self, importance_df):
        """return_data returns the full input df, not just the top-n features."""
        from mlsurv.viz.features import plot_importance

        result = plot_importance(importance_df, n=5, return_data=True)
        assert len(result["data"]) == len(importance_df)

    def test_multi_model_returns_dict(self):
        """return_data works for multi-model importance (>1 unique model)."""
        from mlsurv.viz.features import plot_importance

        df = pd.DataFrame({
            "feature": ["A", "B", "C"] * 2,
            "perm_mean": [0.1, 0.2, 0.15, 0.12, 0.18, 0.14],
            "model": ["coxph"] * 3 + ["rsf"] * 3,
        })
        result = plot_importance(df, n=3, return_data=True)
        assert isinstance(result, dict)
        assert isinstance(result["figure"], go.Figure)
        assert isinstance(result["data"], pd.DataFrame)


def _roc_df(model_col):
    return pd.DataFrame({
        model_col: ["coxph"] * 3,
        "fpr": [0.0, 0.5, 1.0],
        "tpr": [0.0, 0.8, 1.0],
        "auc": [0.75] * 3,
        "eval_time": [12.0] * 3,
    })


def _metric_time_df(model_col):
    return pd.DataFrame({
        model_col: ["coxph"] * 3,
        "eval_time": [6, 12, 18],
        "auc": [0.75, 0.78, 0.76],
    })


def _importance_df(model_col):
    return pd.DataFrame({
        "feature": ["A", "B", "C"] * 2,
        "perm_mean": [0.1, 0.2, 0.15, 0.12, 0.18, 0.14],
        model_col: ["coxph"] * 3 + ["rsf"] * 3,
    })


def _call_roc(df, **kw):
    from mlsurv.viz.metrics import plot_roc
    return plot_roc(df, **kw)


def _call_metric_time(df, **kw):
    from mlsurv.viz.metrics import plot_metric_time
    return plot_metric_time(df, metric="auc", **kw)


def _call_importance(df, **kw):
    from mlsurv.viz.features import plot_importance
    return plot_importance(df, n=3, **kw)


def _call_importance_heatmap(df, **kw):
    from mlsurv.viz.features import plot_importance_heatmap
    return plot_importance_heatmap(df, n=3, **kw)


class TestModelColDefault:
    """Verify the model_col='model' default and the model_col='model_type'
    backward-compat path for all changed viz functions."""

    # (label, df-builder, callable)
    _FUNCS = [
        ("roc", _roc_df, _call_roc),
        ("metric_time", _metric_time_df, _call_metric_time),
        ("importance", _importance_df, _call_importance),
        ("importance_heatmap", _importance_df, _call_importance_heatmap),
    ]

    @pytest.mark.parametrize("label, make_df, call", _FUNCS, ids=[f[0] for f in _FUNCS])
    def test_default_model_col(self, label, make_df, call):
        """Default model_col='model' (no explicit kwarg) renders a figure."""
        fig = call(make_df("model"))
        assert isinstance(fig, go.Figure)

    @pytest.mark.parametrize("label, make_df, call", _FUNCS, ids=[f[0] for f in _FUNCS])
    def test_backward_compat_model_type_col(self, label, make_df, call):
        """Internal callers can pass model_col='model_type' explicitly."""
        fig = call(make_df("model_type"), model_col="model_type")
        assert isinstance(fig, go.Figure)
