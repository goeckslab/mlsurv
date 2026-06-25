"""Unit tests for the shared subpopulation facet-grid scaffold and the
per-family ROC / calibration renderers.

These tests feed synthetic per-level data directly to the renderers — no
``SurvivalLearner`` training — so they stay in the fast tier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv.viz._plot_impl_calibration import _render_calibration_facet_grid
from mlsurv.viz._plot_impl_roc import _render_roc_facet_grid
from mlsurv.viz._plot_impl_shared import (
    _finalize_subpop_facet_layout,
    _format_subpop_facet_axes,
    _make_subpop_facet_grid,
    _subpop_grid_dims,
)


# ------------------------------------------------------------------
# Synthetic data builders
# ------------------------------------------------------------------

def _roc_df(models=("coxph", "rsf"), seed=0):
    """A tidy ROC DataFrame for one subpopulation level."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in models:
        fpr = np.linspace(0, 1, 5)
        tpr = np.clip(fpr + rng.uniform(0, 0.2, size=fpr.size), 0, 1)
        auc = float(np.trapezoid(tpr, fpr))
        for f, t in zip(fpr, tpr):
            rows.append({"model": m, "fpr": f, "tpr": t, "auc": auc})
    return pd.DataFrame(rows)


def _per_level_roc(levels, models=("coxph", "rsf")):
    return [(lvl, _roc_df(models=models, seed=i))
            for i, lvl in enumerate(levels)]


def _cal_data(models=("coxph", "rsf")):
    """A calibration data dict for one subpopulation level."""
    return {
        m: {
            "predicted": np.linspace(0.1, 0.9, 5),
            "observed": np.linspace(0.1, 0.9, 5),
        }
        for m in models
    }


def _per_level_cal(levels, models=("coxph", "rsf")):
    return [(lvl, _cal_data(models=models)) for lvl in levels]


# ------------------------------------------------------------------
# _subpop_grid_dims
# ------------------------------------------------------------------

class TestSubpopGridDims:
    @pytest.mark.parametrize(
        "n_levels, expected",
        [(1, (1, 1)), (2, (2, 1)), (3, (3, 1)), (4, (3, 2)),
         (6, (3, 2)), (7, (3, 3))],
    )
    def test_dims(self, n_levels, expected):
        assert _subpop_grid_dims(n_levels) == expected


# ------------------------------------------------------------------
# _make_subpop_facet_grid
# ------------------------------------------------------------------

class TestMakeFacetGrid:
    def test_panel_titles(self):
        fig = _make_subpop_facet_grid(["a", "b", "c", "d"])
        titles = {a.text for a in fig.layout.annotations}
        assert {"a", "b", "c", "d"} <= titles

    def test_is_figure(self):
        assert isinstance(_make_subpop_facet_grid(["x"]), go.Figure)


# ------------------------------------------------------------------
# _format_subpop_facet_axes
# ------------------------------------------------------------------

class TestFormatFacetAxes:
    def test_dtick_and_constrain_applied_to_both_axes(self):
        # 3 levels -> single full row, no empty cells: every panel formatted.
        fig = _make_subpop_facet_grid(["a", "b", "c"])
        _format_subpop_facet_axes(
            fig, 3, x_title="X", y_title="Y", dtick=0.2, constrain="domain",
        )
        assert all(ax.dtick == 0.2 for ax in fig.select_xaxes())
        assert all(ax.dtick == 0.2 for ax in fig.select_yaxes())
        assert all(ax.constrain == "domain" for ax in fig.select_xaxes())
        assert all(ax.constrain == "domain" for ax in fig.select_yaxes())

    def test_no_dtick_when_unset(self):
        fig = _make_subpop_facet_grid(["a", "b"])
        _format_subpop_facet_axes(fig, 2, x_title="X", y_title="Y")
        assert all(ax.dtick is None for ax in fig.select_xaxes())
        assert all(ax.dtick is None for ax in fig.select_yaxes())

    def test_titles_on_edges_only(self):
        # 4 levels -> 3 cols x 2 rows. Bottom row (row 2) holds only panel
        # index 3; first column holds panel indices 0 and 3.
        fig = _make_subpop_facet_grid(["a", "b", "c", "d"])
        _format_subpop_facet_axes(fig, 4, x_title="FPR", y_title="TPR")
        x_titles = [ax.title.text for ax in fig.select_xaxes()]
        y_titles = [ax.title.text for ax in fig.select_yaxes()]
        assert x_titles.count("FPR") == 1
        assert y_titles.count("TPR") == 2
        # Interior panels carry a blank (not None) title.
        assert "" in x_titles


# ------------------------------------------------------------------
# _finalize_subpop_facet_layout
# ------------------------------------------------------------------

class TestFinalizeFacetLayout:
    def test_bottom_right_anchor_no_title(self):
        fig = go.Figure()
        _finalize_subpop_facet_layout(
            fig, title="T", legend_anchor="bottom-right",
        )
        assert fig.layout.title.text == "T"
        assert fig.layout.legend.xanchor == "right"
        assert fig.layout.legend.yanchor == "bottom"
        assert fig.layout.legend.title.text is None

    def test_top_right_anchor_with_title(self):
        fig = go.Figure()
        _finalize_subpop_facet_layout(
            fig, title="T", legend_anchor="top-right", legend_title="Model",
        )
        assert fig.layout.legend.xanchor == "right"
        assert fig.layout.legend.yanchor == "top"
        assert fig.layout.legend.title.text == "Model"

    def test_size_overrides(self):
        fig = go.Figure()
        _finalize_subpop_facet_layout(
            fig, title="T", legend_anchor="top-right",
            width=640, height=480,
        )
        assert fig.layout.width == 640
        assert fig.layout.height == 480


# ------------------------------------------------------------------
# _render_roc_facet_grid
# ------------------------------------------------------------------

class TestRenderRocFacetGrid:
    def test_panel_count(self):
        fig = _render_roc_facet_grid(
            _per_level_roc(["a", "b", "c", "d"]),
            show_auc=True, eval_time=12.0, group_variable="grp",
        )
        titles = {a.text for a in fig.layout.annotations}
        assert {"a", "b", "c", "d"} <= titles

    def test_diagonal_shape_per_panel(self):
        levels = ["a", "b", "c"]
        fig = _render_roc_facet_grid(
            _per_level_roc(levels), show_auc=True, eval_time=12.0,
            group_variable="grp",
        )
        dashed = [
            s for s in fig.layout.shapes
            if s.line is not None and s.line.dash == "dash"
        ]
        assert len(dashed) == len(levels)

    def test_line_trace_count(self):
        levels, models = ["a", "b", "c"], ("coxph", "rsf")
        fig = _render_roc_facet_grid(
            _per_level_roc(levels, models=models), show_auc=True,
            eval_time=12.0, group_variable="grp",
        )
        # ROC diagonals are shapes, not traces -> traces are model lines only.
        assert len(fig.data) == len(levels) * len(models)

    def test_axis_titles(self):
        fig = _render_roc_facet_grid(
            _per_level_roc(["a", "b", "c", "d"]), show_auc=True,
            eval_time=12.0, group_variable="grp",
        )
        x_titles = [ax.title.text for ax in fig.select_xaxes()]
        y_titles = [ax.title.text for ax in fig.select_yaxes()]
        assert "False Positive Rate" in x_titles
        assert "True Positive Rate" in y_titles

    def test_show_auc_true_in_hover(self):
        fig = _render_roc_facet_grid(
            _per_level_roc(["a", "b"]), show_auc=True, eval_time=12.0,
            group_variable="grp",
        )
        assert any(
            "AUC" in (t.hovertemplate or "") for t in fig.data
        )

    def test_show_auc_false_not_in_hover(self):
        fig = _render_roc_facet_grid(
            _per_level_roc(["a", "b"]), show_auc=False, eval_time=12.0,
            group_variable="grp",
        )
        assert all("AUC" not in (t.hovertemplate or "") for t in fig.data)

    def test_legend_bottom_right_no_title(self):
        fig = _render_roc_facet_grid(
            _per_level_roc(["a", "b"]), show_auc=True, eval_time=12.0,
            group_variable="grp",
        )
        assert fig.layout.legend.xanchor == "right"
        assert fig.layout.legend.yanchor == "bottom"
        assert fig.layout.legend.title.text is None

    def test_default_and_custom_title(self):
        fig = _render_roc_facet_grid(
            _per_level_roc(["a", "b"]), show_auc=True, eval_time=12.0,
            group_variable="grp",
        )
        assert fig.layout.title.text == "Subpopulation ROC: grp (t = 12.0)"
        fig2 = _render_roc_facet_grid(
            _per_level_roc(["a", "b"]), show_auc=True, eval_time=12.0,
            group_variable="grp", title="Custom",
        )
        assert fig2.layout.title.text == "Custom"


# ------------------------------------------------------------------
# _render_calibration_facet_grid
# ------------------------------------------------------------------

class TestRenderCalibrationFacetGrid:
    def test_panel_count(self):
        fig = _render_calibration_facet_grid(
            _per_level_cal(["a", "b", "c", "d"]),
            eval_time=12.0, group_variable="grp",
        )
        titles = {a.text for a in fig.layout.annotations}
        assert {"a", "b", "c", "d"} <= titles

    def test_perfect_diagonal_trace_per_panel(self):
        levels = ["a", "b", "c"]
        fig = _render_calibration_facet_grid(
            _per_level_cal(levels), eval_time=12.0, group_variable="grp",
        )
        perfect = [t for t in fig.data if t.name == "Perfect"]
        assert len(perfect) == len(levels)

    def test_total_trace_count(self):
        levels, models = ["a", "b"], ("coxph", "rsf")
        fig = _render_calibration_facet_grid(
            _per_level_cal(levels, models=models), eval_time=12.0,
            group_variable="grp",
        )
        # Each panel: 1 "Perfect" diagonal trace + 1 trace per model.
        assert len(fig.data) == len(levels) * (1 + len(models))

    def test_axis_titles(self):
        fig = _render_calibration_facet_grid(
            _per_level_cal(["a", "b", "c", "d"]), eval_time=12.0,
            group_variable="grp",
        )
        x_titles = [ax.title.text for ax in fig.select_xaxes()]
        y_titles = [ax.title.text for ax in fig.select_yaxes()]
        assert "Predicted" in x_titles
        assert "Observed (KM)" in y_titles

    def test_legend_top_right_with_title(self):
        fig = _render_calibration_facet_grid(
            _per_level_cal(["a", "b"]), eval_time=12.0, group_variable="grp",
        )
        assert fig.layout.legend.xanchor == "right"
        assert fig.layout.legend.yanchor == "top"
        assert fig.layout.legend.title.text == "Model"

    def test_no_dtick_on_axes(self):
        # Calibration axes do NOT set dtick (unlike ROC).
        fig = _render_calibration_facet_grid(
            _per_level_cal(["a", "b"]), eval_time=12.0, group_variable="grp",
        )
        assert all(ax.dtick is None for ax in fig.select_xaxes())

    def test_default_and_custom_title(self):
        fig = _render_calibration_facet_grid(
            _per_level_cal(["a", "b"]), eval_time=12.0, group_variable="grp",
        )
        assert fig.layout.title.text == (
            "Subpopulation Calibration: grp (t = 12.0)"
        )
        fig2 = _render_calibration_facet_grid(
            _per_level_cal(["a", "b"]), eval_time=12.0, group_variable="grp",
            title="Custom",
        )
        assert fig2.layout.title.text == "Custom"
