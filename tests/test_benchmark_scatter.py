"""Tests for the benchmark scatter figure.

Covers:
- BenchmarkResult persists raw model/reference arrays.
- ``plot_benchmark_scatter`` viz primitive returns a Plotly figure with
  one dropdown button per model and the expected trace count.
- ``learner.plot_benchmark_scatter()`` returns a figure and ``return_data=True``
  yields the long-form DataFrame.
- The benchmarking report section exposes the new figure / data key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv import SurvivalLearner
from mlsurv.viz.benchmarking_scatter import plot_benchmark_scatter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic():
    """Inline synthetic survival data (module-scope-safe).

    Mirrors ``synthetic_data_clean`` from conftest without taking a
    function-scoped dependency.
    """
    rng = np.random.default_rng(42)
    n, p = 200, 20
    X = pd.DataFrame(
        rng.normal(size=(n, p)),
        columns=[f"feature_{i}" for i in range(p)],
    )
    risk = X["feature_0"].values + 0.5 * X["feature_1"].values
    time = rng.exponential(scale=np.exp(-0.5 * risk))
    event = rng.binomial(1, 0.7, size=n).astype(bool)
    y = np.array(list(zip(event, time)),
                 dtype=[("event", bool), ("time", float)])
    return X, y


@pytest.fixture(scope="module")
def benchmarked_learner():
    """Learner with a completed benchmark (two references, one model).

    Module-scoped — no test mutates the result.
    """
    X, y = _make_synthetic()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, verbose=False)
    learner.train("coxph", verbose=False)
    learner.benchmark(
        references={
            "ref_a": X["feature_0"].values,
            "ref_b": X["feature_1"].values,
        },
        bootstrap=False,
        verbose=False,
    )
    return learner


# ---------------------------------------------------------------------------
# BenchmarkResult persists raw arrays
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBenchmarkResultRawArrays:
    def test_raw_scores_populated(self, benchmarked_learner):
        br = benchmarked_learner._state.benchmark_result
        assert br.model_risk_scores is not None
        assert br.reference_arrays is not None
        assert set(br.model_risk_scores) == set(br.model_names)
        assert set(br.reference_arrays) == set(br.reference_names)

    def test_raw_arrays_row_aligned(self, benchmarked_learner):
        br = benchmarked_learner._state.benchmark_result
        m_lens = {len(v) for v in br.model_risk_scores.values()}
        r_lens = {len(v) for v in br.reference_arrays.values()}
        assert m_lens == r_lens
        assert len(m_lens) == 1


# ---------------------------------------------------------------------------
# Viz primitive
# ---------------------------------------------------------------------------


def _make_toy_inputs(n=30, seed=0):
    rng = np.random.default_rng(seed)
    refs = {
        "ref_a": rng.normal(size=n),
        "ref_b": rng.normal(size=n),
    }
    models = {
        "coxph": refs["ref_a"] + 0.3 * rng.normal(size=n),
        "rsf": refs["ref_b"] + 0.3 * rng.normal(size=n),
    }
    rows = []
    for m in models:
        for r in refs:
            rows.append({
                "model": m,
                "reference": r,
                "spearman_rho": 0.5,
                "spearman_p": 0.01,
                "pearson_r": 0.45,
                "pearson_p": 0.02,
            })
    return models, refs, pd.DataFrame(rows)


def test_plot_benchmark_scatter_returns_figure():
    models, refs, corr = _make_toy_inputs()
    fig = plot_benchmark_scatter(models, refs, corr, add_fit=False)
    assert isinstance(fig, go.Figure)


def test_plot_benchmark_scatter_dropdown_has_one_button_per_model():
    models, refs, corr = _make_toy_inputs()
    fig = plot_benchmark_scatter(models, refs, corr, add_fit=False)
    menus = fig.layout.updatemenus
    assert len(menus) == 1
    assert len(menus[0].buttons) == len(models)


def test_plot_benchmark_scatter_trace_count_without_fit():
    models, refs, corr = _make_toy_inputs()
    fig = plot_benchmark_scatter(models, refs, corr, add_fit=False)
    # One scatter trace per (model, reference)
    assert len(fig.data) == len(models) * len(refs)


def test_plot_benchmark_scatter_trace_count_with_fit():
    models, refs, corr = _make_toy_inputs()
    fig = plot_benchmark_scatter(models, refs, corr, add_fit=True)
    # Scatter + OLS line per (model, reference)
    assert len(fig.data) == 2 * len(models) * len(refs)


def test_plot_benchmark_scatter_legend_hidden():
    models, refs, corr = _make_toy_inputs()
    fig = plot_benchmark_scatter(models, refs, corr, add_fit=False)
    assert fig.layout.showlegend is False


def test_plot_benchmark_scatter_initial_visibility():
    """Only the first model's traces should be visible initially.

    With 2 models x 2 references and no OLS line, the first half of
    the trace list (first model) is visible; the rest hidden.
    """
    models, refs, corr = _make_toy_inputs()
    fig = plot_benchmark_scatter(models, refs, corr, add_fit=False)
    per_model = len(refs)
    for i, trace in enumerate(fig.data):
        expected = i < per_model  # first model's block only
        assert bool(trace.visible) is expected

    first_button = fig.layout.updatemenus[0].buttons[0]
    vis = first_button.args[0]["visible"]
    for trace, v in zip(fig.data, vis):
        assert bool(trace.visible) is bool(v)


def test_plot_benchmark_scatter_length_mismatch_raises():
    models, refs, corr = _make_toy_inputs()
    refs["ref_a"] = refs["ref_a"][:-1]
    with pytest.raises(ValueError, match="Length mismatch"):
        plot_benchmark_scatter(models, refs, corr, add_fit=False)


def test_plot_benchmark_scatter_empty_inputs_raise():
    _, refs, corr = _make_toy_inputs()
    with pytest.raises(ValueError, match="model_risk_scores"):
        plot_benchmark_scatter({}, refs, corr)


# ---------------------------------------------------------------------------
# Learner API
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLearnerPlotBenchmarkScatter:
    def test_returns_figure(self, benchmarked_learner):
        fig = benchmarked_learner.plot_benchmark_scatter(add_fit=False)
        assert isinstance(fig, go.Figure)

    def test_return_data_long_form(self, benchmarked_learner):
        df = benchmarked_learner.plot_benchmark_scatter(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {
            "model", "reference", "model_score", "reference_score",
        }
        br = benchmarked_learner._state.benchmark_result
        n_samples = len(next(iter(br.model_risk_scores.values())))
        assert len(df) == (
            len(br.model_names) * len(br.reference_names) * n_samples
        )

    def test_raises_without_benchmark(self):
        X, y = _make_synthetic()
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        with pytest.raises(RuntimeError, match="No benchmark result"):
            learner.plot_benchmark_scatter()


# ---------------------------------------------------------------------------
# Report section integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBenchmarkingReportSection:
    def test_section_exposes_scatter_figure(self, benchmarked_learner):
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        section = BenchmarkSection(benchmarked_learner)
        output = section.generate()
        figure_ids = {fi.figure_id for fi in output["figure_info"]}
        filenames = {fname for fname, _ in output["figures"]}

        assert "benchmark_correlation_scatter" in figure_ids
        assert "benchmark_correlation_scatter.png" in filenames
        assert "correlation_scatter" in section.data
        df = section.data["correlation_scatter"]
        assert set(df.columns) == {
            "model", "reference", "model_score", "reference_score",
        }
