"""Tests for the unified ``metric=`` parameter on plot_metric / plot_metric_time.

``plot_metric`` and ``plot_metric_time`` share a single ``metric: str | list[str]``
parameter (the old ``metrics=`` keyword is gone). This file covers:

- the ``normalize_metric_param`` helper that all entry points share,
- public-API equivalence between a scalar and a 1-element list (collapse),
- public multi-panel output for 2+ metrics, and
- the hard break when the removed ``metrics=`` keyword is passed.

Tests that train models are marked ``@pytest.mark.integration``.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv.viz._metrics_helpers import normalize_metric_param


# ---------------------------------------------------------------------------
# Module-scoped fixture: train CoxPH + RSF, evaluate, bootstrap (small/fast)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def learner_with_bootstrap():
    """Trained CoxPH + RSF learner with evaluation + a tiny bootstrap.

    Module-scoped so the fit happens once.  None of the tests mutate the
    learner, so sharing it is safe.
    """
    from mlsurv.learner import SurvivalLearner
    from mlsurv.utils import SurvivalArray

    rng = np.random.RandomState(42)
    n_samples, n_features = 300, 8
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    true_coef = np.zeros(n_features)
    true_coef[:3] = [0.5, -0.3, 0.4]
    lp = X.values @ true_coef
    times = np.clip(np.exp(-lp + rng.randn(n_samples) * 0.5), 0.1, 10)
    events = rng.binomial(1, 0.7, n_samples).astype(bool)
    y = SurvivalArray(
        np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        )
    )

    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(test_fraction=0.3, random_state=42)
        learner.run(
            models=["coxph", "rsf"],
            tune=False,
            n_importance_repeats=0,
            bootstrap=False,
            verbose=False,
        )
        learner.bootstrap(n_bootstrap=2)
    return learner


def _panel_count(fig):
    """Count subplot panels in a make_subplots figure via its grid ref."""
    grid = getattr(fig, "_grid_ref", None)
    assert grid is not None, "expected a make_subplots figure with a grid ref"
    return sum(cell is not None for row in grid for cell in row)


# ===================================================================
# normalize_metric_param unit tests (no fitting needed)
# ===================================================================


class TestNormalizeMetricParam:
    """Unit tests for the shared metric-parameter normalizer."""

    def test_none(self):
        assert normalize_metric_param(None) == (None, None)

    def test_string(self):
        assert normalize_metric_param("auc") == ("auc", None)

    def test_single_element_list_collapses(self):
        assert normalize_metric_param(["auc"]) == ("auc", None)

    def test_multi_element_list(self):
        assert normalize_metric_param(["a", "b"]) == (None, ["a", "b"])

    def test_single_element_list_no_collapse(self):
        assert normalize_metric_param(
            ["auc"], collapse_singleton=False
        ) == (None, ["auc"])

    def test_tuple_accepted(self):
        assert normalize_metric_param(("a", "b")) == (None, ["a", "b"])

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError):
            normalize_metric_param([])

    def test_non_str_non_list_raises_type_error(self):
        with pytest.raises(TypeError):
            normalize_metric_param(123)

    def test_list_with_non_str_raises_type_error(self):
        with pytest.raises(TypeError):
            normalize_metric_param(["a", 1])


# ===================================================================
# Public-API equivalence: scalar == 1-element list (collapse)
# ===================================================================


@pytest.mark.integration
class TestPublicEquivalence:
    """A scalar metric and a 1-element list collapse to the same output."""

    def test_plot_metric_scalar_equals_singleton_list(self, learner_with_bootstrap):
        scalar = learner_with_bootstrap.plot_metric(
            metric="harrell", return_data=True,
        )
        singleton = learner_with_bootstrap.plot_metric(
            metric=["harrell"], return_data=True,
        )
        pd.testing.assert_frame_equal(scalar, singleton)

    def test_plot_metric_time_scalar_equals_singleton_list(
        self, learner_with_bootstrap,
    ):
        scalar = learner_with_bootstrap.plot_metric_time(
            metric="auc", return_data=True,
        )
        singleton = learner_with_bootstrap.plot_metric_time(
            metric=["auc"], return_data=True,
        )
        default = learner_with_bootstrap.plot_metric_time(return_data=True)
        pd.testing.assert_frame_equal(scalar, singleton)
        pd.testing.assert_frame_equal(scalar, default)


# ===================================================================
# Public multi-panel: 2+ metrics -> multi-panel figure
# ===================================================================


@pytest.mark.integration
class TestPublicMultiPanel:
    """A list of 2+ metric names produces a multi-panel figure."""

    def test_plot_metric_multi_panel(self, learner_with_bootstrap):
        fig = learner_with_bootstrap.plot_metric(
            metric=["harrell", "ipcw"], source="bootstrap",
        )
        assert isinstance(fig, go.Figure)
        assert _panel_count(fig) >= 2

    def test_plot_metric_time_two_panel(self, learner_with_bootstrap):
        fig = learner_with_bootstrap.plot_metric_time(
            metric=["auc", "brier"],
        )
        assert isinstance(fig, go.Figure)
        assert _panel_count(fig) == 2


# ===================================================================
# Hard break: the removed metrics= keyword raises TypeError
# ===================================================================


@pytest.mark.integration
class TestRemovedMetricsKeyword:
    """Passing the removed ``metrics=`` keyword is a hard TypeError."""

    def test_plot_metric_metrics_kwarg_raises(self, learner_with_bootstrap):
        with pytest.raises(TypeError) as exc_info:
            learner_with_bootstrap.plot_metric(metrics=["harrell", "ipcw"])
        assert "unexpected keyword argument 'metrics'" in str(exc_info.value)

    def test_plot_metric_time_metrics_kwarg_raises(self, learner_with_bootstrap):
        with pytest.raises(TypeError) as exc_info:
            learner_with_bootstrap.plot_metric_time(metrics=["auc", "brier"])
        assert "unexpected keyword argument 'metrics'" in str(exc_info.value)
