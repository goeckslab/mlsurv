"""Tests for ``learner.plot_validation_heatmap`` (faceted cross-cohort).

The validation metrics heatmap was report-only (its Programmatic Access
snippet was a bare ``from mlsurv.viz.metrics import ...`` with no runnable
call). This adds a public learner method so the snippet becomes
``fig = learner.plot_validation_heatmap()``. The figure is faceted across
cohorts, so it needs >= 2 validation cohorts with >= 2 comparable metrics.
"""

import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv import SurvivalLearner


def _gbsg2_xy():
    """Load gbsg2 as cat-coded X and a structured survival y."""
    from sksurv.datasets import load_gbsg2

    X_raw, y_raw = load_gbsg2()
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array(
        [(bool(e), float(t)) for e, t in y_raw],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


@pytest.fixture(scope="module")
def _trained_template():
    """Setup + train two models once; no validation cohorts yet.

    Tests deepcopy this and add their own cohorts, so the (expensive) train
    happens a single time for the module.
    """
    X, y = _gbsg2_xy()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None, test_fraction=0.3)
    learner.run(["coxph", "coxnet"], tune=False, bootstrap=False, verbose=False)
    return learner, X, y


def _with_cohorts(template, n_cohorts):
    """Deepcopy the template and register ``n_cohorts`` validation cohorts."""
    learner, X, y = template
    learner = copy.deepcopy(learner)
    half = len(X) // 2
    splits = [
        (X.iloc[half:], y[half:], "pilot"),
        (X.iloc[:half], y[:half], "external"),
    ]
    for Xv, yv, name in splits[:n_cohorts]:
        learner.validate(Xv, yv, name=name, bootstrap=False, verbose=False)
    return learner


@pytest.mark.integration
def test_two_cohorts_returns_figure(_trained_template):
    """>= 2 cohorts -> a Plotly Figure."""
    learner = _with_cohorts(_trained_template, 2)
    fig = learner.plot_validation_heatmap()
    assert isinstance(fig, go.Figure)


@pytest.mark.integration
def test_return_data_is_tidy_frame(_trained_template):
    """``return_data=True`` -> a non-empty tidy DataFrame keyed by cohort."""
    learner = _with_cohorts(_trained_template, 2)
    df = learner.plot_validation_heatmap(return_data=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Cohort" in df.columns and "Model" in df.columns
    # Two cohorts x two models = four rows.
    assert set(df["Cohort"]) == {"pilot", "external"}


@pytest.mark.integration
def test_models_filter(_trained_template):
    """A ``models=`` filter narrows the rows to one model per cohort."""
    learner = _with_cohorts(_trained_template, 2)
    df = learner.plot_validation_heatmap(models=["coxph"], return_data=True)
    assert isinstance(df, pd.DataFrame)
    # One model x two cohorts = two rows, all the same (coxph) model.
    assert len(df) == 2
    assert df["Model"].nunique() == 1
    assert set(df["Cohort"]) == {"pilot", "external"}
    fig = learner.plot_validation_heatmap(models=["coxph"])
    assert isinstance(fig, go.Figure)


@pytest.mark.integration
def test_single_metric_arg_raises(_trained_template):
    """An explicit ``metrics=`` of one entry can't form a cross-cohort z-score."""
    from mlsurv.constants import ALL_METRICS

    learner = _with_cohorts(_trained_template, 2)
    with pytest.raises(ValueError, match="at least 2 metrics"):
        learner.plot_validation_heatmap(metrics=[ALL_METRICS[0]])


@pytest.mark.integration
def test_one_cohort_raises(_trained_template):
    """A single cohort cannot form a cross-cohort facet -> ValueError."""
    learner = _with_cohorts(_trained_template, 1)
    with pytest.raises(ValueError, match="at least 2 validation cohorts"):
        learner.plot_validation_heatmap()


@pytest.mark.integration
def test_zero_cohorts_raises(_trained_template):
    """No validation cohorts at all -> the same clear ValueError."""
    learner = _with_cohorts(_trained_template, 0)
    with pytest.raises(ValueError, match="at least 2 validation cohorts"):
        learner.plot_validation_heatmap()


# --- fast unit tests for the shared builder (no training) ---------------

def _fake_cohort(recal):
    """A minimal ValidationResult stand-in for the heatmap builder.

    Carries the two attributes the builder reads: a ``concordance`` table
    (one row per model, with the first two metrics' concordance columns) and
    a ``recalibrated_models`` tuple.
    """
    import types

    from mlsurv.constants import ALL_METRICS, get_concordance_column

    cols = [get_concordance_column(m) for m in ALL_METRICS[:2]]
    df = pd.DataFrame([
        {"model_type": "coxph", cols[0]: 0.70, cols[1]: 0.66},
        {"model_type": "coxnet", cols[0]: 0.68, cols[1]: 0.64},
    ])
    return types.SimpleNamespace(concordance=df, recalibrated_models=recal)


def test_builder_metric_cols_and_no_recal_marking():
    """``metric_cols`` are the metric keys; ``heatmap_recal`` is always empty.

    The validation heatmap no longer marks a separate "recalibrated" set —
    the deployed (calibrated) metrics ARE the normal metrics — so no model
    rows carry a "(recal)" suffix.
    """
    from mlsurv.constants import ALL_METRICS
    from mlsurv.viz._data_metrics import build_validation_heatmap_data

    vr = {
        "A": _fake_cohort(("coxph", "coxnet")),
        "B": _fake_cohort(("coxph",)),
    }
    cohort_dfs, metric_cols, heatmap_recal, tidy_df = (
        build_validation_heatmap_data(vr)
    )

    assert set(cohort_dfs) == {"A", "B"}
    assert metric_cols == ALL_METRICS[:2]
    assert heatmap_recal == set()
    assert not any("(recal)" in m for m in tidy_df["Model"])


def test_builder_model_names_filter_is_consistent():
    """``model_names`` filters wide and tidy frames together."""
    from mlsurv.viz._data_metrics import build_validation_heatmap_data

    vr = {"A": _fake_cohort(()), "B": _fake_cohort(())}
    cohort_dfs, _metric_cols, _recal, tidy_df = (
        build_validation_heatmap_data(vr, model_names=["coxph"])
    )

    assert all(set(df["model_type"]) == {"coxph"} for df in cohort_dfs.values())
    # One model x two cohorts in the tidy frame, none labelled coxnet.
    assert len(tidy_df) == 2
    assert not tidy_df["Model"].str.contains("Net", case=False).any()
