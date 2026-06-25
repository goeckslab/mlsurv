"""Regression tests for the plot_roc length-mismatch follow-up.

``learner.plot_roc(references='all')`` had the same length-mismatch bug
as the report's ROC overlay — `_plot_roc_impl_main` called
``compute_roc_curves`` once with all models+refs against the full
``ctx.y_data``, but NaN-bearing references in ``risk_scores`` have
length = non-NaN count. The fix in ``mlsurv/viz/_plot_impl_roc.py``
partitions ``risk_scores`` into models vs references and masks
``ctx.y_data`` per-reference using
``state.benchmark_result.reference_arrays[name]``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlsurv import SurvivalLearner


def _setup_learner_with_partial_ref(X_raw, y_raw, nan_frac=0.30, seed=42):
    """Build a SurvivalLearner trained + benchmarked with one NaN-bearing ref."""
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array(
        [(bool(e), t) for e, t in y_raw],
        dtype=[("event", bool), ("time", float)],
    )
    rng = np.random.default_rng(seed)
    ref_partial = rng.normal(size=len(X))
    ref_partial[rng.random(len(X)) < nan_frac] = np.nan

    learner = SurvivalLearner(X, y)
    learner.setup(feature_selector="topk", random_state=seed, test_fraction=0.3)
    learner.train("coxph")
    learner.benchmark(
        references={"ref_partial": ref_partial},
        n_bootstrap=2,
    )
    return learner


@pytest.mark.integration
def test_plot_roc_with_nan_bearing_reference(gbsg2_data):
    """plot_roc(references='all') must not raise length-mismatch."""
    X_raw, y_raw = gbsg2_data
    learner = _setup_learner_with_partial_ref(X_raw, y_raw)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = learner.plot_roc(references="all", return_data=True)

    counts = df.groupby("model").size()
    assert "coxph" in counts.index
    assert "ref_partial" in counts.index
    # Reference rows should be strictly fewer than model rows because the
    # masked y_data is smaller — same horizons × fewer FPR points.
    assert counts["ref_partial"] < counts["coxph"]


@pytest.mark.integration
def test_plot_roc_multi_ref_no_cross_contamination(gbsg2_data):
    """Two refs with different NaN counts must each get their own mask."""
    X_raw, y_raw = gbsg2_data
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array(
        [(bool(e), t) for e, t in y_raw],
        dtype=[("event", bool), ("time", float)],
    )
    rng = np.random.default_rng(7)
    ref_a = rng.normal(size=len(X))
    ref_a[rng.random(len(X)) < 0.10] = np.nan
    ref_b = rng.normal(size=len(X))
    ref_b[rng.random(len(X)) < 0.40] = np.nan

    learner = SurvivalLearner(X, y)
    learner.setup(feature_selector="topk", random_state=7, test_fraction=0.3)
    learner.train("coxph")
    learner.benchmark(
        references={"ref_a": ref_a, "ref_b": ref_b}, n_bootstrap=2,
    )

    df = learner.plot_roc(references="all", return_data=True)
    counts = df.groupby("model").size()
    # ref_a (more valid samples) should have more ROC points than ref_b.
    assert counts["ref_a"] > counts["ref_b"], (
        f"ref_a should have more ROC points than ref_b (less NaN), got "
        f"{dict(counts)}"
    )
    assert counts["coxph"] >= counts["ref_a"]


@pytest.mark.integration
def test_plot_roc_eval_times_match_between_model_and_ref(gbsg2_data):
    """Model and reference share the same eval_time grid in returned data."""
    X_raw, y_raw = gbsg2_data
    learner = _setup_learner_with_partial_ref(X_raw, y_raw)
    df = learner.plot_roc(references="all", return_data=True)
    model_times = set(df[df["model"] == "coxph"]["eval_time"].unique())
    ref_times = set(df[df["model"] == "ref_partial"]["eval_time"].unique())
    # The masked ref subset might fail at some horizons (no cases/controls);
    # any reference horizon present must be a subset of model horizons.
    assert ref_times.issubset(model_times), (
        f"reference eval_times {ref_times - model_times} not in model grid"
    )
    assert len(ref_times) > 0, "reference produced no ROC rows at any horizon"
