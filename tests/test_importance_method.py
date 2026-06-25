"""Tests for ``learner.importance()`` — the one-call importance entry point.

Thin dispatcher over the private ``_permutation_importance`` backend (default)
and the private ``_shap`` backend.  ``method='shap'`` aggregates the per-sample
SHAP values into a per-feature summary.  Requires ``evaluate()`` first, just
like the backends it wraps.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mlsurv import SurvivalLearner

_PERM_COLS = {"model_type", "feature", "coefficient", "perm_mean", "perm_std"}
_SHAP_COLS = ["model_type", "feature", "mean_abs_shap"]


def _has_shap():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


requires_shap = pytest.mark.skipif(not _has_shap(), reason="shap required")


# --- Fast (no SHAP needed) -------------------------------------------------

def test_shap_max_explain_default_agrees_across_entry_points():
    """importance(method='shap') and evaluate() explain the same #samples by default.

    Regression guard: the two SHAP entry points must share a
    ``max_explain`` / ``shap_max_explain`` default so the same model yields the
    same mean|SHAP| ranking regardless of which one you call.
    """
    import inspect

    importance_default = (
        inspect.signature(SurvivalLearner.importance)
        .parameters["max_explain"].default
    )
    evaluate_default = (
        inspect.signature(SurvivalLearner.evaluate)
        .parameters["shap_max_explain"].default
    )
    shap_backend_default = (
        inspect.signature(SurvivalLearner._shap)
        .parameters["max_explain"].default
    )

    assert importance_default == evaluate_default == shap_backend_default == 50


@pytest.mark.integration
def test_importance_bad_method_raises(small_data):
    """An unknown method must raise ValueError before touching state."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    with pytest.raises(ValueError, match="Unknown importance method 'bogus'"):
        learner.importance("coxph", method="bogus")


@pytest.mark.integration
def test_importance_before_evaluate_raises(small_data):
    """Default path inherits the 'call evaluate() first' guard from the wrapper."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")

    with pytest.raises(
        ValueError, match=r"Call evaluate\(\) before importance\(method='permutation'\)",
    ):
        learner.importance("coxph")


@pytest.mark.integration
def test_importance_default_matches_permutation_importance(small_data):
    """method='permutation' (default) returns exactly the backend result."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    out = learner.importance(
        "coxph", n_importance_repeats=3, random_state=7, show_progress=False,
    )
    expected = learner._permutation_importance(
        learner.models["coxph"], n_importance_repeats=3, random_state=7,
        show_progress=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert _PERM_COLS.issubset(out.columns)
    assert len(out) == X.shape[1]  # one row per feature (feature_selector=None)
    pd.testing.assert_frame_equal(out, expected)


# --- SHAP integration ------------------------------------------------------

@pytest.mark.shap
@pytest.mark.integration
@requires_shap
def test_importance_shap_returns_per_feature_summary(small_data):
    """method='shap' returns a sorted per-feature summary; full SHAP retained."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)
    assert learner._state.models["coxph"].evaluation.shap_values is None

    out = learner.importance(
        "coxph", method="shap",
        max_explain=10, max_background=10, show_progress=False,
    )

    # One row per feature, exact summary columns.
    assert list(out.columns) == _SHAP_COLS
    assert len(out) == out["feature"].nunique()
    assert len(out) == X.shape[1]
    assert (out["model_type"] == "coxph").all()

    # Mean absolute SHAP is non-negative and sorted descending.
    assert (out["mean_abs_shap"] >= 0).all()
    assert out["mean_abs_shap"].is_monotonic_decreasing

    # Full per-sample SHAP retained on the evaluation result.
    eval_after = learner._state.models["coxph"].evaluation
    assert eval_after.shap_values is not None
    assert "shap_value" in eval_after.shap_values.columns

    # mean_abs_shap is exactly mean(|shap_value|) per feature — pins the
    # aggregation formula so a dropped .abs() or .sum()-for-.mean() regression
    # can't slip through the structural checks above.
    expected = (
        eval_after.shap_values
        .assign(_abs=lambda d: d["shap_value"].abs())
        .groupby("feature")["_abs"].mean()
    )
    actual = out.set_index("feature")["mean_abs_shap"]
    pd.testing.assert_series_equal(
        actual.sort_index(), expected.sort_index(),
        check_names=False,
    )
