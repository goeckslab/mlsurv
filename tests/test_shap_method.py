"""Tests for the SHAP backend behind ``importance(method='shap')``.

The public ``learner.shap()`` method was demoted to
the private ``learner._shap()`` backend, with ``importance(method='shap')`` as
the single public front door.  The backend still attaches whole-cohort SHAP
values to an existing ``EvaluationResult`` without re-running other metrics or
emitting the "Existing evaluation result will be overwritten" warning.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import SuggestionWarning


def _has_shap():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


requires_shap = pytest.mark.skipif(not _has_shap(), reason="shap required")


# --- Fast (no SHAP needed) -------------------------------------------------

@pytest.mark.integration
def test_shap_before_evaluate_raises(small_data):
    """importance(method='shap') before evaluate() must raise a helpful ValueError."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")

    with pytest.raises(
        ValueError, match="Call evaluate\\(\\) before importance\\(method='shap'\\)",
    ):
        learner.importance("coxph", method="shap")


# --- SHAP integration ------------------------------------------------------

@pytest.mark.shap
@pytest.mark.integration
@requires_shap
def test_shap_attaches_without_overwriting(small_data):
    """_shap() must attach shap_values without clobbering other evaluation fields."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    eval_result = learner._state.models["coxph"].evaluation
    importance_before = eval_result.importance.copy()
    concordance_before = eval_result.concordance.copy()
    auc_t_before = eval_result.auc_t.copy() if eval_result.auc_t is not None else None
    assert eval_result.shap_values is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = learner._shap(learner.models["coxph"], max_explain=10,
                            max_background=10, show_progress=False)

    overwrite_warnings = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning)
        and "overwrit" in str(w.message).lower()
    ]
    assert not overwrite_warnings, f"Unexpected overwrite warning: {overwrite_warnings}"

    eval_after = learner._state.models["coxph"].evaluation
    assert eval_after.shap_values is not None
    assert isinstance(eval_after.shap_values, pd.DataFrame)
    assert len(eval_after.shap_values) > 0
    assert set(["model_type", "feature", "sample_idx", "shap_value", "feature_value"]).issubset(
        eval_after.shap_values.columns,
    )
    assert (eval_after.shap_values["model_type"] == "coxph").all()

    pd.testing.assert_frame_equal(eval_after.importance, importance_before)
    pd.testing.assert_frame_equal(eval_after.concordance, concordance_before)
    if auc_t_before is not None:
        pd.testing.assert_frame_equal(eval_after.auc_t, auc_t_before)

    pd.testing.assert_frame_equal(out, eval_after.shap_values)


@pytest.mark.shap
@pytest.mark.integration
@requires_shap
def test_shap_resolves_default_when_one_model(small_data):
    """models=None resolves to the only trained model (single-form result)."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    out = learner.importance(
        method="shap", max_explain=5, max_background=5, show_progress=False,
    )
    assert (out["model_type"] == "coxph").all()


@pytest.mark.shap
@pytest.mark.integration
@requires_shap
def test_shap_then_plot_importance(small_data):
    """plot_importance() must render after importance(method='shap') (SHAP-flavored bars)."""
    import plotly.graph_objects as go

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)
    learner.importance(
        "coxph", method="shap", max_explain=10, max_background=10, show_progress=False,
    )

    fig = learner.plot_importance("coxph")
    assert isinstance(fig, go.Figure)
