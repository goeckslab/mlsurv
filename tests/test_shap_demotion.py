"""Tests for the demotion of ``shap()`` -> ``_shap()``; ``importance(method='shap')``
is the single public front door for global feature importance.

Covers:
- the public method ``shap`` is gone; the private ``_shap`` backend remains,
- ``importance(method='shap')`` returns the global summary
  ``[model_type, feature, mean_abs_shap]`` sorted descending,
- the per-patient values are stored on ``result.evaluation.shap_values`` by
  default, so ``plot_shap(kind='beeswarm')`` works without an extra opt-in,
- re-calling a populated importance prints a replace notice (both backends);
  the first call does not.
"""
from __future__ import annotations

import pytest

from mlsurv import SurvivalLearner
from mlsurv.evaluation.importance import _compute_permutation_importance


def _has_shap():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


requires_shap = pytest.mark.skipif(not _has_shap(), reason="shap required")


# --- Fast (no training) ----------------------------------------------------

def test_public_shap_removed(small_data):
    """The public ``shap`` method is gone; the private ``_shap`` backend stays."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    assert not hasattr(learner, "shap"), "public shap() should be demoted"
    assert hasattr(learner, "_shap"), "private _shap backend should exist"


# --- SHAP integration ------------------------------------------------------

@pytest.mark.shap
@pytest.mark.integration
@requires_shap
def test_importance_shap_return_shape(small_data):
    """importance(method='shap') returns [model_type, feature, mean_abs_shap] desc."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    summary = learner.importance(
        "coxph", method="shap", max_explain=10, max_background=10, show_progress=False,
    )

    assert list(summary.columns) == ["model_type", "feature", "mean_abs_shap"]
    assert "shap_std" not in summary.columns
    assert (summary["model_type"] == "coxph").all()
    assert summary["mean_abs_shap"].is_monotonic_decreasing


@pytest.mark.shap
@pytest.mark.integration
@requires_shap
def test_importance_shap_populates_attribute_and_beeswarm(small_data):
    """Per-patient shap_values is stored by default; beeswarm reads it (no opt-in)."""
    import plotly.graph_objects as go

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)
    assert learner._state.models["coxph"].evaluation.shap_values is None

    learner.importance(
        "coxph", method="shap", max_explain=10, max_background=10, show_progress=False,
    )

    ev = learner._state.models["coxph"].evaluation
    assert ev.shap_values is not None
    assert len(ev.shap_values) > 0
    fig = learner.plot_shap("coxph", kind="beeswarm")
    assert isinstance(fig, go.Figure)


@pytest.mark.shap
@pytest.mark.integration
@requires_shap
def test_shap_overwrite_notice(small_data, capsys):
    """First SHAP computation has no replace notice; the second does."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)
    capsys.readouterr()  # drop evaluate() output

    learner.importance("coxph", method="shap", max_explain=8, max_background=8)
    first = capsys.readouterr().out
    assert "Computing SHAP for coxph" in first
    assert "(replacing previous results)" not in first

    learner.importance("coxph", method="shap", max_explain=8, max_background=8)
    second = capsys.readouterr().out
    assert "(replacing previous results)" in second


@pytest.mark.integration
def test_permutation_overwrite_notice(small_data, capsys):
    """evaluate()'s importance line has no notice; a post-hoc recompute does.

    evaluate() is the permutation backend's *first* (no-prior) computation —
    it runs with the default ``replacing=False`` and so prints no notice.  A
    later ``importance(method='permutation')`` recompute sees the populated
    result and prints the replace notice.  (A first standalone
    ``importance(method='permutation')`` with no prior is unreachable: evaluate()
    rejects ``n_importance_repeats=0`` and always populates importance first.)
    """
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")

    learner.evaluate("coxph", n_importance_repeats=1)
    eval_out = capsys.readouterr().out
    assert "Computing permutation importance for coxph" in eval_out
    assert "(replacing previous results)" not in eval_out

    learner.importance("coxph", method="permutation", n_importance_repeats=1)
    post = capsys.readouterr().out
    assert "(replacing previous results)" in post


@pytest.mark.integration
def test_compute_permutation_importance_replacing_flag(small_data, capsys):
    """The helper's `replacing` flag toggles the notice deterministically (both branches)."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    model = learner._state.models["coxph"].model

    _compute_permutation_importance(
        model, learner.X_test, learner.y_test, 1, 42,
        show_progress=True, model_name="coxph", replacing=False,
    )
    no_notice = capsys.readouterr().out
    assert "Computing permutation importance for coxph" in no_notice
    assert "(replacing previous results)" not in no_notice

    _compute_permutation_importance(
        model, learner.X_test, learner.y_test, 1, 42,
        show_progress=True, model_name="coxph", replacing=True,
    )
    notice = capsys.readouterr().out
    assert "(replacing previous results)" in notice


@pytest.mark.integration
def test_plot_shap_error_mentions_include_shap(small_data):
    """plot_shap before any SHAP computation errors and names include_shap (criterion 4)."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)  # no include_shap -> shap_values is None

    with pytest.raises(ValueError, match="include_shap"):
        learner.plot_shap("coxph", kind="beeswarm")


@pytest.mark.shap
@pytest.mark.integration
@requires_shap
def test_evaluate_include_shap_then_plot_shap(small_data):
    """evaluate(include_shap=True) stores shap_values; learner.plot_shap works (criterion 2)."""
    import plotly.graph_objects as go

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate(
        "coxph", n_importance_repeats=1,
        include_shap=True, shap_max_explain=10, shap_max_background=10,
    )

    assert learner._state.models["coxph"].evaluation.shap_values is not None
    fig = learner.plot_shap("coxph", kind="beeswarm")
    assert isinstance(fig, go.Figure)
