"""Tests for standalone permutation-importance recompute.

Exercised through the public entry point ``learner.importance(method='permutation')``
(the backend, ``_permutation_importance``, is private).
Recomputes permutation importance and attaches it to an existing
``EvaluationResult`` in place, returning the summary DataFrame.  Requires
``evaluate()`` first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner

_SUMMARY_COLS = {"model_type", "feature", "coefficient", "perm_mean", "perm_std"}


@pytest.mark.integration
def test_permutation_importance_before_evaluate_raises(small_data):
    """Calling before evaluate() must raise ValueError with a helpful message."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")

    with pytest.raises(
        ValueError,
        match=r"Call evaluate\(\) before importance\(method='permutation'\)",
    ):
        learner.importance("coxph", method="permutation")


@pytest.mark.integration
def test_permutation_importance_attaches_and_returns(small_data):
    """Recomputes importance, attaches in place, leaves other fields intact."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    eval_result = learner._state.models["coxph"].evaluation
    concordance_before = eval_result.concordance.copy()
    test_set_before = eval_result.test_set.copy()

    out = learner.importance(
        "coxph", method="permutation", n_importance_repeats=3, show_progress=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert _SUMMARY_COLS.issubset(out.columns)
    assert len(out) == X.shape[1]  # one row per feature (feature_selector=None)
    assert (out["model_type"] == "coxph").all()

    eval_after = learner._state.models["coxph"].evaluation
    # Attached in place (same DataFrame object returned).
    pd.testing.assert_frame_equal(out, eval_after.importance)
    assert eval_after.importance_iterations is not None
    assert (eval_after.importance_iterations["model_type"] == "coxph").all()

    # Cache invalidated: the aggregated property reflects the recomputed values.
    agg = learner.evaluation_result.importance.set_index("feature")["perm_mean"].sort_index()
    exp = out.set_index("feature")["perm_mean"].sort_index()
    pd.testing.assert_series_equal(agg, exp)

    # Unrelated evaluation fields untouched.
    pd.testing.assert_frame_equal(eval_after.concordance, concordance_before)
    pd.testing.assert_frame_equal(eval_after.test_set, test_set_before)


@pytest.mark.integration
def test_permutation_importance_resolves_default_when_one_model(small_data):
    """model=None resolves to the only trained model."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    out = learner.importance(
        method="permutation", n_importance_repeats=2, show_progress=False,
    )
    assert (out["model_type"] == "coxph").all()


@pytest.mark.integration
def test_permutation_importance_multiple_models_stacks_all(small_data):
    """models=None with >1 trained model computes importance for all of them."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.run(["coxph", "coxnet"], tune=False)

    out = learner.importance(
        method="permutation", n_importance_repeats=2, show_progress=False,
    )
    assert set(out["model_type"]) == {"coxph", "coxnet"}


@pytest.mark.integration
def test_permutation_importance_accepts_modelresult(small_data):
    """A ModelResult object resolves the same as its name string."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    mr = learner._state.models["coxph"]
    out = learner.importance(
        mr, method="permutation", n_importance_repeats=2, show_progress=False,
    )
    assert (out["model_type"] == "coxph").all()


@pytest.mark.integration
def test_permutation_importance_preserves_results_on_failure(small_data, monkeypatch):
    """If recomputation fails (empty sentinel), keep prior results and warn."""
    import warnings

    from mlsurv._warnings import SuggestionWarning
    import mlsurv.evaluation.importance as imp_mod

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=2)

    good = learner._state.models["coxph"].evaluation.importance.copy()

    def _fail(*args, **kwargs):
        empty, empty_iters = imp_mod._empty_importance_results()
        return empty, empty_iters, None

    monkeypatch.setattr(imp_mod, "_compute_permutation_importance", _fail)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = learner.importance("coxph", method="permutation", show_progress=False)

    assert any(issubclass(w.category, SuggestionWarning) for w in caught)
    # Prior good importance preserved (not overwritten with empties).
    pd.testing.assert_frame_equal(out, good)
    pd.testing.assert_frame_equal(
        learner._state.models["coxph"].evaluation.importance, good,
    )


@pytest.mark.integration
def test_permutation_importance_random_state_falls_back_to_learner(small_data):
    """random_state=None reuses the learner seed, making repeats reproducible."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    first = learner.importance(
        "coxph", method="permutation", n_importance_repeats=3, show_progress=False,
    )
    second = learner.importance(
        "coxph", method="permutation", n_importance_repeats=3, show_progress=False,
    )
    pd.testing.assert_frame_equal(
        first.set_index("feature")[["perm_mean", "perm_std"]],
        second.set_index("feature")[["perm_mean", "perm_std"]],
    )


@pytest.mark.integration
def test_permutation_importance_groups_collapse_dummies():
    """categorical_groups collapses dummy columns and fills importance_per_column."""
    rng = np.random.RandomState(0)
    n = 80
    stage = rng.choice(3, n)
    X = pd.DataFrame({
        "age": rng.normal(50, 10, n),
        "stage_I": (stage == 0).astype(float),
        "stage_II": (stage == 1).astype(float),
        "stage_III": (stage == 2).astype(float),
    })
    risk = 0.05 * X["age"].values
    time = rng.exponential(1.0 / np.exp(risk * 0.1))
    event = rng.random(n) < 0.7
    y = np.array(
        [(e, t) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )

    groups = {"stage": ["stage_I", "stage_II", "stage_III"]}
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None, categorical_groups=groups)
    learner.train("coxph")
    learner.evaluate("coxph", n_importance_repeats=1)

    out = learner.importance(
        "coxph", method="permutation", n_importance_repeats=3, show_progress=False,
    )

    assert "stage" in out["feature"].values
    assert "age" in out["feature"].values
    assert "stage_I" not in out["feature"].values

    eval_after = learner._state.models["coxph"].evaluation
    assert eval_after.importance_per_column is not None
    per_col_feats = set(eval_after.importance_per_column["feature"])
    assert {"stage_I", "stage_II", "stage_III"}.issubset(per_col_feats)

    # Grouped coefficient handling (dedup regression guard): the collapsed
    # group row carries NaN (not aggregatable), the ungrouped feature keeps its
    # real coefficient. A wrong-index regression in the deduped grouped branch
    # would surface here.
    coef = out.set_index("feature")["coefficient"]
    assert np.isnan(coef["stage"])
    assert np.isfinite(coef["age"])
