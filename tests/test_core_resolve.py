"""Tests for ``SurvivalLearner._resolve_all_models`` ordering.

The helper must rank models best-first by ``cv_metric_mean``, which
is always stored in higher-is-better space — ``score_survival_model``
negates lower-is-better scorers (e.g. integrated Brier returns
``-ibs``) before fold scores are aggregated. Models with missing /
non-finite ``cv_metric_mean`` are placed last.
"""

import numpy as np
import pytest

from mlsurv.learner import SurvivalLearner
from mlsurv.results.containers import ModelResult


def _stub_result(name: str, cv_mean):
    """Minimal ModelResult sufficient for ordering by cv_metric_mean."""
    cv_mean_f = float("nan") if cv_mean is None else float(cv_mean)
    return ModelResult(
        name=name,
        model=object(),  # _resolve_all_models only reads .model attribute
        cv_metric_scores=np.array([cv_mean_f]),
        cv_metric_mean=cv_mean_f,
        cv_metric_std=0.0,
    )


@pytest.fixture
def setup_learner(small_data):
    X, y = small_data

    def _make(scoring):
        learner = SurvivalLearner(X, y)
        learner.setup(scoring=scoring, verbose=False)
        return learner

    return _make


class TestResolveAllModelsOrdering:
    def test_higher_is_better_orders_descending(self, setup_learner):
        learner = setup_learner("harrell")
        learner._state.models = {
            "low": _stub_result("low", 0.60),
            "high": _stub_result("high", 0.70),
        }

        _, order, _ = learner._resolve_all_models()

        assert order == ["high", "low"]

    def test_lower_is_better_orders_descending_in_negated_space(self, setup_learner):
        # ``cv_metric_mean`` is stored in higher-is-better space — for
        # IBS, ``score_survival_model`` returns ``-ibs``. The
        # less-negative value (i.e. lower raw IBS) should therefore
        # rank first.
        learner = setup_learner("integrated_brier")
        learner._state.models = {
            "worse_ibs": _stub_result("worse_ibs", -0.20),  # raw IBS = 0.20
            "better_ibs": _stub_result("better_ibs", -0.15),  # raw IBS = 0.15
        }

        models, order, _ = learner._resolve_all_models()

        assert order == ["better_ibs", "worse_ibs"]
        # Returned dict iteration order must match cv_model_order so downstream
        # consumers that walk the dict see best-first.
        assert list(models.keys()) == order

    @pytest.mark.parametrize("scoring", ["harrell", "integrated_brier"])
    def test_missing_cv_placed_last(self, setup_learner, scoring):
        learner = setup_learner(scoring)
        learner._state.models = {
            "missing": _stub_result("missing", None),
            "a": _stub_result("a", 0.30),
            "b": _stub_result("b", 0.50),
        }

        _, order, _ = learner._resolve_all_models()

        assert order[-1] == "missing"
        assert set(order[:-1]) == {"a", "b"}

    @pytest.mark.parametrize("bad_cv", [np.nan, np.inf, -np.inf])
    def test_non_finite_cv_placed_last(self, setup_learner, bad_cv):
        learner = setup_learner("harrell")
        learner._state.models = {
            "bad": _stub_result("bad", bad_cv),
            "good": _stub_result("good", 0.65),
        }

        _, order, _ = learner._resolve_all_models()

        assert order[-1] == "bad"

    def test_empty_models_raises(self, setup_learner):
        learner = setup_learner("harrell")
        learner._state.models = {}

        with pytest.raises(ValueError, match="No trained models"):
            learner._resolve_all_models()
