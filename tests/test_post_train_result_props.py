"""Read-only property access for post-train analysis results.

Covers the properties that expose stashed results without reaching into
``learner._state``: singular ``benchmark_result`` / ``kang_comparison_result``,
and the per-model dicts ``logrank_test_results`` / ``rmst_test_results``
and the per-method importance-test dicts
``pimp_importance_test_results`` / ``bootstrap_importance_test_results``,
each backed by the same-named state field. The single-model-
per-call tests (``logrank_test``, ``rmst_test``, ``importance_test``)
intentionally expose ONLY per-model dicts — there is no singular
``*_result`` property, because a singular accessor would silently reflect
only the latest model. Also pins the method renames (``rmst_comparison``
-> ``rmst_test``, ``importance_pvalues`` -> ``importance_test``) and the
state-field renames.

These tests only call ``setup()`` on small synthetic data (no training), so
they stay in the fast tier. The "populated after the real call" behavior is
covered by ``test_persist_stat_tests.py`` (integration).
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from mlsurv.learner import SurvivalLearner
from mlsurv.learner._state import LearnerState
from mlsurv.results.containers import ModelResult
from mlsurv.utils import SurvivalArray


def _fake_model_result(name="coxph", is_tuned=False):
    return ModelResult(
        name=name, model=None, cv_metric_scores=np.array([0.7]),
        cv_metric_mean=0.7, cv_metric_std=0.0, is_tuned=is_tuned,
    )

_RESULT_PROPERTIES = [
    "benchmark_result",
    "kang_comparison_result",
]

# property name -> backing state field (1:1 here, but spelled out so a future
# divergence is caught)
_PROPERTY_TO_FIELD = {
    "benchmark_result": "benchmark_result",
    "kang_comparison_result": "kang_comparison_result",
}


@pytest.fixture
def setup_learner():
    """A learner that is set up but has run no post-train analysis."""
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({
        "age": rng.standard_normal(n) * 10 + 60,
        "biomarker": rng.standard_normal(n),
    })
    times = np.clip(np.exp(rng.standard_normal(n)), 0.1, 10)
    events = rng.binomial(1, 0.7, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    ))
    learner = SurvivalLearner(X, y)
    learner.setup(verbose=False)
    return learner


class TestResultPropertiesBeforeCompute:
    @pytest.mark.parametrize("prop", _RESULT_PROPERTIES)
    def test_none_before_relevant_step(self, setup_learner, prop):
        assert getattr(setup_learner, prop) is None


class TestResultPropertyPassthrough:
    """Each property returns exactly the stashed ``_state`` object."""

    @pytest.mark.parametrize("prop,field", list(_PROPERTY_TO_FIELD.items()))
    def test_property_returns_state_field(self, setup_learner, prop, field):
        sentinel = pd.DataFrame({"x": [1, 2, 3]})
        setattr(setup_learner._state, field, sentinel)
        assert getattr(setup_learner, prop) is sentinel

    @pytest.mark.parametrize("prop", _RESULT_PROPERTIES)
    def test_property_is_read_only(self, setup_learner, prop):
        with pytest.raises(AttributeError):
            setattr(setup_learner, prop, pd.DataFrame())


# Per-model dict properties (default ``{}``, not ``None``): the importance-test
# dicts and the risk-group test dicts.
_PER_MODEL_DICT_PROPERTIES = [
    "pimp_importance_test_results",
    "bootstrap_importance_test_results",
    "logrank_test_results",
    "rmst_test_results",
]

# Per-model view name -> the ModelResult field it now reads from.
_PER_MODEL_PROP_TO_RESULT_ATTR = {
    "pimp_importance_test_results": "pimp_importance_test",
    "bootstrap_importance_test_results": "bootstrap_importance_test",
    "logrank_test_results": "logrank_test",
    "rmst_test_results": "rmst_test",
}


class TestPerModelResultsDicts:
    """The per-model dict views (default empty, not ``None``)."""

    @pytest.mark.parametrize("prop", _PER_MODEL_DICT_PROPERTIES)
    def test_empty_before_call(self, setup_learner, prop):
        assert getattr(setup_learner, prop) == {}

    @pytest.mark.parametrize("prop", _PER_MODEL_DICT_PROPERTIES)
    def test_view_reflects_modelresult_field(self, setup_learner, prop):
        # The view assembles from each ModelResult's table, returning
        # the exact stored object.
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = _fake_model_result("coxph")
        setattr(result, _PER_MODEL_PROP_TO_RESULT_ATTR[prop], df)
        setup_learner._state.models["coxph"] = result
        assert getattr(setup_learner, prop)["coxph"] is df

    @pytest.mark.parametrize("prop", _PER_MODEL_DICT_PROPERTIES)
    def test_read_only(self, setup_learner, prop):
        # Both the learner property and the assembled mappingproxy reject writes.
        with pytest.raises(AttributeError):
            setattr(setup_learner, prop, {})
        with pytest.raises(TypeError):
            getattr(setup_learner, prop)["coxph"] = pd.DataFrame()


# Every per-model derived-state view (importance/risk-group test tables), on
# both the learner and the underlying LearnerState.
_ALL_VIEW_PROPERTIES = list(_PER_MODEL_DICT_PROPERTIES)


class TestViewsAreReadOnlyProxies:
    """Every derived-state view is a read-only ``mappingproxy``."""

    @pytest.mark.parametrize("prop", _ALL_VIEW_PROPERTIES)
    def test_learner_view_is_mappingproxy(self, setup_learner, prop):
        from types import MappingProxyType
        assert isinstance(getattr(setup_learner, prop), MappingProxyType)
        assert isinstance(getattr(setup_learner._state, prop), MappingProxyType)

    @pytest.mark.parametrize("prop", _ALL_VIEW_PROPERTIES)
    def test_item_assignment_raises(self, setup_learner, prop):
        with pytest.raises(TypeError):
            getattr(setup_learner, prop)["coxph"] = object()

    def test_no_singular_or_legacy_property(self, setup_learner):
        """Only the per-model dicts are public; singular names dropped."""
        # importance_test
        assert not hasattr(setup_learner, "importance_test_result")
        assert not hasattr(setup_learner, "importance_test_results")
        # logrank_test / rmst_test
        assert not hasattr(setup_learner, "logrank_test_result")
        assert not hasattr(setup_learner, "rmst_test_result")


class TestRiskGroupTestInvalidation:
    """Risk-group tables ride on the ModelResult: they drop when the model is
    overwritten. Exercised via ``_install_tuned_result`` with no training, so
    they stay in the fast tier."""

    def _seed(self, learner, key, *, empty=False):
        if empty:
            df = pd.DataFrame(columns=["model_type", "comparison", "p_value"])
        else:
            df = pd.DataFrame({
                "model_type": [key],
                "comparison": ["omnibus"],
                "p_value": [0.01],
            })
        df.attrs["model_name"] = key
        result = learner._state.models.get(key) or _fake_model_result(key)
        result.logrank_test = df
        result.rmst_test = df
        learner._state.models[key] = result
        return df

    def test_overwrite_drops_both(self, setup_learner):
        self._seed(setup_learner, "rsf")
        self._seed(setup_learner, "coxph")
        setup_learner._install_tuned_result(
            "rsf", _fake_model_result("rsf", is_tuned=True),
        )
        assert "rsf" not in setup_learner.logrank_test_results
        assert "rsf" not in setup_learner.rmst_test_results
        # An unrelated model is untouched.
        assert "coxph" in setup_learner.logrank_test_results


class TestMethodRename:
    def test_rmst_test_method_exists(self, setup_learner):
        assert callable(getattr(setup_learner, "rmst_test", None))

    def test_old_rmst_comparison_method_gone(self, setup_learner):
        assert not hasattr(setup_learner, "rmst_comparison")

    def test_importance_test_method_exists(self, setup_learner):
        assert callable(getattr(setup_learner, "importance_test", None))

    def test_old_importance_pvalues_method_gone(self, setup_learner):
        assert not hasattr(setup_learner, "importance_pvalues")

    def test_logrank_test_method_kept(self, setup_learner):
        # logrank_test() keeps its name; no logrank_comparison introduced.
        assert callable(getattr(setup_learner, "logrank_test", None))
        assert not hasattr(setup_learner, "logrank_comparison")


class TestStateFieldRename:
    def test_new_fields_present(self):
        names = {f.name for f in dataclasses.fields(LearnerState)}
        assert {"benchmark_result", "kang_comparison_result"} <= names

    def test_per_model_views_are_properties_not_fields(self):
        """The per-model tables now ride on ``ModelResult``. The old dict
        names survive as read-only ``LearnerState`` properties, not fields."""
        names = {f.name for f in dataclasses.fields(LearnerState)}
        for prop in (
            "logrank_test_results", "rmst_test_results",
            "pimp_importance_test_results", "bootstrap_importance_test_results",
        ):
            assert prop not in names, f"{prop} should no longer be a field"
            assert isinstance(getattr(LearnerState, prop), property)

    def test_old_fields_absent(self):
        names = {f.name for f in dataclasses.fields(LearnerState)}
        assert "rmst_comparison" not in names
        assert "logrank_result" not in names
        assert "kang_comparison" not in names
        # singular dropped.
        assert "pimp_result" not in names
        assert "pimp_results" not in names
        # singular risk-group fields dropped in favor of per-model views.
        assert "rmst_test_result" not in names
        assert "logrank_test_result" not in names
