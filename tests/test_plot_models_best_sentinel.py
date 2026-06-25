"""Tests for the identifier-based ``plot_*(models=)`` contract.

Every plural ``models=`` plot method routes through
``viz._plot_impl_shared._resolve_models_arg``, which now:

- resolves the ``'best'`` sentinel (standalone or as a list element) to the
  top model by CV score, de-duplicating the result, and
- rejects a bare ``ModelResult`` with a guiding ``TypeError`` pointing at
  ``.name`` / ``'best'``.

The singular ``model=`` plot methods resolve ``'best'`` via the learner-level
``_resolve_best`` helper. ``mlsurv._shared.best_model_key`` is the shared
selector behind both ``learner.best_model`` and the sentinel.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlsurv._shared import best_model_key
from mlsurv._warnings import InternalWarning, SuggestionWarning
from mlsurv.learner import SurvivalLearner
from mlsurv.results.containers import ModelResult
from mlsurv.viz._context import PlotContext
from mlsurv.viz._plot_impl_shared import (
    _models_type_error,
    _resolve_best_name,
    _resolve_models_arg,
)


# ── best_model_key (shared selector) ───────────────────────────────────────


_UNSET = object()


def _mr(cv, *, synthetic=False, scoring_metric=None, harrell=None, cv_metrics=_UNSET):
    """Minimal stand-in for a ModelResult: only the attrs the selector reads.

    ``harrell`` (when given) populates ``cv_metrics['harrell']['mean']`` —
    the value the heterogeneous-scoring fallback ranks by. Pass ``cv_metrics``
    explicitly to inject a malformed structure (overrides ``harrell``).
    """
    if cv_metrics is _UNSET:
        cv_metrics = {"harrell": {"mean": harrell}} if harrell is not None else None
    return SimpleNamespace(
        cv_metric_mean=cv,
        is_synthetic=synthetic,
        scoring_metric=scoring_metric,
        cv_metrics=cv_metrics,
    )


class TestBestModelKey:
    def test_picks_highest_cv_metric_mean(self):
        models = {"coxph": _mr(0.70), "coxnet": _mr(0.82), "rsf": _mr(0.66)}
        assert best_model_key(models) == "coxnet"

    def test_excludes_synthetic(self):
        # The synthetic model has the highest score but must be ignored.
        models = {"coxph": _mr(0.70), "ref": _mr(0.99, synthetic=True)}
        assert best_model_key(models) == "coxph"

    def test_empty_returns_none(self):
        assert best_model_key({}) is None

    def test_all_synthetic_returns_none(self):
        assert best_model_key({"ref": _mr(0.9, synthetic=True)}) is None


class TestBestModelKeyHeterogeneous:
    """Cross-metric ``best_model`` ranking (heterogeneous ``scoring_metric``)."""

    def test_homogeneous_explicit_metric_no_warning(self):
        # Both scored on the same metric: rank by cv_metric_mean, no warning.
        models = {
            "coxph": _mr(0.70, scoring_metric="harrell"),
            "rsf": _mr(0.82, scoring_metric="harrell"),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert best_model_key(models) == "rsf"
        assert not [w for w in caught if issubclass(w.category, SuggestionWarning)]

    def test_mix_none_and_one_metric_is_homogeneous(self):
        # One real metric + one None metric → only one scale, no warning.
        models = {
            "coxph": _mr(0.70, scoring_metric="harrell"),
            "rsf": _mr(0.82, scoring_metric=None),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert best_model_key(models) == "rsf"
        assert not [w for w in caught if issubclass(w.category, SuggestionWarning)]

    def test_heterogeneous_emits_suggestion_warning(self):
        models = {
            "coxph": _mr(-0.15, scoring_metric="integrated_brier", harrell=0.65),
            "rsf": _mr(0.70, scoring_metric="harrell", harrell=0.72),
        }
        with pytest.warns(SuggestionWarning) as record:
            best_model_key(models)
        msg = str(record[0].message)
        assert "integrated_brier" in msg and "harrell" in msg

    def test_heterogeneous_ranks_by_harrell_not_cv_metric_mean(self):
        # cv_metric_mean max (coxph, 0.95) disagrees with Harrell max (rsf, 0.80).
        models = {
            "coxph": _mr(0.95, scoring_metric="integrated_brier", harrell=0.60),
            "rsf": _mr(0.70, scoring_metric="harrell", harrell=0.80),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert best_model_key(models) == "rsf"

    def test_heterogeneous_harrell_tie_keeps_first(self):
        # Equal Harrell means → first-inserted key wins (enumerate tie-break).
        models = {
            "coxph": _mr(-0.15, scoring_metric="integrated_brier", harrell=0.70),
            "rsf": _mr(0.71, scoring_metric="harrell", harrell=0.70),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert best_model_key(models) == "coxph"

    def test_heterogeneous_skips_model_missing_harrell(self):
        # coxph has no cv_metrics → skipped; rsf wins; InternalWarning lists coxph.
        models = {
            "coxph": _mr(0.95, scoring_metric="integrated_brier", harrell=None),
            "rsf": _mr(0.70, scoring_metric="harrell", harrell=0.80),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert best_model_key(models) == "rsf"
        notes = [w for w in caught if issubclass(w.category, InternalWarning)]
        assert notes and "coxph" in str(notes[0].message)

    def test_heterogeneous_skips_nan_harrell(self):
        models = {
            "coxph": _mr(0.95, scoring_metric="integrated_brier", harrell=np.nan),
            "rsf": _mr(0.70, scoring_metric="harrell", harrell=0.80),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert best_model_key(models) == "rsf"

    def test_heterogeneous_all_missing_harrell_falls_back(self):
        # No model exposes a Harrell mean → never None; raw ordering wins, and
        # the fallback is user-visible (SuggestionWarning) because it
        # contradicts the "ranking by Harrell" promise.
        models = {
            "coxph": _mr(0.95, scoring_metric="integrated_brier", harrell=None),
            "rsf": _mr(0.70, scoring_metric="harrell", harrell=None),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = best_model_key(models)
        assert result == "coxph"  # falls back to highest cv_metric_mean
        fallback = [w for w in caught
                    if issubclass(w.category, SuggestionWarning)
                    and "raw cross-validated ordering" in str(w.message)]
        assert fallback

    @pytest.mark.parametrize("bad_cv_metrics", [
        {"ipcw": {"mean": 0.5}},          # no 'harrell' key
        {"harrell": {}},                  # empty harrell entry
        {"harrell": {"std": 0.1}},        # harrell entry without 'mean'
        {"harrell": {"mean": "0.7"}},     # non-numeric mean
        {"harrell": {"mean": True}},      # bool mean (must not coerce to 1.0)
        {"harrell": 0.7},                 # harrell entry is not a dict
    ])
    def test_heterogeneous_skips_malformed_harrell(self, bad_cv_metrics):
        # A malformed cv_metrics is skipped (not fabricated); the well-formed
        # model wins and an InternalWarning lists the skipped one.
        models = {
            "coxph": _mr(0.95, scoring_metric="integrated_brier",
                         cv_metrics=bad_cv_metrics),
            "rsf": _mr(0.70, scoring_metric="harrell", harrell=0.80),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert best_model_key(models) == "rsf"
        notes = [w for w in caught if issubclass(w.category, InternalWarning)]
        assert notes and "coxph" in str(notes[0].message)

    def test_heterogeneous_ranks_among_finite_with_one_skipped(self):
        # Two finite-Harrell models + one skipped: the among-finite ranking
        # must still pick the higher Harrell, not the higher cv_metric_mean.
        models = {
            "coxph": _mr(0.95, scoring_metric="integrated_brier", harrell=0.60),
            "gbsa": _mr(0.50, scoring_metric="harrell", harrell=None),  # skipped
            "rsf": _mr(0.70, scoring_metric="harrell", harrell=0.80),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert best_model_key(models) == "rsf"  # 0.80 > 0.60 Harrell
        notes = [w for w in caught if issubclass(w.category, InternalWarning)]
        assert notes and "gbsa" in str(notes[0].message)

    def test_synthetic_metric_excluded_from_detection(self):
        # A synthetic model's third metric must not trigger the cross-metric path.
        models = {
            "coxph": _mr(0.70, scoring_metric="harrell"),
            "rsf": _mr(0.82, scoring_metric="harrell"),
            "ref": _mr(0.99, synthetic=True, scoring_metric="integrated_brier"),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert best_model_key(models) == "rsf"
        assert not [w for w in caught if issubclass(w.category, SuggestionWarning)]


# ── _resolve_models_arg: 'best' sentinel + type errors (resolver unit) ──────


def _ctx_with(models_cv):
    """A main-source PlotContext whose state + availability cover ``models_cv``.

    ``state_ref.models`` drives ``'best'`` selection; ``fitted_models`` drives
    the availability check the resolver runs after resolving names.
    """
    state = SimpleNamespace(models={n: _mr(cv) for n, cv in models_cv.items()})
    return PlotContext(
        name="main",
        source="main",
        fitted_models={n: object() for n in models_cv},
        state_ref=state,
    )


class TestBestSentinelResolution:
    def test_standalone_best_resolves_to_top(self):
        ctx = _ctx_with({"coxph": 0.70, "coxnet": 0.82})
        assert _resolve_models_arg(ctx, "best") == ["coxnet"]

    def test_best_as_list_element(self):
        ctx = _ctx_with({"coxph": 0.70, "coxnet": 0.82})
        assert _resolve_models_arg(ctx, ["best", "coxph"]) == ["coxnet", "coxph"]

    def test_best_dedups_when_it_repeats_a_named_model(self):
        # 'best' is coxnet; passing both must not draw coxnet twice.
        ctx = _ctx_with({"coxph": 0.70, "coxnet": 0.82})
        assert _resolve_models_arg(ctx, ["best", "coxnet"]) == ["coxnet"]

    def test_best_with_no_models_raises_valueerror(self):
        ctx = _ctx_with({})
        with pytest.raises(ValueError, match="No trained models to resolve"):
            _resolve_models_arg(ctx, "best")

    def test_resolve_best_name_helper(self):
        ctx = _ctx_with({"coxph": 0.70, "coxnet": 0.82})
        assert _resolve_best_name(ctx) == "coxnet"


class TestModelsTypeErrors:
    def test_bare_int_raises_generic_typeerror(self):
        ctx = _ctx_with({"coxph": 0.70})
        with pytest.raises(TypeError, match="models must be None, str, or list"):
            _resolve_models_arg(ctx, 42)  # type: ignore[arg-type]

    def test_models_type_error_message_for_modelresult(self):
        # Build a real ModelResult so the isinstance branch fires.
        mr = _make_modelresult("coxph")
        err = _models_type_error(mr)
        assert isinstance(err, TypeError)
        msg = str(err)
        assert "ModelResult" in msg
        assert "'coxph'" in msg
        assert "best" in msg


def _make_modelresult(name):
    """Construct a minimal real ModelResult for isinstance/message tests."""
    return ModelResult(
        name=name,
        model=None,
        cv_metric_scores=np.array([0.5]),
        cv_metric_mean=0.5,
        cv_metric_std=0.0,
    )


# ── End-to-end on a trained learner (integration) ──────────────────────────


def _make_data():
    rng = np.random.RandomState(0)
    n = 200
    X = pd.DataFrame(
        rng.randn(n, 6), columns=[f"f{i}" for i in range(6)]
    )
    time = rng.exponential(2, n)
    event = rng.binomial(1, 0.7, n).astype(bool)
    y = np.array(list(zip(event, time)),
                 dtype=[("event", bool), ("time", float)])
    return X, y


@pytest.fixture(scope="module")
def _trained():
    X, y = _make_data()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)
    learner.run(models=["coxph", "coxnet"], tune=False)
    learner.evaluate(n_importance_repeats=0)
    return learner


def _model_col(df):
    return "model" if "model" in df.columns else "model_type"


@pytest.mark.integration
class TestBestSentinelEndToEnd:
    @pytest.mark.parametrize("method", ["plot_calibration", "plot_metric"])
    def test_best_selects_top_model(self, _trained, method):
        best = _trained.best_model.name
        df = getattr(_trained, method)("best", return_data=True)
        assert set(df[_model_col(df)].unique()) == {best}

    def test_best_matches_explicit_name(self, _trained):
        best = _trained.best_model.name
        by_best = _trained.plot_metric("best", return_data=True)
        by_name = _trained.plot_metric(best, return_data=True)
        col = _model_col(by_best)
        assert set(by_best[col].unique()) == set(by_name[col].unique())

    def test_best_in_list_keeps_both_distinct_models(self, _trained):
        best = _trained.best_model.name
        other = next(n for n in _trained._state.models if n != best)
        df = _trained.plot_metric(["best", other], return_data=True)
        assert set(df[_model_col(df)].unique()) == {best, other}

    @pytest.mark.parametrize(
        "method", ["plot_calibration", "plot_metric", "plot_roc",
                   "plot_survival", "plot_importance"],
    )
    def test_bare_modelresult_raises_guiding_typeerror(self, _trained, method):
        mr = _trained.best_model
        with pytest.raises(TypeError) as exc:
            getattr(_trained, method)(mr)
        msg = str(exc.value)
        assert "ModelResult" in msg
        assert mr.name in msg
        assert "best" in msg

    def test_bare_evaluationresult_raises_typeerror(self, _trained):
        er = _trained.evaluation_result
        assert er is not None
        with pytest.raises(TypeError, match="models must be None, str, or list"):
            _trained.plot_metric(er)


# ── Singular model= methods: _resolve_best helper ──────────────────


@pytest.mark.integration
class TestResolveBestSingular:
    def test_best_maps_to_best_model_name(self, _trained):
        assert _trained._resolve_best("best") == _trained.best_model.name

    def test_plain_name_passes_through(self, _trained):
        assert _trained._resolve_best("coxph") == "coxph"

    def test_modelresult_passes_through_unchanged(self, _trained):
        mr = _trained.best_model
        assert _trained._resolve_best(mr) is mr

    def test_none_passes_through(self, _trained):
        assert _trained._resolve_best(None) is None


def test_resolve_best_without_models_raises():
    # No setup/train: best_model is None, so 'best' cannot resolve.
    X, y = _make_data()
    learner = SurvivalLearner(X, y)
    with pytest.raises(ValueError, match="No trained models to resolve"):
        learner._resolve_best("best")


# ── viz.export.figures_to_dict passes names, not objects (regression) ─


@pytest.mark.integration
def test_figures_to_dict_real_learner_uses_names(_trained):
    """Regression: export must pass model names to plot_*, not ModelResult.

    These calls previously passed a ``ModelResult``/``Pipeline`` object,
    which now raises ``TypeError`` — a real-learner test (the prior tests
    were all mock-based) guards the fix end-to-end.
    """
    from mlsurv.viz.export import figures_to_dict

    figures = figures_to_dict(_trained, model_result=_trained.best_model)
    names = [name for name, _ in figures]
    # The survival + importance figures depend on model_result and must
    # render (not crash to None) now that names are passed.
    assert "Survival Curves by Risk Score" in names
    assert len(figures) >= 2


# ── Singular model= methods end-to-end with 'best' (interactions) ──────────


@pytest.fixture(scope="module")
def _trained_interactions():
    X, y = _make_data()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)
    learner.run(models=["rsf"], tune=False)
    learner.evaluate(n_importance_repeats=0)
    # Populate H-statistic interaction data for the (tree-based) model.
    learner.interactions(method="hstat", top_k=3, max_samples=20)
    return learner


@pytest.mark.integration
class TestSingularBestEndToEnd:
    def test_plot_interactions_best(self, _trained_interactions):
        import plotly.graph_objects as go

        best = _trained_interactions.best_model.name
        fig = _trained_interactions.plot_interactions(model="best")
        assert isinstance(fig, go.Figure)
        # 'best' resolves to the only trained model.
        assert best == "rsf"

    def test_plot_interaction_pdp_best(self, _trained_interactions):
        import plotly.graph_objects as go

        fig = _trained_interactions.plot_interaction_pdp(
            "f0", "f1", model="best", max_samples=20, grid_resolution=5,
        )
        assert isinstance(fig, go.Figure)
