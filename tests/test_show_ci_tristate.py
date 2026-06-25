"""Tri-state ``show_ci`` contract for plot_roc and plot_metric_time.

``show_ci`` accepts ``"auto"`` (default), ``True``, and ``False``:

- ``"auto"`` — draw the CI ribbon when bootstrap data is present, otherwise
  the point curve; silent either way.
- ``True`` — explicitly request the ribbon; when the bootstrap data is
  absent, draw the point curve and emit one ``SuggestionWarning`` naming the
  ``bootstrap(...)`` flag to re-run.
- ``False`` — never draw a ribbon, no warning.

Covers the main, validation, and subpopulation views.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import SuggestionWarning

ROC_FLAG = "include_predictions=True"
MT_FLAG = "include_time_resolved=True"


def _make_data(n=200, p=6, seed=7):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    grp = np.where(X["f0"] > 0, "high", "low")
    lin = X["f0"] * 0.8 - X["f1"] * 0.5
    t = rng.exponential(np.exp(-lin) * 12 + 1.0)
    c = rng.exponential(20, size=n)
    y = np.array(list(zip(t <= c, np.minimum(t, c))),
                 dtype=[("event", bool), ("time", float)])
    return X, y, grp


@pytest.fixture(scope="module")
def learner_full():
    """Bootstrap with predictions + time-resolved AUC: CI data present."""
    X, y, grp = _make_data()
    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(test_fraction=0.3, random_state=42,
                      subpopulations={"grp": grp})
        learner.run(["coxph"], tune=False, n_importance_repeats=0,
                    verbose=False)
        learner.evaluate()
        learner.bootstrap(n_bootstrap=4, include_time_resolved=True,
                          include_predictions=True, verbose=False,
                          show_progress=False)
    return learner


@pytest.fixture(scope="module")
def learner_bare():
    """Bootstrap without predictions / time-resolved AUC: CI data absent."""
    X, y, _ = _make_data(seed=8)
    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(test_fraction=0.3, random_state=42)
        learner.run(["coxph"], tune=False, n_importance_repeats=0,
                    verbose=False)
        learner.evaluate()
        learner.bootstrap(n_bootstrap=4, verbose=False, show_progress=False)
    return learner


def _suggestions(fn, needle):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = fn()
    matched = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning) and needle in str(w.message)
    ]
    return fig, matched


def _has_ribbon(fig):
    """A CI ribbon sets ``fill`` on its trace (e.g. ``fill='tonexty'`` for
    line ribbons, ``fill='tozeroy'`` elsewhere); plain curves leave it unset."""
    return any(getattr(t, "fill", None) for t in fig.data)


# --------------------------------------------------------------------------
# Main ROC
# --------------------------------------------------------------------------
@pytest.mark.integration
class TestMainRoc:
    def test_auto_with_data_draws_ribbon_silent(self, learner_full):
        fig, warns = _suggestions(lambda: learner_full.plot_roc(), ROC_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert _has_ribbon(fig)  # auto + data → ribbon actually drawn

    def test_auto_without_data_silent(self, learner_bare):
        fig, warns = _suggestions(lambda: learner_bare.plot_roc(), ROC_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns

    def test_true_with_data_draws_ribbon_silent(self, learner_full):
        fig, warns = _suggestions(
            lambda: learner_full.plot_roc(show_ci=True), ROC_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert _has_ribbon(fig)  # True honors present data

    def test_true_without_data_warns(self, learner_bare):
        with pytest.warns(SuggestionWarning, match=ROC_FLAG):
            fig = learner_bare.plot_roc(show_ci=True)
        assert isinstance(fig, go.Figure)

    def test_false_no_ribbon_no_warning(self, learner_full):
        fig, warns = _suggestions(
            lambda: learner_full.plot_roc(t=2.0, show_ci=False), ROC_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert not _has_ribbon(fig)


# --------------------------------------------------------------------------
# Main plot_metric_time
# --------------------------------------------------------------------------
@pytest.mark.integration
class TestMainMetricTime:
    def test_auto_with_data_silent_draws_ribbon(self, learner_full):
        fig, warns = _suggestions(
            lambda: learner_full.plot_metric_time(), MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert _has_ribbon(fig)

    def test_auto_without_data_silent(self, learner_bare):
        fig, warns = _suggestions(
            lambda: learner_bare.plot_metric_time(), MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns

    def test_true_with_data_draws_ribbon_silent(self, learner_full):
        fig, warns = _suggestions(
            lambda: learner_full.plot_metric_time(show_ci=True), MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert _has_ribbon(fig)  # True honors present data

    def test_true_without_data_warns_once(self, learner_bare):
        # AUC(t) is collected on every bootstrap (cheap), so the missing-data
        # subject is Brier(t), which is opt-in via include_time_resolved.
        fig, warns = _suggestions(
            lambda: learner_bare.plot_metric_time(metric="brier", show_ci=True),
            MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert len(warns) == 1  # exactly one, not per-panel duplicates

    def test_auc_band_free_without_flag(self, learner_bare):
        # Cheap-auto: a plain bootstrap() (no include_time_resolved) still
        # collects AUC(t), so the AUC ribbon draws and show_ci=True is silent.
        fig, warns = _suggestions(
            lambda: learner_bare.plot_metric_time(show_ci=True), MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert _has_ribbon(fig)

    def test_true_without_data_warns_multipanel_once(self, learner_bare):
        # AUC band is present (always collected); only the Brier band is
        # missing without include_time_resolved, so exactly one warning fires.
        fig, warns = _suggestions(
            lambda: learner_bare.plot_metric_time(
                metric=["auc", "brier"], show_ci=True),
            MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert len(warns) == 1  # one Brier-ribbon warning, not per-panel

    def test_brier_band_present_with_time_resolved(self, learner_full):
        # include_time_resolved=True collects Brier(t); the multi-panel plot
        # draws a ribbon in BOTH the AUC and Brier panels (symmetric).
        fig = learner_full.plot_metric_time(metric=["auc", "brier"])
        fill_axes = {getattr(t, "yaxis", "y") or "y"
                     for t in fig.data if getattr(t, "fill", None)}
        assert {"y", "y2"} <= fill_axes  # ribbon in each metric panel

    def test_false_no_ribbon(self, learner_full):
        fig, warns = _suggestions(
            lambda: learner_full.plot_metric_time(show_ci=False), MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert not _has_ribbon(fig)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@pytest.mark.integration
class TestValidation:
    def test_roc_true_without_data_warns(self, learner_bare):
        X = learner_bare._state.X_train
        y = learner_bare._state.y_train
        learner_bare.validate(X.iloc[:40], y[:40], name="V1",
                              bootstrap=False, verbose=False)
        view = learner_bare.validation("V1")
        with pytest.warns(SuggestionWarning, match=ROC_FLAG):
            assert isinstance(view.plot_roc(show_ci=True), go.Figure)

    def test_roc_auto_without_data_silent(self, learner_bare):
        view = learner_bare.validation("V1")
        fig, warns = _suggestions(lambda: view.plot_roc(), ROC_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns

    def test_metric_time_true_without_data_warns(self, learner_bare):
        view = learner_bare.validation("V1")
        with pytest.warns(SuggestionWarning, match=MT_FLAG):
            view.plot_metric_time(show_ci=True)

    def test_metric_time_auto_without_data_silent(self, learner_bare):
        view = learner_bare.validation("V1")
        fig, warns = _suggestions(lambda: view.plot_metric_time(), MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns

    def test_metric_time_false_no_ribbon(self, learner_bare):
        view = learner_bare.validation("V1")
        fig, warns = _suggestions(
            lambda: view.plot_metric_time(show_ci=False), MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert not _has_ribbon(fig)


# --------------------------------------------------------------------------
# Subpopulation
# --------------------------------------------------------------------------
@pytest.mark.integration
class TestSubpopulation:
    def test_roc_auto_does_not_raise(self, learner_full):
        view = learner_full.subpopulation("grp")
        # New "auto" default must NOT trip the v1 NotImplementedError guard.
        assert view.plot_roc() is not None

    def test_roc_true_warns_not_raises(self, learner_full):
        """Explicit True is accepted (no raise) but warns: the subpop ROC
        surface cannot draw bootstrap CI bands."""
        view = learner_full.subpopulation("grp")
        with pytest.warns(SuggestionWarning, match="subpopulation surface"):
            assert view.plot_roc(show_ci=True) is not None

    def test_roc_false_raises(self, learner_full):
        view = learner_full.subpopulation("grp")
        with pytest.raises(NotImplementedError, match="show_ci="):
            view.plot_roc(show_ci=False)

    def test_metric_time_auto_with_data_silent_draws_ribbon(self, learner_full):
        view = learner_full.subpopulation("grp")
        fig, warns = _suggestions(lambda: view.plot_metric_time(), MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert _has_ribbon(fig)

    def test_metric_time_false_no_ribbon(self, learner_full):
        view = learner_full.subpopulation("grp")
        fig, warns = _suggestions(
            lambda: view.plot_metric_time(show_ci=False), MT_FLAG)
        assert isinstance(fig, go.Figure)
        assert not warns
        assert not _has_ribbon(fig)

    def test_metric_time_true_without_data_warns(self, learner_full):
        """Null the stored subpop AUC bootstrap so explicit True must warn.

        Works on a deepcopy so the module-scoped fixture stays intact.
        """
        import copy
        import dataclasses
        learner = copy.deepcopy(learner_full)
        spr = learner._state.subpopulation_results["grp"]
        try:
            learner._state.subpopulation_results["grp"] = (
                dataclasses.replace(spr, auc_t_bootstrap=None))
        except (TypeError, ValueError):
            object.__setattr__(spr, "auc_t_bootstrap", None)
        view = learner.subpopulation("grp")
        with pytest.warns(SuggestionWarning, match=MT_FLAG):
            view.plot_metric_time(show_ci=True)


# --------------------------------------------------------------------------
# compute_brier_std_df — fast unit test (no training)
# --------------------------------------------------------------------------
class TestComputeBrierStdDf:
    def test_std_per_model_and_time(self):
        from types import SimpleNamespace
        from mlsurv.viz._data import compute_brier_std_df

        brier_t = pd.DataFrame({
            "model_type": ["coxph"] * 4 + ["rsf"] * 2,
            "eval_time": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
            "brier": [0.10, 0.30, 0.20, 0.20, 0.05, 0.15],
        })
        eval_result = SimpleNamespace(bootstrap=SimpleNamespace(brier_t=brier_t))
        out = compute_brier_std_df(eval_result, model_names=None)

        assert list(out.columns) == ["model_type", "eval_time", "brier_std"]
        cox_t1 = out[(out.model_type == "coxph") & (out.eval_time == 1.0)]
        assert cox_t1["brier_std"].iloc[0] == pytest.approx(np.std([0.1, 0.3], ddof=1))
        cox_t2 = out[(out.model_type == "coxph") & (out.eval_time == 2.0)]
        assert cox_t2["brier_std"].iloc[0] == pytest.approx(0.0)

    def test_model_filter(self):
        from types import SimpleNamespace
        from mlsurv.viz._data import compute_brier_std_df

        brier_t = pd.DataFrame({
            "model_type": ["coxph", "coxph", "rsf", "rsf"],
            "eval_time": [1.0, 1.0, 1.0, 1.0],
            "brier": [0.1, 0.3, 0.2, 0.4],
        })
        eval_result = SimpleNamespace(bootstrap=SimpleNamespace(brier_t=brier_t))
        out = compute_brier_std_df(eval_result, model_names=["coxph"])
        assert set(out["model_type"]) == {"coxph"}

    def test_none_when_absent(self):
        from types import SimpleNamespace
        from mlsurv.viz._data import compute_brier_std_df

        assert compute_brier_std_df(
            SimpleNamespace(bootstrap=SimpleNamespace(brier_t=None)), None) is None
        assert compute_brier_std_df(SimpleNamespace(bootstrap=None), None) is None


class TestFilterEvalResultBrierT:
    """filter_eval_result (benchmark/references path) must preserve brier_t,
    not silently drop it the way auc_t survives — same omission class as the
    _aggregate_bootstrap_results bug this change fixes."""

    def test_filter_preserves_and_subsets_brier_t(self):
        from mlsurv.results.containers import EvaluationResult, BootstrapResult
        from mlsurv.viz._resolve import filter_eval_result

        def mt_df(models):
            return pd.DataFrame({"model_type": models, "x": range(len(models))})

        brier_t = pd.DataFrame({
            "model_type": ["coxph", "coxph", "rsf"],
            "eval_time": [1.0, 2.0, 1.0],
            "brier": [0.1, 0.2, 0.3],
        })
        auc_t = pd.DataFrame({
            "model_type": ["coxph", "rsf"],
            "eval_time": [1.0, 1.0],
            "auc": [0.7, 0.6],
        })
        boot = BootstrapResult(
            summary=mt_df(["coxph", "rsf"]),
            iterations=mt_df(["coxph", "rsf"]),
            auc_t=auc_t, brier_t=brier_t,
        )
        ev = EvaluationResult(
            concordance=mt_df(["coxph", "rsf"]),
            test_set=mt_df(["coxph", "rsf"]),
            auc_t=auc_t,
            importance=mt_df(["coxph", "rsf"]),
            importance_iterations=mt_df(["coxph", "rsf"]),
            bootstrap=boot,
        )
        out = filter_eval_result(ev, ["coxph"])
        assert out.bootstrap.brier_t is not None
        assert set(out.bootstrap.brier_t["model_type"]) == {"coxph"}
        assert set(out.bootstrap.auc_t["model_type"]) == {"coxph"}


# --------------------------------------------------------------------------
# Regression: evaluate(bootstrap=True) materializes brier_t
# --------------------------------------------------------------------------
@pytest.mark.integration
def test_evaluate_bootstrap_populates_brier_t():
    """evaluate(bootstrap=True) hardcodes include_brier_t=True; the combined
    multi-model BootstrapResult must carry brier_t, not silently drop it."""
    X, y, _ = _make_data(seed=11)
    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(test_fraction=0.3, random_state=42)
        learner.run(["coxph", "coxnet"], tune=False, n_importance_repeats=0,
                    verbose=False)
        learner.evaluate(bootstrap=True, n_bootstrap=3,
                         eval_times=[2.0, 5.0, 8.0],
                         show_progress=False, verbose=False)
    for name in ("coxph", "coxnet"):
        boot = learner.models[name].bootstrap
        assert boot.auc_t is not None and not boot.auc_t.empty
        assert boot.brier_t is not None and not boot.brier_t.empty
        assert {"model_type", "eval_time", "brier"} <= set(boot.brier_t.columns)
