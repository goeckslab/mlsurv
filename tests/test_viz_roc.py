"""Tests for time-dependent ROC curve (IPCW-weighted)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import MethodologicalWarning

go = pytest.importorskip("plotly.graph_objects")

from sksurv.metrics import cumulative_dynamic_auc  # noqa: E402
from sksurv.util import check_y_survival  # noqa: E402

from mlsurv.utils import truncate_y_to_train_range  # noqa: E402
from mlsurv.viz.metrics import (  # noqa: E402
    bootstrap_roc_curves,
    compute_roc_curves,
    plot_roc,
    plot_roc_interactive,
)

from tests._helpers import make_survival_array  # noqa: E402


def _normalize_y(y):
    """Convert any sksurv structured array to (event, time) field names."""
    event, time = check_y_survival(y)
    return make_survival_array(event, time)


def _assert_auc_matches(y_train, y_test, risk_scores, t, atol=1e-8):
    """Assert our AUC matches sksurv at time t."""
    y_test_trunc = truncate_y_to_train_range(y_test, y_train)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`trapz` is deprecated")
        sksurv_aucs, _ = cumulative_dynamic_auc(
            y_train, y_test_trunc, risk_scores, np.array([t])
        )
    our_df = compute_roc_curves(y_train, y_test, {"model": risk_scores}, t)
    our_auc = our_df["auc"].iloc[0]
    np.testing.assert_allclose(
        our_auc, sksurv_aucs[0], atol=atol,
        err_msg=f"AUC mismatch at t={t}: ours={our_auc}, sksurv={sksurv_aucs[0]}",
    )


def _make_synth_data(n=200, censoring_rate=0.4, seed=42):
    """Generate synthetic survival data with controllable censoring."""
    rng = np.random.RandomState(seed)
    event_times = rng.exponential(10, n)
    censor_times = rng.exponential(10 / censoring_rate * (1 - censoring_rate), n)
    time = np.minimum(event_times, censor_times)
    event = event_times <= censor_times
    return make_survival_array(event, time)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_data():
    """Synthetic survival data for unit tests."""
    rng = np.random.RandomState(42)
    n = 200
    y_train = _make_synth_data(n, seed=42)
    y_test = _make_synth_data(n, seed=99)
    scores_a = rng.randn(n)
    scores_b = rng.randn(n)
    return y_train, y_test, scores_a, scores_b


@pytest.fixture(scope="module")
def roc_df(synth_data):
    """Pre-computed ROC DataFrame for plot tests."""
    y_train, y_test, scores_a, scores_b = synth_data
    t = float(np.median(y_test["time"][y_test["event"]]))
    return compute_roc_curves(
        y_train, y_test,
        {"coxph": scores_a, "rsf": scores_b},
        t,
    )


# ===================================================================
# TestComputeRocCurves — unit tests
# ===================================================================

class TestComputeRocCurves:
    """Unit tests for compute_roc_curves."""

    def test_returns_correct_columns(self, roc_df):
        expected = {"model", "fpr", "tpr", "threshold", "auc", "eval_time"}
        assert set(roc_df.columns) == expected

    def test_fpr_tpr_range(self, roc_df):
        assert (roc_df["fpr"] >= 0).all() and (roc_df["fpr"] <= 1).all()
        assert (roc_df["tpr"] >= 0).all() and (roc_df["tpr"] <= 1).all()

    def test_monotonic(self, roc_df):
        for _, grp in roc_df.groupby("model"):
            fpr = grp["fpr"].values
            tpr = grp["tpr"].values
            assert np.all(np.diff(fpr) >= -1e-15), "FPR not non-decreasing"
            assert np.all(np.diff(tpr) >= -1e-15), "TPR not non-decreasing"

    def test_starts_at_origin(self, roc_df):
        for _, grp in roc_df.groupby("model"):
            assert grp["fpr"].iloc[0] == 0.0
            assert grp["tpr"].iloc[0] == 0.0

    def test_ends_at_one_one(self, roc_df):
        for _, grp in roc_df.groupby("model"):
            np.testing.assert_allclose(grp["fpr"].iloc[-1], 1.0, atol=1e-10)
            np.testing.assert_allclose(grp["tpr"].iloc[-1], 1.0, atol=1e-10)

    def test_multi_model(self, roc_df):
        models = roc_df["model"].unique()
        assert len(models) == 2
        assert set(models) == {"coxph", "rsf"}

    def test_no_events_before_t_warns(self, synth_data):
        y_train, y_test, scores_a, _ = synth_data
        t_very_small = 0.0001
        with pytest.warns(MethodologicalWarning, match="No cases"):
            with pytest.raises(ValueError, match="Could not compute ROC"):
                compute_roc_curves(
                    y_train, y_test,
                    {"model": scores_a, "backup": scores_a * 2},
                    t_very_small,
                )

    def test_no_controls_warns(self, synth_data):
        y_train, y_test, scores_a, _ = synth_data
        t_very_large = 1e6  # beyond all observations
        with pytest.warns(MethodologicalWarning, match="No controls"):
            with pytest.raises(ValueError, match="Could not compute ROC"):
                compute_roc_curves(
                    y_train, y_test, {"model": scores_a}, t_very_large,
                )

    def test_partial_model_failure(self):
        """One model skipped (no cases at chosen t), the other succeeds."""
        rng = np.random.RandomState(200)
        n = 100
        # Test set: mix of early and late events so t=5 has both cases & controls
        times_test = np.concatenate([rng.uniform(1, 4, n // 2),
                                      rng.uniform(6, 20, n // 2)])
        events_test = np.ones(n, dtype=bool)
        y_test = make_survival_array(events_test, times_test)
        y_train = _make_synth_data(n, seed=201)
        t = 5.0
        scores_valid = rng.randn(n)
        scores_constant = np.zeros(n)  # all tied → still works
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = compute_roc_curves(
                y_train, y_test,
                {"valid": scores_valid, "constant": scores_constant},
                t,
            )
        # Both should succeed (constant scores still produce a valid curve)
        assert len(df["model"].unique()) == 2

    def test_eval_time_column(self, synth_data):
        y_train, y_test, scores_a, _ = synth_data
        t = 5.0
        df = compute_roc_curves(y_train, y_test, {"m": scores_a}, t)
        assert (df["eval_time"] == t).all()


# ===================================================================
# TestAucMatchesSksurv — critical AUC verification
# ===================================================================

class TestAucMatchesSksurv:
    """Verify that compute_roc_curves AUC matches sksurv exactly."""

    def test_synth_random_scores(self):
        rng = np.random.RandomState(0)
        y_train = _make_synth_data(200, seed=0)
        y_test = _make_synth_data(200, seed=1)
        scores = rng.randn(200)
        t = float(np.median(y_test["time"][y_test["event"]]))
        _assert_auc_matches(y_train, y_test, scores, t)

    def test_synth_early_time(self):
        rng = np.random.RandomState(10)
        y_train = _make_synth_data(200, seed=10)
        y_test = _make_synth_data(200, seed=11)
        scores = rng.randn(200)
        event_times = y_test["time"][y_test["event"]]
        t = float(np.percentile(event_times, 25))
        _assert_auc_matches(y_train, y_test, scores, t)

    def test_synth_late_time(self):
        rng = np.random.RandomState(20)
        y_train = _make_synth_data(200, seed=20)
        y_test = _make_synth_data(200, seed=21)
        scores = rng.randn(200)
        event_times = y_test["time"][y_test["event"]]
        t = float(np.percentile(event_times, 75))
        _assert_auc_matches(y_train, y_test, scores, t)

    def test_synth_high_censoring(self):
        rng = np.random.RandomState(30)
        y_train = _make_synth_data(200, censoring_rate=0.7, seed=30)
        y_test = _make_synth_data(200, censoring_rate=0.7, seed=31)
        scores = rng.randn(200)
        t = float(np.median(y_test["time"][y_test["event"]]))
        _assert_auc_matches(y_train, y_test, scores, t)

    def test_synth_low_censoring(self):
        rng = np.random.RandomState(40)
        y_train = _make_synth_data(200, censoring_rate=0.1, seed=40)
        y_test = _make_synth_data(200, censoring_rate=0.1, seed=41)
        scores = rng.randn(200)
        t = float(np.median(y_test["time"][y_test["event"]]))
        _assert_auc_matches(y_train, y_test, scores, t)

    def test_synth_tied_scores(self):
        rng = np.random.RandomState(50)
        y_train = _make_synth_data(200, seed=50)
        y_test = _make_synth_data(200, seed=51)
        scores = np.round(rng.randn(200), 1)  # many ties
        t = float(np.median(y_test["time"][y_test["event"]]))
        _assert_auc_matches(y_train, y_test, scores, t)

    def test_synth_perfect_discrimination(self):
        y_train = _make_synth_data(200, seed=60)
        y_test = _make_synth_data(200, seed=61)
        # Higher risk = shorter time → perfect ordering
        scores = -y_test["time"]
        t = float(np.median(y_test["time"][y_test["event"]]))
        _assert_auc_matches(y_train, y_test, scores, t)

    def test_synth_no_discrimination(self):
        rng = np.random.RandomState(70)
        y_train = _make_synth_data(200, seed=70)
        y_test = _make_synth_data(200, seed=71)
        scores = rng.permutation(y_test["time"])  # shuffled → no signal
        t = float(np.median(y_test["time"][y_test["event"]]))
        _assert_auc_matches(y_train, y_test, scores, t, atol=1e-8)

    def test_real_gbsg2(self):
        from sksurv.datasets import load_gbsg2
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.preprocessing import OneHotEncoder

        X, y = load_gbsg2()
        X = OneHotEncoder().fit_transform(X)
        n_train = int(len(X) * 0.7)
        rng = np.random.RandomState(80)
        idx = rng.permutation(len(X))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        model = CoxPHSurvivalAnalysis(alpha=0.1)
        model.fit(X.iloc[train_idx], y[train_idx])
        scores = model.predict(X.iloc[test_idx])

        # Normalize to (event, time) field names for compute_roc_curves
        y_train_n = _normalize_y(y[train_idx])
        y_test_n = _normalize_y(y[test_idx])
        event_mask = y_test_n["event"]
        t = float(np.median(y_test_n["time"][event_mask]))
        _assert_auc_matches(y_train_n, y_test_n, scores, t)

    def test_real_veterans(self):
        from sksurv.datasets import load_veterans_lung_cancer
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X, y = load_veterans_lung_cancer()
        X = pd.get_dummies(X, drop_first=True).astype(float)
        n_train = int(len(X) * 0.7)
        rng = np.random.RandomState(90)
        idx = rng.permutation(len(X))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        model = CoxPHSurvivalAnalysis(alpha=0.1)
        model.fit(X.iloc[train_idx], y[train_idx])
        scores = model.predict(X.iloc[test_idx])

        y_train_n = _normalize_y(y[train_idx])
        y_test_n = _normalize_y(y[test_idx])
        event_mask = y_test_n["event"]
        t = float(np.percentile(y_test_n["time"][event_mask], 40))
        _assert_auc_matches(y_train_n, y_test_n, scores, t)

    def test_multiple_time_points_loop(self):
        rng = np.random.RandomState(100)
        y_train = _make_synth_data(300, seed=100)
        y_test = _make_synth_data(300, seed=101)
        scores = rng.randn(300)
        event_times = y_test["time"][y_test["event"]]
        for q in [10, 25, 50, 75, 90]:
            t = float(np.percentile(event_times, q))
            _assert_auc_matches(y_train, y_test, scores, t)


# ===================================================================
# TestPlotRoc — viz tests
# ===================================================================

class TestPlotRoc:
    """Test plot_roc visualization function."""

    def test_returns_figure(self, roc_df):
        fig = plot_roc(roc_df)
        assert isinstance(fig, go.Figure)

    def test_diagonal_line(self, roc_df):
        fig = plot_roc(roc_df)
        shapes = fig.layout.shapes
        assert len(shapes) >= 1
        diag = shapes[0]
        assert diag.x0 == 0 and diag.y0 == 0
        assert diag.x1 == 1 and diag.y1 == 1

    def test_auc_in_legend(self, roc_df):
        fig = plot_roc(roc_df, show_auc=True)
        names = [trace.name for trace in fig.data]
        assert any("AUC" in n for n in names)

    def test_no_auc_in_legend(self, roc_df):
        fig = plot_roc(roc_df, show_auc=False)
        names = [trace.name for trace in fig.data]
        assert not any("AUC" in n for n in names)

    def test_multi_model_traces(self, roc_df):
        fig = plot_roc(roc_df)
        assert len(fig.data) == 2  # two models


# ===================================================================
# TestLearnerPlotRoc — integration tests
# ===================================================================

@pytest.fixture(scope="module")
def learner_trained():
    """Trained learner with CoxPH + RSF for integration tests."""
    from mlsurv.learner import SurvivalLearner
    from mlsurv.utils import SurvivalArray

    rng = np.random.RandomState(42)
    n, p = 300, 8
    X = pd.DataFrame(
        rng.randn(n, p),
        columns=[f"feat_{i}" for i in range(p)],
    )
    coef = np.zeros(p)
    coef[:3] = [0.5, -0.3, 0.4]
    lp = X.values @ coef
    times = np.exp(-lp + rng.randn(n) * 0.5)
    times = np.clip(times, 0.1, 10)
    events = rng.binomial(1, 0.7, n).astype(bool)
    y = SurvivalArray(
        np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        )
    )
    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(test_fraction=0.3, random_state=42)
        learner.run(
            models=["coxph", "rsf"],
            tune=False,
            n_importance_repeats=0,
            bootstrap=False,
            verbose=False,
        )
    return learner


@pytest.mark.integration
class TestLearnerPlotRoc:
    """Integration tests for learner.plot_roc()."""

    def test_returns_figure(self, learner_trained):
        fig = learner_trained.plot_roc()
        assert isinstance(fig, go.Figure)

    def test_return_data(self, learner_trained):
        df = learner_trained.plot_roc(return_data=True)
        assert isinstance(df, pd.DataFrame)
        expected = {"model", "fpr", "tpr", "threshold", "auc", "eval_time"}
        assert set(df.columns) == expected

    def test_single_model(self, learner_trained):
        df = learner_trained.plot_roc(models="coxph", return_data=True)
        assert set(df["model"].unique()) == {"coxph"}

    def test_default_t(self, learner_trained):
        # Should not raise when t is not specified
        fig = learner_trained.plot_roc()
        assert fig is not None

    def test_explicit_t(self, learner_trained):
        df = learner_trained.plot_roc(t=5.0, return_data=True)
        assert (df["eval_time"] == 5.0).all()

    def test_round_trip(self, learner_trained):
        df = learner_trained.plot_roc(return_data=True)
        # Both learner and low-level now use 'model' column
        fig = plot_roc(df)
        assert isinstance(fig, go.Figure)


# ===================================================================
# TestLearnerPlotRocCI — CI ribbon integration tests
# ===================================================================

@pytest.fixture(scope="module")
def learner_with_bootstrap():
    """Trained learner with bootstrap predictions for CI tests."""
    from mlsurv.learner import SurvivalLearner
    from mlsurv.utils import SurvivalArray

    rng = np.random.RandomState(42)
    n, p = 300, 8
    X = pd.DataFrame(
        rng.randn(n, p),
        columns=[f"feat_{i}" for i in range(p)],
    )
    coef = np.zeros(p)
    coef[:3] = [0.5, -0.3, 0.4]
    lp = X.values @ coef
    times = np.exp(-lp + rng.randn(n) * 0.5)
    times = np.clip(times, 0.1, 10)
    events = rng.binomial(1, 0.7, n).astype(bool)
    y = SurvivalArray(
        np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        )
    )
    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(test_fraction=0.3, random_state=42)
        learner.run(
            models=["coxph", "rsf"],
            tune=False,
            n_importance_repeats=0,
            n_bootstrap=5,
            verbose=False,
        )
    return learner


@pytest.mark.integration
class TestLearnerPlotRocCI:
    """Integration tests for learner.plot_roc(show_ci=...)."""

    def test_show_ci_default_returns_figure(self, learner_with_bootstrap):
        """Default show_ci="auto" with bootstrap data → ribbon drawn, silent."""
        import warnings as _w
        from mlsurv._warnings import SuggestionWarning
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            fig = learner_with_bootstrap.plot_roc()  # auto default
        assert isinstance(fig, go.Figure)
        # Interactive plot should have more traces than just 2 model lines
        # (dropdown diagonal + model lines + CI ribbon traces)
        assert len(fig.data) > 2
        # "auto" with data present must not warn about missing CI.
        roc_ci = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and "ROC CI bands" in str(w.message)
        ]
        assert not roc_ci

    def test_show_ci_false_returns_static(self, learner_with_bootstrap):
        """show_ci=False with explicit t → static figure (no CI ribbons)."""
        fig = learner_with_bootstrap.plot_roc(
            t=2.0, show_ci=False,
        )
        assert isinstance(fig, go.Figure)
        # Static plot: 1 trace per model (no diagonal, no ribbon)
        assert len(fig.data) == 2

    def test_no_bootstrap_auto_is_silent(self, learner_trained):
        """show_ci="auto" without bootstrap data → point curve, no warning."""
        import warnings as _w
        from mlsurv._warnings import SuggestionWarning
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            fig = learner_trained.plot_roc()  # auto default
        assert isinstance(fig, go.Figure)
        roc_ci = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and "ROC CI bands" in str(w.message)
        ]
        assert not roc_ci

    def test_no_bootstrap_explicit_true_warns(self, learner_trained):
        """show_ci=True without bootstrap data → point curve + SuggestionWarning."""
        from mlsurv._warnings import SuggestionWarning
        with pytest.warns(SuggestionWarning, match="include_predictions=True"):
            fig = learner_trained.plot_roc(show_ci=True)
        assert isinstance(fig, go.Figure)

    def test_explicit_t_with_ci(self, learner_with_bootstrap):
        """Single explicit t with CI → interactive figure."""
        fig = learner_with_bootstrap.plot_roc(t=2.0, show_ci=True)
        assert isinstance(fig, go.Figure)

    def test_return_data_with_ci(self, learner_with_bootstrap):
        """return_data=True returns DataFrame regardless of show_ci."""
        df = learner_with_bootstrap.plot_roc(return_data=True)
        assert isinstance(df, pd.DataFrame)
        expected = {"model", "fpr", "tpr", "threshold", "auc", "eval_time"}
        assert set(df.columns) == expected

    def test_single_model_with_ci(self, learner_with_bootstrap):
        """Single-model plot with CI."""
        fig = learner_with_bootstrap.plot_roc(
            models="coxph", show_ci=True,
        )
        assert isinstance(fig, go.Figure)


# ===================================================================
# TestPlotRocInteractive — dropdown viz tests
# ===================================================================

@pytest.fixture(scope="module")
def roc_data_multi(synth_data):
    """ROC DataFrames at multiple time points for interactive tests."""
    y_train, y_test, scores_a, scores_b = synth_data
    event_times = y_test["time"][y_test["event"]]
    eval_times = [
        float(np.percentile(event_times, q))
        for q in [25, 50, 75]
    ]
    risk_scores = {"coxph": scores_a, "rsf": scores_b}
    roc_data = {}
    for t in eval_times:
        roc_data[t] = compute_roc_curves(y_train, y_test, risk_scores, t)
    return roc_data


class TestPlotRocInteractive:
    """Tests for plot_roc_interactive (dropdown time selector)."""

    def test_returns_figure(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi)
        assert isinstance(fig, go.Figure)

    def test_dropdown_buttons(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi)
        menus = fig.layout.updatemenus
        assert len(menus) == 1
        buttons = menus[0].buttons
        assert len(buttons) == len(roc_data_multi)

    def test_button_labels(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi)
        labels = [b.label for b in fig.layout.updatemenus[0].buttons]
        for t in sorted(roc_data_multi.keys()):
            assert f"t = {t:.1f}" in labels

    def test_diagonal_reference_trace(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi)
        ref = fig.data[0]
        assert list(ref.x) == [0, 1]
        assert list(ref.y) == [0, 1]

    def test_visibility_default(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi)
        # Trace 0 (reference) always visible
        assert fig.data[0].visible is None or fig.data[0].visible is True
        # At least some model traces should be visible (default time)
        visible_count = sum(
            1 for t in fig.data[1:]
            if t.visible is None or t.visible is True
        )
        assert visible_count > 0
        # Some traces should be hidden (other time points)
        hidden_count = sum(1 for t in fig.data[1:] if t.visible is False)
        assert hidden_count > 0

    def test_trace_count(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi)
        n_times = len(roc_data_multi)
        n_models = 2  # coxph + rsf
        # 1 reference + n_models per time point
        assert len(fig.data) == 1 + n_times * n_models

    def test_auc_in_legend(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi, show_auc=True)
        names = [t.name for t in fig.data[1:]]
        assert all("AUC" in n for n in names)

    def test_no_auc_in_legend(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi, show_auc=False)
        names = [t.name for t in fig.data[1:]]
        assert not any("AUC" in n for n in names)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            plot_roc_interactive({})

    def test_single_time_point(self, synth_data):
        y_train, y_test, scores_a, _ = synth_data
        t = float(np.median(y_test["time"][y_test["event"]]))
        roc_data = {t: compute_roc_curves(y_train, y_test, {"coxph": scores_a}, t)}
        fig = plot_roc_interactive(roc_data)
        assert isinstance(fig, go.Figure)
        # Should still have a dropdown with 1 button
        assert len(fig.layout.updatemenus[0].buttons) == 1

    def test_default_time_index(self, roc_data_multi):
        # Explicitly set default to first time
        fig = plot_roc_interactive(roc_data_multi, default_time_index=0)
        assert fig.layout.updatemenus[0].active == 0


# ===================================================================
# Bootstrap ROC CI — helpers and fixtures
# ===================================================================

def _make_bootstrap_predictions(y_train, y_test, risk_scores_dict, n_bootstrap=10, seed=42):
    """Simulate bootstrap predictions DataFrame as produced by BootstrapResult."""
    rng = np.random.RandomState(seed)
    n = len(y_test)
    rows = []
    for model_name, scores in risk_scores_dict.items():
        for i in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            for pos in idx:
                rows.append({
                    "model_type": model_name,
                    "iteration": i,
                    "index": pos,  # using integer positions as index labels
                    "risk_score": scores[pos],
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def bootstrap_ci_fixture(synth_data):
    """Bootstrap predictions + eval times for CI tests."""
    y_train, y_test, scores_a, scores_b = synth_data
    risk_scores = {"coxph": scores_a, "rsf": scores_b}
    event_times = y_test["time"][y_test["event"]]
    eval_times = [float(np.percentile(event_times, q)) for q in [25, 75]]
    preds = _make_bootstrap_predictions(y_train, y_test, risk_scores, n_bootstrap=10)
    X_test_index = pd.RangeIndex(len(y_test))  # default integer index
    return y_train, y_test, preds, X_test_index, eval_times, risk_scores


# ===================================================================
# TestBootstrapRocCurves
# ===================================================================

class TestBootstrapRocCurves:
    """Tests for bootstrap_roc_curves()."""

    def test_happy_path_structure(self, bootstrap_ci_fixture):
        y_train, y_test, preds, idx, eval_times, _ = bootstrap_ci_fixture
        result = bootstrap_roc_curves(y_train, y_test, preds, idx, eval_times)
        assert isinstance(result, dict)
        for t in eval_times:
            assert t in result
            for model_name in result[t]:
                bands = result[t][model_name]
                assert set(bands.keys()) == {"fpr", "tpr_lower", "tpr_median", "tpr_upper"}

    def test_band_shapes(self, bootstrap_ci_fixture):
        y_train, y_test, preds, idx, eval_times, _ = bootstrap_ci_fixture
        n_grid = 50
        result = bootstrap_roc_curves(y_train, y_test, preds, idx, eval_times, n_grid=n_grid)
        for t in result:
            for bands in result[t].values():
                assert len(bands["fpr"]) == n_grid
                assert len(bands["tpr_lower"]) == n_grid
                assert len(bands["tpr_median"]) == n_grid
                assert len(bands["tpr_upper"]) == n_grid

    def test_band_ordering(self, bootstrap_ci_fixture):
        y_train, y_test, preds, idx, eval_times, _ = bootstrap_ci_fixture
        result = bootstrap_roc_curves(y_train, y_test, preds, idx, eval_times)
        for t in result:
            for bands in result[t].values():
                assert np.all(bands["tpr_lower"] <= bands["tpr_median"] + 1e-12)
                assert np.all(bands["tpr_median"] <= bands["tpr_upper"] + 1e-12)

    def test_fpr_grid_range(self, bootstrap_ci_fixture):
        y_train, y_test, preds, idx, eval_times, _ = bootstrap_ci_fixture
        result = bootstrap_roc_curves(y_train, y_test, preds, idx, eval_times)
        for t in result:
            for bands in result[t].values():
                np.testing.assert_allclose(bands["fpr"][0], 0.0)
                np.testing.assert_allclose(bands["fpr"][-1], 1.0)

    def test_single_iteration_returns_empty(self, synth_data):
        y_train, y_test, scores_a, _ = synth_data
        # Only 1 iteration — need ≥2 for bands
        preds = _make_bootstrap_predictions(y_train, y_test, {"coxph": scores_a}, n_bootstrap=1)
        event_times = y_test["time"][y_test["event"]]
        t = float(np.median(event_times))
        result = bootstrap_roc_curves(y_train, y_test, preds, pd.RangeIndex(len(y_test)), [t])
        # Either empty or no model entries
        assert not result or all(not v for v in result.values())

    @pytest.mark.parametrize("ci_val", [0, 1, -0.5, 1.5])
    def test_ci_out_of_range_raises(self, bootstrap_ci_fixture, ci_val):
        """ci outside (0, 1) should raise ValueError."""
        y_train, y_test, preds, idx, eval_times, _ = bootstrap_ci_fixture
        with pytest.raises(ValueError, match="ci must be between"):
            bootstrap_roc_curves(y_train, y_test, preds, idx, eval_times, ci=ci_val)

    @pytest.mark.parametrize("n_grid_val", [0, 1, -1])
    def test_n_grid_too_small_raises(self, bootstrap_ci_fixture, n_grid_val):
        """n_grid < 2 should raise ValueError."""
        y_train, y_test, preds, idx, eval_times, _ = bootstrap_ci_fixture
        with pytest.raises(ValueError, match="n_grid must be"):
            bootstrap_roc_curves(y_train, y_test, preds, idx, eval_times, n_grid=n_grid_val)

    def test_stale_bootstrap_labels_raise_value_error(self, bootstrap_ci_fixture):
        """Bootstrap predictions with index labels not in X_test.index must
        raise ``ValueError`` (not ``KeyError``).

        Regression guard: a stale ``BootstrapResult``
        (e.g., after re-``setup()``) used to trip ``pd.Series.loc`` with a
        bare ``KeyError``, which after the except-narrowing in
        ``_compute_roc_ci_data_main`` would have surfaced as an opaque
        traceback. The upstream validation here converts it to a
        descriptive ``ValueError`` that the fallback still catches.
        """
        y_train, y_test, preds, idx, eval_times, _ = bootstrap_ci_fixture
        stale_preds = preds.copy()
        # Inject a label that is NOT in the X_test index
        stale_label = int(idx.max()) + 999
        stale_preds.loc[stale_preds.index[0], "index"] = stale_label
        with pytest.raises(
            ValueError, match="not present in X_test.index",
        ):
            bootstrap_roc_curves(y_train, y_test, stale_preds, idx, eval_times)

    def test_nondefault_pandas_index(self, synth_data):
        y_train, y_test, scores_a, scores_b = synth_data
        # Pandas index starting at 500
        offset = 500
        custom_index = pd.RangeIndex(offset, offset + len(y_test))
        rng = np.random.RandomState(99)
        n = len(y_test)
        rows = []
        for i in range(5):
            idx = rng.randint(0, n, size=n)
            for pos in idx:
                rows.append({
                    "model_type": "coxph",
                    "iteration": i,
                    "index": offset + pos,  # pandas label, not position
                    "risk_score": scores_a[pos],
                })
        preds = pd.DataFrame(rows)
        event_times = y_test["time"][y_test["event"]]
        t = float(np.median(event_times))
        result = bootstrap_roc_curves(y_train, y_test, preds, custom_index, [t])
        assert result  # should succeed
        assert t in result
        assert "coxph" in result[t]


# ===================================================================
# TestPlotRocInteractiveCI
# ===================================================================

@pytest.fixture(scope="module")
def ci_data(roc_data_multi, bootstrap_ci_fixture):
    """Bootstrap CI bands keyed to the same times as roc_data_multi."""
    y_train, y_test, preds, idx, eval_times, _ = bootstrap_ci_fixture
    return bootstrap_roc_curves(
        y_train, y_test, preds, idx, list(roc_data_multi.keys()),
    )


class TestPlotRocInteractiveCI:
    """Tests for plot_roc_interactive() with bootstrap CI ribbons."""

    def test_ci_adds_ribbon_traces(self, roc_data_multi, ci_data):
        fig_no_ci = plot_roc_interactive(roc_data_multi)
        fig_ci = plot_roc_interactive(roc_data_multi, ci_data=ci_data)

        # CI version should have more traces (2 ribbon traces per model per time with CI)
        assert len(fig_ci.data) > len(fig_no_ci.data)

    def test_no_ci_unchanged(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi, ci_data=None)
        n_times = len(roc_data_multi)
        # 1 diagonal + (2 models x 3 times) = 7 traces
        assert len(fig.data) == 1 + 2 * n_times

    def test_dropdown_toggles_all_traces(self, roc_data_multi, ci_data):
        fig = plot_roc_interactive(roc_data_multi, ci_data=ci_data)

        # Each button's visibility array should show diagonal + all traces for that time
        for button in fig.layout.updatemenus[0].buttons:
            vis = button.args[0]["visible"]
            assert vis[0] is True  # diagonal always visible
            assert any(vis[1:])

    def test_no_ci_entries_without_ci(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi, ci_data=None)
        buttons = fig.layout.updatemenus[0].buttons
        # Without CI data, no "(no CI)" entries
        assert len(buttons) == len(roc_data_multi)
        assert all("no CI" not in b.label for b in buttons)

    def test_ci_one_entry_per_time(self, roc_data_multi, ci_data):
        """With CI bands, the dropdown has exactly one entry per eval time
        (no separate "(no CI)" variants)."""
        fig = plot_roc_interactive(roc_data_multi, ci_data=ci_data)

        buttons = fig.layout.updatemenus[0].buttons
        assert len(buttons) == len(roc_data_multi)
        assert all("no CI" not in b.label for b in buttons)

    def test_ci_traces_visible_in_each_entry(self, roc_data_multi, ci_data):
        """Each dropdown entry shows its CI ribbon traces — CI is always on
        when bands are available (the regression we guard against)."""
        fig = plot_roc_interactive(roc_data_multi, ci_data=ci_data)

        # CI ribbon traces are drawn with hoverinfo="skip" by add_ci_ribbon.
        ribbon_idx = {
            j for j, tr in enumerate(fig.data) if tr.hoverinfo == "skip"
        }
        assert ribbon_idx, "expected CI ribbon traces in the figure"

        buttons = fig.layout.updatemenus[0].buttons
        for button in buttons:
            vis = button.args[0]["visible"]
            visible_ribbons = {j for j in ribbon_idx if vis[j]}
            assert visible_ribbons, (
                f"entry {button.label!r} shows no CI ribbon traces"
            )

        # Each ribbon belongs to exactly one entry — guards against a
        # visibility-mask cross-leak (one time's button revealing another
        # time's ribbon) or a ribbon left visible in every entry.
        for j in ribbon_idx:
            n_visible = sum(b.args[0]["visible"][j] for b in buttons)
            assert n_visible == 1, (
                f"ribbon trace {j} visible in {n_visible} entries (expected 1)"
            )

    def test_empty_ci_data_no_toggle(self, roc_data_multi):
        fig = plot_roc_interactive(roc_data_multi, ci_data={})
        buttons = fig.layout.updatemenus[0].buttons
        assert len(buttons) == len(roc_data_multi)

    def test_single_menu_with_ci(self, roc_data_multi, ci_data):
        """Only one dropdown menu even with CI data (no separate toggle)."""
        fig = plot_roc_interactive(roc_data_multi, ci_data=ci_data)
        assert len(fig.layout.updatemenus) == 1

    def test_partial_ci_data(self, roc_data_multi, ci_data):
        first_t = sorted(ci_data.keys())[0]
        first_model = next(iter(ci_data[first_t]))
        partial = {first_t: {first_model: ci_data[first_t][first_model]}}

        fig = plot_roc_interactive(roc_data_multi, ci_data=partial)
        buttons = fig.layout.updatemenus[0].buttons
        # Still one entry per time; partial CI data must not crash.
        assert len(buttons) == len(roc_data_multi)
        assert all("no CI" not in b.label for b in buttons)

        # Partial CI is honored, not dropped or over-applied: the covered
        # time's button shows a ribbon; an uncovered time's button does not.
        ribbon_idx = {
            j for j, tr in enumerate(fig.data) if tr.hoverinfo == "skip"
        }
        assert ribbon_idx, "partial CI data should still draw a ribbon"
        covered_label = f"t = {first_t:.1f}"
        seen_covered = seen_uncovered = False
        for b in buttons:
            has_ribbon = any(b.args[0]["visible"][j] for j in ribbon_idx)
            if b.label == covered_label:
                assert has_ribbon, "covered time should show its ribbon"
                seen_covered = True
            else:
                assert not has_ribbon, (
                    f"uncovered time {b.label!r} should show no ribbon"
                )
                seen_uncovered = True
        assert seen_covered and seen_uncovered
