"""Tests for the dense, train-anchored IBS integration grid.

IBS is integrated over a dense grid of training event times within the
censoring-stable window ``[t_min, tau]``, decoupled from the coarse display grid
used for the AUC(t)/Brier(t) curves. These tests pin:

- grid construction (train-anchored, within the censoring-stable window, capped);
- that the pipeline still calls sksurv's trapezoid integrator over that grid;
- the headline acceptance criterion: IBS is stable when the display grid changes
  (while Brier(median) — a display-grid metric — still moves);
- bootstrap IBS matches the headline and predicts the dense matrix once;
- subpopulation IBS uses the same (train-anchored) dense grid.
"""

import warnings

import numpy as np
import pytest

from mlsurv.evaluation.metrics import (
    ibs_integration_times,
    compute_safe_ipcw_tau,
    _integrated_brier_from_probs,
    IBS_GRID_CAP,
)
from mlsurv.utils import truncate_y_to_train_range

from tests._helpers import make_y, make_random_survival_array


# ── ibs_integration_times unit tests ────────────────────────────────────


class TestIbsIntegrationTimes:
    """Grid construction: train-anchored, within window, capped."""

    def test_points_are_train_event_times_within_window(self):
        y_train = make_y(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [True, True, False, True, True, True, False, True, True, True],
        )
        y_test = make_y([0.5, 1.5, 3.5, 6.5, 9.5], [True, True, True, True, True])

        grid = ibs_integration_times(y_test, y_train)
        # Mirror the implementation: tau is computed on the train-truncated test
        # cohort, so compute it the same way here.
        y_test_trunc = truncate_y_to_train_range(y_test, y_train, warn=False)
        tau = compute_safe_ipcw_tau(y_train, y_test_trunc)

        train_event_times = set(
            np.unique(y_train["time"][y_train["event"]]).tolist()
        )
        # Every grid point is an actual training event time.
        assert all(g in train_event_times for g in grid)
        # Strictly inside the test cohort's window and within tau.
        assert grid.min() > y_test["time"].min()
        assert grid.max() < y_test["time"].max()
        assert grid.max() <= tau
        # Sorted, unique.
        assert np.all(np.diff(grid) > 0)

    def test_tau_strictly_binds_excludes_in_range_train_events(self):
        # Train events span 1..20. The test cohort's events cluster early so the
        # 90th-percentile cap (tau) lands well below the test follow-up max,
        # excluding train event times that are inside the data range but above
        # tau.
        y_train = make_y(list(np.arange(1.0, 21.0)))
        y_test = make_y(
            [2.0, 3.0, 4.0, 5.0, 6.0, 19.5],
            [True, True, True, True, True, False],  # last is censored, late
        )
        y_test_trunc = truncate_y_to_train_range(y_test, y_train, warn=False)
        tau = compute_safe_ipcw_tau(y_train, y_test_trunc)
        grid = ibs_integration_times(y_test, y_train)
        # tau binds below the test follow-up max.
        assert tau < y_test_trunc["time"].max()
        # No grid point exceeds tau, even though train has events (e.g. 15.0)
        # that are within the test data range but above tau.
        assert grid.max() <= tau
        assert 15.0 not in grid  # in data range, above tau -> excluded

    def test_cap_subsample_preserves_window_endpoints(self):
        # With many train event times and a small cap, the first and last
        # in-window event times are retained (the trapezoid still spans the
        # window).
        y_train = make_random_survival_array(3000, event_rate=0.8, seed=3)
        y_test = make_random_survival_array(400, event_rate=0.8, seed=4)
        full = ibs_integration_times(y_test, y_train, cap=10_000)
        capped = ibs_integration_times(y_test, y_train, cap=30)
        assert len(capped) <= 30
        assert len(full) > 30  # precondition: capping actually engaged
        assert capped[0] == full[0]    # first in-window event time kept
        assert capped[-1] == full[-1]  # last in-window event time kept

    def test_cap_keeps_all_when_below_cap(self):
        y_train = make_y(list(np.arange(1.0, 21.0)))  # events 1..20
        # Test cohort with events spread across the range so tau is high.
        y_test = make_y([1.0, 5.0, 10.0, 15.0, 19.0],
                        [True, True, True, True, True])
        grid = ibs_integration_times(y_test, y_train, cap=100)
        tau = compute_safe_ipcw_tau(y_train, y_test)
        # Below the cap, every train event time inside the window is kept
        # (no subsampling).
        train_event_times = np.unique(y_train["time"][y_train["event"]])
        expected = train_event_times[
            (train_event_times > y_test["time"].min())
            & (train_event_times < min(tau, y_test["time"].max()))
        ]
        np.testing.assert_array_equal(grid, expected)
        assert len(grid) > 2  # sanity: a non-degenerate window

    def test_cap_subsamples_to_actual_event_times(self):
        y_train = make_random_survival_array(2000, event_rate=0.7, seed=1)
        y_test = make_random_survival_array(300, event_rate=0.7, seed=2)
        cap = 40
        grid = ibs_integration_times(y_test, y_train, cap=cap)
        assert len(grid) <= cap
        # Subsampled points are still actual training event times.
        train_event_times = set(
            np.unique(y_train["time"][y_train["event"]]).tolist()
        )
        assert all(g in train_event_times for g in grid)

    def test_default_cap_is_100(self):
        assert IBS_GRID_CAP == 100

    def test_tau_none_falls_back_to_test_max_no_raise(self):
        # No events anywhere -> compute_safe_ipcw_tau returns None.
        y_train = make_y([1.0, 2.0, 3.0], [False, False, False])
        y_test = make_y([1.0, 2.0, 3.0], [False, False, False])
        # Must not raise; no train events -> empty grid.
        grid = ibs_integration_times(y_test, y_train)
        assert len(grid) == 0

    def test_no_train_events_returns_empty(self):
        y_train = make_y([1.0, 2.0, 3.0, 4.0], [False, False, False, False])
        y_test = make_y([1.0, 2.0, 3.0, 4.0], [True, True, False, True])
        grid = ibs_integration_times(y_test, y_train)
        assert len(grid) == 0

    def test_none_inputs_return_empty(self):
        assert len(ibs_integration_times(None, make_y([1.0, 2.0]))) == 0
        assert len(ibs_integration_times(make_y([1.0, 2.0]), None)) == 0

    def test_grid_points_come_from_train_not_test(self):
        # A test-only event time (absent from y_train) must never appear in the
        # grid: points are drawn from y_train's event times. (The window bound
        # tau is test-dependent; the *points* are train-anchored. Downstream
        # consumers compute the grid once on the full test set and reuse it, so
        # subpopulations/resamples share it — see the integration tests.)
        y_train = make_y([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_test = make_y([0.5, 3.7, 6.3, 9.5],  # 3.7/6.3 are test-only times
                        [True, True, True, True])
        grid = ibs_integration_times(y_test, y_train)
        assert 3.7 not in grid and 6.3 not in grid
        train_event_times = set(
            np.unique(y_train["time"][y_train["event"]]).tolist()
        )
        assert all(g in train_event_times for g in grid)


# ── reference match: still sksurv's trapezoid integrator over the grid ───


class TestIntegratedBrierFromProbs:
    """_integrated_brier_from_probs == direct sksurv on the same grid."""

    def test_matches_sksurv_when_no_clipping(self):
        from sksurv.metrics import integrated_brier_score

        rng = np.random.RandomState(0)
        y_train = make_random_survival_array(200, seed=10)
        y_test = make_random_survival_array(120, seed=11)
        y_ipcw = truncate_y_to_train_range(y_test, y_train, warn=False)

        ibs_times = ibs_integration_times(y_test, y_train)
        assert len(ibs_times) >= 2
        # Build a plausible decreasing S(t|x) matrix on the grid.
        base = np.sort(rng.uniform(0.2, 0.95, size=len(ibs_times)))[::-1]
        surv_probs = np.clip(
            base[None, :] + rng.normal(0, 0.02, size=(len(y_test), len(ibs_times))),
            0.0, 1.0,
        )
        surv_probs = np.minimum.accumulate(surv_probs, axis=1)

        ours = _integrated_brier_from_probs(surv_probs, ibs_times, y_test, y_train)
        # All ibs_times lie within (t_min, t_max) of y_ipcw -> no clipping.
        ref = float(integrated_brier_score(y_train, y_ipcw, surv_probs, ibs_times))
        assert ours == pytest.approx(ref, abs=1e-12)

    def test_too_few_points_returns_nan(self):
        y_train = make_y([1.0, 2.0, 3.0])
        y_test = make_y([1.5, 1.6])  # window admits < 2 grid points
        ibs_times = np.array([1.5])  # single point -> NaN after clip
        probs = np.full((2, 1), 0.5)
        assert np.isnan(
            _integrated_brier_from_probs(probs, ibs_times, y_test, y_train)
        )


# ── headline acceptance: grid-independence (integration) ─────────────────


def _gbsg2_learner(eval_times, gbsg2_data):
    """Train CoxPH and evaluate on a fixed split with the given display grid."""
    from mlsurv import SurvivalLearner

    X_raw, y_raw = gbsg2_data
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array([(bool(e), float(t)) for e, t in y_raw],
                 dtype=[("event", bool), ("time", float)])

    lr = SurvivalLearner(X, y)
    lr.setup(feature_selector="topk", random_state=42, test_fraction=0.3)
    lr.train("coxph")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.evaluate(eval_times=eval_times, n_importance_repeats=0)
    return lr


@pytest.mark.integration
class TestIbsGridIndependence:
    """IBS must not move when only the display grid changes."""

    def test_ibs_stable_across_display_grids(self, gbsg2_data):
        from mlsurv.utils import auto_eval_times

        _, y_raw = gbsg2_data
        y = np.array([(bool(e), float(t)) for e, t in y_raw],
                     dtype=[("event", bool), ("time", float)])
        g_coarse = auto_eval_times(y, y_train=y)             # ~10 points
        g_fine = np.linspace(g_coarse.min(), g_coarse.max(), 25)

        lr_a = _gbsg2_learner(g_coarse, gbsg2_data)
        lr_b = _gbsg2_learner(g_fine, gbsg2_data)

        ibs_a = lr_a.best_model.evaluation.concordance.iloc[0]["integrated_brier_test"]
        ibs_b = lr_b.best_model.evaluation.concordance.iloc[0]["integrated_brier_test"]
        # Headline criterion: bit-identical across display grids.
        assert ibs_a == pytest.approx(ibs_b, abs=1e-9)

    def test_brier_median_does_move_negative_control(self, gbsg2_data):
        from mlsurv.utils import auto_eval_times

        _, y_raw = gbsg2_data
        y = np.array([(bool(e), float(t)) for e, t in y_raw],
                     dtype=[("event", bool), ("time", float)])
        g_coarse = auto_eval_times(y, y_train=y)
        # Shift the display grid so the median display time changes.
        g_shifted = np.linspace(
            float(np.percentile(y["time"][y["event"]], 20)),
            float(np.percentile(y["time"][y["event"]], 95)),
            7,
        )
        lr_a = _gbsg2_learner(g_coarse, gbsg2_data)
        lr_b = _gbsg2_learner(g_shifted, gbsg2_data)
        br_a = lr_a.best_model.evaluation.concordance.iloc[0]["brier_score_test"]
        br_b = lr_b.best_model.evaluation.concordance.iloc[0]["brier_score_test"]
        # Brier(median) is a display-grid metric: it SHOULD change. This proves
        # we did not over-decouple (display grid still drives display metrics).
        assert not np.isclose(br_a, br_b)


# ── bootstrap: matches headline + predicts dense matrix once ─────────────


@pytest.mark.integration
class TestBootstrapDenseIbs:
    """Bootstrap IBS uses the dense grid and predicts it once per model."""

    def test_bootstrap_point_matches_headline(self, gbsg2_data):
        from mlsurv import SurvivalLearner

        X_raw, y_raw = gbsg2_data
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array([(bool(e), float(t)) for e, t in y_raw],
                     dtype=[("event", bool), ("time", float)])
        lr = SurvivalLearner(X, y)
        lr.setup(feature_selector="topk", random_state=42, test_fraction=0.3)
        lr.train("coxph")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.evaluate(n_importance_repeats=0)
            lr.bootstrap(n_bootstrap=20, show_progress=False)

        ev_ibs = lr.best_model.evaluation.concordance.iloc[0]["integrated_brier_test"]
        summ = lr.best_model.bootstrap.summary
        row = summ[summ["metric"] == "integrated_brier"].iloc[0]
        assert row["point"] == pytest.approx(float(ev_ibs), abs=1e-9)
        assert np.isfinite(row["ci_lower"]) and np.isfinite(row["ci_upper"])
        assert row["ci_lower"] <= row["ci_upper"]

    def test_dense_matrix_predicted_once_not_per_iteration(self):
        # Spy on _predict_dense_ibs_probs inside the precompute helper: it must
        # be called once per bootstrap() (not once per iteration).
        from mlsurv.evaluation import _bootstrap_workers as bw

        X, y = _small_xy()
        from mlsurv import SurvivalLearner
        lr = SurvivalLearner(X, y)
        lr.setup(random_state=42, test_fraction=0.3)
        lr.train("coxph")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.evaluate(n_importance_repeats=0)

            calls = {"n": 0}
            orig = bw._predict_dense_ibs_probs

            def _counting(*args, **kwargs):
                calls["n"] += 1
                return orig(*args, **kwargs)

            bw._predict_dense_ibs_probs = _counting
            try:
                lr.bootstrap(n_bootstrap=25, show_progress=False, n_jobs=1)
            finally:
                bw._predict_dense_ibs_probs = orig

        # One dense precompute for the single model, independent of n_bootstrap.
        assert calls["n"] == 1


# ── subpopulation: same train-anchored grid (integration) ────────────────


@pytest.mark.integration
class TestSubpopulationDenseIbs:
    """Subpopulation IBS is computed over the shared dense grid."""

    def test_subpopulation_ibs_present_and_finite(self, gbsg2_data):
        from mlsurv import SurvivalLearner

        X_raw, y_raw = gbsg2_data
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array([(bool(e), float(t)) for e, t in y_raw],
                     dtype=[("event", bool), ("time", float)])

        # Binary subpopulation on the first feature's median.
        groups = (X.iloc[:, 0] > X.iloc[:, 0].median()).map(
            {True: "high", False: "low"}
        ).to_numpy()

        lr = SurvivalLearner(X, y)
        lr.setup(feature_selector="topk", random_state=42, test_fraction=0.3,
                 subpopulations={"grp": groups})
        lr.train("coxph")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = lr.evaluate(n_importance_repeats=0)

        # The dense IBS cache must be populated for the model.
        assert ev._ibs_surv_probs_cache is not None
        assert "coxph" in ev._ibs_surv_probs_cache

        sg = lr.subpopulation_results.get("grp")
        assert sg is not None
        ibs_rows = sg.metrics[sg.metrics["metric"] == "integrated_brier"]
        # At least one subpopulation level reports a finite IBS.
        assert ibs_rows["value"].notna().any()

    def test_bootstrap_subpopulation_ibs_finite(self, gbsg2_data):
        # Exercises the per-iteration, double-sliced (resample idx -> subpop
        # mask) dense-IBS bootstrap path end-to-end.
        from mlsurv import SurvivalLearner

        X_raw, y_raw = gbsg2_data
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array([(bool(e), float(t)) for e, t in y_raw],
                     dtype=[("event", bool), ("time", float)])
        groups = (X.iloc[:, 0] > X.iloc[:, 0].median()).map(
            {True: "high", False: "low"}
        ).to_numpy()

        lr = SurvivalLearner(X, y)
        lr.setup(feature_selector="topk", random_state=42, test_fraction=0.3,
                 subpopulations={"grp": groups})
        lr.train("coxph")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.evaluate(n_importance_repeats=0)
            lr.bootstrap(n_bootstrap=15, show_progress=False)

        # Per-subpopulation bootstrap IBS lands in the model's bootstrap summary
        # as ``integrated_brier__subpopulation_<level>`` rows (computed over the
        # dense grid, double-sliced by resample index then subpop mask).
        boot = lr.best_model.bootstrap.summary
        ibs = boot[boot["metric"].astype(str).str.contains(
            "integrated_brier__subpopulation"
        )]
        assert not ibs.empty
        # At least one subpopulation level has a finite, ordered bootstrap CI.
        finite = ibs[ibs["ci_lower"].notna() & ibs["ci_upper"].notna()]
        assert len(finite) >= 1
        assert (finite["ci_lower"] <= finite["ci_upper"]).all()


class TestSubpopulationNoneCacheGuard:
    """A (None, times) dense-cache entry must not crash the subpop IBS slice."""

    def test_none_dense_matrix_skips_ibs_without_crashing(self):
        import pandas as pd
        from mlsurv.evaluation._subpopulation import (
            _compute_model_subpopulation_metrics,
        )

        n = 8
        rng = np.random.RandomState(0)
        test_set = pd.DataFrame({
            "model_type": ["coxph"] * n,
            "risk_score": rng.standard_normal(n),
        })
        mask = np.array([True] * 4 + [False] * 4)
        y_all = make_random_survival_array(n, seed=1)
        y_sub = y_all[mask]
        y_train = make_random_survival_array(40, seed=2)

        # Degenerate cache entry: matrix is None (grid was degenerate). The
        # slice ``ibs_sp[mask, :]`` would raise outside _safe_ibs if unguarded.
        rows, _ = _compute_model_subpopulation_metrics(
            "coxph", test_set, mask, y_sub, y_train, None, None, "lo",
            dense_surv_probs_by_model={"coxph": (None, np.array([1.0, 2.0]))},
        )
        metrics = {r["metric"] for r in rows}
        assert "integrated_brier" not in metrics  # skipped, no crash


def _small_xy():
    """Small but trainable dataset for fast bootstrap structure tests."""
    from tests._helpers import make_feature_dataframe
    X = make_feature_dataframe(120, 4, seed=7)
    y = make_random_survival_array(120, event_rate=0.7, seed=7)
    return X, y
