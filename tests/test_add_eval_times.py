"""Tests for add_eval_times(): additive report horizons with lazy NA semantics.

``add_eval_times`` registers report horizons that flow ADDITIVELY into FUTURE
time-dependent metrics (the effective grid is union(base, extra) clipped to the
censoring-stable window). Registration triggers no recompute, so already-computed
results show NA at the new times until the user re-runs evaluate()/bootstrap();
plots render those NA values honestly with a one-time re-run hint.
"""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
from mlsurv.learner._evaluation_helpers import (
    _censoring_window,
    _validate_eval_times_array,
)
from mlsurv.utils import union_extra_eval_times
from mlsurv.viz._data import align_metric_time_to_grid


def _y(times, events):
    return np.array(
        list(zip(events, times)), dtype=[("event", bool), ("time", float)],
    )


# ----------------------------------------------------------------------
# Pure helpers (no setup / no training)
# ----------------------------------------------------------------------

class TestValidateEvalTimesArray:
    """The shared validation reused by the resolver and add_eval_times."""

    def test_scalar_raises_typeerror_with_hint(self):
        with pytest.raises(TypeError, match=r"times=\[5"):
            _validate_eval_times_array(5.0, param_name="times")

    def test_two_dimensional_raises_valueerror(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            _validate_eval_times_array([[1, 2], [3, 4]])

    @pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-1.0], [0.0]])
    def test_nonfinite_or_nonpositive_raises(self, bad):
        with pytest.raises(ValueError, match="positive and finite"):
            _validate_eval_times_array(bad)

    def test_empty_passes_and_returns_float_array(self):
        out = _validate_eval_times_array([])
        assert out.dtype == float and out.size == 0

    def test_valid_coerced_to_float(self):
        out = _validate_eval_times_array([1, 2, 3])
        assert out.dtype == float
        assert np.array_equal(out, [1.0, 2.0, 3.0])


class TestCensoringWindow:
    def test_returns_min_max(self):
        t_min, t_max = _censoring_window(_y([1, 5, 9], [True, True, False]))
        assert (t_min, t_max) == (1.0, 9.0)

    def test_none_train_returns_none_pair(self):
        assert _censoring_window(None) == (None, None)

    def test_degenerate_returns_none_pair(self):
        assert _censoring_window(_y([4, 4, 4], [True, False, True])) == (None, None)


class TestUnionExtraEvalTimes:
    def test_none_extra_is_noop(self):
        base = np.array([1.0, 2.0])
        assert union_extra_eval_times(base, None, None) is base

    def test_unions_and_sorts_unique(self):
        base = np.array([2.0, 4.0])
        out = union_extra_eval_times(base, [4.0, 1.0, 3.0], None)
        assert np.array_equal(out, [1.0, 2.0, 3.0, 4.0])

    def test_drops_out_of_window_extras_silently(self):
        base = np.array([2.0, 4.0])
        y_train = _y([1, 5, 10], [True, True, False])  # window [1, 10)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # must NOT warn here
            out = union_extra_eval_times(base, [3.0, 99.0], y_train)
        assert np.array_equal(out, [2.0, 3.0, 4.0])  # 99 (>= t_max) dropped


class TestAlignMetricTimeToGrid:
    def test_inserts_na_rows_for_missing_times(self):
        df = pd.DataFrame({
            "model_type": ["coxph", "coxph"],
            "eval_time": [1.0, 2.0],
            "auc": [0.7, 0.8],
        })
        out = align_metric_time_to_grid(df, [1.0, 2.0, 3.0])
        assert sorted(out["eval_time"].unique()) == [1.0, 2.0, 3.0]
        assert out.loc[out["eval_time"] == 3.0, "auc"].isna().all()

    def test_no_op_when_grid_matches(self):
        df = pd.DataFrame({
            "model_type": ["coxph"], "eval_time": [1.0], "auc": [0.7],
        })
        out = align_metric_time_to_grid(df, [1.0])
        assert len(out) == 1 and out["auc"].iloc[0] == 0.7

    def test_per_model_cross_product(self):
        df = pd.DataFrame({
            "model_type": ["a", "b"], "eval_time": [1.0, 2.0], "auc": [0.7, 0.8],
        })
        out = align_metric_time_to_grid(df, [1.0, 2.0])
        # Each model gets every grid time.
        assert len(out) == 4
        assert out.loc[(out.model_type == "a") & (out.eval_time == 2.0),
                       "auc"].isna().all()


# ----------------------------------------------------------------------
# Registration API (setup only, no training — fast tier)
# ----------------------------------------------------------------------

class TestRegistration:
    def _learner(self, small_data):
        X, y = small_data
        lr = SurvivalLearner(X, y)
        lr.setup(random_state=42, verbose=False)
        return lr

    def test_register_sorted_deduped_union(self, small_data):
        lr = self._learner(small_data)
        t_max = float(lr.y_train["time"].max())
        a, b = t_max * 0.3, t_max * 0.6
        lr.add_eval_times([b, a])
        lr.add_eval_times([a])  # dup
        assert np.array_equal(lr.report_eval_times, np.array(sorted([a, b])))

    def test_report_eval_times_empty_by_default(self, small_data):
        lr = self._learner(small_data)
        assert lr.report_eval_times.size == 0

    def test_config_entry(self, small_data):
        lr = self._learner(small_data)
        t = float(lr.y_train["time"].max()) * 0.5
        lr.add_eval_times([t])
        assert lr.config["report_eval_times"] == [t]

    def test_empty_is_noop(self, small_data):
        lr = self._learner(small_data)
        t = float(lr.y_train["time"].max()) * 0.5
        lr.add_eval_times([t])
        before = lr.report_eval_times.copy()
        out = lr.add_eval_times([])
        assert np.array_equal(out, before)

    def test_scalar_raises_typeerror(self, small_data):
        lr = self._learner(small_data)
        with pytest.raises(TypeError):
            lr.add_eval_times(5.0)

    def test_out_of_window_warns_but_keeps(self, small_data):
        lr = self._learner(small_data)
        t_max = float(lr.y_train["time"].max())
        out_t = t_max + 1000.0
        with pytest.warns(MethodologicalWarning, match="censoring-stable window"):
            lr.add_eval_times([out_t])
        assert out_t in lr.report_eval_times  # kept, not dropped

    def test_setup_clears_registered_horizons(self, small_data):
        lr = self._learner(small_data)
        lr.add_eval_times([float(lr.y_train["time"].max()) * 0.5])
        lr.setup(random_state=42, verbose=False)
        assert lr.report_eval_times.size == 0
        assert lr._state.extra_eval_times_render_warned is False


# ----------------------------------------------------------------------
# Integration: compute threading, lazy NA, rendering (training required)
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def _trained_template():
    """Train once on GBSG2-like data; reuse via deepcopy in mutating tests."""
    from sksurv.datasets import load_gbsg2

    X_raw, y_raw = load_gbsg2()
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array([(bool(e), t) for e, t in y_raw],
                 dtype=[("event", bool), ("time", float)])
    lr = SurvivalLearner(X, y)
    lr.setup(random_state=42, verbose=False)
    lr.train("coxph")
    lr.evaluate(verbose=False)
    return lr


@pytest.fixture
def trained(_trained_template):
    return copy.deepcopy(_trained_template)


def _mid_window(lr):
    t_min = float(lr.y_train["time"].min())
    t_max = float(lr.y_train["time"].max())
    return float(np.round((t_min + t_max) / 2.0))


@pytest.mark.integration
class TestComputeThreading:
    def test_lazy_na_then_refill(self, trained):
        before = np.sort(trained.evaluation_result.auc_t["eval_time"].unique())
        new_t = _mid_window(trained)
        trained.add_eval_times([new_t])
        # No recompute: stored auc_t unchanged.
        after_add = np.sort(trained.evaluation_result.auc_t["eval_time"].unique())
        assert np.array_equal(before, after_add)
        # Re-evaluate fills the new horizon.
        trained.evaluate(verbose=False)
        refilled = np.sort(trained.evaluation_result.auc_t["eval_time"].unique())
        assert new_t in refilled

    def test_point_and_ci_grid_match_on_extended_grid(self, trained):
        new_t = _mid_window(trained)
        trained.add_eval_times([new_t])
        trained.evaluate(verbose=False)
        res = trained.bootstrap(
            "coxph", n_bootstrap=3, include_time_resolved=True,
            verbose=False, show_progress=False,
        )
        point = np.sort(trained.evaluation_result.auc_t["eval_time"].unique())
        band = np.sort(res.auc_t["eval_time"].unique())
        assert np.array_equal(point, band)
        assert new_t in band


@pytest.mark.integration
class TestNaAwareRendering:
    def test_uncomputed_horizon_renders_na_with_hint(self, trained):
        new_t = _mid_window(trained)
        trained.add_eval_times([new_t])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = trained.plot_metric_time(return_data=True)
            hints = [x for x in w if issubclass(x.category, SuggestionWarning)
                     and "render as NA" in str(x.message)]
        df = out["data"] if isinstance(out, dict) else out
        assert new_t in df["eval_time"].to_numpy()
        row = df[np.isclose(df["eval_time"], new_t)]
        assert row["auc"].isna().all()
        assert len(hints) == 1

    def test_hint_is_one_time(self, trained):
        trained.add_eval_times([_mid_window(trained)])
        trained.plot_metric_time(return_data=True)  # arms + fires
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained.plot_metric_time(return_data=True)
            hints = [x for x in w if issubclass(x.category, SuggestionWarning)
                     and "render as NA" in str(x.message)]
        assert hints == []

    def test_out_of_window_horizon_stays_na_after_eval(self, trained):
        out_t = float(trained.y_train["time"].max()) + 5000.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MethodologicalWarning)
            trained.add_eval_times([out_t])
        trained.evaluate(verbose=False)  # cannot fill out-of-window
        out = trained.plot_metric_time(return_data=True)
        df = out["data"] if isinstance(out, dict) else out
        assert out_t in df["eval_time"].to_numpy()
        assert df[np.isclose(df["eval_time"], out_t)]["auc"].isna().all()

    def test_no_extras_grid_unchanged(self, trained):
        out = trained.plot_metric_time(return_data=True)
        df = out["data"] if isinstance(out, dict) else out
        auc_times = np.sort(trained.evaluation_result.auc_t["eval_time"].unique())
        assert np.array_equal(np.sort(df["eval_time"].unique()), auc_times)

    def test_figure_renders_with_na(self, trained):
        trained.add_eval_times([_mid_window(trained)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SuggestionWarning)
            fig = trained.plot_metric_time()
        assert len(fig.data) > 0


@pytest.mark.integration
class TestDefaultHorizonFallback:
    def test_calibration_defaults_to_report_horizons(self, trained):
        t_min = float(trained.y_train["time"].min())
        t_max = float(trained.y_train["time"].max())
        h = float(np.round(t_min + (t_max - t_min) * 0.4))
        trained.add_eval_times([h])
        out = trained.plot_calibration(return_data=True)
        df = out["data"] if isinstance(out, dict) else out
        assert h in np.round(df["eval_time"].to_numpy(float), 6)

    def test_compute_brier_curves_default_training_anchored(self, trained):
        from mlsurv.viz._metrics_roc import compute_brier_curves

        bc = compute_brier_curves(
            {"coxph": trained.models["coxph"].model},
            trained.X_train, trained.y_train,
            trained.X_test, trained.y_test,
        )
        assert bc["eval_time"].max() < trained.y_test["time"].max()
        assert bc["eval_time"].nunique() >= 2


@pytest.mark.integration
class TestPersistence:
    def test_report_eval_times_round_trips(self, trained, tmp_path):
        h = _mid_window(trained)
        trained.add_eval_times([h])
        path = tmp_path / "lr.joblib"
        trained.save_learner(str(path))
        reloaded = SurvivalLearner.load_learner(str(path))
        assert np.array_equal(reloaded.report_eval_times, np.array([h]))


# ----------------------------------------------------------------------
# Subpopulation facets honor registered report horizons
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def _subpop_template():
    """Train once on GBSG2 with a subpopulation declared; reuse via deepcopy."""
    from sksurv.datasets import load_gbsg2

    X_raw, y_raw = load_gbsg2()
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array([(bool(e), t) for e, t in y_raw],
                 dtype=[("event", bool), ("time", float)])
    grp = np.where(X["age"] > X["age"].median(), "old", "young")
    lr = SurvivalLearner(X, y)
    lr.setup(subpopulations={"agegrp": grp}, random_state=42, verbose=False)
    lr.train("coxph")
    lr.evaluate(verbose=False)
    return lr


@pytest.fixture
def subpop_trained(_subpop_template):
    return copy.deepcopy(_subpop_template)


def _subpop_data(view_out):
    """Subpopulation plot return_data is a bare DataFrame; tolerate dict too."""
    return view_out["data"] if isinstance(view_out, dict) else view_out


@pytest.mark.integration
class TestSubpopulationHorizons:
    def test_uncomputed_horizon_renders_na_with_hint(self, subpop_trained):
        new_t = _mid_window(subpop_trained)
        subpop_trained.add_eval_times([new_t])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = subpop_trained.subpopulation("agegrp").plot_metric_time(
                return_data=True,
            )
            hints = [x for x in w if issubclass(x.category, SuggestionWarning)
                     and "render as NA" in str(x.message)]
        df = _subpop_data(out)
        assert new_t in df["eval_time"].to_numpy()
        # No recompute yet: the registered horizon is present but NA on every facet.
        assert df[np.isclose(df["eval_time"], new_t)]["auc"].isna().all()
        assert len(hints) == 1

    def test_horizon_filled_after_reevaluate(self, subpop_trained):
        new_t = _mid_window(subpop_trained)
        subpop_trained.add_eval_times([new_t])
        subpop_trained.evaluate(verbose=False)  # union into the subpop grid
        out = subpop_trained.subpopulation("agegrp").plot_metric_time(
            return_data=True,
        )
        df = _subpop_data(out)
        assert new_t in df["eval_time"].to_numpy()
        # At least one facet now carries a real value at the new horizon.
        assert df[np.isclose(df["eval_time"], new_t)]["auc"].notna().any()

    def test_multi_metric_horizon_injected_per_facet(self, subpop_trained):
        # The multi-metric facet path keys alignment on
        # ("subpopulation", "metric_name"); assert the registered horizon is
        # injected into BOTH metric facets across BOTH subpopulation levels
        # (not just one facet, and not duplicated), all NA before recompute.
        new_t = _mid_window(subpop_trained)
        subpop_trained.add_eval_times([new_t])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SuggestionWarning)
            out = subpop_trained.subpopulation("agegrp").plot_metric_time(
                metric=["auc", "brier"], return_data=True,
            )
        df = _subpop_data(out)
        at_t = df[np.isclose(df["eval_time"], new_t)]
        assert set(at_t["metric_name"].unique()) == {"auc", "brier"}
        assert set(at_t["subpopulation"].unique()) == {"old", "young"}
        assert at_t["value"].isna().all()

    def test_out_of_window_horizon_stays_na_after_eval(self, subpop_trained):
        # In-window-of-cohort but unfillable horizons render as honest NA on
        # the facets even after a recompute (compute-grid clip vs render-grid
        # re-add asymmetry).
        out_t = float(subpop_trained.y_train["time"].max()) + 5000.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MethodologicalWarning)
            subpop_trained.add_eval_times([out_t])
        subpop_trained.evaluate(verbose=False)  # cannot fill out-of-window
        out = subpop_trained.subpopulation("agegrp").plot_metric_time(
            return_data=True,
        )
        df = _subpop_data(out)
        assert out_t in df["eval_time"].to_numpy()
        assert df[np.isclose(df["eval_time"], out_t)]["auc"].isna().all()

    def test_hint_shared_across_surfaces(self, subpop_trained):
        # extra_eval_times_render_warned is a single shared flag: a hint fired
        # on the cohort surface suppresses it on the subpopulation surface.
        subpop_trained.add_eval_times([_mid_window(subpop_trained)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SuggestionWarning)
            subpop_trained.plot_metric_time(return_data=True)  # arms + fires
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            subpop_trained.subpopulation("agegrp").plot_metric_time(
                return_data=True,
            )
            hints = [x for x in w if issubclass(x.category, SuggestionWarning)
                     and "render as NA" in str(x.message)]
        assert hints == []

    def test_no_extras_grid_unchanged(self, subpop_trained):
        out = subpop_trained.subpopulation("agegrp").plot_metric_time(
            return_data=True,
        )
        df = _subpop_data(out)
        stored = subpop_trained._state.subpopulation_results["agegrp"].auc_t
        # No registered horizons: no NA rows injected, no row explosion or
        # reordering — same row count, same non-NA count, same grid.
        assert len(df) == len(stored)
        assert int(df["auc"].notna().sum()) == int(stored["auc"].notna().sum())
        assert np.array_equal(
            np.sort(df["eval_time"].unique()),
            np.sort(stored["eval_time"].unique()),
        )
