"""Tests for plot_calibration's interactive dropdown, self-describing titles,
the unified ``eval_times`` parameter (scalar or list), and the default-horizon
grid on the data path.

``learner.plot_calibration`` now returns an interactive figure with a dropdown
to switch evaluation horizon (the same renderer the evaluation/recalibration
reports use). The single-model title names the model (so ``models='best'``
reveals the winner); the multi-model title is "Calibration Comparison". When no
``eval_times`` is given the default is the canonical eval grid (the same
resolution AUC(t)/Brier(t)/ROC use), so the default path is silent — there is no
"25/50/75 percentile" warning.

Also covers ``plot_calibration_overlay``'s title handling: a user-supplied
``title`` is forwarded verbatim, and the auto-built default names the model.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import SuggestionWarning
from mlsurv.viz.calibration import (
    plot_calibration,
    plot_calibration_interactive,
)

_BAND = "95% CI (KM)"

# Synthetic calibration data for the data-path viz functions (no model needed).
_CAL_SINGLE = [
    {"eval_time": 6.0, "time_index": 0,
     "predicted": [0.2, 0.5, 0.8], "observed": [0.25, 0.55, 0.78],
     "ci_lower": [0.15, 0.45, 0.7], "ci_upper": [0.35, 0.65, 0.86]},
    {"eval_time": 12.0, "time_index": 1,
     "predicted": [0.1, 0.4, 0.7], "observed": [0.12, 0.42, 0.69],
     "ci_lower": [0.05, 0.32, 0.6], "ci_upper": [0.2, 0.5, 0.8]},
]
_CAL_MULTI = {
    "coxph": {"eval_time": 12.0, "predicted": [0.2, 0.5, 0.8],
              "observed": [0.25, 0.55, 0.78],
              "ci_lower": [0.15, 0.45, 0.7], "ci_upper": [0.35, 0.65, 0.86]},
    "coxnet": {"eval_time": 12.0, "predicted": [0.3, 0.6, 0.85],
               "observed": [0.28, 0.58, 0.82],
               "ci_lower": [0.2, 0.5, 0.75], "ci_upper": [0.4, 0.7, 0.92]},
}


@pytest.fixture(scope="module")
def cal_learner():
    """coxph + coxnet, evaluated. Read-only across these plot tests."""
    rng = np.random.default_rng(3)
    n, p = 220, 4
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    score = X["x0"].values + 0.5 * X["x1"].values
    times = np.maximum(0.1, rng.exponential(scale=np.exp(-score)))
    events = rng.binomial(1, 0.7, size=n).astype(bool)
    y = np.array(list(zip(events, times)),
                 dtype=[("event", bool), ("time", float)])
    learner = SurvivalLearner(X, y)
    learner.setup(verbose=False, ph_test=False, linearity_test=False,
                  check_leakage=False, cv_folds=3, random_state=42)
    learner.run(models=["coxph", "coxnet"], n_importance_repeats=0,
                bootstrap=False, verbose=False, show_eval_progress=False,
                tune=False)
    return learner


def _suggestions(record):
    return [str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)]


def _n_dropdown_buttons(fig):
    """Number of timepoint buttons in the figure's dropdown (0 if none)."""
    menus = fig.layout.updatemenus
    return len(menus[0].buttons) if menus else 0


class TestCalibrationBandSubtitle:
    """plot_calibration / plot_calibration_interactive name the KM CI band in a
    subtitle when a band is drawn, respecting a user title/subtitle. Pure
    data-path (no model), so fast-tier.
    """

    def test_single_ribbon_names_band(self):
        assert _BAND in plot_calibration(_CAL_SINGLE).layout.title.text

    def test_single_bars_names_band(self):
        title = plot_calibration(_CAL_SINGLE, ci_style="bars").layout.title.text
        assert _BAND in title

    def test_single_none_has_no_band(self):
        title = plot_calibration(_CAL_SINGLE, ci_style="none").layout.title.text
        assert _BAND not in title

    def test_user_title_keeps_band(self):
        title = plot_calibration(_CAL_SINGLE, title="My Cal").layout.title.text
        assert "My Cal" in title
        assert _BAND in title

    def test_empty_subtitle_suppresses(self):
        # subtitle="" is explicit suppression — no subtitle span at all.
        title = plot_calibration(_CAL_SINGLE, subtitle="").layout.title.text
        assert "<span" not in title

    def test_user_subtitle_preserved(self):
        title = plot_calibration(_CAL_SINGLE, subtitle="my note").layout.title.text
        assert "my note" in title
        assert _BAND not in title

    def test_comparison_names_band(self):
        title = plot_calibration(_CAL_MULTI).layout.title.text
        assert "Calibration Comparison" in title
        assert _BAND in title

    def test_comparison_none_has_no_band(self):
        title = plot_calibration(_CAL_MULTI, ci_style="none").layout.title.text
        assert _BAND not in title

    def test_comparison_bars_names_band(self):
        title = plot_calibration(_CAL_MULTI, ci_style="bars").layout.title.text
        assert _BAND in title

    def test_comparison_without_ci_data_has_no_band(self):
        # ci_style defaults to ribbon, but no model carries CI bounds, so no
        # band is drawn and the subtitle must not claim one.
        no_ci = {"coxph": {"eval_time": 12.0, "predicted": [0.2, 0.5, 0.8],
                           "observed": [0.25, 0.55, 0.78]}}
        title = plot_calibration(no_ci).layout.title.text
        assert _BAND not in title

    def test_interactive_band_in_title_and_buttons(self):
        fig = plot_calibration_interactive(_CAL_SINGLE)
        assert _BAND in fig.layout.title.text
        # Every dropdown button keeps the band line under its timepoint.
        for btn in fig.layout.updatemenus[0].buttons:
            assert _BAND in btn.args[1]["title.text"]

    def test_interactive_none_has_no_band(self):
        fig = plot_calibration_interactive(_CAL_SINGLE, ci_style="none")
        assert _BAND not in fig.layout.title.text

    def test_interactive_multi_without_ci_data_has_no_band(self):
        # Multi shape is {eval_time: {model: cal}}; no model carries CI bounds,
        # so the flattened has-CI check is False and no band is named.
        no_ci = {12.0: {"coxph": {"eval_time": 12.0,
                                  "predicted": [0.2, 0.5, 0.8],
                                  "observed": [0.25, 0.55, 0.78]}}}
        fig = plot_calibration_interactive(no_ci)
        assert _BAND not in fig.layout.title.text

    def test_interactive_empty_subtitle_suppresses(self):
        # Each dropdown button rebuilds the title independently, so check both.
        fig = plot_calibration_interactive(_CAL_SINGLE, subtitle="")
        assert "<span" not in fig.layout.title.text
        for btn in fig.layout.updatemenus[0].buttons:
            assert "<span" not in btn.args[1]["title.text"]

    def test_interactive_user_subtitle_in_buttons(self):
        fig = plot_calibration_interactive(_CAL_SINGLE, subtitle="my note")
        assert "my note" in fig.layout.title.text
        for btn in fig.layout.updatemenus[0].buttons:
            assert "my note" in btn.args[1]["title.text"]
            assert _BAND not in btn.args[1]["title.text"]


class TestComparisonHorizon:
    """Single-horizon pick for multi-model / subpopulation comparisons."""

    def test_user_times_take_first(self):
        from mlsurv.viz._plot_impl_calibration import _comparison_horizon
        # User-supplied times: the first one (caller warns about extras).
        assert _comparison_horizon([3.0, 1.0, 2.0], [9.0]) == 3.0

    def test_default_uses_central_grid_horizon(self):
        from mlsurv.viz._plot_impl_calibration import _comparison_horizon
        # No user times: the central element of the sorted default grid.
        assert _comparison_horizon(None, [5.0, 1.0, 3.0]) == 3.0
        assert _comparison_horizon(None, [4.0, 1.0, 3.0, 2.0]) == 3.0


def _expected_default_n(learner):
    """Length of the canonical default eval grid plot_calibration should use."""
    from mlsurv.utils import resolve_display_eval_times

    state = learner._state
    return len(resolve_display_eval_times(state, state.y_test, state.y_train))


@pytest.mark.integration
class TestInteractiveDropdown:
    def test_single_model_has_dropdown(self, cal_learner):
        fig = cal_learner.plot_calibration(models="coxph")
        assert isinstance(fig, go.Figure)
        # Default seeds the dropdown with the canonical eval grid (the same
        # resolution AUC(t)/Brier(t)/ROC use), not a fixed 25/50/75 triple.
        assert _n_dropdown_buttons(fig) == _expected_default_n(cal_learner)

    def test_multi_model_has_dropdown(self, cal_learner):
        fig = cal_learner.plot_calibration(models=["coxph", "coxnet"])
        assert _n_dropdown_buttons(fig) == _expected_default_n(cal_learner)

    def test_scalar_eval_times_one_button(self, cal_learner):
        fig = cal_learner.plot_calibration(models="coxph", eval_times=1.0)
        assert _n_dropdown_buttons(fig) == 1

    def test_list_eval_times_n_buttons(self, cal_learner):
        fig = cal_learner.plot_calibration(models=["coxph", "coxnet"],
                                           eval_times=[0.5, 1.0, 2.0, 3.0])
        assert _n_dropdown_buttons(fig) == 4


@pytest.mark.integration
class TestCalibrationTitles:
    def test_single_model_title_names_model(self, cal_learner):
        fig = cal_learner.plot_calibration(models="coxph")
        assert "CoxPH" in fig.layout.title.text

    def test_best_model_title_names_winner(self, cal_learner):
        fig = cal_learner.plot_calibration(models="best")
        from mlsurv.viz._palettes import get_model_display_name
        winner = get_model_display_name(cal_learner.best_model.name)
        assert winner in fig.layout.title.text

    def test_multi_model_title_is_comparison(self, cal_learner):
        fig = cal_learner.plot_calibration(models=["coxph", "coxnet"])
        assert "Calibration Comparison" in fig.layout.title.text

    def test_user_title_override_wins(self, cal_learner):
        fig = cal_learner.plot_calibration(models="coxph", title="My Title")
        assert "My Title" in fig.layout.title.text


@pytest.mark.integration
class TestCalibrationOverlayTitle:
    # eval_times is kept in-range (the fixture's test times extend past the
    # model's training domain); the title is independent of eval_times.
    def test_user_title_used_verbatim(self, cal_learner):
        """A supplied title is forwarded as-is (no model name appended)."""
        fig = cal_learner.plot_calibration_overlay(
            models="coxph", title="My Title", eval_times=[1.0],
        )
        assert "My Title" in fig.layout.title.text
        # Regression: the wrapper used to append the model name ("My Title: coxph").
        assert "CoxPH" not in fig.layout.title.text

    def test_default_title_names_model(self, cal_learner):
        """With no title, the auto-built default names the model."""
        fig = cal_learner.plot_calibration_overlay(models="coxph", eval_times=[1.0])
        assert "Calibration Overlay" in fig.layout.title.text
        assert "CoxPH" in fig.layout.title.text

    def test_band_subtitle_when_show_ci(self, cal_learner):
        """show_ci draws a KM CI ribbon, so the subtitle names the band."""
        fig = cal_learner.plot_calibration_overlay(
            models="coxph", eval_times=[1.0], show_ci=True,
        )
        assert "95% CI (KM)" in fig.layout.title.text

    def test_no_band_subtitle_without_ci(self, cal_learner):
        fig = cal_learner.plot_calibration_overlay(
            models="coxph", eval_times=[1.0], show_ci=False,
        )
        assert "95% CI (KM)" not in fig.layout.title.text

    def test_multi_panel_overlay_band_when_show_ci(self, cal_learner):
        # The per-group panels draw a KM CI ribbon under show_ci, so the band
        # is both drawn and named.
        fig = cal_learner.plot_calibration_overlay(
            models="coxph", eval_times=[1.0], n_groups=2, show_ci=True,
        )
        assert "95% CI (KM)" in fig.layout.title.text
        assert any(getattr(tr, "fill", None) == "tonexty" for tr in fig.data)

    def test_multi_panel_overlay_no_band_without_ci(self, cal_learner):
        fig = cal_learner.plot_calibration_overlay(
            models="coxph", eval_times=[1.0], n_groups=2, show_ci=False,
        )
        assert "95% CI (KM)" not in fig.layout.title.text
        assert not any(getattr(tr, "fill", None) == "tonexty" for tr in fig.data)


@pytest.mark.integration
class TestNoFigureWarning:
    """The dropdown lists the horizons, so the figure path must not warn."""

    def test_single_model_figure_no_default_warning(self, cal_learner):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cal_learner.plot_calibration(models="coxph")
        assert not _suggestions(rec)

    def test_multi_model_figure_no_default_warning(self, cal_learner):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cal_learner.plot_calibration(models=["coxph", "coxnet"])
        assert not _suggestions(rec)


@pytest.mark.integration
class TestReturnData:
    def test_single_model_spans_all_times(self, cal_learner):
        data = cal_learner.plot_calibration(models="coxph",
                                            eval_times=[0.5, 1.0, 2.0],
                                            return_data=True)
        assert isinstance(data, pd.DataFrame)
        assert set(np.round(data["eval_time"].unique(), 3)) == {0.5, 1.0, 2.0}

    def test_multi_model_spans_all_times_and_models(self, cal_learner):
        data = cal_learner.plot_calibration(models=["coxph", "coxnet"],
                                            eval_times=[0.5, 1.0],
                                            return_data=True)
        assert set(np.round(data["eval_time"].unique(), 3)) == {0.5, 1.0}
        assert set(data["model"].unique()) == {"coxph", "coxnet"}

    def test_default_grid_no_warning(self, cal_learner):
        """The default path uses the canonical eval grid silently — no more
        '25/50/75 percentile' default warning (the grid now matches AUC/ROC)."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cal_learner.plot_calibration(models="coxph", return_data=True)
        msgs = _suggestions(rec)
        assert not any("no eval_times given" in m for m in msgs), msgs

    def test_default_grid_matches_resolver(self, cal_learner):
        """The default return_data grid equals the canonical resolved grid.

        ``n_bins=4`` keeps every horizon's KM bins populated on this cohort, so
        the returned grid spans all resolved horizons rather than a
        sparsity-dropped subset.
        """
        from mlsurv.utils import resolve_display_eval_times
        state = cal_learner._state
        expected = sorted(
            float(t) for t in resolve_display_eval_times(
                state, state.y_test, state.y_train)
        )
        data = cal_learner.plot_calibration(models="coxph", n_bins=4,
                                            return_data=True)
        got = sorted(float(t) for t in data["eval_time"].unique())
        assert got == expected

    def test_explicit_times_no_warning(self, cal_learner):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cal_learner.plot_calibration(models="coxph", eval_times=[1.0],
                                         return_data=True)
        assert not _suggestions(rec)


@pytest.mark.integration
class TestCrossSurfaceGridConsistency:
    """plot_calibration's default grid lines up with AUC(t)/metric_time/ROC.

    The viz layer routes its default through the same materializer the compute
    surfaces use, so after a default (auto) evaluate() the calibration grid,
    plot_metric_time('auc') grid, and stored auc_t share one eval_time set.

    We assert on the *resolved* grid (what calibration asks for) rather than the
    ``return_data`` grid, because ``compute_calibration`` may drop individual
    horizons whose KM bins are too sparse — a property of binning, not of grid
    resolution. (End-to-end ``return_data`` equality on a well-populated cohort
    is exercised separately by an out-of-band validation script on real data.)
    """

    @staticmethod
    def _uniq(df):
        return np.sort(np.unique(np.round(df["eval_time"].to_numpy(float), 6)))

    def _resolved(self, learner):
        from mlsurv.utils import resolve_display_eval_times
        state = learner._state
        return np.sort(np.round(
            resolve_display_eval_times(state, state.y_test, state.y_train), 6))

    def test_default_grid_matches_auc_and_metric_time(self, cal_learner):
        import copy
        lr = copy.deepcopy(cal_learner)
        lr.evaluate(verbose=False)
        auc_grid = self._uniq(lr.evaluation_result.auc_t)

        mt = lr.plot_metric_time(metric="auc", return_data=True)
        mt = mt["data"] if isinstance(mt, dict) else mt

        # Calibration resolves the same grid AUC(t)/metric_time land on.
        assert np.array_equal(self._resolved(lr), auc_grid)
        assert np.array_equal(self._uniq(mt), auc_grid)

    def test_return_data_grid_is_nonempty_subset_of_auc(self, cal_learner):
        import copy
        lr = copy.deepcopy(cal_learner)
        lr.evaluate(verbose=False)
        auc_grid = set(self._uniq(lr.evaluation_result.auc_t).tolist())
        cal = lr.plot_calibration(models="coxph", n_bins=4, return_data=True)
        cal_grid = set(self._uniq(cal).tolist())
        assert cal_grid, "calibration produced no horizons"
        assert cal_grid <= auc_grid

    def test_honors_explicit_evaluate_grid(self, cal_learner):
        import copy
        lr = copy.deepcopy(cal_learner)
        lr.evaluate(eval_times=[1.0, 2.0], verbose=False)
        # The resolved calibration default follows the inherited explicit grid.
        assert np.array_equal(self._resolved(lr), [1.0, 2.0])


@pytest.mark.integration
class TestOverlayDefaultEvalTimesInRange:
    """Default ``eval_times`` must stay within the training range.

    The fixture's test cohort has follow-up beyond the model's training
    domain. With default ``eval_times`` the overlay used to ask the fitted
    survival step functions for out-of-domain horizons and crash with
    ``ValueError: x must be within [...]``. The horizons are now clipped to
    the training range.
    """

    def _train_t_max(self, learner):
        return float(learner._state.y_train["time"].max())

    def test_default_plot_path_returns_figure(self, cal_learner):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SuggestionWarning)
            fig = cal_learner.plot_calibration_overlay(models="coxph")
        assert isinstance(fig, go.Figure)
        # The predicted curve's horizons (the clip happens on the figure
        # path's own line) must stay inside the training domain.
        pred = [tr for tr in fig.data if tr.name == "Predicted (mean)"]
        assert pred, "expected a 'Predicted (mean)' trace"
        assert max(pred[0].x) <= self._train_t_max(cal_learner)

    def test_default_data_path_stays_in_range(self, cal_learner):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SuggestionWarning)
            data = cal_learner.plot_calibration_overlay(
                models="coxph", return_data=True,
            )
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        # Every default horizon is inside the fitted step functions' domain.
        assert data["eval_time"].max() <= self._train_t_max(cal_learner)

    def test_explicit_out_of_range_not_silently_clipped(self, cal_learner):
        # Clipping is scoped to the *default* grid only. An explicit
        # out-of-range horizon is honored verbatim, so it reaches the fitted
        # step functions and raises — it must NOT be silently dropped.
        beyond = self._train_t_max(cal_learner) + 5.0
        with pytest.raises(ValueError):
            cal_learner.plot_calibration_overlay(
                models="coxph", eval_times=[beyond], return_data=True,
            )


@pytest.mark.integration
class TestEvalTimesEdgeCases:
    def test_empty_list_raises_valueerror(self, cal_learner):
        with pytest.raises(ValueError, match="empty list"):
            cal_learner.plot_calibration(models="coxph", eval_times=[])

    def test_palette_accepted_for_multi_model(self, cal_learner):
        # A valid registered palette must thread through without error.
        fig = cal_learner.plot_calibration(models=["coxph", "coxnet"],
                                           palette="tableau10")
        assert isinstance(fig, go.Figure)
