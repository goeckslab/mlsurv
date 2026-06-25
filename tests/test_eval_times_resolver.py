"""Tests for the canonical eval_times resolver and its consumers.

The resolver (`resolve_state_eval_times`) is the single source of truth for the
time grid used by AUC(t), Brier(t), IBS, and calibration. Every consumer
(evaluate/bootstrap/validate/calibrate/benchmark) routes through it so the
point estimates and the bootstrap CI bands land on one `eval_time` set, and
`setup(eval_times=...)` seeds that grid.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import SuggestionWarning
from mlsurv.learner._evaluation_helpers import (
    _inherited_grid_risky,
    resolve_state_eval_times,
)
from mlsurv.utils import auto_clip_union_eval_times, resolve_display_eval_times


class _FakeState:
    """Minimal stand-in carrying just the user_eval_times field."""

    def __init__(self, user_eval_times=None):
        self.user_eval_times = user_eval_times


def _y(times, events):
    """Build a structured survival array."""
    return np.array(
        list(zip(events, times)), dtype=[("event", bool), ("time", float)],
    )


class TestBootstrapPublicParamName:
    """The public bootstrap() time-resolved flag is include_time_resolved."""

    def test_old_include_auc_kwarg_rejected(self):
        """The rename dropped the old public name with no back-compat alias.

        bootstrap() has an explicit signature (no **kwargs), so the old
        include_auc keyword is rejected at call binding before any setup or
        training runs. This guards against a future **kwargs passthrough
        silently resurrecting the old name.
        """
        X = pd.DataFrame({"f0": [0.1, 0.2, 0.3, 0.4]})
        y = _y([1.0, 2.0, 3.0, 4.0], [True, False, True, False])
        learner = SurvivalLearner(X, y)
        with pytest.raises(TypeError, match="include_auc"):
            learner.bootstrap(include_auc=True)


class TestResolveStateEvalTimes:
    """Unit tests for the resolver itself (no model training)."""

    def test_explicit_array_stored_and_returned_as_float(self):
        state = _FakeState()
        out = resolve_state_eval_times(state, [1, 2, 3])
        assert np.array_equal(out, [1.0, 2.0, 3.0])
        assert out.dtype == float
        # Persisted for later consumers to inherit.
        assert np.array_equal(state.user_eval_times, [1.0, 2.0, 3.0])

    def test_none_returns_inherited_grid(self):
        state = _FakeState(user_eval_times=np.array([4.0, 5.0]))
        out = resolve_state_eval_times(state, None)
        assert np.array_equal(out, [4.0, 5.0])

    def test_none_with_no_inherited_returns_none(self):
        state = _FakeState()
        assert resolve_state_eval_times(state, None) is None

    def test_auto_string_clears_inherited(self):
        state = _FakeState(user_eval_times=np.array([4.0, 5.0]))
        out = resolve_state_eval_times(state, "auto")
        assert out is None
        assert state.user_eval_times is None

    def test_scalar_raises_type_error(self):
        state = _FakeState()
        with pytest.raises(TypeError):
            resolve_state_eval_times(state, 5)

    def test_two_dimensional_raises_value_error(self):
        state = _FakeState()
        with pytest.raises(ValueError):
            resolve_state_eval_times(state, [[1, 2], [3, 4]])

    def test_persist_false_makes_no_state_writes(self):
        # Explicit array: validated and returned, but not stored.
        state = _FakeState()
        out = resolve_state_eval_times(state, [5, 6], persist=False)
        assert np.array_equal(out, [5.0, 6.0])
        assert state.user_eval_times is None

        # "auto" with persist=False must not clear an existing grid.
        state = _FakeState(user_eval_times=np.array([7.0]))
        out = resolve_state_eval_times(state, "auto", persist=False)
        assert out is None
        assert np.array_equal(state.user_eval_times, [7.0])

    @pytest.mark.parametrize("bad", [[np.nan, 5.0], [np.inf, 5.0], [-1.0, 2.0], [0.0, 2.0]])
    def test_nonfinite_or_nonpositive_raises(self, bad):
        # NaN/Inf/negative/zero timepoints must raise rather than slip through
        # to clip_eval_times, which would drop them silently.
        state = _FakeState()
        with pytest.raises(ValueError, match="positive and finite"):
            resolve_state_eval_times(state, bad)

    def test_empty_array_passes_through(self):
        # Empty grids are vacuously finite/positive; consumers keep their own
        # size checks (e.g. recalibrate's "non-empty"), so the resolver lets
        # an empty array through unchanged.
        state = _FakeState()
        out = resolve_state_eval_times(state, [])
        assert out.size == 0

    def test_inherited_grid_risk_warns_only_when_misaligned(self):
        # custom_data + inherited grid that the cohort's events overshoot → warn.
        grid = np.array([1.0, 2.0, 3.0])
        risky_y = _y([5, 6, 7, 8], [True, True, True, True])  # all events past 3
        state = _FakeState(user_eval_times=grid)
        with pytest.warns(SuggestionWarning, match="inherited eval_times"):
            out = resolve_state_eval_times(
                state, None, custom_data=True, y_for_risk=risky_y,
            )
        assert np.array_equal(out, grid)

        # Events within the grid → no warning (guards the false-positive direction).
        safe_y = _y([1, 2, 3], [True, True, True])
        state = _FakeState(user_eval_times=grid)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SuggestionWarning)
            resolve_state_eval_times(
                state, None, custom_data=True, y_for_risk=safe_y,
            )

    def test_inherited_grid_risk_suppressed_without_custom_data(self):
        # The common bootstrap-after-evaluate path (custom_data=False) never warns.
        grid = np.array([1.0, 2.0, 3.0])
        risky_y = _y([5, 6, 7, 8], [True, True, True, True])
        state = _FakeState(user_eval_times=grid)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SuggestionWarning)
            resolve_state_eval_times(
                state, None, custom_data=False, y_for_risk=risky_y,
            )


class TestInheritedGridRisky:
    """Direct coverage of the _inherited_grid_risky decision and its fallbacks."""

    def test_events_beyond_grid_is_risky(self):
        assert _inherited_grid_risky([1, 2, 3], _y([5, 6, 7], [True, True, True]))

    def test_events_within_grid_not_risky(self):
        assert not _inherited_grid_risky([1, 2, 3], _y([1, 2, 3], [True, True, True]))

    def test_none_y_is_risky(self):
        # Can't assess risk without times → warn (return True).
        assert _inherited_grid_risky([1, 2, 3], None)

    def test_no_events_not_risky(self):
        # All censored → no events to overshoot the grid.
        assert not _inherited_grid_risky([1, 2, 3], _y([5, 6, 7], [False, False, False]))


class TestResolveDisplayEvalTimes:
    """Unit tests for the viz-facing materializer (read-only, never persists)."""

    def test_inherited_grid_used_as_is(self):
        state = _FakeState(user_eval_times=np.array([2.0, 4.0, 6.0]))
        y = _y([1, 2, 3, 4, 5, 6, 7, 8], [True] * 8)
        out = resolve_display_eval_times(state, y, y)
        assert np.array_equal(sorted(out), [2.0, 4.0, 6.0])
        # Read-only: the inherited grid is not rewritten.
        assert np.array_equal(state.user_eval_times, [2.0, 4.0, 6.0])

    def test_inherited_grid_unions_report_horizons(self):
        # The realized auc_t grid for an explicit evaluate() grid is
        # union(grid, add_eval_times); the display resolver must match it, not
        # return the bare inherited grid (regression guard).
        state = _FakeState(user_eval_times=np.array([2.0, 4.0]))
        state.user_extra_eval_times = np.array([6.0])
        y = _y([1, 2, 3, 4, 5, 6, 7, 8], [True] * 8)
        out = resolve_display_eval_times(state, y, y)
        assert set(np.round(out, 6)) == {2.0, 4.0, 6.0}

    def test_auto_path_materializes_grid(self):
        state = _FakeState()  # no inherited grid
        y = _y(list(range(1, 21)), [True] * 20)
        out = resolve_display_eval_times(state, y, y)
        assert len(out) > 0
        assert np.all(np.isfinite(out))
        # Never persists the auto grid (keeps user_eval_times = "intent").
        assert state.user_eval_times is None

    def test_report_horizons_unioned_into_auto(self):
        state = _FakeState()
        state.user_extra_eval_times = np.array([5.0])
        y = _y(list(range(1, 21)), [True] * 20)
        out = resolve_display_eval_times(state, y, y)
        assert 5.0 in set(np.round(out, 6))

    def test_no_grid_derivable_raises(self):
        # All-censored, single time: auto_eval_times cannot derive a grid.
        state = _FakeState()
        y = _y([3.0, 3.0, 3.0], [False, False, False])
        with pytest.raises(ValueError):
            resolve_display_eval_times(state, y, y)


class TestAutoClipUnionEvalTimes:
    """Unit tests for the shared auto -> clip -> union materializer."""

    def test_auto_grid_in_training_window(self):
        y = _y(list(range(1, 21)), [True] * 20)
        out = auto_clip_union_eval_times(y, y, None)
        assert out is not None and len(out) > 0
        assert out.min() >= 1.0 and out.max() < 20.0  # clipped to [tmin, tmax)

    def test_unions_in_window_extras(self):
        y = _y(list(range(1, 21)), [True] * 20)
        out = auto_clip_union_eval_times(y, y, np.array([7.0]))
        assert 7.0 in set(np.round(out, 6))

    def test_returns_none_when_no_grid(self):
        # Degenerate (single repeated time) -> auto_eval_times raises -> None.
        y = _y([5.0, 5.0, 5.0], [True, True, True])
        assert auto_clip_union_eval_times(y, y, None) is None


def _grid(auc_t):
    """Sorted unique eval_time values from an auc_t DataFrame."""
    return np.sort(np.unique(auc_t["eval_time"].to_numpy()))


@pytest.mark.integration
class TestConsumersHonorGrid:
    """End-to-end: every consumer shares one eval_time grid by construction."""

    def _learner(self, small_data, **setup_kwargs):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, **setup_kwargs)
        learner.train("coxph")
        return learner

    def test_bootstrap_inherits_evaluate_grid(self, small_data):
        """The reported bug: bootstrap() must reuse evaluate()'s explicit grid."""
        learner = self._learner(small_data)
        grid = [1.0, 2.0, 3.0]
        learner.evaluate(eval_times=grid, verbose=False)
        result = learner.bootstrap(
            n_bootstrap=2, include_time_resolved=True, verbose=False, show_progress=False,
        )
        # Persisted grid carried into the bootstrap run config...
        assert np.array_equal(result.run_config["eval_times"], grid)
        # ...and the point curve and the band land on the same eval_time set.
        assert np.array_equal(
            _grid(learner.evaluation_result.auc_t), _grid(result.auc_t),
        )

    def test_setup_seeds_grid(self, small_data):
        """setup(eval_times=...) seeds the grid evaluate/bootstrap inherit."""
        grid = [1.0, 2.0, 3.0]
        learner = self._learner(small_data, eval_times=grid)
        assert np.array_equal(learner._state.user_eval_times, grid)

        learner.evaluate(verbose=False)
        result = learner.bootstrap(
            n_bootstrap=2, include_time_resolved=True, verbose=False, show_progress=False,
        )
        # The seeded grid flows through to bootstrap's run config, and the
        # point and band grids agree (both clip to the train range identically).
        assert np.array_equal(result.run_config["eval_times"], grid)
        assert np.array_equal(
            _grid(learner.evaluation_result.auc_t), _grid(result.auc_t),
        )

    def test_explicit_bootstrap_overrides_inherited(self, small_data):
        """An explicit bootstrap grid supersedes — and re-persists over — the
        grid evaluate() stored."""
        learner = self._learner(small_data)
        learner.evaluate(eval_times=[1.0, 2.0, 3.0], verbose=False)
        override = [1.5, 2.5]
        result = learner.bootstrap(
            eval_times=override, n_bootstrap=2, include_time_resolved=True,
            verbose=False, show_progress=False,
        )
        assert np.array_equal(result.run_config["eval_times"], override)
        assert np.array_equal(learner._state.user_eval_times, override)

    def test_bootstrap_auto_clears_inherited(self, small_data):
        """bootstrap(eval_times='auto') drops the inherited grid and recomputes."""
        learner = self._learner(small_data)
        inherited = [1.0, 2.0, 3.0]
        learner.evaluate(eval_times=inherited, verbose=False)
        result = learner.bootstrap(
            eval_times="auto", n_bootstrap=2, include_time_resolved=True,
            verbose=False, show_progress=False,
        )
        assert learner._state.user_eval_times is None
        # The recomputed band matches a fresh auto evaluate(), not the inherited grid.
        learner.evaluate(eval_times="auto", verbose=False)
        assert np.array_equal(
            _grid(learner.evaluation_result.auc_t), _grid(result.auc_t),
        )
        assert not np.array_equal(_grid(result.auc_t), np.sort(inherited))

    def test_default_auto_grids_match(self, small_data):
        """Behavior-preserving: with no explicit grid, evaluate() and bootstrap()
        still land on identical auto-computed grids."""
        learner = self._learner(small_data)
        learner.evaluate(verbose=False)
        result = learner.bootstrap(
            n_bootstrap=2, include_time_resolved=True, verbose=False, show_progress=False,
        )
        assert np.array_equal(
            _grid(learner.evaluation_result.auc_t), _grid(result.auc_t),
        )

    def test_calibrate_inherits_grid_without_clobbering_it(self, small_data):
        """Attaching a calibrator defaults its horizons to the persisted grid
        and must NOT overwrite user_eval_times (it routes through the resolver
        with persist=False)."""
        learner = self._learner(small_data)
        grid = [1.0, 2.0, 3.0]
        learner.evaluate(eval_times=grid, verbose=False)
        learner._attach_calibrators(learner.models["coxph"], "isotonic")
        cal = learner.models["coxph"].calibrator
        assert np.array_equal(np.sort(cal.horizons), grid)
        # The grid evaluate() persisted survives calibration untouched.
        assert np.array_equal(learner._state.user_eval_times, grid)

    def test_validate_inherits_grid_and_warns_on_mismatch(self, small_data):
        """validate() runs evaluate() with custom data, so the inherited grid is
        reused and the misalignment warning fires when the external cohort's
        events overshoot it."""
        X, _ = small_data
        learner = self._learner(small_data)
        learner.evaluate(eval_times=[1.0, 2.0, 3.0], verbose=False)
        # External cohort with events well beyond the inherited grid's max.
        y_val = _y(np.full(len(X), 6.0), [True] * len(X))
        with pytest.warns(SuggestionWarning, match="inherited eval_times"):
            learner.validate(X, y_val, models="coxph", verbose=False)

    def test_benchmark_inherits_and_persists_grid(self, small_data):
        """benchmark() inherits the persisted grid, and an explicit grid passed
        to benchmark persists for later consumers."""
        X, _ = small_data
        learner = self._learner(small_data)
        learner.evaluate(eval_times=[1.0, 2.0, 3.0], verbose=False)
        # No eval_times → inherits the persisted grid for its model evaluation.
        result = learner.benchmark(
            references={"ext": X["age"].values}, bootstrap=False, verbose=False,
        )
        assert np.array_equal(_grid(result.evaluation.auc_t), [1.0, 2.0, 3.0])
        # Explicit grid passed to benchmark persists (unified resolver, persist=True).
        learner.benchmark(
            references={"ext": X["age"].values}, eval_times=[2.0, 3.0],
            bootstrap=False, verbose=False,
        )
        assert np.array_equal(learner._state.user_eval_times, [2.0, 3.0])

    def test_resetup_clears_seeded_grid(self, small_data):
        """Re-running setup() without eval_times resets a previously seeded grid."""
        learner = self._learner(small_data, eval_times=[1.0, 2.0, 3.0])
        assert np.array_equal(learner._state.user_eval_times, [1.0, 2.0, 3.0])
        learner.setup(random_state=42, verbose=False)
        assert learner._state.user_eval_times is None
