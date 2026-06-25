"""Contract tests for the shared data factory in ``tests/_helpers.py``.

These lock the non-obvious guarantees that the 18 migrated test files depend on
but do not assert themselves — chiefly the byte-for-byte reproduction of the
legacy ``RandomState`` generators. A future edit to ``_helpers.py`` that quietly
changed the RNG, draw order, or distribution would otherwise leave every
dependent green while silently shifting the data those regression tests were
tuned against.

Pure array construction only — no model training, so this stays in the fast tier.
"""

import numpy as np
import pytest

from tests._helpers import (
    make_feature_dataframe,
    make_random_survival_array,
    make_survival_array,
    make_synthetic_data,
    make_y,
)


class TestMakeY:
    def test_events_default_all_observed(self):
        y = make_y([1.0, 5.0, 10.0])
        assert y.dtype.names == ("event", "time")
        assert y["event"].all()
        assert list(y["time"]) == [1.0, 5.0, 10.0]

    def test_explicit_events(self):
        y = make_y([1.0, 2.0, 3.0], [True, False, True])
        assert list(y["event"]) == [True, False, True]

    def test_empty_input_returns_empty_array(self):
        # test_phase1_utils relies on make_y([]) yielding a 0-row array
        # (feeds auto_eval_times' "no observations" path).
        y = make_y([])
        assert len(y) == 0
        assert y.dtype.names == ("event", "time")

    def test_nonpositive_time_rejected_by_construction(self):
        # make_y delegates validation to to_structured_array.
        with pytest.raises(ValueError):
            make_y([0.0, 1.0, 2.0])


class TestMakeSurvivalArray:
    def test_events_first_equals_make_y_times_first(self):
        events = np.array([True, False, True])
        times = np.array([2.0, 4.0, 6.0])
        assert np.array_equal(
            make_survival_array(events, times), make_y(times, events)
        )


class TestMakeRandomSurvivalArrayReproduction:
    """The factory must reproduce each legacy RandomState generator exactly."""

    def test_binomial_exp5_matches_legacy(self):
        # Legacy test_inf_risk_scores / test_xgbaft generator.
        rng = np.random.RandomState(42)
        events = rng.binomial(1, 0.7, 13).astype(bool)
        times = rng.exponential(5.0, 13)
        got = make_random_survival_array(13)  # defaults: 0.7, 5.0, binomial, 42
        assert np.array_equal(got["event"], events)
        assert np.array_equal(got["time"], times)

    def test_uniform_exp10_matches_legacy_rand(self):
        # Legacy test_warning_quality generator (rng.rand < rate, exp(10)).
        rng = np.random.RandomState(42)
        events = rng.rand(13) < 0.5
        times = rng.exponential(10, 13)
        got = make_random_survival_array(
            13, event_rate=0.5, time_scale=10.0, event_dist="uniform"
        )
        assert np.array_equal(got["event"], events)
        assert np.array_equal(got["time"], times)

    def test_uniform_exp5_matches_legacy_random(self):
        # Legacy test_paired_comparison generator (rng.random < rate, exp(5)).
        rng = np.random.RandomState(42)
        events = rng.random(13) < 0.5
        times = rng.exponential(5.0, 13)
        got = make_random_survival_array(
            13, event_rate=0.5, time_scale=5.0, event_dist="uniform"
        )
        assert np.array_equal(got["event"], events)
        assert np.array_equal(got["time"], times)

    def test_invalid_event_dist_raises(self):
        with pytest.raises(ValueError, match="event_dist"):
            make_random_survival_array(5, event_dist="bogus")


class TestMakeRandomSurvivalArrayExternalRng:
    """test_subpopulation_analysis interleaves further draws on the same rng."""

    def test_matches_legacy_external_rng_generator(self):
        rng_a = np.random.RandomState(7)
        legacy = np.array(
            [
                (bool(e), float(t))
                for e, t in zip(
                    rng_a.binomial(1, 0.7, 20), rng_a.exponential(10, 20)
                )
            ],
            dtype=[("event", bool), ("time", float)],
        )
        rng_b = np.random.RandomState(7)
        got = make_random_survival_array(
            20, event_rate=0.7, time_scale=10.0, event_dist="binomial", rng=rng_b
        )
        assert np.array_equal(legacy, got)

    def test_advances_rng_state_by_exactly_two_draws(self):
        rng_b = np.random.RandomState(7)
        make_random_survival_array(20, event_dist="binomial", rng=rng_b)
        # A fresh rng advanced by one binomial(20) + one exponential(20) draw
        # must leave the same downstream stream.
        rng_ref = np.random.RandomState(7)
        rng_ref.binomial(1, 0.7, 20)
        rng_ref.exponential(5.0, 20)
        assert np.array_equal(rng_b.randn(5), rng_ref.randn(5))


class TestFeatureAndSyntheticFactories:
    def test_feature_dataframe_prefix_and_shape(self):
        X = make_feature_dataframe(10, 5, prefix="feat_")
        assert list(X.columns) == [f"feat_{i}" for i in range(5)]
        assert X.shape == (10, 5)

    def test_feature_dataframe_default_prefix(self):
        X = make_feature_dataframe(8, 3)
        assert list(X.columns) == ["f0", "f1", "f2"]

    def test_synthetic_data_shapes(self):
        X, y = make_synthetic_data(30, 4, seed=1)
        assert X.shape == (30, 4)
        assert len(y) == 30
        assert y.dtype.names == ("event", "time")
