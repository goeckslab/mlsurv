"""Tests for the read-only ``ph_test_result`` / ``linearity_test_result``
properties and their ``df.attrs`` param-stamping.

These expose the live ``_state`` fields populated by ``setup()`` and refreshed
by ``ph_test()`` / ``linearity_test()``. All tests call only ``setup()`` and
the cheap ``ph_test()`` / ``linearity_test()`` diagnostics on small synthetic
data — no model training — so they stay in the fast tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner


def _make_data(n=150, p=4, seed=42):
    """Small constant-hazard data where PH/linearity hold (exponential)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)),
                     columns=[f"x{i}" for i in range(p)])
    lp = X["x0"] * 0.5 + X["x1"] * -0.3
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) < 0.7
    y = np.array([(bool(e), t) for e, t in zip(event, time)],
                 dtype=[("event", bool), ("time", float)])
    return X, y


@pytest.fixture
def data():
    return _make_data()


class TestPhTestResultProperty:
    def test_none_before_setup(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        assert learner.ph_test_result is None

    def test_populated_after_setup(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)  # ph_test=True is the default
        result = learner.ph_test_result
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # Plain read-only proxy to the state field.
        assert result is learner._state.ph_test_result

    def test_none_when_skipped(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(ph_test=False, verbose=False)
        assert learner.ph_test_result is None

    def test_refreshed_by_method_call(self, data):
        """ph_test() refreshes the property; the bundle snapshot is separate."""
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        setup_result = learner.ph_test_result

        fresh = learner.ph_test(time_transform="km", verbose=False)
        assert learner.ph_test_result is fresh
        assert learner.ph_test_result is not setup_result
        # The setup-time snapshot in the bundle stays untouched.
        assert learner.model_recommendations.ph_test_result is setup_result

    def test_read_only(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.raises(AttributeError):
            learner.ph_test_result = None

    def test_resetup_with_skip_clears_property(self, data):
        """Re-setup() resets the field — no stale result from a prior setup."""
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner.ph_test_result is not None
        learner.setup(ph_test=False, verbose=False)
        assert learner.ph_test_result is None

    def test_attrs_stamped(self, data):
        """Both store paths stamp df.attrs with the params used."""
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        attrs = learner.ph_test_result.attrs
        assert attrs["time_transform"] == "rank"
        assert attrs["source"] == "setup"

        learner.ph_test(time_transform="km", verbose=False)
        attrs = learner.ph_test_result.attrs
        assert attrs["time_transform"] == "km"
        assert attrs["source"] == "method"
        # The bundle snapshot keeps the setup-time stamp.
        snap_attrs = learner.model_recommendations.ph_test_result.attrs
        assert snap_attrs["time_transform"] == "rank"
        assert snap_attrs["source"] == "setup"


class TestLinearityTestResultProperty:
    def test_none_before_setup(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        assert learner.linearity_test_result is None

    def test_populated_after_setup(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)  # linearity_test=True is the default
        result = learner.linearity_test_result
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result is learner._state.linearity_test_result

    def test_none_when_skipped(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(linearity_test=False, verbose=False)
        assert learner.linearity_test_result is None

    def test_refreshed_by_method_call(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        setup_result = learner.linearity_test_result

        fresh = learner.linearity_test(n_knots=5, verbose=False)
        assert learner.linearity_test_result is fresh
        assert learner.linearity_test_result is not setup_result
        assert (learner.model_recommendations.linearity_test_result
                is setup_result)

    def test_read_only(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.raises(AttributeError):
            learner.linearity_test_result = None

    def test_resetup_with_skip_clears_property(self, data):
        """Re-setup() resets the field — no stale result from a prior setup."""
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner.linearity_test_result is not None
        learner.setup(linearity_test=False, verbose=False)
        assert learner.linearity_test_result is None

    def test_attrs_stamped(self, data):
        """Both store paths stamp df.attrs with the params used."""
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        attrs = learner.linearity_test_result.attrs
        assert attrs["n_knots"] == 4
        assert attrs["source"] == "setup"

        learner.linearity_test(n_knots=5, verbose=False)
        attrs = learner.linearity_test_result.attrs
        assert attrs["n_knots"] == 5
        assert attrs["source"] == "method"
        # The bundle snapshot keeps the setup-time stamp.
        snap_attrs = learner.model_recommendations.linearity_test_result.attrs
        assert snap_attrs["n_knots"] == 4
        assert snap_attrs["source"] == "setup"
