"""Tests for EPV warning deduplication across setup/tuning."""

import warnings
from unittest.mock import MagicMock

import pandas as pd

from mlsurv.learner import SurvivalLearner
from mlsurv.learner._tuning import TuningMixin
from tests.test_small_sample_tuning_warning import _make_data


def _make_mock_learner(n_events, n_features, epv_warned_at_setup=False):
    """Create a mock learner with controlled EPV and flag state."""
    learner = MagicMock()
    learner.y_train = {'event': pd.Series([1] * n_events + [0] * 10)}
    learner.X_train = MagicMock()
    learner.X_train.columns = list(range(n_features))
    learner._state.epv_warned_at_setup = epv_warned_at_setup
    return learner


class TestEpvDedup:
    """EPV warning fires once across setup + tuning lifecycle."""

    def test_setup_warns_tuning_skips(self):
        """When setup warns (EPV < 10), tuning's _warn_epv_if_low is silent."""
        X, y = _make_data(60, n_features=30)
        learner = SurvivalLearner(X, y)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.setup(random_state=42)

        setup_epv = [x for x in w if "EPV" in str(x.message)]
        assert len(setup_epv) == 1, "Setup should fire the EPV warning"
        assert learner._state.epv_warned_at_setup is True

        # Now tuning's EPV check should be silent
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TuningMixin._warn_epv_if_low(learner)

        tuning_epv = [x for x in w if "EPV" in str(x.message)]
        assert len(tuning_epv) == 0, "Tuning should skip EPV warning after setup warned"

    def test_no_setup_warning_tuning_still_warns(self):
        """When setup doesn't warn (EPV >= 10), tuning can still warn (EPV < 2)."""
        # Mock a learner where setup did NOT warn (flag is False)
        # but EPV is very low (simulating e.g. feature selection after setup)
        learner = _make_mock_learner(
            n_events=3, n_features=5, epv_warned_at_setup=False,
        )  # EPV = 0.6

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TuningMixin._warn_epv_if_low(learner)

        epv_warnings = [x for x in w if "EPV" in str(x.message)]
        assert len(epv_warnings) == 1, "Tuning should warn when setup didn't"

    def test_resetup_clears_flag(self):
        """Re-calling setup() on the same learner resets the EPV flag."""
        # First setup with low EPV
        X, y = _make_data(60, n_features=30)
        learner = SurvivalLearner(X, y)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            learner.setup(random_state=42)

        assert learner._state.epv_warned_at_setup is True

        # Re-setup the SAME learner with high-EPV data
        X_few, y_few = _make_data(200, n_features=3)
        learner._state.X = X_few
        learner._state.y = y_few
        learner._state.presplit = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.setup(random_state=42)

        epv_warnings = [x for x in w if "EPV" in str(x.message)]
        assert len(epv_warnings) == 0, "High-EPV re-setup should not warn"
        assert learner._state.epv_warned_at_setup is False
