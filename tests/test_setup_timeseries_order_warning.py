"""Tests for the timeseries row-order warning at setup.

``cv_strategy='timeseries'`` splits train/test with ``shuffle=False`` and
forward-chains the CV folds, which is only a valid past->future design if the
rows arrive pre-sorted chronologically. mlsurv neither sorts nor verifies the
order, so ``setup()`` emits a ``SuggestionWarning`` to surface the
assumption. These tests are fast tier (``setup()`` only, no training)."""

import warnings

import numpy as np
import pandas as pd

from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
from mlsurv.learner import SurvivalLearner
from mlsurv.utils import to_structured_array

_ORDER_SUBSTRING = "pre-sorted chronologically"


def _make_data(n_samples=80, n_features=5, event_rate=0.5, seed=42):
    """Create synthetic survival data."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    time = rng.exponential(10, n_samples).astype(np.float64)
    event = rng.binomial(1, event_rate, n_samples).astype(bool)
    y = to_structured_array(times=time, events=event)
    return X, y


def _order_warnings(caught):
    """Filter captured warnings down to the timeseries row-order warning."""
    return [
        w for w in caught
        if issubclass(w.category, SuggestionWarning)
        and _ORDER_SUBSTRING in str(w.message)
    ]


class TestTimeSeriesOrderWarningAtSetup:
    """SuggestionWarning fires from setup() for cv_strategy='timeseries'."""

    def test_timeseries_setup_emits_order_warning(self):
        """setup(cv_strategy='timeseries') emits exactly one order warning."""
        X, y = _make_data()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                cv_strategy='timeseries',
                stratify=None,
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        order_warns = _order_warnings(caught)
        assert len(order_warns) == 1, (
            f"Expected exactly one timeseries order warning, got: "
            f"{[str(w.message) for w in caught]}"
        )

    def test_warning_message_explains_assumption_and_consequence(self):
        """The message names the pre-sort assumption and the invalid result."""
        X, y = _make_data()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                cv_strategy='timeseries',
                stratify=None,
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        msg = str(_order_warnings(caught)[0].message)
        assert "timeseries" in msg
        assert _ORDER_SUBSTRING in msg
        assert "invalid" in msg

    def test_warning_uses_suggestion_category(self):
        """The warning is a SuggestionWarning (actionable precondition, not a
        detected methodological problem)."""
        X, y = _make_data()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                cv_strategy='timeseries',
                stratify=None,
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        order_warns = _order_warnings(caught)
        assert order_warns[0].category is SuggestionWarning

    def test_warning_is_not_methodological(self):
        """Guards against reverting the tier. The pre-sort reminder is a
        precondition the user can act on, not a defect mlsurv detected, so it
        must not be a MethodologicalWarning. The two tiers are sibling
        UserWarning subclasses (neither subclasses the other), so a revert is a
        real semantic change. This test filters on the message substring alone,
        not the tier, so a revert fails at the issubclass assertion below rather
        than silently dropping out of the tier-keyed _order_warnings filter."""
        X, y = _make_data()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                cv_strategy='timeseries',
                stratify=None,
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        pre_sort = [w for w in caught if _ORDER_SUBSTRING in str(w.message)]
        assert len(pre_sort) == 1
        assert not issubclass(pre_sort[0].category, MethodologicalWarning)
        assert pre_sort[0].category is SuggestionWarning

    def test_default_strategy_does_not_warn(self):
        """The default cv_strategy emits no timeseries order warning."""
        X, y = _make_data()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        assert _order_warnings(caught) == []

    def test_kfold_strategy_does_not_warn(self):
        """cv_strategy='kfold' emits no timeseries order warning."""
        X, y = _make_data()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                cv_strategy='kfold',
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        assert _order_warnings(caught) == []

    def test_warns_when_groups_also_supplied(self):
        """groups + timeseries falls through to the timeseries split (the
        group-aware holdout is skipped for timeseries), so the order warning
        must still fire exactly once. Pins the branch fall-through against a
        future reorder that could route timeseries into the group-aware split
        or suppress the warning."""
        X, y = _make_data()
        # One subject per pair of rows — arbitrary groups, deliberately not a
        # temporal ordering, which is exactly the misuse the warning flags.
        groups = np.repeat(np.arange(len(X) // 2 + 1), 2)[:len(X)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                cv_strategy='timeseries',
                groups=groups,
                stratify=None,
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        assert len(_order_warnings(caught)) == 1, (
            "Order warning must still fire when groups is supplied alongside "
            "cv_strategy='timeseries'"
        )
        # The group-aware holdout must not have fired for timeseries.
        assert learner._state.group_aware_holdout is False

    def test_warning_fires_once_per_setup(self):
        """Re-calling setup() with timeseries re-fires the warning each time."""
        X, y = _make_data()
        learner = SurvivalLearner(X, y)

        for _ in range(2):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                learner.setup(
                    cv_strategy='timeseries',
                    stratify=None,
                    random_state=42,
                    ph_test=False,
                    linearity_test=False,
                )
            assert len(_order_warnings(caught)) == 1, (
                "Each setup(cv_strategy='timeseries') call should warn once"
            )
