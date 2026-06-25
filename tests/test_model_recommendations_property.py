"""Tests for the read-only ``model_recommendations`` property.

The property exposes the setup-time ``RecommendedModels`` bundle, which now
also carries the full PH/linearity test DataFrames as a setup snapshot. These
tests only call ``setup()`` on small synthetic data (no training), so they
stay in the fast tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner, recommend_models
from mlsurv.guidance import RecommendedModels


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


class TestModelRecommendationsProperty:
    """Availability and contents of ``learner.model_recommendations``."""

    def test_none_before_setup(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        assert learner.model_recommendations is None

    def test_populated_after_setup(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        rec = learner.model_recommendations
        assert isinstance(rec, RecommendedModels)
        # Property is a plain read-only proxy to the state field.
        assert rec is learner._state.model_recommendations
        # Every model lands in exactly one tier (the specific tier depends on
        # EPV; small data may leave ``recommended`` empty), and each has a
        # per-model reason string.
        n_tiered = (len(rec.recommended) + len(rec.cautionary)
                    + len(rec.not_recommended))
        assert n_tiered > 0
        assert rec.details
        # Dataset stats live on the bundle (explicit > 0, not truthiness).
        assert rec.n_samples > 0
        assert rec.n_events > 0
        assert rec.epv is not None

    def test_read_only(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.raises(AttributeError):
            learner.model_recommendations = None

    def test_dataframes_present_when_tests_run(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(ph_test=True, linearity_test=True, verbose=False)
        rec = learner.model_recommendations
        assert isinstance(rec.ph_test_result, pd.DataFrame)
        assert not rec.ph_test_result.empty
        assert isinstance(rec.linearity_test_result, pd.DataFrame)
        assert not rec.linearity_test_result.empty
        # Both DataFrames share the schema produced by ph_test() /
        # linearity_test().
        expected_cols = {"feature", "test_statistic", "p_value",
                         "p_adjusted", "significant"}
        assert expected_cols <= set(rec.ph_test_result.columns)
        assert expected_cols <= set(rec.linearity_test_result.columns)

    def test_dataframes_none_when_tests_skipped(self, data):
        X, y = data
        learner = SurvivalLearner(X, y)
        # The model_recommendations diagnostic always runs (gate=True), so the
        # bundle exists even when both assumption tests are turned off.
        learner.setup(ph_test=False, linearity_test=False, verbose=False)
        rec = learner.model_recommendations
        assert isinstance(rec, RecommendedModels)
        assert rec.ph_test_result is None
        assert rec.linearity_test_result is None
        # Tiers are still computed (from EPV / sample size).
        assert rec.recommended or rec.cautionary or rec.not_recommended

    def test_ph_bundle_is_setup_snapshot(self, data):
        """A later ph_test() refreshes _state only, not the bundle."""
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(ph_test=True, verbose=False)
        snapshot = learner.model_recommendations.ph_test_result
        assert snapshot is not None

        fresh = learner.ph_test(time_transform="km", verbose=False)
        # Live field is the freshly recomputed DataFrame...
        assert learner._state.ph_test_result is fresh
        # ...but the bundle keeps the setup-time snapshot, untouched.
        assert learner.model_recommendations.ph_test_result is snapshot

    def test_linearity_bundle_is_setup_snapshot(self, data):
        """Same snapshot semantics for the linearity test.

        Guards against an asymmetric-refresh bug that wired the linearity
        snapshot to the live ``_state`` field while leaving PH correct.
        """
        X, y = data
        learner = SurvivalLearner(X, y)
        learner.setup(linearity_test=True, verbose=False)
        snapshot = learner.model_recommendations.linearity_test_result
        assert snapshot is not None

        fresh = learner.linearity_test(n_knots=5, verbose=False)
        assert learner._state.linearity_test_result is fresh
        assert learner.model_recommendations.linearity_test_result is snapshot


class TestStandaloneRecommendModels:
    """Any standalone recommend_models() call leaves the DataFrame fields
    None — only setup() attaches them."""

    def test_dataframes_none_from_data(self, data):
        X, y = data
        rec = recommend_models(X, y, run_assumption_tests=False, verbose=False)
        assert rec.ph_test_result is None
        assert rec.linearity_test_result is None

    def test_dataframes_none_on_default_path(self, data):
        """Default run_assumption_tests=True runs the tests internally but
        still never attaches the DataFrames — only setup() does."""
        X, y = data
        rec = recommend_models(X, y, verbose=False)
        assert rec.ph_test_result is None
        assert rec.linearity_test_result is None

    def test_dataframes_none_from_numbers(self):
        rec = recommend_models(n_samples=500, n_events=300, n_features=10,
                               verbose=False)
        assert rec.ph_test_result is None
        assert rec.linearity_test_result is None
