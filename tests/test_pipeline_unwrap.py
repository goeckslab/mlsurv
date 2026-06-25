"""Tests for unwrap_pipeline() and propagate_feature_names() utilities."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlsurv.utils import propagate_feature_names, unwrap_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal estimator stub."""
    pass


class _FakeModelWithFeatures:
    """Estimator that already has feature_names_in_."""
    def __init__(self):
        self.feature_names_in_ = np.array(["existing_a", "existing_b"])


class _ScalerWithFeatures(StandardScaler):
    """Scaler stub that has feature_names_in_ set."""
    def __init__(self, names):
        super().__init__()
        self.feature_names_in_ = np.array(names)


# ---------------------------------------------------------------------------
# unwrap_pipeline
# ---------------------------------------------------------------------------

class TestUnwrapPipeline:
    """Tests for unwrap_pipeline()."""

    def test_pipeline_with_model_step(self):
        inner = _FakeModel()
        pipe = Pipeline([("scaler", StandardScaler()), ("model", inner)])
        assert unwrap_pipeline(pipe) is inner

    def test_pipeline_without_model_step(self):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", _FakeModel())])
        assert unwrap_pipeline(pipe) is pipe

    def test_non_pipeline(self):
        model = _FakeModel()
        assert unwrap_pipeline(model) is model

    def test_none_passthrough(self):
        # Edge case: None should be returned as-is (no crash)
        assert unwrap_pipeline(None) is None


# ---------------------------------------------------------------------------
# propagate_feature_names
# ---------------------------------------------------------------------------

class TestPropagateFeatureNames:
    """Tests for propagate_feature_names()."""

    def test_propagates_from_scaler_step(self):
        names = ["feat_a", "feat_b", "feat_c"]
        scaler = _ScalerWithFeatures(names)
        inner = _FakeModel()
        pipe = Pipeline([("scaler", scaler), ("model", inner)])

        propagate_feature_names(inner, pipe)

        np.testing.assert_array_equal(inner.feature_names_in_, names)

    def test_noop_when_already_present(self):
        inner = _FakeModelWithFeatures()
        original = inner.feature_names_in_.copy()
        scaler = _ScalerWithFeatures(["other_a", "other_b"])
        pipe = Pipeline([("scaler", scaler), ("model", inner)])

        propagate_feature_names(inner, pipe)

        np.testing.assert_array_equal(inner.feature_names_in_, original)

    def test_noop_for_non_pipeline(self):
        inner = _FakeModel()
        propagate_feature_names(inner, inner)
        assert not hasattr(inner, 'feature_names_in_')

    def test_noop_when_no_step_has_names(self):
        inner = _FakeModel()
        pipe = Pipeline([("scaler", StandardScaler()), ("model", inner)])

        propagate_feature_names(inner, pipe)

        assert not hasattr(inner, 'feature_names_in_')

    def test_picks_first_step_with_names(self):
        names_1 = ["first_a", "first_b"]
        names_2 = ["second_a", "second_b"]
        step1 = _ScalerWithFeatures(names_1)
        step2 = _ScalerWithFeatures(names_2)
        inner = _FakeModel()
        pipe = Pipeline([("s1", step1), ("s2", step2), ("model", inner)])

        propagate_feature_names(inner, pipe)

        np.testing.assert_array_equal(inner.feature_names_in_, names_1)
