"""Tests for mlsurv.viz.shap — silent fallback warnings."""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import InternalWarning
from mlsurv.viz.shap import _get_feature_values


class TestGetFeatureValuesWarning:
    """_get_feature_values should warn when feature values are unavailable."""

    def _make_feat_shap(self, n=5):
        return pd.DataFrame({
            "sample": range(n),
            "shap_value": np.random.default_rng(0).standard_normal(n),
        })

    def test_warns_when_no_feature_value_column_and_feature_missing_from_X(self):
        feat_shap = self._make_feat_shap()
        X = pd.DataFrame({"other_feature": [1, 2, 3, 4, 5]})
        with pytest.warns(InternalWarning, match="Feature values unavailable"):
            result = _get_feature_values(feat_shap, "missing_feat", X, "sample")
        assert len(result) == len(feat_shap)
        assert np.all(np.isnan(result))

    def test_warns_when_X_is_none(self):
        feat_shap = self._make_feat_shap()
        with pytest.warns(InternalWarning, match="Color encoding will be uniform"):
            result = _get_feature_values(feat_shap, "any_feat", None, "sample")
        assert np.all(np.isnan(result))

    def test_no_warning_when_feature_value_column_present(self):
        feat_shap = pd.DataFrame({
            "sample": [0, 1, 2],
            "shap_value": [0.1, 0.2, 0.3],
            "feature_value": [10, 20, 30],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _get_feature_values(feat_shap, "feat", None, "sample")
        np.testing.assert_array_equal(result, [10, 20, 30])

    def test_no_warning_when_feature_found_in_X(self):
        feat_shap = pd.DataFrame({
            "sample": [0, 1, 2],
            "shap_value": [0.1, 0.2, 0.3],
        })
        X = pd.DataFrame({"my_feat": [10, 20, 30]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _get_feature_values(feat_shap, "my_feat", X, "sample")
        np.testing.assert_array_equal(result, [10, 20, 30])
