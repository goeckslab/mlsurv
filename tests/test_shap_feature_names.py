"""Regression tests: SHAP must not strip column names.

The fitted RSF stores ``feature_names_in_`` because it was trained on a
DataFrame. Without the wrapper around ``model.predict``,
SHAP's PermutationExplainer perturbs the background as a numpy array and
passes plain ndarrays into the inner sksurv model — sklearn then emits
``UserWarning: X does not have valid feature names`` from
``_check_feature_names`` in ``sklearn/utils/validation.py``.

These tests assert that no such warning leaks out of the SHAP path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


_FEATURE_NAMES_MSG = "X does not have valid feature names"


def _has_shap():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def test_wrap_predict_with_columns_coerces_ndarray_to_dataframe():
    """The wrapper must restore feature names when SHAP passes a numpy array."""
    from mlsurv.evaluation._shap import _wrap_predict_with_columns

    captured = {}

    def fake_predict(X):
        captured["type"] = type(X).__name__
        captured["columns"] = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return np.zeros(len(X))

    columns = ["a", "b", "c"]
    wrapped = _wrap_predict_with_columns(fake_predict, columns)

    # ndarray input → wrapped should restore DataFrame with the right columns
    out = wrapped(np.zeros((5, 3)))
    assert captured["type"] == "DataFrame"
    assert captured["columns"] == columns
    assert out.shape == (5,)

    # DataFrame input → passes through unchanged (even when columns differ
    # from training names, so sklearn's downstream `_check_feature_names`
    # still fires on real mismatches — wrapper must not silently relabel).
    wrong_cols = ["x", "y", "z"]
    df = pd.DataFrame(np.ones((2, 3)), columns=wrong_cols)
    wrapped(df)
    assert captured["type"] == "DataFrame"
    assert captured["columns"] == wrong_cols


def test_wrap_predict_with_columns_rejects_width_mismatch():
    """ndarray with wrong width must surface via pandas, not be silently relabeled."""
    from mlsurv.evaluation._shap import _wrap_predict_with_columns

    wrapped = _wrap_predict_with_columns(lambda X: np.zeros(len(X)), ["a", "b", "c"])
    with pytest.raises(ValueError):
        wrapped(np.zeros((4, 2)))  # 2 columns, expected 3


@pytest.mark.shap
@pytest.mark.skipif(not _has_shap(), reason="shap required")
def test_rsf_permutation_shap_no_feature_names_warning(train_test_data):
    """RSF SHAP (permutation path) must not emit the sklearn warning."""
    from mlsurv.models import RSFModel
    from mlsurv.evaluation._shap import _compute_shap

    X_train, X_test, y_train, _y_test = train_test_data
    model = RSFModel(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _compute_shap(model, X_test, max_background=10, max_explain=5)

    leaked = [w for w in caught if _FEATURE_NAMES_MSG in str(w.message)]
    assert not leaked, (
        f"Unexpected sklearn feature-name warnings from RSF "
        f"PermutationExplainer path: {[str(w.message) for w in leaked]}"
    )


@pytest.mark.shap
@pytest.mark.skipif(not _has_shap(), reason="shap required")
def test_coxph_permutation_shap_no_feature_names_warning(train_test_data):
    """CoxPH SHAP (permutation path) must not emit the sklearn warning.

    A second model exercising the same single explainer, to confirm the
    predict wrapper guards the path regardless of model type.
    """
    from mlsurv.models import CoxPHModel
    from mlsurv.evaluation._shap import _compute_shap

    X_train, X_test, y_train, _y_test = train_test_data
    model = CoxPHModel()
    model.fit(X_train, y_train)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _compute_shap(model, X_test, max_background=10, max_explain=5)

    leaked = [w for w in caught if _FEATURE_NAMES_MSG in str(w.message)]
    assert not leaked, (
        f"Unexpected sklearn feature-name warnings from CoxPH "
        f"PermutationExplainer path: {[str(w.message) for w in leaked]}"
    )
