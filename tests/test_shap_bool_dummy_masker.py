"""Regression: SHAP ``explain()`` with bool one-hot (``get_dummies``) columns.

``pd.get_dummies`` returns bool columns (pandas >= 2.0). A DataFrame mixing
bool and float columns has an *object* ``.values`` block, and shap's
``Independent`` masker then calls ``np.isfinite`` on it, raising
``TypeError: ufunc 'isfinite' not supported``. That crashed vignette 07's
``learner.explain(patient='high_risk')`` after the SHAP refactor.

``_run_explainer`` now casts the masker background and the explained frame to
float64 via ``_numeric_frame_for_shap``; bool one-hot is 0/1 so predictions and
SHAP values are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsurv.evaluation._shap import _numeric_frame_for_shap


def _has_shap():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


requires_shap = pytest.mark.skipif(not _has_shap(), reason="shap required")


class TestNumericFrameForShap:
    """The helper that makes a SHAP frame masker-safe."""

    def test_bool_and_float_mix_casts_to_float64(self):
        """A bool/float mix (object .values) becomes a clean float64 frame."""
        df = pd.DataFrame({
            "TMB": [1.0, 2.0, 3.0],
            "ct_A": pd.Series([True, False, True]),
            "ct_B": pd.Series([False, True, False]),
        })
        assert df.values.dtype == object  # the exact trigger condition

        out = _numeric_frame_for_shap(df, "background")

        assert (out.dtypes == np.float64).all()
        assert out.values.dtype == np.float64
        # Values preserved: True -> 1.0, False -> 0.0.
        assert out["ct_A"].tolist() == [1.0, 0.0, 1.0]
        assert out["TMB"].tolist() == [1.0, 2.0, 3.0]

    def test_all_float_is_unchanged(self):
        """Already-numeric frames pass through with identical values."""
        df = pd.DataFrame({"a": [0.5, 1.5], "b": [2.0, 3.0]})
        out = _numeric_frame_for_shap(df, "explained set")
        pd.testing.assert_frame_equal(out, df.astype(np.float64))

    def test_string_column_raises_clear_error(self):
        """A raw string category gets a clear error, not the cryptic isfinite one."""
        df = pd.DataFrame({
            "TMB": [1.0, 2.0],
            "CancerType": ["Melanoma", "Colorectal"],
        })
        with pytest.raises(TypeError, match="non-numeric column"):
            _numeric_frame_for_shap(df, "background")


def _trained_bool_dummy_learner():
    """Train a coxph learner on a get_dummies (bool) feature matrix.

    The feature frame mixes float clinical columns with bool one-hot columns,
    so ``X.values`` is an object block (the exact pre-fix crash trigger).
    """
    from mlsurv import SurvivalLearner
    from mlsurv.utils import to_structured_array

    rng = np.random.RandomState(0)
    n = 80
    X = pd.DataFrame({
        "tmb": rng.rand(n),
        "age": rng.rand(n) * 40 + 40,
        "nlr": rng.rand(n) * 5,
    })
    dummies = pd.get_dummies(rng.choice(["A", "B", "C"], n), prefix="ct")
    X = pd.concat([X, dummies], axis=1)
    # Precondition: the bool dummies make this an object-block frame, which is
    # the exact crash trigger (object .values -> masker np.isfinite).
    assert any(str(t) == "bool" for t in X.dtypes)
    assert X.values.dtype == object

    time = rng.rand(n) * 100 + 1
    event = rng.randint(0, 2, n)
    y = to_structured_array(pd.Series(time), pd.Series(event))

    X_tr, X_te = X.iloc[:60], X.iloc[60:]
    y_tr, y_te = y[:60], y[60:]

    learner = SurvivalLearner(X_train=X_tr, X_test=X_te,
                              y_train=y_tr, y_test=y_te)
    learner.setup(cv_folds=3, random_state=42, n_jobs=1)
    learner.train("coxph", tune=False)
    return learner, X.shape[1]


@pytest.mark.shap
@pytest.mark.integration
class TestExplainWithBoolDummies:
    """Both SHAP seams on a get_dummies feature matrix: patient and cohort."""

    @requires_shap
    def test_explain_runs_and_is_additive(self):
        """Patient path: explain() no longer crashes and stays additive."""
        learner, n_features = _trained_bool_dummy_learner()

        expl = learner.explain(patient="high_risk", model="coxph", plot=False)
        sv = expl.shap_values

        assert len(sv) == n_features
        assert np.isfinite(sv["shap_value"]).all()
        # base + sum(shap) reconstructs the model's risk score exactly.
        risk = sv["risk_score"].iloc[0]
        recon = expl.base_value + sv["shap_value"].sum()
        assert abs(risk - recon) < 1e-6

    @requires_shap
    def test_cohort_shap_runs_and_is_finite(self):
        """Cohort path: _compute_shap explains the multi-row X_test cohort.

        Routes through the same _run_explainer seam but with a multi-row
        explained frame (object .values), so it exercises the explained-frame
        coercion separately from the single-row patient path.
        """
        learner, _ = _trained_bool_dummy_learner()
        learner.evaluate("coxph", n_importance_repeats=0)

        cohort_shap = learner._shap(learner.models["coxph"], max_explain=20,
                                    max_background=20, show_progress=False)

        assert len(cohort_shap) > 0
        assert np.isfinite(cohort_shap["shap_value"]).all()
