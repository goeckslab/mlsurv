"""Tests for cohort SHAP using a shared, summarized ``X_train`` background.

Covers:
- ``_compute_shap`` decouples the explained set (``X_test``) from the SHAP
  background (``X_background``, typically ``X_train``); the explained rows still
  come from ``X_test`` while the background is whatever the caller passes,
- the background falls back to ``X_test`` when no separate background is given,
- the model-agnostic explainer summarizes its background from ``X_background``,
- both cohort call sites (``evaluate(include_shap=True)`` and
  ``importance(method='shap')``) forward ``X_train`` as the background.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv.evaluation import evaluate
from mlsurv.evaluation._shap import _compute_shap


def _has_shap():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


requires_shap = pytest.mark.skipif(not _has_shap(), reason="shap required")


@pytest.mark.shap
class TestComputeShapDecouplesBackground:
    """``_compute_shap`` explains X_test but references a separate background."""

    @requires_shap
    def test_run_explainer_receives_xtrain_background_and_xtest_explained(
        self, train_test_data
    ):
        """The explainer is handed X_background (train) and X_explain (from test)."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        captured = {}

        def _fake_run_explainer(model, X_background, X_explain, max_background,
                                random_state=None):
            captured['X_background'] = X_background
            captured['X_explain'] = X_explain
            return np.zeros((len(X_explain), X_explain.shape[1])), 0.0

        with mock.patch(
            'mlsurv.evaluation._shap._run_explainer',
            side_effect=_fake_run_explainer,
        ):
            _compute_shap(
                model, X_test, X_background=X_train,
                max_explain=len(X_test), random_state=0,
            )

        # Background is the train cohort; explained set is the test cohort.
        pd.testing.assert_frame_equal(captured['X_background'], X_train)
        pd.testing.assert_frame_equal(captured['X_explain'], X_test)

    @requires_shap
    def test_feature_values_come_from_xtest_not_background(self, train_test_data):
        """Returned feature_values reflect X_test even with a train background."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        # max_explain >= len(X_test) → every test row is explained, no subsample.
        result = _compute_shap(
            model, X_test, X_background=X_train,
            max_explain=len(X_test), random_state=0,
        )

        for feat in X_test.columns:
            got = sorted(result.loc[result['feature'] == feat, 'feature_value'])
            assert got == sorted(X_test[feat].tolist()), (
                f"feature_value for {feat} should come from X_test"
            )

    @requires_shap
    def test_background_defaults_to_xtest_when_omitted(self, train_test_data):
        """No X_background → background falls back to X_test (old behavior)."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        captured = {}

        def _fake_run_explainer(model, X_background, X_explain, max_background,
                                random_state=None):
            captured['X_background'] = X_background
            return np.zeros((len(X_explain), X_explain.shape[1])), 0.0

        with mock.patch(
            'mlsurv.evaluation._shap._run_explainer',
            side_effect=_fake_run_explainer,
        ):
            _compute_shap(model, X_test, max_explain=len(X_test), random_state=0)

        pd.testing.assert_frame_equal(captured['X_background'], X_test)

    @requires_shap
    def test_empty_background_raises(self, train_test_data):
        """An empty X_background fails loudly rather than deep inside shap."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        with pytest.raises(ValueError, match="empty"):
            _compute_shap(model, X_test, X_background=X_train.iloc[:0],
                          random_state=0)

    @requires_shap
    def test_mismatched_background_columns_raise(self, train_test_data):
        """A background whose columns differ from X_test raises a clear error."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        # Reorder columns so the background no longer aligns with X_test.
        shuffled = X_train[list(X_train.columns[::-1])]
        with pytest.raises(ValueError, match="columns must match"):
            _compute_shap(model, X_test, X_background=shuffled, random_state=0)


@pytest.mark.shap
@pytest.mark.integration
class TestExplainerSamplesBackground:
    """The explainer summarizes its background from X_background, not X_test."""

    @requires_shap
    def test_background_sample_is_xbackground_not_xtest(self, train_test_data):
        """The explainer's background sample is drawn from X_background, not X_test."""
        import shap as shap_lib
        from mlsurv.models import RSFModel

        X_train, X_test, y_train, y_test = train_test_data
        rsf = RSFModel(n_estimators=10, random_state=42)
        rsf.fit(X_train, y_train)

        seen = {}
        real_sample = shap_lib.sample

        def _spy_sample(data, *a, **k):
            # The first shap.sample call summarizes the background.
            seen.setdefault('first', data)
            return real_sample(data, *a, **k)

        with mock.patch('shap.sample', side_effect=_spy_sample):
            _compute_shap(rsf, X_test, X_background=X_train,
                          max_background=20, max_explain=10, random_state=0)

        assert 'first' in seen, "The explainer should summarize a background via shap.sample"
        pd.testing.assert_frame_equal(seen['first'], X_train)


@pytest.mark.shap
@pytest.mark.integration
class TestCallSitesForwardXTrainBackground:
    """Both cohort call sites pass X_train as the SHAP background."""

    @requires_shap
    def test_evaluate_forwards_xtrain_as_background(self, train_test_data):
        """evaluate(include_shap=True) threads X_train into _compute_shap."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        with mock.patch(
            'mlsurv.evaluation.evaluate._compute_shap',
            wraps=_compute_shap,
        ) as spy:
            evaluate(
                model, X_test, y_test, X_train=X_train,
                include_shap=True,
                shap_max_background=20,
                shap_max_explain=15,
                n_importance_repeats=1,
                show_progress=False,
            )
            spy.assert_called_once()
            passed = spy.call_args.kwargs['X_background']
            pd.testing.assert_frame_equal(passed, X_train)

    @requires_shap
    def test_importance_shap_forwards_xtrain_as_background(self, small_data):
        """importance(method='shap') passes self.X_train as the background."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')
        learner.evaluate('coxph', include_shap=False, n_importance_repeats=1,
                         show_progress=False)

        with mock.patch(
            'mlsurv.evaluation._shap._compute_shap',
            wraps=_compute_shap,
        ) as spy:
            learner.importance(models='coxph', method='shap', show_progress=False)
            spy.assert_called_once()
            passed = spy.call_args.kwargs['X_background']
            pd.testing.assert_frame_equal(passed, learner.X_train)
