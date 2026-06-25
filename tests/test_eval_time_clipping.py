"""Tests for eval_times clipping to data range.

Ensures that bootstrap resamples and subpopulations with shorter follow-up
than the global eval grid produce partial results (NaN for out-of-range
times) instead of raising ValueError.
"""

import numpy as np
import pytest

from mlsurv.utils import clip_eval_times_to_data
from mlsurv._warnings import MethodologicalWarning

from tests._helpers import make_y


# ── clip_eval_times_to_data unit tests ──────────────────────────────────


class TestClipEvalTimesToData:
    """Unit tests for the clip_eval_times_to_data utility."""

    def test_all_times_inside(self):
        """Eval times fully within data range are kept."""
        y = make_y([1.0, 5.0, 10.0])
        eval_times = np.array([2.0, 4.0, 8.0])
        valid, mask, y_out = clip_eval_times_to_data(eval_times, y)
        np.testing.assert_array_equal(valid, [2.0, 4.0, 8.0])
        assert mask.all()
        assert y_out is y  # no truncation when y_train=None

    def test_some_times_outside(self):
        """Only eval times within (t_min, t_max) are kept."""
        y = make_y([2.0, 5.0, 8.0])
        eval_times = np.array([1.0, 3.0, 6.0, 9.0])
        valid, mask, _ = clip_eval_times_to_data(eval_times, y)
        np.testing.assert_array_equal(valid, [3.0, 6.0])
        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_all_times_outside(self):
        """All eval times outside data range returns empty."""
        y = make_y([5.0, 6.0, 7.0])
        eval_times = np.array([1.0, 2.0, 10.0])
        valid, mask, _ = clip_eval_times_to_data(eval_times, y)
        assert len(valid) == 0
        assert not mask.any()

    def test_boundary_times_excluded(self):
        """Eval times equal to t_min or t_max are excluded (strict inequality)."""
        y = make_y([2.0, 5.0, 8.0])
        eval_times = np.array([2.0, 5.0, 8.0])
        valid, mask, _ = clip_eval_times_to_data(eval_times, y)
        # Only 5.0 is strictly between 2.0 and 8.0
        np.testing.assert_array_equal(valid, [5.0])

    def test_single_eval_time_inside(self):
        """Single eval time within range is kept."""
        y = make_y([1.0, 10.0])
        eval_times = np.array([5.0])
        valid, mask, _ = clip_eval_times_to_data(eval_times, y)
        np.testing.assert_array_equal(valid, [5.0])

    def test_truncation_with_y_train(self):
        """y_data is truncated to y_train range when y_train provided."""
        y_data = make_y([1.0, 5.0, 20.0])  # extends beyond y_train
        y_train = make_y([1.0, 3.0, 10.0])
        eval_times = np.array([2.0, 8.0, 15.0])

        valid, mask, y_trunc = clip_eval_times_to_data(eval_times, y_data, y_train)
        # After truncation, y_data max should be 10.0 (capped at y_train max)
        assert float(y_trunc['time'].max()) == 10.0
        # eval_times 2.0 and 8.0 are within (1.0, 10.0); 15.0 is outside
        np.testing.assert_array_equal(valid, [2.0, 8.0])

    def test_empty_eval_times(self):
        """Empty eval_times input returns empty output."""
        y = make_y([1.0, 5.0])
        eval_times = np.array([], dtype=float)
        valid, mask, _ = clip_eval_times_to_data(eval_times, y)
        assert len(valid) == 0
        assert len(mask) == 0

    def test_accepts_python_list_input(self):
        """Python list eval_times is coerced; matches ndarray behavior.

        Subpopulation AUC(t) path forwards the caller's eval_times directly
        to this function. When validate() is called with eval_times=[...],
        the list reached here without coercion and raised
        `TypeError: '>' not supported between instances of 'list' and 'float'`.
        """
        y = make_y([2.0, 5.0, 8.0])
        eval_times_list = [1.0, 3.0, 6.0, 9.0]
        valid, mask, _ = clip_eval_times_to_data(eval_times_list, y)
        np.testing.assert_array_equal(valid, [3.0, 6.0])
        np.testing.assert_array_equal(mask, [False, True, True, False])


# ── Regression tests for bootstrap metric functions ─────────────────────


class TestBootstrapAucClipping:
    """Regression test: bootstrap AUC(t) with short-range resample."""

    def test_short_resample_does_not_crash(self):
        """Resampled data with shorter follow-up returns partial results."""
        from mlsurv.evaluation._bootstrap_iteration import _compute_auc_for_iteration

        # Training data with long follow-up
        y_train = make_y(
            [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0],
            [True, True, True, False, True, True, False, True, True, False],
        )
        # Bootstrap resample with much shorter max follow-up
        y_boot = make_y(
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            [True, True, False, True, True, True, False, True, True, True],
        )
        # Eval times that extend beyond y_boot's range
        eval_times = np.array([1.0, 3.0, 7.0, 10.0, 15.0, 20.0])
        risk_scores = np.random.default_rng(42).standard_normal(len(y_boot))

        failures = []
        result = _compute_auc_for_iteration(
            y_train, y_boot, risk_scores, eval_times,
            model_name="test", iteration=0, failures=failures,
        )

        # Should return rows for ALL eval_times (not crash)
        assert result is not None
        assert len(result) == len(eval_times)

        # Times within y_boot range should have computed AUC (or NaN from sksurv)
        # Times outside should be NaN
        result_by_time = {r['eval_time']: r['auc'] for r in result}
        # 7.0, 10.0, 15.0, 20.0 are outside (0.5, 5.0) → must be NaN
        for t in [7.0, 10.0, 15.0, 20.0]:
            assert np.isnan(result_by_time[t]), f"Expected NaN for t={t}"


class TestBootstrapIBSClipping:
    """Regression test: bootstrap IBS with short-range resample."""

    def test_short_resample_returns_nan(self):
        """IBS returns NaN when eval_times mostly outside data range."""
        from mlsurv.evaluation._bootstrap_iteration import _compute_integrated_brier

        y_train = make_y(
            [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            [True, True, True, True, True, False],
        )
        # Very short follow-up resample
        y_boot = make_y(
            [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            [True, True, False, True, True, True],
        )
        # All eval_times beyond y_boot's range
        eval_times = np.array([5.0, 10.0, 15.0])
        risk_scores = np.random.default_rng(42).standard_normal(len(y_boot))

        # Should return NaN, not raise ValueError
        result = _compute_integrated_brier(
            y_boot, risk_scores, y_train=y_train,
            eval_times=eval_times, surv_probs=None,
        )
        assert np.isnan(result)


class TestBootstrapBrierClipping:
    """Regression test: bootstrap Brier with short-range resample."""

    def test_median_outside_range_returns_nan(self):
        """Brier returns NaN when all valid times are removed."""
        from mlsurv.evaluation._bootstrap_iteration import _compute_brier

        y_train = make_y(
            [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            [True, True, True, True, True, False],
        )
        y_boot = make_y(
            [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            [True, True, False, True, True, True],
        )
        # All eval_times beyond y_boot range
        eval_times = np.array([5.0, 10.0, 15.0])
        risk_scores = np.random.default_rng(42).standard_normal(len(y_boot))

        result = _compute_brier(
            y_boot, risk_scores, y_train=y_train,
            eval_times=eval_times, surv_probs=None,
        )
        assert np.isnan(result)


# ── Regression tests for point-estimate metric functions ────────────────


class TestPointEstimateAucClipping:
    """Regression test: point-estimate AUC(t) with eval_times partially out of range."""

    def test_partial_results_not_crash(self):
        """AUC(t) returns partial results when some eval_times exceed data range."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        y_train = make_y(
            [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0],
            [True, True, True, False, True, True, False, True, True, False],
        )
        # Test set with shorter follow-up than training
        y_test = make_y(
            [0.3, 0.8, 1.5, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            [True, True, False, True, True, True, False, True, True, True],
        )
        # Eval times that extend beyond y_test's range
        eval_times = np.array([1.0, 3.0, 5.0, 10.0, 15.0, 20.0])
        risk_scores = np.random.default_rng(42).standard_normal(len(y_test))

        # Times beyond the test range now surface a MethodologicalWarning
        # (No Silent Data Loss) instead of silently NaN-filling.
        with pytest.warns(MethodologicalWarning, match="AUC"):
            result_df = _compute_auc_t(y_test, y_train, eval_times, risk_scores)

        # Should have rows for all eval_times
        assert len(result_df) == len(eval_times)
        # Times beyond (0.3, 6.0) should be NaN
        for _, row in result_df.iterrows():
            if row['eval_time'] >= 6.0 or row['eval_time'] <= 0.3:
                assert np.isnan(row['auc']), f"Expected NaN for t={row['eval_time']}"


class TestSubpopulationIBSClipping:
    """Regression test: subpopulation IBS with eval_times outside subpopulation range."""

    def test_subpopulation_ibs_clipping(self):
        """_safe_ibs returns NaN when eval_times exceed subpopulation range."""
        from mlsurv.evaluation._subpopulation import _safe_ibs

        y_train = make_y(
            [1.0, 5.0, 10.0, 15.0, 20.0],
            [True, True, True, True, False],
        )
        # Small subpopulation with short follow-up
        y_sub = make_y(
            [1.0, 2.0, 3.0],
            [True, True, True],
        )
        # surv_probs aligned to eval_times (3 subjects × 3 times)
        eval_times = np.array([5.0, 10.0, 15.0])
        surv_probs = np.array([[0.8, 0.5, 0.2], [0.9, 0.6, 0.3], [0.7, 0.4, 0.1]])

        # All eval_times are beyond y_sub's max (3.0) → should return NaN
        result = _safe_ibs(y_sub, surv_probs, eval_times, y_train)
        assert np.isnan(result)


# ── Public API auc_t() clipping tests ───────────────────────────


class TestPublicAucTClipping:
    """Regression tests for public auc_t() eval_times clipping."""

    def test_partial_results(self):
        """auc_t() returns partial results when some eval_times exceed data range."""
        from mlsurv.evaluation.metrics import auc_t

        y_train = make_y(
            [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0],
            [True, True, True, False, True, True, False, True, True, False],
        )
        y_test = make_y(
            [0.3, 0.8, 1.5, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            [True, True, False, True, True, True, False, True, True, True],
        )
        eval_times = np.array([1.0, 3.0, 5.0, 10.0, 15.0, 20.0])
        risk_scores = np.random.default_rng(42).standard_normal(len(y_test))

        auc_values, mean_auc = auc_t(y_train, y_test, risk_scores, eval_times)

        # Should have one AUC per eval_time
        assert len(auc_values) == len(eval_times)
        # Times beyond (0.3, 6.0) should be NaN
        for i, t in enumerate(eval_times):
            if t >= 6.0 or t <= 0.3:
                assert np.isnan(auc_values[i]), f"Expected NaN for t={t}"
        # mean_auc should be a finite number (computed from valid times)
        assert np.isfinite(mean_auc)

    def test_all_out_of_range(self):
        """auc_t() returns all NaN when no eval_times are within data range."""
        from mlsurv.evaluation.metrics import auc_t

        y_train = make_y([1.0, 5.0, 10.0], [True, True, False])
        y_test = make_y([1.0, 2.0, 3.0], [True, True, True])
        eval_times = np.array([5.0, 10.0, 15.0])
        risk_scores = np.random.default_rng(42).standard_normal(len(y_test))

        auc_values, mean_auc = auc_t(y_train, y_test, risk_scores, eval_times)

        assert all(np.isnan(auc_values))
        assert np.isnan(mean_auc)


# ── _get_surv_probs_at_median() clipping tests ──────────────────


class TestSurvProbsAtMedianClipping:
    """Regression tests for _get_surv_probs_at_median eval_times clipping."""

    def test_median_within_data_range(self):
        """Median is selected from valid times within data range."""
        from mlsurv.evaluation._bootstrap_iteration import _get_surv_probs_at_median

        y = make_y([1.0, 3.0, 5.0, 7.0, 9.0], [True, True, False, True, True])
        # eval_times extend well beyond y's range
        eval_times = np.array([2.0, 4.0, 6.0, 15.0, 20.0, 25.0])
        # surv_probs: 5 subjects × 6 times
        surv_probs = np.random.default_rng(42).uniform(0, 1, (5, 6))

        probs, median_time = _get_surv_probs_at_median(
            y, eval_times, surv_probs, model=None, X=None,
        )

        # Median should be within y's range (1.0, 9.0)
        assert 1.0 < median_time < 9.0
        # Should return probabilities for all 5 subjects
        assert probs.shape == (5,)
        assert np.all(np.isfinite(probs))

    def test_all_times_outside_returns_nan(self):
        """Returns NaN when all eval_times are outside data range."""
        from mlsurv.evaluation._bootstrap_iteration import _get_surv_probs_at_median

        y = make_y([1.0, 2.0, 3.0], [True, True, True])
        eval_times = np.array([5.0, 10.0, 15.0])
        surv_probs = np.random.default_rng(42).uniform(0, 1, (3, 3))

        probs, median_time = _get_surv_probs_at_median(
            y, eval_times, surv_probs, model=None, X=None,
        )

        assert np.isnan(median_time)


# ── graceful degradation when auto_eval_times itself fails ─────────


class TestAutoEvalTimesFallback:
    """auto_eval_times failure on degenerate y_train must NOT crash
    evaluate()/bootstrap()/validate(). Time-dependent metrics gracefully
    return NaN and the user gets a SuggestionWarning."""

    def test_compute_auc_t_handles_none_eval_times(self):
        """_compute_auc_t returns empty frame when eval_times is None."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        y_test = make_y([1.0, 2.0, 3.0], [True, False, True])
        y_train = make_y([1.0, 2.0, 3.0], [True, True, False])
        risk_scores = np.array([0.1, 0.2, 0.3])

        df = _compute_auc_t(y_test, y_train, None, risk_scores)
        assert len(df) == 0
        assert set(df.columns) == {'eval_time', 'auc', 'n_events', 'n_at_risk'}

    def test_compute_auc_t_handles_empty_eval_times(self):
        """_compute_auc_t returns empty frame when eval_times is empty array."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        y_test = make_y([1.0, 2.0, 3.0], [True, False, True])
        y_train = make_y([1.0, 2.0, 3.0], [True, True, False])
        risk_scores = np.array([0.1, 0.2, 0.3])

        df = _compute_auc_t(y_test, y_train, np.array([]), risk_scores)
        assert len(df) == 0

    @pytest.mark.integration
    def test_evaluate_with_degenerate_y_train_does_not_crash(self):
        """evaluate() with y_train having <2 unique event times warns and
        returns a result with NaN time-dependent metrics, instead of raising
        ValueError from auto_eval_times.
        """
        import warnings as _warnings
        import pandas as pd
        from mlsurv import SurvivalLearner
        from mlsurv.evaluation.evaluate import evaluate as _evaluate
        from mlsurv._warnings import SuggestionWarning

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((40, 3)), columns=list("abc"))
        y = make_y(
            rng.uniform(1.0, 5.0, 40),
            rng.uniform(0, 1, 40) > 0.4,
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        model = learner._state.models["coxph"].model

        # Degenerate y_train: only one unique event time → auto_eval_times
        # raises "Only 1 unique event time(s)" ValueError.
        y_train_degenerate = make_y(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [True, False, False, False, False],
        )
        X_train_degenerate = pd.DataFrame(
            rng.standard_normal((5, 3)), columns=list("abc"),
        )

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = _evaluate(
                model, learner.X_test, learner.y_test,
                X_train=X_train_degenerate, y_train=y_train_degenerate,
                eval_times=None, n_importance_repeats=0,
            )

        # Did NOT crash and produced a usable EvaluationResult.
        assert result is not None
        # User-visible SuggestionWarning fired with the documented prefix.
        assert any(
            isinstance(w.message, SuggestionWarning)
            and "auto_eval_times failed in evaluate()" in str(w.message)
            for w in caught
        ), [str(w.message) for w in caught]
        # Time-dependent metrics are NaN (or empty), Harrell's C still computed.
        conc = result.concordance
        assert np.isfinite(conc['c_index_test'].iloc[0])
        if 'integrated_brier_test' in conc.columns:
            assert np.isnan(conc['integrated_brier_test'].iloc[0])
        # AUC(t) DataFrame is empty (no eval_times to compute at).
        assert len(result.auc_t) == 0

    @pytest.mark.integration
    def test_bootstrap_with_degenerate_y_train_does_not_crash(self):
        """bootstrap() with include_auc=True on degenerate y_train must not
        crash the iteration workers (_compute_auc_for_iteration
        previously iterated eval_times=None and propagated TypeError).
        """
        import warnings as _warnings
        import pandas as pd
        from mlsurv import SurvivalLearner
        from mlsurv.evaluation.bootstrap import bootstrap as _bootstrap

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((40, 3)), columns=list("abc"))
        y = make_y(
            rng.uniform(1.0, 5.0, 40),
            rng.uniform(0, 1, 40) > 0.4,
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        model = learner._state.models["coxph"].model

        y_train_degenerate = make_y(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [True, False, False, False, False],
        )

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # noisy fold warnings expected
            result = _bootstrap(
                model, learner.X_test, learner.y_test,
                y_train=y_train_degenerate, n_bootstrap=2,
                include_auc=True, show_progress=False,
            )

        assert result is not None
        # Bootstrap completed; AUC iterations short-circuited gracefully.
        assert hasattr(result, 'iterations')
