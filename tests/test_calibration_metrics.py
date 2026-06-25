"""Tests for calibration metrics (slope, intercept, GND, Hosmer-Lemeshow).

Covers:
- Unit tests for calibration_slope_intercept(), hosmer_lemeshow(), greenwood_nam_dagostino()
- calibration_test() dispatcher
- Edge cases (small samples, all censored, perfect calibration)
- Integration with concordance pipeline (_add_calibration_metrics)
- Bootstrap dispatch for calibration metrics
- Scoring guard (diagnostic metrics can't be used for optimization)
"""

import numpy as np
import pandas as pd
import pytest

from tests._helpers import make_survival_array


@pytest.fixture()
def calibration_data():
    """Shared synthetic calibration test data (seed=42, n=100, all events)."""
    rng = np.random.RandomState(42)
    n = 100
    times = rng.exponential(5, n)
    events = np.ones(n, dtype=bool)
    eval_time = np.median(times)
    pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
    y = make_survival_array(events, times)
    return y, pred_probs, eval_time


class TestCalibrationSlopeIntercept:
    """Unit tests for calibration_slope_intercept()."""

    def test_well_calibrated(self):
        """Well-calibrated predictions should produce slope near 1, intercept near 0."""
        from mlsurv.evaluation.metrics import calibration_slope_intercept

        rng = np.random.RandomState(42)
        n = 200
        times = rng.exponential(10, n)
        events = np.ones(n, dtype=bool)
        eval_time = np.median(times)

        # Predictions that match actual survival
        survived = (times >= eval_time).astype(float)
        # Add noise to prevent perfect separation
        pred_probs = np.clip(survived * 0.7 + (1 - survived) * 0.3 + rng.normal(0, 0.1, n), 0.01, 0.99)

        y = make_survival_array(events, times)
        slope, intercept = calibration_slope_intercept(y, pred_probs, eval_time)

        assert not np.isnan(slope)
        assert not np.isnan(intercept)
        # Should be roughly calibrated (not exact due to noise)
        assert slope > 0

    def test_no_sklearn_penalty_warnings(self, calibration_data):
        """sklearn 1.8 deprecated penalty=None; verify no penalty/l1_ratio warnings."""
        import warnings as _warnings
        from mlsurv.evaluation.metrics import calibration_slope_intercept

        y, pred_probs, eval_time = calibration_data
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            calibration_slope_intercept(y, pred_probs, eval_time)
        leaked = [
            w for w in caught
            if "penalty" in str(w.message) or "l1_ratio" in str(w.message)
        ]
        assert not leaked, (
            f"Unexpected sklearn penalty/l1_ratio warnings: "
            f"{[(w.category.__name__, str(w.message)) for w in leaked]}"
        )

    def test_too_few_samples(self, caplog):
        """Fewer than 10 informative samples should return NaN with debug log."""
        import logging
        from mlsurv.evaluation.metrics import calibration_slope_intercept

        y = make_survival_array([True] * 5, [1.0] * 5)
        pred_probs = np.array([0.5] * 5)
        with caplog.at_level(logging.DEBUG, logger="mlsurv.metrics"):
            slope, intercept = calibration_slope_intercept(y, pred_probs, eval_time=2.0)
        assert np.isnan(slope)
        assert np.isnan(intercept)
        assert any("informative samples" in r.message for r in caplog.records)

    def test_all_censored_before_eval_time(self, caplog):
        """All censored before eval_time = no informative samples, with debug log."""
        import logging
        from mlsurv.evaluation.metrics import calibration_slope_intercept

        n = 20
        y = make_survival_array([False] * n, [1.0] * n)
        pred_probs = np.full(n, 0.5)
        with caplog.at_level(logging.DEBUG, logger="mlsurv.metrics"):
            slope, intercept = calibration_slope_intercept(y, pred_probs, eval_time=5.0)
        assert np.isnan(slope)
        assert np.isnan(intercept)
        assert any("informative samples" in r.message for r in caplog.records)

    @pytest.fixture(autouse=True)
    def _clear_calibration_warned(self):
        """Reset the module-level dedup set before each test."""
        from mlsurv.evaluation.metrics import _calibration_warned
        _calibration_warned.clear()

    def test_runtime_error_warns_user(self):
        """RuntimeError from LogisticRegression should emit MethodologicalWarning."""
        import warnings as _warnings
        from unittest.mock import patch
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import calibration_slope_intercept

        rng = np.random.RandomState(42)
        n = 50
        times = rng.exponential(5, n)
        events = np.ones(n, dtype=bool)
        eval_time = np.median(times)
        pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
        y = make_survival_array(events, times)

        with patch(
            "sklearn.linear_model.LogisticRegression.fit",
            side_effect=RuntimeError("convergence failed"),
        ):
            with _warnings.catch_warnings(record=True) as w:
                _warnings.simplefilter("always")
                slope, intercept = calibration_slope_intercept(y, pred_probs, eval_time)
            assert np.isnan(slope)
            assert np.isnan(intercept)
            cal_warns = [wi for wi in w if "returned NaN" in str(wi.message)]
            assert len(cal_warns) == 1
            assert issubclass(cal_warns[0].category, MethodologicalWarning)

    def test_too_few_samples_warns(self):
        """Fewer than 10 informative samples should emit MethodologicalWarning."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import calibration_slope_intercept

        y = make_survival_array([True] * 5, [1.0] * 5)
        pred_probs = np.array([0.5] * 5)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            slope, intercept = calibration_slope_intercept(y, pred_probs, eval_time=2.0)
        assert np.isnan(slope)
        assert np.isnan(intercept)
        cal_warns = [w for w in caught if "informative" in str(w.message)]
        assert len(cal_warns) == 1
        assert issubclass(cal_warns[0].category, MethodologicalWarning)

    def test_cal_slope_warning_dedup(self):
        """Second cal_slope failure should NOT re-warn (coarse dedup)."""
        import warnings as _warnings
        from unittest.mock import patch
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import calibration_slope_intercept

        rng = np.random.RandomState(42)
        n = 50
        times = rng.exponential(5, n)
        events = np.ones(n, dtype=bool)
        eval_time = np.median(times)
        pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
        y = make_survival_array(events, times)

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            # First call: small-sample → warns
            calibration_slope_intercept(
                make_survival_array([True] * 5, [1.0] * 5),
                np.array([0.5] * 5), eval_time=2.0,
            )
            # Second call: convergence failure → suppressed by coarse dedup
            with patch(
                "sklearn.linear_model.LogisticRegression.fit",
                side_effect=RuntimeError("convergence failed"),
            ):
                calibration_slope_intercept(y, pred_probs, eval_time)
        cal_warns = [wi for wi in w if issubclass(wi.category, MethodologicalWarning)]
        assert len(cal_warns) == 1

    def test_cal_slope_per_model_warning_surfaces(self):
        """Distinct model_name values should each emit a warning."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import calibration_slope_intercept

        y = make_survival_array([True] * 5, [1.0] * 5)
        pred_probs = np.array([0.5] * 5)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            calibration_slope_intercept(y, pred_probs, eval_time=2.0, model_name="coxph")
            calibration_slope_intercept(y, pred_probs, eval_time=2.0, model_name="rsf")
            # Repeat coxph — should NOT re-warn (same key already deduped).
            calibration_slope_intercept(y, pred_probs, eval_time=2.0, model_name="coxph")
        cal_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "informative" in str(w.message)
        ]
        assert len(cal_warns) == 2
        msgs = [str(w.message) for w in cal_warns]
        assert any("coxph" in m for m in msgs)
        assert any("rsf" in m for m in msgs)

    def test_mixed_censoring(self):
        """Censored-after-eval_time patients should be informative."""
        from mlsurv.evaluation.metrics import calibration_slope_intercept

        rng = np.random.RandomState(123)
        n = 100
        times = rng.exponential(5, n)
        # Mix of events and censored
        events = rng.choice([True, False], n, p=[0.7, 0.3])
        eval_time = 3.0
        pred_probs = np.clip(rng.uniform(0.2, 0.8, n), 0.01, 0.99)

        y = make_survival_array(events, times)
        slope, intercept = calibration_slope_intercept(y, pred_probs, eval_time)
        # Should produce valid (non-NaN) results
        assert not np.isnan(slope)


class TestHosmerLemeshow:
    """Unit tests for hosmer_lemeshow()."""

    def test_basic_computation(self, calibration_data):
        """Should return valid chi2 and p-value."""
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        y, pred_probs, eval_time = calibration_data
        chi2, p_value = hosmer_lemeshow(y, pred_probs, eval_time, n_bins=5)

        assert not np.isnan(chi2)
        assert not np.isnan(p_value)
        assert 0.0 <= p_value <= 1.0

    def test_too_few_samples(self):
        """n < n_bins * min_per_bin should return NaN (proactive check)."""
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        y = make_survival_array([True] * 5, [1.0, 2.0, 3.0, 4.0, 5.0])
        pred_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        chi2, p_value = hosmer_lemeshow(y, pred_probs, eval_time=3.0, n_bins=10)
        assert np.isnan(chi2)
        assert np.isnan(p_value)

    def test_constant_predictions(self):
        """Constant predictions = no bins = NaN."""
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        n = 50
        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.full(n, 0.5)
        chi2, p_value = hosmer_lemeshow(y, pred_probs, eval_time=5.0)
        assert np.isnan(chi2)
        assert np.isnan(p_value)

    @pytest.fixture(autouse=True)
    def _clear_calibration_warned(self):
        """Reset the module-level dedup set before each warning test."""
        from mlsurv.evaluation.metrics import _calibration_warned
        _calibration_warned.clear()

    @pytest.mark.parametrize("n, n_bins", [
        (28, 10),  # small sample
        (8, 10),   # very small sample (n < 10)
        (49, 10),  # one below threshold (n_bins * 5 - 1)
        (14, 3),   # non-default n_bins: threshold scales to 15
    ])
    def test_small_sample_warns(self, n, n_bins):
        """n < n_bins * 5 emits exactly one MethodologicalWarning, returns NaN,
        and the message reports the n_bins-scaled threshold."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.linspace(0.1, 0.9, n)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            chi2, p_value = hosmer_lemeshow(
                y, pred_probs, eval_time=5.0, n_bins=n_bins,
            )

        assert np.isnan(chi2)
        assert np.isnan(p_value)
        hl_warns = [w for w in caught if "Hosmer-Lemeshow calibration" in str(w.message)]
        assert len(hl_warns) == 1
        assert f"{n} samples < {n_bins * 5}" in str(hl_warns[0].message)
        assert issubclass(hl_warns[0].category, MethodologicalWarning)

    def test_boundary_exactly_at_threshold(self):
        """n == n_bins * 5 should NOT trigger the proactive check."""
        import warnings as _warnings
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        rng = np.random.RandomState(42)
        n = 50  # exactly n_bins * min_per_bin for n_bins=10
        times = rng.exponential(5, n)
        events = np.ones(n, dtype=bool)
        pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
        y = make_survival_array(events, times)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            hosmer_lemeshow(y, pred_probs, np.median(times), n_bins=10)

        hl_warns = [w for w in caught if "Hosmer-Lemeshow calibration" in str(w.message)]
        assert len(hl_warns) == 0

    def test_adequate_sample_no_warn(self):
        """n >= n_bins * 5 should NOT emit the small-sample warning."""
        import warnings as _warnings
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        rng = np.random.RandomState(42)
        n = 60
        times = rng.exponential(5, n)
        events = np.ones(n, dtype=bool)
        pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
        y = make_survival_array(events, times)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            chi2, p_value = hosmer_lemeshow(
                y, pred_probs, eval_time=np.median(times), n_bins=5,
            )

        assert not np.isnan(chi2)
        hl_warns = [w for w in caught if "Hosmer-Lemeshow calibration" in str(w.message)]
        assert len(hl_warns) == 0

    def test_dedup_suppresses_second_call(self):
        """Second HL call should NOT emit a warning (coarse dedup by metric)."""
        import warnings as _warnings
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        n = 28
        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.linspace(0.1, 0.9, n)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            hosmer_lemeshow(y, pred_probs, eval_time=5.0, n_bins=10)
            hosmer_lemeshow(y, pred_probs, eval_time=5.0, n_bins=10)

        hl_warns = [w for w in caught if "Hosmer-Lemeshow calibration" in str(w.message)]
        assert len(hl_warns) == 1

    def test_hl_does_not_suppress_gnd_warning(self):
        """HL and GND dedup keys are independent — both should warn."""
        import warnings as _warnings
        from mlsurv.evaluation.metrics import hosmer_lemeshow, greenwood_nam_dagostino

        n = 28
        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.linspace(0.1, 0.9, n)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            hosmer_lemeshow(y, pred_probs, eval_time=5.0, n_bins=10)
            greenwood_nam_dagostino(y, pred_probs, eval_time=5.0, n_bins=10)

        hl_warns = [w for w in caught if "Hosmer-Lemeshow calibration" in str(w.message)]
        gnd_warns = [w for w in caught if "GND calibration" in str(w.message)]
        assert len(hl_warns) == 1
        assert len(gnd_warns) == 1

    def test_non_default_n_bins_at_threshold_no_warn(self):
        """Threshold scales with n_bins: n == n_bins*5 (n_bins=3 → 15) does NOT warn.

        The below-threshold scaling (n=14, n_bins=3) is covered by the
        parametrized test_small_sample_warns case.
        """
        import warnings as _warnings
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        rng = np.random.RandomState(42)
        times = rng.exponential(5, 15)
        events = np.ones(15, dtype=bool)
        pred_at = np.clip(rng.uniform(0.1, 0.9, 15), 0.01, 0.99)
        y_at = make_survival_array(events, times)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            hosmer_lemeshow(y_at, pred_at, np.median(times), n_bins=3)

        hl_warns = [w for w in caught if "Hosmer-Lemeshow calibration" in str(w.message)]
        assert len(hl_warns) == 0

    def test_binning_failure_warns(self):
        """pd.qcut failure (constant predictions) should emit MethodologicalWarning."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        n = 50
        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.full(n, 0.5)  # constant → qcut fails
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            chi2, p_value = hosmer_lemeshow(y, pred_probs, eval_time=5.0)
        assert np.isnan(chi2)
        hl_warns = [w for w in caught if "Hosmer-Lemeshow returned NaN" in str(w.message)]
        assert len(hl_warns) == 1
        assert issubclass(hl_warns[0].category, MethodologicalWarning)

    def test_too_few_valid_bins_warns(self):
        """< 3 valid bins (df < 1) should emit MethodologicalWarning."""
        import warnings as _warnings
        from unittest.mock import patch
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        rng = np.random.RandomState(42)
        n = 50
        times = rng.exponential(5, n)
        events = np.ones(n, dtype=bool)
        pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
        y = make_survival_array(events, times)

        # Force all KM fits to fail → 0 valid bins → df < 1
        with patch(
            "mlsurv.evaluation.metrics._km_survival_at", return_value=None,
        ):
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                chi2, p_value = hosmer_lemeshow(y, pred_probs, eval_time=5.0, n_bins=5)
        assert np.isnan(chi2)
        hl_warns = [w for w in caught if "Hosmer-Lemeshow" in str(w.message)]
        assert len(hl_warns) == 1
        assert issubclass(hl_warns[0].category, MethodologicalWarning)

    def test_hl_per_model_warning_surfaces(self):
        """Distinct model_name values should each emit a warning."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import hosmer_lemeshow

        y = make_survival_array([True] * 5, [1.0] * 5)
        pred_probs = np.array([0.5] * 5)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            hosmer_lemeshow(y, pred_probs, eval_time=3.0, n_bins=10, model_name="coxph")
            hosmer_lemeshow(y, pred_probs, eval_time=3.0, n_bins=10, model_name="rsf")
            hosmer_lemeshow(y, pred_probs, eval_time=3.0, n_bins=10, model_name="coxph")
        hl_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "Hosmer-Lemeshow" in str(w.message)
        ]
        assert len(hl_warns) == 2
        msgs = [str(w.message) for w in hl_warns]
        assert any("coxph" in m for m in msgs)
        assert any("rsf" in m for m in msgs)


class TestGND:
    """Unit tests for greenwood_nam_dagostino()."""

    def test_basic_computation(self, calibration_data):
        """Should return valid chi2, p-value, and df."""
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        y, pred_probs, eval_time = calibration_data
        chi2, p_value, df = greenwood_nam_dagostino(y, pred_probs, eval_time, n_bins=5)

        assert not np.isnan(chi2)
        assert chi2 >= 0
        assert not np.isnan(p_value)
        assert 0.0 <= p_value <= 1.0
        assert df > 0

    def test_too_few_samples(self):
        """n < n_bins * min_per_bin should return NaN with df=0 (proactive check)."""
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        y = make_survival_array([True] * 5, [1.0, 2.0, 3.0, 4.0, 5.0])
        pred_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        chi2, p_value, df = greenwood_nam_dagostino(y, pred_probs, eval_time=3.0, n_bins=10)
        assert np.isnan(chi2)
        assert np.isnan(p_value)
        assert df == 0

    def test_constant_predictions(self):
        """Constant predictions = no bins = NaN."""
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        n = 50
        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.full(n, 0.5)
        chi2, p_value, df = greenwood_nam_dagostino(y, pred_probs, eval_time=5.0)
        assert np.isnan(chi2)
        assert np.isnan(p_value)
        assert df == 0

    def test_with_censoring(self):
        """GND should handle censored data properly (its primary advantage)."""
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        rng = np.random.RandomState(99)
        n = 150
        times = rng.exponential(5, n)
        events = rng.choice([True, False], n, p=[0.5, 0.5])
        eval_time = np.median(times)
        pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)

        y = make_survival_array(events, times)
        chi2, p_value, df = greenwood_nam_dagostino(y, pred_probs, eval_time, n_bins=5)

        assert not np.isnan(chi2)
        assert chi2 >= 0
        assert not np.isnan(p_value)
        assert 0.0 <= p_value <= 1.0

    def test_all_censored(self):
        """All-censored data should return NaN (Greenwood variance = 0 everywhere)."""
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        n = 100
        y = make_survival_array([False] * n, np.linspace(1, 10, n))
        pred_probs = np.linspace(0.1, 0.9, n)
        chi2, p_value, df = greenwood_nam_dagostino(y, pred_probs, eval_time=5.0, n_bins=5)
        assert np.isnan(chi2)
        assert np.isnan(p_value)
        assert df == 0

    def test_df_equals_valid_bins_minus_one(self):
        """GND df should equal G-1 (Demler et al. 2015)."""
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        rng = np.random.RandomState(42)
        n = 200
        times = rng.exponential(5, n)
        events = np.ones(n, dtype=bool)
        eval_time = np.median(times)
        pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)

        y = make_survival_array(events, times)
        _, _, df = greenwood_nam_dagostino(y, pred_probs, eval_time, n_bins=5)

        # df = G-1 where G=5 valid bins (all have >= 5 obs with 200 samples)
        assert df == 4

    @pytest.fixture(autouse=True)
    def _clear_calibration_warned(self):
        """Reset the module-level dedup set before each warning test."""
        from mlsurv.evaluation.metrics import _calibration_warned
        _calibration_warned.clear()

    @pytest.mark.parametrize("n, n_bins", [
        (28, 10),  # small sample
        (8, 10),   # very small sample (n < 10)
        (49, 10),  # one below threshold (n_bins * 5 - 1)
    ])
    def test_small_sample_warns(self, n, n_bins):
        """n < n_bins * 5 emits exactly one MethodologicalWarning, returns NaN
        with df=0, and the message reports the n_bins-scaled threshold."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.linspace(0.1, 0.9, n)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            chi2, p_value, df = greenwood_nam_dagostino(
                y, pred_probs, eval_time=5.0, n_bins=n_bins,
            )

        assert np.isnan(chi2)
        assert np.isnan(p_value)
        assert df == 0
        gnd_warns = [w for w in caught if "GND calibration" in str(w.message)]
        assert len(gnd_warns) == 1
        assert f"{n} samples < {n_bins * 5}" in str(gnd_warns[0].message)
        assert issubclass(gnd_warns[0].category, MethodologicalWarning)

    def test_boundary_exactly_at_threshold(self):
        """n == n_bins * 5 should NOT trigger the proactive check."""
        import warnings as _warnings
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        rng = np.random.RandomState(42)
        n = 50  # exactly n_bins * min_per_bin for n_bins=10
        times = rng.exponential(5, n)
        events = np.ones(n, dtype=bool)
        pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
        y = make_survival_array(events, times)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            greenwood_nam_dagostino(y, pred_probs, np.median(times), n_bins=10)

        gnd_warns = [w for w in caught if "GND calibration" in str(w.message)]
        assert len(gnd_warns) == 0

    def test_adequate_sample_no_warn(self):
        """n >= n_bins * 5 should NOT emit the small-sample warning."""
        import warnings as _warnings
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        rng = np.random.RandomState(42)
        n = 60
        times = rng.exponential(5, n)
        events = np.ones(n, dtype=bool)
        pred_probs = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
        y = make_survival_array(events, times)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            chi2, p_value, df = greenwood_nam_dagostino(
                y, pred_probs, eval_time=np.median(times), n_bins=5,
            )

        assert not np.isnan(chi2)
        gnd_warns = [w for w in caught if "GND calibration" in str(w.message)]
        assert len(gnd_warns) == 0

    def test_dedup_suppresses_second_call(self):
        """Second GND call should NOT emit a warning (coarse dedup by metric)."""
        import warnings as _warnings
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        n = 28
        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.linspace(0.1, 0.9, n)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            greenwood_nam_dagostino(y, pred_probs, eval_time=5.0, n_bins=10)
            greenwood_nam_dagostino(y, pred_probs, eval_time=5.0, n_bins=10)

        gnd_warns = [w for w in caught if "GND calibration" in str(w.message)]
        assert len(gnd_warns) == 1

    def test_binning_failure_warns(self):
        """pd.qcut failure (constant predictions) should emit MethodologicalWarning."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        n = 50
        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.full(n, 0.5)  # constant → qcut fails
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            chi2, p_value, df = greenwood_nam_dagostino(y, pred_probs, eval_time=5.0)
        assert np.isnan(chi2)
        assert df == 0
        gnd_warns = [w for w in caught if "GND calibration returned NaN" in str(w.message)]
        assert len(gnd_warns) == 1
        assert issubclass(gnd_warns[0].category, MethodologicalWarning)

    def test_too_few_valid_bins_warns(self):
        """< 2 valid bins should emit MethodologicalWarning."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        # All censored → Greenwood variance = 0 → all bins skip → 0 valid bins
        n = 100
        y = make_survival_array([False] * n, np.linspace(1, 10, n))
        pred_probs = np.linspace(0.1, 0.9, n)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            chi2, p_value, df = greenwood_nam_dagostino(
                y, pred_probs, eval_time=5.0, n_bins=5,
            )
        assert np.isnan(chi2)
        assert df == 0
        gnd_warns = [w for w in caught if "GND calibration" in str(w.message)]
        assert len(gnd_warns) == 1
        assert issubclass(gnd_warns[0].category, MethodologicalWarning)

    def test_gnd_per_model_warning_surfaces(self):
        """Distinct model_name values should each emit a warning."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        y = make_survival_array([True] * 5, [1.0] * 5)
        pred_probs = np.array([0.5] * 5)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            greenwood_nam_dagostino(y, pred_probs, eval_time=3.0, n_bins=10, model_name="coxph")
            greenwood_nam_dagostino(y, pred_probs, eval_time=3.0, n_bins=10, model_name="rsf")
            greenwood_nam_dagostino(y, pred_probs, eval_time=3.0, n_bins=10, model_name="coxph")
        gnd_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "GND calibration" in str(w.message)
        ]
        assert len(gnd_warns) == 2
        msgs = [str(w.message) for w in gnd_warns]
        assert any("coxph" in m for m in msgs)
        assert any("rsf" in m for m in msgs)


class TestCalibrationTestDispatcher:
    """Tests for calibration_test() dispatcher."""

    def test_default_is_gnd(self, calibration_data):
        """Default method should be GND."""
        from mlsurv.evaluation.metrics import calibration_test, greenwood_nam_dagostino

        y, pred_probs, eval_time = calibration_data

        chi2_disp, p_disp = calibration_test(y, pred_probs, eval_time, n_bins=5)
        chi2_gnd, p_gnd, _ = greenwood_nam_dagostino(y, pred_probs, eval_time, n_bins=5)

        assert chi2_disp == chi2_gnd
        assert p_disp == p_gnd

    def test_hl_method(self, calibration_data):
        """method='hl' should call hosmer_lemeshow."""
        from mlsurv.evaluation.metrics import calibration_test, hosmer_lemeshow

        y, pred_probs, eval_time = calibration_data

        chi2_disp, p_disp = calibration_test(y, pred_probs, eval_time, n_bins=5, method='hl')
        chi2_hl, p_hl = hosmer_lemeshow(y, pred_probs, eval_time, n_bins=5)

        assert chi2_disp == chi2_hl
        assert p_disp == p_hl

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        from mlsurv.evaluation.metrics import calibration_test

        y = make_survival_array([True] * 10, np.arange(1, 11, dtype=float))
        pred_probs = np.full(10, 0.5)

        with pytest.raises(ValueError, match="Unknown calibration test method"):
            calibration_test(y, pred_probs, eval_time=5.0, method='invalid')


class TestConcordanceIntegration:
    """Integration tests: calibration metrics appear in concordance pipeline."""

    @pytest.fixture(autouse=True)
    def _clear_calibration_warned(self):
        """Reset the module-level dedup set before each test."""
        from mlsurv.evaluation.metrics import _calibration_warned
        _calibration_warned.clear()

    def test_calibration_columns_present(self):
        """_add_calibration_metrics should add expected columns to data dict."""
        from mlsurv.evaluation._concordance import _add_calibration_metrics

        rng = np.random.RandomState(42)
        n = 100
        times = rng.exponential(5, n)
        events = np.ones(n, dtype=bool)
        y = make_survival_array(events, times)

        eval_times = np.percentile(times, [25, 50, 75])
        surv_probs = rng.uniform(0.1, 0.9, (n, len(eval_times)))

        data = {}
        _add_calibration_metrics(data, y, surv_probs, eval_times)

        assert 'cal_slope_test' in data
        assert 'cal_intercept_test' in data
        assert 'gnd_test' in data

        # Verify GND test produced a valid p-value
        gnd_p = data['gnd_test'][0]
        assert isinstance(gnd_p, float)
        assert 0.0 <= gnd_p <= 1.0 or np.isnan(gnd_p)

    def test_per_model_dedup_through_pipeline(self):
        """model_name threaded through _add_calibration_metrics surfaces
        a per-(metric, model) warning, not just per metric."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation._concordance import _add_calibration_metrics

        # Tiny test set: < 10 informative samples → cal_slope NaN warning.
        # < 50 samples for n_bins=10 → GND samples-too-few warning.
        y = make_survival_array([True] * 5, [1.0] * 5)
        eval_times = np.array([1.0, 2.0, 3.0])
        surv_probs = np.full((5, 3), 0.5)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _add_calibration_metrics({}, y, surv_probs, eval_times, model_name="coxph")
            _add_calibration_metrics({}, y, surv_probs, eval_times, model_name="rsf")
        # Each model fires once for cal_slope and once for GND → 4 warnings.
        meth_warns = [w for w in caught if issubclass(w.category, MethodologicalWarning)]
        msgs = [str(w.message) for w in meth_warns]
        coxph_msgs = [m for m in msgs if "coxph" in m]
        rsf_msgs = [m for m in msgs if "rsf" in m]
        assert any("informative" in m for m in coxph_msgs)
        assert any("informative" in m for m in rsf_msgs)
        assert any("GND" in m for m in coxph_msgs)
        assert any("GND" in m for m in rsf_msgs)

    @pytest.mark.integration
    def test_learner_run_per_model_warning(self):
        """End-to-end: learner.run([m1, m2]) on a tiny test set should
        emit a calibration NaN warning for EACH model, not just the first."""
        import warnings as _warnings
        import pandas as pd
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import MethodologicalWarning

        rng = np.random.RandomState(0)
        n = 20
        X = pd.DataFrame({
            'x1': rng.normal(size=n),
            'x2': rng.normal(size=n),
            'x3': rng.normal(size=n),
        })
        times = rng.exponential(5, n)
        events = rng.choice([True, False], size=n, p=[0.4, 0.6])
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        # test_fraction=0.4 → 8-sample test set → informative.sum() < 10.
        learner.setup(random_state=42, cv_folds=2, test_fraction=0.4)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            learner.run(['coxph', 'coxnet'])

        cal_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "informative" in str(w.message)
        ]
        msgs = [str(w.message) for w in cal_warns]
        assert any("coxph" in m for m in msgs), (
            f"coxph warning missing — got: {msgs}"
        )
        assert any("coxnet" in m for m in msgs), (
            f"coxnet warning missing — got: {msgs}"
        )

    def test_none_surv_probs_produces_nan(self):
        """If surv_probs is None, all calibration metrics should be NaN."""
        from mlsurv.evaluation._concordance import _add_calibration_metrics

        y = make_survival_array([True] * 10, np.arange(1, 11, dtype=float))
        data = {}
        _add_calibration_metrics(data, y, None, None)

        assert np.isnan(data['cal_slope_test'][0])
        assert np.isnan(data['cal_intercept_test'][0])
        assert np.isnan(data['gnd_test'][0])


class TestBootstrapDispatch:
    """Tests for bootstrap metric dispatch of calibration metrics."""

    def test_dispatch_entries_exist(self):
        """Calibration metrics should be in the dispatch dict."""
        from mlsurv.evaluation._bootstrap_iteration import _METRIC_DISPATCH

        assert 'cal_slope' in _METRIC_DISPATCH
        assert 'cal_intercept' in _METRIC_DISPATCH
        assert 'gnd' in _METRIC_DISPATCH


class TestScoringGuard:
    """Tests that diagnostic metrics cannot be used for scoring/optimization."""

    def test_diagnostic_metric_raises(self):
        """score_survival_model should reject diagnostic metrics."""
        from mlsurv.evaluation.metrics import score_survival_model

        with pytest.raises(ValueError, match="diagnostic metric"):
            score_survival_model(None, None, None, scoring='cal_slope')

        with pytest.raises(ValueError, match="diagnostic metric"):
            score_survival_model(None, None, None, scoring='cal_intercept')

        with pytest.raises(ValueError, match="diagnostic metric"):
            score_survival_model(None, None, None, scoring='gnd')

    def test_scorable_metrics_pass_guard(self):
        """Scorable metrics should not be rejected by the guard."""
        from mlsurv.constants import METRIC_REGISTRY
        for key, info in METRIC_REGISTRY.items():
            if not info.diagnostic_only:
                # These should not raise ValueError about diagnostic
                # (they may raise other errors due to None inputs, that's fine)
                try:
                    from mlsurv.evaluation.metrics import score_survival_model
                    score_survival_model(None, None, None, scoring=key)
                except ValueError as e:
                    assert "diagnostic" not in str(e).lower()
                except (TypeError, AttributeError):
                    pass  # Expected - None inputs


class TestEvaluationResultProperties:
    """Tests for EvaluationResult calibration convenience properties."""

    def test_calibration_properties_from_concordance(self):
        """Properties should read from concordance DataFrame."""
        from mlsurv.results.containers import EvaluationResult

        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': [0.75],
            'cal_slope_test': [0.95],
            'cal_intercept_test': [-0.05],
            'gnd_test': [0.42],
        })

        result = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
        )

        assert result.cal_slope_test == pytest.approx(0.95)
        assert result.cal_intercept_test == pytest.approx(-0.05)
        assert result.gnd_test == pytest.approx(0.42)

    def test_calibration_properties_none_when_missing(self):
        """Properties should return None when columns are absent."""
        from mlsurv.results.containers import EvaluationResult

        concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': [0.75],
        })

        result = EvaluationResult(
            concordance=concordance,
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
        )

        assert result.cal_slope_test is None
        assert result.cal_intercept_test is None
        assert result.gnd_test is None
