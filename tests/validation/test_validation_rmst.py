"""Reference-implementation validation for RMST computation and comparison.

Independent Python implementation of RMST compared against the production
``rmst()`` and ``rmst_test()`` on real clinical datasets and simulated data
with known analytical solutions.

Reference
---------
Tian L, Zhao L, Wei LJ. Predicting the restricted mean event time with
the subject's baseline covariates in survival analysis. Biostatistics.
2014;15(2):222-233.
"""

import numpy as np
import pytest

from mlsurv.evaluation.metrics import rmst, rmst_test


def _extract_times_events(y):
    """Extract times and events from a structured array with varying field names."""
    names = y.dtype.names
    # sksurv datasets: ('cens', 'time'), ('Status', 'Survival_in_days'), etc.
    # First field is event indicator (bool), second is time (float).
    events = y[names[0]].astype(bool)
    times = y[names[1]].astype(float)
    return times, events


# ======================================================================
# Reference implementation — independent from production code
# ======================================================================


def _ref_km_table(times, events):
    """Build a KM event table from raw data (loop-based, no lifelines).

    Returns sorted arrays: (event_times, n_events, n_at_risk, survival).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)

    # Unique event times (only where events occurred)
    event_mask = events
    unique_event_times = np.unique(times[event_mask])
    unique_event_times.sort()

    surv = 1.0
    km_times = []
    km_surv = []
    km_d = []
    km_n = []

    for t in unique_event_times:
        n_at_risk = int(np.sum(times >= t))
        n_events_at_t = int(np.sum((times == t) & events))

        if n_at_risk == 0:
            continue

        surv *= (1.0 - n_events_at_t / n_at_risk)

        km_times.append(t)
        km_d.append(n_events_at_t)
        km_n.append(n_at_risk)
        km_surv.append(surv)

    return (
        np.array(km_times),
        np.array(km_d),
        np.array(km_n),
        np.array(km_surv),
    )


def _ref_rmst(times, events, tau):
    """Reference RMST: pure-Python KM integration + Greenwood variance.

    Deliberately different style from production (loop-based, no lifelines).
    """
    km_times, km_d, km_n, km_surv = _ref_km_table(times, events)

    # Build step function points for integration
    # S(0) = 1.0, then step down at each event time
    t_pts = [0.0]
    s_pts = [1.0]
    for i in range(len(km_times)):
        if km_times[i] > tau:
            break
        t_pts.append(km_times[i])
        s_pts.append(km_surv[i])
    t_pts.append(tau)
    s_pts.append(s_pts[-1])  # extend last survival to tau

    # Step-function integration: sum S(t_i) * (t_{i+1} - t_i)
    rmst_val = 0.0
    for i in range(len(t_pts) - 1):
        rmst_val += s_pts[i] * (t_pts[i + 1] - t_pts[i])

    # Greenwood variance for RMST
    var_rmst = 0.0
    for k in range(len(km_times)):
        t_j = km_times[k]
        if t_j > tau:
            break
        d_j = km_d[k]
        n_j = km_n[k]
        if n_j <= d_j:
            continue

        # Tail integral from t_j to tau
        tail_area = 0.0
        # Find start index in t_pts
        for i in range(len(t_pts) - 1):
            lo = max(t_pts[i], t_j)
            hi = t_pts[i + 1]
            if hi <= t_j:
                continue
            if lo >= tau:
                break
            hi = min(hi, tau)
            tail_area += s_pts[i] * (hi - lo)

        var_rmst += d_j / (n_j * (n_j - d_j)) * tail_area ** 2

    se_rmst = float(np.sqrt(var_rmst))
    return float(rmst_val), se_rmst


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def exponential_data():
    """Large exponential sample for analytical validation."""
    rng = np.random.RandomState(42)
    lam = 0.3
    n = 10000
    times = rng.exponential(1.0 / lam, size=n)
    events = np.ones(n, dtype=bool)
    tau = 5.0
    analytical_rmst = (1 - np.exp(-lam * tau)) / lam
    return times, events, tau, analytical_rmst


# ======================================================================
# Tests
# ======================================================================


@pytest.mark.validation
class TestRMSTValidation:
    """Validate rmst() against independent reference implementation."""

    def test_exponential_analytical(self, exponential_data):
        """Exponential survival -> RMST = (1-exp(-lambda*tau))/lambda."""
        times, events, tau, analytical_rmst = exponential_data

        prod_val, prod_se = rmst(times, events, tau=tau)
        ref_val, ref_se = _ref_rmst(times, events, tau)

        # Production matches reference
        np.testing.assert_allclose(prod_val, ref_val, rtol=1e-6)
        np.testing.assert_allclose(prod_se, ref_se, rtol=1e-4)

        # Both close to analytical
        np.testing.assert_allclose(prod_val, analytical_rmst, rtol=0.02)
        np.testing.assert_allclose(ref_val, analytical_rmst, rtol=0.02)

    def test_gbsg2_two_groups(self, gbsg2_data):
        """RMST on GBSG2 matches pure-Python reference (median split)."""
        X, y = gbsg2_data
        times, events = _extract_times_events(y)

        # Median split on first numeric feature
        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            pytest.skip("No numeric features in GBSG2")
        scores = X_num.iloc[:, 0].values
        median_val = np.median(scores)

        # Low group
        low_mask = scores <= median_val
        high_mask = ~low_mask

        tau = float(min(times[low_mask].max(), times[high_mask].max()))

        prod_low, prod_se_low = rmst(times[low_mask], events[low_mask], tau=tau)
        ref_low, ref_se_low = _ref_rmst(times[low_mask], events[low_mask], tau)

        prod_high, prod_se_high = rmst(times[high_mask], events[high_mask], tau=tau)
        ref_high, ref_se_high = _ref_rmst(times[high_mask], events[high_mask], tau)

        np.testing.assert_allclose(prod_low, ref_low, rtol=1e-6)
        np.testing.assert_allclose(prod_high, ref_high, rtol=1e-6)
        np.testing.assert_allclose(prod_se_low, ref_se_low, rtol=1e-4)
        np.testing.assert_allclose(prod_se_high, ref_se_high, rtol=1e-4)

    def test_veterans_two_groups(self, veterans_data):
        """RMST on Veterans matches pure-Python reference."""
        X, y = veterans_data
        times, events = _extract_times_events(y)

        X_num = X.select_dtypes(include=[np.number])
        scores = X_num.iloc[:, 0].values
        median_val = np.median(scores)

        low_mask = scores <= median_val
        high_mask = ~low_mask

        tau = float(min(times[low_mask].max(), times[high_mask].max()))

        prod_low, prod_se_low = rmst(times[low_mask], events[low_mask], tau=tau)
        ref_low, ref_se_low = _ref_rmst(times[low_mask], events[low_mask], tau)

        prod_high, prod_se_high = rmst(times[high_mask], events[high_mask], tau=tau)
        ref_high, ref_se_high = _ref_rmst(times[high_mask], events[high_mask], tau)

        np.testing.assert_allclose(prod_low, ref_low, rtol=1e-6)
        np.testing.assert_allclose(prod_high, ref_high, rtol=1e-6)
        np.testing.assert_allclose(prod_se_low, ref_se_low, rtol=1e-4)
        np.testing.assert_allclose(prod_se_high, ref_se_high, rtol=1e-4)

    def test_heavy_censoring(self):
        """High censoring rate: reference and production agree."""
        rng = np.random.RandomState(123)
        n = 500
        event_times = rng.exponential(5.0, size=n)
        censor_times = rng.exponential(1.0, size=n)  # heavy censoring
        times = np.minimum(event_times, censor_times)
        events = (event_times <= censor_times)
        tau = 3.0

        prod_val, prod_se = rmst(times, events, tau=tau)
        ref_val, ref_se = _ref_rmst(times, events, tau)

        np.testing.assert_allclose(prod_val, ref_val, rtol=1e-6)
        np.testing.assert_allclose(prod_se, ref_se, rtol=1e-4)


@pytest.mark.validation
class TestRMSTDifferenceValidation:
    """Validate rmst_test() difference and p-value."""

    def test_gbsg2_difference_matches_ref(self, gbsg2_data):
        """Difference, SE, z-stat, p-value match reference computation."""
        from scipy.stats import norm

        X, y = gbsg2_data
        times, events = _extract_times_events(y)

        X_num = X.select_dtypes(include=[np.number])
        scores = X_num.iloc[:, 0].values
        median_val = np.median(scores)
        groups = (scores > median_val).astype(int)

        tau = float(min(times[groups == 0].max(), times[groups == 1].max()))

        # Production
        result = rmst_test(times, events, groups, tau=tau)

        # Reference
        ref0, ref_se0 = _ref_rmst(times[groups == 0], events[groups == 0], tau)
        ref1, ref_se1 = _ref_rmst(times[groups == 1], events[groups == 1], tau)
        ref_diff = ref1 - ref0
        ref_diff_se = float(np.sqrt(ref_se0 ** 2 + ref_se1 ** 2))
        ref_z = ref_diff / ref_diff_se if ref_diff_se > 0 else 0.0
        ref_p = float(2 * norm.sf(abs(ref_z)))

        np.testing.assert_allclose(result.diff, ref_diff, rtol=1e-6)
        np.testing.assert_allclose(result.diff_se, ref_diff_se, rtol=1e-4)
        np.testing.assert_allclose(result.z_stat, ref_z, rtol=1e-4)
        np.testing.assert_allclose(result.p_value, ref_p, rtol=1e-4)

    def test_no_difference_simulated(self):
        """Two groups from same distribution -> p > 0.05."""
        rng = np.random.RandomState(42)
        n = 500
        times = rng.exponential(5.0, size=2 * n)
        events = rng.binomial(1, 0.7, size=2 * n).astype(bool)
        groups = np.array([0] * n + [1] * n)

        result = rmst_test(times, events, groups, tau=5.0)
        assert result.p_value > 0.05

    def test_large_difference_simulated(self):
        """Groups with very different hazards -> p < 0.001."""
        rng = np.random.RandomState(42)
        n = 500
        times0 = rng.exponential(20.0, size=n)
        times1 = rng.exponential(1.0, size=n)
        events = np.ones(2 * n, dtype=bool)
        times = np.concatenate([times0, times1])
        groups = np.array([0] * n + [1] * n)

        result = rmst_test(times, events, groups, tau=5.0)
        assert result.p_value < 0.001


@pytest.mark.validation
class TestRMSTCrossLanguageValidation:
    """Validate rmst() and rmst_test() against R survRM2 package output.

    Reference values generated via R 4.5.2 with survRM2 package using
    identical data splits (median split on first numeric column).
    """

    # --- GBSG2 reference values (R survRM2::rmst2) ---
    # Dataset: TH.data::GBSG2, split on age (median=53), tau=2563
    # arm=0: age<=53 (n=363), arm=1: age>53 (n=323)
    GBSG2_TAU = 2563.0
    GBSG2_RMST_0 = 1669.091  # arm=0 (low age)
    GBSG2_SE_0 = 54.381
    GBSG2_RMST_1 = 1591.400  # arm=1 (high age)
    GBSG2_SE_1 = 54.148
    GBSG2_DIFF = -77.691  # arm1 - arm0
    GBSG2_DIFF_SE = 76.742  # sqrt(SE0^2 + SE1^2)
    GBSG2_P = 0.3114

    # --- Veterans reference values (R survRM2::rmst2) ---
    # Dataset: survival::veteran, split on age (median=62), tau=467
    # arm=0: age<=62 (n=74), arm=1: age>62 (n=63)
    VET_TAU = 467.0
    VET_RMST_0 = 132.192  # arm=0 (young)
    VET_SE_0 = 16.717
    VET_RMST_1 = 109.538  # arm=1 (old)
    VET_SE_1 = 14.411
    VET_DIFF = -22.655  # arm1 - arm0
    VET_DIFF_SE = 22.071  # sqrt(SE0^2 + SE1^2)
    VET_P = 0.3047

    def test_gbsg2_rmst_against_r(self, gbsg2_data):
        """GBSG2 RMST point estimates match R survRM2 output."""
        X, y = gbsg2_data
        times, events = _extract_times_events(y)

        X_num = X.select_dtypes(include=[np.number])
        scores = X_num.iloc[:, 0].values
        median_val = np.median(scores)

        low_mask = scores <= median_val
        high_mask = ~low_mask

        tau = float(min(times[low_mask].max(), times[high_mask].max()))
        assert tau == pytest.approx(self.GBSG2_TAU, abs=1)

        rmst_low, se_low = rmst(times[low_mask], events[low_mask], tau=tau)
        rmst_high, se_high = rmst(times[high_mask], events[high_mask], tau=tau)

        # Cross-language tolerance: rtol=1e-3
        np.testing.assert_allclose(rmst_low, self.GBSG2_RMST_0, rtol=1e-3)
        np.testing.assert_allclose(rmst_high, self.GBSG2_RMST_1, rtol=1e-3)
        np.testing.assert_allclose(se_low, self.GBSG2_SE_0, rtol=1e-2)
        np.testing.assert_allclose(se_high, self.GBSG2_SE_1, rtol=1e-2)

    def test_gbsg2_difference_against_r(self, gbsg2_data):
        """GBSG2 RMST difference and p-value match R survRM2."""
        X, y = gbsg2_data
        times, events = _extract_times_events(y)

        X_num = X.select_dtypes(include=[np.number])
        scores = X_num.iloc[:, 0].values
        median_val = np.median(scores)
        groups = (scores > median_val).astype(int)

        tau = float(min(times[groups == 0].max(), times[groups == 1].max()))
        result = rmst_test(times, events, groups, tau=tau)

        np.testing.assert_allclose(result.diff, self.GBSG2_DIFF, rtol=1e-2)
        np.testing.assert_allclose(result.diff_se, self.GBSG2_DIFF_SE, rtol=1e-2)
        np.testing.assert_allclose(result.p_value, self.GBSG2_P, rtol=0.05)

    def test_veterans_rmst_against_r(self, veterans_data):
        """Veterans RMST point estimates match R survRM2 output."""
        X, y = veterans_data
        times, events = _extract_times_events(y)

        X_num = X.select_dtypes(include=[np.number])
        scores = X_num.iloc[:, 0].values
        median_val = np.median(scores)

        low_mask = scores <= median_val
        high_mask = ~low_mask

        tau = float(min(times[low_mask].max(), times[high_mask].max()))
        assert tau == pytest.approx(self.VET_TAU, abs=1)

        rmst_low, se_low = rmst(times[low_mask], events[low_mask], tau=tau)
        rmst_high, se_high = rmst(times[high_mask], events[high_mask], tau=tau)

        np.testing.assert_allclose(rmst_low, self.VET_RMST_0, rtol=1e-3)
        np.testing.assert_allclose(rmst_high, self.VET_RMST_1, rtol=1e-3)
        np.testing.assert_allclose(se_low, self.VET_SE_0, rtol=1e-2)
        np.testing.assert_allclose(se_high, self.VET_SE_1, rtol=1e-2)

    def test_veterans_difference_against_r(self, veterans_data):
        """Veterans RMST difference and p-value match R survRM2."""
        X, y = veterans_data
        times, events = _extract_times_events(y)

        X_num = X.select_dtypes(include=[np.number])
        scores = X_num.iloc[:, 0].values
        median_val = np.median(scores)
        groups = (scores > median_val).astype(int)

        tau = float(min(times[groups == 0].max(), times[groups == 1].max()))
        result = rmst_test(times, events, groups, tau=tau)

        np.testing.assert_allclose(result.diff, self.VET_DIFF, rtol=1e-2)
        np.testing.assert_allclose(result.diff_se, self.VET_DIFF_SE, rtol=1e-2)
        np.testing.assert_allclose(result.p_value, self.VET_P, rtol=0.05)
