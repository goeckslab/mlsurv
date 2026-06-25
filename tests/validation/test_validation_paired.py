"""Validation tests for paired model comparison against independent reference.

Validates compare_kang() using a pure-Python reference implementation that
directly follows the Kang et al. (2015) algorithm.  Also checks cross-method
consistency between bootstrap and Kang approaches.

Reference
---------
Kang L, Chen W, Petrick NA, Gallas BD (2015).
    Comparing two correlated C indices with right-censored survival outcome:
    a one-shot nonparametric approach.
    Statistics in Medicine, 34(4), 685-703.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mlsurv.evaluation import compare_kang
from mlsurv.evaluation._paired_comparison import _paired_bootstrap_test
from mlsurv.utils import to_structured_array


# ---------------------------------------------------------------------------
# Pure-Python reference implementation (Kang et al. 2015)
# ---------------------------------------------------------------------------
# This is intentionally written in a different style from the production code
# (double-nested loops, no vectorisation) so it serves as a truly independent
# cross-check.


def _ref_concordance_c(events, times, scores):
    """Compute Harrell's C-index via brute-force pair enumeration."""
    n = len(events)
    concordant = 0.0
    total = 0
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if times[i] >= times[j]:
                continue
            total += 1
            if scores[i] > scores[j]:
                concordant += 1.0
            elif scores[i] == scores[j]:
                concordant += 0.5
    if total == 0:
        return np.nan
    return concordant / total


def _ref_kang_test(events, times, scores_a, scores_b):
    """Reference Kang z-test: pure Python, O(n^2) pair enumeration.

    Returns (c_a, c_b, delta, z_statistic, p_value).
    """
    n = len(events)

    # --- Step 1: enumerate comparable pairs and concordance indicators ---
    h_a = []
    h_b = []
    pair_i_list = []
    pair_j_list = []

    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if times[i] >= times[j]:
                continue
            # Concordance indicator for model A
            if scores_a[i] > scores_a[j]:
                ha = 1.0
            elif scores_a[i] < scores_a[j]:
                ha = 0.0
            else:
                ha = 0.5
            # Concordance indicator for model B
            if scores_b[i] > scores_b[j]:
                hb = 1.0
            elif scores_b[i] < scores_b[j]:
                hb = 0.0
            else:
                hb = 0.5
            h_a.append(ha)
            h_b.append(hb)
            pair_i_list.append(i)
            pair_j_list.append(j)

    h_a = np.array(h_a)
    h_b = np.array(h_b)
    pair_i = np.array(pair_i_list)
    pair_j = np.array(pair_j_list)
    n_pairs = len(h_a)

    if n_pairs == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    c_a = np.mean(h_a)
    c_b = np.mean(h_b)
    delta = c_a - c_b

    # --- Step 2: Hoeffding decomposition for Var(delta_C) ---
    d = h_a - h_b

    # Per-observation kernel means
    kernel_sums = np.zeros(n)
    kernel_counts = np.zeros(n)
    for k in range(n_pairs):
        i_idx = pair_i[k]
        j_idx = pair_j[k]
        kernel_sums[i_idx] += d[k]
        kernel_sums[j_idx] += d[k]
        kernel_counts[i_idx] += 1
        kernel_counts[j_idx] += 1

    active = kernel_counts > 0
    if active.sum() < 2:
        return c_a, c_b, delta, np.nan, np.nan

    kernel_means = np.zeros(n)
    kernel_means[active] = kernel_sums[active] / kernel_counts[active]

    var_kernels = np.var(kernel_means, ddof=1)
    var_delta = 4.0 * var_kernels / n

    if var_delta <= 0 or np.isnan(var_delta):
        return c_a, c_b, delta, np.nan, np.nan

    se = np.sqrt(var_delta)
    z_stat = delta / se
    p_value = 2.0 * norm.sf(np.abs(z_stat))

    return c_a, c_b, delta, z_stat, p_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_dataset(raw_data):
    """Encode categoricals and convert y to SurvivalArray."""
    X, y_raw = raw_data
    X_enc = pd.get_dummies(X, drop_first=True)
    y = to_structured_array(y_raw)
    return X_enc, y


def _fit_coxph(X, y, alpha=0.01):
    """Fit CoxPH and return risk scores."""
    from mlsurv.models import CoxPHModel
    model = CoxPHModel(alpha=alpha)
    model.fit(X, y)
    return model.predict(X)


def _fit_rsf(X, y, random_state=42, n_estimators=50):
    """Fit RSF and return risk scores."""
    from mlsurv.models import RSFModel
    model = RSFModel(
        n_estimators=n_estimators, random_state=random_state, n_jobs=1,
    )
    model.fit(X, y)
    return model.predict(X)


def _survival_arrays(y):
    """Extract boolean events and float64 times from SurvivalArray."""
    events = y["event"].astype(bool)
    times = y["time"].astype(np.float64)
    return events, times


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@pytest.mark.validation
class TestKangValidation:
    """Validate compare_kang() against pure-Python reference implementation."""

    def test_gbsg2_coxph_vs_rsf(self, gbsg2_data):
        """Two real models on GBSG2 — z-stat and p-value match reference."""
        X, y = _prepare_dataset(gbsg2_data)
        scores_a = _fit_coxph(X, y)
        scores_b = _fit_rsf(X, y)

        result = compare_kang(y, scores_a, scores_b)
        events, times = _survival_arrays(y)
        _, _, ref_delta, ref_z, ref_p = _ref_kang_test(
            events, times, scores_a, scores_b,
        )

        np.testing.assert_allclose(result["delta"].iloc[0], ref_delta, rtol=1e-8)
        np.testing.assert_allclose(result["z_statistic"].iloc[0], ref_z, rtol=1e-6)
        np.testing.assert_allclose(result["p_value"].iloc[0], ref_p, rtol=1e-6)

    def test_gbsg2_two_coxph_subsets(self, gbsg2_data):
        """CoxPH on different feature subsets — second independent check."""
        X, y = _prepare_dataset(gbsg2_data)
        cols = list(X.columns)

        # Split features into two overlapping subsets
        mid = len(cols) // 2
        subset_a = cols[:mid + 2]
        subset_b = cols[mid:]

        scores_a = _fit_coxph(X[subset_a], y)
        scores_b = _fit_coxph(X[subset_b], y)

        result = compare_kang(y, scores_a, scores_b)
        events, times = _survival_arrays(y)
        _, _, ref_delta, ref_z, ref_p = _ref_kang_test(
            events, times, scores_a, scores_b,
        )

        np.testing.assert_allclose(result["delta"].iloc[0], ref_delta, rtol=1e-8)
        np.testing.assert_allclose(result["z_statistic"].iloc[0], ref_z, rtol=1e-6)
        np.testing.assert_allclose(result["p_value"].iloc[0], ref_p, rtol=1e-6)

    def test_veterans_dataset(self, veterans_data):
        """Veterans lung cancer (137 pts) — small dataset check."""
        X, y = _prepare_dataset(veterans_data)
        scores_a = _fit_coxph(X, y)
        scores_b = _fit_rsf(X, y, n_estimators=30)

        result = compare_kang(y, scores_a, scores_b)
        events, times = _survival_arrays(y)
        _, _, ref_delta, ref_z, ref_p = _ref_kang_test(
            events, times, scores_a, scores_b,
        )

        np.testing.assert_allclose(result["delta"].iloc[0], ref_delta, rtol=1e-8)
        np.testing.assert_allclose(result["z_statistic"].iloc[0], ref_z, rtol=1e-6)
        np.testing.assert_allclose(result["p_value"].iloc[0], ref_p, rtol=1e-6)

    def test_identical_scores(self, gbsg2_data):
        """Identical scores → delta=0, p≈1 for both implementations."""
        X, y = _prepare_dataset(gbsg2_data)
        scores = _fit_coxph(X, y)

        result = compare_kang(y, scores, scores)
        assert result["delta"].iloc[0] == 0.0
        # Degenerate variance → NaN z/p (zero numerator and zero denominator)
        assert np.isnan(result["z_statistic"].iloc[0])
        assert np.isnan(result["p_value"].iloc[0])

        events, times = _survival_arrays(y)
        _, _, ref_delta, ref_z, ref_p = _ref_kang_test(
            events, times, scores, scores,
        )
        assert ref_delta == 0.0
        assert np.isnan(ref_z)
        assert np.isnan(ref_p)

    def test_perfect_vs_random(self, gbsg2_data):
        """Perfect predictor vs random noise — large delta, significant."""
        _, y = _prepare_dataset(gbsg2_data)
        rng = np.random.RandomState(42)

        # Perfect predictor: negative time (higher risk for shorter survival)
        perfect = -y["time"].astype(np.float64)
        random_scores = rng.randn(len(y))

        result = compare_kang(y, perfect, random_scores)
        events, times = _survival_arrays(y)
        _, _, ref_delta, ref_z, ref_p = _ref_kang_test(
            events, times, perfect, random_scores,
        )

        # Both implementations agree on direction and significance
        np.testing.assert_allclose(result["delta"].iloc[0], ref_delta, rtol=1e-8)
        np.testing.assert_allclose(result["z_statistic"].iloc[0], ref_z, rtol=1e-6)
        np.testing.assert_allclose(result["p_value"].iloc[0], ref_p, rtol=1e-6)

        # Perfect predictor should dominate random
        our_delta = result["delta"].iloc[0]
        our_p = result["p_value"].iloc[0]
        assert our_delta > 0.3, f"Expected large delta, got {our_delta}"
        assert our_p < 0.001, f"Expected p<0.001, got {our_p}"

    def test_reference_c_index_matches(self, gbsg2_data):
        """C-index from Kang internals matches brute-force computation."""
        X, y = _prepare_dataset(gbsg2_data)
        scores = _fit_coxph(X, y)

        events, times = _survival_arrays(y)
        ref_c = _ref_concordance_c(events, times, scores)

        # Kang reference returns c_a as first value
        c_a, _, _, _, _ = _ref_kang_test(events, times, scores, scores)

        np.testing.assert_allclose(c_a, ref_c, rtol=1e-12)


@pytest.mark.validation
class TestCrossMethodConsistency:
    """Bootstrap and Kang should broadly agree on real data."""

    @pytest.fixture
    def gbsg2_scores(self, gbsg2_data):
        """Fit two models and return (y, scores_a, scores_b)."""
        X, y = _prepare_dataset(gbsg2_data)
        scores_a = _fit_coxph(X, y)
        scores_b = _fit_rsf(X, y)
        return y, scores_a, scores_b

    @staticmethod
    def _bootstrap_c_distributions(y, scores_a, scores_b, n_boot=500, seed=0):
        """Generate paired bootstrap C-index distributions.

        Returns (c_a_dist, c_b_dist) and asserts < 5% NaN rate.
        """
        rng = np.random.RandomState(seed)
        n = len(y)
        events, times = _survival_arrays(y)

        c_a_dist = np.empty(n_boot)
        c_b_dist = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            c_a_dist[b] = _ref_concordance_c(
                events[idx], times[idx], scores_a[idx],
            )
            c_b_dist[b] = _ref_concordance_c(
                events[idx], times[idx], scores_b[idx],
            )

        # Guard: too many NaN iterations would inflate CI and weaken tests
        nan_count = np.isnan(c_a_dist).sum() + np.isnan(c_b_dist).sum()
        assert nan_count < n_boot * 0.05, (
            f"Too many NaN bootstrap iterations: {nan_count}/{n_boot}"
        )
        return c_a_dist, c_b_dist

    def test_bootstrap_ci_contains_kang_delta(self, gbsg2_scores):
        """Kang point estimate falls within bootstrap CI."""
        y, scores_a, scores_b = gbsg2_scores

        kang_result = compare_kang(y, scores_a, scores_b)
        kang_delta = kang_result["delta"].iloc[0]

        c_a_dist, c_b_dist = self._bootstrap_c_distributions(
            y, scores_a, scores_b,
        )
        boot_result = _paired_bootstrap_test(c_a_dist, c_b_dist, ci=0.95)

        assert boot_result["ci_lower"] <= kang_delta <= boot_result["ci_upper"], (
            f"Kang delta {kang_delta:.4f} outside bootstrap CI "
            f"[{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}]"
        )

    def test_direction_agreement(self, gbsg2_scores):
        """Both methods agree on sign of delta."""
        y, scores_a, scores_b = gbsg2_scores

        kang_result = compare_kang(y, scores_a, scores_b)
        kang_delta = kang_result["delta"].iloc[0]

        c_a_dist, c_b_dist = self._bootstrap_c_distributions(
            y, scores_a, scores_b,
        )
        boot_result = _paired_bootstrap_test(c_a_dist, c_b_dist)

        if abs(kang_delta) > 0.01 and abs(boot_result["delta"]) > 0.01:
            assert np.sign(kang_delta) == np.sign(boot_result["delta"]), (
                f"Direction mismatch: Kang delta={kang_delta:.4f}, "
                f"bootstrap delta={boot_result['delta']:.4f}"
            )
        else:
            # Both near zero — that's valid agreement (no meaningful diff)
            assert abs(kang_delta) < 0.05, (
                f"Kang delta {kang_delta:.4f} is non-trivial but bootstrap "
                f"delta {boot_result['delta']:.4f} is near zero"
            )
            assert abs(boot_result["delta"]) < 0.05, (
                f"Bootstrap delta {boot_result['delta']:.4f} is non-trivial "
                f"but Kang delta {kang_delta:.4f} is near zero"
            )

    def test_significance_agreement_clear_case(self, gbsg2_data):
        """For perfect vs random, both methods report p < 0.05."""
        _, y = _prepare_dataset(gbsg2_data)
        rng = np.random.RandomState(42)

        perfect = -y["time"].astype(np.float64)
        random_scores = rng.randn(len(y))

        kang_result = compare_kang(y, perfect, random_scores)
        kang_p = kang_result["p_value"].iloc[0]

        c_a_dist, c_b_dist = self._bootstrap_c_distributions(
            y, perfect, random_scores, seed=42,
        )
        boot_result = _paired_bootstrap_test(c_a_dist, c_b_dist)
        boot_p = boot_result["p_value"]

        assert kang_p < 0.05, f"Kang p={kang_p:.4f}, expected < 0.05"
        assert boot_p < 0.05, f"Bootstrap p={boot_p:.4f}, expected < 0.05"
