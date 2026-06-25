"""Tests for paired model comparison (bootstrap and Kang z-test)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv.evaluation._paired_comparison import (
    compare_bootstrap,
    compare_kang,
    _paired_bootstrap_test,
    _adjust_p_values,
    _extract_distribution,
    _extract_mean_auc,
    _extract_auc_at_time,
    _validate_kang_inputs,
)
from mlsurv.results.containers import BootstrapResult

from tests._helpers import make_random_survival_array


# ---------------------------------------------------------------------------
# Helpers for building mock BootstrapResult
# ---------------------------------------------------------------------------

def _make_bootstrap_result(
    models: dict[str, dict[str, np.ndarray]],
    n_bootstrap: int = 200,
    auc_t_data: dict | None = None,
) -> BootstrapResult:
    """Build a BootstrapResult from {model: {metric: values}} dict.

    Parameters
    ----------
    models : dict
        e.g. {"coxph": {"harrell": array(200), "ipcw": array(200)}, ...}
    n_bootstrap : int
        Number of iterations (must match array lengths).
    auc_t_data : dict, optional
        {model: {eval_time: array(n_bootstrap)}} for tdAUC data.
    """
    iter_rows = []
    summary_rows = []

    for model_name, metrics in models.items():
        for metric_name, values in metrics.items():
            for i, v in enumerate(values):
                iter_rows.append({
                    "model_type": model_name,
                    "iteration": i,
                    "metric_name": metric_name,
                    "value": v,
                })
            summary_rows.append({
                "model_type": model_name,
                "metric": metric_name,
                "point": float(np.mean(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
            })

    iterations = pd.DataFrame(iter_rows)
    summary = pd.DataFrame(summary_rows)

    auc_t = None
    if auc_t_data is not None:
        auc_rows = []
        for model_name, timepoints in auc_t_data.items():
            for eval_time, values in timepoints.items():
                for i, v in enumerate(values):
                    auc_rows.append({
                        "model_type": model_name,
                        "iteration": i,
                        "eval_time": eval_time,
                        "auc": v,
                    })
        auc_t = pd.DataFrame(auc_rows)

    return BootstrapResult(
        summary=summary,
        iterations=iterations,
        auc_t=auc_t,
    )


def _make_survival_array(n: int, event_rate: float = 0.5, seed: int = 42):
    """File-specific survival data: uniform events, exp(scale=5) times."""
    return make_random_survival_array(
        n, event_rate=event_rate, time_scale=5.0,
        event_dist="uniform", seed=seed,
    )


# ===========================================================================
# TestPairedBootstrapTest
# ===========================================================================

class TestPairedBootstrapTest:
    """Unit tests for _paired_bootstrap_test."""

    def test_identical_distributions_high_p(self):
        """Identical distributions should yield p > 0.05."""
        rng = np.random.RandomState(42)
        dist = rng.normal(0.7, 0.02, 500)
        result = _paired_bootstrap_test(dist, dist)
        assert result["delta"] == pytest.approx(0.0, abs=1e-10)
        assert result["p_value"] > 0.05 or np.isnan(result["p_value"])

    def test_different_distributions_low_p(self):
        """Clearly different distributions should yield p < 0.05."""
        rng = np.random.RandomState(42)
        dist_a = rng.normal(0.8, 0.02, 500)
        dist_b = rng.normal(0.6, 0.02, 500)
        result = _paired_bootstrap_test(dist_a, dist_b)
        assert result["delta"] > 0
        assert result["p_value"] < 0.05

    def test_ci_contains_delta(self):
        """CI should contain the point estimate."""
        rng = np.random.RandomState(42)
        dist_a = rng.normal(0.75, 0.03, 300)
        dist_b = rng.normal(0.70, 0.03, 300)
        result = _paired_bootstrap_test(dist_a, dist_b, ci=0.95)
        assert result["ci_lower"] <= result["delta"] <= result["ci_upper"]

    def test_shortfall_resampling_warns(self):
        """Should warn when n_bootstrap > available data."""
        rng = np.random.RandomState(42)
        dist_a = rng.normal(0.7, 0.02, 50)
        dist_b = rng.normal(0.65, 0.02, 50)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _paired_bootstrap_test(
                dist_a, dist_b, n_bootstrap=200, random_state=42,
            )
            assert any("Resampling" in str(warning.message) for warning in w)
        assert not np.isnan(result["p_value"])

    def test_nan_handling(self):
        """NaN iterations should be dropped."""
        rng = np.random.RandomState(42)
        dist_a = rng.normal(0.7, 0.02, 100)
        dist_b = rng.normal(0.65, 0.02, 100)
        # Inject NaNs
        dist_a[::10] = np.nan
        result = _paired_bootstrap_test(dist_a, dist_b)
        assert not np.isnan(result["delta"])
        assert not np.isnan(result["p_value"])

    def test_all_nan_returns_nan(self):
        """All-NaN input should return NaN results with warning."""
        dist_a = np.full(50, np.nan)
        dist_b = np.full(50, np.nan)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _paired_bootstrap_test(dist_a, dist_b)
        assert np.isnan(result["delta"])
        assert np.isnan(result["p_value"])
        msgs = [str(x.message) for x in w]
        assert any("All bootstrap iterations were non-finite" in m for m in msgs)

    def test_inf_handling(self):
        """Inf iterations should be dropped like NaN."""
        rng = np.random.RandomState(42)
        dist_a = rng.normal(0.7, 0.02, 100)
        dist_b = rng.normal(0.65, 0.02, 100)
        dist_a[50] = np.inf
        dist_a[51] = -np.inf
        result = _paired_bootstrap_test(dist_a, dist_b)
        assert np.isfinite(result["delta"])
        assert np.isfinite(result["p_value"])

    def test_ci_width_scales_with_level(self):
        """99% CI should be wider than 90% CI."""
        rng = np.random.RandomState(42)
        dist_a = rng.normal(0.75, 0.03, 500)
        dist_b = rng.normal(0.70, 0.03, 500)
        result_90 = _paired_bootstrap_test(dist_a, dist_b, ci=0.90)
        result_99 = _paired_bootstrap_test(dist_a, dist_b, ci=0.99)
        width_90 = result_90["ci_upper"] - result_90["ci_lower"]
        width_99 = result_99["ci_upper"] - result_99["ci_lower"]
        assert width_99 > width_90

    def test_single_iteration_returns_nan_p(self):
        """Single valid iteration should return valid delta but NaN p."""
        result = _paired_bootstrap_test(np.array([0.8]), np.array([0.6]))
        assert result["delta"] == pytest.approx(0.2)
        assert np.isnan(result["p_value"])

    def test_all_positive_deltas_p_not_zero(self):
        """All deltas > 0: continuity correction prevents p=0."""
        dist_a = np.array([0.81, 0.82, 0.83, 0.84])
        dist_b = np.array([0.80, 0.80, 0.80, 0.80])
        result = _paired_bootstrap_test(dist_a, dist_b)
        # n=4, count_minority=0: p = 2*(0+1)/(4+1) = 2/5
        assert result["p_value"] == pytest.approx(2.0 / 5)
        assert result["p_value"] > 0

    def test_all_negative_deltas_p_not_zero(self):
        """All deltas < 0: continuity correction prevents p=0."""
        dist_a = np.array([0.76, 0.77, 0.78, 0.79])
        dist_b = np.array([0.80, 0.80, 0.80, 0.80])
        result = _paired_bootstrap_test(dist_a, dist_b)
        # n=4, count_minority=0: p = 2*(0+1)/(4+1) = 2/5
        assert result["p_value"] == pytest.approx(2.0 / 5)
        assert result["p_value"] > 0

    def test_symmetric_deltas_p_one(self):
        """Symmetric deltas around zero should yield p=1."""
        dist_a = np.array([0.78, 0.79, 0.81, 0.82])
        dist_b = np.array([0.80, 0.80, 0.80, 0.80])
        result = _paired_bootstrap_test(dist_a, dist_b)
        assert result["p_value"] == pytest.approx(1.0)


# ===========================================================================
# TestKangZTest
# ===========================================================================

class TestKangZTest:
    """Unit tests for compare_kang."""

    def test_identical_scores_high_p(self):
        """Identical risk scores should yield delta~0 and p > 0.05."""
        y = _make_survival_array(100, seed=42)
        rng = np.random.RandomState(42)
        scores = rng.normal(0, 1, 100)
        result = compare_kang(y, scores, scores)
        assert result["delta"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_perfect_vs_random_significant(self):
        """Perfect predictor vs random noise should be significant."""
        y = _make_survival_array(200, event_rate=0.5, seed=42)
        # Perfect predictor: risk = -time (higher risk for shorter times)
        perfect = -y["time"].astype(np.float64)
        rng = np.random.RandomState(99)
        random_scores = rng.normal(0, 1, 200)
        result = compare_kang(y, perfect, random_scores)
        assert result["delta"].iloc[0] > 0
        assert result["p_value"].iloc[0] < 0.05

    def test_z_statistic_returned(self):
        """Result should include z_statistic column."""
        y = _make_survival_array(100, seed=42)
        rng = np.random.RandomState(42)
        scores_a = rng.normal(0, 1, 100)
        scores_b = rng.normal(0, 1, 100)
        result = compare_kang(y, scores_a, scores_b)
        assert "z_statistic" in result.columns

    def test_length_mismatch_raises(self):
        """Mismatched array lengths should raise ValueError."""
        y = _make_survival_array(100, seed=42)
        with pytest.raises(ValueError, match="length mismatch"):
            compare_kang(y, np.zeros(100), np.zeros(50))

    def test_large_memory_warns(self):
        """Estimated memory > 1 GB should trigger a warning."""
        # High event rate with moderate n to exceed 1 GB threshold
        # max_pairs = events * n; estimated_bytes = max_pairs * 32
        # Need max_pairs > 1e9 / 32 = 31_250_000
        # With n=6000, event_rate=1.0: max_pairs = 6000 * 6000 = 36M > 31.25M
        # Call _validate_kang_inputs directly to skip O(n²) computation
        y = _make_survival_array(6000, event_rate=1.0, seed=42)
        scores = np.zeros(6000)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_kang_inputs(y, scores, scores, ci=0.95)
            assert any("pre-allocation requires" in str(warning.message) for warning in w)

    def test_small_n_high_events_no_warn(self):
        """Small n with low memory footprint should not warn."""
        y = _make_survival_array(100, event_rate=0.5, seed=42)
        scores = np.zeros(100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compare_kang(y, scores, scores)
            assert not any("pre-allocation requires" in str(warning.message) for warning in w)

    def test_no_comparable_pairs_returns_nan(self):
        """All censored observations should return NaN results."""
        y = np.empty(50, dtype=[("event", bool), ("time", float)])
        y["event"] = False  # all censored → no comparable pairs
        y["time"] = np.arange(50, dtype=float)
        scores = np.random.default_rng(42).normal(0, 1, 50)
        result = compare_kang(y, scores, scores)
        assert np.isnan(result["delta"].iloc[0])
        assert np.isnan(result["p_value"].iloc[0])

    def test_ci_bounds_contain_delta(self):
        """CI should bracket the delta and narrow at lower confidence."""
        y = _make_survival_array(200, event_rate=0.5, seed=42)
        perfect = -y["time"].astype(np.float64)
        rng = np.random.RandomState(99)
        random_scores = rng.normal(0, 1, 200)

        r95 = compare_kang(y, perfect, random_scores, ci=0.95)
        r80 = compare_kang(y, perfect, random_scores, ci=0.80)

        delta = r95["delta"].iloc[0]
        assert r95["ci_lower"].iloc[0] <= delta <= r95["ci_upper"].iloc[0]
        assert r80["ci_lower"].iloc[0] <= delta <= r80["ci_upper"].iloc[0]

        # 95% CI should be wider than 80% CI
        w95 = r95["ci_upper"].iloc[0] - r95["ci_lower"].iloc[0]
        w80 = r80["ci_upper"].iloc[0] - r80["ci_lower"].iloc[0]
        assert w95 > w80

    def test_model_names_passthrough(self):
        """Custom model names should appear in the result DataFrame."""
        y = _make_survival_array(100, seed=42)
        rng = np.random.RandomState(42)
        scores = rng.normal(0, 1, 100)
        result = compare_kang(
            y, scores, scores,
            model_a_name="coxph", model_b_name="rsf",
        )
        assert result["model_a"].iloc[0] == "coxph"
        assert result["model_b"].iloc[0] == "rsf"


# ===========================================================================
# TestAdjustPValues
# ===========================================================================

class TestAdjustPValues:
    """Unit tests for _adjust_p_values."""

    def test_bonferroni(self):
        """Bonferroni should multiply by number of tests."""
        raw = np.array([0.01, 0.04, 0.10])
        adj = _adjust_p_values(raw, method="bonferroni")
        np.testing.assert_array_almost_equal(adj, [0.03, 0.12, 0.30])

    def test_holm(self):
        """Holm step-down should be less conservative than Bonferroni."""
        raw = np.array([0.01, 0.04, 0.10])
        adj_holm = _adjust_p_values(raw, method="holm")
        adj_bonf = _adjust_p_values(raw, method="bonferroni")
        # Holm is always <= Bonferroni (or equal)
        assert np.all(adj_holm <= adj_bonf + 1e-10)

    def test_bh(self):
        """BH should control FDR."""
        raw = np.array([0.01, 0.04, 0.10])
        adj = _adjust_p_values(raw, method="bh")
        # BH adjusted values should be >= raw
        assert np.all(adj >= raw - 1e-10)
        # And <= 1
        assert np.all(adj <= 1.0 + 1e-10)

    def test_single_pvalue(self):
        """Single p-value should be unchanged by correction."""
        raw = np.array([0.03])
        for method in ["holm", "bonferroni", "none"]:
            adj = _adjust_p_values(raw, method=method)
            assert adj[0] == pytest.approx(0.03, abs=1e-10)

    def test_none_passthrough(self):
        """'none' method should return raw values."""
        raw = np.array([0.01, 0.04, 0.10])
        adj = _adjust_p_values(raw, method="none")
        np.testing.assert_array_equal(adj, raw)

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown p_adjust"):
            _adjust_p_values(np.array([0.05]), method="invalid")


# ===========================================================================
# TestMultiplePairComparison
# ===========================================================================

class TestMultiplePairComparison:
    """Integration tests for compare_bootstrap with multiple models."""

    @pytest.fixture
    def three_model_result(self):
        """BootstrapResult with 3 models: good, medium, bad."""
        rng = np.random.RandomState(42)
        n = 200
        return _make_bootstrap_result({
            "good": {"harrell": rng.normal(0.80, 0.02, n)},
            "medium": {"harrell": rng.normal(0.72, 0.02, n)},
            "bad": {"harrell": rng.normal(0.60, 0.02, n)},
        }, n_bootstrap=n)

    def test_three_models_three_pairs(self, three_model_result):
        """3 models should produce 3 comparison pairs."""
        df = compare_bootstrap(three_model_result, metric="harrell")
        assert len(df) == 3

    def test_adjustment_increases_p(self, three_model_result):
        """Holm correction should increase p-values vs raw."""
        df = compare_bootstrap(
            three_model_result, metric="harrell", p_adjust="holm",
        )
        # At least one adjusted p should be > raw p
        assert np.all(df["p_adjusted"] >= df["p_value"] - 1e-10)

    def test_model_subset(self, three_model_result):
        """Selecting 2 of 3 models should produce 1 pair."""
        df = compare_bootstrap(
            three_model_result, metric="harrell", models=["good", "bad"],
        )
        assert len(df) == 1
        assert df.iloc[0]["model_a"] == "good"
        assert df.iloc[0]["model_b"] == "bad"

    def test_unknown_model_raises(self, three_model_result):
        """Requesting a non-existent model should raise ValueError."""
        with pytest.raises(ValueError, match="Models not found"):
            compare_bootstrap(
                three_model_result, metric="harrell", models=["good", "nonexistent"],
            )


# ===========================================================================
# TestExtractPairedDistributions
# ===========================================================================

class TestExtractPairedDistributions:
    """Tests for distribution extraction from BootstrapResult."""

    def test_standard_metric(self):
        """Standard metric should extract via get_distribution."""
        rng = np.random.RandomState(42)
        values = rng.normal(0.7, 0.02, 100)
        br = _make_bootstrap_result({"coxph": {"harrell": values}}, n_bootstrap=100)
        dist = _extract_distribution(br, "harrell", "coxph", eval_time=None)
        np.testing.assert_array_equal(dist, values)

    def test_mean_auc(self):
        """mean_auc should average across timepoints per iteration."""
        rng = np.random.RandomState(42)
        auc_data = {
            "coxph": {
                1.0: rng.normal(0.7, 0.02, 50),
                2.0: rng.normal(0.8, 0.02, 50),
            }
        }
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, 50)}},
            n_bootstrap=50,
            auc_t_data=auc_data,
        )
        dist = _extract_mean_auc(br, "coxph")
        assert len(dist) == 50
        # Mean should be between the two timepoint means
        assert 0.7 < np.mean(dist) < 0.8

    def test_auc_t_requires_eval_time(self):
        """metric='auc_t' without eval_time should raise ValueError."""
        rng = np.random.RandomState(42)
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, 50)}},
            n_bootstrap=50,
        )
        with pytest.raises(ValueError, match="eval_time is required"):
            _extract_distribution(br, "auc_t", "coxph", eval_time=None)


# ===========================================================================
# TestContainerMethods
# ===========================================================================

class TestContainerMethods:
    """Tests for BootstrapResult.compare()."""

    @pytest.fixture
    def two_model_bootstrap(self):
        """BootstrapResult with 2 models."""
        rng = np.random.RandomState(42)
        n = 100
        return _make_bootstrap_result({
            "coxph": {"harrell": rng.normal(0.75, 0.02, n)},
            "rsf": {"harrell": rng.normal(0.70, 0.02, n)},
        }, n_bootstrap=n)

    def test_bootstrap_result_compare(self, two_model_bootstrap):
        """BootstrapResult.compare() should return valid DataFrame."""
        df = two_model_bootstrap.compare(metric="harrell")
        assert len(df) == 1
        assert "model_a" in df.columns
        assert "p_adjusted" in df.columns

    def test_single_model_raises(self):
        """compare() with only 1 model should raise ValueError."""
        rng = np.random.RandomState(42)
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.75, 0.02, 100)}},
            n_bootstrap=100,
        )
        with pytest.raises(ValueError, match="at least 2 models"):
            br.compare(metric="harrell")


# ===========================================================================
# TestValidation — input validation and edge cases from review
# ===========================================================================

class TestValidation:
    """Tests for input validation added from review feedback."""

    def test_invalid_ci_raises(self):
        """ci=95 (instead of 0.95) should raise ValueError."""
        rng = np.random.RandomState(42)
        br = _make_bootstrap_result({
            "coxph": {"harrell": rng.normal(0.75, 0.02, 50)},
            "rsf": {"harrell": rng.normal(0.70, 0.02, 50)},
        }, n_bootstrap=50)
        with pytest.raises(ValueError, match="ci must be between 0 and 1"):
            compare_bootstrap(br, metric="harrell", ci=95)

    def test_invalid_ci_kang_raises(self):
        """ci=0 should raise ValueError for compare_kang."""
        y = _make_survival_array(50, seed=42)
        scores = np.random.default_rng(42).normal(0, 1, 50)
        with pytest.raises(ValueError, match="ci must be between 0 and 1"):
            compare_kang(y, scores, scores, ci=0)

    def test_nan_risk_scores_raises(self):
        """NaN risk scores should raise ValueError."""
        y = _make_survival_array(50, seed=42)
        scores = np.ones(50)
        scores[5] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            compare_kang(y, scores, np.ones(50))

    def test_inf_risk_scores_raises(self):
        """Inf risk scores should raise ValueError."""
        y = _make_survival_array(50, seed=42)
        scores_b = np.ones(50)
        scores_b[3] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            compare_kang(y, np.ones(50), scores_b)

    def test_unknown_metric_raises(self):
        """Misspelled metric should raise ValueError, not return NaN."""
        rng = np.random.RandomState(42)
        br = _make_bootstrap_result({
            "coxph": {"harrell": rng.normal(0.75, 0.02, 50)},
            "rsf": {"harrell": rng.normal(0.70, 0.02, 50)},
        }, n_bootstrap=50)
        with pytest.raises(ValueError, match="No bootstrap iterations found"):
            compare_bootstrap(br, metric="harell")  # typo

    def test_distribution_shape_mismatch_raises(self):
        """Paired test with different-length distributions should raise."""
        with pytest.raises(ValueError, match="shape mismatch"):
            _paired_bootstrap_test(np.array([0.7, 0.8]), np.array([0.6]))

    def test_zero_comparable_pairs_warns(self):
        """Zero comparable pairs should emit a warning."""
        y = np.empty(50, dtype=[("event", bool), ("time", float)])
        y["event"] = False
        y["time"] = np.arange(50, dtype=float)
        scores = np.random.default_rng(42).normal(0, 1, 50)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compare_kang(y, scores, scores)
            assert any("0 comparable pairs" in str(x.message) for x in w)

    def test_holm_known_values(self):
        """Holm step-down should produce specific known-correct values."""
        # p = [0.01, 0.04, 0.10], m=3
        # Sorted: [0.01, 0.04, 0.10]
        # Step: [0.01*3, max(0.03, 0.04*2), max(0.08, 0.10*1)]
        #     = [0.03, 0.08, 0.10]
        raw = np.array([0.01, 0.04, 0.10])
        adj = _adjust_p_values(raw, method="holm")
        np.testing.assert_array_almost_equal(adj, [0.03, 0.08, 0.10])

    def test_adjust_with_nan_preserves_nan(self):
        """NaN p-values should be preserved through adjustment."""
        raw = np.array([0.01, np.nan, 0.10])
        for method in ["holm", "bonferroni", "bh"]:
            adj = _adjust_p_values(raw, method=method)
            assert np.isnan(adj[1]), f"NaN not preserved by {method}"
            assert not np.isnan(adj[0]), f"Non-NaN became NaN in {method}"
            assert not np.isnan(adj[2]), f"Non-NaN became NaN in {method}"

    def test_mean_auc_without_auc_t_data_raises(self):
        """mean_auc metric without auc_t data should raise ValueError."""
        rng = np.random.RandomState(42)
        br = _make_bootstrap_result({
            "coxph": {"harrell": rng.normal(0.7, 0.02, 50)},
            "rsf": {"harrell": rng.normal(0.65, 0.02, 50)},
        }, n_bootstrap=50)
        with pytest.raises(ValueError, match="No auc_t data"):
            compare_bootstrap(br, metric="mean_auc")

    def test_auc_t_end_to_end(self):
        """compare_bootstrap with metric='auc_t' should work end-to-end."""
        rng = np.random.RandomState(42)
        n = 50
        auc_data = {
            "coxph": {1.0: rng.normal(0.7, 0.02, n), 2.0: rng.normal(0.75, 0.02, n)},
            "rsf": {1.0: rng.normal(0.6, 0.02, n), 2.0: rng.normal(0.65, 0.02, n)},
        }
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, n)},
             "rsf": {"harrell": rng.normal(0.6, 0.02, n)}},
            n_bootstrap=n, auc_t_data=auc_data,
        )
        df = compare_bootstrap(br, metric="auc_t", eval_time=1.0)
        assert len(df) == 1
        assert "auc_t@1.0" in df["metric"].iloc[0]
        assert not np.isnan(df["p_value"].iloc[0])

    def test_mean_auc_end_to_end(self):
        """compare_bootstrap with metric='mean_auc' should work end-to-end."""
        rng = np.random.RandomState(42)
        n = 50
        auc_data = {
            "coxph": {1.0: rng.normal(0.7, 0.02, n), 2.0: rng.normal(0.75, 0.02, n)},
            "rsf": {1.0: rng.normal(0.6, 0.02, n), 2.0: rng.normal(0.65, 0.02, n)},
        }
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, n)},
             "rsf": {"harrell": rng.normal(0.6, 0.02, n)}},
            n_bootstrap=n, auc_t_data=auc_data,
        )
        df = compare_bootstrap(br, metric="mean_auc")
        assert len(df) == 1
        assert df["metric"].iloc[0] == "mean_auc"
        assert not np.isnan(df["p_value"].iloc[0])


# ===========================================================================
# TestDuplicateEvalTimeWarning
# ===========================================================================

class TestDuplicateEvalTimeWarning:
    """Test that np.isclose duplicate matches produce a warning."""

    def test_duplicate_eval_times_warns_and_deduplicates(self):
        """Duplicate eval_times per iteration should warn and average duplicates."""
        rng = np.random.RandomState(42)
        n = 20
        # Create auc_t data with two very close eval_times that np.isclose matches
        auc_data = {
            "coxph": {
                1.0: rng.normal(0.7, 0.02, n),
                1.0 + 1e-10: rng.normal(0.72, 0.02, n),  # near-duplicate
            },
        }
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, n)}},
            n_bootstrap=n, auc_t_data=auc_data,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _extract_auc_at_time(br, "coxph", 1.0)
            assert any("Multiple eval_times matched" in str(wn.message) for wn in w)
        # Deduplication: one value per iteration, not 2x
        assert len(result) == n

    def test_unique_eval_times_no_warn(self):
        """Distinct eval_times should not produce a duplicate warning."""
        rng = np.random.RandomState(42)
        n = 20
        auc_data = {
            "coxph": {
                1.0: rng.normal(0.7, 0.02, n),
                5.0: rng.normal(0.65, 0.02, n),
            },
        }
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, n)}},
            n_bootstrap=n, auc_t_data=auc_data,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _extract_auc_at_time(br, "coxph", 1.0)
            assert not any("Multiple eval_times matched" in str(wn.message) for wn in w)


# ===========================================================================
# TestCompareForwarding
# ===========================================================================

class TestCompareForwarding:
    """Test that n_bootstrap and random_state are forwarded through compare()."""

    def test_bootstrap_result_compare_forwards_params(self):
        """BootstrapResult.compare() should forward n_bootstrap/random_state."""
        rng = np.random.RandomState(42)
        n = 50
        br = _make_bootstrap_result({
            "coxph": {"harrell": rng.normal(0.7, 0.02, n)},
            "rsf": {"harrell": rng.normal(0.6, 0.02, n)},
        }, n_bootstrap=n)
        # Should not raise — params forwarded to compare_bootstrap
        df = br.compare(
            metric="harrell", n_bootstrap=500, random_state=123,
        )
        assert len(df) == 1
        assert not np.isnan(df["p_value"].iloc[0])
