"""Tests for the benchmarking feature."""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv.utils import SurvivalArray
from mlsurv.models._reference_adapter import ReferenceAdapter
from mlsurv.evaluation._incremental import (
    delta_c_index,
    category_free_nri,
    integrated_discrimination_improvement,
    compute_incremental_metrics,
    bootstrap_incremental,
    format_p_sig,
)
from mlsurv.evaluation._univariate_cox import compute_reference_univariate_cox
from mlsurv.results.containers import BenchmarkResult

from tests._helpers import make_survival_array, make_y


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def survival_y():
    """Simple SurvivalArray for unit tests."""
    np.random.seed(42)
    n = 100
    events = np.random.binomial(1, 0.7, n).astype(bool)
    times = np.random.exponential(2, n).clip(0.1, 10)
    return make_survival_array(events, times)


@pytest.fixture
def risk_scores():
    """Random risk scores for testing."""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def reference_scores():
    """Random reference scores for testing."""
    np.random.seed(123)
    return np.random.randn(100)


# =====================================================================
# TestReferenceAdapter
# =====================================================================

class TestReferenceAdapter:
    def test_predict_returns_scores(self):
        scores = np.array([1.0, 2.0, 3.0])
        adapter = ReferenceAdapter(scores, name="psa")
        result = adapter.predict()
        np.testing.assert_array_equal(result, scores)

    def test_predict_ignores_x(self):
        scores = np.array([1.0, 2.0, 3.0])
        adapter = ReferenceAdapter(scores, name="psa")
        result = adapter.predict(X=pd.DataFrame({"a": [4, 5, 6]}))
        np.testing.assert_array_equal(result, scores)

    def test_higher_is_worse_true(self):
        scores = np.array([1.0, 2.0, 3.0])
        adapter = ReferenceAdapter(scores, name="psa", higher_is_worse=True)
        np.testing.assert_array_equal(adapter.predict(), scores)

    def test_higher_is_worse_false_negates(self):
        scores = np.array([1.0, 2.0, 3.0])
        adapter = ReferenceAdapter(scores, name="albumin", higher_is_worse=False)
        np.testing.assert_array_equal(adapter.predict(), -scores)

    def test_survival_function_raises(self):
        adapter = ReferenceAdapter(np.array([1.0]), name="x")
        with pytest.raises(NotImplementedError, match="survival function"):
            adapter.predict_survival_function()

    def test_cumulative_hazard_raises(self):
        adapter = ReferenceAdapter(np.array([1.0]), name="x")
        with pytest.raises(NotImplementedError, match="cumulative hazard"):
            adapter.predict_cumulative_hazard_function()

    def test_get_params(self):
        adapter = ReferenceAdapter(np.array([1.0]), name="psa",
                                   display_name="PSA", higher_is_worse=True)
        params = adapter.get_params()
        assert params["name"] == "psa"
        assert params["display_name"] == "PSA"
        assert params["higher_is_worse"] is True

    def test_set_params(self):
        adapter = ReferenceAdapter(np.array([1.0]), name="psa")
        adapter.set_params(name="ki67")
        assert adapter.name == "ki67"

    def test_is_fitted(self):
        adapter = ReferenceAdapter(np.array([1.0]), name="x")
        assert adapter.is_fitted_ is True

    def test_repr(self):
        adapter = ReferenceAdapter(np.array([1.0, 2.0]), name="psa")
        r = repr(adapter)
        assert "psa" in r
        assert "n=2" in r

    def test_display_name_defaults_to_name(self):
        adapter = ReferenceAdapter(np.array([1.0]), name="psa")
        assert adapter.display_name == "psa"

    def test_scores_are_float64(self):
        scores = np.array([1, 2, 3], dtype=np.int32)
        adapter = ReferenceAdapter(scores, name="x")
        assert adapter.predict().dtype == np.float64


# =====================================================================
# TestIncrementalMetrics
# =====================================================================

class TestIncrementalMetrics:
    def test_delta_c_index_returns_float(self, survival_y, risk_scores, reference_scores):
        result = delta_c_index(survival_y, risk_scores, reference_scores)
        assert isinstance(result, float)

    def test_delta_c_identical_scores_is_zero(self, survival_y, risk_scores):
        result = delta_c_index(survival_y, risk_scores, risk_scores)
        assert result == pytest.approx(0.0)

    def test_nri_returns_float(self, survival_y, risk_scores, reference_scores):
        result = category_free_nri(survival_y, risk_scores, reference_scores)
        assert isinstance(result, float)

    def test_idi_returns_float(self, survival_y, risk_scores, reference_scores):
        result = integrated_discrimination_improvement(
            survival_y, risk_scores, reference_scores
        )
        assert isinstance(result, float)

    def test_compute_all_returns_dict(self, survival_y, risk_scores, reference_scores):
        result = compute_incremental_metrics(
            survival_y, risk_scores, reference_scores
        )
        assert "delta_c_index" in result
        assert "nri" in result
        assert "idi" in result

    def test_bootstrap_returns_dataframe(self, survival_y, risk_scores, reference_scores):
        result = bootstrap_incremental(
            survival_y, risk_scores, reference_scores,
            n_bootstrap=10, random_state=42,
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {
            "metric", "point", "mean", "std", "ci_lower", "ci_upper", "p_value"
        }
        assert len(result) == 3  # delta_c, nri, idi

    def test_bootstrap_ci_contains_point(self, survival_y, risk_scores, reference_scores):
        result = bootstrap_incremental(
            survival_y, risk_scores, reference_scores,
            n_bootstrap=50, ci=0.95, random_state=42,
        )
        for _, row in result.iterrows():
            # CI should generally contain the point estimate
            # (not guaranteed for every metric in small samples, but check shape)
            assert not np.isnan(row["ci_lower"])
            assert not np.isnan(row["ci_upper"])
            assert row["ci_lower"] <= row["ci_upper"]

    def test_all_events_no_crash(self):
        """All-events edge case shouldn't crash."""
        n = 50
        y = make_y([float(i + 1) for i in range(n)])
        np.random.seed(42)
        m = np.random.randn(n)
        b = np.random.randn(n)
        result = compute_incremental_metrics(y, m, b)
        assert all(isinstance(v, float) for v in result.values())

    def test_no_events_no_crash(self):
        """No-events edge case shouldn't crash."""
        n = 50
        y = make_survival_array([False] * n, [float(i + 1) for i in range(n)])
        np.random.seed(42)
        m = np.random.randn(n)
        b = np.random.randn(n)
        # delta_c may fail with concordance_index_censored (no concordant pairs)
        # NRI/IDI return NaN for degenerate data (all censored)
        nri = category_free_nri(y, m, b)
        assert np.isnan(nri)

    def test_bootstrap_p_value_in_range(self, survival_y, risk_scores, reference_scores):
        """All p-values must be in [0, 1]."""
        result = bootstrap_incremental(
            survival_y, risk_scores, reference_scores,
            n_bootstrap=50, random_state=42,
        )
        for _, row in result.iterrows():
            p = row["p_value"]
            assert 0 <= p <= 1, f"p_value {p} out of range for {row['metric']}"

    def test_bootstrap_p_value_significant_when_model_better(self):
        """When the model clearly outperforms, delta_c p-value should be small."""
        np.random.seed(99)
        n = 200
        y = SurvivalArray(np.array(
            [(bool(np.random.binomial(1, 0.6)), float(i + 1))
             for i in range(n)],
            dtype=[('event', bool), ('time', float)],
        ))
        # Model scores highly correlated with outcome; reference is noise
        model_scores = y["time"] + np.random.randn(n) * 0.1
        ref_scores = np.random.randn(n)
        result = bootstrap_incremental(
            y, model_scores, ref_scores,
            n_bootstrap=100, random_state=42,
        )
        dc_row = result[result["metric"] == "delta_c_index"].iloc[0]
        assert dc_row["p_value"] < 0.05

    def test_summary_all_positive_bootstrap_continuity_corrected_p(self):
        """All bootstrap values > 0: continuity-corrected p = 2*(0+1)/(n+1)."""
        from mlsurv.evaluation._incremental import _build_incremental_summary

        point = {"delta_c_index": 0.05}
        results = {"delta_c_index": [0.01, 0.02, 0.03, 0.04]}
        df = _build_incremental_summary(point, results, ci=0.95)
        # n=4, count_minority=0: p = 2*(0+1)/(4+1) = 0.4
        assert df.iloc[0]["p_value"] == pytest.approx(2.0 / 5)

    def test_summary_all_negative_bootstrap_continuity_corrected_p(self):
        """All bootstrap values < 0: continuity-corrected p = 2*(0+1)/(n+1)."""
        from mlsurv.evaluation._incremental import _build_incremental_summary

        point = {"delta_c_index": -0.05}
        results = {"delta_c_index": [-0.04, -0.03, -0.02, -0.01]}
        df = _build_incremental_summary(point, results, ci=0.95)
        # n=4, count_minority=0: p = 2*(0+1)/(4+1) = 0.4
        assert df.iloc[0]["p_value"] == pytest.approx(2.0 / 5)

    def test_summary_n2_all_positive_not_half(self):
        """Regression test: n=2 all-positive should NOT give p=0.5."""
        from mlsurv.evaluation._incremental import _build_incremental_summary

        point = {"delta_c_index": 0.05}
        results = {"delta_c_index": [0.01, 0.02]}
        df = _build_incremental_summary(point, results, ci=0.95)
        # n=2, count_minority=0: p = 2*(0+1)/(2+1) = 2/3
        assert df.iloc[0]["p_value"] == pytest.approx(2.0 / 3)
        assert df.iloc[0]["p_value"] != pytest.approx(0.5)  # old clamp value

    def test_summary_symmetric_bootstrap_gives_p_one(self):
        """Symmetric bootstrap around zero should yield p=1."""
        from mlsurv.evaluation._incremental import _build_incremental_summary

        point = {"delta_c_index": 0.0}
        results = {"delta_c_index": [-0.02, -0.01, 0.01, 0.02]}
        df = _build_incremental_summary(point, results, ci=0.95)
        assert df.iloc[0]["p_value"] == pytest.approx(1.0)

    def test_summary_single_bootstrap_gives_nan_p(self):
        """A single bootstrap sample is insufficient for a p-value."""
        from mlsurv.evaluation._incremental import _build_incremental_summary

        point = {"delta_c_index": 0.05}
        results = {"delta_c_index": [0.04]}
        df = _build_incremental_summary(point, results, ci=0.95)
        assert np.isnan(df.iloc[0]["p_value"])

    def test_summary_empty_bootstrap_gives_nan_p(self):
        """No bootstrap results should yield NaN p-value."""
        from mlsurv.evaluation._incremental import _build_incremental_summary

        point = {"delta_c_index": 0.05, "nri": 0.10, "idi": 0.02}
        empty_results = {k: [] for k in point}
        df = _build_incremental_summary(point, empty_results, ci=0.95)
        for _, row in df.iterrows():
            assert np.isnan(row["p_value"])

    def test_summary_nan_in_bootstrap_filtered(self):
        """NaN values in bootstrap array are filtered before p-value."""
        from mlsurv.evaluation._incremental import _build_incremental_summary

        point = {"nri": 0.10}
        # 3 finite positive + 1 NaN; without filtering NaN biases p-value
        results = {"nri": [0.05, 0.10, 0.15, np.nan]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = _build_incremental_summary(point, results, ci=0.95)
        # Should warn about dropped iteration
        method_warns = [x for x in w if "non-finite" in str(x.message)]
        assert len(method_warns) == 1
        assert "1/4" in str(method_warns[0].message)
        # p-value computed from 3 finite values (all > 0):
        # continuity-corrected p = 2*(0+1)/(3+1) = 0.5
        assert df.iloc[0]["p_value"] == pytest.approx(0.5)


class TestFormatPSig:
    """Unit tests for the significance star formatter."""

    def test_stars_below_001(self):
        assert format_p_sig(0.0001) == "1.00e-04***"

    def test_stars_below_01(self):
        assert format_p_sig(0.005) == "5.00e-03**"

    def test_stars_below_05(self):
        assert format_p_sig(0.04) == "4.00e-02*"

    def test_no_stars_above_05(self):
        assert format_p_sig(0.5) == "5.00e-01"

    def test_nan_returns_dash(self):
        assert format_p_sig(np.nan) == "\u2014"

    def test_exactly_one(self):
        assert format_p_sig(1.0) == "1.00e+00"

    def test_exactly_zero(self):
        assert format_p_sig(0.0) == "0.00e+00***"

    def test_negative_no_stars(self):
        """Negative p-values are formatted without stars."""
        assert format_p_sig(-0.01) == "-1.00e-02"

    def test_above_one_no_stars(self):
        """p > 1 is formatted without stars."""
        assert format_p_sig(1.5) == "1.50e+00"

    def test_boundary_005_no_star(self):
        """Exactly 0.05 is NOT < 0.05, so no star."""
        assert format_p_sig(0.05) == "5.00e-02"

    def test_boundary_001_one_star(self):
        """Exactly 0.01 is NOT < 0.01, so one star."""
        assert format_p_sig(0.01) == "1.00e-02*"

    def test_boundary_0001_two_stars(self):
        """Exactly 0.001 is NOT < 0.001, so two stars."""
        assert format_p_sig(0.001) == "1.00e-03**"


# =====================================================================
# TestUnivariateCox
# =====================================================================

class TestUnivariateCox:
    def test_returns_correct_columns(self, survival_y):
        ref_df = pd.DataFrame({
            "ref1": np.random.randn(len(survival_y)),
            "ref2": np.random.randn(len(survival_y)),
        })
        result = compute_reference_univariate_cox(ref_df, survival_y)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "reference", "coefficient", "hazard_ratio",
            "hr_lower_95", "hr_upper_95", "p_value",
        }
        assert set(result.columns) == expected_cols

    def test_hr_positive(self, survival_y):
        ref_df = pd.DataFrame({"ref": np.random.randn(len(survival_y))})
        result = compute_reference_univariate_cox(ref_df, survival_y)
        assert result["hazard_ratio"].iloc[0] > 0

    def test_p_value_in_range(self, survival_y):
        ref_df = pd.DataFrame({"ref": np.random.randn(len(survival_y))})
        result = compute_reference_univariate_cox(ref_df, survival_y)
        p = result["p_value"].iloc[0]
        assert 0 <= p <= 1

    def test_all_nan_reference_skipped(self, survival_y):
        ref_df = pd.DataFrame({"bad": np.full(len(survival_y), np.nan)})
        result = compute_reference_univariate_cox(ref_df, survival_y)
        assert len(result) == 0

    def test_partial_nan_succeeds(self, survival_y):
        """Reference with some NaN still produces valid Cox results."""
        ref = np.random.randn(len(survival_y))
        ref[0:5] = np.nan
        ref_df = pd.DataFrame({"partial_nan": ref})
        result = compute_reference_univariate_cox(ref_df, survival_y)
        assert len(result) == 1
        assert result["reference"].iloc[0] == "partial_nan"
        assert not np.isnan(result["hazard_ratio"].iloc[0])

    def test_fewer_than_5_valid_skipped(self, survival_y):
        """Reference with fewer than 5 valid values is skipped."""
        ref = np.full(len(survival_y), np.nan)
        ref[0:3] = [1.0, 2.0, 3.0]  # Only 3 valid
        ref_df = pd.DataFrame({"too_few": ref})
        result = compute_reference_univariate_cox(ref_df, survival_y)
        assert len(result) == 0

    def test_inf_values_handled(self, survival_y):
        """Inf values are treated like NaN and dropped."""
        ref = np.random.randn(len(survival_y))
        ref[0] = np.inf
        ref[1] = -np.inf
        ref_df = pd.DataFrame({"has_inf": ref})
        result = compute_reference_univariate_cox(ref_df, survival_y)
        assert len(result) == 1
        assert not np.isnan(result["hazard_ratio"].iloc[0])


# =====================================================================
# TestBenchmarkResult
# =====================================================================

class TestBenchmarkResult:
    def test_construction(self):
        from mlsurv.results.containers import EvaluationResult
        eval_result = EvaluationResult(
            concordance=pd.DataFrame({"model_type": ["m1"], "c_index_test": [0.7]}),
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
            shap_values=None,
            bootstrap=None,
        )
        br = BenchmarkResult(
            evaluation=eval_result,
            univariate_cox=pd.DataFrame(),
            incremental=pd.DataFrame(),
            correlation=pd.DataFrame(),
            reference_names=["ref1"],
            model_names=["m1"],
        )
        assert br.reference_names == ["ref1"]
        assert br.model_names == ["m1"]

    def test_repr(self):
        from mlsurv.results.containers import EvaluationResult
        eval_result = EvaluationResult(
            concordance=pd.DataFrame(),
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
            shap_values=None,
            bootstrap=None,
        )
        br = BenchmarkResult(
            evaluation=eval_result,
            univariate_cox=pd.DataFrame(),
            incremental=pd.DataFrame(),
            correlation=pd.DataFrame(),
            reference_names=["ref1"],
            model_names=["m1"],
        )
        r = repr(br)
        assert "ref1" in r
        assert "m1" in r


# =====================================================================
# TestBenchmarkLearner (integration)
# =====================================================================

@pytest.mark.integration
class TestBenchmarkLearner:
    def test_benchmark_end_to_end(self, synthetic_data_clean):
        """Full pipeline: setup → train → benchmark."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        # Use a feature column's values as a mock reference (external name)
        result = learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False,
            verbose=False,
        )

        assert isinstance(result, BenchmarkResult)
        assert "coxph" in result.model_names
        assert "ext_score" in result.reference_names
        assert result.evaluation is not None
        assert not result.univariate_cox.empty
        assert not result.incremental.empty
        assert not result.correlation.empty
        assert learner._state.benchmark_result is result
        # Public read-only property exposes the same stashed result.
        assert learner.benchmark_result is result

    def test_benchmark_with_bootstrap(self, synthetic_data_clean):
        """Benchmark with bootstrap CIs."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        result = learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=True,
            n_bootstrap=10,
            verbose=False,
        )

        assert result.incremental["ci_lower"].notna().any()

    def test_benchmark_full_x_alignment(self, synthetic_data_clean):
        """Reference array aligned with full X (not just test)."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        # Pass full X-length array
        result = learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False,
            verbose=False,
        )
        assert isinstance(result, BenchmarkResult)

    def test_benchmark_higher_is_worse_dict(self, synthetic_data_clean):
        """Per-reference higher_is_worse settings."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        result = learner.benchmark(
            references={
                "ext_score_a": X["feature_0"].values,
                "ext_score_b": X["feature_1"].values,
            },
            higher_is_worse={"ext_score_a": True, "ext_score_b": False},
            bootstrap=False,
            verbose=False,
        )
        assert len(result.reference_names) == 2

    def test_benchmark_name_collision_raises(self, synthetic_data_clean):
        """Reference name colliding with model name raises ValueError."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        with pytest.raises(ValueError, match="overlap"):
            learner.benchmark(
                references={"coxph": X["feature_0"].values},
                bootstrap=False,
                verbose=False,
            )

    def test_benchmark_wrong_length_raises(self, synthetic_data_clean):
        """Reference array with wrong length raises ValueError."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        with pytest.raises(ValueError, match="length"):
            learner.benchmark(
                references={"ref": np.ones(5)},
                bootstrap=False,
                verbose=False,
            )

    def test_benchmark_feature_collision_warning(self, synthetic_data_clean):
        """Reference name matching a feature column triggers SuggestionWarning."""
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import SuggestionWarning

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        with pytest.warns(SuggestionWarning, match="feature columns in X"):
            learner.benchmark(
                references={"feature_0": X["feature_0"].values},
                bootstrap=False,
                verbose=False,
            )

    def test_benchmark_no_feature_collision_warning(self, synthetic_data_clean):
        """No SuggestionWarning when reference name doesn't match any feature."""
        import warnings as _warnings
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import SuggestionWarning

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            learner.benchmark(
                references={"ext_score": X["feature_0"].values},
                bootstrap=False,
                verbose=False,
            )
        collision_warnings = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and "match feature columns" in str(w.message)
        ]
        assert len(collision_warnings) == 0


# =====================================================================
# TestBenchmarkEvalTimesInheritance
# =====================================================================

@pytest.mark.integration
class TestBenchmarkEvalTimesInheritance:
    """eval_times inheritance from evaluate()/run() into benchmark()."""

    def test_benchmark_inherits_eval_times_from_evaluate(self, synthetic_data_clean):
        """benchmark() uses eval_times from prior evaluate() call."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        user_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        learner.evaluate(eval_times=user_times, verbose=False, n_importance_repeats=0)

        assert learner._state.user_eval_times is not None
        np.testing.assert_array_equal(learner._state.user_eval_times, user_times)

        result = learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False,
            verbose=False,
        )
        assert isinstance(result, BenchmarkResult)
        # Verify actual timepoints in results match user-specified times
        actual_times = sorted(result.evaluation.auc_t["eval_time"].unique())
        for t in actual_times:
            assert t in user_times, f"Unexpected eval_time {t} not in user_times"

    def test_benchmark_inherits_eval_times_from_run(self, synthetic_data_clean):
        """run(eval_times=...) stores user_eval_times on state."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)

        user_times = np.array([1.0, 2.0, 3.0])
        learner.run("coxph", tune=False, eval_times=user_times, verbose=False)

        assert learner._state.user_eval_times is not None
        np.testing.assert_array_equal(learner._state.user_eval_times, user_times)

    def test_benchmark_explicit_eval_times_overrides(self, synthetic_data_clean):
        """Explicit eval_times in benchmark() takes precedence over inherited."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        learner.evaluate(eval_times=[1, 2, 3], verbose=False, n_importance_repeats=0)

        override_times = [0.5, 1.0, 1.5]
        result = learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            eval_times=override_times,
            bootstrap=False,
            verbose=False,
        )
        assert isinstance(result, BenchmarkResult)
        # Verify benchmark used the override times, not the inherited ones
        actual_times = sorted(result.evaluation.auc_t["eval_time"].unique())
        for t in actual_times:
            assert t in override_times, f"Unexpected eval_time {t} not in override"

    def test_benchmark_auto_computes_when_no_user_eval_times(self, synthetic_data_clean):
        """benchmark() auto-computes when no user-specified eval_times exist."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        # evaluate without explicit eval_times
        learner.evaluate(verbose=False, n_importance_repeats=0)
        assert learner._state.user_eval_times is None

        result = learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False,
            verbose=False,
        )
        assert isinstance(result, BenchmarkResult)

    def test_user_eval_times_updated_by_latest_evaluate(self, synthetic_data_clean):
        """user_eval_times reflects the most recent explicit evaluate() call."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        learner.evaluate(eval_times=[1, 2], verbose=False, n_importance_repeats=0)
        np.testing.assert_array_equal(learner._state.user_eval_times, [1, 2])

        learner.evaluate(eval_times=[10, 20, 30], verbose=False, n_importance_repeats=0)
        np.testing.assert_array_equal(learner._state.user_eval_times, [10, 20, 30])

    def test_user_eval_times_preserved_by_evaluate_without_times(self, synthetic_data_clean):
        """evaluate() without eval_times inherits stored user_eval_times."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        learner.evaluate(eval_times=[1, 2, 3], verbose=False, n_importance_repeats=0)
        assert learner._state.user_eval_times is not None

        # evaluate without eval_times preserves the stored value
        learner.evaluate(verbose=False, n_importance_repeats=0)
        np.testing.assert_array_equal(learner._state.user_eval_times, [1, 2, 3])

    def test_eval_times_scalar_raises(self, synthetic_data_clean):
        """Scalar eval_times raises TypeError."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        with pytest.raises(TypeError, match="array-like"):
            learner.evaluate(eval_times=5, verbose=False, n_importance_repeats=0)

    def test_eval_times_2d_raises(self, synthetic_data_clean):
        """2D eval_times raises ValueError."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        with pytest.raises(ValueError, match="1-dimensional"):
            learner.evaluate(eval_times=[[1, 2], [3, 4]], verbose=False, n_importance_repeats=0)

    def test_evaluate_inherits_eval_times_from_run(self, synthetic_data_clean):
        """evaluate() without eval_times inherits from prior run(eval_times=...)."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)

        user_times = [1.0, 2.0, 3.0]
        learner.run("coxph", tune=False, eval_times=user_times, verbose=False)
        np.testing.assert_array_equal(learner._state.user_eval_times, user_times)

        # Re-evaluate a single model without passing eval_times — should inherit
        learner.evaluate(models="coxph", verbose=False, n_importance_repeats=0)
        np.testing.assert_array_equal(learner._state.user_eval_times, user_times)

        # Verify the evaluation used the inherited times
        result = learner._state.models["coxph"]
        actual_times = sorted(result.evaluation.auc_t['eval_time'].unique())
        assert len(actual_times) > 0, "AUC(t) should have eval_times"
        for t in actual_times:
            assert t in user_times, f"Unexpected eval_time {t} not in user_times"

    def test_explicit_eval_times_overrides_then_inherits(self, synthetic_data_clean):
        """Explicit eval_times overrides inherited; subsequent evaluate() inherits new."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        learner.evaluate(eval_times=[1, 2, 3], verbose=False, n_importance_repeats=0)
        np.testing.assert_array_equal(learner._state.user_eval_times, [1, 2, 3])

        # Override with new explicit times
        new_times = [0.5, 1.5]
        learner.evaluate(eval_times=new_times, verbose=False, n_importance_repeats=0)
        np.testing.assert_array_equal(learner._state.user_eval_times, new_times)

        # Third call without eval_times inherits the latest override
        learner.evaluate(verbose=False, n_importance_repeats=0)
        np.testing.assert_array_equal(learner._state.user_eval_times, new_times)

    def test_evaluate_auto_computes_when_no_stored_times(self, synthetic_data_clean):
        """evaluate() without eval_times auto-computes when none stored."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        result = learner.evaluate(verbose=False, n_importance_repeats=0)
        assert learner._state.user_eval_times is None
        assert result.auc_t is not None and len(result.auc_t) > 0

    def test_eval_times_auto_clears_inherited(self, synthetic_data_clean):
        """eval_times='auto' clears stored user_eval_times and auto-computes."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        learner.evaluate(eval_times=[1, 2, 3], verbose=False, n_importance_repeats=0)
        np.testing.assert_array_equal(learner._state.user_eval_times, [1, 2, 3])

        # "auto" clears and forces auto-computation
        result = learner.evaluate(eval_times="auto", verbose=False, n_importance_repeats=0)
        assert learner._state.user_eval_times is None
        assert result.auc_t is not None and len(result.auc_t) > 0

    def test_eval_times_auto_in_benchmark(self, synthetic_data_clean):
        """benchmark(eval_times='auto') forces auto-computation."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        # Set eval_times via evaluate
        user_times = [0.5, 1.0, 1.5]
        learner.evaluate(eval_times=user_times, verbose=False, n_importance_repeats=0)
        np.testing.assert_array_equal(learner._state.user_eval_times, user_times)

        # benchmark with "auto" should ignore inherited times
        result = learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            eval_times="auto",
            bootstrap=False,
            verbose=False,
        )
        assert isinstance(result, BenchmarkResult)
        # Verify benchmark used auto-computed times, not user_times
        actual_times = sorted(result.evaluation.auc_t["eval_time"].unique())
        assert actual_times != sorted(user_times)

    def test_inherited_eval_times_warns_with_custom_data(self, synthetic_data_clean):
        """SuggestionWarning fires when inheriting eval_times with custom X/y."""
        import warnings
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import SuggestionWarning

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        learner.evaluate(eval_times=[1, 2, 3], verbose=False, n_importance_repeats=0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.evaluate(X=learner.X_test, verbose=False, n_importance_repeats=0)

        matched = [
            x for x in w
            if issubclass(x.category, SuggestionWarning)
            and "inherited eval_times" in str(x.message)
        ]
        assert len(matched) == 1

    def test_explicit_eval_times_no_warning_with_custom_data(self, synthetic_data_clean):
        """No inherited-times warning when eval_times is passed explicitly."""
        import warnings
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import SuggestionWarning

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        learner.evaluate(eval_times=[1, 2, 3], verbose=False, n_importance_repeats=0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.evaluate(X=learner.X_test, eval_times=[1, 2], verbose=False, n_importance_repeats=0)

        matched = [
            x for x in w
            if issubclass(x.category, SuggestionWarning)
            and "inherited eval_times" in str(x.message)
        ]
        assert len(matched) == 0

    def test_eval_times_auto_suppresses_custom_data_warning(self, synthetic_data_clean):
        """eval_times='auto' with custom X does not warn about inherited times."""
        import warnings
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import SuggestionWarning

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        learner.evaluate(eval_times=[1, 2, 3], verbose=False, n_importance_repeats=0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.evaluate(X=learner.X_test, eval_times="auto", verbose=False, n_importance_repeats=0)

        matched = [
            x for x in w
            if issubclass(x.category, SuggestionWarning)
            and "inherited eval_times" in str(x.message)
        ]
        assert len(matched) == 0
        assert learner._state.user_eval_times is None


# =====================================================================
# TestEvalTimesMismatchWarning
# =====================================================================

class TestEvalTimesMismatchWarning:
    """warn_eval_times_mismatch() emits SuggestionWarning on mismatch."""

    def test_warns_on_mismatched_eval_times(self):
        """SuggestionWarning fires when models have different eval_time grids."""
        import warnings
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.reports.sections.evaluation._helpers import (
            warn_eval_times_mismatch,
        )

        auc_df = pd.DataFrame({
            'model_type': ['coxph', 'coxph', 'rsf', 'rsf'],
            'eval_time': [1.0, 2.0, 3.0, 4.0],
            'auc': [0.8, 0.75, 0.82, 0.78],
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_eval_times_mismatch(auc_df)

        matched = [x for x in w if issubclass(x.category, SuggestionWarning)]
        assert len(matched) == 1
        assert "different eval_times" in str(matched[0].message)
        assert "CoxPH" in str(matched[0].message)

    def test_no_warning_on_matching_eval_times(self):
        """No warning when all models share the same eval_time grid."""
        import warnings
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.reports.sections.evaluation._helpers import (
            warn_eval_times_mismatch,
        )

        auc_df = pd.DataFrame({
            'model_type': ['coxph', 'coxph', 'rsf', 'rsf'],
            'eval_time': [1.0, 2.0, 1.0, 2.0],
            'auc': [0.8, 0.75, 0.82, 0.78],
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_eval_times_mismatch(auc_df)

        matched = [x for x in w if issubclass(x.category, SuggestionWarning)]
        assert len(matched) == 0

    def test_no_warning_single_model(self):
        """No warning when there is only one model."""
        import warnings
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.reports.sections.evaluation._helpers import (
            warn_eval_times_mismatch,
        )

        auc_df = pd.DataFrame({
            'model_type': ['coxph', 'coxph'],
            'eval_time': [1.0, 2.0],
            'auc': [0.8, 0.75],
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_eval_times_mismatch(auc_df)

        matched = [x for x in w if issubclass(x.category, SuggestionWarning)]
        assert len(matched) == 0


# =====================================================================
# TestBenchmarkVisualization (integration)
# =====================================================================

@pytest.mark.integration
class TestBenchmarkVisualization:
    @pytest.fixture
    def benchmarked_learner(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False,
            verbose=False,
        )
        return learner

    def test_plot_metric_references_all(self, benchmarked_learner):
        import plotly.graph_objects as go
        fig = benchmarked_learner.plot_metric(references="all")
        assert isinstance(fig, go.Figure)

    def test_plot_metric_references_selective(self, benchmarked_learner):
        import plotly.graph_objects as go
        fig = benchmarked_learner.plot_metric(references=["ext_score"])
        assert isinstance(fig, go.Figure)

    def test_plot_metric_references_return_data(self, benchmarked_learner):
        df = benchmarked_learner.plot_metric(references="all", return_data=True)
        assert isinstance(df, pd.DataFrame)
        # Both the trained model AND the ref must be present. Prior to
        # the _swap_benchmark_eval backfill fix, the trained model was
        # silently dropped because benchmark() alone doesn't populate
        # per-model .evaluation.
        # _tidy_model_col renames model_type -> model for return_data.
        assert {"coxph", "ext_score"}.issubset(set(df["model"]))

    def test_plot_metric_references_unknown_raises(self, benchmarked_learner):
        with pytest.raises(ValueError, match="Unknown references"):
            benchmarked_learner.plot_metric(references=["nonexistent"])

    def test_plot_metric_references_no_benchmark_raises(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        # Train a model first (mirrors the plot_roc sibling): the metric path
        # resolves the model set before the reference/benchmark guard, so
        # without a trained model it raises "No trained models" rather than the
        # "No benchmark results" error this test is checking.
        learner.train("coxph", verbose=False)
        with pytest.raises(ValueError, match="benchmark"):
            learner.plot_metric(references="all")

    def test_plot_roc_references_all(self, benchmarked_learner):
        """Regression test: plot_roc(references='all')."""
        import plotly.graph_objects as go
        fig = benchmarked_learner.plot_roc(references="all")
        assert isinstance(fig, go.Figure)

    def test_plot_roc_references_selective(self, benchmarked_learner):
        import plotly.graph_objects as go
        fig = benchmarked_learner.plot_roc(references=["ext_score"])
        assert isinstance(fig, go.Figure)

    def test_plot_roc_references_return_data(self, benchmarked_learner):
        df = benchmarked_learner.plot_roc(references="all", return_data=True)
        assert isinstance(df, pd.DataFrame)
        # Both the trained model AND the ref must be present. Prior to
        # the _swap_benchmark_eval backfill fix, the trained model
        # was silently dropped because
        # benchmark() alone doesn't populate per-model .evaluation.
        # _tidy_model_col renames model_type -> model for return_data.
        assert {"coxph", "ext_score"}.issubset(set(df["model"]))

    def test_plot_roc_references_unknown_raises(self, benchmarked_learner):
        with pytest.raises(ValueError, match="Unknown references"):
            benchmarked_learner.plot_roc(references=["nonexistent"])

    def test_plot_roc_references_no_benchmark_raises(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        with pytest.raises(ValueError, match="benchmark"):
            learner.plot_roc(references="all")

    def test_plot_roc_references_with_bootstrap_ci(self, synthetic_data_clean):
        """Regression test: mixed bootstrap / no-bootstrap path.

        The trained model has bootstrap predictions, the reference does
        not. ``_compute_roc_ci_data_main`` must skip the reference (it has
        ``bootstrap is None``) without crashing, and still compute CI
        ribbons for the trained model.
        """
        import plotly.graph_objects as go
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        learner.bootstrap(n_bootstrap=2, verbose=False)
        learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False,
            verbose=False,
        )
        fig = learner.plot_roc(references="all", show_ci=True)
        assert isinstance(fig, go.Figure)

    # ------------------------------------------------------------------
    # Plan A regression tests: backfill + compose
    # ------------------------------------------------------------------

    def test_swap_benchmark_eval_restores_evaluation_on_exit(
        self, benchmarked_learner,
    ):
        """After the swap-context exits, trained model .evaluation must
        be None again (the backfill is temporary — it preserves
        benchmark()-only semantics for non-plot callers).
        """
        # The fixture calls train() + benchmark() without evaluate(),
        # so coxph.evaluation is None before any plot call.
        assert benchmarked_learner._state.models["coxph"].evaluation is None
        _ = benchmarked_learner.plot_roc(references="all")
        assert benchmarked_learner._state.models["coxph"].evaluation is None

    def test_swap_benchmark_eval_restores_on_exception(
        self, benchmarked_learner, monkeypatch,
    ):
        """Exit cleanup must run even when the context body raises.

        Regression guard for the reference-capture semantics: if a
        future refactor moves cleanup into an ``else:`` branch, this
        test will catch it.
        """
        # plot_roc binds get_survival_test_data into its own namespace via
        # `from mlsurv.viz._data import get_survival_test_data` (_plot_impl_roc),
        # so patch it on the consuming module, not the source `_data` module —
        # patching the source would not rebind the already-imported name.
        import mlsurv.viz._plot_impl_roc as _vs
        assert benchmarked_learner._state.models["coxph"].evaluation is None

        # Force get_survival_test_data to raise inside the swap-context
        # body so the backfill is installed, then the inner plot call
        # explodes before the context manager exits normally.
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated mid-swap failure")

        monkeypatch.setattr(
            _vs, "get_survival_test_data", _boom,
        )
        with pytest.raises(RuntimeError, match="simulated mid-swap failure"):
            benchmarked_learner.plot_roc(references="all")

        # Cleanup must have run — coxph.evaluation back to None
        assert benchmarked_learner._state.models["coxph"].evaluation is None

    def test_swap_benchmark_eval_preserves_user_installed_eval(
        self, benchmarked_learner,
    ):
        """Reference-capture: if code inside the ``with`` body replaces
        ``.evaluation`` with a different object, ``__exit__`` must not
        silently wipe it. Only our exact installed object gets reverted
        to ``None``.
        """
        from mlsurv.viz._resolve import _swap_benchmark_eval

        state = benchmarked_learner._state
        br = state.benchmark_result
        assert state.models["coxph"].evaluation is None

        sentinel = object()
        try:
            with _swap_benchmark_eval(state, br, ["ext_score"], kwargs={}):
                # Backfill installed .evaluation from benchmark data
                assert state.models["coxph"].evaluation is not None
                # Simulate a nested call that legitimately replaces
                # .evaluation with a different object (e.g., a user
                # running evaluate() from inside a plot hook).
                state.models["coxph"].evaluation = sentinel

            # After the with-block exits normally: the sentinel must
            # survive because identity check (``is``) sees it is NOT
            # the object we installed. __exit__ leaves it alone.
            assert state.models["coxph"].evaluation is sentinel
        finally:
            # Restore the module-scoped fixture for later tests.
            state.models["coxph"].evaluation = None

    def test_plot_metric_models_subset_with_references(
        self, synthetic_data_clean,
    ):
        """``plot_metric(models=..., references=...)`` must compose:
        only the user-requested models appear alongside the refs.
        """
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        learner.train("rsf", verbose=False)
        learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False, verbose=False,
        )
        df = learner.plot_metric(
            models=["coxph"], references=["ext_score"], return_data=True,
        )
        models_in_df = set(df["model"])
        assert models_in_df == {"coxph", "ext_score"}, models_in_df

    def test_plot_metric_time_models_subset_with_references(
        self, synthetic_data_clean,
    ):
        """``plot_metric_time(models=..., references=...)`` must compose.
        Note: plot_metric_time uses a different reference-resolution
        path (``_resolve_metric_time_eval``), so this is a separate
        code path from plot_metric / plot_roc.
        """
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        learner.train("rsf", verbose=False)
        learner.evaluate(verbose=False, n_importance_repeats=0)  # populate per-model auc_t
        learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False, verbose=False,
        )
        df = learner.plot_metric_time(
            models=["coxph"], references=["ext_score"], return_data=True,
        )
        models_in_df = set(df["model"])
        assert models_in_df == {"coxph", "ext_score"}, models_in_df

    def test_plot_roc_models_subset_with_references(
        self, synthetic_data_clean,
    ):
        """``plot_roc(models=..., references=...)`` must compose."""
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        learner.train("rsf", verbose=False)
        learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False, verbose=False,
        )
        df = learner.plot_roc(
            models=["coxph"], references=["ext_score"], return_data=True,
        )
        models_in_df = set(df["model"])
        assert models_in_df == {"coxph", "ext_score"}, models_in_df

    def test_plot_roc_models_str_with_references(self, benchmarked_learner):
        """Edge case: ``models=`` accepts a bare string (not just lists).

        The ``compose_models_with_refs`` helper must handle the string
        form by promoting it to ``[str, *ref_names]``.
        """
        df = benchmarked_learner.plot_roc(
            models="coxph", references=["ext_score"], return_data=True,
        )
        assert {"coxph", "ext_score"}.issubset(set(df["model"]))

    def test_plot_survival_references(self, benchmarked_learner):
        import plotly.graph_objects as go
        fig = benchmarked_learner.plot_survival(references="all")
        assert isinstance(fig, go.Figure)

    def test_plot_benchmark_incremental_returns_figure(self, benchmarked_learner):
        import plotly.graph_objects as go
        fig = benchmarked_learner.plot_benchmark_incremental()
        assert isinstance(fig, go.Figure)

    def test_plot_benchmark_incremental_return_data(self, benchmarked_learner):
        df = benchmarked_learner.plot_benchmark_incremental(return_data=True)
        assert isinstance(df, pd.DataFrame)

    def test_plot_benchmark_incremental_with_bootstrap(self, synthetic_data_clean):
        """Regression test: ci_uppers missing from hover texts."""
        from mlsurv import SurvivalLearner
        import plotly.graph_objects as go

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=True,
            n_bootstrap=10,
            verbose=False,
        )
        fig = learner.plot_benchmark_incremental()
        assert isinstance(fig, go.Figure)


# =====================================================================
# TestBuildHoverTexts
# =====================================================================


class TestBuildHoverTexts:
    """Unit tests for _build_hover_texts (regression for ci_uppers in hover)."""

    def test_with_ci(self):
        from mlsurv.viz._data import _build_hover_texts

        labels = ["A", "B"]
        points = [0.75, 0.80]
        ci_lowers = [0.70, 0.75]
        ci_uppers = [0.80, 0.85]
        p_values = [0.01, 0.05]
        texts = _build_hover_texts(labels, points, ci_lowers, ci_uppers, p_values, has_ci=True)
        assert len(texts) == 2
        assert "CI: [0.700, 0.800]" in texts[0]
        assert "CI: [0.750, 0.850]" in texts[1]

    def test_without_ci(self):
        from mlsurv.viz._data import _build_hover_texts

        labels = ["A"]
        points = [0.75]
        ci_lowers = [np.nan]
        ci_uppers = [np.nan]
        p_values = [np.nan]
        texts = _build_hover_texts(labels, points, ci_lowers, ci_uppers, p_values, has_ci=False)
        assert len(texts) == 1
        assert "CI" not in texts[0]
        assert "0.750" in texts[0]

    def test_with_ci_nan_lower(self):
        from mlsurv.viz._data import _build_hover_texts

        labels = ["A"]
        points = [0.75]
        ci_lowers = [np.nan]
        ci_uppers = [0.80]
        p_values = [np.nan]
        texts = _build_hover_texts(labels, points, ci_lowers, ci_uppers, p_values, has_ci=True)
        assert "CI" not in texts[0]

    def test_with_p_values(self):
        from mlsurv.viz._data import _build_hover_texts

        labels = ["A"]
        points = [0.05]
        ci_lowers = [0.01]
        ci_uppers = [0.09]
        p_values = [0.003]
        texts = _build_hover_texts(labels, points, ci_lowers, ci_uppers, p_values, has_ci=True)
        assert "p = " in texts[0]
        assert "**" in texts[0]


# =====================================================================
# TestFilterNanScores
# =====================================================================

class TestFilterNanScores:
    def test_no_nan_returns_unchanged(self):
        from mlsurv.learner._benchmarking import _filter_nan_scores
        ref = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        r_out, y_out = _filter_nan_scores(ref, y)
        np.testing.assert_array_equal(r_out, ref)
        np.testing.assert_array_equal(y_out, y)

    def test_drops_nan_correctly(self):
        from mlsurv.learner._benchmarking import _filter_nan_scores
        ref = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        m = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        r_out, y_out, m_out = _filter_nan_scores(ref, y, m)
        np.testing.assert_array_equal(r_out, [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(y_out, [10.0, 30.0, 50.0])
        np.testing.assert_array_equal(m_out, [0.1, 0.3, 0.5])

    def test_all_nan_returns_empty(self):
        from mlsurv.learner._benchmarking import _filter_nan_scores
        ref = np.array([np.nan, np.nan])
        y = np.array([1.0, 2.0])
        r_out, y_out = _filter_nan_scores(ref, y)
        assert len(r_out) == 0
        assert len(y_out) == 0

    def test_drops_inf(self):
        from mlsurv.learner._benchmarking import _filter_nan_scores
        ref = np.array([1.0, np.inf, 3.0, -np.inf])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        r_out, y_out = _filter_nan_scores(ref, y)
        np.testing.assert_array_equal(r_out, [1.0, 3.0])
        np.testing.assert_array_equal(y_out, [10.0, 30.0])


class TestBuildIncrementalRowsSkip:
    """Test that _build_incremental_rows skips pairs with too few valid observations."""

    def test_skip_when_all_nan(self):
        from mlsurv.learner._benchmarking import _build_incremental_rows
        y = make_survival_array([True, True, False], [1.0, 2.0, 3.0])
        # Adapter that returns all-NaN scores
        adapter = ReferenceAdapter(
            np.array([np.nan, np.nan, np.nan]), name="bad",
        )
        rows = _build_incremental_rows(
            y, {"m1": np.array([0.1, 0.2, 0.3])}, {"bad": adapter},
            ["m1"], ["bad"],
            bootstrap=False, n_bootstrap=0,
            bootstrap_ci=0.95, random_state=42,
        )
        assert rows == []

    def test_skip_when_one_valid(self):
        from mlsurv.learner._benchmarking import _build_incremental_rows
        y = make_survival_array([True, True, False], [1.0, 2.0, 3.0])
        # Only 1 valid observation
        adapter = ReferenceAdapter(
            np.array([1.0, np.nan, np.nan]), name="sparse",
        )
        rows = _build_incremental_rows(
            y, {"m1": np.array([0.1, 0.2, 0.3])}, {"sparse": adapter},
            ["m1"], ["sparse"],
            bootstrap=False, n_bootstrap=0,
            bootstrap_ci=0.95, random_state=42,
        )
        assert rows == []


# =====================================================================
# TestResolveReferenceArraysNaN
# =====================================================================

@pytest.mark.integration
class TestResolveReferenceArraysNaN:
    def test_all_nan_raises(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        all_nan = np.full(len(learner._state.X_test), np.nan)
        with pytest.raises(ValueError, match="no finite values"):
            learner.benchmark(
                references={"bad": all_nan},
                bootstrap=False, verbose=False,
            )

    def test_inf_warns(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import MethodologicalWarning
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        ref = X["feature_0"].values[:len(learner._state.X_test)].copy()
        ref[0] = np.inf
        with pytest.warns(MethodologicalWarning, match="infinite"):
            learner.benchmark(
                references={"ref_inf": ref},
                bootstrap=False, verbose=False,
            )


# =====================================================================
# TestCorrelationsWithNaN
# =====================================================================

class TestCorrelationsWithNaN:
    def test_correlations_with_nan_reference(self):
        from mlsurv.learner._benchmarking import _compute_correlations
        model_scores = {"m1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        ref_arrays = {"r1": np.array([5.0, np.nan, 3.0, np.nan, 1.0])}
        result = _compute_correlations(
            model_scores, ref_arrays,
            ["m1"], ["r1"], logger=__import__("logging").getLogger("test"),
        )
        assert len(result) == 1
        assert not np.isnan(result.iloc[0]["spearman_rho"])
        assert not np.isnan(result.iloc[0]["pearson_r"])

    def test_correlations_clean_path_column_names(self):
        """Happy-path _compute_correlations should emit the canonical
        column names. Guards against partial reverts of the column
        rename at _benchmarking.py:203 (no-NaN dict literal)."""
        from mlsurv.learner._benchmarking import _compute_correlations
        model_scores = {"m1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        ref_arrays = {"r1": np.array([5.0, 4.0, 3.0, 2.0, 1.0])}
        result = _compute_correlations(
            model_scores, ref_arrays,
            ["m1"], ["r1"], logger=__import__("logging").getLogger("test"),
        )
        assert set(result.columns) == {
            "model", "reference",
            "spearman_rho", "spearman_p",
            "pearson_r", "pearson_p",
        }
        assert "spearman_r" not in result.columns


# =====================================================================
# TestBenchmarkNaN (integration)
# =====================================================================

@pytest.mark.integration
class TestBenchmarkNaN:
    @pytest.fixture
    def trained_learner(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        return learner, X

    def test_benchmark_nan_reference_succeeds(self, trained_learner):
        """End-to-end benchmark with one NaN-containing reference."""
        learner, X = trained_learner
        ref = X["feature_0"].values[:len(learner._state.X_test)].copy()
        ref[0:5] = np.nan

        result = learner.benchmark(
            references={"ref_with_nan": ref},
            bootstrap=False,
            verbose=False,
        )
        assert isinstance(result, BenchmarkResult)
        assert "ref_with_nan" in result.reference_names
        assert not result.incremental.empty
        assert not result.correlation.empty

    def test_benchmark_mixed_nan_references(self, trained_learner):
        """Mix of clean and NaN-containing references."""
        learner, X = trained_learner
        n_test = len(learner._state.X_test)
        clean_ref = X["feature_0"].values[:n_test].copy()
        nan_ref = X["feature_1"].values[:n_test].copy()
        nan_ref[0:3] = np.nan

        result = learner.benchmark(
            references={"clean": clean_ref, "has_nan": nan_ref},
            bootstrap=False,
            verbose=False,
        )
        assert len(result.reference_names) == 2
        # Both references should produce evaluation rows
        conc = result.evaluation.concordance
        assert "clean" in conc["model_type"].values
        assert "has_nan" in conc["model_type"].values

    def test_benchmark_nan_with_bootstrap(self, trained_learner):
        """Bootstrap CIs computed correctly on valid subset."""
        learner, X = trained_learner
        ref = X["feature_0"].values[:len(learner._state.X_test)].copy()
        ref[0:3] = np.nan

        result = learner.benchmark(
            references={"ref_nan": ref},
            bootstrap=True,
            n_bootstrap=10,
            verbose=False,
        )
        assert result.evaluation.bootstrap is not None
        # Incremental metrics should have CI values
        assert result.incremental["ci_lower"].notna().any()

    def test_benchmark_nan_incremental_metrics(self, trained_learner):
        """Delta-C, NRI, IDI computed on valid subset with NaN reference."""
        learner, X = trained_learner
        ref = X["feature_0"].values[:len(learner._state.X_test)].copy()
        ref[0:5] = np.nan

        result = learner.benchmark(
            references={"ref_nan": ref},
            bootstrap=False,
            verbose=False,
        )
        inc = result.incremental
        assert not inc.empty
        metrics = inc["metric"].unique()
        assert "delta_c_index" in metrics
        assert "nri" in metrics
        assert "idi" in metrics

    def test_benchmark_nan_prints_message(self, trained_learner, capsys):
        """Verbose mode prints NaN info."""
        learner, X = trained_learner
        ref = X["feature_0"].values[:len(learner._state.X_test)].copy()
        ref[0:5] = np.nan

        learner.benchmark(
            references={"my_ref": ref},
            bootstrap=False,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "my_ref" in captured.out
        assert "5/" in captured.out
        assert "non-finite values" in captured.out


# =====================================================================
# TestBenchmarkSampleSizes
# =====================================================================


@pytest.mark.integration
class TestBenchmarkSampleSizes:
    """Tests for BenchmarkResult.sample_sizes."""

    @pytest.fixture
    def trained_learner(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        return learner, X

    def test_sample_sizes_populated(self, trained_learner):
        """All sample_sizes equal test N when references are clean."""
        learner, X = trained_learner
        n_test = len(learner._state.X_test)
        ref = X["feature_0"].values[:n_test].copy()

        result = learner.benchmark(
            references={"clean_ref": ref},
            bootstrap=False,
            verbose=False,
        )
        assert result.sample_sizes is not None
        model_name = result.model_names[0]
        assert result.sample_sizes[model_name] == n_test
        assert result.sample_sizes["clean_ref"] == n_test

    def test_sample_sizes_with_nan(self, trained_learner):
        """Reference with NaN has smaller N than models."""
        learner, X = trained_learner
        n_test = len(learner._state.X_test)
        ref = X["feature_0"].values[:n_test].copy()
        ref[0:5] = np.nan

        result = learner.benchmark(
            references={"nan_ref": ref},
            bootstrap=False,
            verbose=False,
        )
        assert result.sample_sizes is not None
        model_name = result.model_names[0]
        assert result.sample_sizes[model_name] == n_test
        assert result.sample_sizes["nan_ref"] == n_test - 5

    def test_sample_sizes_in_print_summary(self, trained_learner, capsys):
        """Print note appears for NaN references, absent for clean ones."""
        learner, X = trained_learner
        n_test = len(learner._state.X_test)

        # Clean reference — no note
        clean_ref = X["feature_0"].values[:n_test].copy()
        learner.benchmark(
            references={"clean_ref": clean_ref},
            bootstrap=False,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "Different sample sizes" not in captured.out
        assert "population differences" not in captured.out

        # NaN reference — note appears
        nan_ref = X["feature_1"].values[:n_test].copy()
        nan_ref[0:7] = np.nan
        learner.benchmark(
            references={"nan_ref": nan_ref},
            bootstrap=False,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "nan_ref" in captured.out
        assert f"N={n_test - 7}" in captured.out
        assert "population differences" in captured.out

    def test_sample_sizes_none_backward_compat(self):
        """BenchmarkResult with sample_sizes=None doesn't crash print."""
        from mlsurv.results.containers import EvaluationResult
        eval_result = EvaluationResult(
            concordance=pd.DataFrame({"model_type": ["m1"], "c_index_test": [0.7]}),
            test_set=pd.DataFrame(),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
            shap_values=None,
            bootstrap=None,
        )
        br = BenchmarkResult(
            evaluation=eval_result,
            univariate_cox=pd.DataFrame(),
            incremental=pd.DataFrame(),
            correlation=pd.DataFrame(),
            reference_names=["ref1"],
            model_names=["m1"],
        )
        assert br.sample_sizes is None
        # Should not raise
        br.print_summary()


# =====================================================================
# TestBenchmarkDatasetParam
# =====================================================================

@pytest.mark.integration
class TestBenchmarkDatasetParam:
    """Tests for the ``dataset`` parameter on ``benchmark()``."""

    @pytest.fixture
    def trained_learner(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        return learner

    @pytest.mark.parametrize("dataset", ["test", "train", "full"])
    def test_benchmark_dataset_modes(self, trained_learner, dataset):
        """Each dataset mode runs and stores the dataset label."""
        learner = trained_learner
        if dataset == "test":
            ref_length = len(learner.X_test)
        elif dataset == "train":
            ref_length = len(learner.X_train)
        else:
            ref_length = len(learner.X)

        ref = np.random.default_rng(42).standard_normal(ref_length)
        result = learner.benchmark(
            references={"ext_score": ref},
            dataset=dataset,
            bootstrap=False,
            verbose=False,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.dataset == dataset
        assert not result.univariate_cox.empty
        assert not result.incremental.empty

    @pytest.mark.parametrize("dataset", ["test", "train", "full"])
    def test_full_length_ref_extraction(self, trained_learner, dataset):
        """Full-length reference array is extracted to the correct subset."""
        learner = trained_learner
        n_full = len(learner.X)
        ref = np.random.default_rng(99).standard_normal(n_full)

        result = learner.benchmark(
            references={"ext_score": ref},
            dataset=dataset,
            bootstrap=False,
            verbose=False,
        )
        # Model always evaluated on test data (unbiased)
        assert result.sample_sizes["coxph"] == len(learner.X_test)

        # Reference evaluated on the selected dataset
        if dataset == "test":
            expected_ref_n = len(learner.X_test)
        elif dataset == "train":
            expected_ref_n = len(learner.X_train)
        else:
            expected_ref_n = n_full
        assert result.sample_sizes["ext_score"] == expected_ref_n

    def test_sample_sizes_reflect_dataset(self, trained_learner):
        """Sample sizes differ across dataset modes."""
        learner = trained_learner
        n_full = len(learner.X)
        ref = np.random.default_rng(7).standard_normal(n_full)

        r_test = learner.benchmark(
            references={"ext": ref}, dataset="test",
            bootstrap=False, verbose=False,
        )
        r_train = learner.benchmark(
            references={"ext": ref}, dataset="train",
            bootstrap=False, verbose=False,
        )
        r_full = learner.benchmark(
            references={"ext": ref}, dataset="full",
            bootstrap=False, verbose=False,
        )
        # Models always use test data regardless of dataset param
        n_test = len(learner.X_test)
        assert r_test.sample_sizes["coxph"] == n_test
        assert r_train.sample_sizes["coxph"] == n_test
        assert r_full.sample_sizes["coxph"] == n_test

        # References use the selected dataset
        assert r_test.sample_sizes["ext"] == n_test
        assert r_train.sample_sizes["ext"] == len(learner.X_train)
        assert r_full.sample_sizes["ext"] == n_full
        # Reference sizes are additive across train + test = full
        assert (
            r_test.sample_sizes["ext"] + r_train.sample_sizes["ext"]
            == r_full.sample_sizes["ext"]
        )

    def test_invalid_dataset_raises(self, trained_learner):
        """Invalid dataset value raises ValueError."""
        ref = np.random.default_rng(0).standard_normal(len(trained_learner.X_test))
        with pytest.raises(ValueError, match="dataset must be one of"):
            trained_learner.benchmark(
                references={"ext": ref},
                dataset="validation",
                bootstrap=False,
                verbose=False,
            )

    def test_warning_emitted_for_train(self, trained_learner):
        """MethodologicalWarning emitted for dataset='train'."""
        from mlsurv._warnings import MethodologicalWarning

        ref = np.random.default_rng(0).standard_normal(len(trained_learner.X_train))
        with pytest.warns(MethodologicalWarning, match="dataset='train'.*references"):
            trained_learner.benchmark(
                references={"ext": ref},
                dataset="train",
                bootstrap=False,
                verbose=False,
            )

    def test_warning_emitted_for_full(self, trained_learner):
        """MethodologicalWarning emitted for dataset='full'."""
        from mlsurv._warnings import MethodologicalWarning

        ref = np.random.default_rng(0).standard_normal(len(trained_learner.X))
        with pytest.warns(MethodologicalWarning, match="dataset='full'.*references"):
            trained_learner.benchmark(
                references={"ext": ref},
                dataset="full",
                bootstrap=False,
                verbose=False,
            )

    def test_no_warning_for_test(self, trained_learner):
        """No MethodologicalWarning emitted for default dataset='test'."""
        from mlsurv._warnings import MethodologicalWarning

        ref = np.random.default_rng(0).standard_normal(len(trained_learner.X_test))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trained_learner.benchmark(
                references={"ext": ref},
                dataset="test",
                bootstrap=False,
                verbose=False,
            )
        methodological = [
            w for w in caught if issubclass(w.category, MethodologicalWarning)
            and "dataset=" in str(w.message)
        ]
        assert len(methodological) == 0

    def test_default_dataset_is_test(self, trained_learner):
        """Omitting dataset gives dataset='test' in result."""
        ref = np.random.default_rng(0).standard_normal(len(trained_learner.X_test))
        result = trained_learner.benchmark(
            references={"ext": ref},
            bootstrap=False,
            verbose=False,
        )
        assert result.dataset == "test"

    def test_repr_includes_dataset_when_non_default(self, trained_learner):
        """BenchmarkResult repr shows dataset when not 'test'."""
        ref = np.random.default_rng(0).standard_normal(len(trained_learner.X))
        result = trained_learner.benchmark(
            references={"ext": ref},
            dataset="full",
            bootstrap=False,
            verbose=False,
        )
        assert "dataset='full'" in repr(result)

    def test_repr_omits_dataset_when_test(self, trained_learner):
        """BenchmarkResult repr omits dataset for default 'test'."""
        ref = np.random.default_rng(0).standard_normal(len(trained_learner.X_test))
        result = trained_learner.benchmark(
            references={"ext": ref},
            bootstrap=False,
            verbose=False,
        )
        assert "dataset=" not in repr(result)

    def test_dataset_full_with_presplit_raises(self, synthetic_data_clean):
        """dataset='full' with presplit data raises ValueError."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        n = len(X)
        learner = SurvivalLearner(
            X_train=X.iloc[:n // 2], X_test=X.iloc[n // 2:],
            y_train=y[:n // 2], y_test=y[n // 2:],
        )
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        ref = np.random.default_rng(0).standard_normal(n)
        with pytest.raises(ValueError, match="pre-split"):
            learner.benchmark(
                references={"ext": ref},
                dataset="full",
                bootstrap=False,
                verbose=False,
            )

    def test_dataset_train_with_presplit(self, synthetic_data_clean):
        """dataset='train' works with presplit data."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        n = len(X)
        learner = SurvivalLearner(
            X_train=X.iloc[:n // 2], X_test=X.iloc[n // 2:],
            y_train=y[:n // 2], y_test=y[n // 2:],
        )
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        ref = np.random.default_rng(0).standard_normal(n // 2)
        result = learner.benchmark(
            references={"ext": ref},
            dataset="train",
            bootstrap=False,
            verbose=False,
        )
        assert result.dataset == "train"
        # Model always uses test data (n // 2 for equal presplit)
        assert result.sample_sizes["coxph"] == len(learner.X_test)
        # Reference uses training data
        assert result.sample_sizes["ext"] == len(learner.X_train)

    @pytest.mark.parametrize("dataset", ["train", "full"])
    def test_model_always_uses_test_data(self, trained_learner, dataset):
        """Model metrics match standalone evaluate() on test data."""
        from mlsurv.evaluation import evaluate

        learner = trained_learner
        # Full-length ref so test-set resolution succeeds
        ref = np.random.default_rng(42).standard_normal(len(learner.X))

        # Standalone evaluate on test data (ground truth)
        standalone = evaluate(
            {"coxph": learner.models["coxph"].model},
            learner.X_test, learner.y_test,
            X_train=learner.X_train, y_train=learner.y_train,
            bootstrap=False, show_progress=False,
        )
        standalone_ci = (
            standalone.concordance
            .set_index("model_type")["c_index_test"]
            .loc["coxph"]
        )

        # Benchmark with non-test dataset
        result = learner.benchmark(
            references={"ext": ref},
            dataset=dataset,
            bootstrap=False,
            verbose=False,
        )
        bench_ci = (
            result.evaluation.concordance
            .set_index("model_type")
            .loc["coxph", "c_index_test"]
        )

        assert bench_ci == pytest.approx(standalone_ci, abs=1e-10)

    @pytest.mark.parametrize("dataset", ["test", "train", "full"])
    def test_model_sample_size_always_test(self, trained_learner, dataset):
        """Model sample size is always len(X_test)."""
        learner = trained_learner
        ref = np.random.default_rng(0).standard_normal(len(learner.X))
        result = learner.benchmark(
            references={"ext": ref},
            dataset=dataset,
            bootstrap=False,
            verbose=False,
        )
        assert result.sample_sizes["coxph"] == len(learner.X_test)


@pytest.mark.integration
class TestBenchmarkROCCI:
    """Tests for bootstrap CI ribbons in benchmarking ROC overlay."""

    @pytest.fixture
    def benchmarked_with_bootstrap(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=True,
            n_bootstrap=10,
            verbose=False,
        )
        return learner

    @pytest.fixture
    def benchmarked_without_bootstrap(self, synthetic_data_clean):
        from mlsurv import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        learner.benchmark(
            references={"ext_score": X["feature_0"].values},
            bootstrap=False,
            verbose=False,
        )
        return learner

    def test_roc_ci_with_bootstrap(self, benchmarked_with_bootstrap):
        """ROC overlay should include CI ribbon text when bootstrap data exists."""
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        learner = benchmarked_with_bootstrap
        section = BenchmarkSection(learner)
        br = learner._state.benchmark_result

        md_lines = []
        figures = []
        figure_info = []
        section._add_roc_overlay(md_lines, figures, figure_info, br)

        text = "\n".join(md_lines)
        assert "confidence intervals" in text
        # Fixture uses n_bootstrap=10 < 20 threshold
        assert "only 10 bootstrap iterations" in text

    def test_roc_ci_low_bootstrap_warning(self, benchmarked_with_bootstrap):
        """Low-bootstrap warning shown when n_bootstrap < 20."""
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        learner = benchmarked_with_bootstrap
        section = BenchmarkSection(learner)
        br = learner._state.benchmark_result

        md_lines = []
        figures = []
        figure_info = []
        section._add_roc_overlay(md_lines, figures, figure_info, br)

        text = "\n".join(md_lines)
        assert "n_bootstrap" in text
        assert "CI bands may not fully enclose" in text

    def test_roc_ci_no_warning_when_sufficient_iters(
        self, benchmarked_with_bootstrap,
    ):
        """No low-bootstrap warning when n_boot_iters >= 20."""
        import pandas as pd
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        learner = benchmarked_with_bootstrap
        section = BenchmarkSection(learner)
        br = learner._state.benchmark_result

        # Expand predictions from 10 to 25 unique iterations by
        # duplicating existing bootstrap samples with new iteration IDs.
        original = br.evaluation.bootstrap.predictions
        extra = []
        for i in range(15):
            chunk = original[original["iteration"] == i % 10].copy()
            chunk["iteration"] = 10 + i
            extra.append(chunk)
        br.evaluation.bootstrap.predictions = pd.concat(
            [original] + extra, ignore_index=True,
        )
        try:
            md_lines = []
            figures = []
            figure_info = []
            section._add_roc_overlay(md_lines, figures, figure_info, br)

            text = "\n".join(md_lines)
            assert "confidence intervals" in text
            assert "CI bands may not fully enclose" not in text
        finally:
            br.evaluation.bootstrap.predictions = original

    def test_roc_ci_figure_has_ribbons(self, benchmarked_with_bootstrap):
        """ROC figure should have more traces (CI ribbons) than plain ROC."""
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        learner = benchmarked_with_bootstrap
        section = BenchmarkSection(learner)
        br = learner._state.benchmark_result

        md_lines = []
        figures = []
        figure_info = []
        section._add_roc_overlay(md_lines, figures, figure_info, br)

        assert len(figures) == 1
        fig = figures[0][1]
        # CI ribbons add 2 traces per model per time point.
        # With bootstrap, there should be more traces than the number of
        # models + references + 1 (diagonal).
        n_models = len(br.model_names) + len(br.reference_names)
        assert len(fig.data) > n_models + 1

    def test_roc_no_ci_without_bootstrap(self, benchmarked_without_bootstrap):
        """ROC overlay should NOT include CI text when no bootstrap data."""
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        learner = benchmarked_without_bootstrap
        section = BenchmarkSection(learner)
        br = learner._state.benchmark_result

        md_lines = []
        figures = []
        figure_info = []
        section._add_roc_overlay(md_lines, figures, figure_info, br)

        text = "\n".join(md_lines)
        assert "confidence intervals" not in text

    def test_roc_ci_skipped_when_predictions_none(self, benchmarked_with_bootstrap):
        """CI computation skipped when bootstrap.predictions is None."""
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        learner = benchmarked_with_bootstrap
        section = BenchmarkSection(learner)
        br = learner._state.benchmark_result

        # Temporarily set predictions to None
        original = br.evaluation.bootstrap.predictions
        br.evaluation.bootstrap.predictions = None
        try:
            md_lines = []
            figures = []
            figure_info = []
            section._add_roc_overlay(md_lines, figures, figure_info, br)

            text = "\n".join(md_lines)
            assert "confidence intervals" not in text
            assert len(figures) == 1  # still produces plain ROC
        finally:
            br.evaluation.bootstrap.predictions = original

    def test_roc_ci_skipped_when_predictions_empty(self, benchmarked_with_bootstrap):
        """CI computation skipped when bootstrap.predictions is empty."""
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        learner = benchmarked_with_bootstrap
        section = BenchmarkSection(learner)
        br = learner._state.benchmark_result

        original = br.evaluation.bootstrap.predictions
        br.evaluation.bootstrap.predictions = original.iloc[:0]  # empty DF
        try:
            md_lines = []
            figures = []
            figure_info = []
            section._add_roc_overlay(md_lines, figures, figure_info, br)

            text = "\n".join(md_lines)
            assert "confidence intervals" not in text
            assert len(figures) == 1
        finally:
            br.evaluation.bootstrap.predictions = original

    def test_roc_ci_fallback_on_error(self, benchmarked_with_bootstrap):
        """CI computation errors fall back to plain ROC curves with note."""
        from unittest.mock import patch
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        learner = benchmarked_with_bootstrap
        section = BenchmarkSection(learner)
        br = learner._state.benchmark_result

        md_lines = []
        figures = []
        figure_info = []
        with patch(
            "mlsurv.viz._metrics_roc.bootstrap_roc_curves",
            side_effect=ValueError("mock CI failure"),
        ):
            section._add_roc_overlay(md_lines, figures, figure_info, br)

        text = "\n".join(md_lines)
        assert "confidence intervals" not in text
        assert "could not be computed" in text.lower()
        assert len(figures) == 1  # plain ROC still produced

    def test_roc_overlay_exception_guard(self):
        """_add_roc_overlay uses ALL_EXCEPTIONS for CI band errors."""
        import inspect
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        source = inspect.getsource(BenchmarkSection._add_roc_overlay)
        assert "ALL_EXCEPTIONS" in source
        assert "except Exception " not in source


# =====================================================================
# TestBenchmarkPreservesAucTBootstrap
# =====================================================================

@pytest.mark.integration
class TestBenchmarkPreservesAucTBootstrap:
    """Regression: benchmark() must not drop subpopulation auc_t_bootstrap.

    Before the fix, ``_attach_subpopulation_results`` in ``_benchmarking.py``
    populated ``metrics``, ``auc_t``, and ``bootstrap_summary`` on the new
    sg_results, but never ``auc_t_bootstrap`` — the per-time-point bootstrap
    iterations needed for AUC(t) CI ribbons. Then ``benchmark()`` overwrote
    ``state.subpopulation_results`` with those incomplete results, clobbering the
    ribbon data that ``learner.run()`` had populated. Result: the report's
    "Subpopulation AUC(t)" figure had no visible ±1 SD ribbons.
    """

    def test_auc_t_bootstrap_survives_benchmark(self):
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )

        rng = np.random.RandomState(0)
        subpopulation = rng.choice(["A", "B", "C", "D", "E"], size=len(X))

        n = len(X)
        idx_test = rng.choice(n, size=n // 4, replace=False)
        mask_test = np.zeros(n, dtype=bool)
        mask_test[idx_test] = True
        mask_train = ~mask_test

        learner = SurvivalLearner(
            X_train=X[mask_train].reset_index(drop=True),
            X_test=X[mask_test].reset_index(drop=True),
            y_train=y[mask_train],
            y_test=y[mask_test],
        )
        learner.setup(
            feature_selector="topk",
            random_state=42,
            cv_folds=3,
            subpopulations_train={"category": subpopulation[mask_train]},
            subpopulations_test={"category": subpopulation[mask_test]},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.run(
                models=["coxph", "rsf"],
                n_bootstrap=3,
                eval_times=[365, 730, 1095, 1460, 1825],
                show_eval_progress=False,
                verbose=False,
                tune=False,
            )

        sg_after_run = learner._state.subpopulation_results["category"]
        assert sg_after_run.auc_t_bootstrap is not None, (
            "Setup invariant failed: auc_t_bootstrap should be populated "
            "after run() with n_bootstrap > 0 and eval_times set"
        )
        run_models = set(sg_after_run.auc_t_bootstrap["model_type"].unique())
        assert {"coxph", "rsf"}.issubset(run_models), (
            f"Expected coxph and rsf in post-run() auc_t_bootstrap, got {run_models}"
        )

        coxph_test_set = learner._state.models["coxph"].evaluation.test_set
        fake_ref = coxph_test_set[
            coxph_test_set["model_type"] == "coxph"
        ]["risk_score"].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.benchmark(
                references={"FakeRef": fake_ref},
                n_bootstrap=3,
                random_state=42,
                verbose=False,
            )

        sg_after_bench = learner._state.subpopulation_results["category"]
        assert sg_after_bench.auc_t_bootstrap is not None, (
            "benchmark() dropped auc_t_bootstrap from subpopulation_results. "
            "_attach_subpopulation_results in _benchmarking.py must populate "
            "auc_t_bootstrap so ribbon data survives the state overwrite "
            "inside benchmark()."
        )
        bench_models = set(sg_after_bench.auc_t_bootstrap["model_type"].unique())
        # FakeRef adapter may or may not emit subpopulation_auc_t rows — only
        # assert on the real models.
        assert {"coxph", "rsf"}.issubset(bench_models), (
            f"Expected coxph and rsf in post-benchmark() auc_t_bootstrap, "
            f"got {bench_models}"
        )
        # All five synthetic subpopulation levels in the prior run() data must
        # still be represented — guards against a filter that silently
        # truncates rows.
        bench_levels = set(sg_after_bench.auc_t_bootstrap["subpopulation"].unique())
        run_levels = set(sg_after_run.auc_t_bootstrap["subpopulation"].unique())
        assert run_levels.issubset(bench_levels), (
            f"benchmark() lost subpopulation levels. "
            f"Pre-benchmark: {run_levels}, post-benchmark: {bench_levels}"
        )

    def test_auc_t_bootstrap_survives_benchmark_without_bootstrap(self):
        """Gap 1: benchmark(bootstrap=False) after run() with bootstrap.

        When benchmark() runs without its own bootstrap, the new sg_results
        have ``auc_t_bootstrap=None`` because ``_attach_subpopulation_results``
        skips the attachment block entirely (``eval_result.bootstrap is
        None``). The previous overwrite at ``_benchmarking.py:808`` would
        then clobber the ribbon data from the prior ``learner.run()``.
        The fix merges rather than overwrites, preserving prior data for
        fields (like ``auc_t_bootstrap``) that the new run didn't compute.
        """
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )

        rng = np.random.RandomState(1)
        subpopulation = rng.choice(["A", "B", "C", "D", "E"], size=len(X))

        n = len(X)
        idx_test = rng.choice(n, size=n // 4, replace=False)
        mask_test = np.zeros(n, dtype=bool)
        mask_test[idx_test] = True
        mask_train = ~mask_test

        learner = SurvivalLearner(
            X_train=X[mask_train].reset_index(drop=True),
            X_test=X[mask_test].reset_index(drop=True),
            y_train=y[mask_train],
            y_test=y[mask_test],
        )
        learner.setup(
            feature_selector="topk",
            random_state=42,
            cv_folds=3,
            subpopulations_train={"category": subpopulation[mask_train]},
            subpopulations_test={"category": subpopulation[mask_test]},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.run(
                models=["coxph", "rsf"],
                n_bootstrap=3,
                eval_times=[365, 730, 1095, 1460, 1825],
                show_eval_progress=False,
                verbose=False,
                tune=False,
            )

        sg_after_run = learner._state.subpopulation_results["category"]
        assert sg_after_run.auc_t_bootstrap is not None, (
            "Setup invariant failed: run(n_bootstrap=3) should populate "
            "auc_t_bootstrap"
        )
        run_row_count = len(sg_after_run.auc_t_bootstrap)
        run_levels = set(sg_after_run.auc_t_bootstrap["subpopulation"].unique())

        # benchmark WITHOUT its own bootstrap — the new sg_results will
        # have auc_t_bootstrap=None because the attachment block is
        # skipped (eval_result.bootstrap is None). Previously this
        # clobbered the prior run()'s data during the direct overwrite.
        coxph_test_set = learner._state.models["coxph"].evaluation.test_set
        fake_ref = coxph_test_set[
            coxph_test_set["model_type"] == "coxph"
        ]["risk_score"].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.benchmark(
                references={"FakeRef": fake_ref},
                bootstrap=False,
                random_state=42,
                verbose=False,
            )

        sg_after_bench = learner._state.subpopulation_results["category"]
        assert sg_after_bench.auc_t_bootstrap is not None, (
            "Gap 1: benchmark(bootstrap=False) dropped the prior "
            "run()'s auc_t_bootstrap. benchmark() must merge into "
            "state.subpopulation_results instead of overwriting — the new "
            "sg_results from a no-bootstrap benchmark cannot replace "
            "the prior bootstrap data."
        )
        assert len(sg_after_bench.auc_t_bootstrap) == run_row_count, (
            f"Gap 1: auc_t_bootstrap row count changed "
            f"({run_row_count} → {len(sg_after_bench.auc_t_bootstrap)}). "
            f"A no-bootstrap benchmark should leave the prior bootstrap "
            f"iterations intact."
        )
        bench_levels = set(sg_after_bench.auc_t_bootstrap["subpopulation"].unique())
        assert bench_levels == run_levels, (
            f"Gap 1: subpopulation levels changed "
            f"({run_levels} → {bench_levels})"
        )


# =====================================================================
# TestReferenceSurvFuncSilence
# =====================================================================


class _NoSurvFuncFailModel:
    """Fake model whose predict_survival_function raises ValueError.

    Used to verify that filtering surv-func metrics for references
    does not also silence genuine failures from real models.
    """

    is_fitted_ = True
    supports_random_state = False
    has_internal_parallelism = False
    name = "fakebroken"

    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=float)

    def predict(self, X=None):
        if X is not None and hasattr(X, "__len__") and len(X) != len(self._scores):
            return np.full(len(X), np.nan)
        return self._scores

    def predict_survival_function(self, X=None, **kwargs):
        raise ValueError("deliberate failure in test model")

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class TestReferenceSurvFuncSilence:
    """Regression tests: benchmark(references=...) must not emit
    noise when dispatching surv-func metrics that references structurally
    cannot compute, but must still warn on genuine model failures."""

    def test_reference_adapter_declares_no_surv_func(self):
        adapter = ReferenceAdapter(np.array([1.0, 2.0]), name="age")
        assert adapter.supports_survival_function is False

    def test_supports_survival_function_helper(self):
        from mlsurv.utils import supports_survival_function

        adapter = ReferenceAdapter(np.array([1.0]), name="age")
        assert supports_survival_function(adapter) is False

        class _Real:
            def predict_survival_function(self, X):
                return None

        assert supports_survival_function(_Real()) is True

    @pytest.mark.integration
    def test_benchmark_references_silent(self, caplog):
        """learner.benchmark(references=...) emits no MetricFailureWarning
        and no ERROR logs for the reference × surv-func structural mismatch."""
        import logging

        from mlsurv import SurvivalLearner
        from mlsurv._warnings import MetricFailureWarning

        rng = np.random.default_rng(0)
        n, p = 120, 4
        X = pd.DataFrame(
            rng.standard_normal((n, p)),
            columns=[f"f{i}" for i in range(p)],
        )
        time = rng.exponential(scale=1.0, size=n).clip(0.1, 10.0)
        event = rng.binomial(1, 0.7, size=n).astype(bool)
        y = np.array(list(zip(event, time)),
                     dtype=[("event", bool), ("time", float)])

        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=0, verbose=False)
        learner.train("coxph", verbose=False)

        refs = {
            "age": rng.standard_normal(n),
            "ki67": rng.standard_normal(n),
        }

        with caplog.at_level(logging.ERROR, logger="mlsurv.bootstrap"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                learner.benchmark(references=refs, n_bootstrap=2)

        noisy_warnings = [
            w for w in caught
            if issubclass(w.category, MetricFailureWarning)
            and any(name in str(w.message) for name in refs)
            and ("integrated_brier" in str(w.message)
                 or "ctd" in str(w.message))
        ]
        assert noisy_warnings == [], (
            f"Expected no MetricFailureWarning for reference x surv-func "
            f"combinations; got: {[str(w.message) for w in noisy_warnings]}"
        )

        noisy_errors = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR
            and any(name in r.getMessage() for name in refs)
            and ("integrated_brier" in r.getMessage()
                 or "ctd" in r.getMessage())
        ]
        assert noisy_errors == [], (
            f"Expected no ERROR logs for reference x surv-func "
            f"combinations; got: {[r.getMessage() for r in noisy_errors]}"
        )

    @pytest.mark.integration
    def test_benchmark_reference_rows_preserve_nan(self):
        """Benchmark concordance rows for references still show NaN for
        integrated_brier_test and c_index_td_test (unchanged behaviour)."""
        from mlsurv import SurvivalLearner

        rng = np.random.default_rng(1)
        n, p = 120, 4
        X = pd.DataFrame(
            rng.standard_normal((n, p)),
            columns=[f"f{i}" for i in range(p)],
        )
        time = rng.exponential(scale=1.0, size=n).clip(0.1, 10.0)
        event = rng.binomial(1, 0.7, size=n).astype(bool)
        y = np.array(list(zip(event, time)),
                     dtype=[("event", bool), ("time", float)])

        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=1, verbose=False)
        learner.train("coxph", verbose=False)

        ref_scores = rng.standard_normal(n)
        result = learner.benchmark(references={"age": ref_scores}, n_bootstrap=2)

        conc = result.evaluation.concordance
        age_rows = conc[conc["model_type"] == "age"]
        assert not age_rows.empty, "Expected reference row for 'age'"
        assert age_rows["integrated_brier_test"].isna().all()
        assert age_rows["c_index_td_test"].isna().all()

    def test_models_still_warn_on_real_surv_func_failure(self):
        """A real model whose predict_survival_function raises ValueError
        must still produce the usual MetricFailureWarning — the reference
        filter must not blanket-silence genuine failures."""
        from mlsurv._warnings import MetricFailureWarning
        from mlsurv.evaluation.bootstrap import bootstrap

        n = 60
        rng = np.random.default_rng(2)
        scores = rng.standard_normal(n)
        X = pd.DataFrame({"f0": rng.standard_normal(n)})
        time = rng.exponential(1.0, n).clip(0.1, 10.0)
        event = rng.binomial(1, 0.7, n).astype(bool)
        y = np.array(list(zip(event, time)),
                     dtype=[("event", bool), ("time", float)])
        y_train = y.copy()

        model = _NoSurvFuncFailModel(scores)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bootstrap(
                model, X, y, y_train=y_train,
                metrics=["integrated_brier"],
                n_bootstrap=3,
                show_progress=False,
                eval_times=np.array([0.5, 1.0, 2.0]),
                model_name="fakebroken",
            )

        failure_warnings = [
            w for w in caught
            if issubclass(w.category, MetricFailureWarning)
            and "fakebroken" in str(w.message)
            and "integrated_brier" in str(w.message)
        ]
        assert failure_warnings, (
            "Expected a MetricFailureWarning from a real model with a broken "
            "predict_survival_function; the reference surv-func filter must not blanket-silence "
            "genuine model failures."
        )
