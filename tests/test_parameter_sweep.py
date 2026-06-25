"""Tests for SurvivalLearner.parameter_sweep()."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import SuggestionWarning
from mlsurv.results.containers import SweepResult


EXPECTED_COLUMNS = {"param_value", "model", "phase", "metric", "mean",
                    "std", "ci_lower", "ci_upper"}


class TestParameterSweepValidation:
    """Input validation and error paths."""

    def test_invalid_target_raises(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with pytest.raises(ValueError, match="target must be one of"):
            learner.parameter_sweep("k", [3, 5], target="bogus")

    def test_empty_values_raises(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with pytest.raises(ValueError, match="values must be non-empty"):
            learner.parameter_sweep("k", [], target="feature_selector")

    def test_no_models_no_trained_raises(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with pytest.raises(ValueError, match="No models have been trained"):
            learner.parameter_sweep("k", [3, 5], target="feature_selector")

    def test_no_selector_configured_raises(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)  # no feature_selector

        with pytest.raises(ValueError, match="No feature_selector is configured"):
            learner.parameter_sweep(
                "k", [3, 5], target="feature_selector", models="coxph",
            )

    def test_njobs_zero_raises(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        with pytest.raises(ValueError, match="n_jobs=0 has no meaning"):
            learner.parameter_sweep(
                "k", [3, 5], target="feature_selector",
                models="coxph", n_jobs=0,
            )


@pytest.mark.integration
class TestParameterSweepSelector:
    """Sweep feature_selector parameters."""

    def test_sweep_selector_k(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", verbose=False,
        )

        assert isinstance(result, SweepResult)
        assert set(result.data.columns) == EXPECTED_COLUMNS
        assert result.param_name == "k"
        assert result.target == "feature_selector"
        assert set(result.data["param_value"].unique()) == {2, 3}
        assert list(result.data["model"].unique()) == ["coxph"]
        assert len(result.data) > 0
        # Data sanity: metrics are present and values are reasonable
        assert "harrell" in result.data["metric"].values
        assert result.data["mean"].notna().all()
        assert (result.data["std"] >= 0).all()
        assert (result.data["ci_lower"] <= result.data["mean"]).all()
        assert (result.data["mean"] <= result.data["ci_upper"]).all()

    def test_sweep_selector_k_multi_model(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models=["coxph", "rsf"], verbose=False,
        )

        assert set(result.data["model"].unique()) == {"coxph", "rsf"}


@pytest.mark.integration
class TestParameterSweepModel:
    """Sweep model hyperparameters."""

    def test_sweep_model_param(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.parameter_sweep(
            "n_estimators", [10, 50], target="model",
            models="rsf", verbose=False,
        )

        assert isinstance(result, SweepResult)
        assert result.target == "model"
        assert set(result.data["param_value"].unique()) == {10, 50}


@pytest.mark.integration
class TestParameterSweepIncludeTest:
    """Test include_test=True adds test-set metrics."""

    def test_include_test_adds_test_phase(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", include_test=True, verbose=False,
        )

        assert "phase" in result.data.columns
        phases = set(result.data["phase"].unique())
        assert "cv" in phases
        assert "test" in phases

        # Test rows have NaN std/ci (point estimates)
        test_rows = result.data[result.data["phase"] == "test"]
        assert test_rows["std"].isna().all()
        assert test_rows["ci_lower"].isna().all()

    def test_include_test_bootstrap_ci(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", include_test=True, n_bootstrap=2,
            verbose=False,
        )

        test_rows = result.data[result.data["phase"] == "test"]
        # Bootstrap produces std and CI
        assert test_rows["std"].notna().all()
        assert test_rows["ci_lower"].notna().all()
        assert test_rows["ci_upper"].notna().all()
        assert (test_rows["ci_lower"] <= test_rows["mean"]).all()
        assert (test_rows["mean"] <= test_rows["ci_upper"]).all()

    def test_include_test_plot_test_phase(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", include_test=True, verbose=False,
        )

        fig = result.plot(phase="test")
        assert isinstance(fig, go.Figure)


@pytest.mark.integration
class TestParameterSweepStateUnchanged:
    """Verify the learner is not mutated by parameter_sweep."""

    def test_state_unchanged_after_sweep(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        models_before = dict(learner._state.models)

        learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", verbose=False,
        )

        assert learner._state.models == models_before

    def test_state_unchanged_with_existing_model(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)
        learner.train("coxph")

        result_before = learner._state.models["coxph"]

        learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", verbose=False,
        )

        assert learner._state.models["coxph"] is result_before


@pytest.mark.integration
class TestParameterSweepPlot:
    """Verify plotting works."""

    def test_plot_returns_figure(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", verbose=False,
        )

        fig = result.plot(metric="harrell")
        assert isinstance(fig, go.Figure)

    def test_plot_invalid_metric_raises(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", verbose=False,
        )

        with pytest.raises(ValueError, match="No data for metric"):
            result.plot(metric="nonexistent_metric")


@pytest.mark.integration
class TestSweepResultRepr:
    """SweepResult repr."""

    def test_repr(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", verbose=False,
        )

        r = repr(result)
        assert "SweepResult" in r
        assert "param='k'" in r
        assert "target='feature_selector'" in r

    def test_repr_empty(self):
        result = SweepResult(
            data=pd.DataFrame(), param_name="k", target="feature_selector",
        )
        r = repr(result)
        assert "n_values=0" in r


@pytest.mark.integration
class TestParameterSweepModelsDefault:
    """Test models=None default behavior."""

    def test_models_default_uses_trained(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)
        learner.train("coxph")

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector", verbose=False,
        )

        assert list(result.data["model"].unique()) == ["coxph"]


@pytest.mark.integration
class TestParameterSweepScaler:
    """Sweep scaler target (covers imputer/scaler code path)."""

    def test_sweep_scaler_with_mean(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(scaler="standard", random_state=42)

        result = learner.parameter_sweep(
            "with_mean", [True, False], target="scaler",
            models="coxph", verbose=False,
        )

        assert isinstance(result, SweepResult)
        assert result.target == "scaler"
        assert set(result.data["param_value"].unique()) == {True, False}


@pytest.mark.integration
class TestParameterSweepFailurePath:
    """Test graceful degradation when sweep points fail."""

    def test_invalid_param_value_warns_and_continues(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = learner.parameter_sweep(
                "k", [-1, 2], target="feature_selector",
                models="coxph", verbose=False,
            )

        # k=-1 should have triggered a SuggestionWarning
        sweep_warns = [w for w in caught if issubclass(w.category, SuggestionWarning)
                       and "failed" in str(w.message)]
        assert len(sweep_warns) >= 1
        # k=2 should still produce valid results
        assert len(result.data) > 0
        assert set(result.data["param_value"].unique()) == {2}

    def test_all_points_fail_warns_empty_result(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = learner.parameter_sweep(
                "k", [-1, -2], target="feature_selector",
                models="coxph", verbose=False,
            )

        # Should have warned about all points failing
        all_failed = [w for w in caught if issubclass(w.category, SuggestionWarning)
                      and "All" in str(w.message)]
        assert len(all_failed) >= 1
        assert result.data.empty

    def test_plot_empty_result_raises(self):
        result = SweepResult(
            data=pd.DataFrame(), param_name="k", target="feature_selector",
        )
        with pytest.raises(ValueError, match="empty"):
            result.plot()


@pytest.mark.integration
class TestParameterSweepParallel:
    """Tests for parallel sweep execution (n_jobs > 1)."""

    @pytest.mark.parallel
    def test_parallel_matches_sequential(self, small_data):
        """Parallel sweep produces same metrics as sequential."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result_seq = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", n_jobs=1, verbose=False,
        )
        result_par = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", n_jobs=2, verbose=False,
        )

        assert len(result_seq.data) == len(result_par.data)
        merged = result_seq.data.merge(
            result_par.data,
            on=["param_value", "model", "phase", "metric"],
            suffixes=("_seq", "_par"),
        )
        np.testing.assert_array_almost_equal(
            merged["mean_seq"].values,
            merged["mean_par"].values,
            decimal=8,
        )

    @pytest.mark.parallel
    def test_parallel_njobs_minus_1(self, small_data):
        """n_jobs=-1 works without error."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", n_jobs=-1, verbose=False,
        )
        assert len(result.data) > 0

    @pytest.mark.parallel
    def test_parallel_failure_path(self, small_data):
        """Failed sweep points in parallel mode emit warnings."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = learner.parameter_sweep(
                "k", [-1, 2], target="feature_selector",
                models="coxph", n_jobs=2, verbose=False,
            )

        sweep_warns = [w for w in caught if issubclass(w.category, SuggestionWarning)
                       and "failed" in str(w.message)]
        assert len(sweep_warns) >= 1
        assert len(result.data) > 0

    @pytest.mark.parallel
    def test_parallel_include_test(self, small_data):
        """include_test=True works in parallel mode."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", include_test=True, n_jobs=2, verbose=False,
        )

        phases = set(result.data["phase"].unique())
        assert "cv" in phases
        assert "test" in phases

    @pytest.mark.parallel
    def test_parallel_include_test_bootstrap(self, small_data):
        """include_test + n_bootstrap works in parallel mode."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", include_test=True, n_bootstrap=5,
            n_jobs=2, verbose=False,
        )

        test_rows = result.data[result.data["phase"] == "test"]
        assert len(test_rows) > 0
        assert test_rows["std"].notna().all()
        assert test_rows["ci_lower"].notna().all()
        assert test_rows["ci_upper"].notna().all()
        assert (test_rows["ci_lower"] <= test_rows["mean"]).all()
        assert (test_rows["mean"] <= test_rows["ci_upper"]).all()

    @pytest.mark.parallel
    def test_parallel_multi_model(self, small_data):
        """Multiple models with n_jobs=2."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models=["coxph", "rsf"], n_jobs=2, verbose=False,
        )

        assert set(result.data["model"].unique()) == {"coxph", "rsf"}
        assert set(result.data["param_value"].unique()) == {2, 3}

    @pytest.mark.parallel
    def test_parallel_state_unchanged(self, small_data):
        """Learner state is not mutated by parallel sweep."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        models_before = dict(learner._state.models)

        learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", n_jobs=2, verbose=False,
        )

        assert learner._state.models == models_before


@pytest.mark.integration
class TestSweepResultToDataframes:
    """Tests for SweepResult.to_dataframes()."""

    def test_to_dataframes_returns_sweep_key(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", verbose=False,
        )

        dfs = result.to_dataframes()
        assert "sweep" in dfs
        assert isinstance(dfs["sweep"], pd.DataFrame)
        assert len(dfs["sweep"]) == len(result.data)

    def test_to_dataframes_returns_same_object(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="topk", random_state=42)

        result = learner.parameter_sweep(
            "k", [2, 3], target="feature_selector",
            models="coxph", verbose=False,
        )

        dfs = result.to_dataframes()
        assert dfs["sweep"] is result.data

    def test_to_dataframes_empty_result(self):
        result = SweepResult(
            data=pd.DataFrame(), param_name="k", target="feature_selector",
        )
        dfs = result.to_dataframes()
        assert dfs == {}
