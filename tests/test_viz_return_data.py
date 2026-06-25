"""Integration tests: learner.plot_*(return_data=True) → viz.plot_*(data).

Validates the round-trip contract between high-level learner methods that
export DataFrames via ``return_data=True`` and the low-level ``mlsurv.viz``
functions that consume those DataFrames.

Marked ``@pytest.mark.integration`` — runs with ``--run-full``, not the fast tier.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv.viz._validation import validate_columns

# ---------------------------------------------------------------------------
# Module-scoped fixture: train CoxPH + RSF, evaluate, bootstrap
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def learner_with_bootstrap():
    """Create a learner trained with CoxPH + RSF, evaluated, and bootstrapped.

    Module-scoped to avoid re-fitting per test.  Uses synthetic data with
    correct ``('event', 'time')`` field names.

    .. note::

       ``bootstrap()`` is called here because ``TestPlotMetricReturnData``
       tests require bootstrap data for iteration-style round-trips.
    """
    from mlsurv.learner import SurvivalLearner
    from mlsurv.utils import SurvivalArray

    rng = np.random.RandomState(42)
    n_samples, n_features = 300, 8
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    true_coef = np.zeros(n_features)
    true_coef[:3] = [0.5, -0.3, 0.4]
    lp = X.values @ true_coef
    times = np.exp(-lp + rng.randn(n_samples) * 0.5)
    times = np.clip(times, 0.1, 10)
    events = rng.binomial(1, 0.7, n_samples).astype(bool)
    y = SurvivalArray(
        np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        )
    )

    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(test_fraction=0.3, random_state=42)
        learner.run(
            models=["coxph", "rsf"],
            tune=False,
            n_importance_repeats=5,
            bootstrap=False,
            verbose=False,
        )
        learner.bootstrap(n_bootstrap=2)
    return learner


# ===================================================================
# TestPlotMetricReturnData
# ===================================================================


@pytest.mark.integration
class TestPlotMetricReturnData:
    """Round-trip tests for plot_metric(return_data=True)."""

    def test_forest_round_trip(self, learner_with_bootstrap):
        """Summary forest: return_data → plot_metric."""
        from mlsurv.viz import plot_metric

        df = learner_with_bootstrap.plot_metric(style="forest", return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2  # at least coxph + rsf
        for col in ("model", "point", "ci_lower", "ci_upper"):
            assert col in df.columns, f"Missing column: {col}"
        assert df["point"].notna().all()
        fig = plot_metric(df, style="forest")
        assert isinstance(fig, go.Figure)

    def test_bar_round_trip(self, learner_with_bootstrap):
        """Summary bar: return_data → plot_metric."""
        from mlsurv.viz import plot_metric

        df = learner_with_bootstrap.plot_metric(style="bar", return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2
        for col in ("model", "point", "ci_lower", "ci_upper"):
            assert col in df.columns
        fig = plot_metric(df, style="bar")
        assert isinstance(fig, go.Figure)

    def test_raincloud_round_trip(self, learner_with_bootstrap):
        """Iteration raincloud: return_data → plot_metric with model_col override."""
        from mlsurv.viz import plot_metric

        df = learner_with_bootstrap.plot_metric(style="raincloud", return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ("model", "value", "metric_name"):
            assert col in df.columns, f"Missing column: {col}"
        fig = plot_metric(df, style="raincloud")
        assert isinstance(fig, go.Figure)

    def test_boxplot_round_trip(self, learner_with_bootstrap):
        """Iteration boxplot: return_data → plot_metric with model_col override."""
        from mlsurv.viz import plot_metric

        df = learner_with_bootstrap.plot_metric(style="boxplot", return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ("model", "value"):
            assert col in df.columns, f"Missing column: {col}"
        fig = plot_metric(df, style="boxplot")
        assert isinstance(fig, go.Figure)

    def test_source_cv_round_trip(self, learner_with_bootstrap):
        """source='cv' produces summary DataFrame → plot_metric."""
        from mlsurv.viz import plot_metric

        df = learner_with_bootstrap.plot_metric(
            source="cv", style="forest", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2
        for col in ("model", "point", "ci_lower", "ci_upper"):
            assert col in df.columns
        fig = plot_metric(df, style="forest")
        assert isinstance(fig, go.Figure)

    def test_metric_filter_applied_to_return_data(self, learner_with_bootstrap):
        """metric= must filter the return_data DataFrame, not just the figure.

        The figure path filters iterations to the requested metric at render
        time; the return_data path used to skip that filter and leak all
        metrics (and per-subpopulation rows).
        """
        boot_metrics = set(
            learner_with_bootstrap.bootstrap_result.iterations["metric_name"].unique()
        )
        # Guard: the fixture must expose >1 metric, else the test can't catch
        # the bug (a single-metric df would pass trivially).
        assert len(boot_metrics) > 1, (
            "fixture iterations should contain multiple metrics; "
            f"got {sorted(boot_metrics)}"
        )
        for style in ("raincloud", "boxplot"):
            df = learner_with_bootstrap.plot_metric(
                style=style, metric="harrell", source="bootstrap",
                return_data=True,
            )
            assert set(df["metric_name"].unique()) == {"harrell"}, (
                f"{style}: return_data leaked non-requested metrics: "
                f"{sorted(df['metric_name'].unique())}"
            )

    @pytest.mark.parametrize("source", ["bootstrap", "test", "cv"])
    @pytest.mark.parametrize("style", ["forest", "bar", "raincloud", "boxplot"])
    def test_multi_metric_source_style_grid(
        self, learner_with_bootstrap, source, style,
    ):
        """The only illegal multi-metric cell is iteration style + source='test'.

        Test data is point estimates with no distribution, so ``raincloud`` and
        ``boxplot`` need ``bootstrap`` or ``cv``. Every other ``source`` x
        ``style`` cell -- including ``forest``/``bar`` + ``cv``, which used to be
        wrongly rejected -- must build data without raising. This mirrors the
        single-metric contract.
        """
        illegal = style in ("raincloud", "boxplot") and source == "test"
        if illegal:
            with pytest.raises(ValueError):
                learner_with_bootstrap.plot_metric(
                    metric=["harrell", "ipcw"], source=source, style=style,
                    return_data=True,
                )
        else:
            df = learner_with_bootstrap.plot_metric(
                metric=["harrell", "ipcw"], source=source, style=style,
                return_data=True,
            )
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            # CV forest/bar must carry the spread columns, else it silently
            # degraded to the point-only test builder.
            if source == "cv" and style in ("forest", "bar"):
                for col in ("harrell_std", "ipcw_std"):
                    assert col in df.columns, f"cv {style}: missing {col}"

    def test_multi_metric_cv_summary_shape(self, learner_with_bootstrap):
        """metric= + source='cv' + forest/bar yields a wide CV summary and renders.

        Regression: this combination used to raise ValueError claiming CV tracks
        only the scoring metric. CV records several metrics per fold, so the
        forest/bar band (``point +/- z*std`` from ``cv_metric_std``) is well
        defined.
        """
        df = learner_with_bootstrap.plot_metric(
            metric=["harrell", "ipcw"], source="cv", style="forest",
            return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2  # coxph + rsf
        # return_data renames model_type -> model; metric/std columns survive.
        for col in ("model", "harrell", "harrell_std", "ipcw", "ipcw_std"):
            assert col in df.columns, f"missing column: {col}"
        stds = df[["harrell_std", "ipcw_std"]].to_numpy()
        assert np.all(stds[np.isfinite(stds)] >= 0)

        # Both forest and bar must render an actual figure end-to-end.
        for style in ("forest", "bar"):
            fig = learner_with_bootstrap.plot_metric(
                metric=["harrell", "ipcw"], source="cv", style=style,
            )
            assert isinstance(fig, go.Figure)


# ===================================================================
# TestPlotMetricTrainSource
# ===================================================================


@pytest.mark.integration
class TestPlotMetricTrainSource:
    """Opt-in apparent train-set source for plot_metric.

    Train behaves like test: point-only (no error bars), reuses the test
    rendering path, and is never selected by source='auto'.
    """

    TRAIN_METRICS = ["harrell", "ipcw", "ctd", "integrated_brier"]

    def test_single_metric_point_only(self, learner_with_bootstrap):
        """source='train' single metric → point-only data (no whiskers)."""
        from mlsurv.viz import plot_metric

        df = learner_with_bootstrap.plot_metric(
            metric="harrell", source="train", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2  # coxph + rsf
        for col in ("model", "point", "ci_lower", "ci_upper"):
            assert col in df.columns, f"missing column: {col}"
        assert df["point"].notna().all()
        # Point-only: ci bounds equal the point so _extract_ci draws no error bars.
        assert np.allclose(df["ci_lower"], df["point"])
        assert np.allclose(df["ci_upper"], df["point"])
        fig = plot_metric(df, style="bar")
        assert isinstance(fig, go.Figure)

    def test_multi_metric_point_only(self, learner_with_bootstrap):
        """source='train' multi-metric → wide point-only frame, no _std columns."""
        df = learner_with_bootstrap.plot_metric(
            metric=self.TRAIN_METRICS, source="train", style="forest",
            return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2
        for m in self.TRAIN_METRICS:
            assert m in df.columns, f"missing metric column: {m}"
            assert f"{m}_std" not in df.columns, (
                f"train is point-only; {m}_std should be absent"
            )
        fig = learner_with_bootstrap.plot_metric(
            metric=self.TRAIN_METRICS, source="train", style="forest",
        )
        assert isinstance(fig, go.Figure)

    def test_auto_never_selects_train(self, learner_with_bootstrap):
        """source='auto' resolves to bootstrap/test/cv, never train.

        The fixture has a bootstrap, so auto resolves to bootstrap. Forcing a
        summary style ('forest') yields ci_lower/ci_upper; bootstrap carries
        genuine CI width, whereas train would be point-only (ci_lower==ci_upper).
        """
        df_auto = learner_with_bootstrap.plot_metric(
            metric="harrell", source="auto", style="forest", return_data=True,
        )
        # Bootstrap source has genuine CI width; train would be point-only.
        assert not np.allclose(df_auto["ci_lower"], df_auto["ci_upper"])

    @pytest.mark.parametrize("style", ["raincloud", "boxplot"])
    def test_iteration_style_rejected(self, learner_with_bootstrap, style):
        """Iteration styles need a distribution; source='train' is point-only."""
        with pytest.raises(ValueError):
            learner_with_bootstrap.plot_metric(
                metric="harrell", source="train", style=style,
                return_data=True,
            )


# ===================================================================
# TestPlotSurvivalReturnData
# ===================================================================


@pytest.mark.integration
class TestPlotSurvivalReturnData:
    """Round-trip tests for plot_survival(return_data=True)."""

    def test_single_model_round_trip(self, learner_with_bootstrap):
        """Single model → tidy DataFrame with model column."""
        data = learner_with_bootstrap.plot_survival(
            models="coxph", return_data=True,
        )
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        for col in ("model", "time", "event", "risk_score"):
            assert col in data.columns
        assert set(data["model"].unique()) == {"coxph"}

    def test_multi_model_panel_grid(self, learner_with_bootstrap):
        """Multi-model → tidy DataFrame with model column."""
        data = learner_with_bootstrap.plot_survival(return_data=True)
        assert isinstance(data, pd.DataFrame)
        assert set(data["model"].unique()) == {"coxph", "rsf"}
        for col in ("model", "time", "event", "risk_score"):
            assert col in data.columns


# ===================================================================
# TestPlotImportanceReturnData
# ===================================================================


@pytest.mark.integration
class TestPlotImportanceReturnData:
    """Round-trip tests for plot_importance(return_data=True)."""

    def test_basic_round_trip(self, learner_with_bootstrap):
        """Multi-model importance → plot_importance."""
        from mlsurv.viz import plot_importance

        df = learner_with_bootstrap.plot_importance(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ("feature", "perm_mean", "perm_std", "model"):
            assert col in df.columns, f"Missing column: {col}"
        fig = plot_importance(df, n=5, model_col="model")
        assert isinstance(fig, go.Figure)

    def test_single_model_round_trip(self, learner_with_bootstrap):
        """Single-model importance → plot_importance."""
        from mlsurv.viz import plot_importance

        df = learner_with_bootstrap.plot_importance(
            models="coxph", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ("feature", "perm_mean"):
            assert col in df.columns
        fig = plot_importance(df, n=5, model_col="model")
        assert isinstance(fig, go.Figure)

    def test_heatmap_cross_compatibility(self, learner_with_bootstrap):
        """Multi-model importance data → plot_importance_heatmap."""
        from mlsurv.viz import plot_importance_heatmap

        df = learner_with_bootstrap.plot_importance(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert "model" in df.columns
        fig = plot_importance_heatmap(df, n=5, model_col="model")
        assert isinstance(fig, go.Figure)


# ===================================================================
# TestPlotCalibrationReturnData
# ===================================================================


@pytest.mark.integration
class TestPlotCalibrationReturnData:
    """Round-trip tests for plot_calibration(return_data=True)."""

    def test_single_model_round_trip(self, learner_with_bootstrap):
        """Single-model → tidy DataFrame."""
        data = learner_with_bootstrap.plot_calibration(
            models="coxph", eval_times=[1, 3, 5], return_data=True,
        )
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        for col in ("model", "eval_time", "predicted", "observed"):
            assert col in data.columns, f"Missing column: {col}"

    def test_multi_model_round_trip(self, learner_with_bootstrap):
        """Multi-model → tidy DataFrame."""
        data = learner_with_bootstrap.plot_calibration(
            eval_times=3.0, return_data=True,
        )
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        for col in ("model", "eval_time", "predicted", "observed"):
            assert col in data.columns, f"Missing column: {col}"


# ===================================================================
# TestPlotHazardRatiosMultivariateReturnData
# ===================================================================


@pytest.mark.integration
class TestPlotHazardRatiosMultivariateReturnData:
    """Round-trip test for plot_hazard_ratios(kind='multivariate', return_data=True)."""

    def test_hr_round_trip(self, learner_with_bootstrap):
        """HR DataFrame → plot_hazard_ratios."""
        from mlsurv.viz import plot_hazard_ratios

        df = learner_with_bootstrap.plot_hazard_ratios(
            kind='multivariate', return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ("feature", "hr", "ci_lower", "ci_upper"):
            assert col in df.columns
        assert (df["hr"] > 0).all()
        fig = plot_hazard_ratios(df)
        assert isinstance(fig, go.Figure)


# ===================================================================
# TestColumnValidation
# ===================================================================


class TestColumnValidation:
    """Tests for validate_columns error reporting."""

    def test_passes_when_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_columns(df, ["a", "b"], "test_func")

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_columns(df, ["a", "c"], "test_func")

    def test_error_lists_missing_and_available(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="Available:.*'x'"):
            validate_columns(df, ["y", "z"], "my_plot")

    def test_rejects_none_input(self):
        with pytest.raises(TypeError, match="expected a pandas DataFrame"):
            validate_columns(None, ["a"], "test_func")

    def test_rejects_dict_input(self):
        with pytest.raises(TypeError, match="expected a pandas DataFrame"):
            validate_columns({"a": [1]}, ["a"], "test_func")

    def test_rejects_duplicate_columns(self):
        """validate_columns raises ValueError on duplicate column names."""
        df = pd.DataFrame([[1, 2]], columns=["a", "a"])
        with pytest.raises(ValueError, match="duplicate columns"):
            validate_columns(df, ["a"], "test_func")

    def test_plot_metric_rejects_wrong_summary_columns(self):
        """plot_metric raises ValueError for summary data with wrong columns."""
        from mlsurv.viz import plot_metric

        bad_df = pd.DataFrame({"model": ["A"], "point": [0.5],
                                "wrong": [0.4], "also_wrong": [0.6]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_metric(bad_df, style="forest")

    def test_plot_metric_rejects_wrong_iteration_columns(self):
        """plot_metric raises ValueError for iteration data missing value col."""
        from mlsurv.viz import plot_metric

        bad_df = pd.DataFrame({"model_type": ["A"], "wrong": [0.5]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_metric(bad_df, style="raincloud", model_col="model_type")

    def test_plot_importance_rejects_wrong_columns(self):
        """plot_importance raises ValueError on wrong column names."""
        from mlsurv.viz import plot_importance

        bad_df = pd.DataFrame({"wrong_col": [0.5], "another": [0.1]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_importance(bad_df)

    def test_plot_hazard_ratios_rejects_wrong_columns(self):
        """plot_hazard_ratios raises ValueError on wrong column names."""
        from mlsurv.viz import plot_hazard_ratios

        bad_df = pd.DataFrame({"a": [1.0], "b": [0.5], "c": [1.5], "d": ["f1"]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_hazard_ratios(bad_df)

    def test_plot_shap_importance_rejects_wrong_columns(self):
        """plot_shap_importance raises ValueError on wrong column names."""
        from mlsurv.viz import plot_shap_importance

        bad_df = pd.DataFrame({"wrong": [0.1], "cols": [0.2]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_shap_importance(bad_df)

    def test_plot_shap_beeswarm_rejects_wrong_columns(self):
        """plot_shap_beeswarm raises ValueError on wrong column names."""
        from mlsurv.viz import plot_shap_beeswarm

        # With feature_value present, bypasses X requirement — hits validate_columns
        bad_df = pd.DataFrame({"wrong": [0.1], "feature_value": [0.2]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_shap_beeswarm(bad_df)

    def test_plot_km_rejects_missing_columns(self):
        """plot_km raises ValueError on DataFrame missing time/event."""
        from mlsurv.viz import plot_km

        bad_df = pd.DataFrame({"wrong": [1.0], "cols": [True]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_km(bad_df)

    def test_plot_km_dict_rejects_missing_columns(self):
        """plot_km raises ValueError when a dict entry is missing time/event."""
        from mlsurv.viz import plot_km

        bad_data = {"coxph": pd.DataFrame({"wrong": [1.0], "cols": [True]})}
        with pytest.raises(ValueError, match=r"plot_km\[coxph\].*missing required"):
            plot_km(bad_data)

    def test_plot_km_rejects_wrong_type(self):
        """plot_km raises TypeError for non-DataFrame/dict input."""
        from mlsurv.viz import plot_km

        with pytest.raises(TypeError, match="expected a pandas DataFrame or dict"):
            plot_km([1, 2, 3])

    def test_plot_metric_multi_forest_rejects_missing_metric_columns(self):
        """plot_metric with metric=[...] raises ValueError when metric cols missing."""
        from mlsurv.viz import plot_metric

        df = pd.DataFrame({"model_type": ["A"], "harrell": [0.7]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_metric(df, model_col="model_type", metric=["harrell", "nonexistent"])

    def test_plot_metric_style_data_mismatch_raises(self):
        """plot_metric raises ValueError when style conflicts with data type."""
        from mlsurv.viz import plot_metric

        iteration_df = pd.DataFrame({
            "model": ["A", "A"], "value": [0.7, 0.8],
        })
        with pytest.raises(ValueError, match="requires summary data"):
            plot_metric(iteration_df, style="forest")

    def test_plot_coefficients_rejects_wrong_columns(self):
        """plot_coefficients raises ValueError on wrong column names."""
        from mlsurv.viz import plot_coefficients

        bad_df = pd.DataFrame({"wrong": [0.5], "cols": ["f1"]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_coefficients(bad_df)

    def test_plot_importance_heatmap_rejects_wrong_columns(self):
        """plot_importance_heatmap raises ValueError on wrong column names."""
        from mlsurv.viz import plot_importance_heatmap

        bad_df = pd.DataFrame({"wrong": [0.5], "cols": ["f1"]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_importance_heatmap(bad_df)

    def test_plot_calibration_single_rejects_missing_keys(self):
        """plot_calibration raises ValueError for list[dict] with missing keys."""
        from mlsurv.viz import plot_calibration

        bad_data = [{"eval_time": 12, "predicted": [0.5]}]
        with pytest.raises(ValueError, match=r"data\[0\].*missing required keys"):
            plot_calibration(bad_data)

    def test_plot_calibration_multi_rejects_missing_keys(self):
        """plot_calibration raises ValueError for dict[str, dict] with missing keys."""
        from mlsurv.viz import plot_calibration

        bad_data = {"coxph": {"eval_time": 12}}
        with pytest.raises(ValueError, match=r"data\['coxph'\].*missing required keys"):
            plot_calibration(bad_data)

    def test_plot_calibration_rejects_wrong_type(self):
        """plot_calibration raises TypeError for non-list/dict input."""
        from mlsurv.viz import plot_calibration

        with pytest.raises(TypeError, match="expected list.*or dict"):
            plot_calibration("not valid")

    def test_plot_calibration_rejects_non_dict_list_item(self):
        """plot_calibration raises TypeError when list contains non-dict."""
        from mlsurv.viz import plot_calibration

        with pytest.raises(TypeError, match=r"data\[0\] expected a dict"):
            plot_calibration(["not a dict"])

    def test_plot_calibration_rejects_non_dict_multi_value(self):
        """plot_calibration raises TypeError when dict value is not a dict."""
        from mlsurv.viz import plot_calibration

        with pytest.raises(TypeError, match=r"data\['coxph'\] expected a dict"):
            plot_calibration({"coxph": "not a dict"})

    def test_plot_calibration_rejects_empty_list(self):
        """plot_calibration raises ValueError for empty list."""
        from mlsurv.viz import plot_calibration

        with pytest.raises(ValueError, match="empty list"):
            plot_calibration([])

    def test_plot_calibration_rejects_empty_dict(self):
        """plot_calibration raises ValueError for empty dict."""
        from mlsurv.viz import plot_calibration

        with pytest.raises(ValueError, match="empty dict"):
            plot_calibration({})

    def test_plot_metric_style_raincloud_rejects_summary_data(self):
        """plot_metric raises ValueError when raincloud used with summary data."""
        from mlsurv.viz import plot_metric

        summary_df = pd.DataFrame({
            "model": ["A"], "point": [0.7],
            "ci_lower": [0.6], "ci_upper": [0.8],
        })
        with pytest.raises(ValueError, match="requires iteration data"):
            plot_metric(summary_df, style="raincloud")

    def test_plot_metric_point_column_not_misclassified(self):
        """Iteration data with a coincidental 'point' column is not misclassified as summary."""
        from mlsurv.viz import plot_metric

        # Iteration data that happens to have a 'point' column (not CI columns)
        df = pd.DataFrame({
            "model": ["A"] * 10 + ["B"] * 10,
            "value": np.random.RandomState(42).normal(0.7, 0.05, 20),
            "point": range(20),  # coincidental column name
        })
        # Should detect as iteration data (no ci_lower), not summary
        fig = plot_metric(df, style="auto")
        assert isinstance(fig, go.Figure)

    def test_plot_km_dict_rejects_missing_risk_col(self):
        """plot_km dict path validates risk_col in sub-DataFrames."""
        from mlsurv.viz import plot_km

        # Dict with DataFrames missing the default 'risk_score' column
        data = {
            "CoxPH": pd.DataFrame({
                "time": [1.0, 2.0, 3.0],
                "event": [True, False, True],
                # missing "risk_score"
            }),
        }
        with pytest.raises(ValueError, match="missing required columns.*risk_score"):
            plot_km(data)

    def test_plot_km_dataframe_rejects_missing_risk_col(self):
        """plot_km DataFrame path validates explicit risk_col."""
        from mlsurv.viz import plot_km

        df = pd.DataFrame({
            "time": [1.0, 2.0, 3.0],
            "event": [True, False, True],
        })
        with pytest.raises(ValueError, match="missing required columns.*my_risk"):
            plot_km(df, risk_col="my_risk")

    def test_plot_km_dataframe_rejects_missing_group_col(self):
        """plot_km DataFrame path validates explicit group_col."""
        from mlsurv.viz import plot_km

        df = pd.DataFrame({
            "time": [1.0, 2.0, 3.0],
            "event": [True, False, True],
        })
        with pytest.raises(ValueError, match="missing required columns.*stratum"):
            plot_km(df, group_col="stratum")


# ===================================================================
# TestLowLevelStandalone
# ===================================================================


class TestLowLevelStandalone:
    """Low-level viz functions work with synthetic DataFrames."""

    def test_plot_metric_summary(self):
        """Synthetic summary DataFrame → plot_metric(style='forest')."""
        from mlsurv.viz import plot_metric

        df = pd.DataFrame({
            "model": ["CoxPH", "RSF", "GBSA"],
            "point": [0.72, 0.75, 0.70],
            "ci_lower": [0.68, 0.71, 0.65],
            "ci_upper": [0.76, 0.79, 0.75],
        })
        fig = plot_metric(df, style="forest", metric="harrell")
        assert isinstance(fig, go.Figure)

    def test_plot_metric_iteration(self):
        """Synthetic iteration DataFrame → plot_metric(style='raincloud')."""
        from mlsurv.viz import plot_metric

        rng = np.random.RandomState(42)
        rows = []
        for model in ("CoxPH", "RSF"):
            for i in range(20):
                rows.append({
                    "model_type": model,
                    "metric_name": "harrell",
                    "value": rng.normal(0.7, 0.05),
                    "iteration": i,
                })
        df = pd.DataFrame(rows)
        fig = plot_metric(df, style="raincloud", model_col="model_type")
        assert isinstance(fig, go.Figure)

    def test_plot_coefficients(self):
        """Synthetic coefficient DataFrame → plot_coefficients."""
        from mlsurv.viz import plot_coefficients

        df = pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(10)],
            "coefficient": np.linspace(-0.5, 0.5, 10),
            "coef_std": np.full(10, 0.1),
        })
        fig = plot_coefficients(df, n=5)
        assert isinstance(fig, go.Figure)

    def test_plot_shap_importance_synthetic(self):
        """Synthetic SHAP DataFrame → plot_shap_importance."""
        from mlsurv.viz import plot_shap_importance

        rng = np.random.RandomState(42)
        rows = []
        for feat in ("age", "stage", "grade"):
            for idx in range(30):
                rows.append({
                    "feature": feat,
                    "sample_idx": idx,
                    "shap_value": rng.normal(0, 0.1),
                })
        df = pd.DataFrame(rows)
        fig = plot_shap_importance(df, n=3)
        assert isinstance(fig, go.Figure)

    def test_plot_shap_beeswarm_synthetic(self):
        """Synthetic SHAP + X DataFrame → plot_shap_beeswarm."""
        from mlsurv.viz import plot_shap_beeswarm

        rng = np.random.RandomState(42)
        n_samples = 30
        rows = []
        for feat in ("age", "stage"):
            for idx in range(n_samples):
                rows.append({
                    "feature": feat,
                    "sample_idx": idx,
                    "shap_value": rng.normal(0, 0.1),
                })
        shap_df = pd.DataFrame(rows)
        X = pd.DataFrame({
            "age": rng.uniform(30, 80, n_samples),
            "stage": rng.randint(1, 5, n_samples),
        })
        fig = plot_shap_beeswarm(shap_df, X=X, n=2)
        assert isinstance(fig, go.Figure)

    def test_plot_hazard_ratios_synthetic(self):
        """Synthetic HR DataFrame → plot_hazard_ratios."""
        from mlsurv.viz import plot_hazard_ratios

        df = pd.DataFrame({
            "feature": ["age", "stage", "grade"],
            "hr": [1.2, 0.8, 1.5],
            "ci_lower": [1.0, 0.6, 1.1],
            "ci_upper": [1.4, 1.0, 1.9],
        })
        fig = plot_hazard_ratios(df)
        assert isinstance(fig, go.Figure)

    def test_plot_schoenfeld_synthetic(self):
        """Synthetic Schoenfeld DataFrame → plot_schoenfeld."""
        from mlsurv.viz import plot_schoenfeld

        rng = np.random.RandomState(42)
        rows = []
        for feat in ("age", "stage"):
            for t in rng.uniform(1, 50, 30):
                rows.append({"feature": feat, "time": t, "residual": rng.normal()})
        df = pd.DataFrame(rows)
        fig = plot_schoenfeld(df)
        assert isinstance(fig, go.Figure)

    def test_plot_deviance_synthetic(self):
        """Synthetic deviance DataFrame → plot_deviance."""
        from mlsurv.viz import plot_deviance

        rng = np.random.RandomState(42)
        n = 50
        df = pd.DataFrame({
            "linear_predictor": rng.normal(0, 1, n),
            "deviance": rng.normal(0, 1.5, n),
            "event": rng.binomial(1, 0.6, n).astype(bool),
        })
        fig = plot_deviance(df)
        assert isinstance(fig, go.Figure)


# ===================================================================
# TestPlotDiagnosticsReturnData
# ===================================================================


@pytest.mark.integration
class TestPlotDiagnosticsReturnData:
    """Round-trip tests for plot_diagnostics(return_data=True)."""

    def test_schoenfeld_round_trip(self, learner_with_bootstrap):
        """learner.plot_diagnostics(kind='schoenfeld', return_data=True) → plot_schoenfeld(df)."""
        from mlsurv.viz import plot_schoenfeld

        df = learner_with_bootstrap.plot_diagnostics(
            kind="schoenfeld", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        for col in ("feature", "time", "residual"):
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) > 0
        assert df["feature"].nunique() >= 1

        fig = plot_schoenfeld(df)
        assert isinstance(fig, go.Figure)

    def test_deviance_round_trip(self, learner_with_bootstrap):
        """learner.plot_diagnostics(kind='deviance', return_data=True) → plot_deviance(df)."""
        from mlsurv.viz import plot_deviance

        df = learner_with_bootstrap.plot_diagnostics(
            kind="deviance", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        for col in ("linear_predictor", "deviance", "event"):
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) > 0

        fig = plot_deviance(df)
        assert isinstance(fig, go.Figure)


# ===================================================================
# TestPlotMetricMultiReturnData
# ===================================================================


@pytest.mark.integration
class TestPlotMetricMultiReturnData:
    """Round-trip tests for plot_metric(metric=[...], return_data=True)."""

    def test_multi_metric_bootstrap_round_trip(self, learner_with_bootstrap):
        """Bootstrap wide DF has metric + _std cols; round-trips via low-level plot_metric."""
        from mlsurv.viz import plot_metric

        df = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
            style="forest", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert "model" in df.columns
        assert len(df) >= 2  # coxph + rsf
        for m in ("harrell", "integrated_brier"):
            assert m in df.columns, f"Missing metric column: {m}"
            assert f"{m}_std" in df.columns, f"Missing std column: {m}_std"
        assert df["harrell"].notna().all()
        # Round-trip through low-level renderer
        fig = plot_metric(
            df, metric=["harrell", "integrated_brier"],
        )
        assert isinstance(fig, go.Figure)

    def test_multi_metric_test_source(self, learner_with_bootstrap):
        """source='test' produces wide DF without _std cols."""
        df = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
            source="test", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert "model" in df.columns
        for m in ("harrell", "integrated_brier"):
            assert m in df.columns
        # Test source has no uncertainty — no _std columns
        std_cols = [c for c in df.columns if c.endswith("_std")]
        assert len(std_cols) == 0

    def test_multi_metric_renders_figure(self, learner_with_bootstrap):
        """return_data=False returns a plotly Figure."""
        fig = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
        )
        assert isinstance(fig, go.Figure)

    def test_multi_metric_cv_raincloud_succeeds(self, learner_with_bootstrap):
        """source='cv', style='raincloud' succeeds and returns a DataFrame via return_data=True."""
        df = learner_with_bootstrap.plot_metric(
            metric=["harrell", "ipcw"], source="cv",
            style="raincloud", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_multi_metric_cv_forest_succeeds(self, learner_with_bootstrap):
        """source='cv', style='forest' succeeds: CV tracks several metrics per fold."""
        df = learner_with_bootstrap.plot_metric(
            metric=["harrell", "ipcw"], source="cv",
            style="forest", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    # ---- CV forest/bar default to ± SD (mirrors single-metric CV) ----

    _CV_METRICS = ["harrell", "integrated_brier"]

    def _error_arrays(self, fig):
        """Sorted half-width arrays for every trace carrying error_x."""
        out = []
        for tr in fig.data:
            ex = getattr(tr, "error_x", None)
            if ex is not None and ex.array is not None:
                out.append(np.sort(np.asarray(ex.array, dtype=float)))
        return out

    def _error_pair(self, fig):
        """First error_x trace's (sorted plus, sorted minus) half-widths.

        For a symmetric error bar ``arrayminus`` is None, so the minus side
        falls back to the plus side. Used to compare asymmetric percentile
        CIs (plus != minus) between forest and bar.
        """
        for tr in fig.data:
            ex = getattr(tr, "error_x", None)
            if ex is not None and ex.array is not None:
                plus = np.sort(np.asarray(ex.array, dtype=float))
                minus = ex.arrayminus
                minus = (
                    np.sort(np.asarray(minus, dtype=float))
                    if minus is not None else plus
                )
                return plus, minus
        return None

    def _error_map(self, fig):
        """First error_x trace's {model: (plus, minus)} keyed off the y labels.

        Unlike :meth:`_error_pair`, this keeps each model paired with its own
        half-widths so a forest/bar comparison catches per-model misassignment
        (the two renderers emit models in different y-order).
        """
        for tr in fig.data:
            ex = getattr(tr, "error_x", None)
            if ex is not None and ex.array is not None:
                plus = np.asarray(ex.array, dtype=float)
                minus = (
                    np.asarray(ex.arrayminus, dtype=float)
                    if ex.arrayminus is not None else plus
                )
                return {
                    model: (p, m)
                    for model, p, m in zip(tr.y, plus, minus)
                }
        return None

    def test_cv_forest_defaults_to_sd(self, learner_with_bootstrap):
        """Multi-metric CV forest shows point ± std (not z*std) by default."""
        wide = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="cv", style="forest",
            return_data=True,
        )
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="cv", style="forest",
        )
        hovers = [t.hovertemplate for t in fig.data if t.hovertemplate]
        assert any("1 SD" in h for h in hovers)
        assert not any("% CI" in h for h in hovers)
        # First panel (harrell) error bars equal raw harrell_std (z=1).
        errs = self._error_arrays(fig)
        np.testing.assert_allclose(
            errs[0], np.sort(wide["harrell_std"].values), atol=1e-9,
        )

    def test_cv_forest_explicit_float_uses_ci(self, learner_with_bootstrap):
        """An explicit float still forces a z*std CI band for CV."""
        from scipy.stats import norm

        wide = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="cv", style="forest",
            return_data=True,
        )
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="cv", style="forest",
            uncertainty=0.95,
        )
        hovers = [t.hovertemplate for t in fig.data if t.hovertemplate]
        assert any("95% CI" in h for h in hovers)
        assert not any("1 SD" in h for h in hovers)
        errs = self._error_arrays(fig)
        expected = norm.ppf(0.975) * np.sort(wide["harrell_std"].values)
        np.testing.assert_allclose(errs[0], expected, atol=1e-9)

    def test_cv_bar_defaults_to_sd(self, learner_with_bootstrap):
        """Multi-metric CV bar shows point ± std error bars by default."""
        wide = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="cv", style="forest",
            return_data=True,
        )
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="cv", style="bar",
        )
        errs = self._error_arrays(fig)
        assert errs, "bar plot has no error bars"
        np.testing.assert_allclose(
            errs[0], np.sort(wide["harrell_std"].values), atol=1e-9,
        )

    def test_cv_bar_explicit_float_uses_ci(self, learner_with_bootstrap):
        """An explicit float forces a z*std CI band on the CV bar too."""
        from scipy.stats import norm

        wide = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="cv", style="forest",
            return_data=True,
        )
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="cv", style="bar",
            uncertainty=0.95,
        )
        errs = self._error_arrays(fig)
        expected = norm.ppf(0.975) * np.sort(wide["harrell_std"].values)
        np.testing.assert_allclose(errs[0], expected, atol=1e-9)

    # ---- Bootstrap forest/bar both draw the percentile CI ----

    def test_bootstrap_bar_uses_percentile_ci(self, learner_with_bootstrap):
        """Multi-metric bootstrap BAR draws the percentile CI, not z*std.

        Regression guard: the bar used to reconstruct ``point +/- z*std`` from
        ``{m}_std`` and so disagreed with the forest's percentile CI.
        """
        from scipy.stats import norm

        wide = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="forest",
            return_data=True,
        )
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="bar",
        )
        m = self._CV_METRICS[0]
        exp_plus = np.sort((wide[f"{m}_ci_upper"] - wide[m]).values)
        exp_minus = np.sort((wide[m] - wide[f"{m}_ci_lower"]).values)
        plus, minus = self._error_pair(fig)
        np.testing.assert_allclose(plus, exp_plus, atol=1e-9)
        np.testing.assert_allclose(minus, exp_minus, atol=1e-9)
        # And NOT the old normal-approx z*std reconstruction.
        z_std = np.sort(norm.ppf(0.975) * wide[f"{m}_std"].values)
        assert not np.allclose(plus, z_std, atol=1e-6)

    def test_bootstrap_forest_and_bar_match(self, learner_with_bootstrap):
        """Bootstrap forest and bar draw the same CI per model.

        Compared per model (not as a sorted multiset) so a swap of one model's
        band onto another would fail -- the two renderers emit models in
        different y-order, so a sorted comparison would miss it.
        """
        forest = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="forest",
        )
        bar = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="bar",
        )
        fmap = self._error_map(forest)
        bmap = self._error_map(bar)
        assert fmap and bmap and set(fmap) == set(bmap)
        for model in fmap:
            np.testing.assert_allclose(fmap[model], bmap[model], atol=1e-9)

    def test_bootstrap_bar_hover_says_ci(self, learner_with_bootstrap):
        """The bootstrap bar hover states the percentile CI, not '1 SD'."""
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="bar",
        )
        hovers = [t.hovertemplate for t in fig.data if t.hovertemplate]
        assert hovers, "bootstrap bar has no hovertemplate"
        assert any("% CI" in h for h in hovers)
        assert not any("1 SD" in h for h in hovers)

    def test_bootstrap_bar_explicit_float_uses_ci(self, learner_with_bootstrap):
        """uncertainty=0.90 recomputes the bar's percentile CI at that level."""
        m = self._CV_METRICS[0]
        wide = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="forest",
            uncertainty=0.90, return_data=True,
        )
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="bar",
            uncertainty=0.90,
        )
        plus, minus = self._error_pair(fig)
        np.testing.assert_allclose(
            plus, np.sort((wide[f"{m}_ci_upper"] - wide[m]).values), atol=1e-9,
        )
        np.testing.assert_allclose(
            minus, np.sort((wide[m] - wide[f"{m}_ci_lower"]).values), atol=1e-9,
        )

    def test_test_source_bar_has_no_band(self, learner_with_bootstrap):
        """source='test' bar draws no error band and no CI hover."""
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="test", style="bar",
        )
        assert self._error_pair(fig) is None
        hovers = [t.hovertemplate for t in fig.data if t.hovertemplate]
        assert not any("% CI" in h for h in hovers)

    def test_bootstrap_bar_sd_override(self, learner_with_bootstrap):
        """uncertainty='sd' overrides the bootstrap bar to a ± std band."""
        wide = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="forest",
            return_data=True,
        )
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="bar",
            uncertainty="sd",
        )
        m = self._CV_METRICS[0]
        exp = np.sort(wide[f"{m}_std"].values)
        plus, minus = self._error_pair(fig)
        np.testing.assert_allclose(plus, exp, atol=1e-9)
        np.testing.assert_allclose(minus, exp, atol=1e-9)

    def test_bootstrap_forest_sd_override(self, learner_with_bootstrap):
        """uncertainty='sd' overrides the bootstrap forest to a ± std band."""
        wide = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="forest",
            return_data=True,
        )
        fig = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="bootstrap", style="forest",
            uncertainty="sd",
        )
        m = self._CV_METRICS[0]
        exp = np.sort(wide[f"{m}_std"].values)
        plus, minus = self._error_pair(fig)
        np.testing.assert_allclose(plus, exp, atol=1e-9)
        np.testing.assert_allclose(minus, exp, atol=1e-9)

    def test_cv_summary_title_states_mean_and_folds(self, learner_with_bootstrap):
        """CV forest/bar title states reps/folds and 'CV mean ± SD'."""
        for style in ("forest", "bar"):
            fig = learner_with_bootstrap.plot_metric(
                metric=self._CV_METRICS, source="cv", style=style,
            )
            title = fig.layout.title.text
            assert "CV mean ± SD" in title, (style, title)
            assert "fold" in title, (style, title)
        # Explicit float switches the wording to the CI band.
        fig_ci = learner_with_bootstrap.plot_metric(
            metric=self._CV_METRICS, source="cv", style="forest",
            uncertainty=0.95,
        )
        assert "CV mean ± 95% CI" in fig_ci.layout.title.text

    def test_sd_rejected_for_test_source(self, learner_with_bootstrap):
        """uncertainty='sd' stays unsupported for the test source (no spread).

        Bootstrap now supports 'sd' (it carries a per-model std), but the test
        set yields a single point estimate, so 'sd' has no meaning there.
        """
        with pytest.raises(ValueError,
                           match="not supported with source='test'"):
            learner_with_bootstrap.plot_metric(
                metric=self._CV_METRICS, source="test",
                style="forest", uncertainty="sd",
            )

    def test_single_element_list_collapses_to_single_panel(self, learner_with_bootstrap):
        """metric=[single] is now valid and collapses to a single-panel plot."""
        fig = learner_with_bootstrap.plot_metric(metric=["harrell"])
        assert isinstance(fig, go.Figure)
        df = learner_with_bootstrap.plot_metric(
            metric=["harrell"], style="forest", return_data=True,
        )
        # Single-panel summary frame, not the wide multi-metric layout.
        assert isinstance(df, pd.DataFrame)
        assert "point" in df.columns

    def test_multi_metric_invalid_style_raises(self, learner_with_bootstrap):
        """Truly invalid style raises ValueError for multi-metric."""
        with pytest.raises(ValueError, match="not supported with multi-metric"):
            learner_with_bootstrap.plot_metric(
                metric=["harrell", "ipcw"], style="beeswarm",
            )

    def test_multi_metric_unknown_metric_raises(self, learner_with_bootstrap):
        """Unknown metric name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric 'nonexistent'"):
            learner_with_bootstrap.plot_metric(
                metric=["harrell", "nonexistent"],
            )

    def test_multi_metric_models_filter(self, learner_with_bootstrap):
        """models='coxph' filters to 1 row (forest) or 1 model (raincloud)."""
        df = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
            style="forest",
            models="coxph", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["model"].iloc[0] == "coxph"

    def test_single_string_metric_is_single_panel(self, learner_with_bootstrap):
        """metric='harrell' (string) is now valid and draws a single panel."""
        fig = learner_with_bootstrap.plot_metric(metric="harrell")
        assert isinstance(fig, go.Figure)

    def test_multi_metric_nonexistent_model_raises(self, learner_with_bootstrap):
        """Filtering to a nonexistent model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown models"):
            learner_with_bootstrap.plot_metric(
                metric=["harrell", "integrated_brier"],
                models=["nonexistent_model"],
            )

    def test_multi_metric_test_round_trip(self, learner_with_bootstrap):
        """source='test' wide DF round-trips through low-level plot_metric."""
        from mlsurv.viz import plot_metric

        df = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
            source="test", return_data=True,
        )
        fig = plot_metric(
            df, metric=["harrell", "integrated_brier"],
        )
        assert isinstance(fig, go.Figure)

    def test_multi_metric_bootstrap_sd_override_forest(self, learner_with_bootstrap):
        """uncertainty='sd' draws a ± std band on the bootstrap forest.

        Default source here resolves to bootstrap (the fixture is bootstrapped),
        which carries a per-model std, so 'sd' overrides the percentile CI with
        a ± std band whose hover says "1 SD" (not "% CI").
        """
        fig = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
            style="forest",
            uncertainty="sd",
        )
        assert isinstance(fig, go.Figure)
        hovers = [t.hovertemplate for t in fig.data if t.hovertemplate]
        assert any("1 SD" in h for h in hovers)
        assert not any("% CI" in h for h in hovers)

    # -- Multi-metric iteration-style tests --

    def test_multi_metric_raincloud_bootstrap(self, learner_with_bootstrap):
        """style='raincloud' with metric=[...] produces a Figure from bootstrap."""
        fig = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
            style="raincloud",
        )
        assert isinstance(fig, go.Figure)

    def test_multi_metric_boxplot_bootstrap(self, learner_with_bootstrap):
        """style='boxplot' with metric=[...] produces a Figure from bootstrap."""
        fig = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
            style="boxplot",
        )
        assert isinstance(fig, go.Figure)

    def test_multi_metric_bar_bootstrap(self, learner_with_bootstrap):
        """style='bar' with metric=[...] produces a Figure from bootstrap."""
        fig = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
            style="bar",
        )
        assert isinstance(fig, go.Figure)

    def test_multi_metric_raincloud_return_data(self, learner_with_bootstrap):
        """style='raincloud' return_data gives long-format iteration DataFrame."""
        df = learner_with_bootstrap.plot_metric(
            metric=["harrell", "integrated_brier"],
            style="raincloud", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        for col in ("model", "metric_name", "value", "iteration"):
            assert col in df.columns, f"Missing column: {col}"
        assert set(df["metric_name"].unique()) == {"harrell", "integrated_brier"}

    def test_multi_metric_iteration_test_source_raises(self, learner_with_bootstrap):
        """style='raincloud' with source='test' raises ValueError."""
        with pytest.raises(ValueError, match="requires source='bootstrap'"):
            learner_with_bootstrap.plot_metric(
                metric=["harrell", "integrated_brier"],
                style="raincloud", source="test",
            )


# ===================================================================
# TestMultiPanelLowLevel
# ===================================================================


class TestMultiPanelLowLevel:
    """Low-level tests for _plot_metric_multi_panel."""

    def test_multi_panel_raincloud_basic(self):
        """Long-format data, 2 metrics, 3 models → Figure."""
        from mlsurv.viz import plot_metric

        rng = np.random.RandomState(42)
        rows = []
        for model in ("CoxPH", "RSF", "GBSA"):
            for metric in ("harrell", "integrated_brier"):
                for i in range(20):
                    rows.append({
                        "model_type": model, "metric_name": metric,
                        "value": rng.normal(0.7, 0.05), "iteration": i,
                    })
        df = pd.DataFrame(rows)
        fig = plot_metric(
            df, style="raincloud", model_col="model_type",
            metric=["harrell", "integrated_brier"],
        )
        assert isinstance(fig, go.Figure)

    def test_multi_panel_boxplot_basic(self):
        """Long-format data, 2 metrics, 3 models → boxplot Figure."""
        from mlsurv.viz import plot_metric

        rng = np.random.RandomState(42)
        rows = []
        for model in ("CoxPH", "RSF"):
            for metric in ("harrell", "ipcw"):
                for i in range(15):
                    rows.append({
                        "model_type": model, "metric_name": metric,
                        "value": rng.normal(0.7, 0.05), "iteration": i,
                    })
        df = pd.DataFrame(rows)
        fig = plot_metric(
            df, style="boxplot", model_col="model_type",
            metric=["harrell", "ipcw"],
        )
        assert isinstance(fig, go.Figure)

    def test_multi_panel_bar_basic(self):
        """Long-format summary data, 2 metrics → bar Figure."""
        from mlsurv.viz import plot_metric

        df = pd.DataFrame({
            "model": ["CoxPH", "CoxPH", "RSF", "RSF"],
            "metric_name": ["harrell", "ipcw", "harrell", "ipcw"],
            "point": [0.72, 0.71, 0.75, 0.74],
            "ci_lower": [0.68, 0.67, 0.71, 0.70],
            "ci_upper": [0.76, 0.75, 0.79, 0.78],
        })
        fig = plot_metric(
            df, style="bar",
            metric=["harrell", "ipcw"],
        )
        assert isinstance(fig, go.Figure)

    def test_multi_panel_grid_layout(self):
        """4 metrics → 2 rows × 2 cols (check subplot annotations)."""
        from mlsurv.viz import plot_metric

        rng = np.random.RandomState(42)
        metrics = ["harrell", "ipcw", "integrated_brier", "brier"]
        rows = []
        for model in ("CoxPH", "RSF"):
            for metric in metrics:
                for i in range(10):
                    rows.append({
                        "model_type": model, "metric_name": metric,
                        "value": rng.normal(0.7, 0.05), "iteration": i,
                    })
        df = pd.DataFrame(rows)
        fig = plot_metric(
            df, style="raincloud", model_col="model_type",
            metric=metrics,
        )
        assert isinstance(fig, go.Figure)
        # 4 metrics → 4 subplot title annotations
        annotations = [a for a in fig.layout.annotations if a.text]
        assert len(annotations) >= 4

    def test_multi_panel_reference_lines(self):
        """Each panel gets its metric's reference line."""
        from mlsurv.viz import plot_metric

        rng = np.random.RandomState(42)
        rows = []
        for model in ("CoxPH", "RSF"):
            for metric in ("harrell", "integrated_brier"):
                for i in range(10):
                    rows.append({
                        "model_type": model, "metric_name": metric,
                        "value": rng.normal(0.7, 0.05), "iteration": i,
                    })
        df = pd.DataFrame(rows)
        fig = plot_metric(
            df, style="raincloud", model_col="model_type",
            metric=["harrell", "integrated_brier"],
        )
        # Should have vertical reference lines (shapes)
        vlines = [s for s in fig.layout.shapes if s.type == "line"]
        assert len(vlines) >= 2  # one per panel

    def test_multi_panel_empty_raises(self):
        """Empty DataFrame raises ValueError."""
        from mlsurv.viz import plot_metric

        df = pd.DataFrame(columns=["model_type", "metric_name", "value"])
        with pytest.raises(ValueError, match="empty"):
            plot_metric(
                df, style="raincloud", model_col="model_type",
                metric=["harrell"],
            )

    def test_wide_to_long_nan_ci_yields_no_band(self):
        """A NaN percentile bound yields no band, mirroring the forest.

        When ``{m}_ci_lower``/``{m}_ci_upper`` columns are present the bar path
        uses them as-is (column presence, not value finiteness), so a NaN bound
        means no band for that model -- the same as the wide-forest renderer --
        rather than falling back to a ``point +/- z*std`` band the forest never
        draws.
        """
        from mlsurv.viz._metrics_helpers import _wide_to_long_summary

        df = pd.DataFrame({
            "model_type": ["good", "bad"],
            "harrell": [0.70, 0.60],
            "harrell_std": [0.04, 0.05],
            "harrell_ci_lower": [0.65, np.nan],
            "harrell_ci_upper": [0.74, np.nan],
        })
        long = _wide_to_long_summary(df, ["harrell"], "model_type")
        good = long[long["model_type"] == "good"].iloc[0]
        bad = long[long["model_type"] == "bad"].iloc[0]
        # Finite percentile CI is used verbatim.
        assert good["ci_lower"] == 0.65 and good["ci_upper"] == 0.74
        # NaN percentile CI -> no band, NOT 0.60 +/- z*0.05.
        assert np.isnan(bad["ci_lower"]) and np.isnan(bad["ci_upper"])
