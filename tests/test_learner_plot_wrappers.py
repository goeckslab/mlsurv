"""Tests for learner plot wrapper methods."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv._warnings import InternalWarning
from mlsurv.learner import SurvivalLearner

pytestmark = pytest.mark.integration


# ── Module-level synthetic data generator ──────────────────────────────────


def _make_synthetic_data():
    """Generate synthetic survival data (no NaN) for module-level fixtures."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 200, 20
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    time = rng.exponential(2, n_samples)
    event = rng.binomial(1, 0.7, n_samples).astype(bool)
    y = np.array(
        list(zip(event, time)),
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


# ── Shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _synth_data():
    return _make_synthetic_data()


@pytest.fixture(scope="module")
def trained_learner(_synth_data):
    """Learner with coxph + rsf trained and evaluated (untuned, no bootstrap).

    Tuning and bootstrap are deliberately off: the plot tests here only need
    trained models plus the explicit ``evaluate()`` below. Tests that need a
    tuned model use ``tuned_learner`` instead.
    """
    X, y = _synth_data
    learner = SurvivalLearner(X, y)
    learner.setup()
    learner.run(models=["coxph", "rsf"], tune=False, evaluate=False,
                bootstrap=False, verbose=False)
    learner.evaluate(n_importance_repeats=1)
    return learner


@pytest.fixture(scope="module")
def cox_pair_learner(_synth_data):
    """Learner with both coxph and coxnet for comparison tests."""
    X, y = _synth_data
    learner = SurvivalLearner(X, y)
    learner.setup()
    learner.run(models=["coxph", "coxnet"], tune=False, evaluate=False,
                bootstrap=False, verbose=False)
    learner.evaluate(n_importance_repeats=1)
    return learner


@pytest.fixture(scope="module")
def rsf_only_learner(_synth_data):
    """Learner with only RSF (no Cox models)."""
    X, y = _synth_data
    learner = SurvivalLearner(X, y)
    learner.setup()
    learner.run(models=["rsf"], tune=False, evaluate=False, bootstrap=False,
                verbose=False)
    learner.evaluate(n_importance_repeats=1)
    return learner


@pytest.fixture(scope="module")
def tuned_learner(_synth_data):
    """Learner with one cheaply tuned model (coxnet) for plot_optuna tests.

    coxph cannot serve here: its search space is empty, so tune() skips it
    without attaching an optuna_study.
    """
    X, y = _synth_data
    learner = SurvivalLearner(X, y)
    learner.setup()
    learner.tune(models="coxnet", n_trials=3, evaluate=False,
                 verbose=False, show_progress_bar=False)
    return learner


# ── find_cox_models helper ────────────────────────────────────────────────

from mlsurv.viz._data import find_cox_models


class TestFindCoxModels:
    def test_finds_coxph(self, trained_learner):
        result = find_cox_models(trained_learner._state)
        assert "coxph" in result
        assert hasattr(result["coxph"], "get_coefficients")

    def test_excludes_non_cox(self, trained_learner):
        result = find_cox_models(trained_learner._state)
        assert "rsf" not in result

    def test_empty_for_rsf_only(self, rsf_only_learner):
        result = find_cox_models(rsf_only_learner._state)
        assert len(result) == 0

    def test_filter_by_name(self, cox_pair_learner):
        result = find_cox_models(cox_pair_learner._state, models="coxph")
        assert list(result.keys()) == ["coxph"]

    def test_no_models_raises(self):
        X, y = _make_synthetic_data()
        learner = SurvivalLearner(X, y)
        with pytest.raises(ValueError, match="No trained models"):
            find_cox_models(learner._state)


# ── plot_importance style="heatmap" ────────────────────────────────


class TestPlotImportanceHeatmapStyle:
    """Tests for plot_importance(style='heatmap') — the consolidated API."""

    # test_returns_figure dropped: the figure-build path is covered by
    # test_default_style_unchanged / test_heatmap_kwargs_forwarded /
    # test_single_model_title_injection_models_none below.
    def test_return_data(self, trained_learner):
        df = trained_learner.plot_importance(style="heatmap", return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "coefficient" not in df.columns

    def test_no_eval_raises(self):
        X, y = _make_synthetic_data()
        learner = SurvivalLearner(X, y)
        learner.setup()
        learner.train(models="coxph")
        with pytest.raises(ValueError, match="evaluation|evaluate"):
            learner.plot_importance(style="heatmap")

    def test_default_style_unchanged(self, trained_learner):
        """Default style still returns a bar/raincloud figure (not heatmap)."""
        fig = trained_learner.plot_importance()
        assert isinstance(fig, go.Figure)

    def test_invalid_style_raises(self, trained_learner):
        with pytest.raises(ValueError, match="style.*not supported"):
            trained_learner.plot_importance(style="invalid")

    def test_heatmap_kwargs_forwarded(self, trained_learner):
        """Heatmap-specific kwargs reach the viz function."""
        fig = trained_learner.plot_importance(
            style="heatmap", n=5, annotations=False, log_scale=False,
        )
        assert isinstance(fig, go.Figure)

    def test_single_model_title_injection_models_none(self, rsf_only_learner):
        """models=None with 1 trained model injects model name into title."""
        fig = rsf_only_learner.plot_importance(n=5)
        assert fig.layout.title and "RSF" in fig.layout.title.text


# ── plot_coefficients ──────────────────────────────────────────────


class TestPlotCoefficients:
    def test_returns_figure(self, trained_learner):
        fig = trained_learner.plot_coefficients()
        assert isinstance(fig, go.Figure)

    def test_return_data(self, trained_learner):
        df = trained_learner.plot_coefficients(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "coefficient" in df.columns

    def test_no_cox_raises(self, rsf_only_learner):
        with pytest.raises(ValueError, match="No Cox-family"):
            rsf_only_learner.plot_coefficients()


# ── plot_coefficient_comparison ────────────────────────────────────


class TestPlotCoefficientComparison:
    # test_returns_figure dropped: figure-build path covered by
    # test_n_features_limits_annotations below.
    def test_return_data(self, cox_pair_learner):
        df = cox_pair_learner.plot_coefficient_comparison(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns

    def test_single_cox_raises(self, trained_learner):
        with pytest.raises(ValueError, match="at least 2"):
            trained_learner.plot_coefficient_comparison()

    def test_n_features_limits_annotations(self, cox_pair_learner):
        """Wrapper should forward n_features as max_annotations."""
        fig = cox_pair_learner.plot_coefficient_comparison(n_features=5)
        label_annotations = [
            a for a in fig.layout.annotations if a.showarrow
        ]
        assert len(label_annotations) <= 5


# ── plot_hazard_ratios ─────────────────────────────────────────────


class TestPlotHazardRatios:
    def test_univariate_returns_figure(self, trained_learner):
        fig = trained_learner.plot_hazard_ratios(kind="univariate")
        assert isinstance(fig, go.Figure)

    def test_univariate_return_data(self, trained_learner):
        df = trained_learner.plot_hazard_ratios(
            kind="univariate", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert "hr" in df.columns

    def test_multivariate_returns_figure(self, trained_learner):
        fig = trained_learner.plot_hazard_ratios(kind="multivariate")
        assert isinstance(fig, go.Figure)

    def test_multivariate_return_data(self, trained_learner):
        df = trained_learner.plot_hazard_ratios(
            kind="multivariate", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert "hr" in df.columns
        assert "ci_lower" in df.columns

    def test_invalid_kind_raises(self, trained_learner):
        with pytest.raises(ValueError, match="Unknown kind"):
            trained_learner.plot_hazard_ratios(kind="invalid")

    def test_no_cox_multivariate_raises(self, rsf_only_learner):
        with pytest.raises(ValueError, match="valid hazard ratios"):
            rsf_only_learner.plot_hazard_ratios(kind="multivariate")

    def test_univariate_no_cox_results_raises(self, trained_learner):
        orig = trained_learner._state.univariate_cox
        trained_learner._state.univariate_cox = None
        try:
            with pytest.raises(ValueError, match="univariate Cox"):
                trained_learner.plot_hazard_ratios(kind="univariate")
        finally:
            trained_learner._state.univariate_cox = orig


# ── plot_hazard_ratios_comparison ──────────────────────────────────


class TestPlotHazardRatiosComparison:
    def test_returns_figure(self, trained_learner):
        fig = trained_learner.plot_hazard_ratios_comparison()
        assert isinstance(fig, go.Figure)

    def test_return_data(self, trained_learner):
        df = trained_learner.plot_hazard_ratios_comparison(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert "hr" in df.columns
        assert "model" in df.columns
        # Should contain both univariate and multivariate
        assert set(df["model"].unique()) == {"Univariate", "Multivariate"}

    def test_no_cox_raises(self, rsf_only_learner):
        with pytest.raises(ValueError, match="valid hazard ratios"):
            rsf_only_learner.plot_hazard_ratios_comparison()


# ── plot_feature_correlation ───────────────────────────────


class TestPlotFeatureCorrelation:
    # test_returns_figure dropped: figure-build path covered by
    # test_custom_threshold / test_read_only_array_no_crash below.
    def test_return_data(self, trained_learner):
        df = trained_learner.plot_feature_correlation(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        # n rows × (n+1) cols: feature column + n numeric columns
        assert df.shape[0] == df.shape[1] - 1

    def test_custom_threshold(self, trained_learner):
        fig = trained_learner.plot_feature_correlation(threshold=0.5)
        assert isinstance(fig, go.Figure)

    def test_read_only_array_no_crash(self):
        """plot_feature_correlation should not crash on DataFrames with read-only arrays."""
        from mlsurv.viz.features import plot_feature_correlation

        arr = np.random.default_rng(42).standard_normal((50, 5))
        arr.flags.writeable = False
        df = pd.DataFrame(arr, columns=[f"f{i}" for i in range(5)])
        fig = plot_feature_correlation(df)
        assert isinstance(fig, go.Figure)


# ── plot_shap ─────────────────────────────────────────────────────


class TestPlotShap:
    @pytest.fixture()
    def shap_learner(self, trained_learner):
        """Trained learner with synthetic SHAP values injected."""
        # Build synthetic SHAP DataFrame matching expected schema
        features = list(trained_learner.X_test.columns)
        n_samples = len(trained_learner.X_test)
        rng = np.random.RandomState(99)
        rows = []
        for i in range(n_samples):
            for feat in features:
                rows.append({
                    'model_type': 'coxph',
                    'feature': feat,
                    'shap_value': rng.randn(),
                    'sample_idx': i,
                })
        shap_df = pd.DataFrame(rows)

        # Inject into per-model evaluation result so it survives
        # EvaluationResult.combine() in _resolve_eval_result()
        model_eval = trained_learner._state.models['coxph'].evaluation
        orig_shap = model_eval.shap_values
        model_eval.shap_values = shap_df
        try:
            yield trained_learner
        finally:
            model_eval.shap_values = orig_shap

    def test_importance_returns_figure(self, shap_learner):
        fig = shap_learner.plot_shap(kind="importance")
        assert isinstance(fig, go.Figure)

    def test_beeswarm_returns_figure(self, shap_learner):
        fig = shap_learner.plot_shap(kind="beeswarm")
        assert isinstance(fig, go.Figure)

    def test_return_data(self, shap_learner):
        df = shap_learner.plot_shap(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "shap_value" in df.columns
        assert len(df) > 0

    def test_no_shap_raises(self, trained_learner):
        with pytest.raises(ValueError, match="SHAP"):
            trained_learner.plot_shap()

    def test_invalid_kind_raises(self, shap_learner):
        with pytest.raises(ValueError, match="Unknown kind"):
            shap_learner.plot_shap(kind="invalid")


# ── plot_calibration_overlay ──────────────────────────────────────


class TestPlotCalibrationOverlay:
    def test_returns_figure(self, trained_learner):
        fig = trained_learner.plot_calibration_overlay()
        assert isinstance(fig, go.Figure)

    def test_return_data(self, trained_learner):
        data = trained_learner.plot_calibration_overlay(return_data=True)
        assert isinstance(data, pd.DataFrame)
        for col in ("model", "eval_time", "predicted", "observed"):
            assert col in data.columns, f"Missing column: {col}"
        assert len(data) > 0
        assert all(np.isfinite(data["predicted"]))
        assert all(np.isfinite(data["observed"]))

    def test_no_compatible_model_raises(self, trained_learner):
        """Test error when no model supports predict_survival_function."""
        # Temporarily swap models to only have a fake model
        orig_models = trained_learner._state.models

        class _FakeModel:
            pass  # no predict_survival_function

        class _FakeResult:
            # Faithful ModelResult stand-in: real results always expose
            # .evaluation/.bootstrap (read by the evaluation_result /
            # bootstrap_result properties that PlotContext.from_main calls),
            # and a .model whose lack of predict_survival_function is what
            # the calibration overlay rejects.
            model = _FakeModel()
            evaluation = None
            bootstrap = None

        trained_learner._state.models = {"fake": _FakeResult()}
        try:
            with pytest.raises(ValueError, match="predict_survival_function"):
                trained_learner.plot_calibration_overlay()
        finally:
            trained_learner._state.models = orig_models


# ── plot_metric title kwarg ────────────────────────────────────────


class TestPlotMetricTitle:
    def test_explicit_title_overrides_auto(self, trained_learner):
        """Caller-provided title should override auto-generated title."""
        fig = trained_learner.plot_metric(title="Custom Title")
        assert "Custom Title" in fig.layout.title.text

    def test_title_does_not_raise_typeerror(self, trained_learner):
        """Regression: title= kwarg should not cause duplicate keyword error."""
        # Before fix, this raised TypeError: got multiple values for 'title'
        fig = trained_learner.plot_metric(title="My Metric")
        assert isinstance(fig, go.Figure)


# ── plot_coefficient_comparison max_annotations ────────────────────


def _get_identity_traces(fig):
    """Return the gray dotted identity-line traces from a coefficient comparison figure."""
    return [
        t for t in fig.data
        if isinstance(t, go.Scatter)
        and t.mode == "lines"
        and t.line.color == "gray"
        and t.line.dash == "dot"
    ]


class TestCoefficientComparisonAnnotations:
    def test_standalone_max_annotations_limits_labels(self):
        """max_annotations=10 should cap annotation count on 50-row df."""
        from mlsurv.viz.features import plot_coefficient_comparison

        rng = np.random.RandomState(99)
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(50)],
            'coxph_coef': rng.randn(50),
            'coxnet_coef': rng.randn(50),
        })
        fig = plot_coefficient_comparison(df, max_annotations=10)
        # Annotations include feature labels + axis cross-hairs (none here)
        label_annotations = [
            a for a in fig.layout.annotations if a.showarrow
        ]
        assert len(label_annotations) == 10

    def test_standalone_max_annotations_none_annotates_all(self):
        """max_annotations=None should annotate every feature."""
        from mlsurv.viz.features import plot_coefficient_comparison

        rng = np.random.RandomState(99)
        n = 15
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': rng.randn(n),
            'coxnet_coef': rng.randn(n),
        })
        fig = plot_coefficient_comparison(df, max_annotations=None)
        label_annotations = [
            a for a in fig.layout.annotations if a.showarrow
        ]
        assert len(label_annotations) == n

    def test_standalone_small_df_annotates_all(self):
        """When df has fewer rows than max_annotations, all are annotated."""
        from mlsurv.viz.features import plot_coefficient_comparison

        rng = np.random.RandomState(99)
        n = 5
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': rng.randn(n),
            'coxnet_coef': rng.randn(n),
        })
        fig = plot_coefficient_comparison(df, max_annotations=20)
        label_annotations = [
            a for a in fig.layout.annotations if a.showarrow
        ]
        assert len(label_annotations) == n

    def test_all_nan_column_identity_line_present(self):
        """Identity line should still appear when one column is all NaN."""
        from mlsurv.viz.features import plot_coefficient_comparison

        rng = np.random.RandomState(42)
        n = 10
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': rng.randn(n),
            'coxnet_coef': [np.nan] * n,
        })
        with pytest.warns(InternalWarning, match="NaN coefficients"):
            fig = plot_coefficient_comparison(df)
        assert len(_get_identity_traces(fig)) == 1

    def test_both_columns_all_nan_identity_line(self):
        """Identity line with fallback range when both columns are all NaN."""
        from mlsurv.viz.features import plot_coefficient_comparison

        n = 5
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': [np.nan] * n,
            'coxnet_coef': [np.nan] * n,
        })
        with pytest.warns(InternalWarning, match="NaN coefficients"):
            fig = plot_coefficient_comparison(df)
        identity_traces = _get_identity_traces(fig)
        assert len(identity_traces) == 1
        # Fallback range: max_val=1.0 * 1.1 = 1.1
        assert identity_traces[0].x[0] == pytest.approx(-1.1)

    def test_all_zero_coefficients_identity_line(self):
        """Identity line with fallback range when all coefficients are zero."""
        from mlsurv.viz.features import plot_coefficient_comparison

        n = 5
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': [0.0] * n,
            'coxnet_coef': [0.0] * n,
        })
        fig = plot_coefficient_comparison(df)
        identity_traces = _get_identity_traces(fig)
        assert len(identity_traces) == 1
        assert identity_traces[0].x[0] == pytest.approx(-1.1)

    def test_nan_coefficients_warns(self):
        """NaN coefficients should trigger a warning when using max_annotations."""
        from mlsurv.viz.features import plot_coefficient_comparison

        rng = np.random.RandomState(42)
        n = 20
        coefs = rng.randn(n)
        coefs[:3] = np.nan  # 3 NaN values
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': rng.randn(n),
            'coxnet_coef': coefs,
        })
        with pytest.warns(InternalWarning, match="3 features have NaN coefficients"):
            fig = plot_coefficient_comparison(df, max_annotations=5)
        # nlargest skips NaN distances, so all 5 annotations are non-NaN
        label_annotations = [
            a for a in fig.layout.annotations if a.showarrow
        ]
        assert len(label_annotations) == 5

    def test_nan_warns_without_max_annotations_pruning(self):
        """NaN warning fires even when len(df) <= max_annotations."""
        from mlsurv.viz.features import plot_coefficient_comparison

        rng = np.random.RandomState(42)
        n = 10
        coefs = rng.randn(n)
        coefs[:2] = np.nan
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': rng.randn(n),
            'coxnet_coef': coefs,
        })
        with pytest.warns(InternalWarning, match="2 features have NaN coefficients"):
            fig = plot_coefficient_comparison(df, max_annotations=20)
        # NaN features should NOT be annotated
        label_annotations = [
            a for a in fig.layout.annotations if a.showarrow
        ]
        assert len(label_annotations) == n - 2

    def test_nan_coefficients_no_warn_when_no_nan(self):
        """No warning should be raised when there are no NaN coefficients."""
        from mlsurv.viz.features import plot_coefficient_comparison

        rng = np.random.RandomState(42)
        n = 20
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': rng.randn(n),
            'coxnet_coef': rng.randn(n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot_coefficient_comparison(df, max_annotations=5)

    def test_nan_with_max_annotations_none(self):
        """NaN warning fires and NaN features excluded when max_annotations=None."""
        from mlsurv.viz.features import plot_coefficient_comparison

        rng = np.random.RandomState(42)
        n = 10
        coefs = rng.randn(n)
        coefs[:2] = np.nan
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': rng.randn(n),
            'coxnet_coef': coefs,
        })
        with pytest.warns(InternalWarning, match="2 features have NaN coefficients"):
            fig = plot_coefficient_comparison(df, max_annotations=None)
        label_annotations = [
            a for a in fig.layout.annotations if a.showarrow
        ]
        assert len(label_annotations) == n - 2

    def test_nan_in_different_columns(self):
        """NaN scattered across both columns counts correctly."""
        from mlsurv.viz.features import plot_coefficient_comparison

        rng = np.random.RandomState(42)
        n = 10
        x_coefs = rng.randn(n)
        y_coefs = rng.randn(n)
        x_coefs[0] = np.nan   # feat_0: NaN in x
        y_coefs[1] = np.nan   # feat_1: NaN in y
        x_coefs[2] = np.nan   # feat_2: NaN in both
        y_coefs[2] = np.nan
        df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in range(n)],
            'coxph_coef': x_coefs,
            'coxnet_coef': y_coefs,
        })
        # 3 features have NaN in at least one column
        with pytest.warns(InternalWarning, match="3 features have NaN"):
            fig = plot_coefficient_comparison(df, max_annotations=None)
        label_annotations = [
            a for a in fig.layout.annotations if a.showarrow
        ]
        assert len(label_annotations) == n - 3


# ── Fixtures for new plot_* methods ────────────────────────────────


@pytest.fixture(scope="module")
def setup_only_learner(_synth_data):
    """Learner with setup() only — no models trained."""
    X, y = _synth_data
    learner = SurvivalLearner(X, y)
    learner.setup()
    return learner


# ── TestPlotFeatureDistributions ──────────────────────────────────────────


class TestPlotFeatureDistributions:
    """Tests for learner.plot_feature_distributions."""

    def test_returns_figure(self, setup_only_learner):
        fig = setup_only_learner.plot_feature_distributions()
        assert isinstance(fig, go.Figure)

    def test_return_data(self, setup_only_learner):
        df = setup_only_learner.plot_feature_distributions(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["feature", "split", "value"]
        assert set(df["split"].unique()) == {"train", "test"}

    def test_features_filter(self, setup_only_learner):
        fig = setup_only_learner.plot_feature_distributions(
            features=["feature_0", "feature_1"],
        )
        assert isinstance(fig, go.Figure)

    def test_features_filter_return_data(self, setup_only_learner):
        df = setup_only_learner.plot_feature_distributions(
            features=["feature_0", "feature_1"], return_data=True,
        )
        assert set(df["feature"].unique()) == {"feature_0", "feature_1"}

    def test_invalid_feature_raises(self, setup_only_learner):
        with pytest.raises(ValueError, match="not found"):
            setup_only_learner.plot_feature_distributions(
                features=["nonexistent"],
            )


# ── TestPlotMetricHeatmap ─────────────────────────────────────────────────


class TestPlotMetricHeatmap:
    """Tests for learner.plot_metric_heatmap."""

    def test_returns_figure(self, trained_learner):
        fig = trained_learner.plot_metric_heatmap()
        assert isinstance(fig, go.Figure)

    def test_return_data(self, trained_learner):
        df = trained_learner.plot_metric_heatmap(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert "model" in df.columns
        assert len(df) >= 2

    def test_single_model_raises(self, rsf_only_learner):
        with pytest.raises(ValueError, match="at least 2"):
            rsf_only_learner.plot_metric_heatmap()

    def test_models_filter(self, trained_learner):
        # Filtering to one model should raise (< 2)
        with pytest.raises(ValueError, match="at least 2"):
            trained_learner.plot_metric_heatmap(models="coxph")


# ── TestPlotOptuna ────────────────────────────────────────────────────────


class TestPlotOptuna:
    """Tests for learner.plot_optuna."""

    def test_no_tuned_models_raises(self, trained_learner):
        # trained_learner runs with tune=False, so no optuna studies exist.
        with pytest.raises(ValueError, match="tun"):
            trained_learner.plot_optuna()

    def test_returns_figure(self, tuned_learner):
        fig = tuned_learner.plot_optuna()
        assert isinstance(fig, go.Figure)

    def test_return_data(self, tuned_learner):
        data = tuned_learner.plot_optuna(return_data=True)
        assert isinstance(data, dict)
        assert "tuning_history" in data
        assert "param_importance" in data
        assert isinstance(data["tuning_history"], pd.DataFrame)
        if not data["tuning_history"].empty:
            assert "model" in data["tuning_history"].columns


# TestPlotSubpopulationSurvival was removed (the
# plot_subpopulation_survival learner method no longer exists).
# See tests/test_learner_views.py::TestSubpopulationViewPlotSurvival
# for the replacement coverage via SubpopulationView.
