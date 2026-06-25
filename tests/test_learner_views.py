"""Unit tests for :mod:`mlsurv.learner._views` view-proxy classes.

Covers:

- :class:`ValidationView` construction contract
- All five plot methods (``plot_metric_time``, ``plot_metric``,
  ``plot_survival``, ``plot_calibration``, ``plot_roc``) returning
  figures and supporting ``return_data=True``
- ``SurvivalLearner.validation()`` accessor auto-pick / error paths
- NotImplementedError guards for non-validation sources and
  unsupported parameters
- IPCW invariant: ``ctx.y_train`` is always the main training outcome,
  even on validation cohorts

``plot_metric_time`` shipped first as the minimal viable cut; the
other four plot methods were added later.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import SuggestionWarning
from mlsurv.learner._views import SubpopulationView, ValidationView


def _make_small_data_views():
    """Inline small_data for module-scoped fixtures (50 samples)."""
    rng = np.random.RandomState(42)
    n = 50
    X = pd.DataFrame({
        'age': rng.uniform(30, 80, n),
        'stage': rng.randint(1, 5, n),
        'grade': rng.uniform(1, 4, n),
        'tumor_size': rng.uniform(0.5, 10, n),
        'biomarker': rng.randn(n),
    })
    from mlsurv.utils import SurvivalArray
    times = rng.exponential(2, n).clip(0.1, 10)
    events = rng.binomial(1, 0.6, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
    ))
    return X, y


@pytest.fixture(scope="module")
def _validated_learner_template():
    """Module-scoped template — train once, deepcopy per test."""
    X, y = _make_small_data_views()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.run(["coxph"], tune=False)
    learner.bootstrap(n_bootstrap=2)
    half = len(X) // 2
    learner.validate(X.iloc[:half], y[:half], name="Holdout")
    return learner


@pytest.fixture
def validated_learner(_validated_learner_template):
    """Per-test deep copy — safe for tests that mutate state."""
    import copy
    return copy.deepcopy(_validated_learner_template)


@pytest.fixture(scope="module")
def _multi_model_template():
    """Module-scoped template for multi-model learner."""
    X, y = _make_small_data_views()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.run(["coxph", "rsf"], tune=False)
    learner.bootstrap(n_bootstrap=2)
    half = len(X) // 2
    learner.validate(X.iloc[:half], y[:half], name="Holdout")
    return learner


@pytest.fixture
def multi_model_validated_learner(_multi_model_template):
    """Per-test deep copy of multi-model learner."""
    import copy
    return copy.deepcopy(_multi_model_template)


@pytest.fixture
def multi_cohort_learner(_validated_learner_template):
    """Learner with two cohorts — fresh copy with second cohort added."""
    import copy
    X, y = _make_small_data_views()
    learner = copy.deepcopy(_validated_learner_template)
    learner.validate(X.iloc[-20:], y[-20:], name="Secondary")
    return learner


# ------------------------------------------------------------------
# Construction contract
# ------------------------------------------------------------------


class TestValidationViewConstruction:
    def test_stores_cohort(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        assert view.cohort == "Holdout"

    def test_result_is_validation_result(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        assert view.result is validated_learner._state.validation_results["Holdout"]

    def test_repr(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        assert repr(view) == "ValidationView(cohort='Holdout')"

    def test_unknown_cohort_raises_keyerror(self, validated_learner):
        with pytest.raises(KeyError, match="not found"):
            ValidationView(validated_learner, "DoesNotExist")

    def test_keyerror_lists_available_cohorts(self, validated_learner):
        with pytest.raises(KeyError, match="Holdout"):
            ValidationView(validated_learner, "DoesNotExist")


# ------------------------------------------------------------------
# plot_metric_time
# ------------------------------------------------------------------


class TestValidationViewPlotMetricTime:
    def test_returns_figure(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_metric_time()
        assert isinstance(fig, go.Figure)

    def test_return_data_returns_dataframe(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        df = view.plot_metric_time(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Should be the tidy auc_t frame
        assert "eval_time" in df.columns
        assert "auc" in df.columns

    def test_filter_to_specific_model(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        df = view.plot_metric_time(models=["coxph"], return_data=True)
        # tidy_model_col renames model_type → model but keeps raw names
        assert set(df["model"].unique()) == {"coxph"}

    # Canonical models= resolver-validation tests for ValidationView. All plot
    # methods route models= through the same _resolve_models_arg() helper, so
    # the str-acceptance / type-error / unknown-model checks live here once
    # rather than being repeated per plot method. Each other method
    # keeps its own positive test_filter_to_specific_model.
    def test_filter_with_str_model(self, validated_learner):
        """A bare string models= argument should be accepted."""
        view = ValidationView(validated_learner, "Holdout")
        df = view.plot_metric_time(models="coxph", return_data=True)
        assert set(df["model"].unique()) == {"coxph"}

    def test_invalid_models_type_raises_typeerror(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(TypeError, match="models must be"):
            view.plot_metric_time(models=42)

    def test_unknown_model_raises(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(ValueError, match="Unknown models"):
            view.plot_metric_time(models=["nonexistent_model"])

    def test_empty_auc_t_raises_valueerror(self, validated_learner):
        """Cohort with no AUC(t) data must raise a clear ValueError."""
        val_result = validated_learner._state.validation_results["Holdout"]
        val_result.evaluation.auc_t = pd.DataFrame(
            columns=["model_type", "eval_time", "auc"]
        )
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(ValueError, match=r"has no AUC\(t\) data"):
            view.plot_metric_time()

    def test_plot_without_bootstrap(self, small_data):
        """Cohorts without bootstrap should still plot (no CI ribbons)."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None)
        learner.run(["coxph"])
        # NO bootstrap() call — compute_auc_std_df should return None
        learner.validate(X.iloc[:25], y[:25], name="Holdout")
        fig = learner.validation("Holdout").plot_metric_time()
        assert isinstance(fig, go.Figure)

    def test_multi_metric_returns_figure(self, validated_learner):
        """``metric=['auc', 'brier']`` should produce a multi-panel Figure."""
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_metric_time(metric=["auc", "brier"])
        assert isinstance(fig, go.Figure)

    def test_multi_metric_return_data(self, validated_learner):
        """``return_data=True`` should yield a tidy long-format frame covering
        both requested metrics."""
        view = ValidationView(validated_learner, "Holdout")
        df = view.plot_metric_time(
            metric=["auc", "brier"], return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert "metric_name" in df.columns
        assert set(df["metric_name"].unique()) >= {"auc", "brier"}

    def test_multi_metric_rejects_unknown(self, validated_learner):
        """Only ``auc`` / ``brier`` are supported on views; other keys raise
        ``ValueError`` rather than silently dropping."""
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(ValueError, match="Unknown metric"):
            view.plot_metric_time(metric=["made_up"])

    def test_single_string_metric_is_single_panel(self, validated_learner):
        """A string metric is now valid and draws a single-panel plot."""
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_metric_time(metric="auc")
        assert isinstance(fig, go.Figure)

    def test_multi_metric_rejects_non_string_entry(self, validated_learner):
        """A non-string element in the metric list raises TypeError."""
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(TypeError, match="must be strings"):
            view.plot_metric_time(metric=["auc", 1])

    def test_multi_metric_rejects_empty(self, validated_learner):
        """Empty metric list is a user error, not a silent no-op."""
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(ValueError, match="empty"):
            view.plot_metric_time(metric=[])

    def test_references_not_supported_yet(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(NotImplementedError, match="references="):
            view.plot_metric_time(references="all")


# ------------------------------------------------------------------
# SurvivalLearner.validation() accessor
# ------------------------------------------------------------------


class TestValidationAccessor:
    def test_explicit_cohort(self, validated_learner):
        view = validated_learner.validation("Holdout")
        assert isinstance(view, ValidationView)
        assert view.cohort == "Holdout"

    def test_autoselect_single_cohort(self, validated_learner):
        view = validated_learner.validation()
        assert view.cohort == "Holdout"

    def test_autoselect_multi_cohort_raises(self, multi_cohort_learner):
        with pytest.raises(ValueError, match="Multiple validation cohorts"):
            multi_cohort_learner.validation()

    def test_no_cohorts_raises(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None)
        learner.train("coxph")
        with pytest.raises(ValueError, match="No validation cohorts"):
            learner.validation()

    def test_unknown_cohort_raises_keyerror(self, validated_learner):
        with pytest.raises(KeyError, match="not found"):
            validated_learner.validation("Nope")


# ======================================================================
# ValidationView ports: plot_metric / plot_survival /
# plot_calibration / plot_roc
# ======================================================================


# ------------------------------------------------------------------
# plot_metric
# ------------------------------------------------------------------


class TestValidationViewPlotMetric:
    def test_returns_figure(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_metric()
        assert isinstance(fig, go.Figure)

    def test_return_data_returns_dataframe(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        df = view.plot_metric(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "model" in df.columns

    def test_filter_to_specific_model(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        df = view.plot_metric(models=["coxph"], return_data=True)
        assert set(df["model"].unique()) == {"coxph"}

    # models= str/type/unknown validation is covered once in
    # TestValidationViewPlotMetricTime (shared resolver).

    def test_cv_source_raises_notimplementederror(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(NotImplementedError, match="source='cv'"):
            view.plot_metric(source="cv")

    def test_multi_metric_raises_notimplementederror(self, validated_learner):
        """A 2+ element metric list (multi-panel) is not supported on views."""
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(NotImplementedError, match="multi-metric"):
            view.plot_metric(metric=["harrell", "ipcw"])

    def test_single_element_metric_list_succeeds(self, validated_learner):
        """A 1-element metric list collapses to a single-panel plot (no raise)."""
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_metric(metric=["harrell"])
        assert isinstance(fig, go.Figure)

    def test_references_raises_notimplementederror(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(NotImplementedError, match="references="):
            view.plot_metric(references="all")

    def test_auto_source_falls_back_to_test_when_no_bootstrap(self, small_data):
        """Cohort validated without bootstrap auto-resolves to source='test'."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None)
        learner.run(["coxph"])
        # Validate WITHOUT bootstrap
        learner.validate(X.iloc[:25], y[:25], name="Holdout", bootstrap=False)
        view = learner.validation("Holdout")
        # auto → test (bootstrap is None on this cohort)
        fig = view.plot_metric()
        assert isinstance(fig, go.Figure)

    def test_explicit_test_source(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_metric(source="test")
        assert isinstance(fig, go.Figure)

    def test_default_title_uses_metric_display_name(self, validated_learner):
        """Default title shows the metric's display name, not its raw key."""
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_metric(metric="harrell")
        title = fig.layout.title.text
        assert "Harrell's C-index" in title
        assert "harrell on" not in title
        assert "Holdout" in title

    def test_user_supplied_title_is_respected(self, validated_learner):
        """An explicit title= overrides the default cohort title."""
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_metric(metric="harrell", title="Custom title")
        # The renderer may wrap the title in <b>...</b>; substring is enough.
        assert "Custom title" in fig.layout.title.text
        assert "Harrell" not in fig.layout.title.text

    # test_main_source_raises_notimplementederror removed —
    # _plot_metric_impl now supports source="main".


# ------------------------------------------------------------------
# plot_survival
# ------------------------------------------------------------------


class TestValidationViewPlotSurvival:
    def test_returns_figure_single_model(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_survival()
        assert isinstance(fig, go.Figure)

    def test_returns_figure_multi_model(self, multi_model_validated_learner):
        view = ValidationView(multi_model_validated_learner, "Holdout")
        fig = view.plot_survival()
        assert isinstance(fig, go.Figure)

    def test_return_data_returns_dataframe(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        df = view.plot_survival(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # survival_test_data_to_tidy produces time/event/risk_score
        for col in ("time", "event", "risk_score"):
            assert col in df.columns, f"missing column {col!r}"

    def test_filter_to_specific_model(self, multi_model_validated_learner):
        view = ValidationView(multi_model_validated_learner, "Holdout")
        df = view.plot_survival(models=["coxph"], return_data=True)
        assert set(df["model"].unique()) == {"coxph"}

    # models= str/type/unknown validation is covered once in
    # TestValidationViewPlotMetricTime (shared resolver).

    def test_references_raises_notimplementederror(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(NotImplementedError, match="references="):
            view.plot_survival(references="all")

    def test_no_test_set_raises_valueerror(self, validated_learner):
        val_result = validated_learner._state.validation_results["Holdout"]
        val_result.evaluation.test_set = None
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(ValueError, match="no test_set data"):
            view.plot_survival()


# ------------------------------------------------------------------
# plot_calibration
# ------------------------------------------------------------------


class TestValidationViewPlotCalibration:
    # small_data has max time ~9.2 — pick eval_times inside that range
    # to keep these tests deterministic regardless of the percentile-
    # based default chosen by compute_calibration.

    def test_returns_figure_single_model(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_calibration(eval_times=[2.0, 4.0], n_bins=3)
        assert isinstance(fig, go.Figure)

    def test_returns_figure_multi_model(self, multi_model_validated_learner):
        view = ValidationView(multi_model_validated_learner, "Holdout")
        fig = view.plot_calibration(eval_times=2.0, n_bins=3)
        assert isinstance(fig, go.Figure)

    def test_multi_model_default_eval_time(self, multi_model_validated_learner):
        """Multi-model mode with eval_time=None uses median event time."""
        view = ValidationView(multi_model_validated_learner, "Holdout")
        fig = view.plot_calibration(n_bins=2)
        assert isinstance(fig, go.Figure)

    def test_empty_calibration_returns_figure(self):
        """When no models produce valid calibration data, return an
        annotated empty figure with SuggestionWarning."""
        from mlsurv.viz._plot_impl import _empty_calibration_figure
        from mlsurv._warnings import SuggestionWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = _empty_calibration_figure(eval_time=2.0, return_data=False)
        assert isinstance(fig, go.Figure)
        suggestion_warnings = [
            w for w in caught if issubclass(w.category, SuggestionWarning)
        ]
        assert len(suggestion_warnings) == 1
        assert "No models produced valid calibration data" in str(
            suggestion_warnings[0].message,
        )

    def test_empty_calibration_returns_empty_dataframe(self):
        """return_data=True path returns empty DataFrame with correct
        column schema."""
        from mlsurv.viz._plot_impl import _empty_calibration_figure
        from mlsurv._warnings import SuggestionWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = _empty_calibration_figure(eval_time=2.0, return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert "model" in df.columns
        assert "eval_time" in df.columns
        suggestion_warnings = [
            w for w in caught if issubclass(w.category, SuggestionWarning)
        ]
        assert len(suggestion_warnings) == 1

    def test_return_data_returns_dataframe(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        df = view.plot_calibration(
            eval_times=[2.0, 4.0], n_bins=3, return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_filter_to_specific_model(self, multi_model_validated_learner):
        view = ValidationView(multi_model_validated_learner, "Holdout")
        fig = view.plot_calibration(
            models=["coxph"], eval_times=[2.0], n_bins=3,
        )
        assert isinstance(fig, go.Figure)

    # models= str/type/unknown validation is covered once in
    # TestValidationViewPlotMetricTime (shared resolver).

    def test_missing_xy_raises_valueerror(self, validated_learner):
        val_result = validated_learner._state.validation_results["Holdout"]
        val_result.X_val = None
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(ValueError, match="no stored X_val/y_val"):
            view.plot_calibration()

    def test_all_models_unsupported_raises(self, validated_learner):
        """If NO validated model supports predict_survival_function, raise.

        Stubs out the fitted coxph model with a minimal wrapper that
        lacks ``predict_survival_function``. This is the only portable
        way to hit the whitelist filter's dead-end branch — every
        built-in survival model in mlsurv implements the method.
        """
        class NoSurvFunc:
            def predict(self, X):
                return np.zeros(len(X))

        validated_learner._state.models["coxph"].model = NoSurvFunc()
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(
            ValueError, match="No validated models support predict_survival_function",
        ):
            view.plot_calibration()


# ------------------------------------------------------------------
# plot_roc
# ------------------------------------------------------------------


class TestValidationViewPlotRoc:
    def test_returns_figure_default(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_roc()
        assert isinstance(fig, go.Figure)

    def test_returns_figure_explicit_t(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        # Pick an eval time in the middle of the data range
        test_set = validated_learner._state.validation_results["Holdout"].test_set
        t = float(test_set["time"].median())
        fig = view.plot_roc(t=t)
        assert isinstance(fig, go.Figure)

    def test_return_data_returns_dataframe(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        df = view.plot_roc(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ("model", "fpr", "tpr", "auc", "eval_time"):
            assert col in df.columns, f"missing column {col!r}"

    def test_filter_to_specific_model(self, multi_model_validated_learner):
        view = ValidationView(multi_model_validated_learner, "Holdout")
        df = view.plot_roc(models=["coxph"], return_data=True)
        assert set(df["model"].unique()) == {"coxph"}

    # models= str/type/unknown validation is covered once in
    # TestValidationViewPlotMetricTime (shared resolver).

    def test_show_ci_false(self, validated_learner):
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_roc(show_ci=False)
        assert isinstance(fig, go.Figure)

    def test_show_ci_auto_without_bootstrap_is_silent(self, small_data):
        """Default show_ci="auto" on a no-bootstrap cohort: point curve, no warning."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None)
        learner.run(["coxph"])
        learner.validate(X.iloc[:25], y[:25], name="Holdout", bootstrap=False)
        view = learner.validation("Holdout")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = view.plot_roc()  # auto default
        assert isinstance(fig, go.Figure)
        roc_ci = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and "ROC CI bands" in str(w.message)
        ]
        assert not roc_ci, f"auto must be silent; got {[str(w.message) for w in roc_ci]}"

    def test_show_ci_true_without_bootstrap_warns(self, small_data):
        """Explicit show_ci=True on a no-bootstrap cohort: point curve + SuggestionWarning."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None)
        learner.run(["coxph"])
        learner.validate(X.iloc[:25], y[:25], name="Holdout", bootstrap=False)
        view = learner.validation("Holdout")
        with pytest.warns(SuggestionWarning, match="include_predictions=True"):
            fig = view.plot_roc(show_ci=True)
        assert isinstance(fig, go.Figure)

    def test_y_train_is_main_training_set(self, validated_learner):
        """IPCW invariant: ctx.y_train is the learner's training outcome.

        DO NOT "fix" this to y_val — the censoring distribution used
        for IPCW weighting must be the training distribution.
        """
        view = ValidationView(validated_learner, "Holdout")
        ctx = view._ctx()
        np.testing.assert_array_equal(
            ctx.y_train["time"], validated_learner._state.y_train["time"],
        )
        np.testing.assert_array_equal(
            ctx.y_train["event"], validated_learner._state.y_train["event"],
        )
        # Verify it is NOT the validation outcome
        val_result = validated_learner._state.validation_results["Holdout"]
        assert len(ctx.y_train) != len(val_result.y_val), (
            "y_train should be training set, not validation set"
        )

    def test_no_test_set_raises_valueerror(self, validated_learner):
        val_result = validated_learner._state.validation_results["Holdout"]
        val_result.evaluation.test_set = None
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(ValueError, match="no test_set data"):
            view.plot_roc()

    def test_show_ci_with_bootstrap_but_no_xval(self, validated_learner):
        """Bootstrap result present but X_val cleared — must not crash.

        Regression for a review finding: ``ctx.X_data.index`` was
        unconditionally dereferenced inside the show_ci branch. After
        the fix, the branch no-ops when ``X_val`` is missing and emits
        a ``SuggestionWarning`` so the user knows CI bands were
        omitted for an actionable reason (re-run validate with X_val).
        """
        # Build a real validation context, then swap in a fake
        # bootstrap_result with non-empty predictions so the
        # X_data-None branch (with bootstrap present) is exercised.
        class FakeBootstrap:
            predictions = pd.DataFrame({
                "model_type": ["coxph"],
                "iteration": [0],
                "index": [0],
                "risk_score": [0.5],
            })

        view = ValidationView(validated_learner, "Holdout")
        ctx = view._ctx()
        object.__setattr__(ctx, "bootstrap_result", FakeBootstrap())
        object.__setattr__(ctx, "X_data", None)

        from mlsurv.viz._plot_impl import _plot_roc_impl

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = _plot_roc_impl(ctx, show_ci=True, t=2.0)
        assert isinstance(fig, go.Figure)
        # SuggestionWarning must name the cohort and explain the fix.
        suggestion = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and "Holdout" in str(w.message)
            and "X_val" in str(w.message)
        ]
        assert suggestion, (
            f"expected SuggestionWarning naming cohort + X_val; got: "
            f"{[(w.category.__name__, str(w.message)) for w in caught]}"
        )

    def test_auc_t_fallback_to_median_event_time(self, validated_learner):
        """When auc_t is None, fall back to median event time."""
        val_result = validated_learner._state.validation_results["Holdout"]
        val_result.evaluation.auc_t = None
        view = ValidationView(validated_learner, "Holdout")
        fig = view.plot_roc()
        assert isinstance(fig, go.Figure)


# ------------------------------------------------------------------
# Cross-impl invariants and gap-fill tests (from review)
# ------------------------------------------------------------------


# TestPlotImplValidationGuard removed — all five impls now
# support source="main" so _validation_source_guard is dead code.


class TestCalibrationCrossCohortWhitelist:
    """Models without concordance rows on this cohort must be excluded.

    The whitelist filter in _plot_calibration_impl is the load-bearing
    safeguard against silent scope creep where calibration would be
    drawn for models that were never validated on this cohort.
    """

    def test_drops_models_not_in_concordance(self, multi_model_validated_learner):
        """Drop rsf from concordance — only coxph should appear in the figure.

        No models= filter here — we want the impl's own
        validated_names whitelist to restrict to coxph, even though
        rsf is in fitted_models. (If we passed ``models=["coxph", "rsf"]``
        explicitly, ``_resolve_models_arg`` would already raise
        ValueError for rsf, short-circuiting the whitelist branch.)
        """
        val_result = multi_model_validated_learner._state.validation_results["Holdout"]
        conc = val_result.evaluation.concordance
        val_result.evaluation.concordance = conc[conc["model_type"] == "coxph"]
        view = ValidationView(multi_model_validated_learner, "Holdout")
        fig = view.plot_calibration(eval_times=2.0, n_bins=3)
        assert isinstance(fig, go.Figure)
        # The single-model branch (len(model_dict)==1) should fire,
        # producing a figure with coxph traces only.

    def test_explicit_non_validated_model_raises(
        self, multi_model_validated_learner,
    ):
        """Explicit ``models=[...]`` with a non-validated model raises early.

        ``_resolve_models_arg`` uses ``_available_model_names(ctx)`` —
        which returns ``eval_result.concordance['model_type'].unique()``
        — so a requested-but-not-validated model is rejected up front
        with a clear ``ValueError``. This is the short-circuit that
        makes a ``dropped_not_validated`` warning inside
        ``_plot_calibration_impl`` unreachable (dead code).
        """
        val_result = multi_model_validated_learner._state.validation_results[
            "Holdout"
        ]
        conc = val_result.evaluation.concordance
        val_result.evaluation.concordance = conc[conc["model_type"] == "coxph"]
        view = ValidationView(multi_model_validated_learner, "Holdout")
        with pytest.raises(ValueError, match=r"Unknown models \['rsf'\]"):
            view.plot_calibration(
                models=["coxph", "rsf"],
                eval_times=[2.0],
                n_bins=3,
            )


class TestPlotMetricEmptyEvalResult:
    """Coverage gap from review: _plot_metric_impl ValueError paths."""

    def test_empty_concordance_raises_valueerror(self, validated_learner):
        val_result = validated_learner._state.validation_results["Holdout"]
        val_result.evaluation.concordance = None
        view = ValidationView(validated_learner, "Holdout")
        with pytest.raises(ValueError, match="no evaluation data"):
            view.plot_metric()


class TestCalibrationExplicitDropWarning:
    """Coverage gap from review: silent drop of explicitly-requested models.

    When a user passes ``models=["coxph", "stub"]`` and ``stub`` lacks
    ``predict_survival_function``, the impl must warn — silent removal
    leaves the user wondering why their requested model is missing.
    """

    def test_explicit_drop_emits_suggestion_warning(self, validated_learner):
        class NoSurvFunc:
            def predict(self, X):
                return np.zeros(len(X))

        # Add a stub model whose ModelResult.model lacks
        # predict_survival_function. We graft it onto the existing
        # state so it appears in fitted_models AND gets a concordance
        # row (we copy the coxph row and rename it).
        from copy import deepcopy

        coxph_result = validated_learner._state.models["coxph"]
        stub_result = deepcopy(coxph_result)
        stub_result.model = NoSurvFunc()
        stub_result.name = "stub"
        validated_learner._state.models["stub"] = stub_result
        # Add a concordance row for "stub" so the validated_names
        # whitelist includes it.
        val_result = validated_learner._state.validation_results["Holdout"]
        conc = val_result.evaluation.concordance
        stub_row = conc[conc["model_type"] == "coxph"].copy()
        stub_row["model_type"] = "stub"
        val_result.evaluation.concordance = pd.concat(
            [conc, stub_row], ignore_index=True,
        )

        view = ValidationView(validated_learner, "Holdout")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = view.plot_calibration(
                models=["coxph", "stub"],
                eval_times=[2.0],
                n_bins=3,
            )
        assert isinstance(fig, go.Figure)
        # The warning must mention the dropped model by name and
        # be a SuggestionWarning.
        suggestion_warnings = [
            w for w in caught if issubclass(w.category, SuggestionWarning)
        ]
        explicit_drop = [
            w for w in suggestion_warnings
            if "stub" in str(w.message)
            and "predict_survival_function" in str(w.message)
        ]
        assert explicit_drop, (
            f"expected SuggestionWarning naming 'stub' and "
            f"predict_survival_function; got: "
            f"{[str(w.message) for w in suggestion_warnings]}"
        )


class TestRocCiInternalWarning:
    """Coverage gap from review: bootstrap CI failure must emit InternalWarning.

    Tests the impl directly with a hand-constructed PlotContext so we
    don't depend on whether ``validate(bootstrap=True)`` happens to
    populate ``bootstrap_result.predictions`` (an optional field).
    """

    def test_ci_computation_failure_warns_and_continues(
        self, validated_learner, monkeypatch,
    ):
        from mlsurv._warnings import InternalWarning
        from mlsurv.viz import metrics as _viz_metrics
        from mlsurv.viz._plot_impl import _plot_roc_impl

        # Build a fake bootstrap result with non-empty predictions so
        # the show_ci branch is entered. Only the .predictions
        # attribute matters here — _plot_roc_impl reads it via
        # getattr() and immediately hands it to bootstrap_roc_curves.
        class FakeBootstrap:
            predictions = pd.DataFrame({
                "model_type": ["coxph"],
                "iteration": [0],
                "index": [0],
                "risk_score": [0.5],
            })

        # Build a real validation context, then swap in the fake
        # bootstrap. PlotContext is frozen, so use object.__setattr__.
        view = ValidationView(validated_learner, "Holdout")
        ctx = view._ctx()
        object.__setattr__(ctx, "bootstrap_result", FakeBootstrap())

        def boom(*args, **kwargs):
            raise ValueError("simulated CI failure")

        monkeypatch.setattr(_viz_metrics, "bootstrap_roc_curves", boom)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = _plot_roc_impl(ctx, show_ci=True, t=2.0)
        # Figure must still come back (CI bands silently omitted).
        assert isinstance(fig, go.Figure)
        # InternalWarning naming the cohort must have been emitted.
        internal = [
            w for w in caught
            if issubclass(w.category, InternalWarning)
            and "Holdout" in str(w.message)
        ]
        assert internal, (
            f"expected InternalWarning naming the cohort; got: "
            f"{[(w.category.__name__, str(w.message)) for w in caught]}"
        )


# ======================================================================
# SubpopulationView ports: plot_metric / plot_metric_time /
# plot_survival
# ======================================================================


def _make_synthetic_data():
    """Generate synthetic survival data (no NaN) for subpopulation fixtures.

    Mirrors ``tests/test_learner_plot_wrappers.py::_make_synthetic_data``
    — duplicated here to keep the subpopulation view tests self-contained
    (the legacy plot_subpopulation_survival wrapper, which used to hold
    the shared fixture, was deleted).
    """
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


@pytest.fixture(scope="module")
def _synth_data():
    return _make_synthetic_data()


@pytest.fixture(scope="module")
def subpopulation_learner(_synth_data):
    """Learner with one subpopulation variable + two trained models.

    Mirrors the earlier fixture that used to live in
    ``test_learner_plot_wrappers.py``. Scoped at the module level so the
    subpopulation-view test classes below all share the same training
    run — the trained models + evaluated metrics are several seconds
    to rebuild.
    """
    X, y = _synth_data
    median_val = X["feature_0"].median()
    labels = np.where(X["feature_0"] > median_val, "high", "low")
    learner = SurvivalLearner(X, y)
    learner.setup(subpopulations={"risk_group": labels})
    # tune=False: these subpopulation-view tests assert only on plot/metric
    # return-data, never on tuned hyperparameters, so the default 50-trial RSF
    # Optuna search is pure waste here (~40s). Mirrors the tune=False sibling
    # fixtures (_validated_learner_template, _multi_model_template).
    learner.run(models=["coxph", "rsf"], tune=False)
    learner.evaluate()
    return learner


@pytest.fixture
def multi_subpopulation_learner(_synth_data):
    """Learner with two subpopulation variables (for auto-pick tests)."""
    X, y = _synth_data
    median = X["feature_0"].median()
    a = np.where(X["feature_0"] > median, "high", "low")
    b = np.where(X["feature_1"] > X["feature_1"].median(), "top", "bot")
    learner = SurvivalLearner(X, y)
    learner.setup(subpopulations={"risk_group": a, "size_group": b})
    learner.run(models=["coxph"])
    learner.evaluate()
    return learner


# ------------------------------------------------------------------
# Construction contract
# ------------------------------------------------------------------


class TestSubpopulationViewConstruction:
    def test_stores_group(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        assert view.group == "risk_group"

    def test_result_is_subpopulation_result(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        assert view.result is (
            subpopulation_learner._state.subpopulation_results["risk_group"]
        )

    def test_repr(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        assert repr(view) == "SubpopulationView(group='risk_group')"

    def test_unknown_group_raises_keyerror(self, subpopulation_learner):
        with pytest.raises(KeyError, match="not found"):
            SubpopulationView(subpopulation_learner, "DoesNotExist")

    def test_keyerror_lists_available_groups(self, subpopulation_learner):
        with pytest.raises(KeyError, match="risk_group"):
            SubpopulationView(subpopulation_learner, "DoesNotExist")


# ------------------------------------------------------------------
# SurvivalLearner.subpopulation() accessor
# ------------------------------------------------------------------


class TestSubpopulationAccessor:
    def test_explicit_group(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        assert isinstance(view, SubpopulationView)
        assert view.group == "risk_group"

    def test_autoselect_single_group(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation()
        assert view.group == "risk_group"

    def test_autoselect_multi_group_raises(self, multi_subpopulation_learner):
        with pytest.raises(ValueError, match="Multiple subpopulation"):
            multi_subpopulation_learner.subpopulation()

    def test_no_subpopulations_raises(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None)
        learner.train("coxph")
        with pytest.raises(ValueError, match="No subpopulation variables"):
            learner.subpopulation()

    def test_unknown_group_raises_keyerror(self, subpopulation_learner):
        with pytest.raises(KeyError, match="not found"):
            subpopulation_learner.subpopulation("Nope")


# ------------------------------------------------------------------
# plot_metric (grouped forest)
# ------------------------------------------------------------------


class TestSubpopulationViewPlotMetric:
    def test_returns_figure(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_metric(style="forest")
        assert isinstance(fig, go.Figure)

    def test_auto_style_defaults_to_forest(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_metric()
        assert isinstance(fig, go.Figure)

    def test_return_data_returns_dataframe(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        df = view.plot_metric(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "model" in df.columns
        assert "subpopulation" in df.columns

    def test_filter_to_specific_model(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        df = view.plot_metric(models=["coxph"], return_data=True)
        assert set(df["model"].unique()) == {"coxph"}

    def test_return_data_filters_to_single_metric(self, subpopulation_learner):
        """return_data must match the metric the forest plots, not leak all
        computed metrics.

        evaluate() auto-computes several subpopulation metrics (harrell, ipcw,
        ctd, integrated_brier); the figure renders only one. Before the fix the
        return_data frame carried every metric.
        """
        view = subpopulation_learner.subpopulation("risk_group")
        # Default resolves to the learner scoring (harrell) — exactly one metric.
        df = view.plot_metric(return_data=True)
        assert set(df["metric"].unique()) == {"harrell"}, (
            f"return_data leaked metrics: {sorted(df['metric'].unique())}"
        )
        # An explicitly requested available metric is selected (proves the
        # filter follows metric=, and that >1 metric was present to leak).
        df_ipcw = view.plot_metric(metric="ipcw", return_data=True)
        assert set(df_ipcw["metric"].unique()) == {"ipcw"}

    # Canonical models= unknown-model resolver check for SubpopulationView.
    # All subpopulation plot methods share _resolve_models_arg(), so the
    # other plot-method classes don't repeat this.
    def test_unknown_model_raises(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(ValueError, match="Unknown models"):
            view.plot_metric(models=["nonexistent_model"])

    def test_bar_style_raises_notimplementederror(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="style='bar'"):
            view.plot_metric(style="bar")

    def test_references_raises_notimplementederror(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="references="):
            view.plot_metric(references="all")

    def test_multi_metric_raises_notimplementederror(self, subpopulation_learner):
        """A 2+ element metric list (multi-panel) is not supported on views."""
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="multi-metric"):
            view.plot_metric(metric=["harrell", "uno"])

    def test_single_element_metric_list_succeeds(self, subpopulation_learner):
        """A 1-element metric list collapses to a single-panel plot (no raise)."""
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_metric(metric=["harrell"])
        assert isinstance(fig, go.Figure)

    def test_cv_source_raises_notimplementederror(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="source="):
            view.plot_metric(source="cv")

    def test_vertical_orientation_raises_notimplementederror(
        self, subpopulation_learner
    ):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="orientation"):
            view.plot_metric(orientation="vertical")

    def test_level_single_panel(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_metric(level="high")
        assert isinstance(fig, go.Figure)

    def test_level_filters_data(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        df = view.plot_metric(level="high", return_data=True)
        assert set(df["subpopulation"].unique()) == {"high"}

    def test_level_unknown_raises(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(ValueError, match="Unknown subpopulation level"):
            view.plot_metric(level="nonexistent")


# ------------------------------------------------------------------
# plot_metric_time (faceted AUC(t))
# ------------------------------------------------------------------


class TestSubpopulationViewPlotMetricTime:
    def test_returns_figure(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_metric_time()
        assert isinstance(fig, go.Figure)

    def test_return_data_returns_dataframe(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        df = view.plot_metric_time(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "eval_time" in df.columns
        assert "auc" in df.columns
        assert "subpopulation" in df.columns

    def test_filter_to_specific_model(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        df = view.plot_metric_time(models=["coxph"], return_data=True)
        assert set(df["model"].unique()) == {"coxph"}

    # Unknown-model validation covered once in TestSubpopulationViewPlotMetric
    # (shared resolver).

    def test_empty_auc_t_raises_valueerror(self, subpopulation_learner):
        """Group with empty ``auc_t`` must raise a clear ValueError."""
        sp_result = subpopulation_learner._state.subpopulation_results[
            "risk_group"
        ]
        saved = sp_result.auc_t
        try:
            sp_result.auc_t = pd.DataFrame(
                columns=["model_type", "subpopulation", "eval_time", "auc"]
            )
            view = subpopulation_learner.subpopulation("risk_group")
            with pytest.raises(ValueError, match=r"no AUC\(t\) data"):
                view.plot_metric_time()
        finally:
            sp_result.auc_t = saved

    def test_references_raises_notimplementederror(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="references="):
            view.plot_metric_time(references="all")

    def test_multi_metric_returns_figure(self, subpopulation_learner):
        """``metric=['auc', 'brier']`` produces a faceted (metrics × subgroups)
        Figure."""
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_metric_time(metric=["auc", "brier"])
        assert isinstance(fig, go.Figure)

    def test_multi_metric_uses_wider_grid_than_single(self, subpopulation_learner):
        """Two metrics still route to the multi-metric grid (one row per metric),
        which has more subplot axes than the single-metric wrapped grid."""
        view = subpopulation_learner.subpopulation("risk_group")
        fig_single = view.plot_metric_time(metric="brier")
        fig_multi = view.plot_metric_time(metric=["auc", "brier"])
        n_ax_single = sum(
            1 for k in fig_single.layout
            if k == "xaxis" or k.startswith("xaxis")
        )
        n_ax_multi = sum(
            1 for k in fig_multi.layout
            if k == "xaxis" or k.startswith("xaxis")
        )
        # 2 metrics x 2 subpopulations = 4 axes vs 1 row x 2 subpopulations = 2.
        assert n_ax_multi > n_ax_single

    def test_multi_metric_return_data(self, subpopulation_learner):
        """Long-format frame must include both metrics and every subgroup."""
        view = subpopulation_learner.subpopulation("risk_group")
        df = view.plot_metric_time(
            metric=["auc", "brier"], return_data=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert set(df["metric_name"].unique()) >= {"auc", "brier"}
        assert "subpopulation" in df.columns
        assert df["subpopulation"].nunique() >= 2

    def test_multi_metric_rejects_unknown(self, subpopulation_learner):
        """Subpop views reuse the same ``{"auc","brier"}`` allowlist as
        validation views."""
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(ValueError, match="Unknown metric"):
            view.plot_metric_time(metric=["made_up"])

    def test_single_string_metric_is_single_panel(self, subpopulation_learner):
        """A string metric is now valid and draws a single-panel plot."""
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_metric_time(metric="auc")
        assert isinstance(fig, go.Figure)

    def test_single_brier_matches_auc_layout(self, subpopulation_learner):
        """Single brier shares the AUC facet grid and dimensions.

        Regression guard: a single non-AUC metric must not render through the
        wider, un-wrapped multi-metric grid.
        """
        view = subpopulation_learner.subpopulation("risk_group")
        fig_auc = view.plot_metric_time(metric="auc")
        fig_brier = view.plot_metric_time(metric="brier")
        assert fig_auc.layout.width == fig_brier.layout.width
        assert fig_auc.layout.height == fig_brier.layout.height
        # Same wrapped grid → same number of subplot x-axes.
        n_ax_auc = sum(
            1 for k in fig_auc.layout if k == "xaxis" or k.startswith("xaxis")
        )
        n_ax_brier = sum(
            1 for k in fig_brier.layout if k == "xaxis" or k.startswith("xaxis")
        )
        assert n_ax_auc == n_ax_brier

    def test_multi_metric_rejects_non_string_entry(self, subpopulation_learner):
        """A non-string element in the metric list raises TypeError."""
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(TypeError, match="must be strings"):
            view.plot_metric_time(metric=["auc", 1])

    def test_multi_metric_rejects_empty(self, subpopulation_learner):
        """Empty metric list is a user error, not a silent no-op."""
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(ValueError, match="empty"):
            view.plot_metric_time(metric=[])

    def test_level_single_panel(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_metric_time(level="high")
        assert isinstance(fig, go.Figure)

    def test_level_filters_data(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        df = view.plot_metric_time(level="high", return_data=True)
        assert set(df["subpopulation"].unique()) == {"high"}

    def test_level_unknown_raises(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(ValueError, match="Unknown subpopulation level"):
            view.plot_metric_time(level="nonexistent")

    def test_level_with_multi_metric_raises(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="level=.*multi-metric"):
            view.plot_metric_time(level="high", metric=["auc", "brier"])


# ------------------------------------------------------------------
# plot_survival (per-level KM)
# ------------------------------------------------------------------


class TestSubpopulationViewPlotSurvival:
    def test_returns_figure(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_survival()
        assert isinstance(fig, go.Figure)

    def test_return_data_returns_dataframe(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        df = view.plot_survival(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"time", "event", "Subpopulation"}
        # Both levels of risk_group must be present post-drop-NaN
        assert set(df["Subpopulation"].unique()) == {"high", "low"}

    def test_models_filter_raises_notimplementederror(
        self, subpopulation_learner
    ):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="models="):
            view.plot_survival(models=["coxph"])

    def test_n_groups_raises_notimplementederror(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="n_groups"):
            view.plot_survival(n_groups=3)

    def test_references_raises_notimplementederror(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(NotImplementedError, match="references="):
            view.plot_survival(references="all")

    def test_nan_labels_emit_warning_attributed_to_user_code(
        self, _synth_data,
    ):
        """Partial NaN subpopulation labels must emit SuggestionWarning
        with a stacklevel that attributes the warning to user code
        (the test file), not framework internals.

        Guards the ``stacklevel=5`` locked in at
        ``mlsurv/viz/_plot_impl.py::_build_subpopulation_survival_df``.
        A refactor that changes the call depth without updating
        stacklevel will misattribute the warning and fail this test.
        """
        X, y = _synth_data
        # Build a learner with one subpopulation variable, then
        # corrupt half the test-set labels with NaN to trigger the
        # drop-and-warn path.
        median_val = X["feature_0"].median()
        labels = np.where(X["feature_0"] > median_val, "high", "low")
        learner = SurvivalLearner(X, y)
        learner.setup(subpopulations={"risk_group": labels})
        learner.run(models=["coxph"])
        learner.evaluate()

        # Inject NaNs into half the test-set labels.
        test_labels = learner._state.subpopulations_test["risk_group"]
        corrupted = test_labels.astype(object).copy()
        corrupted[: len(corrupted) // 2] = np.nan
        learner._state.subpopulations_test["risk_group"] = corrupted

        view = learner.subpopulation("risk_group")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = view.plot_survival()

        assert isinstance(fig, go.Figure)
        drop_warnings = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and "Dropped" in str(w.message)
            and "risk_group" in str(w.message)
        ]
        assert drop_warnings, (
            f"expected SuggestionWarning mentioning 'Dropped' and "
            f"'risk_group'; got: "
            f"{[(w.category.__name__, str(w.message)) for w in caught]}"
        )
        # Stacklevel correctness: the warning must point at user code
        # (this test file), not at framework internals. A stacklevel
        # that lands inside mlsurv/viz/_plot_impl.py would let
        # warnings.filterwarnings(..., module=user) miss the warning
        # and defeat the "no silent data loss" contract.
        warn_filename = drop_warnings[0].filename.replace("\\", "/")
        assert warn_filename.endswith("test_learner_views.py"), (
            f"expected warning attributed to test_learner_views.py; "
            f"got {warn_filename!r}"
        )

    def test_level_single_panel(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        fig = view.plot_survival(level="high")
        assert isinstance(fig, go.Figure)

    def test_level_filters_data(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        df = view.plot_survival(level="high", return_data=True)
        assert set(df["Subpopulation"].unique()) == {"high"}

    def test_level_unknown_raises(self, subpopulation_learner):
        view = subpopulation_learner.subpopulation("risk_group")
        with pytest.raises(ValueError, match="Unknown subpopulation level"):
            view.plot_survival(level="nonexistent")

    def test_level_with_nan_labels_emits_single_warning(self, _synth_data):
        """plot_survival(level=...) on NaN-containing labels must emit
        the SuggestionWarning exactly once.

        Both _build_subpopulation_survival_df and
        _iter_subpopulation_levels emit the same "Dropped N/M test
        samples with NaN subpopulation labels…" SuggestionWarning when
        NaN labels are present. The survival path validates level
        inline against the just-built df rather than reusing the shared
        helper to avoid emitting the warning twice for one plot call.
        """
        X, y = _synth_data
        median_val = X["feature_0"].median()
        labels = np.where(X["feature_0"] > median_val, "high", "low")
        learner = SurvivalLearner(X, y)
        learner.setup(subpopulations={"risk_group": labels})
        learner.run(models=["coxph"])
        learner.evaluate()

        test_labels = learner._state.subpopulations_test["risk_group"]
        corrupted = test_labels.astype(object).copy()
        corrupted[: len(corrupted) // 2] = np.nan
        learner._state.subpopulations_test["risk_group"] = corrupted

        view = learner.subpopulation("risk_group")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = view.plot_survival(level="high")

        assert isinstance(fig, go.Figure)
        drop_warnings = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and "Dropped" in str(w.message)
            and "risk_group" in str(w.message)
        ]
        assert len(drop_warnings) == 1, (
            f"expected exactly 1 NaN-drop SuggestionWarning, got "
            f"{len(drop_warnings)}: "
            f"{[str(w.message) for w in drop_warnings]}"
        )


# ------------------------------------------------------------------
# Ribbon invariant through the view (test-analyzer Gap #1)
# ------------------------------------------------------------------


@pytest.mark.integration
class TestSubpopulationViewRibbonThroughView:
    """End-to-end ribbon assertion through ``SubpopulationView``.

    ``tests/test_subpopulation_analysis.py::TestSubpopulationAucTPlot``
    pins the ribbon invariants against
    :func:`plot_grouped_metric_time` directly, but that bypasses
    ``_plot_metric_time_subpopulation`` / ``_plot_metric_time_impl`` /
    :class:`SubpopulationView.plot_metric_time`. A regression that
    silently dropped ``axis.auc_t_bootstrap`` from the impl wiring
    would leave the direct-call test green while breaking the
    user-facing ``learner.subpopulation(...).plot_metric_time()``
    path. This class closes that gap with a single integration test.
    """

    @pytest.fixture(scope="class")
    def ribbon_learner(self, _synth_data):
        """Learner with subpopulations + bootstrap → auc_t_bootstrap populated.

        Passing ``bootstrap=True`` to ``evaluate()`` on a learner that
        was ``setup(subpopulations={...})`` triggers
        :func:`mlsurv.evaluation._subpopulation._attach_subpopulation_auc_bootstrap`
        which populates ``SubpopulationResult.auc_t_bootstrap`` — the
        frame :func:`plot_grouped_metric_time` consumes to draw ribbons.
        """
        X, y = _synth_data
        median_val = X["feature_0"].median()
        labels = np.where(X["feature_0"] > median_val, "high", "low")
        learner = SurvivalLearner(X, y)
        learner.setup(subpopulations={"risk_group": labels})
        learner.train("coxph")
        learner.train("rsf")
        learner.evaluate(bootstrap=True, n_bootstrap=5)
        return learner

    def test_ribbons_rendered_through_view(self, ribbon_learner):
        """plot_metric_time() via SubpopulationView must draw ribbons
        when auc_t_bootstrap is populated.

        Guards the wiring from view → impl → subpopulation branch →
        plot_grouped_metric_time. A bug that dropped
        ``axis.auc_t_bootstrap`` anywhere in that chain would produce
        a ribbon-less figure and fail this assertion.
        """
        sp = ribbon_learner._state.subpopulation_results.get("risk_group")
        if sp is None or sp.auc_t_bootstrap is None or sp.auc_t_bootstrap.empty:
            pytest.skip(
                "bootstrap(subpopulation_labels=...) did not populate "
                "auc_t_bootstrap on this fixture; ribbon path is "
                "unreachable through the view and the sibling test "
                "TestSubpopulationAucTPlot exercises the renderer directly."
            )

        fig = ribbon_learner.subpopulation("risk_group").plot_metric_time()
        assert isinstance(fig, go.Figure)

        fill_traces = [
            t for t in fig.data
            if getattr(t, "fill", None) in ("toself", "tonexty")
        ]
        assert fill_traces, (
            "Expected at least one ribbon fill trace through the "
            "SubpopulationView path; got none. "
            f"All trace fills: {[getattr(t, 'fill', None) for t in fig.data]}"
        )


# ------------------------------------------------------------------
# SubpopulationView.plot_calibration
# ------------------------------------------------------------------


class TestSubpopulationViewPlotCalibration:
    def test_returns_figure(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        fig = view.plot_calibration()
        assert isinstance(fig, go.Figure)

    def test_return_data_returns_dataframe(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        df = view.plot_calibration(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ("subpopulation", "model", "predicted", "observed"):
            assert col in df.columns, f"Missing column {col!r}"
        assert set(df["subpopulation"].unique()) == {"high", "low"}

    def test_panel_count_matches_levels(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        fig = view.plot_calibration()
        # Subplot titles are annotations at the top of each panel.
        # The fixture has "high" and "low" levels.
        annotations = [
            a for a in fig.layout.annotations
            if a.text in ("high", "low")
        ]
        assert len(annotations) == 2

    def test_filter_to_specific_model(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        df = view.plot_calibration(models="coxph", return_data=True)
        assert set(df["model"].unique()) == {"coxph"}

    # Unknown-model validation covered once in TestSubpopulationViewPlotMetric
    # (shared resolver).

    def test_level_single_panel(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        fig = view.plot_calibration(level="high")
        assert isinstance(fig, go.Figure)

    def test_level_unknown_raises(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        with pytest.raises(ValueError, match="Unknown subpopulation level"):
            view.plot_calibration(level="nonexistent")

    def test_no_survfunc_models_raises(self, subpopulation_learner):
        """Models lacking predict_survival_function raise ValueError."""
        # Replace all fitted models with stubs that lack the method.
        saved_models = {}
        for name, mr in subpopulation_learner._state.models.items():
            saved_models[name] = mr.model

        class _StubModel:
            def predict(self, X):
                return np.zeros(len(X))

        try:
            for name, mr in subpopulation_learner._state.models.items():
                mr.model = _StubModel()
            view = SubpopulationView(subpopulation_learner, "risk_group")
            with pytest.raises(ValueError, match="predict_survival_function"):
                view.plot_calibration()
        finally:
            for name, mr in subpopulation_learner._state.models.items():
                mr.model = saved_models[name]


# ------------------------------------------------------------------
# SubpopulationView.plot_roc
# ------------------------------------------------------------------


class TestSubpopulationViewPlotRoc:
    def test_returns_figure(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        fig = view.plot_roc()
        assert isinstance(fig, go.Figure)

    def test_return_data_returns_dataframe(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        df = view.plot_roc(return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ("subpopulation", "model", "fpr", "tpr", "auc"):
            assert col in df.columns, f"Missing column {col!r}"
        assert set(df["subpopulation"].unique()) == {"high", "low"}

    def test_panel_count_matches_levels(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        fig = view.plot_roc()
        annotations = [
            a for a in fig.layout.annotations
            if a.text in ("high", "low")
        ]
        assert len(annotations) == 2

    def test_filter_to_specific_model(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        df = view.plot_roc(models="coxph", return_data=True)
        assert set(df["model"].unique()) == {"coxph"}

    # Unknown-model validation covered once in TestSubpopulationViewPlotMetric
    # (shared resolver).

    def test_level_single_panel(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        fig = view.plot_roc(level="high")
        assert isinstance(fig, go.Figure)

    def test_level_unknown_raises(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        with pytest.raises(ValueError, match="Unknown subpopulation level"):
            view.plot_roc(level="nonexistent")

    def test_show_auc_false(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        fig = view.plot_roc(show_auc=False)
        assert isinstance(fig, go.Figure)
        # Hover templates should NOT contain "AUC" when show_auc=False.
        for trace in fig.data:
            ht = getattr(trace, "hovertemplate", "") or ""
            assert "AUC" not in ht, (
                f"show_auc=False but hovertemplate contains AUC: {ht!r}"
            )

    def test_show_auc_true_in_hover(self, subpopulation_learner):
        view = SubpopulationView(subpopulation_learner, "risk_group")
        fig = view.plot_roc(show_auc=True)
        assert isinstance(fig, go.Figure)
        # At least one trace's hover should contain "AUC".
        auc_traces = [
            t for t in fig.data
            if "AUC" in (getattr(t, "hovertemplate", "") or "")
        ]
        assert auc_traces, "show_auc=True but no hover contains AUC"


# ------------------------------------------------------------------
# SubpopulationView zero-sample level warning
# ------------------------------------------------------------------


class TestSubpopulationEdgeCases:
    """Edge-case tests for zero-sample levels and NaN labels."""

    def test_single_level_after_label_collapse(
        self, subpopulation_learner
    ):
        """When all labels collapse to one level, a single-panel figure
        is returned (no faceting needed).

        Note: the zero-sample warning in ``_iter_subpopulation_levels``
        is defensive but unreachable with string labels — if a level
        appears in ``sorted(set(labels[~nan_mask]))`` it necessarily
        has >= 1 non-NaN sample.
        """
        state = subpopulation_learner._state
        original = state.subpopulations_test["risk_group"].copy()

        state.subpopulations_test["risk_group"] = np.full(
            len(original), "high"
        )
        try:
            view = SubpopulationView(subpopulation_learner, "risk_group")
            fig = view.plot_calibration()
            assert isinstance(fig, go.Figure)
        finally:
            state.subpopulations_test["risk_group"] = original

    def test_nan_labels_emit_warning_on_calibration(
        self, subpopulation_learner
    ):
        """NaN subpopulation labels are dropped with SuggestionWarning."""
        state = subpopulation_learner._state
        original = state.subpopulations_test["risk_group"].copy()

        labels = original.copy().astype(object)
        labels[0] = None  # inject one NaN
        state.subpopulations_test["risk_group"] = labels
        try:
            view = SubpopulationView(subpopulation_learner, "risk_group")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig = view.plot_calibration()
            assert isinstance(fig, go.Figure)
            dropped_warnings = [
                w for w in caught
                if issubclass(w.category, SuggestionWarning)
                and "Dropped" in str(w.message)
                and "NaN" in str(w.message)
            ]
            assert dropped_warnings, (
                f"Expected SuggestionWarning about dropped NaN labels; "
                f"got {[str(w.message) for w in caught]}"
            )
        finally:
            state.subpopulations_test["risk_group"] = original

    def test_nan_labels_emit_warning_on_roc(
        self, subpopulation_learner
    ):
        """NaN subpopulation labels are dropped with SuggestionWarning."""
        state = subpopulation_learner._state
        original = state.subpopulations_test["risk_group"].copy()

        labels = original.copy().astype(object)
        labels[0] = None  # inject one NaN
        state.subpopulations_test["risk_group"] = labels
        try:
            view = SubpopulationView(subpopulation_learner, "risk_group")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig = view.plot_roc()
            assert isinstance(fig, go.Figure)
            dropped_warnings = [
                w for w in caught
                if issubclass(w.category, SuggestionWarning)
                and "Dropped" in str(w.message)
                and "NaN" in str(w.message)
            ]
            assert dropped_warnings, (
                f"Expected SuggestionWarning about dropped NaN labels; "
                f"got {[str(w.message) for w in caught]}"
            )
        finally:
            state.subpopulations_test["risk_group"] = original
