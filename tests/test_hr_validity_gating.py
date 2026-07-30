"""Integration tests for hazard-ratio validity gating.

``exp(coef)`` is a genuine hazard ratio only for models whose
``hazard_ratio_valid`` flag is True (currently unpenalized Cox PH). CoxNet
(penalized) and FSSVM (ranking) coefficients are not log-hazards, so the
forest plots, the HR table, and the report's multivariate HR forest must not
present them as hazard ratios.

Companion fast/mock-based tests live in ``test_phase4_viz.py::TestForest`` and
``test_phase4_tables.py::TestFormatHazardRatios``.
"""

import pytest


@pytest.fixture(scope="module")
def coxnet_only_learner():
    """A learner with only CoxNet trained (no HR-valid model)."""
    from mlsurv import SurvivalLearner
    from sksurv.datasets import load_gbsg2
    import numpy as np

    X_raw, y_raw = load_gbsg2()
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array([(bool(e), t) for e, t in y_raw],
                 dtype=[("event", bool), ("time", float)])

    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)
    learner.train("coxnet", tune=False)
    return learner


@pytest.fixture(scope="module")
def coxph_learner():
    """A learner with CoxPH trained (HR-valid) for the regression control."""
    from mlsurv import SurvivalLearner
    from sksurv.datasets import load_gbsg2
    import numpy as np

    X_raw, y_raw = load_gbsg2()
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array([(bool(e), t) for e, t in y_raw],
                 dtype=[("event", bool), ("time", float)])

    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)
    learner.train("coxph", tune=False)
    return learner


@pytest.fixture(scope="module")
def penalized_coxph_learner():
    """A CoxPH trained with a user-specified ridge penalty (alpha>0).

    A penalized CoxPH is no longer HR-valid, so it must be gated exactly like
    CoxNet even though the model name is 'coxph'.
    """
    from mlsurv import SurvivalLearner
    from sksurv.datasets import load_gbsg2
    import numpy as np
    import warnings

    X_raw, y_raw = load_gbsg2()
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array([(bool(e), t) for e, t in y_raw],
                 dtype=[("event", bool), ("time", float)])

    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)
    # The ridge penalty makes the fit emit a MethodologicalWarning by design.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.train("coxph", model_params={"coxph": {"alpha": 0.5}}, tune=False)
    return learner


@pytest.mark.integration
class TestLearnerHRGating:
    """plot_hazard_ratios paths gate on hazard_ratio_valid."""

    def test_multivariate_plot_raises_for_coxnet_only(self, coxnet_only_learner):
        """No HR-valid model -> clear ValueError, not a mislabeled CoxNet forest."""
        with pytest.raises(ValueError, match="valid hazard ratios"):
            coxnet_only_learner.plot_hazard_ratios(kind="multivariate")

    def test_multivariate_plot_raises_for_penalized_coxph(
        self, penalized_coxph_learner,
    ):
        """A penalized CoxPH (alpha>0) is gated like any non-HR-valid model."""
        with pytest.raises(ValueError, match="valid hazard ratios"):
            penalized_coxph_learner.plot_hazard_ratios(kind="multivariate")

    def test_coefficients_suppress_hr_and_pvalue_for_penalized_coxph(
        self, penalized_coxph_learner,
    ):
        """Evaluation-layer gate: the coefficients table keeps the coef column
        but NaNs HR/CI/p-value for a penalized CoxPH (not just the viz path)."""
        df = penalized_coxph_learner.coefficients
        coxph_rows = df[df["model_type"] == "coxph"]
        assert not coxph_rows.empty
        assert coxph_rows["coefficient"].notna().any()  # coefficients still reported
        for col in ["hazard_ratio", "hr_lower_95", "hr_upper_95", "p_value"]:
            assert coxph_rows[col].isna().all(), f"{col} must be NaN for penalized CoxPH"

    def test_comparison_plot_raises_for_coxnet_only(self, coxnet_only_learner):
        """Univariate-vs-multivariate comparison needs an HR-valid multivariate model."""
        with pytest.raises(ValueError, match="valid hazard ratios"):
            coxnet_only_learner.plot_hazard_ratios_comparison()

    def test_multivariate_plot_works_for_coxph(self, coxph_learner):
        """Regression: CoxPH still renders the multivariate HR forest."""
        import plotly.graph_objects as go

        fig = coxph_learner.plot_hazard_ratios(kind="multivariate")
        assert isinstance(fig, go.Figure)


@pytest.mark.integration
class TestReportHRGating:
    """The report's multivariate HR forest skips cleanly for non-HR models."""

    def test_section_skips_multivariate_for_coxnet_only(self, coxnet_only_learner):
        """coxnet-only + no univariate -> empty section, no error placeholder."""
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
        from mlsurv.reports.sections.feature_analysis._hazard_ratios import (
            generate_hazard_ratio_analysis,
        )

        section = FeatureAnalysisSection(coxnet_only_learner)
        out = generate_hazard_ratio_analysis(section, univar_cox=None)

        # Clean skip: empty markdown, no figures, and crucially no
        # "*[Hazard ratio plot unavailable: ...]*" exception placeholder.
        assert out["markdown"] == ""
        assert out["figures"] == []
        assert "unavailable" not in out["markdown"]

    def test_section_renders_univariate_but_skips_multivariate(
        self, coxnet_only_learner,
    ):
        """Realistic coxnet-only report: univariate Cox renders, multivariate is gated.

        Univariate Cox is per-feature and model-independent (genuinely valid),
        so it must still appear. The multivariate forest and the
        univariate-vs-multivariate comparison must be suppressed because no
        HR-valid multivariate model exists (the bug was rendering a
        mislabeled CoxNet forest here).
        """
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
        from mlsurv.reports.sections.feature_analysis._hazard_ratios import (
            generate_hazard_ratio_analysis,
        )

        univar_cox = coxnet_only_learner._state.univariate_cox
        assert univar_cox is not None and "hr_lower_95" in univar_cox.columns

        section = FeatureAnalysisSection(coxnet_only_learner)
        out = generate_hazard_ratio_analysis(section, univar_cox=univar_cox)

        assert "Univariate Hazard Ratios" in out["markdown"]
        assert "Multivariate Hazard Ratios" not in out["markdown"]
        assert "Univariate vs Multivariate" not in out["markdown"]
        # No swallowed-exception placeholder.
        assert "unavailable" not in out["markdown"]

    def test_section_renders_multivariate_for_coxph(self, coxph_learner):
        """Regression: CoxPH report still produces the multivariate HR forest."""
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
        from mlsurv.reports.sections.feature_analysis._hazard_ratios import (
            generate_hazard_ratio_analysis,
        )

        section = FeatureAnalysisSection(coxph_learner)
        out = generate_hazard_ratio_analysis(section, univar_cox=None)

        assert "Multivariate Hazard Ratios" in out["markdown"]
        assert out["figures"]


@pytest.mark.integration
class TestSuppressionIsExplainedWhereTheGapIs:
    """The reason belongs in the section the reader is looking at.

    Model Development already carried it, but a reader who notices the missing
    forest plot is in Feature Analysis and has no pointer to the other section.
    """

    def _render(self, learner, univar_cox=None):
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
        from mlsurv.reports.sections.feature_analysis._hazard_ratios import (
            generate_hazard_ratio_analysis,
        )

        section = FeatureAnalysisSection(learner)
        return generate_hazard_ratio_analysis(section, univar_cox=univar_cox)

    def _copy_without_covariance(self, learner):
        """Return ``(learner copy, inner CoxPH)`` with the covariance removed.

        Deleting the attribute beats trying to provoke a singular Hessian:
        scipy's inv succeeds on every degenerate design a caller can build,
        because the alpha fallback regularizes the fit first. The copy keeps
        the module-scoped fixture intact for the other tests.
        """
        import copy

        from mlsurv.utils import unwrap_pipeline

        learner = copy.deepcopy(learner)
        inner = unwrap_pipeline(learner.models["coxph"].model)
        del inner.model_.covariance_matrix_
        return learner, inner

    def test_penalized_coxph_explains_the_missing_forest(
        self, penalized_coxph_learner,
    ):
        """The gap used to be unexplained here even after the reason existed."""
        out = self._render(penalized_coxph_learner)

        assert "Not reported for this model" in out["markdown"]
        assert "alpha=" in out["markdown"]
        # Explaining the gap must not resurrect the plot it explains.
        assert out["figures"] == []

    def test_coxnet_only_still_returns_nothing(self, coxnet_only_learner):
        """CoxNet records no reason, so the clean skip is unchanged.

        Guards the boundary of the new branch: it must fire for a model that
        deliberately withheld its hazard ratios, not for one that never had
        any to withhold.
        """
        out = self._render(coxnet_only_learner)

        assert out["markdown"] == ""
        assert out["figures"] == []

    def test_healthy_coxph_still_promises_its_intervals(self, coxph_learner):
        """The common case: the CI clause is conditional, not removed."""
        out = self._render(coxph_learner)

        assert "with 95% confidence intervals" in out["markdown"]
        assert "Confidence intervals are not shown" not in out["markdown"]
        assert "Not reported for this model" not in out["markdown"]

    def test_missing_covariance_drops_the_interval_promise(self, coxph_learner):
        """The missing-interval branch, asserted positively.

        The healthy case is covered above; without this, nothing pins that the
        prose actually stops promising intervals when the plot has none.
        """
        learner, _ = self._copy_without_covariance(coxph_learner)

        out = self._render(learner)

        assert "with 95% confidence intervals" not in out["markdown"]
        assert "Confidence intervals are not shown" in out["markdown"]
        # The forest plot itself still belongs on the page: the hazard ratios
        # are unaffected, only the whiskers are gone.
        assert "Multivariate Hazard Ratios" in out["markdown"]
        assert out["figures"]

    def test_interval_note_falls_back_when_no_reason_was_recorded(
        self, coxph_learner,
    ):
        """A model that never recorded a reason still gets an explanation.

        Deleting covariance_matrix_ by hand leaves
        covariance_unavailable_reason_ at None, which is also what an older
        pickle looks like. The note must not render as an empty sentence.
        """
        learner, inner = self._copy_without_covariance(coxph_learner)
        assert inner.covariance_unavailable_reason_ is None

        out = self._render(learner)

        assert "No covariance matrix was available for this fit." in out["markdown"]
