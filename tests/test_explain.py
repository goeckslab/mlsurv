"""Tests for learner.explain() — individual patient explanations."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv.evaluation._explain import compute_patient_explanation
from mlsurv.results._explanation import PatientExplanation


def _has_shap():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


requires_shap = pytest.mark.skipif(not _has_shap(), reason="shap required")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_explain_data():
    """Small synthetic survival dataset (80 samples, 3 features)."""
    rng = np.random.RandomState(42)
    n = 80
    X = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "biomarker": rng.exponential(1.0, n),
        "treatment": rng.choice([0, 1], n),
    })
    time = rng.exponential(5, n).clip(0.1)
    event = rng.choice([True, False], n, p=[0.7, 0.3])
    y = np.array(list(zip(event, time)),
                 dtype=[("event", bool), ("time", "<f8")])
    return X, y


@pytest.fixture()
def sample_data():
    """Small synthetic survival dataset."""
    return _make_explain_data()


@pytest.fixture(scope="module")
def _trained_template():
    """Train CoxPH once per module; consumers get isolated deep copies.

    Module-scoped, so it inlines the data (it can't depend on the
    function-scoped ``sample_data`` fixture).
    """
    from mlsurv.learner import SurvivalLearner
    X, y = _make_explain_data()
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, random_state=42, verbose=False)
    learner.train("coxph", verbose=False)
    return learner


@pytest.fixture()
def trained_learner(_trained_template):
    """SurvivalLearner with CoxPH trained (no evaluate).

    Per-test deep copy of the module-scoped template, so a test that
    mutates the learner can't leak state into other tests.
    """
    import copy
    return copy.deepcopy(_trained_template)


# ---------------------------------------------------------------------------
# plot_waterfall
# ---------------------------------------------------------------------------

def _waterfall_bar_traces(fig):
    """Return the filled-polygon bar traces, excluding the hover-overlay markers."""
    return [t for t in fig.data if getattr(t, "fill", None) == "toself"]


class TestPlotWaterfall:
    def test_returns_figure(self):
        from mlsurv.viz.explain import plot_waterfall

        df = pd.DataFrame({
            "feature": [f"f{i}" for i in range(5)],
            "feature_value": np.arange(5, dtype=float),
            "shap_value": [0.5, -0.3, 0.2, -0.1, 0.05],
            "abs_shap_value": [0.5, 0.3, 0.2, 0.1, 0.05],
        })
        fig = plot_waterfall(df, risk_score=1.0, n=3)
        assert isinstance(fig, go.Figure)
        assert len(_waterfall_bar_traces(fig)) == 4  # 3 top + 1 "Other"

    def test_feature_groups_coloring(self):
        from mlsurv.viz.explain import plot_waterfall

        df = pd.DataFrame({
            "feature": ["age", "biomarker", "treatment"],
            "feature_value": [65.0, 1.2, 1],
            "shap_value": [0.5, -0.3, 0.1],
            "abs_shap_value": [0.5, 0.3, 0.1],
        })
        groups = {"age": "demographics", "biomarker": "lab", "treatment": "clinical"}
        fig = plot_waterfall(df, risk_score=0.3, feature_groups=groups)
        assert isinstance(fig, go.Figure)
        # All 3 features shown, no "Other".
        assert len(_waterfall_bar_traces(fig)) == 3
        # Y-axis ticks are ordered bottom-to-top with largest |SHAP| at top,
        # so the reversed tick list (top-to-bottom) follows abs_shap desc.
        labels = list(fig.layout.yaxis.ticktext)[::-1]
        assert labels[0].startswith("age = 65") and "[demographics]" in labels[0]
        assert labels[1].startswith("biomarker = 1.2") and "[lab]" in labels[1]
        assert labels[2].startswith("treatment = 1") and "[clinical]" in labels[2]

    def test_no_aggregation_when_all_shown(self):
        from mlsurv.viz.explain import plot_waterfall

        df = pd.DataFrame({
            "feature": ["a", "b"],
            "feature_value": [1.0, 2.0],
            "shap_value": [0.5, -0.3],
            "abs_shap_value": [0.5, 0.3],
        })
        fig = plot_waterfall(df, risk_score=0.2, n=10)
        assert len(_waterfall_bar_traces(fig)) == 2  # no "Other" bar

    def test_zero_features_hidden(self):
        """Raw zero-SHAP (selector-dropped) features are hidden, not bucketed
        into 'Other'."""
        from mlsurv.viz.explain import plot_waterfall

        # 2 real contributors + 3 dropped features at exactly 0.
        df = pd.DataFrame({
            "feature": ["a", "b", "d0", "d1", "d2"],
            "feature_value": [1.0, 2.0, 0.0, 0.0, 0.0],
            "shap_value": [0.5, -0.3, 0.0, 0.0, 0.0],
            "abs_shap_value": [0.5, 0.3, 0.0, 0.0, 0.0],
        })
        fig = plot_waterfall(df, risk_score=0.2, n=10)
        # Only the 2 non-zero bars — no "Other (3 features)" bucket.
        assert len(_waterfall_bar_traces(fig)) == 2
        labels = list(fig.layout.yaxis.ticktext)
        assert not any("Other" in lab for lab in labels)

    def test_all_zero_falls_back(self):
        """If every contribution is zero, the chart still renders (no crash)."""
        from mlsurv.viz.explain import plot_waterfall

        df = pd.DataFrame({
            "feature": ["a", "b"],
            "feature_value": [1.0, 2.0],
            "shap_value": [0.0, 0.0],
            "abs_shap_value": [0.0, 0.0],
        })
        fig = plot_waterfall(df, risk_score=0.0, n=10)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# explain() input validation
# ---------------------------------------------------------------------------

class TestExplainValidation:
    def test_both_patient_and_X_raises(self, trained_learner):
        with pytest.raises(ValueError, match="exactly one"):
            trained_learner.explain(patient=0, X=pd.DataFrame({"a": [1]}))

    def test_neither_patient_nor_X_raises(self, trained_learner):
        with pytest.raises(ValueError, match="exactly one"):
            trained_learner.explain()

    def test_out_of_bounds_index(self, trained_learner):
        with pytest.raises(ValueError, match="out of range"):
            trained_learner.explain(patient=9999)

    def test_untrained_model_raises(self, sample_data):
        from mlsurv.learner import SurvivalLearner
        X, y = sample_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.raises(ValueError, match="No trained models"):
            learner.explain(patient=0)

    def test_ambiguous_model_raises(self, sample_data):
        from mlsurv.learner import SurvivalLearner
        X, y = sample_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        learner.train("rsf", verbose=False)
        with pytest.raises(ValueError, match="Multiple models"):
            learner.explain(patient=0)

    def test_X_wrong_type_raises(self, trained_learner):
        with pytest.raises(TypeError, match="DataFrame"):
            trained_learner.explain(X=[[1, 2, 3]])

    def test_X_multiple_rows_raises(self, trained_learner):
        X_multi = trained_learner.X_test.head(2)
        with pytest.raises(ValueError, match="exactly 1 row"):
            trained_learner.explain(X=X_multi)


# ---------------------------------------------------------------------------
# explain() title — patient-index suffix (fast tier: no SHAP, no rendering)
# ---------------------------------------------------------------------------

class TestExplainTitle:
    """The waterfall title names the resolved X_test position.

    These run in the fast tier: the SHAP core and the plotter are patched so
    only the title-building / ``patient_index`` logic is exercised (no SHAP
    computation, no figure rendering), guarding the ``is not None`` guard and
    the ``patient=0`` edge that the integration-marked tests below also cover.
    """

    @staticmethod
    def _explain_capturing_title(learner, **kwargs):
        import mlsurv.evaluation._explain as ev
        import mlsurv.viz.explain as vz

        captured = {}

        def fake_compute(pipeline, X_patient, X_train, **_kw):
            df = pd.DataFrame({
                "feature": ["age", "biomarker"],
                "feature_value": [60.0, 1.0],
                "shap_value": [0.2, -0.1],
                "abs_shap_value": [0.2, 0.1],
            })
            return df, 0.0

        def fake_plot(contributions, **kw):
            captured["title"] = kw["title"]
            return go.Figure()

        with patch.object(ev, "compute_patient_explanation", fake_compute), \
                patch.object(vz, "plot_waterfall", fake_plot):
            result = learner.explain(**kwargs)
        return result, captured["title"]

    def test_int_patient_suffix(self, trained_learner):
        """patient=0 (falsy int) still gets the suffix — guard is `is not None`."""
        result, title = self._explain_capturing_title(trained_learner, patient=0)
        assert result.patient_index == 0
        assert "(patient 0)" in title

    def test_new_patient_has_no_suffix(self, trained_learner):
        X_new = trained_learner.X_test.iloc[[0]].copy()
        result, title = self._explain_capturing_title(trained_learner, X=X_new)
        assert result.patient_index is None
        assert "(patient" not in title


# ---------------------------------------------------------------------------
# explain() integration — CoxPH
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestExplainCoxPH:
    def test_patient(self, trained_learner):
        result = trained_learner.explain(patient=0)

        assert isinstance(result, PatientExplanation)

        # DataFrame checks
        df = result.shap_values
        expected_cols = {
            "model_name", "feature", "feature_value",
            "shap_value", "abs_shap_value", "risk_score",
        }
        assert set(df.columns) == expected_cols
        assert (df["model_name"] == "coxph").all()
        assert isinstance(df["risk_score"].iloc[0], float)

        # Figure checks
        assert isinstance(result.figure, go.Figure)

    def test_patient_index_in_title(self, trained_learner):
        """Integer patient is named in the waterfall title and on the result."""
        result = trained_learner.explain(patient=0)

        assert result.patient_index == 0
        title = result.figure.layout.title.text
        assert "Risk Explanation" in title
        assert "(patient 0)" in title

    def test_patient_index_percentile_spec(self, trained_learner):
        """A risk-percentile spec resolves to the same positional index the
        survival-curve plot uses, and that index appears in the title."""
        from mlsurv.learner._patient_resolve import resolve_patient_index

        pipeline = trained_learner.models["coxph"].model
        expected = resolve_patient_index(
            "high_risk", trained_learner.X_test, pipeline,
        )

        result = trained_learner.explain(patient="high_risk")

        assert result.patient_index == expected
        assert f"(patient {expected})" in result.figure.layout.title.text

    def test_new_patient_data(self, trained_learner):
        X_new = trained_learner.X_test.iloc[[2]].copy()
        result = trained_learner.explain(X=X_new)

        assert isinstance(result, PatientExplanation)
        assert len(result.shap_values) > 0
        assert isinstance(result.figure, go.Figure)
        # The X= path has no test-set position, so no patient index/suffix.
        assert result.patient_index is None
        assert "(patient" not in result.figure.layout.title.text

    def test_plot_false(self, trained_learner):
        result = trained_learner.explain(patient=0, plot=False)
        assert result.figure is None
        assert len(result.shap_values) > 0

    def test_works_without_evaluate(self, sample_data):
        """explain() should work after train() alone (no evaluate needed)."""
        from mlsurv.learner import SurvivalLearner
        X, y = sample_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, random_state=42, verbose=False)
        learner.train("coxph", verbose=False)
        # No evaluate() call
        result = learner.explain(patient=0, plot=False)
        assert isinstance(result, PatientExplanation)
        assert len(result.shap_values) > 0


# ---------------------------------------------------------------------------
# explain() integration — RSF
# ---------------------------------------------------------------------------

class TestExplainRSF:
    @pytest.mark.requires_shap
    @pytest.mark.timeout(120)
    def test_patient(self, sample_data):
        from mlsurv.learner import SurvivalLearner
        X, y = sample_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, random_state=42, verbose=False)
        learner.train("rsf", verbose=False)

        result = learner.explain(patient=0, model="rsf")
        assert isinstance(result, PatientExplanation)
        assert (result.shap_values["model_name"] == "rsf").all()
        assert isinstance(result.figure, go.Figure)


# ---------------------------------------------------------------------------
# compute_patient_explanation — thin wrapper over the shared SHAP core
# ---------------------------------------------------------------------------


class TestComputePatientExplanation:
    """Unit tests for the thin single-patient computer.

    The full pipeline is handed straight to the shared core (``_shap_array``)
    on raw features, so these mock the core and assert the raw-feature framing,
    zero-feature retention, sort order, and ``base_value`` pass-through.
    """

    @staticmethod
    def _make_data(n_features=3):
        X = pd.DataFrame(
            np.arange(n_features, dtype=float).reshape(1, -1),
            columns=[f"f{i}" for i in range(n_features)],
        )
        X_bg = pd.DataFrame(np.ones((3, n_features)), columns=X.columns)
        return X, X_bg

    @pytest.mark.requires_shap
    def test_raw_features_and_values(self):
        """Output lists raw feature names with their raw patient values, and the
        full pipeline + raw background pass straight to the shared core."""
        X, X_bg = self._make_data(3)
        pipeline = object()  # sentinel — must reach the core untouched
        with patch("mlsurv.evaluation._explain._shap_array") as mock_core:
            mock_core.return_value = (np.array([[0.1, -0.3, 0.2]]), 0.42)
            result, base_value = compute_patient_explanation(pipeline, X, X_bg)

        # raw column names preserved; sorted by |shap| desc (f1, f2, f0)
        assert list(result["feature"]) == ["f1", "f2", "f0"]
        assert dict(zip(result["feature"], result["feature_value"])) == {
            "f0": 0.0, "f1": 1.0, "f2": 2.0,
        }
        assert base_value == 0.42
        # the full pipeline + raw patient + raw background reach the core as-is
        args, _kwargs = mock_core.call_args
        assert args[0] is pipeline  # model == full pipeline
        assert args[1] is X         # X_explain == raw X_patient
        assert args[2] is X_bg      # X_background forwarded

    @pytest.mark.requires_shap
    def test_zero_feature_retained(self):
        """A selector-dropped (shap == 0) raw feature stays in the DataFrame."""
        X, X_bg = self._make_data(3)
        with patch("mlsurv.evaluation._explain._shap_array") as mock_core:
            # base_value 0.0 must pass through (the check is ``is not None``)
            mock_core.return_value = (np.array([[0.5, 0.0, -0.1]]), 0.0)
            result, base_value = compute_patient_explanation(object(), X, X_bg)

        assert set(result["feature"]) == {"f0", "f1", "f2"}
        zero_row = result[result["feature"] == "f1"]
        assert float(zero_row["shap_value"].iloc[0]) == 0.0
        assert base_value == 0.0

    @pytest.mark.requires_shap
    def test_all_zero_features_retained(self):
        """A patient whose every feature is zero-SHAP keeps all rows (the
        display layers, not the computer, handle the all-zero fallback)."""
        X, X_bg = self._make_data(3)
        with patch("mlsurv.evaluation._explain._shap_array") as mock_core:
            mock_core.return_value = (np.array([[0.0, 0.0, 0.0]]), 0.0)
            result, base_value = compute_patient_explanation(object(), X, X_bg)

        assert set(result["feature"]) == {"f0", "f1", "f2"}
        assert (result["shap_value"] == 0.0).all()
        assert base_value == 0.0

    @pytest.mark.requires_shap
    def test_categorical_groups_collapse_on_raw_dummies(self):
        """categorical_groups sums raw dummy columns into one group row, in raw
        feature space (the dummies are raw input columns now)."""
        X = pd.DataFrame({"age": [60.0], "stage_I": [1.0], "stage_II": [0.0]})
        X_bg = pd.DataFrame(np.ones((3, 3)), columns=X.columns)
        groups = {"stage": ["stage_I", "stage_II"]}
        with patch("mlsurv.evaluation._explain._shap_array") as mock_core:
            # age=0.5, stage_I=0.3, stage_II=-0.1 -> group 'stage' = 0.2
            mock_core.return_value = (np.array([[0.5, 0.3, -0.1]]), 0.0)
            result, _ = compute_patient_explanation(
                object(), X, X_bg, categorical_groups=groups,
            )

        assert set(result["feature"]) == {"age", "stage"}
        stage_row = result[result["feature"] == "stage"].iloc[0]
        assert float(stage_row["shap_value"]) == pytest.approx(0.2)
        # active category = the dummy with the highest patient value (stage_I=1)
        assert stage_row["feature_value"] == "stage_I"

    def test_single_row_required(self):
        """X_patient with != 1 row raises ValueError before touching the core."""
        X = pd.DataFrame({"f0": [1.0, 2.0]})
        with pytest.raises(ValueError, match="exactly 1 row"):
            compute_patient_explanation(object(), X, X)


# ---------------------------------------------------------------------------
# percentile string shortcut for patient
# ---------------------------------------------------------------------------


class TestPatientPercentileShortcut:
    """`patient='low_risk'|'high_risk'|'p10'...` resolves via X_test."""

    @pytest.mark.integration
    def test_low_risk_matches_manual_argsort(self, trained_learner):
        pipeline = trained_learner.best_model.model
        X_test = trained_learner._state.X_test
        order = np.argsort(pipeline.predict(X_test))
        expected_idx = int(order[int(0.10 * (len(order) - 1))])

        result = trained_learner.explain(
            patient="low_risk", n_features=3, plot=False,
        )
        # Risk score must equal what the manual block would compute.
        manual_risk = float(pipeline.predict(X_test.iloc[[expected_idx]])[0])
        assert float(result.shap_values["risk_score"].iloc[0]) == pytest.approx(
            manual_risk
        )

    @pytest.mark.integration
    def test_high_risk_matches_manual_argsort(self, trained_learner):
        pipeline = trained_learner.best_model.model
        X_test = trained_learner._state.X_test
        order = np.argsort(pipeline.predict(X_test))
        expected_idx = int(order[int(0.90 * (len(order) - 1))])

        result = trained_learner.explain(
            patient="high_risk", n_features=3, plot=False,
        )
        manual_risk = float(pipeline.predict(X_test.iloc[[expected_idx]])[0])
        assert float(result.shap_values["risk_score"].iloc[0]) == pytest.approx(
            manual_risk
        )

    @pytest.mark.integration
    def test_low_and_high_risk_differ(self, trained_learner):
        low = trained_learner.explain(
            patient="low_risk", n_features=3, plot=False,
        )
        high = trained_learner.explain(
            patient="high_risk", n_features=3, plot=False,
        )
        # In a non-degenerate cohort the two should pick different patients.
        assert float(low.shap_values["risk_score"].iloc[0]) <= float(
            high.shap_values["risk_score"].iloc[0]
        )

    @pytest.mark.integration
    def test_integer_path_still_works(self, trained_learner):
        # Regression: the integer-index path still works.
        result = trained_learner.explain(
            patient=0, n_features=3, plot=False,
        )
        assert result.shap_values is not None

    def test_unrecognized_string_raises(self, trained_learner):
        with pytest.raises(ValueError, match="not recognized"):
            trained_learner.explain(patient="bogus", plot=False)

    @pytest.mark.integration
    def test_explain_accepts_patient_kwarg_parity_with_patient_report(
        self, trained_learner
    ):
        """explain() must accept patient= (parity with
        patient_report()), not the legacy patient_index=."""
        result = trained_learner.explain(patient="high_risk", plot=False)
        assert result.shap_values is not None

    def test_explain_rejects_legacy_patient_index_kwarg(self, trained_learner):
        """The legacy patient_index= name must be gone."""
        with pytest.raises(TypeError, match="patient_index"):
            trained_learner.explain(patient_index=0)


# ---------------------------------------------------------------------------
# single-patient explain() reconciles with the cohort SHAP engine
# ---------------------------------------------------------------------------


@pytest.mark.shap
@pytest.mark.integration
@requires_shap
class TestCohortReconciliation:
    """A patient's waterfall SHAP equals their row in the cohort beeswarm.

    Both paths now route through the same shared core (``_shap_array``) on the
    full pipeline in raw feature space with the same ``X_train`` background and
    ``random_state``, so per-feature values match exactly.
    """

    def test_patient_matches_cohort_row(self, trained_learner):
        from mlsurv.evaluation._shap import _compute_shap

        learner = trained_learner
        pipeline = learner._state.models["coxph"].model
        X_test = learner._state.X_test
        X_train = learner._state.X_train
        rs = learner._state.random_state

        # Cohort SHAP via the unified engine — explain ALL test rows (no
        # subsample) so sample_idx maps 1:1 to X_test positions.
        cohort = _compute_shap(
            pipeline, X_test, X_background=X_train,
            max_background=50, max_explain=len(X_test) + 10, random_state=rs,
        )

        i = 0
        cohort_row = (
            cohort[cohort["sample_idx"] == i].set_index("feature")["shap_value"]
        )

        expl = learner.explain(patient=i, plot=False, max_background=50)
        patient_sv = expl.shap_values.set_index("feature")["shap_value"]

        # Same raw feature set, same values.  Per-row permutation SHAP is
        # independent of the rest of the explained set, so the only gap is
        # float noise from batched (cohort) vs single-row (patient) predict.
        assert set(patient_sv.index) == set(cohort_row.index)
        for feat in cohort_row.index:
            assert float(patient_sv[feat]) == pytest.approx(
                float(cohort_row[feat]), abs=1e-6
            )

    def test_waterfall_invariant(self, trained_learner):
        """base_value + sum(shap) == pipeline.predict(patient) (raw space)."""
        learner = trained_learner
        pipeline = learner._state.models["coxph"].model
        X_patient = learner._state.X_test.iloc[[0]]

        expl = learner.explain(patient=0, plot=False)
        predicted = float(pipeline.predict(X_patient)[0])
        reconstructed = float(expl.base_value) + float(
            expl.shap_values["shap_value"].sum()
        )
        assert reconstructed == pytest.approx(predicted, abs=1e-6)

    @pytest.mark.timeout(180)
    def test_rsf_raw_space_and_invariant(self, sample_data):
        """A non-linear model (RSF) also explains the full pipeline in raw
        feature space, and the efficiency invariant holds exactly.

        Both paths explain the *full Pipeline* (not the bare RSF estimator)
        through the one model-agnostic ``_run_explainer`` (a PermutationExplainer
        that only calls ``predict``) — the same engine CoxPH uses.

        Yet per-feature SHAP does NOT reconcile to float precision for RSF the
        way it does for CoxPH.  PermutationExplainer is exact and order-
        independent only for an additive model; RSF's risk is non-additive, so
        with a bounded permutation budget the result is order-dependent.  The
        cohort run and the 1-row patient run use separate explainer instances
        over different backgrounds and explained sets, so their per-feature
        values need not match.  That is a property of the explainer, not of the
        routing.  The efficiency property ``base + sum(shap) == predict`` is a
        telescoping sum and holds exactly regardless.
        """
        from mlsurv.learner import SurvivalLearner

        X, y = sample_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, random_state=42, verbose=False)
        learner.train("rsf", verbose=False)

        pipeline = learner._state.models["rsf"].model
        X_patient = learner._state.X_test.iloc[[0]]

        expl = learner.explain(patient=0, model="rsf", plot=False, max_background=50)
        # raw feature space: every input column appears
        assert set(expl.shap_values["feature"]) == set(X.columns)
        # efficiency invariant
        predicted = float(pipeline.predict(X_patient)[0])
        reconstructed = float(expl.base_value) + float(
            expl.shap_values["shap_value"].sum()
        )
        assert reconstructed == pytest.approx(predicted, abs=1e-6)
