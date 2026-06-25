"""Unit tests for PIMP importance p-values."""

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import MetricFailureWarning
from mlsurv.evaluation.importance import (
    _permute_y,
    _fit_null_importance,
    compute_pimp,
)
from mlsurv.utils import SurvivalArray


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestPermuteY:
    """Tests for _permute_y helper."""

    def test_preserves_length(self):
        y = SurvivalArray(np.array(
            [(True, 1.0), (False, 2.0), (True, 3.0)],
            dtype=[("event", bool), ("time", float)],
        ))
        rng = np.random.RandomState(42)
        y_perm = _permute_y(y, rng)
        assert len(y_perm) == len(y)

    def test_preserves_dtype(self):
        y = SurvivalArray(np.array(
            [(True, 1.0), (False, 2.0), (True, 3.0), (False, 4.0)],
            dtype=[("event", bool), ("time", float)],
        ))
        rng = np.random.RandomState(42)
        y_perm = _permute_y(y, rng)
        assert y_perm.dtype == y.dtype

    def test_preserves_values(self):
        """Same (event, time) pairs, just reordered."""
        y = SurvivalArray(np.array(
            [(True, 1.0), (False, 2.0), (True, 3.0), (False, 4.0)],
            dtype=[("event", bool), ("time", float)],
        ))
        rng = np.random.RandomState(42)
        y_perm = _permute_y(y, rng)
        # Same multiset of values
        orig_pairs = sorted(zip(y["event"], y["time"]))
        perm_pairs = sorted(zip(y_perm["event"], y_perm["time"]))
        assert orig_pairs == perm_pairs

    def test_deterministic_with_seed(self):
        y = SurvivalArray(np.array(
            [(True, 1.0), (False, 2.0), (True, 3.0), (False, 4.0), (True, 5.0)],
            dtype=[("event", bool), ("time", float)],
        ))
        perm1 = _permute_y(y, np.random.RandomState(7))
        perm2 = _permute_y(y, np.random.RandomState(7))
        np.testing.assert_array_equal(perm1, perm2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestComputePimpValidation:
    """Input validation for compute_pimp."""

    @pytest.fixture
    def dummy_data(self):
        """Minimal data for validation tests (not meant to be meaningful)."""
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        rng = np.random.RandomState(0)
        X = pd.DataFrame({"x1": rng.normal(0, 1, 30), "x2": rng.normal(0, 1, 30)})
        y = SurvivalArray(np.array(
            [(bool(rng.binomial(1, 0.5)), abs(rng.normal(5, 2)))
             for _ in range(30)],
            dtype=[("event", bool), ("time", float)],
        ))
        model = CoxPHSurvivalAnalysis(alpha=0.5)
        model.fit(X.iloc[:20], y[:20])
        return model, X.iloc[:20], y[:20], X.iloc[20:], y[20:]

    def test_bad_n_permutations(self, dummy_data):
        model, X_tr, y_tr, X_te, y_te = dummy_data
        with pytest.raises(ValueError, match="n_permutations must be >= 1"):
            compute_pimp(model, X_tr, y_tr, X_te, y_te, n_permutations=0,
                        show_progress=False)

    def test_bad_n_importance_repeats(self, dummy_data):
        model, X_tr, y_tr, X_te, y_te = dummy_data
        with pytest.raises(ValueError, match="n_importance_repeats must be >= 1"):
            compute_pimp(model, X_tr, y_tr, X_te, y_te, n_importance_repeats=0,
                        show_progress=False)

    def test_output_contract(self, dummy_data):
        """compute_pimp returns DataFrame with correct columns, types, sorting."""
        model, X_tr, y_tr, X_te, y_te = dummy_data
        result = compute_pimp(
            model, X_tr, y_tr, X_te, y_te,
            n_permutations=3, n_importance_repeats=2,
            show_progress=False,
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"feature", "observed_importance", "null_mean",
                        "null_std", "p_value", "p_adjusted"}
        assert set(result.columns) == expected_cols
        # p-values in (0, 1]
        assert (result["p_value"] > 0).all()
        assert (result["p_value"] <= 1).all()
        # p_adjusted in [0, 1] (can be exactly 1 after correction)
        assert (result["p_adjusted"] >= 0).all()
        assert (result["p_adjusted"] <= 1).all()
        # Sorted by p_value
        assert list(result["p_value"]) == sorted(result["p_value"])
        # null_mean and null_std are finite
        assert np.isfinite(result["null_mean"]).all()
        assert np.isfinite(result["null_std"]).all()

    def test_n_permutations_one(self, dummy_data):
        """n_permutations=1 produces valid output."""
        model, X_tr, y_tr, X_te, y_te = dummy_data
        result = compute_pimp(
            model, X_tr, y_tr, X_te, y_te,
            n_permutations=1, n_importance_repeats=2,
            show_progress=False, p_adjust="none",
        )
        assert len(result) == 2  # 2 features
        # With s=1, p can only be 1/2 or 2/2
        for p in result["p_value"]:
            assert p in (0.5, 1.0)

    def test_reproducibility(self, dummy_data):
        """Same random_state produces identical results."""
        model, X_tr, y_tr, X_te, y_te = dummy_data
        r1 = compute_pimp(model, X_tr, y_tr, X_te, y_te,
                         n_permutations=3, n_importance_repeats=2,
                         random_state=99, show_progress=False)
        r2 = compute_pimp(model, X_tr, y_tr, X_te, y_te,
                         n_permutations=3, n_importance_repeats=2,
                         random_state=99, show_progress=False)
        pd.testing.assert_frame_equal(r1, r2)

    def test_parallel_matches_sequential(self, dummy_data):
        """n_jobs=2 produces same results as n_jobs=1."""
        model, X_tr, y_tr, X_te, y_te = dummy_data
        seq = compute_pimp(model, X_tr, y_tr, X_te, y_te,
                          n_permutations=5, n_importance_repeats=2,
                          random_state=42, n_jobs=1, show_progress=False)
        par = compute_pimp(model, X_tr, y_tr, X_te, y_te,
                          n_permutations=5, n_importance_repeats=2,
                          random_state=42, n_jobs=2, show_progress=False)
        pd.testing.assert_frame_equal(seq, par)

    def test_uncloneable_model_raises_typeerror(self):
        """Uncloneable model raises TypeError with clear message."""
        class BadModel:
            pass

        rng = np.random.RandomState(0)
        X = pd.DataFrame({"x": rng.normal(0, 1, 10)})
        y = SurvivalArray(np.array(
            [(True, 1.0)] * 5 + [(False, 2.0)] * 5,
            dtype=[("event", bool), ("time", float)],
        ))
        with pytest.raises(TypeError, match="Cannot clone"):
            compute_pimp(BadModel(), X, y, X, y, n_permutations=2,
                        show_progress=False)

    def test_all_null_models_fail_raises_runtime_error(self, dummy_data):
        """RuntimeError when all null models fail to fit."""
        from unittest.mock import patch
        model, X_tr, y_tr, X_te, y_te = dummy_data
        with patch("mlsurv.evaluation.importance._fit_null_importance",
                   return_value=None):
            with pytest.raises(RuntimeError, match="All null models failed"):
                compute_pimp(model, X_tr, y_tr, X_te, y_te,
                            n_permutations=3, show_progress=False)

    def test_partial_null_failures_warn(self, dummy_data):
        """Warning when some null models fail (above minimum threshold)."""
        from unittest.mock import patch
        model, X_tr, y_tr, X_te, y_te = dummy_data
        n_features = X_te.shape[1]
        valid_result = np.zeros(n_features)

        # 11 succeed, 1 fails (above min threshold of 10)
        side_effects = [valid_result] * 5 + [None] + [valid_result] * 6
        with patch("mlsurv.evaluation.importance._fit_null_importance",
                   side_effect=side_effects):
            with pytest.warns(MetricFailureWarning, match="1/12 null models failed"):
                compute_pimp(model, X_tr, y_tr, X_te, y_te,
                            n_permutations=12, show_progress=False)


class TestFitNullImportance:
    """Tests for _fit_null_importance helper."""

    def test_returns_correct_shape(self):
        """Returns array of shape (n_features,)."""
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        rng = np.random.RandomState(0)
        X = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30),
                          "c": rng.normal(0, 1, 30)})
        y = SurvivalArray(np.array(
            [(bool(rng.binomial(1, 0.5)), abs(rng.normal(5, 2)))
             for _ in range(30)],
            dtype=[("event", bool), ("time", float)],
        ))
        model = CoxPHSurvivalAnalysis(alpha=0.5)
        model.fit(X.iloc[:20], y[:20])

        y_perm = _permute_y(y[:20], np.random.RandomState(7))
        result = _fit_null_importance(
            model, X.iloc[:20], y_perm, X.iloc[20:], y[20:],
            n_repeats=3, seed=42,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)  # 3 features

    def test_returns_none_on_fit_failure(self):
        """Returns None when the cloned model fails to fit."""
        from sklearn.base import BaseEstimator

        class FailingModel(BaseEstimator):
            def fit(self, X, y):
                raise ValueError("Intentional failure")
            def predict(self, X):
                return np.zeros(len(X))

        rng = np.random.RandomState(0)
        X = pd.DataFrame({"a": rng.normal(0, 1, 20)})
        y = SurvivalArray(np.array(
            [(True, 1.0)] * 10 + [(False, 2.0)] * 10,
            dtype=[("event", bool), ("time", float)],
        ))
        result = _fit_null_importance(FailingModel(), X, y, X, y, 3, 42)
        assert result is None


# ---------------------------------------------------------------------------
# Integration: learner.importance_test()
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLearnerImportanceTest:
    """Integration tests for SurvivalLearner.importance_test()."""

    @pytest.fixture
    def trained_learner(self, small_data):
        """Learner with a trained CoxPH model."""
        import sys
        import io

        from mlsurv import SurvivalLearner

        X, y = small_data

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            learner = SurvivalLearner(X, y)
            learner.setup(test_fraction=0.3, random_state=42)
            learner.train("coxph")
        finally:
            sys.stdout = old_stdout
        return learner

    def test_pimp_returns_dataframe(self, trained_learner):
        """PIMP method returns a DataFrame with expected columns."""
        import sys, io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = trained_learner.importance_test(
                method="pimp", n_permutations=5, n_importance_repeats=3,
                show_progress=False, verbose=False,
            )
        finally:
            sys.stdout = old_stdout

        assert isinstance(result, pd.DataFrame)
        expected_cols = {"feature", "observed_importance", "null_mean",
                        "null_std", "p_value", "p_adjusted"}
        assert expected_cols == set(result.columns)

    def test_result_property_populates(self, trained_learner):
        """importance_test() exposes its table via pimp_importance_test_results."""
        import sys, io

        # Before the call: empty dict (no singular or legacy property).
        assert trained_learner.pimp_importance_test_results == {}
        assert not hasattr(trained_learner, "importance_test_result")
        assert not hasattr(trained_learner, "importance_test_results")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = trained_learner.importance_test(
                models="coxph", method="pimp", n_permutations=5,
                n_importance_repeats=3, show_progress=False, verbose=False,
            )
        finally:
            sys.stdout = old_stdout

        # After the call: the per-model dict returns the stashed table.
        assert "coxph" in trained_learner.pimp_importance_test_results
        assert trained_learner.pimp_importance_test_results["coxph"] is result

    def test_results_dict_accumulates_per_model(self, trained_learner):
        """The per-model dict keeps every model across calls."""
        import sys, io

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            trained_learner.train("coxnet")
            first = trained_learner.importance_test(
                models="coxph", method="pimp", n_permutations=5,
                n_importance_repeats=3, show_progress=False, verbose=False,
            )
            second = trained_learner.importance_test(
                models="coxnet", method="pimp", n_permutations=5,
                n_importance_repeats=3, show_progress=False, verbose=False,
            )
        finally:
            sys.stdout = old_stdout

        # Both models accumulate in the dict (the second call must not clobber
        # the first).
        results = trained_learner.pimp_importance_test_results
        assert set(results) == {"coxph", "coxnet"}
        assert results["coxph"] is first
        assert results["coxnet"] is second

    def test_bootstrap_method_delegates(self, trained_learner):
        """Bootstrap method delegates to importance_summary() and stores."""
        import sys, io

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            trained_learner.bootstrap(
                "coxph", n_bootstrap=2,
                include_importance=True, random_state=42,
                show_progress=False, verbose=False,
            )
            result = trained_learner.importance_test(
                method="bootstrap", verbose=False,
            )
        finally:
            sys.stdout = old_stdout

        assert isinstance(result, pd.DataFrame)
        assert "p_value" in result.columns
        assert "p_adjusted" in result.columns
        # model=None with a single bootstrapped model: the storage key
        # derives from the unique model_type, and provenance is stamped.
        stored = trained_learner.bootstrap_importance_test_results
        assert set(stored) == {"coxph"}
        assert stored["coxph"] is result
        assert result.attrs["p_adjust"] == "bh"
        assert result.attrs["model_name"] == "coxph"
        # The PIMP dict stays untouched by the bootstrap method.
        assert trained_learner.pimp_importance_test_results == {}

    def test_bootstrap_method_no_data_raises(self, trained_learner):
        """Bootstrap method raises ValueError without bootstrap data."""
        with pytest.raises(ValueError, match="No bootstrap importance data"):
            trained_learner.importance_test(method="bootstrap", verbose=False)

    def test_invalid_method_raises(self, trained_learner):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            trained_learner.importance_test(method="invalid", verbose=False)

    def test_pimp_attrs_stamped(self, trained_learner):
        """Every result-affecting parameter is recorded in result.attrs."""
        result = trained_learner.importance_test(
            models="coxph", method="pimp", n_permutations=3,
            n_importance_repeats=2, p_adjust="holm", random_state=7,
            show_progress=False, verbose=False,
        )
        assert result.attrs["n_permutations"] == 3
        assert result.attrs["n_importance_repeats"] == 2
        assert result.attrs["random_state"] == 7
        assert result.attrs["p_adjust"] == "holm"
        assert result.attrs["model_name"] == "coxph"

    def test_pimp_attrs_default_random_state(self, trained_learner):
        """random_state=None stamps the learner's resolved default."""
        result = trained_learner.importance_test(
            models="coxph", method="pimp", n_permutations=3,
            n_importance_repeats=2, show_progress=False, verbose=False,
        )
        assert result.attrs["random_state"] == trained_learner.random_state

    def test_same_model_overwrite_preserves_others(self, trained_learner):
        """Re-running a model replaces only its entry; others survive."""
        from mlsurv.results.containers import ModelResult

        other = pd.DataFrame({"feature": ["x"], "p_value": [0.5]})
        other_result = ModelResult(
            name="other", model=None, cv_metric_scores=np.array([0.7]),
            cv_metric_mean=0.7, cv_metric_std=0.0,
        )
        other_result.pimp_importance_test = other
        trained_learner._state.models["other"] = other_result

        first = trained_learner.importance_test(
            models="coxph", method="pimp", n_permutations=3,
            n_importance_repeats=2, show_progress=False, verbose=False,
        )
        second = trained_learner.importance_test(
            models="coxph", method="pimp", n_permutations=3,
            n_importance_repeats=2, show_progress=False, verbose=False,
        )

        results = trained_learner.pimp_importance_test_results
        assert results["coxph"] is second
        assert first is not second
        assert results["other"] is other

    def test_foreign_model_result_name_collision_warns(self, trained_learner):
        """A ModelResult whose name matches a registered model but wraps a
        different pipeline object warns on store."""
        import dataclasses

        from sklearn.base import clone

        from mlsurv._warnings import SuggestionWarning

        # A ModelResult named 'coxph' but carrying a foreign (re-fit) pipeline:
        # orig.name resolves to the registered key, yet registered.model is a
        # different object, so the misattribution warning must fire.
        registered = trained_learner._state.models["coxph"]
        foreign_pipe = clone(registered.model)
        foreign_pipe.fit(trained_learner.X_train, trained_learner.y_train)
        foreign_result = dataclasses.replace(registered, model=foreign_pipe)

        with pytest.warns(SuggestionWarning, match="different pipeline object"):
            result = trained_learner.importance_test(
                models=foreign_result, method="pimp", n_permutations=3,
                n_importance_repeats=2, show_progress=False, verbose=False,
            )
        assert trained_learner.pimp_importance_test_results["coxph"] is result

    def test_registered_model_no_collision_warning(self, trained_learner):
        """Naming the registered model itself must not trip the warning."""
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            trained_learner.importance_test(
                models="coxph", method="pimp", n_permutations=3,
                n_importance_repeats=2, show_progress=False, verbose=False,
            )
        collision = [w for w in rec
                     if "different pipeline object" in str(w.message)]
        assert not collision

    def test_raw_pipeline_rejected(self, trained_learner):
        """PIMP on a raw Pipeline (not trained through the learner) raises
        ValueError and stores nothing."""
        from sklearn.pipeline import Pipeline
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        pipeline = Pipeline([("model", CoxPHSurvivalAnalysis())])

        with pytest.raises(ValueError, match="unregistered"):
            trained_learner.importance_test(
                models=pipeline, method="pimp", n_permutations=3,
                n_importance_repeats=2, show_progress=False, verbose=False,
            )
        assert trained_learner.pimp_importance_test_results == {}

    def test_bare_estimator_rejected(self, trained_learner):
        """PIMP on a bare estimator raises ValueError and stores nothing."""
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        with pytest.raises(ValueError, match="compute_pimp"):
            trained_learner.importance_test(
                models=CoxPHSurvivalAnalysis(), method="pimp", n_permutations=3,
                n_importance_repeats=2, show_progress=False, verbose=False,
            )
        assert trained_learner.pimp_importance_test_results == {}

    def test_dict_unregistered_key_rejected(self, trained_learner):
        """A dict naming an unregistered model raises ValueError (not KeyError)
        before storing — the store rides on a registered ModelResult."""
        registered = trained_learner._state.models["coxph"]
        with pytest.raises(ValueError, match="not a registered model"):
            trained_learner.importance_test(
                models={"not_registered": registered.model}, method="pimp",
                n_permutations=3, n_importance_repeats=2,
                show_progress=False, verbose=False,
            )
        assert trained_learner.pimp_importance_test_results == {}

    def test_bootstrap_explicit_model_storage(self, trained_learner):
        """method='bootstrap' with an explicit model stores under its name."""
        import sys, io

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            trained_learner.bootstrap(
                "coxph", n_bootstrap=2,
                include_importance=True, random_state=42,
                show_progress=False, verbose=False,
            )
            result = trained_learner.importance_test(
                method="bootstrap", models="coxph", p_adjust="holm",
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout

        stored = trained_learner.bootstrap_importance_test_results
        assert stored["coxph"] is result
        assert result.attrs["model_name"] == "coxph"
        assert result.attrs["p_adjust"] == "holm"
