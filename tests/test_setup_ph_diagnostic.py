"""Tests for PH diagnostic and model recommendations during setup()."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning
from mlsurv.guidance import RecommendedModels, recommend_models


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_constant_hazard_data(n=200, p=5, seed=42):
    """Synthetic data where PH assumption holds (exponential, constant HR)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)),
                     columns=[f"x{i}" for i in range(p)])
    # Constant log-HR -> PH satisfied
    lp = X["x0"] * 0.5 + X["x1"] * -0.3
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) < 0.7  # ~70% event rate
    y = np.array([(bool(e), t) for e, t in zip(event, time)],
                 dtype=[("event", bool), ("time", float)])
    return X, y


def _make_ph_violating_data(n=300, seed=42):
    """Synthetic data with strong time-varying coefficient -> PH violation."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x_const": rng.standard_normal(n),
        "x_tv": rng.standard_normal(n),
        "x_noise1": rng.standard_normal(n),
        "x_noise2": rng.standard_normal(n),
    })
    # Time-varying effect: x_tv effect flips sign at median time
    base_time = rng.exponential(scale=2.0, size=n)
    # Introduce strong time-dependence by making hazard ratio change
    # quadratically with time for x_tv
    lp = X["x_const"] * 0.5
    # Create PH violation: event probability depends on time * x_tv interaction
    time = base_time * np.exp(-lp)
    event = np.ones(n, dtype=bool)
    # Censor ~30% at random
    censor_mask = rng.random(n) < 0.3
    event[censor_mask] = False
    y = np.array([(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
                 dtype=[("event", bool), ("time", float)])
    return X, y


@pytest.fixture
def constant_hazard_data():
    return _make_constant_hazard_data()


@pytest.fixture
def ph_violating_data():
    return _make_ph_violating_data()


# ---------------------------------------------------------------------------
# Tests: PH test during setup
# ---------------------------------------------------------------------------

class TestSetupPHDiagnostic:
    """Tests for ph_test parameter in setup()."""

    def test_setup_runs_ph_test_by_default(self, constant_hazard_data):
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner._state.ph_test_result is not None
        assert isinstance(learner._state.ph_test_result, pd.DataFrame)
        assert not learner._state.ph_test_result.empty

    def test_setup_ph_test_false_skips(self, constant_hazard_data):
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(ph_test=False, verbose=False)
        assert learner._state.ph_test_result is None

    def test_setup_stores_model_recommendations(self, constant_hazard_data):
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner._state.model_recommendations is not None
        assert isinstance(learner._state.model_recommendations,
                          RecommendedModels)

    def test_setup_recommendations_even_without_ph(self, constant_hazard_data):
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(ph_test=False, verbose=False)
        # Recommendations always run regardless of ph_test
        assert learner._state.model_recommendations is not None
        assert isinstance(learner._state.model_recommendations,
                          RecommendedModels)
        # No PH violations recorded when ph_test=False
        assert learner._state.model_recommendations.ph_violations == ()

    def test_setup_verbose_false_suppresses_print(self, constant_hazard_data,
                                                  capsys):
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        captured = capsys.readouterr()
        # No PH diagnostic or recommendation table in output
        assert "PH Diagnostic" not in captured.out
        assert "Model Recommendations" not in captured.out

    def test_setup_verbose_true_prints_full_table(self, constant_hazard_data,
                                                  capsys):
        """setup(verbose=True) must still print the full recommendation
        table (not the new short repr). Guards against regression now that
        ``__repr__`` returns a one-liner — the setup path goes through
        ``print(rec)`` → ``__str__``, which must remain the full table.
        """
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=True)
        captured = capsys.readouterr()
        assert "Model Recommendations" in captured.out
        # The short repr must NOT appear in setup output.
        assert "<RecommendedModels:" not in captured.out

    def test_re_setup_clears_diagnostics(self, constant_hazard_data):
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner._state.ph_test_result is not None
        assert learner._state.model_recommendations is not None

        # Re-setup with ph_test=False clears PH result
        learner.setup(ph_test=False, verbose=False)
        assert learner._state.ph_test_result is None
        # But recommendations are recomputed
        assert learner._state.model_recommendations is not None

    def test_ph_test_standalone_still_works(self, constant_hazard_data):
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        setup_result = learner._state.ph_test_result

        # Standalone call overwrites the result
        standalone_result = learner.ph_test(time_transform='km',
                                            verbose=False)
        assert learner._state.ph_test_result is standalone_result
        # Different time_transform may give different p-values
        assert isinstance(standalone_result, pd.DataFrame)
        assert isinstance(setup_result, pd.DataFrame)
        assert setup_result is not standalone_result

    def test_setup_ph_result_has_correct_columns(self, constant_hazard_data):
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        df = learner._state.ph_test_result
        expected = {'feature', 'test_statistic', 'p_value', 'p_adjusted',
                    'significant'}
        assert set(df.columns) == expected

    def test_setup_ph_result_has_global_row(self, constant_hazard_data):
        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        df = learner._state.ph_test_result
        assert "GLOBAL" in df['feature'].values


# ---------------------------------------------------------------------------
# Tests: PH violation warning emission
# ---------------------------------------------------------------------------

class TestPHViolationWarning:
    """Tests for MethodologicalWarning when PH violations are detected."""

    def test_setup_ph_warning_emitted_on_violation(self):
        """setup() emits a MethodologicalWarning AND surfaces the
        violations in the recommendation table."""
        import pandas as pd
        from unittest.mock import patch

        violation_df = pd.DataFrame({
            'feature': ['x0', 'x1', 'GLOBAL'],
            'test_statistic': [10.0, 8.0, 18.0],
            'p_value': [0.001, 0.005, 0.0001],
            'p_adjusted': [0.002, 0.01, float('nan')],
            'significant': [True, True, True],
        })

        X, y = _make_constant_hazard_data()
        learner = SurvivalLearner(X, y)
        with patch('mlsurv.evaluation.metrics.ph_test',
                   return_value=violation_df):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                learner.setup(verbose=False)

        ph_warnings = [x for x in w
                       if issubclass(x.category, MethodologicalWarning)
                       and "Proportional hazards" in str(x.message)]
        assert len(ph_warnings) == 1
        msg = str(ph_warnings[0].message)
        assert "x0" in msg and "x1" in msg
        # The warning points at the no-recompute read path.
        assert "learner.ph_test_result" in msg
        # Violations also surface in the recommendation.
        rec = learner._state.model_recommendations
        assert "x0" in rec.ph_violations
        assert "x1" in rec.ph_violations

    def test_setup_ph_no_warning_when_not_significant(self,
                                                       constant_hazard_data):
        """When PH global test is not significant, no PH warning fires."""
        X, y = constant_hazard_data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(verbose=False)

        ph_warnings = [x for x in w
                       if issubclass(x.category, MethodologicalWarning)
                       and "Proportional hazards" in str(x.message)]
        assert len(ph_warnings) == 0

    def test_ph_violations_flow_to_recommendations(self):
        """PH violations propagate from setup to model_recommendations.ph_violations."""
        import pandas as pd
        from unittest.mock import patch

        violation_df = pd.DataFrame({
            'feature': ['age', 'bmi', 'GLOBAL'],
            'test_statistic': [12.0, 9.0, 21.0],
            'p_value': [0.0005, 0.003, 0.00001],
            'p_adjusted': [0.001, 0.006, float('nan')],
            'significant': [True, True, True],
        })

        X, y = _make_constant_hazard_data()
        learner = SurvivalLearner(X, y)
        with patch('mlsurv.evaluation.metrics.ph_test',
                   return_value=violation_df):
            learner.setup(verbose=False)

        rec = learner._state.model_recommendations
        assert "age" in rec.ph_violations
        assert "bmi" in rec.ph_violations
        # Reason for PH-assuming models should mention violations.
        assert "PH violation" in rec.details["coxph"]


# ---------------------------------------------------------------------------
# Tests: recommend_models with ph_violations
# ---------------------------------------------------------------------------

class TestRecommendModelsPHViolations:
    """ph_violations parameter in recommend_models()."""

    def test_no_ph_violations_no_violations_field(self):
        rec = recommend_models(
            n_samples=500, n_events=300, n_features=10,
            ph_violations=[], verbose=False,
        )
        assert rec.ph_violations == ()

    def test_ph_violations_stored_on_result(self):
        rec = recommend_models(
            n_samples=500, n_events=300, n_features=10,
            ph_violations=["age", "tumor_size"], verbose=False,
        )
        assert rec.ph_violations == ("age", "tumor_size")

    def test_ph_violations_in_str(self):
        rec = recommend_models(
            n_samples=500, n_events=300, n_features=10,
            ph_violations=["age"], verbose=False,
        )
        text = str(rec)
        assert "Non-PH features (1): age" in text

    def test_many_ph_violations_truncated(self):
        features = [f"feat_{i}" for i in range(10)]
        rec = recommend_models(
            n_samples=500, n_events=300, n_features=10,
            ph_violations=features, verbose=False,
        )
        text = str(rec)
        assert "feat_0" in text
        assert "feat_4" in text
        assert "(and 5 more)" in text


# ---------------------------------------------------------------------------
# Tests: PH diagnostic follow-ups
# ---------------------------------------------------------------------------

class TestDefaultCoxModelSelection:
    """Item 1: Explicit coxph preference instead of ordering dependency."""

    def test_default_cox_model_constant_exists(self):
        from mlsurv.viz._data import _DEFAULT_COX_MODEL
        assert _DEFAULT_COX_MODEL == 'coxph'

    def test_cox_models_dict_returns_coxph_when_present(self):
        """When coxph is trained, it should be selected even if dict order differs."""
        from mlsurv.viz._data import _DEFAULT_COX_MODEL
        # Simulate cox_models dict where coxnet comes first
        cox_models = {"coxnet": "model_coxnet", "coxph": "model_coxph"}
        selected = cox_models.get(_DEFAULT_COX_MODEL,
                                  next(iter(cox_models.values())))
        assert selected == "model_coxph"

    def test_fallback_when_coxph_not_available(self):
        """When only coxnet is trained, fallback to first available."""
        from mlsurv.viz._data import _DEFAULT_COX_MODEL
        cox_models = {"coxnet": "model_coxnet"}
        selected = cox_models.get(_DEFAULT_COX_MODEL,
                                  next(iter(cox_models.values())))
        assert selected == "model_coxnet"


class TestPHLimitationsFlagGating:
    """PH limitations flag fires on violations regardless of trained models."""

    def test_ph_clean_no_cox_returns_none(self):
        """When PH is clean and no Cox models, no flag fires (positive confirmation removed)."""
        from mlsurv.reports.sections.limitations import LimitationsSection

        clean_df = pd.DataFrame({
            'feature': ['x0', 'x1', 'GLOBAL'],
            'test_statistic': [0.5, 0.3, 0.8],
            'p_value': [0.5, 0.6, 0.4],
            'p_adjusted': [0.6, 0.7, float('nan')],
            'significant': [False, False, False],
        })

        X, y = _make_constant_hazard_data()
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner._state.ph_test_result = clean_df

        # Create section with no Cox models (empty models dict)
        section = LimitationsSection.__new__(LimitationsSection)
        section.learner = learner
        section.name = "Limitations"

        result = section._check_ph_violations()
        assert result is None

    def test_ph_violations_with_cox_returns_warning(self):
        """When PH is violated and Cox models trained, return warning."""
        from mlsurv.reports.sections.limitations import LimitationsSection

        violation_df = pd.DataFrame({
            'feature': ['x0', 'GLOBAL'],
            'test_statistic': [10.0, 10.0],
            'p_value': [0.001, 0.001],
            'p_adjusted': [0.002, float('nan')],
            'significant': [True, True],
        })

        X, y = _make_constant_hazard_data()
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner._state.ph_test_result = violation_df
        # Simulate a trained coxph model
        learner._state.models = {"coxph": "dummy"}

        section = LimitationsSection.__new__(LimitationsSection)
        section.learner = learner
        section.name = "Limitations"

        result = section._check_ph_violations()
        assert result is not None
        assert "info" not in result or result.get("info") is False
        assert "violate" in result["detected"]

    def test_no_ph_result_no_cox_returns_none(self):
        """When no PH result and no Cox models, return None."""
        from mlsurv.reports.sections.limitations import LimitationsSection

        X, y = _make_constant_hazard_data()
        learner = SurvivalLearner(X, y)
        learner.setup(ph_test=False, verbose=False)

        section = LimitationsSection.__new__(LimitationsSection)
        section.learner = learner
        section.name = "Limitations"

        result = section._check_ph_violations()
        assert result is None

    def test_ph_clean_with_cox_returns_none(self):
        """When PH is clean and Cox models trained, return None (no flag)."""
        from mlsurv.reports.sections.limitations import LimitationsSection

        clean_df = pd.DataFrame({
            'feature': ['x0', 'GLOBAL'],
            'test_statistic': [0.5, 0.8],
            'p_value': [0.5, 0.4],
            'p_adjusted': [0.6, float('nan')],
            'significant': [False, False],
        })

        X, y = _make_constant_hazard_data()
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner._state.ph_test_result = clean_df
        learner._state.models = {"coxph": "dummy"}

        section = LimitationsSection.__new__(LimitationsSection)
        section.learner = learner
        section.name = "Limitations"

        result = section._check_ph_violations()
        assert result is None

    def test_ph_violations_no_cox_still_warns(self):
        """PH violations return warning even when no Cox models trained."""
        from mlsurv.reports.sections.limitations import LimitationsSection

        violation_df = pd.DataFrame({
            'feature': ['x0', 'GLOBAL'],
            'test_statistic': [10.0, 10.0],
            'p_value': [0.001, 0.001],
            'p_adjusted': [0.002, float('nan')],
            'significant': [True, True],
        })

        X, y = _make_constant_hazard_data()
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner._state.ph_test_result = violation_df
        # No Cox models

        section = LimitationsSection.__new__(LimitationsSection)
        section.learner = learner
        section.name = "Limitations"

        result = section._check_ph_violations()
        assert result is not None
        assert "info" not in result
        assert "violate" in result["detected"]

    def test_format_flag_warning_uses_callout_warning(self):
        """Warning flag should still render with mlsurv-callout-warning."""
        from mlsurv.reports.sections.limitations import LimitationsSection

        result = {
            "detected": "Violations found.",
            "why": "PH violated.",
            "action": "Use non-PH models.",
        }
        lines = LimitationsSection._format_flag("PH Test", result)
        html = "\n".join(lines)
        assert 'mlsurv-callout-warning' in html
        assert '&#9888;' in html  # warning icon

    def test_clean_ph_no_cox_produces_no_triggered_flag(self):
        """Clean PH + no Cox models → ph_violations flag does not appear."""
        from mlsurv.reports.sections.limitations import LimitationsSection

        clean_df = pd.DataFrame({
            'feature': ['x0', 'GLOBAL'],
            'test_statistic': [0.5, 0.8],
            'p_value': [0.5, 0.4],
            'p_adjusted': [0.6, float('nan')],
            'significant': [False, False],
        })

        X, y = _make_constant_hazard_data()
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner._state.ph_test_result = clean_df

        section = LimitationsSection(learner)
        triggered = section.get_triggered_flags()
        ph_ids = [fid for fid, _ in triggered if fid == "ph_violations"]
        assert len(ph_ids) == 0


class TestPHTestResilience:
    """Item 3: Standalone ph_test() handles failures gracefully."""

    def test_ph_test_survives_imputation_failure(self, constant_hazard_data):
        """ph_test() returns empty DataFrame when _get_imputed_X raises."""
        from unittest.mock import patch

        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with patch.object(
            type(learner), '_get_imputed_X',
            side_effect=ValueError("imputation failed"),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = learner.ph_test(verbose=False)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        meth_warnings = [x for x in w
                         if issubclass(x.category, MethodologicalWarning)]
        # Exactly 1 warning — the "failed" warning, not the "no results" one
        assert len(meth_warnings) == 1
        assert "failed" in str(meth_warnings[0].message)

    def test_ph_test_reraises_memory_error(self, constant_hazard_data):
        """MemoryError must propagate."""
        from unittest.mock import patch

        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with patch.object(
            type(learner), '_get_imputed_X',
            side_effect=MemoryError("out of memory"),
        ):
            with pytest.raises(MemoryError):
                learner.ph_test(verbose=False)

    def test_ph_test_reraises_recursion_error(self, constant_hazard_data):
        """RecursionError must propagate."""
        from unittest.mock import patch

        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with patch.object(
            type(learner), '_get_imputed_X',
            side_effect=RecursionError("maximum recursion depth exceeded"),
        ):
            with pytest.raises(RecursionError):
                learner.ph_test(verbose=False)

    def test_ph_test_survives_ph_test_internal_failure(self, constant_hazard_data):
        """ph_test() handles failure from _ph_test itself (not just imputation)."""
        from unittest.mock import patch

        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        with patch(
            'mlsurv.evaluation.metrics.ph_test',
            side_effect=RuntimeError("singular matrix"),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = learner.ph_test(verbose=False)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        meth_warnings = [x for x in w
                         if issubclass(x.category, MethodologicalWarning)]
        assert len(meth_warnings) == 1
        assert "singular matrix" in str(meth_warnings[0].message)

    def test_ph_test_stores_empty_on_failure(self, constant_hazard_data):
        """Failed ph_test() still stores result on state."""
        from unittest.mock import patch

        X, y = constant_hazard_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        # Overwrite with a non-empty result first
        learner._state.ph_test_result = "non-empty"

        with patch.object(
            type(learner), '_get_imputed_X',
            side_effect=RuntimeError("unexpected"),
        ):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = learner.ph_test(verbose=False)

        assert result.empty
        assert learner._state.ph_test_result is result


class TestEmptyColumnDoesNotAbortDiagnostics:
    """A 100%-missing feature used to take the whole diagnostic block down.

    The imputer drops such a column, so the imputed matrix was rebuilt with
    one name too many and pandas raised "Shape of passed values"; setup()
    logged "skipping PH and linearity diagnostics" and left both results
    unset.
    """

    @staticmethod
    def _data_with_empty_column():
        X, y = _make_constant_hazard_data(n=200, p=5)
        X = X.copy()
        X.loc[X.index[:20], "x0"] = np.nan  # ordinary missingness to impute
        X["dz_stage"] = np.nan              # no observed value at all
        return X, y

    def test_ph_and_linearity_survive(self):
        X, y = self._data_with_empty_column()
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.setup(random_state=42, verbose=False)

        ph = learner.ph_test_result
        lin = learner.linearity_test_result
        assert ph is not None and not ph.empty
        assert lin is not None and not lin.empty

    def test_empty_feature_absent_from_results(self):
        X, y = self._data_with_empty_column()
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.setup(random_state=42, verbose=False)

        assert "dz_stage" not in set(learner.ph_test_result["feature"])
        assert "dz_stage" not in set(learner.linearity_test_result["feature"])

    def test_setup_names_the_empty_feature(self):
        """Two warnings, saying two different things: the data has an empty
        feature, and the diagnostics were computed without it."""
        X, y = self._data_with_empty_column()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurvivalLearner(X, y).setup(random_state=42, verbose=False)

        named = [str(x.message) for x in w
                 if issubclass(x.category, MethodologicalWarning)
                 and "no observed values" in str(x.message)]
        assert len(named) == 2
        assert all("dz_stage" in m for m in named)
        assert any("drops them before model fitting" in m for m in named)
        assert any("covers 5 features, not 6" in m for m in named)

    def test_matches_a_run_without_the_empty_column(self):
        """The surviving features must be diagnosed exactly as before."""
        X, y = self._data_with_empty_column()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with_empty = SurvivalLearner(X, y)
            with_empty.setup(random_state=42, verbose=False)
            without = SurvivalLearner(X.drop(columns=["dz_stage"]), y)
            without.setup(random_state=42, verbose=False)

        pd.testing.assert_frame_equal(
            with_empty.ph_test_result, without.ph_test_result,
        )
