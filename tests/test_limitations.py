"""Tests for the Limitations & Caveats report section."""

import numpy as np
import pandas as pd
import pytest

from mlsurv.reports.sections.limitations import LimitationsSection, _is_valid


# ---------------------------------------------------------------------------
# _is_valid helper
# ---------------------------------------------------------------------------

class TestIsValid:
    """Unit tests for the _is_valid helper."""

    def test_finite_number(self):
        assert _is_valid(0.65)

    def test_zero(self):
        assert _is_valid(0.0)

    def test_none(self):
        assert not _is_valid(None)

    def test_nan(self):
        assert not _is_valid(np.nan)

    def test_positive_inf(self):
        assert not _is_valid(np.inf)

    def test_negative_inf(self):
        assert not _is_valid(-np.inf)


# ---------------------------------------------------------------------------
# Minimal learner stub
# ---------------------------------------------------------------------------

class _FakeEvaluation:
    """Stub for EvaluationResult with configurable metrics."""

    def __init__(self, c_index_test=0.65, cal_slope_test=1.0, gnd_test=0.5,
                 integrated_brier_test=None):
        self.c_index_test = c_index_test
        self.cal_slope_test = cal_slope_test
        self.gnd_test = gnd_test
        # Build concordance DataFrame for metric-aware flag checks
        data = {
            "c_index_test": [c_index_test],
            "cal_slope_test": [cal_slope_test],
            "gnd_test": [gnd_test],
        }
        if integrated_brier_test is not None:
            data["integrated_brier_test"] = [integrated_brier_test]
        self.concordance = pd.DataFrame(data)


class _FakeModelResult:
    """Stub for ModelResult."""

    def __init__(
        self,
        name="coxph",
        cv_metric_mean=0.65,
        evaluation=None,
        is_tuned=False,
        calibrator=None,
        precalibration_slope=None,
    ):
        self.name = name
        self.cv_metric_mean = cv_metric_mean
        self.evaluation = evaluation or _FakeEvaluation()
        self.is_tuned = is_tuned
        self.model = None
        # Internal S(t|x) calibrator (train(calibrate=...)); None = uncalibrated.
        self.calibrator = calibrator
        self.precalibration_slope = precalibration_slope


class _FakeState:
    """Minimal _state stub."""

    def __init__(self, *, validation_results=None, random_state=42):
        self.validation_results = validation_results or {}
        self.random_state = random_state
        self.models = {}
        self.imputer = None
        self.ph_test_result = None
        self.linearity_test_result = None
        self.model_recommendations = None
        self.subpopulation_results = {}
        self.benchmark_result = None


class _FakeLearner:
    """Minimal learner stub for LimitationsSection."""

    def __init__(
        self,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        models=None,
        validation_results=None,
        scoring="harrell",
    ):
        n = 300
        rng = np.random.RandomState(42)

        if X_train is None:
            X_train = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        if y_train is None:
            y_train = pd.DataFrame({"time": rng.exponential(10, n), "event": rng.binomial(1, 0.6, n)})
        if X_test is None:
            X_test = pd.DataFrame(rng.randn(n // 3, 5), columns=[f"f{i}" for i in range(5)])
        if y_test is None:
            y_test = pd.DataFrame(
                {"time": rng.exponential(10, n // 3), "event": rng.binomial(1, 0.6, n // 3)}
            )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scoring = scoring

        self._state = _FakeState(validation_results=validation_results)

        # Build models dict (acts like learner.models)
        self._models = models or {
            "coxph": _FakeModelResult("coxph"),
            "rsf": _FakeModelResult("rsf"),
        }

    def _get_imputed_X(self, X, *, raise_on_error=False):
        """Impute NaN using median fallback (mirrors SurvivalLearner._get_imputed_X)."""
        if X is None or not X.isna().any().any():
            return X
        from sklearn.impute import SimpleImputer
        fitted = SimpleImputer(strategy="median").fit(X)
        return pd.DataFrame(fitted.transform(X), columns=X.columns, index=X.index)

    @property
    def models(self):
        return self._models

    @property
    def validation_results(self):
        # Mirror the public SurvivalLearner.validation_results accessor:
        # report sections read the public path, not _state.
        return self._state.validation_results

    # Mirror the public diagnostic-result accessors: tests write to
    # ``learner._state.<field>``, sections read ``learner.<field>``.
    @property
    def ph_test_result(self):
        return self._state.ph_test_result

    @property
    def linearity_test_result(self):
        return self._state.linearity_test_result

    @property
    def model_recommendations(self):
        return self._state.model_recommendations

    @property
    def subpopulation_results(self):
        return self._state.subpopulation_results

    @property
    def benchmark_result(self):
        return self._state.benchmark_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLimitationsAvailability:
    def test_is_available_always_true(self):
        section = LimitationsSection(_FakeLearner())
        assert section.is_available() is True


class TestNoFlagsTriggered:
    def test_no_flags_produces_graceful_output(self):
        """Clean data with good metrics should produce no flags."""
        good_eval = _FakeEvaluation(c_index_test=0.75, cal_slope_test=1.0, gnd_test=0.5)
        models = {
            "coxph": _FakeModelResult("coxph", cv_metric_mean=0.75, evaluation=good_eval),
            "rsf": _FakeModelResult("rsf", cv_metric_mean=0.74, evaluation=good_eval),
        }
        learner = _FakeLearner(models=models, validation_results={"cohort_a": True})
        section = LimitationsSection(learner)
        result = section.generate()

        assert "No computable limitations were detected" in result["markdown"]
        assert result["figures"] == []

        flags_df = section.data["flags"]
        assert not flags_df["triggered"].any()


class TestSmallSampleFlag:
    def test_triggers_below_200(self):
        rng = np.random.RandomState(42)
        X_small = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y_small = pd.DataFrame({"time": rng.exponential(10, 100), "event": rng.binomial(1, 0.6, 100)})
        X_test = pd.DataFrame(rng.randn(30, 5), columns=[f"f{i}" for i in range(5)])
        y_test = pd.DataFrame({"time": rng.exponential(10, 30), "event": rng.binomial(1, 0.6, 30)})

        learner = _FakeLearner(X_train=X_small, y_train=y_small, X_test=X_test, y_test=y_test)
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Small Sample Size" in result["markdown"]
        assert "n_total=130" in section.data["flags"].query("flag_id == 'small_sample'")["detail"].iloc[0]


class TestHighCensoringFlag:
    def test_triggers_above_80_pct(self):
        rng = np.random.RandomState(42)
        n = 300
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        X_test = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
        # Very high censoring: only 10% events in both splits
        y = pd.DataFrame({"time": rng.exponential(10, n), "event": rng.binomial(1, 0.10, n)})
        y_test = pd.DataFrame({"time": rng.exponential(10, 100), "event": rng.binomial(1, 0.10, 100)})

        learner = _FakeLearner(X_train=X, y_train=y, X_test=X_test, y_test=y_test)
        section = LimitationsSection(learner)
        result = section.generate()

        assert "High Censoring Rate" in result["markdown"]


class TestMissingDataActionText:
    def test_action_does_not_suggest_multiple_imputation(self):
        """Action text should not suggest multiple imputation."""
        rng = np.random.RandomState(42)
        n = 300
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        X.iloc[:, 0] = np.nan  # 100% missing on f0
        y = pd.DataFrame({"time": rng.exponential(10, n), "event": rng.binomial(1, 0.6, n)})
        X_test = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y_test = pd.DataFrame({"time": rng.exponential(10, 100), "event": rng.binomial(1, 0.6, 100)})

        learner = _FakeLearner(X_train=X, y_train=y, X_test=X_test, y_test=y_test)
        section = LimitationsSection(learner)
        result = section._check_missing_data()

        assert result is not None
        action = result["action"].lower()
        # Should not suggest multiple imputation as a direct mlsurv action
        assert "consider multiple imputation" not in action
        # Multiple imputation can be mentioned as an external option
        assert "sensitivity analyses" in action
        assert ">50% missing" in action


class TestNoValidationFlag:
    def test_triggers_when_empty(self):
        learner = _FakeLearner(validation_results={})
        section = LimitationsSection(learner)
        result = section.generate()

        assert "No External Validation" in result["markdown"]

    def test_no_trigger_when_present(self):
        learner = _FakeLearner(validation_results={"cohort_a": True})
        section = LimitationsSection(learner)
        result = section.generate()

        assert "No External Validation" not in result["markdown"]


class TestFlagsDataFrameStructure:
    def test_has_correct_columns(self):
        learner = _FakeLearner()
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        assert set(flags_df.columns) == {"flag_id", "flag_name", "triggered", "detail"}
        assert len(flags_df) == 14  # 13 original + ipcw_support

    def test_triggered_is_bool(self):
        learner = _FakeLearner()
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        assert flags_df["triggered"].dtype == bool


class TestErrorResilience:
    def test_broken_flag_does_not_crash(self, monkeypatch):
        """If one flag checker raises, others still run."""
        learner = _FakeLearner()
        section = LimitationsSection(learner)

        # Break the small_sample checker
        def _broken_checker():
            raise RuntimeError("boom")

        monkeypatch.setattr(section, "_check_small_sample", _broken_checker)
        result = section.generate()

        # Section still generates
        assert "Limitations & Caveats" in section.title
        assert isinstance(result["markdown"], str)

        # The broken flag is marked as not triggered
        flags_df = section.data["flags"]
        small_row = flags_df[flags_df["flag_id"] == "small_sample"]
        assert not small_row["triggered"].iloc[0]

        # Other flags still evaluated
        assert len(flags_df) == 14


class TestLowDiscriminationFlag:
    def test_triggers_below_06(self):
        poor_eval = _FakeEvaluation(c_index_test=0.55)
        models = {"coxph": _FakeModelResult("coxph", evaluation=poor_eval)}
        learner = _FakeLearner(models=models)
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Low Discrimination" in result["markdown"]
        assert "0.550" in result["markdown"]


class TestOverfittingGapFlag:
    def test_triggers_with_large_gap(self):
        eval_r = _FakeEvaluation(c_index_test=0.55)
        models = {"coxph": _FakeModelResult("coxph", cv_metric_mean=0.70, evaluation=eval_r)}
        learner = _FakeLearner(models=models)
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Train/Test Performance Gap" in result["markdown"]
        assert "0.150" in result["markdown"]


class TestOverfittingGapActionText:
    def test_action_does_not_mention_nested_cv(self):
        """Action should not mention nested CV at all."""
        eval_r = _FakeEvaluation(c_index_test=0.55)
        models = {"coxph": _FakeModelResult("coxph", cv_metric_mean=0.70, evaluation=eval_r)}
        learner = _FakeLearner(models=models)
        section = LimitationsSection(learner)
        result = section._check_overfitting_gap()

        assert result is not None
        assert "nested" not in result["action"].lower()

    def test_action_mentions_tuning_reduction(self):
        """Action should mention reducing tuning trials."""
        eval_r = _FakeEvaluation(c_index_test=0.55)
        models = {"coxph": _FakeModelResult("coxph", cv_metric_mean=0.70, evaluation=eval_r)}
        learner = _FakeLearner(models=models)
        section = LimitationsSection(learner)
        result = section._check_overfitting_gap()

        assert result is not None
        action = result["action"]
        assert "tuning" in action.lower() or "n_trials" in action or "tune=False" in action


class TestOverfittingGapMetricAware:
    """Overfitting gap must compare like-for-like metrics."""

    def test_ibs_scoring_compares_ibs_not_c_index(self):
        """When scoring=integrated_brier, compare CV IBS to test IBS."""
        # IBS is lower-is-better: cv=0.10 (good), test=0.20 (bad) -> gap=0.10
        eval_r = _FakeEvaluation(
            c_index_test=0.75,  # good C-index — would NOT trigger if compared
            integrated_brier_test=0.20,
        )
        models = {"rsf": _FakeModelResult("rsf", cv_metric_mean=0.10, evaluation=eval_r)}
        learner = _FakeLearner(models=models, scoring="integrated_brier")
        section = LimitationsSection(learner)
        result = section._check_overfitting_gap()

        assert result is not None, "Should trigger: IBS gap is 0.10 > 0.05"
        assert "0.100" in result["detail"]

    def test_ibs_scoring_no_false_positive(self):
        """When scoring=integrated_brier, do NOT compare IBS cv to C-index test."""
        # With correct logic: IBS cv=0.15 vs IBS test=0.16 -> gap=0.01 -> no trigger
        eval_r = _FakeEvaluation(
            c_index_test=0.72,
            integrated_brier_test=0.16,
        )
        models = {"rsf": _FakeModelResult("rsf", cv_metric_mean=0.15, evaluation=eval_r)}
        learner = _FakeLearner(models=models, scoring="integrated_brier")
        section = LimitationsSection(learner)
        result = section._check_overfitting_gap()

        assert result is None, "Should NOT trigger: IBS gap is only 0.01"

    def test_harrell_scoring_still_works(self):
        """Default harrell scoring still compares CV to test C-index."""
        eval_r = _FakeEvaluation(c_index_test=0.55)
        models = {"coxph": _FakeModelResult("coxph", cv_metric_mean=0.70, evaluation=eval_r)}
        learner = _FakeLearner(models=models, scoring="harrell")
        section = LimitationsSection(learner)
        result = section._check_overfitting_gap()

        assert result is not None
        assert "0.150" in result["detail"]

    def test_detected_text_uses_metric_name(self):
        """Detected text should name the actual metric, not always 'C-index'."""
        eval_r = _FakeEvaluation(
            c_index_test=0.75,
            integrated_brier_test=0.25,
        )
        models = {"rsf": _FakeModelResult("rsf", cv_metric_mean=0.10, evaluation=eval_r)}
        learner = _FakeLearner(models=models, scoring="integrated_brier")
        section = LimitationsSection(learner)
        result = section._check_overfitting_gap()

        assert result is not None
        assert "IBS" in result["detected"] or "Brier" in result["detected"]
        assert "C-index" not in result["detected"]

    def test_fallback_for_missing_concordance_column(self):
        """If concordance lacks the scoring column, fall back to c_index_test."""
        eval_r = _FakeEvaluation(c_index_test=0.55)
        # scoring is 'ipcw' but concordance has no c_index_ipcw_test column
        models = {"coxph": _FakeModelResult("coxph", cv_metric_mean=0.70, evaluation=eval_r)}
        learner = _FakeLearner(models=models, scoring="ipcw")
        section = LimitationsSection(learner)
        result = section._check_overfitting_gap()

        # Should fall back to harrell and still detect the gap
        assert result is not None
        assert "0.150" in result["detail"]
        # Metric label should reflect the fallback (Harrell), not the scoring (IPCW)
        assert "Harrell" in result["detected"]

    def test_callable_scoring_falls_back_to_harrell(self):
        """Callable scorer falls back to Harrell comparison."""
        eval_r = _FakeEvaluation(c_index_test=0.55)
        models = {"coxph": _FakeModelResult("coxph", cv_metric_mean=0.70, evaluation=eval_r)}
        learner = _FakeLearner(models=models)
        learner.scoring = lambda y_true, y_pred: 0.5  # callable, not a string
        section = LimitationsSection(learner)
        result = section._check_overfitting_gap()

        assert result is not None
        assert "0.150" in result["detail"]

    def test_diagnostic_only_metric_falls_back_to_harrell(self):
        """Diagnostic-only metrics (cal_slope) fall back to Harrell."""
        eval_r = _FakeEvaluation(c_index_test=0.55, cal_slope_test=1.0)
        models = {"coxph": _FakeModelResult("coxph", cv_metric_mean=0.70, evaluation=eval_r)}
        learner = _FakeLearner(models=models, scoring="cal_slope")
        section = LimitationsSection(learner)
        result = section._check_overfitting_gap()

        assert result is not None
        assert "0.150" in result["detail"]
        assert "Harrell" in result["detected"]


class TestPoorCalibrationFlag:
    """Slope-only calibration flag, re-pointed to train(calibrate=...).

    The drift trigger is the test calibration slope outside [0.5, 1.5]; the GND
    p-value is shown for context only (never a standalone trigger).
    """

    @staticmethod
    def _section(models):
        return LimitationsSection(_FakeLearner(models=models))

    def test_triggers_with_bad_slope(self):
        bad_eval = _FakeEvaluation(cal_slope_test=0.4, gnd_test=0.5)
        section = self._section(
            {"coxph": _FakeModelResult("coxph", evaluation=bad_eval)}
        )
        result = section.generate()
        assert "Poor Calibration" in result["markdown"]

    def test_significant_gnd_alone_does_not_trigger(self):
        """A significant GND with an in-band slope is NOT flagged."""
        ev = _FakeEvaluation(cal_slope_test=1.0, gnd_test=0.001)
        section = self._section(
            {"coxph": _FakeModelResult("coxph", evaluation=ev)}
        )
        assert section._check_poor_calibration() is None

    def test_uncalibrated_action_suggests_train_calibrate(self):
        bad_eval = _FakeEvaluation(cal_slope_test=0.4, gnd_test=0.5)
        section = self._section(
            {"coxph": _FakeModelResult("coxph", evaluation=bad_eval)}
        )
        flag = section._check_poor_calibration()
        assert flag is not None
        assert "train(calibrate=" in flag["action"]
        assert "recalibrate(" not in flag["action"]
        assert "n_uncalibrated=1" in flag["detail"]
        assert "after calibration" not in flag["detected"]

    def test_residual_drift_after_calibration_suggests_other_method(self):
        """A model that already carries a calibrator but still drifts is flagged
        as residual, with the before -> after slope and a 'try a different
        method' suggestion."""
        bad_eval = _FakeEvaluation(cal_slope_test=1.9, gnd_test=0.5)
        result = _FakeModelResult(
            "deephit", evaluation=bad_eval,
            calibrator=object(), precalibration_slope=2.6,
        )
        section = self._section({"deephit": result})
        flag = section._check_poor_calibration()
        assert flag is not None
        assert "after calibration" in flag["detected"]
        assert "2.600 -> 1.900" in flag["detected"]
        assert "residual drift" in flag["action"]
        assert "different `train(calibrate=...)` method" in flag["action"]
        assert "n_residual_after_calibration=1" in flag["detail"]

    @pytest.mark.parametrize("boundary", [0.5, 1.5])
    def test_slope_band_inclusive_does_not_trigger(self, boundary):
        ev = _FakeEvaluation(cal_slope_test=boundary, gnd_test=0.001)
        section = self._section(
            {"coxph": _FakeModelResult("coxph", evaluation=ev)}
        )
        assert section._check_poor_calibration() is None

    @pytest.mark.parametrize("slope", [0.49, 1.51])
    def test_slope_just_outside_band_triggers(self, slope):
        ev = _FakeEvaluation(cal_slope_test=slope, gnd_test=0.5)
        section = self._section(
            {"coxph": _FakeModelResult("coxph", evaluation=ev)}
        )
        assert section._check_poor_calibration() is not None

    def test_mixed_fresh_and_residual_partition(self):
        models = {
            "fresh": _FakeModelResult(
                "fresh", evaluation=_FakeEvaluation(cal_slope_test=0.3)),
            "resid": _FakeModelResult(
                "resid", evaluation=_FakeEvaluation(cal_slope_test=1.8),
                calibrator=object()),
            "ok": _FakeModelResult(
                "ok", evaluation=_FakeEvaluation(cal_slope_test=1.0)),
        }
        flag = self._section(models)._check_poor_calibration()
        assert flag is not None
        assert "n_models=2" in flag["detail"]
        assert "n_uncalibrated=1" in flag["detail"]
        assert "n_residual_after_calibration=1" in flag["detail"]
        # 'ok' (in-band) is absent from the flag entirely.
        assert "'ok'" not in flag["detected"]
        assert "Consider calibrating" in flag["action"]
        assert "residual drift" in flag["action"]


class TestLowEPVFlag:
    def test_triggers_below_10(self):
        rng = np.random.RandomState(42)
        # 20 events across 5 features = EPV 4.0
        X = pd.DataFrame(rng.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        events = np.zeros(200)
        events[:20] = 1
        y = pd.DataFrame({"time": rng.exponential(10, 200), "event": events})

        learner = _FakeLearner(X_train=X, y_train=y)
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Low Events Per Variable" in result["markdown"]
        assert "epv=4.0" in section.data["flags"].query("flag_id == 'low_epv'")["detail"].iloc[0]

    def test_no_trigger_above_10(self):
        """Default learner has ~180 events / 5 features = EPV ~36."""
        learner = _FakeLearner()
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "low_epv"]
        assert not row["triggered"].iloc[0]


class TestMissingDataFlag:
    def test_triggers_high_overall(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
        # Inject 30% missing overall
        mask = rng.random(X.shape) < 0.30
        X[mask] = np.nan

        learner = _FakeLearner(X_train=X, X_test=None)
        learner.X_test = None
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Extensive Missing Data" in result["markdown"]

    def test_triggers_single_var_above_50(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
        # One variable with 60% missing, others clean
        X.loc[X.index[:60], "b"] = np.nan

        learner = _FakeLearner(X_train=X, X_test=None)
        learner.X_test = None
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Extensive Missing Data" in result["markdown"]
        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "missing_data"]
        assert row["triggered"].iloc[0]
        assert "60.0%" in row["detail"].iloc[0]

    def test_no_trigger_clean_data(self):
        learner = _FakeLearner()
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "missing_data"]
        assert not row["triggered"].iloc[0]


class TestPHViolationsFlag:
    @staticmethod
    def _ph_result(features=("age", "grade"), global_sig=True):
        """Build a cached ph_test_result DataFrame with given violating features."""
        rows = []
        for feat in features:
            rows.append({
                "feature": feat,
                "test_statistic": 10.0,
                "p_value": 0.001,
                "p_adjusted": 0.002,
                "significant": True,
            })
        rows.append({
            "feature": "GLOBAL",
            "test_statistic": 15.0,
            "p_value": 0.0005,
            "p_adjusted": float("nan"),
            "significant": global_sig,
        })
        return pd.DataFrame(rows)

    def test_triggers_with_violations_when_cox_trained(self):
        """Cached ph_test_result with violations + CoxPH → flag fires, names coxph."""
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result()
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Proportional Hazards Violations" in result["markdown"]
        assert "age" in result["markdown"]
        assert "grade" in result["markdown"]
        assert "Affected trained models: coxph" in result["markdown"]

    def test_triggers_without_cox_when_violations_present(self):
        """Violations present but no PH-assumption model → flag still fires."""
        models = {"rsf": _FakeModelResult("rsf")}
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result()
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Proportional Hazards Violations" in result["markdown"]
        assert "No affected models" in result["markdown"]
        assert "Affected trained models" not in result["markdown"]

    def test_no_trigger_when_no_cached_result(self):
        """No cached PH result (user opted out of setup's ph_test) → no flag."""
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        # No ph_test_result set on _state
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "ph_violations"]
        assert not row["triggered"].iloc[0]

    def test_no_trigger_when_no_violations(self):
        """Cached result with no significant features → no flag (no positive confirmation here)."""
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        # Empty violations
        learner._state.ph_test_result = self._ph_result(features=(), global_sig=False)
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "ph_violations"]
        assert not row["triggered"].iloc[0]

    def test_action_leads_with_mlsurv_alternatives(self):
        """Action should lead with mlsurv-supported non-PH models."""
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result(features=("age",))
        section = LimitationsSection(learner)
        result = section._check_ph_violations()

        assert result is not None
        action = result["action"]
        non_ph_pos = action.find("non-PH alternatives")
        time_varying_pos = action.find("time-varying")
        assert non_ph_pos != -1
        assert non_ph_pos < time_varying_pos, (
            "Non-PH alternatives (mlsurv-supported) should appear before "
            "time-varying coefficients (not supported)"
        )
        assert "RSF" in action
        assert "DeepHit" in action

    def test_affected_models_sorted_with_multiple_cox_family(self):
        """Multiple Cox-family models trained → affected list is sorted and complete."""
        models = {
            "coxph": _FakeModelResult("coxph"),
            "coxnet": _FakeModelResult("coxnet"),
            "rsf": _FakeModelResult("rsf"),
        }
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result()
        section = LimitationsSection(learner)
        result = section._check_ph_violations()

        assert result is not None
        assert "Affected trained models: coxnet, coxph" in result["detected"]

    def test_fssvm_not_recognized_as_ph_assumption_model(self):
        """FSSVM optimizes a concordance-ranking objective, not a hazard model
        — it should NOT appear in PH-violation 'Affected trained models'.
        """
        models = {"fssvm": _FakeModelResult("fssvm")}
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result()
        section = LimitationsSection(learner)
        result = section._check_ph_violations()

        # No PH-assuming models trained -> falls back to the "no Cox-family
        # models trained, but PH-assuming alternatives include..." sentence.
        # FSSVM must not be listed as affected.
        if result is not None:
            assert "Affected trained models: fssvm" not in result["detected"]

    def test_xgbcox_recognized_as_ph_assumption_model(self):
        """xgbcox fits the Cox partial likelihood → flagged as PH-affected."""
        models = {"xgbcox": _FakeModelResult("xgbcox")}
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result()
        section = LimitationsSection(learner)
        result = section._check_ph_violations()

        assert result is not None
        assert "Affected trained models: xgbcox" in result["detected"]

    def test_gbsa_recognized_as_ph_assumption_model(self):
        """gbsa defaults to loss='coxph' → flagged as PH-affected."""
        models = {"gbsa": _FakeModelResult("gbsa")}
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result()
        section = LimitationsSection(learner)
        result = section._check_ph_violations()

        assert result is not None
        assert "Affected trained models: gbsa" in result["detected"]

    def test_weibull_aft_recognized_as_ph_assumption_model(self):
        """Weibull is simultaneously PH and AFT (Zhang 2016) → flagged as PH-affected."""
        models = {"weibull_aft": _FakeModelResult("weibull_aft")}
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result()
        section = LimitationsSection(learner)
        result = section._check_ph_violations()

        assert result is not None
        assert "Affected trained models: weibull_aft" in result["detected"]

    def test_fallback_sentence_lists_all_ph_models(self):
        """Fallback sentence is derived from PH_ASSUMPTION_MODELS so it stays in sync."""
        from mlsurv.constants import MODEL_DISPLAY_NAMES, PH_ASSUMPTION_MODELS

        models = {"rsf": _FakeModelResult("rsf")}
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result()
        section = LimitationsSection(learner)
        result = section._check_ph_violations()

        assert result is not None
        detected = result["detected"]
        for key in PH_ASSUMPTION_MODELS:
            assert MODEL_DISPLAY_NAMES[key] in detected, (
                f"Fallback sentence should list {key} display name"
            )

    def test_no_on_demand_ph_test_recompute(self, monkeypatch):
        """Report reads cache only — ph_test must never be called during rendering."""
        call_count = {"n": 0}

        def _should_not_be_called(X, y, *args, **kwargs):
            call_count["n"] += 1
            raise AssertionError("ph_test must not be called from the report path")

        monkeypatch.setattr(
            "mlsurv.evaluation.metrics.ph_test", _should_not_be_called,
        )
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        learner._state.ph_test_result = self._ph_result()
        section = LimitationsSection(learner)
        section.generate()

        assert call_count["n"] == 0


class TestLinearityViolationFlag:
    """Tests for the linearity_violation flag."""

    def _lin_result(self, *, violations=("x0",), global_sig=True):
        """Build a fake linearity_test_result DataFrame with specified violations."""
        rows = []
        for feat in ("x0", "x1", "x2", "x3"):
            sig = feat in violations
            rows.append({
                "feature": feat,
                "test_statistic": 100.0 if sig else 2.0,
                "p_value": 1e-20 if sig else 0.4,
                "p_adjusted": 1e-19 if sig else 0.5,
                "significant": sig,
            })
        rows.append({
            "feature": "GLOBAL",
            "test_statistic": 120.0 if global_sig else 5.0,
            "p_value": 1e-22 if global_sig else 0.3,
            "p_adjusted": float("nan"),
            "significant": global_sig,
        })
        return pd.DataFrame(rows)

    def test_triggers_with_violations_when_cox_trained(self):
        """Cached linearity_test_result with violations + Cox model → flag fires, names coxph."""
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        learner._state.linearity_test_result = self._lin_result(violations=("x0",))
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Log-Hazard Linearity Violations" in result["markdown"]
        assert "x0" in result["markdown"]
        assert "Affected trained models: coxph" in result["markdown"]

    def test_triggers_with_violations_when_weibull_aft_trained(self):
        """Weibull AFT implies log-linearity in covariates → flagged as linearity-affected."""
        models = {"weibull_aft": _FakeModelResult("weibull_aft")}
        learner = _FakeLearner(models=models)
        learner._state.linearity_test_result = self._lin_result(violations=("x0",))
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Log-Hazard Linearity Violations" in result["markdown"]
        assert "x0" in result["markdown"]
        assert "Affected trained models: weibull_aft" in result["markdown"]

    def test_triggers_without_cox_when_violations_present(self):
        """Violations present but only non-linear-assumption models → flag still fires."""
        models = {"rsf": _FakeModelResult("rsf")}
        learner = _FakeLearner(models=models)
        learner._state.linearity_test_result = self._lin_result(violations=("x0",))
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Log-Hazard Linearity Violations" in result["markdown"]
        assert "No affected models" in result["markdown"]
        assert "Affected trained models" not in result["markdown"]

    def test_no_trigger_when_no_violations(self):
        """Cached result with no significant features → no flag."""
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        learner._state.linearity_test_result = self._lin_result(
            violations=(), global_sig=False,
        )
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "linearity_violation"]
        assert not row["triggered"].iloc[0]

    def test_detail_lists_violating_features(self):
        """Multiple violations are named in the detail text."""
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        learner._state.linearity_test_result = self._lin_result(
            violations=("x0", "x2"),
        )
        section = LimitationsSection(learner)
        result = section._check_linearity_violation()

        assert result is not None
        assert "x0" in result["detected"]
        assert "x2" in result["detected"]

    def test_action_mentions_transformations_and_flexible_models(self):
        """Action text recommends transformations or flexible models."""
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        learner._state.linearity_test_result = self._lin_result(violations=("x0",))
        section = LimitationsSection(learner)
        result = section._check_linearity_violation()

        assert result is not None
        action = result["action"]
        assert "transformation" in action.lower() or "log" in action
        assert any(m in action for m in ("RSF", "GBSA", "DeepSurv"))

    def test_no_trigger_when_no_cached_result(self):
        """No cached linearity result (user opted out) → no flag."""
        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(models=models)
        # No linearity_test_result set on _state
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "linearity_violation"]
        assert not row["triggered"].iloc[0]

    def test_fallback_sentence_lists_all_linearity_models(self):
        """Fallback sentence is derived from LINEARITY_ASSUMPTION_MODELS."""
        from mlsurv.constants import (
            LINEARITY_ASSUMPTION_MODELS,
            MODEL_DISPLAY_NAMES,
        )

        models = {"rsf": _FakeModelResult("rsf")}
        learner = _FakeLearner(models=models)
        learner._state.linearity_test_result = self._lin_result(violations=("x0",))
        section = LimitationsSection(learner)
        result = section._check_linearity_violation()

        assert result is not None
        detected = result["detected"]
        for key in LINEARITY_ASSUMPTION_MODELS:
            assert MODEL_DISPLAY_NAMES[key] in detected, (
                f"Fallback sentence should list {key} display name"
            )


class TestMulticollinearityFlag:
    def test_triggers_substantial(self, monkeypatch):
        """Mock detect_correlated_features to return substantial collinearity."""
        monkeypatch.setattr(
            "mlsurv.evaluation.importance.detect_correlated_features",
            lambda X, threshold=0.8: [("f0", "f1", 0.95), ("f2", "f3", 0.88)],
        )
        monkeypatch.setattr(
            "mlsurv.reports.sections.feature_analysis._assess_correlation_severity",
            lambda n, pct, mx: ("substantial", "Consider PCA."),
        )
        learner = _FakeLearner()
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Substantial Multicollinearity" in result["markdown"]
        assert "substantial" in result["markdown"].lower()

    def test_no_trigger_minor(self, monkeypatch):
        """Minor severity -> flag not triggered."""
        monkeypatch.setattr(
            "mlsurv.evaluation.importance.detect_correlated_features",
            lambda X, threshold=0.8: [("f0", "f1", 0.82)],
        )
        monkeypatch.setattr(
            "mlsurv.reports.sections.feature_analysis._assess_correlation_severity",
            lambda n, pct, mx: ("minor", "No action needed."),
        )
        learner = _FakeLearner()
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "multicollinearity"]
        assert not row["triggered"].iloc[0]


class TestNaNHandling:
    def test_nan_c_index_no_crash(self):
        """NaN evaluation metrics should not crash or trigger flags."""
        nan_eval = _FakeEvaluation(
            c_index_test=float("nan"),
            cal_slope_test=float("nan"),
            gnd_test=float("nan"),
        )
        models = {"coxph": _FakeModelResult(
            "coxph", cv_metric_mean=float("nan"), evaluation=nan_eval
        )}
        learner = _FakeLearner(models=models)
        section = LimitationsSection(learner)
        result = section.generate()

        # Should not crash
        assert isinstance(result["markdown"], str)
        # NaN metrics should not trigger discrimination/calibration/gap flags
        flags_df = section.data["flags"]
        for fid in ("low_discrimination", "poor_calibration", "overfitting_gap"):
            row = flags_df[flags_df["flag_id"] == fid]
            assert not row["triggered"].iloc[0], f"{fid} should not trigger with NaN"


class TestInfHandling:
    def test_inf_c_index_no_crash(self):
        """Inf evaluation metrics should not crash or trigger flags."""
        inf_eval = _FakeEvaluation(
            c_index_test=np.inf,
            cal_slope_test=np.inf,
            gnd_test=np.inf,
        )
        models = {"coxph": _FakeModelResult(
            "coxph", cv_metric_mean=np.inf, evaluation=inf_eval
        )}
        learner = _FakeLearner(models=models)
        section = LimitationsSection(learner)
        result = section.generate()

        assert isinstance(result["markdown"], str)
        flags_df = section.data["flags"]
        for fid in ("low_discrimination", "poor_calibration", "overfitting_gap"):
            row = flags_df[flags_df["flag_id"] == fid]
            assert not row["triggered"].iloc[0], f"{fid} should not trigger with inf"


class TestModelAppropriatenessFlag:
    def test_triggers_on_small_data_with_tree_model(self):
        """Tree model on tiny data -> flag triggered."""
        rng = np.random.RandomState(42)
        # N=50, 10 events, 5 features -> EPV=2
        X = pd.DataFrame(rng.randn(50, 5), columns=[f"f{i}" for i in range(5)])
        events = np.zeros(50)
        events[:10] = 1
        y = pd.DataFrame({"time": rng.exponential(10, 50), "event": events})

        models = {
            "rsf": _FakeModelResult("rsf"),
            "coxph": _FakeModelResult("coxph"),
        }
        learner = _FakeLearner(X_train=X, y_train=y, models=models)
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Model Appropriateness" in result["markdown"]
        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "model_appropriateness"]
        assert row["triggered"].iloc[0]
        assert "rsf" in row["detail"].iloc[0]

    def test_empty_recommended_action_message(self):
        """When ALL models are not_recommended, action says 'collect more data'."""
        rng = np.random.RandomState(42)
        # N=30, 3 events, 10 features -> EPV=0.3 -> all not_recommended
        X = pd.DataFrame(rng.randn(30, 10), columns=[f"f{i}" for i in range(10)])
        events = np.zeros(30)
        events[:3] = 1
        y = pd.DataFrame({"time": rng.exponential(10, 30), "event": events})

        models = {"rsf": _FakeModelResult("rsf")}
        learner = _FakeLearner(X_train=X, y_train=y, models=models)
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Model Appropriateness" in result["markdown"]
        # With EPV=0.3, no model is recommended -> empty-recommended branch
        assert "collecting more data" in result["markdown"].lower()

    def test_no_trigger_on_large_data(self):
        """All models appropriate for large data -> no flag."""
        rng = np.random.RandomState(42)
        n = 600
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({"time": rng.exponential(10, n), "event": rng.binomial(1, 0.6, n)})

        models = {
            "coxph": _FakeModelResult("coxph"),
            "rsf": _FakeModelResult("rsf"),
        }
        learner = _FakeLearner(X_train=X, y_train=y, models=models)
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "model_appropriateness"]
        assert not row["triggered"].iloc[0]

    def test_no_trigger_with_only_linear_models(self):
        """Linear models on medium data -> no flag."""
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({"time": rng.exponential(10, n), "event": rng.binomial(1, 0.6, n)})

        models = {"coxph": _FakeModelResult("coxph")}
        learner = _FakeLearner(X_train=X, y_train=y, models=models)
        section = LimitationsSection(learner)
        section.generate()

        flags_df = section.data["flags"]
        row = flags_df[flags_df["flag_id"] == "model_appropriateness"]
        assert not row["triggered"].iloc[0]

    def test_assumption_driven_demotion_omits_sample_size_framing(self):
        """coxph demoted for PH/linearity on ample data -> the text
        names the assumption cause and drops sample-size framing/citations."""
        from mlsurv.guidance import recommend_models

        # METABRIC-like: N=1427, EPV=91.8 (plenty), but PH + linearity violated.
        rec = recommend_models(
            n_samples=1427, n_events=459, n_features=5,
            ph_violations=["f0", "f1", "f2", "f3"],
            linearity_violations=["f0", "f1"],
            verbose=False,
        )
        assert "coxph" in rec.not_recommended  # demoted purely by assumptions

        learner = _FakeLearner(models={"coxph": _FakeModelResult("coxph")})
        learner._state.model_recommendations = rec
        section = LimitationsSection(learner)

        result = section._check_model_appropriateness()
        assert result is not None
        why = result["why"].lower()
        # Names the real cause and points to the assumption sections.
        assert "proportional-hazards assumption" in why
        assert "log-hazard linearity assumption" in why
        # No size framing or sample-size citations for an assumption-only demotion.
        assert "insufficient data" not in why
        assert "riley" not in why
        assert "van der ploeg" not in why
        assert "epv=" not in why
        # detected echoes the real per-model reason (matches the recs table).
        assert "PH" in result["detected"]
        assert "linearity" in result["detected"]

    def test_mixed_causes_attributed_per_model(self):
        """Size-driven and assumption-driven models in one report each
        get the right framing, attributed to the right model."""
        from mlsurv.guidance import recommend_models

        # N=150, EPV=20: coxph (linear, EPV_rec=20) is size-OK but assumption-
        # demoted; rsf (tree_ensemble, N_rec=500/N_caut=200) is size-demoted
        # (N=150 < 200) and assumption-free.
        rec = recommend_models(
            n_samples=150, n_events=100, n_features=5,
            ph_violations=["f0", "f1", "f2", "f3"],
            linearity_violations=["f0", "f1"],
            verbose=False,
        )
        assert "coxph" in rec.not_recommended
        assert "rsf" in rec.not_recommended

        learner = _FakeLearner(models={
            "coxph": _FakeModelResult("coxph"),
            "rsf": _FakeModelResult("rsf"),
        })
        learner._state.model_recommendations = rec
        section = LimitationsSection(learner)

        result = section._check_model_appropriateness()
        why = result["why"]
        # rsf is the size-driven model: its clause carries the sample-size papers.
        assert "rsf: too few events" in why
        assert "Riley et al. 2019" in why
        assert "van der Ploeg et al. 2014" in why
        # coxph is the assumption-driven model: its clause cites the PH section,
        # NOT sample size.
        assert "coxph: violate the proportional-hazards assumption" in why
        # coxph must not be swept into the size clause.
        assert "coxph: too few events" not in why

    def test_defensive_branch_when_causes_unavailable(self):
        """A flagged model whose demotion_causes() is empty (rec and
        causes disagree) falls back to generic text, never a false size cause."""
        from mlsurv.guidance import recommend_models

        # Ample data, no violations -> coxph is *recommended* with no causes.
        rec = recommend_models(
            n_samples=1427, n_events=459, n_features=5,
            ph_violations=[], linearity_violations=[], verbose=False,
        )
        assert rec.demotion_causes("coxph") == ()
        # Force the inconsistency the defensive branch guards against.
        rec.not_recommended.append("coxph")

        learner = _FakeLearner(models={"coxph": _FakeModelResult("coxph")})
        learner._state.model_recommendations = rec
        section = LimitationsSection(learner)

        result = section._check_model_appropriateness()
        assert result is not None
        why = result["why"].lower()
        assert "demoted below the recommended tier" in why
        # Must not invent a size cause or cite sample-size papers.
        assert "insufficient data" not in why
        assert "riley" not in why
        assert "van der ploeg" not in why


# ---------------------------------------------------------------------------
# get_triggered_flags()
# ---------------------------------------------------------------------------


class TestGetTriggeredFlags:
    def test_returns_triggered_pairs(self):
        """Small dataset triggers at least small_sample; returns (id, name) tuples."""
        rng = np.random.RandomState(42)
        X_small = pd.DataFrame(rng.randn(80, 5), columns=[f"f{i}" for i in range(5)])
        y_small = pd.DataFrame({"time": rng.exponential(10, 80), "event": rng.binomial(1, 0.6, 80)})
        X_test = pd.DataFrame(rng.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        y_test = pd.DataFrame({"time": rng.exponential(10, 20), "event": rng.binomial(1, 0.6, 20)})

        learner = _FakeLearner(X_train=X_small, y_train=y_small, X_test=X_test, y_test=y_test)
        section = LimitationsSection(learner)
        triggered = section.get_triggered_flags()

        assert len(triggered) >= 1
        flag_ids = [fid for fid, _ in triggered]
        assert "small_sample" in flag_ids
        # Verify tuple structure
        for fid, fname in triggered:
            assert isinstance(fid, str)
            assert isinstance(fname, str)

    def test_returns_empty_for_clean_data(self):
        """Clean data with good metrics -> empty list."""
        good_eval = _FakeEvaluation(c_index_test=0.75, cal_slope_test=1.0, gnd_test=0.5)
        models = {
            "coxph": _FakeModelResult("coxph", cv_metric_mean=0.75, evaluation=good_eval),
            "rsf": _FakeModelResult("rsf", cv_metric_mean=0.74, evaluation=good_eval),
        }
        learner = _FakeLearner(models=models, validation_results={"cohort_a": True})
        section = LimitationsSection(learner)
        triggered = section.get_triggered_flags()

        assert triggered == []

    def test_does_not_store_data(self):
        """get_triggered_flags should not populate section.data['flags']."""
        learner = _FakeLearner()
        section = LimitationsSection(learner)
        section.get_triggered_flags()

        assert "flags" not in section.data

    def test_logs_debug_on_checker_failure(self):
        """Checker crash emits logger.debug, not silent skip."""
        learner = _FakeLearner()
        section = LimitationsSection(learner)

        # Patch a checker to raise
        def _boom():
            raise ValueError("boom")

        with pytest.MonkeyPatch.context() as m:
            logs = []
            m.setattr(section, "_check_small_sample", _boom)
            m.setattr(
                section._logger, "debug",
                lambda *args, **kw: logs.append(args),
            )
            section.get_triggered_flags()

        assert any("small_sample" in str(entry) for entry in logs)
        assert any("boom" in str(entry) for entry in logs)


# ---------------------------------------------------------------------------
# Summary section cross-reference
# ---------------------------------------------------------------------------


class _FakeSummaryState(_FakeState):
    """State stub with fields needed by SummarySection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_setup = True
        self.outcome_name = "event"
        self.time_unit = None
        self.subpopulation_results = None


class _FakeSummaryLearner(_FakeLearner):
    """Extended learner stub with fields needed by SummarySection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scoring = "harrell"
        self.cv_strategy = "stratifiedkfold"
        self.cv_folds = 5
        self.cv_repeats = 1
        self.X = self.X_train
        self._state = _FakeSummaryState(
            validation_results=kwargs.get("validation_results") or {},
        )
        self._state.models = self._models


class TestSummaryLimitationsLine:
    def test_small_dataset_shows_limitations_line(self):
        """Executive summary should mention limitations when flags trigger."""
        from mlsurv.reports.sections.summary import SummarySection

        rng = np.random.RandomState(42)
        X_small = pd.DataFrame(rng.randn(80, 5), columns=[f"f{i}" for i in range(5)])
        y_small = pd.DataFrame({"time": rng.exponential(10, 80), "event": rng.binomial(1, 0.6, 80)})
        X_test = pd.DataFrame(rng.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        y_test = pd.DataFrame({"time": rng.exponential(10, 20), "event": rng.binomial(1, 0.6, 20)})

        learner = _FakeSummaryLearner(
            X_train=X_small, y_train=y_small, X_test=X_test, y_test=y_test,
        )
        section = SummarySection(learner)
        result = section.generate()
        md = result["markdown"]

        assert "limitation" in md.lower()
        assert "Small Sample Size" in md
        assert "Limitations & Caveats" in md

    def test_clean_dataset_no_limitations_line(self):
        """Executive summary omits the line when no flags trigger."""
        from mlsurv.reports.sections.summary import SummarySection

        good_eval = _FakeEvaluation(c_index_test=0.75, cal_slope_test=1.0, gnd_test=0.5)
        models = {
            "coxph": _FakeModelResult("coxph", cv_metric_mean=0.75, evaluation=good_eval),
            "rsf": _FakeModelResult("rsf", cv_metric_mean=0.74, evaluation=good_eval),
        }
        learner = _FakeSummaryLearner(
            models=models, validation_results={"cohort_a": True},
        )
        section = SummarySection(learner)
        result = section.generate()
        md = result["markdown"]

        assert "limitation" not in md.lower()

    def test_singular_label_for_one_flag(self):
        """One triggered flag -> 'limitation' (singular)."""
        from mlsurv.reports.sections.summary import SummarySection

        # Only low_discrimination triggers (large data, good calibration, external validation)
        low_eval = _FakeEvaluation(c_index_test=0.55, cal_slope_test=1.0, gnd_test=0.5)
        models = {
            "coxph": _FakeModelResult("coxph", cv_metric_mean=0.55, evaluation=low_eval),
        }
        learner = _FakeSummaryLearner(
            models=models, validation_results={"cohort_a": True},
        )
        section = SummarySection(learner)
        result = section.generate()
        md = result["markdown"]

        assert "**1 limitation flagged:**" in md


class TestIPCWCompositionRendering:
    """The Limitations IPCW flag must describe the composition failure mode
    (G_hat not estimable) distinctly from numerical instability."""

    def _section_with_flag(self, flag):
        learner = _FakeLearner()
        learner._state.ipcw_support_flag = flag
        return LimitationsSection(learner)

    def test_composition_only_renders_distinct_text(self):
        """Composition-only trigger (no overhang, G_hat fine): detected_parts
        is non-empty and the prose names the not-estimable failure mode."""
        section = self._section_with_flag({
            "n_beyond": 0,
            "min_g_hat": 1.0,
            "t_max_train": 50.0,
            "tau_used": 20.0,
            "g_hat_source": "train",
            "n_g_events": 40,
            "n_g_censored": 0,  # all-events cohort: the worst trigger
            "min_ipcw_support": 10,
        })
        out = section._check_ipcw_support()
        assert out is not None
        # detected_parts is non-empty -> prose is not "... support: ."
        assert "G_hat not estimable" in out["detected"]
        assert "0 censored obs" in out["detected"]
        assert not out["detected"].endswith(": .")
        # Distinct failure-mode wording in why/action.
        assert "unweighted forms" in out["why"]
        assert "effectively off" in out["action"]
        assert "n_g_censored=0" in out["detail"]

    def test_instability_only_omits_composition_text(self):
        """A healthy-composition cohort that only trips the instability/overhang
        branch must NOT show the composition wording."""
        section = self._section_with_flag({
            "n_beyond": 5,
            "min_g_hat": 0.02,
            "t_max_train": 10.0,
            "tau_used": 8.0,
            "g_hat_source": "train",
            "n_g_events": 40,
            "n_g_censored": 40,
            "min_ipcw_support": 10,
        })
        out = section._check_ipcw_support()
        assert out is not None
        assert "G_hat not estimable" not in out["detected"]
        assert "effectively off" not in out["action"]


class TestIPCWSupportPerCohort:
    """Flag 14 covers every cohort, not just the last one evaluated."""

    _FLAG = {
        "n_beyond": 3, "min_g_hat": 0.02, "t_max_train": 10.0, "tau_used": 8.0,
        "g_hat_source": "test", "n_g_events": 30, "n_g_censored": 30,
        "min_ipcw_support": 10,
    }

    def _section(self, test_set_flag=None, by_cohort=None):
        learner = _FakeLearner()
        learner._state.ipcw_support_flag = test_set_flag
        learner._state.ipcw_support_flags = dict(by_cohort or {})
        return LimitationsSection(learner)

    def test_no_flags_returns_none(self):
        assert self._section()._check_ipcw_support() is None

    def test_test_set_only_keeps_single_cohort_text(self):
        out = self._section(self._FLAG, {"main": self._FLAG})._check_ipcw_support()
        assert out is not None
        assert "Validation cohort" not in out["detected"]
        assert out["detected"].startswith("IPCW-weighted metrics")

    def test_every_flagged_external_cohort_is_listed(self):
        """Two external cohorts flagged and a clean test set: both are named,
        in the order they were validated."""
        out = self._section(None, {"MSK2": self._FLAG, "Kato": self._FLAG})._check_ipcw_support()
        assert out is not None
        assert "Validation cohort MSK2:" in out["detected"]
        assert "Validation cohort Kato:" in out["detected"]
        assert out["detected"].index("MSK2") < out["detected"].index("Kato")
        assert "Test set" not in out["detected"]
        assert out["detail"].count("Validation cohort") == 2

    def test_combined_advice_does_not_quote_one_cohorts_settings(self):
        """With the test set (train G_hat) and an external cohort (test G_hat)
        both flagged, the shared advice must not attribute one cohort's tau or
        G_hat source to every cohort; those stay in each cohort's detail."""
        test_flag = dict(self._FLAG, g_hat_source="train", tau_used=5.0)
        out = self._section(test_flag, {"main": test_flag, "Kato": self._FLAG})._check_ipcw_support()
        assert "Test set:" in out["detected"] and "Validation cohort Kato:" in out["detected"]
        assert "use tau=" not in out["action"]  # no single cohort's tau
        assert "validation-cohort G_hat" not in out["action"]
        assert "tau=5.0" in out["detail"] and "tau=8.0" in out["detail"]


def test_record_ipcw_support_flag_keeps_test_set_flag_for_external_cohorts():
    from types import SimpleNamespace

    from mlsurv.learner._evaluation_helpers import _record_ipcw_support_flag

    state = SimpleNamespace(ipcw_support_flag=None, ipcw_support_flags={})
    test_flag, ext_flag = {"n_beyond": 1}, {"n_beyond": 2}

    _record_ipcw_support_flag(state, "main", test_flag, is_test_set=True)
    _record_ipcw_support_flag(state, "Kato", ext_flag, is_test_set=False)
    assert state.ipcw_support_flag is test_flag
    assert state.ipcw_support_flags == {"main": test_flag, "Kato": ext_flag}

    # A clean re-run of a cohort clears only that cohort's stale flag.
    _record_ipcw_support_flag(state, "Kato", None, is_test_set=False)
    assert state.ipcw_support_flags == {"main": test_flag}
    assert state.ipcw_support_flag is test_flag
