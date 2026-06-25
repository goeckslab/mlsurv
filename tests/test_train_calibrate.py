"""Tests for train(calibrate=...) — internal S(t|x) calibration."""
import io

import numpy as np
import pytest
import joblib

from mlsurv.evaluation._calibrators import (
    BreslowSlopeCalibrator, IsotonicCalibrator, LowDoFCalibrator,
    _CALIBRATE_UNSET, _DEFAULT_CALIBRATION, create_calibrator,
    resolve_calibration_spec,
)


def _toy_survival(n=120, seed=0):
    """Small structured survival array + a single risk score per subject."""
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    # Higher score -> shorter time. Exponential times, ~30% censored.
    time = rng.exponential(scale=np.exp(-0.5 * score)) + 0.05
    event = rng.uniform(size=n) > 0.3
    y = np.array([(bool(e), float(t)) for e, t in zip(event, time)],
                 dtype=[("event", bool), ("time", float)])
    return score, y


def _toy_oof_surv(horizons, n=120, seed=1):
    """A monotone-decreasing OOF S(t|x) block in (0, 1)."""
    rng = np.random.default_rng(seed)
    base = np.sort(rng.uniform(0.05, 0.95, size=(n, len(horizons))), axis=1)[:, ::-1]
    return np.clip(base, 1e-3, 1 - 1e-3)


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

class TestGrammar:
    SCORE = {"fssvm": True, "rsf": False, "deephit": False, "coxph": True}

    def test_unset_applies_registry(self):
        spec = resolve_calibration_spec(
            _CALIBRATE_UNSET, ["fssvm", "rsf", "deephit", "coxph"], self.SCORE,
        )
        # rsf is NOT a default (stays raw); fssvm/deephit are.
        assert spec == {"fssvm": "breslow_slope", "rsf": None,
                        "deephit": "isotonic", "coxph": None}

    @pytest.mark.parametrize("value", [False, None])
    def test_false_or_none_all_raw(self, value):
        spec = resolve_calibration_spec(value, ["fssvm", "rsf"], self.SCORE)
        assert spec == {"fssvm": None, "rsf": None}

    def test_string_applies_to_all(self):
        spec = resolve_calibration_spec("isotonic", ["rsf", "coxph"], self.SCORE)
        assert spec == {"rsf": "isotonic", "coxph": "isotonic"}

    def test_dict_missing_key_is_raw(self):
        spec = resolve_calibration_spec(
            {"coxph": "lowdof"}, ["coxph", "rsf"], self.SCORE,
        )
        assert spec == {"coxph": "lowdof", "rsf": None}

    def test_dict_unknown_model_warns(self):
        with pytest.warns(Warning, match="not being trained"):
            resolve_calibration_spec({"ghost": "isotonic"}, ["coxph"], self.SCORE)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            resolve_calibration_spec(3.14, ["coxph"], self.SCORE)


class TestApplicability:
    SCORE = {"fssvm": True, "rsf": False}

    def test_breslow_on_non_score_raises(self):
        with pytest.raises(ValueError, match="score"):
            resolve_calibration_spec({"rsf": "breslow_slope"}, ["rsf"], self.SCORE)

    def test_breslow_on_score_ok(self):
        spec = resolve_calibration_spec(
            {"fssvm": "breslow_slope"}, ["fssvm"], self.SCORE,
        )
        assert spec == {"fssvm": "breslow_slope"}

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            resolve_calibration_spec({"rsf": "nope"}, ["rsf"], self.SCORE)

    def test_warpers_apply_to_any_model(self):
        for method in ("isotonic", "lowdof"):
            spec = resolve_calibration_spec({"rsf": method}, ["rsf"], self.SCORE)
            assert spec == {"rsf": method}

    def test_create_calibrator_unknown_raises(self):
        with pytest.raises(ValueError):
            create_calibrator("nope")

    def test_registry_defaults_are_fssvm_and_deephit(self):
        # rsf was dropped from the defaults (lowdof regresses on several
        # datasets); it stays raw unless explicitly elected.
        assert set(_DEFAULT_CALIBRATION) == {"fssvm", "deephit"}
        assert "rsf" not in _DEFAULT_CALIBRATION


# ---------------------------------------------------------------------------
# Calibrator math: fit / apply shapes and monotonicity
# ---------------------------------------------------------------------------

class TestCalibratorMath:
    HORIZONS = np.array([0.5, 1.0, 2.0, 3.0])

    def _monotone(self, S):
        return bool((np.diff(S, axis=1) <= 1e-9).all())

    def test_isotonic_fit_apply(self):
        _, y = _toy_survival()
        oof = _toy_oof_surv(self.HORIZONS)
        cal = IsotonicCalibrator().fit(
            y_train=y, horizons=self.HORIZONS, oof_S=oof, model_name="m",
        )
        out = cal.apply(pipeline=None, X=None, surv_probs=oof,
                        eval_times=self.HORIZONS)
        assert out.shape == oof.shape
        assert ((out >= 0) & (out <= 1)).all()
        assert self._monotone(out)

    def test_lowdof_fit_apply(self):
        _, y = _toy_survival()
        oof = _toy_oof_surv(self.HORIZONS)
        cal = LowDoFCalibrator().fit(
            y_train=y, horizons=self.HORIZONS, oof_S=oof, model_name="m",
        )
        assert np.isfinite(cal.a) and np.isfinite(cal.b)
        out = cal.apply(pipeline=None, X=None, surv_probs=oof,
                        eval_times=self.HORIZONS)
        assert out.shape == oof.shape
        assert ((out >= 0) & (out <= 1)).all()
        assert self._monotone(out)

    def test_breslow_slope_fit_apply(self):
        score, y = _toy_survival()
        cal = BreslowSlopeCalibrator().fit(
            y_train=y, horizons=self.HORIZONS, oof_scores=score, model_name="m",
        )
        assert np.isfinite(cal.beta) and cal.baseline_ is not None

        class _Pipe:
            def predict(self, X):
                return X
        out = cal.apply(pipeline=_Pipe(), X=score, surv_probs=None,
                        eval_times=self.HORIZONS)
        assert out.shape == (len(score), len(self.HORIZONS))
        assert ((out >= 0) & (out <= 1)).all()
        assert self._monotone(out)

    def test_breslow_requires_oof_scores(self):
        _, y = _toy_survival()
        with pytest.raises(ValueError, match="oof_scores"):
            BreslowSlopeCalibrator().fit(
                y_train=y, horizons=self.HORIZONS, oof_S=None, model_name="m",
            )

    def test_breslow_oof_folds_pinned_to_five(self):
        assert BreslowSlopeCalibrator.oof_cv_folds == 5
        assert IsotonicCalibrator.oof_cv_folds is None
        assert LowDoFCalibrator.oof_cv_folds is None

    def test_lowdof_empty_oof_falls_back_to_identity_with_warning(self):
        # All-NaN OOF -> every horizon masks out -> no usable samples. The fit
        # must warn and return the identity map rather than a bad (a, b).
        _, y = _toy_survival()
        oof = np.full((len(y), len(self.HORIZONS)), np.nan)
        with pytest.warns(Warning, match="no usable OOF samples"):
            cal = LowDoFCalibrator().fit(
                y_train=y, horizons=self.HORIZONS, oof_S=oof, model_name="m",
            )
        assert (cal.a, cal.b) == (0.0, 1.0)
        probs = _toy_oof_surv(self.HORIZONS)
        out = cal.apply(pipeline=None, X=None, surv_probs=probs,
                        eval_times=self.HORIZONS)
        # Identity map -> output equals clipped input.
        np.testing.assert_allclose(out, np.clip(probs, 1e-6, 1 - 1e-6), atol=1e-9)


class TestApplyCalibratorSafe:
    """The shared guard used by both the learner and the viz/report data-prep.

    Keeps calibrate-or-raw (+ MethodologicalWarning) behaviour identical on
    every surface without the viz layer importing the learner.
    """

    HORIZONS = np.array([0.5, 1.0, 2.0, 3.0])

    def test_none_calibrator_is_passthrough(self):
        from mlsurv.evaluation._calibrators import apply_calibrator_safe
        probs = _toy_oof_surv(self.HORIZONS)
        out = apply_calibrator_safe(
            None, pipeline=None, X=None, surv_probs=probs,
            eval_times=self.HORIZONS, model_name="m",
        )
        np.testing.assert_array_equal(out, probs)

    def test_warper_matches_calibrator_apply(self):
        from mlsurv.evaluation._calibrators import apply_calibrator_safe
        _, y = _toy_survival()
        oof = _toy_oof_surv(self.HORIZONS)
        cal = LowDoFCalibrator().fit(
            y_train=y, horizons=self.HORIZONS, oof_S=oof, model_name="m",
        )
        probs = _toy_oof_surv(self.HORIZONS, seed=7)
        direct = cal.apply(pipeline=None, X=None, surv_probs=probs,
                           eval_times=self.HORIZONS)
        via_safe = apply_calibrator_safe(
            cal, pipeline=None, X=None, surv_probs=probs,
            eval_times=self.HORIZONS, model_name="m",
        )
        np.testing.assert_allclose(via_safe, direct)

    def test_breslow_uses_pipeline_and_X(self):
        from mlsurv.evaluation._calibrators import apply_calibrator_safe
        score, y = _toy_survival()
        cal = BreslowSlopeCalibrator().fit(
            y_train=y, horizons=self.HORIZONS, oof_scores=score, model_name="m",
        )

        class _Pipe:
            def predict(self, X):
                return X

        # surv_probs is ignored by breslow_slope; it recomputes from
        # pipeline.predict(X), so passing None must still work.
        out = apply_calibrator_safe(
            cal, pipeline=_Pipe(), X=score, surv_probs=None,
            eval_times=self.HORIZONS, model_name="m",
        )
        assert out.shape == (len(score), len(self.HORIZONS))
        assert ((out >= 0) & (out <= 1)).all()

    def test_apply_failure_warns_and_falls_back_to_raw(self):
        from mlsurv.evaluation._calibrators import apply_calibrator_safe
        from mlsurv._warnings import MethodologicalWarning

        class _Boom:
            def apply(self, *, pipeline, X, surv_probs, eval_times):
                raise ValueError("kaboom")

        probs = _toy_oof_surv(self.HORIZONS)
        with pytest.warns(MethodologicalWarning, match="failed to apply"):
            out = apply_calibrator_safe(
                _Boom(), pipeline=None, X=None, surv_probs=probs,
                eval_times=self.HORIZONS, model_name="bad",
            )
        np.testing.assert_array_equal(out, probs)

    def test_garbage_return_warns_and_falls_back_to_raw(self):
        # A calibrator that runs without raising but returns a non-finite array
        # where the raw was finite must not silently corrupt the curve.
        from mlsurv.evaluation._calibrators import apply_calibrator_safe
        from mlsurv._warnings import MethodologicalWarning

        probs = _toy_oof_surv(self.HORIZONS)

        class _NaNs:
            def apply(self, *, pipeline, X, surv_probs, eval_times):
                out = np.array(surv_probs, dtype=float)
                out[0, 0] = np.nan
                return out

        with pytest.warns(MethodologicalWarning, match="non-finite or"):
            out = apply_calibrator_safe(
                _NaNs(), pipeline=None, X=None, surv_probs=probs,
                eval_times=self.HORIZONS, model_name="bad",
            )
        np.testing.assert_array_equal(out, probs)

    def test_misshaped_return_warns_and_falls_back_to_raw(self):
        from mlsurv.evaluation._calibrators import apply_calibrator_safe
        from mlsurv._warnings import MethodologicalWarning

        probs = _toy_oof_surv(self.HORIZONS)

        class _WrongShape:
            def apply(self, *, pipeline, X, surv_probs, eval_times):
                return np.asarray(surv_probs, dtype=float)[:, :1]

        with pytest.warns(MethodologicalWarning, match="mis-shaped"):
            out = apply_calibrator_safe(
                _WrongShape(), pipeline=None, X=None, surv_probs=probs,
                eval_times=self.HORIZONS, model_name="bad",
            )
        np.testing.assert_array_equal(out, probs)


class TestPickleRoundTrip:
    HORIZONS = np.array([0.5, 1.0, 2.0])

    def _roundtrip(self, cal):
        buf = io.BytesIO()
        joblib.dump(cal, buf)
        buf.seek(0)
        return joblib.load(buf)

    def test_isotonic_pickles(self):
        _, y = _toy_survival()
        oof = _toy_oof_surv(self.HORIZONS)
        cal = IsotonicCalibrator().fit(
            y_train=y, horizons=self.HORIZONS, oof_S=oof, model_name="m",
        )
        loaded = self._roundtrip(cal)
        a = cal.apply(pipeline=None, X=None, surv_probs=oof, eval_times=self.HORIZONS)
        b = loaded.apply(pipeline=None, X=None, surv_probs=oof, eval_times=self.HORIZONS)
        np.testing.assert_allclose(a, b)

    def test_lowdof_pickles(self):
        _, y = _toy_survival()
        oof = _toy_oof_surv(self.HORIZONS)
        cal = LowDoFCalibrator().fit(
            y_train=y, horizons=self.HORIZONS, oof_S=oof, model_name="m",
        )
        loaded = self._roundtrip(cal)
        assert (loaded.a, loaded.b) == (cal.a, cal.b)

    def test_breslow_pickles(self):
        score, y = _toy_survival()
        cal = BreslowSlopeCalibrator().fit(
            y_train=y, horizons=self.HORIZONS, oof_scores=score, model_name="m",
        )
        loaded = self._roundtrip(cal)
        assert loaded.beta == cal.beta

        class _Pipe:
            def predict(self, X):
                return X
        a = cal.apply(pipeline=_Pipe(), X=score, surv_probs=None, eval_times=self.HORIZONS)
        b = loaded.apply(pipeline=_Pipe(), X=score, surv_probs=None, eval_times=self.HORIZONS)
        np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# Integration: end-to-end through the learner
# ---------------------------------------------------------------------------

class TestReportCalibrationLine:
    """The Model Development report subsection names the attached calibrator."""

    def _subsection(self):
        from mlsurv.reports.sections._trained_details import _TrainedDetailsSubsection
        return _TrainedDetailsSubsection(section=None)

    def _result(self, calibrator, precal=None, post_slope=None):
        import pandas as pd
        from types import SimpleNamespace
        evaluation = None
        if post_slope is not None:
            evaluation = SimpleNamespace(
                concordance=pd.DataFrame({"cal_slope_test": [post_slope]}))
        return SimpleNamespace(
            calibrator=calibrator, precalibration_slope=precal,
            evaluation=evaluation)

    def test_uncalibrated_emits_nothing(self):
        md = []
        self._subsection()._add_calibration_info(md, self._result(None))
        assert md == []

    def test_calibrated_names_method_and_slope(self):
        cal = BreslowSlopeCalibrator()
        cal.method = "breslow_slope"
        md = []
        self._subsection()._add_calibration_info(
            md, self._result(cal, precal=2.84, post_slope=1.76))
        text = "\n".join(md)
        assert "Calibration:" in text
        assert "breslow_slope" in text
        assert "2.84" in text and "1.76" in text

    def test_calibrated_without_eval_omits_slope_line(self):
        cal = LowDoFCalibrator()
        md = []
        self._subsection()._add_calibration_info(
            md, self._result(cal, precal=1.3, post_slope=None))
        text = "\n".join(md)
        assert "Calibration:" in text
        assert "Test calibration slope" not in text


def _gbsg2():
    from sksurv.datasets import load_gbsg2
    Xr, yr = load_gbsg2()
    X = Xr.copy()
    for c in X.select_dtypes(include=["category", "object"]).columns:
        X[c] = X[c].cat.codes.astype(float)
    y = np.array([(bool(e), float(t)) for e, t in yr],
                 dtype=[("event", bool), ("time", float)])
    return X, y


@pytest.mark.integration
class TestTrainCalibrateIntegration:
    def _setup(self):
        from mlsurv import SurvivalLearner
        X, y = _gbsg2()
        lr = SurvivalLearner(X, y)
        lr.setup(feature_selector="topk", random_state=42, n_jobs=1)
        return lr

    def test_registry_defaults_attached(self):
        lr = self._setup()
        lr.train(["coxph", "rsf", "fssvm"], verbose=False)
        assert lr.models["coxph"].calibrator is None
        # rsf is raw by default (dropped from the registry); fssvm calibrated.
        assert lr.models["rsf"].calibrator is None
        assert isinstance(lr.models["fssvm"].calibrator, BreslowSlopeCalibrator)

    def test_rsf_can_opt_in_to_lowdof(self):
        lr = self._setup()
        lr.train(["rsf"], calibrate={"rsf": "lowdof"}, verbose=False)
        assert isinstance(lr.models["rsf"].calibrator, LowDoFCalibrator)

    def test_calibrate_false_is_raw(self):
        lr = self._setup()
        lr.train(["rsf", "fssvm"], calibrate=False, verbose=False)
        assert lr.models["rsf"].calibrator is None
        assert lr.models["fssvm"].calibrator is None

    def test_any_model_can_elect_warper(self):
        lr = self._setup()
        lr.train(["coxph"], calibrate={"coxph": "isotonic"}, verbose=False)
        assert isinstance(lr.models["coxph"].calibrator, IsotonicCalibrator)

    def test_breslow_on_rsf_raises(self):
        lr = self._setup()
        with pytest.raises(ValueError, match="score"):
            lr.train(["rsf"], calibrate={"rsf": "breslow_slope"}, verbose=False)

    def test_cindex_unchanged_and_precalibration_stored(self):
        # fssvm via the default registry; rsf via explicit opt-in (no longer a
        # default). Both are monotone, so the C-index must be unchanged.
        for model, spec in (("rsf", {"rsf": "lowdof"}), ("fssvm", _CALIBRATE_UNSET)):
            raw = self._setup()
            raw.train([model], calibrate=False, verbose=False)
            raw.evaluate([model], verbose=False, show_progress=False)
            cal = self._setup()
            cal.train([model], calibrate=spec, verbose=False)
            cal.evaluate([model], verbose=False, show_progress=False)
            assert cal.models[model].calibrator is not None
            c_raw = raw.models[model].evaluation.c_index_test
            c_cal = cal.models[model].evaluation.c_index_test
            assert abs(c_raw - c_cal) < 1e-6
            assert cal.models[model].precalibration_slope is not None

    def test_save_load_identical_predict(self, tmp_path):
        from mlsurv import SurvivalLearner
        lr = self._setup()
        # rsf opted into lowdof, fssvm via default breslow_slope — covers both
        # a probs-warp and a score->baseline calibrator through save/load.
        lr.train(["rsf", "fssvm"], calibrate={"rsf": "lowdof"}, verbose=False)
        newX = lr._state.X_test.iloc[:5]
        before = {
            m: np.asarray(lr.predict(newX, models=m, verbose=False)._surv_probs, float)
            for m in ("rsf", "fssvm")
        }
        path = tmp_path / "learner.joblib"
        lr.save_learner(str(path))
        loaded = SurvivalLearner.load_learner(str(path))
        for m in ("rsf", "fssvm"):
            after = np.asarray(
                loaded.predict(newX, models=m, verbose=False)._surv_probs, float)
            np.testing.assert_allclose(before[m], after, rtol=1e-6, atol=1e-8)


@pytest.mark.integration
class TestCalibratedBootstrapBand:
    """predict(ci=True, ci_band=True) puts the bootstrap CI band on the same
    (calibrated) scale as the point estimate, for all internal calibrators.

    Warpers (isotonic, lowdof) refit per resample by default
    (band_calibration='refit') and reapply the one fitted probs->probs map
    under band_calibration='fixed'; breslow_slope always refits a
    resample-specific Breslow baseline per iteration.
    """

    def _setup(self):
        from mlsurv import SurvivalLearner
        X, y = _gbsg2()
        lr = SurvivalLearner(X, y)
        lr.setup(feature_selector="topk", random_state=42, n_jobs=1)
        return lr

    @staticmethod
    def _fraction_point_in_band(result):
        ci = result.survival_function_ci
        assert ci is not None
        point = result.survival_function.to_numpy()
        lo, hi = ci["lower"].to_numpy(), ci["upper"].to_numpy()
        inside = (point >= lo - 1e-9) & (point <= hi + 1e-9)
        return float(np.mean(inside))

    @pytest.mark.bootstrap_heavy
    @pytest.mark.parametrize(
        "method, cls_name",
        [("isotonic", "IsotonicCalibrator"), ("lowdof", "LowDoFCalibrator")],
    )
    def test_warper_band_is_calibrated(self, method, cls_name, monkeypatch):
        # coxph is a fast model; under band_calibration='fixed' both warpers
        # reapply a fixed map, so the choice of model is incidental.
        import mlsurv.evaluation._calibrators as cal_mod

        lr = self._setup()
        lr.train(["coxph"], calibrate={"coxph": method}, verbose=False)

        # Spy AFTER training: with band_calibration='fixed' the warper map is
        # fixed, so the bootstrap must REAPPLY it, never refit it. (The 'refit'
        # default is exercised in test_warper_refits_per_iteration.)
        warper_cls = getattr(cal_mod, cls_name)
        fits = {"n": 0}
        orig_fit = warper_cls.fit

        def counting_fit(self, **kwargs):
            fits["n"] += 1
            return orig_fit(self, **kwargs)

        monkeypatch.setattr(warper_cls, "fit", counting_fit)

        newX = lr._state.X_test.iloc[:4]
        common = dict(models="coxph", ci=True, ci_band=True,
                      n_bootstrap=10, random_state=0, verbose=False,
                      band_calibration="fixed")
        cal = lr.predict(newX, use_recalibrated=True, **common)
        raw = lr.predict(newX, use_recalibrated=False, **common)

        # No warper refit happened during predict (fixed map reapplied).
        assert fits["n"] == 0
        # The calibrated point estimate sits inside the calibrated band.
        assert self._fraction_point_in_band(cal) >= 0.9
        # The band actually moved onto the calibrated scale (differs from the
        # raw-scale band rather than staying raw).
        hi_cal = cal.survival_function_ci["upper"].to_numpy()
        hi_raw = raw.survival_function_ci["upper"].to_numpy()
        assert np.nanmax(np.abs(hi_cal - hi_raw)) > 1e-6
        # The raw path is unaffected — it still returns a band, no calibration.
        assert raw.survival_function_ci is not None

    @pytest.mark.bootstrap_heavy
    @pytest.mark.parametrize(
        "method, cls_name",
        [("isotonic", "IsotonicCalibrator"), ("lowdof", "LowDoFCalibrator")],
    )
    def test_warper_refits_per_iteration(self, method, cls_name, monkeypatch):
        # Default band_calibration='refit' replicates the warper fit on every
        # resample, mirroring the breslow path: one fresh warper fit per
        # bootstrap iteration (vs zero on the 'fixed' path above).
        import mlsurv.evaluation._calibrators as cal_mod

        lr = self._setup()
        lr.train(["coxph"], calibrate={"coxph": method}, verbose=False)

        warper_cls = getattr(cal_mod, cls_name)
        calls = {"n": 0}
        orig_fit = warper_cls.fit

        def counting_fit(self, **kwargs):
            calls["n"] += 1
            return orig_fit(self, **kwargs)

        monkeypatch.setattr(warper_cls, "fit", counting_fit)

        n_bootstrap = 4
        newX = lr._state.X_test.iloc[:3]
        common = dict(models="coxph", ci=True, ci_band=True,
                      n_bootstrap=n_bootstrap, random_state=0, verbose=False)
        # band_calibration defaults to 'refit'.
        cal = lr.predict(newX, use_recalibrated=True, **common)
        # One fresh resample-specific warper refit per bootstrap iteration.
        assert calls["n"] == n_bootstrap
        # The band is on the calibrated scale: it differs from the raw band.
        # (Containment is a weak/unreliable check on the refit path — a
        # plateau-prone isotonic refit band can sit off the fixed-map point
        # estimate — so assert the calibration shift directly, like breslow.)
        raw = lr.predict(newX, use_recalibrated=False, **common)
        hi_cal = cal.survival_function_ci["upper"].to_numpy()
        hi_raw = raw.survival_function_ci["upper"].to_numpy()
        assert np.nanmax(np.abs(hi_cal - hi_raw)) > 1e-6

    @pytest.mark.bootstrap_heavy
    def test_warper_refit_band_wider_than_fixed(self):
        # The refit band carries the calibration step's own sampling
        # variability, so it is wider than the fixed-map band (which freezes
        # that step and, on isotonic plateaus, compresses the band). Same
        # random_state -> identical resamples, so the only difference is
        # refit-map vs fixed-map. isotonic is the plateau-prone warper.
        lr = self._setup()
        lr.train(["coxph"], calibrate={"coxph": "isotonic"}, verbose=False)

        newX = lr._state.X_test.iloc[:5]
        common = dict(models="coxph", ci=True, ci_band=True, n_bootstrap=30,
                      random_state=0, use_recalibrated=True, verbose=False)
        refit = lr.predict(newX, band_calibration="refit", **common)
        fixed = lr.predict(newX, band_calibration="fixed", **common)

        def _mean_width(result):
            ci = result.survival_function_ci
            w = ci["upper"].to_numpy() - ci["lower"].to_numpy()
            return float(np.nanmean(w))

        assert _mean_width(refit) > _mean_width(fixed)

    def test_band_calibration_invalid_raises(self):
        # Validation fires before setup/training, so a minimal learner is
        # enough to exercise the guard.
        from mlsurv import SurvivalLearner
        X, y = _gbsg2()
        lr = SurvivalLearner(X, y)
        with pytest.raises(ValueError, match="band_calibration"):
            lr.predict(X.iloc[:2], band_calibration="bogus")

    @pytest.mark.bootstrap_heavy
    def test_breslow_refits_per_iteration(self, monkeypatch):
        import mlsurv.evaluation._calibrators as cal_mod

        lr = self._setup()
        lr.train(["fssvm"], verbose=False)  # default registry -> breslow_slope
        assert isinstance(lr.models["fssvm"].calibrator, BreslowSlopeCalibrator)

        # Spy AFTER training so only the per-resample refits are counted.
        calls = {"n": 0}
        orig_fit = cal_mod.BreslowSlopeCalibrator.fit

        def counting_fit(self, **kwargs):
            calls["n"] += 1
            return orig_fit(self, **kwargs)

        monkeypatch.setattr(
            cal_mod.BreslowSlopeCalibrator, "fit", counting_fit,
        )

        n_bootstrap = 4
        newX = lr._state.X_test.iloc[:3]
        common = dict(models="fssvm", ci=True, ci_band=True,
                      n_bootstrap=n_bootstrap, random_state=0, verbose=False)
        cal = lr.predict(newX, use_recalibrated=True, **common)
        # One fresh resample-specific Breslow refit per bootstrap iteration.
        assert calls["n"] == n_bootstrap
        # The band is on the calibrated scale: it differs from the raw band
        # (the per-resample Breslow refit moves S(t|x) off the raw pipeline
        # curve). Containment is a weak discriminator here — a raw band would
        # often bracket the point too — so assert the shift directly.
        raw = lr.predict(newX, use_recalibrated=False, **common)
        hi_cal = cal.survival_function_ci["upper"].to_numpy()
        hi_raw = raw.survival_function_ci["upper"].to_numpy()
        assert np.nanmax(np.abs(hi_cal - hi_raw)) > 1e-6

    def test_no_raw_band_warning(self):
        import warnings as _warnings

        lr = self._setup()
        lr.train(["coxph"], calibrate={"coxph": "isotonic"}, verbose=False)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            lr.predict(
                lr._state.X_test.iloc[:3], models="coxph", ci=True,
                ci_band=True, n_bootstrap=3, random_state=0, verbose=False,
                use_recalibrated=True,
            )
        offenders = [
            str(w.message) for w in caught
            if "raw S(t|x)" in str(w.message)
            or "band reflects" in str(w.message)
        ]
        assert not offenders, offenders

    def test_uncalibrated_model_band_stays_raw(self):
        # The common production case: a calibrated-band request on a model
        # trained WITHOUT calibrate=. The plan is None and the band is the
        # plain bootstrap band — no calibration, no crash.
        lr = self._setup()
        lr.train(["coxph"], calibrate=False, verbose=False)
        assert lr._build_prediction_calib_plan(lr.models["coxph"]) is None
        newX = lr._state.X_test.iloc[:3]
        common = dict(models="coxph", ci=True, ci_band=True,
                      n_bootstrap=6, random_state=0, verbose=False)
        cal_req = lr.predict(newX, use_recalibrated=True, **common)
        raw = lr.predict(newX, use_recalibrated=False, **common)
        # With no calibrator, use_recalibrated=True and =False are identical.
        np.testing.assert_allclose(
            cal_req.survival_function_ci["upper"].to_numpy(),
            raw.survival_function_ci["upper"].to_numpy(),
            rtol=1e-9, atol=1e-9,
        )

    @pytest.mark.bootstrap_heavy
    def test_ci_band_false_calibrates_median_ci(self):
        # The calibration is applied before deriving medians/RMST, so the
        # median CI is calibrated even when the full band is not retained
        # (ci_band=False). Otherwise the median CI would stay raw.
        lr = self._setup()
        lr.train(["coxph"], calibrate={"coxph": "isotonic"}, verbose=False)
        newX = lr._state.X_test.iloc[:4]
        common = dict(models="coxph", ci=True, ci_band=False,
                      n_bootstrap=10, random_state=0, verbose=False)
        cal = lr.predict(newX, use_recalibrated=True, **common)
        raw = lr.predict(newX, use_recalibrated=False, **common)
        assert cal._bootstrap_surv_probs is None  # band not retained
        # The per-iteration median draws differ once calibration is applied.
        assert np.nanmax(np.abs(
            cal._bootstrap_medians - raw._bootstrap_medians)) > 1e-6

    @pytest.mark.parametrize("model, spec", [
        ("coxph", {"coxph": "isotonic"}),
        ("fssvm", _CALIBRATE_UNSET),  # default -> breslow_slope (carries splitter)
    ])
    def test_calib_plan_is_picklable(self, model, spec):
        # The plan crosses the joblib/loky boundary when n_jobs>1, so every
        # field (fitted warper calibrator, StratifiedKFold splitter, horizon
        # array) must round-trip through pickle.
        import pickle

        lr = self._setup()
        lr.train([model], calibrate=spec, verbose=False)
        plan = lr._build_prediction_calib_plan(lr.models[model])
        assert plan is not None
        restored = pickle.loads(pickle.dumps(plan))
        assert restored["method"] == plan["method"]

    @pytest.mark.bootstrap_heavy
    def test_parallel_njobs_pickles_calib_plan(self):
        # rsf has internal parallelism, so n_bootstrap above the parallel
        # threshold actually dispatches the calib_plan to worker processes.
        # A regression that made the plan unpicklable would raise here.
        lr = self._setup()
        lr.train(["rsf"], calibrate={"rsf": "lowdof"}, verbose=False)
        res = lr.predict(
            lr._state.X_test.iloc[:3], models="rsf", ci=True, ci_band=True,
            n_bootstrap=22, random_state=0, n_jobs=2, verbose=False,
            use_recalibrated=True,
        )
        assert res.survival_function_ci is not None

    @pytest.mark.bootstrap_heavy
    def test_breslow_refit_failure_warns_and_falls_back(self, monkeypatch):
        # When a per-resample breslow refit fails, that iteration is left raw.
        # The worker can't warn (joblib), so the parent surfaces a count.
        import warnings as _warnings
        import mlsurv.evaluation._recalibration as recal_mod
        from mlsurv._warnings import SuggestionWarning

        lr = self._setup()
        lr.train(["fssvm"], verbose=False)  # default -> breslow_slope

        def boom(*args, **kwargs):
            raise ValueError("forced refit failure")

        # _build_iter_calibrate imports this name locally per call, so patching
        # the module attribute takes effect (predict runs n_jobs=1, in-process).
        monkeypatch.setattr(recal_mod, "_collect_oof_survival", boom)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            res = lr.predict(
                lr._state.X_test.iloc[:3], models="fssvm", ci=True,
                ci_band=True, n_bootstrap=4, random_state=0, verbose=False,
                use_recalibrated=True,
            )
        # Band is still produced (degraded, not crashed) ...
        assert res.survival_function_ci is not None
        # ... and the skip is surfaced from the parent.
        assert any(
            isinstance(w.message, SuggestionWarning)
            and "left on the raw" in str(w.message)
            for w in caught
        )

    @pytest.mark.bootstrap_heavy
    @pytest.mark.parametrize("method", ["isotonic", "lowdof"])
    def test_warper_refit_failure_warns_and_falls_back(self, method, monkeypatch):
        # Default band_calibration='refit' refits the warper per resample. When
        # that refit fails, the iteration is left raw; the worker can't warn
        # (joblib), so the parent surfaces a count. Mirrors the breslow case.
        import warnings as _warnings
        import mlsurv.evaluation._recalibration as recal_mod
        from mlsurv._warnings import SuggestionWarning

        lr = self._setup()
        lr.train(["coxph"], calibrate={"coxph": method}, verbose=False)

        def boom(*args, **kwargs):
            raise ValueError("forced refit failure")

        # The warper refit branch imports this name locally per call, so
        # patching the module attribute takes effect (predict runs n_jobs=1).
        monkeypatch.setattr(recal_mod, "_collect_oof_survival", boom)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            res = lr.predict(
                lr._state.X_test.iloc[:3], models="coxph", ci=True,
                ci_band=True, n_bootstrap=4, random_state=0, verbose=False,
                use_recalibrated=True,
            )
        # Band is still produced (degraded, not crashed) ...
        assert res.survival_function_ci is not None
        # ... and the skip is surfaced from the parent.
        assert any(
            isinstance(w.message, SuggestionWarning)
            and "left on the raw" in str(w.message)
            for w in caught
        )

    def test_warper_refit_no_splitter_falls_back_to_fixed(self, monkeypatch):
        # A 'refit' warper request degrades to the fixed-map plan (band stays on
        # the calibrated scale) rather than dropping to raw when no OOF splitter
        # is available. Drive _calibration_splitter to None and assert the plan
        # carries the fitted calibrator, not a refit recipe. Pure plan-shape
        # assertion, so no bootstrap needed.
        import warnings as _warnings
        from mlsurv._warnings import SuggestionWarning

        lr = self._setup()
        lr.train(["coxph"], calibrate={"coxph": "isotonic"}, verbose=False)
        monkeypatch.setattr(lr, "_calibration_splitter", lambda *a, **k: None)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            plan = lr._build_prediction_calib_plan(
                lr.models["coxph"], band_calibration="refit",
            )
        assert plan["method"] == "isotonic"
        assert "calibrator" in plan        # fixed shape, not a refit recipe
        assert "splitter" not in plan
        # The silent downgrade is surfaced to the user.
        assert any(
            isinstance(w.message, SuggestionWarning)
            and "no out-of-fold splitter" in str(w.message)
            for w in caught
        )


def _predicted(df):
    """Pull the predicted-curve column from a plot return_data frame."""
    return np.asarray(df["predicted"], dtype=float)


def _raw_overlay_mean(model, X_test, eval_times):
    """Independent (no-calibrator) mean predicted S(t) — the raw-anchor."""
    from mlsurv.evaluation.metrics import _surv_funcs_to_probs
    sf = model.predict_survival_function(X_test)
    return _surv_funcs_to_probs(sf, np.asarray(eval_times)).mean(axis=0)


@pytest.fixture(scope="module")
def _calibrated_plot_learner():
    """Train once: coxph (breslow_slope), rsf (lowdof warper), gbsa (raw).

    Covers both calibrator families end-to-end through the plots: the
    score->baseline path (coxph/breslow_slope, recomputes from pipeline.predict)
    and the probability-warper path (rsf/lowdof, maps surv_probs). gbsa is the
    uncalibrated control.
    """
    from mlsurv import SurvivalLearner
    X, y = _gbsg2()
    lr = SurvivalLearner(X, y)
    lr.setup(feature_selector="topk", random_state=42, n_jobs=1)
    lr.run(["coxph", "rsf", "gbsa"],
           calibrate={"coxph": "breslow_slope", "rsf": "lowdof"},
           tune=False, verbose=False)
    lr.evaluate(verbose=False, show_progress=False, n_importance_repeats=0)
    return lr


@pytest.mark.integration
class TestCalibratedPlots:
    """Calibration plots + metric-time Brier(t) show the calibrated curve by
    default for a model with an attached calibrator, and raw=True opts out.
    A model with no calibrator is identical either way."""

    # --- breslow_slope (score->baseline) path --------------------------------

    def test_plot_calibration_calibrated_by_default_breslow(
            self, _calibrated_plot_learner):
        lr = _calibrated_plot_learner
        cal = lr.plot_calibration(models="coxph", return_data=True)
        raw = lr.plot_calibration(models="coxph", return_data=True, raw=True)
        # Calibrating warps the predicted S(t|x), which re-bins the curve, so the
        # frames can differ in row count too — assert the predicted curve moves.
        assert not cal.empty and not raw.empty
        assert abs(_predicted(cal).mean() - _predicted(raw).mean()) > 1e-3

    # --- lowdof (probability-warper) path ------------------------------------

    def test_plot_calibration_calibrated_by_default_warper(
            self, _calibrated_plot_learner):
        # rsf/lowdof exercises the warper branch (apply_calibrator_safe maps the
        # passed-in surv_probs columns) — a structurally different code path from
        # breslow's score recompute.
        lr = _calibrated_plot_learner
        cal = lr.plot_calibration(models="rsf", return_data=True)
        raw = lr.plot_calibration(models="rsf", return_data=True, raw=True)
        assert not cal.empty and not raw.empty
        assert abs(_predicted(cal).mean() - _predicted(raw).mean()) > 1e-3

    def test_plot_calibration_overlay_warper(self, _calibrated_plot_learner):
        lr = _calibrated_plot_learner
        cal = lr.plot_calibration_overlay(models="rsf", return_data=True)
        raw = lr.plot_calibration_overlay(models="rsf", return_data=True, raw=True)
        assert np.nanmax(np.abs(_predicted(cal) - _predicted(raw))) > 1e-6

    # --- raw=True is the genuinely-uncalibrated curve, not a third curve ------

    def test_overlay_raw_equals_independent_raw_curve(
            self, _calibrated_plot_learner):
        # raw=True must reproduce the uncalibrated predicted curve exactly (no
        # residual/double application), anchored to an independent computation.
        lr = _calibrated_plot_learner
        for name in ("coxph", "rsf"):
            raw = lr.plot_calibration_overlay(models=name, return_data=True, raw=True)
            anchor = _raw_overlay_mean(
                lr._state.models[name].model,
                lr._state.X_test, raw["eval_time"].to_numpy(),
            )
            np.testing.assert_allclose(_predicted(raw), anchor, rtol=1e-6, atol=1e-9)

    def test_plot_calibration_raw_for_uncalibrated_model_unchanged(
            self, _calibrated_plot_learner):
        lr = _calibrated_plot_learner
        cal = lr.plot_calibration(models="gbsa", return_data=True)
        raw = lr.plot_calibration(models="gbsa", return_data=True, raw=True)
        np.testing.assert_allclose(_predicted(cal), _predicted(raw))

    def test_plot_calibration_overlay_calibrated_by_default(
            self, _calibrated_plot_learner):
        lr = _calibrated_plot_learner
        cal = lr.plot_calibration_overlay(models="coxph", return_data=True)
        raw = lr.plot_calibration_overlay(models="coxph", return_data=True, raw=True)
        assert np.nanmax(np.abs(_predicted(cal) - _predicted(raw))) > 1e-6

    # --- metric-time: Brier(t) calibrates, AUC(t) must NOT --------------------

    def test_metric_time_brier_calibrated_by_default(
            self, _calibrated_plot_learner):
        lr = _calibrated_plot_learner
        cal = lr.plot_metric_time(metric="brier", return_data=True)
        raw = lr.plot_metric_time(metric="brier", return_data=True, raw=True)
        model_col = "model" if "model" in cal.columns else "model_type"
        keys = [c for c in cal.columns if c != "value"]
        merged = cal.merge(raw, on=keys, suffixes=("_cal", "_raw"))
        # calibrated models' Brier(t) must move; gbsa (raw) rows stay put.
        for m in ("coxph", "rsf"):
            rows = merged[merged[model_col] == m]
            assert np.nanmax(np.abs(rows["value_cal"] - rows["value_raw"])) > 1e-6
        gbsa = merged[merged[model_col] == "gbsa"]
        np.testing.assert_allclose(
            gbsa["value_cal"].to_numpy(), gbsa["value_raw"].to_numpy())

    def test_metric_time_auc_unaffected_by_calibration(
            self, _calibrated_plot_learner):
        # AUC(t) is rank-based: a monotone calibrator can't move it, and it has
        # no raw switch. In a multi-panel call the auc rows must be identical
        # whether or not Brier is calibrated.
        lr = _calibrated_plot_learner
        cal = lr.plot_metric_time(metric=["auc", "brier"], return_data=True)
        raw = lr.plot_metric_time(metric=["auc", "brier"], return_data=True, raw=True)
        auc_cal = cal[cal["metric_name"] == "auc"].reset_index(drop=True)
        auc_raw = raw[raw["metric_name"] == "auc"].reset_index(drop=True)
        np.testing.assert_array_equal(
            auc_cal["value"].to_numpy(), auc_raw["value"].to_numpy())


@pytest.fixture(scope="module")
def _validated_cal_learner():
    """Train coxph (breslow_slope) + validate on an external GBSG2 slice."""
    from mlsurv import SurvivalLearner
    X, y = _gbsg2()
    lr = SurvivalLearner(X, y)
    lr.setup(feature_selector="topk", random_state=42, n_jobs=1)
    lr.run(["coxph"], calibrate={"coxph": "breslow_slope"},
           tune=False, verbose=False)
    lr.evaluate(verbose=False, show_progress=False, n_importance_repeats=0)
    lr.validate(X.iloc[:150], y[:150], name="Holdout")
    return lr


@pytest.mark.integration
class TestCalibratedPlotsValidationCohort:
    """The calibrator routes to the validation-cohort calibration plot too,
    matching validate()'s calibrated metrics (validate() applies the same
    calibrator)."""

    def test_validation_calibration_calibrated_by_default(
            self, _validated_cal_learner):
        view = _validated_cal_learner.validation("Holdout")
        cal = view.plot_calibration(models="coxph", return_data=True)
        raw = view.plot_calibration(models="coxph", return_data=True, raw=True)
        assert not cal.empty and not raw.empty
        assert abs(_predicted(cal).mean() - _predicted(raw).mean()) > 1e-3
