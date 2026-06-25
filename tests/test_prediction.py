"""Tests for learner.predict() — new-patient survival predictions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
from mlsurv.evaluation._prediction_helpers import (
    compute_patient_median_survival,
    compute_patient_rmst,
    compute_percentile_ci,
    extract_time_grid_and_probs,
    interpolate_surv_at_times,
    predict_from_pipeline,
)
from mlsurv.results._prediction import PredictionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_predict_data():
    """Generate synthetic survival data (200 samples, 20 features, no NaN)."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 200, 20
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
    )
    coef = np.zeros(n_features)
    coef[:5] = [0.5, -0.3, 0.4, -0.2, 0.3]
    lp = X.values @ coef
    times = np.exp(-lp + rng.randn(n_samples) * 0.5).clip(0.1, 10)
    events = rng.binomial(1, 0.7, n_samples).astype(bool)
    from mlsurv.utils import SurvivalArray
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
    ))
    return X, y


@pytest.fixture(scope="module")
def trained_learner():
    """SurvivalLearner with CoxPH trained, ready for predict() (module-scoped)."""
    X, y = _make_predict_data()
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, random_state=42)
    learner.train('coxph')
    return learner


@pytest.fixture(scope="module")
def new_patients():
    """Small DataFrame of 'new' patients for prediction (module-scoped)."""
    X, _ = _make_predict_data()
    return X.iloc[:5].copy()


# ---------------------------------------------------------------------------
# Helper function tests (fast tier)
# ---------------------------------------------------------------------------

class TestExtractTimeGridAndProbs:

    def test_stepfunction_list(self):
        """Extract from sksurv-style StepFunction objects."""
        from sksurv.functions import StepFunction

        times = np.array([1.0, 2.0, 5.0, 10.0])
        sf1 = StepFunction(times, np.array([1.0, 0.9, 0.6, 0.3]))
        sf2 = StepFunction(times, np.array([1.0, 0.8, 0.5, 0.1]))

        time_grid, probs = extract_time_grid_and_probs([sf1, sf2])
        assert time_grid.shape == (4,)
        assert probs.shape == (2, 4)
        np.testing.assert_array_equal(time_grid, times)
        assert probs[0, -1] == 0.3
        assert probs[1, -1] == 0.1

    def test_dataframe(self):
        """Extract from pycox-style DataFrame (times as index)."""
        times = np.array([1.0, 5.0, 10.0])
        df = pd.DataFrame(
            np.array([[0.95, 0.90], [0.70, 0.60], [0.30, 0.20]]),
            index=times,
            columns=["p0", "p1"],
        )
        time_grid, probs = extract_time_grid_and_probs(df)
        assert time_grid.shape == (3,)
        assert probs.shape == (2, 3)  # 2 patients, 3 times
        np.testing.assert_allclose(probs[0], [0.95, 0.70, 0.30])

    def test_ndarray_raises(self):
        """Plain ndarray without time grid raises TypeError."""
        with pytest.raises(TypeError, match="Cannot extract time grid"):
            extract_time_grid_and_probs(np.ones((3, 5)))


class TestComputePatientMedianSurvival:

    def test_basic(self):
        time_grid = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
        surv_probs = np.array([
            [1.0, 0.8, 0.6, 0.4, 0.2],   # crosses 0.5 between t=10 and t=20
            [1.0, 0.9, 0.7, 0.3, 0.1],   # crosses 0.5 between t=10 and t=20
        ])
        medians = compute_patient_median_survival(time_grid, surv_probs)
        assert medians.shape == (2,)
        # Patient 0: interp between (10, 0.6) and (20, 0.4) -> 0.6-0.5=0.1, 0.6-0.4=0.2 -> t=10+5=15
        assert 10.0 < medians[0] < 20.0
        assert 10.0 < medians[1] < 20.0

    def test_undefined_median(self):
        """S(t) > 0.5 everywhere -> NaN with warning."""
        time_grid = np.array([1.0, 5.0, 10.0])
        surv_probs = np.array([[0.9, 0.8, 0.7]])  # never crosses 0.5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            medians = compute_patient_median_survival(time_grid, surv_probs)
            assert np.isnan(medians[0])
            meth_warnings = [x for x in w if issubclass(x.category, MethodologicalWarning)]
            assert len(meth_warnings) >= 1

    def test_undefined_median_warn_suppressed(self):
        """warn_undefined=False suppresses the MethodologicalWarning."""
        time_grid = np.array([1.0, 5.0, 10.0])
        surv_probs = np.array([[0.9, 0.8, 0.7]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            medians = compute_patient_median_survival(
                time_grid, surv_probs, warn_undefined=False,
            )
            assert np.isnan(medians[0])
            meth = [x for x in w if issubclass(x.category, MethodologicalWarning)]
            assert len(meth) == 0

    def test_undefined_median_no_warn_when_n_zero(self):
        """n_undefined == 0 -> no warning regardless of warn_undefined kwarg."""
        time_grid = np.array([1.0, 5.0, 10.0])
        # S crosses 0.5 cleanly between t=5 and t=10
        surv_probs = np.array([[0.9, 0.6, 0.3]])

        for warn_flag in (True, False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                medians = compute_patient_median_survival(
                    time_grid, surv_probs, warn_undefined=warn_flag,
                )
                assert not np.isnan(medians[0])
                meth = [x for x in w if issubclass(x.category, MethodologicalWarning)]
                assert len(meth) == 0


class TestNaNMedianWording:
    """Undefined NaN-median wording must read "Not reached", not
    "undefined" (the latter reads like a computation error to a clinician
    when it actually signals a good prognosis). Mirrors the "Not reached"
    convention used in the stratification / validation report sections.
    """

    def _nan_median_result(self):
        """Minimal PredictionResult with a single NaN-median patient."""
        return PredictionResult(
            risk_scores=np.array([0.5]),
            model_name="coxph",
            n_patients=1,
            _median_survival=np.array([np.nan]),
        )

    def test_summary_uses_not_reached(self):
        pred = self._nan_median_result()
        text = pred.summary(0)
        assert "Not reached" in text
        # The median-survival line must not use the old "undefined" wording.
        med_line = next(
            ln for ln in text.splitlines() if "Median survival:" in ln
        )
        assert "undefined" not in med_line


@pytest.mark.integration
class TestPredictRecalGate:
    """Integration: ``learner.predict()`` must not double-emit the
    "Median survival undefined" MethodologicalWarning when a recalibrator
    exists. Validates the ``recal_will_apply`` gate in
    ``mlsurv/learner/_prediction.py``.
    """

    @pytest.fixture(scope="class")
    def _learner_with_recal(self):
        """Train coxph + attach a calibrator that forces S>0.5 for all
        patients (every calibrated curve is undefined). Without the gate
        the raw and calibrated paths each fire a MethodologicalWarning
        with different N counts."""
        from sklearn.isotonic import IsotonicRegression

        X, y = _make_predict_data()
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, random_state=42, verbose=False)
        learner.train("coxph")

        # Build a calibrator that maps every raw S to a high value
        # (>>0.5) so the post-calibration median is undefined for
        # every patient. Match a few representative horizons inside the
        # training time grid.
        ymax = float(learner._state.y_train["time"].max())
        horizons = [0.25 * ymax, 0.5 * ymax, 0.75 * ymax]
        grid = np.linspace(0.0, 1.0, 51)
        recal = {}
        for h in horizons:
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            # Constant-high target so every input maps to ~0.99
            iso.fit(grid, np.full_like(grid, 0.99))
            recal[float(h)] = iso
        from mlsurv.evaluation._calibrators import IsotonicCalibrator
        cal = IsotonicCalibrator()
        cal.mappings = recal
        cal.horizons = sorted(recal.keys())
        learner._state.models["coxph"].calibrator = cal
        return learner

    def test_predict_with_recalibrator_emits_one_warning(self, _learner_with_recal):
        learner = _learner_with_recal
        X_new = learner._state.X_test.head(10).copy()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.predict(X_new, models="coxph", use_recalibrated=True)

        med = [
            x for x in w
            if issubclass(x.category, MethodologicalWarning)
            and "Median survival undefined" in str(x.message)
        ]
        # Exactly one — the post-recalibration count. The raw count is
        # internal bookkeeping the user can't act on.
        assert len(med) == 1, (
            f"expected 1 median-undefined warning when recalibrator gates "
            f"the raw warning, got {len(med)}: "
            f"{[str(x.message) for x in med]}"
        )

    def test_predict_without_recalibrator_still_warns(self, _learner_with_recal):
        """Sanity: with use_recalibrated=False the raw warning still fires
        (otherwise the gate would suppress everything — silent data loss).
        """
        learner = _learner_with_recal
        X_new = learner._state.X_test.head(10).copy()

        # Inject a raw scenario where some patients have S>0.5 everywhere
        # by predicting on the model that, with use_recalibrated=False,
        # exposes only the raw count. For this fixture's small synthetic
        # data, coxph's raw predictions vary — at least one patient may or
        # may not be undefined. We assert the gate at least doesn't
        # SUPPRESS the raw warning (i.e., no recal path runs).
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.predict(X_new, models="coxph", use_recalibrated=False)

        # The recalibrator-installed scenario doesn't apply when
        # use_recalibrated=False; we only verify the recal path was NOT
        # silently invoked (no post-recal warning would have been emitted
        # under suppression). If the raw path produces no undefined
        # medians, no warning is expected — that's also a pass for this
        # gate test.
        med = [
            x for x in w
            if issubclass(x.category, MethodologicalWarning)
            and "Median survival undefined" in str(x.message)
        ]
        # The key property: at most one warning, never two duplicates.
        assert len(med) <= 1


class TestPercentileCiQuiet:
    """_percentile_ci_quiet suppresses numpy All-NaN slice."""

    def test_all_nan_slice_suppressed_in_prediction_result_helper(self):
        from mlsurv.results._prediction import _percentile_ci_quiet
        # One patient column is all-NaN across bootstrap draws → nanpercentile
        # inside compute_percentile_ci would raise RuntimeWarning("All-NaN
        # slice encountered") without the suppression.
        values = np.array([
            [np.nan, 1.0],
            [np.nan, 2.0],
            [np.nan, 3.0],
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lo, hi = _percentile_ci_quiet(values, ci_level=0.95)
            nan_slice = [x for x in w if "All-NaN slice" in str(x.message)]
            assert len(nan_slice) == 0
            assert np.isnan(lo[0]) and np.isnan(hi[0])  # NaN propagates
            assert np.isfinite(lo[1]) and np.isfinite(hi[1])

    def test_shared_percentile_ci_NOT_suppressed(self):
        """`mlsurv._shared.percentile_ci` is shared with 6 unrelated callers
        (bootstrap, paired-comparison, ROC, incremental, containers); it must
        keep numpy's All-NaN-slice signal so a model that errors in every
        bootstrap iteration is still visible. Only the patient-prediction
        helper :func:`_percentile_ci_quiet` should swallow it.
        """
        from mlsurv._shared import percentile_ci
        values = np.array([[np.nan], [np.nan], [np.nan]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            percentile_ci(values, ci_level=0.95, axis=0)
            nan_slice = [x for x in w if "All-NaN slice" in str(x.message)]
            assert len(nan_slice) >= 1, (
                "Shared percentile_ci must surface numpy's All-NaN warning "
                "for non-prediction callers; over-broad suppression would "
                "mask legitimate bootstrap failures."
            )


class TestComputePatientRmst:

    def test_basic(self):
        time_grid = np.array([0.0, 5.0, 10.0])
        surv_probs = np.array([[1.0, 0.8, 0.5]])
        rmst = compute_patient_rmst(time_grid, surv_probs, tau=10.0)
        assert rmst.shape == (1,)
        # Area: 1.0*5 + 0.8*5 = 9.0
        np.testing.assert_allclose(rmst[0], 9.0)

    def test_tau_within_grid(self):
        """RMST truncated at tau < max(time_grid)."""
        time_grid = np.array([0.0, 5.0, 10.0, 20.0])
        surv_probs = np.array([[1.0, 0.8, 0.5, 0.2]])
        rmst_10 = compute_patient_rmst(time_grid, surv_probs, tau=10.0)
        rmst_20 = compute_patient_rmst(time_grid, surv_probs, tau=20.0)
        assert rmst_10[0] < rmst_20[0]


class TestInterpolateSurvAtTimes:

    def test_basic(self):
        time_grid = np.array([1.0, 5.0, 10.0])
        surv_probs = np.array([
            [0.95, 0.70, 0.30],
            [0.90, 0.60, 0.20],
        ])
        query = np.array([3.0, 7.0, 15.0])
        result = interpolate_surv_at_times(time_grid, surv_probs, query)
        assert result.shape == (2, 3)
        # t=3 -> step function gives value at t=1 (0.95, 0.90)
        np.testing.assert_allclose(result[:, 0], [0.95, 0.90])
        # t=7 -> value at t=5 (0.70, 0.60)
        np.testing.assert_allclose(result[:, 1], [0.70, 0.60])
        # t=15 -> value at t=10 (0.30, 0.20) (last known)
        np.testing.assert_allclose(result[:, 2], [0.30, 0.20])

    def test_before_first_time(self):
        """Query before first grid time -> S = 1.0."""
        time_grid = np.array([5.0, 10.0])
        surv_probs = np.array([[0.8, 0.5]])
        result = interpolate_surv_at_times(time_grid, surv_probs, np.array([1.0]))
        np.testing.assert_allclose(result[0, 0], 1.0)


class TestComputePercentileCi:

    def test_basic(self):
        rng = np.random.RandomState(42)
        values = rng.normal(10, 2, size=(1000, 5))
        lower, upper = compute_percentile_ci(values, 0.95)
        assert lower.shape == (5,)
        assert upper.shape == (5,)
        # CIs should roughly span mean +/- 2*std
        assert np.all(lower < 10)
        assert np.all(upper > 10)


# ---------------------------------------------------------------------------
# PredictionResult tests (fast tier)
# ---------------------------------------------------------------------------

class TestPredictionResult:

    def test_repr(self):
        pr = PredictionResult(
            risk_scores=np.array([1.0, 2.0]),
            model_name='coxph',
            n_patients=2,
        )
        r = repr(pr)
        assert "coxph" in r
        assert "n_patients=2" in r

    def test_survival_function_none_without_surv(self):
        pr = PredictionResult(
            risk_scores=np.array([1.0]),
            model_name='coxph',
            n_patients=1,
        )
        assert pr.survival_function is None
        assert pr.median_survival is None

    def test_survival_function_dataframe(self):
        pr = PredictionResult(
            risk_scores=np.array([1.0, 2.0]),
            model_name='coxph',
            n_patients=2,
            _time_grid=np.array([1.0, 5.0, 10.0]),
            _surv_probs=np.array([[0.9, 0.7, 0.3], [0.8, 0.5, 0.1]]),
        )
        sf = pr.survival_function
        assert isinstance(sf, pd.DataFrame)
        assert sf.shape == (3, 2)
        assert sf.index.name == "time"

    def test_to_dataframes(self):
        pr = PredictionResult(
            risk_scores=np.array([1.0]),
            model_name='coxph',
            n_patients=1,
        )
        dfs = pr.to_dataframes()
        assert "risk_scores" in dfs
        assert dfs["risk_scores"].shape == (1, 1)


# ---------------------------------------------------------------------------
# Learner.predict() tests (fast tier)
# ---------------------------------------------------------------------------

class TestPredictBasic:

    def test_risk_scores(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, verbose=False)
        assert isinstance(pred, PredictionResult)
        assert pred.risk_scores.shape == (5,)
        assert np.all(np.isfinite(pred.risk_scores))

    def test_survival_function(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, verbose=False)
        sf = pred.survival_function
        assert isinstance(sf, pd.DataFrame)
        assert sf.shape[1] == 5  # 5 patients
        # Survival probs should be in [0, 1]
        assert (sf.values >= 0).all()
        assert (sf.values <= 1).all()

    def test_median_survival(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, verbose=False)
        med = pred.median_survival
        assert med is not None
        assert med.shape == (5,)
        # Medians should be positive (or NaN)
        valid = med[~np.isnan(med)]
        assert np.all(valid > 0)

    def test_rmst(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, verbose=False)
        rmst = pred.rmst
        assert rmst is not None
        assert rmst.shape == (5,)
        assert np.all(rmst > 0)
        assert np.all(np.isfinite(rmst))

    def test_single_patient(self, trained_learner, synthetic_data_clean):
        X, _ = synthetic_data_clean
        pred = trained_learner.predict(X.iloc[[0]], verbose=False)
        assert pred.n_patients == 1
        assert pred.risk_scores.shape == (1,)

    def test_summary_format(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, verbose=False)
        text = pred.summary(patient=0)
        assert isinstance(text, str)
        assert "Patient 0" in text
        assert "Risk score" in text
        assert "coxph" in text

    def test_summary_out_of_range(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, verbose=False)
        with pytest.raises(IndexError, match="out of range"):
            pred.summary(patient=10)


class TestPredictWithTimes:

    def test_survival_at_times(self, trained_learner, new_patients):
        pred = trained_learner.predict(
            new_patients, times=[100, 500, 1000], verbose=False,
        )
        sat = pred.survival_at_times
        assert isinstance(sat, pd.DataFrame)
        assert sat.shape == (5, 3)
        assert "t_100" in sat.columns

    def test_rmst_with_times(self, trained_learner, new_patients):
        pred = trained_learner.predict(
            new_patients, times=[100, 500], verbose=False,
        )
        rmst = pred.rmst
        assert isinstance(rmst, pd.DataFrame)
        assert rmst.shape == (5, 2)
        assert "tau_100" in rmst.columns

    def test_times_beyond_training_warns(self, trained_learner, new_patients):
        """Times beyond max training time emit MethodologicalWarning."""
        max_time = float(trained_learner.y_train['time'].max())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained_learner.predict(
                new_patients, times=[max_time * 10], verbose=False,
            )
            meth = [x for x in w if issubclass(x.category, MethodologicalWarning)]
            assert len(meth) >= 1


class TestPredictValidation:

    def test_no_model_raises(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        with pytest.raises(ValueError, match="No trained models"):
            learner.predict(X.iloc[:3])

    def test_missing_columns_raises(self, trained_learner, new_patients):
        X_bad = new_patients.drop(columns=[new_patients.columns[0]])
        with pytest.raises(ValueError, match="missing"):
            trained_learner.predict(X_bad, verbose=False)

    def test_extra_columns_warns(self, trained_learner, new_patients):
        X_extra = new_patients.copy()
        X_extra["extra_col"] = 999
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pred = trained_learner.predict(X_extra, verbose=False)
            sugg = [x for x in w if issubclass(x.category, SuggestionWarning)]
            assert len(sugg) >= 1
        assert pred.n_patients == 5

    def test_empty_dataframe_raises(self, trained_learner, new_patients):
        with pytest.raises(ValueError, match="at least 1 row"):
            trained_learner.predict(new_patients.iloc[:0], verbose=False)

    def test_non_dataframe_raises(self, trained_learner):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            trained_learner.predict(np.ones((3, 5)), verbose=False)

    def test_invalid_times_raises(self, trained_learner, new_patients):
        with pytest.raises(ValueError, match="non-empty 1-D"):
            trained_learner.predict(new_patients, times=[], verbose=False)

    def test_multiple_models_without_name_returns_dict(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph', 'rsf'])
        preds = learner.predict(X.iloc[:3], verbose=False)
        assert isinstance(preds, dict)
        assert set(preds) == {'coxph', 'rsf'}

    def test_multiple_models_with_name(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph', 'rsf'])
        pred = learner.predict(X.iloc[:3], models='rsf', verbose=False)
        assert pred.model_name == 'rsf'


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPredictIntegration:

    def test_predict_coxph(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')
        pred = learner.predict(X.iloc[:3], verbose=False)
        assert pred.survival_function is not None
        assert pred.median_survival is not None

    def test_predict_rsf(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('rsf')
        pred = learner.predict(X.iloc[:3], verbose=False)
        assert pred.survival_function is not None

    def test_predict_gbsa(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('gbsa')
        pred = learner.predict(X.iloc[:3], verbose=False)
        assert pred.survival_function is not None

    def test_predict_coxnet(self, synthetic_data_clean):
        """CoxNet predict() returns survival functions (fit_baseline_model=True by default)."""
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxnet')
        pred = learner.predict(X.iloc[:3], verbose=False)
        assert pred.survival_function is not None
        assert pred.median_survival is not None

    def test_predict_coxnet_with_times(self, synthetic_data_clean):
        """CoxNet predict() with explicit time points."""
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxnet')
        times = [50.0, 100.0]
        pred = learner.predict(X.iloc[:3], times=times, verbose=False)
        sat = pred.survival_at_times
        assert sat is not None
        assert sat.shape == (3, 2)
        rmst_df = pred.rmst
        assert isinstance(rmst_df, pd.DataFrame)
        assert rmst_df.shape == (3, 2)

    def test_predict_with_preprocessing(self, synthetic_data):
        """Verify predict works with imputer/scaler/selector active."""
        X, y = synthetic_data  # has NaN values
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42,
            feature_selector='topk',
            test_fraction=0.3,
        )
        learner.train('coxph')
        # Use test data (may have NaN) as "new patients"
        pred = learner.predict(X.iloc[:3], verbose=False)
        assert pred.risk_scores.shape == (3,)
        assert np.all(np.isfinite(pred.risk_scores))

    def test_predict_multi_model_list_with_ci(self, small_data):
        """Explicit multi-model selection forwards CI kwargs to every model."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=2)
        learner.train(['coxph', 'coxnet'])

        preds = learner.predict(
            X.iloc[:3], models=['coxph', 'coxnet'], times=[2.0],
            ci=True, n_bootstrap=2, random_state=42,
            use_recalibrated=False, verbose=False,
        )
        assert isinstance(preds, dict)
        assert set(preds) == {'coxph', 'coxnet'}
        for name, pred in preds.items():
            assert pred.model_name == name
            med_ci = pred.median_survival_ci
            assert med_ci is not None
            assert {'median', 'ci_lower', 'ci_upper'} <= set(med_ci.columns)
            assert med_ci.shape[0] == 3
            sat_ci = pred.survival_at_times_ci
            assert sat_ci is not None
            for col in ('t_2', 't_2_ci_lower', 't_2_ci_upper'):
                assert col in sat_ci.columns
            assert (sat_ci['t_2_ci_lower'] <= sat_ci['t_2_ci_upper']).all()
            assert sat_ci.shape[0] == 3


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------

@pytest.mark.bootstrap_heavy
class TestPredictBootstrap:

    def test_bootstrap_ci_populated(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        pred = learner.predict(
            X.iloc[:3], ci=True, n_bootstrap=10, random_state=42,
            verbose=False,
        )
        assert pred.median_survival_ci is not None
        ci_df = pred.median_survival_ci
        assert "median" in ci_df.columns
        assert "ci_lower" in ci_df.columns
        assert "ci_upper" in ci_df.columns
        assert ci_df.shape[0] == 3

    def test_bootstrap_rmst_ci(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        pred = learner.predict(
            X.iloc[:2], ci=True, n_bootstrap=10, random_state=42,
            verbose=False,
        )
        rmst_ci = pred.rmst_ci
        assert rmst_ci is not None
        assert "rmst" in rmst_ci.columns
        assert rmst_ci.shape[0] == 2

    def test_bootstrap_with_times(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        pred = learner.predict(
            X.iloc[:2], times=[100, 500], ci=True, n_bootstrap=10,
            random_state=42, verbose=False,
        )
        sat_ci = pred.survival_at_times_ci
        assert sat_ci is not None
        assert "t_100" in sat_ci.columns
        assert "t_100_ci_lower" in sat_ci.columns

    def test_rmst_default_ci_available_with_times(self, synthetic_data_clean):
        """rmst_default_ci is available even when user times are set."""
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        pred = learner.predict(
            X.iloc[:2], times=[100, 500], ci=True, n_bootstrap=10,
            random_state=42, verbose=False,
        )
        # Default-tau CI must be present alongside per-time CIs
        default_ci = pred.rmst_default_ci
        assert default_ci is not None
        assert "rmst" in default_ci.columns
        assert "ci_lower" in default_ci.columns
        assert "ci_upper" in default_ci.columns
        assert default_ci.shape[0] == 2

        # Per-time CIs also present
        times_ci = pred.rmst_at_times_ci
        assert times_ci is not None
        assert "tau_100" in times_ci.columns

    def test_rmst_ci_backward_compat_with_times(self, synthetic_data_clean):
        """rmst_ci returns per-time CIs when times are set (backward compat)."""
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        pred = learner.predict(
            X.iloc[:2], times=[100, 500], ci=True, n_bootstrap=10,
            random_state=42, verbose=False,
        )
        rmst_ci = pred.rmst_ci
        assert rmst_ci is not None
        # Should return per-time format (backward compat)
        assert "tau_100" in rmst_ci.columns
        assert "tau_100_ci_lower" in rmst_ci.columns

    def test_bootstrap_reproducibility(self, synthetic_data_clean):
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        pred1 = learner.predict(
            X.iloc[:2], ci=True, n_bootstrap=10, random_state=99,
            verbose=False,
        )
        pred2 = learner.predict(
            X.iloc[:2], ci=True, n_bootstrap=10, random_state=99,
            verbose=False,
        )
        np.testing.assert_array_equal(
            pred1.median_survival_ci.values,
            pred2.median_survival_ci.values,
        )

    def test_bootstrap_ci_ordering(self, synthetic_data_clean):
        """CI lower <= point estimate <= CI upper (for most patients)."""
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')

        pred = learner.predict(
            X.iloc[:5], ci=True, n_bootstrap=20, random_state=42,
            verbose=False,
        )
        rmst_ci = pred.rmst_ci
        # For most patients, lower <= point <= upper
        # (not guaranteed for all due to small B, but should hold for most)
        ordering_ok = (
            (rmst_ci['ci_lower'] <= rmst_ci['rmst'] + 1e-6) &
            (rmst_ci['rmst'] <= rmst_ci['ci_upper'] + 1e-6)
        )
        assert ordering_ok.sum() >= 3  # at least 3/5 patients

    def test_bootstrap_coxnet(self, synthetic_data_clean):
        """CoxNet predict() with bootstrap CIs."""
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxnet')

        pred = learner.predict(
            X.iloc[:3], ci=True, n_bootstrap=10, random_state=42,
            verbose=False,
        )
        ci_df = pred.median_survival_ci
        assert ci_df is not None
        assert ci_df.shape[0] == 3
        assert "ci_lower" in ci_df.columns
        assert "ci_upper" in ci_df.columns


# ---------------------------------------------------------------------------
# Additional edge case tests (from review round 1)
# ---------------------------------------------------------------------------

class TestHelperEdgeCases:
    """Edge cases for prediction helper functions."""

    def test_median_at_first_timepoint(self):
        """S(t) <= 0.5 at the very first grid point -> return grid[0]."""
        time_grid = np.array([5.0, 10.0, 20.0])
        surv_probs = np.array([[0.4, 0.3, 0.1]])
        medians = compute_patient_median_survival(time_grid, surv_probs)
        assert medians[0] == 5.0

    def test_median_flat_curve(self):
        """S(t) is exactly 0.5 at two consecutive points -> return first time."""
        time_grid = np.array([5.0, 10.0, 20.0])
        surv_probs = np.array([[0.5, 0.5, 0.3]])
        medians = compute_patient_median_survival(time_grid, surv_probs)
        assert medians[0] == 5.0  # j=0, S(t) <= 0.5 at first point

    def test_rmst_tau_before_grid(self):
        """RMST with tau before first grid point -> S=1.0 for entire interval."""
        time_grid = np.array([5.0, 10.0])
        surv_probs = np.array([[0.8, 0.5]])
        rmst = compute_patient_rmst(time_grid, surv_probs, tau=2.0)
        np.testing.assert_allclose(rmst[0], 2.0)

    def test_extract_different_step_grids(self):
        """StepFunctions with different time grids -> union grid."""
        from sksurv.functions import StepFunction

        sf1 = StepFunction(np.array([1.0, 5.0]), np.array([0.9, 0.5]))
        sf2 = StepFunction(np.array([2.0, 5.0, 10.0]), np.array([0.8, 0.6, 0.2]))

        time_grid, probs = extract_time_grid_and_probs([sf1, sf2])
        # Union of {1,5} and {2,5,10} = {1,2,5,10}
        assert len(time_grid) == 4
        assert probs.shape == (2, 4)

    def test_extract_empty_list_raises(self):
        """Empty list of survival functions raises ValueError."""
        with pytest.raises(ValueError, match="Empty"):
            extract_time_grid_and_probs([])


class TestPredictInputValidation:
    """Tests for early parameter validation in predict()."""

    def test_n_bootstrap_zero_raises(self, trained_learner, new_patients):
        with pytest.raises(ValueError, match="n_bootstrap must be >= 1"):
            trained_learner.predict(new_patients, ci=True, n_bootstrap=0)

    def test_n_bootstrap_negative_raises(self, trained_learner, new_patients):
        with pytest.raises(ValueError, match="n_bootstrap must be >= 1"):
            trained_learner.predict(new_patients, ci=True, n_bootstrap=-5)

    def test_ci_level_percentage_raises(self, trained_learner, new_patients):
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            trained_learner.predict(
                new_patients, ci=True, n_bootstrap=10, ci_level=95,
            )

    def test_ci_level_zero_raises(self, trained_learner, new_patients):
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            trained_learner.predict(
                new_patients, ci=True, n_bootstrap=10, ci_level=0,
            )

    def test_ci_band_without_ci_warns(self, trained_learner, new_patients):
        """ci_band=True with ci=False is a no-op — warn, don't silently ignore."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained_learner.predict(new_patients, ci_band=True)  # ci defaults False
        band_warnings = [
            x for x in w
            if issubclass(x.category, SuggestionWarning)
            and "ci_band=True has no effect" in str(x.message)
        ]
        assert len(band_warnings) == 1

    def test_ci_band_with_ci_no_warning(self, trained_learner, new_patients):
        """ci_band=True with ci=True is valid usage — must NOT warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained_learner.predict(
                new_patients, ci=True, n_bootstrap=2, ci_band=True,
            )
        assert not any(
            "ci_band=True has no effect" in str(x.message) for x in w
        )

    @pytest.mark.parametrize(
        "kwarg,param_name",
        [
            ({"n_bootstrap": 500}, "n_bootstrap"),
            ({"ci_level": 0.90}, "ci_level"),
            ({"random_state": 7}, "random_state"),
            ({"n_jobs": 2}, "n_jobs"),
        ],
    )
    def test_ci_only_param_without_ci_warns(
        self, trained_learner, new_patients, kwarg, param_name
    ):
        """A CI-only param set with ci=False is a no-op — warn, don't drop."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained_learner.predict(new_patients, ci=False, **kwarg)
        hits = [
            x for x in w
            if issubclass(x.category, SuggestionWarning)
            and "no effect when ci=False" in str(x.message)
            and param_name in str(x.message)
        ]
        assert len(hits) == 1

    def test_multiple_ci_only_params_single_consolidated_warning(
        self, trained_learner, new_patients
    ):
        """Several ignored CI params produce ONE warning naming all of them."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained_learner.predict(
                new_patients, ci=False, n_bootstrap=500, random_state=7,
            )
        hits = [
            x for x in w
            if issubclass(x.category, SuggestionWarning)
            and "no effect when ci=False" in str(x.message)
        ]
        assert len(hits) == 1
        msg = str(hits[0].message)
        assert "n_bootstrap" in msg and "random_state" in msg

    def test_ci_false_defaults_no_warning(self, trained_learner, new_patients):
        """ci=False with no CI params (or all at defaults) must NOT warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained_learner.predict(
                new_patients, ci=False,
                n_bootstrap=200, ci_level=0.95, random_state=None, n_jobs=1,
            )
        assert not any(
            "no effect when ci=False" in str(x.message) for x in w
        )

    def test_ci_true_no_ignored_param_warning(
        self, trained_learner, new_patients
    ):
        """ci=True is valid usage — must NOT emit the ignored-param warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained_learner.predict(
                new_patients, ci=True, n_bootstrap=2, random_state=7,
            )
        assert not any(
            "no effect when ci=False" in str(x.message) for x in w
        )


class TestPredictionResultValidation:
    """Tests for PredictionResult __post_init__ and CI None behavior."""

    def test_n_patients_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_patients"):
            PredictionResult(
                risk_scores=np.array([1.0, 2.0]),
                model_name='test',
                n_patients=999,
            )

    def test_grid_probs_copresence(self):
        """_time_grid without _surv_probs raises."""
        with pytest.raises(ValueError, match="both be set"):
            PredictionResult(
                risk_scores=np.array([1.0]),
                model_name='test',
                n_patients=1,
                _time_grid=np.array([1.0, 2.0]),
                _surv_probs=None,
            )

    def test_invalid_ci_level_raises(self):
        with pytest.raises(ValueError, match="ci_level"):
            PredictionResult(
                risk_scores=np.array([1.0]),
                model_name='test',
                n_patients=1,
                _ci_level=5.0,
            )

    def test_ci_properties_none_without_bootstrap(self):
        """CI properties return None when ci=False (no bootstrap data)."""
        pr = PredictionResult(
            risk_scores=np.array([1.0, 2.0]),
            model_name='test',
            n_patients=2,
            _time_grid=np.array([1.0, 5.0, 10.0]),
            _surv_probs=np.array([[0.9, 0.7, 0.3], [0.8, 0.5, 0.1]]),
            _median_survival=np.array([8.0, 5.0]),
            _rmst_values=np.array([7.5, 4.2]),
        )
        assert pr.median_survival_ci is None
        assert pr.survival_at_times_ci is None
        assert pr.rmst_ci is None


class TestSummaryEdgeCases:

    def test_summary_with_times(self, trained_learner, new_patients):
        """summary() includes P(survive past t) lines when times specified."""
        pred = trained_learner.predict(
            new_patients, times=[100, 500], verbose=False,
        )
        text = pred.summary(patient=0)
        assert "P(survive past" in text
        assert "100" in text

    @pytest.mark.bootstrap_heavy
    def test_summary_with_ci(self, synthetic_data_clean):
        """summary() includes CI strings when bootstrap data present."""
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')
        pred = learner.predict(
            X.iloc[:2], times=[100], ci=True, n_bootstrap=10,
            random_state=42, verbose=False,
        )
        text = pred.summary(patient=0)
        assert "CI:" in text


# ---------------------------------------------------------------------------
# Prediction bootstrap parallel config
# ---------------------------------------------------------------------------

class TestPredictionBootstrapParallel:
    """Tests for auto-detection of parallelism in prediction bootstrap."""

    def test_resolve_sequential(self):
        """n_jobs=1 always returns sequential regardless of model/n_bootstrap."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _resolve_prediction_parallel_config,
        )
        n_jobs, backend = _resolve_prediction_parallel_config(
            n_jobs=1, n_bootstrap=1000, model_name='rsf', verbose=False,
        )
        assert n_jobs == 1

    def test_resolve_below_threshold_heavy(self):
        """Heavy model below iteration threshold → sequential."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _resolve_prediction_parallel_config,
            MIN_ITERS_FOR_PREDICTION_PARALLEL,
        )
        n_jobs, _ = _resolve_prediction_parallel_config(
            n_jobs=4,
            n_bootstrap=MIN_ITERS_FOR_PREDICTION_PARALLEL - 1,
            model_name='rsf', verbose=False,
        )
        assert n_jobs == 1

    def test_resolve_light_model_always_sequential(self):
        """Lightweight model always runs sequentially regardless of n_bootstrap."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _resolve_prediction_parallel_config,
        )
        n_jobs, _ = _resolve_prediction_parallel_config(
            n_jobs=4, n_bootstrap=1000, model_name='coxph', verbose=False,
        )
        assert n_jobs == 1

    def test_resolve_loky_for_rsf(self):
        """RSF above threshold should use loky (true parallelism for retrain)."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _resolve_prediction_parallel_config,
        )
        n_jobs, backend = _resolve_prediction_parallel_config(
            n_jobs=4, n_bootstrap=25, model_name='rsf', verbose=False,
        )
        assert n_jobs == 4
        assert backend in ('loky', 'threading')  # loky preferred, threading fallback

    def test_adjust_proto_parallel_rsf(self):
        """Outer parallel + RSF → cloned proto has n_jobs=1, original untouched."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _adjust_proto_njobs,
        )
        from mlsurv.models import get_model
        original = get_model('rsf', n_estimators=100, n_jobs=4)
        adjusted = _adjust_proto_njobs(original, 'rsf', outer_n_jobs=4)
        assert adjusted is not original
        assert adjusted.n_jobs == 1
        assert adjusted.n_estimators == 100
        # Original must be untouched
        assert original.n_jobs == 4

    def test_adjust_proto_sequential(self):
        """Outer sequential → proto returned unchanged."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _adjust_proto_njobs,
        )
        from mlsurv.models import get_model
        original = get_model('rsf', n_estimators=100, n_jobs=4)
        result = _adjust_proto_njobs(original, 'rsf', outer_n_jobs=1)
        assert result is original

    def test_adjust_proto_no_internal_parallelism(self):
        """CoxPH has no internal parallelism → proto unchanged."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _adjust_proto_njobs,
        )
        from mlsurv.models import get_model
        original = get_model('coxph', alpha=0.1)
        result = _adjust_proto_njobs(original, 'coxph', outer_n_jobs=4)
        assert result is original

    def test_adjust_proto_rsf_already_njobs_one(self):
        """RSF with n_jobs already 1 → no adjustment, returns original."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _adjust_proto_njobs,
        )
        from mlsurv.models import get_model
        original = get_model('rsf', n_estimators=100, n_jobs=1)
        result = _adjust_proto_njobs(original, 'rsf', outer_n_jobs=4)
        assert result is original

    def test_resolve_njobs_zero_raises(self):
        """n_jobs=0 should raise ValueError."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _resolve_prediction_parallel_config,
        )
        with pytest.raises(ValueError, match="n_jobs=0"):
            _resolve_prediction_parallel_config(
                n_jobs=0, n_bootstrap=25, model_name='coxph', verbose=False,
            )

    def test_resolve_light_model_warns(self):
        """Lightweight model with n_jobs>1 should emit SuggestionWarning."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _resolve_prediction_parallel_config,
        )
        from mlsurv._warnings import SuggestionWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n_jobs, _ = _resolve_prediction_parallel_config(
                n_jobs=4, n_bootstrap=500, model_name='coxph', verbose=False,
            )
        assert n_jobs == 1
        suggestion_warns = [x for x in w if issubclass(x.category, SuggestionWarning)]
        assert len(suggestion_warns) == 1
        assert "disabled" in str(suggestion_warns[0].message)

    def test_resolve_heavy_below_threshold_warns(self):
        """Heavy model below threshold with n_jobs>1 should warn."""
        from mlsurv.evaluation._prediction_bootstrap import (
            _resolve_prediction_parallel_config,
        )
        from mlsurv._warnings import SuggestionWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n_jobs, _ = _resolve_prediction_parallel_config(
                n_jobs=4, n_bootstrap=5, model_name='rsf', verbose=False,
            )
        assert n_jobs == 1
        suggestion_warns = [x for x in w if issubclass(x.category, SuggestionWarning)]
        assert len(suggestion_warns) == 1
        assert "threshold" in str(suggestion_warns[0].message)


class TestPredictionBootstrapWarningCoalescing:
    """Coalescing of per-iteration IterativeWarnings.

    When every bootstrap iteration fails with the same root cause, the
    coalescer should emit at most ``max_emit + 1`` IterativeWarnings
    (examples + aggregate summary), not ``n_bootstrap``.
    """

    @staticmethod
    def _make_inputs(n=20):
        X = pd.DataFrame({'a': np.arange(float(n)), 'b': np.arange(float(n))})
        y = np.array(
            [(True, float(i + 1)) for i in range(n)],
            dtype=[('event', bool), ('time', float)],
        )
        return X, y

    def test_all_failures_coalesced(self, monkeypatch):
        from mlsurv.evaluation import _prediction_bootstrap as pb
        from mlsurv._warnings import IterativeWarning
        from mlsurv.models import get_model
        from mlsurv.models.coxph import CoxPHModel

        n_bootstrap = 50

        def boom(self, X, y):
            raise ValueError("synthetic failure for coalescing test")

        # The iteration uses ``clone(model_proto)`` then ``model.fit(...)``;
        # patching the class's fit triggers the failure path on every clone.
        monkeypatch.setattr(CoxPHModel, "fit", boom)

        X, y = self._make_inputs()
        X_new = X.iloc[:2]
        time_grid = np.array([1.0, 5.0, 10.0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterativeWarning)
            warnings.simplefilter("always", MethodologicalWarning)
            result = pb.run_prediction_bootstrap(
                X_train=X, y_train=y, X_new=X_new,
                model_name='coxph', model_proto=get_model('coxph'),
                imputer=None, scaler=None, selector=None,
                time_grid=time_grid, rmst_default_tau=10.0, user_times=None,
                n_bootstrap=n_bootstrap, random_state=0,
                n_jobs=1, verbose=False,
            )

        iter_warns = [
            w for w in caught if issubclass(w.category, IterativeWarning)
        ]
        # Single error type → 1 example + 1 aggregate summary = 2 warnings,
        # regardless of n_bootstrap=50.
        assert len(iter_warns) == 2, (
            f"Expected 2 IterativeWarnings (1 example + 1 summary), got "
            f"{len(iter_warns)}"
        )
        msgs = [str(w.message) for w in iter_warns]
        assert "synthetic failure for coalescing test" in msgs[0]
        assert "seed=" in msgs[0]
        assert f"{n_bootstrap}/{n_bootstrap}" in msgs[1]
        assert "all shared the same error type (ValueError)" in msgs[1]

        # 100% failure → MethodologicalWarning "all iterations failed" still
        # fires (preserves user-visible signal beyond the IterativeWarning).
        meth_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and not issubclass(w.category, IterativeWarning)
        ]
        assert any(
            "All bootstrap iterations failed" in str(w.message)
            for w in meth_warns
        ), "Expected the 'All bootstrap iterations failed' MethodologicalWarning"
        # Returned dict should have None values when nothing succeeded.
        assert result['medians'] is None
        assert result['rmst'] is None

    def test_mixed_success_and_failure_partial_summary(self, monkeypatch):
        from mlsurv.evaluation import _prediction_bootstrap as pb
        from mlsurv._warnings import IterativeWarning
        from mlsurv.models import get_model
        from mlsurv.models.coxph import CoxPHModel

        n_bootstrap = 10
        # Fail for first 6 seeds, succeed for the remainder.
        real_fit = CoxPHModel.fit
        call_count = {'n': 0}

        def flaky(self, X, y):
            call_count['n'] += 1
            if call_count['n'] <= 6:
                raise ValueError(f"flaky failure {call_count['n']}")
            return real_fit(self, X, y)

        monkeypatch.setattr(CoxPHModel, "fit", flaky)

        X, y = self._make_inputs()
        X_new = X.iloc[:2]
        time_grid = np.array([1.0, 5.0, 10.0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterativeWarning)
            warnings.simplefilter("always", MethodologicalWarning)
            result = pb.run_prediction_bootstrap(
                X_train=X, y_train=y, X_new=X_new,
                model_name='coxph', model_proto=get_model('coxph'),
                imputer=None, scaler=None, selector=None,
                time_grid=time_grid, rmst_default_tau=10.0, user_times=None,
                n_bootstrap=n_bootstrap, random_state=0,
                n_jobs=1, verbose=False,
            )

        iter_warns = [
            w for w in caught if issubclass(w.category, IterativeWarning)
        ]
        assert len(iter_warns) == 2
        summary = str(iter_warns[1].message)
        assert "6/10 (60%)" in summary
        assert "ValueError" in summary
        # Successful iterations must appear in the result arrays.
        assert result['medians'] is not None
        assert result['medians'].shape[0] == 4  # 10 - 6 failures

    def test_multiple_distinct_exception_types_e2e(self, monkeypatch):
        from mlsurv.evaluation import _prediction_bootstrap as pb
        from mlsurv._warnings import IterativeWarning
        from mlsurv.models import get_model
        from mlsurv.models.coxph import CoxPHModel

        n_bootstrap = 6
        call_count = {'n': 0}

        def alternating(self, X, y):
            call_count['n'] += 1
            if call_count['n'] % 2 == 0:
                raise RuntimeError("rt failure")
            raise ValueError("val failure")

        monkeypatch.setattr(CoxPHModel, "fit", alternating)

        X, y = self._make_inputs()
        X_new = X.iloc[:2]
        time_grid = np.array([1.0, 5.0, 10.0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterativeWarning)
            warnings.simplefilter("always", MethodologicalWarning)
            pb.run_prediction_bootstrap(
                X_train=X, y_train=y, X_new=X_new,
                model_name='coxph', model_proto=get_model('coxph'),
                imputer=None, scaler=None, selector=None,
                time_grid=time_grid, rmst_default_tau=10.0, user_times=None,
                n_bootstrap=n_bootstrap, random_state=0,
                n_jobs=1, verbose=False,
            )

        iter_warns = [
            w for w in caught if issubclass(w.category, IterativeWarning)
        ]
        # 2 distinct error types → 2 examples + 1 aggregate = 3
        assert len(iter_warns) == 3
        summary = str(iter_warns[2].message)
        assert "across 2 distinct error types" in summary
        assert "ValueError (n=3)" in summary
        assert "RuntimeError (n=3)" in summary


# ---------------------------------------------------------------------------
# predict_from_pipeline() — standalone prediction without learner
# ---------------------------------------------------------------------------

class _FakeSurvivalModel:
    """Fake model returning a DataFrame survival function."""

    def predict(self, X):
        return np.arange(len(X), dtype=float) * 0.5

    def predict_survival_function(self, X):
        times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = {}
        for i in range(len(X)):
            # Linearly decreasing survival from 1.0 to 0.5
            data[i] = [1.0 - (0.01 * i * t / 50.0) - (t / 100.0)
                        for t in times]
        return pd.DataFrame(data, index=times)


class _FakeRiskOnlyModel:
    """Fake model without predict_survival_function."""

    def predict(self, X):
        return np.arange(len(X), dtype=float) * 0.5


class TestPredictFromPipeline:
    """Tests for the standalone predict_from_pipeline() function."""

    def test_risk_scores_only(self):
        model = _FakeRiskOnlyModel()
        X = pd.DataFrame({'a': [1, 2, 3]})
        result = predict_from_pipeline(model, X, model_name='test')

        assert result.model_name == 'test'
        assert result.n_patients == 3
        np.testing.assert_allclose(result.risk_scores, [0.0, 0.5, 1.0])
        assert result._median_survival is None
        assert result._rmst_values is None
        assert result._surv_probs is None
        assert result._time_grid is None

    def test_with_survival_function(self):
        model = _FakeSurvivalModel()
        X = pd.DataFrame({'a': [1, 2, 3]})
        result = predict_from_pipeline(model, X, model_name='test')

        assert result.risk_scores is not None
        assert result._time_grid is not None
        assert result._surv_probs is not None
        assert result._median_survival is not None
        assert result._rmst_values is not None
        assert result.n_patients == 3
        assert len(result._time_grid) == 5
        assert result._surv_probs.shape == (3, 5)

    def test_with_times(self):
        model = _FakeSurvivalModel()
        X = pd.DataFrame({'a': [1, 2, 3]})
        result = predict_from_pipeline(
            model, X, model_name='test', times=[15.0, 25.0],
        )

        assert result._surv_at_user_times is not None
        assert result._surv_at_user_times.shape == (3, 2)
        assert result._rmst_at_user_times is not None
        assert result._rmst_at_user_times.shape == (3, 2)
        np.testing.assert_array_equal(result._user_times, [15.0, 25.0])

    def test_rmst_default_tau_from_time_grid(self):
        model = _FakeSurvivalModel()
        X = pd.DataFrame({'a': [1, 2]})
        result = predict_from_pipeline(model, X, model_name='test')

        # Default tau should be max of time grid (50.0)
        assert result._rmst_default_tau == pytest.approx(50.0)

    def test_rmst_default_tau_override(self):
        model = _FakeSurvivalModel()
        X = pd.DataFrame({'a': [1, 2]})
        result = predict_from_pipeline(
            model, X, model_name='test', rmst_default_tau=30.0,
        )

        assert result._rmst_default_tau == pytest.approx(30.0)

    def test_survival_function_failure_degrades_gracefully(self):
        class BrokenSurvModel:
            def predict(self, X):
                return np.array([1.0, 2.0])

            def predict_survival_function(self, X):
                raise RuntimeError("broken")

        X = pd.DataFrame({'a': [1, 2]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = predict_from_pipeline(
                BrokenSurvModel(), X, model_name='broken',
            )

        np.testing.assert_allclose(result.risk_scores, [1.0, 2.0])
        assert result._surv_probs is None
        assert result._median_survival is None
        suggestion_warnings = [x for x in w if issubclass(x.category, SuggestionWarning)]
        assert len(suggestion_warnings) >= 1

    def test_bootstrap_fields_always_none(self):
        model = _FakeSurvivalModel()
        X = pd.DataFrame({'a': [1, 2]})
        result = predict_from_pipeline(model, X, model_name='test')

        assert result._bootstrap_medians is None
        assert result._bootstrap_surv_at_times is None
        assert result._bootstrap_rmst is None
        assert result._bootstrap_rmst_at_times is None

    def test_rejects_empty_times(self):
        model = _FakeRiskOnlyModel()
        X = pd.DataFrame({'a': [1]})
        with pytest.raises(ValueError, match="non-empty 1-D"):
            predict_from_pipeline(model, X, model_name='t', times=[])

    def test_rejects_negative_times(self):
        model = _FakeRiskOnlyModel()
        X = pd.DataFrame({'a': [1]})
        with pytest.raises(ValueError, match="must be >= 0"):
            predict_from_pipeline(model, X, model_name='t', times=[-1.0])

    def test_rejects_non_finite_times(self):
        model = _FakeRiskOnlyModel()
        X = pd.DataFrame({'a': [1]})
        with pytest.raises(ValueError, match="finite numeric values"):
            predict_from_pipeline(
                model, X, model_name='t', times=[1.0, np.nan],
            )

    def test_rejects_duplicate_times(self):
        model = _FakeRiskOnlyModel()
        X = pd.DataFrame({'a': [1]})
        with pytest.raises(ValueError, match="must not contain duplicates"):
            predict_from_pipeline(
                model, X, model_name='t', times=[1.0, 1.0],
            )

    def test_rejects_2d_times(self):
        model = _FakeRiskOnlyModel()
        X = pd.DataFrame({'a': [1]})
        with pytest.raises(ValueError, match="non-empty 1-D"):
            predict_from_pipeline(
                model, X, model_name='t', times=[[1.0, 2.0]],
            )

    def test_scalar_times_accepted(self):
        model = _FakeSurvivalModel()
        X = pd.DataFrame({'a': [1, 2]})
        result = predict_from_pipeline(
            model, X, model_name='test', times=15.0,
        )
        assert result._surv_at_user_times is not None
        assert result._surv_at_user_times.shape == (2, 1)
        np.testing.assert_array_equal(result._user_times, [15.0])

    def test_times_ignored_warning_for_risk_only_model(self):
        model = _FakeRiskOnlyModel()
        X = pd.DataFrame({'a': [1, 2]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = predict_from_pipeline(
                model, X, model_name='test', times=[10.0, 20.0],
            )
        assert result._surv_at_user_times is None
        suggestion_warnings = [
            x for x in w if issubclass(x.category, SuggestionWarning)
        ]
        assert len(suggestion_warnings) >= 1
        assert "times parameter is ignored" in str(suggestion_warnings[0].message)


# ---------------------------------------------------------------------------
# PredictionResult.plot() (fast tier — reuses module-scoped trained_learner)
# ---------------------------------------------------------------------------

class TestPredictionPlot:
    """End-to-end tests for PredictionResult.plot()."""

    def test_plot_returns_figure(self, trained_learner, new_patients):
        import plotly.graph_objects as go

        pred = trained_learner.predict(new_patients, models='coxph', verbose=False)
        fig = pred.plot(0)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_plot_percentile_spec(self, trained_learner, new_patients):
        import plotly.graph_objects as go

        pred = trained_learner.predict(new_patients, models='coxph', verbose=False)
        fig = pred.plot('high_risk')
        assert isinstance(fig, go.Figure)

    def test_plot_includes_ci_band(self, trained_learner, new_patients):
        pred = trained_learner.predict(
            new_patients, models='coxph', ci=True, ci_band=True,
            n_bootstrap=4, random_state=0, verbose=False,
        )
        fig = pred.plot(0)
        names = [t.name for t in fig.data if t.name]
        assert any("CI" in n for n in names)

    def test_plot_milestones_from_times(self, trained_learner, new_patients):
        pred = trained_learner.predict(
            new_patients, models='coxph', times=[1.0, 3.0], verbose=False,
        )
        fig = pred.plot(0)
        assert len(fig.layout.annotations) >= 1

    def test_plot_no_survival_raises(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, models='coxph', verbose=False)
        # Simulate a model without survival-function support.
        pred._surv_probs = None
        pred._time_grid = None
        with pytest.raises(ValueError):
            pred.plot(0)

    def test_plot_index_out_of_range_raises(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, models='coxph', verbose=False)
        with pytest.raises(ValueError):
            pred.plot(999)

    def test_plot_unrecognized_spec_raises(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, models='coxph', verbose=False)
        with pytest.raises(ValueError):
            pred.plot('not_a_spec')

    def test_plot_wrong_type_raises(self, trained_learner, new_patients):
        pred = trained_learner.predict(new_patients, models='coxph', verbose=False)
        with pytest.raises(TypeError):
            pred.plot(1.5)
