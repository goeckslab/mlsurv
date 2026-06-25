"""Tests for the shared S(t|x) calibration compute helpers.

Covers the leakage-free OOF collection, the per-horizon isotonic fit, the
isotonic apply-to-array path, the IPCW calibration weights, and the
calibration-index helper — the pieces that ``train(calibrate=...)`` and the
``IsotonicCalibrator`` are built on.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsurv.evaluation._recalibration import (
    _MIN_OOF_SAMPLES_PER_HORIZON,
    _apply_recalibrators_to_array,
    _fit_censoring_distribution,
    _fit_isotonic_recalibrators,
    _ipcw_calibration_weights,
    compute_calibration_indices,
)


def _make_y(n=200, event_rate=0.6, max_time=5.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.array(
        [(bool(rng.binomial(1, event_rate)), float(rng.uniform(0.0, max_time)))
         for _ in range(n)],
        dtype=[("event", bool), ("time", float)],
    )


def _encode_gbsg2(gbsg2_data):
    """GBSG2 ships y as ('cens', 'time') with categorical X; mlsurv wants
    ('event', 'time') and numeric columns."""
    import pandas as pd
    X_raw, y_raw = gbsg2_data
    y = np.array([(bool(r["cens"]), r["time"]) for r in y_raw],
                 dtype=[("event", bool), ("time", float)])
    X = pd.get_dummies(X_raw, drop_first=True)
    return X, y


# ----------------------------------------------------------------------
# IPCW calibration weights
# ----------------------------------------------------------------------

def test_ipcw_weights_zero_for_censored_before_horizon():
    # one event-before-h, one at-risk past-h, one censored-before-h
    y = np.array([(True, 1.0), (False, 5.0), (False, 1.0)],
                 dtype=[("event", bool), ("time", float)])
    cde = _fit_censoring_distribution(y)
    w = _ipcw_calibration_weights(y["time"], y["event"], horizon=2.0, cde=cde)
    assert w[0] > 0          # event before horizon -> 1/G(T)
    assert w[1] > 0          # at risk past horizon -> 1/G(h)
    assert w[2] == 0.0       # censored before horizon -> uninformative
    assert np.all(w >= 0)


# ----------------------------------------------------------------------
# Isotonic fit + apply
# ----------------------------------------------------------------------

def _make_oof(y, horizons, seed=1):
    """A miscalibrated OOF S(t|x): overconfident survival, monotone in time."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 0.95, size=len(y))
    cols = [np.clip(base ** (1.0 + j * 0.5), 1e-3, 1 - 1e-3)
            for j in range(len(horizons))]
    return np.column_stack(cols)


def test_fit_and_apply_isotonic_roundtrip():
    horizons = np.array([1.0, 2.0, 3.0])
    y = _make_y(n=300)
    oof = _make_oof(y, horizons)
    per_horizon, dropped = _fit_isotonic_recalibrators(oof, horizons, y)
    assert per_horizon, "expected at least one fitted horizon"
    assert not dropped, "well-powered cohort should not drop horizons"

    out = _apply_recalibrators_to_array(per_horizon, oof, horizons)
    assert out.shape == oof.shape
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    # each row is a valid (non-increasing) survival function in time
    assert np.all(np.diff(out, axis=1) <= 1e-9)


def test_apply_empty_mapping_raises():
    with pytest.raises(ValueError, match="No fitted recalibrators"):
        _apply_recalibrators_to_array(
            {}, np.array([[0.5, 0.4]]), np.array([1.0, 2.0]),
        )


def test_fit_drops_underpowered_horizon():
    # A cohort smaller than the per-horizon minimum: every horizon is dropped
    # for insufficient usable samples, so nothing is fitted.
    horizons = np.array([1.0, 2.0])
    y = _make_y(n=8, max_time=5.0)
    oof = _make_oof(y, horizons)
    per_horizon, dropped = _fit_isotonic_recalibrators(oof, horizons, y)
    assert not per_horizon
    assert {h for h, _ in dropped} == {1.0, 2.0}
    for _, n_used in dropped:
        assert n_used < _MIN_OOF_SAMPLES_PER_HORIZON


# ----------------------------------------------------------------------
# Calibration indices
# ----------------------------------------------------------------------

def test_calibration_indices_returns_valid_range():
    # Constant prediction at the empirical survival rate -> finite indices in
    # [0, 1] with a positive used-sample count.
    y = _make_y(n=400, seed=3)
    h = 2.5
    emp = float(np.mean(y["time"] > h))
    preds = np.full(len(y), emp)
    idx = compute_calibration_indices(y, preds, h)
    assert np.isfinite(idx.ici) and 0.0 <= idx.ici <= 1.0
    assert idx.n_used > 0


# ----------------------------------------------------------------------
# OOF collection (re-runs CV folds -> integration)
# ----------------------------------------------------------------------

@pytest.mark.integration
def test_calibration_flagged_models_excludes_calibrated(gbsg2_data):
    """A flagged model drops off ``calibration_flagged_models`` once a
    calibrator is attached (the property must filter on ``result.calibrator``,
    not on a stored recalibration field)."""
    from mlsurv import SurvivalLearner
    from mlsurv.evaluation._calibrators import IsotonicCalibrator

    X, y = _encode_gbsg2(gbsg2_data)
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=0, n_jobs=1)
    learner.run(["coxph"], calibrate=False, n_importance_repeats=0,
                bootstrap=False, verbose=False)
    mr = learner._state.models["coxph"]
    # Stamp the flag deterministically rather than depend on real drift.
    mr.calibration_flagged = True
    assert "coxph" in learner.calibration_flagged_models

    # Attaching a calibrator must exclude it from the flagged set.
    mr.calibrator = IsotonicCalibrator()
    assert "coxph" not in learner.calibration_flagged_models


@pytest.mark.integration
def test_validate_applies_attached_calibrator(gbsg2_data):
    """validate() applies the attached calibrator to the external cohort by
    default; use_recalibrated=False forces raw (routing through validate)."""
    from mlsurv import SurvivalLearner

    X, y = _encode_gbsg2(gbsg2_data)
    n_ext = 150
    Xdev, ydev = X.iloc[n_ext:].reset_index(drop=True), y[n_ext:]
    Xext, yext = X.iloc[:n_ext].reset_index(drop=True), y[:n_ext]

    learner = SurvivalLearner(Xdev, ydev)
    learner.setup(random_state=0, n_jobs=1)
    learner.run(["coxph"], calibrate={"coxph": "isotonic"},
                n_importance_repeats=0, bootstrap=False, verbose=False)
    assert learner._has_calibrator("coxph")

    cal = learner.validate(Xext, yext, name="cal", bootstrap=False,
                           verbose=False, use_recalibrated=True)
    raw = learner.validate(Xext, yext, name="raw", bootstrap=False,
                           verbose=False, use_recalibrated=False)
    cal_row = cal.evaluation.concordance.set_index("model_type").loc["coxph"]
    raw_row = raw.evaluation.concordance.set_index("model_type").loc["coxph"]
    # A probability-magnitude metric must move — the calibrator changed the
    # external-cohort S(t|x). Rank metrics (Harrell C) are invariant.
    assert (
        abs(float(cal_row["integrated_brier_test"])
            - float(raw_row["integrated_brier_test"])) > 1e-9
        or abs(float(cal_row["cal_slope_test"])
               - float(raw_row["cal_slope_test"])) > 1e-9
    ), "validate(use_recalibrated=True) did not apply the attached calibrator"
    # invalid use_recalibrated is rejected, not silently treated as raw.
    with pytest.raises(ValueError, match="use_recalibrated"):
        learner.validate(Xext, yext, name="bad", bootstrap=False,
                         verbose=False, use_recalibrated="yes")


@pytest.mark.integration
def test_subpopulation_metrics_use_calibrated_sprobs(gbsg2_data):
    """Per-subpopulation time-dependent metrics reflect the calibrated S(t|x)
    when a calibrator is attached (routed through the same transform cache)."""
    import pandas as pd
    from mlsurv import SurvivalLearner

    X, y = _encode_gbsg2(gbsg2_data)
    # A binary subpopulation on the first column (full-data labels; setup splits).
    grp = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int).to_numpy()
    subpops = {"grp": grp}

    def _ibs_by_subpop(calibrate):
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=0, n_jobs=1, subpopulations=subpops)
        learner.run(["coxph"], calibrate=calibrate, n_importance_repeats=0,
                    bootstrap=False, verbose=False)
        # Per-subpopulation point metrics for the 'grp' variable.
        return learner._state.subpopulation_results["grp"].metrics

    def _brier_values(df):
        # Long format: columns [subpopulation, model_type, metric, value].
        m = df[df["metric"].str.contains("brier", case=False, na=False)]
        m = m.sort_values(["subpopulation", "metric"])
        return pd.to_numeric(m["value"], errors="coerce").to_numpy()

    cal_v = _brier_values(_ibs_by_subpop({"coxph": "isotonic"}))
    raw_v = _brier_values(_ibs_by_subpop(False))
    assert cal_v.size > 0, "no brier metric in subpopulation table"
    # The calibrator changed the per-subpopulation S(t|x), so the Brier-type
    # magnitude metrics must move (rank metrics would be invariant).
    assert not np.allclose(cal_v, raw_v, equal_nan=True), (
        "subpopulation metrics did not reflect the attached calibrator"
    )


@pytest.mark.integration
def test_collect_oof_survival_shape(gbsg2_data):
    from sklearn.model_selection import KFold

    from mlsurv import SurvivalLearner
    from mlsurv.evaluation._recalibration import _collect_oof_survival

    X, y = _encode_gbsg2(gbsg2_data)
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=0, n_jobs=1)
    learner.train("coxph")
    horizons = np.array([500.0, 1000.0, 1500.0])
    oof = _collect_oof_survival(
        pipeline=learner._state.models["coxph"].model,
        X_train=learner._state.X_train,
        y_train=learner._state.y_train,
        splitter=KFold(n_splits=3, shuffle=True, random_state=0),
        horizons=horizons,
    )
    assert oof.shape == (len(learner._state.X_train), len(horizons))
    assert np.isfinite(oof).any()


def test_calibration_splitter_degrades_and_skips_small_cohort(small_data):
    """``_calibration_splitter`` pins breslow_slope to 5 folds, auto-degrades
    ``k`` to ``min(5, n_events, n_censored)``, and skips (returns ``None`` with a
    ``SuggestionWarning``) when fewer than 2 events or 2 censored are available.
    A warper (``oof_cv_folds is None``) inherits the learner's CV splitter.

    Direct-call unit test of the kept calibration splitter — the small-cohort
    degrade/skip path the (removed) FSSVM in-fit() recalibration once owned.
    """
    from sklearn.model_selection import StratifiedKFold

    from mlsurv import SurvivalLearner
    from mlsurv._warnings import SuggestionWarning
    from mlsurv.evaluation._calibrators import create_calibrator

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=0, n_jobs=1)
    base = learner._state.cv_splitter
    breslow = create_calibrator("breslow_slope")

    def _y(events):
        return np.array(
            [(bool(e), float(i + 1)) for i, e in enumerate(events)],
            dtype=[("event", bool), ("time", float)],
        )

    # Degrade: 3 events, 4 censored -> k = min(5, 3, 4) = 3.
    learner._state.y_train = _y([1, 1, 1, 0, 0, 0, 0])
    splitter = learner._calibration_splitter(breslow, base, "fssvm")
    assert isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 3

    # Skip: only 1 censored -> k = min(5, 4, 1) = 1 < 2 -> warn + None.
    learner._state.y_train = _y([1, 1, 1, 1, 0])
    with pytest.warns(SuggestionWarning, match="at least 2 events and 2 censored"):
        assert learner._calibration_splitter(breslow, base, "fssvm") is None

    # A warper (oof_cv_folds is None) inherits the learner CV splitter unchanged.
    iso = create_calibrator("isotonic")
    assert learner._calibration_splitter(iso, base, "fssvm") is base
