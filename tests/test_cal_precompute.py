"""Survival-function metrics NaN out in the bootstrap precompute path.

``bootstrap()`` takes the precompute path whenever no custom metrics or
importance are requested (the common case). On that path it precomputes risk
scores once up front and drops the per-iteration test features
(``X_boot=None``). The precompute step used to build survival probability curves
only for Brier-family metrics, so requesting a calibration metric
(``cal_slope``/``cal_intercept``/``gnd``) or Antolini's ``ctd`` *without* a Brier
metric left ``precomputed_surv_probs=None`` and ``X_boot=None`` — every
iteration's metric fell back to ``_get_surv_funcs(model, X=None)``, raised, and
was recorded as NaN (with a ``MetricFailureWarning``). This bit *any* model on
the precompute path (loky CoxPH as well as threading RSF/XGB).

The fix widens the precompute trigger to the full ``SURV_FUNC_METRICS`` set, so
these metrics compute correctly on their own. The Brier-path test deliberately adds
``integrated_brier`` to dodge this bug; these tests cover the no-Brier case.

These tests use RSF + ``force_parallel=True`` so they also exercise the threading
backend (the most complex precompute path); the fix is model-agnostic.
"""

import warnings

import numpy as np
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MetricFailureWarning
from mlsurv.evaluation.bootstrap import bootstrap as _bootstrap
from mlsurv.evaluation.helpers import _has_internal_parallelism


@pytest.fixture(scope="module")
def _rsf_eval_inputs(_gbsg2_data_raw):
    """Train RSF on GBSG2 once and expose the held-out evaluation inputs.

    RSF has internal parallelism, so ``bootstrap(force_parallel=True)`` takes the
    precompute/threading path this regression targets. Depends on the
    session-scoped raw GBSG2 fixture (the public ``gbsg2_data`` is function-scoped
    and can't back a module-scoped fixture).
    """
    import pandas as pd

    X_raw, y_raw = _gbsg2_data_raw
    # GBSG2 ships y as ('cens', 'time'); mlsurv wants ('event', 'time').
    y = np.array([(bool(r["cens"]), r["time"]) for r in y_raw],
                 dtype=[("event", bool), ("time", float)])
    X = pd.get_dummies(X_raw, drop_first=True)

    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, n_jobs=4)
    fitted = learner.train("rsf").model
    assert _has_internal_parallelism(fitted)

    state = learner._state
    eval_times = np.quantile(state.y_test["time"], [0.25, 0.5, 0.75])
    return fitted, state.X_test, state.y_test, state.y_train, eval_times


def _run_bootstrap(inputs, metrics):
    fitted, X_test, y_test, y_train, eval_times = inputs
    boot = _bootstrap(
        fitted, X_test, y_test, y_train=y_train, n_bootstrap=8,
        metrics=metrics, eval_times=eval_times, random_state=123,
        n_jobs=4, force_parallel=True, show_progress=False, model_name="rsf",
    )
    return boot.iterations


@pytest.mark.integration
def test_calibration_precomputes_without_brier(_rsf_eval_inputs):
    """cal_slope/cal_intercept must be finite when requested without a Brier metric.

    On ``main`` the precompute trigger omitted calibration, so X_boot=None and
    every iteration NaN'd out. The fix makes the curves get precomputed.
    """
    cal_metrics = ["cal_slope", "cal_intercept"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = _run_bootstrap(_rsf_eval_inputs, cal_metrics)

    cal = df[df["metric_name"].isin(cal_metrics)]
    assert not cal.empty, "calibration metrics produced no iteration rows"
    assert np.isfinite(cal["value"].to_numpy()).any(), \
        "calibration values are all NaN — surv_probs precompute did not engage"

    # The MetricFailureWarning ("Calibration requires model and X when surv_probs
    # not provided") must no longer fire for this valid request. Match the stable
    # "requires model and X" phrase from _get_surv_funcs rather than the metric
    # label, so the check is robust to label wording.
    cal_failures = [
        w for w in caught
        if issubclass(w.category, MetricFailureWarning)
        and "requires model and X" in str(w.message)
    ]
    assert not cal_failures, \
        f"unexpected calibration MetricFailureWarning(s): {[str(w.message) for w in cal_failures]}"


@pytest.mark.integration
def test_gnd_precomputes_without_brier(_rsf_eval_inputs):
    """GND must precompute alone.

    gnd takes a different consumption path than cal_slope/cal_intercept — it
    reads a single survival-curve column via ``_get_surv_probs_at_median`` rather
    than the full matrix — so it is not covered transitively by the calibration
    test. gnd returns a hypothesis-test p-value, so ``.any()`` finite is the
    right strength (a degenerate-but-finite p-value is plausible per iteration).
    """
    df = _run_bootstrap(_rsf_eval_inputs, ["gnd"])
    gnd = df[df["metric_name"] == "gnd"]
    assert not gnd.empty, "gnd produced no iteration rows"
    assert np.isfinite(gnd["value"].to_numpy()).any(), \
        "gnd values are all NaN — surv_probs precompute did not engage"


@pytest.mark.integration
def test_ctd_precomputes_without_brier(_rsf_eval_inputs):
    """Antolini's ctd is also survival-function-based and must precompute alone."""
    df = _run_bootstrap(_rsf_eval_inputs, ["ctd"])
    ctd = df[df["metric_name"] == "ctd"]
    assert not ctd.empty, "ctd produced no iteration rows"
    assert np.isfinite(ctd["value"].to_numpy()).any(), \
        "ctd values are all NaN — surv_probs precompute did not engage"
