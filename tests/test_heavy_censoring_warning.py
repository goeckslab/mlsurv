"""Heavy-censoring warning on the headline Harrell C-index.

Under heavy censoring, ``evaluate()`` reports Harrell's ``c_index_test`` as the
headline, which is upward-biased in that regime. These tests cover the runtime
``MethodologicalWarning`` that points the user to ``c_index_ipcw_test``
(Uno's IPCW C).

The fast unit tests call the helper directly (no model training); the
integration test exercises the full ``setup() -> train() -> evaluate()`` path.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning
from mlsurv.evaluation.metrics import HIGH_CENSORING_THRESHOLD
from mlsurv.learner._evaluation_helpers import _emit_heavy_censoring_warning
from mlsurv.learner._state import LearnerState

from tests._helpers import make_survival_array


def _y_with_censoring(censoring, n=200, seed=0):
    """Structured array with an (approximate) target censoring fraction."""
    rng = np.random.RandomState(seed)
    events = (rng.random(n) >= censoring).astype(bool)
    times = rng.exponential(5.0, n) + 0.1
    return make_survival_array(events, times)


def _capture_methodological(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ret = fn()
    msgs = [str(w.message) for w in caught if _is_heavy_censoring(w)]
    return ret, msgs


def _is_heavy_censoring(w):
    """True for the heavy-censoring warning only.

    Filters on ``c_index_ipcw_test`` — the redirect unique to this warning.
    The co-located IPCW *truncation* MethodologicalWarning also contains the
    word "censored" ("administratively censored"), so a "censored" substring
    filter would wrongly catch it too.
    """
    return (
        issubclass(w.category, MethodologicalWarning)
        and "c_index_ipcw_test" in str(w.message)
    )


# --------------------------------------------------------------------------
# Fast unit tests — helper in isolation
# --------------------------------------------------------------------------

def test_fires_above_threshold_and_points_to_ipcw():
    state = LearnerState()
    y = _y_with_censoring(0.90)
    censoring, msgs = _capture_methodological(
        lambda: _emit_heavy_censoring_warning(state, "main", y)
    )
    assert censoring is not None and censoring > HIGH_CENSORING_THRESHOLD
    assert len(msgs) == 1
    assert "c_index_ipcw_test" in msgs[0]
    # Threshold clause is rendered so a non-strict/mis-formatted slip is caught.
    assert "> 80%" in msgs[0]


def test_silent_below_threshold():
    state = LearnerState()
    y = _y_with_censoring(0.40)
    censoring, msgs = _capture_methodological(
        lambda: _emit_heavy_censoring_warning(state, "main", y)
    )
    assert censoring is None
    assert msgs == []


def test_boundary_at_threshold_does_not_fire():
    # Exactly 80% censored (strict ``>`` comparison must not fire).
    state = LearnerState()
    events = np.array([True] * 20 + [False] * 80)
    times = np.linspace(1.0, 10.0, 100)
    y = make_survival_array(events, times)
    censoring, msgs = _capture_methodological(
        lambda: _emit_heavy_censoring_warning(state, "main", y)
    )
    assert censoring is None
    assert msgs == []


def test_boundary_just_above_threshold_fires():
    # 81% censored -> just above the 80% threshold.
    state = LearnerState()
    events = np.array([True] * 19 + [False] * 81)
    times = np.linspace(1.0, 10.0, 100)
    y = make_survival_array(events, times)
    censoring, msgs = _capture_methodological(
        lambda: _emit_heavy_censoring_warning(state, "main", y)
    )
    assert censoring is not None
    assert len(msgs) == 1


def test_none_cohort_is_noop():
    state = LearnerState()
    censoring, msgs = _capture_methodological(
        lambda: _emit_heavy_censoring_warning(state, "main", None)
    )
    assert censoring is None
    assert msgs == []


def test_empty_cohort_is_noop():
    state = LearnerState()
    y = make_survival_array(np.array([], dtype=bool), np.array([], dtype=float))
    censoring, msgs = _capture_methodological(
        lambda: _emit_heavy_censoring_warning(state, "main", y)
    )
    assert censoring is None
    assert msgs == []


def test_dedup_warns_once_per_cohort():
    state = LearnerState()
    y = _y_with_censoring(0.90)
    _capture_methodological(lambda: _emit_heavy_censoring_warning(state, "main", y))
    _, msgs2 = _capture_methodological(
        lambda: _emit_heavy_censoring_warning(state, "main", y)
    )
    assert msgs2 == []
    # A different cohort still warns.
    _, msgs3 = _capture_methodological(
        lambda: _emit_heavy_censoring_warning(state, "other", y)
    )
    assert len(msgs3) == 1


def test_setup_clears_dedup_state():
    """Re-running setup() re-arms the warning (the censoring fraction derives
    from the old test split, so the dedup flag must be cleared)."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"x1": rng.normal(size=60), "x2": rng.normal(size=60)})
    y = _y_with_censoring(0.30, n=60)
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)
    learner._state.heavy_censoring_warned["main"] = True

    learner.setup(random_state=42)
    assert learner._state.heavy_censoring_warned == {}


# --------------------------------------------------------------------------
# Integration test — full evaluate() path
# --------------------------------------------------------------------------

def _make_heavy_censoring_split(censoring=0.90, n=300, seed=42):
    """Train/test split with heavy censoring, mirroring the issue repro."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })
    events = rng.random(n) >= censoring
    times = rng.exponential(5.0, size=n) + 0.1
    y = make_survival_array(events.astype(bool), times)
    return X, y


@pytest.mark.integration
def test_evaluate_emits_heavy_censoring_warning():
    """evaluate() warns via MethodologicalWarning under ~90% censoring."""
    X, y = _make_heavy_censoring_split(censoring=0.90)
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)
    learner.train("coxph")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.evaluate(verbose=False)

    msgs = [str(w.message) for w in caught if _is_heavy_censoring(w)]
    assert len(msgs) == 1, f"Expected exactly one heavy-censoring warning, got: {msgs}"


@pytest.mark.integration
def test_evaluate_silent_under_moderate_censoring():
    """evaluate() stays silent on the heavy-censoring concern at ~40%."""
    X, y = _make_heavy_censoring_split(censoring=0.40)
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)
    learner.train("coxph")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.evaluate(verbose=False)

    msgs = [str(w.message) for w in caught if _is_heavy_censoring(w)]
    assert msgs == [], f"Did not expect a heavy-censoring warning, got: {msgs}"


@pytest.mark.integration
def test_run_emits_warning_once_no_duplication():
    """run() (evaluate + bootstrap over multiple models) warns exactly once.

    The dedup is per-cohort ("main"), so multiple models and the bootstrap
    pass do not produce duplicate warnings; a follow-up evaluate() stays
    silent.
    """
    X, y = _make_heavy_censoring_split(censoring=0.90)
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.run(
            ["coxph", "coxnet"],
            tune=False,
            bootstrap=True,
            n_bootstrap=2,
            n_importance_repeats=0,
            verbose=False,
            show_eval_progress=False,
            show_progress_bar=False,
        )
    msgs = [str(w.message) for w in caught if _is_heavy_censoring(w)]
    assert len(msgs) == 1, f"run() should warn exactly once, got: {msgs}"

    # A follow-up evaluate() on the same cohort stays silent.
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        learner.evaluate("coxph", verbose=False)
    msgs2 = [str(w.message) for w in caught2 if _is_heavy_censoring(w)]
    assert msgs2 == [], f"Expected dedup after run(), got: {msgs2}"
