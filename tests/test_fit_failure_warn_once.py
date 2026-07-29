"""FitFailureWarning surfaces once per learner, including across tuning.

``LearnerState.fit_failure_warned`` already collapsed the CV -> final-fit ->
calibration copies of a numerical-instability message. The tuner kept its own
count dict and emitted independently, so a ``tune()`` call surfaced the same
message twice: once from the Optuna trials (``" (xN)"``) and once from the
post-tune CV re-run (``" (N occurrences across K CV folds)"``). These tests pin
the shared-set behavior and the unsuffixed-key invariant that makes the phases
match each other.
"""

import threading
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import FitFailureWarning, MethodologicalWarning
from mlsurv.tuning.base import BaseTuner

MSG = "CoxPH(alpha=0) produced numerical instability."


@pytest.fixture
def tuner():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 3), columns=list("abc"))
    y = np.array(
        list(zip(rng.rand(40) < 0.7, rng.exponential(5, 40) + 0.5)),
        dtype=[("event", bool), ("time", float)],
    )
    t = BaseTuner(X, y)
    t._trial_warnings_lock = threading.Lock()
    return t


def _emit(tuner):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tuner._emit_trial_warnings()
    return [str(w.message) for w in caught]


def test_dedups_against_the_learner_set_and_stores_unsuffixed_key(tuner):
    seen = set()
    tuner._seen_warning_keys = seen
    tuner._trial_warnings = {(FitFailureWarning, MSG): 12}

    messages = _emit(tuner)

    # The user still sees the trial count.
    assert messages == [f"{MSG} (x12)"]
    # But the stored key is the raw message: the CV path builds its keys from
    # raw messages, so a suffixed key would never match and dedup would no-op.
    assert seen == {("FitFailureWarning", MSG)}

    # A later phase emitting the same message is suppressed.
    tuner._trial_warnings = {(FitFailureWarning, MSG): 3}
    assert _emit(tuner) == []


def test_other_categories_are_never_suppressed(tuner):
    """Only FitFailureWarning takes part here.

    The set itself is not scoped to that category: the CV path stores a key for
    every warning it re-emits. So CoxPH's ridge-penalty caveat, a plain
    MethodologicalWarning, can already be in the set and must still reach the
    user.
    """
    ridge = "CoxPH(alpha=0.5) applies an L2 ridge penalty."
    tuner._seen_warning_keys = {("MethodologicalWarning", ridge)}
    tuner._trial_warnings = {(MethodologicalWarning, ridge): 4}

    assert _emit(tuner) == [f"{ridge} (x4)"]


def test_unbound_set_keeps_previous_behavior(tuner):
    """A standalone tuner (no learner) behaves exactly as it did before."""
    assert tuner._seen_warning_keys is None
    tuner._trial_warnings = {(FitFailureWarning, MSG): 2}

    assert _emit(tuner) == [f"{MSG} (x2)"]
    assert _emit(tuner) == []      # dict cleared, not deduped


def test_tuner_and_cv_emitters_share_one_key_space():
    """The two emitters must agree on the key, or dedup silently does nothing.

    The tuner writes a key; the CV path's re-emitter must recognize it. Driving
    both against one set catches a divergence that asserting the tuple shape
    by hand would miss.
    """
    from mlsurv._warnings import _reemit_deduped_warnings

    shared = set()
    t = BaseTuner(*_unstable_data())
    t._trial_warnings_lock = threading.Lock()
    t._seen_warning_keys = shared
    t._trial_warnings = {(FitFailureWarning, MSG): 4}
    assert _emit(t) == [f"{MSG} (x4)"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reemit_deduped_warnings(
            {("FitFailureWarning", MSG): (3, "f.py", 1)},
            total_iterations=5,
            seen_messages=shared,
        )
    assert caught == []


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

def _unstable_data(n=80, seed=3):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, 4), columns=[f"f{i}" for i in range(4)])
    y = np.array(
        list(zip(rng.rand(n) < 0.7, rng.exponential(5, n) + 0.5)),
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


@pytest.mark.integration
def test_tuner_receives_the_learner_set(monkeypatch):
    """_create_tuner_for_model binds the learner's set onto every tuner."""
    X, y = _unstable_data()
    learner = SurvivalLearner(X, y)
    learner.setup(cv_folds=3, random_state=42, n_jobs=1, verbose=False)

    tuner = learner._create_tuner_for_model("coxnet", None, None, None, None)

    assert tuner._seen_warning_keys is learner._state.fit_failure_warned


def test_setup_clears_the_set():
    X, y = _unstable_data()
    learner = SurvivalLearner(X, y)
    learner.setup(cv_folds=3, random_state=42, n_jobs=1, verbose=False)
    learner._state.fit_failure_warned.add(("FitFailureWarning", MSG))

    learner.setup(cv_folds=3, random_state=42, n_jobs=1, verbose=False)

    assert learner._state.fit_failure_warned == set()


def test_each_learners_set_starts_empty():
    """The set is per learner, so a new learner does not inherit suppressions.

    Note this is about the *set*, not about what the user sees. Python's own
    "once" filter on IterativeWarning is process-global and keyed on message
    text, so a second learner with identical instability is usually quiet at
    the console anyway. `_emit` below neutralizes that filter on purpose, to
    isolate the per-learner set.
    """
    X, y = _unstable_data()
    emitted = []
    for _ in range(2):
        learner = SurvivalLearner(X, y)
        learner.setup(cv_folds=3, random_state=42, n_jobs=1, verbose=False)
        t = BaseTuner(X, y)
        t._trial_warnings_lock = threading.Lock()
        t._seen_warning_keys = learner._state.fit_failure_warned
        # Two phases of the same learner: the second is suppressed.
        for _ in range(2):
            t._trial_warnings = {(FitFailureWarning, MSG): 1}
            emitted.extend(_emit(t))

    assert emitted == [MSG, MSG]     # one per learner, not one per phase
