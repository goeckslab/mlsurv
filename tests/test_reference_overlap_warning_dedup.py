"""Tests for deduping the reference feature-overlap warning by signature.

The data-leakage warning emitted when a reference name collides with an
``X_val`` column is a property of ``(reference_names, columns)``, not of the
call. In a multi-cohort ``validate()`` loop (vignette 07 §4) with the same
references dict and same column schema, it should fire **once per signature
per session**, not once per call. The same signature should also dedupe
across ``validate()`` and ``benchmark()`` because the underlying leakage
concern is identical.
"""

from __future__ import annotations

import copy
import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import SuggestionWarning
from mlsurv.utils import SurvivalArray


def _make_small_data():
    """50-sample synthetic survival dataset (matches test_validate_references)."""
    rng = np.random.RandomState(42)
    n = 50
    X = pd.DataFrame({
        "age": rng.uniform(30, 80, n),
        "stage": rng.randint(1, 5, n),
        "grade": rng.uniform(1, 4, n),
        "tumor_size": rng.uniform(0.5, 10, n),
        "biomarker": rng.randn(n),
    })
    times = rng.exponential(2, n).clip(0.1, 10)
    events = rng.binomial(1, 0.6, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    ))
    return X, y


@pytest.fixture(scope="module")
def _trained_template():
    X, y = _make_small_data()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.run(["coxph"], tune=False)
    return learner


@pytest.fixture
def trained_learner(_trained_template):
    return copy.deepcopy(_trained_template)


def _val_cohort(learner):
    X = learner._state.X
    y = learner._state.y
    half = len(X) // 2
    return X.iloc[:half].copy(), y[:half]


def _count_leak_warnings(records):
    return [
        r for r in records
        if issubclass(r.category, SuggestionWarning)
        and "data leakage" in str(r.message)
    ]


@pytest.mark.integration
def test_reference_overlap_warning_fires_once_across_validate_calls(trained_learner):
    """5 validate() calls with the same overlap signature → 1 warning."""
    X_val, y_val = _val_cohort(trained_learner)
    refs = {"age": np.random.RandomState(7).randn(len(X_val))}

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        for cohort in range(5):
            trained_learner.validate(
                X_val, y_val, name=f"Cohort{cohort}", references=refs,
                bootstrap=False, verbose=False,
            )

    leaks = _count_leak_warnings(w)
    assert len(leaks) == 1, (
        f"Expected exactly 1 leakage warning across 5 calls; got {len(leaks)}: "
        f"{[str(r.message) for r in leaks]}"
    )


@pytest.mark.integration
def test_reference_overlap_warning_fires_again_for_new_signature(trained_learner):
    """Different overlap signature → separate warning."""
    X_val, y_val = _val_cohort(trained_learner)
    refs_age = {"age": np.random.RandomState(1).randn(len(X_val))}
    refs_stage = {"stage": np.random.RandomState(2).randn(len(X_val))}

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        trained_learner.validate(
            X_val, y_val, name="A1", references=refs_age,
            bootstrap=False, verbose=False,
        )
        trained_learner.validate(
            X_val, y_val, name="A2", references=refs_age,
            bootstrap=False, verbose=False,
        )
        trained_learner.validate(
            X_val, y_val, name="S1", references=refs_stage,
            bootstrap=False, verbose=False,
        )
        trained_learner.validate(
            X_val, y_val, name="S2", references=refs_stage,
            bootstrap=False, verbose=False,
        )

    leaks = _count_leak_warnings(w)
    assert len(leaks) == 2, (
        f"Expected one warning per distinct signature (2 total); got {len(leaks)}"
    )


@pytest.mark.integration
def test_reference_overlap_warning_resets_on_resetup(trained_learner):
    """setup() re-init clears the dedup set so the same signature warns again."""
    X_val, y_val = _val_cohort(trained_learner)
    refs = {"age": np.random.RandomState(3).randn(len(X_val))}

    with _warnings.catch_warnings(record=True) as w1:
        _warnings.simplefilter("always")
        trained_learner.validate(
            X_val, y_val, name="Pre", references=refs,
            bootstrap=False, verbose=False,
        )
    assert len(_count_leak_warnings(w1)) == 1

    # Re-setup clears all results + the dedup set.
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        trained_learner.setup(random_state=42, feature_selector=None)
        trained_learner.run(["coxph"], tune=False)

    with _warnings.catch_warnings(record=True) as w2:
        _warnings.simplefilter("always")
        trained_learner.validate(
            X_val, y_val, name="Post", references=refs,
            bootstrap=False, verbose=False,
        )
    assert len(_count_leak_warnings(w2)) == 1, (
        "Expected the warning to fire again after setup() re-init"
    )


@pytest.mark.integration
def test_no_overlap_emits_no_warning_and_no_dedup_entry(trained_learner):
    """Negative: refs that don't collide with any column → no warning, empty dedup set."""
    X_val, y_val = _val_cohort(trained_learner)
    refs = {"novel_score": np.random.RandomState(11).randn(len(X_val))}
    assert "novel_score" not in X_val.columns

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        trained_learner.validate(
            X_val, y_val, name="NoLeak", references=refs,
            bootstrap=False, verbose=False,
        )

    assert _count_leak_warnings(w) == []
    # Guards against a refactor that records the signature outside the
    # ``if feature_overlap:`` branch.
    assert trained_learner._state.reference_overlap_warned == set()


@pytest.mark.integration
def test_partial_overlap_signatures_are_distinct(trained_learner):
    """`{age}` and `{age, stage}` are different signatures → each warns once."""
    X_val, y_val = _val_cohort(trained_learner)
    rng = np.random.RandomState(12)
    refs_one = {"age": rng.randn(len(X_val))}
    refs_two = {"age": rng.randn(len(X_val)), "stage": rng.randn(len(X_val))}

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        # Each signature appears twice; expect 2 warnings (one per signature).
        for refs in (refs_one, refs_two, refs_one, refs_two):
            trained_learner.validate(
                X_val, y_val, name="P", references=refs,
                bootstrap=False, verbose=False,
            )

    assert len(_count_leak_warnings(w)) == 2


@pytest.mark.integration
def test_benchmark_and_validate_share_overlap_dedup(trained_learner):
    """benchmark() warns first; subsequent validate() with same signature stays quiet."""
    X_val, y_val = _val_cohort(trained_learner)
    # ``age`` is a column in both training X and X_val — same signature on
    # both code paths.
    n_train = len(trained_learner._state.X)
    refs_train = {"age": np.random.RandomState(4).randn(n_train)}
    refs_val = {"age": np.random.RandomState(5).randn(len(X_val))}

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        trained_learner.benchmark(
            references=refs_train, bootstrap=False, verbose=False,
        )
        trained_learner.validate(
            X_val, y_val, name="AfterBench", references=refs_val,
            bootstrap=False, verbose=False,
        )

    leaks = _count_leak_warnings(w)
    assert len(leaks) == 1, (
        f"Expected 1 warning total (benchmark first, validate suppressed); "
        f"got {len(leaks)}: {[str(r.message) for r in leaks]}"
    )
