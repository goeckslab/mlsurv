"""Integration tests for the ``validate()`` held-out-test delta printout.

``validate(verbose=True)`` annotates each headline metric with its change (Δ)
from the model's held-out-test (development) performance, so users see the
generalization gap directly. When no prior ``evaluate()`` established a
baseline, it prints a one-line hint instead. These tests exercise the real
``SurvivalLearner`` path; the per-printer formatting is covered by the fast
``TestDeltaAnnotation`` unit tests in ``test_printing.py``.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv.utils import SurvivalArray


def _make_small_data():
    """50-sample synthetic survival dataset."""
    rng = np.random.RandomState(42)
    n = 50
    X = pd.DataFrame({
        'age': rng.uniform(30, 80, n),
        'stage': rng.randint(1, 5, n),
        'grade': rng.uniform(1, 4, n),
        'tumor_size': rng.uniform(0.5, 10, n),
        'biomarker': rng.randn(n),
    })
    times = rng.exponential(2, n).clip(0.1, 10)
    events = rng.binomial(1, 0.6, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
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
    """Half-cohort drawn from training X/y — cheap and aligned."""
    X = learner._state.X
    y = learner._state.y
    half = len(X) // 2
    return X.iloc[:half].copy(), y[:half]


@pytest.mark.integration
def test_validate_printout_shows_delta_after_evaluate(trained_learner, capsys):
    """With a held-out-test baseline, the validate() table annotates Δ + footnote."""
    learner = trained_learner
    learner.evaluate(verbose=False)  # establishes the held-out-test baseline
    X_val, y_val = _val_cohort(learner)

    capsys.readouterr()  # discard anything printed so far
    learner.validate(X_val, y_val, name="C1", bootstrap=False, verbose=True)
    out = capsys.readouterr().out

    assert "(Δ" in out
    assert "change from held-out test set" in out


@pytest.mark.integration
def test_validate_printout_hint_without_evaluate(capsys):
    """Without a prior evaluate(), no Δ is shown — a one-line hint is printed.

    ``train()`` (unlike ``run()``) does not auto-evaluate on the held-out test
    split, so ``evaluation_result`` is ``None`` and there is no baseline.
    """
    X, y = _make_small_data()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")  # no evaluate() -> no held-out-test baseline
    assert learner.evaluation_result is None
    X_val, y_val = _val_cohort(learner)

    capsys.readouterr()
    learner.validate(X_val, y_val, name="C1", bootstrap=False, verbose=True)
    out = capsys.readouterr().out

    assert "Run evaluate() first" in out
    assert "(Δ" not in out
