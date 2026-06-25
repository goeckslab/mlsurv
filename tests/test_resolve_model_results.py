"""Resolver scaffolding.

Adds ``SurvivalLearner._resolve_model_results`` (returns ``ModelResult``
objects rather than the string keys returned by ``_resolve_model_names``)
and a convention test that pins down the rule that direct
``_state.models[<key>]`` access only happens in files on the allowlist.

After a later audit, the
allowlist was reduced. Two former entries had real ``_tuned``-collision
bugs and were migrated through ``_lookup_model_by_name``
(``_recalibration.py``, ``cli/_commands.py``). The remaining readers
receive names that are already validated storage keys (storage writes,
or iteration over ``_state.models`` itself), so ``_lookup_model_by_name``
would be a no-op at those sites. New direct accesses elsewhere must go
through a resolver. See ``CONTRIBUTING.md`` State Consistency.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv.results.containers import ModelResult


# ---------------------------------------------------------------------------
# Convention test (no-cost: just a regex over .py files)
# ---------------------------------------------------------------------------

_MLSURV_ROOT = Path(__file__).resolve().parent.parent / 'mlsurv'

# Files allowed to access ``_state.models[...]`` directly. The resolver audit
# resolved the migration question: ``_recalibration.py`` and ``cli/_commands.py``
# were migrated through ``_lookup_model_by_name`` (they had real ``_tuned``
# collision bugs), and the remaining entries are permanent because the lookup
# name at each site is already a validated storage key (storage write, or a
# ``ModelResult`` iteration where the key originated from the registry itself).
_ALLOWED_STATE_MODELS_FILES = {
    # Resolvers + writes — these *are* the resolver layer.
    _MLSURV_ROOT / 'learner' / '_core.py',
    _MLSURV_ROOT / 'learner' / '_training.py',
    _MLSURV_ROOT / 'learner' / '_tuning.py',
    _MLSURV_ROOT / 'learner' / '_cv_training.py',
    # ``_swap_benchmark_eval`` writes a synthetic benchmark ``ModelResult``
    # (line ~386) and reads/back-fills its ``.evaluation`` (lines ~396, ~426).
    # ``name`` is iterated over ``br.evaluation.concordance.index`` (already a
    # storage-key column) and over ``self._ref_names`` (populated by the same
    # iteration) — never sourced from user-supplied aliases.
    _MLSURV_ROOT / 'viz' / '_resolve.py',
    # Per-model derived-state write sites: importance_test() and
    # rmst/logrank_test() store their result tables on the registered
    # ``ModelResult`` (``_state.models[name].pimp_importance_test = ...`` etc.).
    # ``name`` is a validated storage key at each site — resolved via
    # ``_resolve_model_input`` (PIMP guards an unregistered key with a
    # ``ValueError``) or ``_get_test_risk_groups`` (coerces to a registered
    # test-set model type).
    _MLSURV_ROOT / 'learner' / '_evaluation_interactions.py',
    # Receives the already-resolved storage key from
    # ``predict()`` -> ``_resolve_single_model()`` upstream.
    _MLSURV_ROOT / 'learner' / '_patient_report_html.py',
    # ``viz/_context.py`` only mentions ``_state.models[`` in a docstring;
    # listed for forward compatibility if a real lookup is added.
    _MLSURV_ROOT / 'viz' / '_context.py',
}


def test_resolve_model_results_exists():
    """Phase 1 deliverable: the resolver helper is on the learner."""
    assert hasattr(SurvivalLearner, '_resolve_model_results'), (
        "the resolver layer introduces SurvivalLearner._resolve_model_results"
    )


def test_no_unlisted_state_models_access():
    """Direct ``_state.models[...]`` access stays in the allowlist.

    Catches *new* readers landing in unlisted files. Adding a new
    direct accessor without going through a resolver requires either a
    resolver fix or a justified addition to ``_ALLOWED_STATE_MODELS_FILES``.
    """
    assert _MLSURV_ROOT.is_dir(), f"mlsurv package not found at {_MLSURV_ROOT}"

    pattern = re.compile(r"_state\.models\[")
    offenders = []
    for path in _MLSURV_ROOT.rglob('*.py'):
        if path in _ALLOWED_STATE_MODELS_FILES:
            continue
        text = path.read_text(encoding='utf-8')
        if pattern.search(text):
            offenders.append(str(path.relative_to(_MLSURV_ROOT)))

    assert not offenders, (
        "Direct ``_state.models[...]`` access added in unlisted file(s). "
        "Use ``learner._resolve_model_results`` / "
        "``_resolve_single_model`` / ``_resolve_model_input`` instead — or, "
        "for a legitimate write site, extend ``_ALLOWED_STATE_MODELS_FILES`` "
        "in this test. Offenders:\n  - " + "\n  - ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Unit tests for ``_resolve_model_results``
# ---------------------------------------------------------------------------

def _make_small_data(n=80, seed=0):
    """Tiny synthetic dataset for the resolver tests."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        'a': rng.randn(n),
        'b': rng.randn(n),
        'c': rng.randn(n),
    })
    times = rng.exponential(scale=10.0, size=n) + 1.0
    events = rng.randint(0, 2, size=n).astype(bool)
    y = np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
    )
    return X, y


@pytest.fixture(scope='module')
def trained_learner():
    """One-coxph learner shared across input-shape tests (no mutation)."""
    X, y = _make_small_data()
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=0)
    learner.train('coxph')
    return learner


def test_resolve_results_none_returns_all(trained_learner):
    out = trained_learner._resolve_model_results(None)
    assert all(isinstance(r, ModelResult) for r in out)
    assert [r.name for r in out] == ['coxph']


def test_resolve_results_none_empty_raises():
    """Match ``_resolve_all_models``: fail loudly on an empty learner."""
    X, y = _make_small_data()
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=0)
    # Note: no train/run; _state.models is empty.
    with pytest.raises(ValueError, match="No trained models"):
        learner._resolve_model_results(None)


def test_resolve_results_str(trained_learner):
    out = trained_learner._resolve_model_results('coxph')
    assert len(out) == 1
    assert isinstance(out[0], ModelResult)
    assert out[0].name == 'coxph'


def test_resolve_results_modelresult(trained_learner):
    mr = trained_learner._state.models['coxph']
    out = trained_learner._resolve_model_results(mr)
    # Identity, not equality — the resolver's contract is "return the
    # stored object, not a copy". Dataclass __eq__ would mask a refactor
    # that accidentally returned a clone.
    assert len(out) == 1
    assert out[0] is mr


def test_resolve_results_rejects_non_modelresult_object(trained_learner):
    """Tighter than ``_resolve_model_names``'s ``hasattr(.., 'name')``:
    a duck-typed object that happens to have ``.name`` (Series, Path,
    Enum, ...) gets a clean TypeError instead of a confusing
    'model not found' from ``_lookup_model_by_name``.
    """
    class FakeNamed:
        name = 'coxph'

    # Scalar non-ModelResult: hits the top-level type guard.
    with pytest.raises(TypeError, match="expected None, str, ModelResult"):
        trained_learner._resolve_model_results(FakeNamed())
    # As a list element: hits the per-element type guard.
    with pytest.raises(TypeError, match="expected str or ModelResult"):
        trained_learner._resolve_model_results([FakeNamed()])


def test_resolve_results_empty_list(trained_learner):
    """Pre-filtered empty list is caller intent (symmetric with empty
    filter_fn result), not an error. Pins the docstring contract.
    """
    assert trained_learner._resolve_model_results([]) == []


def test_resolve_results_unknown_name_raises(trained_learner):
    """Lookup failures propagate from ``_lookup_model_by_name``. This is
    the centralization-of-failure-semantics test: any future refactor
    that swallows lookup errors will break here. Covers both the str
    scalar path and the list-element path.
    """
    with pytest.raises(ValueError):
        trained_learner._resolve_model_results('not_a_model')
    with pytest.raises(ValueError):
        trained_learner._resolve_model_results(['coxph', 'not_a_model'])


def test_resolve_results_mixed_list(trained_learner):
    mr = trained_learner._state.models['coxph']
    # Same model twice via two different input shapes — proves both
    # branches of _resolve_model_names + _lookup_model_by_name agree.
    out = trained_learner._resolve_model_results(['coxph', mr])
    assert [r.name for r in out] == ['coxph', 'coxph']
    assert all(isinstance(r, ModelResult) for r in out)


def test_resolve_results_filter_fn(trained_learner):
    # filter_fn applies only when models is None
    out = trained_learner._resolve_model_results(
        None, filter_fn=lambda k, v: k == 'coxph'
    )
    assert [r.name for r in out] == ['coxph']
    out_empty = trained_learner._resolve_model_results(
        None, filter_fn=lambda k, v: False
    )
    # Empty after filtering is NOT the same as the "no trained models"
    # error path — `_resolve_model_names` returned [] from a populated
    # store, so we just return [].
    assert out_empty == []


@pytest.mark.integration
def test_resolve_results_after_tune_overwrite():
    """Load-bearing: ``tune()`` overwrites the baseline at 'coxnet'.
    Both 'coxnet' (str) and a now-stale baseline ``ModelResult`` reference
    (whose ``.name`` is still 'coxnet') resolve to the **tuned** variant now
    stored under 'coxnet' — name-based resolution always returns the current
    stored object. Directly proves the stale-reference bug class the resolver prevents.
    """
    X, y = _make_small_data()
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=0)
    learner.run(['coxnet'], tune=False, evaluate=False, verbose=False)
    baseline_mr = learner._state.models['coxnet']
    learner.tune(models='coxnet', n_trials=2, evaluate=False, verbose=False)

    # Single storage namespace: tuned overwrote the baseline; no _tuned key.
    assert 'coxnet' in learner._state.models
    assert 'coxnet_tuned' not in learner._state.models
    assert learner._state.models['coxnet'].is_tuned
    assert baseline_mr.name == 'coxnet'

    # str path: 'coxnet' resolves to the current (tuned) variant.
    by_str = learner._resolve_model_results('coxnet')
    assert [r.name for r in by_str] == ['coxnet']
    assert by_str[0].is_tuned

    # ModelResult path: stale baseline ref resolves by .name to the tuned one.
    by_obj = learner._resolve_model_results(baseline_mr)
    assert [r.name for r in by_obj] == ['coxnet']
    assert by_obj[0].is_tuned
