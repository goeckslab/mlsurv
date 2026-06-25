"""Report sections use the public API, not ``learner._state``.

Three guards:

1. A convention test (no-cost line scan over the sections tree) that pins down
   the rule that ``learner._state`` inside ``mlsurv/reports/sections/`` is only
   touched by ``base.py`` — the single report-layer adapter — plus a small
   documented set of exempt ``getattr`` reads. ``base.py`` holds the
   display-config adapters (``outcome_name``/``time_unit``/``feature_groups``/
   ``group_colors``) plus the preprocessing-template adapters
   (``imputer``/``scaler``/``feature_selector``); every other field a leaf
   section needs is read through the public ``SurvivalLearner`` API.

   An earlier guard pinned only the dotted form ``._state.<field>``. The
   scan was later broadened to every form that couples a section to the
   ``LearnerState`` schema:

   - dotted: ``learner._state.field``
   - getattr-comma: ``getattr(learner._state, 'field', default)``
   - alias: ``state = learner._state``
   - quoted: ``getattr(learner, '_state', None)``

   The getattr-comma form is the dangerous one: with a default, a renamed
   ``LearnerState`` field makes the section silently drop content instead of
   erroring. So it is allowed ONLY for the fields in ``_EXEMPT_STATE_FIELDS``
   below — config/diagnostic fields that deliberately have no public accessor.
   The exemption contract:

   - the read must be the 3-arg form
     ``getattr(<...>._state, '<field>', <default>)``, with the ``getattr(``
     opener through the field-name comma on one line;
   - the field must be in ``_EXEMPT_STATE_FIELDS``;
   - the exemption auto-expires: ``test_exempt_fields_have_no_public_accessor``
     fails the moment a public ``property`` is added for an exempt field,
     forcing the call sites to migrate.

2. Unit assertions that the public accessors exist and behave
   as read-only views: ``is_setup`` (bool) and ``validation_results`` /
   ``subpopulation_results`` (read-only ``mappingproxy``).

3. Unit assertions on the line matcher itself, so the scan's coverage of each
   access form is pinned.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner


# ---------------------------------------------------------------------------
# Convention test (no-cost: a line scan over .py files)
# ---------------------------------------------------------------------------

_SECTIONS_ROOT = Path(__file__).resolve().parent.parent / 'mlsurv' / 'reports' / 'sections'

# The only file allowed to touch ``learner._state`` directly: it is the report
# layer's single adapter, exposing the handful of fields reports need as
# ``@property`` accessors (display config + unfitted preprocessing templates).
_ALLOWED_STATE_FILES = {
    _SECTIONS_ROOT / 'base.py',
}

# Config/diagnostic fields with no public accessor. Sections may
# read these ONLY via the single-line 3-arg defensive form
# ``getattr(<...>._state, '<field>', <default>)``. Adding a public accessor for
# one of these makes ``test_exempt_fields_have_no_public_accessor`` fail, which
# is the signal to migrate the call sites and remove the field from this set.
_EXEMPT_STATE_FIELDS = {
    'categorical_groups',
    'device',
    'artifact_path',
    'user_eval_times',
    'subpopulations_test',
    'subpopulations_train',
    'stratify_spec',
    'stratify_subpopulations',
    'presplit',
    'stratify_array',
    'ipcw_support_flag',
}

# Matches dotted (``._state.field``), getattr-comma (``._state,``), and alias
# (``= learner._state``) forms — any token-level mention of ``._state``.
_STATE_TOKEN = re.compile(r"\._state\b")
# An exempt read: 3-arg getattr on any receiver path ending in ``._state``
# (``learner._state``, ``self.learner._state``, ``self._s.learner._state``).
# The trailing comma requires the 3-arg form; the opener through the
# field-name comma must sit on one line (the default may wrap).
_EXEMPT_GETATTR = re.compile(r"getattr\(\s*[\w.]+\._state\s*,\s*['\"](\w+)['\"]\s*,")
# The quoted form ``getattr(learner, '_state', ...)`` carries no ``._state``
# token, so ban the quoted attribute name outright.
_QUOTED_STATE = re.compile(r"['\"]_state['\"]")


def _offending_lines(text):
    """Return 1-based line numbers in *text* that touch ``_state`` illegally."""
    offenders = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = _EXEMPT_GETATTR.sub(
            lambda m: '' if m.group(1) in _EXEMPT_STATE_FIELDS else m.group(0),
            line,
        )
        if _STATE_TOKEN.search(stripped) or _QUOTED_STATE.search(line):
            offenders.append(lineno)
    return offenders


def test_no_unlisted_state_access_in_sections():
    """``learner._state`` access in reports/sections stays in ``base.py``.

    Catches a leaf section coupling itself to the ``LearnerState`` schema in
    ANY form — dotted ``._state.<field>``, comma-form
    ``getattr(.._state, 'field', default)``, alias ``state = learner._state``,
    or quoted ``getattr(learner, '_state', ...)``. The fix is to read through
    a public ``SurvivalLearner`` accessor (e.g. ``learner.models``,
    ``learner.validation_results``) — or, for a
    presentation-only template, add an adapter ``@property`` to ``base.py``
    and read ``self.<name>``.

    Sole exception: the documented ``_EXEMPT_STATE_FIELDS`` may be read via
    the single-line 3-arg form ``getattr(<...>._state, '<field>', <default>)``.

    Scope: the scan is line-literal, so it also flags ``._state`` inside
    docstrings, comments, and user-facing code snippets — those must teach
    the public path.
    """
    assert _SECTIONS_ROOT.is_dir(), f"sections package not found at {_SECTIONS_ROOT}"

    offenders = []
    for path in _SECTIONS_ROOT.rglob('*.py'):
        if path in _ALLOWED_STATE_FILES:
            continue
        text = path.read_text(encoding='utf-8')
        for lineno in _offending_lines(text):
            offenders.append(f"{path.relative_to(_SECTIONS_ROOT)}:{lineno}")

    assert not offenders, (
        "``learner._state`` access added in report section file(s) outside "
        "base.py. Route it through the public SurvivalLearner API, or add a "
        "presentation-only adapter @property to reports/sections/base.py and "
        "read ``self.<name>``. Fields in _EXEMPT_STATE_FIELDS may only use "
        "the single-line 3-arg form getattr(<...>._state, '<field>', "
        "<default>). This also covers docstrings/comments and user-facing "
        "code snippets. Offenders:\n  - "
        + "\n  - ".join(offenders)
    )


def test_exempt_fields_have_no_public_accessor():
    """Each exemption expires the moment a public accessor appears.

    If this fails, a ``SurvivalLearner`` property now exists for an exempt
    field: migrate the section call sites to the accessor and remove the
    field from ``_EXEMPT_STATE_FIELDS``.
    """
    stale = [
        field for field in sorted(_EXEMPT_STATE_FIELDS)
        if isinstance(getattr(SurvivalLearner, field, None), property)
    ]
    assert not stale, (
        "Exempt _state field(s) now have a public SurvivalLearner accessor; "
        "migrate the getattr call sites in reports/sections and drop the "
        "exemption(s): " + ", ".join(stale)
    )


def test_exempt_fields_exist_on_learner_state():
    """Each exempt field must still exist on ``LearnerState``.

    The exempt reads use ``getattr(..., default)``, so renaming a
    ``LearnerState`` field would silently turn them into their defaults
    (vanishing report content) with the line scan still green. This test
    makes that rename loud.
    """
    import dataclasses

    from mlsurv.learner._state import LearnerState

    names = {f.name for f in dataclasses.fields(LearnerState)}
    missing = sorted(_EXEMPT_STATE_FIELDS - names)
    assert not missing, (
        "Exempt _state field(s) no longer exist on LearnerState — update the "
        "report-section getattr call sites and _EXEMPT_STATE_FIELDS: "
        + ", ".join(missing)
    )


def test_offending_lines_matcher():
    """Pin the matcher's coverage of every ``_state`` access form."""
    sample = "\n".join([
        "x = learner._state.models",                                   # 1 dotted
        "y = getattr(self.learner._state, 'benchmark_result', None)",  # 2 comma, not exempt
        "state = self.learner._state",                                 # 3 alias
        "z = getattr(self.learner._state, 'device', None)",            # 4 comma, exempt
        "w = getattr(learner, '_state', None)",                        # 5 quoted
        "ok = self.learner.models",                                    # 6 clean
        "d = getattr(section.learner._state, 'categorical_groups', None)",  # 7 exempt
        "e = getattr(self._s.learner._state, 'ipcw_support_flag', None)",   # 8 exempt
        'f = getattr(learner._state, "device", None)',                 # 9 exempt, double-quoted
        "g = getattr(learner._state, 'device')",                       # 10 2-arg: NOT exempt
        "h = getattr(learner._state,",                                 # 11 wrapped field name: NOT exempt
        "            'device', None)",                                 # 12 continuation (clean)
    ])
    assert _offending_lines(sample) == [1, 2, 3, 5, 10, 11]


# ---------------------------------------------------------------------------
# Unit tests for the 3 public accessors
# ---------------------------------------------------------------------------

def _tiny_learner():
    """A minimal learner for accessor checks (no training needed)."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=['a', 'b', 'c'])
    y = np.array(
        [(bool(e), float(t)) for e, t in zip(rng.integers(0, 2, n),
                                             rng.uniform(1, 100, n))],
        dtype=[('event', bool), ('time', float)],
    )
    return SurvivalLearner(X, y)


def test_is_setup_property():
    """``is_setup`` is a public bool flipping False -> True at setup()."""
    learner = _tiny_learner()
    assert learner.is_setup is False
    learner.setup(random_state=0)
    assert learner.is_setup is True


def test_validation_results_is_readonly_view():
    """``validation_results`` is a read-only mappingproxy over the cohort dict."""
    learner = _tiny_learner()
    assert isinstance(learner.validation_results, MappingProxyType)
    assert len(learner.validation_results) == 0
    with pytest.raises(TypeError):
        learner.validation_results['x'] = object()


def test_subpopulation_results_is_readonly_view():
    """``subpopulation_results`` is a read-only mappingproxy over the group dict."""
    learner = _tiny_learner()
    assert isinstance(learner.subpopulation_results, MappingProxyType)
    assert len(learner.subpopulation_results) == 0
    with pytest.raises(TypeError):
        learner.subpopulation_results['x'] = object()
